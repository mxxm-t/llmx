import contextlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from unittest.mock import patch

import baseline
import baseline_8b as consumer
import baseline_qwen35
import baseline_layered as layered
import common


def logits_text(case):
    return "tokens: %d\n" % case["n_tokens"] + "".join(
        "%d %.6f\n" % pair for pair in zip(case["top_ids"], case["top_logits"]))


def ppl_text(case, nll=None, tokens=247, context=consumer.MODEL_CONTEXT):
    nll = case["mean_nll"] if nll is None else nll
    return ("tokens: %d\nused tokens: %d\nscored tokens: %d\nchunks: %d\n"
            "context size: %d\nmean NLL: %.6g\nperplexity: %.6g\n") % (
        tokens, case["used_tokens"], case["n_scored"], case["chunks"],
        case["context_size"] or context, nll, math.exp(nll))


def simulated_llmx(docs, tokens, context, refuse=None):
    """A stand-in for subprocess.run that answers every command of a consumer's run from the goldens `docs`, as a passing llmx would.
    With `refuse`, every `logits` and `perplexity` command fails with that error instead."""
    ids = {case["text"]: case["ids"] for case in docs["baseline_tokenizer.json"]["cases"]}
    ids.update((case["text"], case["token_ids"]) for case in docs["baseline_logits.json"]["cases"])
    doc = docs["baseline_perplexity.json"]
    ids[doc["text"]] = doc["token_ids"]
    logits = {case["text"]: logits_text(case) for case in docs["baseline_logits.json"]["cases"]}

    def flag(command, name):
        return int(command[command.index(name) + 1]) if name in command else 0

    def llmx(command, **kwargs):
        if refuse and command[1] in ("logits", "perplexity"):
            return subprocess.CompletedProcess(command, 1, b"", refuse.encode("utf-8"))
        if command[1] == "tokenize":
            out = " ".join(map(str, ids[command[3]]))
        elif command[1] == "logits":
            out = logits[command[3]]
        elif command[1] == "perplexity":
            window = (flag(command, "--ctx-size"), flag(command, "--chunks"))
            out = ppl_text(next(case for case in common.ppl_cases(doc)
                                if (case["context_size"], case["max_chunks"]) == window), tokens=tokens, context=context)
        else:
            out = "llmx 0\n"
        return subprocess.CompletedProcess(command, 0, out.encode("utf-8"), b"")
    return llmx


class ReferenceConsumer(unittest.TestCase):

    def test_device_type_refusal_is_a_skip_only_for_a_selected_device(self):
        message = "error: inference: layer 3's feed-forward part needs tensor blk.3.ffn_up_exps.weight of type 39, which the backend of device 0 does not support"
        for text in (message, message.replace("type 39", "type MXFP4 (39)"), message.replace("type 39", "type Q8_0 (8)")):
            for device in ("", "vulkan:0"):
                with patch.dict(os.environ, {"LLMX_DEVICE": device}):
                    self.assertEqual(common.device_lacks_kernel(1, text), bool(device))
                    self.assertFalse(common.device_lacks_kernel(0, text))
                    self.assertFalse(common.device_lacks_kernel(1, "error: inference: incompatible tensor layout blk.3.ffn_up_exps.weight"))
                    self.assertFalse(common.device_lacks_kernel(1, text.replace("does not support", "failed during upload")))

    @classmethod
    def setUpClass(cls):
        cls.docs = consumer.load_goldens()

    def test_token_ids_and_fixture_tampering(self):
        for case in self.docs["baseline_tokenizer.json"]["cases"]:
            output = " ".join(map(str, case["ids"]))
            consumer.check_ids(output, case["ids"])
            with self.assertRaises(ValueError):
                consumer.check_ids(output + " 42", case["ids"])
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            for name in consumer.FIXTURE_SHA256:
                (data / name).write_bytes((consumer.DATA / name).read_bytes())
            name = "baseline_logits.json"
            (data / name).write_text((data / name).read_text(encoding="utf-8") + " ", encoding="utf-8")
            with patch.object(consumer, "DATA", data), self.assertRaisesRegex(ValueError, "fixture changed"):
                consumer.load_goldens()

    def test_logit_damage_is_rejected(self):
        for case in self.docs["baseline_logits.json"]["cases"]:
            consumer.check_logits(logits_text(case), case)
        case = self.docs["baseline_logits.json"]["cases"][0]
        good = logits_text(case)
        lines = good.splitlines()
        invalid = ["\n".join(lines[:-1]), good + lines[-1] + "\n",
                   good.replace("tokens: 5", "tokens: 6")]
        for row in ["%d nan" % case["top_ids"][8], "%d inf" % case["top_ids"][8],
                    "%d 101" % case["top_ids"][8], "151936 17.8", "-1 17.8",
                    "%d 17.8" % case["top_ids"][0]]:
            changed = lines.copy()
            changed[-2] = row
            invalid.append("\n".join(changed))
        for indices in [(1, 2), (5, 6)]:
            changed = lines.copy()
            a, b = indices
            ids = [changed[i].split()[0] for i in indices]
            changed[a] = ids[1] + " " + changed[a].split()[1]
            changed[b] = ids[0] + " " + changed[b].split()[1]
            invalid.append("\n".join(changed))
        changed = lines.copy()
        changed[-1] = changed[-1].split()[0] + " 25.0"
        invalid.append("\n".join(changed))
        for output in invalid:
            with self.subTest(output=output), self.assertRaises(ValueError):
                consumer.check_logits(output, case)

    def test_top5_boundary_swaps(self):
        # llmx's logits from the 4th place down sit `step` apart, so a step of 0.01 makes any two of them an llmx tie.
        def output(case, order, step):
            values = case["top_logits"][:3] + [case["top_logits"][4] - step * i for i in range(7)]
            return "tokens: %d\n" % case["n_tokens"] + "".join(
                "%d %.6f\n" % (case["top_ids"][rank], value) for rank, value in zip(order, values))

        # The first case's reference puts its 6th token 0.125 below its 5th and its 4th 0.369 above; the second puts its 6th 0.023 below.
        apart, near = self.docs["baseline_logits.json"]["cases"][:2]
        swap = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9]
        # The 5th and 6th trading places is forgiven once, whether llmx or the reference puts them within 0.1.
        for case in (apart, near):
            self.assertEqual(consumer.check_logits(output(case, swap, 0.01), case)["top5_overlap"], 5)
        # An llmx gap of 0.15, a strong 4th place dropping out, or the 7th place coming in, alone or beside a forgiven swap, is a miss.
        for order, step in [(swap, 0.15), ([0, 1, 2, 5, 4, 3, 6, 7, 8, 9], 0.01),
                            ([0, 1, 2, 3, 6, 4, 5, 7, 8, 9], 0.01), ([0, 1, 2, 5, 6, 4, 3, 7, 8, 9], 0.01)]:
            with self.subTest(order=order, step=step), self.assertRaisesRegex(ValueError, "top-5 overlap 4/5"):
                consumer.check_logits(output(apart, order, step), apart)

    def test_ppl_counters_and_bounds(self):
        doc = self.docs["baseline_perplexity.json"]
        for case in common.ppl_cases(doc):
            good = ppl_text(case)
            consumer.check_ppl(good, case, 247)
            invalid = [good + "chunks: 1\n", good.replace("tokens: 247", "tokens: 246"),
                       good.replace("mean NLL:", "wrong field:"),
                       ppl_text(case, case["mean_nll"] + 0.03)]
            for key in ["mean NLL", "perplexity"]:
                for value in ["nan", "inf", "-1"]:
                    invalid.append("\n".join(key + ": " + value if line.startswith(key + ":") else line
                                             for line in good.splitlines()))
            for key in ["used tokens", "scored tokens", "chunks", "context size"]:
                invalid.append("\n".join(key + ": 0" if line.startswith(key + ":") else line
                                         for line in good.splitlines()))
            for output in invalid:
                with self.subTest(output=output), self.assertRaises(ValueError):
                    consumer.check_ppl(output, case, 247)

    def test_passing_run_scores_each_nll_case_both_ways(self):
        sha256 = consumer.file_sha256
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ["--exe", str(root / "llmx"), "--model", str(root / "model"), "--output-dir", str(root / "result")]
            with patch.object(consumer, "file_sha256",
                              side_effect=lambda path: sha256(path) if path.name == "excerpt.txt" else consumer.MODEL_SHA256), \
                 patch.object(consumer.subprocess, "run", side_effect=simulated_llmx(self.docs, 247, consumer.MODEL_CONTEXT)), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(consumer.main(args), 0)
            report = json.loads((root / "result/report.json").read_text())
        self.assertEqual(report["status"], "pass")
        self.assertEqual(len(report["checks"]), 41)
        scored = [(record["label"], "--per-token" in record["command"]) for record in report["commands"]
                  if record["label"].startswith("ppl-0")]
        self.assertEqual(scored, [("ppl-%02d" % index + ("-per-token" if per_token else ""), per_token)
                                  for index in range(4) for per_token in (False, True)])

    def test_failed_launches_leave_failure_report(self):
        for scenario in ["wrong-model", "timeout", "nonzero", "launch-error"]:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = ["--exe", str(root / "llmx"), "--model", str(root / "model"),
                        "--output-dir", str(root / "result")]
                digest = consumer.MODEL_SHA256 if scenario != "wrong-model" else "0" * 64
                with patch.object(consumer, "file_sha256", return_value=digest), \
                     patch.object(consumer.subprocess, "run") as run, \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    if scenario == "timeout":
                        run.side_effect = subprocess.TimeoutExpired("llmx", 900, output=b"partial", stderr=b"detail")
                    elif scenario == "launch-error":
                        run.side_effect = OSError("cannot launch executable")
                    else:
                        run.return_value = subprocess.CompletedProcess("llmx", 1, b"", b"failed")
                    self.assertEqual(consumer.main(args), 1)
                    if scenario == "wrong-model":
                        run.assert_not_called()
                report = json.loads((root / "result/report.json").read_text())
                self.assertEqual(report["status"], "fail")
                if scenario == "timeout":
                    self.assertEqual((root / "result/version.stdout").read_bytes(), b"partial")
                    self.assertEqual(report["commands"][0]["status"], "timeout")
                elif scenario == "launch-error":
                    self.assertEqual(report["commands"][0]["status"], "launch-failed")


class Qwen35Consumer(unittest.TestCase):
    """tests/baseline_qwen35.py over simulated llmx outputs on the pinned 0.8B Q8_0 file and its model's committed goldens."""

    FILE = "Qwen3.5-0.8B-Q8_0.gguf"

    @classmethod
    def setUpClass(cls):
        cls.docs = baseline_qwen35.goldens_for(cls.FILE)[1]
        cls.tokenizer = baseline_qwen35.load_golden(baseline_qwen35.TOKENIZER_GOLDEN)
        templates = json.loads((baseline_qwen35.DATA / "baseline_chat_template.json").read_text(encoding="utf-8"))["templates"]
        cls.template = next(t["template"] for t in templates if t["sha256"] == cls.docs["baseline_chat.json"]["template_sha256"])
        cls.spec = baseline.pinned_fixture(cls.FILE)

    def double(self, refuse=False):
        """llmx simulated on the pinned file, given a command with the executable first: every text tokenized as the goldens say, and logits and perplexity as HF's, unless `refuse` refuses the architecture."""
        ids = {case["text"]: case["ids"] for case in self.tokenizer["cases"]}
        cases = self.docs["baseline_logits.json"]["cases"] + self.docs["baseline_chat.json"]["cases"]
        ids.update((case["text"], case["token_ids"]) for case in cases)
        by_text = {case["text"]: case for case in cases}
        doc = self.docs[baseline_qwen35.PPL_GOLDENS[512]]

        def flag(command, name):
            return command[command.index(name) + 1] if name in command else None

        def llmx(command, **kwargs):
            rc, out, err = 0, "", ""
            if command[1] == "tokenize":
                out = " ".join(map(str, ids.get(command[3], [0] * doc["n_tokens"])))
            elif command[1] == "logits" and refuse:
                rc, err = 1, "error: inference: unsupported metadata general.architecture\n"
            elif command[1] == "logits":
                text = Path(flag(command, "--file")).read_text(encoding="utf-8") if "--file" in command else command[3]
                out = logits_text(by_text[text])
            elif command[1] == "perplexity":
                window = (int(flag(command, "--ctx-size") or 0), int(flag(command, "--chunks") or 0))
                case = next(case for case in common.ppl_cases(doc) if (case["context_size"], case["max_chunks"]) == window)
                out = ppl_text(case, context=baseline_qwen35.MODEL_CONTEXT, tokens=doc["n_tokens"])
            else:
                out = "llmx 0\n"
            return subprocess.CompletedProcess(command, rc, out.encode("utf-8"), err.encode("utf-8"))

        return llmx

    def simulate(self, root, refuse=False, digest=None, bounds=None, template=None):
        """main() on the pinned file under `root` with llmx simulated by double()."""
        llmx = self.double(refuse)
        doc = self.docs[baseline_qwen35.PPL_GOLDENS[512]]
        gguf = unittest.mock.MagicMock()
        gguf.return_value.value.return_value = self.template if template is None else template
        args = ["--exe", str(root / "llmx"), "--model", str(root / self.FILE), "--output-dir", str(root / "result")]
        # The excerpt's ids are held to a digest the simulation cannot invert, so that one check passes here and is tested on its own.
        with patch.object(baseline_qwen35, "file_sha256", return_value=digest or self.spec["sha256"]), \
             patch.object(baseline_qwen35, "check_ids_digest", return_value={"tokens": doc["n_tokens"]}), \
             patch.object(baseline_qwen35.spec_decode, "GGUF", gguf), patch.object(baseline_qwen35, "BOUNDS", bounds or {}), \
             patch.object(baseline_qwen35.subprocess, "run", side_effect=llmx), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = baseline_qwen35.main(args)
        return code, json.loads((root / "result/report.json").read_text())

    def test_goldens_are_pinned_and_found_by_file(self):
        self.assertEqual(baseline_qwen35.goldens_for("Qwen3.5-0.8B-Q4_K_M.gguf")[0], "qwen35-0.8b")
        self.assertEqual(baseline_qwen35.goldens_for("Qwen3.5-4B-Q4_K_M.gguf")[0], "qwen35-4b")
        with self.assertRaisesRegex(ValueError, "no qwen35 goldens"):
            baseline_qwen35.goldens_for("Qwen3-0.6B-Q8_0.gguf")
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            for name in baseline_qwen35.GOLDEN_SHA256:
                (data / name).parent.mkdir(exist_ok=True)
                (data / name).write_bytes((baseline_qwen35.DATA / name).read_bytes())
            name = "qwen35-0.8b/baseline_chat.json"
            (data / name).write_text((data / name).read_text(encoding="utf-8") + " ", encoding="utf-8")
            with patch.object(baseline_qwen35, "DATA", data), self.assertRaisesRegex(ValueError, "golden changed"):
                baseline_qwen35.goldens_for(self.FILE)

    def test_ids_digest(self):
        ids = [11751, 279, 7172]
        self.assertEqual(baseline_qwen35.check_ids_digest("11751, 279, 7172", 3, baseline_qwen35.ids_sha256(ids)), {"tokens": 3})
        for output in ("11751, 279", "11751, 279, 7171", "279, 11751, 7172"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "differ from HF"):
                baseline_qwen35.check_ids_digest(output, 3, baseline_qwen35.ids_sha256(ids))

    def test_refused_architecture_is_one_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report = self.simulate(Path(directory), refuse=True)
        # The 37 tokenizer cases, the file's chat template, and the ids of both chat renders, of the six prompts and of the perplexity excerpt.
        self.assertEqual((code, report["status"], len(report["checks"])), (0, "skip", 47))
        self.assertTrue(all(check["status"] == "pass" for check in report["checks"]))
        self.assertEqual([record["label"] for record in report["commands"] if "logits" in record["command"]], ["logits-00"])

    def test_measured_run_needs_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report = self.simulate(Path(directory))
        # Beyond the 47, the six prompts and both chat renders ranked, and the whole excerpt and its windows of 512 scored both ways.
        self.assertEqual((code, report["status"], len(report["checks"])), (1, "unbounded", 59))
        self.assertTrue(all(check["status"] == "pass" for check in report["checks"]))
        bounds = {self.FILE: {"top5_overlap": 5, "continuous_nll": 0.01, "window_nll": 0.02}}
        with tempfile.TemporaryDirectory() as directory:
            code, report = self.simulate(Path(directory), bounds=bounds)
        self.assertEqual((code, report["status"], report["bounds"]), (0, "pass", bounds[self.FILE]))

    def test_wrong_file_or_template_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            code, report = self.simulate(Path(directory), digest="0" * 64)
        self.assertEqual((code, report["status"]), (1, "fail"))
        self.assertIn("pinned SHA-256", report["error"])
        with tempfile.TemporaryDirectory() as directory:
            code, report = self.simulate(Path(directory), refuse=True, template=self.template + " ")
        self.assertEqual((code, report["status"]), (1, "fail"))
        self.assertEqual([check["label"] for check in report["checks"] if check["status"] == "fail"], ["chat-template"])

    def test_file_exact_goldens_of_another_file_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("baseline_logits.json", "baseline_perplexity.json"):
                Path(directory, name).write_text(json.dumps({"weights": {"sha256": "0" * 64}}), encoding="utf-8")
            with patch.object(baseline_qwen35, "file_sha256", return_value=self.spec["sha256"]), self.assertRaisesRegex(ValueError, "not made from"):
                baseline_qwen35.model_goldens(str(Path(directory) / self.FILE), directory)

    def hosted(self, root, refuse=False, bounds=None, present=True):
        """run_hosted, the suite's form, with the pinned file the one qwen35 fixture, on disk under `root` when `present`, and llmx simulated by double(): its result and what it printed."""
        llmx = self.double(refuse)

        def run(arguments):
            process = llmx(["llmx"] + arguments)
            return process.returncode, (process.stdout + (process.stderr if process.returncode else b"")).decode("utf-8")

        path = root / self.FILE
        if present:
            path.write_bytes(b"")
        gguf = unittest.mock.MagicMock()
        gguf.return_value.value.return_value = self.template
        printed = io.StringIO()
        with patch.dict(os.environ), patch.object(baseline, "PINNED", [self.spec]), patch.object(baseline, "snapshot_path", return_value=path), \
             patch.object(baseline_qwen35, "file_sha256", return_value=self.spec["sha256"]), \
             patch.object(baseline_qwen35, "check_ids_digest", return_value={"tokens": self.docs[baseline_qwen35.PPL_GOLDENS[512]]["n_tokens"]}), \
             patch.object(baseline_qwen35.spec_decode, "GGUF", gguf), patch.object(baseline_qwen35, "BOUNDS", bounds or {}), \
             patch.object(baseline_qwen35.common, "run", side_effect=run), contextlib.redirect_stdout(printed):
            os.environ.pop("LLMX_BASELINE_GGUF", None)
            ok = baseline_qwen35.run_hosted()
        return ok, printed.getvalue()

    def test_suite_form_skips_only_what_it_cannot_run(self):
        name = "baseline-qwen35[%s]: " % self.FILE
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.hosted(Path(directory), present=False), (True, name + "SKIP - fixture model not on disk\n"))
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.hosted(Path(directory), refuse=True),
                             (True, name + "SKIP - llmx refuses the qwen35 architecture (47 tokenizer and chat id checks pass)\n"))
        # Once llmx runs the model, a file without bounds fails the suite, and one with bounds passes all 59 checks.
        with tempfile.TemporaryDirectory() as directory:
            ok, printed = self.hosted(Path(directory))
        self.assertFalse(ok)
        self.assertTrue(printed.startswith(name + "FAIL - no bounds"), printed)
        bounds = {self.FILE: {"top5_overlap": 5, "continuous_nll": 0.01, "window_nll": 0.02}}
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.hosted(Path(directory), bounds=bounds), (True, name + "59 checks  [ok]\n"))

    def test_required_baseline_counts_qwen35_gate_models(self):
        # A qwen35 file that joins the gate is required as a Qwen3 one is, though the Qwen3 gate's list leaves it out.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / self.FILE
            with patch.object(baseline, "PINNED", [dict(self.spec, gate=True)]), patch.object(baseline, "BASELINE_MODELS", []), \
                 patch.object(baseline, "snapshot_path", return_value=path):
                self.assertEqual(baseline.missing_gate_models(), [self.FILE])
                path.write_bytes(b"")
                self.assertEqual(baseline.missing_gate_models(), [])


class LayeredConsumer(unittest.TestCase):
    """tests/baseline_layered.py: the layered HF reference's goldens, chosen by the model's SHA-256 and checked by the 8B consumer's run."""

    def consume(self, digest, llmx):
        """baseline_layered.main on a model with SHA-256 `digest`, with `llmx` answering its commands: the exit code, the report and what it printed."""
        sha256 = consumer.file_sha256
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ["--exe", str(root / "llmx"), "--model", str(root / "model"), "--output-dir", str(root / "result")]
            printed = io.StringIO()
            with patch.object(consumer, "file_sha256", side_effect=lambda path: sha256(path) if path.name == "excerpt.txt" else digest), \
                 patch.object(consumer.subprocess, "run", side_effect=llmx) as run, \
                 contextlib.redirect_stdout(printed), contextlib.redirect_stderr(printed):
                code = layered.main(args)
            return code, json.loads((root / "result/report.json").read_text()), printed.getvalue(), run.call_count

    def test_goldens_record_the_gguf_they_are_chosen_for(self):
        for digest, golden in layered.GOLDENS.items():
            with self.subTest(golden=golden["name"]):
                docs = consumer.load_goldens(golden["data"], golden["fixture_sha256"])
                self.assertEqual({doc["gguf_sha256"] for doc in docs.values()}, {digest})
                self.assertEqual(len({doc["gguf_file"] for doc in docs.values()}), 1)
                # One checkpoint gives all three: the tokenizer's repository and commit are the model's.
                ran = {(doc["reference_repo"], doc["reference_revision"]) for name, doc in docs.items() if name != "baseline_tokenizer.json"}
                self.assertEqual(ran, {(docs["baseline_tokenizer.json"]["tokenizer_repo"], docs["baseline_tokenizer.json"]["tokenizer_revision"])})
                # The GGUF was converted from these weights: every F32 tensor equals the checkpoint's but for ssm_a values one float32 step from the exp here, which the consumer's provenance line counts.
                record = docs["baseline_logits.json"]["layered"]["gguf_provenance"]
                self.assertEqual(docs["baseline_perplexity.json"]["layered"]["gguf_provenance"], record)
                self.assertEqual(record["f32_tensors_compared"], record["equal"] + len(record["differ"]))
                self.assertNotIn("f32_not_compared", record)
                for item in record["differ"]:
                    self.assertRegex(item["tensor"], r"^blk\.\d+\.ssm_a$")
                    self.assertTrue(0 < item["values"] <= item["of"] and item["max_float32_steps"] == 1)
                counts = "%d F32 tensors equal" % record["equal"] if not record["differ"] else "%d of its %d F32 tensors equal" % (record["equal"], record["f32_tensors_compared"])
                self.assertIn(counts, golden["provenance_limit"])
                if record["differ"]:
                    self.assertIn("%d ssm_a tensors differ in %d values" % (len(record["differ"]), sum(item["values"] for item in record["differ"])), golden["provenance_limit"])

    def test_passing_run_for_each_model(self):
        for digest, golden in layered.GOLDENS.items():
            with self.subTest(golden=golden["name"]):
                docs = consumer.load_goldens(golden["data"], golden["fixture_sha256"])
                code, report, printed, _ = self.consume(digest, simulated_llmx(docs, docs["baseline_perplexity.json"]["n_tokens"], layered.MODEL_CONTEXT))
                self.assertEqual((code, report["status"], report["golden"]), (0, "pass", golden["name"]))
                self.assertEqual(len(report["checks"]), 41)
                self.assertIn(golden["name"] + " HF check PASS", printed)

    def test_refusal_of_the_architecture_is_one_skip_line(self):
        refusal = "error: inference: unsupported metadata general.architecture\n"
        digest, golden = next(iter(layered.GOLDENS.items()))
        docs = consumer.load_goldens(golden["data"], golden["fixture_sha256"])
        code, report, printed, _ = self.consume(digest, simulated_llmx(docs, docs["baseline_perplexity.json"]["n_tokens"], layered.MODEL_CONTEXT, refuse=refusal))
        self.assertEqual((code, report["status"]), (0, "skip"))
        self.assertEqual([line for line in printed.splitlines() if "HF check" in line],
                         [golden["name"] + " HF check SKIP: llmx does not run this model yet (" + refusal.strip() + ")"])
        # The tokenizer cases ran before the first logits command was refused, and passed.
        self.assertEqual(len(report["checks"]), 21)
        self.assertTrue(all(item["status"] == "pass" for item in report["checks"]))
        # A check that failed before the refusal fails the run, and any other error is a failure, not a skip.
        answer = simulated_llmx(docs, docs["baseline_perplexity.json"]["n_tokens"], layered.MODEL_CONTEXT, refuse=refusal)
        first = docs["baseline_tokenizer.json"]["cases"][0]["text"]
        wrong = lambda command, **kwargs: (subprocess.CompletedProcess(command, 0, b"1 2 3", b"")
                                           if command[1] == "tokenize" and command[3] == first else answer(command, **kwargs))
        for llmx in (wrong, simulated_llmx(docs, 0, 0, refuse="error: vulkan: this model's mixer has no kernel\n")):
            code, report, _, _ = self.consume(digest, llmx)
            self.assertEqual((code, report["status"]), (1, "fail"))

    def test_a_model_without_goldens_fails_before_running(self):
        code, report, printed, calls = self.consume("0" * 64, lambda command, **kwargs: None)
        self.assertEqual((code, report["status"], calls), (1, "fail", 0))
        self.assertIn("no goldens here", report["error"])


def run():
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in (ReferenceConsumer, Qwen35Consumer, LayeredConsumer))
    result = unittest.TextTestRunner(stream=sys.stdout).run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
