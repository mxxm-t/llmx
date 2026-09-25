import contextlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import baseline_8b as consumer
import common


def logits_text(case):
    return "tokens: %d\n" % case["n_tokens"] + "".join(
        "%d %.6f\n" % pair for pair in zip(case["top_ids"], case["top_logits"]))


def ppl_text(case, nll=None):
    nll = case["mean_nll"] if nll is None else nll
    return ("tokens: 247\nused tokens: %d\nscored tokens: %d\nchunks: %d\n"
            "context size: %d\nmean NLL: %.6g\nperplexity: %.6g\n") % (
        case["used_tokens"], case["n_scored"], case["chunks"],
        case["context_size"] or consumer.MODEL_CONTEXT, nll, math.exp(nll))


class ReferenceConsumer(unittest.TestCase):
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
        doc = self.docs["baseline_perplexity.json"]
        ids = {case["text"]: case["ids"] for case in self.docs["baseline_tokenizer.json"]["cases"]}
        ids.update((case["text"], case["token_ids"]) for case in self.docs["baseline_logits.json"]["cases"])
        ids[doc["text"]] = doc["token_ids"]
        logits = {case["text"]: logits_text(case) for case in self.docs["baseline_logits.json"]["cases"]}

        def flag(command, name):
            return int(command[command.index(name) + 1]) if name in command else 0

        def llmx(command, **kwargs):
            if command[1] == "tokenize":
                out = " ".join(map(str, ids[command[3]]))
            elif command[1] == "logits":
                out = logits[command[3]]
            elif command[1] == "perplexity":
                window = (flag(command, "--ctx-size"), flag(command, "--chunks"))
                out = ppl_text(next(case for case in common.ppl_cases(doc)
                                    if (case["context_size"], case["max_chunks"]) == window))
            else:
                out = "llmx 0\n"
            return subprocess.CompletedProcess(command, 0, out.encode("utf-8"), b"")

        sha256 = consumer.file_sha256
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = ["--exe", str(root / "llmx"), "--model", str(root / "model"), "--output-dir", str(root / "result")]
            with patch.object(consumer, "file_sha256",
                              side_effect=lambda path: sha256(path) if path.name == "excerpt.txt" else consumer.MODEL_SHA256), \
                 patch.object(consumer.subprocess, "run", side_effect=llmx), contextlib.redirect_stdout(io.StringIO()):
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


def run():
    result = unittest.TextTestRunner(stream=sys.stdout).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceConsumer))
    return result.wasSuccessful()


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
