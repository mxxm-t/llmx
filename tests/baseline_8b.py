import argparse
import functools
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys

import common
from common import device_args, require


DATA = Path(__file__).resolve().parent / "data" / "qwen3-8b"
MODEL_SHA256 = "408b955510e196121c1c375201744783b5c9a43c7956d73fc78df54c66e883d6"
FIXTURE_SHA256 = {
    "baseline_tokenizer.json": "9f8cdedcd91e89e388ac6081c6e496ea7e2f95d2565664cc5117144e29b1fe2a",
    "baseline_logits.json": "e3eb726e06ff3bd9d5ecd69712c324a9842ccd58ec33fda4bb19dc4476f9680e",
    "baseline_perplexity.json": "c6e84d0d0661e07b285c7882ac0dd9d128dd45ad7215c2d83a345f3e84f71178",
}
BOUNDS = {"top5_overlap": 5, "max_abs_logit": 100.0,
          "continuous_nll": 0.01, "window_nll": 0.02}
VOCAB_SIZE = 151936
MODEL_CONTEXT = 40960


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_goldens(data=None, fixture_sha256=None):
    """The goldens in `data`, each refused unless its text has the SHA-256 `fixture_sha256` gives it; the 8B's when neither is given."""
    data = DATA if data is None else data
    docs = {}
    for name, expected in (FIXTURE_SHA256 if fixture_sha256 is None else fixture_sha256).items():
        # Git may check JSON out with CRLF; the generator writes LF.
        text = (data / name).read_text(encoding="utf-8")
        require(hashlib.sha256(text.encode("utf-8")).hexdigest() == expected,
                "HF fixture changed: " + name)
        docs[name] = json.loads(text)
    return docs


# The shared validators at this model's vocabulary, context and bounds, as tests/reference_consumer.py checks them.
check_ids = common.check_ids
check_logits = functools.partial(common.check_logits, vocab=VOCAB_SIZE, bounds=BOUNDS)
check_ppl = functools.partial(common.check_ppl, context=MODEL_CONTEXT, bounds=BOUNDS)


def qwen3_8b():
    """The 8B's goldens as consume() takes a model's goldens, read from this module's names when a run starts."""
    return {"name": "8B", "data": DATA, "fixture_sha256": FIXTURE_SHA256, "bounds": BOUNDS,
            "vocab": VOCAB_SIZE, "context": MODEL_CONTEXT,
            "scope": "20 tokenizer cases, six short prefill rankings and four NLL cases, each scored in batched passes and per token; not full-corpus or deep-context coverage",
            "provenance_limit": "Official GGUF base model and file digest match; exact original conversion revision is undocumented."}


class Refused(Exception):
    """llmx refused the model with the error that a consumer reads as llmx not running the model yet."""


def main(argv=None):
    return consume(argv, "Optional pinned Qwen3-8B Q8_0 HF check; no downloads or skips.",
                   lambda digest: qwen3_8b() if digest == MODEL_SHA256 else None)


def consume(argv, description, select, refusals=()):
    """Hold llmx on a model to the HF goldens that `select` gives for its SHA-256, and write report.json and each command's output into a new directory.
    A model `select` gives no goldens for fails, as do a golden whose text changed, a failed command and a check out of bounds.
    Goldens that name file-exact goldens (`file_exact`: their data, SHA-256 and bounds) run those first, their labels starting with "file-exact-", and the model's goldens only if they pass.
    A command that fails with one of the texts of `refusals` in its error ends the run as a skip, exit 0, unless a check had already failed."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for raw output and report")
    parser.add_argument("--device", help="run the commands that take --device on this device or comma-separated list, e.g. vulkan:0,vulkan:1")
    parser.add_argument("--layer-shares", help="with several devices in --device, their proportions of the layers, e.g. 1,1")
    args = parser.parse_args(argv)
    # The suite's own configuration, reaching the binary only as flags through device_args.
    if args.device:
        os.environ["LLMX_DEVICE"] = args.device
    if args.layer_shares:
        os.environ["LLMX_LAYER_SHARES"] = args.layer_shares
    exe, model, out = args.exe.resolve(), args.model.resolve(), args.output_dir.resolve()
    if out.exists():
        parser.error("output directory already exists; use a new path")
    out.mkdir(parents=True)
    report = {"status": "running", "threads": 6, "ubatch": 128,
              "model": str(model), "executable": str(exe), "device": args.device or "cpu", "layer_shares": args.layer_shares,
              "commands": [], "checks": []}
    name = None

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="ascii")

    def run(label, arguments):
        command = [str(exe)] + device_args(arguments)
        record = {"label": label, "command": command, "status": "running"}
        report["commands"].append(record)
        save()
        print(label, flush=True)
        try:
            result = subprocess.run(command, capture_output=True, timeout=900)
            stdout, stderr = result.stdout, result.stderr
            record.update(status="exited", returncode=result.returncode)
        except subprocess.TimeoutExpired as error:
            stdout, stderr = error.stdout or b"", error.stderr or b""
            record.update(status="timeout", timeout_seconds=900)
        except OSError as error:
            record.update(status="launch-failed", error=str(error))
            save()
            raise
        (out / (label + ".stdout")).write_bytes(stdout)
        (out / (label + ".stderr")).write_bytes(stderr)
        save()
        error = stderr.decode("utf-8", "replace").strip()
        if record.get("returncode") not in (None, 0) and any(text in error for text in refusals):
            raise Refused(error)
        require(record.get("returncode") == 0, label + " failed: " + record["status"])
        return stdout.decode("utf-8")

    def check(label, validator, *arguments):
        try:
            result = dict(status="pass", **validator(*arguments))
        except (ValueError, OverflowError) as error:
            result = dict(status="fail", error=str(error))
        report["checks"].append(dict(label=label, **result))
        save()

    def failures():
        return [item["label"] for item in report["checks"] if item["status"] == "fail"]

    def title():
        return name + " HF check" if name else "HF check"

    save()
    try:
        report["executable_sha256"] = file_sha256(exe)
        print("Verifying the model's SHA-256", flush=True)
        report["model_sha256"] = file_sha256(model)
        golden = select(report["model_sha256"])
        require(golden is not None, "no goldens here for a GGUF with SHA-256 " + report["model_sha256"])
        name = golden["name"]
        report.update(golden=name, bounds=dict(golden["bounds"]), fixture_sha256_lf=golden["fixture_sha256"],
                      scope=golden["scope"], provenance_limit=golden["provenance_limit"])
        docs = load_goldens(golden["data"], golden["fixture_sha256"])
        report["version"] = run("version", ["--version"]).strip()
        for index, case in enumerate(docs["baseline_tokenizer.json"]["cases"]):
            label = "tokenizer-%02d" % index
            check(label, check_ids, run(label, ["tokenize", str(model), case["text"]]), case["ids"])

        def rankings_and_nll(docs, bounds, prefix=""):
            """The six rankings and four NLL cases of `docs` at `bounds`, each check's label starting with `prefix`.
            Every top-1 must be HF's, unless `bounds` names how many the file keeps (`top1_matches`), which one more check counts over them all."""
            needed = bounds.get("top1_matches")
            check_model_logits = functools.partial(common.check_logits, vocab=golden["vocab"], bounds=bounds, require_top1=needed is None)
            check_model_ppl = functools.partial(common.check_ppl, context=golden["context"], bounds=bounds)
            cases = docs["baseline_logits.json"]["cases"]
            matched = 0
            for index, case in enumerate(cases):
                label = prefix + "logits-%02d" % index
                check(label + "-ids", check_ids, run(label + "-ids", ["tokenize", str(model), case["text"]]), case["token_ids"])
                printed = run(label, ["logits", str(model), case["text"], "--top", "10", "--threads", "6", "--ubatch", "128"])
                check(label, check_model_logits, printed, case)
                matched += common.first_id(printed) == case["top_ids"][0]
            if needed is not None:
                check(prefix + "top1", common.check_top1, matched, len(cases), needed)
            doc = docs["baseline_perplexity.json"]
            excerpt = out / "excerpt.txt"
            excerpt.write_bytes(doc["text"].encode("utf-8"))
            require(file_sha256(excerpt) == doc["text_sha256"], "PPL text hash differs")
            check(prefix + "ppl-ids", check_ids, run(prefix + "ppl-ids", ["tokenize", str(model), doc["text"]]), doc["token_ids"])
            for index, case in enumerate(common.ppl_cases(doc)):
                for mode in common.PPL_MODES:
                    label = prefix + "ppl-%02d" % index + ("-per-token" if mode == "per-token" else "")
                    command = common.ppl_command(str(model), str(excerpt), case, mode, ubatch=128)
                    check(label, check_model_ppl, run(label, command), case, doc["n_tokens"])

        # A file with file-exact goldens is held first to them, HF run on its own weights, at their bounds; only if those pass is it held to its model's goldens.
        exact = golden.get("file_exact")
        if exact:
            report["file_exact_bounds"] = dict(exact["bounds"])
            exact_docs = load_goldens(exact["data"], exact["fixture_sha256"])
            require(all(doc["weights"]["sha256"] == report["model_sha256"] for doc in exact_docs.values()),
                    "the file-exact goldens in %s were not made from this file" % exact["data"])
            rankings_and_nll(exact_docs, exact["bounds"], "file-exact-")
            require(not failures(), "failed file-exact checks: " + ", ".join(failures()))
        rankings_and_nll(docs, golden["bounds"])
        require(not failures(), "failed checks: " + ", ".join(failures()))
        report["status"] = "pass"
        print("%s PASS: %d checks, 20 tokenizer cases, %ssix rankings and four NLL cases batched and per token"
              % (title(), len(report["checks"]), "the file-exact goldens' and the model's " if exact else ""), flush=True)
    except Refused as refused:
        if failures():
            report.update(status="fail", error="failed checks: %s, then llmx refused the model: %s" % (", ".join(failures()), refused))
            print("%s FAIL: %s" % (title(), report["error"]), file=sys.stderr, flush=True)
        else:
            report.update(status="skip", reason=str(refused))
            print("%s SKIP: llmx does not run this model yet (%s)" % (title(), refused), flush=True)
    except (OSError, ValueError, OverflowError) as error:
        report.update(status="fail", error=str(error))
        print("%s FAIL: %s" % (title(), error), file=sys.stderr, flush=True)
    finally:
        save()
    return 0 if report["status"] in ("pass", "skip") else 1


if __name__ == "__main__":
    sys.exit(main())
