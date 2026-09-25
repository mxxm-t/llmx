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


def load_goldens():
    docs = {}
    for name, expected in FIXTURE_SHA256.items():
        # Git may check JSON out with CRLF; the generator writes LF.
        text = (DATA / name).read_text(encoding="utf-8")
        require(hashlib.sha256(text.encode("utf-8")).hexdigest() == expected,
                "HF fixture changed: " + name)
        docs[name] = json.loads(text)
    return docs


# The shared validators at this model's vocabulary, context and bounds, as tests/reference_consumer.py checks them.
check_ids = common.check_ids
check_logits = functools.partial(common.check_logits, vocab=VOCAB_SIZE, bounds=BOUNDS)
check_ppl = functools.partial(common.check_ppl, context=MODEL_CONTEXT, bounds=BOUNDS)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Optional pinned Qwen3-8B Q8_0 HF check; no downloads or skips.")
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
    report = {"status": "running", "bounds": dict(BOUNDS), "threads": 6, "ubatch": 128,
              "model": str(model), "executable": str(exe), "device": args.device or "cpu", "layer_shares": args.layer_shares,
              "fixture_sha256_lf": FIXTURE_SHA256,
              "scope": "20 tokenizer cases, six short prefill rankings and four NLL cases, each scored in batched passes and per token; not full-corpus or deep-context coverage",
              "provenance_limit": "Official GGUF base model and file digest match; exact original conversion revision is undocumented.",
              "commands": [], "checks": []}

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
        require(record.get("returncode") == 0, label + " failed: " + record["status"])
        return stdout.decode("utf-8")

    def check(label, validator, *arguments):
        try:
            result = dict(status="pass", **validator(*arguments))
        except (ValueError, OverflowError) as error:
            result = dict(status="fail", error=str(error))
        report["checks"].append(dict(label=label, **result))
        save()

    save()
    try:
        docs = load_goldens()
        report["executable_sha256"] = file_sha256(exe)
        print("Verifying 8B model SHA-256", flush=True)
        report["model_sha256"] = file_sha256(model)
        require(report["model_sha256"] == MODEL_SHA256, "wrong 8B GGUF digest")
        report["version"] = run("version", ["--version"]).strip()
        for index, case in enumerate(docs["baseline_tokenizer.json"]["cases"]):
            label = "tokenizer-%02d" % index
            check(label, check_ids, run(label, ["tokenize", str(model), case["text"]]), case["ids"])
        for index, case in enumerate(docs["baseline_logits.json"]["cases"]):
            label = "logits-%02d" % index
            check(label + "-ids", check_ids, run(label + "-ids", ["tokenize", str(model), case["text"]]), case["token_ids"])
            check(label, check_logits, run(label, ["logits", str(model), case["text"],
                  "--top", "10", "--threads", "6", "--ubatch", "128"]), case)
        doc = docs["baseline_perplexity.json"]
        excerpt = out / "excerpt.txt"
        excerpt.write_bytes(doc["text"].encode("utf-8"))
        require(file_sha256(excerpt) == doc["text_sha256"], "PPL text hash differs")
        check("ppl-ids", check_ids, run("ppl-ids", ["tokenize", str(model), doc["text"]]), doc["token_ids"])
        for index, case in enumerate(common.ppl_cases(doc)):
            for mode in common.PPL_MODES:
                label = "ppl-%02d" % index + ("-per-token" if mode == "per-token" else "")
                command = common.ppl_command(str(model), str(excerpt), case, mode, ubatch=128)
                check(label, check_ppl, run(label, command), case, doc["n_tokens"])
        failures = [item["label"] for item in report["checks"] if item["status"] == "fail"]
        require(not failures, "failed checks: " + ", ".join(failures))
        report["status"] = "pass"
        print("8B HF check PASS: %d checks, 20 tokenizer cases, six rankings, four NLL cases batched and per token"
              % len(report["checks"]), flush=True)
    except (OSError, ValueError, OverflowError) as error:
        report.update(status="fail", error=str(error))
        print("8B HF check FAIL: " + str(error), file=sys.stderr, flush=True)
    finally:
        save()
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
