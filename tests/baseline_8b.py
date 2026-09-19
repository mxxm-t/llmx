import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


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


def require(condition, message):
    if not condition:
        raise ValueError(message)


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


def check_ids(output, expected):
    ids = [int(value) for value in output.strip().replace(",", " ").split()]
    require(ids == expected, "token IDs differ from HF")
    return {"tokens": len(ids)}


def check_logits(output, case):
    lines = output.strip().splitlines()
    require(len(lines) == 11 and lines[0] == "tokens: " + str(case["n_tokens"]),
            "wrong logit count or prompt token count")
    pairs = [line.split() for line in lines[1:]]
    require(all(len(pair) == 2 for pair in pairs), "malformed logits")
    ids = [int(pair[0]) for pair in pairs]
    values = [float(pair[1]) for pair in pairs]
    require(len(set(ids)) == 10 and all(0 <= token < VOCAB_SIZE for token in ids),
            "duplicate or invalid logit token IDs")
    require(all(math.isfinite(value) and abs(value) <= BOUNDS["max_abs_logit"] for value in values),
            "non-finite or implausible logits")
    require(all(a >= b for a, b in zip(values, values[1:])), "logits not sorted")
    require(ids[0] == case["top_ids"][0], "top-1 differs from HF")
    overlap = len(set(ids[:5]) & set(case["top_ids"][:5]))
    require(overlap >= BOUNDS["top5_overlap"], "top-5 overlap below frozen bound")
    return {"top1": ids[0], "top5_overlap": overlap, "top_ids": ids, "top_logits": values}


def check_ppl(output, case, total_tokens):
    fields = {}
    for line in output.strip().splitlines():
        require(":" in line, "malformed PPL line")
        key, value = line.split(":", 1)
        require(key not in fields, "duplicate PPL field: " + key)
        fields[key] = value.strip()
    counts = {"tokens": total_tokens, "used tokens": case["used_tokens"],
              "scored tokens": case["n_scored"], "chunks": case["chunks"],
              "context size": case["context_size"] or MODEL_CONTEXT}
    require(set(fields) == set(counts) | {"mean NLL", "perplexity"}, "missing or unexpected PPL fields")
    for key, expected in counts.items():
        require(int(fields[key]) == expected, "wrong PPL " + key)
    nll, ppl = float(fields["mean NLL"]), float(fields["perplexity"])
    require(math.isfinite(nll) and math.isfinite(ppl) and nll >= 0 and ppl >= 1,
            "invalid NLL/PPL")
    bound = BOUNDS["window_nll"] if case["context_size"] else BOUNDS["continuous_nll"]
    delta = abs(nll - case["mean_nll"])
    require(delta <= bound, "NLL difference %.9g exceeds frozen bound %.9g" % (delta, bound))
    # The CLI prints six significant digits.
    require(math.isclose(ppl, math.exp(nll), rel_tol=2e-5), "inconsistent NLL/PPL")
    return dict(counts, mean_nll=nll, perplexity=ppl, hf_mean_nll=case["mean_nll"],
                absolute_nll_delta=delta, bound=bound)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Optional pinned Qwen3-8B Q8_0 HF check; no downloads or skips.")
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new directory for raw output and report")
    args = parser.parse_args(argv)
    exe, model, out = args.exe.resolve(), args.model.resolve(), args.output_dir.resolve()
    if out.exists():
        parser.error("output directory already exists; use a new path")
    out.mkdir(parents=True)
    report = {"status": "running", "bounds": dict(BOUNDS), "threads": 6, "ubatch": 128,
              "model": str(model), "executable": str(exe), "fixture_sha256_lf": FIXTURE_SHA256,
              "scope": "20 tokenizer cases, six short prefill rankings and four serial-step NLL cases; not full-corpus or deep-context coverage",
              "provenance_limit": "Official GGUF base model and file digest match; exact original conversion revision is undocumented.",
              "commands": [], "checks": []}

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="ascii")

    def run(label, arguments):
        command = [str(exe)] + arguments
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
        continuous = dict(doc, context_size=0, max_chunks=0, chunks=1, used_tokens=doc["n_tokens"])
        for index, case in enumerate([continuous] + doc["chunk_cases"]):
            label = "ppl-%02d" % index
            command = ["perplexity", str(model), "--file", str(excerpt), "--threads", "6", "--ubatch", "128"]
            if case["context_size"]:
                command += ["--ctx-size", str(case["context_size"])]
            if case["max_chunks"]:
                command += ["--chunks", str(case["max_chunks"])]
            check(label, check_ppl, run(label, command), case, doc["n_tokens"])
        failures = [item["label"] for item in report["checks"] if item["status"] == "fail"]
        require(not failures, "failed checks: " + ", ".join(failures))
        report["status"] = "pass"
        print("8B HF check PASS: 20 tokenizer cases, six rankings, four NLL cases", flush=True)
    except (OSError, ValueError, OverflowError) as error:
        report.update(status="fail", error=str(error))
        print("8B HF check FAIL: " + str(error), file=sys.stderr, flush=True)
    finally:
        save()
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
