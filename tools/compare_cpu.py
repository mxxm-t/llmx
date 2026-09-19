"""Compare the two builds of compare_cpu.cpp on the pinned HF excerpt."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llmx", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--reference-revision", required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    arms = {"llmx": args.llmx.resolve(), "reference": args.reference.resolve()}
    model = args.model.resolve()
    fixture_path = Path(__file__).resolve().parents[1] / "tests/data/baseline_perplexity.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    ids = fixture["token_ids"]
    if len(ids) != 247:
        raise ValueError("benchmark requires the pinned 247-token HF excerpt")
    artifacts = {str(path): digest(path) for path in (*arms.values(), model, fixture_path)}
    if os.name == "nt":
        for name in ("llama.dll", "ggml.dll", "ggml-base.dll", "ggml-cpu.dll"):
            path = arms["reference"].parent / name
            artifacts[str(path)] = digest(path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tokens = output / "tokens.txt"
    tokens.write_text(" ".join(map(str, ids)), encoding="ascii")
    metadata = {
        "sha256": artifacts,
        "reference_revision": args.reference_revision,
        "threads": 6, "ubatch": 128, "kv_type": "f32",
        "prompt_tokens": 215, "forced_decode_tokens": 32,
        "warmup_sequences_per_process": 1,
        "rounds": args.rounds,
        "timing": "model inference only; no load, tokenization or sampling",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows = []
    for pair in range(args.rounds):
        order = ("llmx", "reference") if pair % 2 == 0 else ("reference", "llmx")
        for arm in order:
            command = [str(arms[arm]), str(model), str(tokens)]
            proc = subprocess.run(command, capture_output=True, timeout=300)
            prefix = output / ("%02d-%s" % (pair, arm))
            prefix.with_suffix(".stdout").write_bytes(proc.stdout)
            prefix.with_suffix(".stderr").write_bytes(proc.stderr)
            proc.check_returncode()
            samples = [json.loads(line) for line in proc.stdout.decode("utf-8").splitlines() if line]
            if len(samples) != 2 or [s["run"] for s in samples] != [0, 1]:
                raise ValueError("missing warmup or measured sequence: " + str(prefix))
            for sample in samples:
                if sample["pp_tokens"] != 215 or sample["tg_tokens"] != 32:
                    raise ValueError("mismatched workload: " + str(prefix))
                if not all(math.isfinite(sample[k]) and sample[k] > 0 for k in ("pp_ms", "tg_ms")):
                    raise ValueError("invalid timings: " + str(prefix))
                if not math.isfinite(sample["logit_sum"]):
                    raise ValueError("non-finite output: " + str(prefix))
            row = dict(pair=pair, arm=arm, **samples[1])
            rows.append(row)
            (output / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
    summary = {}
    for arm in arms:
        summary[arm] = {}
        for phase in ("pp", "tg"):
            rates = [r[phase + "_tokens"] * 1000 / r[phase + "_ms"] for r in rows if r["arm"] == arm]
            summary[arm][phase] = dict(mean=statistics.mean(rates), median=statistics.median(rates),
                                       minimum=min(rates), maximum=max(rates))
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
