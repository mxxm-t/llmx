import glob
import hashlib
import io
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import run as cli

# Correctness baseline: llmx against golden fixtures generated once from the HF reference (tools/gen_baseline.py); with f32.py it is the suite's external ground truth.
# It needs a real model, so it skips when none is present; point it at one with LLMX_BASELINE_GGUF, or let it find the fixture model in the HF cache.

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "data", "baseline_tokenizer.json")


GOLDEN_LOGITS = os.path.join(HERE, "data", "baseline_logits.json")
GOLDEN_PPL = os.path.join(HERE, "data", "baseline_perplexity.json")

# Fixture models for the logit/PPL gates, each with the top-5 overlap it reaches against the full-precision reference, measured per model since coarser quantization reorders more of the tail.
# The Q4_0 entry covers subnormal f16 scales: its token_embd is Q6_K with a subnormal super-block scale, which the Q8_0 fixture almost never has.
BASELINE_MODELS = [
    {"repo": "Qwen/Qwen3-0.6B-GGUF", "file": "Qwen3-0.6B-Q8_0.gguf",
     "revision": "23749fefcc72300e3a2ad315e1317431b06b590a",
     "sha256": "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
     "min_overlap": 5, "max_nll_delta": 0.01, "max_chunk_nll_delta": 0.02},
    {"repo": "unsloth/Qwen3-0.6B-GGUF", "file": "Qwen3-0.6B-Q4_0.gguf",
     "revision": "50968a4468ef4233ed78cd7c3de230dd1d61a56b",
     "sha256": "33bcc57074ec7b6eada5a90651ee546ec0c2b271002c22baf9f1b2dd1e8f75cb",
     "min_overlap": 4, "max_nll_delta": 0.16, "max_chunk_nll_delta": 0.20},
    # 168 Q5_K, 29 Q6_K and 113 F32 tensors: the K-quant path in every matmul and the Q6_K head, on both backends.
    # Same repo and revision as the Q4_0 file, so no third download source.
    {"repo": "unsloth/Qwen3-0.6B-GGUF", "file": "Qwen3-0.6B-Q5_K_M.gguf",
     "revision": "50968a4468ef4233ed78cd7c3de230dd1d61a56b",
     "sha256": "03c6e2127d155b89c21a512954010486b1e00e1a9eebdfad650d03b53ab4c74a",
     "min_overlap": 4, "max_nll_delta": 0.05, "max_chunk_nll_delta": 0.16},
]

# A correct next-token logit for these models sits around 15-25.
# Gross corruption blows this up (the reintroduced f16 bug gave 582), so a magnitude bound catches whole classes of damage that a ranking check can miss.
MAX_PLAUSIBLE_LOGIT = 100.0


def run_logits():
    """Compare llmx's next-token ranking against a FULL-PRECISION reference.

    llmx runs a quantized GGUF while the golden comes from the fp32 model, so
    logit VALUES differ by quantization error and comparing them directly is
    meaningless. What is stable is the ranking, plus a magnitude sanity bound:

      - top-1 must match, on every model.
      - top-5 SET overlap must reach the per-model bound above; a swap at the
        5th place counts as agreement when the reference puts both tokens
        within 0.1 logits of its 5th value (common.top5_overlap).
      - exact top-5 ORDER is deliberately NOT required: it legitimately differs
        when two tokens sit within about 0.01 logits of each other, far below
        quantization noise. Requiring it would flag correct behaviour.
    """
    with io.open(GOLDEN_LOGITS, encoding="utf-8") as f:
        doc = json.load(f)

    ran = 0
    for spec in BASELINE_MODELS:
        model = find_fixture(spec)
        if not model:
            continue
        ran += 1
        failures, ordered = [], 0
        skipped = False
        for case in doc["cases"]:
            rc, out = cli(["logits", model, case["text"], "--top", "10"])
            if common.device_lacks_kernel(rc, out):
                print("baseline-logits[%s]: SKIP - %s has no kernel for this model's matrices"
                      % (spec["file"], os.environ["LLMX_DEVICE"]))
                skipped = True
                break
            if rc != 0:
                failures.append((case["text"], "exit %d" % rc))
                continue
            ids, vals = common.parse_logits(out)
            want = case["top_ids"]
            if not ids:
                failures.append((case["text"], "no logits parsed"))
            elif abs(vals[0]) > MAX_PLAUSIBLE_LOGIT:
                failures.append((case["text"], "top logit %.1f is implausible" % vals[0]))
            elif ids[0] != want[0]:
                failures.append((case["text"], "top-1 %d, reference %d" % (ids[0], want[0])))
            else:
                ov = common.top5_overlap(ids, want, case["top_logits"])
                if ov < spec["min_overlap"]:
                    failures.append((case["text"], "top-5 overlap %d/5, expected >= %d"
                                     % (ov, spec["min_overlap"])))
                elif ids[:5] == want[:5]:
                    ordered += 1

        if skipped:
            continue
        n = len(doc["cases"])
        if failures:
            print("baseline-logits[%s]: %d/%d prompts agree, %d differ:"
                  % (spec["file"], n - len(failures), n, len(failures)))
            for text, why in failures:
                print("    %s" % text.encode("unicode_escape").decode("ascii")[:60])
                print("      %s" % why)
            return False
        print("baseline-logits[%s]: top-1 %d/%d, top-5 set >=%d %d/%d, exact order %d/%d  [ok]"
              % (spec["file"], n, n, spec["min_overlap"], n, n, ordered, n))

    if ran == 0:
        print("baseline-logits: SKIP - no fixture model on disk")
    return True


def run_perplexity():
    with io.open(GOLDEN_PPL, encoding="utf-8") as f:
        doc = json.load(f)
    raw = doc["text"].encode("utf-8")
    assert hashlib.sha256(raw).hexdigest() == doc["text_sha256"], "PPL fixture text changed"
    assert len(doc["token_ids"]) == doc["n_tokens"] == doc["n_scored"] + 1
    assert math.isfinite(doc["mean_nll"]) and math.isfinite(doc["perplexity"])
    assert math.isclose(math.exp(doc["mean_nll"]), doc["perplexity"], rel_tol=1e-12)
    with tempfile.TemporaryDirectory(prefix="llmx_ppl_") as directory:
        path = os.path.join(directory, "excerpt.txt")
        with open(path, "wb") as f:
            f.write(raw)
        for spec in BASELINE_MODELS:
            model = find_fixture(spec)
            if not model:
                print("baseline-ppl[%s]: SKIP - fixture model not on disk" % spec["file"])
                continue
            rc, out = cli(["tokenize", model, doc["text"]])
            assert rc == 0 and parse_ids(out) == doc["token_ids"], "PPL token IDs differ from HF"
            continuous = dict(doc, context_size=0, max_chunks=0, chunks=1, used_tokens=doc["n_tokens"])
            # Every case both ways: in batched passes, the prompt path, and one token at a time, the decode path. On a device they are different kernels, and scoring only one way once left the prompt kernels without an HF check at all.
            cases = [(case, mode) for case in [continuous] + doc["chunk_cases"] for mode in ("batched", "per-token")]
            for case, mode in cases:
                args = ["perplexity", model, "--file", path, "--threads", "6"]
                if mode == "per-token":
                    args.append("--per-token")
                if case["context_size"]:
                    args += ["--ctx-size", str(case["context_size"])]
                if case["max_chunks"]:
                    args += ["--chunks", str(case["max_chunks"])]
                rc, out = cli(args)
                if common.device_lacks_kernel(rc, out):
                    print("baseline-ppl[%s]: SKIP - %s has no kernel for this model's matrices"
                          % (spec["file"], os.environ["LLMX_DEVICE"]))
                    break
                assert rc == 0, "perplexity failed (exit %d): %s" % (rc, out)
                fields = common.perplexity_fields(out)
                required = {"tokens", "used tokens", "scored tokens", "chunks", "context size", "mean NLL", "perplexity"}
                assert required <= fields.keys(), "missing PPL results: " + out
                assert int(fields["tokens"]) == doc["n_tokens"], "PPL token count differs from HF"
                assert int(fields["used tokens"]) == case["used_tokens"], "PPL used tokens differ from HF"
                assert int(fields["scored tokens"]) == case["n_scored"], "PPL scored tokens differ from HF"
                assert int(fields["chunks"]) == case["chunks"], "PPL chunk count differs from HF"
                if case["context_size"]:
                    assert int(fields["context size"]) == case["context_size"], "PPL context flag was ignored"
                nll, ppl = float(fields["mean NLL"]), float(fields["perplexity"])
                assert math.isfinite(nll) and math.isfinite(ppl), "non-finite PPL results: " + out
                delta = abs(nll - case["mean_nll"])
                bound = spec["max_chunk_nll_delta"] if case["context_size"] else spec["max_nll_delta"]
                assert delta <= bound, (
                    "%s context %d %s mean NLL %.6f vs HF %.6f: delta %.6f exceeds %.3f"
                    % (spec["file"], case["context_size"], mode, nll, case["mean_nll"], delta, bound))
                # The CLI prints six significant digits, so allow decimal rounding.
                assert math.isclose(ppl, math.exp(nll), rel_tol=2e-5), "inconsistent NLL/PPL: " + out
                print("baseline-ppl[%s c=%d chunks=%d %s]: PPL %.4f vs HF %.4f, NLL delta %.6f <= %.3f  [ok]"
                      % (spec["file"], case["context_size"], case["chunks"], mode, ppl, case["perplexity"], delta,
                         bound))
    return True


def find_fixture(spec):
    override = os.environ.get("LLMX_BASELINE_GGUF")
    if override:
        name = os.path.basename(override)
        assert name in {s["file"] for s in BASELINE_MODELS}, (
            "LLMX_BASELINE_GGUF must retain the fixture filename to select its quantization bounds")
        if name != spec["file"]:
            return None
    return find_model({"gguf_repo": spec["repo"], "gguf_file": spec["file"],
                       "gguf_revision": spec["revision"]})


def find_model(doc):
    env = os.environ.get("LLMX_BASELINE_GGUF")
    if env:
        return env if os.path.exists(env) else None
    repo = doc["gguf_repo"].replace("/", "--")
    pattern = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub",
        "models--" + repo, "snapshots", doc.get("gguf_revision", "*"), doc["gguf_file"])
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def parse_ids(out):
    out = out.strip()
    if not out:
        return []
    return [int(t) for t in out.replace(",", " ").split()]


def run():
    return run_tokenizer() and run_logits() and run_perplexity()


def run_tokenizer():
    with io.open(GOLDEN, encoding="utf-8") as f:
        doc = json.load(f)

    model = find_model(doc)
    if not model:
        print("baseline: SKIP - no fixture model on disk (%s). "
              "Set LLMX_BASELINE_GGUF or `python tools/gen_baseline.py` deps."
              % doc["gguf_file"])
        return True

    failures = []
    for case in doc["cases"]:
        rc, out = cli(["tokenize", model, case["text"]])
        if rc != 0:
            failures.append((case["text"], case["ids"], "exit %d" % rc))
            continue
        got = parse_ids(out)
        if got != case["ids"]:
            failures.append((case["text"], case["ids"], got))

    n = len(doc["cases"])
    if failures:
        print("baseline: %d/%d cases MATCH the HF reference, %d differ:"
              % (n - len(failures), n, len(failures)))
        for text, want, got in failures:
            # The Windows console is not UTF-8, so escape non-ASCII rather than crashing the report on the cases most likely to fail.
            safe = text.encode("unicode_escape").decode("ascii")
            print("    text : '%s'" % safe)
            print("    want : %s" % want)
            print("    got  : %s" % got)
        return False

    print("baseline: tokenizer %d/%d cases match the HF reference (%s)  [ok]"
          % (n, n, doc["tokenizer_repo"]))
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
