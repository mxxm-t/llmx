import hashlib
import io
import json
import math
import os
from pathlib import Path
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

# The fixture models for the logit/PPL gates are pinned in tests/data/fixtures.json (repo, revision, file and SHA-256), which tools/fetch_test_models.py downloads and the HF job's cache key hashes, so a change of bounds keeps the cached downloads.
FIXTURES = os.path.join(HERE, "data", "fixtures.json")

# Each model's bounds against the full-precision reference: the top-5 overlap it reaches, measured per model since coarser quantization reorders more of the tail, and its NLL deltas (docs/ASSETS.md).
BOUNDS = {
    "Qwen3-0.6B-Q8_0.gguf": {"top5_overlap": 5, "continuous_nll": 0.01, "window_nll": 0.02},
    # Subnormal f16 scales: its token_embd is Q6_K with a subnormal super-block scale, which the Q8_0 fixture almost never has.
    "Qwen3-0.6B-Q4_0.gguf": {"top5_overlap": 4, "continuous_nll": 0.16, "window_nll": 0.20},
    # 168 Q5_K, 29 Q6_K and 113 F32 tensors: the K-quant path in every matmul and the Q6_K head, on both backends.
    "Qwen3-0.6B-Q5_K_M.gguf": {"top5_overlap": 4, "continuous_nll": 0.05, "window_nll": 0.16},
    # 168 Q4_K, 29 Q6_K and 113 F32 tensors: the most common download's type in every matmul, from the Q4_0 file's repo and revision.
    "Qwen3-0.6B-Q4_K_M.gguf": {"top5_overlap": 4, "continuous_nll": 0.13, "window_nll": 0.25},
}

with io.open(FIXTURES, encoding="utf-8") as f:
    PINNED = json.load(f)
# Every pinned fixture model needs bounds, and only those, each pinned once.
assert sorted(spec["file"] for spec in PINNED) == sorted(BOUNDS), (
    "tests/data/fixtures.json pins %s, but tests/baseline.py bounds %s" % (sorted(spec["file"] for spec in PINNED), sorted(BOUNDS)))
BASELINE_MODELS = [dict(spec, **BOUNDS[spec["file"]]) for spec in PINNED]

# A correct next-token logit for these models sits around 15-25.
# Gross corruption blows this up (the reintroduced f16 bug gave 582), so a magnitude bound catches whole classes of damage that a ranking check can miss.
MAX_PLAUSIBLE_LOGIT = 100.0
# The fixture models share the vocabulary and the trained context that the outputs are checked against.
VOCAB_SIZE = 151936
MODEL_CONTEXT = 40960


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

    The output must also be well formed, as common.check_logits defines it for this gate and the 8B one.
    """
    with io.open(GOLDEN_LOGITS, encoding="utf-8") as f:
        doc = json.load(f)

    ran = 0
    for spec in BASELINE_MODELS:
        model = find_fixture(spec)
        if not model:
            continue
        ran += 1
        bounds = dict(spec, max_abs_logit=MAX_PLAUSIBLE_LOGIT)
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
            try:
                ids = common.check_logits(out, case, VOCAB_SIZE, bounds)["top_ids"]
            except ValueError as error:
                failures.append((case["text"], str(error)))
                continue
            if ids[:5] == case["top_ids"][:5]:
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
              % (spec["file"], n, n, spec["top5_overlap"], n, n, ordered, n))

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
            assert rc == 0 and common.parse_ids(out) == doc["token_ids"], "PPL token IDs differ from HF"
            for case, mode in ((case, mode) for case in common.ppl_cases(doc) for mode in common.PPL_MODES):
                rc, out = cli(common.ppl_command(model, path, case, mode))
                if common.device_lacks_kernel(rc, out):
                    print("baseline-ppl[%s]: SKIP - %s has no kernel for this model's matrices"
                          % (spec["file"], os.environ["LLMX_DEVICE"]))
                    break
                assert rc == 0, "perplexity failed (exit %d): %s" % (rc, out)
                try:
                    result = common.check_ppl(out, case, doc["n_tokens"], MODEL_CONTEXT, spec)
                except ValueError as error:
                    raise AssertionError("%s context %d %s: %s: %s"
                                         % (spec["file"], case["context_size"], mode, error, out)) from error
                print("baseline-ppl[%s c=%d chunks=%d %s]: PPL %.4f vs HF %.4f, NLL delta %.6f <= %.3f  [ok]"
                      % (spec["file"], case["context_size"], case["chunks"], mode, result["perplexity"],
                         case["perplexity"], result["absolute_nll_delta"], result["bound"]))
    return True


def find_fixture(spec):
    override = os.environ.get("LLMX_BASELINE_GGUF")
    if override:
        name = os.path.basename(override)
        assert name in {s["file"] for s in BASELINE_MODELS}, (
            "LLMX_BASELINE_GGUF must retain the fixture filename to select its quantization bounds")
        if name != spec["file"]:
            return None
    return find_model(spec)


def snapshot_path(repo, revision, file):
    """Where the HF cache keeps `file` of `repo` at `revision`, which is where tools/fetch_test_models.py writes it."""
    return Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision / file


def find_model(spec):
    """The model LLMX_BASELINE_GGUF names when it is set, otherwise the pinned snapshot of the BASELINE_MODELS entry `spec`; None when that file is absent."""
    env = os.environ.get("LLMX_BASELINE_GGUF")
    if env:
        return env if os.path.exists(env) else None
    path = snapshot_path(spec["repo"], spec["revision"], spec["file"])
    return str(path) if path.exists() else None


def run():
    return run_tokenizer() and run_logits() and run_perplexity()


def run_tokenizer():
    with io.open(GOLDEN, encoding="utf-8") as f:
        doc = json.load(f)

    # The golden names its model by repository and file, and the entry for that file in BASELINE_MODELS pins its revision.
    spec = next((s for s in BASELINE_MODELS if s["repo"] == doc["gguf_repo"] and s["file"] == doc["gguf_file"]), None)
    assert spec, "the tokenizer golden's model %s is not a pinned fixture" % doc["gguf_file"]
    model = find_model(spec)
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
        got = common.parse_ids(out)
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
