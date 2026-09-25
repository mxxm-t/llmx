import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_8b import file_sha256
import common
from common import run as cli

# Correctness baseline: llmx against golden fixtures generated once from the HF reference (tools/gen_baseline.py); with f32.py it is the suite's external ground truth.
# It needs a real model, so it skips when none is present; point it at one with LLMX_BASELINE_GGUF, or let it find the fixture model in the HF cache.

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "data", "baseline_tokenizer.json")


GOLDEN_LOGITS = os.path.join(HERE, "data", "baseline_logits.json")
GOLDEN_PPL = os.path.join(HERE, "data", "baseline_perplexity.json")

# The fixture models are pinned in tests/data/fixtures.json: repo, revision, file, SHA-256 and size, "gate" for the models the logit/PPL gates check, and "hosted" for those the hosted HF job downloads.
# tools/fetch_test_models.py downloads the gate's models, and the HF job's cache key hashes their pins, so a change of bounds keeps the cached downloads.
# The other entries are pinned ahead of the tensor types they hold, each joining the gate with its type's bounds (docs/ASSETS.md).
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
# Each model is pinned once, and every gate model needs bounds, and only those.
assert len({spec["file"] for spec in PINNED}) == len(PINNED), "tests/data/fixtures.json pins a file twice"
GATE_FILES = sorted(spec["file"] for spec in PINNED if spec["gate"])
assert GATE_FILES == sorted(BOUNDS), "tests/data/fixtures.json gates %s, but tests/baseline.py bounds %s" % (GATE_FILES, sorted(BOUNDS))
# The hosted HF job downloads and requires every gate model, so a model it is to leave out joins the gate only with a change that lets the job leave it out.
assert all(spec["hosted"] for spec in PINNED if spec["gate"]), "tests/data/fixtures.json gates a model the hosted HF job does not download"
BASELINE_MODELS = [dict(spec, **BOUNDS[spec["file"]]) for spec in PINNED if spec["gate"]]

# A correct next-token logit for these models sits around 15-25.
# Gross corruption blows this up (the reintroduced f16 bug gave 582), so a magnitude bound catches whole classes of damage that a ranking check can miss.
MAX_PLAUSIBLE_LOGIT = 100.0
# The fixture models share the vocabulary and the trained context that the outputs are checked against.
VOCAB_SIZE = 151936
MODEL_CONTEXT = 40960


def run_logits():
    """Compare llmx's next-token ranking against a FULL-PRECISION reference, on each fixture model on disk (check_model_logits).

    llmx runs a quantized GGUF while the golden comes from the fp32 model, so logit VALUES differ by quantization error and comparing them directly is meaningless.
    What is stable is the ranking, plus a magnitude sanity bound:

      - top-1 must match, on every model.
      - top-5 SET overlap must reach the per-model bound above; a swap at the 5th place counts as agreement when the reference puts both tokens within 0.1 logits of its 5th value, or when the reference's 5th and 6th trade places and llmx's own logits for them are within 0.1 (common.top5_overlap).
      - exact top-5 ORDER is deliberately NOT required: it legitimately differs when two tokens sit within about 0.01 logits of each other, far below quantization noise.
        Requiring it would flag correct behaviour.

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
        if check_model_logits(doc, model, spec) is False:
            return False

    if ran == 0:
        print("baseline-logits: SKIP - no fixture model on disk")
    return True


def check_model_logits(doc, model, spec, name="baseline-logits"):
    """`llmx logits` on `model` against the cases of the logit golden `doc` at the bounds in `spec`.
    True when every case agrees, False after printing the ones that do not, and None when the configured device has no kernel for the model."""
    bounds = dict(spec, max_abs_logit=MAX_PLAUSIBLE_LOGIT)
    failures, ordered = [], 0
    for case in doc["cases"]:
        rc, out = cli(["logits", model, case["text"], "--top", "10"])
        if common.device_lacks_kernel(rc, out):
            print("%s[%s]: SKIP - %s has no kernel for this model's matrices"
                  % (name, spec["file"], os.environ["LLMX_DEVICE"]))
            return None
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

    n = len(doc["cases"])
    if failures:
        print("%s[%s]: %d/%d prompts agree, %d differ:"
              % (name, spec["file"], n - len(failures), n, len(failures)))
        for text, why in failures:
            print("    %s" % text.encode("unicode_escape").decode("ascii")[:60])
            print("      %s" % why)
        return False
    print("%s[%s]: top-1 %d/%d, top-5 set >=%d %d/%d, exact order %d/%d  [ok]"
          % (name, spec["file"], n, n, spec["top5_overlap"], n, n, ordered, n))
    return True


def ppl_excerpt(doc, directory):
    """The perplexity golden's text written to a file in `directory`, once the golden is checked to hold together."""
    raw = doc["text"].encode("utf-8")
    assert hashlib.sha256(raw).hexdigest() == doc["text_sha256"], "PPL fixture text changed"
    assert len(doc["token_ids"]) == doc["n_tokens"] == doc["n_scored"] + 1
    assert math.isfinite(doc["mean_nll"]) and math.isfinite(doc["perplexity"])
    assert math.isclose(math.exp(doc["mean_nll"]), doc["perplexity"], rel_tol=1e-12)
    path = os.path.join(directory, "excerpt.txt")
    with open(path, "wb") as f:
        f.write(raw)
    return path


def run_perplexity():
    with io.open(GOLDEN_PPL, encoding="utf-8") as f:
        doc = json.load(f)
    with tempfile.TemporaryDirectory(prefix="llmx_ppl_") as directory:
        path = ppl_excerpt(doc, directory)
        for spec in BASELINE_MODELS:
            model = find_fixture(spec)
            if not model:
                print("baseline-ppl[%s]: SKIP - fixture model not on disk" % spec["file"])
                continue
            check_model_ppl(doc, path, model, spec)
    return True


def check_model_ppl(doc, path, model, spec, name="baseline-ppl"):
    """`llmx perplexity` on `model` over the text in the file `path` against every case of the perplexity golden `doc`, batched and per token, at the bounds in `spec`.
    The first case out of bounds raises AssertionError; None when the configured device has no kernel for the model."""
    rc, out = cli(["tokenize", model, doc["text"]])
    assert rc == 0 and common.parse_ids(out) == doc["token_ids"], "PPL token IDs differ from HF"
    for case, mode in ((case, mode) for case in common.ppl_cases(doc) for mode in common.PPL_MODES):
        rc, out = cli(common.ppl_command(model, path, case, mode))
        if common.device_lacks_kernel(rc, out):
            print("%s[%s]: SKIP - %s has no kernel for this model's matrices"
                  % (name, spec["file"], os.environ["LLMX_DEVICE"]))
            return None
        assert rc == 0, "perplexity failed (exit %d): %s" % (rc, out)
        try:
            result = common.check_ppl(out, case, doc["n_tokens"], MODEL_CONTEXT, spec)
        except ValueError as error:
            raise AssertionError("%s context %d %s: %s: %s"
                                 % (spec["file"], case["context_size"], mode, error, out)) from error
        print("%s[%s c=%d chunks=%d %s]: PPL %.4f vs HF %.4f, NLL delta %.6f <= %.3f  [ok]"
              % (name, spec["file"], case["context_size"], case["chunks"], mode, result["perplexity"],
                 case["perplexity"], result["absolute_nll_delta"], result["bound"]))
    return True


# A file against its file-exact goldens (tools/gen_baseline.py file-exact), HF run on that file's own weights as tests/spec_decode.py decodes them, is held to the Q8_0 fixture's bounds whatever its type.
# The format's loss is on both sides, so what is left to bound is llmx's arithmetic.
FILE_EXACT_BOUNDS = dict(BOUNDS["Qwen3-0.6B-Q8_0.gguf"])


def run_file_exact(directory, model):
    """`model` against the logit and perplexity goldens in `directory`, which must have been made from this very file.
    A device with no kernel for the model fails the run rather than skipping it, since the run was asked for this one file there."""
    docs = []
    for name in ("baseline_logits.json", "baseline_perplexity.json"):
        with io.open(os.path.join(directory, name), encoding="utf-8") as f:
            docs.append(json.load(f))
    sha256 = file_sha256(Path(model))
    for doc in docs:
        assert "weights" in doc, "%s holds goldens that are not file-exact" % directory
        assert doc["weights"]["sha256"] == sha256, ("the goldens in %s were made from %s with SHA-256 %s, not from %s"
                                                    % (directory, doc["weights"]["file"], doc["weights"]["sha256"], model))
    spec = dict(FILE_EXACT_BOUNDS, file=os.path.basename(model))
    logits, ppl = docs
    checked = check_model_logits(logits, model, spec, "file-exact-logits")
    if checked:
        with tempfile.TemporaryDirectory(prefix="llmx_ppl_") as temporary:
            checked = check_model_ppl(ppl, ppl_excerpt(ppl, temporary), model, spec, "file-exact-ppl")
    if checked is None:
        print("file-exact[%s]: FAIL - %s has no kernel for this model's matrices, which fails a file-exact run"
              % (spec["file"], os.environ["LLMX_DEVICE"]))
    return bool(checked)


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


def pinned_fixture(file):
    """The entry of tests/data/fixtures.json that pins `file`, or None."""
    return next((spec for spec in PINNED if spec["file"] == file), None)


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

    failures = common.tokenize_failures(model, doc["cases"])
    n = len(doc["cases"])
    if failures:
        print("baseline: %d/%d cases MATCH the HF reference, %d differ:"
              % (n - len(failures), n, len(failures)))
        for text, want, got in failures:
            print("    text : '%s'" % text)
            print("    want : %s" % want)
            print("    got  : %s" % got)
        return False

    print("baseline: tokenizer %d/%d cases match the HF reference (%s)  [ok]"
          % (n, n, doc["tokenizer_repo"]))
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="The real-model HF checks on the pinned fixture models, or one file against its file-exact goldens.")
    parser.add_argument("--exe", default=common.EXE, help="path to the built llmx executable")
    parser.add_argument("--device", help="run the commands that take --device on this device, e.g. vulkan:0")
    parser.add_argument("--file-exact", metavar="DIR", help="the goldens tools/gen_baseline.py file-exact wrote for --model")
    parser.add_argument("--model", help="with --file-exact, the GGUF file those goldens were made from")
    args = parser.parse_args(argv)
    if bool(args.file_exact) != bool(args.model):
        parser.error("--file-exact and --model go together")
    common.EXE = os.path.abspath(args.exe)
    common.exe_path()
    if args.device:
        os.environ["LLMX_DEVICE"] = args.device
    return 0 if (run_file_exact(args.file_exact, args.model) if args.file_exact else run()) else 1


if __name__ == "__main__":
    sys.exit(main())
