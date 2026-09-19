import glob
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import run as cli

# Correctness baseline: compare llmx against golden fixtures generated once
# from the HF reference tokenizer (tools/gen_baseline.py). This is the only
# test in the suite with an EXTERNAL ground truth -- roundtrip checks the quant
# kernels against themselves, and the tokenizer test is a self-consistency
# round-trip that a consistently-wrong encoder passes happily.
#
# It needs a real model, so unlike the rest of the suite it SKIPS when one is
# not present rather than failing. Point it at a file with LLMX_BASELINE_GGUF,
# or let it find the fixture model in the HF cache.

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "data", "baseline_tokenizer.json")


GOLDEN_LOGITS = os.path.join(HERE, "data", "baseline_logits.json")

# Fixture models for the logit gate, each with the top-5 overlap it is expected
# to reach against the FULL-PRECISION reference. Coarser quantization reorders
# more of the tail, so the bound is per-model and measured, not guessed.
#
# The Q4_0 entry is not redundant. A "Q4_0" file from llama.cpp is mixed, and
# its token_embd is Q6_K whose super-block scale is a SUBNORMAL half. The Q8_0
# fixture has almost no subnormal scales (0.0061% of blocks, against 5.89% in
# Qwen3-8B), so it is structurally blind to the f16 subnormal bug class - the
# gate passed with that bug deliberately reintroduced until this model was
# added.
LOGIT_MODELS = [
    {"repo": "Qwen/Qwen3-0.6B-GGUF", "file": "Qwen3-0.6B-Q8_0.gguf", "min_overlap": 5},
    {"repo": "unsloth/Qwen3-0.6B-GGUF", "file": "Qwen3-0.6B-Q4_0.gguf", "min_overlap": 4},
]

# A correct next-token logit for these models sits around 15-25. Gross
# corruption blows this up (the reintroduced f16 bug gave 582), so a magnitude
# bound catches whole classes of damage that a ranking check can miss.
MAX_PLAUSIBLE_LOGIT = 100.0


def parse_logits(out):
    ids, vals = [], []
    for ln in out.strip().split("\n"):
        p = ln.split()
        if len(p) == 2 and p[0].isdigit():
            ids.append(int(p[0]))
            vals.append(float(p[1]))
    return ids, vals


def run_logits():
    """Compare llmx's next-token ranking against a FULL-PRECISION reference.

    llmx runs a quantized GGUF while the golden comes from the fp32 model, so
    logit VALUES differ by quantization error and comparing them directly is
    meaningless. What is stable is the ranking, plus a magnitude sanity bound:

      - top-1 must match, on every model.
      - top-5 SET overlap must reach the per-model bound above.
      - exact top-5 ORDER is deliberately NOT required: it legitimately differs
        when two tokens sit within about 0.01 logits of each other, far below
        quantization noise. Requiring it would flag correct behaviour.
    """
    with io.open(GOLDEN_LOGITS, encoding="utf-8") as f:
        doc = json.load(f)

    ran = 0
    for spec in LOGIT_MODELS:
        model = find_model({"gguf_repo": spec["repo"], "gguf_file": spec["file"]})
        if not model:
            continue
        ran += 1
        failures, ordered = [], 0
        for case in doc["cases"]:
            rc, out = cli(["logits", model, case["text"], "--top", "10"])
            if rc != 0:
                failures.append((case["text"], "exit %d" % rc))
                continue
            ids, vals = parse_logits(out)
            want = case["top_ids"]
            if not ids:
                failures.append((case["text"], "no logits parsed"))
            elif abs(vals[0]) > MAX_PLAUSIBLE_LOGIT:
                failures.append((case["text"], "top logit %.1f is implausible" % vals[0]))
            elif ids[0] != want[0]:
                failures.append((case["text"], "top-1 %d, reference %d" % (ids[0], want[0])))
            else:
                ov = len(set(ids[:5]) & set(want[:5]))
                if ov < spec["min_overlap"]:
                    failures.append((case["text"], "top-5 overlap %d/5, expected >= %d"
                                     % (ov, spec["min_overlap"])))
                elif ids[:5] == want[:5]:
                    ordered += 1

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

def find_model(doc):
    env = os.environ.get("LLMX_BASELINE_GGUF")
    if env:
        return env if os.path.exists(env) else None
    repo = doc["gguf_repo"].replace("/", "--")
    pattern = os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface", "hub",
        "models--" + repo, "snapshots", "*", doc["gguf_file"])
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


def parse_ids(out):
    out = out.strip()
    if not out:
        return []
    return [int(t) for t in out.replace(",", " ").split()]


def run():
    return run_tokenizer() and run_logits()


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
            # The Windows console is not UTF-8, so escape non-ASCII rather
            # than crashing the report on the cases most likely to fail.
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
