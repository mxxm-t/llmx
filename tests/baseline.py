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
