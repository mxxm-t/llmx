"""The pinned MXFP4 writer's file against file-exact HF and the original fp32 model.

Run by hand; the hosted suite covers MXFP4 with its tiny raw-block fixtures.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import baseline
import common


MODEL_SHA256 = "84de98ed82dbcd8ee077731a03806b6861839684916b468569e65e8b26ddce45"
# BOSS approved 2026-09-27 for this file only; file-exact correctness bounds stay unchanged.
BOUNDS = dict(file="Qwen3-0.6B-MXFP4.gguf", top1_matches=5, top5_overlap=2,
              continuous_nll=0.21, window_nll=0.35)
FILE_EXACT = Path(__file__).resolve().parent / "data/mxfp4-file-exact"


def run(model):
    assert baseline.file_sha256(Path(model)) == MODEL_SHA256, "MXFP4 quality bounds require the pinned writer's file"
    if not baseline.run_file_exact(str(FILE_EXACT), model):
        return False
    with open(baseline.GOLDEN_LOGITS, encoding="utf-8") as f:
        logits = json.load(f)
    assert len(logits["cases"]) == 6, "MXFP4 top-1 bound requires the six approved prompts"
    if baseline.check_model_logits(logits, model, BOUNDS, "mxfp4-fp32-logits") is not True:
        return False
    with open(baseline.GOLDEN_PPL, encoding="utf-8") as f:
        ppl = json.load(f)
    with tempfile.TemporaryDirectory(prefix="llmx_mxfp4_ppl_") as directory:
        return baseline.check_model_ppl(ppl, baseline.ppl_excerpt(ppl, directory), model, BOUNDS, "mxfp4-fp32-ppl") is True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--exe", default=common.EXE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-type", choices=("f16", "f32"), default="f16")
    args = parser.parse_args()
    common.EXE = os.path.abspath(args.exe)
    os.environ["LLMX_DEVICE"] = args.device
    os.environ["LLMX_CACHE_TYPE"] = args.cache_type
    return 0 if run(args.model) else 1


if __name__ == "__main__":
    sys.exit(main())
