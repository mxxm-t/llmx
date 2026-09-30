"""Optional check of llmx on the qwen35 files that the layered HF reference (tools/gen_layered_reference.py) made goldens for; no downloads.

    python -X utf8 tests/baseline_layered.py --exe build/llmx --model path/to/Qwen3.5-9B-Q4_K_M.gguf --output-dir hf-9b-review

The model's SHA-256 chooses its goldens, and the run is tests/baseline_8b.py's: 20 tokenizer cases, six rankings and four NLL cases, each scored batched and per token, with report.json and every command's output in the new directory.
A file with file-exact goldens, HF run on its own weights, is held to them first at FILE_EXACT_BOUNDS, and to its model's goldens only if they pass.
Where llmx does not run the qwen35 architecture, on a device whose backend lacks its ops, the run says so in one line and exits 0.
"""

from pathlib import Path
import sys

import baseline_8b
import baseline_qwen35
from qwen35 import REFUSALS

DATA = Path(__file__).resolve().parent / "data"
VOCAB_SIZE = 248320
MODEL_CONTEXT = 262144
# The Qwen3-0.6B Q4_K_M fixture's budget in tests/baseline.py, the same file type against its full-precision reference.
BOUNDS = {"top5_overlap": 4, "max_abs_logit": 100.0, "continuous_nll": 0.13, "window_nll": 0.25}
SCOPE = "20 tokenizer cases, six short prefill rankings and four NLL cases, each scored in batched passes and per token; not full-corpus or deep-context coverage"
UNCOMPARED = "; its quantized tensors are not compared."
# The Qwen3.6-35B-A3B Q4_K_M, whose own quantization turns over HF's top-1 of `The three primary colors are red,`, where the checkpoint's top two are 0.0058 logits apart and HF on the file's weights puts them the other way (docs/ASSETS.md, The layered qwen35 reference).
# Once it has passed every check of its file-exact goldens, it is held against the checkpoint's to the top-1 of five rankings of six, every other bound unchanged, for this file's SHA-256 alone, as agreed on 2026-09-30 at 14:30 under the delegation of 14:26.
QWEN36_35B_A3B_Q4_K_M_QUALITY = dict(BOUNDS, top1_matches=5)
# A file against goldens made from its own weights (tools/gen_layered_reference.py goldens --file-exact) is held to the qwen35 family's file-exact bounds, since the format's loss is on both sides.
FILE_EXACT_BOUNDS = dict(baseline_qwen35.BOUNDS[baseline_qwen35.FILE_EXACT_BOUNDS], max_abs_logit=100.0)

# Each GGUF's goldens by the GGUF's SHA-256, with each golden's SHA-256 with LF line ends.
GOLDENS = {
    "cd76ec205963b3b33350093e6904d9de16c4e666fd104e1f632d25c7f15f2a13": {
        "name": "Qwen3.5-9B Q4_K_M", "data": DATA / "qwen3.5-9b",
        "fixture_sha256": {
            "baseline_tokenizer.json": "ac9f09c19163e902b0b86fe73e9b24c8a0a762ee6e4bcd210f168156276bb983",
            "baseline_logits.json": "02b8adba52a6fb354cd04a2e9c6b382c0d480922ced82243c1ffa15b3fc16903",
            "baseline_perplexity.json": "3f084ab54634182dfcdd441c42eed0ea7d3369e2c7a9747af168c1aa512a2f5f",
        },
        "bounds": BOUNDS, "vocab": VOCAB_SIZE, "context": MODEL_CONTEXT, "scope": SCOPE,
        "provenance_limit": "The GGUF's Hub repository and commit are in its goldens, and its 177 F32 tensors equal the checkpoint's under the converter's conventions" + UNCOMPARED,
    },
    "876d304664f4615db8924b421a9a88df51f29be9ce413f0bfd059e0480e8f7fe": {
        "name": "Qwen3.6-27B Q4_K_M", "data": DATA / "qwen3.6-27b",
        "fixture_sha256": {
            "baseline_tokenizer.json": "25d59bac8e2fb151278664c66d1b5ebb42510cd735ca163920ced24096481689",
            "baseline_logits.json": "6715db5e42b2ae7386de25ec6a1b7c8fe3b9c658f6ea78afb464fdd060424ea7",
            "baseline_perplexity.json": "c014a20c804d68f3e68824df6d34ffff8e90ec66e3b62ff99fab4be59267aa36",
        },
        "bounds": BOUNDS, "vocab": VOCAB_SIZE, "context": MODEL_CONTEXT, "scope": SCOPE,
        "provenance_limit": ("The GGUF has no known Hub source; 331 of its 353 F32 tensors equal the checkpoint's under the converter's conventions, "
                             "and its other 22 ssm_a tensors differ in 28 values, each one float32 step from -exp(A_log) as torch computes it here" + UNCOMPARED),
    },
    "a8adba03e892f519579290e30f21abb1a9dbb5bfd092afecccd12a4ee670f0de": {
        "name": "Qwen3.6-35B-A3B Q4_K_M", "data": DATA / "qwen3.6-35b-a3b",
        "file_exact": {"data": DATA / "qwen3.6-35b-a3b-file-exact", "bounds": FILE_EXACT_BOUNDS, "fixture_sha256": {
            "baseline_logits.json": "6f8d22bdecba25f5fa1b5f98a68a6a8339f0ddcc7c1882bba1ccb5f126989a4a",
            "baseline_perplexity.json": "d1d7d354d86e165898a905c1e5a048c747edae3a064052083250aab0b9ce92e0",
        }},
        "fixture_sha256": {
            "baseline_tokenizer.json": "87a914328a35491d7d42b6794a346543f77c37ea57564313147a127e6efc7cdd",
            "baseline_logits.json": "6ad2c1e0a28e2581adb8d8fda403b55bcf70e4f44ac23f4379fa122ffac9dbdf",
            "baseline_perplexity.json": "8ae6351740ea5f9ff0f456c1484486891ac519c24647695c5fff28cb1d054d0d",
        },
        "bounds": QWEN36_35B_A3B_Q4_K_M_QUALITY, "vocab": VOCAB_SIZE, "context": MODEL_CONTEXT, "scope": SCOPE,
        "provenance_limit": ("The GGUF has no known Hub source; 289 of its 301 F32 tensors equal the checkpoint's under the converter's conventions, "
                             "every router and shared-expert gate among them, and its other 12 ssm_a tensors differ in 22 values, "
                             "each one float32 step from -exp(A_log) as torch computes it here" + UNCOMPARED),
    },
}


def main(argv=None):
    return baseline_8b.consume(argv, "Optional check of llmx on a qwen35 file against the layered HF reference's goldens; no downloads.",
                               GOLDENS.get, REFUSALS)


if __name__ == "__main__":
    sys.exit(main())
