"""Optional check of llmx on the qwen35 files that the layered HF reference (tools/gen_layered_reference.py) made goldens for; no downloads.

    python -X utf8 tests/baseline_layered.py --exe build/llmx --model path/to/Qwen3.5-9B-Q4_K_M.gguf --output-dir hf-9b-review

The model's SHA-256 chooses its goldens, and the run is tests/baseline_8b.py's: 20 tokenizer cases, six rankings and four NLL cases, each scored batched and per token, with report.json and every command's output in the new directory.
While llmx refuses the qwen35 architecture, the run says so in one line and exits 0.
"""

from pathlib import Path
import sys

import baseline_8b

DATA = Path(__file__).resolve().parent / "data"
VOCAB_SIZE = 248320
MODEL_CONTEXT = 262144
# Declared before any llmx comparison, as the 8B's were: the Qwen3-0.6B Q4_K_M fixture's budget in tests/baseline.py, the same file type against its full-precision reference.
BOUNDS = {"top5_overlap": 4, "max_abs_logit": 100.0, "continuous_nll": 0.13, "window_nll": 0.25}
# llmx's error for a file whose architecture it does not run.
REFUSAL = "unsupported metadata general.architecture"
SCOPE = "20 tokenizer cases, six short prefill rankings and four NLL cases, each scored in batched passes and per token; not full-corpus or deep-context coverage"
PROVENANCE = ("The GGUF's Hub repository and commit, when known, are in its goldens; its %d F32 tensors equal the checkpoint's under the converter's conventions, "
              "and its quantized tensors are not compared.")

# Each GGUF's goldens by the GGUF's SHA-256, with each golden's SHA-256 with LF line ends.
GOLDENS = {
    "cd76ec205963b3b33350093e6904d9de16c4e666fd104e1f632d25c7f15f2a13": {
        "name": "Qwen3.5-9B Q4_K_M", "data": DATA / "qwen3.5-9b",
        "fixture_sha256": {
            "baseline_tokenizer.json": "ac9f09c19163e902b0b86fe73e9b24c8a0a762ee6e4bcd210f168156276bb983",
            "baseline_logits.json": "02b8adba52a6fb354cd04a2e9c6b382c0d480922ced82243c1ffa15b3fc16903",
            "baseline_perplexity.json": "3f084ab54634182dfcdd441c42eed0ea7d3369e2c7a9747af168c1aa512a2f5f",
        },
        "bounds": BOUNDS, "vocab": VOCAB_SIZE, "context": MODEL_CONTEXT, "scope": SCOPE, "provenance_limit": PROVENANCE % 177,
    },
}


def main(argv=None):
    return baseline_8b.consume(argv, "Optional check of llmx on a qwen35 file against the layered HF reference's goldens; no downloads.",
                               GOLDENS.get, REFUSAL)


if __name__ == "__main__":
    sys.exit(main())
