# llmx — Development Status

Living tracker. This is the disposable file: notes here are only useful while
work is in progress. When a feature ships, delete its block below and mark the
row `Done` in the table. Read it together with `docs/ROADMAP.md` (the stable
plan) and `docs/ARCHITECTURE.md` (the layer rules) — STATUS carries where each
feature currently stands right now.

## Status table

| Feature                                  | Status   |
|------------------------------------------|----------|
| Layered restructure                      | Done     |
| Build config (config.hpp + CMake + build.bat) | Done |
| Test suite (roundtrip / perf / tokenizer)| Done     |
| Perf `bench` command                     | Done     |
| CPU backend optimization                 | Done     |
| More quant formats (Q4_0/Q4_1/Q4_K/Q5_K/Q6_K read) | Done |
| More model architectures (Llama, ...)    | Planned  |
| More formats (safetensors, ...)          | Planned  |
| Device execution model (GPU prerequisite) | Planned |
| GPU backends (ROCm first, Vulkan portability) | Planned |
| Multi-device split                       | Planned  |
| Multi-node / cluster                     | Planned  |
| Multi-user server                        | Planned  |
| Correctness baseline vs HF reference     | In Progress |
| HF fixed-excerpt PPL baseline            | Done     |
| Performance floor vs mx-llama.cpp        | In Progress |
| Perplexity text-file input (-f/--file)    | Done     |
| Chunked corpus perplexity               | Done     |
| F32 embedding/matrix inference          | Planned  |
| GitHub CPU CI                          | Done     |
| Automatic build identification          | Done     |
| HF fixture download retries and CI cache | Done |
| HF integration (pull + Hub formats)      | Planned  |
| HF Hub kernels (additional, after #4a)   | Planned  |

## Active feature blocks

One block per in-flight feature. A block is what lets a fresh agent pick a
feature back up with a "continue feature X" prompt, so keep it current. When the
feature ships, delete its block and mark the row `Done` above.

### Correctness baseline vs HF reference

- **Goal:** give the suite an external ground truth. Correctness is measured
  against the HF reference, never against llmx itself (`docs/ROADMAP.md` #8).
- **Done:**
  - `tools/gen_baseline.py` emits golden fixtures using `tokenizers` +
    `huggingface_hub` only (no torch, no transformers); output committed to
    `tests/data/baseline_tokenizer.json`.
  - `tests/baseline.py` compares llmx against the committed goldens and is
    wired into `tests/run_tests.py`. It SKIPS when no fixture model is on disk,
    so the rest of the suite still runs anywhere.
  - Tokenizer parity: **20/20 cases match pinned Qwen/Qwen3-0.6B.**
  - HF fp32 top-10 logit goldens for six prompts landed in `1c2e102`.
    The baseline checks top-1, top-5 set overlap and a magnitude bound on
    Qwen3-0.6B Q8_0 and mixed Q4_0 fixtures. Both passed all six prompts in
    that commit; the mixed fixture catches the f16 subnormal regression.
  - Local golden generation works in an isolated environment with
    numpy<2.3, torch 2.5.1+cpu and transformers 4.55.2.
  - Pinned HF fp32 PPL golden for a 247-token wikitext excerpt: exact token
    IDs/count and finite NLL/PPL checks, with per-quant absolute NLL bounds.
    HF PPL 28.7974; Q8_0 28.8371 (NLL delta 0.001374 <= 0.01); mixed Q4_0
    32.8463 (delta 0.131554 <= 0.16). Both llmx arms repeated identically at
    printed precision. Generator, provenance and bounds are in `docs/ASSETS.md`.
    Eight injected bad-output cases were rejected; build and full suite pass.
  - Nine bugs found and fixed via this path, all of which survived a green
    suite: attention missing 1/sqrt(head_dim); temperature cancelling in the
    sampler; RoPE read past context_length; the GPT-2 whitespace guard that
    could never fire; attention width hardcoded to n_embd; tied embeddings
    unsupported; Windows argv delivered in the ANSI codepage so any non-ASCII
    prompt was mangled before llmx saw it; and the pretokenizer implementing
    the GPT-2 regex instead of the Qwen2/Qwen3 one; the byte encoder incorrectly
    including soft-hyphen byte 0xAD in its printable set. Four new HF cases
    reject the preserved old encoder. Whole-wikitext file input now tokenizes
    298,938 tokens and scores the requested window limit successfully.
- **Left:**
  - Per-layer activation and full-corpus PPL goldens, plus long-context validation.
    Continuous and chunked excerpt gates are implemented, with an explicit
    disjoint-window scoring policy (`docs/USAGE.md`). They are not full-corpus
    coverage. Two-token windows show large quantized/HF deviations even under
    the unchanged old scorer; diagnostics and bounds are in `docs/ASSETS.md`.
    The existing ranking gate does not bound full-vector numerical error.
  - Tolerance bands: F32 vs reference tight, Q8_0 vs reference needs a
    quantization-appropriate bound.
- **Gotchas:**
  - Round-trip and synthetic tests alone are not an external correctness gate;
    preserve the HF tokenizer/logit checks and extend their coverage.
  - The 8B hides bugs the 0.6B exposes: `n_head * head_dim == n_embd` holds for
    Qwen3-8B (32*128 == 4096) and fails for 0.6B/1.7B/4B.
  - torch is a fixture-GENERATION dependency only, never needed to run the
    suite and never at runtime.
  - Analytic scoring tests exposed unsupported F32 embeddings/matrices.
    Norms work, but the registry's F32 entry has no dequantizer and both
    embedding lookup and backend matmul reject it. Fix and validate this
    before claiming all-F32 inference or a tight F32 numerical gate.
  - Generation's legacy reasoning filter searches `thinking_start/end`, not
    Qwen3's actual `<think>` / `</think>` markers. Its docs now state that limit.
  - Qwen3 does NOT use the GPT-2 pretokenizer regex. Read the Split pattern out
    of `tokenizer.json` before touching `pretokenize`.

### Performance floor vs mx-llama.cpp

- **Goal:** llmx must be at least as fast as mx-llama.cpp on the same model,
  quant, prompt and hardware (`docs/ROADMAP.md` #8), pp and tg both reported.
- **Status: recorded prefill parity with the upstream CPU stand-in; decode
  remains below it. The exact mx-llama.cpp floor is not yet established.**
  Qwen3-8B Q8_0, 343-token wikitext prompt, -t 16, this workstation. Reference
  is the CPU AVX2 llama.cpp shipped with LM Studio, stock `llama-server`, same
  machine, so no rig time was used.
    - pp   llmx 37.23 / 37.86   llama.cpp 37.70 / 37.34   -> parity
    - tg   llmx  3.91           llama.cpp  4.99 / 5.02    -> 22% under
  These are the original matched measurements. Later Q8_0 decode commits
  report 4.12 tok/s with independent accumulators (`54ea063`) and 4.24/4.18
  with F16C (`0c12570`); these are not a new matched mx-llama.cpp comparison.
  Separate decode/prefill thread flags landed in `dada6c0`.
- **How prefill got there, 3.89 -> 37.5 tok/s (9.6x), each step A/B measured:**
  - persistent worker pool instead of spawning threads per call (decode -23%)
  - attention through `Backend::parallel_for` instead of its own threads
  - batched prefill: matrix-matrix instead of one token at a time (3.89 -> 13.2)
  - dequantize each weight row once per batch, not once per column (-> 15.3)
  - four independent accumulators in the f32 dot (-> 17.0)
  - fused 4-row kernel sharing one activation load (-> 24.0)
  - two activation columns per four rows, 0.75 loads/FMA (-> 33.7)
  - three activation columns per four rows, 0.58 loads/FMA (-> 37.5)
- **The lesson worth keeping:** the kernel was LOAD bound, not FMA bound. Each
  naive dot needs 2 loads per FMA and Zen3 sustains about 2 loads/cycle against
  2 FMAs/cycle, so it ran at half of FMA peak no matter how the batch was
  blocked. Every win after the first came from raising the FMA:load ratio.
- **Left:**
  - Close the decode gap and measure both phases against mx-llama.cpp itself.
    Q8_0 decode is memory-bandwidth bound (early thread scaling was flat:
    4/8/16 threads give 3.83/4.13/3.90 tok/s) at about 32 GB/s against
    llama.cpp's 37, so the ceiling on the whole gap is bandwidth efficiency.
    Allocation churn, layout fragmentation and software prefetch are measured
    NULL. mmap has not been established as a throughput improvement.
  - An int8 x int8 inner product is NOT worth it for decode (saving ALU work
    buys nothing while stalled on RAM) but may still help prefill. It is lossy,
    so it needs an appropriate numerical bound; top-token rankings alone do
    not establish that bound.
- **Gotchas:**
  - The synthetic `bench` model (2 layers, 256 embd) shows NONE of these wins:
    its matvecs take the single-threaded fast path. Measure on a real model.
  - Do not tune the row block as a byte budget. Measured worse at every size
    (64/128/196/256 KB gave 22.37/22.04/23.68/21.12 against 24.04 for a flat
    4); the knee follows the fused kernel width, so it is `DOT_ROWS`.
  - ubatch barely matters once the kernel is right, and 343 vs 512 on a
    343-token prompt is the SAME computation - do not read noise as signal.

Nothing else is in flight. When you start a feature, open a block above
before writing code — see `AGENTS.md` → "Starting a feature".
