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
| CPU backend optimization                 | In Progress |
| More quant formats (Q4_0/Q4_1/Q6_K read) | Done     |
| More model architectures (Llama, ...)    | Planned  |
| More formats (safetensors, ...)          | Planned  |
| Device execution model (GPU prerequisite) | Planned |
| GPU backends (ROCm first, Vulkan portability) | Planned |
| Multi-device split                       | Planned  |
| Multi-node / cluster                     | Planned  |
| Multi-user server                        | Planned  |
| Correctness baseline vs HF reference     | In Progress |
| Performance floor vs mx-llama.cpp        | In Progress |
| HF integration (pull + Hub formats)      | Planned  |
| HF Hub kernels (additional, after #4a)   | Planned  |

## Active feature blocks

One block per in-flight feature. A block is what lets a fresh agent pick a
feature back up with a "continue feature X" prompt, so keep it current. When the
feature ships, delete its block and mark the row `Done` above.

### CPU backend optimization

- **Goal:** get the CPU hot paths (Q8_0 matvec, RMSNorm, RoPE) closer to
  theoretical throughput; remove avoidable overhead (per-call thread spawn,
  per-row cpuid) so the runtime actually performs.
- **Done:**
  - Cache AVX2 detection in constructor (was `__cpuid` on every row dot).
  - Vectorize `rms_norm` and `rope` with AVX2 FMA (scalar tails for non-multiples of 8).
  - Results (bench @ 2048):
    - matmul: baseline 3.6 GFLOPS / 8-thread 15.6 → now 19.5 GFLOPS default, 29.1 @ 8 threads.
    - rms_norm: 0.001ms → ~0.000ms (vectorized).
    - rope: 0.001ms → ~0.000ms (vectorized).
  - Correctness: vectorized rms_norm/rope verified vs scalar references across
    n/half in {1,2,7,8,15,16,31,32,33,64,100,1000}; max err ~1e-7 (float rounding only).
  - Full suite green: roundtrip / perf / tokenizer all PASS.
  - `bench` now reports end-to-end prefill/decode TPS on a tiny synthetic Qwen3
    model (2 layers, 256 embd, Q8_0 matmuls + F32 norms), and `tests/perf.py`
    gates on those floors (1000 / 800 tok/s). Decode < prefill as expected
    (longer KV cache). This is the automated regression signal — real-model TPS
    is a manual measurement via `perplexity`/`generate`, not part of the suite.
- **Left:**
  - Re-read docs (ARCHITECTURE/AGENTS mention) for stale claims.
- **Gotchas:**
  - **AVX-512 is deferred.** Dev CPU (Ryzen 7 5800X, Zen 3) has no AVX-512, so
    an AVX-512 path couldn't be benchmarked or proven lossless here. When an
    AVX-512 machine is available, add a runtime-selected kernel dispatch
    (detect AVX512F/VNNI, fall back to AVX2) without touching the Backend seam.
  - `rope` needs a scalar tail when `half` isn't a multiple of 8.
  - The worker pool must be torn down cleanly (join in destructor) and handle
    re-entrancy (task runs on calling thread when threads_ == 1).
  - Don't let the pool add more overhead than it saves for small matvecs;
    keep the single-threaded fast path.

### More quant formats — Q4_0

- **Goal:** add the first non-Q8_0 block quant, Q4_0, end-to-end: kernels +
  registry entry, GGUF format support (`data_size`, dequantize path), a CLI way
  to produce/read Q4_0 files, and a round-trip test. This is roadmap item #1 and
  validates the "one file + one registry entry" drop-in claim.
- **Done:**
  - `format/gguf.hpp`: `GGML_TYPE_Q4_0`, `Q4_0_BLOCK=32`, `Q4_0_TYPESIZE=18`,
    handled in `TensorInfo::data_size()`.
  - `quant/quant.hpp`: `quantize_row_q4_0` / `dequantize_row_q4_0`; registered in
    `register_builtins()`.
  - `cli/main.cpp`: `quantize <...> [q8_0|q4_0]` optional type arg; Q4_0 handled
    in `dequantize` and `type_name`; fixed `general.file_type` (Q8_0 → 7,
    Q4_0 → 2).
  - `model/arch_qwen.hpp`: `matvec`/`dequant_row` dispatch by tensor type via the
    registry. Q8_0 keeps the fused AVX2 path; Q4_0 uses a correct generic
    dequant-then-F32-dot path. `Model` ctor now calls `quant::register_builtins()`
    (idempotent) — the registry was previously never populated.
  - `tests/roundtrip.py`: Q4_0 round-trip leg (bound 1.0; measured 0.328).
  - `docs/USAGE.md` + `docs/src/quant-quant.md` + `docs/src/format-gguf.md`
    updated for Q4_0.
- **Left:**
  - Q4_0 full-model inference is implemented but not yet exercised on a real
    Q4_0 Qwen3 model (none on disk; `quantize` doesn't emit the `qwen3.*`
    metadata a runnable model needs). Kernels are validated by the round-trip
    test and the registry path by the Q8_0 bench; the Q4_0 generic matvec /
    `dequant_row` paths are code-reviewed but not end-to-end run.
  - Fused AVX2 dequant+FMA Q4_0 matvec kernel (see Gotchas).
- **Gotchas:**
  - Q4_0 block = 2-byte f16 scale + 32 nibbles (18 bytes): `qs[j]` holds value
    `j` in the low nibble and value `j+16` in the high nibble, each stored as
    unsigned 0..15 where the true value is `nibble - 8` (so -8..7); scale
    `d = amax/7`. Dequantize must mirror that layout.
  - **Q4_0 matmul is generic (correct-but-slow), not a fused AVX2 kernel.** It
    dequantizes each row to f32 then does an f32 dot. A fused AVX2 dequant+FMA
    Q4_0 kernel is the follow-up (same deferral pattern as AVX-512 in the CPU
    backend block). Until then Q4_0 inference is slower than Q8_0 despite the
    smaller size.
  - The quant `Registry` is a lazy singleton — inference code that uses it must
    call `quant::register_builtins()` (the `Model` ctor does) or `get()` returns
    null and throws.
  - `quantize` input tensors must have element counts divisible by 32 (both Q8_0
    and Q4_0 block size).

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
  - Tokenizer parity: **16/16 cases match Qwen/Qwen3-0.6B.**
  - Eight bugs found and fixed via this path, all of which survived a green
    suite: attention missing 1/sqrt(head_dim); temperature cancelling in the
    sampler; RoPE read past context_length; the GPT-2 whitespace guard that
    could never fire; attention width hardcoded to n_embd; tied embeddings
    unsupported; Windows argv delivered in the ANSI codepage so any non-ASCII
    prompt was mangled before llmx saw it; and the pretokenizer implementing
    the GPT-2 regex instead of the Qwen2/Qwen3 one.
- **Left:**
  - Logits, per-layer activation and corpus-PPL goldens. BLOCKED locally: any
    `from transformers import Auto*` segfaults on this workstation (plain
    `import transformers` and torch alone are both fine), so reference-model
    fixtures must be generated on the rig.
  - Tolerance bands: F32 vs reference tight, Q8_0 vs reference needs a
    quantization-appropriate bound.
- **Gotchas:**
  - A green suite proves nothing on its own - every pre-existing test compares
    llmx against llmx.
  - The 8B hides bugs the 0.6B exposes: `n_head * head_dim == n_embd` holds for
    Qwen3-8B (32*128 == 4096) and fails for 0.6B/1.7B/4B.
  - torch is a fixture-GENERATION dependency only, never needed to run the
    suite and never at runtime.
  - Qwen3 does NOT use the GPT-2 pretokenizer regex. Read the Split pattern out
    of `tokenizer.json` before touching `pretokenize`.

### Performance floor vs mx-llama.cpp

- **Goal:** llmx must be at least as fast as mx-llama.cpp on the same model,
  quant, prompt and hardware (`docs/ROADMAP.md` #8), pp and tg both reported.
- **Status: prefill is AT the floor. Decode is 22% under it.**
  Qwen3-8B Q8_0, 343-token wikitext prompt, -t 16, this workstation. Reference
  is the CPU AVX2 llama.cpp shipped with LM Studio, stock `llama-server`, same
  machine, so no rig time was used.
    - pp   llmx 37.23 / 37.86   llama.cpp 37.70 / 37.34   -> parity
    - tg   llmx  3.91           llama.cpp  4.99 / 5.02    -> 22% under
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
  - Decode, 22% under. It is memory-bandwidth bound (thread scaling is flat:
    4/8/16 threads give 3.83/4.13/3.90 tok/s) at about 32 GB/s against
    llama.cpp's 37, so the ceiling on the whole gap is bandwidth efficiency.
    Allocation churn and layout fragmentation are both measured NULL, so what
    remains is prefetch behaviour and possibly mmap.
  - An int8 x int8 inner product is NOT worth it for decode (saving ALU work
    buys nothing while stalled on RAM) but may still help prefill. It is lossy,
    so it needs the logits golden to bound the cost first.
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
