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
| More quant formats (Q4_0, Q4_1, ...)     | In Progress |
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
- **Done:**
  - Removed both per-call thread-spawn sites. `matvec_q8_0` now uses a
    persistent pool, and `attend_heads` goes through `Backend::parallel_for`
    instead of owning threads in the model layer.
  - Qwen3-8B Q8_0, this workstation, 24 tokens greedy, default threads,
    interleaved A/B/A/B each time:
      - spawn-per-call baseline:  8277 ms decode
      - pooled matvec:            6913 ms  (-16.5%)
      - pooled attention too:     6372 ms  (-23% cumulative, 2.90 -> 3.77 tok/s)
  - Greedy output byte-identical across every arm on both 8B and 0.6B.
  - `generate` now prints tok/s with two decimals; the integer print was
    rounding a 20% change away.
- **Floor measured 2026-09-19.** Reference is the CPU AVX2 llama.cpp that ships
  with LM Studio (`llama.cpp-win-x86_64-avx2-2.28.2`), stock `llama-server` on
  this same workstation and CPU, so no build was needed and no rig time was
  used. The mx fork's changes are gfx906/GPU-specific, so its CPU path is
  upstream; this is a fair stand-in and is labelled as one.
  Qwen3-8B Q8_0, prompt "The capital of France is", 24 tokens greedy, `-t 16`
  on both, `cache_prompt` off, two runs each:
    - tg  llmx 3.81 / 3.96   llama.cpp 4.61 / 4.60   -> llmx 15.7% SLOWER
    - pp  llmx 3.68 / 4.10   llama.cpp 12.20 / 11.84 -> llmx 3.1x SLOWER
  llmx is under the floor on both arms.
- **Output divergence to settle.** At temp 0 on the same weights both emit
  " Paris. The capital of Italy is Rome. The capital of" and then split: llmx
  continues "Spain is", llama.cpp "Germany is". That is an argmax flip around
  token 11. It may be ordinary FP accumulation order, or it may be residual
  llmx inaccuracy - the logits golden is what settles it, so this is tracked
  under the correctness baseline, not assumed benign.
- **Left:**
  - **Decode is memory-bandwidth bound, measured, not assumed.** Qwen3-8B Q8_0
    thread scaling is flat: 4 threads 3.83 tok/s, 8 threads 4.13, 16 threads
    3.90. At 8.1 GB of weights streamed per token that is about 31.6 GB/s,
    close to practical DDR4 dual-channel on this 5800X.
    Consequences, which redirect the remaining work:
      - An int8 x int8 dot kernel (quantizing the activation, as llama.cpp
        does) would NOT help tg. Saving ALU work does nothing while stalled on
        RAM, and it would cost numerics for no gain. NOT worth doing here.
      - tg headroom is bounded: llama.cpp reaches about 37 GB/s, so the ceiling
        on the whole 16% gap is bandwidth EFFICIENCY. The suspect is layout,
        llmx holds tensor data in 399 separate heap allocations while llama.cpp
        mmaps one contiguous file-backed region.
      - pp is where the real headroom is. Prefill is compute-bound because each
        weight byte is reused across the batch, so batching is worth up to the
        full 3.1x and thread scaling there should be real.
  - **Close the pp gap first, it is the 3.1x one.** `infer::prefill` feeds one
    token at a time, so every prompt matvec is matrix-VECTOR where llama.cpp
    runs matrix-matrix over the whole prompt. Batched prefill is also ROADMAP
    #4a's item, so it serves the GPU work too.
  - **Close the tg gap (16%) by layout, not by kernel.** Load tensor data as one
    contiguous region (or mmap it) instead of 399 separate heap allocations, so
    the weight stream is sequential and prefetchable. This is lossless.
  - Remaining known costs, in likely order: `Model::step` heap-allocates
    gate/up/ffn every layer every token and `attend_head` allocates a scores
    vector per head; `read_gguf` loads the whole file with no mmap, so every
    invocation pays a full 8 GB read and double resident memory; prefill feeds
    one token at a time so there is no matrix-matrix work.
- **Gotchas:**
  - The synthetic `bench` model (2 layers, 256 embd) does NOT show these wins:
    its matvecs are small enough to take the single-threaded fast path. Real
    gains only appear on a real model, so measure there and say so.
  - Compare like with like and state it: same quant, thread count, prompt
    length. A single number with no configuration is not a measurement.

Nothing else is in flight. When you start a feature, open a block above
before writing code — see `AGENTS.md` → "Starting a feature".
