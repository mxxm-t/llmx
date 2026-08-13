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
| More quant formats (Q4_0, Q4_1, ...)     | Planned  |
| More model architectures (Llama, ...)    | Planned  |
| More formats (safetensors, ...)          | Planned  |
| GPU backends (ROCm / CUDA / Vulkan)      | Planned  |
| Multi-device split                       | Planned  |
| Multi-node / cluster                     | Planned  |
| Multi-user server                        | Planned  |

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
- **Left:**
  - Bump `tests/perf.py` floor (currently 0.5 GFLOPS, now 18+ default) if you want
    it to catch real regressions rather than only catastrophic ones.
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

Nothing is in flight right now. When you start a feature, open a block above
before writing code — see `AGENTS.md` → "Starting a feature".
