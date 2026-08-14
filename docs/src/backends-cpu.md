# `src/backends/cpu/cpu_backend.hpp` — CPU backend (AVX2)

CPU implementation of the `Backend` interface, in namespace `backend`.

- Detects AVX2 **once** in the constructor (via `__cpuid` on MSVC, `__get_cpuid`
  on GCC/Clang) and caches it — not per row.
- `matvec_q8_0`: fused dequant+FMA AVX2 row dot; uses a thread pool for large
  problems, single-threaded fast path for small ones (threshold `nout < threads*8`).
- `dot_row_impl`: AVX2 fused dequant (f16 scale broadcast) + FMA accumulation
  over int8 blocks, with a scalar fallback.
- `rms_norm`, `rope`: AVX2 vectorized with scalar tails for non-multiples of 8.
- `make_cpu_backend()` factory.

The AVX-512 path is deferred (no dev hardware to benchmark/prove lossless); a
runtime-dispatched AVX512F/VNNI kernel can be added later without touching the
`Backend` seam.
