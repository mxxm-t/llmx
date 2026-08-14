# `src/backends/backend.hpp` — compute backend interface

Device-agnostic compute abstraction in namespace `backend`. The inference graph
runs its primitive ops through a `Backend` so the same model code can target CPU
now and ROCm / CUDA / Vulkan / oneAPI later without changes.

- `set_threads(n)`, `threads_available()`: worker-thread control.
- `dot_q8_0(row, x, nblocks)`: dot product of one Q8_0 block row with `x`.
- `matvec_q8_0(data, x, out, nblocks, nout)`: `out[o] = dot(row_o, x)`.
- `rms_norm(dst, src, w, n, eps)`: RMS norm.
- `rope(x, cos, sin, half)`: rotary position embedding.
- `BackendPtr` / factory (`make_cpu_backend`).

Kept device-agnostic so a model can later be split across several backends
(per-layer / per-tensor / per-row) without changing this interface.
