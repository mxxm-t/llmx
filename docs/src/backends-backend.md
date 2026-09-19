# `src/backends/backend.hpp` — compute backend interface

Device-agnostic compute abstraction in namespace `backend`. The inference graph
runs its primitive ops through a `Backend` so the same model code can target CPU
now. ROCm / CUDA / Vulkan / SYCL need the device execution refactor in
`docs/ROADMAP.md` #4a: current operations take host pointers synchronously.

- `set_threads(n)`, `threads_available()`: worker-thread control.
- `dot_q8_0(row, x, nblocks)`: dot product of one Q8_0 block row with `x`.
- `matvec_q8_0(data, x, out, nblocks, nout)`: `out[o] = dot(row_o, x)`.
- `matmul(ggml_type, data, X, Y, nin, nout, nbatch)`: the type-generic matmul.
  The quant type is resolved through `quant::Registry`, so every block format
  gets the batched path and a new type needs no backend change. `nbatch == 1`
  is the single-column case that `Model::matvec` uses, so there is one dispatch
  path rather than two.
- `parallel_for(n, fn)`: run `fn(i)` across the backend's workers. The model
  layer uses it for attention heads instead of creating threads of its own, so
  there is exactly one pool in the process.
- `rms_norm(dst, src, w, n, eps)`: RMS norm.
- `rope(x, cos, sin, half)`: rotary position embedding.
- `BackendPtr` / factory (`make_cpu_backend`).

Multi-device placement is planned. Device buffers, resident activations,
backend attention and asynchronous execution require interface changes.
