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
- `matmul_group(projections, X, nin, nbatch)`: independent projections sharing
  activations. Each descriptor gives type, weights, output and row count.
  Outputs must be disjoint from one another, inputs and weights. The default
  calls `matmul` sequentially; all outputs are ready when the call returns.
- `attention(Q, K, V, out, n_head, n_head_kv, head_dim, n_past, nbatch, kv_head_stride)`:
  causal GQA, shared by decode and prefill. Queries/output have shape
  `[nbatch, n_head, head_dim]`. K/V positions are contiguous within each head;
  `kv_head_stride` is the distance between heads in floats, including unused
  capacity. It must hold at least `n_past + nbatch` positions. Query `b` sees
  only positions through `n_past + b`. The backend owns temporary score storage.
- `parallel_for(n, fn)`: run `fn(i)` across the backend's workers.
  Task failure waits for active participants before propagating an exception;
  partially written outputs are not rolled back. Concurrent submission is not
  supported by the current CPU implementation.
- `rms_norm(dst, src, w, n, eps)`: RMS norm.
- `rope(x, cos, sin, half)`: rotary position embedding.
- `BackendPtr` / factory (`make_cpu_backend`).

Multi-device placement is planned. Device buffers, resident activations,
and asynchronous execution still require interface changes. Attention is a
backend operation now, but still takes synchronous host pointers.
