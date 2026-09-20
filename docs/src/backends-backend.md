# `src/backends/backend.hpp` - compute backend interface

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
- `kv_layout()`, `kv_alloc(layers, n_head_kv, head_dim, max_tokens)`,
  `kv_write(layer, view, pos, k, v, batch)`: the backend-owned half of the
  paged KV cache in `docs/KV-CACHE.md`. The backend chooses the block size and
  the layout inside a block; the model layer hands it a `KVView` (storage
  handle, block table, committed length) and never computes an offset.
  Storage is backed on demand up to the blocks `max_tokens` needs; it reports
  retained bytes and the peak held during a growth copy.
- `attention(Q, layer, view, out, n_head, n_head_kv, head_dim, nbatch)`:
  causal GQA over the view, shared by decode and prefill. Queries/output have
  shape `[nbatch, n_head, head_dim]`; query `b` sees positions through
  `view.length + b`, so the table must cover `length + nbatch` positions and
  every block it reaches must have been written. The backend owns temporary
  score storage.
- `rms_norm(dst, src, w, n, eps)`: RMS norm of one row.
- `rope(x, cos, sin, half)`: rotary position embedding on one head.
- `rms_norm_rows(dst, src, w, rows, n, stride, eps)`: RMS norm of `rows` rows
  against a shared weight.
- `norm_rope_rows(x, rows, stride, heads, w, eps, cos, sin, half)`: per-head
  RMS norm followed by RoPE over a batch of rows. Row `r` is at position
  `pos0 + r` and reads `cos`/`sin + r*half`. The two are one op because the
  model never applies one without the other.
- `silu_mul(dst, gate, up, n)`: the SwiGLU elementwise stage.
- `add(dst, src, n)`: the residual add.
- `BackendPtr` / factory (`make_cpu_backend`).

The batched forms exist so the model layer holds no elementwise loops and needs
no host parallelism of its own: one call per layer rather than one per row, or
per head per row. `parallel_for` is therefore NOT on this interface - a host
callback across host threads has no device implementation. It remains public on
`CpuBackend`, which its own tests use.

Multi-device placement is planned. Device buffers, resident activations,
and asynchronous execution still require interface changes. Attention is a
backend operation now, but still takes synchronous host pointers.

`run_prefill(work)` invokes the body once on the caller after successful setup
and completes cleanup before returning. Setup or reentrancy errors can reject
the call before body entry. Its default implementation invokes the body directly.
Backends may use this boundary to scope execution policy across all prompt
microbatches without putting platform details in the model layer.

The replacement interface is designed in `docs/DEVICE-EXECUTION.md`. Two
members above do not survive it: `parallel_for` takes a host callback and has
no device implementation, and `dot_q8_0` / `matvec_q8_0` are type-specific,
The replacement interface is designed in `docs/DEVICE-EXECUTION.md`.
`dot_q8_0` / `matvec_q8_0` do not survive it: they are type-specific,
single-row leftovers that `matmul` replaced everywhere except the `bench`
command, and a scalar return per row is one kernel launch per row on a device.
