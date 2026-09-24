# `src/backends/backend.hpp` - compute backend interface

Device-agnostic compute abstraction in namespace `backend`. The inference graph
runs its primitive ops through a `Backend` so the same model code targets the CPU
and the Vulkan backend. Operands are a `Buffer` and a float offset (`Slice` / `CSlice`), so the
backend owns its storage and the model never dereferences it. Every op
enqueues on the backend's single implicit stream. The model submits work and
waits on tickets for results and resource lifetimes; `sync()` also drains work
queued behind no ticket on failure paths. This is the device execution model
of `docs/DEVICE-EXECUTION.md`, extended by the implemented batching and
placement contracts in `docs/EXECUTION.md`.

- `alloc(bytes, where)`, `adopt(src, bytes)`, `read(src, off, dst, bytes)`,
  `copy(dst, dst_off, src, src_off, bytes)`: backend-owned storage. `where`
  is `Memory::device` or `Memory::host_visible`; the logits live in the
  latter and the host reads them through `host_ptr()` after a wait, with no
  copy op. `adopt` makes host data reachable without copying on a host
  backend; the source must outlive the handle.
- `write(dst, off, src, bytes)`: host to storage, enqueued, the source
  consumed before it returns. Its caller is the residual stream crossing
  to another device at a placement boundary; nothing else needs one.
- `submit()` returns a monotonic `Ticket` for everything enqueued so far;
  `wait(t)` blocks until that submission has retired. The model submits
  once per forward pass, waits on that ticket for the logits, and waits on
  the last one again when a conversation is reset.
- `sync()`: `noexcept`, like `wait`, and blocks until everything has
  retired, including ops queued behind no ticket. The model calls it on the
  failure paths before returning KV blocks to the pool, which is why it
  cannot throw.

- `set_threads(n)`, `threads_available()`: worker-thread control.
- `memory_available()`: the bytes the backend can still allocate now, as its device or operating system reports them, or nothing when it cannot tell; `resident_bytes(type, nin, rows, bytes)`: what adopting such a matrix keeps resident, its bytes by default. A split over several devices is fitted against both (`model/layer_split.hpp`).
- `matmul(ggml_type, data, X, Y, nin, nout, nbatch, runs)`: the type-generic
  matmul. The quant type is resolved through `quant::Registry`, so every block
  format gets the generic CPU batched fallback. Vendor backends require kernels
  and validation for each supported type. A decode token is the one-column
  case of the same call.
- `RowRun` / `RowRuns`: the rows of a call grouped by the prompt they belong
  to, each run's `end` and its `extent`, the position one past the prompt's
  last token for prompt rows and 1 for a generated token. A device picks a
  row's kernel by its extent rather than by the call's width, so a prompt
  computes the same however its rows are batched; without runs a backend
  chooses by the width. The CPU reads them the same way: a generated token
  takes its decode dots and a prompt's rows the batched path.
- `matmul_group(projections, X, nin, nbatch, runs)`: independent projections sharing
  activations. Each descriptor gives type, weights, output and row count.
  Outputs must be disjoint from one another, inputs and weights. The default
  calls `matmul` sequentially on the same stream; host observation of outputs
  follows the usual `wait`, `sync` or `read` completion contract.
- `matmul_add(...)`: `matmul` whose product is added to what the output
  already holds.
- `matmul_logits(...)`: the output head, by default `matmul`. Its values
  are the logits a caller reads directly, so a backend may keep more
  precise activations for it; the Vulkan backend keeps the 16-bit twin for
  a Q4_0, Q4_1 or Q6_K head.
- `kv_layout()`, `kv_alloc(layers, n_head_kv, head_dim, max_tokens)`,
  `kv_write(layer, views, n_views, k, v)`: the backend-owned half of the
  paged KV cache in `docs/KV-CACHE.md`. The backend chooses the block size and
  the layout inside a block; the model layer hands it `KVView`s (storage
  handle, block table, committed length, `nq` rows of this pass, and the
  rows' extent as in `RowRuns`) and never computes an offset. Rows are laid out in view order and view `v`'s rows go
  to positions `length .. length + nq` of its sequence. Storage is backed on
  demand up to the blocks `max_tokens` needs; it reports retained bytes and
  the peak held during a growth copy.
- `kv_copy(storage, src, dst)`: every layer's K and V of one block into
  another of the same storage, enqueued. Fills a fork's private tail from
  the block it shares up to.
- `attention(Q, layer, views, n_views, out, n_head, n_head_kv, head_dim)`:
  causal GQA over every view, shared by decode and prefill. Queries/output
  have shape `[rows, n_head, head_dim]` in view order; row `b` of view `v`
  sees positions through `v.length + b`, so each table must cover
  `length + nq` positions and every block it reaches must have been written.
  Several views in one call is what a batch of sequences needs; the model
  passes one per entry. The backend owns temporary score storage.
- `Slice` / `CSlice`: where an operand lives, a buffer and a float offset.
  Every op takes these rather than pointers, so nothing outside a backend
  holds a host address. An empty allocation resolves to no address and is
  read by nothing, which is how a zero-length batch passes through.
- `embed(dst, type, table, nin, nrows, ids, count)`: gather `count` embedding
  rows into `dst`, row-major. An op rather than a model-side read because the
  table is a buffer the model cannot address on a device backend. Rejects a
  token id at or beyond `nrows`.
- `rms_norm(dst, src, w, n, eps)`: RMS norm of one row.
- `rms_norm_rows(dst, src, w, rows, n, stride, eps)`: RMS norm of `rows` rows
  against a shared weight.
- `norm_rope_rows(x, rows, stride, heads, w, eps, cos, sin, half, pos)`:
  per-head RMS norm followed by RoPE over a batch of rows. `cos`/`sin` are
  buffers holding the per-position tables; row `r` reads entry `pos[r]` of
  each, so a batch may carry rows from several sequences. The two are one
  op because the model never applies one without the other.
- `norm_rope_kv(...)`: a layer's attention inputs in one op, q normed and
  rotated in place, k normed and rotated into its KV block and v copied
  into its block. The default runs the three ops; the Vulkan backend fuses
  them.
- `silu_mul(dst, gate, up, n)`: the SwiGLU elementwise stage.
- `add(dst, src, n)`: the residual add.
- `gather_rows(dst, src, width, rows, count)`: row `i` of `dst` is row
  `rows[i]` of `src`. Compacts the rows of a pass that want logits, which a
  batch mixing prefill and decode entries leaves non-contiguous, so the
  output head runs once over exactly them.
- `Routing`, `route_experts(scores, rows, n_expert, k, normalize, ids,
  weights)`: a mixture-of-experts layer's routing, the k experts of highest
  softmax probability per row, most probable first and ties to the lower
  id, with their probabilities renormalized over the k when asked. Slot `j`
  of row `r` is entry `r*k + j`; the ids are 32-bit integers in float-sized
  slots, so they stay in the activation arena.
- `matmul_experts(projections, X, nin, nrows, routing, runs)`: up to three
  routed projections of one X, entry `e` of a projection being its expert
  `ids[e]`'s rows times X row `e / k`. A projection's data holds its
  `n_expert` matrices back to back, as a GGUF stacks them.
- `matmul_experts_add(type, data, X, Y, nin, nout, nrows, routing, runs)`:
  the routed down projection joining the residual, row `r` of `Y` adding
  the weighted sum of its k slots, formed in slot order before the add.
- `BackendPtr` / factory (`make_cpu_backend`).

The batched forms exist so the model layer holds no elementwise loops and needs
no host parallelism of its own: one call per layer rather than one per row, or
per head per row. `parallel_for` is therefore NOT on this interface - a host
callback across host threads has no device implementation. It remains public on
`CpuBackend`, which its own tests use.

Multi-device placement and batching across sequences are designed in
`docs/EXECUTION.md` and implemented over this interface.

`run_prefill(work)` invokes the body once on the caller after successful setup
and completes cleanup before returning. Setup or reentrancy errors can reject
the call before body entry. Its default implementation invokes the body directly.
Backends may use this boundary to scope execution policy across all prompt
microbatches without putting platform details in the model layer.

`dot_q8_0` and `matvec_q8_0` are gone. They were Q8_0-specific single-row
leftovers that `matmul` replaced everywhere, and a scalar return per row is
one kernel launch per row on a device. The Q8_0 single-column path survives
as a private detail of the CPU backend, reached through `matmul`.
