# `src/backends/backend.hpp` - compute backend interface

A backend takes one caller at a time: the CPU backend's pool holds one job (`run_parallel`) and a device's command recording one stream of calls, so nothing calls a backend from two threads at once, and the server's encoder runs on the scheduler's thread for that reason ([VISION](../VISION.md), section 10).

`MatrixPaths` records the actual matrix activation forms selected by dispatch: F32, F16, BF16, block-int16 or block-int8. Direct backend conformance tests wait for their work and consume the evidence with the reader in `tests/matrix_precision.hpp`, using `swap_matrix_paths`. Model stages swap in their own evidence and restore the direct caller's set on every exit; model callers read the retained set through `Model::take_matrix_paths` ([runtime](model-runtime.md)). The records are independent of the capability catalog and have no runtime switch.

The six matrix entry points take an explicit activation `Dtype` after their `RowRuns`, defaulting to F16 for direct calls. Base grouped and output-head forwarding preserve it. The value belongs to a call, so models used in turn can share a backend without a mutable backend-wide dtype selection. `native_dtypes` lists complete policies in preference order: the base offers F32; CPU, Vulkan and ROCm offer F16 then F32, and ROCm also offers int8. A Vulkan device that runs int8 lists `Dtype::int8` last, where auto never reaches it; int8 sets the inputs of every matrix product at 8 bits, as every value sets them at its width (docs/PRECISION.md, section 5). This names the available execution policy; its kernels may use packed integer activations or wider F32 inputs. `dtype_path` describes those implementation families, and `emulates_dtype` reports round/widen support without adding it to auto preferences. Placement consumes these capabilities once before constructing a model.

Device-agnostic compute abstraction in namespace `backend`. The inference graph
runs its primitive ops through a `Backend` so the same model code targets the CPU,
Vulkan and ROCm where the backend implements the model's required ops. Operands are a `Buffer` and a float offset (`Slice` / `CSlice`), so the
backend owns device storage and its arithmetic layout. Host-visible results
and staging may be addressed by the model after the required wait. Every op
enqueues on the backend's single implicit stream. The model submits work and
waits on tickets for results and resource lifetimes; `sync()` also drains work
queued behind no ticket on failure paths. This is the device execution model
of `docs/DEVICE-EXECUTION.md`, extended by the implemented batching and
placement contracts in `docs/EXECUTION.md`.

- `Collective` and `join(members, rows, width)`: a tensor group's sum ([TENSOR-SPLIT](../TENSOR-SPLIT.md), section 4.3), made by the group's first member over at most `kMaxTensorWidth` (16) backends of its kind, the one limit the CLI's `--tensor-width` and every backend's collective read, for sums of up to `rows` rows of `width` floats, or null where the backend has none on this build or platform, the default; `collective_bytes_per_row(members, width)`, what a member keeps of the backend's memory for each row of a sum, which the fit counts, 0 by default. `member_order(members, rows, width)`, asked of a group's listed first backend once before any weight loads (`order_members` in `model/place.hpp`), the order the group takes its members in as indices into `members`, the listed order by default; ROCm times it ([TENSOR-SPLIT](../TENSOR-SPLIT.md), section 4.1). `pci_root()`, the PCI root complex above the backend's device, which a tensor group's startup report gives, empty by default.
  `partial(member)` is where a member writes the partial rows of its next sum, storage the collective owns on that member; `sum_into(residual, rows, width)` adds ((p0 + p1) + ...) + p(W-1) of the members' partial rows to every member's residual rows, so every member keeps the same bits, and refuses other members, more rows or another width than it was made for; `sum_in_place(rows, width)` replaces every member's partial rows by that sum, at any width whose rows the reserved floats hold, and refuses more.
  `sum_norm(residual, rows, width, entry)` fuses the sum with the next part's entry norm (`EntryNorm`: each member's norm weight and output, eps, runs, the products that read the output and their dtype, or `gather` alone), leaving each member its stripe of the residual and of the output with the output's twin whole; it returns false with nothing enqueued where the collective does not take the rows, as the base class always does, and only ROCm's takes them ([gpu](backends-gpu.md)).
  The CPU and ROCm implement collectives. Vulkan implements a collective where its sharing path is available; `join` returns null otherwise.
  `send(lane, base, rows, width)` and `finish(lane, base, rows, width, residual)` make the same sum of the partial rows from row `base` in two calls, in place or added to `residual`, so a pass cut in two parts records one part's work between the other part's send and finish; each of the two lanes alternates send and finish, and their rows do not overlap; `drop()`, after a failure between a lane's send and its finish, forgets the lanes sent with each member's later work ordered behind their sums, so the group's later sums are not refused (on Vulkan as a failed sum leaves the collective), nothing by default; `copies_beside(floats)`, whether a lane's sum of that many floats copies beside the members' work, which decides whether a group cuts its prompt passes, false by default and on the CPU, whose members compute as they are called.
- `member_order(devices, width, rows, summed)`: the permutation of the device list whose consecutive `width` devices form the tensor groups in member order, asked of the listing's first backend once before any weight loads ([TENSOR-SPLIT](../TENSOR-SPLIT.md), section 4.1); the listing by default, a timed pick on ROCm, which `times_groups` reports so the command line prints no grouping by PCI root beside it.
- `Handoff` and `handoff(readers)`: a crossing's order on the devices for one host-visible handoff buffer of the sender ([MULTI-DEVICE](../MULTI-DEVICE.md), Execution structure): `serves(reader)`, whether a reader takes this path; `begin` and `sent` on the sender, around its writes of the buffer; `take` and `taken` on a served reader, its copies out of the buffer behind the last `sent` and the mark that it is done with them. Each records work with its queue's next submission and makes no queue call, and the callers order them on the host: a send's `sent` before any `take` of it, and every served reader's `taken` of it before the sender's next `begin`, which a crossing inside a stage keeps by calling both halves in a row (`Model::cross`). Null, the default, serves no reader, so every crossing goes through the host; ROCm serves its own readers, and Vulkan and the CPU none.
- `supports_type(type)`: whether the implemented weight-reading operations
  support that storage type on this device. The default is false. The model checks the
  types of all assigned roles before adoption, and uses the same query to
  decide whether a host layer can stream to its mixer device.
- `alloc(bytes, where)`, `adopt(src, bytes)`, `read(src, off, dst, bytes)`,
  `copy(dst, dst_off, src, src_off, bytes)`: backend-owned storage. `where`
  is `Memory::device` or `Memory::host_visible`; the logits live in the
  latter and the host reads them through `host_ptr()` after a wait, with no
  copy op. `adopt` makes host data reachable by the backend's ops. A backend
  that reads in place (`reads_in_place()`) borrows the source for the
  buffer's life and does not read it inside `adopt`; one that copies has
  consumed the source when `adopt` returns, so the caller may release it
  then.
  `adopt_file(src, bytes, file, offset)` adopts a weight whose bytes are also
  those at `offset` of `file`: a backend that reads in place may read the
  parts an op uses from the file instead (the CPU's expert stacks,
  `docs/src/backends-cpu.md`), and any other adopts `src`.
- `alloc_weight(bytes)`: storage for a weight the caller fills with `write`
  before any op reads it. It need not be zeroed, and a backend keeps it as it
  keeps an adopted weight (Vulkan's padded F32 copies). The default is `alloc`.
  The loader's streamed load allocates every weight a copying backend takes
  this way while the model is built and streams them in after, so a model that
  cannot be placed fails before any weight is uploaded ([load](inference-load.md)).
- `wrap_host(memory, bytes)`: a buffer over the caller's page-aligned memory
  that `copy` reads in place, or null where the backend cannot read it so,
  which is the default. The caller keeps the memory for the buffer's life and
  leaves it unchanged until the copies out of it retire. The CPU aliases the
  memory; Vulkan imports it (`VK_EXT_external_memory_host`). The loader's
  stream wraps each slot of its read ring once, so a device copies each part
  straight out of the slot rather than through its staging.
- `write(dst, off, src, bytes)`: host to storage, enqueued, the source
  consumed before it returns. Callers include residual transfers at placement
  boundaries, uploads of streamed expert weights, and the loader's streamed
  load of every weight a copying backend took, where the backend cannot wrap
  the read ring (`infer::detail::stream`).
- `submit()` returns a monotonic `Ticket` for everything enqueued so far;
  `TicketLedger` keeps a device backend's tickets, the last handed out and the greatest seen retired, which any thread may read while the recording thread hands them out; the Vulkan and ROCm backends both keep theirs in one.
  `wait(t)` blocks until that submission has retired. The model submits
  each device a stage records on at the end of that stage, and the source
  device again at each crossing inside a stage, such as a feed-forward
  block on another device than its mixer or a streamed layer's host
  rows on the way out and back, so a pass on a single device submits once.
  It waits on its pass's last ticket for the logits, and on a sequence's
  last tickets when that sequence is reset.
  A backend is one thread's: every call is made by the thread recording on
  it, but for `wait(t)` of a ticket `submit()` has returned, which another
  thread may make while that one records.
  `done(t)` says whether submission t has retired, without waiting where the backend can tell and by waiting, then true, by default; any thread may ask, as it may wait.
  It may read not done for a ticket whose work has ended while another thread is still in `wait(t)` (the ROCm backend answers so rather than block behind HIP), so a caller asks again rather than treating false as final.
  A collective's `wait(back)` returns once every member is done with what the last sum submitted on it (`back` 0) or the sum before it (1), and does nothing by default, where the members compute as they are called.
  A group's queue calls may be made on a thread of their own ([Vulkan](backends-vulkan.md)): a ticket is waited for there first, and one whose call failed retires without its work, so `raise(t)`, which does nothing by default, throws for it after `wait(t)` and before a caller reads in place what it wrote, as the logits are read, and `read` throws such a failure before it copies.
- `hold_between_submissions(on)`: a model made over several devices asks it
  of each (true) and gives it back when it goes (false), since each waits
  while the others run their parts of a pass; while any request remains,
  the device waits on itself between submissions rather than going idle,
  so its clock stays up. The CPU, and any backend without such a wait,
  ignores it.
- `sync()`: `noexcept`, like `wait`, and blocks until everything has
  retired, including ops queued behind no ticket. The model calls it on the
  failure paths before returning KV blocks to the pool, during failed loading
  before constructor members unwind, and at model teardown before releasing
  owned buffers, which is why it cannot throw.

- `set_threads(n)`, `threads_available()`: worker-thread control; zero leaves
  the backend's current worker count unchanged.
- `memory_available()`: the bytes the backend can still allocate now, as its device or operating system reports them, or nothing when it cannot tell; `resident_bytes(type, nin, rows, bytes, product)`: what adopting such a matrix keeps resident, its bytes by default; `product` is a matrix a product reads as its weights, for which the runtime calls `prepare_matrix(weight, nin, rows)` as it places the matrix, so the backend makes what it counts there rather than in a pass, nothing by default. `allocation_bytes(bytes)`: what `alloc` takes for a buffer of `bytes` as the device rounds it, so a fit counts each cache buffer so; `allocation_limit()`: the most one buffer may take, past which `alloc` fails however much memory is free, or nothing where the device states none (the default), which the KV fit holds every cache buffer to; its bytes by default. `host_resident()`: host memory the backend holds for itself whatever it loads, such as upload staging, none by default. `memory_is_host()`: whether `memory_available()` is the host's free memory, which the host tier and the rest of the host share, as the CPU's is; false by default, and the automatic KV budget then keeps the model's context. `reads_in_place()`: whether `adopt` aliases the caller's memory rather than copying it, so weights placed there cost none of the backend's memory; false by default, true on the CPU. `PassWidths`: what the scratch of one device's passes follows, a pass's rows and the widest shapes its layers and head run there (heads and their width, the longest view, sparse ids and sink entries, the indexer's compression, routed slots and expert rows, the widest product and the widest product of float weights, a hyper-connection mix, and on the head's device its logits rows and their width); each architecture fills the model-wide ones (`ModelPlan::widths`) and the fit adds the rows and logits, the float products from the footprint's matrices and whether the head's is one (`pass_widths`, `float_product_out`). `scratch_bytes(w)`: the scratch passes of widths `w` take beside weights, caches and activations, each buffer rounded as the device allocates it, none by default; `reserve_scratch(w)` makes it once as a model is built, so no pass grows a buffer and none waits beside its successor. `memory_kept(free)`: what a fit leaves free beside that scratch, out of the `free` bytes the fit read, so a fit takes one reading: the runtime's own memory as passes run on a device, which ignores `free`, or the host's share of it on the CPU (`host_reserve`), none by default. A split over several devices is fitted against these (`model/layer_split.hpp`, `budgets_for`).
- `kernel_times()`, for a backend made to time its work (`make_backend`'s diagnostics): device time per kernel since the last reading, the output head's (`matmul_logits`) named with ` head` after the kernel's, the kernels it timed, the device time between each timed op's end and the next one's start and the timed ops whose events gave no time; reading it waits for the device. The default is empty; Vulkan times every dispatch and copy of the interval, the GPU core of ROCm and CUDA the first 4096 kernels, copies and fills ([backends-gpu](backends-gpu.md)). `bench --profile` prints it.
- `host_times()` and `device_ms()`, for a backend made to time its work (`make_backend`'s diagnostics): where its calls held the caller since it was made, waiting on tickets, for a free command slot and for staging, and in uploads apart from those waits; and the device time of the work recorded since the last call, by timestamps, which waits for its queue to read. The defaults report nothing and a negative time, which a backend that computes as it records, the CPU, keeps; the server's `--timing` reads them (`server/scheduler.hpp`).
- `busy_ms(a, b)`: the device time submissions a+1 through b held the device, each from its first operation after its waits on other devices to its last, asked after b retired and read without waiting; negative once the backend no longer keeps them or the device keeps no timestamps. Every backend gives it, so a pipelined prefill prices each stage by its own device time (`SlicePeriods`, [model-slicing](model-slicing.md)); the CPU counts the span from a submission's first dispatch, write or copy to its submit, ROCm a start event recorded after the submission's cross-device waits and its ticket event, Vulkan a timestamp pair a submission.
- `prompt_rows()`: the prompt rows a pass takes on this backend where the caller names none and the fit leaves room ([place](model-place.md), the prompt chunk); 0 by default, the model's `kDefaultUbatch`, and the GPU core's device profile number (`DeviceProfile::prompt_rows`) on ROCm and CUDA.
- `is_cpu()`: whether the backend is the CPU itself, which experts on the CPU (`infer::adds_host_for_experts`) ask so that no second CPU is placed beside it, and which the split's fit does not ask; false by default, true on the CPU, and a device that reads in place still answers false.
- `unordered(on)`: while on, the ops recorded need not run in order, the caller's promise that none reads or writes what another writes; turned off, everything recorded before runs before what follows. The promise covers only the buffers the caller passes: a backend that records still orders each op's own steps and its own scratch across calls (twins, partial sums, groupings). A rerun of many state layers records them so (`docs/SPECULATIVE.md`, section 7), and qwen4exp its independent products ([qwen4exp](model-arch-qwen4exp.md)); by default nothing, as a backend that computes as it records needs.
- `decode_columns()`: the generated tokens a decode pass's products read each weight once for, past which another row costs a weight read of its own, what a server's drafts ride on beside other requests' rows (`docs/SPECULATIVE.md`, section 3); 1 by default, the CPU's.
- `identity()`: what makes the backend's arithmetic its own, as text, so a history one backend computed continues on another of the same identity and build (`docs/DISK-TIER.md`, The entry file); by default an identity no other backend shares.
- `row_class(extent)`: a key for rows of that `RowRun` extent, equal for two extents only where every op of the backend takes the same arithmetic for both, so they give the same bits; by default every extent is a class of its own, and a backend claims more only where it proves it (`docs/SPECULATIVE.md`, section 1).
- `matmul(weight, X, Y, nin, nout, nbatch, runs, dtype)`: the type-generic
  matmul. The quant type is resolved through `quant::Registry`, so every registered
  block decoder gets the generic CPU batched fallback; storage metadata alone
  does not enable it. Vendor backends require kernels
  and validation for each supported type. A decode token is the one-column
  case of the same call.
  Rows are sized by `quant::row_bytes`, so a row that ends inside a block is refused by every op on every backend.
- `RowRun` / `RowRuns`: the rows of a call grouped by the prompt they belong
  to, each run's `end` and its `extent`, the position one past the prompt's
  last token for prompt rows and 1 for a generated token. A device picks a
  row's kernel by its extent rather than by the call's width, so a prompt
  computes the same however its rows are batched; without runs a backend
  chooses by the width. The CPU reads them the same way: a generated token
  takes its decode dots and a prompt's rows the batched path.
- `for_each_run(n, runs, key, each)`: the one reader of `RowRuns`.
  It checks that the runs are in row order and end at row `n` before any callback, so a malformed list reaches no rows.
  Then it calls `each(first, count, key)` over each stretch of adjacent runs whose `key(run)` is equal, which is how each backend groups the rows that take one kernel.
  A call without runs is left to the caller, which chooses by the call's width.
- `matmul_group(projections, X, nin, nbatch, runs, dtype)`: independent projections sharing
  activations. Each `Projection` holds a `WeightView` (type, data, scales, scale count and scale shift), output and row count, and for `matmul_experts` and `moe_step` a row window of each expert matrix, `rows` rows from `first` in matrices of `pitch` rows (0 for `rows`), so Inkling-2-Mini's fused gate-up stack is read as its gate and up halves without a copy; a window past its matrix or over an F8_E4M3 grid is refused, and only the CPU takes one (`Op::expert_window`).
  Its three-argument constructor keeps the weight view grouped, so the old flat initializer cannot silently fill the scale fields.
  A projection may name its own input rows (`in`), of the call's shape, in place of `X` (`input_of`); the default, the CPU and Vulkan run such a group as a `matmul` a projection (`own_inputs`), and the GPU runs one column of two inputs as one pair launch (`docs/src/backends-gpu.md`).
  Outputs must be disjoint from one another, inputs and weights. The default
  calls `matmul` sequentially on the same stream; host observation of outputs
  follows the usual `wait`, `sync` or `read` completion contract.
- `matmul_add(weight, X, Y, nin, nout, nbatch, runs, dtype)`: `matmul` whose product is added to what the output
  already holds.
- `matmul_logits(weight, X, Y, nin, nout, nbatch, runs, dtype, cap)`: the output head, by default `matmul`. Its values
  are the logits a caller reads directly, so a backend may keep more
  precise activations for it; the Vulkan backend keeps the 16-bit twin for
  a Q4_0, Q4_1 or Q6_K head.
  A `cap` above 0 softcaps the logits as `softcap` does, which a device may apply to the head's output before it leaves device memory.
- `kv_layout()`, `kv_alloc(layers, n_head_kv, head_dim, max_tokens, k_type,
  v_type, auxiliary_widths = {}, auxiliary_pools = {}, layer_shapes = {})`, `kv_write(layer, views, n_views, k, v,
  auxiliary = {})`: the backend-owned half of
  the paged KV cache in `docs/KV-CACHE.md`. Each side is stored as
  `KVType::f32` or `KVType::f16` (the CLI's `--cache-type-k` and
  `--cache-type-v`), and a backend without a type throws rather than
  substituting one. `kv_type_of(name)` and `kv_type_name(type)` hold the
  three names, `f32`, `f16` and `native`, the same on every backend; an unknown name
  throws. `KVType::native` is a model's declared cache format, which the model resolves before its storages (`settle_cache` in `model/runtime.hpp`), so a storage is handed f32 or f16. The backend chooses the block size and
  the layout inside a block; the model layer hands it `KVView`s (storage
  handle, block table, committed length, `nq` rows of this pass, and the
  rows' extent as in `RowRuns`) and never computes an offset. Rows are laid out in view order and view `v`'s rows go
  to positions `length .. length + nq` of its sequence. Storage is backed on
  demand up to the blocks `max_tokens` needs, or all at once by
  `back_all()`; it reports retained bytes and the peak held during a growth
  copy.
  `auxiliary_widths` is empty for zero width on every layer or supplies one F32 width per KV layer.
  `kv_write` requires its token-major `[rows, width]` source on enabled layers, refuses one on disabled layers, and validates all views and sources before writing.
  The optional plane uses the same block ids as K and V; `select_kv` reads it for QSA, where qwen4exp keeps its raw indexer keys.
  `auxiliary_pools` is empty or supplies, per layer, an `AuxiliaryPool`: the groups of `tokens` whose pooled rows of `width` floats each block also keeps after its token rows (0 tokens for none), which `pool_kv` and `select_kv` write.
  A layer of no heads keeps no K or V and holds pooled rows alone, as an entries layer's are.
  `layer_shapes` is empty for `n_head_kv` heads `head_dim` wide on every layer, or supplies one `KVShape` per layer, its heads and width, as a drafter whose layers differ from its target's takes; every op on a layer reads that layer's shape, and a backend refuses a width its kernels do not take on any layer.
  The backends here derive their storage from `BlockKVStorage` (`backends-kv_storage.md`), which holds the buffers, the accounting, the growth rule, the whole-budget check and the view checks once.
- `AttentionTerms`, passed as the optional last argument `terms` of `attention` and `gated_attention`: extra terms of the causal softmax by query row r (in the call's view order) and head h.
  Where `rel` has data, the scaled score of a key at distance d = position(r) - key position gains sum_i rel[r][h][i] * table[i][d] for d below `extent`, `rel` being `[rows][n_head][rank]` and `table` `[rank][extent]`; where `sink` (`[rows][n_head]`) has data, exp(sink[r][h]) joins the denominator as a key of zero value, its logit counted in the row maximum.
  The scale stays 1/sqrt(head_dim); a member of a tensor group passes its own heads' rows, so `rel` and `sink` split with the heads and `table` stays whole.
  `attention_terms_rows` refuses terms on a non-causal view and a bias without its rank, extent or table; empty terms give the bits of none.
  The CPU adds the bias after the q.k scores, takes the row maximum again, and adds the sink's exp to the block sums' total (Inkling-2, docs/INKLING2.md); the GPU core and Vulkan refuse terms until their forms land (inkling-core steps 3b and 3c).
- `gated_attention(Q, layer, views, n_views, out, n_head, n_head_kv, head_dim, g)`: `attention`, then `sigmoid_mul` of the output in place, `OutputGate` holding the gate, its rows, strides and runs; the default makes the two calls, and a backend overrides it to skip what only the ungated output needed (the GPU core writes no twin of it).
- `gated_attention_layer(in, layer, views, n_views, out, n_head, n_head_kv, head_dim, g)`: a gated attention layer from its projections, `GatedLayer` holding the q-and-gate rows, k, v, the norms' weights, the rope tables, positions and streams; the default makes the four calls, `norm_rope_partial` of q and of k, `kv_write` and `gated_attention`, and a backend that forms q and k on chip may leave `in.q` and `in.k` unwritten (the GPU core does for a decode row).
- `attention(Q, layer, views, n_views, out, n_head, n_head_kv, head_dim)`:
  causal GQA over every view, shared by decode and prefill. Queries/output
  have shape `[rows, n_head, head_dim]` in view order; row `b` of view `v`
  sees positions through `v.length + b`, so each table must cover
  `length + nq` positions and every block it reaches must have been written.
  Several views in one call is what a batch of sequences needs; the model
  passes one per entry. The backend owns temporary score storage.
- `pool_kv(layer, views, n_views, index, key_norm, cos, sin)`: the complete groups the views' rows complete, their raw keys averaged, normalized with `key_norm` and rotated at the group's first position, into their pooled rows of the auxiliary blocks.
  `select_kv` pools its own rows' groups; a caller that writes a pooled layer's auxiliary rows without selecting them, as qwen4exp's MTP context rows do, pools them here, so every complete group of a history has its row.
  A layer's pool must be the selection's compression. A backend with `select_kv` has it.
- `select_kv(Q, layer, views, n_views, ids, index, key_norm, cos, sin)`:
  QSA's causal selector. `Q` is already normalized and RoPE-rotated with shape
  `[rows, index.heads, index.dim]`; each view supplies its contiguous visible
  cache prefix and current rows. `IndexArgs` gives heads, key dimension,
  rotary dimension, compression ratio, token budget and RMS epsilon. `key_norm`
  is the stored GGUF multiplier (already including the one added to the HF
  zero-centered gamma); `cos` and `sin` hold `rope_dim / 2` values per
  position. The operation pools each complete group of raw auxiliary F32 keys
  once per view, applies RMSNorm and split-half RoPE at the group's first
  position, then scores each causal query by summing per-head ReLU dot products
  divided by `sqrt(dim)`. It selects at most `budget / compression` groups,
  emits their token IDs in ascending chronological order, appends the visible
  partial tail, and fills the remaining `budget + compression - 1` U32 slots
  with `UINT32_MAX`. Equal scores prefer the earlier group, a deterministic
  contract because HF `topk` does not specify tied-index order. Backends
  validate dimensions, views and operand spans before changing `ids`. The
  default throws; CPU and Vulkan override it. This primitive does not perform
  sparse attention over the selected IDs.
- `sparse_attention(Q, layer, views, n_views, ids, id_stride, out, n_head,
  n_head_kv, head_dim)`: causal GQA over original, full-resolution K/V rows
  selected by `select_kv`. Q and `out` have `[rows, n_head, head_dim]` shape;
  `ids` contains `id_stride` U32 values in float-sized slots for each row, in
  view order. IDs are chronological and unique; `UINT32_MAX` padding and any
  ID at or beyond that row's causal end are ignored before KV access. Each
  query head uses its grouped KV head and a stable softmax over selected keys.
  K and V independently support F32 or F16 storage. A row with no valid IDs
  writes zero. Backends preflight dimensions, views, spans and output aliases
  before writing. The default throws; CPU and Vulkan override it.
- `KVView::causal`, true by default: a view whose `causal` is false lets every row of `attention` attend all `length + nq` positions of the view, as a vision encoder's rows see their whole image (`docs/VISION.md`, section 3); the encoder writes an image's K and V through `kv_write` and attends through the same `attention`, so attention keeps one owner. `implements(Op::attention_full)` says a backend reads such a view.
- `KVView::window`, 0 by default: a row at position p attends only positions above p - window, as a DFlash drafter's block reads its context rows (`docs/SPECULATIVE.md`, section 11); a non-causal row's span runs from there through the view's last row, as a DFlash2 block's rows each see the whole block. `implements(Op::attention_window)` says a backend reads it, the CPU backend, the GPU core (ROCm and CUDA) and Vulkan; the CPU kernel starts each row's walk at its first position, so a window at or past a row's extent gives the bits of none.
- `Slice` / `CSlice`: where an operand lives, a buffer and a float offset.
  Device arithmetic uses these handles; host-visible results and transfer
  staging may expose host addresses. An empty allocation resolves to no address and is
  read by nothing, which is how a zero-length batch passes through.
- `overlaps(a, an, b, bn)`: whether `an` floats at `a` and `bn` floats at `b` share a float of one buffer, the one check every backend makes before writing an output beside its inputs.
- `span(buffer, off, bytes)`, `size_mul`, `size_add`, `blocks_for(tokens,
  block_tokens)`: the checks every backend makes, a byte range inside a
  buffer and size arithmetic that throws rather than wraps, and the whole
  KV blocks a number of positions needs.
- `embed(dst, weight, nin, nrows, ids, count)`: gather `count` embedding
  rows into `dst`, row-major. An op rather than a model-side read because the
  table is a buffer the model cannot address on a device backend. An id at or
  beyond `nrows` writes a zero row, as `embed_ids` does, so a member of a
  table split by rows gathers the ids it does not hold as zeros; the pass
  refuses a token id past the embedding's rows before it runs.
- `embed_ids(dst, weight, nin, nrows, ids, count)`: `embed` with the ids
  read on the device, 32-bit integers in float-sized slots, as an embedded
  drafter's chain feeds a draft back (docs/SPECULATIVE.md, section 7); an id
  of `nrows` or more writes a zero row and reads no row of the table.
- `argmax_rows(ids, logits, rows, n, after, skip)`: each row's largest logit's id,
  ties to the lowest, into `ids` as a 32-bit integer in a float-sized slot;
  `n` where the largest is not finite or a logit is NaN, and `n` where the
  row's id in `after`, when given, is `n` or more, so a draft chain ends at
  its first invalid draft. An id `skip` below `n` is passed over as if the
  row lacked it, as a draw under `ignore_eos` masks the end token. Both throw
  by default, naming the op.
- `argmax_part(cands, logits, rows, n, first, slot, slots, skip)` and
  `argmax_pick(ids, cands, rows, slots, n, after)`: `argmax_rows` over a
  vocabulary split into `slots` slices on a tensor group's members, around a
  sum of the candidates. `argmax_part` writes each row's candidate from one
  slice whose first id is `first` at slot `slot` of a row of zeros: the
  largest logit, -inf counted, `first` plus the lowest id holding it as a
  float value, and 1 where a logit is NaN or the largest is +inf; the row's
  id `skip` is passed over where the slice holds it (`part_skip`).
  `argmax_pick` gives `n` where a flag is set or the largest is not finite,
  else the largest's id, ties to the lowest, then `after`'s rule, so the id
  is `argmax_rows`' over the whole row. Ids travel as float values, exact
  below `kExactIds` (2^24). Both throw by default, naming `argmax_rows`.
- `rms_norm(dst, src, w, n, eps)`: RMS norm of one row, not virtual:
  `rms_norm_rows` over that one row.
- `rms_norm_rows(dst, src, w, rows, n, stride, eps, runs)`: RMS norm of `rows` rows
  against a shared weight. `runs`, when given, are the row runs of the matmul
  that reads `dst` next, so a device can also write `dst` in the form that
  matmul's kernel reads.
- `norm_rope_partial(dst, src, rows, src_stride, src_head_stride, heads, head_dim, rope_dim, w, eps, cos, sin, pos, form, streams)`:
  per-head RMS norm over `head_dim`, then RoPE on `rope_dim` dims of it
  only, over a batch of rows, from heads read at a stride of their own into
  contiguous heads, so qwen35 reads q between its gates in `attn_q`'s rows.
  `cos`/`sin` are buffers holding the per-position tables of `rope_dim / 2`
  entries; row `r` reads entry `pos[r]` of each, so a batch may carry rows
  from several sequences. At the full width over contiguous heads, in place,
  it is Qwen3's rope; for qwen35's text the rope sections give every
  frequency the same position, so they reduce to it. The two are one op
  because the model never applies one without the other.
  `RopeForm` says which dims and which way: the plain form, its default, rotates pairs (j, j + rope_dim / 2) of the head's first `rope_dim` dims; `offset` starts the rotated dims later in the head, `adjacent` takes pairs (2j, 2j + 1) as DeepSeek-V4.1's complex rotation does, and `inverse` turns each pair by minus its angle, which removes a query's rotation from the attention output.
  The CPU, Vulkan and ROCm run every form.
- `norm_rope_kv(...)`: a layer's attention inputs in one op, q normed and
  rotated in place, k normed and rotated into its KV block and v copied
  into its block. The default runs `norm_rope_partial` at the full width on
  q and k in place, then `kv_write`; the Vulkan backend fuses them.
- `silu_mul(dst, gate, up, n, glu, runs)`: the SwiGLU elementwise stage, `glu` (`Glu`) holding the GLU's `limit`, `form` and `up_limit`; with a limit L > 0 silu(min(gate, L)) * clamp(up, -L, L) as DeepSeek-V4.1's experts take it, and with 0 unclamped; a negative or infinite limit is refused.
  `form` (`GluForm`) is `silu` by default; `quick_gelu`, MiniMax-M3.1's, takes (up' + 1) * gate' * sigmoid(1.702 gate') after the same clamps, which only the CPU has.
  `softcap`, Inkling-2-Mini's, is L tanh(gate / L) * sigmoid(gate) * U tanh(up / U) with no clamp, L the limit and U `up_limit`, both positive; an up limit in another form is refused, and only the CPU has it (`Op::glu_softcap`).
- `softcap(x, n, cap)`: x = cap tanh(x / cap) in place, Inkling-2-Mini's final logit softcap; only the CPU has it (`Op::softcap`). `runs`,
  when given, group `dst`'s rows by prompt for the matmul that reads it next,
  as for `rms_norm_rows`; a routed layer passes its entries' runs.
- `add(dst, src, n)`: the residual add.
- `gather_rows(dst, src, width, rows, count)`: row `i` of `dst` is row
  `rows[i]` of `src`. Compacts the rows of a pass that want logits, which a
  batch mixing prefill and decode entries leaves non-contiguous, so the
  output head runs once over exactly them.
- `Routing`, `RouteRule`, `route_experts(scores, rows, n_expert, k, rule, ids,
  weights)`: a mixture-of-experts layer's routing, the k experts the rule ranks highest per row, highest first and ties to the lower id.
  Each rule carries `eps`, which its factory sets and which joins the sum of the picks' weights before a normalizing rule divides by it: 0 for softmax, 1e-20 for sqrt_softplus and the caller's for sigmoid.
  `RouteRule::softmax(normalize)` ranks by softmax probability and weighs by the probabilities, renormalized over the k when asked.
  `RouteRule::sqrt_softplus(bias, scale, normalize)`, DeepSeek-V4.1's, scores sqrt(softplus(x)), softplus being x above 20 as torch's threshold has it, ranks by score + bias and weighs by the unbiased scores over their sum + 1e-20 when `normalize` and k > 1, then times `scale`; a row with a score + bias that is not finite has no candidate.
  A biased rule's `bias_stride`, 0 by default, gives each row its own bias, row r's at `bias + r * bias_stride`; DeepSeek-V4.1's image rows take `exp_probs_b_vl` this way, gathered per row, while a stride of 0 is the shared bias and its bits ([DEEPSEEK4](../DEEPSEEK4.md)).
  `RouteRule::sigmoid(bias, scale, normalize, eps)`, MiniMax-M3.1's and Inkling-2-Mini's, scores sigmoid(x) and ranks and weighs as sqrt_softplus does, the weights over their sum plus `eps` at every k: 0 for MiniMax-M3.1 ([MINIMAX](../MINIMAX.md)), whose quotient it leaves bit for bit, and 1e-20 for Inkling-2-Mini.
  The CPU, Vulkan and the GPU core (ROCm and CUDA) run all three rules; a backend given a rule it lacks, or a softmax rule given a bias or a scale, refuses it by name (`RouteRule::softmax_only`). Slot `j`
  of row `r` is entry `r*k + j`; the ids are 32-bit integers in float-sized
  slots, so they stay in the activation arena. A row whose probabilities are
  not finite has no candidate: its k ids are `kNoExpert` with weights 0, and
  the routed products leave such an entry's rows unwritten and add nothing for it.
  A probability below FLT_MIN counts as 0, so such picks go by lower id on every
  device whatever its denormal mode, where HF orders them by value; their weights
  are below 1e-38, so no output changes.
- `matmul_grouped(p, X, nin, rows, groups, group_ids, runs, dtype, scratch)`: DeepSeek-V4.1's grouped output projection, input g of each row projected by the g-th stack of `p.rows` rows of p's one matrix, as `matmul_experts` at k = 1 over the (row, group) pairs with `group_ids` giving each pair its group and each pair in its token's run, so the placed matrix is read as it lies and the output row is the next projection's input.
- `matmul_experts(projections, X, nin, nrows, routing, runs, dtype)`: up to three
  routed projections of one X, entry `e` of a projection being its expert
  `ids[e]`'s rows times X row `e / k`. A projection's data holds its
  `n_expert` matrices back to back, as a GGUF stacks them.
  The routed calls require `implements(Op::routed_experts)`, true on the CPU, Vulkan and ROCm, which every routed layer's plan declares; differing projection types also require `implements(Op::mixed_experts)`, true on the CPU and false on Vulkan. Architecture plans declare this requirement for their grouped gate/up pair; a separately typed down projection does not need it.
- `matmul_experts_add(weight, X, Y, nin, nout, nrows, routing, runs, dtype, from)`:
  the routed down projection joining the residual, row `r` of `Y` adding
  the weighted sum of its k slots, formed in slot order before the add.
  With `from`, each row of `Y` is first the buffer's first `nout` floats (`repeat_row`), which a device may read in the sum's launch instead of copying them first.
- `RopeStreams` gives `norm_rope_partial` several positions a row: row `r`'s stream `k` is `pos[r * count + k]` and frequency `j` of the rope's half turns by stream `of[j]`, as Qwen's three position components and a vision encoder's patch row and column do; one stream is the plain rope, and a `w` with no buffer takes the rope without the norm. `implements(Op::rope_streams)` says a backend reads more than one stream or no norm.
- The vision encoder's ops (`docs/VISION.md`, section 3), each with a base refusal until a backend implements it and an `Op` that says it does: `layer_norm_rows(dst, src, w, b, rows, n, stride, eps)`, the mean and biased variance of each row then `(x - mean) / sqrt(var + eps) * w + b`; `gelu(dst, src, n, form)`, `GeluForm::erf` or `GeluForm::tanh`; `add_bias(dst, bias, rows, width)`, a bias row added to each row; and `mix_rows(dst, table, width, table_rows, idx, weight, rows, taps)`, each row adding the weighted sum of its taps' table rows, taken in order, a row past the table refused.
- `delta_step(d, mixed, nin, rows, layer, views, n_views, runs, dtype)`: a linear-attention layer's step up to its output product, `DeltaStep` its operands: the qkv product, then the gate, alpha and beta products as one group, both unordered, `causal_conv_silu` into `d.conv`, `gated_delta_rule` into `d.out` and `gated_rms_norm` of `d.out` under the sigmoid of the gate rows; the CPU and Vulkan backends run this base, and the ROCm backend fuses it where its limits hold ([backends-gpu](backends-gpu.md)), leaving every slot but the conv's with the base's bits.
- `moe_input(m, h, nin, rows, runs)`: the step's input where `m.input` (`MoeInput`) is set, its `src` RMS-normed against `norm` into `h` by `rms_norm_rows`; the base router runs it first, every fused `moe_step` path forms `h` before reading it, and a device may fold it into its routing launch where it gives the same bits.
- `moe_router(m, h, nin, rows, runs)`: the step's routing after its input, the router's F32 product into its scores and `route_experts` under `m.rule`; a device may form both in one launch where it gives the same bits.
- `glu_products(gate, up, act, glu, X, nin, rows, runs, dtype)`: the dense form of `moe_act`, `matmul_group` for gate and up and `silu_mul` at `glu` into `act`, which `blocks::swiglu` and `moe_step`'s shared expert call; a device may form the act in one launch where it gives the same bits, leaving `gate.out` and `up.out` unwritten.
- `moe_act(m, h, nin, rows, routing, runs, dtype)`: the step's routed act after its routing, `matmul_experts` for gate and up and the routed `silu_mul` at `m.glu` into `m.act`; a device may form the act in one launch where it gives the same bits, leaving `m.gate.out` and `m.up.out` unwritten.
- `moe_step(m, h, y, nin, rows, runs, dtype)`: a routed feed-forward step, `MoeStep` its operands: the F32 router into its scores, `route_experts` under `m.rule`, `matmul_experts` for gate and up, the routed `silu_mul` at `m.glu` over runs of `end * k` (built in `m.scratch`) and `matmul_experts_add` into `y`, then, with `m.shared` (`MoeShared`), the row gate's F32 product in the step's dtype where the expert has one, `matmul_group` for its gate and up, `silu_mul` at its `glu`, `sigmoid_mul` by the row gate, and `matmul_add` into `y`; `m.zero`, a row of zeros, clears `y` first through `repeat_row`. Its projections and down weights carry `WeightView`s, including optional scale storage, through both the base operations and device override; Vulkan's fused step refuses a rule other than softmax and a clamp by name before it builds, and ROCm, which keeps the base, refuses them in its `route_experts` and `silu_mul`.
  The base runs exactly those calls in that order, so a backend that does not override it, the CPU among them, computes what the routed layers computed before the op; a device overrides it where it fuses the step with the same bits (`backends-vulkan.md`, the fused routed step), and calls the base otherwise.
  With `m.slots` (`ExpertSlots`, a layer's device expert cache) a device that `caches_experts()` routes as without slots, then pulls the step's missing experts into the layer's slots through its ring in waves and runs the entries in rounds of what the slots hold, each round's down products kept apart until one combine in entry order, so the experts run through the same kernels over the same bytes as with every expert on the device ([expert_ring](backends-expert_ring.md)); the base throws, and Vulkan, ROCm and CUDA have it.
  `ExpertSlots` owns its state's layout, `owner_at`, `age_at`, `counts_at` (the picked experts found in slots, the experts the waves pulled and each ring half's last wave's pulls), `flags_at` and `words`, which the model allocates and reads its counts by and the backend pushes to its kernels, and `RingCounts`, the ring reader's counts a layer; `Memory::host_coherent` is the host memory the ring's exchange and entries take.
- `repeat_row(dst, row, rows, width)`: copies a row of `width` floats into each of `rows` rows of dst, then doubles the copied rows, so clearing rows takes no host buffer or per-token upload; the base `moe_step` and qwen4exp's injection into a tensor group's partial rows use it.
- The ops of the qwen35 layers, whose math is in [QWEN35](../QWEN35.md), The forward pass, with `norm_rope_partial` above.
  The CPU (`backends-cpu.md`) and Vulkan (`backends-vulkan.md`) implement them. ROCm does not advertise the qwen35 state ops, so placement there is refused before execution.
  - `Op` names these five, the routed products and the mixed routed projection capability above, the embedded drafter's `argmax_rows` and `embed_ids`, `depthwise_conv` in its PLE subset (a tail, SiLU, a given residual, divisor 1), QSA `select_kv` and `sparse_attention`, the vision encoder's six (`layer_norm`, `gelu`, `add_bias`, `mix_rows`, `rope_streams`, `attention_full`), DeepSeek-V4.1's ten (`hc_coeffs`, `hc_collapse_norm`, `hc_expand`, `sparse_attention_sink`, `topk_rows`, `candidate_blocks`, `compress_pool`, `indexer_scores`, `model_format_qdq`, `pool_write`), which the CPU, Vulkan and ROCm implement, three modes of other calls that all three have: `sqrt_softplus_routing` (`route_experts` under `RouteRule::sqrt_softplus`), `silu_mul_clamp` (`silu_mul` with a limit) and `rope_forms` (`norm_rope_partial` in a form other than the plain one), and `sigmoid_routing` (`route_experts` under `RouteRule::sigmoid`), `glu_quick_gelu` (`silu_mul` in `GluForm::quick_gelu`), `raw_index_scores` (`indexer_scores` in `IndexForm::raw_blocks`), `sparse_attention_blocks` (`sparse_attention` over id groups or blocks above 1) `glu_softcap` (`silu_mul` in `GluForm::softcap`) and `expert_window` (a routed `Projection` with a row window), which only the CPU has, and Inkling-2-Mini's `softcap`, `l2_clip`, `query_mix`, `tanh_gate_mix`, `depthwise_conv_ring` (`depthwise_conv` in any form but the PLE subset), `attention_rel_sink` and `copy_rows`, which only the CPU has so far; `op_name(op)` spells each one. `implements(op)` is false unless the backend says otherwise; both backends support the five qwen35 ops, the drafter's two, `depthwise_conv`, `select_kv` and `sparse_attention`.
    A model's plan names the ops each part issues from this list, and the model refuses at load a placement that puts a part on a backend without one of them (`model-runtime.md`); a stream destination without one of a routed feed-forward part's ops leaves that layer on its host.
    It is the seam an architecture's new op comes through while one backend lacks it (`docs/ADDING-AN-ARCHITECTURE.md`, New backend ops); the five qwen35 ops are pure virtual, while optional additions such as `select_kv` and `sparse_attention` keep the base refusal until implemented by a backend.
    `sparse_attention` takes `groups` and `block_size`, both 1 by default: a row's ids are `groups` runs of `id_stride` slots, group-major, KV head h reading run h / (n_head_kv / groups), and an id names `block_size` positions in order, each masked by the row's causal end, so at 1 and 1 the slots are today's.
  - `StateShape`: the common linear-attention state prefix for one sequence, K and V heads and their widths; `channels()` is the conv's channel count, the width of the raw projection row `[q | k | v]`, and `slot_floats()` counts every V head's `k_dim x v_dim` matrix laid out `[K row][V column]`, then the conv's `kConvTaps - 1` carried raw rows, oldest first, all F32.
    `kConvTaps` is the conv's width, 4 in every qwen35 file, and `kL2NormEps` the L2 norms' epsilon, 1e-6, which no file carries.
  - `state_alloc(layers, slots, shape, tail_floats)`: a `StateStorage` of one buffer per layer holding every slot back to back, allocated through `alloc` and zero-filled when it is made and never grown, so no pass allocates state. `tail_floats` is optional; when supplied it has one count per layer. Each slot then holds the common `StateShape` prefix followed by that layer's additional F32 values, so layers without a tail do not pay for one. A shape of all zeros holds the tails alone, every layer having one, as entries layers' slots do. `slot_floats(layer)` and `slot_bytes(layer)` report the complete stride for that layer.
    It refuses zero layers, zero slots, a shape with zero K or V heads or a zero width, and V heads that are no multiple of the K heads.
    It is the same on every backend, over its own `alloc`.
    It refuses a tail list with the wrong layer count, arithmetic overflow, a missing buffer or one smaller than that layer's full slot strides, so a storage made by hand cannot send an op past a buffer's end. `check_state_views` also requires matching full per-layer strides across views. State ops use the common prefix; an architecture-specific tail remains in the same slot and is untouched unless that architecture updates it.
  - `StateView`: one sequence's `nq` rows of a pass continuing a history of `length` tokens, read from slot `src` and written to slot `dst`, which differ only in a verify.
    Length 0 reads a zero state whatever `src` holds, so a recycled slot needs no clearing.
    `check_state_views(views, n_views, layer)` is every backend's check of them: rows in every view, the layer and slots inside a storage, one shape, and no slot written by one view and read or written by another, so the views do not depend on their order.
  - `causal_conv_silu(out, x, w, layer, views, n_views)`: the causal conv of width `kConvTaps` over each view's carried rows and rows, then SiLU, with `w` as `ssm_conv1d` stores it and rows before a sequence's start zero; each view leaves its last raw rows in slot `dst`.
  - `gated_delta_rule(out, qkv, alpha, b, a, dt_bias, layer, views, n_views)`: the L2 norms of q and k, beta and the decay, then the recurrence token by token for every (view, V head), V head `j` reading K head `j mod k_heads`, from slot `src`'s matrices into slot `dst`'s; the decay is 0 below 2^-126.
  - `gated_rms_norm(dst, x, z, w, rows, heads, dim, eps, runs, gate)`: the RMS norm of each head of `x` against `w`, which every head shares, times `NormGate::silu` (the default) or `NormGate::sigmoid` of `z`. Qwen35 keeps SiLU; Qwen4Exp selects sigmoid.
  - `sigmoid_mul(dst, x, gate, rows, heads, dim, gate_stride, gate_head_stride, runs)`: `x * sigmoid(gate)`, the gate addressed by row, head and element: the output gate reads each query head's gate in place, and a scale of one value per row is `heads` the width, `dim` 1 and `gate_head_stride` 0.
  - `l2_clip(items, rows, runs)` over up to `kClipItems` (4) disjoint `ClipRows` tensors `{x, scale, row_scale, heads, dim, stride, threshold}`, which the GPU backend clips in one launch, `query_mix(items, rows, dim, stride, eps, runs)` over up to `kQueryMixItems` (2) `QueryMixRows` mixes `{out, src, queries, n, nq, norm, norm_eps}` whose outputs meet no other operand, which the GPU backend runs in one launch, a mix with a `norm` (one weight a query, up to `kQueryMixNorms` (3) queries) written as `rms_norm_rows` of each query over rows `nq * dim` apart, and `tanh_gate_mix(items, rows, width, out_stride, h_stride, m_stride, runs, eps)` over `GateMixItem` mixes `{out, h, m, f, g, norm, forget, gain, rank, dtype, down, down_w}` whose outputs meet no other item's operands, an item naming `forget` and `gain` F32 products having the call compute f and g first from their own inputs (`gate_products`, one `matmul_group` for up to three items, on the CPU and Vulkan), and one naming a `down` step (a `NormRows` with its product) having the call form those inputs first as one `group_rms_norm` of two groups of the width against `down_w` with the call's eps (`gate_downs`): Inkling-2-Mini's per-head scale, clip to an L2 norm and row scale in place, its attention-residual mix of a row's n unweighted-RMS-normed items by nq learned queries, and its message gate's tail `(1 - tanh f) h + tanh(g) m`, then with a `norm` the in-place `rms_norm_rows` of it, which the GPU backend runs in the held norm's launch for packed rows of up to 4096 values apart from the inputs, up to three items whose outputs lie one after another a launch, keeping one twin of all their rows that a product of any one item's rows reads (`llmx_tanh_norm_16`) (docs/INKLING2.md, section 5); the CPU, GPU and Vulkan backends implement them.
  - `copy_rows(dst, dst_stride, src, src_stride, rows, width, add)`: dst row r becomes src row r, or gains it when `add`, rows of `width` floats a stride apart, as Inkling-2-Mini moves rows between its residual bundle and contiguous slots (docs/INKLING2.md, section 3); only the CPU implements it so far (`Op::copy_rows`).
- `depthwise_conv(out, x, conv, layer, views, n_views)`: a causal depthwise convolution, `out = (residual + act(acc)) / divisor` with `acc` the sum over taps `k` of `w[c][k]` (plus an optional per-row, per-head dynamic tap) times `x` at `(taps - 1 - k) * dilation` positions back, zero before sequence start, accumulated from the oldest tap with fused multiply-adds toward the current one. `DepthwiseConv` gives the weights `[channel, tap]` (last tap current), taps, dilation, the activation (`ConvAct` none or SiLU), the residual (`ConvResidual` none, the input `x` or a given slice), the divisor, the dynamic taps (`dyn[row][head][lag]`, `dyn_heads`, `dyn_stride`) and the history (`ConvHistory`): `tail` keeps the last `(taps - 1) * dilation` raw rows, oldest first, in the layer's F32 tail after the common state prefix; `ring` keeps raw rows at position mod `ring_rows` from `ring_offset` floats into the slot and writes a pass's last `ring_rows` rows, so a rewind inside the headroom keeps its history. `out` may alias `x` or the given residual exactly, but not partially, nor `w`, `dyn` or state storage. The PLE subset (tail, SiLU, given residual, divisor 1, no dynamic taps, `out` apart from `x`) is `Op::depthwise_conv`, which the CPU, Vulkan and the GPU core implement; any other form is `Op::depthwise_conv_ring`, which only the CPU has so far, and a device refuses it by that name before any launch.
- `depthwise_conv(items, layer, views, n_views)`: a list of `ConvRows` `{out, x, conv, from}` over one layer's state, run as the single calls in order. Both forms are non-virtual and reach the one virtual `depthwise_conv_items(items, n, ...)`, which each backend implements over its per-item convolution; the GPU backend runs up to two ring items whose outputs and rings meet none of the other's operands in one launch of `llmx_conv_ring`, and any other list as the single calls.
- `depthwise_conv(items, layer, views, n_views, then)`: the list, then `then`'s query mix (`RowMix`, its `before` item first) over the views' rows, which reads what the convolutions wrote. The virtual `depthwise_conv_mix` runs both where it returns true; the base returns false and the two run as `conv_items` and `query_mix`. The GPU backend takes one ring item of one row over one view, with no product or post step, and a mix of held rows whose items take at most 4 rows, and runs the mix in the convolution's launch (`llmx_conv_ring_mix4`, backends-gpu.md).
  - An item may name the product that forms its input (`ConvProduct` `{weight, x, y, nin, runs, dtype, step, h}`, its item's x then y), defined as `matmul(weight, x, y, nin, channels, rows, runs, dtype)`, or with a `step` as `moe_step(*step, h, y, nin, rows, runs, dtype)`, before the list's convolutions. CPU and Vulkan run the products with `conv_products` first. The GPU backend runs a single item of one row over one view that keeps its slot, a ring with taps up to 4, its input as residual and no activation, dynamic taps or post step, in the product's store where the product takes the one-column Q8_0 row kernel (`llmx_q8_0_rows16_1_conv`, `llmx_q8_0_rows8_1_conv`, docs/src/backends-gpu.md); a step's conv runs in the store of its routed down and combine where that takes a walk build (`llmx_q8_0_routed16_walk3_add_conv`, `_walk5_add_conv`); otherwise the product and then the convolution. Inkling-2's post-attention conv takes its `attn_output` product this way, and its post-FFN conv on one device its MoE step.
  - A `DepthwiseConv` may name a post step (`ConvPost`): `l2_clip` at `post_limit` or `group_rms_norm` against `post_w` with eps `post_limit`, over `post_heads` heads of its output rows, defined as that op after all the list's convolutions. `depthwise_conv_items` returns true where it took every item's post step itself; otherwise the non-virtual wrapper runs the ops. The GPU backend takes them in the `llmx_conv_ring` launch when every view has one row and each head is a power of two of whole waves within a block, a head's squares summed by `block_sum`'s tree so the bits are the ops'; CPU and Vulkan leave them to the ops.
- Stream operations keep their reductions and norms in F32. `group_rms_norm`
  normalizes each group using its own weight slice; `silu_scaled` applies SiLU
  after multiplying its input by a scale. Both allow exact in-place output.
  `mix_streams` averages elementwise sigmoid-gated stream values into one row.
  `group_rms_norm` may name a product of its normed rows (`NormProduct`), which it runs as `matmul` would after the norm, its dst then scratch; the CPU and Vulkan run the norm and then `norm_product`, a device may form the product from the sources in one launch with the same bits.
  `group_rms_norm(items, w, rows, groups, dim, eps, runs)` takes a list of `NormRows` `{dst, src, from, into}` of one shape and weight, run as the single calls in order, so items naming products may share one scratch dst. Both forms are non-virtual and reach the one virtual `group_rms_norm_items`; the CPU and Vulkan run the items in turn, and the GPU backend runs up to three items whose products all form the norm inside them in one launch (`docs/src/backends-gpu.md`).
  `hyper_mix` is the whole learned mix of a row's streams: its default runs `group_rms_norm`, the down product with the optional inject gates' product beside it under `unordered`, `silu_scaled` by 1 / streams, the up product and `mix_streams`; a backend may run it as one launch with those steps' bits.
  `inject_streams` adds a block's row into every residual stream, weighted by
  twice the sigmoid of that stream's raw gate divided by the stream count.
  `signed_sqrt_gate` takes a normalized key/query dot divided by the square root
  of their width, applies its sign times the square root of its absolute value
  clamped below at 1e-6, then sigmoid, and scales a value row for each group.
  A zero dot has zero sign under `ZeroSign::none`, the default, so its gate is 0.5,
  and under `ZeroSign::kept` the zero's own sign bit, as copysign takes it. Mixing and signed-gate outputs must be disjoint
  from their inputs; injection operands are disjoint. Matrix dtype policy is
  applied by the subsequent matrix call, not by these operations.
  DeepSeek-V4.1's mHC ops (docs/DEEPSEEK4.md, section 4.2) work on hc residual streams of dim floats a row.
  `hc_coeffs` turns each row's (2 + hc) * hc mixes, scaled by the reciprocal root mean square of the row's streams, into pre, post and the hc x hc comb of 20 Sinkhorn rounds in the reference's order; its three outputs are disjoint from each other and its inputs.
  `hc_collapse_norm` sums pre_k X_k in stream order and normalizes the sum with a weight, into an output disjoint from its inputs.
  `hc_coeffs_collapse` is the two on the same streams, the collapse reading a pre the coefficients do not write; its default calls them in turn, and a backend may run them side by side.
  `hc_expand` writes X'_k = post_k f + sum over j of comb[j][k] X_j, the sum in j order before post_k f is added, out of place or exactly over the stream.
  Its selection and attention ops read U32 ids and lengths held in float slices, UINT32_MAX naming none, and rank by score, then lower index, refusing a NaN score.
  `indexer_scores` writes sum over heads of w_h w_scale ReLU(q_h . k_t) for the rows of its KV views in order, key t the last dim values of the view's pooled row t of the layer in the key format, -inf at or past (p + 1) / tokens for a row at position p and outside its candidate blocks when given; a row's groups past the scores' width are refused.
  In `IndexForm::raw_blocks`, MiniMax-M3.1's form, it writes rows x heads x n block maxima instead: score b of head h is the largest F32 dot q_h . k_t over the block_size positions of block b up to the row's own, key t the dim values of t's auxiliary row as `kv_write` stored it, -inf for a block past them, with no weights, candidate blocks or key format but f32, and a row's blocks past n refused.
  `pool_write` stores the pooled rows of the groups each view's rows complete, in view order, into their blocks, as `kv_write` stores token rows, each row the storage's pool width; it is the store alone, so a source's rows and a replica's carried rows go through the same call.
  `topk_rows` keeps each row's k best positions in ascending order plus an offset, a reachable pick of -inf kept and one at or past the length written as none; `candidate_blocks` keeps a row's k best blocks, each scored by its best position, the block of its newest position pinned and picks of -inf dropped.
  A DSpine block's ops (`docs/src/model-arch-qwen35.md`, A DSpine block), the CPU's, the GPU core's and Vulkan's: `unit_rows` divides each row by its root mean square or length, floored at FLT_MIN before the sqrt and eps added after it, or only scales it (`RowScale`), and can give each row's norm; `sigmoid_gate_rows` gates a row's message from host-listed rows, zero where none is listed; `matmul_at_ids` dots each row against its group's listed head rows times a weight vector, one for every group or a row a group (`w_stride`), and adds each candidate's own entry of an `add` row a group where one is given, as DFlash2's selector adds the candidate's logit; `chain_pick` walks a chain of candidates by each step's best score, a tie to the lower index.
  A DFlash2 block's convolution (`docs/src/model-arch-qwen35.md`, A DFlash2 block), the CPU's, the GPU core's and Vulkan's: `tap_weighted_rows` sets or adds to each dst row the sum over its taps of a host-listed src row times a per-channel weight, `base` plus the row's own coefficient shared by each `group` channels, a tap listed as UINT32_MAX adding nothing.
  `compress_pool` pools each group of ratio rows by a softmax per channel and normalizes the pooled row; `sparse_attention_sink` attends each head over the rows its ids name, one row as K and V, the max from -1e30 over the named scores and the sink in the denominator after it, so a row naming nothing gives zeros; with KV views each row's entry ids then name the first dim values of its view's pooled rows of the layer, read in place after the kv rows under the one softmax, an id its row cannot see yet refused, and with none it reads the kv rows alone.
  Without entry ids, entry_k entries name the ids 0 to entry_k - 1 of every row, those its row cannot see yet naming nothing, which are the picks `topk_rows` writes at k equal to the visible count.
  Its `SinkTerms` are both optional: `place_bias`, heads x k, adds to each head's score of the kv id at place t of its row's list, so a caller listing keys nearest first gets a bias by distance; `gate`, rows x heads, scales each head's read, which is then added into out instead of written (the DFly2 drafter's gated block read), and `gate_form` tanh scales it by tanh(gate) instead (the DFly2 correction head's gate); `tail`, `tail_rows` rows of F32 values whatever `kv_format` is, named by the ids from `kv_rows` on, so a draft block's unrounded keys join a packed ring under one softmax. The CPU, the GPU core (ROCm and CUDA) and Vulkan compute them all, the GPU's sparse kernels in each pair's `_terms` build, which a call passing any term takes (`docs/src/backends-gpu.md`), and Vulkan's in the same two kernels by push-constant flags (`docs/src/backends-vulkan.md`): the scores add the bias by each kept row's place, the values add the gated read, tanh'd under the tanh form, into out, and a kept row inside the tail reads as F32.
  `model_format_qdq` quantizes and dequantizes a run of whole blocks in place in one of DeepSeek-V4.1's storage formats (`ModelFormat`: the window's FP8 under a power-of-two scale per 32, the index keys' FP4 under one per 32, the entries' FP4 under an E4M3 scale per 16), each block on its own as `quant/model_formats.hpp` gives it.
  `model_format_pack` writes the same blocks' bytes instead, each row of dim values into `format_words` words from word r times a stride, its blocks then zero bytes to the row's last word, the words past it to the stride untouched; the values read back from those bytes are model_format_qdq's.
  The readers take each source's format: `sparse_attention_sink` the kv rows' and the entries', `indexer_scores` the keys', `ModelFormat::f32` for rows held as F32 values and a packed format for rows of its blocks, decoded value by value as code times scale, the one multiply qdq gives, so a packed row reads as the bits its F32 form holds (the cache type `native`, docs/DEEPSEEK4.md, section 4.3).
- `BackendPtr`: the shared handle a backend is held by. The factories live
  with their backends, `make_cpu_backend` in `cpu/cpu_backend.hpp` and
  `make_vulkan_backend` in `vulkan/vulkan_backend.hpp`, and `devices.hpp`
  picks one by device spec.

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

`dot_q8_0` and `matvec_q8_0` left the device-neutral `Backend` interface for `matmul` (`docs/DEVICE-EXECUTION.md`, step 3), and `CpuBackend` has no public `matvec_q8_0` either: the float reference path of a Q8_0 decode row is its private `row_dot` (`backends-cpu.md`), not a primitive another backend implements.

`WeightView` groups the storage type, weight `CSlice`, optional F32 scale `CSlice`, scale count and scale shift; an F8_E4M3 weight's scales are its grid, read at `(row >> scale_shift) * (nin / 128) + k / 128` ([weight_scales](format-weight_scales.md)), and every other weight's shift is 0. `wide` lets an F32 matrix's prompt rows outside the F32 class take a GPU backend's wide tile (F32 sums split by the run's extent) in place of the exact order; inkling2's message-gate products ask for it, and every other weight keeps the exact order. `Projection` holds that view with its output and row count, so grouped and routed callers carry the same association as direct matrix and embedding calls. The base `moe_step` forwards scale-bearing views through the ordinary matrix owners. Vulkan checks scales before its fused routed dispatch and uses the base step when no compatible fused build is available. A scale is scalar or indexed by expert; it belongs to the weight operand rather than activation dtype selection. CPU scale controls pass. Radeon VII checks pass four sets of 262144 embedding values with global scales 1, -0.125, +0 and -0, plus all six matrix entry points scaled/unscaled, mixed grouping and per-expert residuals. F32/F16/BF16 scale checks pass all six matrix calls and mixed/routed cases, including F32 overflow-before-product/residual controls and grouped invalid-input refusal before writes. Broad HF, released-file and performance qualification remain pending.

## Row products

`rows_product(src, items, nin, rows, runs, dtype)` runs up to `kRowItems` (6) projections of the rows a `RowSource` forms, each a `RowItem` `{p, sink, plane, step, h}`; it reaches the one virtual `rows_product_items`.
They are the backend-level form of a layer's decode products: the model says what feeds a product and what its store does, and the backend decides what fuses. A call is synchronous; no override keeps a pointer past it.

- `RowSource` `{row, planes, take, conv, mix, gate, norm, twin, runs}` forms up to `kRowPlanes` (3) planes in this fixed order, every stage optional and naming its own operands as the op it unrolls to takes them: `take` (the first stage's input is a tensor group member's partial rows, under a `RowSum` handle, whose sum a device may take there), `conv` (`RowConv`, a ring convolution with its residual as one item of `depthwise_conv`'s list), `mix` (`RowMix`, one `query_mix` item with its dim, stride and eps, and optionally a `before` item in the same call that feeds no product), then per plane `gate[j]` (`RowGate`, `tanh_gate_mix`) and `norm[j]` (`RowNorm`, `rms_norm_rows`, or with groups above 1 or a split `NormSource` `group_rms_norm`), and `twin` (the rows feed integer products, so a device may write their twins as it forms them). `row[j]` is what items of plane j read, and `runs` are the runs the stages take, none by default, as the model's own stage calls take none.
- `RowSink` `{store, act, glu, twin, conv, sum, cap}` says what an item's store does, `RowStore`: `plain`; `glu`, the item the gate and the next item its up, `act` = `silu_mul` of the pair; `conv`, the item's rows the input of its `RowConv`; `push`, the rows a member's partial (`Collective::partial`) under `sum`, which a device may send to its peers as it stores them; `logits`, the head's `matmul_logits` softcapped by `cap`.
- An item with a `step` is that routed step (`moe_step`) of the row into `p.out` through `h`, alone in its call, stored plain, pushed or into a convolution.
- `RowSum` is declared here and defined by the collective that opens a sum in the products' stores; until one does, nothing names it.
- The base unrolls a call into today's separate calls, so a backend that overrides nothing runs as before: the stages in order, a norm that renormalizes a one-plane, one-query mix or a gate in place folded into that call (`QueryMixRows::norm`, `tanh_gate_mix`'s norm), a one-plane grouped norm with a lone plain item over its rows as `group_rms_norm` with its `NormProduct`, then `matmul` for one item or `matmul_group` for several (an item of plane j above 0 reading `row[j]` as its own input), the GLU stores as `silu_mul`, and the convolutions as one `depthwise_conv` list. A lone item that stores through a convolution, or a step with a `conv` store, is its convolution's `ConvProduct`, as a device runs a convolution in the product's store (`conv_in_store`, [backends-gpu](backends-gpu.md)). `take`, `push` and `twin` change nothing in the base: the sum is the collective's, and the twin the product's.
- `check_row_items` refuses no items or more than `kRowItems`, no planes or more than `kRowPlanes`, an item of a plane the source does not form, a `glu` item without a plain next item of its rows, a `logits` item or a step beside others, a `conv` store without its `RowConv` or a `push` without its `sum` or the reverse, convolutions over different slots, and a step's store other than plain, push or conv. `source_calls` and `row_projections` are the base's stage and product calls, for an override that fuses only some stages.
- Inkling-2-Mini's head is one row product (mix and norm, logits store); `inkling2_checks.hpp` holds each form against its separate calls bit for bit on the CPU and on the devices (`row_products`).

Comment details moved out of `backend.hpp`:
- A `ConvProduct` with no weight data and no `step` is an item without a product, and a device may run its convolution in the product's store, so `y` is the only other thing it writes.
- `query_mix` refuses `n` above 16 and more than `kQueryMixItems` items, `out` overlaps no item's `src` or queries, and a normed item's norm overlaps no `out`.
- `attention_rel_sink` gates its output in place as `gated_attention` does when given a `gate` (`OutputGate`), and first clips `q` and `rel` when given a `clip` (`RelClip`); `runs` give a device the prompts' extents (`RowRun`).
