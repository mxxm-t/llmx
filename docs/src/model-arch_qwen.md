# `src/model/arch_qwen.hpp` - Qwen3 forward pass

Qwen3-style transformer forward pass, from scratch, in namespace `infer`:
dense Qwen3 and its mixture-of-experts form, `qwen3moe`. The compute
primitives (matmul, attention, RMSNorm, RoPE, expert routing) are delegated
to a `backend::Backend`.

- Mixture of experts: `general.architecture = qwen3moe` reads every key
  under the `qwen3moe.` prefix, plus `expert_count`, `expert_used_count`,
  `expert_feed_forward_length` and an optional `expert_weights_norm`
  (default true); a sigmoid gate, shared experts or scaled expert weights
  are refused. A layer is routed when `blk.N.ffn_gate_inp.weight` is
  present, its experts the stacked `ffn_{gate,up,down}_exps` tensors, so
  dense and routed layers can mix. A routed feed-forward block is the
  router matmul, `route_experts`, `matmul_experts` for gate and up,
  `silu_mul` over every slot and `matmul_experts_add` into the residual;
  the arena gains the router scores, expert ids and weights as slots 9 to
  11, and the feed-forward slots are as wide as a dense layer or k
  experts, whichever is wider.

- `QwenConfig` + `load_config(GGUFModel)`: reads Qwen3 metadata
  (`block_count`, `embedding_length`, `feed_forward_length`,
  `attention.head_count[_kv]`, `attention.key_length`, `context_length`,
  `rope.freq_base`, `attention.layer_norm_rms_epsilon`, under the
  architecture's prefix, `qwen3.` or `qwen3moe.`).
  Consumed integer fields accept positive INT32/UINT32/INT64/UINT64 values up
  to `INT_MAX`. Consumed float fields accept finite positive F32/F64 values
  representable as nonzero F32. Duplicate consumed keys and wrong types fail.
  Optional defaults apply only when absent: KV heads equal query heads, key
  width is an exact embedding/head quotient, context is 4096, RoPE base is
  10000 and RMS epsilon is 1e-6. Explicit key width can differ from that quotient.
  Head width must be even; GQA head counts must divide and projection widths
  fit the runtime's integer indices. Context storage must fit float vectors.
  Declared value/rotary widths must equal key width. Declared architecture
  must be `qwen3` or `qwen3moe` and tensor layout `reference`; absence
  remains accepted for existing synthetic models. RoPE scaling is
  unsupported: type must be absent or `none`, and current/legacy factors
  absent or exactly one. These keys use the [GGUF metadata vocabulary](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md).
- `Weight` / `LayerWeights`: a tensor resolved once at load - type, a buffer
  handle from the backend that hosts it and the two dimensions - and a
  layer's weights grouped together: eleven for a dense layer, the router
  and three stacked expert tensors in place of the three feed-forward
  matrices for a routed one, and a streamed layer's copies of its norm and
  router. `Weight::slice()` names the weight's location; the model never
  dereferences it. The forward pass indexes `layers_[l]` instead of
  rebuilding `"blk.N."` and hashing a tensor name for every projection of
  every layer of every token, and a device backend recognizes the same
  weight across calls. See `docs/DEVICE-EXECUTION.md` step 1.
- `footprint(model, options)`: what this architecture asks of memory, for a split fitted to devices (`model/layer_split.hpp`): each layer's matrices from its `blk.N.` tensors, those a product reads marked by their role (the attention and feed-forward projections and the router, whatever their rank), the embedding, the head's matrix and norm and whether it is tied, one layer's cache for the budgeted positions at the options' cache types, the RoPE tables, a row of the arena and a row of the residual stream handed between devices. `placement_for(split)` turns a `LayerSplit` into a `Placement`. `synthetic_model(...)`: a model of this architecture with a given shape and random weights, Q8_0 matrices and F32 norms, which `bench` times without a file.
- `kDefaultUbatch` (512): the prompt tokens a pass takes unless set otherwise, and so the prompt rows a placement is fitted for. `kv_tokens(cfg, options)` and `kv_bytes_per_position(cfg, options)`: the positions the caches are budgeted for and one position's key and value bytes at the options' cache types, which the fit, the cache allocation and `kv_used_bytes` all take. `routed_layers(m, n_layer)`: the layers whose router tensor (`blk.N.ffn_gate_inp.weight`) is present.
- `place_model(model, backends, request, options)`: the one place a model is placed over the backends its caller made, returning the model with the request's ubatch set and, for a split, its plan (`LayerSplit::describe`). With several backends or layer shares it fits the split for `request.ubatch` (default `kDefaultUbatch`) plus `request.decode_rows` rows (`budgets_for`, `split_layers`, `placement_for`); with one backend and `PlacementRequest::cpu_moe`, the CPU becomes device 0 beside it and the first `cpu_moe` routed layers' feed-forward blocks (every one at -1) run there, with `stream_from` as the placement's; otherwise the model is on the one backend. Experts on the CPU with several devices are refused. The CLI, `bench --model`, the server and `llmx-split-check` all build their model through it. A routed layer's experts are streamed to the device only where attention runs on a backend that copies its weights and the experts on one that reads them in place (`Backend::reads_in_place`). A tied head on the embedding's device reads the buffer adopted for the embedding rather than a second copy.
- `slot_widths(config, dense)`: the floats one row takes in each of the arena's twelve slots, which `ensure` allocates and `footprint` counts.
- `Placement`: a device index per tensor role: each layer's attention and
  feed-forward block, the embedding table and the output head. Empty means
  everything on device 0. Per role rather than per layer so expert offload
  puts a layer's experts on the CPU while its attention stays on the device
  (`docs/EXECUTION.md`). `stream_from` is the count of new prompt tokens (`BatchEntry::fresh`) from which such
  a layer runs on its attention device instead: the norm and router get a
  copy there at load, the experts are written into a per-device window
  (one buffer per projection, sized to the largest such layer) once per
  pass that needs them, and `ffn_split` runs a pass's consecutive entries
  alike as one group, the long ones on the device and the rest on the host
  through a crossing each way. A host backend aliases what it adopts and a
  device copies it, which is how the model tells which side a weight is
  on.
- `ModelOptions`: what is fixed at construction, before the caches are
  allocated: each cache side's type (`kv_k`, `kv_v`, the CLI's
  `--cache-type-k` and `--cache-type-v`) and `kv_tokens`, the positions
  every pool holds, zero for one model context. Both sides default to
  `KVType::f16`, the runtime's one default: the CLI, the server, the
  synthetic bench and the split check all start from it.
- `Sequence`: one request's history over a model's cache, made by
  `Model::make_sequence`: a block table per storage and the committed
  length, and per device the ticket of the last pass that touched it, which
  a reset waits on. Movable, not copyable. The server keeps one per request;
  the CLI's model keeps one.
- `ExecContext`: where a context's passes run, plain data the model fills:
  per device an activation arena (twelve slots at 64-byte offsets in one
  backend allocation), which each device's passes use in turn, and a
  host-visible handoff buffer a crossing goes through, two when a prompt's
  chunks are pipelined; the host-visible
  logits rows; the tickets; and a `Pass` per pass in flight, the entries,
  rows, positions, head rows and cache views its stages read as they are
  recorded. Allocated by the first forward that needs it and grown to the
  largest pass seen. Two contexts let a scheduler keep one pass on the device
  while it reads another's logits. `logits(i)` is row `i` of the last pass,
  in entry order.
- `BatchEntry`: what one sequence contributes to a pass: tokens appended to
  it and whether the logits after its last token are wanted, or with
  `every_logits` the logits after every one of its tokens. A prefill
  microbatch is one entry with many tokens, a decode batch is many entries
  with one, and they mix. `extent` is what a device picks the entry's
  kernels by (`backend.hpp` `RowRuns`): for a prompt's rows the position one
  past its last token, for a generated token 1. `prefill`, `score` and the
  server set it, so a prompt computes the same in one pass or in slices.
- `Model`: loads tensors from a `GGUFModel` over one backend, or over
  several with a `Placement`. Each weight is adopted by the backend that
  hosts its role, which on the CPU aliases the loaded file bytes and costs no
  further RAM.
  Each device that runs attention gets a `KVStorage` for exactly its layers
  with its own pool, block size and adopted RoPE tables. Its stages are
  runs of consecutive layers whose attention sits on one device, each
  writing that device's storage. A device's attention layers must form one
  run, or the placement is refused; a model on one device has one stage. The
  residual stream crosses devices wherever the placement changes, in two
  halves: the source copies the rows into its handoff buffer inside its own
  work (`send`), and the destination waits that submission's ticket and
  writes them (`receive`); inside a stage both run at once (`cross`). One
  default sequence and context serve the single-sequence entry points.
  Read-only after construction apart from pool bookkeeping.
  - `forward(ctx, entries, n)`: one pass over every entry. Each sequence's
    tokens go through the graph at their own positions and attend through
    their own history via one view per entry and per storage; the rows that
    want logits are gathered, normed and projected once on the output
    device. The head's submission is waited on only when logits are wanted;
    a crossing waits on the host for its source's submission. It runs
    its stages in a row (`begin`, `run_stage`, `finish`): each reserves the
    blocks of the storage it writes, submits the devices it recorded on and
    commits. It is one transaction: a failure anywhere drains every device
    and returns every history to where the pass found it, stages already
    committed included. A sequence listed twice is refused.
  - `make_sequence()`, `reset(sequence)`: a fresh history, and one returned
    to the pool after waiting on its last ticket.
  - `kv_pools()`, `kv_pool_block_tokens(s)`, `kv_pool_blocks(s)`: the
    cache pools a scheduler admits against, one per device that runs
    attention, each in its own blocks; `kv_tokens_total()` is the tokens
    every pool can hold, and `kv_block_tokens()` the largest block, which a
    reusable prefix ends on.
  - `fork(sequence, length)`: a second history holding the first `length`
    tokens, which must be whole blocks in every storage, sharing every block
    below `length` on every storage and allocating and copying nothing; the
    server forks a donor at the blocks a prompt shares with it. A forked
    sequence continues exactly as a fresh one fed the same tokens would.
  - `set_threads(n)` applies to every backend and `threads_available()` reports the largest count among them, the host's wherever it sits in a placement.
  - `n_tokens()`, `context_length()`. The thread getter reports the resolved backend count,
    allowing the CLI to restore automatic decode settings after prefill.
  - `step(token_id) -> logits`: one entry of one token through `forward` on
    the model's own sequence and context. This is the decode path.
  - `prefill(ids) -> logits`: the prompt in chunks of `ubatch()` tokens, one
    entry per chunk, inside one backend prefill scope, so each weight row is
    read once per chunk instead of once per token. Only the last chunk asks
    for logits. The prompt is one transaction across its chunks. Over more
    than one stage the chunks run as a software pipeline on the calling
    thread: step t runs stage s of chunk t-s, the first stage first, so
    every device has its next chunk queued before it finishes the one it
    runs, and each takes its chunks in order, which is what lets them share
    its arena. A prompt's positions follow the first stage's storage, which
    a chunk commits first. That needs the embedding on the first stage's
    device, the head on the last's and every feed-forward block beside its
    attention, as every fitted split has; any other placement runs its
    chunks one pass at a time. Chunks are the ubatch either way, so a split
    computes what one device does.
  - `score(ids, each)`: the ids in chunks of `ubatch()` tokens from an empty
    history, each chunk an `every_logits` entry, calling `each(pos, logits)`
    for every position. This is the batched path perplexity scores through.
  - `set_ubatch(n)` / `prefill_batch()`: physical batch, set by `--ubatch`
    through `place_model`; `ubatch()` is the private getter the passes
    use. llmx has no logical batch; see `docs/USAGE.md`.
  - `reset()`: the default sequence's history returns to the pool while
    allocated KV capacity is retained.
  - Both forward paths call `Backend::attention` over the paged KV cache;
    score scratch, causal masking and head scheduling belong to the backend.
    `norm_rope_kv` norms and rotates q and k and writes k and v into the
    views' blocks in one op.
  - Q/K/V and FFN gate/up share activations and use `matmul_group` in both
    forward paths. CPU groups eligible decode projections; batched prefill
    retains sequential matrix calls through the backend fallback.
  - Batched norms, per-head norm/RoPE, SiLU and the residual adds are backend
    ops (`rms_norm_rows`, `norm_rope_kv`, `silu_mul`, and `matmul_add`, which
    folds the residual add into the output projections), so the model
    holds no elementwise loops and needs no host parallelism of its own. Each
    row keeps the same arithmetic, and the operations finish before dependent
    matrix operations or KV writes begin. Whether to spread a stage across
    workers is the backend's decision, not the model's.
  - Before model activation/KV/RoPE allocation, construction checks tensor-name
    uniqueness, offset count/alignment/ranges, supported storage types and all
    required layouts. `resolve_tensors` performs this validation and returns
    the `Weight` for each tensor from the same check, so a resolved handle is
    well-formed by construction and no other path produces one.
    Norms are F32 vectors. Matrices have the expected input
    and output dimensions, with equal embedding/output vocabulary sizes.
    Trailing singleton dimensions up to rank four are accepted. Valid payload
    aliases and unused scalar/empty F32 tensors remain supported. The backend
    can already have allocated its worker pool before these checks.
  - Loading may enqueue uploads before a later tensor or allocation fails. The
    constructor catches failures inside its body and drains each used backend
    while its members are still alive. Normal model destruction also drains
    those backends before releasing weights, caches and RoPE storage. Successful
    loading keeps uploads asynchronous; reader tracking is allocated before
    the first adoption so bookkeeping cannot fail while holding a local weight.

Supports dense and mixture-of-experts Qwen3 with Q8_0 / Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K weights
and F32 embeddings/matrices/norms. F32 embedding rows are copied directly;
F32 matmul reads weight rows without staging. Missing
`output.weight` selects tied token embeddings for the output projection.

The borrowed GGUF model must outlive `Model` and remain unchanged, except that
its payload may be released (`GGUFModel::release_payload`) once
`holds_payload()` is false, that is when every weight was copied into a
device backend's own memory rather than read in place. Construction
does not scan numerical weight contents, validate every possible metadata
extension, check arbitrary token IDs or establish recovery after an execution
failure. Those require separate input/session checks; they are not guarantees
of the configuration and layout validation above.

`prefill` enters one backend-owned synchronous scope around batch-buffer
allocation and every microbatch, including the final projection. Empty input
is rejected before entry; individual decode steps do not enter that scope.
