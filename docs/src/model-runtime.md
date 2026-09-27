# `src/model/runtime.hpp` - the model runtime, with the Qwen3 architecture

The runtime, in namespace `infer`, which runs an architecture's plan and
parts ([architecture](model-architecture.md)) over the weights a reader
gives it ([weights](model-weights.md)), and Qwen3's architecture: dense
Qwen3 and its mixture-of-experts form, `qwen3moe`. The compute primitives
(matmul, attention, RMSNorm, RoPE, expert routing) are delegated to a
`backend::Backend`.

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
- `qwen3::Qwen3`: Qwen3's architecture over a `QwenConfig`, with its roles
  as the ids of `qwen3::Role` and its layer kinds `qwen3::dense` and
  `qwen3::routed`.
  - `plan(index)`: the embedding gives the vocabulary, refused when it is
    not a positive `int`. A layer is routed when
    `blk.N.ffn_gate_inp.weight` is present, which a dense architecture
    refuses, and a dense layer needs the dense width. A layer's roles are
    its attention's norm, q and k norms and four projections, then its
    feed-forward norm, then the router and three expert stacks or the three
    dense matrices; a routed layer's norm and router are `copy` roles and
    its stacks `window` roles. The head's `output.weight` takes
    `token_embd.weight` as its alias, which is the tie. The context is the
    configuration's, the residual the embedding width, the slots
    `qwen3::slot_widths` with the dense width counted when some layer is
    dense, and the two RoPE tables hold the context's positions at half a
    head each.
  - `fill_tables`: the RoPE cos and sin tables, `[pos*(head_dim/2) + i]`.
  - `embed`, `mixer`, `ffn` and `head`: the embedding gather; the attention
    (norm, grouped q, k and v projections, `norm_rope_kv`, `attention` and
    the output projection added into the residual); the feed-forward block,
    dense or routed by the layer's kind, reading whichever row of weights it
    is handed; and the head over the rows that want logits, compacted first.
- `gguf_weights(GGUFModel)`: a GGUF model's weights (`ModelWeights`): the
  Qwen3 architecture over the configuration `load_config` reads once and,
  before any backend storage exists, a view per tensor, refusing a tensor
  table whose storage count does not match its tensors, with a rank above four, or an offset or extent outside the payload, which includes a
  payload its owner released. A view's data is null while its tensor's file
  is not mapped (`gguf::map_payload`). The loader, the tests and the
  synthetic bench call it.
- `plan_model(weights)`: the plan of a model's weights, which `place_model`
  makes once and hands to the fit, the experts placement and the model. It
  indexes the views by name once (`TensorIndex`, which refuses a repeated
  name), asks the architecture for its plan over that index, and sets each
  role's `tensor`, its name's or else its alias's (`aliased`), so neither the
  fit nor the resolution looks a name up again. A plan whose slot 0 is not
  as wide as the residual, or with a role id past `role_ids`, is the
  architecture's error (`std::logic_error`), since the runtime strides the
  residual by one and sizes slot 0 by the other, and indexes rows by the id.
- `footprint(weights, plan, options)`: what a model asks of memory, for a split fitted to devices (`model/layer_split.hpp`), counted from its plan: each layer's matrices are the tensors its roles take, in the file's order and each once, marked as products where a role reads them as a matrix, whatever their rank, so a tensor no role takes costs nothing; the embedding is the embed part's table, the output the head's matrix and the output norm the head's norm, and the head is tied when its role took the alias, its output then the embedding counted as a product; a pass role of any other part and kind, or a second role for one of those three, throws `std::logic_error`, so a new kind of pass role extends `Footprint` rather than fitting a split it overruns; it reads each role's tensor as `plan_model` set it and looks no name up; one layer's cache is the budgeted positions at the plan's K and V geometry and the options' cache types, the tables the plan's, a row of the arena the plan's slots and a row of the residual stream handed between devices the plan's residual. `placement_for(split)` turns a `LayerSplit` into a `Placement`. `synthetic_model(...)`: a model of this architecture with a given shape and random weights, Q8_0 matrices and F32 norms, which `bench` times without a file.
- `kDefaultUbatch` (512): the prompt tokens a pass takes unless set otherwise, and so the prompt rows a placement is fitted for. `kv_tokens(plan, options)` and `kv_bytes_per_position(plan, options)`: the positions the caches are budgeted for, the options' or else the plan's context, which the fit and the cache allocation take, and one position's key and value bytes at the plan's geometry and the options' cache types, which only the fit and `kv_used_bytes` take: the allocation passes the token budget and the two types to `kv_alloc`, and the backend's storage turns them into blocks and bytes (`backends/kv_storage.hpp`).
- `place_model(weights, backends, request, options, adopt = {})`: the one place a model is placed over the backends its caller made, returning the model with the request's ubatch set and, for a split, its plan (`LayerSplit::describe`). It plans the weights once (`plan_model`), and the fit, the experts placement and the model it builds read that plan. A request that names `histories` of `history_tokens` each, as `bench --model` does its sequences, has them counted in whole blocks of each backend, each up to the model's context, which no run passes: where the options' budget would leave any storage short, the budget becomes what they take in the largest blocks, which the fit and every storage then use, and otherwise it is unchanged. With several backends or layer shares it fits the split for `request.ubatch` (default `kDefaultUbatch`) plus `request.decode_rows` rows and `request.slots` pass slots, whose handoff buffers the host holds (`budgets_for`, `split_layers`, `placement_for`); with one backend that is not the CPU (`Backend::is_cpu`) and `PlacementRequest::cpu_moe`, the CPU becomes device 0 beside it and the feed-forward blocks of the first `cpu_moe` layers the plan marks routed (every one at -1) run there, with `stream_from` as the placement's; otherwise the model is on the one backend, the CPU with its experts on it included. Experts on the CPU on a model whose plan marks no layer routed are refused whatever the backends, the CPU included, and so are experts on the CPU with several devices, each refusal naming the flag the request stands for (`--cpu-moe` at -1, `--n-cpu-moe` otherwise); so is a nonzero `stream_from` without experts on the CPU, which would have nothing to stream. `adds_host_for_experts(backends, request)` is the rule for when it adds the CPU for experts, which asks whether the backend is the CPU and not whether it reads its weights in place, since a device may read host memory in place, and `host_reads_in_place(backends, request)` says whether any backend of the placement reads weights in place, one of `backends` or that CPU, which the loader asks before it decides what to map. The CLI, `bench --model`, the server and `llmx-split-check` all build their model through it by way of `infer::load_model` ([load](inference-load.md)), as does `compare_cpu`. A routed layer's experts are streamed to the device only where attention runs on a backend that copies its weights and the experts on one that reads them in place (`Backend::reads_in_place`).
- `qwen3::slot_widths(config, dense)`: the floats one row takes in each of Qwen3's twelve arena slots, which its plan holds, `ensure` allocates and `footprint` counts.
- `Placement`: a device index per tensor role: each layer's attention and
  feed-forward block, the embedding table and the output head. Empty means
  everything on device 0. Per role rather than per layer so expert offload
  puts a layer's experts on the CPU while its attention stays on the device
  (`docs/EXECUTION.md`). `stream_from` is the prompt length (`BatchEntry::extent`) from which such
  a layer runs on its attention device instead for every row the prompt computes, whatever prefix the history already held, 1 counting as 2 since a one-token prompt never streams; rows a server forks from a donor keep the path they were computed on, the one the donor's prompt took for its prompt rows and the host for its generated tokens (`docs/SERVER.md`, Open gaps). Its `copy` roles (the norm
  and router) get a copy there at load, its `window` roles (the expert
  stacks) are written into a per-device window, one buffer per window role
  in role order sized to the largest such layer's, once per pass that needs
  them, and `ffn_split` runs a pass's consecutive entries alike as one
  group, the long ones on the device through the layer's streamed row of
  weights and the rest on the host through its own row, with a crossing
  each way. Which side a weight is on is the backend's `reads_in_place()`:
  a backend that reads what it adopts in place is the host side, whether
  or not it is the CPU, and one that copies it is the device side.
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
  the CLI's model keeps one. `length()` is storage 0's committed length;
  the storages can disagree while a pass is part way through its stages,
  and the model continues a history from the first stage's storage.
  `in_flight()` holds from `begin_pass` until `end_pass` or `abort_pass`,
  when no other pass, reset or fork may take the sequence and it must not
  move.
- `ExecContext`: where a context's passes run, plain data the model fills:
  per device an activation arena (the plan's slots at 64-byte offsets in
  one backend allocation), which each device's passes use in turn, and the
  host-visible handoff buffers on each device the residual leaves, one,
  two when a prompt's chunks are pipelined, and in a context reserved for
  passes those of its slots (`reserve_passes`), while a device it never
  leaves, such as a pipelined split's last, keeps none; the host-visible
  logits rows; the tickets; and a `Pass` per pass in flight, the entries,
  rows, positions, head rows and cache views its stages read as they are
  recorded, the handoff buffer its crossings use, its first logits row and
  its ticket; and the run lists a part rebuilds (`Step::scratch`) and a
  streamed layer's groups, reserved at a run per row so no part grows them.
  Allocated by the first forward that needs it and grown to the
  largest pass seen, or sized once by `reserve_passes` (below), after which
  nothing in it is replaced. `logits(i)` is row `i` of the last pass,
  in entry order.
- `BatchEntry`: what one sequence contributes to a pass: tokens appended to
  it and whether the logits after its last token are wanted, or with
  `every_logits` the logits after every one of its tokens. A prefill
  microbatch is one entry with many tokens, a decode batch is many entries
  with one, and they mix. `extent` is what a device picks the entry's
  kernels by (`backend.hpp` `RowRuns`): for a prompt's rows the position one
  past its last token, for a generated token 1. `prefill`, `score` and the
  server set it, so a prompt computes the same in one pass or in slices.
  An entry of many generated tokens at extent 1, which is how the server recomputes a paused request's reply, takes the decode kernels for every one of them.
- `Model`: built from `ModelWeights` over one backend (the CPU's by
  default) or over several with a `Placement` and an optional
  `AdoptWeight`, planning the weights itself or given their plan
  (`plan_model`), as `place_model` gives it; the fixtures and the synthetic
  bench wrap their file in `gguf_weights`. The model keeps no view and no
  reference to the file, and shares the architecture object with the
  weights. Each weight is put on the backend that hosts its role, which on
  the CPU reads the file's bytes in place and on a device copies them.
  Each device that runs attention gets a `KVStorage` for exactly its layers
  with its own pool, block size and adopted position tables, which the
  architecture filled once (`fill_tables`). A pass calls the architecture's
  parts, each with a `Step` on the device that runs it: the embedding on
  the first stage, each layer's mixer and feed-forward part, and the head
  on the output device. Its stages are
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
    committed included. A sequence listed twice or in flight is refused, and
    so is a context reserved for passes.
  - The pass API, for a scheduler that keeps passes of different sequences
    in flight so that every stage of a pipelined split works on one while
    the host samples another (`docs/MULTI-DEVICE.md`). `stage_count()` and
    `pipelined()` say whether that can pay: several stages, the embedding on
    the first stage's device, the head on the last's and every feed-forward
    block beside its attention. `reserve_passes(ctx, slots, rows,
    logit_rows)` sizes a fresh context once: the arena for `rows` rows,
    which every pass shares, a handoff buffer per slot on each device the
    residual leaves, two at least on a pipelined split and one for the
    single slot of a placement that is not pipelined (`handoff_buffers` in
    `layer_split.hpp`, the rule the fit counts by), and `logit_rows` rows
    of logits the caller hands out. The context is frozen from then on. A
    reservation that fails leaves the context fresh, so a smaller one may
    follow. More than one slot needs a pipelined placement.
    `begin_pass(ctx, slot, entries, n, logits_base)` plans a pass in a free
    slot, copying its tokens, and puts its sequences in flight; a sequence
    in flight or listed twice, a slot in use or beyond the reservation, and
    more rows or logits rows than reserved are refused before any work.
    `run_pass_stage(ctx, slot, s)` records the pass's next stage, which must
    be `s`; a failure aborts the pass before it is rethrown, and the other
    passes go on. `pass_logits(ctx, slot, i)` waits on the pass's own head
    and returns its wanting row `i`, written from row `logits_base` on.
    `end_pass` takes a pass whose last stage has run out of flight, and
    `abort_pass` abandons one at any point: every device drained, then only
    its entries truncated in every storage to where it found them. Each
    pass's stages run in order; passes interleave as the caller likes, since
    each device runs the stages recorded on it in that order and a pass
    keeps its own handoff buffer, logits rows and ticket. `forward`,
    `prefill`, `step` and `score` do not use it.
  - `make_sequence()`, `reset(sequence)`: a fresh history, and one returned
    to the pool after waiting on its last ticket; a sequence in flight is
    refused.
  - `kv_pools()`, `kv_pool_block_tokens(s)`, `kv_pool_blocks(s)`: the
    cache pools a scheduler admits against, one per device that runs
    attention, each in its own blocks; `kv_tokens_total()` is the tokens
    every pool can hold, and `kv_block_tokens()` the largest block, which a
    reusable prefix ends on.
  - `fork(sequence, length)`: a second history holding the first `length`
    tokens, which must be whole blocks in every storage, sharing every block
    below `length` on every storage and allocating and copying nothing; the
    server forks a donor at the blocks a prompt shares with it.
    A forked sequence continues exactly as a fresh one fed the same tokens at the same extents would; rows another extent computed can differ from them by rounding (`docs/SERVER.md`, Open gaps).
    A sequence in flight is not forked.
  - `set_threads(n)` applies to every backend and `threads_available()` reports the largest count among them, the host's wherever it sits in a placement.
  - `n_tokens()`, `context_length()`, the plan's. The thread getter reports the resolved backend count,
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
  - Before model activation/KV/RoPE allocation, `gguf_weights` checks
    tensor-name uniqueness, offset count/alignment/ranges and supported
    storage types, and construction checks all required layouts.
    `resolve_tensors` walks the plan's roles in order - the pass's,
    then each layer's followed by a streamed layer's copies - refusing a
    role whose tensor is absent with `TensorIndex`'s text, checking each by
    its kind and returning
    its `Weight` from the same check, so a resolved handle is well-formed by
    construction and no other path produces one. Each weight goes to the
    device of its part, and a tensor two roles take on one device is adopted
    there once, which is how a tied head beside the embedding reads the
    embedding's buffer and how each tensor reaches each backend at most once.
    Norms are F32 vectors. Matrices have the expected input
    and output dimensions, with equal embedding/output vocabulary sizes.
    Trailing singleton dimensions up to rank four are accepted. Valid payload
    aliases and unused scalar/empty F32 tensors remain supported.
    A CPU backend starts its worker pool on its first parallel dispatch, so a fresh one has started no threads when these checks run.
  - Loading may enqueue uploads before a later tensor or allocation fails. The
    constructor catches failures inside its body and drains each used backend
    while its members are still alive. Normal model destruction also drains
    those backends before releasing weights, caches and position tables. Successful
    loading keeps uploads asynchronous.

Supports dense and mixture-of-experts Qwen3 with Q8_0 / Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K weights
and F32 embeddings/matrices/norms. F32 embedding rows are copied directly;
F32 matmul reads weight rows without staging. Missing
`output.weight` selects tied token embeddings for the output projection.

The model reads the views only while it is built, and then only their
addresses. The bytes a backend that reads in place adopted must stay valid
while its buffer lives, which is the model's lifetime, and unchanged once the
load returns; a direct load fills them after construction, before anything
reads them. A backend that copies has consumed its bytes when `adopt` or
`write` returns (`backend.hpp`). The loader's streamed loads read the file
after construction to fill the storage a copying backend was given, so the
payload of a model whose every weight a copying backend took is released once
those writes are made (`GGUFModel::release_payload`). Construction
does not scan numerical weight contents, validate every possible metadata
extension, check arbitrary token IDs or establish recovery after an execution
failure. Those require separate input/session checks; they are not guarantees
of the configuration and layout validation above.

`prefill` enters one backend-owned synchronous scope around batch-buffer
allocation and every microbatch, including the final projection. Empty input
is rejected before entry; individual decode steps do not enter that scope.
