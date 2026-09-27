# `src/model/runtime.hpp` - the model runtime

The runtime, in namespace `infer`: sequences, passes over batches of them,
stages over devices, the activation arena and the crossings between
devices. It runs an architecture's plan and parts
([architecture](model-architecture.md)) over the weights a reader gives it
([weights](model-weights.md)), and names no architecture or tensor, which
`tests/arch_boundary.py` checks; the architectures are under
`model/arch/` ([registry](model-arch-registry.md)), and a model is placed
over its devices by [place](model-place.md). The compute primitives are
delegated to a `backend::Backend`.

- `plan_model(weights)`: the plan of a model's weights, which `place_model`
  makes once and hands to the fit, the experts placement and the model. It
  indexes the views by name once (`TensorIndex`, which refuses a repeated
  name), asks the architecture for its plan over that index, and sets each
  role's `tensor`, its name's or else its alias's (`aliased`), so neither the
  fit nor the resolution looks a name up again. A plan whose slot 0 is not
  as wide as the residual, or with a role id past `role_ids`, is the
  architecture's error (`std::logic_error`), since the runtime strides the
  residual by one and sizes slot 0 by the other, and indexes rows by the id.
- `kDefaultUbatch` (512): the prompt tokens a pass takes unless set otherwise, and so the prompt rows a placement is fitted for. `kv_tokens(plan, options)` and `kv_bytes_per_position(plan, options)`: the positions the caches are budgeted for, the options' or else the plan's context, which the fit and the cache allocation take, and one position's key and value bytes at the plan's geometry and the options' cache types, which only the fit and `kv_used_bytes`, over the layers that keep KV, take: the allocation passes the token budget and the two types to `kv_alloc`, and the backend's storage turns them into blocks and bytes (`backends/kv_storage.hpp`).
- `Placement`: a device index per tensor role: each layer's mixer
  (`mixer_device`) and feed-forward block (`ffn_device`), the embedding
  table and the output head. Empty means everything on device 0. Per role
  rather than per layer so expert offload puts a layer's experts on the CPU
  while its mixer stays on the device (`docs/EXECUTION.md`). `stream_from` is the prompt length (`BatchEntry::extent`) from which such
  a layer whose streamed weight types the destination supports runs on its mixer device instead for every row the prompt computes, whatever prefix the history already held, 1 counting as 2 since a one-token prompt never streams; rows a server forks from a donor keep the path they were computed on, the one the donor's prompt took for its prompt rows and the host for its generated tokens (`docs/SERVER.md`, Open gaps). Its `copy` roles (the norm
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
  `--cache-type-k` and `--cache-type-v`), `kv_tokens`, the positions
  every pool holds, zero for one model context, and `state_slots`, the
  sequences that may hold a recurrent state at once in a model whose layers
  keep one: one for a command's own sequence, the sequences a pass carries
  for `bench --seqs` (the CLI's `open_model` sets it from them), two for
  `llmx-split-check`. Both sides default to
  `KVType::f16`, the runtime's one default: the CLI, the server, the
  synthetic bench and the split check all start from it.
- `Sequence`: one request's history over a model's cache, made by
  `Model::make_sequence`: a block table per KV storage, the committed length
  of each stage, which is its KV sequence's where its layers keep KV and a
  count of its own otherwise, the state slot it holds from its first pass on in a model
  whose layers keep a state, whether a failed pass lost that state, and per
  device the ticket of the last pass that touched it, which a reset waits
  on. Movable, not copyable. The server keeps one
  per request; the CLI's model keeps one. `length()` is the first stage's
  committed length; the stages can disagree while a pass is part way
  through them, and the model continues a history from the first stage's.
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
  Each device that runs a mixer gets a `KVStorage` for exactly its layers
  whose cache is KV (`LayerPlan::cache`), each layer indexed within it by
  its place among them, with its own pool, block size and adopted position
  tables, which the
  architecture filled once (`fill_tables`). A pass calls the architecture's
  parts, each with a `Step` on the device that runs it: the embedding on
  the first stage, each layer's mixer and feed-forward part, and the head
  on the output device. Its stages are
  runs of consecutive layers whose mixer sits on one device, each
  writing that device's storage. A device's mixer layers must form one
  run, or the placement is refused; a model on one device has one stage.
  Each stage commits its own length, whatever its layers keep, so a stage
  whose layers keep no KV has no storage and still runs. Each device whose
  mixer layers keep a recurrent state holds a `backend::StateStorage` of
  exactly those layers with `state_slots` slots, allocated and zeroed at
  load and never grown (`Backend::state_alloc`), and a sequence takes one
  slot of the model's `SlotPool` in its first pass, before any work, and
  keeps it until its reset. The
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
    block beside its mixer. `stage_on_host(s)` says whether stage s runs
    on the CPU, which computes as it is recorded, so a caller records it
    after its device stages, and `stage_backend(s)` gives the backend a
    caller timing the stages reads. `reserve_passes(ctx, slots, rows,
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
    to the pools, its blocks and its state slot, after waiting on its last
    ticket; a sequence in flight is refused.
  - `keeps_state()`: whether some layer keeps a recurrent state, which
    exists only at the end of what it has read. Such a model is never
    forked. A pass updates the state in place, so a failed pass, or a
    prompt that fails part way, loses the state of every entry it takes
    back to a length other than 0, where the state reads as zero; a lost
    sequence is refused by the next pass until its reset. A server that
    cannot hold states refuses such a model.
  - `kv_pools()`, `kv_pool_block_tokens(s)`, `kv_pool_blocks(s)`: the
    cache pools a scheduler admits against, one per device whose mixer
    layers keep KV, each in its own blocks; `kv_tokens_total()` is the tokens
    every pool can hold, and `kv_block_tokens()` the largest block, which a
    reusable prefix ends on.
  - `fork(sequence, length)`: a second history holding the first `length`
    tokens, which must be whole blocks in every storage, sharing every block
    below `length` on every storage without allocating or copying physical
    KV blocks; the logical block tables and ticket vectors still allocate. The
    server forks a donor at the blocks a prompt shares with it.
    A model whose layers keep a state is not forked.
    A forked sequence continues exactly as a fresh one fed the same tokens at the same extents would; rows another extent computed can differ from them by rounding (`docs/SERVER.md`, Open gaps).
    A sequence in flight is not forked.
  - `set_threads(n)` applies to every backend and `threads_available()` reports the largest count among them, the host's wherever it sits in a placement.
    The thread getter reports the resolved backend count, allowing the CLI to restore automatic decode settings after prefill.
  - `n_tokens()` is the default sequence's length; `context_length()` and `n_vocab()` are the plan's.
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
    its arena. A prompt's positions follow the first stage's length, which
    a chunk commits first. That needs the embedding on the first stage's
    device, the head on the last's and every feed-forward block beside its
    mixer, as every fitted split has; any other placement runs its
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
  - The math is the architecture's: the model calls its parts and holds no
    elementwise loop and no host parallelism of its own, and whether to
    spread an op across workers is the backend's decision. What each part
    issues is on the architecture's page ([qwen3](model-arch-qwen3.md)).
  - A file's own checks come before the model is built, in its reader
    (`gguf_weights`, [registry](model-arch-registry.md)).
    Before any weight is adopted or model buffer allocated, each present
    role's type must be supported by its assigned backend (`supports_type`),
    including a tied head placed apart from the embedding. A refusal names
    the pass role or layer part, tensor, type and device index.
    A routed layer's streaming eligibility is computed here once: its host
    reads in place, its distinct mixer device copies weights, and that
    destination supports every copied or windowed role. If a streamed type
    is unsupported, the whole layer stays on its host; other eligible layers
    can still stream. `resolve_tensors` reuses this decision.
    In the same pass over the layers, every op a layer's plan names
    (`LayerPlan::ops`) must be one the backend of its part's device
    implements (`Backend::implements`), or the model is refused with a text
    naming the layer, the part and the op; a stream destination that lacks
    one of a routed feed-forward part's ops leaves the layer on its host, as
    a type it lacks does.
    `resolve_tensors` walks the plan's roles in order - the pass's, then
    each layer's followed by a streamed layer's copies - refusing a role
    whose tensor is absent with `TensorIndex`'s text, checking each by its
    kind and returning its `Weight` from the same check, so a resolved
    handle is well-formed by construction and no other path produces one:
    a norm is an F32 vector of `in`, a matrix or gathered table is
    `[in, out]` and a table an F32 `[in, out]`, each with trailing axes of
    one up to rank four, and an
    expert stack is exactly `[in, out, experts]`. Each weight goes to the
    device of its part, and a tensor two roles take on one device is adopted
    there once, which is how a tied head beside the embedding reads the
    embedding's buffer and how each tensor reaches each backend at most once.
    A tensor no role takes is never read, so an unused scalar or empty F32
    tensor, and views that share bytes, are accepted.
    A CPU backend starts its worker pool on its first parallel dispatch, so a fresh one has started no threads when these checks run.
  - Loading may enqueue uploads before a later tensor or allocation fails. The
    constructor catches failures inside its body and drains each used backend
    while its members are still alive. Normal model destruction also drains
    those backends before releasing weights, caches and position tables. Successful
    loading keeps uploads asynchronous.

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
