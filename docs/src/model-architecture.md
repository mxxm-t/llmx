# `src/model/architecture.hpp` - the architecture contract

`Step::dtype` carries the model's activation policy to its matrix calls; `HeadStep` inherits it. An architecture forwards it for ordinary projections and explicitly requests F32 for an operation whose arithmetic must remain F32, such as the Qwen3 MoE router. Row grouping itself carries no precision policy.

What an architecture gives the runtime ([runtime](model-runtime.md)), in
namespace `infer`: a plan, which is data the runtime resolves, validates,
adopts, fits, places and streams from, and the math of each part of a pass,
which the runtime calls. The runtime names no architecture; an architecture
never sees devices, placements, stages or caches beyond what a call hands it.

- The plan: `Role` (the id a resolved weight is indexed by, its `Part` -
  `embed`, `mixer`, `ffn`, `head` or `draft` - its `RoleKind`, its tensor name and an
  alias taken when the name is absent, its expected `in`, `out` and
  `experts`, and its `Stream`; then `tensor` and `aliased`, the view it
  reads and whether that is the alias's, which `plan_model` sets and the
  architecture leaves alone), `LayerPlan` (the architecture's own kind for
  a layer, handed back on every call for that layer, whether its
  feed-forward part is routed, what its mixer keeps for each sequence
  between passes, its `Cache`: keys and values for every position, a
  recurrent state of fixed size, or nothing, its roles in adoption order,
  and the ops of its parts that some backends lack, each an `OpUse`: the
  part and the `backend::Op`, and for a layer whose cache is a state the
  inputs its state's update reads, each a `Saved`: an arena slot, the
  floats a row takes in it and which block of the pass's rows it is, so
  plane 1 starts `rows` rows in. `replicated` keeps a saved row whole on every
  tensor-group member; otherwise its width is divided among members.
  `saved_floats(layer)` is their floats a
  row; and `state_tail`, additional F32 values after the common recurrent
  prefix in each slot of this layer; `token_history`, the count of preceding
  token IDs a recurrent layer keeps, and `token_padding`, the ID used before
  sequence start;
  `kv_aux`, an optional F32 width per KV position, stored in the same physical
  blocks as the layer's K and V and held whole on each tensor-group member,
  and `kv_pool`, the groups of positions whose pooled rows of `kv_pool_width` floats those blocks also keep
  (`kv_aux_floats()` counts both for the fit);
  `headroom`, the positions an entries layer's rings hold beyond those its rows read, the most rows a pass after a mark may take;
  `native`, the layer's widths under the cache type `native`, its pooled row and its state tail in 32-bit words of its declared formats, which `settle_cache` swaps with `kv_pool_width` and `state_tail` while `ModelPlan::native` says they are in effect, and `Step::native` tells the parts;
  `entry_source`, an earlier entries layer whose pooled rows an entries layer without its own reads, or -1: its source's storage layer on the source's device, and on a later device a replica there, which the first such reader fills (`Step::appends`);
  `lookup`, an optional `Lookup` naming a mixer gather role and its requested
  table rows per token; and
  `Cache`, what a layer keeps between passes: `kv`, `state`, `entries` (a state slot of its `state_tail` floats over the plan's state and, with `kv_pool`, pooled rows in the KV blocks with no K or V, docs/DEEPSEEK4.md section 4.3) or `none`; `keeps_slot(layer)` is true of a state and an entries layer and `keeps_kv(layer)` of a KV layer and an entries layer with pooled rows;
  `ModelPlan` (the size of a row of resolved weights, the vocabulary, the
  pass's roles and optional `pass_ops` (`OpUse`s issued by embedding or head
  parts beyond the common backend ops), each layer's roles and ops, the context
  length, and what the arena,
  the caches and the tables take: the residual row, the arena's slot
  widths, the K and V heads and head width of every layer whose cache is
  KV, the common recurrent-state prefix shape (`backend::StateShape`) of every layer
  whose cache is a state, with each layer's optional `state_tail` extending only
  its own slots, the tables' fixed floats (`tables`) and the floats each covered position adds (`table_steps`), the slots that hold a float for every `per` covered positions (`position_slots`), `image_streams` (per rotated pair, which of an image row's three positions turns it, empty where the model takes no image rows), `index_width`, the maximum number of host `uint32_t` lookup
  indices per activation row, reused by token embedding expansion and
  gathered-weight lookups (zero means no index scratch), and an embedded
  drafter's `LayerPlan` when a caller asked
  for one, its roles all `Part::draft`, which run on the head's device, and
  `draft_x`, the arena slot of a draft step's residual rows, which a tensor group sums into, and
  `draft_h`, the arena slot its context rows leave the target's
  final-normed rows in, 0 for a drafter whose context rows read no row of the target before each row, and `draft_block`, the most drafts one of its calls gives, 0 for one).
  `carries_row(plan)` holds where the drafter carries each sequence's last normed row from one pass to the next (`draft_h` nonzero), which decides the runtime's carried row, zero row, saved rows and the fit's count of them; `draft_calls(plan, k)` gives the calls that make k drafts.
  The drafter's cache is KV, or a state slot where its `LayerPlan` keeps entries.
  - A `RoleKind` says how a role's tensor is checked and whether the fit
    counts it as a product: `norm` is F32 `[in]`, `matrix` is `[in, out]`
    read by a product, `gather` is checked as a matrix and gathered by the
    embedding, `experts` is exactly `[in, out, experts]`, and `table` is
    an F32 `[in, out]` an op reads whole, such as a convolution's taps,
    which the fit does not count as a product, and `source` is an
    `[in, out]` of any type read on the host only to fill a table, never
    placed on a device nor counted in a device's bytes; a norm, matrix,
    table or source takes trailing axes of one up to rank four.
  - A `Stream` says what a routed layer run beside its mixer for a long
    prompt does with a role: nothing, a copy adopted on the mixer's device
    at load, or a copy written into that device's window in each pass that
    needs it.
  - A `Shard` says how a role splits over a tensor group: its `Axis` (none,
    output rows, or input columns) and the `ShardSection`s that cover it,
    each tiles of units of rows or columns, a unit replicated over members
    where the section allows; the architecture sets it after listing the
    role, and [shard](model-shard.md) turns it into each member's spans.
  - The context length bounds every pass and is the cache
    budget unless the options give one. The tables and the position slots cover `covered_positions`, the context capped at the budget: `table_floats` and `slot_floats` give their sizes at a number of positions, and `cover_positions` narrows the slots once a model knows its pool. The residual is the width of slot
    0, of a handoff row and of a crossing between devices.
  - `plan_model` holds every plan to slot 0 as wide as the residual and to
    role ids below `role_ids`, and the fit to one of its three pass fields
    for each pass role, a role of another part and kind needing a field of
    its own first; each is the architecture's error (`std::logic_error`).
- `Step`: one call of a part: the backend of the device it runs on and that
  device's arena with each slot's offset (`slot(i)`), the residual at the
  call's first row (`x`), the rows and their runs, the row of weights by
  role id (the layer's, the pass's, or a streamed layer's copies and
  windows), the layer's kind and the model's layer it belongs to (`layer`, -1 for the embedding, the head, the drafter and a group's sum), the cache views and the layer's index in its
  storage, the rows' positions, one a row or three where a pass holds an image's rows (`streams`), the position tables on that device, a
  run list the part may rebuild, which the runtime reserves at a run per
  row so the part does not allocate, and for a layer whose cache is a state
  its views (`states`, one per entry, each reading and writing its
  sequence's slot after the history the stage has committed) and its index
  in its device's state storage. If the layer declares `token_history`, the
  step also carries the pass's current token IDs and, per entry, the preceding
  IDs oldest first as `uint32_t`, with their count in `previous_count`.
  Sequence-start padding comes from the layer's `token_padding`. The runtime
  also lends `indices` and `indices_size`, a borrowed host U32 buffer sized
  for that device's arena rows and `index_width`; architecture calls may use
  it synchronously for embedding expansion or gathered-weight lookups and
  must not retain it. It is empty when the plan's width is zero. On a tensor
  group `width` is the group's and `partial` where the part's last projection writes the member's partial rows (`blocks::join`), which a routed part clears from `zero`, a row of zeros on the member; a member's weights are its shards, `Weight::nin` and `nout` its share, and a module runs its share of the heads, the plan's divided by the width, a KV head replicated where the width is a multiple of them. `HeadStep` adds the
  rows that want logits, their runs and the slice their logits go to.
  `DraftRowsStep` adds the rows' token ids and, where the drafter carries a row, per entry the row its first
  row reads as the target's row before it (the sequence's carried row, or
  a zero row for an empty history); `DraftStep` the device slices of a
  draft step's token ids, the rows before them where it carries a row (gathered by `prev_rows`
  where they are rows of the carried rows), the drafted ids, its output
  rows after the drafter's final norm and its logits, `ids` drafts a sequence (`draft_block`, 1 for none), draft j of sequence i at position `pos[i * ids + j]` in row j * `stride` + i, the output rows being the next call's `prev`.
  The cache views of both are the drafter's KV layer's or state layer's.
  An external table lookup adds `lookup`, a borrowed `PreparedLookup`: packed
  unique rows in `data`, their count in `rows`, and ordered request IDs remapped
  into that table in `ids`. The runtime prepares it before recording the graph;
  the architecture gathers it through the existing backend embedding op.
- `Architecture`: the interface an architecture implements, immutable once
  read from a file, so models built from one set of weights share it.
  - `plan(index)`: the plan of a file's tensors, looked up through the
    `TensorIndex` ([weights](model-weights.md)), with the architecture's
    own refusals. It declares optional embedding and head backend ops in
    `ModelPlan::pass_ops`; the runtime checks them on the backend assigned to
    that part before adopting weights.
  - `fill_tables(tables, sources)`: the position tables' values, into
    vectors the runtime sized as the plan says, and any table derived from
    weights, from `sources`, host views of the plan's `source` roles in plan
    order (the layers ascending, a layer's roles in order), read only during
    the call; the loader reads their bytes from each tensor's file span in
    every load mode and drops them once the model is built. The runtime
    keeps the tables for the model's life and adopts them on every device
    that runs a mixer part.
  - `embed(step, ids)`, `mixer(step)`, `ffn(step)` and `head(step)`, whose
    step is a `HeadStep`: the parts as backend ops. The runtime calls each once per layer per
    pass, or once per group of entries when a routed layer streams, and a
    part issues backend ops and nothing else.
  - `lookup_ids(step)`: fills `indices` from the current tokens, preceding
    token snapshots and views for a declared lookup, without backend work,
    allocation or I/O. The default refuses the call. Row reading and staging
    belong to the runtime; addressing belongs to the architecture.
  - `recur(step)`: a state layer's ops that update its state, from the rows
    its `saved` names in their slots, reading the views' `src` slots and
    writing their `dst` slots. The mixer runs them, and the runtime runs
    them again over a mark's kept rows when a retract lands inside a verify
    (`docs/SPECULATIVE.md`, section 1); a layer that keeps no state has
    none.
  - `plan_drafter(index, plan)`, `draft_rows(step)` and `draft(step)`: an
    embedded drafter a file carries (`docs/SPECULATIVE.md`, section 7):
    its plan, added to the model's when a caller asks for one and refused
    by default; its rows of the drafter's cache for every row of a pass,
    after the last stage; and one call through the whole drafter and
    the head into the next drafted ids, one a sequence or a block of `draft_block`. None by default.

On a tensor group each member of an embedded drafter's step computes the logits of its vocabulary rows of the head and their candidate at its slot (`DraftStep::member`, `DraftStep::first`, `blocks::draft_head`), and `Model::draft_record` sums the candidates and picks the id on every member.

A call of an embedded drafter on a tensor group runs in parts (`Step::phase`): the first `ModelPlan::draft_sums` each run a sublayer and end in the members' partial rows, which the group sums into the residual over the call's rows (`draft_block` rows a sequence), and each later part ends in each member's candidate of one draft, from which the group picks it; -1 runs all of it, as one device does.
With stages apart from the head a call runs in the same parts on any width, `draft_sums` an equal number a stage (`LayerPlan::stages`), each stage's parts on its device, and a context call (`DraftRowsStep`) in parts too, 0 the head's and 1 + st stage st's; the plan names what crosses between them (`ModelPlan::draft_carried`, `draft_ids`, `draft_rows_x`, docs/SPECULATIVE.md, section 9).
Qwen's MTP block takes two sum parts, its mixer and its feed-forward block, then one part for the final norm, the head and the argmax; DeepSeek-V4.1's DSpark takes two a stage, then a part a draft of its block, the first with the final norm and the head's logits, each with its Markov bias from the pick before it.
`ModelPlan::partial` is the width of a member's partial rows, 0 for the residual's (`sum_width`, the residual's where `embedding_split` finds the embedding role split, whose members' rows the group sums as the pass begins), which the group's sum adds to the residual; an architecture that sets it narrower (qwen4exp, whose parts end in a block injected into several streams) has the group sum the partial rows in place, and `join(step)` then runs on each member with `Step::partial` holding the sum and adds it into `Step::x` as one device's part ends (`sum_width` gives the width summed, which sizes the collective and the fit's count of it).
