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
  plane 1 starts `rows` rows in; `saved_floats(layer)` is their floats a
  row) and
  `ModelPlan` (the size of a row of resolved weights, the vocabulary, the
  pass's roles and each layer's, the context length, and what the arena,
  the caches and the tables take: the residual row, the arena's slot
  widths, the K and V heads and head width of every layer whose cache is
  KV, the recurrent state's shape (`backend::StateShape`) of every layer
  whose cache is a state, and the position
  tables' sizes, and an embedded drafter's `LayerPlan` when a caller asked
  for one, its roles all `Part::draft`, which run on the head's device, and
  `draft_x`, the arena slot of a draft step's residual rows, which a tensor group sums into, and
  `draft_h`, the arena slot its context rows leave the target's
  final-normed rows in).
  - A `RoleKind` says how a role's tensor is checked and whether the fit
    counts it as a product: `norm` is F32 `[in]`, `matrix` is `[in, out]`
    read by a product, `gather` is checked as a matrix and gathered by the
    embedding, `experts` is exactly `[in, out, experts]`, and `table` is
    an F32 `[in, out]` an op reads whole, such as a convolution's taps,
    which the fit does not count as a product; a norm, matrix or table
    takes trailing axes of one up to rank four.
  - A `Stream` says what a routed layer run beside its mixer for a long
    prompt does with a role: nothing, a copy adopted on the mixer's device
    at load, or a copy written into that device's window in each pass that
    needs it.
  - A `Shard` says how a role splits over a tensor group: its `Axis` (none,
    output rows, or input columns) and the `ShardSection`s that cover it,
    each tiles of units of rows or columns, a unit replicated over members
    where the section allows; the architecture sets it after listing the
    role, and [shard](model-shard.md) turns it into each member's spans.
  - The context length bounds every pass, sizes the tables and is the cache
    budget unless the options give one. The residual is the width of slot
    0, of a handoff row and of a crossing between devices.
  - `plan_model` holds every plan to slot 0 as wide as the residual and to
    role ids below `role_ids`, and the fit to one of its three pass fields
    for each pass role, a role of another part and kind needing a field of
    its own first; each is the architecture's error (`std::logic_error`).
- `Step`: one call of a part: the backend of the device it runs on and that
  device's arena with each slot's offset (`slot(i)`), the residual at the
  call's first row (`x`), the rows and their runs, the row of weights by
  role id (the layer's, the pass's, or a streamed layer's copies and
  windows), the layer's kind, the cache views and the layer's index in its
  storage, the rows' positions, the position tables on that device, a
  run list the part may rebuild, which the runtime reserves at a run per
  row so the part does not allocate, and for a layer whose cache is a state
  its views (`states`, one per entry, each reading and writing its
  sequence's slot after the history the stage has committed) and its index
  in its device's state storage. On a tensor group `width` is the group's and `partial` where the part's last projection writes the member's partial rows (`blocks::join`); a member's weights are its shards, `Weight::nin` and `nout` its share, and a module runs its share of the heads, the plan's divided by the width, a KV head replicated where the width is a multiple of them. `HeadStep` adds the
  rows that want logits, their runs and the slice their logits go to.
  `DraftRowsStep` adds the rows' token ids and, per entry, the row its first
  row reads as the target's row before it (the sequence's carried row, or
  a zero row for an empty history); `DraftStep` the device slices of a
  draft step's token ids, the rows before them (gathered by `prev_rows`
  where they are rows of the carried rows), the drafted ids, its output
  rows after the drafter's final norm and its logits, a row a sequence.
- `Architecture`: the interface an architecture implements, immutable once
  read from a file, so models built from one set of weights share it.
  - `plan(index)`: the plan of a file's tensors, looked up through the
    `TensorIndex` ([weights](model-weights.md)), with the architecture's
    own refusals.
  - `fill_tables(tables)`: the position tables' values, into vectors the
    runtime sized as the plan says; the runtime keeps them for the model's
    life and adopts them on every device that runs a mixer part.
  - `embed(step, ids)`, `mixer(step)`, `ffn(step)` and `head(step)`, whose
    step is a `HeadStep`: the parts as backend ops. The runtime calls each once per layer per
    pass, or once per group of entries when a routed layer streams, and a
    part issues backend ops and nothing else.
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
    after the last stage; and one draft row through the whole drafter and
    the head into the next drafted id. None by default.
