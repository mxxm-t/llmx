# `src/model/architecture.hpp` - the architecture contract

What an architecture gives the runtime ([runtime](model-runtime.md)), in
namespace `infer`: a plan, which is data the runtime resolves, validates,
adopts, fits, places and streams from, and the math of each part of a pass,
which the runtime calls. The runtime names no architecture; an architecture
never sees devices, placements, stages or caches beyond what a call hands it.

- The plan: `Role` (the id a resolved weight is indexed by, its `Part` -
  `embed`, `mixer`, `ffn` or `head` - its `RoleKind`, its tensor name and an
  alias taken when the name is absent, its expected `in`, `out` and
  `experts`, and its `Stream`; then `tensor` and `aliased`, the view it
  reads and whether that is the alias's, which `plan_model` sets and the
  architecture leaves alone), `LayerPlan` (the architecture's own kind for
  a layer, handed back on every call for that layer, whether its
  feed-forward part is routed, and its roles in adoption order) and
  `ModelPlan` (the size of a row of resolved weights, the vocabulary, the
  pass's roles and each layer's, the context length, and what the arena,
  the caches and the tables take: the residual row, the arena's slot
  widths, the K and V heads and head width of every layer, and the position
  tables' sizes).
  - A `RoleKind` says how a role's tensor is checked and whether the fit
    counts it as a product: `norm` is F32 `[in]`, `matrix` is `[in, out]`
    read by a product, `gather` is checked as a matrix and gathered by the
    embedding, and `experts` is exactly `[in, out, experts]`; a norm or
    matrix takes trailing axes of one up to rank four.
  - A `Stream` says what a routed layer run beside its mixer for a long
    prompt does with a role: nothing, a copy adopted on the mixer's device
    at load, or a copy written into that device's window in each pass that
    needs it.
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
  storage, the rows' positions, the position tables on that device, and a
  run list the part may rebuild, which the runtime reserves at a run per
  row so the part does not allocate. `HeadStep` adds the
  rows that want logits, their runs and the slice their logits go to.
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
