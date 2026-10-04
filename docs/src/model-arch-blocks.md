# `src/model/arch/blocks.hpp` - graph pieces the architectures share

The shared feed-forward and output-head matrix calls forward `Step::dtype`; the routed router explicitly keeps F32. These graph pieces do not resolve a dtype or implement its arithmetic.

The shared planning and pass pieces that more than one architecture runs, in namespace
`infer::blocks`, each as backend ops over the weights and arena slots it is
handed. None names a tensor, a metadata key or an architecture, and
`tests/arch_boundary.py` holds every header under `model/arch/` that the
registry does not include to naming no architecture and no tensor, so a module calls them with its own
roles and slots ([ADDING-AN-ARCHITECTURE](../ADDING-AN-ARCHITECTURE.md),
One owner). [qwen3](model-arch-qwen3.md) and [qwen35](model-arch-qwen35.md)
run them.

- `routed_ops(layer, tensors, gate, up)`: finds the two declared roles by id and adds the `mixed_experts` requirement of the part the gate role runs in (the feed-forward part, or an embedded drafter's) when both tensors exist with differing storage types. The down projection is separate. Missing tensors stay with the runtime's existing resolution errors; this helper performs no schema validation.
- `shard(roles, id, axis, sections)`: declares role `id`'s split over a tensor group (`Role::shard`, [shard](model-shard.md)); a role the list lacks is the plan's error.
  `shard_swiglu(roles, gate, up, down, ff)`: the SwiGLU block's split, the gate and up projections by their `ff` rows and the down projection by as many columns.
- `projection(weight, out)`: a weight's product into `out` for
  `matmul_group` and `matmul_experts`, the buffer passed by raw pointer so
  building one copies no shared pointer on the per-token path.
- `embed(step, table, ids)`: the table's rows the ids name, into the
  residual.
- `swiglu(step, gate, up, down, h, g, u, act, row_gate)`: the SwiGLU
  feed-forward block over the normed rows `h`: gate and up grouped through
  `matmul_group`, `silu_mul`, and the down projection added into the
  residual through `matmul_add`, with the step's runs throughout. With
  `row_gate`, one value a row, `sigmoid_mul` scales each row the down
  projection reads by sigmoid of that value, as qwen35moe's shared expert
  is gated.
- `routed_experts(step, router, gate, up, down, k, norm, h, g, u, act,
  scores, ids, weights)`: routed experts over the normed rows `h`: the
  router's matmul into `scores`, `route_experts` choosing each row's k
  experts into `ids` and `weights` (renormalized when `norm`),
  `matmul_experts` for gate and up, `silu_mul`, and `matmul_experts_add`
  into the residual. The routed SiLU reads the expert products as k
  entries a token row, so it rebuilds the runs for them (`end * k`) in
  the step's run list (`Step::scratch`), which the runtime has reserved;
  the down projection takes the step's runs unchanged. qwen3moe and
  qwen35moe run it.
- `nextn_input(step, enorm, hnorm, eh_proj, pair, side, out, eps)`: an MTP
  block's input (docs/SPECULATIVE.md, section 7): `pair` holds the tokens'
  rows then the target's rows before them, each half normed in place, the
  two put side by side a row each, token first, through `side` (a single
  row already is), and projected by `eh_proj` into `out` with the step's
  runs.
- `head(step, norm, out, eps, rows)`: the rows that want logits gathered
  into the slot `rows`, since they are not contiguous once entries mix,
  normed, and projected once through `matmul_logits`.
