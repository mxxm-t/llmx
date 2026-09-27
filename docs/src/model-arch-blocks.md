# `src/model/arch/blocks.hpp` - graph pieces the architectures share

The pieces of a pass that more than one architecture runs, in namespace
`infer::blocks`, each as backend ops over the weights and arena slots it is
handed. None names a tensor, a metadata key or an architecture
(`tests/arch_boundary.py` holds every header under `model/arch/` that the
registry does not include to that), so a module calls them with its own
roles and slots ([ADDING-AN-ARCHITECTURE](../ADDING-AN-ARCHITECTURE.md),
One owner). [qwen3](model-arch-qwen3.md) and [qwen35](model-arch-qwen35.md)
run them.

- `projection(weight, out)`: a weight's product into `out` for
  `matmul_group` and `matmul_experts`, the buffer passed by raw pointer so
  building one copies no shared pointer on the per-token path.
- `embed(step, table, ids)`: the table's rows the ids name, into the
  residual.
- `swiglu(step, gate, up, down, h, g, u, act)`: the SwiGLU feed-forward
  block over the normed rows `h`: gate and up grouped through
  `matmul_group`, `silu_mul`, and the down projection added into the
  residual through `matmul_add`, with the step's runs throughout.
- `head(step, norm, out, eps, rows)`: the rows that want logits gathered
  into the slot `rows`, since they are not contiguous once entries mix,
  normed, and projected once through `matmul_logits`.
