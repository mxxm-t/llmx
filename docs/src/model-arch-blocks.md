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

- `routed_ops(layer, tensors, gate, up)`: finds the two declared roles by id and adds, for the part the gate role runs in (the feed-forward part, or an embedded drafter's), the `routed_experts` requirement and, when both tensors exist with differing storage types, the `mixed_experts` one. The down projection is separate. Missing tensors stay with the runtime's existing resolution errors; this helper performs no schema validation.
- `shard(roles, id, axis, sections)`: declares role `id`'s split over a tensor group (`Role::shard`, [shard](model-shard.md)); a role the list lacks is the plan's error.
  `shard_swiglu(roles, gate, up, down, ff)`: the SwiGLU block's split, the gate and up projections by their `ff` rows and the down projection by as many columns.
  `shard_experts(roles, tensors, gate, up, down, ff, experts)`: a routed layer's split, the gate and up stacks by the `ff` rows of each expert and the down stack by as many columns, the router whole; each section carries the down stack's block size as its alignment, read from the tensor, so a member holds the blocks that cover its share ([shard](model-shard.md)); a shared expert is split by the same call as one expert.
- `join(step, weight, in)`: a part's last projection, whose output joins the residual: `matmul_add` into the residual on one device, and on a tensor group the member's partial rows into `Step::partial`, which the runtime sums into every member's residual; `swiglu`'s down projection and the modules' attention and `ssm_out` projections go through it.
- `projection(weight, out)`: a weight's product into `out` for
  `matmul_group` and `matmul_experts`, the buffer passed by raw pointer so
  building one copies no shared pointer on the per-token path.
- `embed(step, table, ids)`: the table's rows the ids name, into the
  residual.
- `swiglu(step, gate, up, down, h, g, u, act)`: the dense SwiGLU
  feed-forward block over the normed rows `h`: gate and up grouped through
  `matmul_group`, `silu_mul`, and the down projection joining the residual
  through `join`, with the step's runs throughout.
- `SharedExpert`: a routed layer's shared expert, its gate, up and down weights and the F32 vector gating its row, null for an ungated one.
- `moe_step(step, router, gate, up, down, k, rule, h, g, u, act, scores, ids, weights, shared, row_gate, zero, limit)`: a routed layer's feed-forward block as one backend op, `Backend::moe_step` ([backend](backends-backend.md)): the router's scores, each row's k experts picked and weighted by `rule` (`RouteRule`), their SwiGLU clamped at `limit` (0 for none, as `SharedExpert::limit` for the shared one) and weighted down projections, then `shared`'s SwiGLU scaled by sigmoid of its row gate into `row_gate`, all joining the residual, or on a tensor group the member's partial rows, which the step clears from `zero`, or else from the step's own row of zeros (`Step::zero`), as it starts.
  `g`, `u` and `act` hold a token's k routed rows and, past `rows * k` of them, the shared expert's rows, so the slots are as wide as both (the modules' slot widths); `zero`, a row of zeros the model keeps, has the step clear the residual first, as qwen4exp's expert block starts from zero.
  It builds the op's operands from the weights and the step's run list (`Step::scratch`, where the base op rebuilds the routed SiLU's runs) and calls the backend once; qwen3moe, qwen35moe and qwen4exp run it.
- `nextn_input(step, enorm, hnorm, eh_proj, pair, side, out, eps)`: an MTP
  block's input (docs/SPECULATIVE.md, section 7): `pair` holds the tokens'
  rows then the target's rows before them, each half normed in place, the
  two put side by side a row each, token first, through `side` (a single
  row already is), and projected by `eh_proj` into `out` with the step's
  runs.
- `NgramHash` and `ngram_rows(hash, vocab_size, previous, current, output)`: token-major table rows of an n-gram hash, computed without allocating; qwen4exp's PLE runs it, and DeepSeek-V4.1's Engram is to run it with one `NgramHash` a lookup layer (docs/DEEPSEEK4.md, section 4.4).
  The oldest-first history tail may hold at most `ngram_size - 1` tokens.
  Missing predecessors are filled with `fill_id`; with `fill_cuts`, the default, so is history at or before a `fill_id` in text, while without it a `fill_id` in text is an ordinary token; a current `fill_id` token still hashes its own context.
  A nonempty `token_map`, one entry for each id of the vocabulary, maps every id, the fill's included, into `[0, mapped_vocab)` before the fold, and the hash-product bound is then taken over `mapped_vocab - 1`.
  Each order past the first folds the (mapped) token ids, each times its multiplier, by xor and gives `heads_per_ngram` rows, the fold modulo a head's vocabulary size plus its offset.
  The fill id, every token id and its mapped id, the map's size, the output capacity and the signed hash-product bound are checked before output is written.
- `NgramGateWeights`, `NgramGateScratch` and `ngram_gate(step, weights, embeddings, embedding_dim, E, H, eps, gated, scratch, zero)`: the n-gram gate over the gathered rows: the H streams' keys and the value as one `matmul_group`, `group_rms_norm` of the keys and of the stream rows, and `signed_sqrt_gate` of the value into `gated`, a zero dot's sign as `zero` gives it (`ZeroSign::none` by default); qwen4exp's `ple` runs it before its convolution.
- `head(step, norm, out, eps, rows)`: the rows that want logits gathered
  into the slot `rows`, since they are not contiguous once entries mix,
  normed, and projected once through `matmul_logits`.
