# `src/model/passes.hpp` - a pass and its stages

The one owner of how a pass runs, in namespace `infer`: its plan, the stages of its layers over the devices, the crossings of the residual between devices, a streamed layer's split, the embedded drafter's rows after the last stage, and the storage of the context it runs in.
They are members of `Model`, declared in its class in `model/runtime.hpp` ([runtime](model-runtime.md)), which includes this file after the class, and defined here, beside the private steps only a pass and the history operations ([history](model-history.md)) take: `begin`, which plans a pass; `run_stage`, which runs one stage, and `group_stage`, which runs it on a tensor group: every member runs each part over its own residual on its shards (`Step::width`, `Step::partial`), the group's collective sums the members' partial rows into every member's residual after each part, the residual comes in to every member and leaves from the first, and after the last stage each member's vocabulary slice of the logits rows is gathered by the first member into the context's rows; `end_stage`, the submissions of every member and the commits that end a stage; `draft_context`, the embedded drafter's rows after the last stage; `finish` and `roll_back`; `alloc_arena` and `ensure`, the context's arena, handoff buffers and logits rows, and on a tensor split each group's collective for the rows the arenas hold and each member of the head's group its slice of the logits rows; `send`, `receive` and `cross`; `ffn_split`, a streamed layer's groups; and `part` and `mixer_part`, the `Step` a part is called with.
The context and the pass are plain data in `model/runtime.hpp` (`ExecContext`, `Pass`).

The residual stream crosses devices wherever the placement changes, in two halves: the source copies the rows into its handoff buffer inside its own work (`send`), and the destination waits that submission's ticket and writes them (`receive`); inside a stage both run at once (`cross`).

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
  committed included. A sequence listed twice, in flight or whose state
  a failed pass lost is refused before any work, and so is a pass whose
  fresh sequences outnumber the free state slots and a context reserved
  for passes; `begin` takes the slots last, once the pass is planned.
- The pass API, for a scheduler that keeps passes of different sequences
  in flight so that every stage of a pipelined split works on one while
  the host samples another (`docs/MULTI-DEVICE.md`); whether that can pay
  is `Model::pipelined` ([runtime](model-runtime.md)). `reserve_passes(ctx, slots, rows,
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
  more rows or logits rows than reserved are refused before any work, and
  so is a pass whose sequences without a state slot outnumber the free
  ones, which takes none; a pass takes its slots last, so one whose plan
  fails to allocate takes none either.
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
