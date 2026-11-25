# `src/model/passes.hpp` - a pass and its stages

The one owner of how a pass runs, in namespace `infer`: its plan, the stages of its layers over the devices, the crossings of the residual between devices, a streamed layer's split, the embedded drafter's rows after the last stage, and the storage of the context it runs in.
They are members of `Model`, declared in its class in `model/runtime.hpp` ([runtime](model-runtime.md)), which includes this file after the class, and defined here, beside the private steps only a pass and the history operations ([history](model-history.md)) take: `begin`, which plans a pass; `run_stage`, which runs one stage as `prepare_stage`, `record_stage` and `stage_commit`, the pass API's three calls and on a tensor group its group's: every member runs each part over its own residual on its shards (`Step::width`, `Step::partial`), a layer that keeps a state reads and writes each member's own state storage at the slots the group shares and saves a marked entry's inputs on each member, the group's collective sums the members' partial rows into every member's residual after each part, or where the architecture's partial rows are narrower sums them in place and each member joins them (`group_sum`, also the drafter's two sums), the residual comes in to every member and leaves from the first, and after the last stage each member's vocabulary slice of the logits rows is gathered by the first member into the context's rows; `single_prepare` and `single_record`, a stage of one device as the blocks it takes and its device work, and `group_prepare` and `group_record`, a group's; `stage_submit` and `stage_commit`, the submissions of every member and the commits that end a stage; `run_chunks`, a prompt's chunks pipelined over the stages, each stage that waits on its devices recorded on a thread of its own; for a token-history layer, `prepare_tokens` snapshots preceding U32 IDs on the history thread before its stage is recorded, then `stage_commit` advances the slot-indexed tail only after that stage completes; `draft_context`, the embedded drafter's rows after the last stage, on a tensor group every member of the head's group writing the K and V of its own KV heads and carrying the row itself; `finish` and `roll_back`; `alloc_arena` and `ensure`, the context's arena, handoff buffers and logits rows, `argmax_buffers`, an argmax pass's device rows and host ids, allocated once at the rows a verify takes (`argmax_rows`, at most the logits rows), refused for good where the device cannot hold them and passed over by a pass reaching past them, and on a tensor split each group's collective for the rows the arenas hold and each member of the head's group its slice of the logits rows; `cut_row`, `cut_parts`, `narrow` and `cut_record`, a pass cut in two parts (below); `send`, `receive` and `cross`; `ffn_split`, a streamed layer's groups; and `part` and `mixer_part`, the `Step` a part is called with.
The context and the pass are plain data in `model/runtime.hpp` (`ExecContext`, `Pass`).

The residual stream crosses devices wherever the placement changes, in two halves: the source copies the rows into its handoff buffer inside its own work (`send`), and the destination waits that submission's ticket and writes them (`receive`); inside a stage both run at once (`cross`), through buffer 0 serially, since a crossing inside a stage makes a split unpipelined.
Where the sender's backend serves the destination (`Device::served`, asked once at load from `Backend::handoff` over the devices `readers` walks a pass to), each handoff buffer has a `backend::Handoff`: `send` begins by waiting on the device for every served reader's last copy out of the buffer and ends by marking the rows sent, and `receive` enqueues the reader's wait for that mark and its own copies from the buffer, with no host wait or write; other readers of the same sender keep the host path.
Each carried slot of the plan (`ModelPlan::carried`) goes with the residual in every one of them at or past the start of its layer `from`, the first before which its rows are written: a handoff buffer holds slot 0's rows, then each carried slot's, each a region of the buffer's rows, so a plan without carried slots makes the copies it made before them.
`send`, `receive` and `cross` take the first layer whose part has not run where the crossing is and skip a slot written later, keeping its region of the handoff rows.
`send` and `receive` also take the slots a crossing carries in place of the residual and the carried ones (`crossed`), as an embedded drafter's stages apart from the head cross a draft step's rows, its input ids and a context call's rows (docs/SPECULATIVE.md, section 9).
`draft_context` runs nothing for a pass none of whose entries asks for context rows (`BatchEntry::context`), and otherwise the drafter's context rows after the last stage, whole on the head's devices, or with stages apart its first part there, a crossing of its rows to each other device holding stages and each stage's part on its device over that device's drafter layer, the head's state views over its storage, those devices' submissions given to the histories at the last stage's commit (`Pass::draft_tickets`).
`record_stage` holds the stage's devices' recording locks while stages sit apart or a device runs several stages (`Model::Recording`), so the drafter's parts recorded on them from the last stage's thread, the chain and another stage's thread wait for the stage's span; it swaps the matrix-path evidence in and out under those locks too.
A reader of an earlier layer's pooled rows (`LayerPlan::entry_source`) is handed its device's storage views with `kv_layer` the source's storage layer or its replica, and `appends` where it fills that replica.

`start_lookups` starts reading every layer's external lookup rows as the pass is formed (`begin`), into the pass's own scratch, with the preceding ids predicted from the committed tail where the token stage holds the history the pass follows, else from the tail the sequence's last formed pass leaves (`Sequence::formed_tail_`).
`start_lookup` computes the ids again from the stage's own snapshot as the stage begins, keeps the early read where they match and starts the read again where they do not, so the rows are those of a read started there, and counts each layer's read as kept, restarted or late (begun only there, where `start_lookups` started none), which `Model::lookup_reads` gives; `finish_lookup` takes them just before the layer and uploads them, for a single device or a tensor group.
As a stage opens, `single_record` and `group_record` call the architecture's `pass_values` on the stage's device or each member of its group, after the embedding or the received residual and before any mixer.
A group's stage may cut a prompt pass in two parts (`cut_row`, docs/TENSOR-SPLIT.md, section 4.3): stage 0 decides the row (`Pass::cut`) and every later stage cuts the pass at the same row, since a cut stage leaves the rows it carries to the next at its parts' rows; where every group's collective copies both parts' sums beside its work (`Collective::copies_beside`), a pass of at least `Model::kCutRows` rows, of one entry or several, with no image rows, streamed layers or marked entry, and recorded whole rather than a few layers a call (`record_pass_stage`), is cut at the last entry end or 64-row step from an entry's start that is a multiple of 4 rows at or below its middle, which keeps the recurrent layers' chunks where the whole pass has them.
`cut_parts` gives each part the entries its rows hold, the entry the cut falls in in both, and their runs clipped to its rows at the pass's extents, so each keeps the whole pass's row classes, its own rows of every arena slot (each slot's offset moved by the part's first row times the slot's floats a row), KV views whose history the second part continues from the first's rows, a state view whose second part reads the slot the first wrote, and the token history before its first row; `narrow` moves a part's `Step` to its rows and keeps the pass's offsets and rows, so a slot of planes over the pass's rows, which a step over the whole pass writes or reads, is addressed through `Step::plane` the same way in a part.
A failure part way through a cut pass drops the lanes it sent (`Collective::drop`) before it goes on, so the group's later sums are not refused.
`cut_record` records the layers part by part, each part's mixer or feed-forward part, then its sum sent on its lane (`Collective::send`), the sum finished and joined only before that part's next part, so one part's sum crosses while the other part's work runs; the second part's layer is recorded after the first's on every member's one queue, so it reads the cache and state the first wrote, and nothing waits on the host between parts.
The cut gives the whole pass's bits, which `tests/cut_passes.hpp` holds on CPU and Vulkan groups.

A layer with a device expert cache (`ExpertCache`, `Placement::expert_slots`) runs its feed-forward block on its mixer's device in `single_record`, never crossing to the host: the step pulls its missing experts into the slots through the device's ring and runs its entries in rounds of what the slots hold ([expert_ring](backends-expert_ring.md)).
`single_record` and `group_record` start every lookup of their stage before recording the layer operations and finish each before its layer's mixers, so the layers before it record and run while the rows are read.
The architecture's `lookup_ids`
fills dedicated IDs from the pass and its preceding-token snapshots;
[WeightRows](model-weight_rows.md) reads packed unique rows and returns their
request remap. The runtime converts the remap to U32, writes the packed rows
into each member's own backend buffer, and supplies a `PreparedLookup` through
`mixer_part`. The graph gathers with the original storage type and column width
against the unique-row count. The first group member owns the reader scratch
and remapped IDs, so the source is read once per layer and pass, not once per
member. Every member borrows those IDs until its gather calls consume them.
Resident tables retain their ordinary path.

`group_embed` runs the embedding on a group's members: whole rows where the table is replicated; where the plan splits it by rows (`embedding_split`), each member gathers the ids its rows hold and zero rows for the rest into its partial rows, the group sums them in place over the residual width and each member copies the sum into its residual.
The pass refuses a token id past the embedding's rows as it is formed, since a gather writes a zero row for one.

The row reader and its scratch perform no backend work. `Backend::write`
consumes the host source before returning; the packed backend buffer remains
owned by execution scratch through its consuming submissions. A failed read publishes no prepared view; the normal pass
rollback drains work before histories or storage are reused. Recurrent replay
reads saved inputs and performs no table lookup.

- A pass reading an image's rows needs them given (refused by name otherwise) and gives every row three positions, as the plan's `image_streams` turn them, and every other pass one; `begin` takes them from `positions` ([history](model-history.md)), and after the embedding `write_media` copies each image's rows the pass reads over their pad rows of the residual on every embedding device; the last stage's commit drops the rows of the images the history has read whole (`release_media`), and an embedded drafter's rows take the three positions too.
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
  is `Model::pipelined` ([runtime](model-runtime.md)). `reserve_prompt(rows)` makes the model's own context's arena for a prompt pass of `rows` rows as the model is built (`build_fitted`, [place](model-place.md)), so a command's first pass makes no arena; a perplexity run's first pass still makes its logits buffer (`ensure`'s `want`), which the fit prices. On a device that is memory the fit already took from the KV budget; on the CPU it is host memory a command takes at build even for a short prompt, `fitted_rows` rows of activations, about 200 MB on Qwen3.6-27B at 413696 bytes a row. `reserve_passes(ctx, slots, rows,
  logit_rows)` sizes a fresh context once: the arena for `rows` rows and,
  separately on each used device, host U32 lookup indices for `rows *
  plan.index_width` entries (no allocation when the width is zero).
  External lookups additionally reserve packed storage on every group member
  and reader/remap scratch on the first member for the maximum requested rows.
  Passes share these buffers per device. It also allocates a handoff buffer per slot on each device the
  residual leaves, two at least on a pipelined split and one for the
  single slot of a placement that is not pipelined (`handoff_buffers` in
  `layer_split.hpp`, the rule the fit counts by), and `logit_rows` rows
  of logits the caller hands out. The context is frozen from then on. A
  reservation that fails leaves the context fresh, so a smaller one may
  follow. More than one slot needs a pipelined placement.
  `begin_pass(ctx, slot, entries, n, logits_base)` plans a pass in a free
  slot, copying its tokens, and puts its sequences in flight; a sequence
  listed twice, a slot in use or beyond the reservation, and
  more rows or logits rows than reserved are refused before any work, and
  so is a sequence already in flight unless it continues a prompt as prefill's chunks do (docs/MULTI-DEVICE.md, Prefill beside decode): alone in its pass, each of its passes in flight holding it alone and past its first stage, none wanting its logits and none under a mark; it starts at the first stage's committed length. And
  so is a pass whose sequences without a state slot outnumber the free
  ones, which takes none; a pass takes its slots last, so one whose plan
  fails to allocate takes none either.
  `run_pass_stage(ctx, slot, s)` records the pass's next stage, which must
  be `s`, and whose entries' committed length there must be the pass's start (`at_stage`), so a continued prompt's earlier pass has run it first, refused before any work otherwise; a failure aborts the pass alone before it is rethrown, after the caller's `before_undo`, which the scheduler uses to quiet its stages' threads since undoing drains every device, and the other
  passes go on, a later pass of its sequence refused at its next stage until the caller abandons it. `prepare_pass_stage` holds the same check. A caller that records a stage on another thread takes it
  in three calls instead: `prepare_pass_stage` on its own thread, which
  takes the stage's blocks; `record_pass_stage` on the stage's thread,
  which records and submits and touches only that pass and the stage's
  devices, whole or `layers` layers a call, each group submitted with its tickets queued on the pass (`Pass::queued`, `Pass::spans`) and the last one the stage's, until it returns true (`Pass::open`, `Pass::layer` and `Pass::cur` keep the place);
  `pace_pass_stage` before a group waits beside the devices until fewer groups are queued than `queued_groups` gives within the wait the scheduler posted with the stage (`wait_ms`) for the stage's host recording of a replies' stage (`reply_host_ms`, the median of `Stage::reply_ms`, which `record_stage` fills without the recording's waits for the residual, lookup rows and sums) against the device time of the pass's last retired group (`Pass::group_ms`, `Backend::busy_ms`), and counts the waits, those that found every queued group retired and the groups each kept (`pacing`), and `pass_interleaves` says whether a pass in the replies' set may be recorded now between another pass's groups at its next stage, its residual having arrived (`Backend::done`); and `commit_pass_stage` on its own thread again, which commits
  the histories and counts the stage. A failed `record_pass_stage` leaves
  the pass for the caller's `abort_pass`.
  `reserve_passes` with `reply_rows` adds the replies' scratch set (`ExecContext::replies`, `ensure`), each device's own arena and lookup scratch, which the fit counts (`reply_rows` in [place](model-place.md)); a reservation that fails drops what it took and syncs each used device, so the memory is back before a caller tries fewer passes; `begin_pass` with `replies` puts a pass that fits it there (`Pass::scratch`, `pass_apart`), and every part, crossing, save and lookup of a pass goes through `Model::scratch` and `Model::slot` by the pass's set, `part` and `group_sum` taking the set, the context's where no pass runs them: a saved history's rerun and the drafters' chain ([history](model-history.md)).
  A pass keeps each device's submission of its stage (`Pass::tickets`), which its histories take as their last, since the context's last submission may be another pass's.
  `pass_logits(ctx, slot, i)` waits on the pass's own head
  and returns its wanting row `i`, written from row `logits_base` on; after an argmax pass, whose rows stay on the device, the output device first copies the row into its place in the host-visible rows (`logits_buf`), which the pass does not write.
  `argmax_pass(ctx, slot)` says whether the pass kept its rows on the device, and `pass_argmax_id(ctx, slot, i)` gives its row `i`'s argmax id from `argmax_ids`, the vocabulary size where the largest logit is not finite or the row holds a NaN; both wait as `pass_logits` does (`retired_pass`).
  `end_pass` takes a pass whose last stage has run out of flight, and
  `abort_pass` abandons one at any point: every device drained, then only
  its entries truncated in every storage to where it found them, its sequences' later passes rolled back first, newest first, each by the rows up to the furthest end in flight, so a rollback by length counts every chunk against the rings' headroom, and their slots returned. Each
  pass's stages run in order; passes interleave as the caller likes, since
  each device runs the stages recorded on it in that order and a pass
  keeps its own handoff buffer, logits rows and ticket. `forward`,
  `prefill`, `step` and `score` do not use it.

Under `ModelPlan::text_position_images` a pass finds the rows holding the image id once (`Pass::image_marks`, one entry a row, empty where no row holds it, and `Pass::image_runs`), and each step carries them (`Step::image_marks`, offset by the part's first row, and `Step::image_runs`), so a layer reads the image rows without finding them again.
