# `src/model/history.hpp` - a sequence's history

The one owner of the operations on a sequence's history (`docs/SPECULATIVE.md`, section 1), in namespace `infer`.
They are members of `Model`, declared in its class in `model/runtime.hpp` ([runtime](model-runtime.md)), which includes this file after the class, and defined here, beside the private steps only they take: `settle`, which refuses a sequence of another model or in flight and waits on its last tickets; `rewind`, which a retract, a failed pass and a prompt that fails part way go through; `restore_mark` and `drop_mark`; `save`, which a pass runs after each state layer's mixer, and `rerun`, both at the offsets `saved_at` gives.
The blocks, slots and holds these operations move are `model/kv_cache.hpp`'s ([KV cache](model-kv_cache.md)).

- Images ([VISION](../VISION.md), section 4): `add_media(sequence, run)` gives a sequence an image's rows, a `MediaRun` at or past its history and after every image it holds, refused on a model that takes no image rows or beside an embedded drafter, with rows that do not fill the image's merged grid or are not the embedding's width, or past the context.
  `add_media` gives an image: one the sequence carries, from a fork or a host copy, given again must be the same image (place, grid and digest) and fills its rows; any other comes after every image held, below the history only where nothing carries its rows past it, as after a disk restore; rows are needed where the history has not read the image whole, and they may be only the image's last rows, from `MediaRun::from` on, where the history has read the rows before them, which is refused otherwise.
  An image's rows are uploaded to an embedding device the first time a pass there reads them, and dropped, host and device copies alike, when the pass reading the last of them commits (`release_media`), each device copy freed once its last reading submission has retired (`retiring_`); `reset` drops every image and `rewind` those at or past the length it reaches.
  `positions(sequence, first, n, streams, out)` is the one owner of a row's position, as HF's `get_rope_index` places Qwen's images: a text row at its index less what the images below it saved, an image's row at (p, p + row, p + column) of its merged grid, p the position its first row would have had as text, and the text after it at p plus the larger side of the grid; a history without images keeps its indices.
  `fork`, `save_host` and `save_host_blocks` and `fork` with a state carry the images below their length as places, grids and digests without rows (`carried`, `HostHistory::media`), which `restore_host` gives the restored sequence; a disk restore carries none, and the owner gives every image again, as the server does (docs/VISION.md, section 5).
  A drafted token's position follows the images below it, and `image_rows` gives a sequence's image rows, which the lookup drafter does not draft from.

- `reset(sequence)`: the history returned
  to the pools, its blocks and its state slot, after waiting on its last
  ticket; a sequence in flight is refused.
- `fork(sequence, length)`: a second history holding the first `length`
  tokens, which must be whole blocks in every storage, sharing every block
  below `length` on every storage without allocating or copying physical
  KV blocks; the logical block tables and ticket vectors still allocate. The
  server forks a donor at the blocks a prompt shares with it.
  On a model whose layers keep a state `length` must be the source's checkpoint, whose state the fork's first pass reads in place.
  A forked sequence continues exactly as a fresh one fed the same tokens at the same extents would; rows another extent computed can differ from them by rounding, so a caller forks only rows whose class (`row_class`) is its own.
  A sequence in flight is not forked.
- Checkpoints: a model whose layers keep a recurrent state forks and takes a history back only at a checkpoint (`docs/SPECULATIVE.md`,
  section 1). A sequence holds one checkpoint at most, a state kept at a
  position in a slot of the pool's checkpoint side, counted by reference
  since a fork reads it in place: a `BatchEntry::keep` entry writes its
  state into a fresh checkpoint slot (`StateView::dst`), which the next
  pass reads (`src`) as it writes the live slot, and which replaces the
  sequence's older checkpoint once the entry's last stage commits it;
  `keep(sequence)` makes the live state at the current length the
  checkpoint between passes, its slot moving to the checkpoint side.
  `retract(sequence, length)` is the one call that shortens a history: to
  `length` where the caches hold it, else to the checkpoint at or below
  it, else to 0, returning the length reached; a failed pass, or a prompt
  that fails part way, takes its entries back the same way, since a pass
  writes the live state in place. `checkpoint(sequence)`
  gives its position.
  `retract(sequence, length, defer)` does the same, but a rerun inside a mark is handed to `defer` as one `StageItem` a device, its stage and its recording (`rerun_on`), which the caller runs on that stage's thread: the lengths and blocks change at once, the stage's queue orders the rerun before the history's next pass, and the history's `Deferred` holds the parts left, each device's ticket and the mark's hold, which go back when the history next settles (`settle`, waiting with `on_stage_wait` around it).
  The parts and a drafter's chain take the model's context one at a time (`context_m_`).
  `wait_ticket(device, ticket)` is the caller's one wait on a device's ticket, told to `on_stage_wait` where the work has not yet retired, so a server counts every wait of its thread on a device (`settle`, `wait_host`, `release_host`).
  `release_settled(sequence)` takes a retract's parts' tickets and gives the mark's hold back once they have all run, without waiting, so another verify can take that mark before the history's next settle; a restore's record and a part's failure stay.
- On a tensor group each member's storage is copied on its own, its KV blocks and the state slot of its own heads; `restore_host` takes the blocks from the pool on the group's first member, before the others, which copy into the same blocks.
- On a tensor group a rerun runs on each member over the inputs that member saved, at its share of each width, with no sum.
- `save_host(sequence, length, out, limit)` and `restore_host(host)`: a
  history's first `length` tokens, whole blocks of every storage, copied
  into a `HostHistory` in host memory and back (`docs/SPECULATIVE.md`,
  section 2, Host tier): each KV storage's blocks, read through
  `BlockKVStorage`'s buffers in runs of consecutive blocks, and on a model
  that keeps a state its checkpoint's slot, which must be at `length`, from
  every state storage, including each layer's complete per-slot stride and any
  `LayerPlan::state_tail`, with an embedded drafter's carried row of that slot,
  through `Backend::copy` into the backend's
  host-visible memory, enqueued on each device's stream behind the passes
  that wrote the history and not waited for, since whatever writes those
  blocks or that slot next comes after them on the same stream. The memory
  comes in 64 MiB slabs (`host_slab_bytes`) that `release_host`, which waits for the copies
  into and out of them, leaves to the model for the next copy, since
  allocating and pinning host memory costs far more than copying into it;
  the slabs alive, idle or holding a copy (`host_allocated`), stay within
  `limit` (`trim_host` frees idle ones down to a limit after a copy allocated beyond it), a copy freeing idle slabs of other devices before it allocates
  (`detail::slabs_to_free`, and with the slabs it then allocates anew `detail::copy_slabs`) and being refused where they cannot make room,
  or where the slabs it allocates would leave the host less free memory
  than the pool's reserve.
  When a layer declares `token_history`, the exact U32 tail is appended once
  to the host run for the first backend of its owning stage, after that
  device's state slots and carried row. `host_state_ranges` includes it;
  restore and state-only fork copy it back into the new slot with the same
  slot id mapping. This tail is transferred through the backend's host
  `write` and `read` operations, before enqueueing that device's copies.
  Vulkan synchronizes those calls, so transfer can wait for earlier device
  work, but not the new KV, recurrent-state and carried-row copies.
  That reserve is `CpuBackend::host_reserve` of what the host had as the model was made, fixed then, so it does not shrink as memory falls; `host_admits(bytes, optional)` is the one rule every fresh slab passes, a copy's, a read's and an idle step's (`alloc_host`), with twice the reserve for growth nothing waits for, and it reads the host's memory at each call, so it is a sampled guard and no guarantee against another process.
  `host_short()` says the host is below the reserve while idle slabs exist, and `host_yield()` then frees idle slabs, as many as cover what the host is short by, and no slab a copy holds.
  `ModelOptions::host_memory` gives the host's free memory in tests.
  `copies_retired(tickets)` says whether a host history's copies have all retired, asked through each device's `Backend::done`, which a device backend answers without waiting; the server asks it once a round to time a promotion's copies.
  `restore_into(length)` makes the history a restore copied in piece by piece fills, as `restore_host` does from a whole copy: its blocks and checkpoint slot are taken first. `detail::block_runs` and `detail::state_runs` give both `HostSpan` and the restore the same ordered K, V and auxiliary blocks and complete per-layer state slots, followed by the drafter's carried row and host token history. A KV run resolves its storage's layer and side at each copy (`RestoreInto::Run::target`), including auxiliary planes, since growth replaces the buffers. A host destination names the fixed token-history array allocated once when the model is constructed; the restore's checkpoint slot holds that slice until the read and copies have retired. Nothing reads or forks the filling history until then, and `restored` gives its passes the copies' tickets and its stages without KV their length.
  `caches_on_devices` says whether some KV or
  state storage sits off the CPU, without which a copy gains nothing.
  A copy that fails part way retires every device's stream before its slabs
  go back, and the pools reserve room for every slab of their device, so a
  release allocates nothing.
  A restore makes a fresh history over new blocks and, on such a model, a
  checkpoint slot of its own, its copies ahead of the history's first
  pass, and only the model that wrote the copy restores it. If reserving a
  pool block or state slot, queuing a copy or submitting on any device
  throws, the restore retires what it recorded before returning the error
  and releasing its checkpoint slot (`drain_recorded`: a device it submitted
  on waits on that ticket, one it recorded on submits and waits, synced alone
  should that fail, and a device it never touched is left alone, since
  another thread may be recording on it); the host slabs remain safe to
  retry or release. `host_bytes(length)` gives the slabs a copy of `length`
  tokens takes, including the fixed-width U32 token-history tail when the
  model declares one.
  `alloc_host(length, out, limit)` takes the slabs a copy of `length`
  tokens takes, holding nothing yet, as `save_host` takes them before its
  copies, so the server's disk tier reads an entry back into them.
  `host_fresh(length)` counts the slabs such a copy would allocate anew now, each device's need past its own idle slabs, which is what the server bounds an idle step's allocations by.
  `host_copy_fits(length, limit, freed)` says whether a copy would get its slabs now, as `alloc_host` takes them (`copy_slabs`), with `freed` idle slabs more on each device from the copies released first, its fresh slabs within the reserve read as if the idle slabs freed for the limit were given back: it fits, is past the limit, or the host lacks the memory; the server's `host_room` asks it before a copy is kept.
  `wait_host(host)` waits for the copies into and out of a copy's slabs, so
  the host can read its bytes, as the server's disk tier does before it
  writes them to a file.
  `save_host` and `save_host_blocks` with a `Defer` hand their device work to the stages' threads, one `StageItem` a device (`copy_out`): the slabs reused and counted and the block ids, token tail and slot read at once, each part allocating its device's fresh slabs, recording its copies and submitting on its stage's thread, and the idle slabs freed for the limit freed there too (`free_slabs`).
  The copy then holds a `HostPending` record of its parts, which `host_settle` takes once they have run, waiting for them or not: their fresh slabs joined to the copy's, their tickets and the first error (`HostHistory::error`); every reader of a copy settles it first (`release_host`, `wait_host`, `restore_host`, `fork` with a state), and one whose part failed is not restored (`settled_whole`).
  `restore_host` and `fork` with a state take a `Defer` the same way (`copy_in`): the blocks and checkpoint slot are taken and the token tail read from the slabs' host memory at once, and each device's copies run on its stage's thread, the history's passes queued behind them, their tickets going to the copy read from and to the history's `Sequence::Deferred`, marked `restore`.
  A part that fails gives its error to the history alone and throws, so its stage's thread reports the restore (`StageItem::restore`); `settle` keeps a failed restore's record on the history, `Sequence::restoring` finds every holder of it, a fork sharing the record of a history still landing, `reset` forgets it, and a save of such a history is refused.
  `Sequence::landing` says whether a restore's copies are still landing.
  `take_host` takes a copy's slabs as `alloc_host` does, with a `Defer` the fresh ones allocated on their devices' stages' threads and waited for, so the host can use them on return; `grow_slabs` allocates them there with nothing copied, the server's idle growth; `trim_host` and `host_yield` with a `Defer` free their slabs there (`free_slabs`), and `retract` with a `Defer` frees the images it drops there (`drop_media`).
  Without `blocks`, `save_host` copies the checkpoint's state alone (`HostHistory::blocks` false), and `fork(sequence, length, state)` continues from it: a second history sharing the source's blocks below `length`, as `fork` does whatever checkpoint the source holds, with the state copied back into a checkpoint slot of its own at `length`, which its first pass reads in place; the server keeps a conversation's message boundaries this way, the source a later history of the same conversation whose rows below `length` are the ones the state was computed after.
  With an embedded drafter the state alone carries the drafter's carried row of its slot too, and the fork copies it into its own slot, so it drafts as the history it was taken from.
  `restore_host` refuses a state alone, and a fork with a state refuses a
  whole history's copy or another length; a throw from the slot or a copy
  holds nothing, its copies retired by `drain_recorded` as a restore's are.
  `save_host_blocks(sequence, first, length, out, limit)` copies the blocks
  of a token range alone, with no state and needing no checkpoint
  (`HostHistory::first`, `HostHistory::state` false): what a turn added to a
  history whose earlier blocks are kept elsewhere ([DISK-TIER](../DISK-TIER.md),
  Entries written as what changed). `host_ranges(host, first, length)` gives
  where a range's blocks lie in a host history that holds them, a layer's K,
  V and enabled auxiliary plane on each device, in that order, and in the
  order a copy of that range alone holds them end to end, so a range copy
  and the same range of a whole copy are the same bytes in the same order;
  `host_state_ranges(host)` gives where the state lies, after the blocks.
  `restore_host` takes a whole history only.
  A copy records its bytes per device (`HostHistory::device_bytes`), the run of each device's slabs.
- `host_identity()`: what a copy to host memory holds and how, beyond the model file and the build, as text: every device's identity (`Backend::identity`) and effective dtype, the cache types, each device's runs (its KV storage's layers, block tokens and block bytes, its state layers and core slot bytes, the carried row) and the placement, then the row classes from extent 1 to the most a history holds (the model context or the KV pool), given where they change. When any auxiliary width is nonzero, identity appends the per-layer F32 widths; with all widths zero, the previous identity text is unchanged. If any state layer has a tail, the identity appends the full per-layer slot strides; if token history is declared, identity version 2 also records its owner stage, width and padding ID. Two models of one file and build restore each other's copies only when these identity strings match (`docs/DISK-TIER.md`, The entry file).
- `can_draft(sequence)`: whether a draft of it may be asked for: it holds a history and, on a model that keeps a state, a pass has fed it since its fork, a forked sequence reading its checkpoint in place until then.
- `mark(sequence)`: the history kept at its length while one pass runs
  past it, so a retract into that pass reaches any of its rows exactly, as
  a verify of drafts needs (`docs/SPECULATIVE.md`, section 1). On a model
  that keeps no state it does nothing, since its caches reach every
  length; on one that does, the live slot moves to the pool's mark side
  and becomes the next pass's `src`, the pass writing a fresh live slot,
  and after each state layer's mixer the pass copies the marked entry's
  rows of the inputs its update read (`LayerPlan::saved`) into the mark's
  buffer. For declared token history, the owner stage also saves the pass's
  current U32 token IDs in the host mark buffer after it commits. A retract
  replays only the accepted IDs from the mark's slot tail into the live slot;
  rejected IDs never become its preceding-token history. A layer's
  `state_tail` stays in its ordinary state slot; an
  architecture that updates it must make `Architecture::recur` reproduce
  that tail over the kept rows from the saved inputs. The sequence holds the mark through a `MarkHold`, so a marked
  sequence destroyed or moved from returns its holds once, and every
  allocation comes before a hold is taken, so a mark that fails changes
  nothing. It returns false, marking nothing, when no mark is free; a
  second mark, a fork or a `keep` of a marked sequence, a second pass
  after a mark, a pass of more rows than `mark_rows` and a `keep` entry
  after a mark are refused. `retract(sequence, length)` inside the pass
  after a mark runs the state's update again from the mark over the kept
  rows' saved inputs into the live slot (`Architecture::recur`, in the
  model's own arena), at the mark's position takes the mark's state back,
  and at the pass's end keeps it; the mark then goes. A failed pass goes
  back to the mark, and a rerun that throws turns every device's unordered recording off, drains the devices and keeps
  the mark, so the retract may be called again.
  With an embedded drafter the pass after the mark saves its rows' final-normed rows too (`save_h`), and the rerun copies the last kept row's into the live slot's carried row, so the history carries the row of its last kept token.
- `draft(asks, n)`: for each of `n` sequences (`DraftAsk`), up to its `k` drafts of the tokens after its `last`, the history's last pick not yet fed, from an embedded drafter (docs/SPECULATIVE.md, section 7), in its `out`: one submission on the head's device of a step a draft (`Architecture::draft`), step m a row for each sequence whose chain is longer than m, at that history's length plus m, reading the token drafted before it, the first `last`, and the row before it, the first the row the history carries; each row's id is written and read on the device, and the host reads the ids once.
  The sequences take their rows longest chain first, so a step's rows are those of the step before it less the last ones, and the drafter's weights and the head are read once a step for all of them; a row computes what it computes alone.
  The rows write the drafter's KV at their positions into blocks taken for the chain and returned after it, a failure included, so the committed lengths are unchanged and the verify overwrites those rows before anything reads them; a sequence's drafts end before its first that is not an id of the vocabulary.
- `draft_begin(asks, n, behind)`, `draft_record(chain)`, `draft_end(chain)`: `draft` in three parts (`Chain`), so a server records the chain on the head stage's thread: the begin settles each history, takes the KV rows its drafts write and lays out every step's rows, positions and views, its sequences in flight until the end; the record runs the steps on the head's devices under the context lock, waits for them and reads the ids back into the chain, since the next chain writes the same buffers; the end returns the KV rows, takes the sequences out of flight and, where the record ran, writes each history's drafts.
  With `behind` the begin settles no history, for a caller that records the chain behind the histories' rerun parts on the head stage's thread, which order it after them on the head's devices.
- `reserve_drafting()`: the model's context for a mark's rerun (`mark_rows` rows on a model that keeps a state) and for a chain of every mark's sequence, and the drafter's ids and logits for those chains at `mark_rows` - 1 drafts, taken at once, so a server's chains and reruns allocate nothing while a stage's thread records; the fit counts both (`Footprint::context_rows`, `Footprint::drafter_cache`).
  A sequence in flight, one no pass has fed and a model without a drafter are refused; `draft(sequence, last, k, out)` is the batch of one.
  On a tensor group every member of the head's group runs each step on its shards of the block in three calls (`DraftStep`, `Step::phase`), the group's collective summing the members' partial rows into the step's residual (`ModelPlan::draft_x`) after the block's mixer and after its feed-forward block; each member reads the head whole and finds the row's id itself, into ids of its own, and a sequence's drafts end before the first id the members do not agree on.
