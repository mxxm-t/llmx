# `src/model/history.hpp` - a sequence's history

The one owner of the operations on a sequence's history (`docs/SPECULATIVE.md`, section 1), in namespace `infer`.
They are members of `Model`, declared in its class in `model/runtime.hpp` ([runtime](model-runtime.md)), which includes this file after the class, and defined here, beside the private steps only they take: `settle`, which refuses a sequence of another model or in flight and waits on its last tickets; `rewind`, which a retract, a failed pass and a prompt that fails part way go through; `restore_mark` and `drop_mark`; `save`, which a pass runs after each state layer's mixer, and `rerun`, both at the offsets `saved_at` gives.
The blocks, slots and holds these operations move are `model/kv_cache.hpp`'s ([KV cache](model-kv_cache.md)).

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
- On a tensor group each member's storage is copied on its own, its KV blocks and the state slot of its own heads; `restore_host` takes the blocks from the pool on the group's first member, before the others, which copy into the same blocks.
- On a tensor group a rerun runs on each member over the inputs that member saved, at its share of each width, with no sum.
- `save_host(sequence, length, out, limit)` and `restore_host(host)`: a
  history's first `length` tokens, whole blocks of every storage, copied
  into a `HostHistory` in host memory and back (`docs/SPECULATIVE.md`,
  section 2, Host tier): each KV storage's blocks, read through
  `BlockKVStorage`'s buffers in runs of consecutive blocks, and on a model
  that keeps a state its checkpoint's slot, which must be at `length`, from
  every state storage, with an embedded drafter's carried row of that slot, through `Backend::copy` into the backend's
  host-visible memory, enqueued on each device's stream behind the passes
  that wrote the history and not waited for, since whatever writes those
  blocks or that slot next comes after them on the same stream. The memory
  comes in 64 MiB slabs (`host_slab_bytes`) that `release_host`, which waits for the copies
  into and out of them, leaves to the model for the next copy, since
  allocating and pinning host memory costs far more than copying into it;
  the slabs alive, idle or holding a copy (`host_allocated`), stay within
  `limit` (`trim_host` frees idle ones down to a limit after a copy allocated beyond it), a copy freeing idle slabs of other devices before it allocates
  (`detail::slabs_to_free`) and being refused where they cannot make room,
  or where the slabs it allocates would leave the host less free memory
  than the reserve the fit keeps on it (`detail::host_room`, over
  `CpuBackend::host_reserve`). `caches_on_devices` says whether some KV or
  state storage sits off the CPU, without which a copy gains nothing.
  A copy that fails part way retires every device's stream before its slabs
  go back, and the pools reserve room for every slab of their device, so a
  release allocates nothing.
  A restore makes a fresh history over new blocks and, on such a model, a
  checkpoint slot of its own, its copies ahead of the history's first
  pass, and only the model that wrote the copy restores it; a throw from a
  pool, a slot or a copy leaves nothing held. `host_bytes(length)` gives
  the slabs a copy of `length` tokens takes.
  `alloc_host(length, out, limit)` takes the slabs a copy of `length`
  tokens takes, holding nothing yet, as `save_host` takes them before its
  copies, so the server's disk tier reads an entry back into them.
  `wait_host(host)` waits for the copies into and out of a copy's slabs, so
  the host can read its bytes, as the server's disk tier does before it
  writes them to a file.
  Without `blocks`, `save_host` copies the checkpoint's state alone (`HostHistory::blocks` false), and `fork(sequence, length, state)` continues from it: a second history sharing the source's blocks below `length`, as `fork` does whatever checkpoint the source holds, with the state copied back into a checkpoint slot of its own at `length`, which its first pass reads in place; the server keeps a conversation's message boundaries this way, the source a later history of the same conversation whose rows below `length` are the ones the state was computed after.
  With an embedded drafter the state alone carries the drafter's carried row of its slot too, and the fork copies it into its own slot, so it drafts as the history it was taken from.
  `restore_host` refuses a state alone, and a fork with a state refuses a whole history's copy or another length; a throw from the slot or a copy holds nothing.
  `save_host_blocks(sequence, first, length, out, limit)` copies the blocks of a token range alone, with no state and needing no checkpoint (`HostHistory::first`, `HostHistory::state` false): what a turn added to a history whose earlier blocks are kept elsewhere ([DISK-TIER](../DISK-TIER.md), Entries written as what changed). `host_ranges(host, first, length)` gives where a range's blocks lie in a host history that holds them, a layer's K and then its V on each device, in the order a copy of that range alone holds them end to end, so a range copy and the same range of a whole copy are the same bytes in the same order; `host_state_ranges(host)` gives where the state lies, after the blocks. `restore_host` takes a whole history only.
  A copy records its bytes per device (`HostHistory::device_bytes`), the run of each device's slabs.
- `host_identity()`: what a copy to host memory holds and how, beyond the model file and the build, as text: every device's identity (`Backend::identity`) and effective dtype, the cache types, each device's runs (its KV storage's layers, block tokens and block bytes, its state layers and slot bytes, the carried row) and the placement, then the row classes from extent 1 to the most a history holds (the model context or the KV pool), given where they change; two models of one file and build with equal identities restore each other's copies (`docs/DISK-TIER.md`, The entry file).
- `mark(sequence)`: the history kept at its length while one pass runs
  past it, so a retract into that pass reaches any of its rows exactly, as
  a verify of drafts needs (`docs/SPECULATIVE.md`, section 1). On a model
  that keeps no state it does nothing, since its caches reach every
  length; on one that does, the live slot moves to the pool's mark side
  and becomes the next pass's `src`, the pass writing a fresh live slot,
  and after each state layer's mixer the pass copies the marked entry's
  rows of the inputs its update read (`LayerPlan::saved`) into the mark's
  buffer. The sequence holds the mark through a `MarkHold`, so a marked
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
  A sequence in flight, one no pass has fed and a model without a drafter are refused; `draft(sequence, last, k, out)` is the batch of one.
