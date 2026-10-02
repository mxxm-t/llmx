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
- `save_host(sequence, length, out, limit)` and `restore_host(host)`: a
  history's first `length` tokens, whole blocks of every storage, copied
  into a `HostHistory` in host memory and back (`docs/SPECULATIVE.md`,
  section 2, Host tier): each KV storage's blocks, read through
  `BlockKVStorage`'s buffers in runs of consecutive blocks, and on a model
  that keeps a state its checkpoint's slot, which must be at `length`, from
  every state storage, through `Backend::copy` into the backend's
  host-visible memory, enqueued on each device's stream behind the passes
  that wrote the history and not waited for, since whatever writes those
  blocks or that slot next comes after them on the same stream. The memory
  comes in 64 MiB slabs that `release_host`, which waits for the copies
  into and out of them, leaves to the model for the next copy, since
  allocating and pinning host memory costs far more than copying into it;
  the slabs alive, idle or holding a copy (`host_allocated`), stay within
  `limit`, a copy freeing idle slabs of other devices before it allocates
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
  back to the mark, and a rerun that throws drains the devices and keeps
  the mark, so the retract may be called again.
