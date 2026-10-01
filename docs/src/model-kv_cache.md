# `src/model/kv_cache.hpp` - logical KV cache and state slots

`infer::BlockPool` and `infer::KVSequence` are the backend-neutral half of the
paged KV cache designed in [KV-CACHE](../KV-CACHE.md). They hold no floats:
physical blocks live in a `backend::KVStorage` and the model layer never
computes an offset into them. `infer::SlotPool`, `infer::StateSlot` and `infer::Checkpoint` are
the same half of the recurrent state ([QWEN35](../QWEN35.md), The recurrent
state), whose slots live in a `backend::StateStorage`.

- `BlockPool(max_blocks)` hands out dense block ids from a free list with a
  refcount per id. `alloc` throws when the budget is exhausted; `release`
  returns an id to the free list at refcount zero and rejects a second
  release. Ids ascend from zero until one is freed, which is what lets the
  backend back storage on demand instead of allocating the budget.
- `KVSequence(pool, block_tokens)` is one sequence's block table and committed
  `length`. `prepare(n)` takes the blocks positions `length .. length+n` need;
  `commit()` advances `length`; `abort()` returns the attempt's blocks and
  keeps the committed history. A failed `prepare` leaves the sequence as it
  was. `truncate(length)` rolls the committed history back to `length`,
  returning the blocks past it; `reset()` is `truncate(0)` and returns every
  block. The backend keeps the storage they used.
- `view(storage)` produces the `backend::KVView` that `kv_write` and
  `attention` consume: storage handle, block table, committed length and
  the rows prepared for this pass.
- `fork(length)` is a second sequence holding the first `length` committed
  tokens, a whole number of blocks: every block below `length` shared by
  refcount, with no new physical blocks or copying of their contents. The
  fork allocates its own logical block table. A length inside a block or past
  the history is refused. Shared blocks are read-only: `prepare` refuses to
  append into one, which a history truncated into a shared block would do.
- Ownership: neither class is copyable and the pool is not movable, since
  sequences hold its address; `configure` sets the budget in place and is
  refused while blocks are held. A sequence returns its blocks when destroyed
  or when another is moved into it, a move transfers them, and `prepare` on
  an unbound sequence is rejected. Vectors
  are reserved to the budget, so `alloc`, `release`, `abort` and `reset`
  never allocate and cannot fail half way. `Model` is not copyable or
  movable for the same reason.

- `SlotPool` hands out the state slots of a model, the same slot in every
  state storage, on three sides counted apart over the slots they share
  (`configure(live, checkpoints, marks = 0)`, refused while any is held): the live
  side, one per sequence that holds a state, where `acquire` throws when
  every live slot is held, `release` returns one and `available` counts
  the free ones, so a sequence the live side admits always finds a slot;
  and the checkpoint side, states kept at a position
  (`docs/SPECULATIVE.md`, section 1), where `acquire_kept` takes a slot
  with one reference, `retain` and `release_kept` count references and
  free it with the last, `keep_live` moves a held live slot to the
  checkpoint side when that side has room, and `kept_available` counts
  its free slots; and the mark side, a verify's starting states
  (`Model::mark`), where `mark_live` moves a held live slot there when the
  side has room, `unmark` moves it back to the live side and `release_mark`
  frees it, beside a buffer of saved inputs for each mark, taken by
  `acquire_mark_buffer` and returned by `release_mark_buffer` without
  allocating. A slot holds nothing a
  sequence must clear, since a history of length 0 reads a zero state
  whatever its slot holds. `StateSlot` is a sequence's hold on one slot,
  taken by `take`, which does nothing when one is already held, named by
  `slot`, and returned by `release`, when it is moved over or when it is
  destroyed, and `held` says whether it holds one; `forget` ends the hold
  without returning a slot `keep_live` or `mark_live` moved, and
  `adopt(pool, slot)` returns the slot it holds and takes a mark's slot back
  as its live slot. `MarkHold` is a sequence's hold on a mark, a buffer and,
  where the mark took the live slot, that slot on the mark side (`take`,
  which takes nothing where either has no room), each returned once by
  `release`, a move over it or its destruction; `give_slot` hands the slot
  back as a live slot, leaving the buffer to return. `Checkpoint` is one reference
  to a checkpoint's slot with its position (`slot`, `pos`), copied by
  taking another and returning the slot with the last; neither `StateSlot`
  nor the pool is copyable and the pool is not movable.

`Model` owns one block pool per device whose mixer layers keep KV, one slot
pool when a layer keeps a state, and one default sequence; a `Sequence`
holds a table per storage and `Model::fork` forks every table at one
length, and refuses a model that keeps a state. The server keeps one sequence per request over
a shared pool, finds prefix donors by comparing tokens in
`server/scheduler.hpp` and forks a donor at the whole blocks it shares;
nothing here indexes prefixes.

CTest's `kv-cache` test covers pool reuse and exhaustion, sequence
prepare/commit/abort/reset, on-demand CPU storage growth and retained reset
across block boundaries, paged attention over two different block tables
against a double-precision reference, and forks: a whole-block length
shares its blocks and allocates none, a length inside a block or past the
history is refused, appends into shared blocks are refused and release
follows the refcounts. HF/model history checks remain separate.
It also holds the backends' shared storage (`backends/kv_storage.hpp`) to the doubling rule it replaced: every growth step and the peak, and a failed growth leaving the accounting as it was.
