# `src/model/kv_cache.hpp` - logical KV cache

`infer::BlockPool` and `infer::KVSequence` are the backend-neutral half of the
paged KV cache designed in [KV-CACHE](../KV-CACHE.md). They hold no floats:
physical blocks live in a `backend::KVStorage` and the model layer never
computes an offset into them.

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
- `fork(length, tail)` is a second sequence holding the first `length`
  committed tokens: every full block below `length` shared by refcount and,
  when `length` ends inside a block, a fresh block for that partial tail,
  whose ids come back in `tail` for the backend's `kv_copy`. A length past
  the history is refused. Shared blocks are read-only: `prepare` refuses to
  append into one, which a history truncated into a shared block would do.
- Ownership: neither class is copyable and the pool is not movable, since
  sequences hold its address; `configure` sets the budget in place and is
  refused while blocks are held. A sequence returns its blocks when destroyed
  or moved from, and `prepare` on an unbound sequence is rejected. Vectors
  are reserved to the budget, so `alloc`, `release`, `abort` and `reset`
  never allocate and cannot fail half way. `Model` is not copyable or
  movable for the same reason.

`Model` owns one pool per device that runs attention, and one default
sequence; a `Sequence` holds a table per storage and `Model::fork` forks
every table at one length and copies every tail. The server keeps one
sequence per request over a shared pool, finds prefix donors by comparing
tokens in `server/scheduler.hpp` and forks a donor at the whole blocks it
shares, so it never takes a tail; nothing here indexes prefixes.

CTest's `kv-cache` test covers pool reuse and exhaustion, sequence
prepare/commit/abort/reset, on-demand CPU storage growth and retained reset
across block boundaries, paged attention over two different block tables
against a double-precision reference, and forks: a whole-block length
shares its blocks and allocates none, a length past the history is refused,
a length inside a block copies its tail, appends into shared blocks are
refused and release follows the refcounts. HF/model history checks remain
separate.
