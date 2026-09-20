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
  was. `reset()` returns every block; the backend keeps the storage they used.
- `view(storage)` produces the `backend::KVView` that `kv_write` and
  `attention` consume: storage handle, block table and committed length.
- Ownership: neither class is copyable and the pool is not movable, since
  sequences hold its address; `configure` sets the budget in place and is
  refused while blocks are held. A sequence returns its blocks when destroyed
  or moved from, and `prepare` on an unbound sequence is rejected. Vectors
  are reserved to the budget, so `alloc`, `release`, `abort` and `reset`
  never allocate and cannot fail half way. `Model` is not copyable or
  movable for the same reason.

`Model` owns one pool and one sequence today. The server keeps one sequence per
request over a shared pool; fork, prefix sharing and completion-gated release
are the later steps in the design, not implemented here.

CTest's `kv-cache` test covers pool reuse and exhaustion, sequence
prepare/commit/abort/reset, on-demand CPU storage growth and retained reset
across block boundaries, and paged attention over two different block tables
against a double-precision reference. HF/model history checks remain separate.
