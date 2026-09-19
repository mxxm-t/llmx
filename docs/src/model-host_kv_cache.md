# `src/model/host_kv_cache.hpp` — CPU KV storage

`infer::HostKVCache` owns F32 keys and values for one CPU sequence, arranged
per layer and then per head, with contiguous positions within each head.
It depends only on the standard library. Logical token count, reset and model
execution stay in `Model`; attention computation stays in the backend.

- Construction records layer/head dimensions and the context limit without
  allocating storage for the full context.
- `reserve(want, used)` grows capacity geometrically, capped at the context
  limit, preserving every head's valid prefix. Replacement buffers are fully
  allocated and populated before replacing live storage. Old and new storage
  coexist during growth, so memory budgeting must include that temporary peak.
- `write(layer, k, v, pos, batch)` accepts token-major projections and writes
  them into the allocated per-head histories. Writes must fit the reservation.
- `keys(layer)`, `values(layer)` expose read-only host pointers;
  `head_stride()` gives the physical distance between heads in floats.

Reset changes the sequence's valid length to zero and may retain capacity.
Unused storage can contain old values and must never be included in attention.
Pointers must be reacquired after growth and are consumed synchronously today.
Concurrent requests, device buffers, paging and shared-prefix ownership are
future execution/server work, not implemented by this class.

CTest's `kv-cache` test checks distinct layer/head/position/lane values across
growth, retained-capacity reset and invalid extents. HF/model history checks
remain separate; this direct storage oracle does not replace them.
