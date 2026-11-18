#pragma once
// What the GPU backends' shared core (backend_core.hpp) shows a vendor's public header, so its tests can name it.

namespace backend {

// The forms of a tensor group's fast sum (docs/src/backends-gpu.md, the collective), each adding every element in member order, so each gives the one-shot's bits.
// wrap is the chain whose total goes on around through member 0 rather than back down; pulls is the copies behind events, timed beside the others where the rows go through host memory.
enum class GpuSumForm { one_shot, two_shots, chain, wrap, pulls };

// How a tensor group's members reach each other's rows (docs/src/backends-gpu.md, the collective): stores into each other's device memory, stores into pinned host memory every member maps, or copies the host orders behind events.
// The first two run the same kernels and forms; the first a probe passes is taken, in this order.
enum class GroupTransport { peer_writes, host_writes, pulls };

} // namespace backend
