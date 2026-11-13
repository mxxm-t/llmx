#pragma once
// What the GPU backends' shared core (backend_core.hpp) shows a vendor's public header, so its tests can name it.

namespace backend {

// The forms of a tensor group's fast sum (docs/src/backends-gpu.md, the collective), each adding every element in member order, so each gives the one-shot's bits.
// wrap is the chain whose total goes on around through member 0 rather than back down.
enum class GpuSumForm { one_shot, two_shots, chain, wrap };

} // namespace backend
