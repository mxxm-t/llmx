// Emits the CPU backend's Q6_K fused dot on its own, so its scalar FMA instructions can be counted: clang++ or g++ -std=c++17 -O3 -mavx2 -mfma -mf16c -I<tree>/src -c, then objdump -d over dot_row_q6_K.
#include "backends/backend.hpp"
#include "backends/kv_storage.hpp"
#include "backends/cpu/prefill_placement.hpp"
#include "backends/cpu/q8_dots.hpp"
#include "core/fp16.hpp"
#include "core/host_memory.hpp"
#include "format/gguf.hpp"
#include "quant/quant.hpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#define private public
#include "backends/cpu/cpu_backend.hpp"
#undef private
float q6(backend::CpuBackend& b, const uint8_t* r, const float* x, size_t n) { return b.dot_row_q6_K(r, x, n); }
