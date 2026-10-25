#pragma once
// VulkanBackend's activation twins: the packed copies of matmul inputs kept in a few slots, and what a write drops; vulkan_backend.cpp includes it after the class.
#include <cstdint>
#include <cstddef>
#include "backends/vulkan/vulkan_kernels.hpp"

namespace backend {
namespace {

// The twins and the grouping name a buffer by its handle, which a new buffer can take over once the buffer it named is freed, so a new buffer drops them all.
inline void VulkanBackend::drop_tags() {
    for (Twin& t : twins_) t.n = 0;
    group_tag_ = GroupTag{};
}

// Forget the twins whose input, and the grouping whose ids, overlap bytes [begin, end) of a buffer that is about to be written.
inline void VulkanBackend::forget_written(VkBuffer buffer, VkDeviceSize begin, VkDeviceSize end) {
    for (Twin& t : twins_)
        if (t.n && t.x.buffer == buffer && t.x.offset < end && begin < t.x.offset + t.n * sizeof(float)) t.n = 0;
    if (group_tag_.entries && group_tag_.ids.buffer == buffer && group_tag_.ids.offset < end &&
        begin < group_tag_.ids.offset + group_tag_.entries * sizeof(float))
        group_tag_ = GroupTag{};
}

// Bind original floats for a float row build or policy, otherwise the packed activation twin (shaders/xquant.glsl).
// Missing twins are quantized here into stream-ordered scratch.
inline VkDescriptorBufferInfo VulkanBackend::row_twin(CSlice X, const RowPlan& plan, size_t n, Dtype dtype) {
    if (weight_kernels(plan.type)->layout == RowLayout::values || plan.kernel == weight_kernels(plan.type)->float_row || dtype == Dtype::f32) return bind(X);
    return twin(X, n, reads_x8(plan.kernel), 0);
}

// Bind a kept twin or pack it for the reader: tile_nin is the tile's input width, zero for a row kernel; `kept_major` receives the bound twin's major.
// Column-major int8 copies are reusable; block-major copies require the reader's row width.
inline VkDescriptorBufferInfo VulkanBackend::twin(CSlice X, size_t n, bool x8, size_t tile_nin, uint32_t* kept_major) {
    // The first matmul reading the 8-bit twin has it made here; producers after it write both.
    if (x8) want_x8_ = true;
    const VkDescriptorBufferInfo xf = bind(X);
    const uint32_t major = tile_nin ? x8_major(tile_nin, n / tile_nin) : 0;
    size_t slot = kTwins;
    for (size_t i = 0; i < kTwins; ++i) {
        const Twin& t = twins_[i];
        if (t.n == n && t.x.buffer == xf.buffer && t.x.offset == xf.offset && (!x8 || (t.has8 && (!t.major || t.major == major)))) slot = i;
    }
    if (slot == kTwins) {
        slot = take_twin(n, xf);
        const uint32_t pc[2] = {u32(n), major};
        if (tile_nin) dispatch(K_QUANTIZE_XW, {xf, twin_binding(slot)}, pc, sizeof(pc), groups(n / 4, 256), 1, twin_variant());
        else dispatch(K_QUANTIZE_X, {xf, twin_binding(slot)}, pc, sizeof(uint32_t), groups(n, 256), 1, twin_variant());
        keep_twin(slot, xf, n, major);
    }
    Twin& t = twins_[slot];
    t.used = ++twin_clock_;
    if (kept_major) *kept_major = t.major;
    VkDescriptorBufferInfo xq = twin_binding(slot);
    if (x8) xq.offset = x8_base_bytes(n);
    return xq;
}

// A slot for the twin of an n-value input, emptied and with scratch to hold it, the 8-bit twin after the 16-bit one once a matmul has read it: the slot already holding `x`'s twin, else an empty one, else the least recently used.
inline size_t VulkanBackend::take_twin(size_t n, VkDescriptorBufferInfo x) {
    size_t slot = 0;
    for (size_t i = 0; i < kTwins; ++i) {
        const Twin &t = twins_[i], &best = twins_[slot];
        const bool same = t.n && x.buffer && t.x.buffer == x.buffer && t.x.offset == x.offset;
        if (same) { slot = i; break; }
        if ((!t.n && best.n) || (!t.n == !best.n && t.used < best.used)) slot = i;
    }
    Twin& t = twins_[slot];
    t.n = 0;
    // A block-major 8-bit twin pads its columns by at most seven, which from kX8MajorRows columns on is under a quarter.
    const size_t bytes = want_x8_ ? x8_base_bytes(n) + (n + (n / 32) * 8) / 4 * 5 : n * 2 + (n / 32) * 8;
    if (!t.buffer || t.buffer->size() < bytes) grow(t.buffer, bytes);
    return slot;
}

inline VkDescriptorBufferInfo VulkanBackend::twin_binding(size_t slot) const { return VkDescriptorBufferInfo{twins_[slot].buffer->handle(), 0, VK_WHOLE_SIZE}; }

// Record what a slot's scratch now holds, once the dispatch writing it is recorded.
inline void VulkanBackend::keep_twin(size_t slot, VkDescriptorBufferInfo x, size_t n, uint32_t major) {
    Twin& t = twins_[slot];
    t.x = x;
    t.n = n;
    t.has8 = want_x8_;
    t.major = major;
    t.used = ++twin_clock_;
}

// The twin the integer-dot tile reads (twin); `xpad` receives the padded columns of a block-major 8-bit twin, zero for one that lies column after column.
inline VkDescriptorBufferInfo VulkanBackend::tile_twin(CSlice X, size_t n, size_t nin, bool x8, uint32_t& xpad) {
    uint32_t major = 0;
    const VkDescriptorBufferInfo xt = twin(X, n, x8, nin, &major);
    xpad = x8 && major ? x8_columns(u32(n / 32 / major)) : 0;
    return xt;
}

// The blocks of a column where a producer whose output the tile reads next writes the 8-bit twin block-major, zero where it writes it column after column (shaders/xquant.glsl, xquant8_word).
// It changes where the twin's blocks lie, never a value.
inline uint32_t VulkanBackend::x8_major(size_t nin, size_t rows) const { return want_x8_ && rows >= kX8MajorRows && nin % 32 == 0 ? u32(nin / 32) : 0; }

inline uint32_t VulkanBackend::x8_columns(uint32_t cols) { return (((cols + 3) >> 2) | 1) << 2; }

// Where the 8-bit twin starts after the 16-bit one, in bytes, rounded up to 256 so it is a valid binding offset (shaders/xquant.glsl, xquant8_base).
inline size_t VulkanBackend::x8_base_bytes(size_t n) { return ((n / 2 + n / 16 + 63) & ~size_t(63)) * 4; }

// The producers' build: the one that also writes the 8-bit twin once a matmul has read it (shaders/xquant.glsl, TWIN8).
inline int VulkanBackend::twin_variant() const { return want_x8_ ? 1 : 0; }

} // namespace
} // namespace backend
