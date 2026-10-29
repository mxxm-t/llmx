#pragma once
// VulkanBackend's memory: allocation and adoption, reads, writes and copies through staging, stream-ordered scratch, padded copies, the argument arena and the lookup grids; vulkan_backend.cpp includes it after the class.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>
#include "backends/vulkan/vulkan_buffer.hpp"

namespace backend {
namespace {

// What the device-local heap can still take: the driver's budget less what is in use, or without the budget extension the heap's size, which overstates.
inline std::optional<size_t> VulkanBackend::memory_available() const {
    const Fn& fn = dev_->fn;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2 mp{};
    mp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    if (dev_->memory_budget) mp.pNext = &budget;
    fn.vkGetPhysicalDeviceMemoryProperties2(dev_->physical, &mp);
    size_t free = 0;
    for (uint32_t h = 0; h < mp.memoryProperties.memoryHeapCount; ++h) {
        const VkMemoryHeap& heap = mp.memoryProperties.memoryHeaps[h];
        // The small device-local window the host can map (256 MiB on these cards) is not where weights go.
        if (!(heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) || heap.size < (VkDeviceSize(1) << 30)) continue;
        const VkDeviceSize total = dev_->memory_budget ? budget.heapBudget[h] : heap.size;
        const VkDeviceSize used = dev_->memory_budget ? budget.heapUsage[h] : 0;
        free = std::max(free, (size_t)(total > used ? total - used : 0));
    }
    return free;
}

inline bool VulkanBackend::memory_is_host() const { return host_memory_device(dev_->props.deviceType); }

// The upload staging and each ring slot's argument arena, host-visible memory held for the backend's life.
inline size_t VulkanBackend::host_resident() const { return kStagingBytes + kRing * kArenaBytes; }

// Split partials and attention keep their reserve; devices that use the copied integer tile also reserve its capacity.
inline size_t VulkanBackend::scratch_reserve(size_t free) const {
    return ((size_t)256 << 20) + free / 20 + (dev_->profile.prefer_integer_dot ? kMxCopyBytes : 0);
}

// An adopted F32 matrix a product reads, whose rows are a multiple of 256 floats wide, keeps a padded copy beside it once a float tile has read it (padded_f32); routed stacks bind their data as it is.
inline size_t VulkanBackend::resident_bytes(uint32_t type, size_t nin, size_t rows, size_t bytes, bool product) const {
    return bytes + (product && pads_f32(type, nin) ? rows * (nin + kF32Pad) * sizeof(float) : 0);
}

inline BufferPtr VulkanBackend::alloc(size_t bytes, Memory where) {
    drop_tags();
    auto b = std::make_shared<VulkanBuffer>(dev_, bytes, where == Memory::host_visible);
    if (bytes && !b->host_visible()) {
        // Zeroed like every other allocation, in stream order.
        // The slot holds the buffer until the fill retires, so a caller may drop it before anything is submitted.
        VkCommandBuffer cmd = open();
        pending_[ring_index_].push_back(b);
        dev_->fn.vkCmdFillBuffer(cmd, b->handle(), 0, VK_WHOLE_SIZE, 0);
        barrier(cmd);
    }
    return b;
}

// Device-local and not filled, since the caller writes every byte before an op reads it; adopted, so a float tile keeps a padded copy of it as it does of any adopted weight.
inline BufferPtr VulkanBackend::alloc_weight(size_t bytes) {
    drop_tags();
    auto b = std::make_shared<VulkanBuffer>(dev_, bytes, false);
    b->adopted = true;
    return b;
}

// Host memory the device imports and copies from in place, where the device imports host memory and `memory` and `bytes` are on its alignment; null otherwise, and where the import fails, so the caller writes through staging instead.
inline BufferPtr VulkanBackend::wrap_host(void* memory, size_t bytes) {
    const Device& d = *dev_;
    const size_t align = d.host_import_align;
    if (!d.host_pointer_props || !memory || !bytes || (uintptr_t)memory % align || bytes % align) return nullptr;
    VkMemoryHostPointerPropertiesEXT hp{};
    hp.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT;
    if (d.host_pointer_props(d.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, memory, &hp) != VK_SUCCESS || !hp.memoryTypeBits)
        return nullptr;
    try {
        return std::make_shared<VulkanBuffer>(dev_, memory, bytes, hp.memoryTypeBits);
    } catch (const std::exception&) {
        return nullptr;
    }
}

// alloc_weight's storage filled in chunks through staging, for a weight the model adopts as it is built.
inline BufferPtr VulkanBackend::adopt(const void* src, size_t bytes) {
    if (!src && bytes) throw std::runtime_error("vulkan: adopting null storage");
    auto b = std::static_pointer_cast<VulkanBuffer>(alloc_weight(bytes));
    try {
        upload(*b, 0, src, bytes, b);
    } catch (...) {
        // Earlier chunks may still target this local buffer when a later submission fails.
        sync();
        throw;
    }
    return b;
}

inline void VulkanBackend::read(const Buffer& src_b, size_t off, void* dst, size_t bytes) {
    const VulkanBuffer& src = as_vulkan(src_b);
    span(src, off, bytes);
    if (!bytes) return;
    if (src.host_visible()) {
        sync();
        if (submitter_) submitter_->raise();
        std::memcpy(dst, (const uint8_t*)src.mapped() + off, bytes);
        return;
    }
    VulkanBuffer& st = staging();
    size_t done = 0;
    while (done < bytes) {
        const size_t n = std::min(bytes - done, st.size());
        VkCommandBuffer cmd = open();
        barrier(cmd);
        VkBufferCopy region{off + done, 0, n};
        dev_->fn.vkCmdCopyBuffer(cmd, src.handle(), st.handle(), 1, &region);
        barrier(cmd);
        wait(flush());
        if (submitter_) submitter_->raise();
        std::memcpy((uint8_t*)dst + done, st.mapped(), n);
        done += n;
    }
}

inline void VulkanBackend::write(Buffer& dst_b, size_t off, const void* src, size_t bytes) {
    if (!timed_) return write_now(dst_b, off, src, bytes);
    const auto t0 = std::chrono::steady_clock::now();
    const HostTimes before = host_;
    write_now(dst_b, off, src, bytes);
    host_.write_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() -
                      (host_.ticket_ms - before.ticket_ms) - (host_.slot_ms - before.slot_ms) - (host_.staging_ms - before.staging_ms);
}

inline void VulkanBackend::write_now(Buffer& dst_b, size_t off, const void* src, size_t bytes) {
    if (!src && bytes) throw std::runtime_error("vulkan: writing from null storage");
    VulkanBuffer& dst = as_vulkan(dst_b);
    drop_padded(dst);
    span(dst, off, bytes);
    forget_written(dst.handle(), off, off + bytes);
    if (!bytes) return;
    if (dst.host_visible()) {
        // An enqueued op may still be reading or writing the range.
        sync();
        std::memcpy((uint8_t*)dst.mapped() + off, src, bytes);
        return;
    }
    upload(dst, off, src, bytes);
}

inline void VulkanBackend::copy(Buffer& dst_b, size_t dst_off, const Buffer& src_b, size_t src_off,
                                size_t bytes) {
    VulkanBuffer& dst = as_vulkan(dst_b);
    const VulkanBuffer& src = as_vulkan(src_b);
    drop_padded(dst);
    span(dst, dst_off, bytes);
    span(src, src_off, bytes);
    forget_written(dst.handle(), dst_off, dst_off + bytes);
    if (!bytes) return;
    VkCommandBuffer cmd = open();
    if (!unordered_) barrier(cmd);
    VkBufferCopy region{src_off, dst_off, bytes};
    dev_->fn.vkCmdCopyBuffer(cmd, src.handle(), dst.handle(), 1, &region);
    if (!unordered_) barrier(cmd);
}

// The binding a float tile reads a matrix through: for an adopted matrix pads_f32 picks, its padded copy, made on first use and kept with the buffer.
inline VkDescriptorBufferInfo VulkanBackend::padded_f32(CSlice data, uint32_t type, size_t rows, size_t nin) {
    if (!pads_f32(type, nin) || !rows) return bind(data);
    VulkanBuffer& src = const_cast<VulkanBuffer&>(as_vulkan(*data.buffer));
    if (!src.adopted) return bind(data);
    const size_t off = data.offset * sizeof(float);
    auto it = src.padded.find(off);
    if (it == src.padded.end() || it->second.rows != rows || it->second.nin != nin) {
        auto copy = std::make_shared<VulkanBuffer>(dev_, rows * (nin + kF32Pad) * sizeof(float), false);
        std::vector<VkBufferCopy> regions(rows);
        for (size_t r = 0; r < rows; ++r)
            regions[r] = VkBufferCopy{off + r * nin * sizeof(float), r * (nin + kF32Pad) * sizeof(float), nin * sizeof(float)};
        const uint32_t count = u32(regions.size());
        VkCommandBuffer cmd = open();
        if (it != src.padded.end()) pending_[ring_index_].push_back(it->second.copy);
        it = src.padded.insert_or_assign(off, VulkanBuffer::Padded{rows, nin, copy}).first;
        barrier(cmd);
        dev_->fn.vkCmdCopyBuffer(cmd, src.handle(), copy->handle(), count, regions.data());
        barrier(cmd);
    }
    return VkDescriptorBufferInfo{it->second.copy->handle(), 0, VK_WHOLE_SIZE};
}

// Copies of a buffer about to change, retired in stream order.
inline void VulkanBackend::drop_padded(VulkanBuffer& b) {
    if (b.padded.empty()) return;
    open();
    auto& pending = pending_[ring_index_];
    // Reserve before moving any cache entry so allocation failure leaves the cache intact.
    pending.reserve(pending.size() + b.padded.size());
    for (auto& e : b.padded) pending.push_back(std::move(e.second.copy));
    b.padded.clear();
}

// A stream-ordered scratch outgrown mid-pass: recorded commands still name the old buffer, so it retires with the ring slot.
inline void VulkanBackend::grow(std::shared_ptr<VulkanBuffer>& buffer, size_t bytes) {
    open();
    if (buffer) pending_[ring_index_].push_back(std::move(buffer));
    buffer = std::make_shared<VulkanBuffer>(dev_, bytes, false);
}

inline void VulkanBackend::check_weight_scales(WeightView weight, size_t experts) {
    if (!weight.scales.buffer) {
        if (weight.scale_count || weight.scales.offset) throw std::runtime_error("vulkan: weight scale count without storage");
        return;
    }
    if (weight.type != quant::GGML_TYPE_NVFP4) throw std::runtime_error("vulkan: weight scales require NVFP4 storage");
    if (!weight.scale_count || (weight.scale_count != 1 && weight.scale_count != experts))
        throw std::runtime_error("vulkan: weight scale count differs from its matrices");
    const size_t offset = size_mul(weight.scales.offset, sizeof(float)), bytes = size_mul(weight.scale_count, sizeof(float));
    if (offset > weight.scales.buffer->size() || bytes > weight.scales.buffer->size() - offset)
        throw std::runtime_error("vulkan: weight scales outside the allocation");
}

// NVFP4 uses its spare F32 binding for scales, with an exact range so the shader distinguishes one scale from an expert vector.
inline VkDescriptorBufferInfo VulkanBackend::weight_floats(WeightView weight, VkDescriptorBufferInfo otherwise) {
    if (weight.type != quant::GGML_TYPE_NVFP4) return otherwise;
    if (!weight.scales.buffer) {
        if (!unit_scale_) {
            const float one = 1;
            unit_scale_ = adopt(&one, sizeof(one));
        }
        VkDescriptorBufferInfo info = bind(CSlice{unit_scale_.get(), 0});
        info.range = sizeof(float);
        return info;
    }
    VkDescriptorBufferInfo info = bind(weight.scales);
    info.range = size_mul(weight.scale_count, sizeof(float));
    return info;
}

// Immutable format dictionaries alias storage views unused by these weight formats.
inline VkDescriptorBufferInfo VulkanBackend::lookup_grid(uint32_t type, VkDescriptorBufferInfo otherwise) {
    const WeightKernels* format = weight_kernels(type);
    if (!format || !format->dictionary) return otherwise;
    BufferPtr& grid = lookup_grids_[size_t(format - weight_formats)];
    if (!grid) {
        for (size_t i = 0; i < lookup_grids_.size(); ++i)
            if (lookup_grids_[i] && weight_formats[i].dictionary == format->dictionary &&
                weight_formats[i].dictionary_bytes == format->dictionary_bytes) {
                grid = lookup_grids_[i];
                break;
            }
        if (!grid) grid = adopt(format->dictionary, format->dictionary_bytes);
    }
    return bind(CSlice{grid.get(), 0});
}

// Small per-call host inputs (ids, positions, row lists) go through a host-visible arena per ring slot, bumped per call and reset when the slot retires; a call larger than the arena gets its own buffer, kept the same way.
// A dispatch reads only its own slot's arena, so each dispatch takes its own copy: the chunk flush can fall between two dispatches of one op, and a slot reopens after waiting for its own submission alone.
inline VkDescriptorBufferInfo VulkanBackend::args(const void* data, size_t bytes) {
    open();
    Arena& a = arena_[ring_index_];
    const size_t need = (bytes + 15) / 16 * 16;
    if (need > kArenaBytes) {
        auto b = std::make_shared<VulkanBuffer>(dev_, bytes, true);
        std::memcpy(b->mapped(), data, bytes);
        pending_[ring_index_].push_back(b);
        return VkDescriptorBufferInfo{b->handle(), 0, VK_WHOLE_SIZE};
    }
    if (!a.buffer) a.buffer = std::make_unique<VulkanBuffer>(dev_, kArenaBytes, true);
    if (a.used + need > kArenaBytes) {
        // The slot's arena is full before its command buffer retired: the overflow gets a second arena kept alongside.
        pending_[ring_index_].emplace_back(std::move(a.buffer));
        a.used = 0;
        a.buffer = std::make_unique<VulkanBuffer>(dev_, kArenaBytes, true);
    }
    std::memcpy((uint8_t*)a.buffer->mapped() + a.used, data, bytes);
    const VkDescriptorBufferInfo info{a.buffer->handle(), a.used, bytes};
    a.used += need;
    return info;
}

inline VulkanBuffer& VulkanBackend::staging() {
    if (!staging_) staging_ = std::make_unique<VulkanBuffer>(dev_, kStagingBytes, true);
    return *staging_;
}

// Alternate staging halves across calls so host fills overlap device copies, returning once the source is consumed.
// Every copying slot retains keep until its commands retire.
inline void VulkanBackend::upload(VulkanBuffer& dst, size_t off, const void* src, size_t bytes, const std::shared_ptr<VulkanBuffer>& keep) {
    if (!bytes) return;
    VulkanBuffer& st = staging();
    const size_t half = st.size() / 2;
    size_t done = 0;
    for (size_t& i = next_half_; done < bytes; i ^= 1) {
        const size_t n = std::min(bytes - done, half);
        waited(host_.staging_ms, staged_[i]);
        std::memcpy((uint8_t*)st.mapped() + i * half, (const uint8_t*)src + done, n);
        VkCommandBuffer cmd = open();
        if (keep) pending_[ring_index_].push_back(keep);
        barrier(cmd);
        VkBufferCopy region{i * half, off + done, n};
        dev_->fn.vkCmdCopyBuffer(cmd, st.handle(), dst.handle(), 1, &region);
        barrier(cmd);
        staged_[i] = flush();
        done += n;
    }
}

} // namespace
} // namespace backend
