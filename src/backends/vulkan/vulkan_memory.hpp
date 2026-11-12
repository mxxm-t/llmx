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

// What the device-local heap can still take: the driver's budget, at most the ceiling the heap stated when it was last full (Device::ceiling), less what is in use; without the budget extension the heap's size, which overstates.
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
        const VkDeviceSize used = dev_->memory_budget ? budget.heapUsage[h] : 0;
        // A heap whose usage reaches its budget less what the backend keeps (none on Vulkan) is full, so that budget is its ceiling from now on.
        if (dev_->memory_budget && used + memory_kept(0) >= budget.heapBudget[h]) dev_->lower_ceiling(h, budget.heapBudget[h]);
        const VkDeviceSize ceiling = dev_->ceiling[h].load();
        const VkDeviceSize total = dev_->memory_budget ? (ceiling ? std::min(ceiling, budget.heapBudget[h]) : budget.heapBudget[h]) : heap.size;
        free = std::max(free, (size_t)(total > used ? total - used : 0));
    }
    return free;
}

inline bool VulkanBackend::memory_is_host() const { return host_memory_device(dev_->props.deviceType); }

// The upload staging and each ring slot's argument arena, host-visible memory held for the backend's life.
inline size_t VulkanBackend::host_resident() const { return kStagingBytes + kRing * kArenaBytes; }

// Each scratch buffer's bytes at the widest call passes of widths `w` make, by the formulas its grow site follows (docs/src/backends-vulkan.md, Scratch).
inline VulkanBackend::ScratchSizes VulkanBackend::scratch_sizes(const PassWidths& w) const {
    ScratchSizes z{};
    const size_t R = w.rows, H = w.heads, D = w.head_dim, stride = D + 2;
    if (!R) return z;
    const DeviceProfile& prof = dev_->profile;
    size_t scratch = 0;
    if (w.dense_attention && H) {
        // The per-row kernel's splits up to the profile's limit, and the tiled kernel's parts of attention_tile_part tokens in pieces whose parts fit attention_scratch_cap_.
        const size_t chunk = std::max<size_t>(1, prof.attention_split_chunk), most = D == 256 ? prof.attention_split_max_wide : prof.attention_split_max;
        const size_t nsplit = std::min(most, std::max<size_t>(1, (w.context + chunk - 1) / chunk));
        if (nsplit > 1) scratch = size_mul(size_mul(size_mul(R, H), nsplit), stride * sizeof(float));
        const size_t part = std::max<size_t>(1, prof.attention_tile_part), wide = (w.context + part - 1) / part;
        if (wide > 1) {
            const size_t per_row = size_mul(size_mul(H, wide), stride * sizeof(float));
            const size_t cap_rows = std::max(kAttentionTileRowsOne, attention_scratch_cap_ / per_row / kAttentionTileRowsOne * kAttentionTileRowsOne);
            scratch = std::max(scratch, size_mul(std::min(R, cap_rows), per_row));
        }
    }
    if (w.index_compression) scratch = std::max(scratch, size_mul(size_mul(std::min<size_t>(R, 16), w.context / w.index_compression), sizeof(float)));
    if (w.sparse_named && H) {
        const size_t named = w.sparse_named, row_words = size_add(size_add(size_mul(named, H), H), size_add(named, 2));
        const size_t chunk = std::min(R, std::max<size_t>(1, ((size_t)8 << 20) / row_words));
        scratch = std::max(scratch, size_mul(size_add(size_add(size_mul(size_mul(chunk, named), H), size_mul(chunk, H)), size_add(size_mul(chunk, named), size_mul(chunk, 2))), sizeof(float)));
    }
    z[0] = scratch;
    if (w.index_compression) {
        const size_t groups = w.context / w.index_compression;
        if (groups >= prof.select_wide_from) z[1] = size_mul(size_add(kSelectRows * kSelectRowWords, size_mul(size_mul(kSelectRows, (groups + kSelectSlice - 1) / kSelectSlice), 2)), sizeof(uint32_t));
    }
    // A split tile call's parts, at the 8 bytes an MXFP4 part takes; the float tile's are bounded by kFloatPartBytes within that.
    z[2] = size_mul(split_partials_most(prof, dev_->caps, R, w.product_in, w.product_out, kTileRowsSmall), sizeof(double));
    const size_t entries = size_mul(R, w.routed);
    if (entries) {
        z[3] = size_mul(size_mul(entries, w.expert_out), sizeof(float));
        z[4] = size_mul(size_add(size_mul(4, size_add((entries + 1) / 2, entries)), entries), sizeof(uint32_t));
        z[6] = sizeof(uint32_t);
    }
    if (prof.prefer_integer_dot) z[5] = kMxCopyBytes;
    if (w.logit_rows && w.vocab > kVulkanArgmaxSlice) z[7] = size_mul(size_mul(w.logit_rows, (w.vocab + kVulkanArgmaxSlice - 1) / kVulkanArgmaxSlice), 3 * sizeof(uint32_t));
    // A twin slot at the 8-bit form's bytes, which follows the 16-bit one, for a product's rows padded by at most seven columns or a routed call's entries.
    const auto twin = [](size_t n) { return x8_base_bytes(n) + (n + (n / 32) * 8) / 4 * 5; };
    const size_t n = std::max(size_mul(size_add(R, 7), w.product_in), size_mul(size_add(entries, 7), w.expert_in));
    for (size_t i = 0; i < kTwins; ++i) z[8 + i] = twin(n);
    return z;
}

inline std::shared_ptr<VulkanBuffer>* VulkanBackend::scratch_buffer(size_t i) {
    std::shared_ptr<VulkanBuffer>* const held[kScratchBuffers] = {&scratch_, &select_state_, &parts_, &moe_out_, &moe_tab_, &mx_copy_, &moe_count_, &argmax_parts_,
                                                                  &twins_[0].buffer, &twins_[1].buffer, &twins_[2].buffer, &twins_[3].buffer};
    return held[i];
}

inline size_t VulkanBackend::scratch_bytes(const PassWidths& w) const {
    size_t n = 0;
    for (size_t b : scratch_sizes(w)) n = size_add(n, alloc_bytes(b));
    return n;
}

// Each scratch buffer made at its priced size and zeroed in stream order before the first pass, so no pass grows one beside the one it replaces.
inline void VulkanBackend::reserve_scratch(const PassWidths& w) {
    const ScratchSizes z = scratch_sizes(w);
    static_assert(kTwins == 4, "scratch_buffer names each twin slot");
    for (size_t i = 0; i < z.size(); ++i) {
        std::shared_ptr<VulkanBuffer>& b = *scratch_buffer(i);
        if (!z[i] || (b && b->size() >= z[i])) continue;
        grow(b, z[i]);
        VkCommandBuffer cmd = open();
        dev_->fn.vkCmdFillBuffer(cmd, b->handle(), 0, VK_WHOLE_SIZE, 0);
        barrier(cmd);
    }
}

inline size_t VulkanBackend::alloc_bytes(size_t bytes) const {
    if (!bytes) return 0;
    const size_t unit = bytes > kLargeFrom ? large_unit_ : small_unit_;
    return size_mul((bytes + unit - 1) / unit, unit);
}

// The allocation units from the heap budget's readings around fresh buffers; without the budget extension both stay 2 MiB.
inline void VulkanBackend::probe_memory() {
    if (!dev_->memory_budget) return;
    const auto taken = [&](size_t bytes, size_t n) -> size_t {
        const std::optional<size_t> before = memory_available();
        std::vector<std::shared_ptr<VulkanBuffer>> held;
        try {
            for (size_t i = 0; i < n; ++i) held.push_back(std::make_shared<VulkanBuffer>(dev_, bytes, false));
        } catch (const std::exception&) {
            return size_t(0);
        }
        const std::optional<size_t> after = memory_available();
        return before && after && *before > *after ? (*before - *after) / n : 0;
    };
    small_unit_ = allocation_unit(4, taken(4, 64));
    large_unit_ = allocation_unit(kLargeFrom + 4, taken(kLargeFrom + 4, 4));
}

// An adopted F32 matrix a product reads, whose rows are a multiple of 256 floats wide, keeps a padded copy beside it once a float tile has read it (padded_f32); routed stacks bind their data as it is.
// Each buffer counts as the device takes it (alloc_bytes).
inline size_t VulkanBackend::resident_bytes(uint32_t type, size_t nin, size_t rows, size_t bytes, bool product) const {
    return alloc_bytes(bytes) + (product && pads_f32(type, nin) ? alloc_bytes(rows * (nin + kF32Pad) * sizeof(float)) : 0);
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
    span(dst, off, bytes);
    forget_written(dst.handle(), off, off + bytes);
    if (!bytes) return;
    if (dst.host_visible()) {
        // An enqueued op may still be reading or writing the range.
        sync();
        std::memcpy((uint8_t*)dst.mapped() + off, src, bytes);
    } else {
        upload(dst, off, src, bytes);
    }
    refresh_padded(dst, off, bytes);
}

inline void VulkanBackend::copy(Buffer& dst_b, size_t dst_off, const Buffer& src_b, size_t src_off,
                                size_t bytes) {
    VulkanBuffer& dst = as_vulkan(dst_b);
    const VulkanBuffer& src = as_vulkan(src_b);
    span(dst, dst_off, bytes);
    span(src, src_off, bytes);
    forget_written(dst.handle(), dst_off, dst_off + bytes);
    if (!bytes) return;
    VkCommandBuffer cmd = open();
    if (!unordered_) barrier(cmd);
    else {
        const std::pair<VkBuffer, bool> uses[2] = {{src.handle(), false}, {dst.handle(), true}};
        order_step(cmd, uses, 2);
    }
    VkBufferCopy region{src_off, dst_off, bytes};
    if (dev_->timestamps) stamp_before(cmd);
    dev_->fn.vkCmdCopyBuffer(cmd, src.handle(), dst.handle(), 1, &region);
    if (dev_->timestamps) stamp_after(cmd, kTimedCopy);
    if (!unordered_) barrier(cmd);
    refresh_padded(dst, dst_off, bytes);
}

// The binding a float tile reads a matrix through: for an adopted matrix pads_f32 picks, its padded copy, made by prepare_matrix or on first use and kept with the buffer; a product of fewer rows reads the first rows of a copy made for more.
inline VkDescriptorBufferInfo VulkanBackend::padded_f32(CSlice data, uint32_t type, size_t rows, size_t nin) {
    if (!pads_f32(type, nin) || !rows) return bind(data);
    VulkanBuffer& src = const_cast<VulkanBuffer&>(as_vulkan(*data.buffer));
    if (!src.adopted) return bind(data);
    const size_t off = data.offset * sizeof(float);
    auto it = src.padded.find(off);
    if (it == src.padded.end() || it->second.rows < rows || it->second.nin != nin) {
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

// A product matrix's padded copy made as the model is built, before its bytes are uploaded, which the writes then keep current (refresh_padded).
inline void VulkanBackend::prepare_matrix(WeightView weight, size_t nin, size_t rows) {
    if (weight.data.buffer) padded_f32(weight.data, weight.type, rows, nin);
}

// The rows of each padded copy that bytes [off, off + bytes) of its matrix reach, copied again in stream order after the write that changed them.
inline void VulkanBackend::refresh_padded(VulkanBuffer& b, size_t off, size_t bytes) {
    if (b.padded.empty() || !bytes) return;
    std::vector<VkBufferCopy> regions;
    for (const auto& [at, p] : b.padded) {
        const size_t row = p.nin * sizeof(float), end = at + p.rows * row;
        if (off >= end || off + bytes <= at) continue;
        regions.clear();
        for (size_t r = (std::max(off, at) - at) / row; r < p.rows && at + r * row < off + bytes; ++r)
            regions.push_back(VkBufferCopy{at + r * row, r * (p.nin + kF32Pad) * sizeof(float), row});
        VkCommandBuffer cmd = open();
        barrier(cmd);
        dev_->fn.vkCmdCopyBuffer(cmd, b.handle(), p.copy->handle(), u32(regions.size()), regions.data());
        barrier(cmd);
    }
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
        if (dev_->timestamps) stamp_before(cmd);
        dev_->fn.vkCmdCopyBuffer(cmd, st.handle(), dst.handle(), 1, &region);
        if (dev_->timestamps) stamp_after(cmd, kTimedUpload);
        barrier(cmd);
        staged_[i] = flush();
        done += n;
    }
}

} // namespace
} // namespace backend
