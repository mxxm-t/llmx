#pragma once
// Device memory: the buffer, the rule for padded F32 copies, a table's storage views and the KV storage.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>
#include "backends/kv_storage.hpp"
#include "backends/vulkan/vulkan_device.hpp"
#include "backends/vulkan/vulkan_kernels.hpp"
#include "quant/quant.hpp"

namespace backend {
namespace {

const size_t kMxCopyBytes = size_t(256) << 20;
const size_t kFloatPartBytes = size_t(256) << 20;
const size_t kMxCopyBlock = 36; // one F32 scale and 32 signed codes
const size_t kF32Pad = 32;   // floats after each row of a padded F32 matrix (padded_f32)
// The matrix shapes that get a copy with kF32Pad floats after each row, one rule for padded_f32, which makes the copy, and resident_bytes, which counts it.
// F32 rows a multiple of 256 floats wide would otherwise all read the same memory channel (docs/VULKAN.md).
static bool pads_f32(uint32_t type, size_t nin) { return type == quant::GGML_TYPE_F32 && nin && nin % 256 == 0; }

// A table a gather reads (shaders/embed.comp), as storage views of whole rows: each view's first row `skew` bytes past its offset, rows first to end.
struct TableView {
    VkDeviceSize offset = 0, range = 0;
    uint32_t skew = 0, first = 0, end = 0;
};
// Each view begins on the device's storage-buffer offset alignment and spans whole rows within its storage-buffer range and 32-bit byte indices; a table that fits one binding is one view.
inline std::vector<TableView> table_views(VkDeviceSize base, size_t rows, size_t row_bytes, VkDeviceSize max_range, VkDeviceSize align) {
    if (rows > UINT32_MAX) throw std::runtime_error("vulkan: a table of more rows than 32-bit ids name");
    align = std::max<VkDeviceSize>(align, sizeof(float));
    const VkDeviceSize reach = std::min<VkDeviceSize>(max_range, UINT32_MAX);
    std::vector<TableView> views;
    for (size_t first = 0; first < rows || views.empty();) {
        const VkDeviceSize at = base + VkDeviceSize(first) * row_bytes, offset = at - at % align, skew = at - offset;
        const size_t n = std::min<size_t>(rows - first, size_t((reach - std::min(reach, skew)) / std::max<size_t>(row_bytes, 1)));
        if (!n && rows) throw std::runtime_error("vulkan: a table row longer than one storage binding reaches");
        views.push_back(TableView{offset, skew + VkDeviceSize(n) * row_bytes, uint32_t(skew), uint32_t(first), uint32_t(first + n)});
        first += n;
        if (!rows) break;
    }
    return views;
}

// The variant of a cache kernel for a storage's K and V types.
class VulkanKVStorage;
inline KernelId kv_variant(KernelId f32, KernelId k16, const VulkanKVStorage& s);
// The per-row attention build of 128- or 256-wide heads taken `hg` a workgroup, for a storage's K and V types.
inline KernelId vec_attention_kernel(int head_dim, uint32_t hg, const VulkanKVStorage& s);

// 64 tokens per KV block, half the CPU's: the attention workgroup reads a block per iteration, and a smaller block wastes less tail per sequence.
const size_t kVkBlockTokens = 64;
// A wide selection's slice of groups, its rows a call and a row's state words, as shaders/kv_index_select_wide.comp lays them out.
const size_t kSelectSlice = 1024, kSelectRows = 16, kSelectRowWords = 5122;
// The groups a workgroup of the scoring takes (shaders/kv_index_scores_tile.comp), one to each of its eight lane groups of 32.
const size_t kScoreTile = 8;

class VulkanBackend;

// KV blocks on the device, kept and grown by BlockKVStorage (backends/kv_storage.hpp).
// A growth's copies run on the queue after it returns, so the old buffers are kept until the command buffer that recorded the copies retires.
class VulkanKVStorage final : public BlockKVStorage {
public:
    VulkanKVStorage(VulkanBackend& owner, size_t layers, size_t heads, size_t dim, size_t max_tokens, KVType kt,
                    KVType vt, const std::vector<size_t>& auxiliary_widths, const std::vector<size_t>& auxiliary_pools);

private:
    void retire(const BufferPtr& old) override;
};

class AllocationError final : public std::bad_alloc {
    char message_[512];
public:
    AllocationError(const Device& device, VkResult result, const VkMemoryAllocateInfo& info) noexcept {
        std::snprintf(message_, sizeof message_, "vulkan: %s: vkAllocateMemory failed with %s for %llu bytes (memory type %u)",
                      device.caps.device.c_str(), vk_result_name(result),
                      static_cast<unsigned long long>(info.allocationSize), info.memoryTypeIndex);
    }
    const char* what() const noexcept override { return message_; }
};

class VulkanBuffer final : public Buffer {
public:
    // With `uncached`, host memory the device reads past its caches where it offers such a type (Device::uncached_memory), and host memory the CPU does not cache otherwise.
    VulkanBuffer(std::shared_ptr<Device> dev, size_t bytes, bool host_visible, bool uncached = false)
        : dev_(std::move(dev)), size_(bytes) {
        if (!bytes) return;   // an empty allocation has no address and no object
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        // Whole words, so a 32-bit view can reach a byte-sized tensor's last bytes (a Q8_0 or Q6_K tensor with an odd block count ends two bytes into a word).
        bi.size = (bytes + 3) & ~size_t(3);
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(dev_->fn.vkCreateBuffer(dev_->device, &bi, nullptr, &buffer_), "vkCreateBuffer");
        try {
            VkMemoryRequirements req{};
            dev_->fn.vkGetBufferMemoryRequirements(dev_->device, buffer_, &req);
            VkMemoryAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = req.size;
            // Device memory prefers not to be host visible, so it comes from the device-local heap rather than the BAR window; host-visible memory prefers to be cached.
            // The device-uncached types are taken only for `uncached` and only where the device enabled them.
            constexpr VkMemoryPropertyFlags device_uncached = VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
            const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            const auto uncached_type = [&] {
                if (dev_->uncached_memory)
                    for (uint32_t i = 0; i < dev_->memory.memoryTypeCount; ++i)
                        if ((req.memoryTypeBits >> i & 1) && (dev_->memory.memoryTypes[i].propertyFlags & (host | device_uncached)) == (host | device_uncached)) return i;
                return dev_->memory_type(req.memoryTypeBits, host, 0, VK_MEMORY_PROPERTY_HOST_CACHED_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | device_uncached);
            };
            ai.memoryTypeIndex = uncached
                ? uncached_type()
                : host_visible
                ? dev_->memory_type(req.memoryTypeBits, host, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, device_uncached)
                : dev_->memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | device_uncached);
            VkDeviceMemory memory = VK_NULL_HANDLE;
            const VkResult r = dev_->fn.vkAllocateMemory(dev_->device, &ai, nullptr, &memory);
            if (r != VK_SUCCESS) {
                if (r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY)
                    throw AllocationError(*dev_, r, ai);
                check(r, "vkAllocateMemory");
            }
            memory_ = memory;
            check(dev_->fn.vkBindBufferMemory(dev_->device, buffer_, memory_, 0), "vkBindBufferMemory");
            if (host_visible) {
                void* mapped = nullptr;
                check(dev_->fn.vkMapMemory(dev_->device, memory_, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory");
                mapped_ = mapped;
                std::memset(mapped_, 0, bytes);
            }
        } catch (...) {
            release();
            throw;
        }
    }
    // Device memory shared with other devices as a dma-buf (`fd` < 0, exported by this device), or another device's imported from `fd`, whose ownership the import takes (docs/TENSOR-SPLIT.md, section 4.3).
    VulkanBuffer(std::shared_ptr<Device> dev, size_t bytes, int fd)
        : dev_(std::move(dev)), size_(bytes) {
        VkExternalMemoryBufferCreateInfo ei{};
        ei.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        ei.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.pNext = &ei;
        bi.size = (bytes + 3) & ~size_t(3);
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        try {
            check(dev_->fn.vkCreateBuffer(dev_->device, &bi, nullptr, &buffer_), "vkCreateBuffer");
            VkMemoryRequirements req{};
            dev_->fn.vkGetBufferMemoryRequirements(dev_->device, buffer_, &req);
            VkExportMemoryAllocateInfo ex{};
            ex.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
            ex.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
            VkImportMemoryFdInfoKHR im{};
            im.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
            im.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
            im.fd = fd;
            VkMemoryAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.allocationSize = req.size;
            uint32_t types = req.memoryTypeBits;
            if (fd >= 0) {
                VkMemoryFdPropertiesKHR fp{};
                fp.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR;
                check(dev_->memory_fd_props(dev_->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fp), "vkGetMemoryFdPropertiesKHR");
                types &= fp.memoryTypeBits;
                ai.pNext = &im;
                ai.memoryTypeIndex = dev_->memory_type(types, 0, 0, 0);
            } else {
                ai.pNext = &ex;
                ai.memoryTypeIndex = dev_->memory_type(types, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            }
            check(dev_->fn.vkAllocateMemory(dev_->device, &ai, nullptr, &memory_), "vkAllocateMemory (shared)");
            fd = -1;
            check(dev_->fn.vkBindBufferMemory(dev_->device, buffer_, memory_, 0), "vkBindBufferMemory (shared)");
        } catch (...) {
#if !defined(_WIN32)
            if (fd >= 0) close(fd);
#endif
            release();
            throw;
        }
    }
    // A dma-buf of this device's exported memory, which the caller owns.
    int export_fd() const {
        VkMemoryGetFdInfoKHR gi{};
        gi.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        gi.memory = memory_;
        gi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
        int fd = -1;
        check(dev_->get_memory_fd(dev_->device, &gi, &fd), "vkGetMemoryFdKHR");
        return fd;
    }
    // A copy source or destination over host memory the device reads and writes in place: `memory` imported as device memory of a type among `types`, both it and `bytes` on the import alignment.
    VulkanBuffer(std::shared_ptr<Device> dev, void* memory, size_t bytes, uint32_t types)
        : dev_(std::move(dev)), size_(bytes) {
        VkExternalMemoryBufferCreateInfo ei{};
        ei.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        ei.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.pNext = &ei;
        bi.size = bytes;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(dev_->fn.vkCreateBuffer(dev_->device, &bi, nullptr, &buffer_), "vkCreateBuffer");
        try {
            VkMemoryRequirements req{};
            dev_->fn.vkGetBufferMemoryRequirements(dev_->device, buffer_, &req);
            if (req.size > bytes) throw std::runtime_error("vulkan: an imported buffer needs more than the memory it wraps");
            VkImportMemoryHostPointerInfoEXT hi{};
            hi.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT;
            hi.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
            hi.pHostPointer = memory;
            VkMemoryAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            ai.pNext = &hi;
            ai.allocationSize = bytes;
            ai.memoryTypeIndex = dev_->memory_type(req.memoryTypeBits & types, 0, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
            check(dev_->fn.vkAllocateMemory(dev_->device, &ai, nullptr, &memory_), "vkAllocateMemory");
            check(dev_->fn.vkBindBufferMemory(dev_->device, buffer_, memory_, 0), "vkBindBufferMemory");
        } catch (...) {
            release();
            throw;
        }
    }
    ~VulkanBuffer() override { release(); }
    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;

    size_t size() const override { return size_; }
    // Weights arrive through adopt; only those get padded copies, which a write or copy into the buffer drops (VulkanBackend::padded_f32).
    bool adopted = false;
    struct Padded {
        size_t rows, nin;
        std::shared_ptr<VulkanBuffer> copy;
    };
    std::map<size_t, Padded> padded;   // by byte offset
    const void* host_ptr() const override { return mapped_; }
    void* mapped() const { return mapped_; }
    VkBuffer handle() const { return buffer_; }
    bool host_visible() const { return mapped_ != nullptr; }

private:
    void release() noexcept {
        if (mapped_) dev_->fn.vkUnmapMemory(dev_->device, memory_);
        if (buffer_) dev_->fn.vkDestroyBuffer(dev_->device, buffer_, nullptr);
        if (memory_) dev_->fn.vkFreeMemory(dev_->device, memory_, nullptr);
    }

    std::shared_ptr<Device> dev_;
    size_t size_ = 0;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    void* mapped_ = nullptr;
};

const VulkanBuffer& as_vulkan(const Buffer& b) {
    const auto* v = dynamic_cast<const VulkanBuffer*>(&b);
    if (!v) throw std::runtime_error("vulkan: buffer of another backend");
    return *v;
}
VulkanBuffer& as_vulkan(Buffer& b) {
    auto* v = dynamic_cast<VulkanBuffer*>(&b);
    if (!v) throw std::runtime_error("vulkan: buffer of another backend");
    return *v;
}

} // namespace
} // namespace backend
