#pragma once
// VulkanBackend's dispatch: storage bindings, kernel pipelines made on first use, and one recorded dispatch; vulkan_backend.cpp includes it after the class.
#include <array>
#include <initializer_list>
#include <stdexcept>
#include <vector>
#include "backends/vulkan/vulkan_buffer.hpp"
#include "backends/vulkan/vulkan_kernels.hpp"

namespace backend {
namespace {

// A slice as a storage buffer binding: the buffer at a byte offset of four times the float offset, through to the end of the allocation.
inline VkDescriptorBufferInfo VulkanBackend::bind(CSlice s) {
    if (!s.buffer) throw std::runtime_error("vulkan: operand without storage");
    const VulkanBuffer& b = as_vulkan(*s.buffer);
    const VkDeviceSize off = (VkDeviceSize)s.offset * sizeof(float);
    if (off > b.size()) throw std::runtime_error("vulkan: operand outside the allocation");
    if (!b.size()) return VkDescriptorBufferInfo{VK_NULL_HANDLE, 0, VK_WHOLE_SIZE};
    return VkDescriptorBufferInfo{b.handle(), off, VK_WHOLE_SIZE};
}

inline VkDescriptorBufferInfo VulkanBackend::bind(Slice s) { return bind(CSlice(s)); }

// The buffers the backend keeps across calls: twins, partial sums, groupings and split states.
inline bool VulkanBackend::is_scratch(VkBuffer h) const {
    auto is = [h](const std::shared_ptr<VulkanBuffer>& b) { return b && b->handle() == h; };
    for (const Twin& t : twins_)
        if (is(t.buffer)) return true;
    return is(parts_) || is(scratch_) || is(moe_out_) || is(moe_tab_) || is(moe_count_) || is(argmax_parts_) || is(select_state_) || is(mx_copy_);
}

// While unordered, a barrier before a step that follows another of its op call, or that uses scratch written, or writes scratch used, since the last barrier; a caller's buffers across calls are its own promise (Backend::unordered).
inline void VulkanBackend::order_step(VkCommandBuffer cmd, const std::pair<VkBuffer, bool>* uses, size_t n) {
    auto has = [](const std::vector<VkBuffer>& v, VkBuffer h) { return std::find(v.begin(), v.end(), h) != v.end(); };
    bool wait = op_depth_ > 0 && op_steps_++ > 0;
    for (size_t i = 0; !wait && i < n; ++i)
        wait = has(scratch_written_, uses[i].first) || (uses[i].second && has(scratch_read_, uses[i].first));
    if (wait) barrier(cmd);
    for (size_t i = 0; i < n; ++i)
        if (is_scratch(uses[i].first)) (uses[i].second ? scratch_written_ : scratch_read_).push_back(uses[i].first);
}

inline void VulkanBackend::destroy_kernel(Kernel& k) noexcept {
    Device& d = *dev_;
    if (k.pipeline) d.fn.vkDestroyPipeline(d.device, k.pipeline, nullptr);
    if (k.layout) d.fn.vkDestroyPipelineLayout(d.device, k.layout, nullptr);
    if (k.set_layout) d.fn.vkDestroyDescriptorSetLayout(d.device, k.set_layout, nullptr);
    if (k.module) d.fn.vkDestroyShaderModule(d.device, k.module, nullptr);
    k = Kernel{};
}

inline Kernel& VulkanBackend::kernel(KernelId id, int variant) {
    Kernel& cached = kernels_[id][variant];
    if (cached.pipeline) return cached;
    Kernel k;
    try {
        Device& d = *dev_;
        const KernelSource& src = kKernels[id];
        const bool preserve = d.preserve_float32 && src.preserved_words;
        std::vector<uint32_t> words;
        // Keep any patched SPIR-V alive until Vulkan consumes the create info.
        const auto mi = kernel_module(id, preserve, d.float_controls, words);
        VkShaderModule module = VK_NULL_HANDLE;
        check(d.fn.vkCreateShaderModule(d.device, &mi, nullptr, &module), "vkCreateShaderModule");
        k.module = module;
        std::vector<VkDescriptorSetLayoutBinding> bindings(src.bindings);
        k.buffers = 0;
        for (uint32_t i = 0; i < src.bindings; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = src.counts ? src.counts[i] : 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            k.buffers += bindings[i].descriptorCount;
        }
        VkDescriptorSetLayoutCreateInfo li{};
        li.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        li.bindingCount = src.bindings;
        li.pBindings = bindings.data();
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        check(d.fn.vkCreateDescriptorSetLayout(d.device, &li, nullptr, &set_layout),
              "vkCreateDescriptorSetLayout");
        k.set_layout = set_layout;
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, kPushBytes};
        VkPipelineLayoutCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pi.setLayoutCount = 1;
        pi.pSetLayouts = &k.set_layout;
        pi.pushConstantRangeCount = 1;
        pi.pPushConstantRanges = &range;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        check(d.fn.vkCreatePipelineLayout(d.device, &pi, nullptr, &layout), "vkCreatePipelineLayout");
        k.layout = layout;
        VkComputePipelineCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        ci.stage.module = k.module;
        ci.stage.pName = "main";
        // The tile kernels take their row count as specialization constant 0 and the row kernels their column count.
        const bool tile = is_tile_kernel(id);
        const bool tall_tile = id == K_MATMUL_TILE_Q5_TALL || id == K_MATMUL_TILE_Q50_TALL || id == K_MATMUL_TILE_Q5I8_TALL || id == K_MATMUL_TILE_Q50I8_TALL || id == K_MATMUL_TILE_IQ4_TALL || id == K_MATMUL_TILE_IQ4_XS_TALL || id == K_MATMUL_TILE_WF16_TALL || id == K_MATMUL_TILE_WF16_BF16_TALL ||
                               id == K_MATMUL_TILE_IQ1_S_DOT_TALL || id == K_MATMUL_TILE_IQ1_M_DOT_TALL || id == K_MATMUL_TILE_IQ1_S_TALL || id == K_MATMUL_TILE_IQ1_M_TALL || id == K_MATMUL_TILE_IQ1_S_BF16_TALL || id == K_MATMUL_TILE_IQ1_M_BF16_TALL ||
                               id == K_MATMUL_TILE_WBF16_TALL || id == K_MATMUL_TILE_WBF16_BF16_TALL || id == K_MATMUL_TILE_Q8MX_TALL || id == K_MATMUL_TILE_MXFP4_TALL || id == K_MATMUL_TILE_MXFP4_BF16_TALL || (id == K_MATMUL_TILE_IQ3_XXS_DOT_TALL || id == K_MATMUL_TILE_IQ2_XXS_DOT_TALL || id == K_MATMUL_TILE_IQ2_XS_DOT_TALL) || id == K_MATMUL_TILE_TQ1_0_DOT_TALL || id == K_MATMUL_TILE_TQ2_0_DOT_TALL || id == K_MATMUL_TILE_IQ2_S_DOT_TALL || id == K_MATMUL_TILE_IQ3_S_DOT_TALL || (id == K_MATMUL_TILE_IQ3_XXS_TALL || id == K_MATMUL_TILE_IQ2_XXS_TALL || id == K_MATMUL_TILE_IQ2_XS_TALL) || id == K_MATMUL_TILE_TQ1_0_TALL || id == K_MATMUL_TILE_TQ2_0_TALL || id == K_MATMUL_TILE_IQ2_S_TALL || (id == K_MATMUL_TILE_IQ3_XXS_BF16_TALL || id == K_MATMUL_TILE_IQ2_XXS_BF16_TALL || id == K_MATMUL_TILE_IQ2_XS_BF16_TALL) || id == K_MATMUL_TILE_TQ1_0_BF16_TALL || id == K_MATMUL_TILE_TQ2_0_BF16_TALL || id == K_MATMUL_TILE_IQ2_S_BF16_TALL || id == K_MATMUL_TILE_IQ3_S_TALL || id == K_MATMUL_TILE_IQ3_S_BF16_TALL || id == K_MATMUL_TILE_TALL || id == K_MATMUL_TILE_BF16_TALL || id == K_MATMUL_TILE_Q_TALL || id == K_MATMUL_TILE_Q8_TALL ||
                               id == K_MATMUL_TILE_QI8_TALL || id == K_MATMUL_TILE_Q8I8_TALL || id == K_MATMUL_TILE_NVFP4_TALL || id == K_MATMUL_TILE_NVFP4_F16_TALL || id == K_MATMUL_TILE_NVFP4_BF16_TALL;
        const uint32_t spec_value = tile ? (tall_tile ? kTileRowsTall : variant == 1 ? kTileRowsSmall : kTileRowsShort)
                                         : build_cols(id, variant);
        // Specialization constants 8-13 and 16 describe grouped builds, rows, load steps, forms, column groups and halves.
        // Their mapping is in docs/src/backends-vulkan.md, Decode builds; modules ignore entries they do not declare.
        const VecBuild& vb = kVecBuilds[variant];
        const bool vec = is_vec_kernel(id) || moe_step_q8(id);
        const uint32_t forms = vec ? vec_forms(variant, d.profile) : 0;
        // Constant 7 is a twin producer's build that also writes the 8-bit twin, its variant 1 (twin_variant).
        // The entries go in constant order.
        const bool twin8 = variant == 1 && !tile && !is_row_kernel(id);
        // Constant 14 keeps a fused kernel's row products; a diagnostic backend also builds each fused kernel without them (role_free_).
        // Constant 15 fixes NVFP4's format independently of the fused role control, and 16 is a Q8_0 decode build's halves.
        uint32_t spec_data[12] = {vec ? vb.cols : spec_value, twin8 ? 1u : 0u, variant == 2 ? 1u : 0u, vec ? vb.rows : build_rows(id, variant), vb.steps,
                                  (forms & kQ8Tree) ? 1u : 0u, (forms & kQ8Half) ? 1u : 0u, vb.span, 1u, (is_nvfp4_tile(id) || id == K_MATMUL_ROW_NVFP4) ? uint32_t(quant::GGML_TYPE_NVFP4) : UINT32_MAX,
                                  vec ? vb.halves : 1u, row_tree_cluster(id, variant, d.profile, d.caps.subgroup_size)};
        const uint32_t spec_ids[12] = {0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
        VkSpecializationMapEntry entries[12];
        for (uint32_t i = 0; i < 12; ++i) entries[i] = {spec_ids[i], i * uint32_t(sizeof(uint32_t)), sizeof(uint32_t)};
        VkSpecializationInfo spec{};
        spec.mapEntryCount = 12;
        spec.pMapEntries = entries;
        spec.dataSize = sizeof(spec_data);
        spec.pData = spec_data;
        ci.stage.pSpecializationInfo = &spec;
        ci.layout = k.layout;
        if (d.exec_stats) ci.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
        if (d.exec_ir) ci.flags |= VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
        check(d.fn.vkCreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &ci, nullptr, &k.pipeline),
              "vkCreateComputePipelines");
        if (d.exec_ir && moe_step_kernel(id) && !role_free_[id][variant]) {
            spec_data[8] = 0u;
            check(d.fn.vkCreateComputePipelines(d.device, VK_NULL_HANDLE, 1, &ci, nullptr, &role_free_[id][variant]), "vkCreateComputePipelines");
        }
        // The fused route step's rows are held to the F32 family alone, a build only a diagnostic backend makes.
        if (d.exec_ir && moe_step_route_kernel(id)) kernel(K_MATMUL_ROW_F32_ONLY, variant);
        k.bindings = src.bindings;
        cached = k;
        return cached;
    } catch (...) {
        destroy_kernel(k);
        throw;
    }
}

// One dispatch: bind the pipeline, push the buffers and constants, launch the workgroups, and fence it off from the next command.
// `extents` gives, buffer by buffer, the floats a written binding's writes reach from its offset where the call knows them, 0 for the rest of the buffer; it bounds only which twins the writes drop.
inline void VulkanBackend::dispatch(KernelId id, std::initializer_list<VkDescriptorBufferInfo> buffers,
                                    const void* push, size_t push_bytes, uint32_t groups_x, uint32_t groups_y,
                                    int variant, std::initializer_list<size_t> extents) {
    Kernel& k = kernel(id, variant);
    if (buffers.size() != k.buffers || extents.size() > buffers.size()) throw std::logic_error("vulkan: kernel binding count");
    if (push_bytes > kPushBytes) throw std::logic_error("vulkan: push constants exceed 128 bytes");
    for (const auto& b : buffers)
        if (!b.buffer) throw std::runtime_error("vulkan: dispatch over an empty allocation");
    const KernelSource& src = kKernels[id];
    const uint32_t written = kernel_writes()[id];
    for (uint32_t i = 0, e = 0; i < src.bindings; ++i)
        for (uint32_t j = 0; j < (src.counts ? src.counts[i] : 1); ++j, ++e)
            if (written >> i & 1) {
                const VkDescriptorBufferInfo& b = buffers.begin()[e];
                const size_t floats = e < extents.size() ? extents.begin()[e] : 0;
                forget_written(b.buffer, b.offset, floats ? b.offset + floats * sizeof(float) : VK_WHOLE_SIZE);
            }
    VkCommandBuffer cmd = open();
    if (unordered_) {
        std::array<std::pair<VkBuffer, bool>, kMaxBuffers> uses;
        if (buffers.size() > uses.size()) throw std::logic_error("vulkan: a kernel of more buffers than order_step holds");
        size_t e = 0;
        for (uint32_t i = 0; i < src.bindings; ++i)
            for (uint32_t j = 0; j < (src.counts ? src.counts[i] : 1); ++j, ++e) uses[e] = {buffers.begin()[e].buffer, (written >> i & 1) != 0};
        order_step(cmd, uses.data(), e);
    }
    std::vector<VkWriteDescriptorSet> writes(k.bindings);
    const VkDescriptorBufferInfo* next = buffers.begin();
    for (uint32_t i = 0; i < k.bindings; ++i) {
        VkWriteDescriptorSet& w = writes[i];
        w = VkWriteDescriptorSet{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstBinding = i;
        w.descriptorCount = src.counts ? src.counts[i] : 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = next;
        next += w.descriptorCount;
    }
    dev_->fn.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipeline);
    dev_->fn.vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0,
                                       (uint32_t)writes.size(), writes.data());
    dev_->fn.vkCmdPushConstants(cmd, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                (uint32_t)push_bytes, push);
    if (dev_->timestamps) {
        // A reading interval takes as many pools as its dispatches need, each reset as the interval first reaches it.
        const size_t pool = query_next_ / kQueries;
        if (pool == queries_.size()) {
            VkQueryPoolCreateInfo qp{};
            qp.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            qp.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qp.queryCount = kQueries;
            VkQueryPool queries = VK_NULL_HANDLE;
            check(dev_->fn.vkCreateQueryPool(dev_->device, &qp, nullptr, &queries), "vkCreateQueryPool");
            queries_.push_back(queries);
        }
        if (pool == pools_reset_) {
            dev_->fn.vkCmdResetQueryPool(cmd, queries_[pool], 0, kQueries);
            ++pools_reset_;
        }
        const uint32_t q = (uint32_t)(query_next_ % kQueries);
        dev_->fn.vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries_[pool], q);
        dev_->fn.vkCmdDispatch(cmd, groups_x, groups_y, 1);
        dev_->fn.vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries_[pool], q + 1);
        query_kernel_.push_back(id * kVariants + variant + (head_ ? kTimedKernels : 0));
        query_next_ += 2;
    } else {
        dev_->fn.vkCmdDispatch(cmd, groups_x, groups_y, 1);
    }
    if (!unordered_) barrier(cmd);
    // A pass is submitted in chunks so the device starts on the first while the host records the rest; the ordered timeline makes the last chunk's ticket cover them all.
    if (++chunk_ >= chunk_limit_) flush();
}

} // namespace
} // namespace backend
