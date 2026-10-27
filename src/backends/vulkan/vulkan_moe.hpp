#pragma once
// VulkanBackend's fused routed step (Backend::moe_step): a decode pass's router, routing, experts and shared expert in four dispatches where the unfused step takes eleven or more; vulkan_backend.cpp includes it after the class.
#include <algorithm>
#include <vector>
#include "backends/vulkan/vulkan_buffer.hpp"
#include "backends/vulkan/vulkan_kernels.hpp"
#include "quant/quant.hpp"

namespace backend {
namespace {

// The type combinations the fused step builds, a routed layer's of the target files: the experts' gate and up, their down, the shared expert's gate and up and its down, none for a layer without one.
// Each names its gate-and-up kernel (K_COUNT where the routed row kernel runs alone) and its down kernel, in the 16-bit and 8-bit builds.
struct MoeBuild {
    uint32_t gate, down, sh_gate, sh_down;
    KernelId up, up8, dn, dn8;
};
const MoeBuild kMoeBuilds[] = {
    {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q8_0, 0, 0, K_COUNT, K_COUNT, K_MOE_STEP_DOWN_Q8, K_MOE_STEP_DOWN_Q8_I8},
    {quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q8_0, K_MOE_STEP_UP_K4Q8, K_MOE_STEP_UP_K4Q8_I8, K_MOE_STEP_DOWN_Q5Q8, K_MOE_STEP_DOWN_Q5Q8_I8},
    {quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q6_K, K_MOE_STEP_UP_K4, K_MOE_STEP_UP_K4_I8, K_MOE_STEP_DOWN_K, K_MOE_STEP_DOWN_K_I8},
};

// The route build holding the fewest experts an invocation needs, as route_experts picks moe_route's.
inline KernelId moe_route_kernel(size_t n_expert) {
    return n_expert <= 64 ? K_MOE_STEP_ROUTE_S1 : n_expert <= 128 ? K_MOE_STEP_ROUTE_S2 : n_expert <= 256 ? K_MOE_STEP_ROUTE_S4 : n_expert <= 512 ? K_MOE_STEP_ROUTE_S8 : K_MOE_STEP_ROUTE;
}

// The build a step fuses through, or null for the base: decode rows alone on the per-entry row kernels, the types of a kMoeBuilds entry in the kernels row_plan picks, f16 or int8 on an integer-dot profile.
inline const MoeBuild* VulkanBackend::moe_build(const MoeStep& m, size_t nin, size_t rows, RowRuns runs, Dtype dtype) const {
    const DeviceProfile& prof = dev_->profile;
    if (!dev_->workgroup_counter || !prof.prefer_integer_dot || unordered_) return nullptr;
    if ((dtype != Dtype::f16 && dtype != Dtype::int8) || !rows || rows > kMoeStepRows || !runs.n || runs.runs[runs.n - 1].end != rows) return nullptr;
    for (size_t i = 0; i < runs.n; ++i)
        if (runs.runs[i].extent != 1) return nullptr;
    const size_t n_expert = m.router.rows, k = m.k, ff = m.gate.rows;
    if (m.router.weight.type != quant::GGML_TYPE_F32 || !k || k > n_expert || k > 256 || n_expert > 1024 || rows * k >= 2 * n_expert) return nullptr;
    if (m.gate.weight.type != m.up.weight.type || m.up.rows != ff || ff % 32 || nin % 32) return nullptr;
    const MoeShared* s = m.shared;
    if (s && (!s->row_gate.weight.data.buffer || s->row_gate.weight.type != quant::GGML_TYPE_F32 || s->gate.weight.type != s->up.weight.type || s->gate.rows != ff || s->up.rows != ff))
        return nullptr;
    const MoeBuild* build = nullptr;
    for (const MoeBuild& b : kMoeBuilds)
        if (b.gate == m.gate.weight.type && b.down == m.down.type && b.sh_gate == (s ? s->gate.weight.type : 0u) && b.sh_down == (s ? s->down.type : 0u)) build = &b;
    if (!build) return nullptr;
    // Every product the unfused step makes takes a row kernel: the routed calls below their tile crossover, the plain ones below theirs.
    auto routed_rows = [&](uint32_t type) { return 1 < prof.*(weight_kernels(type)->moe_from); };
    if (!routed_rows(m.gate.weight.type) || !routed_rows(m.down.type) || dtype == Dtype::bf16) return nullptr;
    const Projection router{m.router.weight, m.router.out, n_expert};
    if (tile_from(&router, 1, nin) <= 1) return nullptr;
    if (s) {
        const Projection group[2] = {s->gate, s->up}, down{s->down, Slice{}, nin};
        if (tile_from(group, 2, nin) <= 1 || tile_from(&down, 1, ff) <= 1) return nullptr;
    }
    // The kernels the fused builds share their bodies with: the routed row kernel of each type and the Q8_0 decode kernel.
    auto kernel_of = [&](uint32_t type, size_t width) {
        const KernelId id = row_plan(type, width, dtype).kernel;
        const bool i8 = dtype == Dtype::int8;
        if (type == quant::GGML_TYPE_Q8_0) return id == (i8 ? K_MATMUL_VEC_Q8_I8 : K_MATMUL_VEC_Q8);
        if (type == quant::GGML_TYPE_Q4_K) return id == (i8 ? K_MATMUL_ROW_K4_I8 : K_MATMUL_ROW_K4_DOT);
        if (type == quant::GGML_TYPE_Q5_1) return id == (i8 ? K_MATMUL_ROW_Q5_I8 : K_MATMUL_ROW_Q5_DOT);
        return id == (i8 ? K_MATMUL_ROW_K_I8 : K_MATMUL_ROW_K_DOT);
    };
    if (!kernel_of(m.gate.weight.type, nin) || !kernel_of(m.down.type, ff) || (s && (!kernel_of(s->gate.weight.type, nin) || !kernel_of(s->down.type, ff)))) return nullptr;
    return build;
}

inline void VulkanBackend::moe_step(const MoeStep& m, CSlice h, Slice y, size_t nin, size_t rows, RowRuns runs, Dtype dtype) {
    const MoeBuild* build = moe_build(m, nin, rows, runs, dtype);
    if (!build) return Backend::moe_step(m, h, y, nin, rows, runs, dtype);
    const size_t n_expert = m.router.rows, k = m.k, ff = m.gate.rows, entries = rows * k, E = nin;
    const MoeShared* s = m.shared;
    const uint32_t S = dev_->caps.subgroup_size;
    const bool i8 = dtype == Dtype::int8;
    // The operands as the unfused calls check them.
    check_matrix(m.router.weight.type, m.router.weight.data, nin, n_expert);
    check_weight_scales(m.router.weight);
    check_matrix(m.gate.weight.type, m.gate.weight.data, nin, size_mul(n_expert, ff));
    check_weight_scales(m.gate.weight, n_expert);
    check_matrix(m.up.weight.type, m.up.weight.data, nin, size_mul(n_expert, ff));
    check_weight_scales(m.up.weight, n_expert);
    check_matrix(m.down.type, m.down.data, ff, size_mul(n_expert, E));
    check_weight_scales(m.down, n_expert);
    const size_t act_n = size_mul(rows, k * ff + (s ? ff : 0));
    if (floats_from(h) < rows * E || floats_from(y) < rows * E || floats_from(m.router.out) < rows * n_expert || floats_from(m.ids) < entries ||
        floats_from(m.weights) < entries || floats_from(m.gate.out) < act_n || floats_from(m.up.out) < act_n || floats_from(m.act) < act_n)
        throw std::runtime_error("vulkan: routed step operand outside its allocation");
    if (s) {
        check_matrix(s->gate.weight.type, s->gate.weight.data, nin, ff);
        check_weight_scales(s->gate.weight);
        check_matrix(s->up.weight.type, s->up.weight.data, nin, ff);
        check_weight_scales(s->up.weight);
        check_matrix(s->down.type, s->down.data, ff, E);
        check_weight_scales(s->down);
        check_matrix(s->row_gate.weight.type, s->row_gate.weight.data, nin, 1);
        check_weight_scales(s->row_gate.weight);
        // The shared rows sit right after the routed ones, which the fused kernels read as one run of entries.
        if (s->gate.out.buffer != m.gate.out.buffer || s->gate.out.offset != m.gate.out.offset + entries * ff || s->up.out.buffer != m.up.out.buffer ||
            s->up.out.offset != m.up.out.offset + entries * ff || s->act.buffer != m.act.buffer || s->act.offset != m.act.offset + entries * ff ||
            floats_from(s->row_gate.out) < rows)
            return Backend::moe_step(m, h, y, nin, rows, runs, dtype);
    }
    if (m.zero && bytes_from(CSlice(m.zero, 0)) < E * sizeof(float)) throw std::runtime_error("vulkan: routed step zero row outside its allocation");
    if (!moe_count_) {
        // Zeroed once in stream order; every call's last workgroup sets it back to zero.
        grow(moe_count_, sizeof(uint32_t));
        VkCommandBuffer cmd = open();
        dev_->fn.vkCmdFillBuffer(cmd, moe_count_->handle(), 0, VK_WHOLE_SIZE, 0);
        barrier(cmd);
    }

    // D1: the router's and the row gate's F32 rows, then the routing by the last workgroup.
    {
        const RowPlan plan = row_plan(quant::GGML_TYPE_F32, nin, Dtype::f32);
        const int variant = rows == 1 ? 1 : 0;
        const uint32_t per_group = (256 / S) * sg_rows(plan, variant), cols = build_cols(plan.kernel, variant);
        const uint32_t gr = u32(groups(n_expert, per_group)), gg = s ? 1u : 0u, gy = u32((rows + cols - 1) / cols);
        const uint32_t pc[26] = {u32(nin), u32(rows), 0, 0, plan.cluster, S / plan.cluster, s ? 2u : 1u,
                                 u32(n_expert), quant::GGML_TYPE_F32, 0, 0, 1, quant::GGML_TYPE_F32, 0, gr, 0, 0, 0, 0, 0, 0, 0,
                                 u32(n_expert), u32(k), m.normalize ? 1u : 0u, (gr + gg) * gy};
        const VkDescriptorBufferInfo sc = bind(m.router.out), hf = bind(h), rw = bind(m.router.weight.data);
        const VkDescriptorBufferInfo sg = s ? bind(s->row_gate.out) : sc, gw = s ? bind(s->row_gate.weight.data) : rw;
        dispatch(moe_route_kernel(n_expert), {sc, sg, sc, rw, gw, rw, rw, gw, rw, rw, gw, rw, hf, rw, gw, rw, hf, hf, hf, hf, hf, hf,
                                             sc, bind(m.ids), bind(m.weights), VkDescriptorBufferInfo{moe_count_->handle(), 0, VK_WHOLE_SIZE}},
                 pc, sizeof(pc), gr + gg, gy, variant, {rows * n_expert, s ? rows : 0, rows * n_expert, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                       0, entries, entries, 1});
        group_tag_ = GroupTag{};
    }
    const Routing routing{m.ids, m.weights, k, n_expert};
    std::vector<RowRun>& entry_runs = *m.scratch;
    entry_runs.clear();
    for (size_t i = 0; i < runs.n; ++i) entry_runs.push_back(RowRun{runs.runs[i].end * k, runs.runs[i].extent});
    const RowPlan down = row_plan(m.down.type, ff, dtype);
    if (!s) {
        // D2 and D2b: without a shared expert the routed row kernel and the SiLU run as the unfused step runs them.
        matmul_experts({m.gate, m.up}, h, nin, rows, routing, runs, dtype);
        silu_mul(m.act, m.gate.out, m.up.out, entries * ff, {entry_runs.data(), entry_runs.size()});
    } else {
        // D2: the routed entries' gate and up rows beside each token's shared expert's.
        const RowPlan rp = row_plan(m.gate.weight.type, nin, dtype), sp = row_plan(s->gate.weight.type, nin, dtype);
        const bool q8 = is_vec_kernel(sp.kernel);
        const int variant = q8 ? one_column_build(nin, 2 * ff, reads_x8(sp.kernel)) : 1;
        const VkDescriptorBufferInfo xqi = row_twin(h, rp, rows * nin, dtype);
        const uint32_t rg = (256 / S) * sg_rows(rp, 1), sgr = q8 ? (256 / S) * kVecBuilds[variant].rows : rg;
        const uint32_t start1 = u32(groups(ff, rg)), sh_start1 = u32(groups(ff, sgr));
        const uint32_t pc[26] = {u32(nin), u32(rows), 0, 1, rp.cluster, S / rp.cluster, 4,
                                 u32(ff), rp.type, rp.wide, 0, u32(ff), rp.type, rp.wide, start1, 0, 0, 0, 0, 0, u32(k), 0,
                                 u32(entries), sh_start1, sp.type, sp.wide};
        const VkDescriptorBufferInfo g = bind(m.gate.out), u = bind(m.up.out), a = bind(m.gate.weight.data), b = bind(m.up.weight.data);
        const VkDescriptorBufferInfo c = bind(s->gate.weight.data), d = bind(s->up.weight.data), ids = bind(m.ids);
        dispatch(i8 ? build->up8 : build->up, {g, u, g, u, a, b, c, d, a, b, c, d, a, b, c, d, bind(h), a, b, c, d, xqi, xqi, xqi, xqi, ids, ids},
                 pc, sizeof(pc), std::max(2 * start1, 2 * sh_start1), u32(entries + rows), variant, {act_n, act_n, act_n, act_n});
        // D2b: every entry's SiLU product and each token's shared one under its gate, with the twin of all of them that the down kernel reads.
        if (reads_x8(down.kernel)) want_x8_ = true;
        const uint32_t ac[3] = {u32(act_n), u32(entries * ff), u32(ff)};
        const size_t slot = take_twin(act_n);
        dispatch(K_MOE_STEP_ACT, {bind(m.act), bind(m.gate.out), bind(m.up.out), bind(s->row_gate.out), twin_binding(slot)}, ac, sizeof(ac),
                 u32(groups(act_n, 256)), 1, twin_variant(), {act_n, 0, 0, 0, act_n});
        keep_twin(slot, bind(m.act), act_n, 0);
    }
    // D3: each token's routed down rows joined in slot order, then its shared expert's, into y.
    {
        const size_t columns = entries + (s ? rows : 0);
        const VkDescriptorBufferInfo xqi = row_twin(m.act, down, columns * ff, dtype);
        const bool routed_q8 = is_vec_kernel(down.kernel);
        const RowPlan sp = s ? row_plan(s->down.type, ff, dtype) : down;
        const bool shared_q8 = s && is_vec_kernel(sp.kernel);
        const int variant = routed_q8 ? one_column_build(ff, E * entries, reads_x8(down.kernel)) : shared_q8 ? one_column_build(ff, E, reads_x8(sp.kernel)) : 1;
        const uint32_t rows_wg = routed_q8 ? (256 / S) * kVecBuilds[variant].rows : 256 / down.cluster;
        const uint32_t sh_rows = shared_q8 ? (256 / S) * kVecBuilds[variant].rows : rows_wg;
        const uint32_t pc[29] = {u32(ff), u32(columns), 0, 1, down.cluster, S / down.cluster, 2,
                                 u32(E), down.type, down.wide, 0, u32(E), sp.type, sp.wide, 0, 0, 0, 0, 0, 0, 1, 0,
                                 u32(k), u32(rows), m.zero ? 1u : 0u, rows_wg, sh_rows, sp.type, sp.wide};
        const VkDescriptorBufferInfo yb = bind(y), a = bind(m.down.data), b = s ? bind(s->down.data) : a, ids = bind(m.ids);
        dispatch(i8 ? build->dn8 : build->dn, {yb, yb, a, b, a, b, a, b, bind(m.act), a, b, xqi, xqi, xqi, xqi, ids, ids, bind(m.weights),
                                               m.zero ? bind(CSlice(m.zero, 0)) : yb},
                 pc, sizeof(pc), u32(groups(E, rows_wg)), u32(rows), variant, {rows * E, rows * E});
    }
}

} // namespace
} // namespace backend
