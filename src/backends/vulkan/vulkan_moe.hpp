#pragma once
// VulkanBackend's fused routed step (Backend::moe_step): a decode pass's router, routing, experts and shared expert in four dispatches where the unfused step takes eleven or more; vulkan_backend.cpp includes it after the class.
#include <algorithm>
#include <vector>
#include "backends/vulkan/vulkan_buffer.hpp"
#include "backends/vulkan/vulkan_kernels.hpp"
#include "quant/quant.hpp"

namespace backend {
namespace {

// A fused build of one product slot pair: the routed rows' kernel and the shared expert's (K_COUNT for none) that row_plan picks, whose bodies the build compiles.
// A slot matches only by that kernel, since a build decodes its family alone and a shared type of another family would run the wrong decoder.
struct MoeSlots {
    KernelId routed, shared, build;
};
const MoeSlots kMoeUpBuilds[] = {
    {K_MATMUL_ROW_K4_DOT, K_MATMUL_VEC_Q8, K_MOE_STEP_UP_K4Q8}, {K_MATMUL_ROW_K4_I8, K_MATMUL_VEC_Q8_I8, K_MOE_STEP_UP_K4Q8_I8},
    {K_MATMUL_ROW_K4_DOT, K_MATMUL_ROW_K4_DOT, K_MOE_STEP_UP_K4}, {K_MATMUL_ROW_K4_I8, K_MATMUL_ROW_K4_I8, K_MOE_STEP_UP_K4_I8},
};
// Every down build also runs its routed rows alone (sh_rows 0), the step adding a shared down of another kernel after them.
const MoeSlots kMoeDownBuilds[] = {
    {K_MATMUL_VEC_Q8, K_COUNT, K_MOE_STEP_DOWN_Q8}, {K_MATMUL_VEC_Q8_I8, K_COUNT, K_MOE_STEP_DOWN_Q8_I8},
    {K_MATMUL_VEC_Q8, K_MATMUL_VEC_Q8, K_MOE_STEP_DOWN_Q8Q8}, {K_MATMUL_VEC_Q8_I8, K_MATMUL_VEC_Q8_I8, K_MOE_STEP_DOWN_Q8Q8_I8},
    {K_MATMUL_ROW_Q5_DOT, K_MATMUL_VEC_Q8, K_MOE_STEP_DOWN_Q5Q8}, {K_MATMUL_ROW_Q5_I8, K_MATMUL_VEC_Q8_I8, K_MOE_STEP_DOWN_Q5Q8_I8},
    {K_MATMUL_ROW_K_DOT, K_MATMUL_ROW_K_DOT, K_MOE_STEP_DOWN_K}, {K_MATMUL_ROW_K_I8, K_MATMUL_ROW_K_I8, K_MOE_STEP_DOWN_K_I8},
    {K_MATMUL_ROW_K_DOT, K_MATMUL_VEC_Q8, K_MOE_STEP_DOWN_KQ8}, {K_MATMUL_ROW_K_I8, K_MATMUL_VEC_Q8_I8, K_MOE_STEP_DOWN_KQ8_I8},
    {K_MATMUL_ROW_K5_DOT, K_MATMUL_VEC_Q8, K_MOE_STEP_DOWN_K5Q8}, {K_MATMUL_ROW_K5_I8, K_MATMUL_VEC_Q8_I8, K_MOE_STEP_DOWN_K5Q8_I8},
    {K_MATMUL_ROW_K4_DOT, K_MATMUL_ROW_K4_DOT, K_MOE_STEP_DOWN_K4}, {K_MATMUL_ROW_K4_I8, K_MATMUL_ROW_K4_I8, K_MOE_STEP_DOWN_K4_I8},
};

// What a fused step fuses (moe_build): the route, the gate and up products (K_COUNT: the routed and shared calls), the down build and whether it holds the shared down.
struct MoePlan {
    bool route = false;
    KernelId up = K_COUNT, down = K_COUNT;
    bool shared_down = false;
};

// The route build holding the fewest experts an invocation needs, as route_experts picks moe_route's.
inline KernelId moe_route_kernel(size_t n_expert) {
    return n_expert <= 64 ? K_MOE_STEP_ROUTE_S1 : n_expert <= 128 ? K_MOE_STEP_ROUTE_S2 : n_expert <= 256 ? K_MOE_STEP_ROUTE_S4 : n_expert <= 512 ? K_MOE_STEP_ROUTE_S8 : K_MOE_STEP_ROUTE;
}

// Whether a step fuses, and what (MoePlan): decode rows alone with the silu GLU and no clamp, f16 or int8 on an integer-dot profile, a fused down build for the routed down's kernel.
// Each other slot fuses where its kernel has a build and its unfused call would take the same row kernel, and is otherwise the base's own call.
inline bool VulkanBackend::moe_build(const MoeStep& m, size_t nin, size_t rows, RowRuns runs, Dtype dtype, MoePlan& plan) const {
    const DeviceProfile& prof = dev_->profile;
    plan = MoePlan{};
    const MoeShared* s = m.shared;
    if (m.glu.limit != 0.0f || m.glu.form != GluForm::silu || (s && (s->glu.limit != 0.0f || s->glu.form != GluForm::silu))) return false;
    if (!dev_->workgroup_counter || !prof.prefer_integer_dot || unordered_) return false;
    if ((dtype != Dtype::f16 && dtype != Dtype::int8) || !rows || rows > kMoeStepRows || !runs.n || runs.runs[runs.n - 1].end != rows) return false;
    for (size_t i = 0; i < runs.n; ++i)
        if (runs.runs[i].extent != 1) return false;
    const size_t n_expert = m.router.rows, k = m.k, ff = m.gate.rows;
    if (!k || k > n_expert || k > 256 || n_expert > 1024 || rows * k >= 2 * n_expert) return false;
    if (m.gate.weight.type != m.up.weight.type || m.up.rows != ff || ff % 32 || nin % 32) return false;
    // The fused kernels read whole expert matrices, so row windows (Op::expert_window) take the base's ops, whose matmul_experts reads them before any run.
    if (m.gate.first || m.gate.pitch || m.up.first || m.up.pitch) return false;
    // The activation kernel scales the shared expert's product by its row gate, so a shared expert without one takes the base.
    if (s && (!s->row_gate.weight.data.buffer || s->gate.weight.type != s->up.weight.type || s->gate.rows != ff || s->up.rows != ff)) return false;
    // A routed call takes its row kernel below its tile crossover, a plain call below its own.
    auto routed_rows = [&](uint32_t type) { return 1 < prof.*(weight_kernels(type)->moe_from); };
    const RowPlan down = row_plan(m.down.type, ff, dtype);
    if (!routed_rows(m.down.type)) return false;
    const Projection sh_down{s ? s->down : m.down, Slice{}, nin};
    const KernelId sh_dn = s && tile_from(&sh_down, 1, ff) > 1 ? row_plan(s->down.type, ff, dtype).kernel : K_COUNT;
    // A row-bodied shared slot runs in the routed rows' lanes, so its plan's cluster must be theirs.
    const bool sh_lanes = sh_dn == K_COUNT || is_vec_kernel(sh_dn) || row_plan(s->down.type, ff, dtype).cluster == down.cluster;
    const MoeSlots *both = nullptr, *alone = nullptr;
    for (const MoeSlots& b : kMoeDownBuilds) {
        if (b.routed != down.kernel) continue;
        if (sh_dn != K_COUNT && b.shared == sh_dn && sh_lanes) both = &b;
        if (!alone || b.shared == K_COUNT) alone = &b;
    }
    if (!both && !alone) return false;
    plan.down = both ? both->build : alone->build;
    plan.shared_down = both != nullptr;
    const Projection router{m.router.weight, m.router.out, n_expert};
    plan.route = m.rule.score == RouteRule::Score::softmax && m.router.weight.type == quant::GGML_TYPE_F32 && (!s || s->row_gate.weight.type == quant::GGML_TYPE_F32) &&
                 tile_from(&router, 1, nin) > 1;
    if (s && routed_rows(m.gate.weight.type)) {
        const Projection group[2] = {s->gate, s->up};
        const KernelId rk = row_plan(m.gate.weight.type, nin, dtype).kernel, sk = tile_from(group, 2, nin) > 1 ? row_plan(s->gate.weight.type, nin, dtype).kernel : K_COUNT;
        for (const MoeSlots& b : kMoeUpBuilds)
            if (b.routed == rk && b.shared == sk) plan.up = b.build;
    }
    return true;
}

// D1 of the fused step: the router's and the row gate's F32 rows, then the routing by the last workgroup.
inline void VulkanBackend::moe_route_step(const MoeStep& m, CSlice h, size_t nin, size_t rows, bool normalize) {
    const size_t n_expert = m.router.rows, k = m.k, entries = rows * k;
    const MoeShared* s = m.shared;
    const uint32_t S = dev_->caps.subgroup_size;
    if (!moe_count_) {
        // Zeroed once in stream order; every call's last workgroup sets it back to zero.
        grow(moe_count_, sizeof(uint32_t));
        VkCommandBuffer cmd = open();
        settle(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT);
        dev_->fn.vkCmdFillBuffer(cmd, moe_count_->handle(), 0, VK_WHOLE_SIZE, 0);
        barrier(cmd);
    }

    {
        const RowPlan plan = row_plan(quant::GGML_TYPE_F32, nin, Dtype::f32);
        const int variant = rows == 1 ? 1 : 0;
        const uint32_t per_group = (256 / S) * sg_rows(plan, variant), cols = build_cols(plan.kernel, variant);
        const uint32_t gr = u32(groups(n_expert, per_group)), gg = s ? 1u : 0u, gy = u32((rows + cols - 1) / cols);
        const uint32_t pc[26] = {u32(nin), u32(rows), 0, 0, plan.cluster, S / plan.cluster, s ? 2u : 1u,
                                 u32(n_expert), quant::GGML_TYPE_F32, 0, 0, 1, quant::GGML_TYPE_F32, 0, gr, 0, 0, 0, 0, 0, 0, 0,
                                 u32(n_expert), u32(k), normalize ? 1u : 0u, (gr + gg) * gy};
        const VkDescriptorBufferInfo sc = bind(m.router.out), hf = bind(h), rw = bind(m.router.weight.data);
        const VkDescriptorBufferInfo sg = s ? bind(s->row_gate.out) : sc, gw = s ? bind(s->row_gate.weight.data) : rw;
        dispatch(moe_route_kernel(n_expert), {sc, sg, sc, rw, gw, rw, rw, gw, rw, rw, gw, rw, hf, rw, gw, rw, hf, hf, hf, hf, hf, hf,
                                             sc, bind(m.ids), bind(m.weights), VkDescriptorBufferInfo{moe_count_->handle(), 0, VK_WHOLE_SIZE}},
                 pc, sizeof(pc), gr + gg, gy, variant, {rows * n_expert, s ? rows : 0, rows * n_expert, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                       0, entries, entries, 1});
        group_tag_ = GroupTag{};
    }
}

// A cached step (ExpertSlots): route as the step without the cache would, mark the picks, then rounds of waves and products in ExpertRing::run's order.
// Each round's down products stay parts in moe_out_ until one combine in j order, then the shared expert joins y as the base's ops add it.
inline void VulkanBackend::cached_step(const MoeStep& m, Slice h, Slice y, size_t nin, size_t rows, RowRuns runs, Dtype dtype) {
    const ExpertSlots& c = *m.slots;
    const size_t n_expert = m.router.rows, k = m.k, ff = m.gate.rows, entries = size_mul(rows, k);
    if (n_expert != c.n_expert || c.slots > 0xffff || n_expert > 0xffff || m.gate.weight.scales.buffer || m.up.weight.scales.buffer || m.down.scales.buffer)
        throw std::runtime_error("vulkan: an expert cache step of another expert count, over 65535 slots or experts, or of scaled stacks");
    const OpSteps steps(*this);
    ExpertRing& ring = ExpertRing::of(expert_ring_, *this, c, "vulkan: ");
    constexpr size_t chunk = 65536;
    ExpertRing::Source src{};
    uint32_t strides[3];
    size_t chunks = 0;
    for (int i = 0; i < 3; ++i) {
        const ExpertSlots::Stack& st = c.stack[i];
        if (st.stride % 16 || !st.host || bytes_from(CSlice(st.dst, 0)) < c.slots * st.stride || st.stride > ring.entry(i))
            throw std::runtime_error("vulkan: an expert cache stack off its alignment, outside its allocation or wider than the ring");
        strides[i] = u32(st.stride / 16);
        src.at[i] = st.host;
        src.stride[i] = st.stride;
        src.file[i] = st.file;
        src.file_at[i] = st.file_at;
        chunks = std::max<size_t>(chunks, groups(st.stride, chunk));
    }
    ExpertRing::count_step(src, c, rows);
    // The routing the step without the cache takes: the fused route where that step fuses, else the base's.
    MoePlan plan;
    const bool fused = moe_build(m, nin, rows, runs, dtype, plan) && plan.route;
    if (m.zero) repeat_row(y, *m.zero, rows, nin);
    if (fused) {
        moe_input(m, h, nin, rows, runs);
        moe_route_step(m, h, nin, rows, m.rule.softmax_only("vulkan"));
    } else {
        moe_router(m, h, nin, rows, runs);
    }
    if (!expert_ids_ || expert_ids_->size() < entries * sizeof(uint32_t)) grow(expert_ids_, entries * sizeof(uint32_t));
    const uint32_t ne = u32(n_expert), en = u32(entries), sl = u32(c.slots), hw = u32(ring.half()), fl = u32(c.flags_at());
    const VkDescriptorBufferInfo state = bind(c.state), xchg = bind(CSlice(&ring.exchange_buffer(), 0)), round_ids = bind(CSlice(expert_ids_.get(), 0));
    const size_t xwords = ExpertRing::words(ring.half());
    const uint32_t mc[5] = {ne, en, u32(c.age_at()), u32(c.counts_at()), fl};
    dispatch(K_EXPERT_MARK, {bind(m.ids), state}, mc, sizeof(mc), 1, 1, 0, {0, c.words()});
    const uint32_t ticks = (uint32_t)std::min<double>(4.0e9, kRingWaitMs * 1.0e6 / std::max(dev_->timestamp_ns, 1e-3));
    const auto wave = [&] {
        const uint64_t seq = ring.next();
        const uint32_t wc[9] = {ne, sl, hw, uint32_t(seq), u32(c.owner_at()), u32(c.age_at()), u32(c.counts_at()), fl, u32(ring.half_at(seq))};
        dispatch(K_EXPERT_WAVE, {state, xchg}, wc, sizeof(wc), 1, 1, 0, {c.words(), xwords});
        ring.request(submit(), seq, src);
        return seq;
    };
    // The hold spins on the half's ticket, which the reader writes once the wave's experts are in the ring, then the pull copies them into their slots.
    const auto whole = [&](const Buffer* b) { return bind(CSlice(b, 0)); };
    const auto pull = [&](uint64_t seq) {
        const size_t at = ring.half_at(seq);
        const uint32_t hc[2] = {uint32_t(seq), ticks};
        dispatch(K_HOLD, {bind(CSlice(&ring.exchange_buffer(), at))}, hc, sizeof(hc), 1);
        const uint32_t pc[9] = {u32(chunks), u32(chunk / 16), strides[0], strides[1], strides[2], u32(at), uint32_t(seq), u32(ExpertRing::kFault), u32(c.counts_at() + 2 + (seq & 1))};
        dispatch(K_EXPERT_PULL, {xchg, whole(&ring.entries(0)), whole(&ring.entries(1)), whole(&ring.entries(2)), whole(c.stack[0].dst), whole(c.stack[1].dst),
                                 whole(c.stack[2].dst), state},
                 pc, sizeof(pc), u32(ring.half() * chunks), 3);
    };
    const auto round = [&](size_t, bool last) {
        const uint32_t rc[5] = {ne, en, fl, last ? 1u : 0u, u32(ExpertRing::kUnplaced)};
        dispatch(K_EXPERT_ROUND, {bind(m.ids), round_ids, state, xchg}, rc, sizeof(rc), 1, 1, 0, {0, entries, c.words(), xwords});
    };
    if (!moe_out_ || moe_out_->size() < entries * nin * sizeof(float)) grow(moe_out_, entries * nin * sizeof(float));
    std::vector<RowRun>& entry_runs = *m.scratch;
    entry_runs.clear();
    for (size_t i = 0; i < runs.n; ++i) entry_runs.push_back(RowRun{runs.runs[i].end * k, runs.runs[i].extent});
    // A round's products: entries outside it read no expert, so their parts in moe_out_ keep the rounds before, which nothing else writes until the combine.
    const auto products = [&] {
        const Routing routing{CSlice(expert_ids_.get(), 0), m.weights, k, c.slots};
        matmul_experts({m.gate, m.up}, h, nin, rows, routing, runs, dtype);
        silu_mul(m.act, m.gate.out, m.up.out, entries * ff, m.glu, {entry_runs.data(), entry_runs.size()});
        routed_parts(m.down, m.act, ff, nin, rows, routing, runs, dtype);
    };
    ExpertRing::run(ring.plan(entries, n_expert, c.slots), wave, pull, round, products);
    combine_parts(y, nin, rows, Routing{m.ids, m.weights, k, n_expert});
    moe_shared(m, h, y, nin, rows, runs, dtype);
}

inline void VulkanBackend::moe_step(const MoeStep& m, Slice h, Slice y, size_t nin, size_t rows, RowRuns runs, Dtype dtype) {
    if (m.slots) return cached_step(m, h, y, nin, rows, runs, dtype);
    const OpSteps steps(*this);
    MoePlan plan;
    if (!moe_build(m, nin, rows, runs, dtype, plan)) return Backend::moe_step(m, h, y, nin, rows, runs, dtype);
    const size_t n_expert = m.router.rows, k = m.k, ff = m.gate.rows, entries = rows * k, E = nin;
    const MoeShared* s = m.shared;
    const uint32_t S = dev_->caps.subgroup_size;
    // The operands as the unfused calls check them.
    check_matrix(m.router.weight.type, m.router.weight.data, nin, n_expert);
    check_weight_scales(m.router.weight, nin, n_expert);
    check_matrix(m.gate.weight.type, m.gate.weight.data, nin, size_mul(n_expert, ff));
    check_weight_scales(m.gate.weight, nin, ff, n_expert);
    check_matrix(m.up.weight.type, m.up.weight.data, nin, size_mul(n_expert, ff));
    check_weight_scales(m.up.weight, nin, ff, n_expert);
    check_matrix(m.down.type, m.down.data, ff, size_mul(n_expert, E));
    check_weight_scales(m.down, ff, E, n_expert);
    const size_t act_n = size_mul(rows, k * ff + (s ? ff : 0));
    if (floats_from(h) < rows * E || floats_from(y) < rows * E || floats_from(m.router.out) < rows * n_expert || floats_from(m.ids) < entries ||
        floats_from(m.weights) < entries || floats_from(m.gate.out) < act_n || floats_from(m.up.out) < act_n || floats_from(m.act) < act_n)
        throw std::runtime_error("vulkan: routed step operand outside its allocation");
    if (s) {
        check_matrix(s->gate.weight.type, s->gate.weight.data, nin, ff);
        check_weight_scales(s->gate.weight, nin, ff);
        check_matrix(s->up.weight.type, s->up.weight.data, nin, ff);
        check_weight_scales(s->up.weight, nin, ff);
        check_matrix(s->down.type, s->down.data, ff, E);
        check_weight_scales(s->down, ff, E);
        check_matrix(s->row_gate.weight.type, s->row_gate.weight.data, nin, 1);
        check_weight_scales(s->row_gate.weight, nin, 1);
        // The shared rows sit right after the routed ones, which the fused kernels read as one run of entries.
        if (s->gate.out.buffer != m.gate.out.buffer || s->gate.out.offset != m.gate.out.offset + entries * ff || s->up.out.buffer != m.up.out.buffer ||
            s->up.out.offset != m.up.out.offset + entries * ff || s->act.buffer != m.act.buffer || s->act.offset != m.act.offset + entries * ff ||
            floats_from(s->row_gate.out) < rows)
            return Backend::moe_step(m, h, y, nin, rows, runs, dtype);
    }
    if (m.zero && bytes_from(CSlice(m.zero, 0)) < E * sizeof(float)) throw std::runtime_error("vulkan: routed step zero row outside its allocation");
    // D1: the router's and the row gate's rows and the routing in one launch, or the base's router and row gate where that launch does not take them.
    if (plan.route) {
        moe_input(m, h, nin, rows, runs);
        moe_route_step(m, h, nin, rows, m.rule.softmax_only("vulkan"));
    } else {
        moe_router(m, h, nin, rows, runs);
        if (s) matmul(s->row_gate.weight, h, s->row_gate.out, nin, 1, rows, runs, dtype);
    }
    const Routing routing{m.ids, m.weights, k, n_expert};
    std::vector<RowRun>& entry_runs = *m.scratch;
    entry_runs.clear();
    for (size_t i = 0; i < runs.n; ++i) entry_runs.push_back(RowRun{runs.runs[i].end * k, runs.runs[i].extent});
    const RowPlan down = row_plan(m.down.type, ff, dtype);
    if (!s) {
        // D2 and D2b: without a shared expert the routed row kernel and the SiLU run as the unfused step runs them.
        matmul_experts({m.gate, m.up}, h, nin, rows, routing, runs, dtype);
        silu_mul(m.act, m.gate.out, m.up.out, entries * ff, {}, {entry_runs.data(), entry_runs.size()});
    } else {
        if (plan.up == K_COUNT) {
            // D2 unfused: the routed calls and the shared expert's group call, as the base makes them.
            matmul_experts({m.gate, m.up}, h, nin, rows, routing, runs, dtype);
            matmul_group({s->gate, s->up}, h, nin, rows, runs, dtype);
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
            dispatch(plan.up, {g, u, g, u, a, b, c, d, a, b, c, d, a, b, c, d, bind(h), a, b, c, d, xqi, xqi, xqi, xqi, ids, ids},
                     pc, sizeof(pc), std::max(2 * start1, 2 * sh_start1), u32(entries + rows), variant, {act_n, act_n, act_n, act_n});
        }
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
        // The twin covers every activation column, the one D2b keeps, even where the shared down runs after.
        const size_t columns = entries + (s ? rows : 0);
        const VkDescriptorBufferInfo xqi = row_twin(m.act, down, columns * ff, dtype);
        const bool routed_q8 = is_vec_kernel(down.kernel), fused_shared = plan.shared_down;
        const RowPlan sp = fused_shared ? row_plan(s->down.type, ff, dtype) : down;
        const bool shared_q8 = fused_shared && is_vec_kernel(sp.kernel);
        const int variant = routed_q8 ? one_column_build(ff, E * entries, reads_x8(down.kernel)) : shared_q8 ? one_column_build(ff, E, reads_x8(sp.kernel)) : 1;
        const uint32_t rows_wg = routed_q8 ? (256 / S) * kVecBuilds[variant].rows : 256 / down.cluster;
        const uint32_t sh_rows = !fused_shared ? 0u : shared_q8 ? (256 / S) * kVecBuilds[variant].rows : rows_wg;
        const uint32_t pc[29] = {u32(ff), u32(columns), 0, 1, down.cluster, S / down.cluster, 2,
                                 u32(E), down.type, down.wide, 0, u32(E), sp.type, sp.wide, 0, 0, 0, 0, 0, 0, 1, 0,
                                 u32(k), u32(rows), m.zero ? 1u : 0u, rows_wg, sh_rows, sp.type, sp.wide};
        const VkDescriptorBufferInfo yb = bind(y), a = bind(m.down.data), b = fused_shared ? bind(s->down.data) : a, ids = bind(m.ids);
        dispatch(plan.down, {yb, yb, a, b, a, b, a, b, bind(m.act), a, b, xqi, xqi, xqi, xqi, ids, ids, bind(m.weights),
                                               m.zero ? bind(CSlice(m.zero, 0)) : yb},
                 pc, sizeof(pc), u32(groups(E, rows_wg)), u32(rows), variant, {rows * E, rows * E});
        // A shared down without a build beside the routed one joins y after it, as the base adds it.
        if (s && !fused_shared) matmul_add(s->down, s->act, y, ff, nin, rows, runs, dtype);
    }
}

} // namespace
} // namespace backend
