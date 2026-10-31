#pragma once
// VulkanBackend's KV cache and attention: storage, the view tables, the KV write and the attention kernels; vulkan_backend.cpp includes it after the class.
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "backends/vulkan/vulkan_buffer.hpp"
#include "backends/vulkan/vulkan_device.hpp"
#include "backends/vulkan/vulkan_kernels.hpp"

namespace backend {
namespace {

// A non-causal attention call's workgroups cut into dispatches of at most `budget` multiply-adds, as boundaries from 0 to the call's workgroup count.
// `runs` gives, view by view, its workgroups and each one's multiply-adds; a view whose workgroups do not fit beside the open piece is cut into near-equal pieces, and a workgroup over the budget is a piece alone.
inline std::vector<size_t> attention_pieces(const std::vector<std::pair<size_t, size_t>>& runs, size_t budget) {
    std::vector<size_t> cuts{0};
    size_t at = 0, open = 0;
    for (const auto& [count, each] : runs) {
        if (!count) continue;
        if (size_add(open, size_mul(count, each)) <= budget) {
            at += count;
            open += count * each;
            continue;
        }
        if (open) cuts.push_back(at);
        const size_t per = std::max<size_t>(1, budget / std::max<size_t>(1, each)), pieces = (count + per - 1) / per;
        for (size_t i = 1; i < pieces; ++i) cuts.push_back(at + count * i / pieces);
        open = (count - count * (pieces - 1) / pieces) * each;
        at += count;
    }
    if (at > cuts.back()) cuts.push_back(at);
    return cuts;
}

// One dispatch for q and k norm-rope and the KV write, every view of the batch going through the view table.
inline void VulkanBackend::norm_rope_kv(Slice q, size_t q_stride, size_t n_head, CSlice q_w,
                                        Slice k, CSlice v, size_t kv_stride, size_t n_head_kv, CSlice k_w,
                                        const RopeArgs& rope, size_t rows, size_t layer,
                                        const KVView* views, size_t n_views) {
    if (!rows || !n_head || !n_head_kv || !rope.half) {
        Backend::norm_rope_kv(q, q_stride, n_head, q_w, k, v, kv_stride, n_head_kv, k_w,
                              rope, rows, layer, views, n_views);
        return;
    }
    if (!rope.pos) throw std::runtime_error("vulkan: rope without positions");
    const size_t table = floats_from(rope.cos) / rope.half;
    for (size_t r = 0; r < rows; ++r)
        if (rope.pos[r] >= table) throw std::runtime_error("vulkan: position outside the RoPE table");
    std::vector<Placed> placed = place_views(views, n_views);
    ViewTable t = view_table(layer, placed, true);
    if (!t.storage) return;
    VulkanKVStorage& s = *t.storage;
    const size_t dim = 2 * rope.half, hd = s.heads() * s.dim();
    if (s.dim() != dim || s.heads() != n_head_kv || t.rows != rows)
        throw std::runtime_error("vulkan: attention inputs do not match the KV storage");
    if (floats_from(q) < rows * q_stride || floats_from(k) < rows * kv_stride ||
        floats_from(v) < rows * kv_stride || q_stride < n_head * dim || kv_stride < hd)
        throw std::runtime_error("vulkan: attention rows outside their allocation");
    struct { uint32_t q_stride, n_head, kv_stride, n_head_kv, half; float eps; uint32_t bt; }
        pc{u32(q_stride), u32(n_head), u32(kv_stride), u32(n_head_kv), u32(rope.half),
           rope.eps, u32(kVkBlockTokens)};
    dispatch(kv_variant(K_NORM_ROPE_KV, K_NORM_ROPE_KV_K16, s),
             {bind(q), bind(k), bind(v), bind(q_w), bind(k_w), bind(rope.cos), bind(rope.sin),
              args(rope.pos, rows * sizeof(uint32_t)),
              bind(CSlice{s.k_buffer(layer).get(), 0}), bind(CSlice{s.v_buffer(layer).get(), 0}),
              args(t.words.data(), t.words.size() * sizeof(uint32_t))},
             &pc, sizeof(pc), u32(rows * (n_head + 2 * n_head_kv)));
}

inline KVLayout VulkanBackend::kv_layout() const { return KVLayout{kVkBlockTokens}; }

inline std::unique_ptr<KVStorage> VulkanBackend::kv_alloc(size_t layers, size_t n_head_kv, size_t head_dim,
                                                          size_t max_tokens, KVType k_type,
                                                          KVType v_type,
                                                          const std::vector<size_t>& auxiliary_widths, const std::vector<size_t>& auxiliary_pools) {
    if (!layers || !n_head_kv || !head_dim)
        throw std::runtime_error("vulkan: KV storage without layers, heads or width");
    if (!attention_head_fits(head_dim, dev_->caps.subgroup_size))
        throw std::runtime_error("vulkan: a head " + std::to_string(head_dim) + " wide is not supported on this device, whose attention takes heads 128 or 256 wide or at most " +
                                 std::to_string(attention_row_width(dev_->caps.subgroup_size)));
    return std::make_unique<VulkanKVStorage>(*this, layers, n_head_kv, head_dim, max_tokens, k_type, v_type, auxiliary_widths, auxiliary_pools);
}

// One dispatch for every view of the batch through the view table, each view's rows scattering into its blocks.
inline void VulkanBackend::kv_write(size_t layer, const KVView* views, size_t n_views, CSlice k,
                                    CSlice v, CSlice auxiliary) {
    std::vector<Placed> placed = place_views(views, n_views);
    if (placed.empty()) {
        if (auxiliary.buffer) throw std::runtime_error("vulkan: auxiliary KV input supplied without views");
        return;
    }
    VulkanKVStorage* storage = nullptr;
    size_t rows = 0;
    for (const Placed& pv : placed) {
        VulkanKVStorage& current = storage_of(pv.view->storage);
        if (storage && storage != &current) throw std::runtime_error("vulkan: views of two storages in one call");
        storage = &current;
        storage->validate_write_view(*pv.view, layer);
        rows = size_add(rows, pv.view->nq);
    }
    const size_t hd = size_mul(storage->heads(), storage->dim());
    const size_t kv_floats = size_mul(rows, hd);
    if (!k.buffer || !v.buffer) throw std::runtime_error("vulkan: operand without storage");
    span(*k.buffer, size_mul(k.offset, sizeof(float)), size_mul(kv_floats, sizeof(float)));
    span(*v.buffer, size_mul(v.offset, sizeof(float)), size_mul(kv_floats, sizeof(float)));
    storage->validate_auxiliary(layer, auxiliary, rows);
    ViewTable t = view_table(layer, placed, true);
    if (!t.storage || !t.rows) return;
    VulkanKVStorage& s = *t.storage;
    s.write_aux(layer, views, n_views, auxiliary);
    const uint32_t pc[4] = {u32(t.rows), u32(s.heads()), u32(s.dim()), u32(kVkBlockTokens)};
    dispatch(kv_variant(K_KV_WRITE, K_KV_WRITE_K16, s),
             {bind(CSlice{s.k_buffer(layer).get(), 0}), bind(CSlice{s.v_buffer(layer).get(), 0}),
              bind(k), bind(v), args(t.words.data(), t.words.size() * sizeof(uint32_t))},
             pc, sizeof(pc), groups(t.rows * hd, 256));
}

// Every view of the batch through the view table: one tiled and one per-row dispatch at most, and a merge when the per-row one splits histories.
inline void VulkanBackend::attention(CSlice Q, size_t layer, const KVView* views, size_t n_views, Slice out,
                                     int n_head, int n_head_kv, int head_dim) {
    if (n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0 || head_dim <= 0 || !attention_head_fits((size_t)head_dim, dev_->caps.subgroup_size))
        throw std::runtime_error("vulkan: invalid attention dimensions");
    std::vector<Placed> placed = place_views(views, n_views);
    if (placed.empty()) return;
    const OpSteps steps(*this);
    // A call's views are causal or not alike; non-causal ones, a vision encoder's, take the per-row kernel's full build over f32 caches.
    const bool full = !placed.front().view->causal;
    for (const Placed& pv : placed) {
        if (!pv.view->nq) throw std::runtime_error("vulkan: invalid attention dimensions");
        if (pv.view->causal == full) throw std::runtime_error("vulkan: causal and non-causal views in one attention call");
    }
    const size_t qstride = (size_t)n_head * head_dim;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const size_t rows = placed.back().row0 + placed.back().view->nq;
    if (floats_from(Q) < rows * qstride || floats_from(out) < rows * qstride)
        throw std::runtime_error("vulkan: attention rows outside their allocation");
    // Non-causal views of heads 64 or 72 wide, a vision encoder's, take the flash kernel: a tile of 64 query rows shares each staged tile of keys (shaders/attention_flash.comp).
    if (full && (head_dim == 64 || head_dim == 72)) {
        ViewTable t = view_table(layer, placed, false);
        VulkanKVStorage& s = *t.storage;
        check_storage(s, n_head_kv, head_dim);
        if (s.k_type() != KVType::f32 || s.v_type() != KVType::f32) throw std::runtime_error("vulkan: non-causal attention over a cache that is not f32");
        // A workgroup's multiply-adds: its 64 rows' dots and weighted sums over every key.
        std::vector<std::pair<size_t, size_t>> runs;
        for (const Placed& pv : placed)
            runs.push_back({(pv.view->nq + kFullTileRows - 1) / kFullTileRows * (size_t)n_head,
                            size_mul(2 * kFullTileRows * (size_t)head_dim, size_add(pv.view->length, pv.view->nq))});
        const std::vector<size_t> cuts = attention_pieces(runs, kFullTileProducts);
        for (size_t i = 0; i + 1 < cuts.size(); ++i) {
            if (cuts.size() > 2) isolate();
            struct { uint32_t n_head, n_head_kv, bt; float scale; uint32_t quant, rows, major, wg0; }
                tc{(uint32_t)n_head, (uint32_t)n_head_kv, u32(kVkBlockTokens), scale, 0u, u32(rows), 0u, u32(cuts[i])};
            dispatch(head_dim == 64 ? K_ATTENTION_FULL_TILE_D64 : K_ATTENTION_FULL_TILE_D72,
                     {bind(Q), bind(out), bind(CSlice{s.k_buffer(layer).get(), 0}), bind(CSlice{s.v_buffer(layer).get(), 0}),
                      args(t.words.data(), t.words.size() * sizeof(uint32_t)), bind(out)},
                     &tc, sizeof(tc), groups(cuts[i + 1] - cuts[i], 1), 1, 0, {0, rows * qstride, 0, 0, 0, 0});
        }
        if (cuts.size() > 2) isolate();
        return;
    }
    // Views of 128- or 256-wide heads whose prompt reaches the profile's attention_tile_rows take the tiled kernel, the rest the per-row kernel: at most two dispatches per layer.
    // The choice is by the view's extent rather than its row count, so a prompt's rows take the same kernel however they were batched.
    std::vector<Placed> wide, narrow;
    for (const Placed& pv : placed) {
        const size_t extent = pv.view->extent ? pv.view->extent : pv.view->nq;
        (!full && extent >= dev_->profile.attention_tile_rows && (head_dim == 128 || head_dim == 256) ? wide : narrow).push_back(pv);
    }
    if (!wide.empty()) {
        ViewTable t = view_table(layer, wide, false);
        VulkanKVStorage& s = *t.storage;
        check_storage(s, n_head_kv, head_dim);
        size_t tiles = 0;
        for (const Placed& pv : wide) tiles += (pv.view->nq + kAttentionTileRows - 1) / kAttentionTileRows;
        // With every view here and the integer-dot tile reading the output next, the output's twin written beside it (shaders/attention_tile.comp).
        std::vector<RowRun> vruns;
        for (const Placed& pv : placed) vruns.push_back(RowRun{pv.row0 + pv.view->nq, pv.view->extent});
        const bool tile = narrow.empty() && tile_reads(qstride, rows, RowRuns{vruns.data(), vruns.size()});
        const uint32_t major = tile ? x8_major(qstride, rows) : 0;
        struct { uint32_t n_head, n_head_kv, bt; float scale; uint32_t quant, rows, major; }
            tc{(uint32_t)n_head, (uint32_t)n_head_kv, u32(kVkBlockTokens), scale, tile ? 1u : 0u, u32(rows), major};
        const size_t slot = tile ? take_twin(rows * qstride) : 0, n = rows * qstride;
        dispatch(head_dim == 256 ? kv_variant(K_ATTENTION_TILE_D256, K_ATTENTION_TILE_D256_K16, s) : kv_variant(K_ATTENTION_TILE, K_ATTENTION_TILE_K16, s),
                 {bind(Q), bind(out), bind(CSlice{s.k_buffer(layer).get(), 0}), bind(CSlice{s.v_buffer(layer).get(), 0}),
                  args(t.words.data(), t.words.size() * sizeof(uint32_t)), tile ? twin_binding(slot) : bind(out)},
                 &tc, sizeof(tc), u32(tiles * (size_t)n_head), 1, tile ? twin_variant() : 0, {0, n, 0, 0, 0, n});
        if (tile) keep_twin(slot, bind(out), n, major);
    }
    if (!narrow.empty()) {
        const VulkanKVStorage& s = storage_of(narrow.front().view->storage);
        check_storage(s, n_head_kv, head_dim);
        if (full && (s.k_type() != KVType::f32 || s.v_type() != KVType::f32)) throw std::runtime_error("vulkan: non-causal attention over a cache that is not f32");
        // The output's 16-bit twin, written by whichever kernel writes the output, when the whole batch is this call and a head is whole blocks.
        const bool quant = wide.empty() && head_dim % 32 == 0;
        const size_t slot = quant ? take_twin(rows * qstride) : 0, n = rows * qstride;
        const VkDescriptorBufferInfo xq = quant ? twin_binding(slot) : bind(out);
        // A KV head's query heads share a workgroup once a view's own history reaches `from`, each head's arithmetic unchanged (shaders/attention_vec.comp).
        // A non-causal view takes one head a workgroup: attention_full has no grouped form.
        const size_t group = (size_t)(n_head / n_head_kv);
        const uint32_t hg = full ? 1u : group % 4 == 0 ? 4u : group % 3 == 0 ? 3u : group % 2 == 0 ? 2u : 1u;
        const size_t from = dev_->profile.attention_group_from;
        std::vector<Placed> sets[2];
        for (const Placed& pv : narrow) sets[hg > 1 && size_add(pv.view->length, pv.view->nq) >= from ? 1 : 0].push_back(pv);
        for (int grouped = 0; grouped < 2; ++grouped)
            if (!sets[grouped].empty()) narrow_attention(Q, layer, sets[grouped], out, n_head, n_head_kv, head_dim, grouped ? hg : 1u, quant, xq, n);
        if (quant) keep_twin(slot, bind(out), n, 0);
    }
}

// The per-row kernel over views of one grouping: splits chosen from each row's history length, doubling the chunk until the profile's split limit fits, and a merge when a row splits.
// A row computes identically beside longer rows, whose extra splits are empty for it.
inline void VulkanBackend::narrow_attention(CSlice Q, size_t layer, const std::vector<Placed>& views, Slice out, int n_head, int n_head_kv, int head_dim, uint32_t hg,
                                            bool quant, VkDescriptorBufferInfo xq, size_t n) {
    ViewTable t = view_table(layer, views, false);
    VulkanKVStorage& s = *t.storage;
    const bool full = !views.front().view->causal;
    size_t longest = 0;
    for (const Placed& pv : views)
        longest = std::max(longest, size_add(pv.view->length, pv.view->nq));
    const size_t pairs = t.rows * (size_t)n_head;
    const DeviceProfile& prof = dev_->profile;
    const size_t chunk = prof.attention_split_chunk;
    const size_t split_max = head_dim == 256 ? prof.attention_split_max_wide : prof.attention_split_max;
    const size_t nsplit = std::min(split_max, std::max<size_t>(1, (longest + chunk - 1) / chunk));
    const size_t scratch_floats = nsplit > 1 ? pairs * nsplit * ((size_t)head_dim + 2) : 0;
    if (scratch_floats && (!scratch_ || scratch_->size() < scratch_floats * sizeof(float)))
        grow(scratch_, scratch_floats * sizeof(float));
    // The full build's `wg0`, past the other builds' constants, which they do not read.
    struct { uint32_t rows, n_head, n_head_kv, dim, bt; float scale; uint32_t nsplit, chunk, quant, max_parts, hg, wg0, xrows; }
        pc{u32(t.rows), (uint32_t)n_head, (uint32_t)n_head_kv, (uint32_t)head_dim, u32(kVkBlockTokens),
           1.0f / std::sqrt((float)head_dim), u32(nsplit), u32(chunk), quant ? 1u : 0u, u32(split_max), hg, 0u, u32(n / ((size_t)n_head * head_dim))};
    const VkDescriptorBufferInfo scratch = scratch_
        ? VkDescriptorBufferInfo{scratch_->handle(), 0, VK_WHOLE_SIZE} : bind(out);
    // Heads 128 wide take the kernel whose 16 lanes read a token's row in one load each, several tokens a subgroup, and heads 256 wide its build of 32 lanes a token (shaders/attention_vec.comp).
    const KernelId kernel = full ? K_ATTENTION_FULL
                          : head_dim == 128 || head_dim == 256 ? vec_attention_kernel(head_dim, hg, s)
                          : (hg > 1 ? kv_variant(K_ATTENTION_G4, K_ATTENTION_K16_G4, s) : kv_variant(K_ATTENTION, K_ATTENTION_K16, s));
    // A non-causal call is cut into dispatches of bounded work by its (row, head) pairs, each pair's splits in one dispatch.
    std::vector<size_t> cuts{0, pairs / hg};
    if (full) {
        std::vector<std::pair<size_t, size_t>> runs;
        for (const Placed& pv : views) runs.push_back({pv.view->nq * (size_t)n_head, size_mul(2 * (size_t)head_dim, size_add(pv.view->length, pv.view->nq))});
        cuts = attention_pieces(runs, kFullRowProducts);
    }
    for (size_t i = 0; i + 1 < cuts.size(); ++i) {
        if (cuts.size() > 2) isolate();
        pc.wg0 = u32(cuts[i] * nsplit);
        dispatch(kernel,
                 {bind(Q), bind(out), bind(CSlice{s.k_buffer(layer).get(), 0}), bind(CSlice{s.v_buffer(layer).get(), 0}),
                  args(t.words.data(), t.words.size() * sizeof(uint32_t)), scratch, xq},
                 &pc, sizeof(pc), groups((cuts[i + 1] - cuts[i]) * nsplit, 1), 1, quant ? twin_variant() : 0, {0, n, 0, 0, 0, scratch_ ? 0 : n, n});
    }
    if (cuts.size() > 2) isolate();
    if (nsplit > 1) {
        const uint32_t mc[6] = {u32(t.rows), (uint32_t)n_head, (uint32_t)head_dim, u32(nsplit), quant ? 1u : 0u, u32(n / ((size_t)n_head * head_dim))};
        dispatch(K_ATTENTION_MERGE, {bind(out), scratch, args(t.words.data(), t.words.size() * sizeof(uint32_t)), xq}, mc, sizeof(mc), groups(pairs, 1), 1, quant ? twin_variant() : 0, {n, 0, 0, n});
    }
}

inline void VulkanBackend::sparse_attention(CSlice query, size_t layer, const KVView* views, size_t n_views,
                                            CSlice ids, size_t id_stride, Slice out, int n_head, int n_head_kv, int head_dim) {
    if (n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv || head_dim <= 0 || !id_stride ||
        id_stride > UINT32_MAX / 2 || !attention_head_fits(size_t(head_dim), dev_->caps.subgroup_size))
        throw std::runtime_error("vulkan: invalid sparse attention dimensions");
    const auto placed = place_views(views, n_views);
    if (placed.empty()) return;
    ViewTable table = view_table(layer, placed, false);
    if (!table.rows) return;
    VulkanKVStorage& storage = *table.storage;
    check_storage(storage, n_head_kv, head_dim);
    const size_t width = size_mul(size_t(n_head), size_t(head_dim));
    const size_t count = size_mul(table.rows, width), selections = size_mul(table.rows, id_stride);
    auto checked = [](CSlice s, size_t elements) {
        if (!s.buffer) throw std::runtime_error("vulkan: operand without storage");
        span(*s.buffer, size_mul(s.offset, sizeof(float)), size_mul(elements, sizeof(float)));
        u32(elements);
    };
    checked(query, count); checked(ids, selections); checked(out, count);
    if (overlaps(out, count, query, count) || overlaps(out, count, ids, selections) ||
        out.buffer == storage.k_buffer(layer).get() || out.buffer == storage.v_buffer(layer).get() ||
        out.buffer == storage.auxiliary_buffer(layer).get())
        throw std::runtime_error("vulkan: sparse attention output overlaps an input");
    u32(storage.k_buffer(layer)->size() / kv_elem_bytes(storage.k_type()));
    u32(storage.v_buffer(layer)->size() / kv_elem_bytes(storage.v_type()));
    const size_t pairs = size_mul(table.rows, size_t(n_head));
    const size_t gx = std::min(pairs, size_t(dev_->props.limits.maxComputeWorkGroupCount[0]));
    const size_t gy = blocks_for(pairs, gx);
    if (gy > dev_->props.limits.maxComputeWorkGroupCount[1])
        throw std::runtime_error("vulkan: sparse attention exceeds the workgroup count limit");
    const KernelId kernel = head_dim == 256 ? kv_variant(K_SPARSE_ATTENTION_VEC_D256, K_SPARSE_ATTENTION_VEC_D256_K16, storage)
                          : head_dim == 128 ? kv_variant(K_SPARSE_ATTENTION_VEC, K_SPARSE_ATTENTION_VEC_K16, storage)
                                           : kv_variant(K_SPARSE_ATTENTION, K_SPARSE_ATTENTION_K16, storage);
    struct { uint32_t rows, heads, kv_heads, dim, bt; float scale; uint32_t splits, chunk, quant, max_parts, hg, stride; }
        pc{u32(table.rows), uint32_t(n_head), uint32_t(n_head_kv), uint32_t(head_dim), u32(kVkBlockTokens),
           1.0f / std::sqrt(float(head_dim)), 1, 1, 0, 1, 1, u32(id_stride)};
    dispatch(kernel, {bind(query), bind(out), bind(CSlice{storage.k_buffer(layer).get(), 0}),
                      bind(CSlice{storage.v_buffer(layer).get(), 0}), args(table.words.data(), table.words.size() * sizeof(uint32_t)),
                      bind(out), bind(out), bind(ids)}, &pc, sizeof(pc), u32(gx), u32(gy));
}

inline void VulkanBackend::select_kv(CSlice query, size_t layer, const KVView* views, size_t n_views, Slice ids,
                                     const IndexArgs& index, CSlice key_norm, CSlice cos, CSlice sin, const GroupPositions& positions) {
    if (!n_views) return;
    if (!index.heads || !index.dim || index.dim > 256 || !index.rope_dim || index.rope_dim % 2 ||
        index.rope_dim > index.dim || !index.compression || !index.budget || index.budget % index.compression ||
        !std::isfinite(index.eps) || index.eps <= 0)
        throw std::runtime_error("vulkan: invalid index selection dimensions");
    const size_t half = index.rope_dim / 2, stride = size_add(index.budget, index.compression - 1);
    const size_t width = size_mul(index.heads, index.dim);
    const auto placed = place_views(views, n_views);
    ViewTable table = view_table(layer, placed, false);
    if (!table.rows) return;
    VulkanKVStorage& storage = *table.storage;
    if (storage.auxiliary_width(layer) != index.dim)
        throw std::runtime_error("vulkan: index selection auxiliary width differs from its key dimension");
    if (storage.auxiliary_pool(layer) != index.compression)
        throw std::runtime_error("vulkan: index selection compression differs from the groups its auxiliary blocks pool");
    size_t longest = 0, max_rows = 0;
    for (size_t i = 0; i < n_views; ++i) {
        const size_t visible = size_add(views[i].length, views[i].nq);
        u32(visible);
        longest = std::max(longest, visible);
        max_rows = std::max(max_rows, views[i].nq);
    }
    const size_t queries = size_mul(table.rows, width), outputs = size_mul(table.rows, stride);
    const size_t rope_values = size_mul(longest, half);
    auto checked = [](CSlice s, size_t count) {
        if (!s.buffer) throw std::runtime_error("vulkan: operand without storage");
        span(*s.buffer, size_mul(s.offset, sizeof(float)), size_mul(count, sizeof(float)));
        u32(count);
    };
    checked(query, queries); checked(ids, outputs); checked(key_norm, index.dim);
    checked(cos, rope_values); checked(sin, rope_values);
    if (overlaps(ids, outputs, query, queries) || overlaps(ids, outputs, key_norm, index.dim) ||
        overlaps(ids, outputs, cos, rope_values) || overlaps(ids, outputs, sin, rope_values))
        throw std::runtime_error("vulkan: index selection output overlaps an input");
    const auto& auxiliary = storage.auxiliary_buffer(layer);
    if (!auxiliary) throw std::runtime_error("vulkan: index selection without auxiliary key storage");
    if (ids.buffer == auxiliary.get() || ids.buffer == storage.k_buffer(layer).get() || ids.buffer == storage.v_buffer(layer).get())
        throw std::runtime_error("vulkan: index selection output overlaps KV storage");
    u32(auxiliary->size() / sizeof(float));
    const uint32_t heads = u32(index.heads), dim = u32(index.dim), compression = u32(index.compression);
    const uint32_t budget = u32(index.budget), output_stride = u32(stride);
    const auto qb = bind(query), ob = bind(ids), wb = bind(key_norm), cb = bind(cos), sb = bind(sin);
    const auto ab = bind(CSlice{auxiliary.get(), 0});
    // Pool the groups a view's rows complete into their pooled rows of the auxiliary blocks, then score at most 16 query rows at once, bounding the score matrix independently of the prompt batch.
    const size_t max_groups = longest / index.compression;
    const size_t score_bytes = size_mul(size_mul(std::min<size_t>(max_rows, 16), max_groups), sizeof(float));
    u32(score_bytes / sizeof(float));
    if (score_bytes && (!scratch_ || scratch_->size() < score_bytes)) grow(scratch_, score_bytes);
    const auto scores = scratch_ ? bind(CSlice{scratch_.get(), 0}) : ob;
    const PlaneShape plane{u32(kVkBlockTokens), u32(storage.auxiliary_block_bytes(layer) / sizeof(float)), u32(storage.auxiliary_rows_bytes(layer) / sizeof(float))};
    size_t at = 0;
    for (size_t vi = 0; vi < n_views; ++vi) {
        const size_t count = (views[vi].length + views[vi].nq) / index.compression;
        const size_t at_view = at;
        at += (count - views[vi].length / index.compression) * positions.streams.count;
        if (!views[vi].nq) continue;
        pool_view(ab, plane, wb, cb, sb, table.words, u32(vi), views[vi], index, positions, at_view);
        for (size_t row = 0; row < views[vi].nq; row += 16) {
            const size_t rows = std::min<size_t>(16, views[vi].nq - row);
            score_rows(qb, ab, plane, scores, table.words, u32(vi), row, rows, count, heads, dim, compression, 1.0f / std::sqrt(float(index.dim)));
            select_rows(scores, ob, table.words, u32(vi), row, rows, count, compression, budget, output_stride, count >= dev_->profile.select_wide_from);
        }
    }
}

inline std::vector<VulkanBackend::Placed> VulkanBackend::place_views(const KVView* views, size_t n_views) {
    if (n_views && !views) throw std::runtime_error("vulkan: cache op without views");
    std::vector<Placed> placed;
    size_t row0 = 0;
    for (size_t i = 0; i < n_views; ++i) {
        placed.push_back(Placed{&views[i], row0});
        row0 = size_add(row0, views[i].nq);
    }
    return placed;
}

inline void VulkanBackend::pool_kv(size_t layer, const KVView* views, size_t n_views, const IndexArgs& index, CSlice key_norm, CSlice cos, CSlice sin,
                                   const GroupPositions& positions) {
    if (!n_views) return;
    if (!index.dim || index.dim > 256 || !index.rope_dim || index.rope_dim % 2 || index.rope_dim > index.dim || !index.compression || !std::isfinite(index.eps) ||
        index.eps <= 0)
        throw std::runtime_error("vulkan: invalid index pooling dimensions");
    const auto placed = place_views(views, n_views);
    ViewTable table = view_table(layer, placed, false);
    if (!table.rows) return;
    VulkanKVStorage& storage = *table.storage;
    if (storage.auxiliary_width(layer) != index.dim || storage.auxiliary_pool(layer) != index.compression)
        throw std::runtime_error("vulkan: index pooling differs from the rows and groups its auxiliary blocks keep");
    const auto& auxiliary = storage.auxiliary_buffer(layer);
    if (!auxiliary) throw std::runtime_error("vulkan: index pooling without auxiliary key storage");
    size_t longest = 0;
    for (size_t i = 0; i < n_views; ++i) longest = std::max(longest, size_add(views[i].length, views[i].nq));
    const size_t rope_values = size_mul(longest, index.rope_dim / 2);
    for (const auto& [s, count] : {std::pair<CSlice, size_t>{key_norm, index.dim}, {cos, rope_values}, {sin, rope_values}}) {
        if (!s.buffer) throw std::runtime_error("vulkan: operand without storage");
        span(*s.buffer, size_mul(s.offset, sizeof(float)), size_mul(count, sizeof(float)));
    }
    const PlaneShape plane{u32(kVkBlockTokens), u32(storage.auxiliary_block_bytes(layer) / sizeof(float)), u32(storage.auxiliary_rows_bytes(layer) / sizeof(float))};
    const auto ab = bind(CSlice{auxiliary.get(), 0}), wb = bind(key_norm), cb = bind(cos), sb = bind(sin);
    for (size_t vi = 0, at = 0; vi < n_views; ++vi) {
        pool_view(ab, plane, wb, cb, sb, table.words, u32(vi), views[vi], index, positions, at);
        at += ((views[vi].length + views[vi].nq) / index.compression - views[vi].length / index.compression) * positions.streams.count;
    }
}

inline void VulkanBackend::pool_view(VkDescriptorBufferInfo aux, PlaneShape plane, VkDescriptorBufferInfo key_norm, VkDescriptorBufferInfo cos, VkDescriptorBufferInfo sin,
                                     const std::vector<uint32_t>& table, uint32_t view, const KVView& v, const IndexArgs& index, const GroupPositions& positions, size_t at) {
    const size_t done = v.length / index.compression, count = (v.length + v.nq) / index.compression;
    if (count <= done) return;
    const size_t limit = dev_->props.limits.maxComputeWorkGroupCount[0], half = index.rope_dim / 2;
    const uint32_t streams = positions.pos ? u32(positions.streams.count) : 0u;
    if (positions.pos && (!streams || (streams > 1 && !positions.streams.of))) throw std::runtime_error("vulkan: invalid group position streams");
    const uint32_t none = 0;
    std::vector<uint32_t> of(half, 0);
    for (size_t i = 0; streams > 1 && i < half; ++i) of[i] = positions.streams.of[i];
    struct { uint32_t view, first, dim, half, compression, bt, block_floats, rows_floats; float eps; uint32_t streams, base; }
        pc{view, 0, u32(index.dim), u32(half), u32(index.compression), plane.bt, plane.block_floats, plane.rows_floats, index.eps, streams, u32(done)};
    for (size_t first = done; first < count; first += limit) {
        pc.first = u32(first);
        dispatch(K_KV_INDEX_POOL,
                 {aux, key_norm, cos, sin, args(table.data(), table.size() * sizeof(uint32_t)),
                  streams ? args(positions.pos + at, (count - done) * streams * sizeof(uint32_t)) : args(&none, sizeof(none)),
                  args(of.data(), of.size() * sizeof(uint32_t))},
                 &pc, sizeof(pc), u32(std::min(count - first, limit)));
    }
    if (unordered_) barrier(open());
}

inline void VulkanBackend::score_rows(VkDescriptorBufferInfo query, VkDescriptorBufferInfo pooled, PlaneShape plane, VkDescriptorBufferInfo scores, const std::vector<uint32_t>& table,
                                      uint32_t view, size_t row, size_t rows, size_t groups, uint32_t heads, uint32_t dim, uint32_t compression, float scale) {
    const size_t limit = dev_->props.limits.maxComputeWorkGroupCount[0];
    struct { uint32_t view, row, first, groups, heads, dim, compression; float scale; uint32_t tile, bt, block_floats, rows_floats; }
        sc{view, u32(row), 0, u32(groups), heads, dim, compression, scale, u32(kScoreTile), plane.bt, plane.block_floats, plane.rows_floats};
    const size_t tiles = (groups + kScoreTile - 1) / kScoreTile;
    for (size_t first = 0; first < tiles; first += limit) {
        sc.first = u32(first);
        dispatch(K_KV_INDEX_SCORES_TILE, {query, pooled, scores, args(table.data(), table.size() * sizeof(uint32_t))}, &sc, sizeof(sc), u32(std::min(tiles - first, limit)), u32(rows));
    }
    if (unordered_) barrier(open());
}

inline void VulkanBackend::select_rows(VkDescriptorBufferInfo scores, VkDescriptorBufferInfo ids, const std::vector<uint32_t>& table, uint32_t view, size_t row, size_t rows,
                                       size_t groups, uint32_t compression, uint32_t budget, uint32_t stride, bool wide) {
    if (!wide) {
        const uint32_t tc[] = {view, u32(row), u32(groups), compression, budget, stride};
        dispatch(K_KV_INDEX_SELECT, {scores, ids, args(table.data(), table.size() * sizeof(uint32_t))}, tc, sizeof(tc), u32(rows));
        if (unordered_) barrier(open());
        return;
    }
    if (rows > kSelectRows) throw std::runtime_error("vulkan: a wide selection of more rows than its state holds");
    const size_t slices = (groups + kSelectSlice - 1) / kSelectSlice;
    const size_t words = size_add(kSelectRows * kSelectRowWords, size_mul(size_mul(kSelectRows, slices), 2));
    if (!select_state_ || select_state_->size() < words * sizeof(uint32_t)) {
        // Zeroed once: the histograms are zero between calls, since each call's emission clears them.
        grow(select_state_, words * sizeof(uint32_t));
        VkCommandBuffer cmd = open();
        dev_->fn.vkCmdFillBuffer(cmd, select_state_->handle(), 0, VK_WHOLE_SIZE, 0);
        barrier(cmd);
    }
    const VkDescriptorBufferInfo state = bind(CSlice{select_state_.get(), 0});
    struct { uint32_t view, row, groups, compression, budget, stride, stage, slices; } pc{view, u32(row), u32(groups), compression, budget, stride, 0, u32(slices)};
    for (uint32_t s = 0; s < 5; ++s) {
        pc.stage = s;
        dispatch(K_KV_INDEX_SELECT_WIDE, {scores, ids, args(table.data(), table.size() * sizeof(uint32_t)), state}, &pc, sizeof(pc), u32(slices), u32(rows));
        if (unordered_) barrier(open());
    }
}

inline VulkanBackend::ViewTable VulkanBackend::view_table(size_t layer, const std::vector<Placed>& placed, bool writing) {
    ViewTable t;
    t.words.push_back(u32(placed.size()));
    std::vector<uint32_t> blocks;
    size_t local = 0;
    for (const Placed& pv : placed) {
        const KVView& view = *pv.view;
        VulkanKVStorage& s = storage_of(view.storage);
        if (t.storage && t.storage != &s) throw std::runtime_error("vulkan: views of two storages in one call");
        t.storage = &s;
        const size_t used = s.check_view(view, layer, writing);
        t.words.push_back(u32(pv.row0));
        t.words.push_back(u32(local));
        t.words.push_back(u32(view.nq));
        t.words.push_back(u32(view.length));
        t.words.push_back(u32(blocks.size()));
        t.words.push_back(u32(used));
        for (size_t b = 0; b < used; ++b) blocks.push_back((uint32_t)view.blocks[b]);
        local = size_add(local, view.nq);
    }
    t.words.insert(t.words.end(), blocks.begin(), blocks.end());
    t.rows = local;
    return t;
}

// The heads and width attention is asked for against the storage's; the view table has checked the layer.
inline void VulkanBackend::check_storage(const VulkanKVStorage& s, int n_head_kv, int head_dim) {
    if ((size_t)head_dim != s.dim() || (size_t)n_head_kv != s.heads())
        throw std::runtime_error("vulkan: attention outside the KV view");
}

// Buffers a storage growth copy reads from, kept until the command buffer that recorded the copy retires.
inline void VulkanBackend::keep_until_retired(BufferPtr b) {
    open();
    pending_[ring_index_].push_back(std::dynamic_pointer_cast<VulkanBuffer>(b));
}

inline VulkanKVStorage& VulkanBackend::storage_of(KVStorage* storage) {
    auto* s = dynamic_cast<VulkanKVStorage*>(storage);
    if (!s) throw std::runtime_error("vulkan: KV storage of another backend");
    return *s;
}

inline KernelId kv_variant(KernelId f32, KernelId k16, const VulkanKVStorage& s) {
    const int i = (s.k_type() == KVType::f16 ? 1 : 0) + (s.v_type() == KVType::f16 ? 2 : 0);
    return i == 0 ? f32 : (KernelId)((int)k16 + i - 1);
}

inline KernelId vec_attention_kernel(int head_dim, uint32_t hg, const VulkanKVStorage& s) {
    const bool w = head_dim == 256;
    switch (hg) {
    case 2: return w ? kv_variant(K_ATTENTION_VEC_D256_G2, K_ATTENTION_VEC_D256_K16_G2, s) : kv_variant(K_ATTENTION_VEC_G2, K_ATTENTION_VEC_K16_G2, s);
    case 3: return w ? kv_variant(K_ATTENTION_VEC_D256_G3, K_ATTENTION_VEC_D256_K16_G3, s) : kv_variant(K_ATTENTION_VEC_G3, K_ATTENTION_VEC_K16_G3, s);
    case 4: return w ? kv_variant(K_ATTENTION_VEC_D256_G4, K_ATTENTION_VEC_D256_K16_G4, s) : kv_variant(K_ATTENTION_VEC_G4, K_ATTENTION_VEC_K16_G4, s);
    default: return w ? kv_variant(K_ATTENTION_VEC_D256, K_ATTENTION_VEC_D256_K16, s) : kv_variant(K_ATTENTION_VEC, K_ATTENTION_VEC_K16, s);
    }
}

VulkanKVStorage::VulkanKVStorage(VulkanBackend& owner, size_t layers, size_t heads, size_t dim, size_t max_tokens,
                                 KVType kt, KVType vt, const std::vector<size_t>& auxiliary_widths, const std::vector<size_t>& auxiliary_pools)
    : BlockKVStorage(owner, "vulkan", kVkBlockTokens, layers, heads, dim, max_tokens, kt, vt, auxiliary_widths, auxiliary_pools) {}

void VulkanKVStorage::retire(const BufferPtr& old) {
    static_cast<VulkanBackend&>(owner()).keep_until_retired(old);
}

} // namespace
} // namespace backend
