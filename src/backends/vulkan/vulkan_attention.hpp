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

// The columns of a row a sparse_columns workgroup takes, its COLUMNS (shaders/sparse_columns.comp).
constexpr size_t kSparseColumns = 8;

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
    const size_t dim = 2 * rope.half, hd = n_head_kv * dim;
    if (!s.holds(layer, n_head_kv, dim) || t.rows != rows)
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
                                                          const std::vector<size_t>& auxiliary_widths, const std::vector<AuxiliaryPool>& auxiliary_pools,
                                                          const std::vector<KVShape>& layer_shapes) {
    const std::vector<KVShape> shapes = BlockKVStorage::shapes("vulkan", layers, n_head_kv, head_dim, layer_shapes);
    if (shapes.empty()) throw std::runtime_error("vulkan: KV storage without layers, or heads without width");
    for (const KVShape& shape : shapes) {
        if (shape.heads && !shape.dim) throw std::runtime_error("vulkan: KV storage without layers, or heads without width");
        if (shape.heads && !attention_head_fits(shape.dim, dev_->caps.subgroup_size))
            throw std::runtime_error("vulkan: a head " + std::to_string(shape.dim) + " wide is not supported on this device, whose attention takes heads 128 or 256 wide or at most " +
                                     std::to_string(attention_row_width(dev_->caps.subgroup_size)));
    }
    return std::make_unique<VulkanKVStorage>(*this, shapes, max_tokens, k_type, v_type, auxiliary_widths, auxiliary_pools);
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
    const size_t hd = size_mul(storage->heads(layer), storage->dim(layer));
    const size_t kv_floats = size_mul(rows, hd);
    if (!k.buffer || !v.buffer) throw std::runtime_error("vulkan: operand without storage");
    span(*k.buffer, size_mul(k.offset, sizeof(float)), size_mul(kv_floats, sizeof(float)));
    span(*v.buffer, size_mul(v.offset, sizeof(float)), size_mul(kv_floats, sizeof(float)));
    storage->validate_auxiliary(layer, auxiliary, rows);
    ViewTable t = view_table(layer, placed, true);
    if (!t.storage || !t.rows) return;
    VulkanKVStorage& s = *t.storage;
    s.write_aux(layer, views, n_views, auxiliary);
    const uint32_t pc[4] = {u32(t.rows), u32(s.heads(layer)), u32(s.dim(layer)), u32(kVkBlockTokens)};
    dispatch(kv_variant(K_KV_WRITE, K_KV_WRITE_K16, s),
             {bind(CSlice{s.k_buffer(layer).get(), 0}), bind(CSlice{s.v_buffer(layer).get(), 0}),
              bind(k), bind(v), args(t.words.data(), t.words.size() * sizeof(uint32_t))},
             pc, sizeof(pc), groups(t.rows * hd, 256));
}

// Every view of the batch through the view table: one tiled dispatch at most, and a per-row dispatch for each grouping the views take, each with a merge when it splits histories.
inline void VulkanBackend::attention(CSlice Q, size_t layer, const KVView* views, size_t n_views, Slice out,
                                     int n_head, int n_head_kv, int head_dim, const AttentionTerms* terms) {
    if (terms) throw std::runtime_error("vulkan: attention terms (inkling-core step 3c)");
    if (n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv != 0 || head_dim <= 0 || !attention_head_fits((size_t)head_dim, dev_->caps.subgroup_size))
        throw std::runtime_error("vulkan: invalid attention dimensions");
    std::vector<Placed> placed = place_views(views, n_views);
    if (placed.empty()) return;
    const OpSteps steps(*this);
    // A call's views are causal or not alike; non-causal ones without a window, a vision encoder's, take the full builds over f32 sides, which have no f16 form.
    // Non-causal views over a window, a DFlash2 drafter's, take the window builds read through the block's last row, whatever their sides, as do a DFlash drafter's unwindowed ones over f16 under a window past every position.
    const bool full = !placed.front().view->causal;
    const VulkanKVStorage& sides = storage_of(placed.front().view->storage);
    const bool f32_sides = sides.k_type() == KVType::f32 && sides.v_type() == KVType::f32;
    bool windowed = false;
    for (const Placed& pv : placed) {
        if (!pv.view->nq) throw std::runtime_error("vulkan: invalid attention dimensions");
        if (pv.view->causal == full) throw std::runtime_error("vulkan: causal and non-causal views in one attention call");
        windowed = windowed || pv.view->window;
    }
    // A non-causal call is windowed in every view or none, so a view's build does not follow the batch beside it.
    for (const Placed& pv : placed)
        if (full && !pv.view->window == windowed) throw std::runtime_error("vulkan: windowed and unwindowed non-causal views in one attention call");
    const size_t qstride = (size_t)n_head * head_dim;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const size_t rows = placed.back().row0 + placed.back().view->nq;
    if (floats_from(Q) < rows * qstride || floats_from(out) < rows * qstride)
        throw std::runtime_error("vulkan: attention rows outside their allocation");
    // Non-causal views of heads 64 or 72 wide, a vision encoder's, take the flash kernel: a tile of 64 query rows shares each staged tile of keys (shaders/attention_flash.comp).
    if (full && !windowed && f32_sides && (head_dim == 64 || head_dim == 72)) {
        ViewTable t = view_table(layer, placed, false);
        VulkanKVStorage& s = *t.storage;
        check_storage(s, layer, n_head_kv, head_dim);
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
    // Views of 128- or 256-wide heads whose prompt reaches the profile's attention_tile_rows take the tiled kernel, the rest the per-row kernel, once for the views below attention_group_from and once for those at or past it.
    // The choice is by the view's extent rather than its row count, so a prompt's rows take the same kernel however they were batched.
    std::vector<Placed> wide, narrow;
    for (const Placed& pv : placed) {
        const size_t extent = pv.view->extent ? pv.view->extent : pv.view->nq;
        (!full && !pv.view->window && extent >= dev_->profile.attention_tile_rows && (head_dim == 128 || head_dim == 256) ? wide : narrow).push_back(pv);
    }
    if (!wide.empty()) {
        // With every view here and the integer-dot tile reading the output next, the output's twin written beside it (shaders/attention_tile.comp).
        std::vector<RowRun> vruns;
        for (const Placed& pv : placed) vruns.push_back(RowRun{pv.row0 + pv.view->nq, pv.view->extent});
        const bool tile = narrow.empty() && tile_reads(qstride, rows, RowRuns{vruns.data(), vruns.size()});
        const uint32_t major = tile ? x8_major(qstride, rows) : 0;
        const size_t slot = tile ? take_twin(rows * qstride) : 0, n = rows * qstride;
        wide_attention(Q, layer, wide, out, n_head, n_head_kv, head_dim, tile ? twin_binding(slot) : bind(out), tile, major, rows);
        if (tile) keep_twin(slot, bind(out), n, major);
    }
    if (!narrow.empty()) {
        const VulkanKVStorage& s = storage_of(narrow.front().view->storage);
        check_storage(s, layer, n_head_kv, head_dim);
        // The output's 16-bit twin, written by whichever kernel writes the output, when the whole batch is this call and a head is whole blocks.
        const bool quant = wide.empty() && head_dim % 32 == 0;
        const size_t slot = quant ? take_twin(rows * qstride) : 0, n = rows * qstride;
        const VkDescriptorBufferInfo xq = quant ? twin_binding(slot) : bind(out);
        // A KV head's query heads share a workgroup once a view's own history reaches the profile's attention_group_from, each head's arithmetic unchanged (shaders/attention_vec.comp).
        // The choice follows the view alone, so a row takes the same grouping beside any batch; a non-causal view takes one head a workgroup, attention_full having no grouped form.
        const size_t group = (size_t)(n_head / n_head_kv);
        const uint32_t hg = full ? 1u : group % 4 == 0 ? 4u : group % 3 == 0 ? 3u : group % 2 == 0 ? 2u : 1u;
        // A view whose window starts a row past position 0, and every non-causal view over a window, takes the window builds, one head a workgroup, a dispatch for each window; its other views run the builds they ran before.
        std::vector<Placed> sets[2];
        std::map<size_t, std::vector<Placed>> windows;
        size_t unwindowed = 0;
        for (const Placed& pv : narrow) {
            const size_t end = size_add(pv.view->length, pv.view->nq);
            if (pv.view->window && (end > pv.view->window || full)) windows[pv.view->window].push_back(pv);
            else if (full && !f32_sides) {
                windows[UINT32_MAX].push_back(pv);
                unwindowed = size_add(unwindowed, size_mul(pv.view->nq * (size_t)n_head, size_mul(2 * (size_t)head_dim, end)));
            } else sets[hg > 1 && end >= dev_->profile.attention_group_from ? 1 : 0].push_back(pv);
        }
        // The window builds run uncut, so a vision encoder's long prompt over f16 sides is refused rather than left to outrun a job's time limit.
        if (unwindowed > kFullRowProducts) throw std::runtime_error("vulkan: non-causal attention this long over f16 sides; run with --cache-type-k f32 --cache-type-v f32");
        for (int grouped = 0; grouped < 2; ++grouped)
            if (!sets[grouped].empty()) narrow_attention(Q, layer, sets[grouped], out, n_head, n_head_kv, head_dim, grouped ? hg : 1u, xq, quant, rows);
        for (const auto& [window, set] : windows) narrow_attention(Q, layer, set, out, n_head, n_head_kv, head_dim, 1u, xq, quant, rows, window);
        if (quant) keep_twin(slot, bind(out), n, 0);
    }
}

// The per-row kernel over views taken `hg` query heads a workgroup: splits chosen from each row's history length, doubling the chunk until the profile's split limit fits, and a merge when a row splits.
// A row computes identically beside longer rows, whose extra splits are empty for it; `rows` is the call's, whose twin `xq` holds where `quant`.
inline void VulkanBackend::narrow_attention(CSlice Q, size_t layer, const std::vector<Placed>& views, Slice out, int n_head, int n_head_kv, int head_dim, uint32_t hg,
                                            VkDescriptorBufferInfo xq, bool quant, size_t rows, size_t window) {
    ViewTable t = view_table(layer, views, false);
    VulkanKVStorage& s = *t.storage;
    // A non-causal call over a window runs the window builds, its rows read through the block's last row (`open`).
    const bool open = !views.front().view->causal && window, full = !views.front().view->causal && !window;
    const size_t qstride = (size_t)n_head * head_dim, n = rows * qstride;
    // A windowed row's splits cover its span alone, which an open row's block rows past it lengthen.
    size_t longest = 0;
    for (const Placed& pv : views) {
        const size_t end = size_add(pv.view->length, pv.view->nq);
        longest = std::max(longest, window ? std::min(open ? window + pv.view->nq - 1 : window, end) : end);
    }
    const size_t pairs = t.rows * (size_t)n_head;
    const DeviceProfile& prof = dev_->profile;
    const size_t chunk = prof.attention_split_chunk;
    const size_t split_max = head_dim == 256 ? prof.attention_split_max_wide : prof.attention_split_max;
    const size_t nsplit = std::min(split_max, std::max<size_t>(1, (longest + chunk - 1) / chunk));
    const size_t scratch_floats = nsplit > 1 ? pairs * nsplit * ((size_t)head_dim + 2) : 0;
    if (scratch_floats && (!scratch_ || scratch_->size() < scratch_floats * sizeof(float)))
        grow(scratch_, scratch_floats * sizeof(float));
    // The full build's `wg0` and every causal build's `xrows`, past the sparse builds' constants, which these builds do not have.
    // The window builds' `open` follows, past what the other builds declare, which the fixed push range allows.
    struct { uint32_t rows, n_head, n_head_kv, dim, bt; float scale; uint32_t nsplit, chunk, quant, max_parts, hg, wg0, xrows, open; }
        pc{u32(t.rows), (uint32_t)n_head, (uint32_t)n_head_kv, (uint32_t)head_dim, u32(kVkBlockTokens),
           1.0f / std::sqrt((float)head_dim), u32(nsplit), u32(chunk), quant ? 1u : 0u, u32(split_max), hg, 0u, u32(rows), open ? 1u : 0u};
    const VkDescriptorBufferInfo scratch = scratch_
        ? VkDescriptorBufferInfo{scratch_->handle(), 0, VK_WHOLE_SIZE} : bind(out);
    // Heads 128 wide take the kernel whose 16 lanes read a token's row in one load each, several tokens a subgroup, and heads 256 wide its build of 32 lanes a token (shaders/attention_vec.comp).
    const KernelId kernel = full ? K_ATTENTION_FULL
                          : window ? (head_dim == 256 ? kv_variant(K_ATTENTION_VEC_D256_WINDOW, K_ATTENTION_VEC_D256_K16_WINDOW, s)
                                      : head_dim == 128 ? kv_variant(K_ATTENTION_VEC_WINDOW, K_ATTENTION_VEC_K16_WINDOW, s)
                                      : kv_variant(K_ATTENTION_WINDOW, K_ATTENTION_K16_WINDOW, s))
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
        // The window builds read the window in wg0's place.
        pc.wg0 = window ? uint32_t(std::min<size_t>(window, UINT32_MAX)) : u32(cuts[i] * nsplit);
        dispatch(kernel,
                 {bind(Q), bind(out), bind(CSlice{s.k_buffer(layer).get(), 0}), bind(CSlice{s.v_buffer(layer).get(), 0}),
                  args(t.words.data(), t.words.size() * sizeof(uint32_t)), scratch, xq},
                 &pc, sizeof(pc), groups((cuts[i + 1] - cuts[i]) * nsplit, 1), 1, quant ? twin_variant() : 0, {0, n, 0, 0, 0, scratch_ ? 0 : n, n});
    }
    if (cuts.size() > 2) isolate();
    if (nsplit > 1) {
        const uint32_t mc[7] = {u32(t.rows), (uint32_t)n_head, (uint32_t)head_dim, u32(nsplit), quant ? 1u : 0u, u32(rows), 0u};
        dispatch(K_ATTENTION_MERGE, {bind(out), scratch, args(t.words.data(), t.words.size() * sizeof(uint32_t)), xq}, mc, sizeof(mc), groups(pairs, 1), 1, quant ? twin_variant() : 0, {n, 0, 0, n});
    }
}

// The tiled kernel over the wide views, its history split into parts of attention_tile_part tokens and merged once a row passes one part.
// Parts start at absolute positions, so a row's parts follow its own history (docs/src/backends-vulkan.md).
inline void VulkanBackend::wide_attention(CSlice Q, size_t layer, const std::vector<Placed>& views, Slice out, int n_head, int n_head_kv, int head_dim,
                                          VkDescriptorBufferInfo xq, bool quant, uint32_t major, size_t rows) {
    const size_t qstride = (size_t)n_head * head_dim, n = rows * qstride, stride = (size_t)head_dim + 2;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const size_t part = dev_->profile.attention_tile_part;
    if (part == 0 || part % 16 != 0) throw std::logic_error("vulkan: attention_tile_part is not a whole number of key tiles");
    const size_t limit = dev_->props.limits.maxComputeWorkGroupCount[0];
    size_t longest = 0;
    for (const Placed& pv : views) longest = std::max(longest, size_add(pv.view->length, pv.view->nq));
    const size_t nsplit = (longest + part - 1) / part;
    const VulkanKVStorage& s = storage_of(views.front().view->storage);
    const KernelId kernel = head_dim == 256 ? kv_variant(K_ATTENTION_TILE_D256, K_ATTENTION_TILE_D256_K16, s) : kv_variant(K_ATTENTION_TILE, K_ATTENTION_TILE_K16, s);
    // Two rows a lane, a workgroup of two 32-row tiles, once the paired workgroups of every part still give every compute unit one; below that the call is too small to fill the device and halving its workgroups slows it.
    // A row's bits are the same either way (shaders/attention_tile.comp).
    size_t pairs = 0;
    for (const Placed& pv : views) pairs += (pv.view->nq + kAttentionTileRows - 1) / kAttentionTileRows;
    const bool pair = head_dim == 128 && size_mul(pairs * (size_t)n_head, nsplit) >= dev_->caps.compute_units;
    const size_t wg_rows = pair ? kAttentionTileRows : kAttentionTileRowsOne;
    struct { uint32_t n_head, n_head_kv, bt; float scale; uint32_t quant, rows, major, pair, chunk, nsplit, wg0; }
        tc{(uint32_t)n_head, (uint32_t)n_head_kv, u32(kVkBlockTokens), scale, quant ? 1u : 0u, u32(rows), major, pair ? 1u : 0u, u32(part), 1u, 0u};
    // The workgroups of `pieces`, whole or in `parts` parts each, in dispatches within the device's workgroup count.
    auto tiles_of = [&](const std::vector<Placed>& pieces, const ViewTable& t, VkDescriptorBufferInfo scratch, size_t scratch_floats, uint32_t parts) {
        size_t tiles = 0;
        for (const Placed& pv : pieces) tiles += (pv.view->nq + wg_rows - 1) / wg_rows;
        const size_t total = size_mul(tiles * (size_t)n_head, parts);
        tc.nsplit = parts;
        for (size_t g0 = 0; g0 < total; g0 += limit) {
            tc.wg0 = u32(g0);
            dispatch(kernel,
                     {bind(Q), bind(out), bind(CSlice{t.storage->k_buffer(layer).get(), 0}), bind(CSlice{t.storage->v_buffer(layer).get(), 0}),
                      args(t.words.data(), t.words.size() * sizeof(uint32_t)), xq, scratch},
                     &tc, sizeof(tc), u32(std::min(limit, total - g0)), 1, quant ? twin_variant() : 0, {0, n, 0, 0, 0, n, scratch_floats});
        }
    };
    if (nsplit <= 1) {
        ViewTable t = view_table(layer, views, false);
        check_storage(*t.storage, layer, n_head_kv, head_dim);
        tiles_of(views, t, bind(out), n, 1);
        return;
    }
    // Pieces of whole tiles whose parts fit the cap; a view cut at a tile boundary goes on as a view whose history holds the rows before the cut.
    const size_t per_row = size_mul(size_mul((size_t)n_head, nsplit), stride * sizeof(float));
    const size_t cap_rows = std::max(kAttentionTileRowsOne, attention_scratch_cap_ / per_row / kAttentionTileRowsOne * kAttentionTileRowsOne);
    std::vector<KVView> cut;
    size_t cuts = 0;
    for (const Placed& pv : views) cuts += pv.view->nq / kAttentionTileRowsOne + 1;
    cut.reserve(cuts);
    std::vector<std::vector<Placed>> pieces(1);
    size_t held = 0;
    for (const Placed& pv : views)
        for (size_t b0 = 0; b0 < pv.view->nq;) {
            if (held == cap_rows) { pieces.emplace_back(); held = 0; }
            const size_t take = std::min(pv.view->nq - b0, cap_rows - held);
            KVView v = *pv.view;
            v.length += b0;
            v.nq = take;
            cut.push_back(v);
            pieces.back().push_back(Placed{&cut.back(), pv.row0 + b0});
            held += (take + kAttentionTileRowsOne - 1) / kAttentionTileRowsOne * kAttentionTileRowsOne;
            b0 += take;
        }
    for (const std::vector<Placed>& piece : pieces) {
        ViewTable t = view_table(layer, piece, false);
        check_storage(*t.storage, layer, n_head_kv, head_dim);
        const size_t scratch_floats = size_mul(size_mul(t.rows, (size_t)n_head), nsplit * stride);
        if (!scratch_ || scratch_->size() < scratch_floats * sizeof(float)) grow(scratch_, scratch_floats * sizeof(float));
        const VkDescriptorBufferInfo scratch{scratch_->handle(), 0, VK_WHOLE_SIZE};
        tiles_of(piece, t, scratch, 0, u32(nsplit));
        const uint32_t mc[7] = {u32(t.rows), (uint32_t)n_head, (uint32_t)head_dim, u32(nsplit), quant ? 1u : 0u, u32(rows), major};
        dispatch(K_ATTENTION_MERGE, {bind(out), scratch, args(t.words.data(), t.words.size() * sizeof(uint32_t)), xq}, mc, sizeof(mc),
                 groups(t.rows * (size_t)n_head, 1), 1, quant ? twin_variant() : 0, {n, 0, 0, n});
    }
}

inline void VulkanBackend::sparse_attention(CSlice query, size_t layer, const KVView* views, size_t n_views,
                                            CSlice ids, size_t id_stride, Slice out, int n_head, int n_head_kv, int head_dim, size_t groups, size_t block_size) {
    // A block id names a power of two of positions, so a slot splits by shift and mask (shaders/sparse_slots.glsl).
    if (!block_size || (block_size & (block_size - 1))) throw std::runtime_error("vulkan: sparse attention blocks of a size that is not a power of two");
    uint32_t shift = 0;
    while ((size_t(1) << shift) < block_size) ++shift;
    if (n_head <= 0 || n_head_kv <= 0 || n_head % n_head_kv || head_dim <= 0 || !id_stride || !groups || size_t(n_head_kv) % groups ||
        shift > 16 || id_stride > (UINT32_MAX / 2) >> shift || !attention_head_fits(size_t(head_dim), dev_->caps.subgroup_size))
        throw std::runtime_error("vulkan: invalid sparse attention dimensions");
    const auto placed = place_views(views, n_views);
    if (placed.empty()) return;
    ViewTable table = view_table(layer, placed, false);
    if (!table.rows) return;
    VulkanKVStorage& storage = *table.storage;
    check_storage(storage, layer, n_head_kv, head_dim);
    const size_t width = size_mul(size_t(n_head), size_t(head_dim));
    const size_t count = size_mul(table.rows, width), selections = size_mul(size_mul(table.rows, groups), id_stride);
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
    // Rows enough to give every compute unit a workgroup group the query heads sharing a KV head, each head's arithmetic and token order unchanged (shaders/attention_vec.comp); fewer keep one head a workgroup, filling the device.
    const size_t group = size_t(n_head / n_head_kv);
    uint32_t hg = head_dim != 256 ? 1u : group % 4 == 0 ? 4u : group % 3 == 0 ? 3u : group % 2 == 0 ? 2u : 1u;
    if (size_mul(table.rows, size_t(n_head)) / hg < size_t(dev_->caps.compute_units)) hg = 1;
    const size_t pairs = size_mul(table.rows, size_t(n_head)) / hg;
    const size_t gx = std::min(pairs, size_t(dev_->props.limits.maxComputeWorkGroupCount[0]));
    const size_t gy = blocks_for(pairs, gx);
    if (gy > dev_->props.limits.maxComputeWorkGroupCount[1])
        throw std::runtime_error("vulkan: sparse attention exceeds the workgroup count limit");
    const KernelId kernel = head_dim == 256 ? sparse_attention_kernel(hg, storage)
                          : head_dim == 128 ? kv_variant(K_SPARSE_ATTENTION_VEC, K_SPARSE_ATTENTION_VEC_K16, storage)
                                           : kv_variant(K_SPARSE_ATTENTION, K_SPARSE_ATTENTION_K16, storage);
    struct { uint32_t rows, heads, kv_heads, dim, bt; float scale; uint32_t splits, chunk, quant, max_parts, hg, stride, id_shift, id_groups; }
        pc{u32(table.rows), uint32_t(n_head), uint32_t(n_head_kv), uint32_t(head_dim), u32(kVkBlockTokens),
           1.0f / std::sqrt(float(head_dim)), 1, 1, 0, 1, hg, u32(id_stride), shift, u32(groups)};
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

// The pooled rows a call's views read or write, and the storage's plane shape for them; every view is of one storage whose `layer` keeps rows at least `dim` wide.
inline VulkanBackend::PooledPlane VulkanBackend::pooled_plane(size_t layer, const KVView* views, size_t n_views, size_t dim, bool writing, const char* op) {
    PooledPlane out;
    out.table = view_table(layer, place_views(views, n_views), writing);
    VulkanKVStorage& s = *out.table.storage;
    if (!s.auxiliary_pool(layer) || s.auxiliary_pool_width(layer) < dim || !s.auxiliary_buffer(layer))
        throw std::runtime_error(std::string("vulkan: ") + op + " reads pooled rows its layer does not keep");
    u32(s.auxiliary_buffer(layer)->size() / sizeof(float));
    out.aux = bind(CSlice{s.auxiliary_buffer(layer).get(), 0});
    out.tokens = u32(s.auxiliary_pool(layer));
    out.width = u32(s.auxiliary_pool_width(layer));
    out.plane = PlaneShape{u32(kVkBlockTokens), u32(s.auxiliary_block_bytes(layer) / sizeof(float)), u32(s.auxiliary_rows_bytes(layer) / sizeof(float))};
    return out;
}

inline void VulkanBackend::sparse_attention_sink(Slice out, CSlice q, CSlice kv, CSlice ids, CSlice sink, size_t rows, size_t heads, size_t dim, size_t kv_rows,
                                                 size_t k, float scale, size_t layer, const KVView* views, size_t n_views, CSlice entry_ids, size_t entry_k,
                                                 ModelFormat kv_format, ModelFormat entry_format, const SinkTerms& terms) {
    if (terms.gate_form != GateForm::identity && !terms.gate.buffer) throw std::runtime_error("vulkan: sparse attention gate form without its gate");
    if (!heads || !dim || !std::isfinite(scale)) throw std::runtime_error("vulkan: invalid sparse attention dimensions or scale");
    if (!n_views && (entry_k || entry_ids.buffer)) throw std::runtime_error("vulkan: sparse attention entries without views");
    if (size_add(k, entry_k) > 1024) throw std::runtime_error("vulkan: sparse attention names at most 1024 rows a row");
    if (terms.tail_rows && !terms.tail.buffer) throw std::runtime_error("vulkan: sparse attention tail rows without their buffer");
    if (!rows) return;
    const size_t kw = format_words(kv_format, dim), ew = format_words(entry_format, dim);
    const size_t qn = size_mul(rows, size_mul(heads, dim)), kvn = size_mul(kv_rows, kw), idn = size_mul(rows, k), en = entry_ids.buffer ? size_mul(rows, entry_k) : 0;
    const size_t bn = terms.place_bias.buffer ? size_mul(heads, k) : 0, gn = terms.gate.buffer ? size_mul(rows, heads) : 0;
    const size_t tail_rows = terms.tail.buffer ? terms.tail_rows : 0, tn = size_mul(tail_rows, dim);
    // A kept tail row's first word carries the top bit (shaders/sparse_softmax.comp), so with a tail both sources stay below 2^31 words.
    if (tn && (kvn >> 31 || tn >> 31 || kv_rows + tail_rows > UINT32_MAX)) throw std::runtime_error("vulkan: sparse attention tail past 31-bit word offsets");
    if (floats_from(out) < qn || floats_from(q) < qn || floats_from(kv) < kvn || floats_from(ids) < idn || floats_from(sink) < heads || (en && floats_from(entry_ids) < en) ||
        (bn && floats_from(terms.place_bias) < bn) || (gn && floats_from(terms.gate) < gn) || (tn && floats_from(terms.tail) < tn))
        throw std::runtime_error("vulkan: sparse attention operand outside its allocation");
    if (overlaps(CSlice(out), qn, q, qn) || overlaps(CSlice(out), qn, kv, kvn) || overlaps(CSlice(out), qn, ids, idn) || overlaps(CSlice(out), qn, sink, heads) ||
        (en && overlaps(CSlice(out), qn, entry_ids, en)) || (bn && overlaps(CSlice(out), qn, terms.place_bias, bn)) || (gn && overlaps(CSlice(out), qn, terms.gate, gn)) ||
        (tn && overlaps(CSlice(out), qn, terms.tail, tn)))
        throw std::runtime_error("vulkan: sparse attention output must be disjoint from its inputs");
    PooledPlane pooled;
    const uint32_t none = 0;
    VkDescriptorBufferInfo aux = bind(kv), tab = args(&none, sizeof(none));
    if (n_views) {
        pooled = pooled_plane(layer, views, n_views, ew, false, "sparse attention");
        if (pooled.table.rows != rows) throw std::runtime_error("vulkan: sparse attention views do not cover its rows");
        aux = pooled.aux;
        tab = args(pooled.table.words.data(), pooled.table.words.size() * sizeof(uint32_t));
    }
    // The scratch holds a chunk of rows of at most 32 MiB: p by kept row and head, the denominators, the kept rows' first words and their two counts (shaders/sparse_softmax.comp).
    const size_t named = size_add(k, entry_k), row_words = size_add(size_add(size_mul(named, heads), heads), size_add(named, 2));
    const size_t chunk = std::min(rows, std::max<size_t>(1, ((size_t)8 << 20) / row_words));
    const size_t den_at = size_mul(size_mul(chunk, named), heads), kept_at = size_add(den_at, size_mul(chunk, heads)), counts_at = size_add(kept_at, size_mul(chunk, named));
    const size_t words = size_add(counts_at, size_mul(chunk, 2));
    // The kernels address the scratch by 32-bit word offsets.
    if (words > UINT32_MAX) throw std::runtime_error("vulkan: sparse attention scratch past 32-bit word offsets");
    if (!scratch_ || scratch_->size() < words * sizeof(float)) grow(scratch_, words * sizeof(float));
    const VkDescriptorBufferInfo sc{scratch_->handle(), 0, VK_WHOLE_SIZE};
    for (size_t first = 0; first < rows; first += chunk) {
        const size_t n = std::min(chunk, rows - first);
        const struct { uint32_t rows, heads, dim, k, entry_k; float scale; uint32_t views, tokens, bt, block_floats, rows_floats, width, kv_rows, kv_format, entry_format, kv_words,
                       first_row, den_at, kept_at, counts_at, tail_rows, biased; }
            pc{u32(rows), u32(heads), u32(dim), u32(k), u32(entry_k), scale, n_views ? en ? 1u : 2u : 0u, pooled.tokens, pooled.plane.bt, pooled.plane.block_floats,
               pooled.plane.rows_floats, pooled.width, u32(kv_rows), uint32_t(kv_format), uint32_t(entry_format), u32(kw), u32(first), u32(den_at), u32(kept_at), u32(counts_at), u32(tail_rows), bn ? 1u : 0u};
        dispatch(K_SPARSE_SOFTMAX, {sc, bind(q), bind(kv), bind(ids), bind(sink), aux, tab, en ? bind(entry_ids) : bind(ids), tn ? bind(terms.tail) : bind(kv),
                                    bn ? bind(terms.place_bias) : bind(sink)}, &pc, sizeof(pc), u32(size_mul(n, heads)));
        const struct { uint32_t heads, dim, named, kv_format, entry_format, first_row, den_at, kept_at, counts_at, gated, tailed; }
            cc{u32(heads), u32(dim), u32(named), uint32_t(kv_format), uint32_t(entry_format), u32(first), u32(den_at), u32(kept_at), u32(counts_at), gn ? terms.gate_form == GateForm::tanh ? 2u : 1u : 0u, tn ? 1u : 0u};
        dispatch(K_SPARSE_COLUMNS, {bind(out), sc, bind(kv), aux, tn ? bind(terms.tail) : bind(kv), gn ? bind(terms.gate) : bind(sink)}, &cc, sizeof(cc), u32((dim + kSparseColumns - 1) / kSparseColumns), u32(n));
    }
}

inline void VulkanBackend::indexer_scores(Slice out, CSlice q, size_t layer, const KVView* views, size_t n_views, CSlice w, float w_scale, size_t heads,
                                          size_t dim, ModelFormat key_format, size_t n, CSlice blocks, size_t block_k, size_t block_size, IndexForm form) {
    if (form == IndexForm::raw_blocks) {
        if (w.buffer || blocks.buffer || block_k || key_format != ModelFormat::f32) throw std::runtime_error("vulkan: raw index scores take no weights, candidate blocks or key format but f32");
        return raw_index_scores(out, q, layer, views, n_views, heads, dim, n, block_size);
    }
    if (!heads || heads > 1024 || !dim || n >= UINT32_MAX || (blocks.buffer && !block_size) || !std::isfinite(w_scale))
        throw std::runtime_error("vulkan: invalid index score dimensions");
    size_t rows = 0;
    for (size_t vi = 0; vi < n_views && views; ++vi) rows = size_add(rows, views[vi].nq);
    if (!rows || !n) return;
    const size_t key_words = format_words(key_format, dim);
    const PooledPlane pooled = pooled_plane(layer, views, n_views, key_words, false, "index scores");
    const size_t on = size_mul(rows, n), qn = size_mul(rows, size_mul(heads, dim)), wn = size_mul(rows, heads), bn = blocks.buffer ? size_mul(rows, block_k) : 0;
    if (floats_from(out) < on || floats_from(q) < qn || floats_from(w) < wn || (bn && floats_from(blocks) < bn))
        throw std::runtime_error("vulkan: index scores operand outside its allocation");
    if (overlaps(CSlice(out), on, q, qn) || overlaps(CSlice(out), on, w, wn) || (bn && overlaps(CSlice(out), on, blocks, bn)))
        throw std::runtime_error("vulkan: index scores output must be disjoint from its inputs");
    for (size_t vi = 0; vi < n_views; ++vi)
        if (views[vi].nq && size_add(views[vi].length, views[vi].nq) / pooled.tokens > n) throw std::runtime_error("vulkan: index scores narrower than a row's groups");
    if (rows > dev_->props.limits.maxComputeWorkGroupCount[1]) throw std::runtime_error("vulkan: index scores of more rows than a dispatch holds");
    const struct { uint32_t rows, heads, dim, n; float w_scale; uint32_t tokens, bt, block_floats, rows_floats, width, block_k, block_size, key_format, key_words; }
        pc{u32(rows), u32(heads), u32(dim), u32(n), w_scale, pooled.tokens, pooled.plane.bt, pooled.plane.block_floats, pooled.plane.rows_floats, pooled.width,
           bn ? u32(block_k) : 0u, u32(block_size), uint32_t(key_format), u32(key_words)};
    const VkDescriptorBufferInfo tab = args(pooled.table.words.data(), pooled.table.words.size() * sizeof(uint32_t));
    // Rows of a view in runs of 16 share each decoded key in the tile shader (shaders/indexer_scores_tile.comp), taken where its work fills the device; rows a view each, as decode's, keep a workgroup a row.
    constexpr size_t tile_bytes = (32 + 16) * 132 * sizeof(float) + 16 * 32 * sizeof(uint32_t);   // the tile shader's ks, qs and seen
    if (dim % 4 == 0 && dim <= 128 && rows >= size_mul(n_views, 4) && tile_fills_twice(dev_->caps, tile_bytes, 16, 32, rows, n)) {
        if (groups(n, 32) > dev_->props.limits.maxComputeWorkGroupCount[0]) throw std::runtime_error("vulkan: index scores of more positions than a dispatch holds");
        return dispatch(K_INDEXER_SCORES_TILE, {bind(out), bind(q), pooled.aux, tab, bind(w), bn ? bind(blocks) : bind(w)}, &pc, sizeof(pc), groups(n, 32), u32((rows + 15) / 16));
    }
    dispatch(K_INDEXER_SCORES, {bind(out), bind(q), pooled.aux, tab, bind(w), bn ? bind(blocks) : bind(w)}, &pc, sizeof(pc),
             std::min<uint32_t>(groups(n, std::max<size_t>(1, 256 / heads)), 65535), u32(rows));
}

inline void VulkanBackend::raw_index_scores(Slice out, CSlice q, size_t layer, const KVView* views, size_t n_views, size_t heads, size_t dim, size_t n, size_t block_size) {
    if (!heads || !dim || !block_size || n >= UINT32_MAX) throw std::runtime_error("vulkan: invalid index score dimensions");
    if (!views && n_views) throw std::runtime_error("vulkan: raw index scores without views");
    const ViewTable table = view_table(layer, place_views(views, n_views), false);
    if (!table.rows) return;
    const VulkanKVStorage& s = *table.storage;
    if (s.auxiliary_width(layer) != dim || !s.auxiliary_buffer(layer)) throw std::runtime_error("vulkan: raw index scores read auxiliary rows its layer does not keep at their width");
    for (size_t vi = 0; vi < n_views; ++vi)
        if (views[vi].nq && (size_add(views[vi].length, views[vi].nq) + block_size - 1) / block_size > n) throw std::runtime_error("vulkan: raw index scores narrower than a row's blocks");
    const size_t rows = table.rows, on = size_mul(rows, size_mul(heads, n)), qn = size_mul(rows, size_mul(heads, dim));
    if (floats_from(out) < on || floats_from(q) < qn) throw std::runtime_error("vulkan: index scores operand outside its allocation");
    if (overlaps(CSlice(out), on, q, qn)) throw std::runtime_error("vulkan: index scores output must be disjoint from its inputs");
    if (!n) return;
    if (n > dev_->props.limits.maxComputeWorkGroupCount[0] || (rows + 15) / 16 > dev_->props.limits.maxComputeWorkGroupCount[1])
        throw std::runtime_error("vulkan: raw index scores of more blocks or rows than a dispatch holds");
    u32(s.auxiliary_buffer(layer)->size() / sizeof(float));
    const struct { uint32_t rows, heads, dim, n, block_size, bt, block_floats, width; }
        pc{u32(rows), u32(heads), u32(dim), u32(n), u32(block_size), u32(kVkBlockTokens), u32(s.auxiliary_block_bytes(layer) / sizeof(float)), u32(dim)};
    const VkDescriptorBufferInfo tab = args(table.words.data(), table.words.size() * sizeof(uint32_t));
    dispatch(K_RAW_INDEX_SCORES, {bind(out), bind(q), bind(CSlice{s.auxiliary_buffer(layer).get(), 0}), tab}, &pc, sizeof(pc), u32(n), u32((rows + 15) / 16));
}

inline void VulkanBackend::pool_write(size_t layer, const KVView* views, size_t n_views, CSlice rows) {
    if (!n_views) return;
    if (!views) throw std::runtime_error("vulkan: pooled rows written without views");
    const PooledPlane pooled = pooled_plane(layer, views, n_views, 0, true, "pooled rows written to a layer that keeps none");
    size_t groups_count = 0;
    for (size_t vi = 0; vi < n_views; ++vi)
        groups_count = size_add(groups_count, size_add(views[vi].length, views[vi].nq) / pooled.tokens - views[vi].length / pooled.tokens);
    if (!groups_count) return;
    if (floats_from(rows) < size_mul(groups_count, pooled.width)) throw std::runtime_error("vulkan: pooled rows past their buffer");
    const VkDescriptorBufferInfo rb = bind(rows), tab = args(pooled.table.words.data(), pooled.table.words.size() * sizeof(uint32_t));
    size_t src = 0;
    for (size_t vi = 0; vi < n_views; ++vi) {
        const size_t first = views[vi].length / pooled.tokens, count = size_add(views[vi].length, views[vi].nq) / pooled.tokens - first;
        if (!count) continue;
        const size_t floats = size_mul(count, pooled.width);
        const struct { uint32_t view, first, count, width, src, tokens, bt, block_floats, rows_floats; }
            pc{u32(vi), u32(first), u32(count), pooled.width, u32(src), pooled.tokens, pooled.plane.bt, pooled.plane.block_floats, pooled.plane.rows_floats};
        dispatch(K_POOL_WRITE, {pooled.aux, rb, tab}, &pc, sizeof(pc), groups(floats, 256));
        src = size_add(src, floats);
    }
    if (unordered_) barrier(open());
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
    const size_t tiles = (groups + kScoreTile - 1) / kScoreTile;
    struct { uint32_t view, row, first, groups, heads, dim, compression; float scale; uint32_t rows, bt, block_floats, rows_floats; }
        sc{view, u32(row), 0, u32(groups), heads, dim, compression, scale, u32(rows), plane.bt, plane.block_floats, plane.rows_floats};
    for (size_t first = 0; first < tiles; first += limit) {
        sc.first = u32(first);
        dispatch(K_KV_INDEX_SCORES_TILE, {query, pooled, scores, args(table.data(), table.size() * sizeof(uint32_t))}, &sc, sizeof(sc), u32(std::min(tiles - first, limit)),
                 u32((rows + kScoreRows - 1) / kScoreRows));
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
        settle(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT);
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

// The heads and width attention is asked for against the layer's.
inline void VulkanBackend::check_storage(const VulkanKVStorage& s, size_t layer, int n_head_kv, int head_dim) {
    if (!s.holds(layer, (size_t)n_head_kv, (size_t)head_dim))
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

inline KernelId sparse_attention_kernel(uint32_t hg, const VulkanKVStorage& s) {
    switch (hg) {
    case 2: return kv_variant(K_SPARSE_ATTENTION_VEC_D256_G2, K_SPARSE_ATTENTION_VEC_D256_K16_G2, s);
    case 3: return kv_variant(K_SPARSE_ATTENTION_VEC_D256_G3, K_SPARSE_ATTENTION_VEC_D256_K16_G3, s);
    case 4: return kv_variant(K_SPARSE_ATTENTION_VEC_D256_G4, K_SPARSE_ATTENTION_VEC_D256_K16_G4, s);
    default: return kv_variant(K_SPARSE_ATTENTION_VEC_D256, K_SPARSE_ATTENTION_VEC_D256_K16, s);
    }
}

VulkanKVStorage::VulkanKVStorage(VulkanBackend& owner, const std::vector<KVShape>& shapes, size_t max_tokens,
                                 KVType kt, KVType vt, const std::vector<size_t>& auxiliary_widths, const std::vector<AuxiliaryPool>& auxiliary_pools)
    : BlockKVStorage(owner, "vulkan", kVkBlockTokens, shapes, max_tokens, kt, vt, auxiliary_widths, auxiliary_pools) {}

void VulkanKVStorage::retire(const BufferPtr& old) {
    static_cast<VulkanBackend&>(owner()).keep_until_retired(old);
}

} // namespace
} // namespace backend
