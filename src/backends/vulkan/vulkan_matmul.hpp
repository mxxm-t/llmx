#pragma once
// VulkanBackend's products: the choice of row kernel build or tile, column chunks, row and tile dispatch, and routed expert calls; vulkan_backend.cpp includes it after the class.
#include <algorithm>
#include <array>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "backends/vulkan/vulkan_buffer.hpp"
#include "backends/vulkan/vulkan_kernels.hpp"
#include "quant/quant.hpp"

namespace backend {
namespace {

// The narrower of the Q8_0 decode kernel's widest build and the other row kernels', each reading a weight once for its columns.
inline size_t VulkanBackend::decode_columns() const {
    return std::min<size_t>(dev_->profile.q8_decode_cols, dev_->profile.row_decode_cols ? dev_->profile.row_decode_cols : kRowColsWide);
}

inline void VulkanBackend::matmul(WeightView weight, CSlice X, Slice Y, size_t nin, size_t nout,
                                  size_t nbatch, RowRuns runs, Dtype dtype) {
    const Projection one{weight, Y, nout};
    matmul_runs({one}, X, nin, nbatch, false, runs, dtype);
}

// The output head is matmul, as the interface's own default has it; a backend that times its dispatches names this call's apart (kernel_times), and no other backend keeps any state for it.
inline void VulkanBackend::matmul_logits(WeightView weight, CSlice X, Slice Y, size_t nin, size_t nout, size_t nbatch, RowRuns runs,
                                         Dtype dtype) {
    if (!dev_->timestamps) return matmul(weight, X, Y, nin, nout, nbatch, runs, dtype);
    struct Head {
        bool& on;
        ~Head() { on = false; }
    } head{head_};
    head_ = true;
    matmul(weight, X, Y, nin, nout, nbatch, runs, dtype);
}

// The residual add folded into the kernels' store: Y += W X.
inline void VulkanBackend::matmul_add(WeightView weight, CSlice X, Slice Y, size_t nin, size_t nout,
                                      size_t nbatch, RowRuns runs, Dtype dtype) {
    const Projection one{weight, Y, nout};
    matmul_runs({one}, X, nin, nbatch, true, runs, dtype);
}

inline void VulkanBackend::matmul_group(std::initializer_list<Projection> projections, CSlice X,
                                        size_t nin, size_t nbatch, RowRuns runs, Dtype dtype) {
    matmul_runs(projections, X, nin, nbatch, false, runs, dtype);
}

// Batch rows from which a call takes the tile rather than the row kernel, by the types of the projections that have rows.
// The row kernel costs a weight pass per eight columns and the tile a whole tile however little is filled, so the crossover depends on the row width and the types; measured per device (backends/device_profile.hpp).
inline size_t VulkanBackend::tile_from(const Projection* projections, size_t count, size_t nin) const {
    bool eight_bit_or_float = true;
    for (size_t i = 0; i < count; ++i) {
        const Projection& pr = projections[i];
        if (pr.rows && !weight_kernels(pr.weight.type)->fast_tile) eight_bit_or_float = false;
    }
    return tile_from_for(dev_->profile, eight_bit_or_float, nin);
}

// Row runs choose kernels and tile splits by prompt extent, preserving results across batching (docs/VULKAN.md, batch invariance).
// Adjacent runs share a call only when both choices match.
inline size_t VulkanBackend::split_tiles_of(size_t extent) { return (std::min<size_t>(extent, 512) + 63) / 64; }

inline void VulkanBackend::matmul_runs(std::initializer_list<Projection> projections, CSlice X, size_t nin, size_t nbatch,
                                       bool accumulate, RowRuns runs, Dtype dtype) {
    if (!nbatch) return;
    check_group(projections.begin(), projections.size(), X, nin, nbatch);
    if (!runs.n) {
        matmul_group_impl(projections.begin(), projections.size(), X, nin, nbatch, accumulate, dtype);
        return;
    }
    const size_t from = tile_from(projections.begin(), projections.size(), nin);
    auto tile_split = [&](const RowRun& r) {
        const bool tile = dtype == Dtype::bf16 || r.extent >= from;
        return std::make_pair(tile, tile ? split_tiles_of(r.extent) : size_t(0));
    };
    for_each_run(nbatch, runs, tile_split, [&](size_t start, size_t rows, std::pair<bool, size_t> ts) {
        const int k = ts.first ? 1 : 0;
        if (rows == nbatch) {
            matmul_group_impl(projections.begin(), projections.size(), X, nin, nbatch, accumulate, dtype, k, ts.second);
        } else {
            std::vector<Projection> at(projections);
            for (Projection& pr : at) pr.out.offset += start * pr.rows;
            const CSlice xs{X.buffer, X.offset + start * nin};
            matmul_group_impl(at.data(), at.size(), xs, nin, rows, accumulate, dtype, k, ts.second);
        }
    });
}

// Throws unless the weights, outputs and X hold the call's `nbatch` rows.
// matmul_runs checks a call whole before recording any of it, so a call its runs split into several is refused before the first.
inline void VulkanBackend::check_group(const Projection* projections, size_t count, CSlice X, size_t nin, size_t nbatch) const {
    for (size_t i = 0; i < count; ++i) {
        const Projection& pr = projections[i];
        check_matrix(pr.weight.type, pr.weight.data, nin, pr.rows);
        check_weight_scales(pr.weight);
        if (floats_from(pr.out) < size_mul(nbatch, pr.rows)) throw std::runtime_error("vulkan: matmul operand outside its allocation");
    }
    if (floats_from(X) < size_mul(nbatch, nin)) throw std::runtime_error("vulkan: matmul operand outside its allocation");
}

// Dispatch up to three validated projections: kernel_choice selects row (0), tile (1), or the batch crossover (negative).
// A nonzero split_tiles fixes the split to that many column tiles; zero uses the call's width.
inline void VulkanBackend::matmul_group_impl(const Projection* projections, size_t count, CSlice X,
                                             size_t nin, size_t nbatch, bool accumulate, Dtype dtype, int kernel_choice, size_t split_tiles) {
    // The kernels' limit; no caller passes more.
    if (count > 3) throw std::logic_error("vulkan: a matmul call of more than three projections");
    if (!nbatch) return;
    std::vector<const Projection*> live;
    for (size_t i = 0; i < count; ++i)
        if (projections[i].rows) live.push_back(&projections[i]);
    if (live.empty()) return;
    // int8 takes f16's kernels but where a type has an 8-bit build.
    const bool sixteen = dtype == Dtype::f16 || dtype == Dtype::int8;
    if (dtype == Dtype::bf16 || kernel_choice == 1 || (kernel_choice < 0 && nbatch >= tile_from(projections, count, nin))) {
        const size_t gy = (nbatch + 63) / 64;
        std::vector<Projection> copies;
        std::vector<CSlice> packed;
        copies.reserve(live.size());
        auto original = [&](const Projection* pr) -> const CSlice* {
            for (size_t i = 0; i < copies.size(); ++i) if (pr == &copies[i]) return &packed[i];
            return nullptr;
        };
        size_t copy_at = 0;
        if (sixteen && dev_->profile.prefer_integer_dot) {
            for (const Projection*& pr : live) {
                if (pr->weight.type != quant::GGML_TYPE_MXFP4) continue;
                const size_t bytes = size_mul(pr->rows, size_mul(nin / 32, kMxCopyBlock));
                if (bytes > kMxCopyBytes - copy_at) continue;
                copy_mxfp4(pr->weight.data, pr->rows, nin, copy_at);
                packed.push_back(pr->weight.data);
                WeightView copied = pr->weight;
                copied.data = CSlice(mx_copy_.get(), copy_at / sizeof(float));
                copies.push_back(Projection{copied, pr->out, pr->rows});
                pr = &copies.back();
                copy_at += (bytes + 255) & ~size_t(255);
            }
        }
        auto integer = [&](const Projection* pr) {
            return sixteen && integer_dot_tile(pr->weight.type) &&
                   (pr->weight.type != quant::GGML_TYPE_MXFP4 || original(pr));
        };
        if (gy > dev_->props.limits.maxComputeWorkGroupCount[1])
            throw std::runtime_error("vulkan: dispatch exceeds the workgroup count limit");
        // On a device whose integer dot is native, quantized types take the integer-dot tile (shaders/matmul_tile_q.comp), which reads X's 16-bit twin, or under int8 its 8-bit twin, each made once for every projection that needs it.
        const size_t nblk = nin / 32;
        // The float tile, one projection a dispatch.
        for (const Projection* pr : live) {
            if (integer(pr)) continue;
            // Split as the integer-dot tile is, by the rows' whole prompt (matmul_runs), so a projection of few rows, such as a router's, still fills the device.
            const size_t st = split_tiles ? split_tiles : gy;
            const uint32_t hs = tile_rows_for(dev_->caps, dev_->profile, kTileRowsSmall, kTileRowsShort, kTileRowsTall, pr->rows, st, nin);
            const size_t steps = (nin + 31) / 32;   // an F32 row's last step may be partial
            // Split only below a quarter of a workgroup per compute unit to limit reduction overhead (docs/VULKAN.md).
            const size_t fwg = groups(pr->rows, hs) * st;
            const bool starved = fwg * 4 < dev_->caps.compute_units;
            const size_t base_kper = starved ? split_blocks(groups(pr->rows, kTileRowsSmall) * st, steps, dev_->profile.float_tile_split_per_cu) : steps;
            // Shorter F32 sums keep the BF16 depth agreement; retain a finer split chosen for occupancy.
            const size_t kper = dtype == Dtype::bf16 ? std::min(base_kper, (steps + 3) / 4) : base_kper;
            const size_t parts = (steps + kper - 1) / kper;
            // A taller tile reads less shared memory per product but halves the workgroups; below one per compute unit the call takes a shorter one, and a starved call the shortest, for the most workgroups.
            const uint32_t height = starved ? kTileRowsSmall
                                            : float_tile_rows_for(dev_->caps, dev_->profile, pr->weight.type, pr->rows, gy, nin);
            const bool tall = height == kTileRowsTall;
            const VkDescriptorBufferInfo wf = padded_f32(pr->weight.data, pr->weight.type, pr->rows, nin);
            const size_t wstride = wf.buffer == bind(pr->weight.data).buffer ? nin : nin + kF32Pad;
            const VkDescriptorBufferInfo words = weight_floats(pr->weight, lookup_grid(pr->weight.type, wf));
            uint32_t pc[10] = {u32(nin), u32(pr->rows), u32(nbatch), pr->weight.type, accumulate && parts == 1 ? 1u : 0u, 0, 0,
                                     u32(kper * 32), u32(gy), u32(wstride)};
            const KernelId kernel = float_tile_kernel(tall, dtype, pr->weight.type, dev_->f16_inputs);
            const int small = height == kTileRowsSmall ? 1 : 0;
            if (parts > 1) {
                const size_t column_bytes = size_mul(size_mul(parts, pr->rows), pr->weight.type == quant::GGML_TYPE_MXFP4 ? sizeof(double) : sizeof(float));
                const size_t alignment = std::max<size_t>(1, dev_->props.limits.minStorageBufferOffsetAlignment / sizeof(float));
                const size_t aligned = alignment / std::gcd(alignment, std::gcd(nin, pr->rows));
                // Keep the same K parts and tile height while bounding their storage by column slices; each binding starts at the device's alignment.
                const size_t block = column_bytes <= kFloatPartBytes / 64 ? std::max<size_t>(64, aligned) : aligned;
                const size_t columns = std::max(block, kFloatPartBytes / column_bytes / block * block);
                const size_t bytes = size_mul(std::min(nbatch, columns), column_bytes);
                if (!parts_ || parts_->size() < bytes) grow(parts_, bytes);
                const VkDescriptorBufferInfo pb{parts_->handle(), 0, VK_WHOLE_SIZE};
                for (size_t first = 0; first < nbatch; first += columns) {
                    const size_t taken = std::min(nbatch - first, columns), tiles = (taken + 63) / 64;
                    const CSlice x{X.buffer, X.offset + first * nin};
                    const Slice y{pr->out.buffer, pr->out.offset + first * pr->rows};
                    pc[2] = u32(taken);
                    pc[8] = u32(tiles);
                    dispatch(kernel, {pb, bind(pr->weight.data), words, bind(x), bind(pr->weight.data), bind(x)},
                             pc, sizeof(pc), groups(pr->rows, height), u32(tiles * parts), small);
                    const size_t n = taken * pr->rows;
                    const uint32_t start = u32(groups(n, 256));
                    const uint32_t rc[7] = {u32(n), 0, 0, u32(parts), accumulate ? 1u : 0u, start, start + 1};
                    dispatch(pr->weight.type == quant::GGML_TYPE_MXFP4 ? K_MATMUL_REDUCE_MXFP4 : K_MATMUL_REDUCE, {bind(y), bind(y), bind(y), pb}, rc, sizeof(rc), start, 1, 0,
                             {n, n, n});
                }
            } else if (const KernelId wide = wide_tile_kernel(dtype, pr->weight.type); wide != K_COUNT && wide_tile_fits(dev_->caps, pr->rows, nbatch)) {
                dispatch(wide, {bind(pr->out), bind(pr->weight.data), words, bind(X), bind(pr->weight.data), bind(X)},
                         pc, sizeof(pc), groups(pr->rows, kWideTile), groups(nbatch, kWideTile), 0, {nbatch * pr->rows});
            } else {
                dispatch(kernel, {bind(pr->out), bind(pr->weight.data), words, bind(X), bind(pr->weight.data), bind(X)},
                         pc, sizeof(pc), groups(pr->rows, height), (uint32_t)gy, small, {nbatch * pr->rows});
            }
            record_matrix_path(is_nvfp4_f16_tile(kernel) ? MatrixPath::f16 : dtype == Dtype::bf16 ? MatrixPath::bf16 : MatrixPath::f32);
        }
        // The integer-dot tile takes the projections of one type in one dispatch, since each alone can be too small to fill the device.
        std::vector<const Projection*> pending;
        for (const Projection* pr : live)
            if (integer(pr)) pending.push_back(pr);
        while (!pending.empty()) {
            std::vector<const Projection*> group, rest;
            for (const Projection* pr : pending)
                (pr->weight.type == pending[0]->weight.type ? group : rest).push_back(pr);
            pending.swap(rest);
            const bool i8 = dtype == Dtype::int8 && weight_kernels(group[0]->weight.type)->int8_tile != K_COUNT;
            uint32_t xpad = 0;
            const VkDescriptorBufferInfo xt = tile_twin(X, nbatch * nin, nin, i8, xpad);
            const QTile t = qtile(group, gy, nin, i8);
            // The split is the one the rows' whole prompt would take (matmul_runs).
            const size_t st = split_tiles ? split_tiles : gy;
            const uint32_t hs = tile_rows_for(dev_->caps, dev_->profile, kTileRowsSmall, kTileRowsShort, kTileRowsTall, t.rows, st, nin);
            size_t gxs = 0;
            for (const Projection* pr : group) gxs += groups(pr->rows, hs);
            const size_t kper = split_blocks(gxs * st, nblk);
            const size_t parts = (nblk + kper - 1) / kper;
            const uint32_t pc[16] = {u32(nin), u32(nbatch), group[0]->weight.type, accumulate && parts == 1 ? 1u : 0u, u32(kper),
                                     u32(group.size()), t.nout[0], t.start[0], t.nout[1], t.start[1], t.nout[2], t.start[2], 0, 0, u32(nbatch), xpad};
            const Projection &a = *t.p[0], &b = *t.p[1], &c = *t.p[2];
            const int small = t.height == kTileRowsSmall ? 1 : 0;
            const bool mx = a.weight.type == quant::GGML_TYPE_MXFP4;
            // Each output's floats, an unused slot repeating the first's.
            const std::array<size_t, 3> reach{nbatch * a.rows, nbatch * b.rows, nbatch * c.rows};
            if (parts > 1) {
                const size_t n = nbatch * t.rows;
                const size_t bytes = size_mul(size_mul(parts, n), mx ? sizeof(double) : sizeof(float));
                if (!parts_ || parts_->size() < bytes) grow(parts_, bytes);
                const VkDescriptorBufferInfo pb{parts_->handle(), 0, VK_WHOLE_SIZE};
                if (mx)
                    dispatch(t.kernel, {pb, pb, pb, bind(a.weight.data), bind(b.weight.data), bind(c.weight.data), xt, xt, xt,
                                       bind(*original(t.p[0])), bind(*original(t.p[1])), bind(*original(t.p[2]))},
                             pc, sizeof(pc), u32(t.gx), u32(gy * parts), small);
                else
                    dispatch_qtile(t, {pb, pb, pb}, xt, xt, pc, u32(gy * parts));
                const size_t n0 = nbatch * a.rows, n1 = group.size() > 1 ? nbatch * b.rows : 0,
                             n2 = group.size() > 2 ? nbatch * c.rows : 0;
                const size_t start1 = groups(n0, 256), start2 = start1 + groups(n1, 256);
                // An unused projection's start is past every workgroup.
                const uint32_t rc[7] = {u32(n0), u32(n1), u32(n2), u32(parts), accumulate ? 1u : 0u,
                                        u32(n1 ? start1 : start2 + groups(n2, 256)), u32(n2 ? start2 : start2 + groups(n2, 256) + 1)};
                dispatch(mx ? K_MATMUL_REDUCE_MXFP4 : K_MATMUL_REDUCE, {bind(a.out), bind(b.out), bind(c.out), pb}, rc, sizeof(rc),
                         u32(start2 + groups(n2, 256)), 1, 0, {reach[0], reach[1], reach[2]});
            } else if (mx) {
                dispatch(t.kernel, {bind(a.out), bind(b.out), bind(c.out), bind(a.weight.data), bind(b.weight.data), bind(c.weight.data), xt, xt, xt,
                                   bind(*original(t.p[0])), bind(*original(t.p[1])), bind(*original(t.p[2]))},
                         pc, sizeof(pc), u32(t.gx), u32(gy), small, {reach[0], reach[1], reach[2]});
            } else {
                dispatch_qtile(t, {bind(a.out), bind(b.out), bind(c.out)}, xt, xt, pc, u32(gy), reach);
            }
            record_matrix_path(i8 ? MatrixPath::block_int8 : MatrixPath::block_int16);
        }
        return;
    }
    // A dispatch shares one type, module and cluster size, so mixed groups partition by type.
    // Each partition retains the whole group's kernel choice to preserve the arithmetic at crossovers.
    for (size_t i = 1; i < live.size(); ++i)
        if (live[i]->weight.type != live[0]->weight.type) {
            std::vector<Projection> same, rest;
            for (const Projection* pr : live)
                (pr->weight.type == live[0]->weight.type ? same : rest).push_back(*pr);
            matmul_group_impl(same.data(), same.size(), X, nin, nbatch, accumulate, dtype, 0);
            matmul_group_impl(rest.data(), rest.size(), X, nin, nbatch, accumulate, dtype, 0);
            return;
        }
    const RowPlan plan = row_plan(live[0]->weight.type, nin, dtype);
    const VkDescriptorBufferInfo xqi = row_twin(X, plan, nbatch * nin, dtype);
    size_t rows = 0;
    for (const Projection* pr : live) rows += pr->rows;
    for_each_column_chunk(plan.kernel, nbatch, nin, rows, [&](size_t col0, size_t ncols, int variant) {
        row_dispatch(plan, live, X, xqi, nin, nbatch, col0, ncols, accumulate, ncols == 1 && paired_build(plan.kernel, nin) ? kVecPaired : variant);
    });
}

// Whether a plain one-column call of Q8_0 rows `nin` wide takes the paired build: half-block order and the reduction tree on 64-lane subgroups, and an even block count of at most 16, so a half's lanes cover the row.
// It gives every column the bits of the build it replaces (docs/src/backends-vulkan.md, Decode builds).
inline bool VulkanBackend::paired_build(KernelId id, size_t nin) const {
    const size_t nblk = nin / 32;
    const uint32_t forms = vec_forms(kVecPaired, dev_->profile);
    return is_vec_kernel(id) && dev_->caps.subgroup_size == 64 && (forms & kQ8Tree) && (forms & kQ8Half) && nblk % 2 == 0 && nblk <= 16;
}

// Return the count of available decode builds, narrowest first, limited by the profile's Q8_0 or Q4/K-quant column limit.
// Other row kernels use their one-column and wide builds; the Q8_0 decode kernel's one-column build follows the call's shape (one_column_build).
inline size_t VulkanBackend::column_builds(KernelId id, size_t nin, size_t rows, int (&out)[kVariants]) const {
    size_t n = 0;
    if (is_vec_kernel(id)) {
        for (int v : kVecByWidth)
            if (kVecBuilds[v].capacity() <= dev_->profile.q8_decode_cols) out[n++] = v == 1 ? one_column_build(nin, rows, reads_x8(id)) : v;
        return n;
    }
    if (row_kernel_builds_one_column(id)) out[n++] = 1;
    if (row_kernel_builds_two_rows(id) && dev_->profile.row_decode_cols >= kRowBuilds[kRowByWidth[0]].cols) {
        for (int v : kRowByWidth)
            if (kRowBuilds[v].cols <= dev_->profile.row_decode_cols) out[n++] = v;
        return n;
    }
    out[n++] = 0;
    return n;
}

// The Q8_0 decode kernel's one-column build for `rows` output rows `nin` wide: two rows and one step below the profile's narrow width, else one row and four steps below the knee of its activations' width.
// Above that knee 8-bit activations take two rows and two steps and 16-bit ones two rows and one step.
inline int VulkanBackend::one_column_build(size_t nin, size_t rows, bool x8) const {
    if (nin < dev_->profile.q8_decode_narrow_below) return 1;
    if (!x8) return rows < dev_->profile.q8_decode_one_row_below_16bit ? kVecOneRow : 1;
    return rows < dev_->profile.q8_decode_two_rows_from ? kVecOneRow : kVecTwoSteps;
}

// Call each(col0, ncols, variant), filling widest builds first and using the narrowest build for the remainder; `rows` are the output rows a dispatch computes, each `nin` wide.
// Every build gives a column the same bits (docs/src/backends-vulkan.md, Decode builds).
template <typename Fn>
inline void VulkanBackend::for_each_column_chunk(KernelId id, size_t nbatch, size_t nin, size_t rows, const Fn& each) const {
    int builds[kVariants] = {};
    const size_t n = column_builds(id, nin, rows, builds);
    const size_t wide = build_cols(id, builds[n - 1]);
    size_t col0 = 0;
    for (; nbatch - col0 > wide; col0 += wide) each(col0, wide, builds[n - 1]);
    size_t i = 0;
    while (build_cols(id, builds[i]) < nbatch - col0) ++i;
    each(col0, nbatch - col0, builds[i]);
}

inline VulkanBackend::RowPlan VulkanBackend::row_plan(uint32_t type, size_t nin, Dtype dtype) const {
    const WeightKernels& format = *weight_kernels(type);
    const size_t nblocks = nin / quant::storage_type(type)->block_size;
    uint32_t wide = 0, lanes = 1;
    size_t units = nin;
    const bool dot = dev_->profile.*format.dot_from;
    KernelId kernel = dot ? format.dot_row : format.row;
    switch (format.layout) {
    case RowLayout::values: break;
    case RowLayout::groups32: units = nin / 32; break;
    case RowLayout::q8_pairs:
        wide = nblocks % 2 == 0 && nblocks / 2 >= kQ8LanesPerPair;
        lanes = wide ? kQ8LanesPerPair : 1;
        units = wide ? nblocks / 2 * lanes : nblocks;
        if (wide) kernel = K_MATMUL_ROW_Q8W;
        break;
    case RowLayout::q4_pairs:
        wide = nblocks % 2 == 0;
        lanes = wide ? 2 : 1;
        units = wide ? nblocks / 2 * lanes : nblocks;
        break;
    case RowLayout::k_blocks:
        lanes = kKQuantLanes;
        units = nblocks * lanes;
        break;
    }
    if ((dtype == Dtype::f32 || kernel == format.float_row) && format.layout != RowLayout::values) {
        kernel = format.float_row;
        units = nin;
        lanes = 1;
        wide = 0;
    }
    uint32_t cluster = lanes;
    while (cluster < dev_->caps.subgroup_size && cluster < units) cluster *= 2;
    // Under int8 a type with an 8-bit build takes it, in the 16-bit build's layout.
    if (dtype == Dtype::int8 && dev_->profile.prefer_integer_dot && format.int8_row != K_COUNT) kernel = format.int8_row;
    if (kernel == K_MATMUL_ROW_K4_DOT || kernel == K_MATMUL_ROW_K5_DOT || kernel == K_MATMUL_ROW_K4_I8 || kernel == K_MATMUL_ROW_K5_I8)
        cluster = std::min(cluster, std::max(lanes, dev_->profile.k45_row_lanes));
    // Where the integer dot is native, Q8_0 rows take the integer dot over the 16-bit twin (shaders/matmul_vec_q8.comp).
    if (dtype == Dtype::f16 && type == quant::GGML_TYPE_Q8_0 && dev_->profile.prefer_integer_dot) kernel = K_MATMUL_VEC_Q8;
    return RowPlan{kernel, type, wide, cluster};
}

// Rows a subgroup takes in a row kernel's build: in matmul_row.comp the build's rows a cluster of lanes (build_rows), in the Q8_0 decode kernel its build's own (kVecBuilds).
inline uint32_t VulkanBackend::sg_rows(const RowPlan& plan, int variant) const {
    return is_vec_kernel(plan.kernel) ? kVecBuilds[variant].rows * kVecBuilds[variant].halves : dev_->caps.subgroup_size / plan.cluster * build_rows(plan.kernel, variant);
}

// Dispatch a row build over the selected columns and projections.
// A routed call instead reads the expert ids and runs one workgroup row per entry.
inline void VulkanBackend::row_dispatch(const RowPlan& plan, const std::vector<const Projection*>& live, CSlice X, VkDescriptorBufferInfo xqi,
                                        size_t nin, size_t nbatch, size_t col0, size_t ncols, bool accumulate, int variant,
                                        uint32_t per, size_t entries, VkDescriptorBufferInfo ids,
                                        uint32_t order0, VkDescriptorBufferInfo tab, size_t routed) {
    // A Q8_0 decode build whose column groups take the same rows gives each workgroup's rows that many adjacent workgroups.
    const uint32_t per_group = (256 / dev_->caps.subgroup_size) * sg_rows(plan, variant);
    const uint32_t span = is_vec_kernel(plan.kernel) ? kVecBuilds[variant].span : 1;
    uint32_t nout[3] = {0, 0, 0}, start[3] = {0, 0, 0};
    uint32_t total = 0;
    for (size_t i = 0; i < live.size(); ++i) {
        nout[i] = u32(live[i]->rows);
        start[i] = total;
        total += groups(live[i]->rows, per_group) * span;
    }
    if (total > dev_->props.limits.maxComputeWorkGroupCount[0] || entries > dev_->props.limits.maxComputeWorkGroupCount[1])
        throw std::runtime_error("vulkan: dispatch exceeds the workgroup count limit");
    // Unused projection slots bind the first one's buffers; no workgroup reaches them.
    const Projection& a = *live[0];
    const Projection& b = live.size() > 1 ? *live[1] : a;
    const Projection& c = live.size() > 2 ? *live[2] : a;
    const uint32_t t = plan.type, w = plan.wide;
    // Dictionary float rows alias the first twin view; integer rows alias the unused F32 weight view.
    const VkDescriptorBufferInfo dictionary = (plan.kernel == K_MATMUL_ROW_IQ1_S || plan.kernel == K_MATMUL_ROW_IQ1_M || plan.kernel == K_MATMUL_ROW_IQ3_S || plan.kernel == K_MATMUL_ROW_IQ2_S || plan.kernel == K_MATMUL_ROW_IQ3_XXS || plan.kernel == K_MATMUL_ROW_IQ2_XXS || plan.kernel == K_MATMUL_ROW_IQ2_XS) ? lookup_grid(t, xqi) : xqi;
    const bool lookup_dot = plan.kernel == K_MATMUL_ROW_IQ1_S_DOT || plan.kernel == K_MATMUL_ROW_IQ1_M_DOT || plan.kernel == K_MATMUL_ROW_IQ3_S_DOT || plan.kernel == K_MATMUL_ROW_IQ2_S_DOT || (plan.kernel == K_MATMUL_ROW_IQ3_XXS_DOT || plan.kernel == K_MATMUL_ROW_IQ2_XXS_DOT || plan.kernel == K_MATMUL_ROW_IQ2_XS_DOT);
    // The call's outputs are the projections' whole columns, a routed call's its entries'.
    const size_t columns = routed ? routed : per ? entries : nbatch;
    // matmul_row.comp counts its rows by cluster, the Q8_0 decode kernel reads neither.
    const uint32_t pc[22] = {u32(nin), u32(nbatch), u32(col0), u32(ncols), plan.cluster, dev_->caps.subgroup_size / plan.cluster,
                             (uint32_t)live.size(),
                             nout[0], t, w, start[0],
                             nout[1], t, w, start[1],
                             nout[2], t, w, start[2], accumulate ? 1u : 0u, per, order0};
    dispatch(plan.kernel,
             {bind(a.out), bind(b.out), bind(c.out),
              bind(a.weight.data), bind(b.weight.data), bind(c.weight.data),
              bind(a.weight.data), bind(b.weight.data), bind(c.weight.data),
              weight_floats(a.weight, lookup_dot ? lookup_grid(t, bind(a.weight.data)) : bind(a.weight.data)),
              weight_floats(b.weight, lookup_dot ? lookup_grid(t, bind(b.weight.data)) : bind(b.weight.data)),
              weight_floats(c.weight, lookup_dot ? lookup_grid(t, bind(c.weight.data)) : bind(c.weight.data)),
              bind(X),
              bind(a.weight.data), bind(b.weight.data), bind(c.weight.data),
              dictionary, xqi, xqi, xqi, ids.buffer ? ids : bind(X), tab.buffer ? tab : bind(X)},
             pc, sizeof(pc), total, u32(entries), variant, {columns * a.rows, columns * b.rows, columns * c.rows});
    record_matrix_path(weight_kernels(a.weight.type)->layout == RowLayout::values || plan.kernel == weight_kernels(a.weight.type)->float_row ? MatrixPath::f32
                       : reads_x8(plan.kernel) ? MatrixPath::block_int8 : MatrixPath::block_int16);
}

// Routed entries take row kernels or per-expert tiles according to prompt extent and the type's crossover.
// Routed tiles never split, keeping each entry's arithmetic independent of neighbouring entries.
inline void VulkanBackend::route_experts(CSlice scores, size_t rows, size_t n_expert, size_t k, bool normalize, Slice ids, Slice weights) {
    if (!k || k > n_expert || k > 256 || n_expert > 1024)
        throw std::runtime_error("vulkan: routing takes 1 to 256 of at most 1024 experts");
    if (!rows) return;
    if (floats_from(scores) < rows * n_expert || floats_from(ids) < rows * k || floats_from(weights) < rows * k)
        throw std::runtime_error("vulkan: routing operand outside its allocation");
    const uint32_t pc[3] = {u32(n_expert), u32(k), normalize ? 1u : 0u};
    // The build holding the fewest experts an invocation needs, e = t + 64 q (shaders/moe_route.comp).
    const KernelId kernel = n_expert <= 64 ? K_MOE_ROUTE_S1 : n_expert <= 128 ? K_MOE_ROUTE_S2 : n_expert <= 256 ? K_MOE_ROUTE_S4 : n_expert <= 512 ? K_MOE_ROUTE_S8 : K_MOE_ROUTE;
    dispatch(kernel, {bind(scores), bind(ids), bind(weights)}, pc, sizeof(pc), u32(rows), 1, 0, {0, rows * k, rows * k});
    group_tag_ = GroupTag{};
}

// Calls `each(first, count, tile)` over the token rows of a routed call, a call per stretch of rows that take the same kernel.
template <typename Fn>
inline void VulkanBackend::expert_runs(uint32_t type, size_t nrows, RowRuns runs, const Fn& each) {
    const size_t from = dev_->profile.*(weight_kernels(type)->moe_from);
    if (!runs.n) { each(0, nrows, nrows >= from); return; }
    for_each_run(nrows, runs, [&](const RowRun& r) { return r.extent >= from; }, each);
}

inline CSlice VulkanBackend::at_float(CSlice s, size_t floats) { return CSlice{s.buffer, s.offset + floats}; }

inline Slice VulkanBackend::at_float(Slice s, size_t floats) { return Slice{s.buffer, s.offset + floats}; }

inline void VulkanBackend::matmul_experts(std::initializer_list<Projection> projections, CSlice X, size_t nin, size_t nrows,
                                          const Routing& routing, RowRuns runs, Dtype dtype) {
    const size_t k = routing.k, entries = nrows * k;
    if (!entries || !projections.size()) return;
    if (projections.size() > 3) throw std::logic_error("vulkan: more than three routed projections");
    for (const Projection& pr : projections) {
        check_matrix(pr.weight.type, pr.weight.data, nin, size_mul(routing.n_expert, pr.rows));
        check_weight_scales(pr.weight, routing.n_expert);
        if (floats_from(pr.out) < entries * pr.rows) throw std::runtime_error("vulkan: matmul operand outside its allocation");
        if (pr.weight.type != projections.begin()->weight.type) throw std::runtime_error("vulkan: routed projections of different types");
    }
    if (floats_from(X) < nrows * nin || floats_from(routing.ids) < entries)
        throw std::runtime_error("vulkan: matmul operand outside its allocation");
    expert_runs(projections.begin()->weight.type, nrows, runs, [&](size_t first, size_t count, bool tile) {
        std::vector<Projection> at(projections);
        for (Projection& pr : at) pr.out = at_float(pr.out, first * k * pr.rows);
        std::vector<const Projection*> live;
        for (const Projection& pr : at) live.push_back(&pr);
        routed_call(live, at_float(X, first * nin), nin, count, count, k, at_float(routing.ids, first * k), routing.n_expert, tile, dtype);
    });
}

inline void VulkanBackend::matmul_experts_add(WeightView weight, CSlice X, Slice Y, size_t nin, size_t nout, size_t nrows,
                                              const Routing& routing, RowRuns runs, Dtype dtype) {
    const uint32_t type = weight.type;
    const size_t k = routing.k, entries = nrows * k;
    if (!entries) return;
    check_matrix(type, weight.data, nin, size_mul(routing.n_expert, nout));
    check_weight_scales(weight, routing.n_expert);
    if (floats_from(X) < entries * nin || floats_from(Y) < nrows * nout || floats_from(routing.ids) < entries ||
        floats_from(routing.weights) < entries)
        throw std::runtime_error("vulkan: matmul operand outside its allocation");
    if (!moe_out_ || moe_out_->size() < entries * nout * sizeof(float)) grow(moe_out_, entries * nout * sizeof(float));
    expert_runs(type, nrows, runs, [&](size_t first, size_t count, bool tile) {
        const Slice part{moe_out_.get(), first * k * nout};
        const Projection one{weight, part, nout};
        routed_call({&one}, at_float(X, first * k * nin), nin, count * k, count, k, at_float(routing.ids, first * k),
                    routing.n_expert, tile, dtype, 1);
        const uint32_t pc[3] = {u32(count), u32(nout), u32(k)};
        dispatch(K_MOE_COMBINE, {bind(at_float(Y, first * nout)), bind(CSlice(part)), bind(at_float(routing.weights, first * k)), bind(at_float(routing.ids, first * k))},
                 pc, sizeof(pc), groups(count * nout, 256), 1, 0, {count * nout});
    });
}

// One routed call over `nrows` token rows, k entries each, whose X has `xcols` columns: entry e reads column e / per, per being k when X is the token rows and 1 when it holds a column per entry.
inline void VulkanBackend::routed_call(const std::vector<const Projection*>& live, CSlice X, size_t nin, size_t xcols, size_t nrows, size_t k,
                                       CSlice ids, size_t n_expert, bool tile, Dtype dtype, size_t per) {
    if (dtype == Dtype::bf16) tile = true;
    if (!per) per = k;
    const size_t entries = nrows * k;
    const uint32_t type = live[0]->weight.type;
    if (!tile && entries < 2 * n_expert) {
        // Generated tokens whose entries average fewer than two an expert: each entry its own workgroup row, through the one-column build, since grouping them would save few reads and costs a dispatch.
        const RowPlan plan = row_plan(type, nin, dtype);
        const VkDescriptorBufferInfo xqi = row_twin(X, plan, xcols * nin, dtype);
        size_t rows = 0;
        for (const Projection* pr : live) rows += pr->rows * entries;
        for_each_column_chunk(plan.kernel, 1, nin, rows, [&](size_t, size_t, int variant) {
            row_dispatch(plan, live, X, xqi, nin, xcols, 0, 1, false, variant, u32(per), entries, bind(ids));
        });
        return;
    }
    // Grouped by expert: the tiles in runs of 64, the row kernels in runs of their column count, so an expert's rows are read once per run rather than once per entry; at most one partial run per expert that has any.
    // Where the entries average 16 an expert or fewer the integer tile takes its narrow build, in runs of 16, whose columns give the 64-column build's bits.
    const bool integer = tile && (dtype == Dtype::f16 || dtype == Dtype::int8) && integer_dot_tile(type) && type != quant::GGML_TYPE_MXFP4;
    const bool narrow = integer && entries <= kNarrowCols * std::min(n_expert, entries);
    const size_t chunk = narrow ? kNarrowCols : tile ? 64 : kRowColsWide;
    const size_t max_tiles = (entries + chunk - 1) / chunk + std::min(n_expert, entries);
    const size_t tab_bytes = (4 * max_tiles + entries) * sizeof(uint32_t);
    if (!moe_tab_ || moe_tab_->size() < tab_bytes) {
        grow(moe_tab_, tab_bytes);
        group_tag_ = GroupTag{};
    }
    const VkDescriptorBufferInfo tab{moe_tab_->handle(), 0, VK_WHOLE_SIZE};
    // The down projection routes the same ids as gate and up, so their grouping is reused.
    const VkDescriptorBufferInfo idb = bind(ids);
    if (!(group_tag_.ids.buffer == idb.buffer && group_tag_.ids.offset == idb.offset && group_tag_.entries == entries &&
          group_tag_.n_expert == n_expert && group_tag_.chunk == chunk)) {
        const uint32_t gpc[4] = {u32(entries), u32(n_expert), u32(max_tiles), u32(chunk)};
        dispatch(K_MOE_GROUP, {idb, tab}, gpc, sizeof(gpc), u32(n_expert));
        group_tag_ = GroupTag{idb, entries, n_expert, chunk};
    }
    if (max_tiles > dev_->props.limits.maxComputeWorkGroupCount[1])
        throw std::runtime_error("vulkan: dispatch exceeds the workgroup count limit");
    const uint32_t order0 = u32(4 * max_tiles);
    if (!tile) {
        // Enough generated tokens that experts repeat: each run of one expert's entries a workgroup row of the row kernel's wide build, a column per entry, so the expert's rows are read once per run.
        // A column computes the same in either build and as it would alone, so an entry does not depend on what else is routed beside it.
        const RowPlan plan = row_plan(type, nin, dtype);
        const VkDescriptorBufferInfo xqi = row_twin(X, plan, xcols * nin, dtype);
        row_dispatch(plan, live, X, xqi, nin, xcols, 0, kRowColsWide, false, 2, u32(per), max_tiles, bind(ids), order0, tab, entries);
        return;
    }
    if (integer) {
        const bool i8 = dtype == Dtype::int8 && weight_kernels(type)->int8_tile != K_COUNT;
        uint32_t xpad = 0;
        const VkDescriptorBufferInfo xt = tile_twin(X, xcols * nin, nin, i8, xpad);
        const QTile t = qtile(live, max_tiles, nin, i8, narrow);
        const Projection &a = *t.p[0], &b = *t.p[1], &c = *t.p[2];
        const uint32_t pc[16] = {u32(nin), u32(xcols), type, 0, u32(nin / 32), u32(live.size()),
                                 t.nout[0], t.start[0], t.nout[1], t.start[1], t.nout[2], t.start[2], u32(per), order0, u32(xcols), xpad};
        dispatch_qtile(t, {bind(a.out), bind(b.out), bind(c.out)}, xt, tab, pc, u32(max_tiles), {entries * a.rows, entries * b.rows, entries * c.rows});
        record_matrix_path(i8 ? MatrixPath::block_int8 : MatrixPath::block_int16);
    } else {
        for (const Projection* pr : live) {
            const uint32_t height = float_tile_rows_for(dev_->caps, dev_->profile, type, pr->rows, max_tiles, nin);
            const uint32_t pc[10] = {u32(nin), u32(pr->rows), u32(xcols), type, 0, u32(per), order0, u32(nin), u32(max_tiles), u32(nin)};
            const KernelId kernel = float_tile_kernel(height == kTileRowsTall, dtype, type, dev_->f16_inputs);
            dispatch(kernel,
                     {bind(pr->out), bind(pr->weight.data), weight_floats(pr->weight, lookup_grid(type, bind(pr->weight.data))), bind(X), bind(pr->weight.data), tab},
                     pc, sizeof(pc), groups(pr->rows, height), u32(max_tiles), height == kTileRowsSmall ? 1 : 0, {entries * pr->rows});
            record_matrix_path(is_nvfp4_f16_tile(kernel) ? MatrixPath::f16 : dtype == Dtype::bf16 ? MatrixPath::bf16 : MatrixPath::f32);
        }
    }
}

// Throws unless the kernels decode `type` and `data` holds `rows` rows of `nin` values of it, each whole blocks.
// `what` names the operand in the errors; tests/common.py matches the unsupported-type one to skip a model the device has no kernel for.
inline void VulkanBackend::check_matrix(uint32_t type, CSlice data, size_t nin, size_t rows, const char* what) const {
    if (!supports_type(type))
        throw std::runtime_error(std::string("vulkan: unsupported ") + what + " type " + std::to_string(type) +
                                 " (docs/VULKAN.md lists the types the kernels decode)");
    if (bytes_from(data) < size_mul(rows, quant::row_bytes(type, nin)))
        throw std::runtime_error(std::string("vulkan: ") + what + " outside its allocation");
}

// Whether a type's wide matmul goes through the integer-dot tile on this device; profile_for prefers the integer dot only where the device has it.
inline bool VulkanBackend::integer_dot_tile(uint32_t type) const {
    return dev_->profile.prefer_integer_dot && weight_kernels(type)->integer_tile != K_COUNT;
}

inline VulkanBackend::QTile VulkanBackend::qtile(const std::vector<const Projection*>& ps, size_t column_groups, size_t nin, bool i8, bool narrow) const {
    QTile t;
    for (const Projection* pr : ps) t.rows += pr->rows;
    t.height = narrow ? kTileRowsTall : tile_rows_for(dev_->caps, dev_->profile, kTileRowsSmall, kTileRowsShort, kTileRowsTall, t.rows, column_groups, nin);
    const WeightKernels& format = *weight_kernels(ps[0]->weight.type);
    const KernelId tile = i8 ? format.int8_tile : format.integer_tile, tall = i8 ? format.int8_tall_tile : format.integer_tall_tile;
    // Types with one integer module keep the short height to bound live half sums.
    if (tile == tall && t.height == kTileRowsTall) t.height = kTileRowsShort;
    t.kernel = t.height == kTileRowsTall ? tall : tile;
    t.variant = narrow ? 2 : t.height == kTileRowsSmall ? 1 : 0;
    for (size_t i = 0; i < 3; ++i) {
        t.p[i] = i < ps.size() ? ps[i] : ps[0];
        if (i >= ps.size()) continue;
        t.start[i] = u32(t.gx);
        t.nout[i] = u32(ps[i]->rows);
        t.gx += groups(ps[i]->rows, t.height);
    }
    return t;
}

// Ordinary and routed integer tiles share bindings; dictionary formats add their immutable table.
// `reach` is each output's floats, zero for a split call's partial sums.
inline void VulkanBackend::dispatch_qtile(const QTile& tile, std::array<VkDescriptorBufferInfo, 3> output,
                                          VkDescriptorBufferInfo activation, VkDescriptorBufferInfo grouping,
                                          const uint32_t (&pc)[16], uint32_t groups_y, std::array<size_t, 3> reach) {
    const Projection &a = *tile.p[0], &b = *tile.p[1], &c = *tile.p[2];
    const int variant = tile.variant;
    if (a.weight.type == quant::GGML_TYPE_IQ1_S || a.weight.type == quant::GGML_TYPE_IQ1_M || a.weight.type == quant::GGML_TYPE_IQ3_S || a.weight.type == quant::GGML_TYPE_IQ2_S || a.weight.type == quant::GGML_TYPE_IQ3_XXS || a.weight.type == quant::GGML_TYPE_IQ2_XXS || a.weight.type == quant::GGML_TYPE_IQ2_XS)
        dispatch(tile.kernel, {output[0], output[1], output[2], bind(a.weight.data), bind(b.weight.data), bind(c.weight.data), activation, activation, grouping,
                               lookup_grid(a.weight.type, activation)}, pc, sizeof(pc), u32(tile.gx), groups_y, variant, {reach[0], reach[1], reach[2]});
    else
        dispatch(tile.kernel, {output[0], output[1], output[2], bind(a.weight.data), bind(b.weight.data), bind(c.weight.data), activation, activation, grouping},
                 pc, sizeof(pc), u32(tile.gx), groups_y, variant, {reach[0], reach[1], reach[2]});
}

// One bounded stream-ordered scratch holds this call's exact float-scale copies.
inline void VulkanBackend::copy_mxfp4(CSlice data, size_t rows, size_t nin, size_t at) {
    const size_t blocks = size_mul(rows, nin / 32);
    if (!mx_copy_) grow(mx_copy_, kMxCopyBytes);
    const VkDescriptorBufferInfo output{mx_copy_->handle(), at, VK_WHOLE_SIZE};
    const uint32_t pc[] = {u32(blocks)};
    const size_t total = (size_mul(blocks, kMxCopyBlock / 4) + 255) / 256;
    const size_t gx = std::min(total, size_t(dev_->props.limits.maxComputeWorkGroupCount[0]));
    dispatch(K_COPY_MXFP4, {bind(data), output}, pc, sizeof(pc), u32(gx), u32(groups(total, gx)));
}

// Split below the profile's workgroup target, retaining at least tile_split_min_blocks per part; per_cu overrides the target.
inline size_t VulkanBackend::split_blocks(size_t workgroups, size_t nblk, uint32_t per_cu) const {
    const bool narrow = nblk * 32 < dev_->profile.tile_narrow_nin;
    const size_t target = (size_t)(per_cu ? per_cu : narrow ? dev_->profile.tile_split_per_cu_narrow : dev_->profile.tile_split_per_cu) *
                          dev_->caps.compute_units;
    const size_t floor_blocks = std::max<size_t>(dev_->profile.tile_split_min_blocks, 2);
    if (workgroups >= target) return nblk;
    size_t parts = std::min((target + workgroups - 1) / workgroups, std::max<size_t>(1, nblk / floor_blocks));
    if (parts <= 1) return nblk;
    size_t kper = (nblk + parts - 1) / parts;
    kper = (kper + 1) & ~size_t(1);   // a split call's parts are a multiple of the tile's STEP of two
    return std::min(kper, nblk);
}

// Whether all runs take one integer-dot tile call, independent of weight type.
// Such producers write the twin four values per lane; other batches retain the separate quantization pass (docs/src/backends-vulkan.md).
inline bool VulkanBackend::tile_reads(size_t nin, size_t nbatch, RowRuns runs) const {
    const DeviceProfile& p = dev_->profile;
    if (!p.prefer_integer_dot || nin % 32 || !runs.n || runs.runs[runs.n - 1].end != nbatch) return false;
    const size_t from = std::max({tile_from_for(p, true, nin), tile_from_for(p, false, nin), p.moe_tile_from,
                                  p.moe_tile_from_q4, p.moe_tile_from_q4k, p.moe_tile_from_q5k});
    const size_t split = split_tiles_of(runs.runs[0].extent);
    for (size_t i = 0; i < runs.n; ++i)
        if (runs.runs[i].extent < from || split_tiles_of(runs.runs[i].extent) != split) return false;
    return true;
}

} // namespace
} // namespace backend
