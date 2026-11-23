#pragma once
// A DSpine drafter's device ops held to the CPU's, shared by the GPU backends' tests: windowed attention (KVView::window) and the four ops of its block (Backend::unit_rows to chain_pick).
#include "backends/backend.hpp"
#include "backends/kv_storage.hpp"
#include "model/kv_cache.hpp"
#include "row_classes.hpp"
#include "twin_checks.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
namespace dspine_checks {
using backend::Backend;
using backend::KVType;
using backend::KVView;
using twin_checks::Launches;
using twin_checks::Require;
using twin_checks::read;

// The window builds a backend runs for heads 64, 128 and 256 wide over f16 sides, by the names its launches report.
using WindowBuilds = std::array<std::string, 3>;
inline const WindowBuilds kGpuWindowBuilds{"llmx_attention_chunk_window", "llmx_attention_vec_window", "llmx_attention_vec256_window"};

// A drafter block's 16 rows after `hist` positions under windows below, at and past the history, heads 64 (the chunk kernel), 128 and 256 wide (the per-row kernels).
inline void window(Backend& g, Backend& c, const Launches& take, const Require& require, const WindowBuilds& builds = kGpuWindowBuilds) {
    const int n_head = 4, n_head_kv = 2;
    const size_t rows = 16;
    for (const int dim : {64, 128, 256})
        for (const size_t hist : {size_t(0), size_t(40), size_t(300)})
            for (const size_t win : {size_t(7), size_t(64), size_t(1000)}) {
                const size_t qw = size_t(n_head) * dim, kvw = size_t(n_head_kv) * dim, end = hist + rows;
                const std::vector<float> keys = row_classes::values(end * kvw, 71), vals = row_classes::values(end * kvw, 72);
                const std::vector<float> q = row_classes::values(rows * qw, 73, 4.0f);
                const std::string what = "windowed attention, heads " + std::to_string(dim) + " wide, history " + std::to_string(hist) + ", window " + std::to_string(win);
                std::vector<std::vector<float>> out(2);
                Backend* bs[2] = {&g, &c};
                for (int k = 0; k < 2; ++k) {
                    Backend& b = *bs[k];
                    auto kb = b.adopt(keys.data(), keys.size() * 4), vb = b.adopt(vals.data(), vals.size() * 4);
                    auto qb = b.adopt(q.data(), q.size() * 4), ob = b.alloc(rows * qw * 4), o1 = b.alloc(qw * 4), o2 = b.alloc(qw * 4);
                    auto st = b.kv_alloc(1, n_head_kv, dim, 2048, KVType::f16, KVType::f16);
                    infer::BlockPool pool;
                    pool.configure(st->max_blocks());
                    infer::KVSequence s(&pool, b.kv_layout().block_tokens), span(&pool, b.kv_layout().block_tokens);
                    s.prepare(end);
                    KVView all = s.view(st.get());
                    b.kv_write(0, &all, 1, {kb.get(), 0}, {vb.get(), 0});
                    KVView v = all;
                    v.length = hist;
                    v.nq = rows;
                    v.extent = rows;
                    v.window = win;
                    if (k == 0) take(b);
                    b.attention({qb.get(), 0}, 0, &v, 1, {ob.get(), 0}, n_head, n_head_kv, dim);
                    out[k] = read(b, *ob, rows * qw);
                    if (k == 0) {
                        const auto ran = take(b);
                        const std::string& body = builds[dim == 64 ? 0 : dim == 128 ? 1 : 2];
                        require(ran.count(body) == (end > win), what + (end > win ? " runs its window build" : " runs no window build, since no row starts past 0"));
                        bool alone = true;
                        for (size_t r = 0; r < rows; ++r) {
                            KVView one = v;
                            one.length = hist + r;
                            one.nq = 1;
                            b.attention({qb.get(), r * qw}, 0, &one, 1, {o1.get(), 0}, n_head, n_head_kv, dim);
                            alone = alone && std::memcmp(read(b, *o1, qw).data(), out[0].data() + r * qw, qw * 4) == 0;
                        }
                        require(alone, what + ": each row alone gives its bits in the block");
                        // The last row over a sequence of its window's positions alone, unwindowed, gives the same bits: a row's result follows its span.
                        const size_t begin = end > win ? end - win : 0, n = end - begin;
                        span.prepare(n);
                        KVView sv = span.view(st.get());
                        b.kv_write(0, &sv, 1, {kb.get(), begin * kvw}, {vb.get(), begin * kvw});
                        sv.length = n - 1;
                        sv.nq = 1;
                        sv.extent = rows;
                        KVView last = v;
                        last.length = end - 1;
                        last.nq = 1;
                        b.attention({qb.get(), (rows - 1) * qw}, 0, &last, 1, {o1.get(), 0}, n_head, n_head_kv, dim);
                        b.attention({qb.get(), (rows - 1) * qw}, 0, &sv, 1, {o2.get(), 0}, n_head, n_head_kv, dim);
                        require(std::memcmp(read(b, *o1, qw).data(), read(b, *o2, qw).data(), qw * 4) == 0, what + ": the last row equals its window's positions alone, bit for bit");
                        span.abort();
                    }
                    s.abort();
                }
                require(twin_checks::f32_close(out[0], out[1]), what + ", against the CPU");
            }
}

// The largest distance in units of the last place between two runs' floats, the same bits 0; a NaN on either side counts as far.
inline uint32_t max_ulps(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return UINT32_MAX;
    uint32_t worst = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != a[i] || b[i] != b[i]) return UINT32_MAX;
        int32_t x, y;
        std::memcpy(&x, &a[i], 4);
        std::memcpy(&y, &b[i], 4);
        const int64_t ox = x < 0 ? int64_t(INT32_MIN) - x : x, oy = y < 0 ? int64_t(INT32_MIN) - y : y;
        worst = std::max<uint32_t>(worst, uint32_t(std::min<int64_t>(UINT32_MAX, ox > oy ? ox - oy : oy - ox)));
    }
    return worst;
}

// The drafter's four ops against the CPU on the same inputs, within `bound` units of the last place each (chain_pick always bit for bit), at the 27B's widths and a narrow one.
struct Bounds {
    uint32_t unit_rows, sigmoid_gate_rows, matmul_at_ids;
};
inline void ops(Backend& g, Backend& c, const Require& require, Bounds bound) {
    Backend* bs[2] = {&g, &c};
    // unit_rows: every mode, in place and apart, with and without norms.
    for (const size_t n : {size_t(1024), size_t(5120), size_t(37)})
        for (const backend::RowScale mode : {backend::RowScale::rms, backend::RowScale::l2, backend::RowScale::none})
            for (const bool in_place : {false, true}) {
                const size_t rows = 48;
                const std::vector<float> x = row_classes::values(rows * n, 81, 3.0f);
                std::vector<std::vector<float>> out(2), norms(2);
                for (int k = 0; k < 2; ++k) {
                    Backend& b = *bs[k];
                    auto xb = b.adopt(x.data(), x.size() * 4), yb = b.alloc(rows * n * 4), nb = b.alloc(rows * 4);
                    const backend::Slice dst = in_place ? backend::Slice{xb.get(), 0} : backend::Slice{yb.get(), 0};
                    b.unit_rows(dst, mode == backend::RowScale::none ? backend::Slice{} : backend::Slice{nb.get(), 0}, {xb.get(), 0}, rows, n, 32.0f, 1e-6f, mode);
                    out[k] = read(b, *dst.buffer, rows * n);
                    if (mode != backend::RowScale::none) norms[k] = read(b, *nb, rows);
                }
                const std::string what = "unit_rows " + std::to_string(n) + " wide, mode " + std::to_string(int(mode)) + (in_place ? ", in place" : "");
                require(max_ulps(out[0], out[1]) <= bound.unit_rows && max_ulps(norms[0], norms[1]) <= bound.unit_rows, what + ", against the CPU within its bound");
            }
    // sigmoid_gate_rows: rows with no sender, repeated senders and arguments past the exponent's range.
    {
        const size_t rows = 33, width = 512, src = 40;
        std::vector<float> q = row_classes::values(src * width, 82, 4.0f), s = row_classes::values(src * width, 83, 4.0f), v = row_classes::values(src * width, 84),
                           bias = row_classes::values(width, 85), scale = row_classes::values(src, 86);
        for (size_t i = 0; i < width; i += 7) q[i] = i % 2 ? 95.0f : -95.0f;
        std::vector<uint32_t> q_rows(rows), from(rows);
        for (size_t r = 0; r < rows; ++r) {
            q_rows[r] = uint32_t((r * 7) % src);
            from[r] = r % 16 == 0 ? UINT32_MAX : uint32_t((r * 5 + 3) % src);
        }
        std::vector<std::vector<float>> out(2);
        for (int k = 0; k < 2; ++k) {
            Backend& b = *bs[k];
            auto qb = b.adopt(q.data(), q.size() * 4), sb = b.adopt(s.data(), s.size() * 4), vb = b.adopt(v.data(), v.size() * 4), bb = b.adopt(bias.data(), bias.size() * 4),
                 gb = b.adopt(scale.data(), scale.size() * 4), mb = b.alloc(rows * width * 4);
            b.sigmoid_gate_rows({mb.get(), 0}, {qb.get(), 0}, {sb.get(), 0}, {vb.get(), 0}, {bb.get(), 0}, {gb.get(), 0}, rows, width, q_rows.data(), from.data());
            out[k] = read(b, *mb, rows * width);
        }
        require(max_ulps(out[0], out[1]) <= bound.sigmoid_gate_rows, "sigmoid_gate_rows against the CPU within its bound");
    }
    // matmul_at_ids: one group's row alone and every candidate after each, an id past the table scoring -inf.
    for (const size_t width : {size_t(5120), size_t(96)})
        for (const size_t per : {size_t(1), size_t(16)}) {
            const size_t groups = 3, k = 16, nrows = 300;
            const std::vector<float> x = row_classes::values(groups * per * width, 87), w = row_classes::values(width, 88), head = row_classes::values(nrows * width, 89);
            std::vector<uint32_t> ids(groups * k);
            for (size_t i = 0; i < ids.size(); ++i) ids[i] = i == 5 ? uint32_t(nrows) : uint32_t((i * 37) % nrows);
            std::vector<std::vector<float>> out(2);
            for (int kk = 0; kk < 2; ++kk) {
                Backend& b = *bs[kk];
                auto xb = b.adopt(x.data(), x.size() * 4), wb = b.adopt(w.data(), w.size() * 4), hb = b.adopt(head.data(), head.size() * 4),
                     ib = b.adopt(ids.data(), ids.size() * 4), ob = b.alloc(groups * k * k * 4);
                const std::vector<float> zero(groups * k * k, 0.0f);
                b.write(*ob, 0, zero.data(), zero.size() * 4);
                b.matmul_at_ids({ob.get(), 0}, {xb.get(), 0}, {wb.get(), 0}, {quant::GGML_TYPE_F32, {hb.get(), 0}}, width, nrows, {ib.get(), 0}, groups, k, per);
                out[kk] = read(b, *ob, groups * k * k);
            }
            require(max_ulps(out[0], out[1]) <= bound.matmul_at_ids,
                    "matmul_at_ids " + std::to_string(width) + " wide, " + std::to_string(per) + " rows a group, against the CPU within its bound");
        }
    // chain_pick: ties to the lower index, the same picks bit for bit.
    {
        const size_t seqs = 3, steps = 15, k = 16;
        std::vector<float> score = row_classes::values(steps * seqs * k * k, 90);
        for (size_t i = 0; i + 3 < score.size(); i += 11) score[i + 3] = score[i];
        std::vector<uint32_t> cand(steps * seqs * k);
        for (size_t i = 0; i < cand.size(); ++i) cand[i] = uint32_t(i * 2654435761u);
        std::vector<std::vector<float>> out(2);
        for (int kk = 0; kk < 2; ++kk) {
            Backend& b = *bs[kk];
            auto sb = b.adopt(score.data(), score.size() * 4), cb = b.adopt(cand.data(), cand.size() * 4), nb = b.alloc(steps * seqs * 4);
            b.chain_pick({nb.get(), 0}, {sb.get(), 0}, {cb.get(), 0}, seqs, steps, k);
            out[kk] = read(b, *nb, steps * seqs);
        }
        require(std::memcmp(out[0].data(), out[1].data(), steps * seqs * 4) == 0, "chain_pick against the CPU, bit for bit");
    }
}
}  // namespace dspine_checks
