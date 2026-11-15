// minimax-ops: the CPU forms of MiniMax-M3.1's layer ops against the math of docs/MINIMAX.md in double.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/cpu/cpu_backend.hpp"

namespace {

using backend::BufferPtr;
using backend::CpuBackend;

size_t checked = 0;
const double U = std::ldexp(1.0, -24);
const int kThreads[] = {1, 2, 3, 5, 8};

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checked;
}

void refused(const std::function<void()>& run, const std::string& what, const std::string& text) {
    try {
        run();
    } catch (const std::exception& e) {
        require(std::string(e.what()) == text, what + " refused with \"" + e.what() + "\"");
        return;
    }
    throw std::runtime_error(what + " was not refused");
}

uint32_t bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

uint64_t state = 0x243F6A8885A308D3ull;
uint32_t draw() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return uint32_t(state >> 32);
}

BufferPtr upload(CpuBackend& cpu, const std::vector<float>& v) {
    BufferPtr b = cpu.alloc(std::max<size_t>(v.size(), 1) * sizeof(float), backend::Memory::device);
    if (!v.empty()) cpu.write(*b, 0, v.data(), v.size() * sizeof(float));
    return b;
}

std::vector<float> download(CpuBackend& cpu, const backend::Buffer& b, size_t n) {
    std::vector<float> v(n);
    cpu.read(b, 0, v.data(), n * sizeof(float));
    return v;
}

// route_experts' sigmoid rule against docs/MINIMAX.md (The forward pass) in double: scores sigmoid(x), picks by score + bias with ties to the lower id, weights the unbiased scores over their sum times the scale.
void sigmoid_routing() {
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    const float level[] = {-inf, -20.0f, -30.0f, -2.5f, -0.0f, 0.0f, 0.75f, 3.0f, 12.0f, 31.0f, inf};
    const size_t n_expert = 24, rows = 9;
    std::vector<float> x(rows * n_expert), bias(n_expert);
    for (float& v : x) v = level[draw() % 11];
    for (float& v : bias) v = float(int(draw() % 9) - 4) / 64.0f;
    for (size_t e = 0; e < n_expert; ++e) x[n_expert + e] = 1.0f;
    for (size_t e = 0; e < n_expert; ++e) x[4 * n_expert + e] = float(int(draw() % 2001) - 1000) / 100.0f;
    x[2 * n_expert + 5] = nan;
    CpuBackend cpu;
    BufferPtr xb = upload(cpu, x), bb = upload(cpu, bias);
    for (size_t k : {1, 8, 24})
        for (bool with_bias : {false, true})
            for (bool normalize : {false, true}) {
                const float scale = 2.0f;
                const backend::RouteRule rule = backend::RouteRule::sigmoid(with_bias ? backend::CSlice{bb.get(), 0} : backend::CSlice{}, scale, normalize);
                auto run = [&](size_t first, size_t n) {
                    BufferPtr ib = cpu.alloc(n * k * 4, backend::Memory::device), wb = cpu.alloc(n * k * 4, backend::Memory::device);
                    cpu.route_experts({xb.get(), first * n_expert}, n, n_expert, k, rule, {ib.get(), 0}, {wb.get(), 0});
                    std::vector<float> out = download(cpu, *ib, n * k), w = download(cpu, *wb, n * k);
                    out.insert(out.end(), w.begin(), w.end());
                    return out;
                };
                cpu.set_threads(1);
                const std::vector<float> got = run(0, rows);
                const std::string name = "route_experts sigmoid k " + std::to_string(k) + (with_bias ? " with a bias" : "") + (normalize ? " normalized" : "");
                for (size_t r = 0; r < rows; ++r) {
                    std::vector<double> s(n_expert);
                    std::vector<float> key(n_expert);
                    bool finite = true;
                    for (size_t e = 0; e < n_expert; ++e) {
                        s[e] = 1.0 / (1.0 + std::exp(-double(x[r * n_expert + e])));
                        const float b = float(s[e]) + (with_bias ? bias[e] : 0.0f);
                        finite = finite && std::isfinite(b);
                        key[e] = b;
                    }
                    std::vector<uint32_t> order(n_expert);
                    for (size_t i = 0; i < n_expert; ++i) order[i] = uint32_t(i);
                    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return key[a] > key[b]; });
                    double total = 0.0;
                    for (size_t j = 0; j < k; ++j) total += s[order[j]];
                    for (size_t j = 0; j < k; ++j) {
                        const uint32_t id = bits(got[r * k + j]);
                        const float w = got[rows * k + r * k + j];
                        if (!finite) {
                            require(id == CpuBackend::kNoExpert && bits(w) == 0, name + " row " + std::to_string(r) + " without a candidate");
                            continue;
                        }
                        require(id == order[j], name + " row " + std::to_string(r) + " pick " + std::to_string(j));
                        const double want = (normalize ? s[order[j]] / total : s[order[j]]) * scale;
                        require(std::fabs(w - want) <= (k + 8) * U * std::fabs(want) + 1e-44, name + " row " + std::to_string(r) + " weight " + std::to_string(j));
                    }
                    const std::vector<float> one = run(r, 1);
                    require(!std::memcmp(one.data(), got.data() + r * k, k * 4) && !std::memcmp(one.data() + k, got.data() + rows * k + r * k, k * 4),
                            name + " of row " + std::to_string(r) + " alone");
                }
                for (int t : kThreads) {
                    cpu.set_threads(t);
                    const std::vector<float> again = run(0, rows);
                    require(!std::memcmp(again.data(), got.data(), got.size() * 4), name + " at " + std::to_string(t) + " threads");
                }
            }
    refused([] { backend::RouteRule::sigmoid({}, 2.0f, true).softmax_only("device"); }, "a device without sigmoid scores given them",
            "device: route_experts has no sigmoid scores");
}

// silu_mul's quick_gelu form against docs/MINIMAX.md (The forward pass) in double: g' = min(g, L), u' = clamp(u, -L, L), (u' + 1) * g' * sigmoid(1.702 g'), a limit of 0 clamping nothing and 1 to 8 threads bit for bit.
void quick_gelu() {
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    const float special[] = {-inf, -200.0f, -7.5f, -7.0f, -1.0f, -0.0f, 0.0f, 1e-30f, 0.5f, 6.999f, 7.0f, 7.25f, 90.0f, inf, nan};
    const size_t n = 4099;
    std::vector<float> g(n), u(n);
    for (size_t i = 0; i < n; ++i) {
        g[i] = float(int(draw() % 24001) - 12000) / 1000.0f;
        u[i] = float(int(draw() % 24001) - 12000) / 1000.0f;
    }
    for (size_t i = 0; i < 15; ++i)
        for (size_t j = 0; j < 15; ++j) {
            g[i * 15 + j] = special[i];
            u[i * 15 + j] = special[j];
        }
    CpuBackend cpu;
    BufferPtr gb = upload(cpu, g), ub = upload(cpu, u), ob = cpu.alloc(n * 4, backend::Memory::device);
    for (float limit : {7.0f, 0.0f}) {
        const std::string name = "silu_mul quick_gelu at limit " + std::to_string(limit);
        cpu.set_threads(1);
        cpu.silu_mul({ob.get(), 0}, {gb.get(), 0}, {ub.get(), 0}, n, limit, {}, backend::GluForm::quick_gelu);
        const std::vector<float> got = download(cpu, *ob, n);
        for (size_t i = 0; i < n; ++i) {
            double gd = g[i], ud = u[i];
            if (limit > 0.0f) {
                ud = ud < -limit ? -limit : ud > limit ? limit : ud;
                gd = gd > limit ? limit : gd;
            }
            // The GLU half is an F32 value in the op, so the reference rounds it to that class, and an underflow to zero meets an infinite up as F32 arithmetic does.
            const double want = (ud + 1.0) * double(float(gd / (1.0 + std::exp(-1.702 * gd))));
            const std::string at = name + " at " + std::to_string(g[i]) + ", " + std::to_string(u[i]);
            if (std::isnan(want)) require(std::isnan(got[i]), at + " is NaN");
            else if (std::isinf(want)) require(got[i] == float(want), at + " is " + std::to_string(want));
            else require(std::fabs(got[i] - want) <= (8 + 2 * std::fabs(1.702 * gd)) * U * std::fabs(want) + 1e-37, at + ": " + std::to_string(got[i]) + " against " + std::to_string(want));
        }
        for (int t : kThreads) {
            cpu.set_threads(t);
            cpu.silu_mul({ob.get(), 0}, {gb.get(), 0}, {ub.get(), 0}, n, limit, {}, backend::GluForm::quick_gelu);
            require(!std::memcmp(download(cpu, *ob, n).data(), got.data(), n * 4), name + " at " + std::to_string(t) + " threads");
        }
    }
}

// indexer_scores's raw form against docs/MINIMAX.md (The forward pass) in double: per head the largest dot of q and a key over a block's positions up to the row's own, -inf past them.
// Each score is within the dot's F32 bound of the exact maximum, a NaN key makes its block NaN, and 1 to 8 threads agree bit for bit.
void raw_index_scores() {
    const size_t layers = 2, heads = 3, dim = 16, bt = 128;
    const float inf = std::numeric_limits<float>::infinity();
    CpuBackend cpu;
    std::unique_ptr<backend::KVStorage> owner = cpu.kv_alloc(layers, 1, 2, 4 * bt, backend::KVType::f32, backend::KVType::f32, {0, dim});
    const int32_t a_ids[] = {2, 0}, b_ids[] = {1};
    const size_t a_len = 150, a_nq = 4, b_nq = 5;
    // The histories, every position's key written once, then the rows that score: view a's last a_nq positions, crossing storage blocks, and all of b's.
    const backend::KVView fill[] = {{owner.get(), a_ids, 2, 0, a_len + a_nq}, {owner.get(), b_ids, 1, 0, b_nq}};
    const size_t positions = a_len + a_nq + b_nq;
    std::vector<float> keys(positions * dim), kv(positions * 2, 0.0f);
    for (float& x : keys) x = float(int(draw() % 4001) - 2000) / 512.0f;
    for (size_t d = 0; d < dim; ++d) keys[(a_len + 1) * dim + d] = keys[(a_len - 2) * dim + d];   // a tie inside one block of 4
    BufferPtr kb = upload(cpu, kv), keyb = upload(cpu, keys);
    cpu.kv_write(1, fill, 2, {kb.get(), 0}, {kb.get(), 0}, {keyb.get(), 0});
    const backend::KVView views[] = {{owner.get(), a_ids, 2, a_len, a_nq}, {owner.get(), b_ids, 1, 0, b_nq}};
    const size_t rows = a_nq + b_nq;
    std::vector<float> q(rows * heads * dim);
    for (float& x : q) x = float(int(draw() % 4001) - 2000) / 1024.0f;
    BufferPtr qb = upload(cpu, q);
    for (size_t block : {size_t(1), size_t(4), size_t(48), size_t(128)}) {
        const size_t n = (a_len + a_nq + block - 1) / block + 2;
        const std::string name = "raw index scores at block " + std::to_string(block);
        BufferPtr ob = cpu.alloc(rows * heads * n * 4, backend::Memory::device);
        cpu.set_threads(1);
        cpu.indexer_scores({ob.get(), 0}, {qb.get(), 0}, 1, views, 2, {}, 1.0f, heads, dim, backend::ModelFormat::f32, n, {}, 0, block, backend::IndexForm::raw_blocks);
        const std::vector<float> got = download(cpu, *ob, rows * heads * n);
        for (size_t r = 0; r < rows; ++r) {
            const size_t pos = r < a_nq ? a_len + r : r - a_nq, base = r < a_nq ? 0 : a_len + a_nq;
            for (size_t h = 0; h < heads; ++h)
                for (size_t b = 0; b < n; ++b) {
                    const float v = got[(r * heads + h) * n + b];
                    if (b * block > pos) {
                        require(v == -inf, name + ": a block past the row is -inf");
                        continue;
                    }
                    double best = -double(inf), bound = 0.0;
                    for (size_t t = b * block; t < std::min((b + 1) * block, pos + 1); ++t) {
                        double dot = 0.0, mag = 0.0;
                        for (size_t d = 0; d < dim; ++d) {
                            const double x = double(q[(r * heads + h) * dim + d]) * keys[(base + t) * dim + d];
                            dot += x;
                            mag += std::fabs(x);
                        }
                        best = std::max(best, dot);
                        bound = std::max(bound, double(dim) * U * mag);
                    }
                    require(std::fabs(double(v) - best) <= bound, name + ": row " + std::to_string(r) + " head " + std::to_string(h) + " block " + std::to_string(b));
                }
        }
        for (int t : kThreads) {
            cpu.set_threads(t);
            cpu.indexer_scores({ob.get(), 0}, {qb.get(), 0}, 1, views, 2, {}, 1.0f, heads, dim, backend::ModelFormat::f32, n, {}, 0, block, backend::IndexForm::raw_blocks);
            require(!std::memcmp(download(cpu, *ob, rows * heads * n).data(), got.data(), got.size() * 4), name + " at " + std::to_string(t) + " threads");
        }
        refused([&] { cpu.indexer_scores({ob.get(), 0}, {qb.get(), 0}, 1, views, 2, {}, 1.0f, heads, dim, backend::ModelFormat::f32, (a_len + a_nq - 1) / block, {}, 0, block,
                                         backend::IndexForm::raw_blocks); },
                name + " narrower than its blocks", "backend: raw index scores narrower than a row's blocks");
    }
    BufferPtr ob = cpu.alloc(rows * heads * 200 * 4, backend::Memory::device);
    refused([&] { cpu.indexer_scores({ob.get(), 0}, {qb.get(), 0}, 0, views, 2, {}, 1.0f, heads, dim, backend::ModelFormat::f32, 200, {}, 0, 1, backend::IndexForm::raw_blocks); },
            "raw index scores over a layer without index keys", "backend: raw index scores read auxiliary rows its layer does not keep at their width");
    refused([&] { cpu.indexer_scores({ob.get(), 0}, {qb.get(), 0}, 1, views, 2, {qb.get(), 0}, 1.0f, heads, dim, backend::ModelFormat::f32, 200, {}, 0, 1, backend::IndexForm::raw_blocks); },
            "raw index scores with weights", "backend: raw index scores take no weights, candidate blocks or key format but f32");
    // A NaN key component makes its block's score NaN for every head, whichever position of the block it is.
    std::unique_ptr<backend::KVStorage> poisoned = cpu.kv_alloc(layers, 1, 2, 4 * bt, backend::KVType::f32, backend::KVType::f32, {0, dim});
    keys[9 * dim + 5] = std::numeric_limits<float>::quiet_NaN();
    BufferPtr nb = upload(cpu, keys);
    const backend::KVView poison_fill{poisoned.get(), a_ids, 2, 0, a_len + a_nq}, poison{poisoned.get(), a_ids, 2, a_len, a_nq};
    cpu.kv_write(1, &poison_fill, 1, {kb.get(), 0}, {kb.get(), 0}, {nb.get(), 0});
    cpu.set_threads(1);
    cpu.indexer_scores({ob.get(), 0}, {qb.get(), 0}, 1, &poison, 1, {}, 1.0f, heads, dim, backend::ModelFormat::f32, 200, {}, 0, 4, backend::IndexForm::raw_blocks);
    const std::vector<float> got = download(cpu, *ob, a_nq * heads * 200);
    for (size_t r = 0; r < a_nq; ++r)
        for (size_t h = 0; h < heads; ++h)
            for (size_t b = 0; b < 40; ++b) require(std::isnan(got[(r * heads + h) * 200 + b]) == (b == 2), "a NaN key makes only its block NaN");
}

// candidate_blocks over the raw block maxima, block size 1 and lengths in blocks, against HF's selection: the row's own block pinned, the K best by score then lower index, -inf dropped, ascending ids.
// At B 128 and K 16 a row keeps every block through position 2047 and drops one first at 2048; B 4 and K 3 is the tiny fixture's shape.
void block_selection() {
    const size_t heads = 2, dim = 8, bt = 128, total = 2049, short_nq = 5;
    const float inf = std::numeric_limits<float>::infinity();
    CpuBackend cpu;
    std::unique_ptr<backend::KVStorage> owner = cpu.kv_alloc(2, 1, 2, 18 * bt, backend::KVType::f32, backend::KVType::f32, {0, dim});
    std::vector<int32_t> a_ids(17);
    for (size_t i = 0; i < a_ids.size(); ++i) a_ids[i] = int32_t((i * 5 + 3) % 17);
    const int32_t b_ids[] = {17};
    const backend::KVView fill[] = {{owner.get(), a_ids.data(), 17, 0, total}, {owner.get(), b_ids, 1, 0, short_nq}};
    std::vector<float> keys((total + short_nq) * dim), kv((total + short_nq) * 2, 0.0f);
    for (float& x : keys) x = float(int(draw() % 4001) - 2000) / 512.0f;
    BufferPtr kb = upload(cpu, kv), keyb = upload(cpu, keys);
    cpu.kv_write(1, fill, 2, {kb.get(), 0}, {kb.get(), 0}, {keyb.get(), 0});
    const backend::KVView views[] = {{owner.get(), a_ids.data(), 17, total - 3, 3}, {owner.get(), b_ids, 1, 0, short_nq}};
    const size_t rows = 3 + short_nq;
    std::vector<float> q(rows * heads * dim);
    for (float& x : q) x = float(int(draw() % 4001) - 2000) / 1024.0f;
    BufferPtr qb = upload(cpu, q);
    for (const std::pair<size_t, size_t>& shape : {std::make_pair(size_t(128), size_t(16)), std::make_pair(size_t(4), size_t(3))}) {
        const size_t block = shape.first, k = shape.second, n = (total + block - 1) / block + 1;
        const std::string name = "block selection at B " + std::to_string(block) + " and K " + std::to_string(k);
        BufferPtr sb = cpu.alloc(rows * heads * n * 4, backend::Memory::device), pb = cpu.alloc(rows * heads * k * 4, backend::Memory::device);
        cpu.indexer_scores({sb.get(), 0}, {qb.get(), 0}, 1, views, 2, {}, 1.0f, heads, dim, backend::ModelFormat::f32, n, {}, 0, block, backend::IndexForm::raw_blocks);
        const std::vector<float> scores = download(cpu, *sb, rows * heads * n);
        std::vector<float> lens(rows * heads);
        for (size_t r = 0; r < rows; ++r) {
            const uint32_t len = uint32_t((r < 3 ? total - 3 + r : r - 3) / block + 1);
            for (size_t h = 0; h < heads; ++h) std::memcpy(&lens[r * heads + h], &len, 4);
        }
        BufferPtr lb = upload(cpu, lens);
        cpu.candidate_blocks({pb.get(), 0}, {sb.get(), 0}, {lb.get(), 0}, rows * heads, n, 1, k);
        const std::vector<float> picks = download(cpu, *pb, rows * heads * k);
        for (size_t r = 0; r < rows; ++r) {
            const size_t pos = r < 3 ? total - 3 + r : r - 3, own = pos / block;
            for (size_t h = 0; h < heads; ++h) {
                std::vector<std::pair<float, uint32_t>> order;
                for (size_t b = 0; b < n; ++b) order.push_back({b == own ? inf : scores[(r * heads + h) * n + b], uint32_t(b)});
                std::stable_sort(order.begin(), order.end(), [](const std::pair<float, uint32_t>& x, const std::pair<float, uint32_t>& y) { return x.first > y.first; });
                std::vector<uint32_t> want;
                for (size_t i = 0; i < k && i < order.size(); ++i)
                    if (order[i].first > -inf) want.push_back(order[i].second);
                std::sort(want.begin(), want.end());
                require(want.size() == std::min(k, own + 1), name + ": position " + std::to_string(pos) + " keeps min(K, its blocks)");
                for (size_t i = 0; i < k; ++i) {
                    uint32_t got;
                    std::memcpy(&got, &picks[(r * heads + h) * k + i], 4);
                    require(got == (i < want.size() ? want[i] : UINT32_MAX), name + ": position " + std::to_string(pos) + " head " + std::to_string(h) + " pick " + std::to_string(i));
                }
            }
        }
    }
}

// sparse_attention's id groups and blocks against docs/MINIMAX.md (The forward pass) in double, the query heads of KV group h attending the positions of group h's blocks up to their own.
// A blocked call has the bits of the call naming each position alone, and a grouped call each group's heads' bits of a call of that group's ids alone, since the chain over slots is today's.
void blocked_attention() {
    const size_t heads = 8, kv_heads = 4, dim = 16, bt = 128, total = 300;
    CpuBackend cpu;
    std::unique_ptr<backend::KVStorage> owner = cpu.kv_alloc(1, kv_heads, dim, 4 * bt, backend::KVType::f32, backend::KVType::f32);
    const int32_t ids_table[] = {3, 0, 2};
    const backend::KVView fill{owner.get(), ids_table, 3, 0, total};
    std::vector<float> k(total * kv_heads * dim), v(total * kv_heads * dim);
    for (float& x : k) x = float(int(draw() % 4001) - 2000) / 1024.0f;
    for (float& x : v) x = float(int(draw() % 4001) - 2000) / 1024.0f;
    BufferPtr kb = upload(cpu, k), vb = upload(cpu, v);
    cpu.kv_write(0, &fill, 1, {kb.get(), 0}, {vb.get(), 0});
    const size_t nq = 3, first_row = total - nq;
    const backend::KVView view{owner.get(), ids_table, 3, first_row, nq};
    std::vector<float> q(nq * heads * dim);
    for (float& x : q) x = float(int(draw() % 4001) - 2000) / 512.0f;
    BufferPtr qb = upload(cpu, q);
    const uint32_t none = UINT32_MAX;
    for (size_t block : {size_t(4), size_t(128)}) {
        const size_t stride = 3, groups = kv_heads, own = (total - 1) / block;
        // Group g of row r keeps earlier blocks, one in groups above 0 and two in group 0, then the row's own block, causal inside it, and padding.
        std::vector<uint32_t> ids(nq * groups * stride);
        for (size_t r = 0; r < nq; ++r)
            for (size_t g = 0; g < groups; ++g) {
                uint32_t* e = &ids[(r * groups + g) * stride];
                e[0] = uint32_t((g * 7 + r) % (own - 1));
                e[1] = g == 0 ? e[0] + 1 : uint32_t(own);
                e[2] = g == 0 ? uint32_t(own) : none;
            }
        const std::string name = "sparse attention at block " + std::to_string(block);
        std::vector<float> idf(ids.size());
        std::memcpy(idf.data(), ids.data(), ids.size() * 4);
        BufferPtr ib = upload(cpu, idf), ob = cpu.alloc(nq * heads * dim * 4, backend::Memory::device);
        cpu.set_threads(1);
        cpu.sparse_attention({qb.get(), 0}, 0, &view, 1, {ib.get(), 0}, stride, {ob.get(), 0}, int(heads), int(kv_heads), int(dim), groups, block);
        const std::vector<float> got = download(cpu, *ob, nq * heads * dim);
        for (int t : kThreads) {
            cpu.set_threads(t);
            cpu.sparse_attention({qb.get(), 0}, 0, &view, 1, {ib.get(), 0}, stride, {ob.get(), 0}, int(heads), int(kv_heads), int(dim), groups, block);
            require(!std::memcmp(download(cpu, *ob, got.size()).data(), got.data(), got.size() * 4), name + " at " + std::to_string(t) + " threads");
        }
        cpu.set_threads(1);
        for (size_t g = 0; g < groups; ++g) {
            // Group g's ids alone, each block spelled out a position a slot, through today's call.
            std::vector<float> spelled(nq * stride * block);
            for (size_t r = 0; r < nq; ++r)
                for (size_t e = 0; e < stride; ++e)
                    for (size_t o = 0; o < block; ++o) {
                        const uint32_t id = ids[(r * groups + g) * stride + e], pos = id == none ? none : uint32_t(id * block + o);
                        std::memcpy(&spelled[(r * stride + e) * block + o], &pos, 4);
                    }
            BufferPtr sb = upload(cpu, spelled), alone = cpu.alloc(nq * heads * dim * 4, backend::Memory::device);
            static_cast<backend::Backend&>(cpu).sparse_attention({qb.get(), 0}, 0, &view, 1, {sb.get(), 0}, stride * block, {alone.get(), 0}, int(heads), int(kv_heads), int(dim));
            const std::vector<float> ref = download(cpu, *alone, nq * heads * dim);
            for (size_t r = 0; r < nq; ++r)
                for (size_t h = g * heads / groups; h < (g + 1) * heads / groups; ++h) {
                    require(!std::memcmp(&got[(r * heads + h) * dim], &ref[(r * heads + h) * dim], dim * 4), name + ": group " + std::to_string(g) + " head " + std::to_string(h) + " has the spelled call's bits");
                    // The double reference: softmax of q . k / sqrt(dim) over the group's visible positions, in any order.
                    const size_t p = first_row + r, kvh = h / (heads / kv_heads);
                    std::vector<size_t> seen;
                    for (size_t e = 0; e < stride; ++e) {
                        const uint32_t id = ids[(r * groups + g) * stride + e];
                        for (size_t o = 0; id != none && o < block; ++o)
                            if (id * block + o <= p) seen.push_back(id * block + o);
                    }
                    std::vector<double> sc;
                    double top = -1e300;
                    for (size_t t : seen) {
                        double d = 0.0;
                        for (size_t i = 0; i < dim; ++i) d += double(q[(r * heads + h) * dim + i]) * k[(t * kv_heads + kvh) * dim + i];
                        sc.push_back(d / std::sqrt(double(dim)));
                        top = std::max(top, sc.back());
                    }
                    double den = 0.0;
                    for (double x : sc) den += std::exp(x - top);
                    for (size_t i = 0; i < dim; ++i) {
                        double want = 0.0, mag = 0.0;
                        for (size_t m = 0; m < seen.size(); ++m) {
                            want += std::exp(sc[m] - top) / den * v[(seen[m] * kv_heads + kvh) * dim + i];
                            mag += std::exp(sc[m] - top) / den * std::fabs(v[(seen[m] * kv_heads + kvh) * dim + i]);
                        }
                        require(std::fabs(got[(r * heads + h) * dim + i] - want) <= 1e-4 * (mag + 1e-3), name + ": group " + std::to_string(g) + " head " + std::to_string(h) + " against double");
                    }
                }
        }
    }
    BufferPtr ib = upload(cpu, std::vector<float>(nq * 3 * 3)), ob = cpu.alloc(nq * heads * dim * 4, backend::Memory::device);
    refused([&] { cpu.sparse_attention({qb.get(), 0}, 0, &view, 1, {ib.get(), 0}, 3, {ob.get(), 0}, int(heads), int(kv_heads), int(dim), 3, 1); },
            "id groups not dividing the KV heads", "backend: invalid sparse attention dimensions");
    refused([&] { cpu.sparse_attention({qb.get(), 0}, 0, &view, 1, {ib.get(), 0}, 3, {ob.get(), 0}, int(heads), int(kv_heads), int(dim), 1, 0); },
            "a block of no positions", "backend: invalid sparse attention dimensions");
}

} // namespace

int main() {
    try {
        sigmoid_routing();
        quick_gelu();
        raw_index_scores();
        block_selection();
        blocked_attention();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "minimax-ops: " << checked << " checks passed" << std::endl;
    return 0;
}
