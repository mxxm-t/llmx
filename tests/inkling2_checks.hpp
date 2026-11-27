#pragma once
// Inkling-2-Mini's layer ops (docs/INKLING2.md, section 5): their CPU forms against HF's math in double, and a device's forms against the CPU and against themselves, shared by inkling2-ops and the device backends' tests.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/cpu/cpu_backend.hpp"

namespace ink2 {

using backend::BufferPtr;
using backend::CpuBackend;

size_t checked = 0;
const int kThreads[] = {1, 3, 8};

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

uint64_t state = 0x243F6A8885A308D3ull;
float uniform(float lo, float hi) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return lo + (hi - lo) * float(uint32_t(state >> 40)) / float(1u << 24);
}

std::vector<float> random(size_t n, float lo, float hi) {
    std::vector<float> v(n);
    for (float& x : v) x = uniform(lo, hi);
    return v;
}

BufferPtr upload(backend::Backend& cpu, const std::vector<float>& v) {
    BufferPtr b = cpu.alloc(std::max<size_t>(v.size(), 1) * sizeof(float), backend::Memory::device);
    if (!v.empty()) cpu.write(*b, 0, v.data(), v.size() * sizeof(float));
    return b;
}

std::vector<float> download(backend::Backend& cpu, const backend::Buffer& b, size_t n) {
    std::vector<float> v(n);
    cpu.read(b, 0, v.data(), n * sizeof(float));
    return v;
}

// Every element within `tol` * (1 + |reference|) of the double reference, and the untouched floats between strided rows kept.
void close(const std::vector<float>& got, const std::vector<double>& want, double tol, const std::string& what) {
    require(got.size() == want.size(), what + ": size");
    for (size_t i = 0; i < got.size(); ++i)
        if (!(std::fabs(double(got[i]) - want[i]) <= tol * (1.0 + std::fabs(want[i]))))
            throw std::runtime_error(what + ": element " + std::to_string(i) + " is " + std::to_string(got[i]) + ", not " + std::to_string(want[i]));
    ++checked;
}

// l2_clip (HF Inkling2LoopScaledL2Clip, then Inkling2LoopLogScale): heads above and below the threshold, with and without scales, rows strided past their heads.
void l2_clip() {
    const size_t rows = 7, heads = 3, dim = 16, stride = 53;
    const float threshold = 2.5f;
    for (int nt : kThreads)
        for (int scaled = 0; scaled < 2; ++scaled)
            for (int row_scaled = 0; row_scaled < 2; ++row_scaled) {
                CpuBackend cpu;
                cpu.set_threads(nt);
                std::vector<float> x = random(rows * stride, -1.0f, 1.0f), scale = random(heads * dim, 0.2f, 1.5f), tau(rows);
                for (size_t r = 0; r < rows; ++r) tau[r] = 1.0f + 0.1f * std::log(std::max(float(r * 3000 + 1) / 8192.0f, 1.0f));
                std::vector<double> want(x.begin(), x.end());
                for (size_t r = 0; r < rows; ++r)
                    for (size_t h = 0; h < heads; ++h) {
                        double* v = want.data() + r * stride + h * dim;
                        double ss = 0.0;
                        for (size_t d = 0; d < dim; ++d) {
                            if (scaled) v[d] *= scale[h * dim + d];
                            ss += v[d] * v[d];
                        }
                        const double c = std::min(1.0, threshold / (std::sqrt(ss) + 1e-6));
                        for (size_t d = 0; d < dim; ++d) v[d] *= c * (row_scaled ? tau[r] : 1.0);
                    }
                BufferPtr xb = upload(cpu, x), sb = upload(cpu, scale), tb = upload(cpu, tau);
                cpu.l2_clip({{{xb.get(), 0}, scaled ? backend::CSlice{sb.get(), 0} : backend::CSlice{}, row_scaled ? backend::CSlice{tb.get(), 0} : backend::CSlice{},
                              heads, dim, stride, threshold}},
                            rows);
                close(download(cpu, *xb, rows * stride), want, 1e-6, "l2_clip scaled " + std::to_string(scaled) + " row scale " + std::to_string(row_scaled));
            }
    CpuBackend cpu;
    BufferPtr xb = upload(cpu, std::vector<float>(64));
    refused([&] { cpu.l2_clip({{{xb.get(), 0}, {}, {}, 2, 16, 31, 1.0f}}, 2); }, "rows narrower than their heads", "backend: invalid l2 clip dimensions");
    const backend::ClipRows one{{xb.get(), 0}, {}, {}, 1, 16, 16, 1.0f};
    refused([&] { cpu.l2_clip({one, one, one, one, one}, 1); }, "more items than one call takes", "backend: invalid l2 clip dimensions");
}

// query_mix (HF Inkling2LoopBlockAttentionResidual): items of different magnitudes, one to six of them, one to four queries.
void query_mix() {
    const size_t rows = 5, dim = 24, stride = 6 * dim + 5;
    const float eps = 1e-6f;
    for (int nt : kThreads)
        for (size_t n : {size_t(1), size_t(3), size_t(6)})
            for (size_t nq : {size_t(1), size_t(4)}) {
                CpuBackend cpu;
                cpu.set_threads(nt);
                std::vector<float> src = random(rows * stride, -2.0f, 2.0f), queries = random(nq * dim, -3.0f, 3.0f);
                for (size_t r = 0; r < rows; ++r)
                    for (size_t d = 0; d < dim; ++d) src[r * stride + dim + d] *= 40.0f;
                std::vector<double> want(rows * nq * dim);
                for (size_t r = 0; r < rows; ++r) {
                    std::vector<double> normed(n * dim);
                    for (size_t i = 0; i < n; ++i) {
                        double ss = 0.0;
                        for (size_t d = 0; d < dim; ++d) ss += double(src[r * stride + i * dim + d]) * src[r * stride + i * dim + d];
                        for (size_t d = 0; d < dim; ++d) normed[i * dim + d] = src[r * stride + i * dim + d] / std::sqrt(ss / dim + eps);
                    }
                    for (size_t q = 0; q < nq; ++q) {
                        std::vector<double> p(n);
                        double top = -1e300, total = 0.0;
                        for (size_t i = 0; i < n; ++i) {
                            double dot = 0.0;
                            for (size_t d = 0; d < dim; ++d) dot += queries[q * dim + d] * normed[i * dim + d];
                            top = std::max(top, p[i] = dot / std::sqrt(double(dim)));
                        }
                        for (double& v : p) total += (v = std::exp(v - top));
                        for (size_t d = 0; d < dim; ++d) {
                            double acc = 0.0;
                            for (size_t i = 0; i < n; ++i) acc += p[i] / total * normed[i * dim + d];
                            want[(r * nq + q) * dim + d] = acc;
                        }
                    }
                }
                BufferPtr sb = upload(cpu, src), qb = upload(cpu, queries), ob = cpu.alloc(rows * nq * dim * sizeof(float), backend::Memory::device);
                cpu.query_mix({{{ob.get(), 0}, {sb.get(), 0}, {qb.get(), 0}, n, nq}}, rows, dim, stride, eps);
                close(download(cpu, *ob, rows * nq * dim), want, 2e-6, "query_mix of " + std::to_string(n) + " items by " + std::to_string(nq) + " queries");
            }
    CpuBackend cpu;
    BufferPtr b = upload(cpu, std::vector<float>(1024));
    refused([&] { cpu.query_mix({{{b.get(), 0}, {b.get(), 0}, {b.get(), 0}, 17, 1}}, 1, 4, 68, 1e-6f); }, "seventeen items", "backend: invalid query mix dimensions");
    refused([&] { cpu.query_mix({{{b.get(), 0}, {b.get(), 0}, {b.get(), 0}, 3, 1}}, 1, 4, 11, 1e-6f); }, "rows narrower than their items", "backend: invalid query mix dimensions");
    const backend::QueryMixRows one{{b.get(), 0}, {b.get(), 512}, {b.get(), 768}, 2, 1};
    refused([&] { cpu.query_mix({one, one, one}, 1, 4, 8, 1e-6f); }, "more mixes than one call takes", "backend: invalid query mix dimensions");
    refused([&] { cpu.query_mix({{{b.get(), 0}, {b.get(), 512}, {b.get(), 768}, 2, 2, {{{b.get(), 900}}}, 1e-6f}}, 1, 4, 8, 1e-6f); }, "a normed mix of two queries with one norm",
            "backend: a normed query mix without a norm for each query");
}

// tanh_gate_mix (HF Inkling2LoopMessageGate's tail), out written over h and strided rows, saturated gates included.
void tanh_gate_mix() {
    const size_t rows = 6, width = 40, stride = 97;
    for (int nt : kThreads) {
        CpuBackend cpu;
        cpu.set_threads(nt);
        std::vector<float> h = random(rows * stride, -3.0f, 3.0f), m = random(rows * stride, -3.0f, 3.0f), f = random(rows * width, -4.0f, 4.0f), g = random(rows * width, -4.0f, 4.0f);
        f[3] = 30.0f, g[5] = -30.0f;
        std::vector<double> want(h.begin(), h.end());
        for (size_t r = 0; r < rows; ++r)
            for (size_t d = 0; d < width; ++d)
                want[r * stride + d] = (1.0 - std::tanh(double(f[r * width + d]))) * h[r * stride + d] + std::tanh(double(g[r * width + d])) * m[r * stride + d];
        BufferPtr hb = upload(cpu, h), mb = upload(cpu, m), fb = upload(cpu, f), gb = upload(cpu, g);
        cpu.tanh_gate_mix({{{hb.get(), 0}, {hb.get(), 0}, {mb.get(), 0}, {fb.get(), 0}, {gb.get(), 0}}}, rows, width, stride, stride, stride);
        close(download(cpu, *hb, rows * stride), want, 1e-6, "tanh_gate_mix");
    }
}

// depthwise_conv over a ring (HF Inkling2ShortConvolution and its dynamic form) against the whole sequence in double: chunks shorter and longer than the ring, a rewind, a fork and two views.
struct ConvCall {
    size_t src, dst, length, nq;
};

// The ring form of depthwise_conv: x plus the taps, static and optional per-head dynamic ones, over a divisor.
backend::DepthwiseConv ring_conv(backend::CSlice taps, backend::CSlice dyn, size_t C, size_t K, size_t heads, size_t dyn_stride, float divisor, size_t ring_offset,
                                 size_t ring_rows) {
    backend::DepthwiseConv d;
    d.w = taps;
    d.channels = C;
    d.taps = K;
    d.residual_from = backend::ConvResidual::input;
    d.divisor = divisor;
    d.dyn = dyn;
    d.dyn_heads = heads;
    d.dyn_stride = dyn_stride;
    d.history = backend::ConvHistory::ring;
    d.ring_offset = ring_offset;
    d.ring_rows = ring_rows;
    return d;
}

void ring_conv_case(size_t C, bool dynamic, bool in_place, int nt, backend::Backend* device = nullptr) {
    const size_t K = 4, heads = 2, dyn_stride = heads * K + 3, ring_rows = K - 1 + 2, ring_offset = 7, slots = 3;
    const double divisor = dynamic ? 3.0 : std::sqrt(5.0);
    CpuBackend own;
    backend::Backend& cpu = device ? *device : own;
    if (!device) own.set_threads(nt);
    std::unique_ptr<backend::StateStorage> store = cpu.state_alloc(1, slots, backend::StateShape{}, {ring_offset + ring_rows * C + 5});
    const std::vector<float> taps = random(C * K, -1.0f, 1.0f);
    BufferPtr tb = upload(cpu, taps);
    std::vector<std::vector<std::vector<float>>> seq(slots);
    const auto call = [&](const std::vector<ConvCall>& calls) {
        size_t rows = 0;
        for (const ConvCall& c : calls) rows += c.nq;
        const std::vector<float> x = random(rows * C, -2.0f, 2.0f), dyn = random(rows * dyn_stride, -0.5f, 0.5f);
        std::vector<double> want(rows * C);
        std::vector<backend::StateView> views;
        std::vector<std::vector<std::vector<float>>> next = seq;
        for (size_t i = 0, row = 0; i < calls.size(); row += calls[i].nq, ++i) {
            const ConvCall& c = calls[i];
            views.push_back({store.get(), c.src, c.dst, c.length, c.nq});
            std::vector<std::vector<float>> s(seq[c.src].begin(), seq[c.src].begin() + long(c.length));
            for (size_t b = 0; b < c.nq; ++b) s.emplace_back(x.begin() + long((row + b) * C), x.begin() + long((row + b + 1) * C));
            for (size_t b = 0; b < c.nq; ++b) {
                const size_t pos = c.length + b, g = row + b;
                for (size_t ch = 0; ch < C; ++ch) {
                    double acc = 0.0;
                    for (size_t j = 0; j < K && j <= pos; ++j)
                        acc += (double(taps[ch * K + K - 1 - j]) + (dynamic ? double(dyn[g * dyn_stride + (ch / (C / heads)) * K + j]) : 0.0)) * double(s[pos - j][ch]);
                    want[g * C + ch] = (acc + double(x[g * C + ch])) / divisor;
                }
            }
            next[c.dst] = s;
        }
        BufferPtr xb = upload(cpu, x), db = upload(cpu, dyn), apart = upload(cpu, std::vector<float>(rows * C));
        backend::Buffer* ob = in_place ? xb.get() : apart.get();
        cpu.depthwise_conv({ob, 0}, {xb.get(), 0},
                           ring_conv({tb.get(), 0}, dynamic ? backend::CSlice{db.get(), 0} : backend::CSlice{}, C, K, dynamic ? heads : 0, dyn_stride, float(divisor),
                                     ring_offset, ring_rows),
                           0, views.data(), views.size());
        close(download(cpu, *ob, rows * C), want, 2e-6,
              "depthwise_conv over a ring of " + std::to_string(C) + " channels" + (dynamic ? ", dynamic" : "") + (in_place ? ", in place" : ""));
        seq = next;
    };
    call({{0, 0, 0, 3}});
    call({{0, 0, 3, 1}});
    call({{0, 0, 4, 6}});
    call({{0, 0, 10, 1}});
    call({{0, 0, 9, 2}});
    call({{0, 1, 11, 2}});
    call({{0, 0, 11, 1}, {1, 1, 13, 3}});
    call({{1, 1, 14, 1}});
    call({{1, 2, 15, 4}});
    // Rows past a device's staged stretch of 256, then two rows and one row on the ring that pass wrote, at 128 and 256 channels a block.
    call({{2, 2, 19, 300}});
    call({{2, 1, 319, 2}});
    call({{1, 1, 321, 1}});
}

void ring_conv() {
    for (int nt : kThreads)
        for (size_t C : {size_t(10), size_t(130)})
            for (bool dynamic : {false, true})
                for (bool in_place : {false, true}) ring_conv_case(C, dynamic, in_place, nt);
    CpuBackend cpu;
    std::unique_ptr<backend::StateStorage> store = cpu.state_alloc(1, 2, backend::StateShape{}, {40});
    BufferPtr b = upload(cpu, std::vector<float>(1024));
    const backend::StateView v{store.get(), 0, 0, 0, 2};
    refused([&] { cpu.depthwise_conv({b.get(), 0}, {b.get(), 600}, ring_conv({b.get(), 900}, {}, 8, 4, 0, 0, 3.0f, 0, 6), 0, &v, 1); }, "a ring past its slot",
            "backend: depthwise convolution history outside the state slot");
    refused([&] { cpu.depthwise_conv({b.get(), 0}, {b.get(), 600}, ring_conv({b.get(), 900}, {}, 8, 4, 0, 0, 3.0f, 0, 2), 0, &v, 1); }, "a ring shorter than the taps",
            "backend: depthwise convolution ring shorter than its history");
    refused([&] { cpu.depthwise_conv({b.get(), 0}, {b.get(), 4}, ring_conv({b.get(), 900}, {}, 8, 4, 0, 0, 3.0f, 0, 3), 0, &v, 1); }, "an output half over its input",
            "backend: depthwise convolution output must be disjoint or exactly its input or residual");
    refused([&] { cpu.depthwise_conv({b.get(), 0}, {b.get(), 600}, ring_conv({b.get(), 900}, {b.get(), 300}, 8, 4, 3, 12, 3.0f, 0, 3), 0, &v, 1); },
            "heads that do not divide the channels", "backend: invalid depthwise convolution dimensions");
}

// attention_rel_sink (HF eager_attention_forward with Inkling2RelativePositionBias and sinks) against whole sequences in double, over a ring (window, chunks longer than the ring, a rewind, a fork) and over pooled rows across blocks.
// `wide` takes the model's head width, window and bias extent, five query heads a key-value head (a device's four at a time, then one), and a pass longer than a 1024-key chunk.
void attention_rel_sink_case(bool ring, int nt, backend::Backend* device = nullptr, bool wide = false) {
    const size_t heads = wide ? 10 : 4, kv_heads = 2, D = wide ? 128 : 8, R = wide ? 16 : 3, KD = kv_heads * D, HD = heads * D;
    const size_t W = ring ? (wide ? 1024 : 5) : 0, extent = wide ? 1024 : ring ? 4 : 6, ring_rows = wide ? 1087 : 6, ring_offset = 5;
    const float scale = float(1.0 / std::sqrt(double(D)));
    CpuBackend own;
    backend::Backend& cpu = device ? *device : own;
    if (!device) own.set_threads(nt);
    std::unique_ptr<backend::StateStorage> store = cpu.state_alloc(1, 3, backend::StateShape{}, {ring_offset + ring_rows * 2 * KD + 3});
    std::unique_ptr<backend::KVStorage> owned = cpu.kv_alloc(1, 0, 0, wide ? 1280 : 512, backend::KVType::f32, backend::KVType::f32, {0}, {{1, 2 * KD}});
    std::vector<std::vector<int32_t>> blocks = {{2, 0, 1}, {3}};
    if (wide) blocks = {{19, 3, 17, 0, 12, 5, 6, 7, 8, 9, 10, 11, 4, 13, 14, 15, 16, 2, 18, 1}, {}};
    const std::vector<float> table = random(R * extent, -1.0f, 1.0f);
    BufferPtr tb = upload(cpu, table);
    std::vector<std::vector<std::vector<float>>> seq(3);
    const auto call = [&](const std::vector<ConvCall>& calls) {
        size_t rows = 0;
        for (const ConvCall& c : calls) rows += c.nq;
        const std::vector<float> q = random(rows * HD, -1.0f, 1.0f), k = random(rows * KD, -1.0f, 1.0f), v = random(rows * KD, -1.0f, 1.0f);
        const std::vector<float> rel = random(rows * heads * R, -1.0f, 1.0f), sink = random(rows * heads, -2.0f, 2.0f);
        std::vector<double> want(rows * HD);
        std::vector<backend::StateView> rings;
        std::vector<backend::KVView> pool;
        std::vector<std::vector<std::vector<float>>> next = seq;
        for (size_t i = 0, row = 0; i < calls.size(); row += calls[i].nq, ++i) {
            const ConvCall& c = calls[i];
            if (ring) rings.push_back({store.get(), c.src, c.dst, c.length, c.nq});
            else pool.push_back({owned.get(), blocks[c.src].data(), blocks[c.src].size(), c.length, c.nq});
            std::vector<std::vector<float>> s(seq[c.src].begin(), seq[c.src].begin() + long(c.length));
            for (size_t b = 0; b < c.nq; ++b) {
                std::vector<float> kv(k.begin() + long((row + b) * KD), k.begin() + long((row + b + 1) * KD));
                kv.insert(kv.end(), v.begin() + long((row + b) * KD), v.begin() + long((row + b + 1) * KD));
                s.push_back(kv);
            }
            for (size_t b = 0; b < c.nq; ++b) {
                const size_t pos = c.length + b, g = row + b, lo = W && pos + 1 > W ? pos + 1 - W : 0;
                for (size_t h = 0; h < heads; ++h) {
                    const size_t kh = h / (heads / kv_heads);
                    std::vector<double> sc;
                    double top = sink[g * heads + h];
                    for (size_t p = lo; p <= pos; ++p) {
                        double dot = 0.0, bias = 0.0;
                        for (size_t e = 0; e < D; ++e) dot += double(q[g * HD + h * D + e]) * s[p][kh * D + e];
                        if (pos - p < extent)
                            for (size_t r = 0; r < R; ++r) bias += double(rel[(g * heads + h) * R + r]) * table[r * extent + pos - p];
                        sc.push_back(dot * scale + bias);
                        top = std::max(top, sc.back());
                    }
                    double total = std::exp(sink[g * heads + h] - top);
                    for (double& x : sc) total += x = std::exp(x - top);
                    for (size_t e = 0; e < D; ++e) {
                        double o = 0.0;
                        for (size_t p = lo; p <= pos; ++p) o += sc[p - lo] / total * s[p][KD + kh * D + e];
                        want[g * HD + h * D + e] = o;
                    }
                }
            }
            next[c.dst] = s;
        }
        BufferPtr qb = upload(cpu, q), kb = upload(cpu, k), vb = upload(cpu, v), rb = upload(cpu, rel), sb = upload(cpu, sink), ob = upload(cpu, std::vector<float>(rows * HD));
        cpu.attention_rel_sink({ob.get(), 0}, {qb.get(), 0}, {kb.get(), 0}, {vb.get(), 0}, {rb.get(), 0}, {tb.get(), 0}, {sb.get(), 0}, heads, kv_heads, D, R, extent, W, scale, 0,
                               ring ? rings.data() : nullptr, ring ? nullptr : pool.data(), calls.size(), ring_offset, ring_rows);
        close(download(cpu, *ob, rows * HD), want, 2e-6, std::string("attention_rel_sink over ") + (ring ? "a ring" : "pooled rows"));
        seq = next;
    };
    if (wide) {
        call({{0, 0, 0, 1100}});
        call({{0, size_t(ring ? 1 : 0), 1100, 1}});
        if (ring) call({{1, 1, 1101, 30}, {0, 2, 1100, 2}});
        return;
    }
    if (ring) {
        call({{0, 0, 0, 3}});
        call({{0, 0, 3, 9}});
        call({{0, 0, 12, 1}});
        call({{0, 0, 11, 2}});
        call({{0, 1, 13, 2}});
        call({{0, 0, 13, 1}, {1, 1, 15, 2}});
        call({{1, 2, 17, 1}});
    } else {
        call({{0, 0, 0, 5}});
        call({{0, 0, 5, 1}});
        call({{0, 0, 6, 140}});
        call({{0, 0, 146, 2}, {1, 1, 0, 3}});
        call({{0, 0, 147, 2}});
    }
}

// A 16384-position context whose terms a sum in position order drops: head 0's sink holds each softmax term near 4e-8 of a denominator of 1, and head 1's first value holds each later product below half an ulp of its sum.
// Each output lies within 256 ulps of its sum of |w v|, the bound of f32 sums in blocks of 128 positions at 16k positions (docs/INKLING2.md, Row classes).
void attention_rel_sink_long() {
    const size_t n = 16384, heads = 2, D = 4, KD = heads * D, row_w = 2 * KD;
    CpuBackend cpu;
    std::unique_ptr<backend::StateStorage> store = cpu.state_alloc(1, 1, backend::StateShape{}, {n * row_w});
    std::vector<float> k(n * KD, 0.0f), v(n * KD), sink(n * heads);
    for (size_t p = 0; p < n; ++p) {
        sink[p * heads] = 17.0f;
        sink[p * heads + 1] = -100.0f;
        for (size_t e = 0; e < D; ++e) {
            v[p * KD + e] = 0.7f;
            v[p * KD + D + e] = p ? 5e-8f : 1.0f;
        }
    }
    BufferPtr qb = upload(cpu, std::vector<float>(n * KD)), kb = upload(cpu, k), vb = upload(cpu, v), rb = upload(cpu, std::vector<float>(n * heads));
    BufferPtr tb = upload(cpu, std::vector<float>(1)), sb = upload(cpu, sink), ob = upload(cpu, std::vector<float>(n * KD));
    const backend::StateView view{store.get(), 0, 0, 0, n};
    cpu.attention_rel_sink({ob.get(), 0}, {qb.get(), 0}, {kb.get(), 0}, {vb.get(), 0}, {rb.get(), 0}, {tb.get(), 0}, {sb.get(), 0}, heads, heads, D, 1, 1, n, 0.5f, 0, &view,
                           nullptr, 1, 0, n);
    const std::vector<float> got = download(cpu, *ob, n * KD);
    const double u = std::ldexp(1.0, -24), tiny = std::exp(double(-17.0f));
    for (size_t h = 0; h < heads; ++h) {
        // Head 0: n terms of e^-17 beside the sink's 1, each with the value 0.7; head 1: n equal weights, the first value 1 and the others 5e-8.
        const double want = h ? (1.0 + double(n - 1) * double(5e-8f)) / double(n) : double(0.7f) * double(n) * tiny / (1.0 + double(n) * tiny);
        for (size_t e = 0; e < D; ++e) {
            const double error = std::fabs(double(got[(n - 1) * KD + h * D + e]) - want);
            require(error <= 256.0 * u * want, "attention_rel_sink at 16384 positions, head " + std::to_string(h) + ": error " + std::to_string(error / (u * want)) +
                                                   " ulps of the sum of |w v|, above 256");
        }
    }
}

void attention_rel_sink() {
    for (int nt : kThreads)
        for (bool ring : {true, false}) attention_rel_sink_case(ring, nt);
    attention_rel_sink_long();
    CpuBackend cpu;
    std::unique_ptr<backend::StateStorage> store = cpu.state_alloc(1, 2, backend::StateShape{}, {64});
    BufferPtr b = upload(cpu, std::vector<float>(4096));
    const backend::StateView v{store.get(), 0, 0, 0, 2};
    const auto run = [&](size_t window, size_t ring_rows, size_t out_at) {
        cpu.attention_rel_sink({b.get(), out_at}, {b.get(), 100}, {b.get(), 200}, {b.get(), 300}, {b.get(), 400}, {b.get(), 500}, {b.get(), 600}, 2, 1, 4, 2, 4, window, 0.5f, 0,
                               &v, nullptr, 1, 0, ring_rows);
    };
    refused([&] { run(5, 3, 0); }, "a ring shorter than its window", "backend: invalid relative attention dimensions");
    refused([&] { run(0, 8, 0); }, "a ring without a window", "backend: invalid relative attention dimensions");
    refused([&] { run(5, 9, 0); }, "a ring past its slot", "backend: relative attention ring outside the state slot");
    refused([&] { run(5, 8, 96); }, "an output over the queries", "backend: relative attention output overlaps an input");
}

// Core attention with AttentionTerms (the relative bias and the sink) over block KV storage written by kv_write, against the same softmax in double over the stored keys and values.
// Calls of 5, 1, 140 and 2 + 3 rows over two sequences cross the KV blocks; bias and sink alone and together, without and with a window, f32 and f16 sides; a device runs heads 128 wide.
void attention_terms_case(backend::KVType type, size_t W, int nt, int mode, backend::Backend* device = nullptr, size_t heads = 4, size_t D = 8, double tol = 2e-6) {
    const size_t kv_heads = 2, R = 3, extent = 6, KD = kv_heads * D, HD = heads * D;
    CpuBackend own;
    own.set_threads(nt);
    backend::Backend& cpu = device ? *device : own;
    std::unique_ptr<backend::KVStorage> kv = cpu.kv_alloc(1, kv_heads, D, 512, type, type);
    const std::vector<std::vector<int32_t>> blocks = {{2, 0, 1}, {3}};
    std::vector<float> table = random(R * extent, -1.0f, 1.0f);
    table[0] = 0.5f;
    BufferPtr tb = upload(cpu, table);
    std::vector<std::vector<float>> hist[2];
    const auto stored = [&](float x) { return type == backend::KVType::f16 ? f16_to_f32(f32_to_f16(x)) : x; };
    size_t n_call = 0;
    const auto call = [&](const std::vector<ConvCall>& calls) {
        const bool bias = mode != 1, sink_on = mode != 2;
        mode = (mode + 1) % 3;
        size_t rows = 0;
        for (const ConvCall& c : calls) rows += c.nq;
        const std::vector<float> q = random(rows * HD, -1.0f, 1.0f), k = random(rows * KD, -1.0f, 1.0f), v = random(rows * KD, -1.0f, 1.0f);
        std::vector<float> rel = random(rows * heads * R, -1.0f, 1.0f), sink = random(rows * heads, -2.0f, 2.0f);
        // Extreme rows: a sink far above every score, one far below, and a bias of -200 on each row's own key, which must lower the maximum the exps take.
        if (++n_call % 2 == 0)
            for (size_t g = 0; g < rows; ++g) {
                sink[g * heads] = 200.0f;
                sink[g * heads + 1] = -200.0f;
                for (size_t r = 0; r < R; ++r) rel[(g * heads + 2) * R + r] = r ? 0.0f : -400.0f;
            }
        std::vector<backend::KVView> views;
        std::vector<double> want(rows * HD);
        for (size_t i = 0, row = 0; i < calls.size(); row += calls[i].nq, ++i) {
            const ConvCall& c = calls[i];
            views.push_back({kv.get(), blocks[c.src].data(), blocks[c.src].size(), c.length, c.nq});
            views.back().window = W;
            std::vector<std::vector<float>>& s = hist[c.src];
            s.resize(c.length);
            for (size_t b = 0; b < c.nq; ++b) {
                std::vector<float> kvrow(2 * KD);
                for (size_t e = 0; e < KD; ++e) {
                    kvrow[e] = stored(k[(row + b) * KD + e]);
                    kvrow[KD + e] = stored(v[(row + b) * KD + e]);
                }
                s.push_back(kvrow);
            }
            for (size_t b = 0; b < c.nq; ++b) {
                const size_t pos = c.length + b, g = row + b, lo = W && pos + 1 > W ? pos + 1 - W : 0;
                for (size_t h = 0; h < heads; ++h) {
                    const size_t kh = h / (heads / kv_heads);
                    std::vector<double> sc;
                    double top = sink_on ? double(sink[g * heads + h]) : -INFINITY;
                    for (size_t p = lo; p <= pos; ++p) {
                        double dot = 0.0, add = 0.0;
                        for (size_t e = 0; e < D; ++e) dot += double(q[g * HD + h * D + e]) * s[p][kh * D + e];
                        if (bias && pos - p < extent)
                            for (size_t r = 0; r < R; ++r) add += double(rel[(g * heads + h) * R + r]) * table[r * extent + pos - p];
                        sc.push_back(dot / std::sqrt(double(D)) + add);
                        top = std::max(top, sc.back());
                    }
                    double total = sink_on ? std::exp(sink[g * heads + h] - top) : 0.0;
                    for (double& x : sc) total += x = std::exp(x - top);
                    for (size_t e = 0; e < D; ++e) {
                        double o = 0.0;
                        for (size_t p = lo; p <= pos; ++p) o += sc[p - lo] / total * s[p][KD + kh * D + e];
                        want[g * HD + h * D + e] = o;
                    }
                }
            }
        }
        BufferPtr qb = upload(cpu, q), kb = upload(cpu, k), vb = upload(cpu, v), rb = upload(cpu, rel), sb = upload(cpu, sink), ob = upload(cpu, std::vector<float>(rows * HD));
        cpu.kv_write(0, views.data(), views.size(), {kb.get(), 0}, {vb.get(), 0});
        backend::AttentionTerms terms;
        if (bias) terms = {{rb.get(), 0}, {tb.get(), 0}, R, extent, {}};
        if (sink_on) terms.sink = {sb.get(), 0};
        cpu.attention({qb.get(), 0}, 0, views.data(), views.size(), {ob.get(), 0}, int(heads), int(kv_heads), int(D), &terms);
        close(download(cpu, *ob, rows * HD), want, tol,
              std::string("attention with ") + (bias ? sink_on ? "bias and sink" : "bias" : "sink") + (W ? " in a window" : "") + (type == backend::KVType::f16 ? " on f16 sides" : "") +
                  " over " + std::to_string(heads) + " heads " + std::to_string(D) + " wide");
        // Terms of neither part give the bits of no terms.
        BufferPtr ob2 = upload(cpu, std::vector<float>(rows * HD)), ob3 = upload(cpu, std::vector<float>(rows * HD));
        const backend::AttentionTerms none;
        cpu.attention({qb.get(), 0}, 0, views.data(), views.size(), {ob2.get(), 0}, int(heads), int(kv_heads), int(D), &none);
        cpu.attention({qb.get(), 0}, 0, views.data(), views.size(), {ob3.get(), 0}, int(heads), int(kv_heads), int(D));
        require(download(cpu, *ob2, rows * HD) == download(cpu, *ob3, rows * HD), "attention with empty terms gives the bits of none");
        // On a device the gated call gives the bits of the call then sigmoid_mul, and so does a product reading the twin it writes, rows of one key part and of several alike.
        if (device) {
            const size_t gs = HD + 5, nout = 64;
            BufferPtr gate = upload(cpu, random(rows * gs, -4.0f, 4.0f)), w = upload(cpu, random(HD * nout, -0.2f, 0.2f));
            BufferPtr yg = upload(cpu, std::vector<float>(rows * nout)), ya = upload(cpu, std::vector<float>(rows * nout));
            const backend::WeightView wv{quant::GGML_TYPE_F32, {w.get(), 0}};
            for (backend::Dtype dtype : {backend::Dtype::f16, backend::Dtype::int8}) {
                cpu.gated_attention({qb.get(), 0}, 0, views.data(), views.size(), {ob2.get(), 0}, int(heads), int(kv_heads), int(D), {{gate.get(), 0}, rows, gs, D, {}}, &terms);
                cpu.matmul(wv, {ob2.get(), 0}, {yg.get(), 0}, HD, nout, rows, {}, dtype);
                cpu.attention({qb.get(), 0}, 0, views.data(), views.size(), {ob3.get(), 0}, int(heads), int(kv_heads), int(D), &terms);
                cpu.sigmoid_mul({ob3.get(), 0}, {ob3.get(), 0}, {gate.get(), 0}, rows, heads, D, gs, D);
                cpu.matmul(wv, {ob3.get(), 0}, {ya.get(), 0}, HD, nout, rows, {}, dtype);
                require(download(cpu, *ob2, rows * HD) == download(cpu, *ob3, rows * HD) && download(cpu, *yg, rows * nout) == download(cpu, *ya, rows * nout),
                        "gated attention with terms over " + std::to_string(rows) + " rows has the bits of the call then sigmoid_mul, and so has its product");
            }
        }
    };
    call({{0, 0, 0, 5}});
    call({{0, 0, 5, 1}});
    call({{0, 0, 6, 140}});
    call({{0, 0, 146, 2}, {1, 1, 0, 3}});
    call({{0, 0, 148, 1}, {1, 1, 3, 1}});
    call({{1, 1, 4, 2}});
}

void attention_terms() {
    int mode = 0;
    for (int nt : kThreads)
        for (backend::KVType type : {backend::KVType::f32, backend::KVType::f16})
            for (size_t W : {size_t(0), size_t(5), size_t(9)}) attention_terms_case(type, W, nt, mode++ % 3);
    CpuBackend cpu;
    std::unique_ptr<backend::KVStorage> kv = cpu.kv_alloc(1, 1, 4, 128, backend::KVType::f32, backend::KVType::f32);
    BufferPtr b = upload(cpu, std::vector<float>(64));
    const int32_t block = 0;
    backend::KVView view{kv.get(), &block, 1, 0, 2};
    const auto run = [&](const backend::AttentionTerms& t) { cpu.attention({b.get(), 0}, 0, &view, 1, {b.get(), 32}, 1, 1, 4, &t); };
    refused([&] { run({{b.get(), 0}, {}, 2, 4, {}}); }, "a bias without its table", "backend: attention terms with a bias missing its rank, extent or table");
    refused([&] { run({{b.get(), 0}, {b.get(), 0}, 0, 4, {}}); }, "a bias of no rank", "backend: attention terms with a bias missing its rank, extent or table");
    view.causal = false;
    refused([&] { run({{}, {}, 0, 0, {b.get(), 0}}); }, "terms on a non-causal view", "backend: attention terms on a non-causal view");
}

// silu_mul in GluForm::softcap (HF Inkling2LoopSoftcappedSwiGLU) and softcap (Inkling2LoopSoftcap on the logits), with values past both caps, and their refusals.
void softcaps() {
    const size_t n = 300;
    const float L = 4.0f, U = 25.0f, cap = 30.0f;
    for (int nt : kThreads) {
        CpuBackend cpu;
        cpu.set_threads(nt);
        std::vector<float> g = random(n, -40.0f, 40.0f), u = random(n, -200.0f, 200.0f), x = random(n, -300.0f, 300.0f);
        g[0] = -120.0f, g[1] = 90.0f, u[2] = 1e4f, x[3] = 0.0f;
        std::vector<double> glu(n), capped(n);
        for (size_t i = 0; i < n; ++i) {
            const double gd = g[i], ud = u[i];
            glu[i] = L * std::tanh(gd / L) / (1.0 + std::exp(-gd)) * U * std::tanh(ud / U);
            capped[i] = cap * std::tanh(double(x[i]) / cap);
        }
        BufferPtr gb = upload(cpu, g), ub = upload(cpu, u), xb = upload(cpu, x), ob = upload(cpu, std::vector<float>(n));
        cpu.silu_mul({ob.get(), 0}, {gb.get(), 0}, {ub.get(), 0}, n, {L, backend::GluForm::softcap, U});
        close(download(cpu, *ob, n), glu, 1e-6, "silu_mul in the softcap form");
        cpu.softcap({xb.get(), 0}, n, cap);
        close(download(cpu, *xb, n), capped, 1e-6, "softcap");
        const std::string glu_text = "backend: a GLU softcap needs two positive finite caps, and only that form takes an up limit";
        refused([&] { cpu.silu_mul({ob.get(), 0}, {gb.get(), 0}, {ub.get(), 0}, n, {L, backend::GluForm::softcap, 0.0f}); }, "a softcap GLU without an up cap", glu_text);
        refused([&] { cpu.silu_mul({ob.get(), 0}, {gb.get(), 0}, {ub.get(), 0}, n, {0.0f, backend::GluForm::softcap, U}); }, "a softcap GLU without a gate cap", glu_text);
        refused([&] { cpu.silu_mul({ob.get(), 0}, {gb.get(), 0}, {ub.get(), 0}, n, {0.0f, backend::GluForm::silu, U}); }, "an up cap outside the softcap form", glu_text);
        refused([&] { cpu.softcap({xb.get(), 0}, n, 0.0f); }, "a softcap of 0", "backend: a softcap that is not positive and finite");
    }
}

// route_experts under a sigmoid rule: the eps joins the sum before it divides (HF Inkling2LoopTopKRouter's 1e-20), and an eps of 0 gives MiniMax-M3.1's plain quotient bit for bit.
void sigmoid_eps() {
    const size_t rows = 3, E = 6, k = 2;
    std::vector<float> scores(rows * E), bias(E, 0.0f);
    for (size_t i = 0; i < scores.size(); ++i) scores[i] = -46.0f + 0.5f * float(i % E) - float(i / E);
    for (float eps : {0.0f, 1e-20f}) {
        CpuBackend cpu;
        BufferPtr sb = upload(cpu, scores), bb = upload(cpu, bias), ib = upload(cpu, std::vector<float>(rows * k)), wb = upload(cpu, std::vector<float>(rows * k));
        cpu.route_experts({sb.get(), 0}, rows, E, k, backend::RouteRule::sigmoid({bb.get(), 0}, 2.0f, true, eps), {ib.get(), 0}, {wb.get(), 0});
        const std::vector<float> w = download(cpu, *wb, rows * k);
        for (size_t r = 0; r < rows; ++r) {
            float s[2], total = 0.0f;
            for (size_t j = 0; j < k; ++j) total += s[j] = 1.0f / (1.0f + std::exp(-scores[r * E + E - 1 - j]));
            for (size_t j = 0; j < k; ++j) {
                const float want = s[j] / (eps == 0.0f ? total : total + eps) * 2.0f;
                require(w[r * k + j] == want, "a sigmoid route weight with eps " + std::to_string(eps) + " is " + std::to_string(w[r * k + j]));
            }
        }
    }
}

// The fused gate-up expert stack read through row windows (Projection::first, pitch) gives moe_step the bits of separate gate and up stacks, on the decode step and on a prompt, in the softcap GLU under a sigmoid rule with eps.
void expert_window() {
    const size_t E = 4, nin = 64, ff = 32, k = 2, block = 34;
    const size_t row_in = nin / 32 * block, row_ff = ff / 32 * block;
    const auto q8 = [&](size_t rows, size_t row_bytes) {
        std::vector<uint8_t> v(rows * row_bytes);
        for (size_t r = 0; r < rows; ++r)
            for (size_t b = 0; b < row_bytes / block; ++b) {
                uint8_t* p = v.data() + r * row_bytes + b * block;
                p[0] = 0x00, p[1] = 0x20;  // d = 2^-7 as f16
                for (size_t i = 0; i < 32; ++i) p[2 + i] = uint8_t(int8_t(int(uniform(-127.0f, 127.0f))));
            }
        return v;
    };
    const std::vector<uint8_t> fused = q8(E * 2 * ff, row_in), down = q8(E * nin, row_ff);
    std::vector<uint8_t> gate(E * ff * row_in), up(E * ff * row_in);
    for (size_t e = 0; e < E; ++e) {
        std::copy_n(fused.begin() + (e * 2 * ff) * row_in, ff * row_in, gate.begin() + e * ff * row_in);
        std::copy_n(fused.begin() + (e * 2 * ff + ff) * row_in, ff * row_in, up.begin() + e * ff * row_in);
    }
    for (int nt : kThreads)
        for (size_t rows : {size_t(1), size_t(5)}) {
            CpuBackend cpu;
            cpu.set_threads(nt);
            const auto bytes = [&](const std::vector<uint8_t>& v) {
                BufferPtr b = cpu.alloc(v.size(), backend::Memory::device);
                cpu.write(*b, 0, v.data(), v.size());
                return b;
            };
            BufferPtr fb = bytes(fused), gb = bytes(gate), ub = bytes(up), db = bytes(down), rb = upload(cpu, random(E * nin, -0.3f, 0.3f)),
                      bias = upload(cpu, random(E, -0.1f, 0.1f)), hb = upload(cpu, random(rows * nin, -1.0f, 1.0f));
            std::vector<float> y[2];
            for (int windowed = 0; windowed < 2; ++windowed) {
                BufferPtr scores = upload(cpu, std::vector<float>(rows * E)), ids = upload(cpu, std::vector<float>(rows * k)), w = upload(cpu, std::vector<float>(rows * k)),
                          go = upload(cpu, std::vector<float>(rows * k * ff)), uo = upload(cpu, std::vector<float>(rows * k * ff)), act = upload(cpu, std::vector<float>(rows * k * ff)),
                          yb = upload(cpu, std::vector<float>(rows * nin));
                std::vector<backend::RowRun> scratch;
                backend::Projection gp{{quant::GGML_TYPE_Q8_0, {(windowed ? fb : gb).get(), 0}}, {go.get(), 0}, ff}, up_p{{quant::GGML_TYPE_Q8_0, {(windowed ? fb : ub).get(), 0}}, {uo.get(), 0}, ff};
                if (windowed) gp.pitch = up_p.pitch = 2 * ff, up_p.first = ff;
                const backend::MoeStep m{backend::Projection{{quant::GGML_TYPE_F32, {rb.get(), 0}}, {scores.get(), 0}, E}, k, backend::RouteRule::sigmoid({bias.get(), 0}, 2.5f, true, 1e-20f),
                                         {ids.get(), 0}, {w.get(), 0}, gp, up_p, {quant::GGML_TYPE_Q8_0, {db.get(), 0}}, {act.get(), 0}, nullptr, nullptr, &scratch,
                                         {4.0f, backend::GluForm::softcap, 25.0f}};
                const backend::RowRun decode[1] = {{1, 1}};
                cpu.moe_step(m, {hb.get(), 0}, {yb.get(), 0}, nin, rows, rows == 1 ? backend::RowRuns{decode, 1} : backend::RowRuns{});
                y[windowed] = download(cpu, *yb, rows * nin);
                if (windowed) {
                    backend::MoeStep wide = m;
                    wide.up.rows = ff + 1;
                    refused([&] { cpu.moe_step(wide, {hb.get(), 0}, {yb.get(), 0}, nin, rows); }, "an up window past its matrix",
                            "backend: an expert row window outside its matrix, or over an F8_E4M3 grid");
                }
            }
            bool same = true;
            for (size_t i = 0; i < rows * nin; ++i) same = same && std::memcmp(&y[0][i], &y[1][i], sizeof(float)) == 0;
            require(same, "the windowed gate-up stack's moe_step over " + std::to_string(rows) + " rows at " + std::to_string(nt) + " threads gives the split stacks' bits");
        }
}

// copy_rows between strided rows, as the residual bundle's rows move to and from contiguous slots: a copy and an add, bit for bit, and the refusals.
void copy_rows() {
    const size_t rows = 7, width = 5, dst_stride = 30, src_stride = 8;
    for (int nt : kThreads) {
        CpuBackend cpu;
        cpu.set_threads(nt);
        const std::vector<float> d0 = random(rows * dst_stride, -3.0f, 3.0f), src = random(rows * src_stride, -3.0f, 3.0f);
        for (bool add : {false, true}) {
            BufferPtr db = upload(cpu, d0), sb = upload(cpu, src);
            cpu.copy_rows({db.get(), 10}, dst_stride, {sb.get(), 2}, src_stride, rows, width, add);
            std::vector<float> want = d0;
            for (size_t r = 0; r < rows; ++r)
                for (size_t i = 0; i < width; ++i) {
                    float& w = want[10 + r * dst_stride + i];
                    w = add ? w + src[2 + r * src_stride + i] : src[2 + r * src_stride + i];
                }
            const std::vector<float> got = download(cpu, *db, want.size());
            require(std::memcmp(got.data(), want.data(), want.size() * sizeof(float)) == 0,
                    std::string("copy_rows ") + (add ? "adds" : "copies") + " strided rows bit for bit at " + std::to_string(nt) + " threads, other floats untouched");
        }
        BufferPtr b = upload(cpu, d0);
        refused([&] { cpu.copy_rows({b.get(), 0}, 4, {b.get(), 0}, 8, 2, 5, false); }, "a row copy wider than its destination stride", "backend: invalid row copy dimensions");
        refused([&] { cpu.copy_rows({b.get(), 0}, 8, {b.get(), 0}, 8, 2, 0, true); }, "a row copy of no width", "backend: invalid row copy dimensions");
    }
}

// With --device, each op runs on the device and on the CPU from the same inputs, and the device's floats must be within `tol` * (1 + |cpu|) of the CPU's.
struct Pair {
    CpuBackend cpu;
    backend::Backend& dev;
    explicit Pair(backend::Backend& d) : dev(d) {}
    std::pair<BufferPtr, BufferPtr> put(const std::vector<float>& v) {
        BufferPtr c = upload(cpu, v);
        BufferPtr d = dev.alloc(std::max<size_t>(v.size(), 1) * sizeof(float), backend::Memory::device);
        if (!v.empty()) dev.write(*d, 0, v.data(), v.size() * sizeof(float));
        return {c, d};
    }
    void same(const std::pair<BufferPtr, BufferPtr>& b, size_t n, double tol, const std::string& what) {
        dev.sync();
        std::vector<float> d(n);
        dev.read(*b.second, 0, d.data(), n * sizeof(float));
        const std::vector<float> c = download(cpu, *b.first, n);
        std::vector<double> want(c.begin(), c.end());
        close(d, want, tol, "device " + what);
    }
};

void row_products(backend::Backend& b);

void device_ops(backend::Backend& dev) {
    Pair p(dev);
    row_products(dev);
    for (bool add : {false, true}) {
        const size_t rows = 37, width = 2304, ds = 6 * 2304, ss = 2304 + 3;
        auto d = p.put(random(rows * ds + 11, -3.0f, 3.0f)), s = p.put(random(rows * ss, -3.0f, 3.0f));
        p.cpu.copy_rows({d.first.get(), 11}, ds, {s.first.get(), 3}, ss, rows, width, add);
        p.dev.copy_rows({d.second.get(), 11}, ds, {s.second.get(), 3}, ss, rows, width, add);
        p.same(d, rows * ds + 11, 0.0, add ? "copy_rows add" : "copy_rows copy");
    }
    {
        const size_t n = 3 * 201024;
        std::vector<float> v = random(n, -400.0f, 400.0f);
        v[0] = 1e30f; v[1] = -1e30f; v[2] = 0.0f;
        auto x = p.put(v);
        p.cpu.softcap({x.first.get(), 0}, n, 30.0f);
        p.dev.softcap({x.second.get(), 0}, n, 30.0f);
        p.same(x, n, 2e-6, "softcap 30");
    }
    for (size_t n : {size_t(31), size_t(8 * 768), size_t(5 * 768)}) {
        auto g = p.put(random(n, -40.0f, 40.0f)), u = p.put(random(n, -90.0f, 90.0f)), o = p.put(std::vector<float>(n, 0.0f));
        const backend::Glu glu{4.0f, backend::GluForm::softcap, 25.0f};
        p.cpu.silu_mul({o.first.get(), 0}, {g.first.get(), 0}, {u.first.get(), 0}, n, glu);
        p.dev.silu_mul({o.second.get(), 0}, {g.second.get(), 0}, {u.second.get(), 0}, n, glu);
        p.same(o, n, 2e-6, "silu_mul softcap form over " + std::to_string(n));
    }
    {
        // A prompt run, so the GLU takes its four-a-lane build beside the tile's activation twin.
        const size_t rows = 64, width = 768, n = rows * width;
        const backend::RowRun run[1] = {{rows, rows}};
        auto g = p.put(random(n, -40.0f, 40.0f)), u = p.put(random(n, -90.0f, 90.0f)), o = p.put(std::vector<float>(n, 0.0f));
        const backend::Glu glu{4.0f, backend::GluForm::softcap, 25.0f};
        p.cpu.silu_mul({o.first.get(), 0}, {g.first.get(), 0}, {u.first.get(), 0}, n, glu, {run, 1});
        p.dev.silu_mul({o.second.get(), 0}, {g.second.get(), 0}, {u.second.get(), 0}, n, glu, {run, 1});
        p.same(o, n, 2e-6, "silu_mul softcap form over a prompt run");
    }
    for (size_t dim : {size_t(128), size_t(16)}) {
        const size_t rows = 9, heads = 24, stride = heads * dim + 5;
        for (int with : {0, 1}) {
            auto x = p.put(random(rows * stride, -6.0f, 6.0f)), sc = p.put(random(heads * dim, 0.2f, 2.0f)), rs = p.put(random(rows, 1.0f, 1.2f));
            const float t = dim == 16 ? 8.0f : 22.63f;
            const auto run = [&](backend::Backend& b, BufferPtr xb, BufferPtr sb, BufferPtr rb) {
                b.l2_clip({{{xb.get(), 0}, with ? backend::CSlice{sb.get(), 0} : backend::CSlice{}, with ? backend::CSlice{rb.get(), 0} : backend::CSlice{}, heads, dim, stride, t}}, rows);
            };
            run(p.cpu, x.first, sc.first, rs.first);
            run(p.dev, x.second, sc.second, rs.second);
            p.same(x, rows * stride, 2e-6, "l2_clip dim " + std::to_string(dim) + (with ? " with scales" : ""));
        }
    }
    {
        // Inkling's q, rel and k in one call: each item the bits of its own call on the device, and the CPU's within the bound.
        const size_t rows = 9, H = 24, HK = 4, D = 128, R = 16;
        auto q = p.put(random(rows * H * D, -6.0f, 6.0f)), rel = p.put(random(rows * H * R, -6.0f, 6.0f)), k = p.put(random(rows * HK * D, -6.0f, 6.0f));
        auto qs = p.put(random(H * D, 0.2f, 2.0f)), rs = p.put(random(H * R, 0.2f, 2.0f)), tau = p.put(random(rows, 1.0f, 1.2f));
        std::vector<float> alone[3];
        {
            const std::vector<float> xs[3] = {download(p.dev, *q.second, rows * H * D), download(p.dev, *rel.second, rows * H * R), download(p.dev, *k.second, rows * HK * D)};
            const size_t hs[3] = {H, H, HK}, ds[3] = {D, R, D};
            const BufferPtr sc[3] = {qs.second, rs.second, nullptr};
            for (size_t i = 0; i < 3; ++i) {
                BufferPtr xb = upload(p.dev, xs[i]);
                p.dev.l2_clip({{{xb.get(), 0}, sc[i] ? backend::CSlice{sc[i].get(), 0} : backend::CSlice{}, i < 2 ? backend::CSlice{tau.second.get(), 0} : backend::CSlice{}, hs[i], ds[i],
                                hs[i] * ds[i], i == 1 ? 8.0f : 22.63f}},
                              rows);
                alone[i] = download(p.dev, *xb, rows * hs[i] * ds[i]);
            }
        }
        const auto run = [&](backend::Backend& b, BufferPtr qb, BufferPtr rb, BufferPtr kb, BufferPtr qsb, BufferPtr rsb, BufferPtr tb) {
            b.l2_clip({{{qb.get(), 0}, {qsb.get(), 0}, {tb.get(), 0}, H, D, H * D, 22.63f}, {{rb.get(), 0}, {rsb.get(), 0}, {tb.get(), 0}, H, R, H * R, 8.0f},
                       {{kb.get(), 0}, {}, {}, HK, D, HK * D, 22.63f}},
                      rows);
        };
        run(p.cpu, q.first, rel.first, k.first, qs.first, rs.first, tau.first);
        run(p.dev, q.second, rel.second, k.second, qs.second, rs.second, tau.second);
        p.same(q, rows * H * D, 2e-6, "l2_clip of three items, q");
        p.same(rel, rows * H * R, 2e-6, "l2_clip of three items, rel");
        p.same(k, rows * HK * D, 2e-6, "l2_clip of three items, k");
        const BufferPtr got[3] = {q.second, rel.second, k.second};
        for (size_t i = 0; i < 3; ++i)
            if (std::memcmp(download(p.dev, *got[i], alone[i].size()).data(), alone[i].data(), alone[i].size() * sizeof(float))) throw std::runtime_error("l2_clip of three items differs from item " + std::to_string(i) + "'s own call");
    }
    {
        const size_t rows = 7, width = 2304, stride = 2 * width;
        auto raw = p.put(random(rows * stride, -3.0f, 3.0f)), f = p.put(random(rows * width, -4.0f, 4.0f)), g = p.put(random(rows * width, -4.0f, 4.0f));
        p.cpu.tanh_gate_mix({{{raw.first.get(), 0}, {raw.first.get(), 0}, {raw.first.get(), width}, {f.first.get(), 0}, {g.first.get(), 0}}}, rows, width, stride, stride, stride);
        p.dev.tanh_gate_mix({{{raw.second.get(), 0}, {raw.second.get(), 0}, {raw.second.get(), width}, {f.second.get(), 0}, {g.second.get(), 0}}}, rows, width, stride, stride, stride);
        p.same(raw, rows * stride, 1e-6, "tanh_gate_mix in place");
        // Packed out from h and m rows of their own strides, as the message gate writes its output, from fresh rows rather than the in-place results the two sides round differently.
        auto src = p.put(random(rows * stride, -3.0f, 3.0f)), out = p.put(std::vector<float>(rows * width, 0.0f));
        p.cpu.tanh_gate_mix({{{out.first.get(), 0}, {src.first.get(), 0}, {src.first.get(), 3}, {f.first.get(), 0}, {g.first.get(), 0}}}, rows, width, width, stride, width + 5);
        p.dev.tanh_gate_mix({{{out.second.get(), 0}, {src.second.get(), 0}, {src.second.get(), 3}, {f.second.get(), 0}, {g.second.get(), 0}}}, rows, width, width, stride, width + 5);
        p.same(out, rows * width, 1e-6, "tanh_gate_mix packed out");
    }
    // With a norm, the device gives the bits of its mix then its norm in place: one row, several and more rows than compute units, and a width the fused build does not take.
    for (size_t width : {size_t(2304), size_t(1000)})
        for (size_t rows : {size_t(1), size_t(7), size_t(70)}) {
            const size_t stride = 2 * width;
            BufferPtr src = upload(p.dev, random(rows * stride, -3.0f, 3.0f)), f = upload(p.dev, random(rows * width, -4.0f, 4.0f)), g = upload(p.dev, random(rows * width, -4.0f, 4.0f));
            BufferPtr nw = upload(p.dev, random(width, 0.5f, 1.5f)), fused = upload(p.dev, std::vector<float>(rows * width)), apart = upload(p.dev, std::vector<float>(rows * width));
            p.dev.tanh_gate_mix({{{fused.get(), 0}, {src.get(), 0}, {src.get(), 3}, {f.get(), 0}, {g.get(), 0}, {nw.get(), 0}}}, rows, width, width, stride, width + 5, {}, 1e-6f);
            p.dev.tanh_gate_mix({{{apart.get(), 0}, {src.get(), 0}, {src.get(), 3}, {f.get(), 0}, {g.get(), 0}}}, rows, width, width, stride, width + 5);
            p.dev.rms_norm_rows({apart.get(), 0}, {apart.get(), 0}, {nw.get(), 0}, rows, width, width, 1e-6f);
            // Two items in one call, the second from other rows, norm and gates, each the bits of its own call.
            BufferPtr src2 = upload(p.dev, random(rows * stride, -3.0f, 3.0f)), nw2 = upload(p.dev, random(width, 0.5f, 1.5f));
            BufferPtr both = upload(p.dev, std::vector<float>(2 * rows * width)), alone = upload(p.dev, std::vector<float>(rows * width));
            p.dev.tanh_gate_mix({{{both.get(), 0}, {src.get(), 0}, {src.get(), 3}, {f.get(), 0}, {g.get(), 0}, {nw.get(), 0}},
                                 {{both.get(), rows * width}, {src2.get(), 0}, {src2.get(), 3}, {g.get(), 0}, {f.get(), 0}, {nw2.get(), 0}}},
                                rows, width, width, stride, width + 5, {}, 1e-6f);
            p.dev.tanh_gate_mix({{{alone.get(), 0}, {src2.get(), 0}, {src2.get(), 3}, {g.get(), 0}, {f.get(), 0}, {nw2.get(), 0}}}, rows, width, width, stride, width + 5, {}, 1e-6f);
            p.dev.sync();
            require(!std::memcmp(download(p.dev, *fused, rows * width).data(), download(p.dev, *apart, rows * width).data(), rows * width * sizeof(float)),
                    "tanh_gate_mix with a norm over " + std::to_string(rows) + " rows of " + std::to_string(width) + " has the bits of the mix then the norm");
            const std::vector<float> two = download(p.dev, *both, 2 * rows * width);
            require(!std::memcmp(two.data(), download(p.dev, *fused, rows * width).data(), rows * width * sizeof(float)) &&
                        !std::memcmp(two.data() + rows * width, download(p.dev, *alone, rows * width).data(), rows * width * sizeof(float)),
                    "tanh_gate_mix of two normed items over " + std::to_string(rows) + " rows of " + std::to_string(width) + " has each item's own bits");
        }
    // Items naming their forget and gain products give the bits of matmul_group into f and g then the mix without them: norms or none, two items, in place, over a prompt run the tile mix build takes and rows it does not.
    for (size_t rows : {size_t(1), size_t(5), size_t(130)}) {
        const size_t width = 2304, rank = 64, stride = 2 * width;
        const backend::RowRun runs[1] = {{rows, rows == 1 ? size_t(1) : rows}};
        const auto f32w = [](const BufferPtr& w) { return backend::WeightView{quant::GGML_TYPE_F32, {w.get(), 0}}; };
        BufferPtr wf = upload(p.dev, random(width * rank, -0.3f, 0.3f)), wg = upload(p.dev, random(width * rank, -0.3f, 0.3f)), x = upload(p.dev, random(2 * rows * rank, -1.0f, 1.0f));
        BufferPtr src = upload(p.dev, random(rows * stride, -3.0f, 3.0f)), src2 = upload(p.dev, random(rows * stride, -3.0f, 3.0f)), nw = upload(p.dev, random(width, 0.5f, 1.5f));
        BufferPtr fg = upload(p.dev, std::vector<float>(4 * rows * width)), fg2 = upload(p.dev, std::vector<float>(4 * rows * width));
        const auto product = [&](const BufferPtr& w, const BufferPtr& out, size_t plane, size_t xplane) {
            backend::Projection pr{f32w(w), {out.get(), plane * rows * width}, width};
            pr.in = {x.get(), xplane * rows * rank};
            return pr;
        };
        const auto item = [&](const BufferPtr& out, size_t at, const BufferPtr& h, const BufferPtr& g, size_t j, bool norm, bool weights) {
            backend::GateMixItem it{{out.get(), at}, {h.get(), 0}, {h.get(), 3}, {g.get(), 2 * j * rows * width}, {g.get(), (2 * j + 1) * rows * width},
                                    norm ? backend::CSlice{nw.get(), 0} : backend::CSlice{}};
            if (weights) it.forget = product(wf, g, 2 * j, j), it.gain = product(wg, g, 2 * j + 1, j), it.rank = rank;
            return it;
        };
        for (bool norm : {false, true}) {
            BufferPtr want = upload(p.dev, std::vector<float>(2 * rows * width)), got = upload(p.dev, std::vector<float>(2 * rows * width));
            p.dev.matmul_group({product(wf, fg, 0, 0), product(wg, fg, 1, 0), product(wf, fg, 2, 1), product(wg, fg, 3, 1)}, {}, rank, rows, {runs, 1}, backend::Dtype::f16);
            p.dev.tanh_gate_mix({item(want, 0, src, fg, 0, norm, false), item(want, rows * width, src2, fg, 1, norm, false)}, rows, width, width, stride, width + 5, {runs, 1}, 1e-6f);
            p.dev.tanh_gate_mix({item(got, 0, src, fg2, 0, norm, true), item(got, rows * width, src2, fg2, 1, norm, true)}, rows, width, width, stride, width + 5, {runs, 1}, 1e-6f);
            p.dev.sync();
            require(!std::memcmp(download(p.dev, *want, 2 * rows * width).data(), download(p.dev, *got, 2 * rows * width).data(), 2 * rows * width * sizeof(float)),
                    "tanh_gate_mix computing its products over " + std::to_string(rows) + " rows" + (norm ? " with norms" : "") + " has the bits of the products then the mix");
        }
        // In place over h, as the feed-forward gate mixes, rows `width` apart.
        const std::vector<float> h0 = random(rows * width, -3.0f, 3.0f);
        BufferPtr h = upload(p.dev, h0), inplace = upload(p.dev, h0);
        p.dev.matmul_group({product(wf, fg, 0, 0), product(wg, fg, 1, 0)}, {}, rank, rows, {runs, 1}, backend::Dtype::f16);
        p.dev.tanh_gate_mix({{{h.get(), 0}, {h.get(), 0}, {src.get(), 0}, {fg.get(), 0}, {fg.get(), rows * width}}}, rows, width, width, width, stride, {runs, 1});
        backend::GateMixItem it{{inplace.get(), 0}, {inplace.get(), 0}, {src.get(), 0}, {fg2.get(), 0}, {fg2.get(), rows * width}};
        it.forget = product(wf, fg2, 0, 0), it.gain = product(wg, fg2, 1, 0), it.rank = rank;
        p.dev.tanh_gate_mix({it}, rows, width, width, width, stride, {runs, 1});
        p.dev.sync();
        require(!std::memcmp(download(p.dev, *h, rows * width).data(), download(p.dev, *inplace, rows * width).data(), rows * width * sizeof(float)),
                "tanh_gate_mix computing its products in place over " + std::to_string(rows) + " rows has the bits of the products then the mix");
    }
    // A grouped norm with its product gives the bits of the norm then the matmul, on the device and on the CPU: the message gate's two sources, offsets off 16 bytes, a dim not a multiple of 4, and rows the fused build does not take or hold.
    for (size_t dim : {size_t(2304), size_t(1001), size_t(4500)})
        for (size_t off : {size_t(0), size_t(1)})
            for (size_t rows : {size_t(1), size_t(2)})
                for (size_t nout : {size_t(64), size_t(63), size_t(1)}) {
                    const size_t hs = 3 * dim + 4, ms = 4 * dim, width = 2 * dim;
                    std::vector<float> hv = random(off + rows * hs, -3.0f, 3.0f), mv = random(off + rows * ms, -3.0f, 3.0f);
                    hv[off] = -0.0f, mv[off + 1] = 0.0f;
                    for (backend::Backend* b : {static_cast<backend::Backend*>(&p.cpu), &p.dev}) {
                        BufferPtr h = upload(*b, hv), m = upload(*b, mv), nw = upload(*b, random(width, 0.5f, 1.5f)), w = upload(*b, random(width * nout, -0.2f, 0.2f));
                        BufferPtr normed = upload(*b, std::vector<float>(rows * width)), fused = upload(*b, std::vector<float>(rows * nout)), apart = upload(*b, std::vector<float>(rows * nout));
                        const backend::WeightView wv{quant::GGML_TYPE_F32, {w.get(), 0}};
                        const backend::NormSource from{{m.get(), off}, 1, hs, ms};
                        b->group_rms_norm({normed.get(), 0}, {h.get(), off}, {nw.get(), 0}, rows, 2, dim, 1e-6f, {}, from, {wv, {fused.get(), 0}, nout, backend::Dtype::f16});
                        b->group_rms_norm({normed.get(), 0}, {h.get(), off}, {nw.get(), 0}, rows, 2, dim, 1e-6f, {}, from);
                        b->matmul(wv, {normed.get(), 0}, {apart.get(), 0}, width, nout, rows, {}, backend::Dtype::f16);
                        b->sync();
                        require(!std::memcmp(download(*b, *fused, rows * nout).data(), download(*b, *apart, rows * nout).data(), rows * nout * sizeof(float)),
                                std::string(b == &p.dev ? "device" : "cpu") + " group_rms_norm into a product over " + std::to_string(rows) + " rows of 2 x " + std::to_string(dim) + " at offset " +
                                    std::to_string(off) + " to " + std::to_string(nout) + " has the bits of the norm then the matmul");
                    }
                }
    // Three grouped norms with products as one list call over one scratch row have the bits of the single calls in order, on the device and the CPU; where the second's y is the third's source, the list keeps their order.
    for (size_t dim : {size_t(2304), size_t(1001)})
        for (size_t off : {size_t(0), size_t(1)})
            for (size_t rows : {size_t(1), size_t(2)})
                for (bool chain : {false, true}) {
                    const size_t hs = 3 * dim + 4, ms = 4 * dim, width = 2 * dim, nout = 64;
                    const std::vector<float> mv = random(off + rows * ms + 2, -3.0f, 3.0f), nwv = random(width, 0.5f, 1.5f);
                    std::vector<float> hv[3], wv[3];
                    for (size_t i = 0; i < 3; ++i) hv[i] = random(off + rows * hs, -3.0f, 3.0f), wv[i] = random(width * nout, -0.2f, 0.2f);
                    for (backend::Backend* b : {static_cast<backend::Backend*>(&p.cpu), &p.dev}) {
                        std::vector<float> got[2];
                        for (size_t list = 0; list < 2; ++list) {
                            BufferPtr m = upload(*b, mv), nw = upload(*b, nwv), normed = upload(*b, std::vector<float>(rows * width)), y = upload(*b, std::vector<float>(3 * rows * nout));
                            BufferPtr h[3], w[3];
                            backend::NormRows it[3];
                            for (size_t i = 0; i < 3; ++i) {
                                h[i] = upload(*b, hv[i]), w[i] = upload(*b, wv[i]);
                                it[i] = {{normed.get(), 0}, {h[i].get(), off}, {{m.get(), off + i}, 1, hs, ms}, {{quant::GGML_TYPE_F32, {w[i].get(), 0}}, {y.get(), i * rows * nout}, nout, backend::Dtype::f16}};
                            }
                            if (chain) it[1].into.y = {h[2].get(), off};
                            if (list) {
                                b->group_rms_norm(it, 3, {nw.get(), 0}, rows, 2, dim, 1e-6f);
                            } else {
                                for (const backend::NormRows& r : it) b->group_rms_norm(r.dst, r.src, {nw.get(), 0}, rows, 2, dim, 1e-6f, {}, r.from, r.into);
                            }
                            b->sync();
                            got[list] = download(*b, *y, 3 * rows * nout);
                            const std::vector<float> tail = download(*b, *h[2], off + rows * hs);
                            got[list].insert(got[list].end(), tail.begin(), tail.end());
                        }
                        require(!std::memcmp(got[0].data(), got[1].data(), got[0].size() * sizeof(float)),
                                std::string(b == &p.dev ? "device" : "cpu") + " group_rms_norm of three products over " + std::to_string(rows) + " rows of 2 x " + std::to_string(dim) + " at offset " +
                                    std::to_string(off) + (chain ? ", chained," : "") + " has the bits of the single calls");
                    }
                }
    // Rows of at most 2304 values take the held build to 12 items and the looped one past it; 2305 values take the looped one throughout.
    for (size_t dim : {size_t(2304), size_t(1000), size_t(2305)})
    for (size_t n : {size_t(1), size_t(4), size_t(6), size_t(7), size_t(12), size_t(13), size_t(16)})
        for (size_t nq : {size_t(1), size_t(4)}) {
            const size_t rows = 5, stride = 16 * dim;
            auto src = p.put(random(rows * stride, -2.0f, 2.0f)), q = p.put(random(nq * dim, -0.1f, 0.1f)), o = p.put(std::vector<float>(rows * nq * dim, 0.0f));
            p.cpu.query_mix({{{o.first.get(), 0}, {src.first.get(), 0}, {q.first.get(), 0}, n, nq}}, rows, dim, stride, 1e-6f);
            p.dev.query_mix({{{o.second.get(), 0}, {src.second.get(), 0}, {q.second.get(), 0}, n, nq}}, rows, dim, stride, 1e-6f);
            p.same(o, rows * nq * dim, 1e-5, "query_mix of " + std::to_string(n) + " items of " + std::to_string(dim) + " by " + std::to_string(nq) + " queries");
        }
    // Two mixes in one call have the bits of each alone: the loop message's (2 items, 4 queries) beside a residual mix of 1 to 13 items by 1 or 3 queries.
    for (size_t dim : {size_t(2304), size_t(2305)})
        for (size_t n : {size_t(1), size_t(3), size_t(6), size_t(13)})
            for (size_t nq : {size_t(1), size_t(3)}) {
                const size_t rows = 3, stride = 16 * dim, at = 14 * dim;
                auto src = p.put(random(rows * stride, -2.0f, 2.0f)), qm = p.put(random(4 * dim, -0.1f, 0.1f)), qr = p.put(random(nq * dim, -0.1f, 0.1f));
                auto om = p.put(std::vector<float>(rows * 4 * dim, 0.0f)), orr = p.put(std::vector<float>(rows * nq * dim, 0.0f));
                auto sm = p.put(std::vector<float>(rows * 4 * dim, 0.0f)), sr = p.put(std::vector<float>(rows * nq * dim, 0.0f));
                const auto mix = [&](const BufferPtr& out, const BufferPtr& s, const BufferPtr& q, size_t off, size_t items, size_t queries) {
                    return backend::QueryMixRows{{out.get(), 0}, {s.get(), off}, {q.get(), 0}, items, queries};
                };
                p.dev.query_mix({mix(om.second, src.second, qm.second, at, 2, 4), mix(orr.second, src.second, qr.second, 0, n, nq)}, rows, dim, stride, 1e-6f);
                p.dev.query_mix({mix(sm.second, src.second, qm.second, at, 2, 4)}, rows, dim, stride, 1e-6f);
                p.dev.query_mix({mix(sr.second, src.second, qr.second, 0, n, nq)}, rows, dim, stride, 1e-6f);
                p.dev.sync();
                const std::string what = " of two query mixes in one call (" + std::to_string(n) + " items of " + std::to_string(dim) + " by " + std::to_string(nq) + " queries)";
                require(!std::memcmp(download(p.dev, *om.second, rows * 4 * dim).data(), download(p.dev, *sm.second, rows * 4 * dim).data(), rows * 4 * dim * sizeof(float)),
                        "the first" + what + " has the bits of it alone");
                require(!std::memcmp(download(p.dev, *orr.second, rows * nq * dim).data(), download(p.dev, *sr.second, rows * nq * dim).data(), rows * nq * dim * sizeof(float)),
                        "the second" + what + " has the bits of it alone");
            }
    // A call of more rows than any device's compute units takes the row form for held mixes of at most five items: each row has the bits of a five-row call, which takes the one-launch form.
    // The message's 2 items by 4 queries beside residual mixes of 1, 3, 5 and 6 items by 1 or 3 queries, at held rows, a row tail and a row the held builds do not take.
    for (size_t dim : {size_t(2304), size_t(1000), size_t(2305)})
        for (size_t n : {size_t(1), size_t(3), size_t(5), size_t(6)})
            for (size_t nq : {size_t(1), size_t(3)}) {
                const size_t rows = 320, part = 5, stride = 8 * dim, at = 6 * dim;
                auto src = p.put(random(rows * stride, -2.0f, 2.0f)), qm = p.put(random(4 * dim, -0.1f, 0.1f)), qr = p.put(random(nq * dim, -0.1f, 0.1f));
                auto om = p.put(std::vector<float>(rows * 4 * dim, 0.0f)), orr = p.put(std::vector<float>(rows * nq * dim, 0.0f));
                auto sm = p.put(std::vector<float>(rows * 4 * dim, 0.0f)), sr = p.put(std::vector<float>(rows * nq * dim, 0.0f));
                const auto mix = [&](const BufferPtr& out, size_t oo, const BufferPtr& q, size_t off, size_t items, size_t queries) {
                    return backend::QueryMixRows{{out.get(), oo}, {src.second.get(), off}, {q.get(), 0}, items, queries};
                };
                p.dev.query_mix({mix(om.second, 0, qm.second, at, 2, 4), mix(orr.second, 0, qr.second, 0, n, nq)}, rows, dim, stride, 1e-6f);
                for (size_t r = 0; r < rows; r += part)
                    p.dev.query_mix({mix(sm.second, r * 4 * dim, qm.second, r * stride + at, 2, 4), mix(sr.second, r * nq * dim, qr.second, r * stride, n, nq)}, part, dim, stride, 1e-6f);
                p.dev.sync();
                const std::string what = " of " + std::to_string(rows) + " rows (" + std::to_string(n) + " items of " + std::to_string(dim) + " by " + std::to_string(nq) + " queries beside the message)";
                require(!std::memcmp(download(p.dev, *om.second, rows * 4 * dim).data(), download(p.dev, *sm.second, rows * 4 * dim).data(), rows * 4 * dim * sizeof(float)),
                        "the message mix" + what + " has the bits of five-row calls");
                require(!std::memcmp(download(p.dev, *orr.second, rows * nq * dim).data(), download(p.dev, *sr.second, rows * nq * dim).data(), rows * nq * dim * sizeof(float)),
                        "the residual mix" + what + " has the bits of five-row calls");
            }
    // A normed mix has the bits of the mix then rms_norm_rows in place, and so has a product reading its twin: held rows of one to six items, seven items, and a row the held build does not take.
    for (size_t dim : {size_t(2304), size_t(1024), size_t(2305)})
        for (size_t n : {size_t(1), size_t(6), size_t(7)})
            for (size_t rows : {size_t(1), size_t(5)})
                for (backend::Dtype dtype : {backend::Dtype::f16, backend::Dtype::int8}) {
                    const size_t stride = 8 * dim, nout = 64;
                    const bool product = dim % 32 == 0;
                    BufferPtr src = upload(p.dev, random(rows * stride, -2.0f, 2.0f)), q = upload(p.dev, random(dim, -0.1f, 0.1f)), nw = upload(p.dev, random(dim, 0.5f, 1.5f));
                    BufferPtr out[2] = {upload(p.dev, std::vector<float>(rows * dim)), upload(p.dev, std::vector<float>(rows * dim))};
                    BufferPtr y[2] = {upload(p.dev, std::vector<float>(rows * nout)), upload(p.dev, std::vector<float>(rows * nout))};
                    std::vector<uint8_t> wq(nout * (dim / 32) * 34);
                    uint32_t seed = 12345;
                    for (size_t i = 0; i < wq.size(); ++i) wq[i] = uint8_t((seed = seed * 1664525u + 1013904223u) >> 24);
                    for (size_t b = 0; b < wq.size(); b += 34) wq[b] = 0x00, wq[b + 1] = 0x1c;
                    BufferPtr wb = p.dev.alloc(std::max<size_t>(wq.size(), 1), backend::Memory::device);
                    if (product) p.dev.write(*wb, 0, wq.data(), wq.size());
                    const backend::WeightView wv{quant::GGML_TYPE_Q8_0, {wb.get(), 0}};
                    for (int arm = 0; arm < 2; ++arm) {
                        // A product of the class first, so the mix or the norm writes the twin the next product reads.
                        if (product) p.dev.matmul(wv, {out[arm].get(), 0}, {y[arm].get(), 0}, dim, nout, rows, {}, dtype);
                        if (arm == 0) {
                            p.dev.query_mix({{{out[0].get(), 0}, {src.get(), 0}, {q.get(), 0}, n, 1, {{{nw.get(), 0}}}, 1e-5f}}, rows, dim, stride, 1e-6f);
                        } else {
                            p.dev.query_mix({{{out[1].get(), 0}, {src.get(), 0}, {q.get(), 0}, n, 1}}, rows, dim, stride, 1e-6f);
                            p.dev.rms_norm_rows({out[1].get(), 0}, {out[1].get(), 0}, {nw.get(), 0}, rows, dim, dim, 1e-5f);
                        }
                        if (product) p.dev.matmul(wv, {out[arm].get(), 0}, {y[arm].get(), 0}, dim, nout, rows, {}, dtype);
                    }
                    p.dev.sync();
                    require(!std::memcmp(download(p.dev, *out[0], rows * dim).data(), download(p.dev, *out[1], rows * dim).data(), rows * dim * sizeof(float)) &&
                                !std::memcmp(download(p.dev, *y[0], rows * nout).data(), download(p.dev, *y[1], rows * nout).data(), rows * nout * sizeof(float)),
                            "a normed query_mix of " + std::to_string(n) + " items over " + std::to_string(rows) + " rows of " + std::to_string(dim) + " at dtype " + std::to_string(int(dtype)) +
                                " has the bits of the mix then the norm, and so has its product");
                }
    // A mix of three queries normed each against its own weight has the bits of the mix then three rms_norm_rows over rows 3 dim apart, and so has a product of each plane, at one row and several, in either dtype.
    for (size_t dim : {size_t(2304), size_t(1024), size_t(2305)})
        for (size_t n : {size_t(1), size_t(6), size_t(7)})
            for (size_t rows : {size_t(1), size_t(5)})
                for (backend::Dtype dtype : {backend::Dtype::f16, backend::Dtype::int8}) {
                    const size_t stride = 8 * dim, nout = 64, nq = 3;
                    const bool product = dim % 32 == 0;
                    BufferPtr src = upload(p.dev, random(rows * stride, -2.0f, 2.0f)), q = upload(p.dev, random(nq * dim, -0.1f, 0.1f));
                    BufferPtr nw[3] = {upload(p.dev, random(dim, 0.5f, 1.5f)), upload(p.dev, random(dim, 0.5f, 1.5f)), upload(p.dev, random(dim, 0.5f, 1.5f))};
                    BufferPtr out[2] = {upload(p.dev, std::vector<float>(rows * nq * dim)), upload(p.dev, std::vector<float>(rows * nq * dim))};
                    BufferPtr y[2][3];
                    for (auto& arm : y)
                        for (auto& v : arm) v = upload(p.dev, std::vector<float>(rows * nout));
                    std::vector<uint8_t> wq(nout * (dim / 32) * 34);
                    uint32_t seed = 54321;
                    for (size_t i = 0; i < wq.size(); ++i) wq[i] = uint8_t((seed = seed * 1664525u + 1013904223u) >> 24);
                    for (size_t b = 0; b < wq.size(); b += 34) wq[b] = 0x00, wq[b + 1] = 0x1c;
                    BufferPtr wb = p.dev.alloc(std::max<size_t>(wq.size(), 1), backend::Memory::device);
                    if (product) p.dev.write(*wb, 0, wq.data(), wq.size());
                    const backend::WeightView wv{quant::GGML_TYPE_Q8_0, {wb.get(), 0}};
                    for (int arm = 0; arm < 2; ++arm) {
                        if (product) p.dev.matmul(wv, {out[arm].get(), 0}, {y[arm][0].get(), 0}, dim, nout, rows, {}, dtype);
                        if (arm == 0) {
                            backend::QueryMixRows m{{out[0].get(), 0}, {src.get(), 0}, {q.get(), 0}, n, nq};
                            for (size_t j = 0; j < nq; ++j) m.norm[j] = {nw[j].get(), 0};
                            m.norm_eps = 1e-5f;
                            p.dev.query_mix({m}, rows, dim, stride, 1e-6f);
                        } else {
                            p.dev.query_mix({{{out[1].get(), 0}, {src.get(), 0}, {q.get(), 0}, n, nq}}, rows, dim, stride, 1e-6f);
                            for (size_t j = 0; j < nq; ++j)
                                p.dev.rms_norm_rows({out[1].get(), j * dim}, {out[1].get(), j * dim}, {nw[j].get(), 0}, rows, dim, nq * dim, 1e-5f);
                        }
                        for (size_t j = 0; product && j < nq; ++j) p.dev.matmul(wv, {out[arm].get(), j * dim}, {y[arm][j].get(), 0}, dim, nout, 1, {}, dtype);
                    }
                    p.dev.sync();
                    bool same = !std::memcmp(download(p.dev, *out[0], rows * nq * dim).data(), download(p.dev, *out[1], rows * nq * dim).data(), rows * nq * dim * sizeof(float));
                    for (size_t j = 0; j < nq; ++j)
                        same = same && !std::memcmp(download(p.dev, *y[0][j], rows * nout).data(), download(p.dev, *y[1][j], rows * nout).data(), rows * nout * sizeof(float));
                    require(same, "a query_mix of " + std::to_string(n) + " items by 3 normed queries over " + std::to_string(rows) + " rows of " + std::to_string(dim) + " at dtype " + std::to_string(int(dtype)) +
                                      " has the bits of the mix then the norms, and so has each plane's product");
                }
    for (size_t C : {size_t(10), size_t(130), size_t(2304)})
        for (bool dynamic : {false, true})
            for (bool in_place : {false, true}) ring_conv_case(C, dynamic, in_place, 1, &dev);
    for (bool wide : {false, true})
        for (bool ring : {true, false}) attention_rel_sink_case(ring, 1, &dev, wide);
    // moe_step at the model's expert count under the sigmoid rule with eps, the softcap GLU and the fused gate-up stack read through row windows, decode and prompt rows.
    const size_t E = 128, nin = 256, ff = 64, k = 8, block = 34;
    const auto q8 = [&](size_t rows, size_t row_bytes) {
        std::vector<uint8_t> v(rows * row_bytes);
        for (size_t r = 0; r < rows; ++r)
            for (size_t b = 0; b < row_bytes / block; ++b) {
                uint8_t* q = v.data() + r * row_bytes + b * block;
                // A scale of (1 + m / 8) 2^-e of either sign, e from 6 to 8, exact in binary16 and F32.
                const uint32_t m = uint32_t(uniform(0.0f, 8.0f)), e = 6 + uint32_t(uniform(0.0f, 3.0f)), neg = uniform(0.0f, 1.0f) < 0.3f;
                const uint16_t bits = uint16_t(neg << 15 | (15 - e) << 10 | m << 7);
                q[0] = uint8_t(bits & 0xff), q[1] = uint8_t(bits >> 8);
                for (size_t i = 0; i < 32; ++i) q[2 + i] = uint8_t(int8_t(int(uniform(-127.0f, 127.0f))));
            }
        return v;
    };
    const std::vector<uint8_t> fused = q8(E * 2 * ff, nin / 32 * block), down = q8(E * nin, ff / 32 * block);
    for (size_t rows : {size_t(1), size_t(3), size_t(70)}) {
        const auto bytes = [&](const std::vector<uint8_t>& v) {
            std::pair<BufferPtr, BufferPtr> b{p.cpu.alloc(v.size(), backend::Memory::device), p.dev.alloc(v.size(), backend::Memory::device)};
            p.cpu.write(*b.first, 0, v.data(), v.size());
            p.dev.write(*b.second, 0, v.data(), v.size());
            return b;
        };
        auto fb = bytes(fused), db = bytes(down), rb = p.put(random(E * nin, -0.3f, 0.3f)), bias = p.put(random(E, -0.1f, 0.1f)), hb = p.put(random(rows * nin, -1.0f, 1.0f));
        auto scores = p.put(std::vector<float>(rows * E)), ids = p.put(std::vector<float>(rows * k)), w = p.put(std::vector<float>(rows * k)),
             go = p.put(std::vector<float>(rows * k * ff)), uo = p.put(std::vector<float>(rows * k * ff)), act = p.put(std::vector<float>(rows * k * ff)),
             yb = p.put(random(rows * nin, -4.0f, 4.0f));
        const std::vector<float> y0 = download(p.cpu, *yb.first, rows * nin);
        std::vector<backend::RowRun> scratch[2];
        const backend::RowRun runs[1] = {{rows, rows == 1 ? size_t(1) : rows}};
        for (int side = 0; side < 2; ++side) {
            auto pick = [&](const std::pair<BufferPtr, BufferPtr>& b) { return side ? b.second.get() : b.first.get(); };
            backend::Projection gp{{quant::GGML_TYPE_Q8_0, {pick(fb), 0}}, {pick(go), 0}, ff}, up_p{{quant::GGML_TYPE_Q8_0, {pick(fb), 0}}, {pick(uo), 0}, ff};
            gp.pitch = up_p.pitch = 2 * ff, up_p.first = ff;
            const backend::MoeStep m{backend::Projection{{quant::GGML_TYPE_F32, {pick(rb), 0}}, {pick(scores), 0}, E}, k, backend::RouteRule::sigmoid({pick(bias), 0}, 8.0f, true, 1e-20f),
                                     {pick(ids), 0}, {pick(w), 0}, gp, up_p, {quant::GGML_TYPE_Q8_0, {pick(db), 0}}, {pick(act), 0}, nullptr, nullptr, &scratch[side],
                                     {4.0f, backend::GluForm::softcap, 25.0f}};
            (side ? p.dev : static_cast<backend::Backend&>(p.cpu)).moe_step(m, {pick(hb), 0}, {pick(yb), 0}, nin, rows, {runs, 1}, backend::Dtype::f32);
        }
        p.same(ids, rows * k, 0.0, "sigmoid route ids over " + std::to_string(rows) + " rows");
        p.same(w, rows * k, 1e-6, "sigmoid route weights over " + std::to_string(rows) + " rows");
        // Each side's products against double sums of its own inputs, within the F32 accumulation bound gamma(n) * sum |terms| of docs/PRECISION.md (Kernel checks), since the device sums in another order.
        p.dev.sync();
        const size_t entries = rows * k;
        const auto get = [&](const std::pair<BufferPtr, BufferPtr>& b, size_t n, int side) {
            std::vector<float> v(n);
            (side ? p.dev : static_cast<backend::Backend&>(p.cpu)).read(side ? *b.second : *b.first, 0, v.data(), n * sizeof(float));
            return v;
        };
        // Q8_0 decoded exactly from the scale bits the blocks were given.
        const auto weight = [&](const std::vector<uint8_t>& v, size_t row, size_t width, size_t j) {
            const uint8_t* q = v.data() + row * (width / 32 * block) + j / 32 * block;
            const uint32_t bits = uint32_t(q[0]) | uint32_t(q[1]) << 8;
            const double d = (bits >> 15 ? -1.0 : 1.0) * (1.0 + double(bits >> 7 & 7) / 8.0) * std::ldexp(1.0, int(bits >> 10 & 31) - 15);
            return d * double(int8_t(q[2 + j % 32]));
        };
        const auto gamma = [](size_t n) { const double u = std::ldexp(1.0, -24); return double(n) * u / (1.0 - double(n) * u); };
        const std::vector<float> h = get(hb, rows * nin, 0);
        std::vector<float> sides[2][5];
        for (int side = 0; side < 2; ++side) {
            std::vector<float>* s = sides[side];
            s[0] = get(go, entries * ff, side), s[1] = get(uo, entries * ff, side), s[2] = get(act, entries * ff, side), s[3] = get(w, entries, side),
            s[4] = get(yb, rows * nin, side);
            double worst[2] = {0, 0};
            std::vector<uint32_t> id(entries);
            std::memcpy(id.data(), get(ids, entries, side).data(), entries * sizeof(uint32_t));
            for (size_t i = 0; i < entries; ++i)
                for (size_t o = 0; o < 2 * ff; ++o) {
                    double ref = 0, mag = 0;
                    for (size_t j = 0; j < nin; ++j) {
                        const double t = weight(fused, id[i] * 2 * ff + o, nin, j) * h[i / k * nin + j];
                        ref += t, mag += std::fabs(t);
                    }
                    const double err = std::fabs(s[o < ff ? 0 : 1][i * ff + o % ff] - ref) / (gamma(nin + 2) * mag);
                    worst[0] = std::isnan(worst[0]) || err <= worst[0] ? worst[0] : err;   // a NaN error stays the worst
                }
            for (size_t r = 0; r < rows; ++r)
                for (size_t o = 0; o < nin; ++o) {
                    double ref = y0[r * nin + o], mag = std::fabs(y0[r * nin + o]);
                    for (size_t slot = 0; slot < k; ++slot) {
                        const size_t i = r * k + slot;
                        double dot = 0, dmag = 0;
                        for (size_t j = 0; j < ff; ++j) {
                            const double t = weight(down, id[i] * nin + o, ff, j) * s[2][i * ff + j];
                            dot += t, dmag += std::fabs(t);
                        }
                        ref += s[3][i] * dot, mag += std::fabs(s[3][i]) * dmag;
                    }
                    const double err = std::fabs(s[4][r * nin + o] - ref) / (gamma(ff + k + 3) * mag);
                    worst[1] = std::isnan(worst[1]) || err <= worst[1] ? worst[1] : err;
                }
            const std::string what = std::string(side ? "device" : "cpu") + " moe_step over " + std::to_string(rows) + " rows";
            std::cout << what << ": gate/up error " << worst[0] << " and y error " << worst[1] << " of the F32 accumulation bound\n";
            require(worst[0] <= 1.0, what + ": gate and up rows within the F32 accumulation bound of their double sums");
            require(worst[1] <= 1.0, what + ": y within the F32 accumulation bound of the double sum of its own act and weights");
        }
        // The GLU is elementwise, so its rows keep 1e-4 of the CPU's; y then differs from the CPU's by both sides' bounds plus what act's and the weights' differences carry, w_dev (a_dev - a_cpu) + (w_dev - w_cpu) a_cpu.
        p.same(act, entries * ff, 1e-4, "softcap GLU rows over " + std::to_string(rows) + " rows");
        std::vector<uint32_t> id(entries);
        std::memcpy(id.data(), get(ids, entries, 1).data(), entries * sizeof(uint32_t));
        double worst = 0;
        for (size_t r = 0; r < rows; ++r)
            for (size_t o = 0; o < nin; ++o) {
                double carried = 0, mag[2] = {0, 0};
                for (size_t slot = 0; slot < k; ++slot) {
                    const size_t i = r * k + slot;
                    double dot = 0;
                    for (size_t j = 0; j < ff; ++j) {
                        const double d = std::fabs(weight(down, id[i] * nin + o, ff, j));
                        dot += d * std::fabs(sides[0][2][i * ff + j]);
                        carried += std::fabs(sides[1][3][i]) * d * std::fabs(double(sides[1][2][i * ff + j]) - sides[0][2][i * ff + j]);
                        for (int side = 0; side < 2; ++side) mag[side] += std::fabs(sides[side][3][i]) * d * std::fabs(sides[side][2][i * ff + j]);
                    }
                    carried += std::fabs(double(sides[1][3][i]) - sides[0][3][i]) * dot;
                }
                const double bound = carried + gamma(ff + k + 3) * (mag[0] + mag[1] + 2 * std::fabs(y0[r * nin + o]));
                const double diff = std::fabs(double(sides[1][4][r * nin + o]) - sides[0][4][r * nin + o]);
                const double err = bound > 0 ? diff / bound : diff > 0 ? 1e30 : 0.0;
                worst = std::isnan(worst) || err <= worst ? worst : err;
            }
        std::cout << "device moe_step over " << rows << " rows: y differs from the cpu's by " << worst << " of the carried bound\n";
        require(worst <= 1.0, "device moe_step over " + std::to_string(rows) + " rows: y within both sides' F32 bounds and the carried act and weight differences of the cpu's");
    }
}

// The ops over state slots on the device and the CPU from the same inputs: a call sequence over three slots with a prompt longer than the ring, forks and two views a call, outputs and whole state layers compared.
struct SlotCall {
    size_t src, dst, length, nq;
};
const std::vector<std::vector<SlotCall>> kSlotCalls = {{{0, 0, 0, 70}}, {{0, 0, 70, 1}}, {{0, 1, 71, 5}}, {{0, 0, 71, 1}, {1, 1, 76, 3}}, {{1, 2, 79, 130}}, {{2, 2, 209, 1}}, {{2, 2, 210, 1100}}, {{2, 0, 1310, 300}, {1, 1, 79, 33}}};

void same_state(Pair& p, backend::StateStorage& cs, backend::StateStorage& ds, double tol, const std::string& what) {
    p.dev.sync();
    const size_t n = cs.layer(0).size() / sizeof(float);
    std::vector<float> d(n), c(n);
    p.dev.read(ds.layer(0), 0, d.data(), n * sizeof(float));
    p.cpu.read(cs.layer(0), 0, c.data(), n * sizeof(float));
    close(d, std::vector<double>(c.begin(), c.end()), tol, "device " + what);
}

void device_ring_conv(Pair& p, size_t C, bool dynamic, bool in_place) {
    const size_t K = 4, heads = 8, dyn_stride = heads * K + 3, ring_rows = K - 1 + 64, ring_offset = 5, slots = 3;
    const float divisor = dynamic ? 3.0f : std::sqrt(5.0f);
    const std::vector<size_t> tails = {ring_offset + ring_rows * C + 7};
    std::unique_ptr<backend::StateStorage> cs = p.cpu.state_alloc(1, slots, backend::StateShape{}, tails), ds = p.dev.state_alloc(1, slots, backend::StateShape{}, tails);
    auto taps = p.put(random(C * K, -1.0f, 1.0f));
    const std::string at = std::to_string(C) + " channels" + (dynamic ? ", dynamic" : "") + (in_place ? ", in place" : "");
    for (const std::vector<SlotCall>& calls : kSlotCalls) {
        size_t rows = 0;
        std::vector<backend::StateView> cv, dv;
        for (const SlotCall& c : calls) {
            cv.push_back({cs.get(), c.src, c.dst, c.length, c.nq});
            dv.push_back({ds.get(), c.src, c.dst, c.length, c.nq});
            rows += c.nq;
        }
        auto x = p.put(random(rows * C, -2.0f, 2.0f)), dyn = p.put(random(rows * dyn_stride, -0.5f, 0.5f)), apart = p.put(std::vector<float>(rows * C));
        auto& o = in_place ? x : apart;
        for (int side = 0; side < 2; ++side) {
            auto pick = [&](const std::pair<BufferPtr, BufferPtr>& b) { return side ? b.second.get() : b.first.get(); };
            const backend::DepthwiseConv conv = ring_conv({pick(taps), 0}, dynamic ? backend::CSlice{pick(dyn), 0} : backend::CSlice{}, C, K, dynamic ? heads : 0, dyn_stride, divisor,
                                                          ring_offset, ring_rows);
            (side ? p.dev : static_cast<backend::Backend&>(p.cpu)).depthwise_conv({pick(o), 0}, {pick(x), 0}, conv, 0, side ? dv.data() : cv.data(), calls.size());
        }
        p.same(o, rows * C, 1e-5, "depthwise_conv over a ring of " + at + " at length " + std::to_string(calls[0].length));
    }
    same_state(p, *cs, *ds, 0.0, "depthwise_conv rings of " + at);
}

// On the device alone, a list of two ring convolutions gives each item's bits of its own call, outputs and state: apart (one launch), and with the second reading the first's output (in order).
void device_ring_conv_list(backend::Backend& dev, bool chained) {
    const size_t K = 4, heads = 8, dyn_stride = heads * K + 3, ring_rows = K - 1 + 64, slots = 3, C[2] = {1024, chained ? size_t(1024) : size_t(136)};
    const size_t offset[2] = {5, 5 + ring_rows * C[0] + 3};
    const std::vector<size_t> tails = {offset[1] + ring_rows * C[1] + 7};
    std::unique_ptr<backend::StateStorage> listed = dev.state_alloc(1, slots, backend::StateShape{}, tails), single = dev.state_alloc(1, slots, backend::StateShape{}, tails);
    BufferPtr taps[2], dyn[2];
    for (int i = 0; i < 2; ++i) taps[i] = upload(dev, random(C[i] * K, -1.0f, 1.0f));
    const std::string at = chained ? "the second reading the first's output" : "apart";
    for (const std::vector<SlotCall>& calls : kSlotCalls) {
        size_t rows = 0;
        std::vector<backend::StateView> lv, sv;
        for (const SlotCall& c : calls) {
            lv.push_back({listed.get(), c.src, c.dst, c.length, c.nq});
            sv.push_back({single.get(), c.src, c.dst, c.length, c.nq});
            rows += c.nq;
        }
        backend::DepthwiseConv conv[2];
        std::vector<float> x[2];
        for (int i = 0; i < 2; ++i) {
            dyn[i] = upload(dev, random(rows * dyn_stride, -0.5f, 0.5f));
            x[i] = random(rows * C[i], -2.0f, 2.0f);
            conv[i] = ring_conv({taps[i].get(), 0}, i ? backend::CSlice{} : backend::CSlice{dyn[i].get(), 0}, C[i], K, i ? 0 : heads, dyn_stride, 3.0f, offset[i], ring_rows);
        }
        // Each side's x[0], x[1] and the second's output, the first item's output being x[0] in place.
        BufferPtr la = upload(dev, x[0]), lb = upload(dev, x[1]), lo = upload(dev, std::vector<float>(rows * C[1]));
        BufferPtr sa = upload(dev, x[0]), sb = upload(dev, x[1]), so = upload(dev, std::vector<float>(rows * C[1]));
        dev.depthwise_conv({{{la.get(), 0}, {la.get(), 0}, conv[0]}, {{lo.get(), 0}, {chained ? la.get() : lb.get(), 0}, conv[1]}}, 0, lv.data(), calls.size());
        dev.depthwise_conv({sa.get(), 0}, {sa.get(), 0}, conv[0], 0, sv.data(), calls.size());
        dev.depthwise_conv({so.get(), 0}, {chained ? sa.get() : sb.get(), 0}, conv[1], 0, sv.data(), calls.size());
        dev.sync();
        const auto same = [&](BufferPtr& a, BufferPtr& b, size_t n, const std::string& what) {
            if (std::memcmp(download(dev, *a, n).data(), download(dev, *b, n).data(), n * sizeof(float)))
                throw std::runtime_error("depthwise_conv list, " + at + ": " + what + " differs from its own call at length " + std::to_string(calls[0].length));
        };
        same(la, sa, rows * C[0], "the first item");
        same(lo, so, rows * C[1], "the second item");
    }
    const size_t n = listed->layer(0).size() / sizeof(float);
    std::vector<float> a(n), b(n);
    dev.read(listed->layer(0), 0, a.data(), n * sizeof(float));
    dev.read(single->layer(0), 0, b.data(), n * sizeof(float));
    require(!std::memcmp(a.data(), b.data(), n * sizeof(float)), "depthwise_conv list, " + at + ": the state layer has the bits of the single calls");
}

// A clipped and a normed convolution of heads of D channels, one list call, give the bits of the plain convs followed by l2_clip and group_rms_norm on the same device, at decode and over prompts.
// Head 0 is all zero, so the clip's 1e-6 and the norm's eps alone decide it; D 32 heads are left to the ops on a device whose fused heads take whole waves.
void device_conv_post(backend::Backend& dev, size_t D, size_t heads) {
    const size_t K = 4, C = D * heads, ring_rows = K - 1 + 64, slots = 3, offset[2] = {3, 3 + ring_rows * C + 5};
    const float clip = 1.5f, eps = 1e-6f;
    const std::vector<size_t> tails = {offset[1] + ring_rows * C + 2};
    std::unique_ptr<backend::StateStorage> fused = dev.state_alloc(1, slots, backend::StateShape{}, tails), plain = dev.state_alloc(1, slots, backend::StateShape{}, tails);
    // The k and v items' rows also go to KV storage of mixed side types, each slot's sequence on blocks in reverse order, through ConvKv on one arm and kv_write after the ops on the other.
    const size_t bt = dev.kv_layout().block_tokens, nb = backend::blocks_for(2048, bt);
    std::unique_ptr<backend::KVStorage> fkv = dev.kv_alloc(1, heads, D, slots * nb * bt, backend::KVType::f16, backend::KVType::f32),
                                        pkv = dev.kv_alloc(1, heads, D, slots * nb * bt, backend::KVType::f16, backend::KVType::f32);
    std::vector<std::vector<int32_t>> kvb(slots);
    for (size_t s = 0; s < slots; ++s)
        for (size_t j = 0; j < nb; ++j) kvb[s].push_back(int32_t((nb - 1 - j) * slots + s));
    // A fresh storage backs no block, and a backed block reads as zeros.
    fkv->back_all();
    pkv->back_all();
    std::vector<float> taps = random(C * K, -1.0f, 1.0f);
    for (size_t c = 0; c < D; ++c)
        for (size_t k = 0; k < K; ++k) taps[c * K + k] = 0.0f;
    BufferPtr tb = upload(dev, taps), nw = upload(dev, random(C, 0.5f, 1.5f));
    const std::string at = std::to_string(heads) + " heads of " + std::to_string(D);
    for (const std::vector<SlotCall>& calls : kSlotCalls) {
        size_t rows = 0;
        std::vector<backend::StateView> fv, pv;
        std::vector<backend::KVView> fk, pk;
        for (const SlotCall& c : calls) {
            fv.push_back({fused.get(), c.src, c.dst, c.length, c.nq});
            pv.push_back({plain.get(), c.src, c.dst, c.length, c.nq});
            fk.push_back({fkv.get(), kvb[c.dst].data(), nb, c.length, c.nq});
            pk.push_back({pkv.get(), kvb[c.dst].data(), nb, c.length, c.nq});
            rows += c.nq;
        }
        const backend::ConvKv to{0, fk.data(), fk.size(), 0, 1};
        std::vector<float> x[2];
        BufferPtr fx[2], px[2];
        backend::DepthwiseConv conv[2], post[2];
        for (int i = 0; i < 2; ++i) {
            x[i] = random(rows * C, -2.0f, 2.0f);
            for (size_t r = 0; r < rows; ++r)
                for (size_t c = 0; c < D; ++c) x[i][r * C + c] = 0.0f;
            fx[i] = upload(dev, x[i]), px[i] = upload(dev, x[i]);
            conv[i] = post[i] = ring_conv({tb.get(), 0}, {}, C, K, 0, 0, 3.0f, offset[i], ring_rows);
        }
        post[0].post = backend::ConvPost::l2_clip, post[0].post_heads = heads, post[0].post_limit = clip;
        post[1].post = backend::ConvPost::rms_norm, post[1].post_heads = heads, post[1].post_limit = eps, post[1].post_w = {nw.get(), 0};
        dev.depthwise_conv({{{fx[0].get(), 0}, {fx[0].get(), 0}, post[0]}, {{fx[1].get(), 0}, {fx[1].get(), 0}, post[1]}}, 0, fv.data(), calls.size(), &to);
        dev.depthwise_conv({{{px[0].get(), 0}, {px[0].get(), 0}, conv[0]}, {{px[1].get(), 0}, {px[1].get(), 0}, conv[1]}}, 0, pv.data(), calls.size());
        dev.l2_clip({{{px[0].get(), 0}, {}, {}, heads, D, C, clip}}, rows);
        dev.group_rms_norm({px[1].get(), 0}, {px[1].get(), 0}, {nw.get(), 0}, rows, heads, D, eps);
        dev.kv_write(0, pk.data(), pk.size(), {px[0].get(), 0}, {px[1].get(), 0});
        dev.sync();
        for (int i = 0; i < 2; ++i)
            require(!std::memcmp(download(dev, *fx[i], rows * C).data(), download(dev, *px[i], rows * C).data(), rows * C * sizeof(float)),
                    std::string("depthwise_conv with its ") + (i ? "norm" : "clip") + ", " + at + ": the bits of the conv and the op at length " + std::to_string(calls[0].length));
    }
    const size_t n = fused->layer(0).size() / sizeof(float);
    std::vector<float> a(n), b(n);
    dev.read(fused->layer(0), 0, a.data(), n * sizeof(float));
    dev.read(plain->layer(0), 0, b.data(), n * sizeof(float));
    require(!std::memcmp(a.data(), b.data(), n * sizeof(float)), "depthwise_conv with post steps, " + at + ": the state layer has the plain convs' bits");
    const auto& fs = dynamic_cast<const backend::BlockKVStorage&>(*fkv);
    const auto& ps = dynamic_cast<const backend::BlockKVStorage&>(*pkv);
    for (int side = 0; side < 2; ++side) {
        const backend::Buffer& fb = side ? *fs.v_buffer(0) : *fs.k_buffer(0);
        const backend::Buffer& pb = side ? *ps.v_buffer(0) : *ps.k_buffer(0);
        std::vector<uint8_t> x(fb.size()), y(pb.size());
        dev.read(fb, 0, x.data(), x.size());
        dev.read(pb, 0, y.data(), y.size());
        require(x == y, "depthwise_conv with post steps and a KV write, " + at + ": the " + (side ? "v" : "k") + " side has the bits of kv_write after the ops");
    }
}

// On the device alone, the fused gate-up stack read through row windows gives the bits of split gate and up stacks under every dtype's path, decode, a few rows and a prompt.
void device_window_paths(backend::Backend& dev) {
    const size_t E = 128, nin = 2304, ff = 768, k = 8, block = 34, row_in = nin / 32 * block;
    std::vector<uint8_t> fused(E * 2 * ff * row_in), gate(E * ff * row_in), up(E * ff * row_in), down(E * nin * (ff / 32 * block));
    for (std::vector<uint8_t>* v : {&fused, &down})
        for (size_t b = 0; b < v->size() / block; ++b) {
            uint8_t* q = v->data() + b * block;
            q[0] = 0x00, q[1] = 0x20;
            for (size_t i = 0; i < 32; ++i) q[2 + i] = uint8_t(int8_t(int(uniform(-127.0f, 127.0f))));
        }
    for (size_t e = 0; e < E; ++e) {
        std::copy_n(fused.begin() + long(e * 2 * ff * row_in), ff * row_in, gate.begin() + long(e * ff * row_in));
        std::copy_n(fused.begin() + long((e * 2 * ff + ff) * row_in), ff * row_in, up.begin() + long(e * ff * row_in));
    }
    const auto bytes = [&](const std::vector<uint8_t>& v) {
        BufferPtr b = dev.alloc(v.size(), backend::Memory::device);
        dev.write(*b, 0, v.data(), v.size());
        return b;
    };
    const auto floats = [&](const std::vector<float>& v) { return bytes(std::vector<uint8_t>((const uint8_t*)v.data(), (const uint8_t*)(v.data() + v.size()))); };
    BufferPtr fb = bytes(fused), gb = bytes(gate), ub = bytes(up), db = bytes(down), rb = floats(random(E * nin, -0.05f, 0.05f)), bias = floats(random(E, -0.1f, 0.1f));
    for (backend::Dtype dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::int8})
        for (size_t rows : {size_t(1), size_t(3), size_t(70)}) {
            BufferPtr sb = floats(random(rows * nin, -1.0f, 1.0f)), nb = floats(random(nin, 0.5f, 1.5f)), hb = floats(std::vector<float>(rows * nin)),
                      hin = floats(std::vector<float>(rows * nin));
            dev.rms_norm_rows({hb.get(), 0}, {sb.get(), 0}, {nb.get(), 0}, rows, nin, nin, 1e-6f);
            const backend::MoeInput in{{sb.get(), 0}, {nb.get(), 0}, 1e-6f};
            std::vector<float> y[3];
            // The third arm forms h in the step from its unnormed source (MoeStep::input), fused into the routing launch on a decode row.
            for (int arm = 0; arm < 3; ++arm) {
                const int windowed = arm > 0;
                BufferPtr scores = floats(std::vector<float>(rows * E)), ids = floats(std::vector<float>(rows * k)), w = floats(std::vector<float>(rows * k)),
                          go = floats(std::vector<float>(rows * k * ff)), uo = floats(std::vector<float>(rows * k * ff)), act = floats(std::vector<float>(rows * k * ff)),
                          yb = floats(std::vector<float>(rows * nin));
                std::vector<backend::RowRun> scratch;
                backend::Projection gp{{quant::GGML_TYPE_Q8_0, {(windowed ? fb : gb).get(), 0}}, {go.get(), 0}, ff}, up_p{{quant::GGML_TYPE_Q8_0, {(windowed ? fb : ub).get(), 0}}, {uo.get(), 0}, ff};
                if (windowed) gp.pitch = up_p.pitch = 2 * ff, up_p.first = ff;
                const backend::MoeStep m{backend::Projection{{quant::GGML_TYPE_F32, {rb.get(), 0}}, {scores.get(), 0}, E}, k, backend::RouteRule::sigmoid({bias.get(), 0}, 8.0f, true, 1e-20f),
                                         {ids.get(), 0}, {w.get(), 0}, gp, up_p, {quant::GGML_TYPE_Q8_0, {db.get(), 0}}, {act.get(), 0}, nullptr, nullptr, &scratch,
                                         {4.0f, backend::GluForm::softcap, 25.0f}, arm == 2 ? &in : nullptr};
                const backend::RowRun runs[1] = {{rows, rows == 1 ? size_t(1) : rows}};
                dev.moe_step(m, {(arm == 2 ? hin : hb).get(), 0}, {yb.get(), 0}, nin, rows, {runs, 1}, dtype);
                dev.sync();
                y[arm].resize(rows * nin);
                dev.read(*yb, 0, y[arm].data(), rows * nin * sizeof(float));
            }
            const std::string at = std::to_string(rows) + " rows at dtype " + std::to_string(int(dtype));
            require(std::memcmp(y[0].data(), y[1].data(), rows * nin * sizeof(float)) == 0, "device windowed gate-up moe_step over " + at + " gives the split stacks' bits");
            std::vector<float> h[2] = {std::vector<float>(rows * nin), std::vector<float>(rows * nin)};
            dev.read(*hb, 0, h[0].data(), rows * nin * sizeof(float));
            dev.read(*hin, 0, h[1].data(), rows * nin * sizeof(float));
            require(std::memcmp(h[0].data(), h[1].data(), rows * nin * sizeof(float)) == 0 && std::memcmp(y[1].data(), y[2].data(), rows * nin * sizeof(float)) == 0,
                    "device moe_step forming its input over " + at + " gives rms_norm_rows' h and the step's bits");
        }
}

// attention_rel_sink at the model's heads over a ring (a short window that wraps, and the model's window) or pooled rows, outputs at each call and the rings at the end; later calls read what earlier ones stored.
// With `pool_tokens` above 2048, pooled sequence 0 holds every block of a pool that size, and `heads` and `kv_heads` may be cut to keep a long history cheap on the CPU.
void device_rel_sink(Pair& p, bool ring, size_t window, size_t extent, size_t ring_rows, const std::vector<std::vector<SlotCall>>& seq, size_t pool_tokens = 2048,
                     size_t heads = 24, size_t kv_heads = 8) {
    const size_t D = 128, R = 16, KD = kv_heads * D, HD = heads * D, ring_offset = 9;
    const float scale = float(1.0 / std::sqrt(double(D)));
    const std::vector<size_t> tails = {ring_offset + ring_rows * 2 * KD + 3};
    std::unique_ptr<backend::StateStorage> cs = p.cpu.state_alloc(1, 3, backend::StateShape{}, tails), ds = p.dev.state_alloc(1, 3, backend::StateShape{}, tails);
    std::unique_ptr<backend::KVStorage> ck = p.cpu.kv_alloc(1, 0, 0, pool_tokens, backend::KVType::f32, backend::KVType::f32, {0}, {{1, 2 * KD}}),
                                        dk = p.dev.kv_alloc(1, 0, 0, pool_tokens, backend::KVType::f32, backend::KVType::f32, {0}, {{1, 2 * KD}});
    // The CPU and the device hold another number of tokens a block, so a pools whole block list is each storages own.
    std::vector<std::vector<int32_t>> blocks = {{5, 0, 1, 2, 3, 4}, {9, 7, 8, 6, 10, 11}, {12, 13, 14, 15, 16, 17}}, dblocks = blocks;
    if (pool_tokens > 2048) {
        blocks[0].clear();
        dblocks[0].clear();
        for (size_t b = 0; b < ck->max_blocks(); ++b) blocks[0].push_back(int32_t(b));
        for (size_t b = 0; b < dk->max_blocks(); ++b) dblocks[0].push_back(int32_t(b));
    }
    auto table = p.put(random(R * extent, -0.5f, 0.5f));
    const std::string at = std::string(ring ? "a ring" : "pooled rows") + " of window " + std::to_string(window);
    for (const std::vector<SlotCall>& calls : seq) {
        size_t rows = 0;
        std::vector<backend::StateView> cv, dv;
        std::vector<backend::KVView> cp, dp;
        for (const SlotCall& c : calls) {
            cv.push_back({cs.get(), c.src, c.dst, c.length, c.nq});
            dv.push_back({ds.get(), c.src, c.dst, c.length, c.nq});
            cp.push_back({ck.get(), blocks[c.src].data(), blocks[c.src].size(), c.length, c.nq});
            dp.push_back({dk.get(), dblocks[c.src].data(), dblocks[c.src].size(), c.length, c.nq});
            rows += c.nq;
        }
        auto q = p.put(random(rows * HD, -1.0f, 1.0f)), k = p.put(random(rows * KD, -1.0f, 1.0f)), v = p.put(random(rows * KD, -1.0f, 1.0f)),
             rel = p.put(random(rows * heads * R, -1.0f, 1.0f)), sink = p.put(random(rows * heads, -2.0f, 2.0f)), o = p.put(std::vector<float>(rows * HD));
        for (int side = 0; side < 2; ++side) {
            auto pick = [&](const std::pair<BufferPtr, BufferPtr>& b) { return side ? b.second.get() : b.first.get(); };
            (side ? p.dev : static_cast<backend::Backend&>(p.cpu))
                .attention_rel_sink({pick(o), 0}, {pick(q), 0}, {pick(k), 0}, {pick(v), 0}, {pick(rel), 0}, {pick(table), 0}, {pick(sink), 0}, heads, kv_heads, D, R, extent,
                                    window, scale, 0, ring ? (side ? dv : cv).data() : nullptr, ring ? nullptr : (side ? dp : cp).data(), calls.size(), ring_offset, ring_rows);
        }
        p.same(o, rows * HD, 1e-5, "attention_rel_sink over " + at + " at length " + std::to_string(calls[0].length));
        // The device's gated call, again over the keys it stored, gives the bits of the output then sigmoid_mul, and so does a product reading its twin; a ring call reads the same keys again unless its stores overwrote some.
        bool again = true;
        for (const SlotCall& c : calls) again = again && (!ring || c.src != c.dst || c.nq + window <= ring_rows + 1);
        const size_t gs = HD + 5, nout = 64;
        // Buffers written and freed unsynced would leave the device's pending copies naming freed memory, so they exist only when the check runs.
        if (again) {
            BufferPtr gate = upload(p.dev, random(rows * gs, -4.0f, 4.0f)), w = upload(p.dev, random(HD * nout, -0.2f, 0.2f)), og = upload(p.dev, std::vector<float>(rows * HD));
            BufferPtr yg = upload(p.dev, std::vector<float>(rows * nout)), ya = upload(p.dev, std::vector<float>(rows * nout));
            const backend::WeightView wv{quant::GGML_TYPE_F32, {w.get(), 0}};
            for (backend::Dtype dtype : {backend::Dtype::f16, backend::Dtype::int8}) {
                p.dev.attention_rel_sink({og.get(), 0}, {q.second.get(), 0}, {k.second.get(), 0}, {v.second.get(), 0}, {rel.second.get(), 0}, {table.second.get(), 0}, {sink.second.get(), 0},
                                         heads, kv_heads, D, R, extent, window, scale, 0, ring ? dv.data() : nullptr, ring ? nullptr : dp.data(), calls.size(), ring_offset, ring_rows, {},
                                         {{gate.get(), 0}, rows, gs, D, {}});
                p.dev.matmul(wv, {og.get(), 0}, {yg.get(), 0}, HD, nout, rows, {}, dtype);
                p.dev.sigmoid_mul({o.second.get(), 0}, {o.second.get(), 0}, {gate.get(), 0}, rows, heads, D, gs, D);
                p.dev.matmul(wv, {o.second.get(), 0}, {ya.get(), 0}, HD, nout, rows, {}, dtype);
                p.dev.sync();
                require(!std::memcmp(download(p.dev, *og, rows * HD).data(), download(p.dev, *o.second, rows * HD).data(), rows * HD * sizeof(float)) &&
                            !std::memcmp(download(p.dev, *yg, rows * nout).data(), download(p.dev, *ya, rows * nout).data(), rows * nout * sizeof(float)),
                        "device gated attention_rel_sink over " + at + " at length " + std::to_string(calls[0].length) + " has the bits of the output then sigmoid_mul, and so has its product");
                // The second pass gates the gated output again: rerun the ungated call to restore it.
                p.dev.attention_rel_sink({o.second.get(), 0}, {q.second.get(), 0}, {k.second.get(), 0}, {v.second.get(), 0}, {rel.second.get(), 0}, {table.second.get(), 0}, {sink.second.get(), 0},
                                         heads, kv_heads, D, R, extent, window, scale, 0, ring ? dv.data() : nullptr, ring ? nullptr : dp.data(), calls.size(), ring_offset, ring_rows);
            }
            p.dev.sync();
            // A clip in the call gives the bits of l2_clip and then the call (RelClip), with and without a row scale.
            const std::vector<float> qh = download(p.dev, *q.second, rows * HD), rh = download(p.dev, *rel.second, rows * heads * R);
            BufferPtr qw = upload(p.dev, random(HD, 0.5f, 1.5f)), rw = upload(p.dev, random(heads * R, 0.5f, 1.5f)), ls = upload(p.dev, random(rows, 1.0f, 1.5f));
            for (int scaled = 0; scaled < 2; ++scaled) {
                BufferPtr qa = upload(p.dev, qh), ra = upload(p.dev, rh), qb = upload(p.dev, qh), rb = upload(p.dev, rh);
                BufferPtr oa = upload(p.dev, std::vector<float>(rows * HD)), ob = upload(p.dev, std::vector<float>(rows * HD));
                const backend::RelClip clip{{qw.get(), 0}, {rw.get(), 0}, scaled ? backend::CSlice{ls.get(), 0} : backend::CSlice{}, rows, 2.0f, 1.5f};
                p.dev.l2_clip({{{qa.get(), 0}, clip.q_scale, clip.row_scale, heads, D, HD, clip.q_limit}, {{ra.get(), 0}, clip.rel_scale, clip.row_scale, heads, R, heads * R, clip.rel_limit}},
                              rows);
                p.dev.attention_rel_sink({oa.get(), 0}, {qa.get(), 0}, {k.second.get(), 0}, {v.second.get(), 0}, {ra.get(), 0}, {table.second.get(), 0}, {sink.second.get(), 0},
                                         heads, kv_heads, D, R, extent, window, scale, 0, ring ? dv.data() : nullptr, ring ? nullptr : dp.data(), calls.size(), ring_offset, ring_rows);
                p.dev.attention_rel_sink({ob.get(), 0}, {qb.get(), 0}, {k.second.get(), 0}, {v.second.get(), 0}, {rb.get(), 0}, {table.second.get(), 0}, {sink.second.get(), 0},
                                         heads, kv_heads, D, R, extent, window, scale, 0, ring ? dv.data() : nullptr, ring ? nullptr : dp.data(), calls.size(), ring_offset, ring_rows, {}, {},
                                         clip);
                p.dev.sync();
                require(!std::memcmp(download(p.dev, *oa, rows * HD).data(), download(p.dev, *ob, rows * HD).data(), rows * HD * sizeof(float)),
                        "device attention_rel_sink with its clip over " + at + " at length " + std::to_string(calls[0].length) + " has the bits of l2_clip and then the call");
            }
        }
    }
    if (ring) same_state(p, *cs, *ds, 0.0, "attention_rel_sink rings of " + at);
}

// On the device, attention_rel_sink gives a sequence's rows the same bits alone, beside another sequence on either side and sliced over two calls of one prompt, over a ring and pooled rows (docs/INKLING2.md, Row classes).
// A prompt's extent is its class key, given by runs and on pooled views: 1 and 7 rows stay below the tile's rows, 32 to 100 reach it, and a slice below them still belongs to a prompt that reaches it.
void device_rel_sink_rows(backend::Backend& dev) {
    const size_t D = 128, R = 16, heads = 4, kv_heads = 2, KD = kv_heads * D, HD = heads * D, extent = 1024, ring_window = 320, ring_rows = 1087, ring_offset = 9;
    const size_t hist = 300, other_hist = 40, other_nq = 5, most = 100;
    const float scale = float(1.0 / std::sqrt(double(D)));
    const std::vector<float> table = random(R * extent, -0.5f, 0.5f);
    // Every call's inputs by sequence and position, so each arm feeds a row the same values wherever it sits in a call.
    struct Rows {
        std::vector<float> q, k, v, rel, sink;
    };
    const auto rows_of = [&](size_t n) { return Rows{random(n * HD, -1, 1), random(n * KD, -1, 1), random(n * KD, -1, 1), random(n * heads * R, -1, 1), random(n * heads, -2, 2)}; };
    const Rows target_hist = rows_of(hist), other_hist_rows = rows_of(other_hist + other_nq), target = rows_of(most);
    const auto floats = [&](const std::vector<float>& v) {
        BufferPtr b = dev.alloc(std::max<size_t>(v.size(), 1) * sizeof(float), backend::Memory::device);
        if (!v.empty()) dev.write(*b, 0, v.data(), v.size() * sizeof(float));
        return b;
    };
    const BufferPtr tb = floats(table);
    std::string failed;
    for (bool ring : {true, false})
        for (size_t nq : {size_t(1), size_t(7), size_t(32), size_t(33), most}) {
            std::vector<float> got[4];
            for (int arm = 0; arm < 4; ++arm) {
                const std::vector<size_t> tails = {ring_offset + ring_rows * 2 * KD + 3};
                std::unique_ptr<backend::StateStorage> st = dev.state_alloc(1, 2, backend::StateShape{}, tails);
                std::unique_ptr<backend::KVStorage> kv = dev.kv_alloc(1, 0, 0, 2048, backend::KVType::f32, backend::KVType::f32, {0}, {{1, 2 * KD}});
                std::vector<int32_t> blocks[2];
                for (size_t b = 0; b < kv->max_blocks(); ++b) blocks[b % 2].push_back(int32_t(b));
                // One call over `parts`, each {sequence, length, first row of its inputs, rows}, the rows' outputs returned in order.
                const auto call = [&](const std::vector<std::array<size_t, 4>>& parts) {
                    Rows in;
                    std::vector<backend::StateView> sv;
                    std::vector<backend::KVView> kvv;
                    std::vector<backend::RowRun> runs;
                    for (const std::array<size_t, 4>& part : parts) {
                        const size_t seq = part[0], length = part[1], at = part[2], n = part[3];
                        const Rows& from = seq == 1 ? other_hist_rows : length < hist ? target_hist : target;
                        const auto take = [&](std::vector<float>& to, const std::vector<float>& v, size_t w) { to.insert(to.end(), v.begin() + long(at * w), v.begin() + long((at + n) * w)); };
                        take(in.q, from.q, HD), take(in.k, from.k, KD), take(in.v, from.v, KD), take(in.rel, from.rel, heads * R), take(in.sink, from.sink, heads);
                        const size_t key = seq == 0 && length >= hist ? nq : n;
                        sv.push_back({st.get(), seq, seq, length, n});
                        backend::KVView w{kv.get(), blocks[seq].data(), blocks[seq].size(), length, n};
                        w.extent = key;
                        kvv.push_back(w);
                        runs.push_back({in.sink.size() / heads, key});
                    }
                    const size_t rows = in.sink.size() / heads;
                    const BufferPtr q = floats(in.q), k = floats(in.k), v = floats(in.v), rel = floats(in.rel), sink = floats(in.sink), o = floats(std::vector<float>(rows * HD));
                    dev.attention_rel_sink({o.get(), 0}, {q.get(), 0}, {k.get(), 0}, {v.get(), 0}, {rel.get(), 0}, {tb.get(), 0}, {sink.get(), 0}, heads, kv_heads, D, R, extent, ring ? ring_window : 0,
                                           scale, 0, ring ? sv.data() : nullptr, ring ? nullptr : kvv.data(), parts.size(), ring_offset, ring_rows, {runs.data(), runs.size()});
                    dev.sync();
                    std::vector<float> out(rows * HD);
                    dev.read(*o, 0, out.data(), out.size() * sizeof(float));
                    return out;
                };
                call({{0, 0, 0, hist}});
                call({{1, 0, 0, other_hist}});
                std::vector<float>& g = got[arm];
                if (arm == 0) g = call({{0, hist, 0, nq}});
                else if (arm == 1) g = call({{1, other_hist, other_hist, other_nq}, {0, hist, 0, nq}}), g.erase(g.begin(), g.begin() + long(other_nq * HD));
                else if (arm == 2 || nq == 1) g = call({{0, hist, 0, nq}, {1, other_hist, other_hist, other_nq}}), g.resize(nq * HD);
                else {
                    const size_t a = nq - nq / 4;
                    g = call({{0, hist, 0, a}});
                    const std::vector<float> rest = call({{0, hist + a, a, nq - a}});
                    g.insert(g.end(), rest.begin(), rest.end());
                }
            }
            const std::string at = std::string(ring ? "a ring" : "pooled rows") + ", " + std::to_string(nq) + " rows";
            const char* arms[4] = {"", "beside another sequence after it", "beside another sequence before it", "sliced over two calls"};
            bool written = false;
            for (float x : got[0]) written = written || x != 0.0f;
            if (!written) failed += "\n  " + at + ": no output written";
            for (int arm = 1; arm < 4; ++arm)
                if (std::memcmp(got[0].data(), got[arm].data(), nq * HD * sizeof(float)) != 0) failed += "\n  " + at + ": alone and " + arms[arm] + " differ";
            ++checked;
        }
    require(failed.empty(), "device attention_rel_sink gives a row the same bits however its prompt is batched:" + failed);
}

// depthwise_conv into rows out_stride floats apart, set and accumulated, gives the bits of the contiguous conv placed, or added with one f32 add, into those rows, and keeps the floats between them.
void conv_stride_accumulate(backend::Backend& b, size_t C, const std::string& at) {
    const size_t K = 4, S = C + 5, ring_rows = K - 1 + 8, ring_offset = 3;
    const std::vector<SlotCall> seq[] = {{{0, 0, 0, 6}}, {{0, 0, 6, 1}}, {{0, 1, 7, 2}}, {{0, 0, 7, 1}, {1, 1, 9, 3}}};
    std::unique_ptr<backend::StateStorage> st[3];
    for (auto& s : st) s = b.state_alloc(1, 2, backend::StateShape{}, {ring_offset + ring_rows * C + 7});
    BufferPtr taps = upload(b, random(C * K, -1.0f, 1.0f));
    for (const std::vector<SlotCall>& calls : seq) {
        size_t rows = 0;
        std::vector<backend::StateView> v[3];
        for (const SlotCall& c : calls) {
            for (int i = 0; i < 3; ++i) v[i].push_back({st[i].get(), c.src, c.dst, c.length, c.nq});
            rows += c.nq;
        }
        const std::vector<float> base = random(rows * S, -3.0f, 3.0f);
        BufferPtr x = upload(b, random(rows * C, -2.0f, 2.0f)), apart = upload(b, std::vector<float>(rows * C)), set = upload(b, base), add = upload(b, base);
        backend::DepthwiseConv conv = ring_conv({taps.get(), 0}, {}, C, K, 0, 0, std::sqrt(5.0f), ring_offset, ring_rows);
        b.depthwise_conv({apart.get(), 0}, {x.get(), 0}, conv, 0, v[0].data(), calls.size());
        conv.out_stride = S;
        b.depthwise_conv({set.get(), 0}, {x.get(), 0}, conv, 0, v[1].data(), calls.size());
        conv.accumulate = true;
        b.depthwise_conv({add.get(), 0}, {x.get(), 0}, conv, 0, v[2].data(), calls.size());
        b.sync();
        const std::vector<float> want = download(b, *apart, rows * C), got_set = download(b, *set, rows * S), got_add = download(b, *add, rows * S);
        bool same = true;
        for (size_t g = 0; g < rows; ++g)
            for (size_t c = 0; c < S; ++c) {
                const size_t i = g * S + c;
                const float s = c < C ? want[g * C + c] : base[i], a = c < C ? base[i] + want[g * C + c] : base[i];
                same = same && std::memcmp(&got_set[i], &s, 4) == 0 && std::memcmp(&got_add[i], &a, 4) == 0;
            }
        require(same, "depthwise_conv " + at + " into rows " + std::to_string(S) + " apart, set and accumulated, at length " + std::to_string(calls[0].length) + ": bits of the contiguous conv");
    }
    const size_t n = st[0]->layer(0).size() / sizeof(float);
    const std::vector<float> s0 = download(b, st[0]->layer(0), n), s1 = download(b, st[1]->layer(0), n), s2 = download(b, st[2]->layer(0), n);
    require(s0 == s1 && s0 == s2, "depthwise_conv " + at + ": strided and accumulating calls leave the contiguous call's rings");
}

void conv_strides() {
    CpuBackend cpu;
    for (size_t C : {size_t(10), size_t(130)}) conv_stride_accumulate(cpu, C, "of " + std::to_string(C) + " channels");
    std::unique_ptr<backend::StateStorage> store = cpu.state_alloc(1, 2, backend::StateShape{}, {40});
    BufferPtr b = upload(cpu, std::vector<float>(1024));
    const backend::StateView v{store.get(), 0, 0, 0, 2};
    backend::DepthwiseConv conv = ring_conv({b.get(), 900}, {}, 8, 4, 0, 0, 3.0f, 0, 3);
    conv.out_stride = 7;
    refused([&] { cpu.depthwise_conv({b.get(), 0}, {b.get(), 600}, conv, 0, &v, 1); }, "output rows closer than the channels", "backend: depthwise convolution output rows overlap");
    conv.out_stride = 12;
    refused([&] { cpu.depthwise_conv({b.get(), 600}, {b.get(), 600}, conv, 0, &v, 1); }, "a strided output on its input",
            "backend: depthwise convolution output must be disjoint or exactly its input or residual");
    conv.out_stride = 0;
    conv.accumulate = true;
    refused([&] { cpu.depthwise_conv({b.get(), 600}, {b.get(), 600}, conv, 0, &v, 1); }, "an accumulating output on its input",
            "backend: depthwise convolution output must be disjoint or exactly its input or residual");
}

// The core attention with terms on a device against the softmax in double: three query heads a KV head (the grouped build) and two (the one-head build).
// Rows of 480 and 481 keys take different part counts, and each row's output is exactly 1 / (keys + 1) (docs/TESTS.md, attention_terms_mixed_parts).
void attention_terms_mixed_parts(backend::Backend& dev, backend::KVType type) {
    const size_t D = 128, heads = 12, kv_heads = 4, keys = 481;
    std::unique_ptr<backend::KVStorage> kv = dev.kv_alloc(1, kv_heads, D, 512, type, type);
    std::vector<int32_t> blocks(backend::blocks_for(keys, dev.kv_layout().block_tokens));
    std::iota(blocks.begin(), blocks.end(), 0);
    const backend::KVView all{kv.get(), blocks.data(), blocks.size(), 0, keys};
    std::vector<float> values(keys * kv_heads * D);
    BufferPtr k = upload(dev, values);
    std::fill_n(values.begin() + 479 * kv_heads * D, kv_heads * D, 1.0f);
    BufferPtr v = upload(dev, values);
    dev.kv_write(0, &all, 1, {k.get(), 0}, {v.get(), 0});
    const backend::KVView query{kv.get(), blocks.data(), blocks.size(), keys - 2, 2};
    BufferPtr q = upload(dev, std::vector<float>(2 * heads * D)), o = upload(dev, std::vector<float>(2 * heads * D)), sink = upload(dev, std::vector<float>(2 * heads));
    backend::AttentionTerms terms;
    terms.sink = {sink.get(), 0};
    dev.attention({q.get(), 0}, 0, &query, 1, {o.get(), 0}, int(heads), int(kv_heads), int(D), &terms);
    const std::vector<float> got = download(dev, *o, 2 * heads * D);
    for (size_t row = 0; row < 2; ++row)
        for (size_t j = 0; j < heads * D; ++j)
            require(std::fabs(double(got[row * heads * D + j]) - 1.0 / double(keys + row)) <= 1e-6,
                    std::string("attention with terms over rows of 480 and 481 keys on ") + (type == backend::KVType::f16 ? "f16" : "f32") + " sides gives row " +
                        std::to_string(row) + " 1 / " + std::to_string(keys + row));
}

void device_attention_terms(backend::Backend& dev) {
    attention_terms_mixed_parts(dev, backend::KVType::f32);
    attention_terms_mixed_parts(dev, backend::KVType::f16);
    int mode = 0;
    for (size_t heads : {size_t(6), size_t(4)})
        for (backend::KVType type : {backend::KVType::f32, backend::KVType::f16})
            for (size_t W : {size_t(0), size_t(5), size_t(70)}) attention_terms_case(type, W, 1, mode++ % 3, &dev, heads, 128, 1e-5);
}

void device_state_ops(backend::Backend& dev) {
    device_window_paths(dev);
    device_rel_sink_rows(dev);
    Pair p(dev);
    device_rel_sink(p, true, 64, 48, 79, kSlotCalls);
    device_rel_sink(p, true, 1024, 1024, 1087, {{{0, 0, 0, 600}}, {{0, 0, 600, 700}}, {{0, 0, 1300, 1}}, {{0, 1, 1301, 3}}});
    device_rel_sink(p, false, 0, 1024, 0, {{{0, 0, 0, 70}}, {{0, 0, 70, 1}}, {{0, 0, 71, 5}, {1, 1, 0, 3}}, {{1, 1, 3, 130}, {2, 2, 0, 1}}, {{0, 0, 76, 1}}});
    // A pooled decode row past 16384 keys, more 64-key parts than a split once held, so it attends to every key.
    std::vector<std::vector<SlotCall>> past;
    for (size_t at = 0; at < 16400; at += 2050) past.push_back({{0, 0, at, std::min<size_t>(2050, 16400 - at)}});
    past.push_back({{0, 0, 16400, 1}});
    device_rel_sink(p, false, 0, 1024, 0, past, 16512, 1, 1);
    for (size_t C : {size_t(1024), size_t(2304), size_t(136)})
        for (bool dynamic : {false, true})
            for (bool in_place : {false, true}) device_ring_conv(p, C, dynamic, in_place);
    for (bool chained : {false, true}) device_ring_conv_list(dev, chained);
    for (const auto& [D, heads] : {std::pair<size_t, size_t>{128, 8}, {64, 3}, {256, 2}, {32, 5}}) device_conv_post(dev, D, heads);
    for (size_t C : {size_t(2304), size_t(136)}) conv_stride_accumulate(dev, C, "on the device, " + std::to_string(C) + " channels");
}

// Backend::rows_product against the separate calls it stands for on one backend, bit for bit: the head's mix, norm and softcapped logits, a gate and norm into a group with a GLU store, and a grouped norm with its product.
void row_products(backend::Backend& b) {
    const size_t E = 256, S = 6 * E, rows = 3, nout = 96, ff = 64, mixes = 5;
    const auto put = [&](const std::vector<float>& v) {
        BufferPtr d = b.alloc(v.size() * sizeof(float), backend::Memory::device);
        b.write(*d, 0, v.data(), v.size() * sizeof(float));
        return d;
    };
    const auto bits = [&](const BufferPtr& x, const BufferPtr& y, size_t n, const std::string& what) {
        b.sync();
        std::vector<float> a(n), c(n);
        b.read(*x, 0, a.data(), n * sizeof(float));
        b.read(*y, 0, c.data(), n * sizeof(float));
        require(std::memcmp(a.data(), c.data(), n * sizeof(float)) == 0, "row product " + what + " has the bits of its separate calls");
    };
    const auto f32 = [](const BufferPtr& w) { return backend::WeightView{quant::GGML_TYPE_F32, {w.get(), 0}}; };
    BufferPtr src = put(random(rows * S, -3.0f, 3.0f)), queries = put(random(E, -2.0f, 2.0f)), norm = put(random(E, 0.5f, 1.5f));
    BufferPtr head = put(random(nout * E, -0.2f, 0.2f)), wg = put(random(ff * E, -0.2f, 0.2f)), wu = put(random(ff * E, -0.2f, 0.2f)), wq = put(random(nout * E, -0.2f, 0.2f));
    BufferPtr f = put(random(rows * E, -3.0f, 3.0f)), g = put(random(rows * E, -3.0f, 3.0f)), m = put(random(rows * E, -3.0f, 3.0f));
    const auto zeros = [&](size_t n) { return put(std::vector<float>(n, 0.0f)); };
    {
        BufferPtr o1 = zeros(rows * E), o2 = zeros(rows * E), y1 = zeros(rows * nout), y2 = zeros(rows * nout);
        b.query_mix({{{o1.get(), 0}, {src.get(), 0}, {queries.get(), 0}, mixes, 1, {{{norm.get(), 0}}}, 1e-5f}}, rows, E, S, 1e-6f);
        b.matmul_logits(f32(head), {o1.get(), 0}, {y1.get(), 0}, E, nout, rows, {}, backend::Dtype::f16, 30.0f);
        const backend::RowMix mix{{{o2.get(), 0}, {src.get(), 0}, {queries.get(), 0}, mixes, 1}, E, S, 1e-6f};
        const backend::RowNorm rn{{o2.get(), 0}, {o2.get(), 0}, {norm.get(), 0}, E, E, 1e-5f};
        backend::RowSource s;
        s.row[0] = {o2.get(), 0}, s.mix = &mix, s.norm[0] = &rn;
        backend::RowSink k;
        k.store = backend::RowStore::logits, k.cap = 30.0f;
        b.rows_product(s, {{{f32(head), {y2.get(), 0}, nout}, k}}, E, rows);
        bits(o1, o2, rows * E, "mix and norm");
        bits(y1, y2, rows * nout, "logits");
    }
    {
        BufferPtr h1 = put(random(rows * E, -3.0f, 3.0f)), h2 = zeros(rows * E);
        b.copy(*h2, 0, *h1, 0, rows * E * sizeof(float));
        BufferPtr q1 = zeros(rows * nout), q2 = zeros(rows * nout), g1 = zeros(rows * ff), g2 = zeros(rows * ff), u1 = zeros(rows * ff), u2 = zeros(rows * ff), a1 = zeros(rows * ff),
                  a2 = zeros(rows * ff);
        const backend::Glu glu{4.0f, backend::GluForm::softcap, 25.0f};
        b.tanh_gate_mix({{{h1.get(), 0}, {h1.get(), 0}, {m.get(), 0}, {f.get(), 0}, {g.get(), 0}, {norm.get(), 0}}}, rows, E, E, E, E, {}, 1e-5f);
        b.matmul_group({{f32(wq), {q1.get(), 0}, nout}, {f32(wg), {g1.get(), 0}, ff}, {f32(wu), {u1.get(), 0}, ff}}, {h1.get(), 0}, E, rows);
        b.silu_mul({a1.get(), 0}, {g1.get(), 0}, {u1.get(), 0}, rows * ff, glu);
        const backend::RowGate gate{{h2.get(), 0}, {h2.get(), 0}, {m.get(), 0}, {f.get(), 0}, {g.get(), 0}, E, E, E, E};
        const backend::RowNorm rn{{h2.get(), 0}, {h2.get(), 0}, {norm.get(), 0}, E, E, 1e-5f};
        backend::RowSource s;
        s.row[0] = {h2.get(), 0}, s.gate[0] = &gate, s.norm[0] = &rn;
        backend::RowSink pair;
        pair.store = backend::RowStore::glu, pair.act = {a2.get(), 0}, pair.glu = glu;
        b.rows_product(s, {{{f32(wq), {q2.get(), 0}, nout}, {}}, {{f32(wg), {g2.get(), 0}, ff}, pair}, {{f32(wu), {u2.get(), 0}, ff}, {}}}, E, rows);
        bits(h1, h2, rows * E, "gate and norm");
        bits(q1, q2, rows * nout, "group item");
        bits(a1, a2, rows * ff, "GLU store");
    }
    {
        BufferPtr n1 = zeros(rows * E), n2 = zeros(rows * E), y1 = zeros(rows * nout), y2 = zeros(rows * nout);
        BufferPtr gw = put(random(E, 0.5f, 1.5f));
        b.group_rms_norm({n1.get(), 0}, {src.get(), 0}, {gw.get(), 0}, rows, 2, E / 2, 1e-5f, {}, {{src.get(), E}, 1, S, S}, {f32(wq), {y1.get(), 0}, nout});
        const backend::RowNorm rn{{n2.get(), 0}, {src.get(), 0}, {gw.get(), 0}, E, S, 1e-5f, 2, {{src.get(), E}, 1, S, S}};
        backend::RowSource s;
        s.row[0] = {n2.get(), 0}, s.norm[0] = &rn;
        b.rows_product(s, {{{f32(wq), {y2.get(), 0}, nout}, {}}}, E, rows);
        bits(y1, y2, rows * nout, "grouped norm product");
    }
    refused([&] { b.rows_product({}, {}, E, rows); }, "a row product of no items", "backend: a row product of no items or more than it takes");
}

} // namespace ink2
