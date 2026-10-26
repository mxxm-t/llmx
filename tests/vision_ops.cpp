// The CPU backend's ops of the vision encoder (docs/VISION.md, section 3) against references written from the math in double precision, each within a bound stated beside it.
// Every result is the same bits at every thread count and for a row alone or beside others; a non-causal attention view and a rope over position streams leave the causal and one-stream paths as they were.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/cpu/cpu_backend.hpp"
#include "core/fp16.hpp"

namespace {

using backend::BufferPtr;
using backend::CpuBackend;
using backend::CSlice;
using backend::Slice;

int checks = 0;
const int kThreads[] = {1, 2, 3, 5, 8, 16};

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

std::vector<float> uniform(std::mt19937& g, size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (float& x : v) x = d(g);
    return v;
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

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// The largest error of `got` against `ref` in units of `tol(ref)`; at most 1 passes.
template <typename Tol>
double worst(const std::vector<float>& got, const std::vector<double>& ref, Tol tol) {
    double w = 0.0;
    for (size_t i = 0; i < got.size(); ++i) w = std::max(w, std::fabs(double(got[i]) - ref[i]) / tol(ref[i]));
    return w;
}

// LayerNorm with weight and bias: mean and biased variance over each row, within 4 float units of the row's largest output magnitude.
void layer_norm() {
    std::mt19937 g(11);
    for (size_t n : {1u, 7u, 64u, 768u, 1152u, 1153u}) {
        const size_t rows = 9, stride = n + 3;
        std::vector<float> x = uniform(g, rows * stride, -3.0f, 3.0f);
        for (size_t r = 0; r < rows; ++r)
            for (size_t i = 0; i < n; ++i) x[r * stride + i] += float(r) * 40.0f - 100.0f;  // means far from 0
        const std::vector<float> w = uniform(g, n, -2.0f, 2.0f), b = uniform(g, n, -1.0f, 1.0f);
        std::vector<double> ref(rows * n);
        double scale = 0.0;
        for (size_t r = 0; r < rows; ++r) {
            double mean = 0.0, var = 0.0;
            for (size_t i = 0; i < n; ++i) mean += x[r * stride + i];
            mean /= double(n);
            for (size_t i = 0; i < n; ++i) var += (x[r * stride + i] - mean) * (x[r * stride + i] - mean);
            var /= double(n);
            for (size_t i = 0; i < n; ++i) {
                ref[r * n + i] = (x[r * stride + i] - mean) / std::sqrt(var + 1e-6) * w[i] + b[i];
                scale = std::max(scale, std::fabs(ref[r * n + i]));
            }
        }
        std::vector<float> first;
        for (int t : kThreads) {
            CpuBackend cpu;
            cpu.set_threads(t);
            BufferPtr xb = upload(cpu, x), wb = upload(cpu, w), bb = upload(cpu, b), out = upload(cpu, std::vector<float>(rows * stride, 0.0f));
            cpu.layer_norm_rows(Slice{out.get(), 0}, CSlice(xb.get(), 0), CSlice(wb.get(), 0), CSlice(bb.get(), 0), rows, n, stride, 1e-6f);
            const std::vector<float> strided = download(cpu, *out, rows * stride);
            std::vector<float> got(rows * n);
            for (size_t r = 0; r < rows; ++r) std::memcpy(got.data() + r * n, strided.data() + r * stride, n * sizeof(float));
            if (first.empty()) {
                first = got;
                const double e = worst(got, ref, [&](double) { return 4.0 * std::ldexp(std::max(scale, 1.0), -23); });
                require(e <= 1.0, "layer_norm at width " + std::to_string(n) + ": " + std::to_string(e) + " of its bound");
                // A row alone gives the bits it gives beside the others.
                BufferPtr one = upload(cpu, std::vector<float>(n, 0.0f));
                cpu.layer_norm_rows(Slice{one.get(), 0}, CSlice(xb.get(), 4 * stride), CSlice(wb.get(), 0), CSlice(bb.get(), 0), 1, n, stride, 1e-6f);
                const std::vector<float> alone = download(cpu, *one, n);
                require(std::memcmp(alone.data(), got.data() + 4 * n, n * sizeof(float)) == 0, "layer_norm: a row alone differs from it beside others");
            } else {
                require(same_bits(first, got), "layer_norm at width " + std::to_string(n) + " differs at " + std::to_string(t) + " threads");
            }
        }
    }
}

double gelu_ref(double x, backend::GeluForm form) {
    if (form == backend::GeluForm::erf) return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
    const double k = std::sqrt(2.0 / 3.14159265358979323846);
    return 0.5 * x * (1.0 + std::tanh(k * (x + 0.044715 * x * x * x)));
}

// GELU in both forms within 8 float units of each value's magnitude, over a sweep, zeros of both signs, the tiny and the large.
void gelu() {
    std::vector<float> x;
    for (int i = -4000; i <= 4000; ++i) x.push_back(float(i) / 200.0f);
    for (float v : {0.0f, -0.0f, 1e-30f, -1e-30f, 1e-8f, -1e-8f, 1e4f, -1e4f, 3e38f, -3e38f}) x.push_back(v);
    for (backend::GeluForm form : {backend::GeluForm::tanh, backend::GeluForm::erf}) {
        std::vector<double> ref(x.size());
        for (size_t i = 0; i < x.size(); ++i) ref[i] = gelu_ref(x[i], form);
        std::vector<float> first;
        for (int t : kThreads) {
            CpuBackend cpu;
            cpu.set_threads(t);
            BufferPtr xb = upload(cpu, x), out = upload(cpu, std::vector<float>(x.size(), 0.0f));
            cpu.gelu(Slice{out.get(), 0}, CSlice(xb.get(), 0), x.size(), form);
            const std::vector<float> got = download(cpu, *out, x.size());
            if (first.empty()) {
                first = got;
                for (size_t i = 0; i < x.size(); ++i) {
                    if (std::isinf(ref[i]) || std::fabs(ref[i]) > 3.4e38) {
                        require(std::isinf(got[i]) || std::fabs(double(got[i]) - ref[i]) <= 8.0 * std::ldexp(std::fabs(ref[i]), -23),
                                "gelu: a large value went astray");
                        continue;
                    }
                    const double tol = 8.0 * std::ldexp(std::max(std::fabs(ref[i]), 1e-38), -23) + 1e-45;
                    require(std::fabs(double(got[i]) - ref[i]) <= tol,
                            std::string("gelu ") + (form == backend::GeluForm::tanh ? "tanh" : "erf") + " at " + std::to_string(x[i]) + ": " +
                                std::to_string(got[i]) + " against " + std::to_string(ref[i]));
                }
            } else {
                require(same_bits(first, got), "gelu differs at " + std::to_string(t) + " threads");
            }
        }
    }
}

// The row-broadcast bias add, exactly a float add an element.
void add_bias() {
    std::mt19937 g(5);
    const size_t rows = 13, width = 1155;
    const std::vector<float> x = uniform(g, rows * width, -5.0f, 5.0f), b = uniform(g, width, -1.0f, 1.0f);
    std::vector<float> want(x);
    for (size_t r = 0; r < rows; ++r)
        for (size_t i = 0; i < width; ++i) want[r * width + i] = x[r * width + i] + b[i];
    for (int t : kThreads) {
        CpuBackend cpu;
        cpu.set_threads(t);
        BufferPtr xb = upload(cpu, x), bb = upload(cpu, b);
        cpu.add_bias(Slice{xb.get(), 0}, CSlice(bb.get(), 0), rows, width);
        require(same_bits(download(cpu, *xb, rows * width), want), "add_bias is not the float add at " + std::to_string(t) + " threads");
    }
}

// The weighted gather of table rows added to dst, the learned positions' bilinear taps: each tap a fused multiply-add in tap order, then one add to dst.
void mix_rows() {
    std::mt19937 g(9);
    const size_t width = 1152, table_rows = 2304, rows = 37, taps = 4;
    const std::vector<float> table = uniform(g, table_rows * width, -1.0f, 1.0f), base = uniform(g, rows * width, -2.0f, 2.0f);
    std::vector<uint32_t> idx(rows * taps);
    std::vector<float> wt(rows * taps);
    std::uniform_int_distribution<uint32_t> pick(0, uint32_t(table_rows - 1));
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = pick(g);
    const std::vector<float> raw = uniform(g, rows * taps, 0.0f, 1.0f);
    for (size_t i = 0; i < wt.size(); ++i) wt[i] = raw[i];
    std::vector<float> want(base);
    for (size_t r = 0; r < rows; ++r)
        for (size_t i = 0; i < width; ++i) {
            float acc = 0.0f;
            for (size_t k = 0; k < taps; ++k) acc = std::fma(wt[r * taps + k], table[size_t(idx[r * taps + k]) * width + i], acc);
            want[r * width + i] = base[r * width + i] + acc;
        }
    for (int t : kThreads) {
        CpuBackend cpu;
        cpu.set_threads(t);
        BufferPtr tb = upload(cpu, table), db = upload(cpu, base);
        cpu.mix_rows(Slice{db.get(), 0}, CSlice(tb.get(), 0), width, table_rows, idx.data(), wt.data(), rows, taps);
        require(same_bits(download(cpu, *db, rows * width), want), "mix_rows is not its taps in order at " + std::to_string(t) + " threads");
    }
    CpuBackend cpu;
    BufferPtr tb = upload(cpu, table), db = upload(cpu, base);
    std::vector<uint32_t> bad(idx);
    bad[5] = uint32_t(table_rows);
    bool refused = false;
    try {
        cpu.mix_rows(Slice{db.get(), 0}, CSlice(tb.get(), 0), width, table_rows, bad.data(), wt.data(), rows, taps);
    } catch (const std::runtime_error&) {
        refused = true;
    }
    require(refused, "mix_rows read a row past the table");
}

// The rope over position streams: Qwen's three streams at equal positions give the one-stream bits, and the vision encoder's two streams without the norm meet a double rotation within 4 float units.
void rope_streams() {
    std::mt19937 g(3);
    // Text: 8 heads of 256, a rope of 64 as qwen35's, the sections interleaved [11, 11, 10].
    {
        const size_t rows = 21, heads = 8, dim = 256, rope = 64, half = 32, npos = 600;
        std::vector<float> cosv(npos * half), sinv(npos * half);
        for (size_t p = 0; p < npos; ++p)
            for (size_t j = 0; j < half; ++j) {
                const float f = float(1.0 / std::pow(1e7, double(2 * j) / double(rope)));
                cosv[p * half + j] = std::cos(float(p) * f);
                sinv[p * half + j] = std::sin(float(p) * f);
            }
        std::vector<uint8_t> of(half, 0);
        for (size_t j = 1; j < 33 && j < half; j += 3) of[j] = 1;
        for (size_t j = 2; j < 30; j += 3) of[j] = 2;
        std::vector<uint32_t> pos(rows), pos3(rows * 3);
        for (size_t r = 0; r < rows; ++r) {
            pos[r] = uint32_t(r * 23 + 5);
            pos3[r * 3] = pos3[r * 3 + 1] = pos3[r * 3 + 2] = pos[r];
        }
        const std::vector<float> x = uniform(g, rows * heads * dim, -2.0f, 2.0f), w = uniform(g, dim, 0.5f, 1.5f);
        CpuBackend cpu;
        cpu.set_threads(4);
        BufferPtr xb = upload(cpu, x), wb = upload(cpu, w), cb = upload(cpu, cosv), sb = upload(cpu, sinv);
        BufferPtr one = upload(cpu, std::vector<float>(x.size(), 0.0f)), three = upload(cpu, std::vector<float>(x.size(), 0.0f));
        cpu.norm_rope_partial(Slice{one.get(), 0}, CSlice(xb.get(), 0), rows, heads * dim, dim, heads, dim, rope, CSlice(wb.get(), 0), 1e-6f,
                              CSlice(cb.get(), 0), CSlice(sb.get(), 0), pos.data());
        cpu.norm_rope_partial(Slice{three.get(), 0}, CSlice(xb.get(), 0), rows, heads * dim, dim, heads, dim, rope, CSlice(wb.get(), 0), 1e-6f,
                              CSlice(cb.get(), 0), CSlice(sb.get(), 0), pos3.data(), backend::RopeStreams{3, of.data()});
        require(same_bits(download(cpu, *one, x.size()), download(cpu, *three, x.size())), "rope: three equal streams differ from one stream");
    }
    // One stream with a norm rotates any width, as the synthetic bench's one head of 2048 rotated whole does; only the gathering path is bounded, and refuses a wider rope by name.
    {
        const size_t dim = 2048, half = dim / 2;
        std::vector<float> cosv(2 * half), sinv(2 * half);
        for (size_t j = 0; j < half; ++j) {
            const float f = float(1.0 / std::pow(1e4, double(2 * j) / double(dim)));
            cosv[j] = 1.0f;
            sinv[j] = 0.0f;
            cosv[half + j] = std::cos(f);
            sinv[half + j] = std::sin(f);
        }
        const std::vector<float> x = uniform(g, dim, -2.0f, 2.0f), w = uniform(g, dim, 0.5f, 1.5f);
        double ss = 0.0;
        for (float v : x) ss += double(v) * double(v);
        const double inv = 1.0 / std::sqrt(ss / double(dim) + 1e-6);
        std::vector<double> ref(dim);
        for (size_t j = 0; j < half; ++j) {
            const double a = x[j] * inv * w[j], b = x[j + half] * inv * w[j + half];
            ref[j] = a * double(cosv[half + j]) - b * double(sinv[half + j]);
            ref[j + half] = b * double(cosv[half + j]) + a * double(sinv[half + j]);
        }
        CpuBackend cpu;
        BufferPtr xb = upload(cpu, x), wb = upload(cpu, w), cb = upload(cpu, cosv), sb = upload(cpu, sinv);
        const uint32_t one = 1;
        cpu.norm_rope_partial(Slice{xb.get(), 0}, CSlice(xb.get(), 0), 1, dim, dim, 1, dim, dim, CSlice(wb.get(), 0), 1e-6f,
                              CSlice(cb.get(), 0), CSlice(sb.get(), 0), &one);
        const double e = worst(download(cpu, *xb, dim), ref, [](double v) { return 16.0 * std::ldexp(std::max(std::fabs(v), 1.0), -23); });
        require(e <= 1.0, "rope of 1024 pairs with a norm: " + std::to_string(e) + " of its bound");
        bool refused = false;
        try {
            cpu.norm_rope_partial(Slice{xb.get(), 0}, CSlice(xb.get(), 0), 1, dim, dim, 1, dim, dim, CSlice{}, 0.0f, CSlice(cb.get(), 0), CSlice(sb.get(), 0), &one);
        } catch (const std::runtime_error& err) {
            refused = std::string(err.what()).find("at most") != std::string::npos;
        }
        require(refused, "rope: 1024 pairs without a norm were not refused by the gathering path's width");
    }
    // Vision: 16 heads of 72, the whole head rotated, 18 frequencies by patch row then 18 by patch column, no norm.
    const size_t rows = 50, heads = 16, dim = 72, half = 36, side = 40;
    std::vector<float> freq(18);
    for (size_t j = 0; j < 18; ++j) freq[j] = float(1.0 / std::pow(10000.0, double(2 * j) / 36.0));
    std::vector<float> cosv(side * half), sinv(side * half);
    for (size_t p = 0; p < side; ++p)
        for (size_t j = 0; j < half; ++j) {
            cosv[p * half + j] = std::cos(float(p) * freq[j % 18]);
            sinv[p * half + j] = std::sin(float(p) * freq[j % 18]);
        }
    std::vector<uint8_t> of(half);
    for (size_t j = 0; j < half; ++j) of[j] = j < 18 ? 0 : 1;
    std::vector<uint32_t> pos(rows * 2);
    for (size_t r = 0; r < rows; ++r) {
        pos[r * 2] = uint32_t((r * 7) % side);
        pos[r * 2 + 1] = uint32_t((r * 13 + 3) % side);
    }
    const std::vector<float> x = uniform(g, rows * heads * dim, -3.0f, 3.0f);
    std::vector<double> ref(x.size());
    for (size_t r = 0; r < rows; ++r)
        for (size_t h = 0; h < heads; ++h)
            for (size_t j = 0; j < half; ++j) {
                const size_t base = (r * heads + h) * dim;
                const size_t p = pos[r * 2 + of[j]];
                const double c = cosv[p * half + j], s = sinv[p * half + j], a = x[base + j], b = x[base + j + half];
                ref[base + j] = a * c - b * s;
                ref[base + j + half] = b * c + a * s;
            }
    std::vector<float> first;
    for (int t : kThreads) {
        CpuBackend cpu;
        cpu.set_threads(t);
        BufferPtr xb = upload(cpu, x), cb = upload(cpu, cosv), sb = upload(cpu, sinv);
        cpu.norm_rope_partial(Slice{xb.get(), 0}, CSlice(xb.get(), 0), rows, heads * dim, dim, heads, dim, dim, CSlice{}, 0.0f,
                              CSlice(cb.get(), 0), CSlice(sb.get(), 0), pos.data(), backend::RopeStreams{2, of.data()});
        const std::vector<float> got = download(cpu, *xb, x.size());
        if (first.empty()) {
            first = got;
            const double e = worst(got, ref, [](double v) { return 4.0 * std::ldexp(std::max(std::fabs(v), 1.0), -23); });
            require(e <= 1.0, "rope over two streams: " + std::to_string(e) + " of its bound");
        } else {
            require(same_bits(first, got), "rope over two streams differs at " + std::to_string(t) + " threads");
        }
    }
    // A stream map past its streams, and a position past the table, are refused.
    CpuBackend cpu;
    BufferPtr xb = upload(cpu, x), cb = upload(cpu, cosv), sb = upload(cpu, sinv);
    std::vector<uint8_t> bad_of(of);
    bad_of[3] = 2;
    std::vector<uint32_t> bad_pos(pos);
    bad_pos[7] = uint32_t(side);
    for (int k = 0; k < 3; ++k) {
        bool refused = false;
        try {
            cpu.norm_rope_partial(Slice{xb.get(), 0}, CSlice(xb.get(), 0), rows, heads * dim, dim, heads, dim, dim, CSlice{}, 0.0f, CSlice(cb.get(), 0),
                                  CSlice(sb.get(), 0), k == 1 ? bad_pos.data() : pos.data(),
                                  backend::RopeStreams{k == 2 ? size_t(0) : size_t(2), k == 0 ? bad_of.data() : of.data()});
        } catch (const std::runtime_error&) {
            refused = true;
        }
        require(refused, std::string("rope: ") + (k == 0 ? "a stream map past its streams" : k == 1 ? "a position past the table" : "no streams") + " was not refused");
    }
}

float half_round(float v) { return f16_to_f32(f32_to_f16(v)); }

// Attention through a non-causal view: every row of an image attends every row of it, against a double softmax within 4e-6 of the output scale, at head widths 64 and 72 and both cache types.
// Two images side by side give each alone's bits, and the causal view keeps its prefix.
void attention_full() {
    std::mt19937 g(17);
    for (size_t dim : {64u, 72u}) {
        for (backend::KVType type : {backend::KVType::f32, backend::KVType::f16}) {
            const int heads = 12;
            const size_t n0 = 150, n1 = 61, width = size_t(heads) * dim;
            const std::vector<float> q = uniform(g, (n0 + n1) * width, -1.0f, 1.0f), k = uniform(g, (n0 + n1) * width, -1.0f, 1.0f),
                                     v = uniform(g, (n0 + n1) * width, -1.0f, 1.0f);
            const auto stored = [&](float x) { return type == backend::KVType::f16 ? half_round(x) : x; };
            std::vector<double> ref((n0 + n1) * width);
            for (int seg = 0; seg < 2; ++seg) {
                const size_t first = seg ? n0 : 0, n = seg ? n1 : n0;
                for (size_t r = 0; r < n; ++r)
                    for (int h = 0; h < heads; ++h) {
                        std::vector<double> s(n);
                        double m = -1e300, sum = 0.0;
                        for (size_t t = 0; t < n; ++t) {
                            double d = 0.0;
                            for (size_t i = 0; i < dim; ++i) d += double(q[(first + r) * width + h * dim + i]) * stored(k[(first + t) * width + h * dim + i]);
                            s[t] = d / std::sqrt(double(dim));
                            m = std::max(m, s[t]);
                        }
                        for (size_t t = 0; t < n; ++t) sum += (s[t] = std::exp(s[t] - m));
                        for (size_t i = 0; i < dim; ++i) {
                            double o = 0.0;
                            for (size_t t = 0; t < n; ++t) o += s[t] / sum * stored(v[(first + t) * width + h * dim + i]);
                            ref[(first + r) * width + h * dim + i] = o;
                        }
                    }
            }
            std::vector<float> first;
            for (int th : kThreads) {
                CpuBackend cpu;
                cpu.set_threads(th);
                auto kv = cpu.kv_alloc(1, size_t(heads), dim, n0 + n1 + 256, type, type);
                const size_t bt = cpu.kv_layout().block_tokens;
                std::vector<int32_t> b0, b1;
                for (size_t i = 0; i < backend::blocks_for(n0, bt); ++i) b0.push_back(int32_t(i));
                for (size_t i = 0; i < backend::blocks_for(n1, bt); ++i) b1.push_back(int32_t(b0.size() + i));
                backend::KVView views[2] = {{kv.get(), b0.data(), b0.size(), 0, n0, n0, false}, {kv.get(), b1.data(), b1.size(), 0, n1, n1, false}};
                BufferPtr qb = upload(cpu, q), kb = upload(cpu, k), vb = upload(cpu, v), out = upload(cpu, std::vector<float>(q.size(), 0.0f));
                cpu.kv_write(0, views, 2, CSlice(kb.get(), 0), CSlice(vb.get(), 0));
                cpu.attention(CSlice(qb.get(), 0), 0, views, 2, Slice{out.get(), 0}, heads, heads, int(dim));
                const std::vector<float> got = download(cpu, *out, q.size());
                const std::string at = "attention_full at head width " + std::to_string(dim) + (type == backend::KVType::f16 ? ", f16 cache" : ", f32 cache");
                if (!first.empty()) {
                    require(same_bits(first, got), at + " differs at " + std::to_string(th) + " threads");
                    continue;
                }
                first = got;
                const double e = worst(got, ref, [](double r) { return 4e-6 * (1.0 + std::fabs(r)); });
                require(e <= 1.0, at + ": " + std::to_string(e) + " of its bound");
                // The second image alone gives the bits it gives beside the first.
                BufferPtr out1 = upload(cpu, std::vector<float>(n1 * width, 0.0f));
                cpu.attention(CSlice(qb.get(), n0 * width), 0, &views[1], 1, Slice{out1.get(), 0}, heads, heads, int(dim));
                const std::vector<float> alone = download(cpu, *out1, n1 * width);
                require(std::memcmp(alone.data(), got.data() + n0 * width, n1 * width * sizeof(float)) == 0, at + ": an image alone differs from it beside another");
                // A run of the first image's rows, 40 to 99, as a view of all its keys whose rows are the run's, as an encoder's piece takes them: the whole call's bits.
                const size_t from = 40, m = 60;
                const backend::KVView run{kv.get(), b0.data(), b0.size(), n0 - m, m, m, false};
                BufferPtr out3 = upload(cpu, std::vector<float>(m * width, 0.0f));
                cpu.attention(CSlice(qb.get(), from * width), 0, &run, 1, Slice{out3.get(), 0}, heads, heads, int(dim));
                const std::vector<float> rows = download(cpu, *out3, m * width);
                require(std::memcmp(rows.data(), got.data() + from * width, m * width * sizeof(float)) == 0, at + ": a run of rows differs from them in the whole call");
                // The same view causal: its first row attends only itself, so it is its value row.
                views[0].causal = true;
                BufferPtr out2 = upload(cpu, std::vector<float>(n0 * width, 0.0f));
                cpu.attention(CSlice(qb.get(), 0), 0, &views[0], 1, Slice{out2.get(), 0}, heads, heads, int(dim));
                const std::vector<float> causal = download(cpu, *out2, width);
                bool own = true;
                for (size_t i = 0; i < width; ++i) own = own && std::fabs(double(causal[i]) - stored(v[i])) <= 1e-6 * (1.0 + std::fabs(stored(v[i])));
                require(own, at + ": a causal view's first row is not its own value");
            }
        }
    }
}

void declared() {
    CpuBackend cpu;
    for (backend::Op op : {backend::Op::layer_norm, backend::Op::gelu, backend::Op::add_bias, backend::Op::mix_rows, backend::Op::rope_streams,
                           backend::Op::attention_full})
        require(cpu.implements(op), std::string("the CPU does not implement ") + backend::op_name(op));
}

}  // namespace

int main() {
    try {
        declared();
        layer_norm();
        gelu();
        add_bias();
        mix_rows();
        rope_streams();
        attention_full();
    } catch (const std::exception& e) {
        std::cerr << "vision-ops: FAIL: " << e.what() << "\n";
        return 1;
    }
    std::cout << "vision-ops: " << checks << " checks passed\n";
    return 0;
}
