#pragma once
// The GPU core's witnesses shared by the GPU backends' tests: each type's builds by their launches, each output against a double sum over the twin, a call's column or row its bits alone, and the fused delta step.
// A test passes its own types, classes, wave width, launches reader and require, so each keeps its own reporting.
#include "backends/backend.hpp"
#include "backends/device_profile.hpp"
#include "backends/gpu/kernels/shapes.h"
#include "core/bf16.hpp"
#include "core/fp16.hpp"
#include "format/weight_scales.hpp"
#include "quant/fp4.hpp"
#include "quant/types.hpp"
#include "row_classes.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace twin_checks {

using backend::Backend;
using backend::Buffer;
using backend::DeviceCaps;
using backend::DeviceProfile;
using backend::BufferPtr;
using backend::Dtype;
using backend::MatrixPaths;
using backend::Projection;
using backend::RowRun;
using backend::WeightView;
using backend::dtype_name;

// A failed check's report, and the kernels a backend launched since it was last asked, by name and count.
using Require = std::function<void(bool, const std::string&)>;
using Launches = std::function<std::map<std::string, size_t>(Backend&)>;

// Every value within 1e-4 * (1 + |reference|) of the CPU's, the bound the device tests hold F32 ops to; the first value outside it is printed.
inline bool f32_close(const std::vector<float>& got, const std::vector<float>& want) {
    bool ok = got.size() == want.size();
    size_t bad = 0;
    for (size_t i = 0; ok && i < got.size(); ++i) ok = std::fabs(got[i] - want[i]) <= 1e-4f * (1.0f + std::fabs(want[i])), bad = i;
    if (!ok && got.size() == want.size()) std::fprintf(stderr, "  value %zu: %.9g against %.9g\n", bad, got[bad], want[bad]);
    return ok;
}
inline bool same(const std::vector<float>& a, const std::vector<float>& b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0; }
// Double sums of the CPU's decoded weights wd over x, columns of nout, and beside each the sum of |w*x| that bounds an F32 reduction's rounding.
struct DoubleSums {
    std::vector<double> sum, mag;
};
inline DoubleSums double_sums(const std::vector<float>& wd, const std::vector<float>& x, size_t nin, size_t nout) {
    const size_t n = x.size() / nin * nout;
    DoubleSums d{std::vector<double>(n), std::vector<double>(n)};
    for (size_t i = 0; i < n; ++i) {
        const size_t r = i / nout, o = i % nout;
        for (size_t j = 0; j < nin; ++j) {
            const double p = double(wd[o * nin + j]) * x[r * nin + j];
            d.sum[i] += p;
            d.mag[i] += std::fabs(p);
        }
    }
    return d;
}

// An exact product against the double sums, within 16 F32 epsilons of the sum of |w*x|: each backend reduces in its own order, and the bound covers that rounding however the terms cancel.
inline bool exact_close(const std::vector<float>& got, const DoubleSums& d) {
    bool ok = got.size() == d.sum.size();
    for (size_t i = 0; ok && i < got.size(); ++i) {
        const double bound = 16 * double(FLT_EPSILON) * d.mag[i];
        ok = std::fabs(got[i] - d.sum[i]) <= bound + 1e-30;
        if (!ok) std::fprintf(stderr, "  value %zu: %.9g against %.9g, bound %.3g\n", i, got[i], d.sum[i], bound);
    }
    return ok;
}

inline std::vector<float> read(Backend& b, const Buffer& buf, size_t n, size_t off = 0) {
    std::vector<float> v(n);
    b.read(buf, off * sizeof(float), v.data(), n * sizeof(float));
    return v;
}

// A Q8_0 matrix of `rows` rows `nin` wide: each block's scale and values from the seed, every value in range.
inline std::vector<uint8_t> q8_matrix(size_t nin, size_t rows, uint32_t seed) {
    std::vector<uint8_t> m;
    const std::vector<float> scale = row_classes::values(rows * nin / 32, seed, 0.05f), v = row_classes::values(rows * nin, seed + 1, 127.0f);
    for (size_t b = 0; b < rows * nin / 32; ++b) {
        const uint16_t d = f32_to_f16(scale[b]);
        m.push_back(uint8_t(d));
        m.push_back(uint8_t(d >> 8));
        for (size_t j = 0; j < 32; ++j) m.push_back(uint8_t(int8_t(std::max(-127.0f, std::min(127.0f, std::round(v[b * 32 + j]))))));
    }
    return m;
}

// Rows of `type` from seeded bytes, every f16 or bf16 field set to a finite scale, so each value is defined and finite.
inline std::vector<uint8_t> typed_matrix(uint32_t type, size_t nin, size_t rows, uint32_t seed) {
    std::vector<uint8_t> m(quant::row_bytes(type, nin, rows));
    uint32_t r = seed * 2654435761u + 1;
    for (uint8_t& v : m) {
        r = r * 1664525u + 1013904223u;
        v = uint8_t(r >> 24);
    }
    const quant::StorageType& st = *quant::storage_type(type);
    const std::vector<float> scale = row_classes::values(m.size() / st.type_size * 2, seed + 7, 0.02f);
    const auto put = [&](size_t at, uint16_t h) {
        m[at] = uint8_t(h);
        m[at + 1] = uint8_t(h >> 8);
    };
    std::vector<size_t> halves;
    if (type == quant::GGML_TYPE_Q4_K || type == quant::GGML_TYPE_Q5_K || type == quant::GGML_TYPE_Q5_1 || type == quant::GGML_TYPE_Q4_1) halves = {0, 2};
    if (type == quant::GGML_TYPE_Q8_0 || type == quant::GGML_TYPE_Q5_0 || type == quant::GGML_TYPE_IQ4_NL || type == quant::GGML_TYPE_Q4_0) halves = {0};
    if (type == quant::GGML_TYPE_Q6_K) halves = {208};
    if (type == quant::GGML_TYPE_Q2_K) halves = {80, 82};
    if (type == quant::GGML_TYPE_Q3_K) halves = {108};
    if (type == quant::GGML_TYPE_TQ1_0) halves = {52};
    if (type == quant::GGML_TYPE_TQ2_0) halves = {64};
    if (type == quant::GGML_TYPE_IQ1_S || type == quant::GGML_TYPE_Q1_0 || type == quant::GGML_TYPE_Q2_0) halves = {0};
    if (type == quant::GGML_TYPE_IQ4_XS || type == quant::GGML_TYPE_IQ2_XXS || type == quant::GGML_TYPE_IQ2_XS || type == quant::GGML_TYPE_IQ2_S || type == quant::GGML_TYPE_IQ3_XXS || type == quant::GGML_TYPE_IQ3_S) halves = {0};
    for (size_t b = 0, i = 0; b < m.size() / st.type_size; ++b)
        for (size_t h : halves) put(b * st.type_size + h, f32_to_f16(scale[i++]));
    // IQ1_M's binary16 scale is the high nibbles of its four scale words, which take a finite scale's nibbles.
    if (type == quant::GGML_TYPE_IQ1_M)
        for (size_t b = 0, i = 0; b < m.size() / st.type_size; ++b, ++i) {
            const uint16_t h = f32_to_f16(scale[i]);
            for (size_t k = 0; k < 4; ++k) m[b * st.type_size + 49 + 2 * k] = uint8_t((m[b * st.type_size + 49 + 2 * k] & 15) | (((h >> (4 * k)) & 15) << 4));
        }
    // MXFP4's E8M0 scale byte at 2^-10 to 2^-6, as the raw-block fixtures' ordinary blocks take it.
    if (type == quant::GGML_TYPE_MXFP4)
        for (size_t b = 0; b < m.size() / st.type_size; ++b) m[b * st.type_size] = uint8_t(118 + m[b * st.type_size] % 5);
    if (type == quant::GGML_TYPE_F16) {
        const std::vector<float> v = row_classes::values(nin * rows, seed + 9);
        for (size_t i = 0; i < v.size(); ++i) put(2 * i, f32_to_f16(v[i]));
    }
    if (type == quant::GGML_TYPE_BF16) {
        const std::vector<float> v = row_classes::values(nin * rows, seed + 9);
        for (size_t i = 0; i < v.size(); ++i) {
            uint32_t u;
            std::memcpy(&u, &v[i], 4);
            put(2 * i, uint16_t(u >> 16));
        }
    }
    return m;
}

// The table's types beside F32 without twins, which take their exact product in every class.
inline bool exact_only(uint32_t type) {
    return type == quant::GGML_TYPE_BF16 || type == quant::GGML_TYPE_F16 ||
           type == quant::GGML_TYPE_TQ1_0 || type == quant::GGML_TYPE_TQ2_0;
}

// Flash-Next's projection shapes, input by output, that a type's rows fill whole: the K-quants' 256-value blocks the 2560- and 6144-wide rows, the 32-value types the 640-wide expert down rows too.
inline std::vector<std::array<size_t, 2>> flash_shapes(uint32_t type) {
    const size_t all[][2] = {{2560, 640}, {640, 2560}, {2560, 320}, {2560, 512}, {6144, 2560}};
    const uint64_t block = quant::storage_type(type)->block_size;
    std::vector<std::array<size_t, 2>> out;
    for (const auto& sh : all)
        if (sh[0] % block == 0) out.push_back({sh[0], sh[1]});
    return out;
}

// Each given exact type's product over several rows must give each output the bits of its row alone through the one-row build, plain and added, at each side of a block's outputs and rows.
// The widths give lanes chains of unequal length (2561, 2080), lanes without one (100), the hyper-connection maps' 10240 and F32 rows off the vector loads' alignment (2561).
inline void exact_rows(Backend& g, const std::vector<uint32_t>& kinds, const Launches& launches, const Require& require) {
    const size_t nouts[] = {1, 13, 512}, nrows[] = {2, 3, 4, 5, 7, 8, 9, 33};
    for (uint32_t type : kinds) {
        const std::string name = quant::storage_type(type)->name;
        const size_t block = quant::storage_type(type)->block_size;
        std::string tname = name;
        for (char& ch : tname) ch = char(std::tolower(ch));
        const auto build = [&](size_t n) { return type == quant::GGML_TYPE_F32 && n <= 4 ? std::string(n <= 2 ? "llmx_matmul_cols2_f32" : "llmx_matmul_cols4_f32") : "llmx_matmul_rows_" + tname; };
        const std::vector<size_t> widths = block == 1 ? std::vector<size_t>{2560, 2561, 100, 10240} : block == 32 ? std::vector<size_t>{2560, 2080} : std::vector<size_t>{2560};
        for (size_t nin : widths)
        for (size_t nout : nouts) {
            std::vector<uint8_t> w;
            if (type == quant::GGML_TYPE_F32) {
                const std::vector<float> v = row_classes::values(nin * nout, uint32_t(nout + 3));
                w.resize(v.size() * 4);
                std::memcpy(w.data(), v.data(), w.size());
            } else {
                w = typed_matrix(type, nin, nout, uint32_t(type * 13 + nout));
                if (type == quant::GGML_TYPE_Q8_0)
                    for (size_t b = 0; b < w.size() / 34; ++b) {
                        const uint16_t h = f32_to_f16(0.01f + 0.001f * float(b % 7));
                        w[b * 34] = uint8_t(h), w[b * 34 + 1] = uint8_t(h >> 8);
                    }
            }
            const std::vector<float> x = row_classes::values(nin * 33, uint32_t(nout + 11)), y0 = row_classes::values(nout * 33, uint32_t(nout + 17));
            auto wb = g.adopt(w.data(), w.size()), xb = g.adopt(x.data(), x.size() * 4), yb = g.alloc(nout * 33 * 4), y1 = g.alloc(nout * 4);
            for (size_t n : nrows)
                for (int add = 0; add < 2; ++add) {
                    const std::string at = name + " " + std::to_string(nin) + "->" + std::to_string(nout) + " over " + std::to_string(n) + " rows" + (add ? ", added" : "");
                    g.write(*yb, 0, y0.data(), y0.size() * 4);
                    launches(g);
                    if (add) g.matmul_add({type, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, n, {}, Dtype::f32);
                    else g.matmul({type, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, n, {}, Dtype::f32);
                    require(launches(g).count(build(n)) == 1, "a " + at + " exact product takes " + build(n));
                    const std::vector<float> all = read(g, *yb, nout * 33);
                    bool ok = std::memcmp(all.data() + n * nout, y0.data() + n * nout, (33 - n) * nout * 4) == 0;
                    for (size_t r = 0; r < n && ok; ++r) {
                        g.write(*y1, 0, y0.data() + r * nout, nout * 4);
                        if (add) g.matmul_add({type, {wb.get(), 0}}, {xb.get(), r * nin}, {y1.get(), 0}, nin, nout, 1, {}, Dtype::f32);
                        else g.matmul({type, {wb.get(), 0}}, {xb.get(), r * nin}, {y1.get(), 0}, nin, nout, 1, {}, Dtype::f32);
                        ok = std::memcmp(read(g, *y1, nout).data(), all.data() + r * nout, nout * 4) == 0;
                    }
                    require(ok, "every row of a " + at + " exact product is the row alone, bit for bit, and nothing past it is written");
                }
        }
    }
}

// The twin's rule as docs/STATUS.md gives it, on the host, at L levels: a block's scale and its values, 32767 for block-int16 and 127 for block-int8.
inline void twin_rule(const float* x, float& d, int16_t* q, float L) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
    if (amax == 0.0f) {
        d = 0.0f;
        for (int j = 0; j < 32; ++j) q[j] = 0;
        return;
    }
    int e = 0;
    std::frexp(amax, &e);
    const int s = (amax >= 0x1p-100f && amax <= 0x1p100f) ? 0 : 1 - e;
    const float an = std::ldexp(amax, s);
    const float k = L / an;
    for (int j = 0; j < 32; ++j) {
        const float r = std::ldexp(x[j], s) * k;
        const float t = std::min(std::floor(std::fabs(r) + 0.5f), L);
        q[j] = (int16_t)(r < 0.0f ? -t : t);
    }
    d = std::ldexp(an / L, -s);
}

// A Q8_0 row of `nin` values whose block b holds value 1 at position j and scale 1, zero elsewhere: its product reads the twin's value d q at (b, j) alone.
inline std::vector<uint8_t> one_hot_q8(size_t nin, size_t b, size_t j) {
    std::vector<uint8_t> row(nin / 32 * 34, 0);
    const uint16_t one = 0x3c00;
    row[b * 34] = uint8_t(one);
    row[b * 34 + 1] = uint8_t(one >> 8);
    row[b * 34 + 2 + j] = 1;
    return row;
}

// A block's min for its sub-block: Q4_K's and Q5_K's dmin times the 6-bit min as the format packs them, Q4_1's and Q5_1's min, Q2_K's dmin times the larger 4-bit min of the sub-block's halves.
// Q6_K, Q3_K, Q4_0, Q5_0, IQ4_NL, IQ4_XS, the IQ1, IQ2 and IQ3 types, Q1_0, Q2_0 and MXFP4 have none.
inline float k_min(uint32_t type, const uint8_t* block, int sub) {
    if (type == quant::GGML_TYPE_Q6_K || type == quant::GGML_TYPE_Q3_K || type == quant::GGML_TYPE_Q4_0 || type == quant::GGML_TYPE_Q5_0 || type == quant::GGML_TYPE_IQ4_NL || type == quant::GGML_TYPE_IQ4_XS ||
        type == quant::GGML_TYPE_IQ2_XXS || type == quant::GGML_TYPE_IQ2_XS || type == quant::GGML_TYPE_IQ2_S || type == quant::GGML_TYPE_IQ3_XXS || type == quant::GGML_TYPE_IQ3_S || type == quant::GGML_TYPE_IQ1_S || type == quant::GGML_TYPE_IQ1_M || type == quant::GGML_TYPE_Q1_0 || type == quant::GGML_TYPE_Q2_0 || type == quant::GGML_TYPE_MXFP4 ||
        type == quant::GGML_TYPE_NVFP4)
        return 0.0f;
    if (type == quant::GGML_TYPE_Q2_K) return f16_to_f32(uint16_t(block[82] | (block[83] << 8))) * float(std::max(block[2 * sub] >> 4, block[2 * sub + 1] >> 4));
    if (type == quant::GGML_TYPE_Q5_1 || type == quant::GGML_TYPE_Q4_1) return f16_to_f32(uint16_t(block[2] | (block[3] << 8)));
    const uint8_t* q = block + 4;
    const int m = sub < 4 ? q[sub + 4] & 63 : (q[sub + 4] >> 4) | ((q[sub] >> 6) << 4);
    return f16_to_f32(uint16_t(block[2] | (block[3] << 8))) * float(m);
}

// A twin product's output (o, r) against a double sum of the CPU's decoded weights over the twin of x by its rule, within 1e-6 of its magnitude, each min counted apart since a sub-block's term subtracts it.
inline bool twin_close(uint32_t type, const std::vector<uint8_t>& w, const std::vector<float>& wd, const std::vector<float>& x, size_t nin, size_t o, size_t r, float L, float got) {
    const quant::StorageType& st = *quant::storage_type(type);
    double sum = 0, mag = 0;
    float d;
    int16_t q[32];
    for (size_t b = 0; b < nin / 32; ++b) {
        twin_rule(x.data() + r * nin + b * 32, d, q, L);
        const float dm = k_min(type, w.data() + (o * nin + b * 32) / st.block_size * st.type_size, int(b % (st.block_size / 32)));
        for (size_t j = 0; j < 32; ++j) {
            const double xv = double(d) * q[j], wv = wd[o * nin + b * 32 + j];
            sum += wv * xv;
            mag += (std::fabs(wv) + 2 * std::fabs(double(dm))) * std::fabs(xv);
        }
    }
    return std::fabs(got - sum) <= 1e-6 * mag + 1e-30;
}

// Whether a one-column 8-bit MXFP4 row of nin values runs the quad rows' builds on a wave of W lanes: gpu_shape::quad_rows at the 32-lane cluster a 64-lane wave gives rows of 16 blocks or more.
inline bool quad_build(uint32_t type, Dtype dt, size_t nin, size_t W) { return type == quant::GGML_TYPE_MXFP4 && dt == Dtype::int8 && W == 64 && gpu_shape::quad_rows(32, nin / 32); }

// Whether row_form takes the quarter form for a Q4_0, Q4_1, Q5_0 or Q5_1 row of nin values on a wave of W lanes: in the F16 class where four lanes a block fill the cluster's steps fuller than one, never in the int8 class.
inline bool quarter_form(uint32_t type, size_t nin, size_t W, bool int8) {
    if (int8 || (type != quant::GGML_TYPE_Q4_0 && type != quant::GGML_TYPE_Q4_1 && type != quant::GGML_TYPE_Q5_0 && type != quant::GGML_TYPE_Q5_1)) return false;
    const size_t blocks = nin / 32;
    const auto span = [W](size_t units) {
        size_t cl = 1;
        while (cl < W && cl < units) cl *= 2;
        return (units + cl - 1) / cl * cl;
    };
    return 4 * blocks * span(blocks) > blocks * span(4 * blocks);
}

// Widths whose rows take every lane cluster a type's rule gives (kernels/int16.gpu, k_rows): lane units below the wave, at the Q4_K and Q5_K cap, above it and not a multiple of the cluster.
// The 32-value twin types' also lie each side of the lane form's switches on a 64-lane wave (one lane at 1024 and 2048, quarter at 1056 and 2080); 64- and 128-value types take one and three blocks, and 37 outputs no block's rows divide.
inline std::vector<std::array<size_t, 2>> cluster_shapes(uint32_t type) {
    const size_t b = quant::storage_type(type)->block_size;
    const std::vector<size_t> widths = b == 256 ? std::vector<size_t>{256, 512, 768, 1024, 4096}
                                       : b == 32 ? std::vector<size_t>{32, 64, 96, 640, 1024, 1056, 2048, 2080, 4096}
                                                 : std::vector<size_t>{b, 3 * b, 640, 1024, 2048, 4096};
    std::vector<std::array<size_t, 2>> out;
    for (size_t nin : widths) out.push_back({nin, 37});
    return out;
}

// The twin products of the given types at Flash-Next's shapes in the given classes, extents on both sides of each crossover, each output against a double sum over the twin.
// The backend's launches name the build or tile that ran, and a row of a call is the row alone or in another call of its class, every generated column of up to 64; the cluster widths (cluster_shapes) take the generated rows alone.
inline void twin_types(Backend& g, Backend& c, const std::vector<uint32_t>& kinds, const std::vector<Dtype>& dtypes, size_t wave, const Launches& launches,
                       const std::function<bool(uint32_t, Dtype, size_t, size_t, size_t)>& tall, const Require& require) {
    const size_t cols[] = {1, 2, 3, 5, 8, 9, 16, 17, 33}, extents[] = {2, 33, 64, 65, 320};
    for (uint32_t type : kinds) {
        std::string tname = quant::storage_type(type)->name;
        for (char& ch : tname) ch = char(std::tolower(ch));
        std::vector<std::array<size_t, 2>> shapes = flash_shapes(type);
        const size_t flash = shapes.size();
        for (const auto& sh : cluster_shapes(type)) shapes.push_back(sh);
        if (type == quant::GGML_TYPE_Q4_0 || type == quant::GGML_TYPE_Q4_1 || type == quant::GGML_TYPE_Q5_0 || type == quant::GGML_TYPE_Q5_1) {
            size_t quarter = 0;
            for (const auto& sh : shapes) quarter += quarter_form(type, sh[0], wave, false);
            require(quarter > 0 && quarter < shapes.size(), tname + "'s widths take both lane forms in the F16 class");
        }
        for (size_t si = 0; si < shapes.size(); ++si) {
            const auto& sh = shapes[si];
            const size_t nin = sh[0], nout = sh[1], nmax = 320;
            const std::string at = tname + " " + std::to_string(nin) + "->" + std::to_string(nout);
            const std::vector<uint8_t> w = typed_matrix(type, nin, nout, uint32_t(type * 17 + nin + nout));
            const std::vector<float> x = row_classes::values(nin * nmax, uint32_t(nin + 11));
            // The CPU's decoded rows, the embedding of every row.
            std::vector<uint32_t> ids(nout);
            for (size_t i = 0; i < nout; ++i) ids[i] = uint32_t(i);
            auto cw = c.adopt(w.data(), w.size()), ce = c.alloc(nout * nin * 4);
            c.embed({ce.get(), 0}, {type, {cw.get(), 0}}, nin, nout, ids.data(), nout);
            const std::vector<float> wd = read(c, *ce, nout * nin);
            auto wb = g.adopt(w.data(), w.size()), xb = g.adopt(x.data(), x.size() * 4), yb = g.alloc(nout * nmax * 4), y1 = g.alloc(nout * nmax * 4);
            for (Dtype dt : dtypes) {
                const std::string form = dt == Dtype::int8 ? "8" : "16", cls = at + ", " + dtype_name(dt);
                const float L = dt == Dtype::int8 ? 127.0f : 32767.0f;
                const auto run = [&](size_t n, size_t extent, Buffer& y, size_t x0) {
                    const RowRun rr{n, extent};
                    launches(g);
                    MatrixPaths paths;
                    g.swap_matrix_paths(paths);
                    g.matmul({type, {wb.get(), 0}}, {xb.get(), x0 * nin}, {&y, 0}, nin, nout, n, {&rr, 1}, dt);
                    g.swap_matrix_paths(paths);
                    require(paths.take() == std::vector<std::string>{dt == Dtype::int8 ? "block-int8" : "block-int16"}, "a " + cls + " product records its twin's path");
                    std::map<std::string, size_t> mine;
                    for (const auto& [k, v] : launches(g))
                        if (k.rfind("llmx_" + tname, 0) == 0 || k.rfind("llmx_matmul", 0) == 0 || k.rfind("llmx_q8_0", 0) == 0) mine[k] = v;
                    return mine;
                };
                const auto check = [&](const std::vector<float>& y, size_t n, const std::string& what) {
                    bool ok = true;
                    for (size_t r : {size_t(0), n / 2, n - 1})
                        for (size_t o = 0; o < nout && ok; ++o) ok = twin_close(type, w, wd, x, nin, o, r, L, y[r * nout + o]);
                    require(ok, what + " within 1e-6 of a double sum over the twin");
                };
                // Generated rows: each build by the columns a launch takes, sixteen at most, and each column the column alone.
                for (size_t n : cols) {
                    std::map<std::string, size_t> want;
                    for (size_t left = n; left; left -= std::min<size_t>(left, 16)) {
                        const size_t m = std::min<size_t>(left, 16), b = m <= 1 ? 1 : m <= 2 ? 2 : m <= 4 ? 4 : m <= 8 ? 8 : 16;
                        ++want[b == 1 && quad_build(type, dt, nin, wave) ? "llmx_mxfp4_quad_rows8_1" : "llmx_" + tname + (quarter_form(type, nin, wave, dt == Dtype::int8) ? "_quarter" : "") + "_rows" + form + "_" + std::to_string(b)];
                    }
                    const std::string what = "a " + cls + " call of " + std::to_string(n) + " generated rows";
                    require(run(n, 1, *yb, 0) == want, what + " runs its row builds alone");
                    const std::vector<float> all = read(g, *yb, nout * n);
                    check(all, n, what);
                }
                // Every column of 1 to 64 generated rows, as one run and as one run a row as a decode pass of n sequences gives them, is the column alone, bit for bit.
                {
                    const size_t most = 64;
                    std::vector<float> one(most * nout);
                    for (size_t r = 0; r < most; ++r) {
                        run(1, 1, *y1, r);
                        g.read(*y1, 0, one.data() + r * nout, nout * 4);
                    }
                    bool alone = true;
                    for (size_t n = 1; n <= most; ++n) {
                        run(n, 1, *yb, 0);
                        alone = alone && std::memcmp(read(g, *yb, nout * n).data(), one.data(), nout * n * 4) == 0;
                        std::vector<RowRun> each(n);
                        for (size_t r = 0; r < n; ++r) each[r] = {r + 1, 1};
                        g.matmul({type, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, n, {each.data(), n}, dt);
                        alone = alone && std::memcmp(read(g, *yb, nout * n).data(), one.data(), nout * n * 4) == 0;
                    }
                    require(alone, "a " + cls + " call of 1 to 64 generated rows: every column is the column alone, bit for bit, as one run and as one run a row");
                }
                // Prompt rows: the tile, and the first rows of a prompt the same bits in a call of fewer of its rows.
                for (size_t e : extents) {
                    if (si >= flash) break;
                    const std::string what = "a " + cls + " prompt of " + std::to_string(e) + " rows";
                    const std::map<std::string, size_t> ran = run(e, e, *yb, 0);
                    const std::string tile = "llmx_" + tname + "_tile" + form + (tall(type, dt, nin, nout, e) ? "_tall" : "");
                    require(ran.size() == 1 && ran.begin()->first == tile, what + " runs " + tile + " alone");
                    const std::vector<float> all = read(g, *yb, nout * e);
                    check(all, e, what);
                    const size_t part = (e + 1) / 2;
                    run(part, e, *y1, 0);
                    require(std::memcmp(read(g, *y1, nout * part).data(), all.data(), nout * part * 4) == 0, what + ": its first " + std::to_string(part) + " rows in a call of their own, bit for bit");
                }
            }
        }
    }
}

// A class's Q8_0 products, f16 on the block-int16 twin and int8 on the block-int8 one, against the twin's rule and a double oracle on the twin.
// Every decode column must be the column alone, and a prompt row the same bits in any call.
inline void twin_products(Backend& g, const DeviceCaps& caps, Dtype dt, const Launches& launches, const Require& require) {
    const float L = dt == Dtype::int8 ? 127.0f : 32767.0f;
    const std::string form = dt == Dtype::int8 ? "8-bit" : "16-bit";
    // Twin: blocks of ordinary, tiny, huge, tied and zero values, read through one-hot rows on both kernels.
    const size_t nin = 32 * 6;
    std::vector<float> x = row_classes::values(nin * 2, 31);
    for (size_t j = 0; j < 32; ++j) {
        x[32 + j] *= 1e-35f;              // a block whose scale needs the power of two
        x[64 + j] *= 1e35f;               // and one whose largest value would overflow its reciprocal's product
        x[96 + j] = 0.0f;                 // a block of zeros
        x[128 + j] = j == 0 ? L : (float)j + 0.5f;   // exact halves: ties away from zero
    }
    x[160] = -L;
    x[161] = -0.5f;
    std::vector<uint8_t> w;
    for (size_t o = 0; o < nin; ++o) {
        const std::vector<uint8_t> row = one_hot_q8(nin, o / 32, o % 32);
        w.insert(w.end(), row.begin(), row.end());
    }
    auto wb = g.adopt(w.data(), w.size()), xb = g.adopt(x.data(), x.size() * 4), yb = g.alloc(nin * 2 * 4);
    bool rule = true;
    for (size_t extent : {size_t(1), size_t(2)}) {
        const RowRun run[1] = {{2, extent}};
        g.matmul({quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nin, 2, {run, 1}, dt);
        const std::vector<float> y = read(g, *yb, nin * 2);
        for (size_t r = 0; r < 2; ++r)
            for (size_t b = 0; b < nin / 32; ++b) {
                float d;
                int16_t q[32];
                twin_rule(x.data() + r * nin + b * 32, d, q, L);
                for (size_t j = 0; j < 32; ++j) {
                    const float want = d * (float)q[j];
                    rule = rule && std::memcmp(&y[r * nin + b * 32 + j], &want, 4) == 0;
                }
            }
    }
    require(rule, "the " + form + " twin is its rule, through the row kernel and the tile");
    // Products against a double oracle on the twin, rows 4096 wide, generated and prompt rows.
    const size_t width = 4096, nout = 40, rows = 21;
    const std::vector<uint8_t> m = q8_matrix(width, nout, 33);
    const std::vector<float> xs = row_classes::values(width * rows, 34);
    auto mb = g.adopt(m.data(), m.size()), xsb = g.adopt(xs.data(), xs.size() * 4), ys = g.alloc(nout * rows * 4);
    std::vector<std::vector<float>> twin_d(rows, std::vector<float>(width / 32));
    std::vector<std::vector<int16_t>> twin_q(rows, std::vector<int16_t>(width));
    for (size_t r = 0; r < rows; ++r)
        for (size_t b = 0; b < width / 32; ++b) twin_rule(xs.data() + r * width + b * 32, twin_d[r][b], twin_q[r].data() + b * 32, L);
    for (size_t extent : {size_t(1), size_t(2)}) {
        const RowRun run[1] = {{rows, extent}};
        g.matmul({quant::GGML_TYPE_Q8_0, {mb.get(), 0}}, {xsb.get(), 0}, {ys.get(), 0}, width, nout, rows, {run, 1}, dt);
        const std::vector<float> y = read(g, *ys, nout * rows);
        bool close_ok = true;
        for (size_t r = 0; r < rows; ++r)
            for (size_t o = 0; o < nout; ++o) {
                double sum = 0.0, mag = 0.0;
                for (size_t b = 0; b < width / 32; ++b) {
                    const uint8_t* block = m.data() + (o * (width / 32) + b) * 34;
                    const double dw = f16_to_f32((uint16_t)(block[0] | block[1] << 8));
                    long isum = 0;
                    for (size_t j = 0; j < 32; ++j) isum += (long)(int8_t)block[2 + j] * twin_q[r][b * 32 + j];
                    sum += dw * twin_d[r][b] * (double)isum;
                    mag += std::fabs(dw * twin_d[r][b] * (double)isum);
                }
                close_ok = close_ok && std::fabs(y[r * nout + o] - sum) <= 1e-6 * mag + 1e-30;
            }
        require(close_ok, std::string("the ") + (extent == 1 ? "row kernel" : "tile") + " within 1e-6 of a double sum over the " + form + " twin");
    }
    // A group of three projections of one X in one tile call, its inner dimension split for a prompt of 21 rows, each projection within 1e-6 of the double sum over the twin.
    {
        const size_t outs[3] = {40, 104, 64};
        std::vector<std::vector<uint8_t>> gm;
        std::vector<BufferPtr> gw, gy;
        for (size_t p = 0; p < 3; ++p) {
            gm.push_back(q8_matrix(width, outs[p], 81 + (uint32_t)p));
            gw.push_back(g.adopt(gm[p].data(), gm[p].size()));
            gy.push_back(g.alloc(outs[p] * rows * 4));
        }
        const RowRun run[1] = {{rows, size_t(21)}};
        g.matmul_group({Projection({quant::GGML_TYPE_Q8_0, {gw[0].get(), 0}}, {gy[0].get(), 0}, outs[0]),
                        Projection({quant::GGML_TYPE_Q8_0, {gw[1].get(), 0}}, {gy[1].get(), 0}, outs[1]),
                        Projection({quant::GGML_TYPE_Q8_0, {gw[2].get(), 0}}, {gy[2].get(), 0}, outs[2])},
                       {xsb.get(), 0}, width, rows, {run, 1}, dt);
        bool group_ok = true;
        for (size_t p = 0; p < 3; ++p) {
            const std::vector<float> y = read(g, *gy[p], outs[p] * rows);
            for (size_t r = 0; r < rows; ++r)
                for (size_t o = 0; o < outs[p]; ++o) {
                    double sum = 0.0, mag = 0.0;
                    for (size_t b = 0; b < width / 32; ++b) {
                        const uint8_t* block = gm[p].data() + (o * (width / 32) + b) * 34;
                        const double dw = f16_to_f32((uint16_t)(block[0] | block[1] << 8));
                        long isum = 0;
                        for (size_t j = 0; j < 32; ++j) isum += (long)(int8_t)block[2 + j] * twin_q[r][b * 32 + j];
                        sum += dw * twin_d[r][b] * (double)isum;
                        mag += std::fabs(dw * twin_d[r][b] * (double)isum);
                    }
                    group_ok = group_ok && std::fabs(y[r * outs[p] + o] - sum) <= 1e-6 * mag + 1e-30;
                }
        }
        require(group_ok, form + ": a group of three projections in one split tile call within 1e-6 of a double sum over the twin");
    }
    auto y1 = g.alloc(nout * 4);
    // Every decode column of 1 to 64 generated rows, up to four launches of 16, is the column alone, bit for bit.
    // The residual add and a group of three projections are held so at counts either side of each launch's edge.
    {
        // 37 outputs, which neither two nor four divides, so each build's last wave or block holds rows past the end, which it must not store.
        const size_t cols = 64, dn = 37;
        bool alone = true, added = true, grouped = true;
        for (size_t w : {size_t(4096), size_t(1280), size_t(1312)}) {
            const std::vector<uint8_t> mw = q8_matrix(w, dn, 61);
            const std::vector<float> xw = row_classes::values(w * cols, 62), base = row_classes::values(dn * cols, 63);
            auto mwb = g.adopt(mw.data(), mw.size()), xwb = g.adopt(xw.data(), xw.size() * 4);
            auto yw = g.alloc(dn * cols * 4), ya = g.alloc(dn * cols * 4), yb = g.alloc(dn * cols * 4), yc = g.alloc(dn * cols * 4);
            const WeightView wv{quant::GGML_TYPE_Q8_0, {mwb.get(), 0}};
            std::vector<std::vector<float>> one(cols), one_added(cols);
            for (size_t r = 0; r < cols; ++r) {
                const RowRun run[1] = {{1, 1}};
                g.matmul(wv, {xwb.get(), r * w}, {y1.get(), 0}, w, dn, 1, {run, 1}, dt);
                one[r] = read(g, *y1, dn);
                g.write(*y1, 0, base.data() + r * dn, dn * 4);
                g.matmul_add(wv, {xwb.get(), r * w}, {y1.get(), 0}, w, dn, 1, {run, 1}, dt);
                one_added[r] = read(g, *y1, dn);
            }
            for (size_t n = 1; n <= cols; ++n) {
                const RowRun run[1] = {{n, 1}};
                g.matmul(wv, {xwb.get(), 0}, {yw.get(), 0}, w, dn, n, {run, 1}, dt);
                const std::vector<float> y = read(g, *yw, dn * n);
                for (size_t r = 0; r < n; ++r) alone = alone && std::memcmp(y.data() + r * dn, one[r].data(), dn * 4) == 0;
                // The same rows as n sequences' runs of one generated row each, as a decode pass of n sequences gives them.
                std::vector<RowRun> each(n);
                for (size_t r = 0; r < n; ++r) each[r] = {r + 1, 1};
                g.matmul(wv, {xwb.get(), 0}, {yw.get(), 0}, w, dn, n, {each.data(), n}, dt);
                const std::vector<float> ys = read(g, *yw, dn * n);
                for (size_t r = 0; r < n; ++r) alone = alone && std::memcmp(ys.data() + r * dn, one[r].data(), dn * 4) == 0;
            }
            for (size_t n : {size_t(1), size_t(2), size_t(3), size_t(15), size_t(16), size_t(17), size_t(31), size_t(32), size_t(33), size_t(48), size_t(49), size_t(63), size_t(64)}) {
                const RowRun run[1] = {{n, 1}};
                g.write(*ya, 0, base.data(), dn * n * 4);
                g.matmul_add(wv, {xwb.get(), 0}, {ya.get(), 0}, w, dn, n, {run, 1}, dt);
                const std::vector<float> y = read(g, *ya, dn * n);
                for (size_t r = 0; r < n; ++r) added = added && std::memcmp(y.data() + r * dn, one_added[r].data(), dn * 4) == 0;
                g.matmul_group({Projection(wv, {yw.get(), 0}, dn), Projection(wv, {yb.get(), 0}, dn), Projection(wv, {yc.get(), 0}, dn)},
                               {xwb.get(), 0}, w, n, {run, 1}, dt);
                for (BufferPtr* out : {&yw, &yb, &yc}) {
                    const std::vector<float> yg = read(g, **out, dn * n);
                    for (size_t r = 0; r < n; ++r) grouped = grouped && std::memcmp(yg.data() + r * dn, one[r].data(), dn * 4) == 0;
                }
            }
        }
        require(alone, form + ": every decode column of 1 to 64 generated rows is the column alone, bit for bit, as one run and as one run a row, rows 4096, 1280 and 1312 wide (41 blocks, the int8 class's quarter layout)");
        require(added, form + ": every decode column of the residual add at 1 to 64 rows is the column alone, bit for bit");
        require(grouped, form + ": every decode column of a group of three projections at 1 to 64 rows is the column alone, bit for bit");
    }
    // A row gives the same bits whatever rows a wave the call's output count takes: the first 37 rows of one matrix in calls of 37 to 16384 outputs, at 1 to 16 columns in one launch, 1280 wide and 1312 (the int8 quarter layout).
    {
        const size_t big = 16384, head = 37, cols = 16;
        bool same = true;
        for (size_t w : {size_t(1280), size_t(1312)}) {
            const std::vector<uint8_t> mw = q8_matrix(w, big, 67);
            const std::vector<float> xw = row_classes::values(w * cols, 68);
            auto mwb = g.adopt(mw.data(), mw.size()), xwb = g.adopt(xw.data(), xw.size() * 4), yw = g.alloc(big * cols * 4);
            const WeightView wv{quant::GGML_TYPE_Q8_0, {mwb.get(), 0}};
            for (size_t n = 1; n <= cols; ++n) {
                const RowRun run[1] = {{n, 1}};
                std::vector<float> first;
                for (size_t nout : {head, size_t(1024), size_t(2048), size_t(3072), size_t(4096), size_t(6144), size_t(8192), size_t(12288), big}) {
                    g.matmul(wv, {xwb.get(), 0}, {yw.get(), 0}, w, nout, n, {run, 1}, dt);
                    const std::vector<float> y = read(g, *yw, nout * n);
                    std::vector<float> rows;
                    for (size_t r = 0; r < n; ++r) rows.insert(rows.end(), y.begin() + r * nout, y.begin() + r * nout + head);
                    if (first.empty()) first = rows;
                    same = same && std::memcmp(rows.data(), first.data(), rows.size() * 4) == 0;
                }
            }
        }
        require(same, form + ": a decode row gives the same bits in calls of 37 to 16384 outputs, whatever rows a wave they take, at 1 to 16 columns");
    }
    // A decode group of three different projections in one launch gives each projection's own call, bit for bit; 1248 wide takes the int8 quarter layout.
    {
        const size_t outs[3] = {41, 104, 63};
        bool group_ok = true;
        for (size_t w : {size_t(1280), size_t(1248)}) {
            const std::vector<float> xw = row_classes::values(w * 17, 91);
            auto xwb = g.adopt(xw.data(), xw.size() * 4);
            std::vector<std::vector<uint8_t>> gm;
            std::vector<BufferPtr> gw, gy, ys;
            for (size_t p = 0; p < 3; ++p) {
                gm.push_back(q8_matrix(w, outs[p], 92 + (uint32_t)p));
                gw.push_back(g.adopt(gm[p].data(), gm[p].size()));
                gy.push_back(g.alloc(outs[p] * 17 * 4));
                ys.push_back(g.alloc(outs[p] * 17 * 4));
            }
            for (size_t n : {size_t(1), size_t(2), size_t(17)}) {
                const RowRun run[1] = {{n, 1}};
                for (size_t p = 0; p < 3; ++p) g.matmul({quant::GGML_TYPE_Q8_0, {gw[p].get(), 0}}, {xwb.get(), 0}, {ys[p].get(), 0}, w, outs[p], n, {run, 1}, dt);
                g.matmul_group({Projection({quant::GGML_TYPE_Q8_0, {gw[0].get(), 0}}, {gy[0].get(), 0}, outs[0]),
                                Projection({quant::GGML_TYPE_Q8_0, {gw[1].get(), 0}}, {gy[1].get(), 0}, outs[1]),
                                Projection({quant::GGML_TYPE_Q8_0, {gw[2].get(), 0}}, {gy[2].get(), 0}, outs[2])},
                               {xwb.get(), 0}, w, n, {run, 1}, dt);
                for (size_t p = 0; p < 3; ++p) {
                    const std::vector<float> a = read(g, *gy[p], outs[p] * n), b = read(g, *ys[p], outs[p] * n);
                    group_ok = group_ok && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
                }
            }
        }
        require(group_ok, form + ": a decode group of three different projections in one launch is each projection's own call, bit for bit, at 1, 2 and 17 rows");
    }
    // DeepSeek-V4.1's query and window projections, 1280 and 512 rows, and a third of 1024, as two and three projections of one decode call on rows across the step rule's edges.
    // At one column the three take two rows a wave where the two take one, and each projection is its own call at 1 and 2 columns; a prompt's tile splits by the call's outputs, so a prompt group is not held to it.
    {
        const size_t outs[3] = {1280, 512, 1024};
        bool group_ok = true;
        for (size_t w : {size_t(1280), size_t(5120), size_t(6144)}) {
            const std::vector<float> xw = row_classes::values(w * 17, 93);
            auto xwb = g.adopt(xw.data(), xw.size() * 4);
            std::vector<BufferPtr> gw, gy, ys;
            for (size_t p = 0; p < 3; ++p) {
                const std::vector<uint8_t> m = q8_matrix(w, outs[p], 94 + (uint32_t)p);
                gw.push_back(g.adopt(m.data(), m.size()));
                gy.push_back(g.alloc(outs[p] * 17 * 4));
                ys.push_back(g.alloc(outs[p] * 17 * 4));
            }
            const auto proj = [&](size_t p, const BufferPtr& y) { return Projection({quant::GGML_TYPE_Q8_0, {gw[p].get(), 0}}, {y.get(), 0}, outs[p]); };
            for (size_t nproj : {size_t(2), size_t(3)})
                for (const auto& [n, extent] : std::vector<std::pair<size_t, size_t>>{{1, 1}, {2, 1}}) {
                    const RowRun run[1] = {{n, extent}};
                    for (size_t p = 0; p < nproj; ++p) g.matmul({quant::GGML_TYPE_Q8_0, {gw[p].get(), 0}}, {xwb.get(), 0}, {ys[p].get(), 0}, w, outs[p], n, {run, 1}, dt);
                    if (nproj == 2) g.matmul_group({proj(0, gy[0]), proj(1, gy[1])}, {xwb.get(), 0}, w, n, {run, 1}, dt);
                    else g.matmul_group({proj(0, gy[0]), proj(1, gy[1]), proj(2, gy[2])}, {xwb.get(), 0}, w, n, {run, 1}, dt);
                    for (size_t p = 0; p < nproj; ++p) {
                        const std::vector<float> a = read(g, *gy[p], outs[p] * n), b = read(g, *ys[p], outs[p] * n);
                        group_ok = group_ok && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
                    }
                }
        }
        require(group_ok, form + ": groups of 1280 and 512 rows, and of a third of 1024, are each projection's own call, bit for bit, at 1 and 2 columns");
    }
    // A prompt row gives the same bits in calls of 1 to 21 rows and of 63 to 65, 128 and 130, which cross the tile's 64 rows, over 100 outputs, which end inside its 64.
    {
        const size_t pn = 130, po = 100;
        const std::vector<uint8_t> mp = q8_matrix(width, po, 71);
        const std::vector<float> xp = row_classes::values(width * pn, 72);
        auto mpb = g.adopt(mp.data(), mp.size()), xpb = g.adopt(xp.data(), xp.size() * 4), yp = g.alloc(po * pn * 4), yp1 = g.alloc(po * 4);
        const WeightView wv{quant::GGML_TYPE_Q8_0, {mpb.get(), 0}};
        std::vector<std::vector<float>> alone(pn);
        for (size_t r = 0; r < pn; ++r) {
            const RowRun run[1] = {{1, 2}};
            g.matmul(wv, {xpb.get(), r * width}, {yp1.get(), 0}, width, po, 1, {run, 1}, dt);
            alone[r] = read(g, *yp1, po);
        }
        bool tiled = true;
        std::vector<size_t> counts;
        for (size_t n = 1; n <= 21; ++n) counts.push_back(n);
        for (size_t n : {size_t(63), size_t(64), size_t(65), size_t(128), size_t(130)}) counts.push_back(n);
        for (size_t n : counts) {
            const RowRun run[1] = {{n, 2}};
            g.matmul(wv, {xpb.get(), 0}, {yp.get(), 0}, width, po, n, {run, 1}, dt);
            const std::vector<float> y = read(g, *yp, po * n);
            for (size_t r = 0; r < n; ++r) tiled = tiled && std::memcmp(y.data() + r * po, alone[r].data(), po * 4) == 0;
        }
        require(tiled, form + ": a prompt row gives the same bits in calls of 1 to 130 rows across the tile's edges");
    }
    // A matrix of an odd number of blocks, 65 outputs of 3 blocks adopted alone, so its last block starts on a word and the tile's last word load of it runs past the matrix: each prompt row the bits of that row alone.
    {
        const size_t sw = 96, so = 65, sn = 70;
        const std::vector<uint8_t> ms = q8_matrix(sw, so, 73);
        const std::vector<float> xs = row_classes::values(sw * sn, 74);
        auto msb = g.adopt(ms.data(), ms.size()), xsb2 = g.adopt(xs.data(), xs.size() * 4), ysn = g.alloc(so * sn * 4), ys1 = g.alloc(so * 4);
        const WeightView wv{quant::GGML_TYPE_Q8_0, {msb.get(), 0}};
        bool edge = true;
        const RowRun all[1] = {{sn, 2}};
        g.matmul(wv, {xsb2.get(), 0}, {ysn.get(), 0}, sw, so, sn, {all, 1}, dt);
        const std::vector<float> y = read(g, *ysn, so * sn);
        for (size_t r = 0; r < sn; ++r) {
            const RowRun run[1] = {{1, 2}};
            g.matmul(wv, {xsb2.get(), r * sw}, {ys1.get(), 0}, sw, so, 1, {run, 1}, dt);
            edge = edge && std::memcmp(y.data() + r * so, read(g, *ys1, so).data(), so * 4) == 0;
        }
        // The same matrix with room after it in its buffer gives the same bits, so the load past its end reads nothing that reaches a sum.
        std::vector<uint8_t> roomy = ms;
        roomy.resize(ms.size() + 64, 0xa5);
        auto rb = g.adopt(roomy.data(), roomy.size());
        g.matmul({quant::GGML_TYPE_Q8_0, {rb.get(), 0}}, {xsb2.get(), 0}, {ysn.get(), 0}, sw, so, sn, {all, 1}, dt);
        edge = edge && std::memcmp(read(g, *ysn, so * sn).data(), y.data(), so * sn * 4) == 0;
        require(edge, form + ": a prompt row of a matrix whose last block ends inside a word gives the bits of that row alone and of the matrix with room after it");
    }
    // The tall build gives the 64-output build's bits, a whole matrix against slices of the same inner split (docs/src/backends-gpu.md).
    // The split and the build follow the device's units and profile, so a shape counts only where whole and slice split alike, and one shape must set the tall build against the 64-output one.
    const DeviceProfile prof = backend::profile_for(caps);
    const auto tiles_of = [](size_t rows) { return (std::min<size_t>(rows, 512) + 63) / 64; };
    const auto kper_of = [&](size_t n, size_t tw, size_t tr) { return backend::split_blocks(prof, caps, (n + 63) / 64 * tiles_of(tr), tw / 32); };
    const auto tall_of = [&](size_t n, size_t tw, size_t tr) {
        return backend::tile_rows_for(caps, prof, 64, 64, 128, n, tiles_of(tr), tw, dt == Dtype::int8 ? prof.qtile_tall_from_8bit : prof.qtile_tall_from, dt == Dtype::int8) == 128;
    };
    const auto meets = [&](const std::array<size_t, 4>& s) { return kper_of(s[2], s[0], s[1]) == kper_of(s[3], s[0], s[1]) && tall_of(s[2], s[0], s[1]) != tall_of(s[3], s[0], s[1]); };
    std::vector<std::array<size_t, 4>> shapes = {{512, 70, 3940, 100}, {512, 70, 3000, 100}, {4096, 520, 5860, 3840}, {4128, 520, 5860, 3840}, {4096, 70, 3776, 3072}};
    // The fixed shapes meet that only on some numbers of compute units, so where none does, each width adds the first slice and whole, both ending inside a tile, that meet it on this device.
    for (const std::array<size_t, 2> base : {std::array<size_t, 2>{512, 70}, {4096, 520}}) {
        if (std::any_of(shapes.begin(), shapes.end(), meets)) break;
        size_t tall = 2;
        while (tall <= 8192 && !tall_of(tall * 128 - 28, base[0], base[1])) ++tall;
        for (size_t ms = 1; ms < tall && tall <= 8192 && !meets(shapes.back()); ++ms)
            for (size_t m = tall; m < tall + 64 && !meets(shapes.back()); ++m)
                if (meets({base[0], base[1], m * 128 - 28, ms * 128 - 28})) shapes.push_back({base[0], base[1], m * 128 - 28, ms * 128 - 28});
    }
    bool tall_met = false;
    for (const std::array<size_t, 4>& shape : shapes) {
        const size_t tw = shape[0], tr = shape[1], tn = shape[2], slice = shape[3];
        if (kper_of(tn, tw, tr) != kper_of(slice, tw, tr)) continue;
        tall_met = tall_met || tall_of(tn, tw, tr) != tall_of(slice, tw, tr);
        const std::vector<uint8_t> mt = q8_matrix(tw, tn, 91);
        const std::vector<float> xt = row_classes::values(tw * tr, 92);
        const size_t row_bytes = tw / 32 * 34;
        auto mtb = g.adopt(mt.data(), mt.size()), xtb = g.adopt(xt.data(), xt.size() * 4), yt = g.alloc(tn * tr * 4), ys = g.alloc(slice * tr * 4);
        const RowRun run[1] = {{tr, tr}};
        launches(g);
        g.matmul({quant::GGML_TYPE_Q8_0, {mtb.get(), 0}}, {xtb.get(), 0}, {yt.get(), 0}, tw, tn, tr, {run, 1}, dt);
        // The profile's rule, not the kernel table, says which build runs, so a class that should take the tall build fails here without it.
        const std::string tall_build = dt == Dtype::int8 ? "llmx_q8_0_tile8_tall" : "llmx_q8_0_tile16_tall";
        require((launches(g).count(tall_build) > 0) == tall_of(tn, tw, tr), form + ": a " + std::to_string(tn) + "-output matrix " + std::to_string(tw) + " wide takes " + tall_build + " exactly where the profile's rule sets it");
        const std::vector<float> whole = read(g, *yt, tn * tr);
        bool same = true;
        for (size_t first : {size_t(0), tn - slice}) {
            const std::vector<uint8_t> part(mt.begin() + first * row_bytes, mt.begin() + (first + slice) * row_bytes);
            auto pb = g.adopt(part.data(), part.size());
            g.matmul({quant::GGML_TYPE_Q8_0, {pb.get(), 0}}, {xtb.get(), 0}, {ys.get(), 0}, tw, slice, tr, {run, 1}, dt);
            const std::vector<float> y = read(g, *ys, slice * tr);
            for (size_t r = 0; r < tr; ++r) same = same && std::memcmp(y.data() + r * slice, whole.data() + r * tn + first, slice * 4) == 0;
        }
        require(same, form + ": the tall tile's outputs are the 64-output tile's bits, at the matrix's start and at its end inside a partial tile");
    }
    require(tall_met, form + ": a shape of one inner split sets the tall tile against the 64-output tile on this device");
    // The twin's layout: a prompt's rows block-major, padded to an odd multiple of 4 rows, give the bits of the same rows laid row after row, which a call of more regions than the layout holds takes.
    // Row counts either side of each padding step (4, 12, 20) and of a 64-row tile, final tiles cut short, the tall build at 261 and 520 rows, and runs of both kernels side by side, a prompt's region after a row kernel's.
    for (const size_t lw : {size_t(512), size_t(4096)}) {
        const size_t lo = lw == 512 ? 200 : 5860;
        const std::vector<uint8_t> ml = q8_matrix(lw, lo, 95);
        auto mlb = g.adopt(ml.data(), ml.size());
        const WeightView wl{quant::GGML_TYPE_Q8_0, {mlb.get(), 0}};
        const std::vector<size_t> counts = lw == 512 ? std::vector<size_t>{1, 4, 5, 12, 13, 20, 21, 63, 64, 65, 70, 129} : std::vector<size_t>{261, 520};
        bool flat = true, mixed = true;
        for (const size_t n : counts) {
            // xa: the prompt's n rows and 8 more; xo: 3 generated rows, 2 more and a prompt of 5; xm: xo's first 3, the prompt's n, then xo's other 7.
            const std::vector<float> xa = row_classes::values(lw * (n + 8), 96 + (uint32_t)n), xo = row_classes::values(lw * 10, 97);
            std::vector<float> xm(xo.begin(), xo.begin() + 3 * lw);
            xm.insert(xm.end(), xa.begin(), xa.begin() + n * lw);
            xm.insert(xm.end(), xo.begin() + 3 * lw, xo.end());
            auto xab = g.adopt(xa.data(), xa.size() * 4), xob = g.adopt(xo.data(), xo.size() * 4), xmb = g.adopt(xm.data(), xm.size() * 4);
            auto ya = g.alloc(lo * (n + 8) * 4), ym = g.alloc(lo * (n + 10) * 4), yo = g.alloc(lo * 5 * 4);
            const RowRun alone[1] = {{n, n}};
            g.matmul(wl, {xab.get(), 0}, {ya.get(), 0}, lw, lo, n, {alone, 1}, dt);
            const std::vector<float> a = read(g, *ya, lo * n);
            // Nine regions: the prompt, then single rows of the row kernel and the tile in turn.
            std::vector<RowRun> nine = {{n, n}};
            for (size_t i = 1; i <= 8; ++i) nine.push_back({n + i, i % 2 ? size_t(1) : size_t(3)});
            g.matmul(wl, {xab.get(), 0}, {ya.get(), 0}, lw, lo, n + 8, {nine.data(), nine.size()}, dt);
            flat = flat && std::memcmp(read(g, *ya, lo * n).data(), a.data(), lo * n * 4) == 0;
            const RowRun four_runs[4] = {{3, 1}, {3 + n, n}, {5 + n, 1}, {10 + n, 7}};
            g.matmul(wl, {xmb.get(), 0}, {ym.get(), 0}, lw, lo, n + 10, {four_runs, 4}, dt);
            const std::vector<float> m4 = read(g, *ym, lo * (n + 10));
            mixed = mixed && std::memcmp(m4.data() + 3 * lo, a.data(), lo * n * 4) == 0;
            // Two prompts side by side, which share a tile call where their splits match, give each prompt's bits alone.
            const RowRun two[2] = {{n, n}, {n + 8, 8}};
            g.matmul(wl, {xab.get(), 0}, {ya.get(), 0}, lw, lo, n + 8, {two, 2}, dt);
            const std::vector<float> a2 = read(g, *ya, lo * (n + 8));
            const RowRun eight[1] = {{8, 8}};
            g.matmul(wl, {xab.get(), n * lw}, {ym.get(), 0}, lw, lo, 8, {eight, 1}, dt);
            mixed = mixed && std::memcmp(a2.data(), a.data(), lo * n * 4) == 0 && std::memcmp(a2.data() + n * lo, read(g, *ym, lo * 8).data(), lo * 8 * 4) == 0;
            // Each other run alone gives its rows' bits.
            struct Part { size_t row, rows, extent, at; };
            for (const Part p : {Part{0, 3, 1, 0}, Part{3, 2, 1, 3 + n}, Part{5, 5, 7, 5 + n}}) {
                const RowRun run[1] = {{p.rows, p.extent}};
                g.matmul(wl, {xob.get(), p.row * lw}, {yo.get(), 0}, lw, lo, p.rows, {run, 1}, dt);
                mixed = mixed && std::memcmp(read(g, *yo, lo * p.rows).data(), m4.data() + p.at * lo, lo * p.rows * 4) == 0;
            }
        }
        require(flat, form + ": a prompt's rows read block-major give the bits of the same rows read row after row, rows " + std::to_string(lw) + " wide");
        require(mixed, form + ": runs of the row kernel and the tile side by side give each run's bits alone, rows " + std::to_string(lw) + " wide");
    }
    // Every pair of extents of one class: the same bits through a Q8_0 product in this class.
    std::vector<std::vector<float>> out;
    const size_t four = 4;
    for (size_t e : row_classes::kExtents) {
        const RowRun run[1] = {{four, e}};
        g.matmul({quant::GGML_TYPE_Q8_0, {mb.get(), 0}}, {xsb.get(), 0}, {ys.get(), 0}, width, nout, four, {run, 1}, dt);
        out.push_back(read(g, *ys, nout * four));
    }
    require(row_classes::same_within_classes(g, out, "a Q8_0 product, " + form) > 0, "the row-class check compared pairs, " + form);
}


// delta_step of a linear-attention layer against the base calls (Backend::delta_step), bit for bit in every slot both write but the conv, both state slots and the output product.
// It covers the tensor-width shard shapes and other head widths, 1 to 8 rows a view over histories of 0, 1, 2 and 100, one view and three, f16 and int8, F32 and Q8_0 alpha and beta, generated rows, and which launches ran.
inline void delta_step_bits(Backend& g, const DeviceCaps& caps, const Launches& launches, const Require& require) {
    using namespace backend;
    const auto same = [](const std::vector<float>& a, const std::vector<float>& b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0; };
    const auto throws = [](const std::function<void()>& f) {
        try {
            f();
        } catch (const std::exception&) {
            return true;
        }
        return false;
    };
    struct Shape {
        size_t hk, hv, kd, vd, nin;
    };
    const Shape shapes[] = {{16, 48, 128, 128, 2560}, {8, 24, 128, 128, 2560}, {4, 12, 128, 128, 640}, {2, 6, 128, 128, 640}, {4, 4, 64, 64, 512},
                            {2, 6, 128, 64, 512},     {2, 4, 96, 128, 512},     {3, 6, 4, 4, 256},    {1, 2, 36, 8, 256},   {2, 4, 132, 64, 256},
                            {2, 4, 64, 130, 256},     {2, 4, 64, 66, 256}};
    // Each case is its views' (rows, history) pairs; the last view of a case of several reads slot 0 and writes slot 2, as a verify does.
    const std::vector<std::vector<std::pair<size_t, size_t>>> cases = {{{1, 100}}, {{1, 0}}, {{2, 1}}, {{3, 2}}, {{4, 100}}, {{5, 0}}, {{8, 100}}, {{9, 100}},
                                                                     {{1, 100}, {8, 2}, {3, 0}}, {{6, 1}, {7, 100}, {2, 2}}, {{1, 0}, {1, 0}, {1, 0}}};
    for (const Shape& sp : shapes) {
        const StateShape sh{sp.hk, sp.hv, sp.kd, sp.vd};
        const size_t C = sh.channels(), V = sp.hv * sp.vd, E = sp.nin, slots = 6;
        const bool width = sp.kd <= 128 && sp.kd % 4 == 0 && sp.vd <= 128 && sp.vd % 4 == 0;
        // The base refuses K heads past 128 itself, so a step at that width is refused either way.
        const bool base_runs = sp.kd <= 128;
        const std::vector<uint8_t> wqkv = q8_matrix(E, C, 301), wgate = q8_matrix(E, V, 302), wout = q8_matrix(V % 32 ? 32 : V, 64, 303);
        const std::vector<float> wa = row_classes::values(sp.hv * E, 304, 0.02f), wb = row_classes::values(sp.hv * E, 305, 0.02f);
        const std::vector<uint8_t> wa8 = q8_matrix(E, sp.hv, 312), wb8 = q8_matrix(E, sp.hv, 313);
        std::vector<float> a = row_classes::values(sp.hv, 306, 1.0f), dt = row_classes::values(sp.hv, 307, 0.5f), nw = row_classes::values(sp.vd, 308, 1.0f),
                           cw = row_classes::values(C * kConvTaps, 309, 0.5f), state = row_classes::values(slots * sh.slot_floats(), 310, 0.1f);
        for (float& v : a) v = -std::fabs(v) - 0.05f;
        auto qb = g.adopt(wqkv.data(), wqkv.size()), gb = g.adopt(wgate.data(), wgate.size()), ob = g.adopt(wout.data(), wout.size());
        auto ab = g.adopt(wa.data(), wa.size() * 4), bb = g.adopt(wb.data(), wb.size() * 4), ab8 = g.adopt(wa8.data(), wa8.size()), bb8 = g.adopt(wb8.data(), wb8.size()), ap = g.adopt(a.data(), a.size() * 4), db = g.adopt(dt.data(), dt.size() * 4),
             nb = g.adopt(nw.data(), nw.size() * 4), cb = g.adopt(cw.data(), cw.size() * 4);
        const std::string shape = std::to_string(sp.hv) + "/" + std::to_string(sp.hk) + " heads " + std::to_string(sp.kd) + "x" + std::to_string(sp.vd) + ", " + std::to_string(E) + " wide";
        for (const auto& views_at : cases)
            for (const NormGate ng : {NormGate::sigmoid, NormGate::silu})
            for (Dtype dt_ : {Dtype::f32, Dtype::f16, Dtype::int8})
                for (int form = 0; form < 3; ++form) {
                const bool q8 = form > 0, gen = form == 2;
                size_t rows = 0;
                for (const auto& [nq, len] : views_at) rows += nq;
                std::string at = shape + ", rows";
                for (const auto& [nq, len] : views_at) at += " " + std::to_string(nq) + "@" + std::to_string(len);
                at += std::string(", ") + dtype_name(dt_) + (q8 ? gen ? ", Q8_0 alpha as generated rows" : ", Q8_0 alpha" : "") + (ng == NormGate::silu ? ", SiLU gate" : ", sigmoid gate");
                const std::vector<float> mixed = row_classes::values(rows * E, 311 + (uint32_t)rows, 1.0f);
                auto mb = g.adopt(mixed.data(), mixed.size() * 4);
                struct Arm {
                    std::unique_ptr<StateStorage> st;
                    BufferPtr raw, conv, gate, ab, out, y;
                };
                Arm arm[2];
                std::vector<StateView> views[2];
                DeltaStep step[2];
                for (int k = 0; k < 2; ++k) {
                    arm[k] = Arm{g.state_alloc(1, slots, sh), g.alloc(rows * C * 4), g.alloc(rows * C * 4), g.alloc(rows * V * 4), g.alloc(2 * rows * sp.hv * 4),
                                 g.alloc(rows * V * 4), g.alloc(rows * 64 * 4)};
                    g.write(arm[k].st->layer(0), 0, state.data(), state.size() * 4);
                    size_t v = 0;
                    for (const auto& [nq, len] : views_at) {
                        const bool verify = views_at.size() > 1 && v + 1 == views_at.size();
                        views[k].push_back(StateView{arm[k].st.get(), verify ? 0 : v + 1, verify ? 5 : v + 1, len, nq});
                        ++v;
                    }
                    const Slice alpha{arm[k].ab.get(), 0}, beta{arm[k].ab.get(), rows * sp.hv};
                    step[k] = DeltaStep{Projection{{quant::GGML_TYPE_Q8_0, {qb.get(), 0}}, {arm[k].raw.get(), 0}, C},
                                        Projection{{quant::GGML_TYPE_Q8_0, {gb.get(), 0}}, {arm[k].gate.get(), 0}, V},
                                        Projection{{q8 ? quant::GGML_TYPE_Q8_0 : quant::GGML_TYPE_F32, {q8 ? ab8.get() : ab.get(), 0}}, alpha, sp.hv},
                                        Projection{{q8 ? quant::GGML_TYPE_Q8_0 : quant::GGML_TYPE_F32, {q8 ? bb8.get() : bb.get(), 0}}, beta, sp.hv},
                                        {cb.get(), 0}, {ap.get(), 0}, {db.get(), 0}, {nb.get(), 0}, {arm[k].conv.get(), 0}, {arm[k].out.get(), 0}, 1e-6f, ng};
                }
                // A run a view, its extent the view's history and rows, as a decode step or a verify gives them.
                std::vector<RowRun> runs;
                for (const auto& [nq, len] : views_at) runs.push_back(RowRun{(runs.empty() ? 0 : runs.back().end) + nq, len + nq});
                if (gen) {
                    runs.clear();
                    for (size_t r = 0; r < rows; ++r) runs.push_back(RowRun{r + 1, 1});
                }
                const RowRuns rr{runs.data(), runs.size()};
                const auto call = [&](int k) {
                    if (k == 0) g.Backend::delta_step(step[0], {mb.get(), 0}, E, rows, 0, views[0].data(), views[0].size(), rr, dt_);
                    else g.delta_step(step[1], {mb.get(), 0}, E, rows, 0, views[1].data(), views[1].size(), rr, dt_);
                    if (V % 32 == 0) g.matmul({quant::GGML_TYPE_Q8_0, {ob.get(), 0}}, {arm[k].out.get(), 0}, {arm[k].y.get(), 0}, V, 64, rows, rr, dt_);
                };
                launches(g);
                if (!base_runs) {
                    require(throws([&] { call(1); }), "a step past the base's K heads is refused, " + at);
                    launches(g);
                    continue;
                }
                call(0);
                const auto base = launches(g);
                call(1);
                const auto fused = launches(g);
                bool holds = width && views_at.size() <= 64;
                for (const auto& [nq, len] : views_at) holds = holds && nq <= 8;
                // Q8_0 alpha and beta are summed in the fused step in the F32 class's exact order, or in the int8 class's row order where every run is one generated row, as one-row views at history 0 are too.
                bool one_row_runs = true;
                for (const RowRun& r : runs) one_row_runs = one_row_runs && r.extent == 1;
                if (q8) holds = holds && (dt_ == Dtype::f32 || (dt_ == Dtype::int8 && E / 32 % 2 == 0 && one_row_runs));
                // Views of several rows take the fused step only where its blocks, one a (view, V head), fill half the compute units.
                if (rows > views_at.size() && 2 * views_at.size() * sp.hv < caps.compute_units) holds = false;
                require(!base.count("llmx_delta_step"), "the base's delta_step launches no fused kernel, " + at);
                require(fused.count("llmx_delta_step") == (holds ? 1u : 0u), std::string(holds ? "delta_step is one fused launch, " : "a step outside the limits runs the base's ops, ") + at);
                if (holds) require(!fused.count("llmx_gated_delta_rule_short") && !fused.count("llmx_causal_conv_silu") && !fused.count("llmx_gated_rms_norm"), "the fused step runs no chain op, " + at);
                require(same(read(g, *arm[0].raw, rows * C), read(g, *arm[1].raw, rows * C)), "the qkv rows are the base's, " + at);
                require(same(read(g, *arm[0].gate, rows * V), read(g, *arm[1].gate, rows * V)), "the gate rows are the base's, " + at);
                require(same(read(g, *arm[0].ab, 2 * rows * sp.hv), read(g, *arm[1].ab, 2 * rows * sp.hv)), "alpha and beta are the base's, " + at);
                require(same(read(g, *arm[0].out, rows * V), read(g, *arm[1].out, rows * V)), "the output is the base's, " + at);
                require(same(read(g, *arm[0].y, rows * 64), read(g, *arm[1].y, rows * 64)), "the product of the output is the base's, " + at);
                require(same(read(g, arm[0].st->layer(0), state.size()), read(g, arm[1].st->layer(0), state.size())), "every state slot is the base's, " + at);
            }
        // A step of BF16 alpha and beta matrices, or of 65 views, runs the base's ops.
        if (!width) continue;
        const size_t rows = 65;
        const std::vector<float> mixed = row_classes::values(rows * E, 399, 1.0f);
        auto mb = g.adopt(mixed.data(), mixed.size() * 4);
        auto st = g.state_alloc(1, 2 * rows, sh);
        auto raw = g.alloc(rows * C * 4), conv = g.alloc(rows * C * 4), gate = g.alloc(rows * V * 4), abo = g.alloc(2 * rows * sp.hv * 4), out = g.alloc(rows * V * 4);
        std::vector<uint16_t> wa16(wa.size());
        for (size_t i = 0; i < wa.size(); ++i) wa16[i] = f32_to_bf16(wa[i]);
        auto ab16 = g.adopt(wa16.data(), wa16.size() * 2);
        std::vector<StateView> many;
        std::vector<RowRun> each;
        for (size_t v = 0; v < rows; ++v) many.push_back(StateView{st.get(), v, v, 3, 1}), each.push_back(RowRun{v + 1, 4});
        for (const bool f16 : {false, true}) {
            const size_t n = f16 ? 1 : rows;
            const DeltaStep d{Projection{{quant::GGML_TYPE_Q8_0, {qb.get(), 0}}, {raw.get(), 0}, C}, Projection{{quant::GGML_TYPE_Q8_0, {gb.get(), 0}}, {gate.get(), 0}, V},
                              Projection{{f16 ? quant::GGML_TYPE_BF16 : quant::GGML_TYPE_F32, {f16 ? ab16.get() : ab.get(), 0}}, {abo.get(), 0}, sp.hv},
                              Projection{{quant::GGML_TYPE_F32, {bb.get(), 0}}, {abo.get(), n * sp.hv}, sp.hv},
                              {cb.get(), 0}, {ap.get(), 0}, {db.get(), 0}, {nb.get(), 0}, {conv.get(), 0}, {out.get(), 0}, 1e-6f, NormGate::silu};
            launches(g);
            g.delta_step(d, {mb.get(), 0}, E, n, 0, many.data(), n, RowRuns{each.data(), n}, Dtype::int8);
            require(!launches(g).count("llmx_delta_step"), std::string(f16 ? "a step of BF16 alpha runs the base's ops, " : "a step of 65 views runs the base's ops, ") + shape);
        }
    }
}

// The witnesses that read the device's caps and profile: the F16 half tile, the Q8_0 and MXFP4 decode rows against the host's order, and a decode call's columns each its column alone.
// A test passes its device's caps and profile, its launches reader and its require.
struct Bits {
    DeviceCaps caps;
    std::function<DeviceProfile&(Backend&)> profile_of;
    Launches launches;
    Require require;

    void exact_close(const std::vector<float>& got, const DoubleSums& d, const std::string& what) const { require(twin_checks::exact_close(got, d), what); }

    // F16 weights' prompt tile (half_product): outputs within the F32 bound of the double sum over binary16-rounded inputs, subnormal and signed-zero weights and inputs among them, past every tile edge.
    // A tile row has the bits of the same row in a call of another row count, added outputs the plain ones added, and decode columns in a call the exact product's bits.
    void half_tile(Backend& g) {
        const size_t R = 130;
        for (size_t nin : {size_t(32), size_t(96), size_t(2560)})
            for (size_t nout : {size_t(1), size_t(63), size_t(64), size_t(65), size_t(130)}) {
                const std::string shape = "F16 " + std::to_string(nin) + "->" + std::to_string(nout);
                std::vector<uint8_t> w(nin * nout * 2);
                std::vector<float> wf(nin * nout);
                const std::vector<float> wv = row_classes::values(nin * nout, uint32_t(nin + nout), 0.05f);
                for (size_t i = 0; i < wf.size(); ++i) {
                    uint16_t h = f32_to_f16(wv[i]);
                    if (i % 97 == 3) h = uint16_t(1 + i % 1023);   // subnormal weights
                    if (i % 89 == 5) h = i % 2 ? 0x8000 : 0x0000;
                    w[2 * i] = uint8_t(h), w[2 * i + 1] = uint8_t(h >> 8);
                    wf[i] = f16_to_f32(h);
                }
                std::vector<float> x = row_classes::values(nin * R, uint32_t(nin + 3 * nout));
                for (size_t i = 0; i < x.size(); i += 53) x[i] = (i % 2 ? -3e-6f : 4e-7f) * float(1 + i % 5);   // inputs rounding to binary16 subnormals
                std::vector<float> xh(x.size());
                for (size_t i = 0; i < x.size(); ++i) xh[i] = f16_to_f32(f32_to_f16(x[i]));
                const DoubleSums want = double_sums(wf, xh, nin, nout);
                auto wb = g.adopt(w.data(), w.size()), xb = g.adopt(x.data(), x.size() * 4), yb = g.alloc(nout * R * 4), y1 = g.alloc(nout * R * 4);
                const WeightView wt{quant::GGML_TYPE_F16, {wb.get(), 0}};
                for (size_t rows : {size_t(2), size_t(63), size_t(64), size_t(65), size_t(R - 3)}) {
                    const std::string at = shape + " over " + std::to_string(rows) + " rows";
                    launches(g);
                    g.matmul(wt, {xb.get(), 0}, {yb.get(), 0}, nin, nout, rows, {}, Dtype::f16);
                    require(launches(g).count("llmx_f16_tile") == 1, "a " + at + " prompt takes the half tile");
                    const std::vector<float> all = read(g, *yb, nout * rows);
                    const DoubleSums part{std::vector<double>(want.sum.begin(), want.sum.begin() + std::ptrdiff_t(nout * rows)),
                                          std::vector<double>(want.mag.begin(), want.mag.begin() + std::ptrdiff_t(nout * rows))};
                    exact_close(all, part, "a " + at + " half tile against the double sum on binary16 inputs");
                    g.matmul(wt, {xb.get(), 0}, {y1.get(), 0}, nin, nout, rows + 1, {}, Dtype::f16);
                    require(std::memcmp(read(g, *y1, nout * rows).data(), all.data(), nout * rows * 4) == 0, "a " + at + " half tile row keeps its bits in a call of one row more");
                    const std::vector<float> y0 = row_classes::values(nout * rows, uint32_t(rows + nout));
                    std::vector<float> sum(y0.size());
                    for (size_t i = 0; i < y0.size(); ++i) sum[i] = y0[i] + all[i];
                    g.write(*y1, 0, y0.data(), y0.size() * 4);
                    g.matmul_add(wt, {xb.get(), 0}, {y1.get(), 0}, nin, nout, rows, {}, Dtype::f16);
                    require(std::memcmp(read(g, *y1, nout * rows).data(), sum.data(), nout * rows * 4) == 0, "a " + at + " added half tile adds the plain outputs");
                    // Three decode columns, then a prompt: the columns take the exact product's bits, the prompt the half tile's.
                    const RowRun runs[2] = {{3, 1}, {rows + 3, 64}};
                    const size_t n = rows + 3;
                    launches(g);
                    g.matmul(wt, {xb.get(), 0}, {yb.get(), 0}, nin, nout, n, {runs, 2}, Dtype::f16);
                    const auto ran = launches(g);
                    require(ran.count("llmx_f16_tile") == 1 && ran.count("llmx_matmul_rows_f16") == 1, "a " + at + " call of decode columns and a prompt takes the exact rows and the half tile");
                    const std::vector<float> mixed = read(g, *yb, nout * n);
                    g.matmul(wt, {xb.get(), 0}, {y1.get(), 0}, nin, nout, 3, {}, Dtype::f32);
                    require(std::memcmp(read(g, *y1, nout * 3).data(), mixed.data(), nout * 3 * 4) == 0, "a " + at + " call's decode columns have the exact product's bits");
                    g.matmul(wt, {xb.get(), 3 * nin}, {y1.get(), 0}, nin, nout, rows, {}, Dtype::f16);
                    require(std::memcmp(read(g, *y1, nout * rows).data(), mixed.data() + 3 * nout, nout * rows * 4) == 0, "a " + at + " call's prompt rows have the half tile's bits");
                }
            }
    }

    // The one-column Q8_0 decode row of one wave of W lanes, on the host, in the kernels' order, then the xor tree from W / 2.
    // The half-block layout (int8 twin, an even block count) takes two lanes a block, each lane's half-block dots scaled in block order, and the quarter layout four lanes a block.
    float q8_row_host(const uint8_t* row, const std::vector<float>& d, const std::vector<int16_t>& q, size_t blocks, bool half, unsigned W) {
        std::vector<float> acc(W, 0.0f);
        const size_t parts = half ? 2 : 4, per = 32 / parts;
        for (size_t u = 0; u < blocks * parts; ++u) {
            const size_t b = u / parts, p = u % parts;
            const uint8_t* block = row + b * 34;
            int s = 0;
            for (size_t j = p * per; j < (p + 1) * per; ++j) s += int(int8_t(block[2 + j])) * q[b * 32 + j];
            const float dw = f16_to_f32(uint16_t(block[0] | block[1] << 8)), dd = dw * d[b];
            float& a = acc[half ? (b % (W / 2)) * 2 + p : u % W];
            a = std::fmaf(dd, float(s), a);
        }
        for (unsigned m = W / 2; m; m /= 2) {
            std::vector<float> next(W);
            for (unsigned l = 0; l < W; ++l) next[l] = acc[l] + acc[l ^ m];
            acc = next;
        }
        return acc[0];
    }

    // Each one-column Q8_0 decode build against the host's one-row build bit for bit, the paired build (two rows a wave, rows of at most 16 blocks) among them, whose idle half the one-row tree adds as +0.
    // The rows pin the zero's sign: all-zero weights, a negative scale against a zero integer dot (dw * xd = -0), and lanes whose partials cancel to an exact zero from nonzero terms.
    void q8_decode_bits(Backend& g) {
        const unsigned W = caps.subgroup_size;
        const size_t nout = 33;
        DeviceProfile& profile = profile_of(g);
        const uint32_t keep = profile.q8_decode_two_rows_from;
        const size_t four = profile.q8_decode_four_steps_from;
        // Row widths reach both edges of the step rule (63 to 65 and 254 to 257 blocks), both edges of the held rule (32 to 34 and 160 to 162) and step tails, and output counts both edges of the two-row rule at both profiles' values.
        for (size_t blocks0 : {size_t(8), size_t(16), size_t(18), size_t(20), size_t(24), size_t(32), size_t(33), size_t(34), size_t(40), size_t(63), size_t(64), size_t(65), size_t(80), size_t(96),
                               size_t(128), size_t(136), size_t(160), size_t(162), size_t(191), size_t(192), size_t(193), size_t(254), size_t(255), size_t(256), size_t(257)}) {
            const size_t nin = blocks0 * 32;
            const size_t blocks = nin / 32;
            std::vector<uint8_t> w = q8_matrix(nin, nout, uint32_t(nin + 5));
            const auto set_scale = [&](size_t o, size_t b, float s) {
                const uint16_t h = f32_to_f16(s);
                w[(o * blocks + b) * 34] = uint8_t(h), w[(o * blocks + b) * 34 + 1] = uint8_t(h >> 8);
            };
            for (size_t b = 0; b < blocks; ++b) {
                std::memset(&w[(0 * blocks + b) * 34], 0, 34);          // row 0: all zero
                std::memset(&w[(1 * blocks + b) * 34 + 2], 0, 32);      // row 1: values zero under negative scales
                set_scale(1, b, -0.25f - 0.01f * float(b));
                if (b % 2) {
                    // Row 2: each odd block the even block before it negated, so on a block-periodic twin a lane's partial cancels its partner's exactly.
                    std::memcpy(&w[(2 * blocks + b) * 34], &w[(2 * blocks + b - 1) * 34], 34);
                    for (size_t j = 0; j < 32; ++j) w[(2 * blocks + b) * 34 + 2 + j] = uint8_t(-int8_t(w[(2 * blocks + b) * 34 + 2 + j]));
                }
            }
            const std::vector<float> rnd = row_classes::values(nin, uint32_t(nin + 6));
            std::vector<float> periodic(nin);
            for (size_t i = 0; i < nin; ++i) periodic[i] = rnd[i % 32];
            // Rows past the special ones repeat the first 33, so a call of many outputs holds them too.
            const size_t nmax = 4097;
            const std::vector<uint8_t> first = w;
            for (size_t o = nout; o < nmax; ++o) w.insert(w.end(), first.begin() + (o % nout) * blocks * 34, first.begin() + (o % nout + 1) * blocks * 34);
            auto wb = g.adopt(w.data(), w.size()), yb = g.alloc(nmax * 4);
            const std::vector<float>* inputs[] = {&rnd, &periodic};
            for (const std::vector<float>* xv : inputs) {
                auto xb = g.adopt(xv->data(), nin * 4);
                for (Dtype dt : {Dtype::f16, Dtype::int8}) {
                    const bool half = dt == Dtype::int8 && blocks % 2 == 0;
                    const float L = dt == Dtype::int8 ? 127.0f : 32767.0f;
                    std::vector<float> d(blocks);
                    std::vector<int16_t> q(nin);
                    for (size_t b = 0; b < blocks; ++b) twin_rule(xv->data() + b * 32, d[b], q.data() + b * 32, L);
                    for (const auto& [from, n] : std::vector<std::pair<uint32_t, size_t>>{{2560, 1}, {2560, 2}, {2560, nout}, {2560, 2559}, {2560, 2560}, {2560, 4097}, {4096, 4095}, {4096, 4096}}) {
                        if (dt == Dtype::f16 && from == 4096) continue;
                        const RowRun rr{1, 1};
                        profile.q8_decode_two_rows_from = from;
                        launches(g);
                        g.matmul({quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, n, 1, {&rr, 1}, dt);
                        profile.q8_decode_two_rows_from = keep;
                        const std::vector<float> y = read(g, *yb, n);
                        std::string ran;
                        for (const auto& [k, v] : launches(g))
                            if (k.rfind("llmx_q8_0", 0) == 0) ran += k;
                        // A row of two to five spans of blocks takes one row a wave with all its steps loaded at once, whatever the outputs.
                        // Other half-block rows run one row a wave for few outputs and two for many, paired up to 16 blocks a row, and under two spans one step a load, two rows a wave from the profile's four-step width four.
                        const bool narrow = blocks < W;
                        const size_t spans = (blocks + W / 2 - 1) / (W / 2);
                        const std::string want = !half                   ? "llmx_q8_0_rows" + std::string(dt == Dtype::int8 ? "8" : "16") + "_1"
                                                 : W == 64 && blocks <= 16 ? "llmx_q8_0_half8_1p"
                                                 : spans >= 2 && spans <= 5 ? "llmx_q8_0_half8_1h" + std::to_string(spans)
                                                 : n < from              ? (narrow ? "llmx_q8_0_half8_1s1" : "llmx_q8_0_half8_1")
                                                 : narrow                ? "llmx_q8_0_half8_1r2s1"
                                                 : blocks >= four        ? "llmx_q8_0_half8_1r2s4"
                                                                         : "llmx_q8_0_half8_1r2";
                        const std::string what = "a Q8_0 decode row " + std::to_string(nin) + " wide, " + std::to_string(n) + " outputs from " + std::to_string(from) + ", " + dtype_name(dt) +
                                                 (xv == &periodic ? ", periodic twin" : "");
                        require(ran == want, what + " runs " + want + " (ran " + ran + ")");
                        bool ok = true;
                        for (size_t o = 0; o < n; ++o) {
                            const float h = q8_row_host(w.data() + o * blocks * 34, d, q, blocks, half, W);
                            ok = ok && std::memcmp(&h, &y[o], 4) == 0;
                        }
                        require(ok, what + ": every row the one-row build's bits");
                        bool zero = true;
                        for (size_t o = 0; o < std::min<size_t>(n, 2); ++o) zero = zero && y[o] == 0.0f && !std::signbit(y[o]);
                        require(zero, what + ": the zero rows are +0");
                        if (xv == &periodic && n > 2 && blocks % 2 == 0) require(y[2] == 0.0f && !std::signbit(y[2]), what + ": the cancelling row is +0");
                    }
                }
            }
        }
    }

    // One MXFP4 decode row on the host in the order every MXFP4 row kernel keeps: four units a block of eight values, unit u's terms chained in accumulator u % W.
    // W is the power of two covering the units, at most the wave, and the accumulators are summed by the xor tree from W / 2.
    float mxfp4_row_host(const uint8_t* row, const std::vector<float>& d, const int16_t* q, size_t blocks, unsigned wave) {
        unsigned W = 1;
        while (W < wave && W < blocks * 4) W *= 2;
        std::vector<float> acc(W, 0.0f);
        for (size_t u = 0; u < blocks * 4; ++u) {
            const size_t b = u / 4, t = u % 4;
            const uint8_t* block = row + b * 17;
            int s = 0;
            for (size_t j = 8 * t; j < 8 * t + 8; ++j) s += quant::FP4_VALUES[j < 16 ? block[1 + j] & 15 : block[1 + j - 16] >> 4] * q[b * 32 + j];
            const uint32_t bits = block[0] <= 1 ? 0x00200000u << block[0] : uint32_t(block[0] - 1) << 23;
            float ds;
            std::memcpy(&ds, &bits, 4);
            acc[u % W] = std::fmaf(d[b], ds * float(s), acc[u % W]);
        }
        for (unsigned m = W / 2; m; m /= 2) {
            std::vector<float> next(W);
            for (unsigned l = 0; l < W; ++l) next[l] = acc[l] + acc[l ^ m];
            acc = next;
        }
        return acc[0];
    }

    // A Q8_0 int8 decode call of 2 to 16 columns against each column computed alone, bit for bit, by the build it names, over rows of 1024 values (one block step a lane) and 2560 (several).
    // The build each column count and output count takes is listed in docs/TESTS.md (backend-rocm).
    void q8_columns_bits_at(Backend& g, size_t nin) {
        const size_t edge = size_t(4) * profile_of(g).row_four_waves_per_cu * caps.compute_units;
        const uint32_t waves = profile_of(g).row_four_waves_per_cu;
        // The rule is rows_a_cluster at a whole subgroup and a cap of four.
        require(rows_a_cluster(caps, edge - 1, caps.subgroup_size, 4, waves) < 4 && rows_a_cluster(caps, edge, caps.subgroup_size, 4, waves) == 4,
                "rows_a_cluster at a whole subgroup takes four rows from row_build's edge on and fewer one row below it");
        const size_t big = std::max<size_t>(edge, 4097);
        const std::vector<uint8_t> w = q8_matrix(nin, big, 27);
        auto wb = g.adopt(w.data(), w.size());
        const std::vector<float> x = row_classes::values(nin * 16, 28);
        std::vector<BufferPtr> xs;
        for (size_t c = 0; c < 16; ++c) xs.push_back(g.adopt(x.data() + c * nin, nin * 4));
        // The output past a call's columns holds a sentinel, which the build's padding columns must not overwrite.
        const std::vector<float> fill(big * 16, -12345.0f);
        auto xb = g.adopt(x.data(), x.size() * 4), yb = g.alloc(big * 16 * 4);
        for (const size_t n : {size_t(33), edge - 1, edge, big})
            for (size_t cols = 2; cols <= 16; ++cols) {
                if (cols > 8 && cols < 16 && n != big) continue;
                const RowRun rr{cols, 1};
                g.write(*yb, 0, fill.data(), n * 16 * 4);
                launches(g);
                g.matmul({quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, n, cols, {&rr, 1}, Dtype::int8);
                std::string ran;
                for (const auto& [k, v] : launches(g))
                    if (k.rfind("llmx_q8_0", 0) == 0) ran += k;
                const std::vector<float> y = read(g, *yb, n * cols);
                const size_t build = cols <= 2 ? 2 : cols <= 4 ? 4 : cols <= 8 ? 8 : cols;
                const bool four = build == 2 || build == 4 || build == 8 || build == 16;
                const std::string want = "llmx_q8_0_half8_" + std::to_string(build) + (four && n >= edge ? "_r4" : "");
                const std::string what = "a Q8_0 int8 decode call of " + std::to_string(cols) + " columns, " + std::to_string(n) + " outputs";
                require(ran == want, what + " runs " + want + " (ran " + ran + ")");
                const std::vector<float> past = read(g, *yb, n * 16);
                require(std::memcmp(past.data() + n * cols, fill.data(), n * (16 - cols) * 4) == 0, what + ": the output past its columns unchanged");
                bool ok = true;
                for (size_t c = 0; c < cols; ++c) {
                    const RowRun one{1, 1};
                    g.matmul({quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, {xs[c].get(), 0}, {yb.get(), 0}, nin, n, 1, {&one, 1}, Dtype::int8);
                    const std::vector<float> y1 = read(g, *yb, n);
                    ok = ok && std::memcmp(y1.data(), y.data() + c * n, n * 4) == 0;
                }
                require(ok, what + ": every column the one-column call's bits");
            }
    }

    void q8_columns_bits(Backend& g) {
        q8_columns_bits_at(g, 1024);
        q8_columns_bits_at(g, 2560);
    }

    // Every MXFP4 decode build against the host's row bit for bit in both twins: dense rows of 1 to 16 columns, routed entries and the grouped builds of 2, 4 and 8.
    // The widths take 4 to 1040 units a row, on both sides of 32 and 64, with DeepSeek-V4.1-Flash's 2304 and 5120, a 65-block row, and the quad rows' edges: 15 to 18 and 20 blocks, 256 blocks (the staging loop's two passes) and 260.
    void mxfp4_decode_bits(Backend& g) {
        const unsigned wave = caps.subgroup_size;
        const uint32_t T = quant::GGML_TYPE_MXFP4;
        const size_t nout = 40, ne = 4, cols = 16, K = 2;
        size_t quad_runs = 0;
        for (size_t nin : {size_t(32), size_t(224), size_t(256), size_t(288), size_t(480), size_t(512), size_t(544), size_t(576), size_t(640), size_t(2080), size_t(2304), size_t(5120), size_t(8192), size_t(8320)}) {
            const size_t blocks = nin / 32, rowb = blocks * 17;
            // 64 experts of rows, so the routed calls over 64 have them all; the entries pick among the first 4.
            const std::vector<uint8_t> w = typed_matrix(T, nin, nout * 64, uint32_t(nin + 3));
            const std::vector<float> x = row_classes::values(nin * cols, uint32_t(nin + 4));
            std::vector<uint32_t> ids(cols * K);
            for (size_t c = 0; c < cols; ++c)
                for (size_t k = 0; k < K; ++k) ids[c * K + k] = uint32_t((c + 3 * k) % ne);
            const std::vector<float> wts(cols * K, 0.5f);
            auto wb = g.adopt(w.data(), w.size()), xb = g.adopt(x.data(), x.size() * 4), ib = g.adopt(ids.data(), ids.size() * 4), tb = g.adopt(wts.data(), wts.size() * 4);
            auto yb = g.alloc(nout * ne * cols * 4);
            for (Dtype dt : {Dtype::f16, Dtype::int8}) {
                const float L = dt == Dtype::int8 ? 127.0f : 32767.0f;
                std::vector<std::vector<float>> d(cols, std::vector<float>(blocks));
                std::vector<std::vector<int16_t>> q(cols, std::vector<int16_t>(nin));
                for (size_t c = 0; c < cols; ++c)
                    for (size_t b = 0; b < blocks; ++b) twin_rule(x.data() + c * nin + b * 32, d[c][b], q[c].data() + b * 32, L);
                const std::string tw = dt == Dtype::int8 ? "8" : "16", at = std::to_string(nin) + " wide, " + dtype_name(dt);
                const auto bits = [&](const std::vector<float>& y, const std::vector<std::pair<size_t, size_t>>& rc, const std::string& what) {
                    bool ok = true;
                    for (size_t i = 0; ok && i < rc.size(); ++i) {
                        const float h = mxfp4_row_host(w.data() + rc[i].first * rowb, d[rc[i].second], q[rc[i].second].data(), blocks, wave);
                        ok = std::memcmp(&h, &y[i], 4) == 0;
                        if (!ok) std::fprintf(stderr, "  output %zu: %.9g against %.9g\n", i, y[i], h);
                    }
                    require(ok, what + ": every output the host row's bits");
                };
                // A one-column 8-bit call takes MXFP4's quad rows exactly where the shared rule gives them at the 32-lane cluster a 64-lane wave gives rows of 16 blocks or more (twin_checks.hpp, quad_build).
                const bool quad = twin_checks::quad_build(quant::GGML_TYPE_MXFP4, dt, nin, wave);
                const auto ran = [&](const std::string& build, const std::string& quad_build, const std::string& what, bool one_column) {
                    const auto l = launches(g);
                    const bool takes = one_column && quad;
                    require(l.count(takes ? quad_build : build) > 0, what + " runs " + (takes ? quad_build : build));
                    require(l.count(takes ? build : quad_build) == 0, what + " does not run " + (takes ? build : quad_build));
                    quad_runs += takes;
                };
                for (size_t n : {size_t(1), size_t(2), size_t(3), size_t(5), size_t(9), size_t(16)}) {
                    const RowRun rr{n, 1};
                    launches(g);
                    g.matmul({T, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout * ne, n, {&rr, 1}, dt);
                    const std::string what = "MXFP4 decode rows " + at + ", " + std::to_string(n) + " columns";
                    ran("llmx_mxfp4_rows" + tw + "_" + std::to_string(n <= 1 ? 1 : n <= 2 ? 2 : n <= 4 ? 4 : n <= 8 ? 8 : 16), "llmx_mxfp4_quad_rows8_1", what, n == 1);
                    std::vector<std::pair<size_t, size_t>> rc;
                    for (size_t c = 0; c < n; ++c)
                        for (size_t o = 0; o < nout * ne; ++o) rc.push_back({o, c});
                    bits(read(g, *yb, nout * ne * n), rc, what);
                }
                // 64 experts keep the entries below the grouped builds' break-even; over 4 the average entries an expert pick g2, g4 and g8.
                const std::pair<size_t, size_t> cases[] = {{64, 1}, {64, 6}, {4, 1}, {4, 6}, {4, 16}};
                for (const auto& [n_expert, n] : cases) {
                    const RowRun rr{n, 1};
                    const Backend::Routing routing{{ib.get(), 0}, {tb.get(), 0}, K, n_expert};
                    launches(g);
                    g.matmul_experts({Projection{{T, {wb.get(), 0}}, {yb.get(), 0}, nout}}, {xb.get(), 0}, nin, n, routing, {&rr, 1}, dt);
                    const size_t avg = n * K / std::min(n_expert, n * K);
                    const std::string build = "llmx_mxfp4_routed" + tw + (n_expert == 64 ? "" : avg <= 2 ? "_g2" : avg <= 4 ? "_g4" : "_g8");
                    const std::string what = "MXFP4 routed decode " + at + ", " + std::to_string(n) + " columns over " + std::to_string(n_expert) + " experts";
                    ran(build, "llmx_mxfp4_quad_routed8", what, n_expert == 64);
                    std::vector<std::pair<size_t, size_t>> rc;
                    for (size_t e = 0; e < n * K; ++e)
                        for (size_t o = 0; o < nout; ++o) rc.push_back({ids[e] * nout + o, e / K});
                    bits(read(g, *yb, nout * n * K), rc, what);
                }
            }
        }
        require(wave != 64 || quad_runs > 0, "MXFP4 decode takes the quad rows on a 64-lane wave");
    }
};
// A diagnostics backend's kernel_times after an F32 product and an output head's names every kernel its launches name, the head's with " head" after it, each with positive time, and counts them all.
// A backend made without diagnostics times nothing.
inline void kernel_times(Backend& timed, Backend& plain, const Launches& launches, const Require& require) {
    const size_t nin = 1024, nout = 256, rows = 4;
    std::vector<float> w(nin * nout), x(nin * rows);
    for (size_t i = 0; i < w.size(); ++i) w[i] = float(int(i % 13) - 6) / 8.0f;
    for (size_t i = 0; i < x.size(); ++i) x[i] = float(int(i % 7) - 3) / 4.0f;
    for (Backend* b : {&timed, &plain}) {
        const BufferPtr wb = b->adopt(w.data(), w.size() * sizeof(float)), xb = b->adopt(x.data(), x.size() * sizeof(float));
        const BufferPtr yb = b->alloc(nout * rows * sizeof(float), backend::Memory::device);
        b->sync();
        (void)b->kernel_times();
        (void)launches(*b);
        std::map<std::string, size_t> ran;
        for (const bool head : {false, true}) {
            if (head) b->matmul_logits({quant::GGML_TYPE_F32, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, rows, {}, Dtype::f32);
            else b->matmul({quant::GGML_TYPE_F32, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, rows, {}, Dtype::f32);
            for (const auto& [name, count] : launches(*b)) ran[name + (head ? " head" : "")] += count;
        }
        const Backend::KernelTimes t = b->kernel_times();
        if (b == &plain) {
            require(t.kernels.empty() && t.dispatches == 0 && t.gaps.empty(), "a backend made without diagnostics times no kernels");
            continue;
        }
        size_t launched = 0;
        std::string named;
        for (const auto& [name, count] : ran) {
            launched += count;
            const auto it = std::find_if(t.kernels.begin(), t.kernels.end(), [&](const auto& k) { return k.first == name; });
            require(it != t.kernels.end() && it->second > 0.0, "kernel_times names " + name + " with positive time");
            named += " " + name;
        }
        require(!ran.empty() && t.dispatches == launched, "kernel_times counts the " + std::to_string(launched) + " launches of" + named + ": " + std::to_string(t.dispatches));
    }
}

// NVFP4 under its tensor's F32 scales at MiniMax-M3.1-Flash's expert width: the F32 class on the exact family (kernels/f32.gpu, ExactNVFP4), the f16 and int8 classes on the twins, one factor a projection.
// Embeddings are the CPU's rows bit for bit, F32 products the F32 product of those rows bit for bit, twin products within twin_close's bound, and a routed entry its expert's one-row product in its class.
inline void nvfp4_scales(Backend& g, Backend& c, const Launches& launches, const Require& require) {
    const uint32_t T = quant::GGML_TYPE_NVFP4;
    const size_t nin = 3072, nout = 96, ne = 4, K = 2, cols = 5;
    // Every scale byte occurs, 0, 127 and those with the top bit among them, since the seeded bytes fill each block's four scales.
    const std::vector<uint8_t> w = typed_matrix(T, nin, ne * nout, 61);
    const std::vector<float> factors = {0.0123457f, 3.14159274f, 1.0f, -0.70710677f}, one = {0.0123457f};
    const std::vector<float> x = row_classes::values(nin * cols, 67);
    const size_t eb = quant::row_bytes(T, nin, nout);
    auto gw = g.adopt(w.data(), w.size()), gs = g.adopt(factors.data(), ne * 4), g1 = g.adopt(one.data(), 4), gx = g.adopt(x.data(), x.size() * 4);
    auto cw = c.adopt(w.data(), w.size()), cs = c.adopt(factors.data(), ne * 4);
    std::vector<uint32_t> every(nout);
    for (size_t i = 0; i < nout; ++i) every[i] = uint32_t(i);
    std::vector<std::vector<float>> rows(ne);
    std::map<std::string, size_t> ran;
    const auto note = [&] {
        for (const auto& [name, n] : launches(g)) ran[name] += n;
    };
    note();
    ran.clear();
    for (size_t e = 0; e < ne; ++e) {
        const std::string at = "NVFP4 expert " + std::to_string(e) + " under factor " + std::to_string(factors[e]);
        const std::vector<uint8_t> we(w.begin() + e * eb, w.begin() + (e + 1) * eb);
        const WeightView cv{T, {cw.get(), e * eb / 4}, {cs.get(), e}, 1}, gv{T, {gw.get(), e * eb / 4}, {gs.get(), e}, 1};
        auto ce = c.alloc(nout * nin * 4), ge = g.alloc(nout * nin * 4);
        c.embed({ce.get(), 0}, cv, nin, nout, every.data(), nout);
        g.embed({ge.get(), 0}, gv, nin, nout, every.data(), nout);
        note();
        rows[e] = read(c, *ce, nout * nin);
        require(!std::memcmp(rows[e].data(), read(g, *ge, nout * nin).data(), nout * nin * 4), "a scaled " + at + " embedding gives the CPU's rows, bit for bit");
        auto fw = g.adopt(rows[e].data(), nout * nin * 4), want = g.alloc(nout * cols * 4), got = g.alloc(nout * cols * 4);
        for (size_t n : {size_t(1), cols}) {
            g.matmul({quant::GGML_TYPE_F32, {fw.get(), 0}}, {gx.get(), 0}, {want.get(), 0}, nin, nout, n, {}, Dtype::f32);
            const std::vector<float> f = read(g, *want, nout * n);
            for (Dtype dt : {Dtype::f32, Dtype::f16, Dtype::int8}) {
                note();
                g.matmul(gv, {gx.get(), 0}, {got.get(), 0}, nin, nout, n, {}, dt);
                note();
                const std::vector<float> y = read(g, *got, nout * n);
                if (dt == Dtype::f32) {
                    require(!std::memcmp(f.data(), y.data(), nout * n * 4), "a scaled " + at + " product of " + std::to_string(n) + " rows in f32 is the F32 product of its decoded rows, bit for bit");
                    continue;
                }
                bool ok = true;
                for (size_t r = 0; r < n && ok; ++r)
                    for (size_t o = 0; o < nout && ok; ++o) ok = twin_close(T, we, rows[e], x, nin, o, r, dt == Dtype::int8 ? 127.0f : 32767.0f, y[r * nout + o]);
                require(ok, "a scaled " + at + " product of " + std::to_string(n) + " rows in " + dtype_name(dt) + " is within 1e-6 of a double sum over the twin of its decoded rows");
                // A group of two scaled projections takes one joint twin call (matmul_group), each under its factor.
                auto g2 = g.alloc(nout * n * 4);
                note();
                g.matmul_group({Projection(gv, {got.get(), 0}, nout), Projection(gv, {g2.get(), 0}, nout)}, {gx.get(), 0}, nin, n, {}, dt);
                note();
                for (const auto* out : {got.get(), g2.get()}) {
                    const std::vector<float> yg = read(g, *out, nout * n);
                    ok = true;
                    for (size_t r = 0; r < n && ok; ++r)
                        for (size_t o = 0; o < nout && ok; ++o) ok = twin_close(T, we, rows[e], x, nin, o, r, dt == Dtype::int8 ? 127.0f : 32767.0f, yg[r * nout + o]);
                    require(ok, "a group of two scaled " + at + " projections of " + std::to_string(n) + " rows in " + dtype_name(dt) + " is within the twin bound in each");
                }
            }
        }
    }
    std::vector<uint32_t> ids(cols * K);
    for (size_t r = 0; r < cols; ++r)
        for (size_t k = 0; k < K; ++k) ids[r * K + k] = uint32_t((r + 3 * k) % ne);
    const std::vector<float> wts(cols * K, 0.5f);
    auto gi = g.adopt(ids.data(), ids.size() * 4), gt = g.adopt(wts.data(), wts.size() * 4), gy = g.alloc(cols * K * nout * 4), gd = g.alloc(nout * 4);
    const Backend::Routing routing{{gi.get(), 0}, {gt.get(), 0}, K, ne};
    for (size_t count : {ne, size_t(1)}) {
        const std::string at = count == 1 ? "one factor" : "a factor an expert";
        g.matmul_experts({Projection{{T, {gw.get(), 0}, {count == 1 ? g1.get() : gs.get(), 0}, count}, {gy.get(), 0}, nout}}, {gx.get(), 0}, nin, cols, routing, {}, Dtype::f16);
        note();
        const std::vector<float> y = read(g, *gy, cols * K * nout);
        for (size_t e = 0; e < cols * K; ++e) {
            const size_t id = ids[e];
            g.matmul({T, {gw.get(), id * eb / 4}, {count == 1 ? g1.get() : gs.get(), count == 1 ? 0 : id}, 1}, {gx.get(), e / K * nin}, {gd.get(), 0}, nin, nout, 1, {}, count == 1 ? Dtype::f16 : Dtype::f32);
            require(!std::memcmp(y.data() + e * nout, read(g, *gd, nout).data(), nout * 4), "routed NVFP4 entry " + std::to_string(e) + " under " + at + " is its expert's one-row product in its class, bit for bit");
        }
    }
    bool refused = false;
    try {
        const std::vector<uint8_t> q = typed_matrix(quant::GGML_TYPE_Q8_0, nin, nout, 71);
        auto qw = g.adopt(q.data(), q.size());
        g.matmul({quant::GGML_TYPE_Q8_0, {qw.get(), 0}, {g1.get(), 0}, 1}, {gx.get(), 0}, {gd.get(), 0}, nin, nout, 1, {}, Dtype::f32);
    } catch (const std::exception&) {
        refused = true;
    }
    require(refused, "weight scales on a Q8_0 matrix are refused, as the CPU refuses them");
    for (const char* k : {"llmx_gather_nvfp4", "llmx_matmul_nvfp4", "llmx_matmul_rows_nvfp4", "llmx_routed_nvfp4", "llmx_nvfp4_rows16_1", "llmx_nvfp4_rows8_1", "llmx_nvfp4_tile16", "llmx_nvfp4_tile8"})
        require(ran.count(k) > 0, std::string("the NVFP4 checks ran ") + k);
    bool routed_twin = false;
    for (const auto& [name, n] : ran) routed_twin = routed_twin || (n && name.rfind("llmx_nvfp4_routed", 0) == 0);
    require(routed_twin, "the NVFP4 checks ran a routed twin build");
}

// F8_E4M3 under its scale grid through the exact family (kernels/f32.gpu, ExactF8) at row shifts 0 and 7, against the CPU embedding and the F32 product of its rows (docs/TESTS.md).
inline void fp8_grids(Backend& g, Backend& c, const Launches& launches, const Require& require) {
    const uint32_t T = quant::GGML_TYPE_F8_E4M3;
    const size_t nin = 1024, nout = 200, ne = 3, K = 2, cols = 5;
    const auto codes = [](size_t n, size_t seed) {
        std::vector<uint8_t> w(n);
        for (size_t i = 0; i < n; ++i) {
            const uint8_t b = uint8_t(i * 37 + i / 1024 * 11 + i / 7 + seed * 101);
            w[i] = (b & 0x7F) == 0x7F ? 0x3F : b;
        }
        return w;
    };
    const auto grid = [](size_t n, size_t seed) {
        std::vector<float> s(n);
        for (size_t i = 0; i < n; ++i) s[i] = std::ldexp(float(int((i + seed) % 7) - 3) * 0.75f + 0.5f, -int(i % 5) - 6);
        return s;
    };
    std::map<std::string, size_t> ran;
    const auto note = [&] {
        for (const auto& [name, n] : launches(g)) ran[name] += n;
    };
    note();
    ran.clear();
    {
        std::vector<uint8_t> all(3 * 256);
        for (size_t i = 0; i < all.size(); ++i) all[i] = uint8_t(i);
        const std::vector<float> sc = {1.0f, -std::ldexp(1.0f, -7)};
        const uint32_t ids[] = {2, 0, 1, 3};
        auto gw = g.adopt(all.data(), all.size()), gs = g.adopt(sc.data(), 8), cw = c.adopt(all.data(), all.size()), cs = c.adopt(sc.data(), 8);
        auto ge = g.alloc(4 * 256 * 4), ce = c.alloc(4 * 256 * 4);
        c.embed({ce.get(), 0}, {T, {cw.get(), 0}, {cs.get(), 0}, 2, 7}, 256, 3, ids, 4);
        g.embed({ge.get(), 0}, {T, {gw.get(), 0}, {gs.get(), 0}, 2, 7}, 256, 3, ids, 4);
        note();
        require(!std::memcmp(read(c, *ce, 4 * 256).data(), read(g, *ge, 4 * 256).data(), 4 * 256 * 4), "an F8_E4M3 embedding of every code under its grid gives the CPU's rows, bit for bit");
    }
    const std::vector<float> x = row_classes::values(nin * cols, 73);
    auto gx = g.adopt(x.data(), x.size() * 4), cx = c.adopt(x.data(), x.size() * 4);
    std::vector<uint32_t> every(nout);
    for (size_t i = 0; i < nout; ++i) every[i] = uint32_t(i);
    for (unsigned shift : {0u, 7u}) {
        const size_t per = gguf::fp8_grid_count(nin, nout, shift, 1);
        const std::vector<uint8_t> w = codes(ne * nout * nin, shift);
        const std::vector<float> sc = grid(ne * per, shift);
        auto gw = g.adopt(w.data(), w.size()), gs = g.adopt(sc.data(), sc.size() * 4), cw = c.adopt(w.data(), w.size()), cs = c.adopt(sc.data(), sc.size() * 4);
        const std::string at = "F8_E4M3 at row shift " + std::to_string(shift);
        const WeightView gv{T, {gw.get(), 0}, {gs.get(), 0}, per, shift}, cv{T, {cw.get(), 0}, {cs.get(), 0}, per, shift};
        auto ce = c.alloc(nout * nin * 4), ge = g.alloc(nout * nin * 4);
        c.embed({ce.get(), 0}, cv, nin, nout, every.data(), nout);
        g.embed({ge.get(), 0}, gv, nin, nout, every.data(), nout);
        note();
        const std::vector<float> rows = read(c, *ce, nout * nin);
        require(!std::memcmp(rows.data(), read(g, *ge, nout * nin).data(), nout * nin * 4), "an " + at + " embedding gives the CPU's rows, bit for bit");
        auto fw = g.adopt(rows.data(), nout * nin * 4), want = g.alloc(nout * cols * 4), got = g.alloc(nout * cols * 4), cy = c.alloc(nout * cols * 4);
        for (size_t n : {size_t(1), cols}) {
            g.matmul({quant::GGML_TYPE_F32, {fw.get(), 0}}, {gx.get(), 0}, {want.get(), 0}, nin, nout, n, {}, Dtype::f32);
            const std::vector<float> f = read(g, *want, nout * n);
            c.matmul(cv, {cx.get(), 0}, {cy.get(), 0}, nin, nout, n, {}, Dtype::f32);
            const std::vector<float> cf = read(c, *cy, nout * n);
            double worst = 0.0;
            for (size_t r = 0; r < n; ++r)
                for (size_t o = 0; o < nout; ++o) {
                    double size = 0.0;
                    for (size_t k = 0; k < nin; ++k) size += std::fabs(double(rows[o * nin + k]) * x[r * nin + k]);
                    worst = std::max(worst, std::fabs(double(f[r * nout + o]) - cf[r * nout + o]) / (16.0 * FLT_EPSILON * size + 1e-30));
                }
            require(worst <= 1.0, "an " + at + " product of " + std::to_string(n) + " rows is within 16 eps of the CPU's, ratio " + std::to_string(worst));
            c.matmul(cv, {cx.get(), 0}, {cy.get(), 0}, nin, nout, n, {}, Dtype::f16);
            const std::vector<float> ch = read(c, *cy, nout * n);
            for (Dtype dt : {Dtype::f32, Dtype::f16, Dtype::int8}) {
                note();
                g.matmul(gv, {gx.get(), 0}, {got.get(), 0}, nin, nout, n, {}, dt);
                const auto took = launches(g);
                for (const auto& [name, k] : took) ran[name] += k;
                const bool tile = n > 1 && dt != Dtype::f32;
                require((took.count("llmx_f8_e4m3_tile") > 0) == tile, "an " + at + " product of " + std::to_string(n) + " rows in " + dtype_name(dt) + (tile ? " takes" : " does not take") + " the half tile");
                const std::vector<float> y = read(g, *got, nout * n);
                if (!tile) {
                    require(!std::memcmp(f.data(), y.data(), nout * n * 4), "an " + at + " product of " + std::to_string(n) + " rows in " + dtype_name(dt) + " is the F32 product of its decoded rows, bit for bit");
                    continue;
                }
                // The half tile's terms, exact binary16 codes times binary16 inputs, are the CPU's f16 terms, summed in another order.
                double worst = 0.0;
                for (size_t r = 0; r < n; ++r)
                    for (size_t o = 0; o < nout; ++o) {
                        double size = 0.0;
                        for (size_t k = 0; k < nin; ++k) size += std::fabs(double(rows[o * nin + k]) * f16_to_f32(f32_to_f16(x[r * nin + k])));
                        worst = std::max(worst, std::fabs(double(y[r * nout + o]) - ch[r * nout + o]) / (16.0 * FLT_EPSILON * size + 1e-30));
                    }
                require(worst <= 1.0, "an " + at + " half tile of " + std::to_string(n) + " rows in " + dtype_name(dt) + " is within 16 eps of the CPU's f16 product, ratio " + std::to_string(worst));
            }
        }
        std::vector<uint32_t> ids(cols * K);
        for (size_t r = 0; r < cols; ++r)
            for (size_t k = 0; k < K; ++k) ids[r * K + k] = uint32_t((r + 2 * k) % ne);
        const std::vector<float> wts(cols * K, 0.5f);
        auto gi = g.adopt(ids.data(), ids.size() * 4), gt = g.adopt(wts.data(), wts.size() * 4), gy = g.alloc(cols * K * nout * 4), gd = g.alloc(nout * 4);
        const Backend::Routing routing{{gi.get(), 0}, {gt.get(), 0}, K, ne};
        g.matmul_experts({Projection{{T, {gw.get(), 0}, {gs.get(), 0}, ne * per, shift}, {gy.get(), 0}, nout}}, {gx.get(), 0}, nin, cols, routing, {}, Dtype::f16);
        note();
        const std::vector<float> y = read(g, *gy, cols * K * nout);
        const size_t eb = nout * nin;
        for (size_t e = 0; e < cols * K; ++e) {
            const size_t id = ids[e];
            g.matmul({T, {gw.get(), id * eb / 4}, {gs.get(), id * per}, per, shift}, {gx.get(), e / K * nin}, {gd.get(), 0}, nin, nout, 1, {}, Dtype::f32);
            require(!std::memcmp(y.data() + e * nout, read(g, *gd, nout).data(), nout * 4), "routed " + at + " entry " + std::to_string(e) + " is its expert's one-row product, bit for bit");
        }
        const auto refused = [&](const WeightView& v) {
            try {
                g.matmul(v, {gx.get(), 0}, {gd.get(), 0}, nin, nout, 1, {}, Dtype::f32);
            } catch (const std::exception&) {
                return true;
            }
            return false;
        };
        require(refused({T, {gw.get(), 0}, {gs.get(), 0}, per - 1, shift}) && refused({T, {gw.get(), 0}, {gs.get(), 0}, per, shift ? 6u : 1u}) && refused({T, {gw.get(), 0}}),
                "an " + at + " matrix with a grid of another count or row shift, or none, is refused, as the CPU refuses it");
    }
    for (const char* k : {"llmx_gather_f8_e4m3", "llmx_matmul_f8_e4m3", "llmx_matmul_rows_f8_e4m3", "llmx_routed_f8_e4m3", "llmx_f8_e4m3_tile"})
        require(ran.count(k) > 0, std::string("the F8_E4M3 checks ran ") + k);
}

} // namespace twin_checks
