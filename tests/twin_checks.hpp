#pragma once
// The twin products' witnesses shared by the GPU backends' tests: each type's builds by their launches, each output against a double sum over the twin, and a column or row of a call its bits alone.
// A test passes its own types, classes, wave width, launches reader and require, so each keeps its own reporting.
#include "backends/backend.hpp"
#include "backends/device_profile.hpp"
#include "backends/gpu/kernels/shapes.h"
#include "core/fp16.hpp"
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
#include <functional>
#include <map>
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
    if (type == quant::GGML_TYPE_Q4_K || type == quant::GGML_TYPE_Q5_K || type == quant::GGML_TYPE_Q5_1) halves = {0, 2};
    if (type == quant::GGML_TYPE_Q8_0 || type == quant::GGML_TYPE_Q5_0 || type == quant::GGML_TYPE_IQ4_NL) halves = {0};
    if (type == quant::GGML_TYPE_Q6_K) halves = {208};
    for (size_t b = 0, i = 0; b < m.size() / st.type_size; ++b)
        for (size_t h : halves) put(b * st.type_size + h, f32_to_f16(scale[i++]));
    // MXFP4's E8M0 scale byte at 2^-10 to 2^-6, as the raw-block fixtures' ordinary blocks take it.
    if (type == quant::GGML_TYPE_MXFP4)
        for (size_t b = 0; b < m.size() / st.type_size; ++b) m[b * st.type_size] = uint8_t(118 + m[b * st.type_size] % 5);
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

// A block's min for its sub-block: Q4_K's and Q5_K's dmin times the 6-bit min as the format packs them, Q5_1's min; Q6_K, Q5_0, IQ4_NL and MXFP4 have none.
inline float k_min(uint32_t type, const uint8_t* block, int sub) {
    if (type == quant::GGML_TYPE_Q6_K || type == quant::GGML_TYPE_Q5_0 || type == quant::GGML_TYPE_IQ4_NL || type == quant::GGML_TYPE_MXFP4) return 0.0f;
    if (type == quant::GGML_TYPE_Q5_1) return f16_to_f32(uint16_t(block[2] | (block[3] << 8)));
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

// Whether row_form takes the quarter form's builds for a row of nin values on a wave of W lanes, restated from its rule: for Q5_0 and Q5_1, where four lanes a block fill the cluster's steps fuller than one, one lane on a tie.
inline bool quarter_form(uint32_t type, size_t nin, size_t W) {
    if (type != quant::GGML_TYPE_Q5_0 && type != quant::GGML_TYPE_Q5_1) return false;
    const size_t blocks = nin / 32;
    const auto span = [W](size_t units) {
        size_t cl = 1;
        while (cl < W && cl < units) cl *= 2;
        return (units + cl - 1) / cl * cl;
    };
    return 4 * blocks * span(blocks) > blocks * span(4 * blocks);
}

// Widths whose rows take every lane cluster a type's rule gives (kernels/int16.gpu, k_rows): lane units below the wave, at the Q4_K and Q5_K cap, above it and not a multiple of the cluster.
// Q5_0's and Q5_1's also lie each side of the lane form's switches on a 64-lane wave (one lane at 1024 and 2048, quarter at 1056 and 2080); 37 outputs, which no block's rows divide.
inline std::vector<std::array<size_t, 2>> cluster_shapes(uint32_t type) {
    const bool k = quant::storage_type(type)->block_size == 256;
    const std::vector<size_t> widths = k ? std::vector<size_t>{256, 512, 768, 1024, 4096} : std::vector<size_t>{32, 64, 96, 640, 1024, 1056, 2048, 2080, 4096};
    std::vector<std::array<size_t, 2>> out;
    for (size_t nin : widths) out.push_back({nin, 37});
    return out;
}

// The twin products of the given types at Flash-Next's shapes in the given classes, extents on both sides of each crossover, each output against a double sum over the twin.
// The backend's launches name the build or tile that ran, and a row of a call is the row alone or in another call of its class, every generated column of up to 64; the cluster widths (cluster_shapes) take the generated rows alone.
inline void twin_types(Backend& g, Backend& c, const std::vector<uint32_t>& kinds, const std::vector<Dtype>& dtypes, size_t wave, const Launches& launches,
                       const Require& require) {
    const size_t cols[] = {1, 2, 3, 5, 8, 9, 16, 17, 33}, extents[] = {2, 33, 64, 65, 320};
    for (uint32_t type : kinds) {
        std::string tname = quant::storage_type(type)->name;
        for (char& ch : tname) ch = char(std::tolower(ch));
        std::vector<std::array<size_t, 2>> shapes = flash_shapes(type);
        const size_t flash = shapes.size();
        for (const auto& sh : cluster_shapes(type)) shapes.push_back(sh);
        if (type == quant::GGML_TYPE_Q5_0 || type == quant::GGML_TYPE_Q5_1) {
            size_t quarter = 0;
            for (const auto& sh : shapes) quarter += quarter_form(type, sh[0], wave);
            require(quarter > 0 && quarter < shapes.size(), tname + "'s widths take both lane forms");
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
                        ++want[b == 1 && quad_build(type, dt, nin, wave) ? "llmx_mxfp4_quad_rows8_1" : "llmx_" + tname + (quarter_form(type, nin, wave) ? "_quarter" : "") + "_rows" + form + "_" + std::to_string(b)];
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
                    require(ran.size() == 1 && ran.begin()->first == "llmx_" + tname + "_tile" + form, what + " runs the tile alone");
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
inline void twin_products(Backend& g, const DeviceCaps& caps, Dtype dt, const Require& require) {
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
        return backend::tile_rows_for(caps, prof, 64, 64, 128, n, tiles_of(tr), tw, dt == Dtype::int8 ? prof.qtile_tall_from_8bit : prof.qtile_tall_from) == 128;
    };
    bool tall_met = false;
    for (const std::array<size_t, 4> shape : {std::array<size_t, 4>{512, 70, 3940, 100}, {512, 70, 3000, 100}, {4096, 520, 5860, 3840}, {4128, 520, 5860, 3840}}) {
        const size_t tw = shape[0], tr = shape[1], tn = shape[2], slice = shape[3];
        if (kper_of(tn, tw, tr) != kper_of(slice, tw, tr)) continue;
        tall_met = tall_met || tall_of(tn, tw, tr) != tall_of(slice, tw, tr);
        const std::vector<uint8_t> mt = q8_matrix(tw, tn, 91);
        const std::vector<float> xt = row_classes::values(tw * tr, 92);
        const size_t row_bytes = tw / 32 * 34;
        auto mtb = g.adopt(mt.data(), mt.size()), xtb = g.adopt(xt.data(), xt.size() * 4), yt = g.alloc(tn * tr * 4), ys = g.alloc(slice * tr * 4);
        const RowRun run[1] = {{tr, tr}};
        g.matmul({quant::GGML_TYPE_Q8_0, {mtb.get(), 0}}, {xtb.get(), 0}, {yt.get(), 0}, tw, tn, tr, {run, 1}, dt);
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

} // namespace twin_checks
