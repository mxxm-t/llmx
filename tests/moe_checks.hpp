#pragma once
// The routed experts' checks shared by the GPU backends' tests: routing against the CPU and the host's kernel order, and each routed build's entries their expert's one-column product bit for bit.
// A test passes its own wave width, launches reader, routed-build filter and require (twin_checks.hpp), so each keeps its own reporting.
#include "backends/backend.hpp"
#include "quant/types.hpp"
#include "row_classes.hpp"
#include "twin_checks.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace moe_checks {

using namespace backend;
using twin_checks::DoubleSums;
using twin_checks::double_sums;
using twin_checks::q8_matrix;
using twin_checks::read;
using twin_checks::same;
using twin_checks::typed_matrix;

struct Moe {
    size_t wave;   // the device's wave width, which picks the lane forms
    twin_checks::Launches launches;
    std::function<void(Backend&, std::string)> route_only;   // the routed tile builds a call may take, by name, empty for all
    twin_checks::Require require;

    void close(const std::vector<float>& got, const std::vector<float>& want, const std::string& what) const { require(twin_checks::f32_close(got, want), what); }
    void exact_close(const std::vector<float>& got, const DoubleSums& d, const std::string& what) const { require(twin_checks::exact_close(got, d), what); }
    bool quarter_form(uint32_t type, size_t nin, bool int8) const { return twin_checks::quarter_form(type, nin, wave, int8); }

    // Routing on the host in the kernel's order (kernels/f32.gpu, llmx_moe_route): lane t of 64 takes experts t, t + 64, ..., the softmax's sum by the 64-pair tree, then k picks, ties to the lowest id.
    // Its expf is the host's, so its weights may differ from the device's in the last bits.
    void route_host(const float* s, size_t n_expert, size_t k, bool normalize, uint32_t* ids, float* weights) {
        constexpr size_t L = 64;
        float top = -INFINITY;
        for (size_t e = 0; e < n_expert; ++e) top = std::fmax(top, s[e]);
        std::vector<float> p(n_expert), part(L, 0.0f);
        for (size_t e = 0; e < n_expert; ++e) {
            p[e] = std::exp(s[e] - top);
            part[e % L] += p[e];
        }
        for (size_t h = L / 2; h; h /= 2)
            for (size_t t = 0; t < h; ++t) part[t] = part[t] + part[t + h];
        const float sum = part[0];
        if (std::isnan(sum)) {
            for (size_t j = 0; j < k; ++j) ids[j] = Backend::kNoExpert, weights[j] = 0.0f;
            return;
        }
        for (float& v : p) v = v / sum < FLT_MIN ? 0.0f : v / sum;
        float total = 0.0f;
        std::vector<float> chosen(k);
        for (size_t j = 0; j < k; ++j) {
            float m = -2.0f;
            uint32_t id = Backend::kNoExpert;
            for (size_t e = 0; e < n_expert; ++e)
                if (p[e] > m) m = p[e], id = uint32_t(e);
            chosen[j] = id < n_expert ? m : 0.0f;
            total += chosen[j];
            if (id < n_expert) p[id] = -1.0f;
            ids[j] = id;
        }
        for (size_t j = 0; j < k; ++j) weights[j] = ids[j] >= n_expert ? 0.0f : normalize ? chosen[j] / total : chosen[j];
    }

    // Routing against the CPU: the same picks, ties to the lower id, rows without a candidate giving kNoExpert, and weights within the F32 bound.
    void routing(Backend& g, Backend& c) {
        for (size_t n_expert : {16, 128, 1024})
            for (size_t k : {1, 4, 8})
                for (bool normalize : {false, true}) {
                    const size_t rows = 37;
                    std::vector<float> s = row_classes::values(rows * n_expert, uint32_t(n_expert + k), 4.0f);
                    s[1 * n_expert + 3] = s[1 * n_expert + 9] = 30.0f;   // a tie at the top
                    s[2 * n_expert + 5] = NAN;
                    for (size_t e = 0; e < n_expert; ++e) s[3 * n_expert + e] = -INFINITY;
                    s[4 * n_expert] = INFINITY;
                    std::vector<float> ids[2], ws[2];
                    Backend* bs[2] = {&g, &c};
                    for (int b = 0; b < 2; ++b) {
                        auto sb = bs[b]->adopt(s.data(), s.size() * 4), ib = bs[b]->alloc(rows * k * 4), wb = bs[b]->alloc(rows * k * 4);
                        bs[b]->route_experts({sb.get(), 0}, rows, n_expert, k, backend::RouteRule::softmax(normalize), {ib.get(), 0}, {wb.get(), 0});
                        ids[b] = read(*bs[b], *ib, rows * k);
                        ws[b] = read(*bs[b], *wb, rows * k);
                    }
                    const std::string at = std::to_string(k) + " of " + std::to_string(n_expert) + (normalize ? " normalized" : "");
                    require(same(ids[0], ids[1]), "routing " + at + " picks the CPU's experts");
                    close(ws[0], ws[1], "routing " + at + " weighs as the CPU does");
                    uint32_t none;
                    std::memcpy(&none, &ids[0][2 * k], 4);
                    require(none == Backend::kNoExpert, "a row with a NaN score routes to no expert");
                    // The device's picks are the host's in the kernel's order, ties to the lower id, whatever the device's wave width; the weights hold the CPU's bound, since the host's expf is not the device's.
                    std::vector<uint32_t> hid(rows * k);
                    std::vector<float> hw(rows * k);
                    for (size_t r = 0; r < rows; ++r) route_host(s.data() + r * n_expert, n_expert, k, normalize, hid.data() + r * k, hw.data() + r * k);
                    require(std::memcmp(hid.data(), ids[0].data(), rows * k * 4) == 0, "routing " + at + " picks the host's experts in the kernel's order");
                }
    }

    // Whether a twin class's routed call groups its entries by expert, from each build's break-even (backend_core.hpp, TwinKernels::group_from).
    bool groups_at(uint32_t type, Dtype dt, size_t entries, size_t n_expert, size_t nin) {
        const bool i8 = dt == Dtype::int8, quarter = quarter_form(type, nin, i8);
        const double from = type == quant::GGML_TYPE_Q8_0 ? (i8 ? 0 : 1.25) : type == quant::GGML_TYPE_Q4_K || type == quant::GGML_TYPE_Q5_K ? (i8 ? 1.25 : 2.5)
                          : type == quant::GGML_TYPE_Q6_K || type == quant::GGML_TYPE_Q3_K || type == quant::GGML_TYPE_Q2_K || type == quant::GGML_TYPE_Q5_0 ? 1.25
                          : type == quant::GGML_TYPE_IQ4_NL || type == quant::GGML_TYPE_IQ4_XS || type == quant::GGML_TYPE_IQ2_XXS || type == quant::GGML_TYPE_IQ2_XS || type == quant::GGML_TYPE_IQ2_S || type == quant::GGML_TYPE_IQ3_XXS || type == quant::GGML_TYPE_IQ3_S || type == quant::GGML_TYPE_MXFP4 ? 0.3125
                          : type == quant::GGML_TYPE_Q4_0 ? 5.0 : type == quant::GGML_TYPE_Q4_1 ? (quarter ? 2.5 : 5.0) : quarter ? 1.25 : (i8 ? 1.25 : 5.0);
        // The half-block form's calls of at most one entry an expert take the one-column build an entry where it fills its steps.
        const size_t blocks = nin / 32, sub = wave;
        const bool half = type == quant::GGML_TYPE_Q8_0 && i8 && blocks % 2 == 0 && (blocks < sub || blocks >= 2 * sub);
        return entries >= 1 && double(entries) >= from * double(n_expert) && !(half && entries <= n_expert);
    }

    // The prompt extent from which a routed build takes the routed tile, the backend's `TwinKernels::tile_from`: 32, but 128 for block-int16 gate and up.
    // Q8_0, IQ4_NL, IQ4_XS, the IQ2 and IQ3 types and MXFP4 keep 32 there, and Q4_0's and Q4_1's block-int16 down takes 128 too.
    size_t tile_from_at(uint32_t type, Dtype dt, bool down) {
        const bool q4 = type == quant::GGML_TYPE_Q4_0 || type == quant::GGML_TYPE_Q4_1;
        return (down && !q4) || dt == Dtype::int8 || type == quant::GGML_TYPE_Q8_0 || type == quant::GGML_TYPE_IQ4_NL || type == quant::GGML_TYPE_IQ4_XS || type == quant::GGML_TYPE_IQ2_XXS || type == quant::GGML_TYPE_IQ2_XS || type == quant::GGML_TYPE_IQ2_S || type == quant::GGML_TYPE_IQ3_XXS || type == quant::GGML_TYPE_IQ3_S ||
               type == quant::GGML_TYPE_MXFP4 ? 32 : 128;
    }

    // Routed products at Flash-Next's expert shapes, every type and class, each entry its expert's one-column product bit for bit (docs/src/backends-gpu.md).
    void experts(Backend& g, Backend& c) {
        const uint32_t kinds[] = {quant::GGML_TYPE_F32, quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K,
                                  quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_BF16, quant::GGML_TYPE_F16, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_MXFP4,
                                  quant::GGML_TYPE_Q2_K, quant::GGML_TYPE_Q3_K, quant::GGML_TYPE_IQ4_XS,
                                  quant::GGML_TYPE_IQ2_XXS, quant::GGML_TYPE_IQ2_XS, quant::GGML_TYPE_IQ2_S, quant::GGML_TYPE_IQ3_XXS, quant::GGML_TYPE_IQ3_S,
                                  quant::GGML_TYPE_IQ1_S, quant::GGML_TYPE_IQ1_M, quant::GGML_TYPE_TQ1_0, quant::GGML_TYPE_TQ2_0, quant::GGML_TYPE_Q1_0, quant::GGML_TYPE_Q2_0};
        const size_t n_expert = 8, k = 4;
        for (uint32_t type : kinds) {
            const std::string name = quant::storage_type(type)->name;
            for (const auto& sh : std::vector<std::array<size_t, 2>>{{2560, 640}, {640, 2560}}) {
                const size_t nin = sh[0], ff = sh[1];
                if (nin % quant::storage_type(type)->block_size) continue;
                const size_t ebytes = quant::row_bytes(type, nin, ff);
                require(ebytes % 4 == 0, "an expert's matrix ends on a float");
                std::vector<uint8_t> w = type == quant::GGML_TYPE_F32  ? std::vector<uint8_t>()
                                         : type == quant::GGML_TYPE_Q8_0 ? q8_matrix(nin, n_expert * ff, uint32_t(nin))
                                                                         : typed_matrix(type, nin, n_expert * ff, uint32_t(type * 7 + nin));
                if (type == quant::GGML_TYPE_F32) {
                    const std::vector<float> v = row_classes::values(nin * n_expert * ff, uint32_t(nin + 3), 0.05f);
                    w.assign((const uint8_t*)v.data(), (const uint8_t*)v.data() + v.size() * 4);
                }
                auto wa = g.adopt(w.data(), w.size()), wb = g.adopt(w.data(), w.size());
                // The row counts take the one-column build and, past each build's break-even, a grouped one in the twin classes (groups_at).
                for (size_t rows : {1, 3, 4, 5, 33}) {
                    const std::vector<float> x = row_classes::values(rows * nin, uint32_t(rows + nin));
                    std::vector<uint32_t> idv(rows * k);
                    for (size_t e = 0; e < idv.size(); ++e) idv[e] = uint32_t((e * 5 + rows) % n_expert);
                    idv[k + 1] = Backend::kNoExpert;
                    const std::vector<float> wts = row_classes::values(rows * k, uint32_t(rows + 11), 0.5f);
                    auto xb = g.adopt(x.data(), x.size() * 4), ib = g.adopt(idv.data(), idv.size() * 4), tb = g.adopt(wts.data(), wts.size() * 4);
                    const Backend::Routing routing{{ib.get(), 0}, {tb.get(), 0}, k, n_expert};
                    std::vector<RowRun> decode(rows);
                    for (size_t r = 0; r < rows; ++r) decode[r] = {r + 1, 1};
                    const RowRun prompt[1] = {{rows, rows}};
                    for (Dtype dt : {Dtype::f32, Dtype::f16, Dtype::int8}) {
                        const bool twins = type != quant::GGML_TYPE_F32 && !twin_checks::exact_only(type);
                        if (!twins && dt != Dtype::f32) continue;
                        for (const RowRuns runs : {RowRuns{decode.data(), rows}, RowRuns{prompt, 1}}) {
                            const std::string at = name + " " + std::to_string(nin) + "->" + std::to_string(ff) + ", " + std::to_string(rows) + (runs.n == 1 && rows > 1 ? " prompt rows, " : " rows, ") + dtype_name(dt);
                            const std::vector<float> fill(rows * k * ff, 7.0f);
                            auto ga = g.adopt(fill.data(), fill.size() * 4), gb = g.adopt(fill.data(), fill.size() * 4);
                            launches(g);
                            g.matmul_experts({Projection{{type, {wa.get(), 0}}, {ga.get(), 0}, ff}, Projection{{type, {wb.get(), 0}}, {gb.get(), 0}, ff}}, {xb.get(), 0}, nin, rows, routing, runs, dt);
                            // The kernels each class's routed call runs, and nothing else (docs/src/backends-gpu.md).
                            const bool tiled = twins && dt != Dtype::f32 && runs.n == 1 && rows >= tile_from_at(type, dt, false);
                            const bool grouped = twins && dt != Dtype::f32 && groups_at(type, dt, rows * k, n_expert, nin);
                            std::string ran;
                            bool routed_only = true, routed_ran = false, lane_form = true;
                            for (const auto& [kn, v] : launches(g)) {
                                ran += (ran.empty() ? "" : "+") + kn;
                                const bool exact = kn.rfind("llmx_routed_", 0) == 0, twin = kn.find("_routed") != std::string::npos && !exact;
                                const bool group_build = twin && kn.find("_g", kn.find("_routed")) != std::string::npos;
                                const bool tile = kn.find("_routed_tile") != std::string::npos || kn.find("_routed_narrow") != std::string::npos || kn.find("_routed_stream") != std::string::npos ||
                                                  kn.find("_routed_thin") != std::string::npos || kn.find("_routed_mid") != std::string::npos;
                                if (twin && !tile) lane_form = lane_form && (kn.find("_quarter_") != std::string::npos) == quarter_form(type, nin, dt == Dtype::int8);
                                routed_ran = routed_ran || (tiled ? tile : twins && dt != Dtype::f32 ? twin && !tile && group_build == grouped : exact);
                                routed_only = routed_only && (tiled ? tile || kn.rfind("llmx_twin", 0) == 0 || kn == "llmx_moe_group"
                                                                    : exact || (twin && !tile && group_build == grouped) || kn.rfind("llmx_twin", 0) == 0 || (grouped && kn == "llmx_moe_group"));
                            }
                            require(routed_ran && routed_only, at + " runs its routed build alone, ran " + ran);
                            require(lane_form, at + " takes its row's lane form, ran " + ran);
                            const std::vector<float> got[2] = {read(g, *ga, rows * k * ff), read(g, *gb, rows * k * ff)};
                            auto one = g.alloc(ff * 4);
                            bool alone = true, unwritten = true;
                            for (size_t e = 0; e < rows * k; ++e) {
                                if (idv[e] == Backend::kNoExpert) {
                                    unwritten = std::all_of(got[0].begin() + e * ff, got[0].begin() + (e + 1) * ff, [](float v) { return v == 7.0f; });
                                    continue;
                                }
                                if (tiled) {
                                    alone = alone && !std::memcmp(got[0].data() + e * ff, got[1].data() + e * ff, ff * 4);
                                    continue;
                                }
                                g.matmul({type, {wa.get(), idv[e] * ebytes / 4}}, {xb.get(), e / k * nin}, {one.get(), 0}, nin, ff, 1, {}, dt);
                                const std::vector<float> o = read(g, *one, ff);
                                alone = alone && !std::memcmp(o.data(), got[0].data() + e * ff, ff * 4) && !std::memcmp(o.data(), got[1].data() + e * ff, ff * 4);
                            }
                            require(alone, at + (tiled ? ": each entry has the same bits in both projections of one weight" : ": each entry is its expert's one-column product, bit for bit"));
                            require(unwritten, at + ": an entry without an expert is not written");
                            if (dt == Dtype::f32 && rows == 5 && runs.n > 1) {
                                // Every expert's decoded rows by the CPU's embedding, and each entry against the double sum.
                                auto cw = c.adopt(w.data(), w.size()), ce = c.alloc(n_expert * ff * nin * 4);
                                std::vector<uint32_t> every(n_expert * ff);
                                for (size_t i = 0; i < every.size(); ++i) every[i] = uint32_t(i);
                                c.embed({ce.get(), 0}, {type, {cw.get(), 0}}, nin, n_expert * ff, every.data(), every.size());
                                const std::vector<float> wd = read(c, *ce, n_expert * ff * nin);
                                for (size_t e = 0; e < rows * k; ++e) {
                                    if (idv[e] == Backend::kNoExpert) continue;
                                    const std::vector<float> we(wd.begin() + idv[e] * ff * nin, wd.begin() + (idv[e] + 1) * ff * nin);
                                    const std::vector<float> xe(x.begin() + e / k * nin, x.begin() + (e / k + 1) * nin);
                                    exact_close(std::vector<float>(got[0].begin() + e * ff, got[0].begin() + (e + 1) * ff), double_sums(we, xe, nin, ff), at + ": entry " + std::to_string(e) + " against the double sum");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // The grouping's own branches, 300 experts 8 a token over 80 rows and skewed ids, every entry its expert's one-column product bit for bit (docs/src/backends-gpu.md).
    // Every twin type takes its grouped builds and its one-column build here, each entry against the same one-column product, so the builds give the same bits whichever the break-evens choose.
    void experts_grouped(Backend& g) {
        struct Case { size_t n_expert, k, rows; bool skew; };
        // Past each build's break-even the calls below take the 4-, 8-, 4- and 2-column builds, below it the one-column build, and 4 rows of 300 the one-column build below every break-even.
        const Case cases[] = {{300, 8, 80, false}, {16, 8, 33, true}, {16, 8, 5, false}, {64, 2, 64, true}, {300, 8, 4, false}};
        const size_t ff = 256;
        for (const auto& [type, nin] : std::vector<std::pair<uint32_t, size_t>>{{quant::GGML_TYPE_Q8_0, 2080}, {quant::GGML_TYPE_Q8_0, 2560}, {quant::GGML_TYPE_Q8_0, 1280}, {quant::GGML_TYPE_Q8_0, 4096}, {quant::GGML_TYPE_Q4_K, 2560}, {quant::GGML_TYPE_Q5_K, 2560},
                                                                                 {quant::GGML_TYPE_Q6_K, 2560}, {quant::GGML_TYPE_Q4_0, 2560}, {quant::GGML_TYPE_Q4_1, 2560}, {quant::GGML_TYPE_Q5_0, 2560}, {quant::GGML_TYPE_Q5_1, 2560}, {quant::GGML_TYPE_IQ4_NL, 2560}, {quant::GGML_TYPE_MXFP4, 2560},
                                                                                 {quant::GGML_TYPE_Q4_0, 640}, {quant::GGML_TYPE_Q4_1, 640}, {quant::GGML_TYPE_Q5_0, 640}, {quant::GGML_TYPE_Q5_1, 640}}) {
            const std::string name = quant::storage_type(type)->name;
            const size_t ebytes = quant::row_bytes(type, nin, ff);
            const std::vector<uint8_t> w = type == quant::GGML_TYPE_Q8_0 ? q8_matrix(nin, 300 * ff, uint32_t(nin)) : typed_matrix(type, nin, 300 * ff, uint32_t(type * 7 + nin));
            auto wb = g.adopt(w.data(), w.size());
            for (const Case& cs : cases) {
                const size_t entries = cs.rows * cs.k;
                const std::vector<float> x = row_classes::values(cs.rows * nin, uint32_t(cs.rows + nin));
                std::vector<uint32_t> idv(entries);
                for (size_t e = 0; e < entries; ++e) idv[e] = uint32_t((e * 7 + e / cs.k) % cs.n_expert);
                if (cs.skew) {
                    // Expert 0 one entry, expert 1 every other entry (several groups, the last partial), the top quarter unused.
                    const uint32_t spread = uint32_t(cs.n_expert * 3 / 4 - 2);
                    for (size_t e = 0; e < entries; ++e) idv[e] = e == 4 ? 0u : e % 2 ? 1u : uint32_t(2 + (e * 5) % spread);
                    idv[entries - 2] = Backend::kNoExpert;
                }
                const std::vector<float> wts = row_classes::values(entries, uint32_t(cs.rows + 11), 0.5f);
                auto xb = g.adopt(x.data(), x.size() * 4), ib = g.adopt(idv.data(), idv.size() * 4), tb = g.adopt(wts.data(), wts.size() * 4);
                const Backend::Routing routing{{ib.get(), 0}, {tb.get(), 0}, cs.k, cs.n_expert};
                std::vector<RowRun> decode(cs.rows);
                for (size_t r = 0; r < cs.rows; ++r) decode[r] = {r + 1, 1};
                const RowRun prompt[1] = {{cs.rows, cs.rows}};
                for (Dtype dt : {Dtype::f16, Dtype::int8})
                    for (const RowRuns runs : {RowRuns{decode.data(), cs.rows}, RowRuns{prompt, 1}}) {
                        if (runs.n == 1 && cs.rows >= tile_from_at(type, dt, false)) continue;   // the routed tile's (experts_tile)
                        const std::string at = name + " " + std::to_string(nin) + "->" + std::to_string(ff) + ", " + std::to_string(cs.n_expert) + " experts, " + std::to_string(cs.rows) + (runs.n == 1 ? " prompt rows, " : " rows, ") + dtype_name(dt);
                        const std::vector<float> fill(entries * ff, 7.0f);
                        auto gb = g.adopt(fill.data(), fill.size() * 4);
                        launches(g);
                        g.matmul_experts({Projection{{type, {wb.get(), 0}}, {gb.get(), 0}, ff}}, {xb.get(), 0}, nin, cs.rows, routing, runs, dt);
                        const bool grouped = groups_at(type, dt, entries, cs.n_expert, nin);
                        const size_t touched = std::min(cs.n_expert, entries), average = (entries + touched - 1) / touched;
                        const std::string build = !grouped ? "" : average <= 2 ? "_g2" : average <= 4 ? "_g4" : "_g8";
                        // The half-block one-column build takes one step a load on a row under two spans of blocks, as the dense one-column build does.
                        const bool half = type == quant::GGML_TYPE_Q8_0 && dt == Dtype::int8 && nin / 32 % 2 == 0;
                        const std::string one_build = !half ? "" : nin / 32 < wave ? "llmx_q8_0_routed_half8s1" : "llmx_q8_0_routed_half8";
                        std::string ran;
                        bool built = false, grouping = false;
                        for (const auto& [kn, v] : launches(g)) {
                            ran += (ran.empty() ? "" : "+") + kn;
                            const size_t r = kn.find("_routed");
                            built = built || (r != std::string::npos && (grouped ? kn.compare(kn.size() - 3, 3, build) == 0 : kn.find("_g", r) == std::string::npos && (one_build.empty() || kn == one_build)));
                            grouping = grouping || kn == "llmx_moe_group";
                        }
                        require(built && grouping == grouped, at + " takes its " + (grouped ? build.substr(1) + " grouped" : "one-column") + " build, ran " + ran);
                        const std::vector<float> got = read(g, *gb, entries * ff);
                        auto one = g.alloc(ff * 4);
                        bool alone = true, unwritten = true;
                        for (size_t e = 0; e < entries; ++e) {
                            if (idv[e] == Backend::kNoExpert) {
                                unwritten = unwritten && std::all_of(got.begin() + e * ff, got.begin() + (e + 1) * ff, [](float v) { return v == 7.0f; });
                                continue;
                            }
                            g.matmul({type, {wb.get(), idv[e] * ebytes / 4}}, {xb.get(), e / cs.k * nin}, {one.get(), 0}, nin, ff, 1, {}, dt);
                            alone = alone && !std::memcmp(read(g, *one, ff).data(), got.data() + e * ff, ff * 4);
                        }
                        require(alone, at + ": each entry is its expert's one-column product, bit for bit");
                        require(unwritten, at + ": an entry without an expert is not written");
                    }
            }
        }
    }

    // The routed tile's group sizes over rows [first, first + count) of `ids` (llmx_moe_group), each expert's entries in groups of 64, the last holding the rest; per_expert takes each expert's entries.
    std::vector<size_t> tile_groups(const std::vector<uint32_t>& ids, size_t first, size_t count, size_t k, size_t n_expert, std::vector<size_t>& per_expert) {
        std::vector<size_t> n(n_expert, 0), sizes;
        for (size_t e = first * k; e < (first + count) * k; ++e)
            if (ids[e] < n_expert) ++n[ids[e]];
        for (size_t c : n)
            for (size_t at = 0; at < c; at += 64) sizes.push_back(std::min<size_t>(64, c - at));
        per_expert = n;
        return sizes;
    }

    // experts_tile's ids for half extent H over 2H rows: experts 11 to 15 hold 1, 7, 15, 16 and 17 entries in the half call and one group of 17 to 33 each in the whole call.
    // Expert 0's five first-half entries make the whole call launch a narrow group too, and two entries have no expert.
    std::vector<uint32_t> tile_ids(size_t H, size_t k) {
        std::vector<uint32_t> v(2 * H * k);
        for (size_t e = 0; e < v.size(); ++e) v[e] = e % 2 ? 1u : uint32_t(2 + (e * 5 + e / k) % 9);
        const size_t sizes[5] = {1, 7, 15, 16, 17};
        size_t at = H * k;
        for (size_t s = 0; s < 5; ++s)
            for (size_t j = 0; j < sizes[s]; ++j, at += 2) v[at] = uint32_t(11 + s);
        require(at <= 2 * H * k, "experts_tile's second half holds the sparse experts' entries");
        at = 0;
        for (size_t s = 0; s < 5; ++s)
            for (size_t j = 0; j < 16; ++j, at += 2) v[at] = uint32_t(11 + s);
        require(at <= H * k, "experts_tile's first half holds the sparse experts' entries");
        for (size_t e : {1, 5, 13, 17, 21}) v[e] = 0;
        v[9] = v[(H + 5) * k + 1] = Backend::kNoExpert;
        return v;
    }

    // Whether the calls over rows [0, 2H) and [H, 2H) each hold groups of both builds (narrow up to 16 entries), and the half call tile_ids' five sizes as wide groups of the whole call.
    bool tile_groups_cover(const std::vector<uint32_t>& ids, size_t H, size_t k, size_t n_expert) {
        std::vector<size_t> whole_n, half_n;
        const auto whole = tile_groups(ids, 0, 2 * H, k, n_expert, whole_n), half = tile_groups(ids, H, H, k, n_expert, half_n);
        const auto both = [](const std::vector<size_t>& g) {
            return std::any_of(g.begin(), g.end(), [](size_t n) { return n <= 16; }) && std::any_of(g.begin(), g.end(), [](size_t n) { return n > 16; });
        };
        bool sizes = true;
        for (size_t want : {1, 7, 15, 16, 17}) {
            bool found = false;
            for (size_t x = 0; x < n_expert; ++x) found = found || (half_n[x] == want && whole_n[x] > 16 && whole_n[x] <= 64);
            sizes = sizes && found;
        }
        return both(whole) && both(half) && sizes;
    }

    // The routed prompt tile from its build's tile extent T for the call's role, every type, gate and up and the down, held to the same bits at 2T, T and beside generated rows (docs/src/backends-gpu.md).
    // T - 1 rows take the row builds, and the two extents are different row classes.
    void experts_tile(Backend& g, Backend& c) {
        const size_t n_expert = 16, k = 4, ff[2] = {256, 200}, most = 2 * 128;   // twice the widest tile extent of tile_from_at
        const std::vector<std::pair<uint32_t, size_t>> kinds = {{quant::GGML_TYPE_Q8_0, 2560}, {quant::GGML_TYPE_Q8_0, 2080}, {quant::GGML_TYPE_Q4_K, 2560}, {quant::GGML_TYPE_Q5_K, 2560},
                                                                {quant::GGML_TYPE_Q6_K, 2560}, {quant::GGML_TYPE_Q4_0, 2560}, {quant::GGML_TYPE_Q4_1, 2560}, {quant::GGML_TYPE_Q5_0, 2560}, {quant::GGML_TYPE_Q5_1, 2560}, {quant::GGML_TYPE_IQ4_NL, 2560}, {quant::GGML_TYPE_MXFP4, 2560},
                                                                {quant::GGML_TYPE_Q4_0, 2080}};
        for (const auto& [type, nin] : kinds) {
            const std::string name = quant::storage_type(type)->name, tname = std::string(name == "Q8_0" ? "q8_0" : name == "Q4_K" ? "q4_k" : name == "Q5_K" ? "q5_k" : name == "Q6_K" ? "q6_k" : name == "Q4_0" ? "q4_0" : name == "Q4_1" ? "q4_1" : name == "Q5_0" ? "q5_0" : name == "Q5_1" ? "q5_1" : name == "MXFP4" ? "mxfp4" : "iq4_nl");
            const size_t nblk = nin / 32;
            std::vector<uint8_t> w[2];
            BufferPtr wb[2];
            std::vector<float> wd[2];
            for (int p = 0; p < 2; ++p) {
                w[p] = type == quant::GGML_TYPE_Q8_0 ? q8_matrix(nin, n_expert * ff[p], uint32_t(nin + p)) : typed_matrix(type, nin, n_expert * ff[p], uint32_t(type * 13 + nin + p));
                wb[p] = g.adopt(w[p].data(), w[p].size());
                auto cw = c.adopt(w[p].data(), w[p].size()), ce = c.alloc(n_expert * ff[p] * nin * 4);
                std::vector<uint32_t> every(n_expert * ff[p]);
                for (size_t i = 0; i < every.size(); ++i) every[i] = uint32_t(i);
                c.embed({ce.get(), 0}, {type, {cw.get(), 0}}, nin, n_expert * ff[p], every.data(), every.size());
                wd[p] = read(c, *ce, n_expert * ff[p] * nin);
            }
            const std::vector<float> x = row_classes::values(most * nin, uint32_t(nin + 41));
            const std::vector<float> wts(most * k, 0.25f);
            auto xb = g.adopt(x.data(), x.size() * 4), tb = g.adopt(wts.data(), wts.size() * 4);
            for (Dtype dt : {Dtype::f16, Dtype::int8}) {
                const std::string at = name + " " + std::to_string(nin) + ", " + dtype_name(dt);
                const std::string tile = "llmx_" + tname + "_routed_tile" + (dt == Dtype::int8 ? "8" : "16"), narrow = "llmx_" + tname + (dt == Dtype::int8 ? (tname == "q8_0" ? "_routed_narrow8" : "_routed_stream16_8") : "_routed_narrow16");
                for (bool down : {false, true}) {
                    const size_t from = tile_from_at(type, dt, down);
                    require(g.row_class(from - 1) != g.row_class(from), at + (down ? ", down" : ", gate and up") + ": extents " + std::to_string(from - 1) + " and " + std::to_string(from) + " are different row classes");
                }
                // The grouping and the routed tile's two builds launched, no row build; which build a group ran in is witnessed by the bits across extents.
                const auto tiled_alone = [&](const std::string& ran) {
                    return ran.find(tile) != std::string::npos && ran.find(narrow) != std::string::npos && ran.find("llmx_moe_group") != std::string::npos && ran.find("_rows") == std::string::npos && ran.find("_routed1") == std::string::npos &&
                           ran.find("_routed8") == std::string::npos && ran.find("_routed_half") == std::string::npos;
                };
                // Gate and up.
                {
                    const size_t from = tile_from_at(type, dt, false), half_rows = std::max<size_t>(from, 40), rows = 2 * half_rows;
                    require(rows <= most, "experts_tile holds the rows of every build's tile extent");
                    const std::vector<uint32_t> idv = tile_ids(half_rows, k);
                    require(tile_groups_cover(idv, half_rows, k, n_expert), at + ": the calls at extents " + std::to_string(rows) + " and " + std::to_string(half_rows) + " hold groups of both builds, of 1, 7, 15, 16 and 17 entries wide at twice the extent");
                    auto ib = g.adopt(idv.data(), idv.size() * 4);
                    // A call over rows [first, first + count) as one prompt of that extent, or with `decode` generated rows ahead of the prompt, into outputs filled with 7.
                    const auto call = [&](size_t first, size_t count, size_t decode, std::string& ran) {
                        std::vector<RowRun> runs;
                        for (size_t r = 0; r < decode; ++r) runs.push_back({r + 1, 1});
                        if (count > decode) runs.push_back({count, count - decode});
                        const std::vector<float> fill(count * k * 256, 7.0f);
                        auto ga = g.adopt(fill.data(), count * k * ff[0] * 4), gu = g.adopt(fill.data(), count * k * ff[1] * 4);
                        const Backend::Routing routing{{ib.get(), first * k}, {tb.get(), first * k}, k, n_expert};
                        launches(g);
                        g.matmul_experts({Projection{{type, {wb[0].get(), 0}}, {ga.get(), 0}, ff[0]}, Projection{{type, {wb[1].get(), 0}}, {gu.get(), 0}, ff[1]}}, {xb.get(), first * nin}, nin, count, routing,
                                         RowRuns{runs.data(), runs.size()}, dt);
                        ran.clear();
                        for (const auto& [kn, v] : launches(g)) ran += (ran.empty() ? "" : "+") + kn;
                        return std::array<std::vector<float>, 2>{read(g, *ga, count * k * ff[0]), read(g, *gu, count * k * ff[1])};
                    };
                    std::string ran;
                    const auto whole = call(0, rows, 0, ran);
                    require(tiled_alone(ran), at + ": a " + std::to_string(rows) + "-row prompt takes the grouping, " + tile + " and " + narrow + " alone, ran " + ran);
                    call(0, from - 1, 0, ran);
                    require(ran.find("routed_tile") == std::string::npos, at + ": a " + std::to_string(from - 1) + "-row prompt takes the row builds, ran " + ran);
                    call(0, from, 0, ran);
                    require(ran.find(tile) != std::string::npos, at + ": a " + std::to_string(from) + "-row prompt takes " + tile + ", ran " + ran);
                    // An extent off a multiple of 4, which the twin map pads to the next odd multiple of 4.
                    const auto odd = call(0, from + 1, 0, ran);
                    require(ran.find(tile) != std::string::npos, at + ": a " + std::to_string(from + 1) + "-row prompt takes " + tile + ", ran " + ran);
                    // The mixed call's 3 generated rows ahead of a padded tile region give an odd total at an odd block count, 195 + 260*65 = 17095 at Q4_0 2080, whose block-int16 sums plane is only 4-aligned.
                    const auto half = call(half_rows, half_rows, 0, ran), mixed = call(0, rows, 3, ran);
                    bool same_bits = true, unwritten = true, decode_alone = true, near = true;
                    auto one = g.alloc(256 * 4);
                    for (int p = 0; p < 2; ++p)
                        for (size_t e = 0; e < rows * k; ++e) {
                            const float* got = whole[p].data() + e * ff[p];
                            if (idv[e] == Backend::kNoExpert) {
                                unwritten = unwritten && std::all_of(got, got + ff[p], [](float v) { return v == 7.0f; }) &&
                                            std::all_of(mixed[p].begin() + e * ff[p], mixed[p].begin() + (e + 1) * ff[p], [](float v) { return v == 7.0f; });
                                continue;
                            }
                            if (e >= half_rows * k) same_bits = same_bits && !std::memcmp(got, half[p].data() + (e - half_rows * k) * ff[p], ff[p] * 4);
                            if (e < (from + 1) * k) same_bits = same_bits && !std::memcmp(got, odd[p].data() + e * ff[p], ff[p] * 4);
                            if (e >= 3 * k) same_bits = same_bits && !std::memcmp(got, mixed[p].data() + e * ff[p], ff[p] * 4);
                            if (e >= 3 * k && e % 7) continue;
                            g.matmul({type, {wb[p].get(), idv[e] * quant::row_bytes(type, nin, ff[p]) / 4}}, {xb.get(), e / k * nin}, {one.get(), 0}, nin, ff[p], 1, {}, dt);
                            const std::vector<float> o = read(g, *one, ff[p]);
                            if (e < 3 * k) decode_alone = decode_alone && !std::memcmp(mixed[p].data() + e * ff[p], o.data(), ff[p] * 4);
                            for (size_t j = 0; j < ff[p]; ++j) {
                                double mag = 0.0;
                                const float* wr = wd[p].data() + (idv[e] * ff[p] + j) * nin;
                                for (size_t i = 0; i < nin; ++i) mag += std::fabs(double(wr[i]) * x[e / k * nin + i]);
                                const double bound = double(nblk + 8) * double(FLT_EPSILON) * mag;
                                if (near && std::fabs(double(got[j]) - o[j]) > bound) std::fprintf(stderr, "  entry %zu output %zu: %.9g against %.9g, bound %.3g\n", e, j, got[j], o[j], bound);
                                near = near && std::fabs(double(got[j]) - o[j]) <= bound;
                            }
                        }
                    require(same_bits, at + ": each prompt entry has the same bits at extents " + std::to_string(rows) + ", " + std::to_string(half_rows) + " and " + std::to_string(from + 1) + " and beside generated rows");
                    require(decode_alone, at + ": generated rows beside a prompt give their one-column bits");
                    require(unwritten, at + ": an entry without an expert is not written");
                    require(near, at + ": each sampled tile entry is within (blocks + 8) F32 epsilons of the sum of |w*x| of its one-column product");
                }
                // The down, the gate's matrix over a row an entry.
                {
                    const size_t from = tile_from_at(type, dt, true), half_rows = std::max<size_t>(from, 40), rows = 2 * half_rows, nout = ff[0];
                    require(rows <= most, "experts_tile holds the rows of every build's down extent");
                    const std::vector<uint32_t> idv = tile_ids(half_rows, k);
                    require(tile_groups_cover(idv, half_rows, k, n_expert), at + ", down: the calls at extents " + std::to_string(rows) + " and " + std::to_string(half_rows) + " hold groups of both builds, of 1, 7, 15, 16 and 17 entries wide at twice the extent");
                    auto ib = g.adopt(idv.data(), idv.size() * 4);
                    const std::vector<float> xd = row_classes::values(rows * k * nin, uint32_t(nin + 43));
                    auto xdb = g.adopt(xd.data(), xd.size() * 4);
                    // A down over rows [first, first + count) onto zeros, as one prompt of that extent or with `decode` generated rows ahead of it.
                    const auto call = [&](size_t first, size_t count, size_t decode, std::string& ran) {
                        std::vector<RowRun> runs;
                        for (size_t r = 0; r < decode; ++r) runs.push_back({r + 1, 1});
                        if (count > decode) runs.push_back({count, count - decode});
                        const std::vector<float> zeros(count * nout, 0.0f);
                        auto yb = g.adopt(zeros.data(), zeros.size() * 4);
                        const Backend::Routing routing{{ib.get(), first * k}, {tb.get(), first * k}, k, n_expert};
                        launches(g);
                        g.matmul_experts_add({type, {wb[0].get(), 0}}, {xdb.get(), first * k * nin}, {yb.get(), 0}, nin, nout, count, routing, RowRuns{runs.data(), runs.size()}, dt);
                        ran.clear();
                        for (const auto& [kn, v] : launches(g)) ran += (ran.empty() ? "" : "+") + kn;
                        return read(g, *yb, count * nout);
                    };
                    std::string ran;
                    const auto whole = call(0, rows, 0, ran);
                    require(tiled_alone(ran), at + ", down: a " + std::to_string(rows) + "-row prompt takes the grouping, " + tile + " and " + narrow + " alone, ran " + ran);
                    call(0, from - 1, 0, ran);
                    require(ran.find("routed_tile") == std::string::npos, at + ", down: a " + std::to_string(from - 1) + "-row prompt takes the row builds, ran " + ran);
                    call(0, from, 0, ran);
                    require(ran.find(tile) != std::string::npos, at + ", down: a " + std::to_string(from) + "-row prompt takes " + tile + ", ran " + ran);
                    const auto half = call(half_rows, half_rows, 0, ran), mixed = call(0, rows, 3, ran), by_rows = call(0, rows, rows, ran);
                    bool same_bits = true, decode_rows = true, near = true;
                    for (size_t r = 0; r < rows; ++r) {
                        const float* got = whole.data() + r * nout;
                        if (r >= half_rows) same_bits = same_bits && !std::memcmp(got, half.data() + (r - half_rows) * nout, nout * 4);
                        if (r >= 3) same_bits = same_bits && !std::memcmp(got, mixed.data() + r * nout, nout * 4);
                        else decode_rows = decode_rows && !std::memcmp(mixed.data() + r * nout, by_rows.data() + r * nout, nout * 4);
                        if (r >= 3 && r % 7) continue;
                        for (size_t j = 0; j < nout; ++j) {
                            double mag = 0.0;
                            for (size_t s = 0; s < k; ++s) {
                                const uint32_t id = idv[r * k + s];
                                if (id == Backend::kNoExpert) continue;
                                const float* wr = wd[0].data() + (id * nout + j) * nin;
                                for (size_t i = 0; i < nin; ++i) mag += std::fabs(double(wts[r * k + s]) * double(wr[i]) * xd[(r * k + s) * nin + i]);
                            }
                            const float o = by_rows[r * nout + j];
                            const double bound = double(nblk + 8 + k) * double(FLT_EPSILON) * mag;
                            if (near && std::fabs(double(got[j]) - o) > bound) std::fprintf(stderr, "  down row %zu output %zu: %.9g against %.9g, bound %.3g\n", r, j, got[j], o, bound);
                            near = near && std::fabs(double(got[j]) - o) <= bound;
                        }
                    }
                    require(same_bits, at + ", down: each prompt row has the same bits at extents " + std::to_string(rows) + " and " + std::to_string(half_rows) + " and beside generated rows");
                    require(decode_rows, at + ", down: generated rows beside a prompt give the row builds' bits");
                    require(near, at + ", down: each sampled tile row is within (blocks + 8 + k) F32 epsilons of the weighted sum of |w*x| of the row builds' row");
                }
            }
        }
    }

    // The routed builds by group: experts of 1 to 33 entries each side of every build's range and every narrow quad, an empty expert between, every type in f16 and int8.
    // Every entry is the entry routed alone, bit for bit, at inner sizes ending the stream's last chunk full, half full, short and with a one-block tail, and MXFP4 at 65 blocks a row (docs/TESTS.md).
    void experts_quads(Backend& g) {
        const size_t n_expert = 24, ff = 256, sizes[] = {1, 3, 4, 5, 8, 9, 12, 13, 16, 17, 32, 33};
        std::vector<std::pair<uint32_t, size_t>> kinds;
        for (uint32_t type : {quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K})
            for (size_t nin : {size_t(2560), size_t(768)}) kinds.push_back({type, nin});
        for (uint32_t type : {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_MXFP4})
            for (size_t nin : {size_t(2560), size_t(640), size_t(800)}) kinds.push_back({type, nin});
        kinds.push_back({quant::GGML_TYPE_MXFP4, 2080});
        for (const auto& [type, nin] : kinds) {
            const std::string name = quant::storage_type(type)->name;
            const std::vector<uint8_t> w = type == quant::GGML_TYPE_Q8_0 ? q8_matrix(nin, n_expert * ff, uint32_t(nin + 3)) : typed_matrix(type, nin, n_expert * ff, uint32_t(type * 19 + nin));
            auto wb = g.adopt(w.data(), w.size());
            for (Dtype dt : {Dtype::f16, Dtype::int8}) {
                const std::string at = name + " " + std::to_string(nin) + ", " + dtype_name(dt);
                const size_t rows = std::max<size_t>(tile_from_at(type, dt, false), 160);
                const std::vector<float> x = row_classes::values(rows * nin, uint32_t(nin + 47)), wts(rows, 1.0f);
                auto xb = g.adopt(x.data(), x.size() * 4), tb = g.adopt(wts.data(), wts.size() * 4);
                // Expert 2i + 1 takes sizes[i] entries, interleaved with the others' over the rows; the rest have no expert.
                std::vector<uint32_t> ids(rows, Backend::kNoExpert);
                std::vector<std::vector<size_t>> of(std::size(sizes));
                for (size_t r = 0, left = 1; left;) {
                    left = 0;
                    for (size_t i = 0; i < std::size(sizes); ++i)
                        if (of[i].size() < sizes[i]) {
                            ids[r] = uint32_t(2 * i + 1);
                            of[i].push_back(r++);
                            left = 1;
                        }
                }
                const auto call = [&](const std::vector<uint32_t>& v, std::string& ran) {
                    auto ib = g.adopt(v.data(), v.size() * 4);
                    const std::vector<float> fill(rows * ff, 7.0f);
                    auto yb = g.adopt(fill.data(), fill.size() * 4);
                    const RowRun run{rows, rows};
                    launches(g);
                    g.matmul_experts({Projection{{type, {wb.get(), 0}}, {yb.get(), 0}, ff}}, {xb.get(), 0}, nin, rows, {{ib.get(), 0}, {tb.get(), 0}, 1, n_expert}, RowRuns{&run, 1}, dt);
                    ran.clear();
                    for (const auto& [kn, n] : launches(g)) ran += (ran.empty() ? "" : "+") + kn;
                    return read(g, *yb, rows * ff);
                };
                std::string ran;
                const std::vector<float> all = call(ids, ran);
                const bool narrow = dt == Dtype::int8 && type != quant::GGML_TYPE_Q8_0, streams = dt == Dtype::int8 && type == quant::GGML_TYPE_MXFP4;
                require(ran.find(narrow ? "_routed_stream16_8" : "_routed_narrow") != std::string::npos && ran.find("_routed_tile") != std::string::npos && (ran.find("_routed_stream8") != std::string::npos) == (dt == Dtype::int8),
                        at + ": a " + std::to_string(rows) + "-row prompt takes the routed tile and the narrow build, the stream in int8 only, ran " + ran);
                require(type == quant::GGML_TYPE_Q8_0 || (ran.find(streams ? "_routed_stream32_8" : "_routed_mid") != std::string::npos && (ran.find("_routed_thin") != std::string::npos) == (dt == Dtype::f16 && type == quant::GGML_TYPE_MXFP4)),
                        at + ": a K-type's groups of 17 to 32 entries take its mid build, and MXFP4's of up to 8 its thin build in f16, ran " + ran);
                bool same = true;
                for (size_t i = 0; i < std::size(sizes); ++i)
                    for (size_t r : of[i]) {
                        std::vector<uint32_t> one(rows, Backend::kNoExpert);
                        one[r] = ids[r];
                        const std::vector<float> alone = call(one, ran);
                        same = same && !std::memcmp(all.data() + r * ff, alone.data() + r * ff, ff * 4);
                    }
                require(same, at + ": every entry of experts of 1 to 33 entries is the entry routed alone, bit for bit");
            }
        }
    }

    // Which routed build computes a group: each build launched alone over a sentinel output writes the groups its range holds and no other, each the bits of the call with every build.
    // A group of up to 8 takes the stream in int8 and MXFP4's thin build in f16, up to 16 the narrow tile or in int8 the 16-row stream (not Q8_0), up to 32 the mid tile or MXFP4's int8 32-row stream, past that the 64-row tile.
    void routed_builds_witness(Backend& g) {
        // The drawn sizes 8/9, 16/17 and 32/33 sit on each range's edges.
        const size_t n_expert = 20, ff = 128, sizes[] = {4, 5, 8, 9, 16, 17, 32, 33}, rows = 256;
        const char* builds[] = {"thin", "stream8", "stream16", "stream32", "narrow", "mid", "tile"};
        const uint32_t sentinel = 0x7fc0deadu;
        for (const auto& [type, nin] : std::vector<std::pair<uint32_t, size_t>>{{quant::GGML_TYPE_MXFP4, 2048}, {quant::GGML_TYPE_Q4_K, 2560}, {quant::GGML_TYPE_Q5_K, 2560}, {quant::GGML_TYPE_Q6_K, 2560}, {quant::GGML_TYPE_Q4_0, 1024}, {quant::GGML_TYPE_Q4_1, 1024}, {quant::GGML_TYPE_Q5_0, 1024}, {quant::GGML_TYPE_Q5_1, 1024}, {quant::GGML_TYPE_IQ4_NL, 1024}, {quant::GGML_TYPE_Q8_0, 1024}}) {
            const bool ktype = type != quant::GGML_TYPE_Q8_0;
            const std::string name = quant::storage_type(type)->name;
            const std::vector<uint8_t> w = ktype ? typed_matrix(type, nin, n_expert * ff, uint32_t(type * 7 + 3)) : q8_matrix(nin, n_expert * ff, 5);
            auto wb = g.adopt(w.data(), w.size());
            const std::vector<float> x = row_classes::values(rows * nin, 61), wts(rows, 1.0f);
            auto xb = g.adopt(x.data(), x.size() * 4), tb = g.adopt(wts.data(), wts.size() * 4);
            // Expert 2i + 1 takes sizes[i] consecutive rows.
            std::vector<uint32_t> ids(rows, Backend::kNoExpert);
            std::vector<std::pair<size_t, size_t>> span(std::size(sizes));
            for (size_t i = 0, r = 0; i < std::size(sizes); r += sizes[i], ++i) {
                span[i] = {r, sizes[i]};
                for (size_t j = 0; j < sizes[i]; ++j) ids[r + j] = uint32_t(2 * i + 1);
            }
            auto ib = g.adopt(ids.data(), ids.size() * 4);
            for (Dtype dt : {Dtype::f16, Dtype::int8}) {
                require(tile_from_at(type, dt, false) <= rows, name + " " + dtype_name(dt) + ": a prompt of " + std::to_string(rows) + " rows takes the routed tile");
                const auto expect = [&](size_t n) -> std::string {
                    const bool narrow = dt == Dtype::int8 && ktype, streams = dt == Dtype::int8 && type == quant::GGML_TYPE_MXFP4;
                    return n <= 8 ? (dt == Dtype::int8 ? "stream8" : type == quant::GGML_TYPE_MXFP4 ? "thin" : "narrow") : n <= 16 ? (narrow ? "stream16" : "narrow") : n <= 32 && ktype ? (streams ? "stream32" : "mid") : "tile";
                };
                const auto call = [&](const std::string& only) {
                    const std::vector<uint32_t> fill(rows * ff, sentinel);
                    auto yb = g.adopt(fill.data(), fill.size() * 4);
                    const RowRun run{rows, rows};
                    route_only(g, only.empty() ? only : "_routed_" + only);
                    g.matmul_experts({Projection{{type, {wb.get(), 0}}, {yb.get(), 0}, ff}}, {xb.get(), 0}, nin, rows, {{ib.get(), 0}, {tb.get(), 0}, 1, n_expert}, RowRuns{&run, 1}, dt);
                    route_only(g, "");
                    return read(g, *yb, rows * ff);
                };
                const std::vector<float> all = call("");
                for (const std::string build : builds) {
                    const std::vector<float> one = call(build);
                    for (size_t i = 0; i < std::size(sizes); ++i) {
                        const auto [r0, n] = span[i];
                        bool written = true, kept = true;
                        for (size_t v = r0 * ff; v < (r0 + n) * ff; ++v) {
                            uint32_t bits;
                            std::memcpy(&bits, &one[v], 4);
                            written = written && bits != sentinel && !std::memcmp(&one[v], &all[v], 4);
                            kept = kept && bits == sentinel;
                        }
                        const bool mine = expect(n) == build;
                        require(mine ? written : kept, name + " " + dtype_name(dt) + ": a group of " + std::to_string(n) + " entries is " + (mine ? "" : "not ") + "computed by the " + build +
                                                           " build launched alone" + (mine ? ", with the bits of the call of every build" : ""));
                    }
                }
            }
        }
    }

    // matmul_experts_add: Y plus each row's slots, product by product in slot order, the products those of matmul_experts over a row an entry.
    void experts_add(Backend& g) {
        const size_t n_expert = 8, k = 4, nin = 640, nout = 2560;
        for (uint32_t type : {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_F32}) {
            if (nin % quant::storage_type(type)->block_size) continue;
            std::vector<uint8_t> w;
            if (type == quant::GGML_TYPE_F32) {
                const std::vector<float> v = row_classes::values(nin * n_expert * nout, 21, 0.05f);
                w.assign((const uint8_t*)v.data(), (const uint8_t*)v.data() + v.size() * 4);
            } else {
                w = type == quant::GGML_TYPE_Q8_0 ? q8_matrix(nin, n_expert * nout, 5) : typed_matrix(type, nin, n_expert * nout, uint32_t(type + 3));
            }
            auto wb = g.adopt(w.data(), w.size());
            for (size_t rows : {1, 7}) {
                const std::vector<float> x = row_classes::values(rows * k * nin, uint32_t(rows + 2)), y0 = row_classes::values(rows * nout, uint32_t(rows + 4));
                std::vector<uint32_t> idv(rows * k);
                for (size_t e = 0; e < idv.size(); ++e) idv[e] = uint32_t((e * 3 + 1) % n_expert);
                idv[k - 1] = Backend::kNoExpert;
                idv[0] = rows > 1 ? Backend::kNoExpert : idv[0];
                const std::vector<float> wts = row_classes::values(rows * k, uint32_t(rows + 6), 0.5f);
                auto xb = g.adopt(x.data(), x.size() * 4), ib = g.adopt(idv.data(), idv.size() * 4), tb = g.adopt(wts.data(), wts.size() * 4);
                const Backend::Routing routing{{ib.get(), 0}, {tb.get(), 0}, k, n_expert};
                for (Dtype dt : {Dtype::f32, Dtype::f16, Dtype::int8}) {
                    if (type == quant::GGML_TYPE_F32 && dt != Dtype::f32) continue;
                    auto yb = g.adopt(y0.data(), y0.size() * 4), parts = g.alloc(rows * k * nout * 4);
                    g.matmul_experts({Projection{{type, {wb.get(), 0}}, {parts.get(), 0}, nout}}, {xb.get(), 0}, nin, rows * k, Backend::Routing{routing.ids, routing.weights, 1, n_expert}, {}, dt);
                    g.matmul_experts_add({type, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, rows, routing, {}, dt);
                    const std::vector<float> p = read(g, *parts, rows * k * nout), got = read(g, *yb, rows * nout);
                    bool ok = true;
                    for (size_t r = 0; r < rows; ++r)
                        for (size_t o = 0; o < nout; ++o) {
                            float s = 0.0f;
                            for (size_t j = 0; j < k; ++j) {
                                if (idv[r * k + j] == Backend::kNoExpert) continue;
                                volatile float product = wts[r * k + j] * p[(r * k + j) * nout + o];
                                s = j == 0 ? product : s + product;
                            }
                            volatile float sum = y0[r * nout + o] + s;
                            if (ok && std::memcmp((const void*)&sum, &got[r * nout + o], 4))
                                std::fprintf(stderr, "  row %zu output %zu: %.9g against %.9g, y %.9g, first slot %.9g\n", r, o, got[r * nout + o], (float)sum, y0[r * nout + o], p[r * k * nout + o]);
                            ok = ok && !std::memcmp((const void*)&sum, &got[r * nout + o], 4);
                        }
                    require(ok, std::string(quant::storage_type(type)->name) + " routed down of " + std::to_string(rows) + " rows, " + dtype_name(dt) + ": Y plus the slots in order, bit for bit");
                }
            }
        }
    }
};

} // namespace moe_checks
