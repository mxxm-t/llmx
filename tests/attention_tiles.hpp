#pragma once
// The prompt attention tile's witnesses shared by the GPU backends' tests: each build by its launch, each output against a double softmax over the f16 cache's values, and a row alone its bits in the tile.
#include "backends/backend.hpp"
#include "backends/kv_storage.hpp"
#include "core/fp16.hpp"
#include "model/kv_cache.hpp"
#include "row_classes.hpp"
#include "twin_checks.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>
namespace attention_tiles {
using backend::Backend;
using backend::KVType;
using backend::KVView;
using twin_checks::Launches;
using twin_checks::Require;
using twin_checks::read;

// Causal attention of `rows` query rows after `hist` tokens, n_head heads over n_head_kv KV heads, in doubles over the given (already f16-rounded) keys and values.
inline std::vector<float> host(const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, size_t hist, size_t rows, int n_head, int n_head_kv,
                               int dim) {
    std::vector<float> o(rows * n_head * dim);
    const double scale = 1.0 / std::sqrt((double)dim);
    for (size_t b = 0; b < rows; ++b)
        for (int h = 0; h < n_head; ++h) {
            const int kvh = h / (n_head / n_head_kv);
            const size_t n = hist + b + 1;
            std::vector<double> s(n);
            double mx = -1e300;
            for (size_t t = 0; t < n; ++t) {
                double d = 0.0;
                for (int e = 0; e < dim; ++e) d += (double)q[(b * n_head + h) * dim + e] * k[(t * n_head_kv + kvh) * dim + e];
                s[t] = d * scale;
                mx = std::max(mx, s[t]);
            }
            double sum = 0.0;
            for (double& x : s) sum += (x = std::exp(x - mx));
            for (int e = 0; e < dim; ++e) {
                double a = 0.0;
                for (size_t t = 0; t < n; ++t) a += s[t] * v[(t * n_head_kv + kvh) * dim + e];
                o[(b * n_head + h) * dim + e] = (float)(a / sum);
            }
        }
    return o;
}

// Attention of `rows` prompt rows after `hist` tokens through the backend, the view's extent `rows`, from keys and values of (hist + rows) tokens.
inline std::vector<float> run(Backend& g, const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, size_t hist, size_t rows, int n_head,
                              int n_head_kv, int dim, bool alone) {
    const size_t qw = (size_t)n_head * dim;
    auto kb = g.adopt(k.data(), k.size() * 4), vb = g.adopt(v.data(), v.size() * 4), qb = g.adopt(q.data(), q.size() * 4);
    auto ob = g.alloc(rows * qw * 4);
    auto st = g.kv_alloc(1, n_head_kv, dim, std::max<size_t>(1024, hist + rows), KVType::f16, KVType::f16);
    infer::BlockPool pool;
    pool.configure(st->max_blocks());
    infer::KVSequence s(&pool, g.kv_layout().block_tokens);
    s.prepare(hist + rows);
    KVView all = s.view(st.get());
    g.kv_write(0, &all, 1, {kb.get(), 0}, {vb.get(), 0});
    if (!alone) {
        KVView view = all;
        view.length = hist;
        view.nq = rows;
        view.extent = rows;
        g.attention({qb.get(), 0}, 0, &view, 1, {ob.get(), 0}, n_head, n_head_kv, dim);
    } else {
        for (size_t b = 0; b < rows; ++b) {
            KVView one = all;
            one.length = hist + b;
            one.nq = 1;
            one.extent = rows;   // the prompt's extent, so the row alone keeps its row class
            g.attention({qb.get(), b * qw}, 0, &one, 1, {ob.get(), b * qw}, n_head, n_head_kv, dim);
        }
    }
    std::vector<float> out = read(g, *ob, rows * qw);
    s.abort();
    return out;
}

// Heads 128 and 256 wide, 40 rows (a full tile and a part one) after histories across the tile's key steps and the cache's blocks.
// Four heads over two KV heads, so two query heads read each KV head.
inline void check(Backend& g, const Launches& launches, const Require& require) {
    const int n_head = 4;
    const size_t rows = 40;
    for (const int dim : {128, 256})
        for (const size_t hist : {size_t(0), size_t(37), size_t(300)}) {
            const size_t n = hist + rows, kvw = 2 * (size_t)dim;
            std::vector<float> k = row_classes::values(n * kvw, 61), v = row_classes::values(n * kvw, 62);
            for (float& x : k) x = f16_to_f32(f32_to_f16(x));
            for (float& x : v) x = f16_to_f32(f32_to_f16(x));
            const std::vector<float> q = row_classes::values(rows * n_head * dim, 63, 4.0f);
            const std::string at = std::to_string(dim) + " wide after " + std::to_string(hist) + " tokens";
            const std::string build = dim == 256 ? "llmx_attention_tile256" : "llmx_attention_tile";
            launches(g);
            const std::vector<float> got = run(g, q, k, v, hist, rows, n_head, 2, dim, false);
            require(launches(g).count(build) == 1, "a prompt of heads " + at + " takes " + build);
            const std::vector<float> want = host(q, k, v, hist, rows, n_head, 2, dim);
            double worst = 0.0, top = 0.0;
            for (float x : v) top = std::max(top, (double)std::fabs(x));
            for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs((double)got[i] - want[i]) / (top * 1e-5));
            char what[200];
            std::snprintf(what, sizeof(what), "the tile of heads %s is within 1e-5 of the values' largest magnitude of a double softmax, worst %.3f of it", at.c_str(), worst);
            require(worst <= 1.0, what);
            require(std::memcmp(run(g, q, k, v, hist, rows, n_head, 2, dim, true).data(), got.data(), got.size() * 4) == 0,
                    "every row of a tile of heads " + at + " alone is its row in the prompt, bit for bit");
        }
}

// 256-wide heads after 4k and 16k tokens: the error against a double softmax stays near an F32 sum's, and a row alone has its bits (docs/STATUS.md, the 1a fold).
// The drift is the mean over the rows and heads of <error, reference> / <reference, reference>, which a one-sided rounding error moves and a random one does not.
inline void check_long(Backend& g, const Launches& launches, const Require& require) {
    const int n_head = 2, n_head_kv = 1, dim = 256;
    const size_t rows = 40;   // check()'s rows, at least the profile's attention_tile_rows, so the prompt takes the tile
    for (const size_t hist : {size_t(4096 + 37), size_t(16384)}) {
        const size_t n = hist + rows, kvw = (size_t)dim;
        std::vector<float> k = row_classes::values(n * kvw, 71), v = row_classes::values(n * kvw, 72);
        for (float& x : k) x = f16_to_f32(f32_to_f16(x));
        for (float& x : v) x = f16_to_f32(f32_to_f16(x));
        const std::vector<float> q = row_classes::values(rows * n_head * dim, 73, 4.0f);
        const std::string at = std::to_string(hist) + " tokens";
        launches(g);
        const std::vector<float> got = run(g, q, k, v, hist, rows, n_head, n_head_kv, dim, false);
        const std::map<std::string, size_t> ran = launches(g);
        const bool tiled = ran.count("llmx_attention_tile256") && ran.at("llmx_attention_tile256") == 1;
        if (!tiled)
            for (const auto& [name, count] : ran) std::printf("  after %s launched %s x%zu\n", at.c_str(), name.c_str(), count);
        require(tiled, "a prompt of heads 256 wide after " + at + " takes llmx_attention_tile256");
        const std::vector<float> want = host(q, k, v, hist, rows, n_head, n_head_kv, dim);
        double err2 = 0.0, ref2 = 0.0, drift = 0.0;
        for (size_t p = 0; p < rows * n_head; ++p) {
            double e2 = 0.0, r2 = 0.0, er = 0.0;
            for (int e = 0; e < dim; ++e) {
                const double r = want[p * dim + e], x = (double)got[p * dim + e] - r;
                e2 += x * x, r2 += r * r, er += x * r;
            }
            err2 += e2, ref2 += r2, drift += er / r2;
        }
        drift /= (double)(rows * n_head);
        const double rel = std::sqrt(err2 / ref2);
        std::printf("  after %s rms error %.3e, drift %.3e\n", at.c_str(), rel, drift);
        char what[240];
        std::snprintf(what, sizeof(what), "the tile of heads 256 wide after %s has rms error %.2e of the reference's rms, within 1e-6", at.c_str(), rel);
        require(rel <= 1e-6, what);
        std::snprintf(what, sizeof(what), "the tile of heads 256 wide after %s drifts %.2e of the reference, within 2e-8", at.c_str(), drift);
        require(std::fabs(drift) <= 2e-8, what);
        require(std::memcmp(run(g, q, k, v, hist, rows, n_head, n_head_kv, dim, true).data(), got.data(), got.size() * 4) == 0,
                "every row of a tile of heads 256 wide after " + at + " alone is its row in the prompt, bit for bit");
    }
}
} // namespace attention_tiles
