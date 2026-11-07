#pragma once
// DeepSeek-V4.1's layer ops (docs/DEEPSEEK4.md, section 4.2) on a device against the CPU backend, shared by the device backends' tests.
// Exact where the arithmetic is the CPU's without a multiply-add (the format codes and the selections); within a stated bound where an exponential, a root, a reduction order or a multiply-add differs, since a device need not fuse one.
#include "qwen35_checks.hpp"
#include "quant/model_formats.hpp"

#include <cstring>
#include <deque>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace ds4 {
using q35::Pair;
using q35::close;
using q35::require;
using q35::uniform;

const uint32_t kNone = 0xffffffffu;

// An input on both backends whose floats outlive the check, since the CPU reads an adopted buffer in place.
Pair::In held(Pair& p, std::vector<float> v) {
    static std::deque<std::vector<float>> kept;
    kept.push_back(std::move(v));
    return p.in(kept.back());
}

std::vector<float> as_floats(const std::vector<uint32_t>& v) {
    std::vector<float> f(v.size());
    if (!v.empty()) std::memcpy(f.data(), v.data(), v.size() * sizeof(float));
    return f;
}

// Within rel of the CPU relative to 1 + |cpu|, an infinity only where the CPU has the same one.
size_t within_rel(const std::vector<float>& a, const std::vector<float>& b, double rel, const std::string& what) {
    require(a.size() == b.size(), what.c_str());
    for (size_t i = 0; i < a.size(); ++i) {
        const bool inf = std::isinf(a[i]) || std::isinf(b[i]);
        if (inf ? a[i] != b[i] : !(std::fabs(double(a[i]) - b[i]) <= rel * (1.0 + std::fabs(double(a[i]))))) {
            std::fprintf(stderr, "  [%zu] cpu %.9g device %.9g\n", i, a[i], b[i]);
            require(false, what.c_str());
        }
    }
    return a.size();
}

// Bit-for-bit equality, NaN and infinities included.
size_t same(const std::vector<float>& a, const std::vector<float>& b, const std::string& what) {
    require(a.size() == b.size(), what.c_str());
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(float))) {
            uint32_t x, y;
            std::memcpy(&x, &a[i], 4);
            std::memcpy(&y, &b[i], 4);
            std::fprintf(stderr, "  [%zu] cpu %.9g (%08x) device %.9g (%08x)\n", i, a[i], x, b[i], y);
            require(false, what.c_str());
        }
    return a.size();
}

// The ops a device computes in the CPU's order, which it must then meet bit for bit; any other is held to its bound.
using Exact = std::set<std::string>;
size_t meet(const Exact& exact, const char* op, const std::vector<float>& a, const std::vector<float>& b, double rel, const std::string& what, bool infinities = false) {
    if (exact.count(op)) return same(a, b, std::string(op) + " differs from the CPU's bits");
    return infinities ? within_rel(a, b, rel, what) : close(a, b, rel, what.c_str());
}

size_t check_mhc(Pair& p, const Exact& exact) {
    size_t values = 0;
    for (size_t dim : {40, 256}) {
        const size_t rows = 5, hc = 4, mix = (hc + 2) * hc;
        Pair::In mi = held(p, uniform(rows * mix, 1, -2.0f, 2.0f)), xi = held(p, uniform(rows * hc * dim, 2, -1.5f, 1.5f));
        Pair::In si = held(p, std::vector<float>{0.7f, 1.3f, 0.9f}), bi = held(p, uniform(mix, 3, -0.5f, 0.5f));
        Pair::Out pre = p.out(rows * hc), post = p.out(rows * hc), comb = p.out(rows * hc * hc);
        p.cpu.hc_coeffs(pre.cs(), post.cs(), comb.cs(), mi.cs(), xi.cs(), si.cs(), bi.cs(), rows, hc, dim, 1e-6f, 1e-6f, 20);
        p.dev.hc_coeffs(pre.vs(), post.vs(), comb.vs(), mi.vs(), xi.vs(), si.vs(), bi.vs(), rows, hc, dim, 1e-6f, 1e-6f, 20);
        auto a = p.results(pre), b = p.results(post), c = p.results(comb);
        values += close(a.first, a.second, 1e-5, "hc_coeffs pre differs beyond 1e-5");
        values += close(b.first, b.second, 1e-5, "hc_coeffs post differs beyond 1e-5");
        values += close(c.first, c.second, 1e-5, "hc_coeffs comb differs beyond 1e-5");

        // The collapse and the expansion read the CPU's coefficients, so each meets the CPU on its own.
        Pair::In pi = held(p, a.first), wi = held(p, uniform(dim, 4, 0.5f, 1.5f)), fi = held(p, uniform(rows * dim, 5)), qi = held(p, b.first), ci = held(p, c.first);
        Pair::Out u = p.out(rows * dim);
        p.cpu.hc_collapse_norm(u.cs(), xi.cs(), pi.cs(), wi.cs(), rows, hc, dim, 1e-6f);
        p.dev.hc_collapse_norm(u.vs(), xi.vs(), pi.vs(), wi.vs(), rows, hc, dim, 1e-6f);
        auto ur = p.results(u);
        values += meet(exact, "hc_collapse_norm", ur.first, ur.second, 1e-5, "hc_collapse_norm differs beyond 1e-5");
        Pair::Out e = p.out(rows * hc * dim);
        p.cpu.hc_expand(e.cs(), fi.cs(), xi.cs(), qi.cs(), ci.cs(), rows, hc, dim);
        p.dev.hc_expand(e.vs(), fi.vs(), xi.vs(), qi.vs(), ci.vs(), rows, hc, dim);
        auto er = p.results(e);
        values += meet(exact, "hc_expand", er.first, er.second, 1e-6, "hc_expand differs beyond 1e-6");
        // In place over the stream gives the same bits.
        Pair::Out x2 = p.out(rows * hc * dim);
        const std::vector<float> x = uniform(rows * hc * dim, 2, -1.5f, 1.5f);
        p.cpu.write(*x2.c, 0, x.data(), x.size() * 4);
        p.dev.write(*x2.v, 0, x.data(), x.size() * 4);
        p.cpu.hc_expand(x2.cs(), fi.cs(), x2.cs(), qi.cs(), ci.cs(), rows, hc, dim);
        p.dev.hc_expand(x2.vs(), fi.vs(), x2.vs(), qi.vs(), ci.vs(), rows, hc, dim);
        auto xr = p.results(x2);
        values += same(xr.first, er.first, "hc_expand in place differs from the CPU's out of place");
        values += same(xr.second, er.second, "hc_expand in place on the device differs from the device's out of place");
    }
    return values;
}

// Inputs reaching every exponent, zeros of both signs and subnormals, in blocks whose peaks set scales from tiny to large.
size_t check_formats(Pair& p) {
    std::vector<float> x;
    uint32_t s = 7;
    for (int blk = 0; blk < 96; ++blk) {
        const float peak = std::ldexp(1.0f, blk % 48 * 5 - 130);
        for (int i = 0; i < 32; ++i) {
            s = s * 1664525u + 1013904223u;
            float v = peak * (float(s >> 8) / float(1u << 24)) * ((s & 1) ? -1.0f : 1.0f);
            if (i == 3) v = 0.0f;
            if (i == 4) v = -0.0f;
            if (i == 5) v = std::ldexp(1.0f, -140);
            x.push_back(v);
        }
    }
    size_t values = 0;
    for (backend::ModelFormat f : {backend::ModelFormat::fp8_power, backend::ModelFormat::fp4_power, backend::ModelFormat::fp4_e4m3}) {
        Pair::Out o = p.out(x.size());
        p.cpu.write(*o.c, 0, x.data(), x.size() * 4);
        p.dev.write(*o.v, 0, x.data(), x.size() * 4);
        p.cpu.model_format_qdq(o.cs(), x.size(), f);
        p.dev.model_format_qdq(o.vs(), x.size(), f);
        auto r = p.results(o);
        values += same(r.first, r.second, "model_format_qdq differs from the CPU in format " + std::to_string(int(f)));
    }
    return values;
}

size_t check_selections(Pair& p) {
    size_t values = 0;
    const float inf = std::numeric_limits<float>::infinity();
    for (size_t n : {7, 300, 5000}) {
        const size_t rows = 4;
        std::vector<float> s = uniform(rows * n, 11 + uint32_t(n), -3.0f, 3.0f);
        // Five levels in every other row, so ties fall at every cut; -inf and both zeros among them, and no NaN, which the selections refuse (backend.hpp).
        for (size_t i = 0; i < s.size(); ++i) {
            if ((i / n) % 2) s[i] = float(int(s[i] * 1.5f));
            if (i % 17 == 3) s[i] = -inf;
            if (i % 23 == 5) s[i] = -0.0f;
        }
        const std::vector<uint32_t> lens = {uint32_t(n), uint32_t(n / 2), 0, uint32_t(n - 1)};
        Pair::In si = held(p, s), li = held(p, as_floats(lens));
        for (size_t k : {size_t(1), n / 3 + 1, n}) {
            Pair::Out o = p.out(rows * k);
            p.cpu.topk_rows(o.cs(), si.cs(), li.cs(), rows, n, k, 9);
            p.dev.topk_rows(o.vs(), si.vs(), li.vs(), rows, n, k, 9);
            auto r = p.results(o);
            values += same(r.first, r.second, "topk_rows differs from the CPU at n " + std::to_string(n) + " k " + std::to_string(k));
        }
        for (size_t bs : {1, 4, 64})
            for (size_t k : {1, 3, 200}) {
                Pair::Out o = p.out(rows * k);
                p.cpu.candidate_blocks(o.cs(), si.cs(), li.cs(), rows, n, bs, k);
                p.dev.candidate_blocks(o.vs(), si.vs(), li.vs(), rows, n, bs, k);
                auto r = p.results(o);
                values += same(r.first, r.second, "candidate_blocks differs from the CPU at n " + std::to_string(n) + " block " + std::to_string(bs) + " k " + std::to_string(k));
            }
    }
    return values;
}

size_t check_pool_and_routing(Pair& p) {
    size_t values = 0;
    for (size_t ratio : {1, 4, 128}) {
        const size_t groups = 3, dim = 96;
        Pair::In ki = held(p, uniform(groups * ratio * dim, 21, -2.0f, 2.0f)), si = held(p, uniform(groups * ratio * dim, 22, -4.0f, 4.0f)), wi = held(p, uniform(dim, 23, 0.5f, 1.5f));
        Pair::Out o = p.out(groups * dim);
        p.cpu.compress_pool(o.cs(), ki.cs(), si.cs(), wi.cs(), groups, ratio, dim, 1e-6f);
        p.dev.compress_pool(o.vs(), ki.vs(), si.vs(), wi.vs(), groups, ratio, dim, 1e-6f);
        auto r = p.results(o);
        values += close(r.first, r.second, 1e-5, "compress_pool differs beyond 1e-5");
    }
    for (bool biased : {false, true})
        for (bool normalize : {false, true}) {
            const size_t rows = 9, n_expert = 64, k = 6;
            std::vector<float> scores = uniform(rows * n_expert, 31, -8.0f, 24.0f);
            scores[3] = -100.0f;
            Pair::In sc = held(p, scores), bi = held(p, uniform(n_expert, 32, -0.05f, 0.05f));
            Pair::Out ids = p.out(rows * k), w = p.out(rows * k);
            const backend::RouteRule rc = backend::RouteRule::sqrt_softplus(biased ? bi.cs() : backend::CSlice{}, 1.5f, normalize);
            const backend::RouteRule rv = backend::RouteRule::sqrt_softplus(biased ? bi.vs() : backend::CSlice{}, 1.5f, normalize);
            p.cpu.route_experts(sc.cs(), rows, n_expert, k, rc, ids.cs(), w.cs());
            p.dev.route_experts(sc.vs(), rows, n_expert, k, rv, ids.vs(), w.vs());
            auto ri = p.results(ids), rw = p.results(w);
            values += same(ri.first, ri.second, "sqrt_softplus routing picks differ from the CPU");
            values += close(rw.first, rw.second, 1e-5, "sqrt_softplus routing weights differ beyond 1e-5");
        }
    for (size_t n : {96, 100}) {
        std::vector<float> g = uniform(n, 41, -12.0f, 12.0f), u = uniform(n, 42, -12.0f, 12.0f);
        Pair::In gi = held(p, g), ui = held(p, u);
        Pair::Out d = p.out(n);
        p.cpu.silu_mul(d.cs(), gi.cs(), ui.cs(), n, 7.0f);
        p.dev.silu_mul(d.vs(), gi.vs(), ui.vs(), n, 7.0f);
        auto r = p.results(d);
        values += close(r.first, r.second, 1e-6, "silu_mul's clamp differs beyond 1e-6");
    }
    return values;
}

size_t check_rope_forms(Pair& p, const Exact& exact) {
    size_t values = 0;
    const size_t rows = 3, heads = 2, head_dim = 96, rope = 32, half = rope / 2, positions = 40;
    const std::vector<float> x = uniform(rows * heads * head_dim, 51), w = uniform(head_dim, 52, 0.5f, 1.5f);
    std::vector<float> c(positions * half), s(positions * half);
    for (size_t t = 0; t < positions; ++t)
        for (size_t j = 0; j < half; ++j) {
            c[t * half + j] = std::cos(float(t) * std::pow(1e4f, -float(j) / float(half)));
            s[t * half + j] = std::sin(float(t) * std::pow(1e4f, -float(j) / float(half)));
        }
    const uint32_t pos[rows] = {0, 17, 39};
    Pair::In xi = held(p, x), wi = held(p, w), ci = held(p, c), si = held(p, s);
    for (bool adjacent : {false, true})
        for (size_t offset : {size_t(0), head_dim - rope})
            for (bool inverse : {false, true})
                for (bool norm : {false, true}) {
                    const backend::RopeForm form{adjacent, offset, inverse};
                    Pair::Out d = p.out(rows * heads * head_dim);
                    p.cpu.norm_rope_partial(d.cs(), xi.cs(), rows, heads * head_dim, head_dim, heads, head_dim, rope, norm ? wi.cs() : backend::CSlice{}, 1e-6f, ci.cs(), si.cs(), pos, form);
                    p.dev.norm_rope_partial(d.vs(), xi.vs(), rows, heads * head_dim, head_dim, heads, head_dim, rope, norm ? wi.vs() : backend::CSlice{}, 1e-6f, ci.vs(), si.vs(), pos, form);
                    auto r = p.results(d);
                    // The plain form is norm_rope_partial's own, which the exact list names apart from the other forms.
                    values += meet(exact, form.plain() ? "norm_rope_partial" : "rope_forms", r.first, r.second, 1e-5, "norm_rope_partial's form differs beyond 1e-5");
                }
    return values;
}

// Pooled rows as a source layer keeps them, groups of 4 positions of 24-float entries and 32-float index keys, on both backends from one pool_write.
// Sequence A holds 300 positions on its blocks in reverse order, so its groups cross block edges; sequence B holds 16 positions on the block after them.
size_t check_pooled(Pair& p, const Exact& exact) {
    const size_t ratio = 4, entry = 24, key = 32, width = entry + key;
    auto cs = p.cpu.kv_alloc(1, 0, 0, 512, backend::KVType::f32, backend::KVType::f32, {0}, {{ratio, width}});
    auto vs = p.dev.kv_alloc(1, 0, 0, 512, backend::KVType::f32, backend::KVType::f32, {0}, {{ratio, width}});
    // Each backends tables in its own block size, As blocks in reverse order.
    std::vector<int32_t> tables[2][2];
    for (int dev = 0; dev < 2; ++dev) {
        const size_t bt = (dev ? p.dev : static_cast<backend::Backend&>(p.cpu)).kv_layout().block_tokens, na = (300 + bt - 1) / bt;
        for (size_t i = 0; i < na; ++i) tables[dev][0].push_back(int32_t(na - 1 - i));
        tables[dev][1].push_back(int32_t(na));
    }
    auto view = [&](bool dev, bool on_a, size_t length, size_t nq) {
        const std::vector<int32_t>& t = tables[dev][on_a ? 0 : 1];
        return backend::KVView{dev ? vs.get() : cs.get(), t.data(), t.size(), length, nq};
    };
    const std::vector<float> rows = uniform((75 + 4) * width, 61);
    Pair::In ri = held(p, rows);
    const backend::KVView cw[] = {view(false, true, 0, 300), view(false, false, 0, 16)}, vw[] = {view(true, true, 0, 300), view(true, false, 0, 16)};
    p.cpu.pool_write(0, cw, 2, ri.cs());
    p.dev.pool_write(0, vw, 2, ri.vs());
    size_t values = 0;
    {
        const size_t heads = 4, n = 80, bs = 8, bk = 3;
        const std::vector<backend::KVView> cv = {view(false, true, 280, 10), view(false, false, 13, 3)}, dv = {view(true, true, 280, 10), view(true, false, 13, 3)};
        const size_t m = 13;
        std::vector<uint32_t> cand(m * bk, kNone);
        for (size_t i = 0; i < cand.size(); ++i)
            if (i % 3) cand[i] = uint32_t(i * 7 % 10);
        Pair::In qi = held(p, uniform(m * heads * key, 62)), wi = held(p, uniform(m * heads, 63, -0.5f, 0.5f)), bi = held(p, as_floats(cand));
        for (bool blocks : {false, true}) {
            Pair::Out o = p.out(m * n);
            p.cpu.indexer_scores(o.cs(), qi.cs(), 0, cv.data(), cv.size(), wi.cs(), 0.375f, heads, key, backend::ModelFormat::f32, n, blocks ? bi.cs() : backend::CSlice{}, bk, bs);
            p.dev.indexer_scores(o.vs(), qi.vs(), 0, dv.data(), dv.size(), wi.vs(), 0.375f, heads, key, backend::ModelFormat::f32, n, blocks ? bi.vs() : backend::CSlice{}, bk, bs);
            auto r = p.results(o);
            values += meet(exact, "indexer_scores", r.first, r.second, 1e-5, std::string("indexer_scores differs beyond 1e-5") + (blocks ? " over candidate blocks" : ""), true);
        }
    }
    {
        const size_t heads = 2, dim = entry, kv_rows = 50, k = 6, ek = 5, m = 8;
        const std::vector<backend::KVView> cv = {view(false, true, 290, 6), view(false, false, 14, 2)}, dv = {view(true, true, 290, 6), view(true, false, 14, 2)};
        std::vector<uint32_t> ids(m * k, kNone), entries(m * ek, kNone);
        uint32_t s = 5;
        auto draw = [&] { return s = s * 1664525u + 1013904223u, s >> 8; };
        for (size_t r = 0; r < m; ++r) {
            const size_t visible = (r < 6 ? 290 + r + 1 : 14 + r - 6 + 1) / ratio;
            if (r != 1 && r != 3)
                for (size_t t = 0; t < k; ++t)
                    if (draw() % 4) ids[r * k + t] = draw() % kv_rows;
            if (r != 0 && r != 3)
                for (size_t t = 0; t < ek; ++t)
                    if (draw() % 4) entries[r * ek + t] = uint32_t(draw() % visible);
        }
        Pair::In qi = held(p, uniform(m * heads * dim, 64)), ki = held(p, uniform(kv_rows * dim, 65)), ii = held(p, as_floats(ids)), ei = held(p, as_floats(entries));
        Pair::In si = held(p, std::vector<float>{0.25f, -1.0f});
        const float scale = 1.0f / std::sqrt(float(dim));
        for (bool with_views : {false, true}) {
            Pair::Out o = p.out(m * heads * dim);
            p.cpu.sparse_attention_sink(o.cs(), qi.cs(), ki.cs(), ii.cs(), si.cs(), m, heads, dim, kv_rows, k, scale, 0, with_views ? cv.data() : nullptr,
                                        with_views ? cv.size() : 0, with_views ? ei.cs() : backend::CSlice{}, with_views ? ek : 0, backend::ModelFormat::f32, backend::ModelFormat::f32);
            p.dev.sparse_attention_sink(o.vs(), qi.vs(), ki.vs(), ii.vs(), si.vs(), m, heads, dim, kv_rows, k, scale, 0, with_views ? dv.data() : nullptr,
                                        with_views ? dv.size() : 0, with_views ? ei.vs() : backend::CSlice{}, with_views ? ek : 0, backend::ModelFormat::f32, backend::ModelFormat::f32);
            auto r = p.results(o);
            values += close(r.first, r.second, 1e-5, "sparse_attention_sink differs beyond 1e-5");
        }
    }
    return values;
}

// The cache type native (docs/DEEPSEEK4.md, section 4.3): model_format_pack's bytes against the CPU's in rows of a stride whose gaps stay untouched, every row ending in padding.
// On each backend, window and pooled rows packed and read by sparse_attention_sink and indexer_scores give the bits of the same rows held as F32 rounded by model_format_qdq.
size_t check_packed(Pair& p) {
    namespace mf = quant::model_formats;
    using backend::ModelFormat;
    size_t values = 0;
    // Values over many binades with zeros of both signs, so scales of every size and codes at the clamps are reached.
    const auto raw = [](size_t n, uint32_t seed) {
        std::vector<float> x(n);
        uint32_t s = seed;
        for (size_t i = 0; i < n; ++i) {
            s = s * 1664525u + 1013904223u;
            const int e = int((i / 16) % 23) - 12;
            x[i] = std::ldexp(float(s >> 8) / float(1u << 24), e) * ((s & 1) ? -1.0f : 1.0f);
            if (i % 97 == 3) x[i] = 0.0f;
            if (i % 97 == 4) x[i] = -0.0f;
        }
        return x;
    };
    for (ModelFormat f : {ModelFormat::fp8_power, ModelFormat::fp4_power, ModelFormat::fp4_e4m3}) {
        const size_t rows = 7, dim = 96, words = backend::format_words(f, dim), stride = words + 3, n = (rows - 1) * stride + words;
        const std::vector<float> x = raw(rows * dim, 91 + uint32_t(f));
        Pair::In xi = held(p, x);
        Pair::Out o = p.out(n);
        const std::vector<float> fill(n, ds4::as_floats({0xA5A5A5A5u})[0]);
        p.cpu.write(*o.c, 0, fill.data(), n * 4);
        p.dev.write(*o.v, 0, fill.data(), n * 4);
        p.cpu.model_format_pack(o.cs(), stride, xi.cs(), rows, dim, f);
        p.dev.model_format_pack(o.vs(), stride, xi.vs(), rows, dim, f);
        auto r = p.results(o);
        values += same(r.first, r.second, "model_format_pack differs from the CPU in format " + std::to_string(int(f)));
        for (size_t row = 0; row < rows; ++row)
            for (size_t w = words; w < stride && row + 1 < rows; ++w)
                require(!std::memcmp(&r.first[row * stride + w], &fill[0], 4), "model_format_pack wrote between its rows");
    }
    const size_t ratio = 4, entry = 64, key = 32, heads = 2, positions = 96, groups = positions / ratio;
    const size_t ew = backend::format_words(ModelFormat::fp4_e4m3, entry), kw = backend::format_words(ModelFormat::fp4_power, key);
    const size_t f32_width = entry + key, packed_width = ew + kw;
    std::vector<float> latents = raw(groups * entry, 71), keys = raw(groups * key, 72), rounded(groups * f32_width);
    for (size_t g = 0; g < groups; ++g) {
        std::copy(latents.begin() + g * entry, latents.begin() + (g + 1) * entry, rounded.begin() + g * f32_width);
        std::copy(keys.begin() + g * key, keys.begin() + (g + 1) * key, rounded.begin() + g * f32_width + entry);
        for (size_t b = 0; b < entry; b += 16) mf::qdq_fp4_e4m3(&rounded[g * f32_width + b]);
        for (size_t b = 0; b < key; b += 32) mf::qdq_fp4_power(&rounded[g * f32_width + entry + b]);
    }
    const size_t kv_rows = 40, kvw = backend::format_words(ModelFormat::fp8_power, entry);
    std::vector<float> ring = raw(kv_rows * entry, 73), ring_rounded = ring;
    for (size_t i = 0; i < ring_rounded.size(); i += 32) mf::qdq_fp8_power(&ring_rounded[i]);
    Pair::In li = held(p, latents), ki = held(p, keys), ri = held(p, rounded), wi = held(p, ring), wr = held(p, ring_rounded);
    Pair::Out pr = p.out(groups * packed_width), pw = p.out(kv_rows * kvw);
    p.cpu.model_format_pack(pr.cs(), packed_width, li.cs(), groups, entry, ModelFormat::fp4_e4m3);
    p.dev.model_format_pack(pr.vs(), packed_width, li.vs(), groups, entry, ModelFormat::fp4_e4m3);
    p.cpu.model_format_pack({pr.c.get(), ew}, packed_width, ki.cs(), groups, key, ModelFormat::fp4_power);
    p.dev.model_format_pack({pr.v.get(), ew}, packed_width, ki.vs(), groups, key, ModelFormat::fp4_power);
    p.cpu.model_format_pack(pw.cs(), kvw, wi.cs(), kv_rows, entry, ModelFormat::fp8_power);
    p.dev.model_format_pack(pw.vs(), kvw, wi.vs(), kv_rows, entry, ModelFormat::fp8_power);
    auto c32 = p.cpu.kv_alloc(1, 0, 0, 256, backend::KVType::f32, backend::KVType::f32, {0}, {{ratio, f32_width}});
    auto v32 = p.dev.kv_alloc(1, 0, 0, 256, backend::KVType::f32, backend::KVType::f32, {0}, {{ratio, f32_width}});
    auto cpk = p.cpu.kv_alloc(1, 0, 0, 256, backend::KVType::f32, backend::KVType::f32, {0}, {{ratio, packed_width}});
    auto vpk = p.dev.kv_alloc(1, 0, 0, 256, backend::KVType::f32, backend::KVType::f32, {0}, {{ratio, packed_width}});
    std::vector<int32_t> tables[2];
    for (int dev = 0; dev < 2; ++dev) {
        const size_t bt = (dev ? p.dev : static_cast<backend::Backend&>(p.cpu)).kv_layout().block_tokens;
        for (size_t i = 0, nb = (positions + bt - 1) / bt; i < nb; ++i) tables[dev].push_back(int32_t(nb - 1 - i));
    }
    const auto view = [&](backend::KVStorage* st, bool dev, size_t length, size_t nq) {
        return backend::KVView{st, tables[dev].data(), tables[dev].size(), length, nq};
    };
    const backend::KVView w_c32 = view(c32.get(), false, 0, positions), w_v32 = view(v32.get(), true, 0, positions);
    const backend::KVView w_cpk = view(cpk.get(), false, 0, positions), w_vpk = view(vpk.get(), true, 0, positions);
    p.cpu.pool_write(0, &w_c32, 1, ri.cs());
    p.dev.pool_write(0, &w_v32, 1, ri.vs());
    p.cpu.pool_write(0, &w_cpk, 1, pr.cs());
    p.dev.pool_write(0, &w_vpk, 1, pr.vs());
    const size_t m = 6, n = groups;
    const backend::KVView r_c32 = view(c32.get(), false, 90, m), r_v32 = view(v32.get(), true, 90, m);
    const backend::KVView r_cpk = view(cpk.get(), false, 90, m), r_vpk = view(vpk.get(), true, 90, m);
    Pair::In qi = held(p, uniform(m * heads * key, 74)), wq = held(p, uniform(m * heads, 75, -0.5f, 0.5f));
    {
        Pair::Out a = p.out(m * n), b = p.out(m * n);
        p.cpu.indexer_scores(a.cs(), qi.cs(), 0, &r_c32, 1, wq.cs(), 0.375f, heads, key, ModelFormat::f32, n, {}, 0, 0);
        p.dev.indexer_scores(a.vs(), qi.vs(), 0, &r_v32, 1, wq.vs(), 0.375f, heads, key, ModelFormat::f32, n, {}, 0, 0);
        p.cpu.indexer_scores(b.cs(), qi.cs(), 0, &r_cpk, 1, wq.cs(), 0.375f, heads, key, ModelFormat::fp4_power, n, {}, 0, 0);
        p.dev.indexer_scores(b.vs(), qi.vs(), 0, &r_vpk, 1, wq.vs(), 0.375f, heads, key, ModelFormat::fp4_power, n, {}, 0, 0);
        auto ra = p.results(a), rb = p.results(b);
        values += same(ra.first, rb.first, "indexer_scores over packed keys differs from F32 keys on the CPU");
        values += same(ra.second, rb.second, "indexer_scores over packed keys differs from F32 keys on the device");
    }
    {
        const size_t k = 7, ek = 5;
        std::vector<uint32_t> ids(m * k, kNone), picks(m * ek, kNone);
        uint32_t s = 9;
        auto draw = [&] { return s = s * 1664525u + 1013904223u, s >> 8; };
        for (size_t r = 0; r < m; ++r) {
            for (size_t t = 0; t < k; ++t)
                if (draw() % 4) ids[r * k + t] = draw() % kv_rows;
            for (size_t t = 0; t < ek; ++t)
                if (draw() % 4) picks[r * ek + t] = uint32_t(draw() % ((90 + r + 1) / ratio));
        }
        Pair::In qa = held(p, uniform(m * heads * entry, 76)), ii = held(p, as_floats(ids)), ei = held(p, as_floats(picks)), si = held(p, {0.25f, -1.0f});
        const float scale = 1.0f / std::sqrt(float(entry));
        Pair::Out a = p.out(m * heads * entry), b = p.out(m * heads * entry);
        p.cpu.sparse_attention_sink(a.cs(), qa.cs(), wr.cs(), ii.cs(), si.cs(), m, heads, entry, kv_rows, k, scale, 0, &r_c32, 1, ei.cs(), ek, ModelFormat::f32,
                                    ModelFormat::f32);
        p.dev.sparse_attention_sink(a.vs(), qa.vs(), wr.vs(), ii.vs(), si.vs(), m, heads, entry, kv_rows, k, scale, 0, &r_v32, 1, ei.vs(), ek, ModelFormat::f32,
                                    ModelFormat::f32);
        p.cpu.sparse_attention_sink(b.cs(), qa.cs(), pw.cs(), ii.cs(), si.cs(), m, heads, entry, kv_rows, k, scale, 0, &r_cpk, 1, ei.cs(), ek, ModelFormat::fp8_power,
                                    ModelFormat::fp4_e4m3);
        p.dev.sparse_attention_sink(b.vs(), qa.vs(), pw.vs(), ii.vs(), si.vs(), m, heads, entry, kv_rows, k, scale, 0, &r_vpk, 1, ei.vs(), ek, ModelFormat::fp8_power,
                                    ModelFormat::fp4_e4m3);
        auto ra = p.results(a), rb = p.results(b);
        values += same(ra.first, rb.first, "sparse_attention_sink over packed rows differs from F32 rows on the CPU");
        values += same(ra.second, rb.second, "sparse_attention_sink over packed rows differs from F32 rows on the device");
    }
    return values;
}

// Every DeepSeek-V4.1 op against the CPU, the ops named in `exact` bit for bit; a device whose backend does not implement the format op is reported and passed over.
size_t check_ops(Pair& p, const Exact& exact = {}) {
    size_t values = check_mhc(p, exact);
    if (p.dev.implements(backend::Op::model_format_qdq) && p.dev.implements(backend::Op::model_format_pack))
        values += check_formats(p) + check_packed(p);
    else
        std::printf("deepseek4: model formats skipped: the device does not implement model_format_qdq\n");
    values += check_selections(p);
    values += check_pool_and_routing(p);
    values += check_rope_forms(p, exact);
    values += check_pooled(p, exact);
    return values;
}
} // namespace ds4
