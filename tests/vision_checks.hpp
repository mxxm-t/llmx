#pragma once
// The vision encoder's ops (docs/VISION.md, section 3) on a device against the CPU backend at the towers' widths, shared by the device backends' tests.
// The bias is exact, the row mix, norm, gelu and rope over streams within their bounds, and attention over a non-causal view within 1e-4, a run of its query rows the bits of the whole call's.
#include "qwen35_checks.hpp"

namespace vision {

using q35::close;
using q35::Pair;
using q35::require;
using q35::uniform;

// `dst` written with `init` on both sides, so an op that adds to it starts from the same rows.
Pair::Out seeded(Pair& p, const std::vector<float>& init) {
    Pair::Out o = p.out(init.size());
    p.cpu.write(*o.c, 0, init.data(), init.size() * sizeof(float));
    p.dev.write(*o.v, 0, init.data(), init.size() * sizeof(float));
    return o;
}

// Attention of q over one view of n keys, all non-causal, `nq` query rows from row n - nq on, as the encoder runs a piece of its rows.
std::vector<float> attend(backend::Backend& b, const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, size_t n, size_t nq,
                          int heads, int hd) {
    const size_t w = size_t(heads) * size_t(hd);
    auto st = b.kv_alloc(1, size_t(heads), size_t(hd), n);
    std::vector<int32_t> blocks(backend::blocks_for(n, b.kv_layout().block_tokens));
    for (size_t i = 0; i < blocks.size(); ++i) blocks[i] = int32_t(i);
    const backend::KVView all{st.get(), blocks.data(), blocks.size(), 0, n, n, false};
    const auto Kb = b.adopt(k.data(), k.size() * sizeof(float)), Vb = b.adopt(v.data(), v.size() * sizeof(float));
    b.kv_write(0, &all, 1, {Kb.get(), 0}, {Vb.get(), 0});
    const backend::KVView view{st.get(), blocks.data(), blocks.size(), n - nq, nq, nq, false};
    const auto Qb = b.adopt(q.data(), q.size() * sizeof(float));
    const auto ob = b.alloc(nq * w * sizeof(float), backend::Memory::device);
    b.attention({Qb.get(), 0}, 0, &view, 1, {ob.get(), 0}, heads, heads, hd);
    std::vector<float> out(nq * w);
    b.read(*ob, 0, out.data(), out.size() * sizeof(float));
    return out;
}

// Attention over n keys where key 0 scores 18.25 above the rest, its values +1 against -1 for every other key, so each other weight is about 1.2e-8 of the total.
// Returns the worst error of `b`'s outputs as a fraction of 4e-6 (1 + |ref|) against a double softmax: one F32 chain over the keys drops every small product against the leading one (docs/PRECISION.md, Attention value sums).
double dominant_key(backend::Backend& b, int hd, size_t n) {
    const int heads = 2;
    const size_t w = size_t(heads) * size_t(hd);
    const float root = std::sqrt(float(hd));
    std::vector<float> q(n * w, 0.0f), k = uniform(n * w, 420 + uint32_t(hd), -0.25f, 0.25f), v(n * w, -1.0f);
    for (size_t t = 0; t < n; ++t)
        for (int h = 0; h < heads; ++h) {
            q[t * w + size_t(h) * size_t(hd)] = 1.0f;
            k[t * w + size_t(h) * size_t(hd)] *= root;
        }
    for (int h = 0; h < heads; ++h) {
        k[size_t(h) * size_t(hd)] = 18.25f * root;
        for (int i = 0; i < hd; ++i) v[size_t(h) * size_t(hd) + size_t(i)] = 1.0f;
    }
    // Every query row is the same, so a head's output is one value: the weighted mean of +1 and -1 by the keys' first dimensions.
    std::vector<double> ref(static_cast<size_t>(heads));
    for (int h = 0; h < heads; ++h) {
        double m = -1e300, num = 0.0, den = 0.0;
        for (size_t t = 0; t < n; ++t) m = std::max(m, double(k[t * w + size_t(h) * size_t(hd)]) / std::sqrt(double(hd)));
        for (size_t t = 0; t < n; ++t) {
            const double e = std::exp(double(k[t * w + size_t(h) * size_t(hd)]) / std::sqrt(double(hd)) - m);
            num += e * double(v[t * w + size_t(h) * size_t(hd)]);
            den += e;
        }
        ref[size_t(h)] = num / den;
    }
    const std::vector<float> got = attend(b, q, k, v, n, n, heads, hd);
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        const double r = ref[(i % w) / size_t(hd)];
        worst = std::max(worst, std::fabs(double(got[i]) - r) / (4e-6 * (1.0 + std::fabs(r))));
    }
    return worst;
}

size_t check_ops(Pair& p) {
    size_t values = 0;
    // A long view's value sums hold small weights beside a dominant one, at the towers' head widths, an Inkling 1080p image's 1296 patches and 4096.
    for (int hd : {64, 72})
        for (size_t n : {size_t(1296), size_t(4096)}) {
            const double e = dominant_key(p.dev, hd, n);
            const std::string at = "attention over " + std::to_string(n) + " keys with a dominant one, head width " + std::to_string(hd) + ": " + std::to_string(e) + " of its bound";
            require(e <= 1.0, at.c_str());
            values += n * 2 * size_t(hd);
        }
    for (size_t width : {size_t(48), size_t(72), size_t(1152), size_t(4304)}) {
        const size_t rows = 37;
        const auto x = uniform(rows * width, 400 + uint32_t(width), -3.0f, 5.0f), bias = uniform(width, 401 + uint32_t(width));
        Pair::In bi = p.in(bias);
        Pair::Out d = seeded(p, x);
        p.cpu.add_bias(d.cs(), bi.cs(), rows, width);
        p.dev.add_bias(d.vs(), bi.vs(), rows, width);
        auto r = p.results(d);
        require(q35::same_bits(r.first, r.second), "add_bias differs from the CPU");
        values += r.first.size();
        // A layer scale on the biased rows, (row + bias) * scale.
        const auto ls = uniform(width, 404 + uint32_t(width), -0.5f, 0.5f);
        Pair::In si = p.in(ls);
        Pair::Out ds = seeded(p, x);
        p.cpu.add_bias(ds.cs(), bi.cs(), rows, width, si.cs());
        p.dev.add_bias(ds.vs(), bi.vs(), rows, width, si.vs());
        r = p.results(ds);
        require(q35::same_bits(r.first, r.second), "add_bias with a scale differs from the CPU");
        values += r.first.size();
        // The learned positions' four taps of a 48 by 48 table, read in order.
        const size_t side = 48, taps = 4;
        const auto table = uniform(side * side * width, 402 + uint32_t(width)), w = uniform(rows * taps, 403, 0.0f, 1.0f);
        std::vector<uint32_t> idx(rows * taps);
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = uint32_t((i * 2654435761u) % (side * side));
        Pair::In ti = p.in(table);
        Pair::Out m = seeded(p, x);
        p.cpu.mix_rows(m.cs(), ti.cs(), width, side * side, idx.data(), w.data(), rows, taps);
        p.dev.mix_rows(m.vs(), ti.vs(), width, side * side, idx.data(), w.data(), rows, taps);
        r = p.results(m);
        // Vulkan does not require fma fused, and on an MI50 under RADV the taps' last bits differ from the CPU's fused chain, so they are held to a bound.
        values += close(r.first, r.second, 1e-6, "mix_rows differs from the CPU beyond 1e-6");
        // LayerNorm over rows of a nonzero mean, packed and with a stride, out of place and in place.
        for (size_t stride : {width, width + 8}) {
            const auto src = uniform(rows * stride, 404 + uint32_t(stride), 1.0f, 7.0f), lw = uniform(width, 405, 0.5f, 1.5f), lb = uniform(width, 406);
            Pair::In si = p.in(src), wi = p.in(lw), bb = p.in(lb);
            Pair::Out n = seeded(p, src);
            p.cpu.layer_norm_rows(n.cs(), si.cs(), wi.cs(), bb.cs(), rows, width, stride, 1e-6f);
            p.dev.layer_norm_rows(n.vs(), si.vs(), wi.vs(), bb.vs(), rows, width, stride, 1e-6f);
            r = p.results(n);
            values += close(r.first, r.second, 1e-5, "layer_norm_rows differs from the CPU beyond 1e-5");
            Pair::Out ip = seeded(p, src);
            p.dev.layer_norm_rows(ip.vs(), {ip.v.get(), 0}, wi.vs(), bb.vs(), rows, width, stride, 1e-6f);
            std::vector<float> in_place(src.size());
            p.dev.read(*ip.v, 0, in_place.data(), in_place.size() * sizeof(float));
            require(q35::same_bits(in_place, r.second), "layer_norm_rows in place differs on the device");
            // A row alone gives the bits it gives beside the others.
            Pair::Out one = p.out(width);
            p.dev.layer_norm_rows(one.vs(), {si.v.get(), 5 * stride}, wi.vs(), bb.vs(), 1, width, width, 1e-6f);
            std::vector<float> alone(width);
            p.dev.read(*one.v, 0, alone.data(), width * sizeof(float));
            require(std::memcmp(alone.data(), r.second.data() + 5 * stride, width * sizeof(float)) == 0, "layer_norm_rows of a row alone differs from the row beside others");
        }
    }
    // GELU in both forms over values from 1e-30 to 1e5 of both signs, a uniform spread where it bends and a dense sweep of the negative tail.
    // The tanh form is held within 1e-6 absolute; the erf form within 16 float units of the CPU's value and 1e-9 absolute, so a 1 + erf that cancels below -2 fails.
    std::vector<float> g = uniform(65536, 407, -12.0f, 12.0f);
    for (int e = -30; e <= 5; ++e)
        for (float s : {1.0f, -1.0f, 3.7f, -3.7f}) g.push_back(s * std::pow(10.0f, float(e)));
    for (int i = 0; i <= 4000; ++i) g.push_back(-6.0f + float(i) / 1000.0f);
    Pair::In gi = p.in(g);
    for (backend::GeluForm form : {backend::GeluForm::tanh, backend::GeluForm::erf}) {
        Pair::Out d = p.out(g.size());
        p.cpu.gelu(d.cs(), gi.cs(), g.size(), form);
        p.dev.gelu(d.vs(), gi.vs(), g.size(), form);
        auto r = p.results(d);
        if (form == backend::GeluForm::tanh) {
            values += close(r.first, r.second, 1e-6, "gelu (tanh) differs from the CPU beyond 1e-6");
            continue;
        }
        for (size_t i = 0; i < g.size(); ++i) {
            const double want = r.first[i], got = r.second[i];
            if (!(std::fabs(got - want) <= 16.0 * std::ldexp(std::fabs(want), -23) + 1e-9)) {
                std::fprintf(stderr, "  gelu (erf) at %.9g: cpu %.9g device %.9g\n", double(g[i]), want, got);
                require(false, "gelu (erf) differs from the CPU beyond 16 float units and 1e-9");
            }
        }
        values += g.size();
    }
    // The rope over two streams, a head's first quarter of pairs by the patch's row and the rest by its column, with and without a norm, and one stream without a norm.
    for (size_t hd : {size_t(64), size_t(72), size_t(128)}) {
        const size_t rows = 37, heads = 4, half = hd / 2, table = 64, width = heads * hd;
        std::vector<float> cs(table * half), sn(table * half);
        for (size_t t = 0; t < table; ++t)
            for (size_t i = 0; i < half; ++i) {
                const double f = std::pow(10000.0, -2.0 * double(i) / double(half));
                cs[t * half + i] = float(std::cos(double(t) * f));
                sn[t * half + i] = float(std::sin(double(t) * f));
            }
        std::vector<uint8_t> of(half);
        for (size_t j = 0; j < half; ++j) of[j] = j < hd / 4 ? 0 : 1;
        std::vector<uint32_t> pos(rows * 2);
        for (size_t i = 0; i < pos.size(); ++i) pos[i] = uint32_t((i * 7 + 3) % table);
        const auto src = uniform(rows * width, 408 + uint32_t(hd)), nw = uniform(hd, 409, 0.5f, 1.5f);
        Pair::In si = p.in(src), ci = p.in(cs), sni = p.in(sn), wi = p.in(nw);
        for (int mode = 0; mode < 3; ++mode) {
            const backend::RopeStreams streams = mode == 2 ? backend::RopeStreams{} : backend::RopeStreams{2, of.data()};
            const bool norm = mode == 1;
            Pair::Out d = p.out(src.size());
            p.cpu.norm_rope_partial(d.cs(), si.cs(), rows, width, hd, heads, hd, hd, norm ? wi.cs() : backend::CSlice{}, 1e-6f, ci.cs(), sni.cs(), pos.data(), backend::RopeForm{}, streams);
            p.dev.norm_rope_partial(d.vs(), si.vs(), rows, width, hd, heads, hd, hd, norm ? wi.vs() : backend::CSlice{}, 1e-6f, ci.vs(), sni.vs(), pos.data(), backend::RopeForm{}, streams);
            auto r = p.results(d);
            values += close(r.first, r.second, 1e-5, "the rope over streams differs from the CPU beyond 1e-5");
            Pair::Out ip = seeded(p, src);
            p.dev.norm_rope_partial(ip.vs(), {ip.v.get(), 0}, rows, width, hd, heads, hd, hd, norm ? wi.vs() : backend::CSlice{}, 1e-6f, ci.vs(), sni.vs(), pos.data(), backend::RopeForm{}, streams);
            std::vector<float> in_place(src.size());
            p.dev.read(*ip.v, 0, in_place.data(), in_place.size() * sizeof(float));
            require(q35::same_bits(in_place, r.second), "the rope over streams in place differs on the device");
        }
    }
    // Attention over one non-causal view of an image's patches, every row seeing every key, at the towers' head widths and at 128.
    for (int hd : {64, 72, 128})
        for (size_t n : {size_t(37), size_t(300), size_t(4096)}) {
            const int heads = 4;
            const size_t w = size_t(heads) * size_t(hd);
            const auto q = uniform(n * w, 410 + uint32_t(n), -2.0f, 2.0f), k = uniform(n * w, 411 + uint32_t(n), -2.0f, 2.0f), v = uniform(n * w, 412 + uint32_t(n));
            const std::vector<float> whole = attend(p.dev, q, k, v, n, n, heads, hd);
            values += close(attend(p.cpu, q, k, v, n, n, heads, hd), whole, 1e-4, "non-causal attention differs from the CPU beyond 1e-4");
            // The encoder's piece: the last rows as a view whose history is the rest, the bits of the same rows in the whole call.
            const size_t m = std::min<size_t>(n, 33);
            const std::vector<float> tail(q.end() - std::ptrdiff_t(m * w), q.end());
            const std::vector<float> piece = attend(p.dev, tail, k, v, n, m, heads, hd);
            require(std::memcmp(piece.data(), whole.data() + (n - m) * w, piece.size() * sizeof(float)) == 0,
                    "a run of non-causal attention's rows differs from the rows in the whole call");
        }
    return values;
}

}  // namespace vision
