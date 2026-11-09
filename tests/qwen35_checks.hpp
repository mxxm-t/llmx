#pragma once
// The qwen35 layers' ops (docs/QWEN35.md) on a device against the CPU backend, and the device's own bit-for-bit rules for them, shared by the device backends' tests.
// The recurrence sums a column in another order than the CPU, so its rows and states meet the CPU's within a bound; on the device a sequence gives the same bits alone, beside others, in any view order and cut into passes of any length.
#include "backends/cpu/cpu_backend.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace q35 {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

std::vector<float> uniform(size_t n, uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

// The same op on both backends over the same inputs.
// Inputs are adopted (the CPU keeps the pointer, the device uploads a copy); outputs are allocated on each and the device's is read back.
struct Pair {
    backend::CpuBackend cpu;
    backend::Backend& dev;
    // The reference keeps float activations in decode, so a device's rounding is compared against exact arithmetic.
    explicit Pair(backend::Backend& v) : dev(v) { cpu.set_threads(1); }
    struct In {
        backend::BufferPtr c, v;
        backend::CSlice cs() const { return {c.get(), 0}; }
        backend::CSlice vs() const { return {v.get(), 0}; }
    };
    struct Out {
        backend::BufferPtr c, v;
        size_t n;
        backend::Slice cs() const { return {c.get(), 0}; }
        backend::Slice vs() const { return {v.get(), 0}; }
    };
    In in(const void* data, size_t bytes) {
        return {cpu.adopt(data, bytes), dev.adopt(data, bytes)};
    }
    In in(const std::vector<float>& f) { return in(f.data(), f.size() * sizeof(float)); }
    Out out(size_t floats) {
        return {cpu.alloc(floats * sizeof(float), backend::Memory::device),
                dev.alloc(floats * sizeof(float), backend::Memory::device), floats};
    }
    // Both results, with the device's read back.
    std::pair<std::vector<float>, std::vector<float>> results(const Out& o) {
        std::vector<float> a(o.n), b(o.n);
        cpu.read(*o.c, 0, a.data(), o.n * sizeof(float));
        dev.read(*o.v, 0, b.data(), o.n * sizeof(float));
        return {a, b};
    }
};

size_t close(const std::vector<float>& a, const std::vector<float>& b, double rel, const char* what) {
    require(a.size() == b.size(), what);
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(b[i])) throw std::runtime_error(std::string("nonfinite device output: ") + what);
        if (!(std::fabs((double)a[i] - b[i]) <= rel * (1.0 + std::fabs((double)a[i])))) {
            std::fprintf(stderr, "  [%zu] cpu %.9g device %.9g\n", i, a[i], b[i]);
            require(false, what);
        }
    }
    return a.size();
}

using backend::BufferPtr;
using backend::StateShape;
using backend::StateView;

struct Seq {
    size_t length, src, dst, nq;
};
struct Inputs {
    std::vector<float> x, w, alpha, b, a, dt_bias;
};
struct Run {
    std::vector<float> conv, delta;
    std::vector<std::vector<float>> slots;
};

Inputs inputs(uint32_t seed, const StateShape& sh, size_t rows) {
    Inputs in;
    in.x = uniform(rows * sh.channels(), seed, -1.0f, 1.0f);
    in.w = uniform(sh.channels() * backend::kConvTaps, seed + 1, -0.8f, 0.8f);
    in.alpha = uniform(rows * sh.v_heads, seed + 2, -3.0f, 3.0f);
    in.b = uniform(rows * sh.v_heads, seed + 3, -4.0f, 4.0f);
    in.a = uniform(sh.v_heads, seed + 4, -2.0f, -0.05f);
    in.dt_bias = uniform(sh.v_heads, seed + 5, -2.0f, 2.0f);
    return in;
}

std::vector<float> read_floats(backend::Backend& b, const backend::Buffer& buf, size_t n, size_t offset = 0) {
    std::vector<float> v(n);
    if (n) b.read(buf, offset * sizeof(float), v.data(), n * sizeof(float));
    return v;
}

BufferPtr floats_on(backend::Backend& b, const std::vector<float>& v) {
    BufferPtr buf = b.alloc(std::max<size_t>(v.size(), 1) * sizeof(float), backend::Memory::device);
    if (!v.empty()) b.write(*buf, 0, v.data(), v.size() * sizeof(float));
    return buf;
}

// One conv and one delta rule call over the views, on layer 1 of a two-layer storage whose slots start as `start`.
Run run(backend::Backend& b, const StateShape& sh, const std::vector<Seq>& seqs, const Inputs& in,
        const std::vector<std::vector<float>>& start, size_t tail_floats = 0) {
    const size_t layer = 1, core = sh.slot_floats();
    const float tail_sentinel = -12345.5f;
    const std::vector<size_t> tails = tail_floats ? std::vector<size_t>{0, tail_floats} : std::vector<size_t>{};
    auto storage = b.state_alloc(2, start.size(), sh, tails);
    const size_t n = storage->slot_floats(layer), other_n = storage->slot_floats(0);
    for (size_t s = 0; s < start.size(); ++s) {
        require(start[s].size() == core, "recurrent test slot has the wrong core size");
        std::vector<float> slot(n, tail_sentinel);
        std::copy(start[s].begin(), start[s].end(), slot.begin());
        b.write(storage->layer(layer), s * n * sizeof(float), slot.data(), n * sizeof(float));
        if (tail_floats) {
            std::vector<float> other(other_n, tail_sentinel);
            b.write(storage->layer(0), s * other_n * sizeof(float), other.data(), other_n * sizeof(float));
        }
    }
    std::vector<StateView> views;
    size_t rows = 0;
    for (const Seq& q : seqs) {
        views.push_back({storage.get(), q.src, q.dst, q.length, q.nq});
        rows += q.nq;
    }
    const size_t C = sh.channels(), Hv = sh.v_heads;
    BufferPtr x = floats_on(b, in.x), w = floats_on(b, in.w), alpha = floats_on(b, in.alpha), bb = floats_on(b, in.b);
    BufferPtr a = floats_on(b, in.a), dt = floats_on(b, in.dt_bias);
    BufferPtr u = b.alloc(rows * C * sizeof(float), backend::Memory::device);
    BufferPtr o = b.alloc(rows * Hv * sh.v_dim * sizeof(float), backend::Memory::device);
    b.causal_conv_silu({u.get(), 0}, {x.get(), 0}, {w.get(), 0}, layer, views.data(), views.size());
    b.gated_delta_rule({o.get(), 0}, {u.get(), 0}, {alpha.get(), 0}, {bb.get(), 0}, {a.get(), 0}, {dt.get(), 0}, layer, views.data(), views.size());
    Run r;
    r.conv = read_floats(b, *u, rows * C);
    r.delta = read_floats(b, *o, rows * Hv * sh.v_dim);
    for (size_t s = 0; s < start.size(); ++s) r.slots.push_back(read_floats(b, storage->layer(layer), n, s * n));
    for (size_t s = 0; s < start.size(); ++s) {
        const auto other = read_floats(b, storage->layer(0), other_n, s * other_n);
        require(std::all_of(other.begin(), other.end(), [&](float f) { return f == (tail_floats ? tail_sentinel : 0.0f); }), "a state op wrote another layer");
    }
    return r;
}

// A layer-specific tail changes slot spacing, but the recurrent ops may only touch their GDN prefix.
size_t check_layer_tail(backend::Backend& vk) {
    const StateShape sh = {1, 2, 8, 8};
    const size_t core = sh.slot_floats(), tail = 11;
    size_t values = 0;
    for (size_t rows : {size_t(3), size_t(33)}) {
        const std::vector<Seq> seqs = {{7, 1, 2, rows}};
        const Inputs in = inputs(171 + (uint32_t)rows, sh, rows);
        std::vector<std::vector<float>> start(4);
        for (size_t s = 0; s < start.size(); ++s) start[s] = uniform(core, 172 + (uint32_t)s, -0.5f, 0.5f);
        const Run base = run(vk, sh, seqs, in, start);
        const Run with_tail = run(vk, sh, seqs, in, start, tail);
        require(base.conv.size() == with_tail.conv.size() && base.delta.size() == with_tail.delta.size() &&
                    std::memcmp(base.conv.data(), with_tail.conv.data(), base.conv.size() * sizeof(float)) == 0 &&
                    std::memcmp(base.delta.data(), with_tail.delta.data(), base.delta.size() * sizeof(float)) == 0,
                "a recurrent tail changed device conv or delta rows");
        for (size_t s = 0; s < start.size(); ++s) {
            require(std::memcmp(base.slots[s].data(), with_tail.slots[s].data(), core * sizeof(float)) == 0,
                    "a recurrent tail changed the device GDN state");
            require(std::all_of(with_tail.slots[s].begin() + core, with_tail.slots[s].end(),
                                [](float f) { return f == -12345.5f; }),
                    "a recurrent op wrote into its layer tail");
        }
        require(std::memcmp(start[1].data(), with_tail.slots[1].data(), core * sizeof(float)) == 0,
                "a recurrent op changed its nonzero source slot");
        values += with_tail.conv.size() + with_tail.delta.size() + start.size() * (core + tail);
    }
    return values;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}


// A slot's matrices within the bound, and its carried rows, which are raw rows, bit for bit.
size_t close_slot(const StateShape& sh, const std::vector<float>& cpu, const std::vector<float>& dev, const char* what) {
    const size_t m = sh.v_heads * sh.matrix_floats();
    const std::vector<float> a(cpu.begin(), cpu.begin() + m), b(dev.begin(), dev.begin() + m);
    size_t values = close(a, b, 1e-4, what);
    require(std::memcmp(cpu.data() + m, dev.data() + m, (cpu.size() - m) * sizeof(float)) == 0, "carried conv rows differ from the CPU's");
    return values + cpu.size() - m;
}

// Compare fresh, short-history, one-token and verify sequences beside each other against the CPU, with NaN in unread state and history slots.
size_t check_mix(backend::Backend& vk, backend::CpuBackend& cpu, const StateShape& sh, uint32_t seed) {
    const std::vector<Seq> seqs = {{0, 0, 0, 5}, {1, 1, 1, 3}, {2, 2, 2, 1}, {7, 3, 3, 6}, {0, 4, 4, 1}, {9, 5, 6, 4}, {3, 7, 7, 2}};
    size_t rows = 0;
    for (const Seq& q : seqs) rows += q.nq;
    const Inputs in = inputs(seed, sh, rows);
    std::vector<std::vector<float>> start;
    for (uint32_t s = 0; s < 8; ++s) start.push_back(uniform(sh.slot_floats(), seed + 10 + s, -0.5f, 0.5f));
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const size_t C = sh.channels(), first = sh.v_heads * sh.matrix_floats();
    std::vector<std::vector<float>> clean = start;
    clean[0].assign(sh.slot_floats(), 0.0f);
    clean[4].assign(sh.slot_floats(), 0.0f);
    start[0].assign(sh.slot_floats(), nan);
    start[4].assign(sh.slot_floats(), nan);
    for (size_t c = 0; c < C; ++c) {
        start[1][first + c] = start[1][first + C + c] = nan;
        start[2][first + c] = nan;
        clean[1][first + c] = clean[1][first + C + c] = 0.0f;
        clean[2][first + c] = 0.0f;
    }
    const Run d = run(vk, sh, seqs, in, start), z = run(vk, sh, seqs, in, clean), c = run(cpu, sh, seqs, in, clean);
    for (float f : d.conv) require(std::isfinite(f), "a device conv row read a fresh slot or a carried row before the sequence's start");
    for (float f : d.delta) require(std::isfinite(f), "the device delta rule read the slot of a fresh sequence");
    require(same_bits(d.conv, z.conv) && same_bits(d.delta, z.delta), "a fresh slot's contents changed a device result");
    size_t values = close(c.conv, d.conv, 1e-5, "conv differs from the CPU beyond 1e-5");
    values += close(c.delta, d.delta, 1e-4, "delta rule rows differ from the CPU beyond 1e-4");
    for (const Seq& q : seqs) values += close_slot(sh, c.slots[q.dst], d.slots[q.dst], "a delta rule state differs from the CPU beyond 1e-4");
    // A slot only read keeps its bits.
    require(same_bits(d.slots[5], start[5]), "a verify changed its source slot on the device");
    return values;
}

// Sequence rows and state must be identical alone, reordered, batched and sliced (docs/QWEN35.md, Row classes).
// The chosen lengths also compare both delta-rule builds across their threshold.
size_t check_invariance(backend::Backend& vk, const StateShape& sh, uint32_t seed) {
    const std::vector<Seq> seqs = {{0, 0, 0, 9}, {4, 1, 1, 8}, {2, 2, 3, 5}, {6, 4, 4, 40}};
    size_t rows = 0;
    for (const Seq& q : seqs) rows += q.nq;
    const Inputs in = inputs(seed, sh, rows);
    std::vector<std::vector<float>> start;
    for (uint32_t s = 0; s < 5; ++s) start.push_back(uniform(sh.slot_floats(), seed + 20 + s, -0.5f, 0.5f));
    const size_t C = sh.channels(), Hv = sh.v_heads, Dv = sh.v_dim;
    const Run one = run(vk, sh, seqs, in, start);
    std::vector<size_t> first(seqs.size(), 0);
    for (size_t i = 1; i < seqs.size(); ++i) first[i] = first[i - 1] + seqs[i - 1].nq;
    auto slice = [](const std::vector<float>& v, size_t width, size_t r0, size_t n) {
        return std::vector<float>(v.begin() + r0 * width, v.begin() + (r0 + n) * width);
    };
    auto rows_of = [&](size_t vi, size_t r0, size_t n) {
        Inputs part = in;
        part.x = slice(in.x, C, first[vi] + r0, n);
        part.alpha = slice(in.alpha, Hv, first[vi] + r0, n);
        part.b = slice(in.b, Hv, first[vi] + r0, n);
        return part;
    };
    size_t runs = 1;
    {
        std::vector<Seq> rev(seqs.rbegin(), seqs.rend());
        Inputs part = in;
        part.x.clear();
        part.alpha.clear();
        part.b.clear();
        for (size_t k = seqs.size(); k-- > 0;) {
            const Inputs q = rows_of(k, 0, seqs[k].nq);
            part.x.insert(part.x.end(), q.x.begin(), q.x.end());
            part.alpha.insert(part.alpha.end(), q.alpha.begin(), q.alpha.end());
            part.b.insert(part.b.end(), q.b.begin(), q.b.end());
        }
        const Run r = run(vk, sh, rev, part, start);
        size_t at = 0;
        for (size_t k = seqs.size(); k-- > 0;) {
            require(same_bits(slice(r.conv, C, at, seqs[k].nq), slice(one.conv, C, first[k], seqs[k].nq)) &&
                        same_bits(slice(r.delta, Hv * Dv, at, seqs[k].nq), slice(one.delta, Hv * Dv, first[k], seqs[k].nq)),
                    "the view order changed a device result");
            at += seqs[k].nq;
        }
        for (size_t s = 0; s < start.size(); ++s) require(same_bits(r.slots[s], one.slots[s]), "the view order changed a device state");
        ++runs;
    }
    for (size_t vi = 0; vi < seqs.size(); ++vi)
        for (size_t pass : {size_t(0), size_t(1), size_t(2), size_t(3)}) {
            const Seq& q = seqs[vi];
            std::vector<std::vector<float>> slots = start;
            std::vector<float> conv, delta;
            size_t done = 0;
            while (done < q.nq) {
                const size_t n = pass ? std::min(pass, q.nq - done) : q.nq;
                const Seq step = {q.length + done, done ? q.dst : q.src, q.dst, n};
                const Run r = run(vk, sh, {step}, rows_of(vi, done, n), slots);
                conv.insert(conv.end(), r.conv.begin(), r.conv.end());
                delta.insert(delta.end(), r.delta.begin(), r.delta.end());
                slots = r.slots;
                done += n;
                ++runs;
            }
            require(same_bits(conv, slice(one.conv, C, first[vi], q.nq)) && same_bits(delta, slice(one.delta, Hv * Dv, first[vi], q.nq)),
                    "cutting a sequence into device passes changed its rows");
            require(same_bits(slots[q.dst], one.slots[q.dst]), "cutting a sequence into device passes changed its state");
        }
    return runs;
}

// A decay below 2^-126 is 0 and one just above it is kept, on a state whose decayed values stay normal, so the check holds under any denormal handling: with beta 0 the token writes nothing and the state left is the decayed one.
void check_decay_flush(backend::Backend& vk) {
    const StateShape sh = {1, 1, 8, 8};
    const std::vector<Seq> seq = {{5, 0, 0, 1}};
    for (float bias : {88.0f, 87.0f}) {
        Inputs in = inputs(61, sh, 1);
        in.a = {-1.0f};
        in.alpha = {0.0f};
        in.dt_bias = {bias};
        in.b = {-200.0f};
        std::vector<std::vector<float>> start = {uniform(sh.slot_floats(), 62, -0.5f, 0.5f)};
        const auto magnitude = uniform(sh.matrix_floats(), 63, 1.0f, 2.0f);
        for (size_t i = 0; i < sh.matrix_floats(); ++i) start[0][i] = i % 2 ? -magnitude[i] : magnitude[i];
        const Run r = run(vk, sh, seq, in, start);
        const float kept = std::exp(-bias);
        for (size_t i = 0; i < sh.matrix_floats(); ++i) {
            if (bias == 88.0f) {
                require(r.slots[0][i] == 0.0f, "a decay of exp(-88) was not flushed to 0 on the device");
                continue;
            }
            const float want = start[0][i] * kept;
            require(std::fpclassify(r.slots[0][i]) == FP_NORMAL, "a decayed state value below the normal range on the device");
            // The device's exp is exp2 of g log2(e), whose rounded exponent puts about |g| units of 2^-24 into the result.
            require(std::fabs((double)r.slots[0][i] - want) <= 1e-5 * std::fabs((double)want), "a decay of exp(-87) was not kept on the device");
        }
    }
}

// state_alloc zero-fills every slot of every layer on the device, and each slot holds what is written to it.
size_t check_storage(backend::Backend& vk) {
    const StateShape sh = {2, 6, 12, 10};
    const size_t layers = 3, slots = 4, n = sh.slot_floats();
    auto s = vk.state_alloc(layers, slots, sh);
    require(s->layers() == layers && s->slots() == slots, "device state storage counts");
    std::vector<std::vector<std::vector<float>>> held(layers);
    for (size_t l = 0; l < layers; ++l) {
        require(s->layer(l).size() >= slots * n * sizeof(float), "a device state layer's size");
        const auto all = read_floats(vk, s->layer(l), slots * n);
        require(std::all_of(all.begin(), all.end(), [](float f) { return f == 0.0f; }), "state_alloc left a device slot unzeroed");
        for (size_t k = 0; k < slots; ++k) {
            held[l].push_back(uniform(n, 70 + (uint32_t)(l * slots + k)));
            vk.write(s->layer(l), k * n * sizeof(float), held[l][k].data(), n * sizeof(float));
        }
    }
    for (size_t l = 0; l < layers; ++l)
        for (size_t k = 0; k < slots; ++k)
            require(same_bits(read_floats(vk, s->layer(l), n, k * n), held[l][k]), "a device state slot does not hold what was written to it");
    return layers * slots;
}

// The gated norm against the CPU, in place too, heads near 1e-4 holding eps.
size_t check_gated_norm(Pair& p, size_t heads, size_t dim, backend::NormGate gate = backend::NormGate::silu) {
    const size_t rows = 23, n = rows * heads * dim;
    auto x = uniform(n, 80, -2.0f, 2.0f);
    auto z = uniform(n, 81, -6.0f, 6.0f);
    z[0] = 0.0f;
    z[1] = -4.0f;
    const auto w = uniform(dim, 82, 0.5f, 1.5f);
    for (size_t i = 0; i < dim; ++i) x[(5 * heads + 1) * dim + i] *= 1e-4f;
    Pair::In xi = p.in(x), zi = p.in(z), wi = p.in(w);
    Pair::Out d = p.out(n);
    p.cpu.gated_rms_norm(d.cs(), xi.cs(), zi.cs(), wi.cs(), rows, heads, dim, 1e-6f, {}, gate);
    p.dev.gated_rms_norm(d.vs(), xi.vs(), zi.vs(), wi.vs(), rows, heads, dim, 1e-6f, {}, gate);
    auto r = p.results(d);
    size_t values = close(r.first, r.second, 1e-5, "gated_rms_norm differs beyond 1e-5");
    Pair::Out ip = p.out(n);
    p.dev.write(*ip.v, 0, x.data(), n * sizeof(float));
    p.dev.gated_rms_norm(ip.vs(), ip.vs(), zi.vs(), wi.vs(), rows, heads, dim, 1e-6f, {}, gate);
    std::vector<float> in_place(n);
    p.dev.read(*ip.v, 0, in_place.data(), n * sizeof(float));
    require(same_bits(in_place, r.second), "gated_rms_norm in place differs on the device");
    return values;
}

// sigmoid_mul as the output gate, each head's gate read between its q rows, and as a scale of one value per row, against the CPU; in place too.
size_t check_sigmoid_mul(Pair& p, size_t heads, size_t dim) {
    const size_t rows = 21;
    const auto q = uniform(rows * heads * 2 * dim, 83, -8.0f, 8.0f), x = uniform(rows * heads * dim, 84, -3.0f, 3.0f);
    const auto per_row = uniform(rows, 85, -8.0f, 8.0f);
    Pair::In qi = p.in(q), xi = p.in(x), pi = p.in(per_row);
    size_t values = 0;
    for (int mode = 0; mode < 2; ++mode) {
        const size_t h = mode ? heads * dim : heads, dd = mode ? 1 : dim, gs = mode ? 1 : heads * 2 * dim, ghs = mode ? 0 : 2 * dim;
        const backend::CSlice gc = mode ? pi.cs() : backend::CSlice{qi.c.get(), dim}, gv = mode ? pi.vs() : backend::CSlice{qi.v.get(), dim};
        Pair::Out d = p.out(x.size());
        p.cpu.sigmoid_mul(d.cs(), xi.cs(), gc, rows, h, dd, gs, ghs);
        p.dev.sigmoid_mul(d.vs(), xi.vs(), gv, rows, h, dd, gs, ghs);
        auto r = p.results(d);
        values += close(r.first, r.second, 1e-6, "sigmoid_mul differs beyond 1e-6");
        Pair::Out ip = p.out(x.size());
        p.dev.write(*ip.v, 0, x.data(), x.size() * sizeof(float));
        p.dev.sigmoid_mul(ip.vs(), ip.vs(), gv, rows, h, dd, gs, ghs);
        std::vector<float> in_place(x.size());
        p.dev.read(*ip.v, 0, in_place.data(), x.size() * sizeof(float));
        require(same_bits(in_place, r.second), "sigmoid_mul in place differs on the device");
    }
    return values;
}

// The qwen35 ops' checks every device takes (docs/TESTS.md, backend-vulkan); `runs` counts the device runs held bit for bit.
size_t check_ops(Pair& p, size_t& runs) {
    backend::Backend& vk = p.dev;
    size_t values = check_storage(vk);
    // Hv = Hk and Hv = 3 Hk at the tiny fixtures' widths, a shape past one column block, the files' 128 by 128 matrices, and the 0.8B's and the 27B's heads, whose slots are too large to cut into passes of a row here.
    const std::vector<backend::StateShape> shapes = {{2, 2, 12, 10}, {2, 6, 12, 10}, {2, 4, 64, 40}, {2, 4, 128, 128}, {16, 16, 128, 128}, {16, 48, 128, 128}};
    values += check_layer_tail(vk);
    uint32_t seed = 100;
    for (const backend::StateShape& sh : shapes) {
        try {
            values += check_mix(vk, p.cpu, sh, seed);
            if (sh.k_heads == 2) runs += check_invariance(vk, sh, seed + 1);
        } catch (const std::runtime_error&) {
            std::fprintf(stderr, "  linear attention K heads %zu V heads %zu, %zu by %zu\n", sh.k_heads, sh.v_heads, sh.k_dim, sh.v_dim);
            throw;
        }
        seed += 50;
    }
    check_decay_flush(vk);
    values += check_gated_norm(p, 2, 10) + check_gated_norm(p, 16, 128);
    values += check_gated_norm(p, 2, 8, backend::NormGate::sigmoid) + check_gated_norm(p, 16, 128, backend::NormGate::sigmoid);
    values += check_sigmoid_mul(p, 4, 40) + check_sigmoid_mul(p, 3, 256);
    return values;
}

// Embedded-drafter argmax and embedding ops must match the CPU, including ties, nonfinite logits, prior ids and out-of-range ids (docs/TESTS.md), on every device that drafts.
// `slice` is the ids a device's argmax reduces before merging, so rows of a slice less and more one id reach both sides of it; `matrix` writes a table of each type the device embeds.
size_t check_drafter_ops(Pair& p, size_t slice, std::vector<uint8_t> (*matrix)(uint32_t type, size_t in, size_t rows, uint32_t seed)) {
    size_t values = 0;
    auto ids_on = [](backend::Backend& b, const std::vector<uint32_t>& ids) {
        backend::BufferPtr buf = b.alloc(ids.size() * sizeof(float), backend::Memory::device);
        b.write(*buf, 0, ids.data(), ids.size() * sizeof(uint32_t));
        return buf;
    };
    auto ids_of = [](backend::Backend& b, const backend::Buffer& buf, size_t n) {
        std::vector<uint32_t> ids(n);
        b.read(buf, 0, ids.data(), n * sizeof(uint32_t));
        return ids;
    };
    // Rows of one slice of the device's reduction, of a slice less one id and plus one, whose last slice is one id, and of a vocabulary's 61 slices; 64 rows a call, as many as a pass drafts.
    for (size_t n : {size_t(37), slice - 1, slice + 1, size_t(248320)}) {
        const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
        const size_t rows = 64;
        std::vector<float> flat = uniform(rows * n, (uint32_t)(300 + n % 7), -6.0f, 6.0f);
        flat[1 * n + 3] = flat[1 * n + n - 2] = 9.5f;   // a tie
        flat[2 * n + n / 2] = inf;
        flat[3 * n] = nan;                               // a NaN in the first slice, the largest in the last
        flat[3 * n + n - 1] = 9.5f;
        flat[4 * n + n - 1] = nan;                       // and a NaN in the last, the largest in the first
        flat[4 * n] = 9.5f;
        for (size_t i = 0; i < n; ++i) flat[5 * n + i] = -inf;
        flat[6 * n + 7] = 9.0f;                          // its prior id is invalid
        flat[7 * n + n - 1] = 9.25f;                     // the largest in the last slice
        flat[8 * n] = 9.25f;                             // and in the first
        flat[9 * n] = flat[9 * n + n - 1] = 9.75f;       // a tie between the first id and the last
        flat[10 * n + n - 1] = inf;                      // an infinity in the last slice
        flat[11 * n + n / 2 - 1] = flat[11 * n + n / 2] = 9.75f;   // a tie across the middle, where a group of two splits the row
        for (size_t i = 0; i < n / 2; ++i) flat[12 * n + i] = -inf;   // a first half of -inf
        for (size_t i = 0; i < n; ++i) flat[13 * n + i] = i % 2 ? 0.0f : -0.0f;   // zeros of both signs
        for (size_t i = 0; i < n; ++i) flat[14 * n + i] = -1.0f;
        flat[14 * n + 1] = -0.0f;                        // -0 first, +0 last as the largest
        flat[14 * n + n - 1] = 0.0f;
        for (size_t i = 0; i < n; ++i) flat[15 * n + i] = -inf;
        flat[15 * n + n - 1] = nan;                      // -inf and a NaN in the last slice
        std::vector<uint32_t> prior(rows, 0);
        prior[6] = (uint32_t)n;
        const backend::BufferPtr lc = floats_on(p.cpu, flat), lv = floats_on(p.dev, flat), pc = ids_on(p.cpu, prior), pv = ids_on(p.dev, prior);
        std::vector<uint32_t> whole[2];
        for (bool with_prior : {true, false}) {
            const backend::BufferPtr oc = p.cpu.alloc(rows * sizeof(float), backend::Memory::device), ov = p.dev.alloc(rows * sizeof(float), backend::Memory::device);
            p.cpu.argmax_rows({oc.get(), 0}, {lc.get(), 0}, rows, n, with_prior ? backend::CSlice{pc.get(), 0} : backend::CSlice{});
            p.dev.argmax_rows({ov.get(), 0}, {lv.get(), 0}, rows, n, with_prior ? backend::CSlice{pv.get(), 0} : backend::CSlice{});
            p.dev.sync();
            const std::vector<uint32_t> c = ids_of(p.cpu, *oc, rows), v = ids_of(p.dev, *ov, rows);
            require(c == v, ("argmax_rows differs from the CPU over " + std::to_string(n) + " ids").c_str());
            require(c[1] == 3 && c[2] == n && c[3] == n && c[4] == n && c[5] == n && c[6] == (with_prior ? n : 7), "argmax_rows over ties, infinities and NaN");
            require(c[7] == n - 1 && c[8] == 0 && c[9] == 0 && c[10] == n, "argmax_rows over the first and the last slice of a row");
            require(c[11] == n / 2 - 1 && c[13] == 0 && c[14] == 1 && c[15] == n, "argmax_rows over a tie in the middle, signed zeros and -inf");
            whole[with_prior] = c;
            values += rows;
        }
        // The row split into a tensor group's 2 to 4 vocabulary slices: argmax_part on each, the candidates summed as Collective::sum_in_place sums them, and argmax_pick give argmax_rows' ids on both sides.
        for (size_t W : {size_t(2), size_t(3), size_t(4)}) {
            for (backend::Backend* b : std::initializer_list<backend::Backend*>{&p.cpu, &p.dev}) {
                std::vector<float> total(rows * 3 * W);
                for (size_t k = 0; k < W; ++k) {
                    const size_t a = n * k / W, e = n * (k + 1) / W;
                    std::vector<float> part(rows * (e - a));
                    for (size_t r = 0; r < rows; ++r) std::copy(flat.begin() + r * n + a, flat.begin() + r * n + e, part.begin() + r * (e - a));
                    const backend::BufferPtr lp = floats_on(*b, part), cands = b->alloc(total.size() * sizeof(float), backend::Memory::device);
                    b->argmax_part({cands.get(), 0}, {lp.get(), 0}, rows, e - a, a, k, W);
                    b->sync();
                    const std::vector<float> got = read_floats(*b, *cands, total.size());
                    for (size_t i = 0; i < total.size(); ++i) total[i] = k ? total[i] + got[i] : got[i];
                }
                const backend::BufferPtr summed = floats_on(*b, total), pb = ids_on(*b, prior);
                for (bool with_prior : {true, false}) {
                    const backend::BufferPtr out = b->alloc(rows * sizeof(float), backend::Memory::device);
                    b->argmax_pick({out.get(), 0}, {summed.get(), 0}, rows, W, n, with_prior ? backend::CSlice{pb.get(), 0} : backend::CSlice{});
                    b->sync();
                    require(ids_of(*b, *out, rows) == whole[with_prior],
                            ("argmax_part and argmax_pick over " + std::to_string(W) + " slices differ from argmax_rows over " + std::to_string(n) + " ids").c_str());
                    values += rows;
                }
            }
        }
    }
    const size_t width = 64, vocab = 9;
    const std::vector<float> table = uniform(width * vocab, 333, -2.0f, 2.0f);
    std::vector<uint8_t> q8(vocab * (width / quant::Q8_0_BLOCK) * quant::Q8_0_TYPESIZE);
    for (size_t r = 0; r < vocab; ++r)
        quant::quantize_row_q8_0(table.data() + r * width, q8.data() + r * (width / quant::Q8_0_BLOCK) * quant::Q8_0_TYPESIZE, width / quant::Q8_0_BLOCK);
    const std::vector<uint32_t> ids = {4, (uint32_t)vocab, 0, 0xffffffffu, 8};
    for (bool quantized : {false, true}) {
        const uint32_t type = quantized ? quant::GGML_TYPE_Q8_0 : quant::GGML_TYPE_F32;
        const void* data = quantized ? (const void*)q8.data() : (const void*)table.data();
        const size_t bytes = quantized ? q8.size() : table.size() * sizeof(float);
        const backend::BufferPtr tc = p.cpu.adopt(data, bytes), tv = p.dev.adopt(data, bytes), ic = ids_on(p.cpu, ids), iv = ids_on(p.dev, ids);
        const backend::BufferPtr dc = p.cpu.alloc(ids.size() * width * sizeof(float), backend::Memory::device), dv = p.dev.alloc(ids.size() * width * sizeof(float), backend::Memory::device);
        p.cpu.embed_ids({dc.get(), 0}, {type, {tc.get(), 0}}, width, vocab, {ic.get(), 0}, ids.size());
        p.dev.embed_ids({dv.get(), 0}, {type, {tv.get(), 0}}, width, vocab, {iv.get(), 0}, ids.size());
        p.dev.sync();
        const std::vector<float> c = read_floats(p.cpu, *dc, ids.size() * width), v = read_floats(p.dev, *dv, ids.size() * width);
        require(same_bits(c, v), quantized ? "embed_ids differs from the CPU on a Q8_0 table" : "embed_ids differs from the CPU on an F32 table");
        for (size_t i = 0; i < width; ++i) require(v[width + i] == 0.0f && v[3 * width + i] == 0.0f, "embed_ids of an id past the table wrote no zero row");
        values += c.size();
    }
    // Every type the device embeds: embed_ids gives embed's rows for the ids inside the table, through the same decoding, and a zero row past it.
    const size_t wide = 256;
    for (uint32_t type : {quant::GGML_TYPE_F32, quant::GGML_TYPE_F16, quant::GGML_TYPE_BF16, quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_Q4_K,
                          quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_MXFP4, quant::GGML_TYPE_TQ1_0, quant::GGML_TYPE_TQ2_0}) {
        if (!p.dev.supports_type(type)) continue;
        const std::vector<uint8_t> bytes = matrix(type, wide, vocab, 340 + type);
        const backend::BufferPtr tv = p.dev.adopt(bytes.data(), bytes.size()), iv = ids_on(p.dev, ids);
        const backend::BufferPtr dv = p.dev.alloc(ids.size() * wide * sizeof(float), backend::Memory::device), ev = p.dev.alloc(ids.size() * wide * sizeof(float), backend::Memory::device);
        const std::vector<float> sentinel(ids.size() * wide, 17.0f);
        p.dev.write(*dv, 0, sentinel.data(), sentinel.size() * sizeof(float));
        const std::vector<uint32_t> inside = {4, 0, 0, 0, 8};
        p.dev.embed_ids({dv.get(), 0}, {type, {tv.get(), 0}}, wide, vocab, {iv.get(), 0}, ids.size());
        p.dev.embed({ev.get(), 0}, {type, {tv.get(), 0}}, wide, vocab, inside.data(), inside.size());
        p.dev.sync();
        const std::vector<float> v = read_floats(p.dev, *dv, ids.size() * wide), e = read_floats(p.dev, *ev, ids.size() * wide);
        for (size_t r : {size_t(0), size_t(2), size_t(4)})
            require(std::memcmp(v.data() + r * wide, e.data() + r * wide, wide * sizeof(float)) == 0, ("embed_ids differs from embed on a table of type " + std::to_string(type)).c_str());
        for (size_t i = 0; i < wide; ++i) require(v[wide + i] == 0.0f && v[3 * wide + i] == 0.0f, "embed_ids of an id past the table wrote no zero row");
        values += v.size();
    }
    return values;
}
}  // namespace q35
