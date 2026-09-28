// The CPU backend's ops of the qwen35 layers (docs/QWEN35.md, The forward pass) against references written here from the math in double precision, each within the bound stated beside it.
// Also that no result depends on the thread count, on how rows are grouped into calls or on the block a V column runs in, bit for bit, that length 0 reads a zero state, and that a backend without the ops refuses each by name.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "backends/cpu/cpu_backend.hpp"

namespace {
using backend::BufferPtr;
using backend::CpuBackend;
using backend::StateShape;
using backend::StateStorage;
using backend::StateView;

// One unit in the last place of 1 in F32; every bound below is a count of it times a magnitude.
const double U = std::ldexp(1.0, -24);

// The thread counts every op is run at, whose results must be the same bits.
const int kThreadCounts[] = {1, 2, 3, 5, 8, 16};

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

std::vector<float> uniform(std::mt19937& g, size_t n, float lo, float hi) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (float& x : v) x = d(g);
    return v;
}

// Zero-filled storage of `bytes`.
BufferPtr zeros(CpuBackend& cpu, size_t bytes) { return cpu.alloc(bytes, backend::Memory::device); }

BufferPtr upload(CpuBackend& cpu, const std::vector<float>& v) {
    BufferPtr b = zeros(cpu, std::max<size_t>(v.size(), 1) * sizeof(float));
    if (!v.empty()) cpu.write(*b, 0, v.data(), v.size() * sizeof(float));
    return b;
}

std::vector<float> download(CpuBackend& cpu, const backend::Buffer& b, size_t n, size_t offset = 0) {
    std::vector<float> v(n);
    if (n) cpu.read(b, offset * sizeof(float), v.data(), n * sizeof(float));
    return v;
}

std::vector<float> read_slot(CpuBackend& cpu, const StateStorage& s, size_t layer, size_t slot) {
    const size_t n = s.shape().slot_floats();
    return download(cpu, s.layer(layer), n, slot * n);
}

void write_slot(CpuBackend& cpu, StateStorage& s, size_t layer, size_t slot, const std::vector<float>& v) {
    const size_t n = s.shape().slot_floats();
    cpu.write(s.layer(layer), slot * n * sizeof(float), v.data(), n * sizeof(float));
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

bool same_slots(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!same_bits(a[i], b[i])) return false;
    return true;
}

double silu(double x) { return x / (1.0 + std::exp(-x)); }
double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// Worst error seen, as a fraction of its bound, per op.
struct Worst {
    double conv = 0, delta = 0, norm = 0, rope = 0, gate = 0;
};
Worst worst;

void check_close(double got, double ref, double bound, double& seen, const std::string& what) {
    const double err = std::fabs(got - ref);
    if (!(err <= bound)) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "%s: got %.9g, reference %.9g, error %.3g over the bound %.3g", what.c_str(), got, ref, err, bound);
        throw std::runtime_error(msg);
    }
    seen = std::max(seen, bound > 0 ? err / bound : 0.0);
}

// A sequence of the linear-attention tests: its history length before the call, its slots, and its rows' inputs.
struct Seq {
    size_t length, src, dst, nq;
};

// The inputs of one linear-attention layer's rows, in the order of a call's views.
struct LinearInputs {
    std::vector<float> x, w, alpha, b, a, dt_bias;
};

LinearInputs linear_inputs(std::mt19937& g, const StateShape& sh, size_t rows) {
    LinearInputs in;
    in.x = uniform(g, rows * sh.channels(), -1.0f, 1.0f);
    in.w = uniform(g, sh.channels() * backend::kConvTaps, -0.8f, 0.8f);
    in.alpha = uniform(g, rows * sh.v_heads, -3.0f, 3.0f);
    in.b = uniform(g, rows * sh.v_heads, -4.0f, 4.0f);
    in.a = uniform(g, sh.v_heads, -2.0f, -0.05f);
    in.dt_bias = uniform(g, sh.v_heads, -2.0f, 2.0f);
    return in;
}

// A slot as a history would leave it: bounded matrices and carried rows.
std::vector<float> random_slot(std::mt19937& g, const StateShape& sh) {
    return uniform(g, sh.slot_floats(), -0.5f, 0.5f);
}

// ---------------------------------------------------------------------------------------------------------------------
// References from the math (docs/QWEN35.md, Linear attention), in double.

// The conv for one view: out[t][c] = silu(sum_i w[c][i] x[t - 3 + i][c]) over the carried rows and the view's rows, rows before position 0 zero; also the raw rows it must leave, exactly.
void conv_reference(const StateShape& sh, const float* x, size_t nq, const float* w, const std::vector<float>& src, size_t length,
                    std::vector<double>& out, std::vector<double>& magnitude, std::vector<float>& carried) {
    const size_t C = sh.channels(), T = backend::kConvTaps, first = sh.v_heads * sh.matrix_floats();
    // Raw values from position length - (T - 1) on.
    auto raw = [&](size_t index, size_t c) -> float {
        if (index < T - 1) {
            if (length + index < T - 1) return 0.0f;
            return src[first + index * C + c];
        }
        return x[(index - (T - 1)) * C + c];
    };
    out.assign(nq * C, 0.0);
    magnitude.assign(nq * C, 0.0);
    for (size_t t = 0; t < nq; ++t)
        for (size_t c = 0; c < C; ++c) {
            double s = 0, m = 0;
            for (size_t i = 0; i < T; ++i) {
                const double p = (double)w[c * T + i] * (double)raw(t + i, c);
                s += p;
                m += std::fabs(p);
            }
            out[t * C + c] = silu(s);
            magnitude[t * C + c] = m;
        }
    carried.assign((T - 1) * C, 0.0f);
    for (size_t j = 0; j + 1 < T; ++j)
        for (size_t c = 0; c < C; ++c) carried[j * C + c] = raw(nq + j, c);
}

// The gated delta rule for one view from its slot's matrices (zero at length 0): out is nq rows of v_heads * v_dim, and S the matrices after.
void delta_reference(const StateShape& sh, const float* qkv, const float* alpha, const float* b, const float* a, const float* dt_bias,
                     size_t nq, const std::vector<float>& src, size_t length, std::vector<double>& out, std::vector<double>& S) {
    const size_t Hk = sh.k_heads, Hv = sh.v_heads, Dk = sh.k_dim, Dv = sh.v_dim, C = sh.channels(), M = sh.matrix_floats();
    S.assign(Hv * M, 0.0);
    if (length)
        for (size_t i = 0; i < Hv * M; ++i) S[i] = src[i];
    out.assign(nq * Hv * Dv, 0.0);
    std::vector<double> qn(Dk), kn(Dk), m(Dv), d(Dv);
    for (size_t t = 0; t < nq; ++t) {
        const float* row = qkv + t * C;
        for (size_t j = 0; j < Hv; ++j) {
            const size_t kh = j % Hk;
            double sq = 0, sk = 0;
            for (size_t i = 0; i < Dk; ++i) {
                sq += (double)row[kh * Dk + i] * row[kh * Dk + i];
                sk += (double)row[Hk * Dk + kh * Dk + i] * row[Hk * Dk + kh * Dk + i];
            }
            const double iq = 1.0 / std::sqrt(sq + 1e-6) / std::sqrt((double)Dk), ik = 1.0 / std::sqrt(sk + 1e-6);
            for (size_t i = 0; i < Dk; ++i) {
                qn[i] = row[kh * Dk + i] * iq;
                kn[i] = row[Hk * Dk + kh * Dk + i] * ik;
            }
            const double beta = sigmoid(b[t * Hv + j]);
            const double z = (double)alpha[t * Hv + j] + dt_bias[j];
            const double sp = z > 20 ? z : std::log1p(std::exp(z));
            double decay = std::exp((double)a[j] * sp);
            if (decay < std::ldexp(1.0, -126)) decay = 0;
            double* Sj = S.data() + j * M;
            for (size_t i = 0; i < M; ++i) Sj[i] *= decay;
            for (size_t c = 0; c < Dv; ++c) {
                double s = 0;
                for (size_t i = 0; i < Dk; ++i) s += Sj[i * Dv + c] * kn[i];
                m[c] = s;
                d[c] = beta * ((double)row[2 * Hk * Dk + j * Dv + c] - s);
            }
            for (size_t i = 0; i < Dk; ++i)
                for (size_t c = 0; c < Dv; ++c) Sj[i * Dv + c] += kn[i] * d[c];
            for (size_t c = 0; c < Dv; ++c) {
                double o = 0;
                for (size_t i = 0; i < Dk; ++i) o += Sj[i * Dv + c] * qn[i];
                out[(t * Hv + j) * Dv + c] = o;
            }
        }
    }
}

// A view's delta-rule rows and matrices against the reference's: each token adds at most 2 sums of k_dim products to the error of state and output, which the decay and the unit-norm key do not grow, so (t + 1) (2 k_dim + 16) units of the magnitude.
void check_delta(const StateShape& sh, const float* rows, const float* state, const std::vector<double>& out, const std::vector<double>& S, size_t nq,
                 const std::string& what) {
    const size_t Hv = sh.v_heads, Dv = sh.v_dim, Dk = sh.k_dim, M = sh.matrix_floats();
    for (size_t j = 0; j < Hv; ++j) {
        double scale = 1;
        for (size_t i = 0; i < M; ++i) scale = std::max(scale, std::fabs(S[j * M + i]));
        for (size_t t = 0; t < nq; ++t)
            for (size_t c = 0; c < Dv; ++c) {
                const size_t k = (t * Hv + j) * Dv + c;
                check_close(rows[k], out[k], (t + 1) * (2.0 * Dk + 16) * U * scale, worst.delta,
                            what + ": delta rule row " + std::to_string(t) + " head " + std::to_string(j));
            }
        for (size_t i = 0; i < M; ++i)
            check_close(state[j * M + i], S[j * M + i], nq * (2.0 * Dk + 16) * U * scale, worst.delta, what + ": state of head " + std::to_string(j));
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The linear-attention ops against the references.

struct LinearRun {
    std::vector<float> conv, delta;                 // the rows of every view, in view order
    std::vector<std::vector<float>> slots;           // every slot of layer `layer` after the call
};

// One conv and one gated delta rule call over the views on a storage whose slots start as `start`, on layer `layer` of `layers`.
LinearRun run_linear(CpuBackend& cpu, const StateShape& sh, const std::vector<Seq>& seqs, const LinearInputs& in,
                     const std::vector<std::vector<float>>& start, size_t layer = 1, size_t layers = 2) {
    auto storage = cpu.state_alloc(layers, start.size(), sh);
    for (size_t s = 0; s < start.size(); ++s) write_slot(cpu, *storage, layer, s, start[s]);
    std::vector<StateView> views;
    size_t rows = 0;
    for (const Seq& q : seqs) {
        views.push_back({storage.get(), q.src, q.dst, q.length, q.nq});
        rows += q.nq;
    }
    const size_t C = sh.channels(), Hv = sh.v_heads;
    BufferPtr x = upload(cpu, in.x), w = upload(cpu, in.w), alpha = upload(cpu, in.alpha), b = upload(cpu, in.b);
    BufferPtr a = upload(cpu, in.a), dt = upload(cpu, in.dt_bias);
    BufferPtr u = zeros(cpu, rows * C * sizeof(float)), o = zeros(cpu, rows * Hv * sh.v_dim * sizeof(float));
    cpu.causal_conv_silu({u.get(), 0}, {x.get(), 0}, {w.get(), 0}, layer, views.data(), views.size());
    cpu.gated_delta_rule({o.get(), 0}, {u.get(), 0}, {alpha.get(), 0}, {b.get(), 0}, {a.get(), 0}, {dt.get(), 0},
                         layer, views.data(), views.size());
    LinearRun r;
    r.conv = download(cpu, *u, rows * C);
    r.delta = download(cpu, *o, rows * Hv * sh.v_dim);
    for (size_t s = 0; s < start.size(); ++s) r.slots.push_back(read_slot(cpu, *storage, layer, s));
    // The other layer of the storage is never touched.
    for (size_t s = 0; s < start.size(); ++s)
        for (size_t l = 0; l < layers; ++l)
            if (l != layer) {
                const auto other = read_slot(cpu, *storage, l, s);
                require(std::all_of(other.begin(), other.end(), [](float f) { return f == 0.0f; }), "a state op wrote another layer");
            }
    return r;
}

// Every view's conv rows, carried rows, delta-rule rows and matrices against the references.
void check_linear(const StateShape& sh, const std::vector<Seq>& seqs, const LinearInputs& in, const std::vector<std::vector<float>>& start,
                  const LinearRun& r, const std::string& what) {
    const size_t C = sh.channels(), Hv = sh.v_heads, Dv = sh.v_dim;
    const size_t first_carried = Hv * sh.matrix_floats();
    size_t r0 = 0;
    for (size_t vi = 0; vi < seqs.size(); ++vi) {
        const Seq& q = seqs[vi];
        std::vector<double> ref, mag;
        std::vector<float> carried;
        conv_reference(sh, in.x.data() + r0 * C, q.nq, in.w.data(), start[q.src], q.length, ref, mag, carried);
        // A sum of 4 products and a SiLU in F32: 16 units of the products' magnitude, plus 16 of the result's.
        for (size_t i = 0; i < q.nq * C; ++i)
            check_close(r.conv[r0 * C + i], ref[i], 16 * U * (mag[i] + std::fabs(ref[i])) + 1e-30, worst.conv,
                        what + ": conv view " + std::to_string(vi) + " value " + std::to_string(i));
        for (size_t i = 0; i < carried.size(); ++i)
            require(std::memcmp(&r.slots[q.dst][first_carried + i], &carried[i], sizeof(float)) == 0,
                    what + ": carried row value " + std::to_string(i) + " of view " + std::to_string(vi) + " is not the raw row");
        std::vector<double> out, S;
        delta_reference(sh, r.conv.data() + r0 * C, in.alpha.data() + r0 * Hv, in.b.data() + r0 * Hv, in.a.data(), in.dt_bias.data(),
                        q.nq, start[q.src], q.length, out, S);
        check_delta(sh, r.delta.data() + r0 * Hv * Dv, r.slots[q.dst].data(), out, S, q.nq, what + ": view " + std::to_string(vi));
        // A slot only read keeps its bits.
        if (q.src != q.dst) require(same_bits(r.slots[q.src], start[q.src]), what + ": a verify changed its source slot");
        r0 += q.nq;
    }
}

// Sequences of every kind beside each other: fresh ones (length 0 on a slot of garbage), ones continuing a history of 1, 2 and 7 tokens, one-token entries, and a verify reading one slot and writing another.
size_t check_linear_mix(std::mt19937& g, const StateShape& sh) {
    const std::vector<Seq> seqs = {
        {0, 0, 0, 5}, {1, 1, 1, 3}, {2, 2, 2, 1}, {7, 3, 3, 6}, {0, 4, 4, 1}, {9, 5, 6, 4}, {3, 7, 7, 2},
    };
    size_t rows = 0;
    for (const Seq& q : seqs) rows += q.nq;
    const LinearInputs in = linear_inputs(g, sh, rows);
    std::vector<std::vector<float>> start;
    for (size_t s = 0; s < 8; ++s) start.push_back(random_slot(g, sh));
    // Length 0 must not read its slot, so the fresh ones hold NaN.
    start[0].assign(sh.slot_floats(), std::numeric_limits<float>::quiet_NaN());
    start[4].assign(sh.slot_floats(), std::numeric_limits<float>::quiet_NaN());
    // The same for the carried rows before position 0 of the short histories.
    const size_t C = sh.channels(), first = sh.v_heads * sh.matrix_floats();
    for (size_t c = 0; c < C; ++c) {
        start[1][first + 0 * C + c] = start[1][first + 1 * C + c] = std::numeric_limits<float>::quiet_NaN();
        start[2][first + 0 * C + c] = std::numeric_limits<float>::quiet_NaN();
    }
    CpuBackend cpu;
    cpu.set_threads(4);
    const LinearRun r = run_linear(cpu, sh, seqs, in, start);
    std::vector<std::vector<float>> clean = start;
    for (auto* s : {&clean[0], &clean[4]}) std::fill(s->begin(), s->end(), 0.0f);
    check_linear(sh, seqs, in, clean, r, "mix");
    for (float f : r.conv) require(std::isfinite(f), "mix: a conv row read a carried row before position 0 or a fresh slot");
    for (float f : r.delta) require(std::isfinite(f), "mix: the delta rule read the slot of a fresh sequence");
    // The same views on zeroed slots give the same bits: a fresh slot's garbage is never read.
    const LinearRun z = run_linear(cpu, sh, seqs, in, clean);
    require(same_bits(r.conv, z.conv) && same_bits(r.delta, z.delta), "mix: a fresh slot's contents changed a result");
    return rows;
}

// Thread counts and every grouping of rows into calls give the same bits: all sequences in one call, each alone, in another view order, and each cut into passes of 1, 2 and 3 rows that carry the state in their slot.
// The long sequence takes the wider shapes' conv past the rows it keeps on one thread.
size_t check_linear_invariance(std::mt19937& g, const StateShape& sh) {
    const std::vector<Seq> seqs = {{0, 0, 0, 9}, {4, 1, 1, 7}, {2, 2, 3, 5}, {6, 4, 4, 80}};
    size_t rows = 0;
    for (const Seq& q : seqs) rows += q.nq;
    const LinearInputs in = linear_inputs(g, sh, rows);
    std::vector<std::vector<float>> start;
    for (size_t s = 0; s < 5; ++s) start.push_back(random_slot(g, sh));
    const size_t C = sh.channels(), Hv = sh.v_heads, Dv = sh.v_dim;
    CpuBackend cpu;
    cpu.set_threads(1);
    const LinearRun one = run_linear(cpu, sh, seqs, in, start);
    check_linear(sh, seqs, in, start, one, "invariance");
    size_t runs = 1;
    for (int t : kThreadCounts) {
        if (t == 1) continue;
        cpu.set_threads(t);
        const LinearRun r = run_linear(cpu, sh, seqs, in, start);
        require(same_bits(r.conv, one.conv) && same_bits(r.delta, one.delta) && same_slots(r.slots, one.slots),
                "invariance: " + std::to_string(t) + " threads changed a result");
        ++runs;
    }
    // A sequence's rows of the inputs, and where its rows sit in the one-call result.
    std::vector<size_t> first(seqs.size(), 0);
    for (size_t i = 1; i < seqs.size(); ++i) first[i] = first[i - 1] + seqs[i - 1].nq;
    auto slice = [&](const std::vector<float>& v, size_t width, size_t r0, size_t n) {
        return std::vector<float>(v.begin() + r0 * width, v.begin() + (r0 + n) * width);
    };
    auto rows_of = [&](size_t vi, size_t r0, size_t n) {
        LinearInputs part = in;
        part.x = slice(in.x, C, first[vi] + r0, n);
        part.alpha = slice(in.alpha, Hv, first[vi] + r0, n);
        part.b = slice(in.b, Hv, first[vi] + r0, n);
        return part;
    };
    cpu.set_threads(3);
    // In reverse view order in one call.
    {
        std::vector<Seq> rev(seqs.rbegin(), seqs.rend());
        LinearInputs part = in;
        part.x.clear();
        part.alpha.clear();
        part.b.clear();
        for (size_t k = seqs.size(); k-- > 0;) {
            const LinearInputs p = rows_of(k, 0, seqs[k].nq);
            part.x.insert(part.x.end(), p.x.begin(), p.x.end());
            part.alpha.insert(part.alpha.end(), p.alpha.begin(), p.alpha.end());
            part.b.insert(part.b.end(), p.b.begin(), p.b.end());
        }
        const LinearRun r = run_linear(cpu, sh, rev, part, start);
        size_t at = 0;
        for (size_t k = seqs.size(); k-- > 0;) {
            require(same_bits(slice(r.conv, C, at, seqs[k].nq), slice(one.conv, C, first[k], seqs[k].nq)) &&
                    same_bits(slice(r.delta, Hv * Dv, at, seqs[k].nq), slice(one.delta, Hv * Dv, first[k], seqs[k].nq)),
                    "invariance: the view order changed a result");
            at += seqs[k].nq;
        }
        require(same_slots(r.slots, one.slots), "invariance: the view order changed a state");
        ++runs;
    }
    // Each sequence alone, whole and in passes of 1, 2 and 3 rows, each pass continuing the last in the destination slot.
    for (size_t vi = 0; vi < seqs.size(); ++vi) {
        for (size_t pass : {size_t(0), size_t(1), size_t(2), size_t(3)}) {
            const Seq& q = seqs[vi];
            std::vector<std::vector<float>> slots = start;
            std::vector<float> conv, delta;
            size_t done = 0;
            while (done < q.nq) {
                const size_t n = pass ? std::min(pass, q.nq - done) : q.nq;
                const Seq step = {q.length + done, done ? q.dst : q.src, q.dst, n};
                const LinearRun r = run_linear(cpu, sh, {step}, rows_of(vi, done, n), slots);
                conv.insert(conv.end(), r.conv.begin(), r.conv.end());
                delta.insert(delta.end(), r.delta.begin(), r.delta.end());
                slots = r.slots;
                done += n;
                ++runs;
            }
            require(same_bits(conv, slice(one.conv, C, first[vi], q.nq)) && same_bits(delta, slice(one.delta, Hv * Dv, first[vi], q.nq)),
                    "invariance: passes of " + std::to_string(pass) + " rows changed sequence " + std::to_string(vi) + "'s rows");
            require(same_bits(slots[q.dst], one.slots[q.dst]),
                    "invariance: passes of " + std::to_string(pass) + " rows changed sequence " + std::to_string(vi) + "'s state");
        }
    }
    return runs;
}

// A sequence decoded one row per call, as the decode steps are: each step meets the reference and leaves the reference's state for the next.
size_t check_one_token(std::mt19937& g, const StateShape& sh) {
    const size_t steps = 6;
    const LinearInputs in = linear_inputs(g, sh, steps);
    std::vector<std::vector<float>> slots = {random_slot(g, sh)};
    CpuBackend cpu;
    cpu.set_threads(2);
    const size_t C = sh.channels(), Hv = sh.v_heads;
    for (size_t t = 0; t < steps; ++t) {
        LinearInputs step = in;
        step.x.assign(in.x.begin() + t * C, in.x.begin() + (t + 1) * C);
        step.alpha.assign(in.alpha.begin() + t * Hv, in.alpha.begin() + (t + 1) * Hv);
        step.b.assign(in.b.begin() + t * Hv, in.b.begin() + (t + 1) * Hv);
        const std::vector<Seq> seq = {{t, 0, 0, 1}};
        const LinearRun r = run_linear(cpu, sh, seq, step, slots);
        check_linear(sh, seq, step, slots, r, "one-token step " + std::to_string(t));
        slots = r.slots;
    }
    return steps;
}

// A decay below 2^-126 is 0, so the state before the token is dropped whole, and one just above it is kept: with beta 0 the token writes nothing, and what is left is the decayed state.
// The matrix holds values of magnitude 1 to 2, so every decayed value stays normal and the check holds under any denormal handling.
void check_decay_flush(std::mt19937& g) {
    const StateShape sh = {1, 1, 8, 8};
    const std::vector<Seq> seq = {{5, 0, 0, 1}};
    for (float bias : {88.0f, 87.0f}) {
        LinearInputs in = linear_inputs(g, sh, 1);
        in.a = {-1.0f};
        in.alpha = {0.0f};
        in.dt_bias = {bias};
        in.b = {-200.0f};
        std::vector<std::vector<float>> start = {random_slot(g, sh)};
        const auto magnitude = uniform(g, sh.matrix_floats(), 1.0f, 2.0f);
        for (size_t i = 0; i < sh.matrix_floats(); ++i) start[0][i] = i % 2 ? -magnitude[i] : magnitude[i];
        CpuBackend cpu;
        const LinearRun r = run_linear(cpu, sh, seq, in, start);
        check_linear(sh, seq, in, start, r, "decay at " + std::to_string(bias));
        const float kept = std::exp(-bias);
        for (size_t i = 0; i < sh.matrix_floats(); ++i) {
            if (bias == 88.0f) {
                require(r.slots[0][i] == 0.0f, "a decay of exp(-88) was not flushed to 0");
                continue;
            }
            const float want = start[0][i] * kept;
            require(std::fpclassify(want) == FP_NORMAL, "a decayed state value below the normal range");
            require(same_bits(r.slots[0][i], want), "a decay of exp(-87) was not kept");
        }
    }
}

// One gated delta rule call of one view over the rows `qkv` given directly, from a slot holding `start`, which length 0 does not read; the rows, then the slot's matrices after.
struct DeltaRun {
    std::vector<float> rows, state;
};

DeltaRun run_delta(CpuBackend& cpu, const StateShape& sh, const std::vector<float>& qkv, const LinearInputs& in, const std::vector<float>& start,
                   size_t length, size_t nq) {
    auto storage = cpu.state_alloc(1, 1, sh);
    write_slot(cpu, *storage, 0, 0, start);
    const StateView view = {storage.get(), 0, 0, length, nq};
    BufferPtr u = upload(cpu, qkv), alpha = upload(cpu, in.alpha), b = upload(cpu, in.b), a = upload(cpu, in.a), dt = upload(cpu, in.dt_bias);
    BufferPtr o = zeros(cpu, nq * sh.v_heads * sh.v_dim * sizeof(float));
    cpu.gated_delta_rule({o.get(), 0}, {u.get(), 0}, {alpha.get(), 0}, {b.get(), 0}, {a.get(), 0}, {dt.get(), 0}, 0, &view, 1);
    DeltaRun r;
    r.rows = download(cpu, *o, nq * sh.v_heads * sh.v_dim);
    r.state = read_slot(cpu, *storage, 0, 0);
    r.state.resize(sh.v_heads * sh.matrix_floats());
    return r;
}

// Each V column's arithmetic is its own, so a column gives the same bits in a 32-column block, in an 8-column block and alone: columns of a v_dim 40 run against v_dim 8 and v_dim 1 runs fed the same q, k, gates and history.
void check_column_paths(std::mt19937& g) {
    const StateShape wide = {2, 4, 16, 40};
    const size_t nq = 5, Hk = wide.k_heads, Hv = wide.v_heads, Dk = wide.k_dim, Dv = wide.v_dim, qk = 2 * Hk * Dk;
    const LinearInputs in = linear_inputs(g, wide, nq);
    const std::vector<float> qkv = uniform(g, nq * wide.channels(), -1.0f, 1.0f), start = random_slot(g, wide);
    CpuBackend cpu;
    cpu.set_threads(2);
    const DeltaRun all = run_delta(cpu, wide, qkv, in, start, 3, nq);
    // The n columns from column c0 of every V head, as a run of their own.
    auto columns = [&](size_t c0, size_t n) {
        const StateShape sh = {Hk, Hv, Dk, n};
        const size_t C = sh.channels(), W = wide.channels();
        std::vector<float> rows(nq * C), slot(sh.slot_floats(), 0.0f);
        for (size_t r = 0; r < nq; ++r) {
            std::copy(qkv.begin() + r * W, qkv.begin() + r * W + qk, rows.begin() + r * C);
            for (size_t j = 0; j < Hv; ++j)
                for (size_t c = 0; c < n; ++c) rows[r * C + qk + j * n + c] = qkv[r * W + qk + j * Dv + c0 + c];
        }
        for (size_t j = 0; j < Hv; ++j)
            for (size_t i = 0; i < Dk; ++i)
                for (size_t c = 0; c < n; ++c) slot[(j * Dk + i) * n + c] = start[(j * Dk + i) * Dv + c0 + c];
        const DeltaRun r = run_delta(cpu, sh, rows, in, slot, 3, nq);
        const std::string what = "columns " + std::to_string(c0) + " to " + std::to_string(c0 + n - 1) + " alone";
        for (size_t j = 0; j < Hv; ++j)
            for (size_t c = 0; c < n; ++c) {
                for (size_t t = 0; t < nq; ++t)
                    require(same_bits(r.rows[(t * Hv + j) * n + c], all.rows[(t * Hv + j) * Dv + c0 + c]), what + " give other rows than in the wide run");
                for (size_t i = 0; i < Dk; ++i)
                    require(same_bits(r.state[(j * Dk + i) * n + c], all.state[(j * Dk + i) * Dv + c0 + c]), what + " leave another state than in the wide run");
            }
    };
    columns(0, 8);
    columns(24, 8);
    for (size_t c = 0; c < Dv; ++c) columns(c, 1);
}

// q and k heads near 1e-4, whose sums of squares lie far below kL2NormEps, hold the epsilon and its place inside the root.
void check_l2_eps(std::mt19937& g) {
    const StateShape sh = {2, 2, 16, 8};
    const size_t nq = 3, Hk = sh.k_heads, Dk = sh.k_dim, C = sh.channels();
    const LinearInputs in = linear_inputs(g, sh, nq);
    std::vector<float> qkv = uniform(g, nq * C, -1.0f, 1.0f);
    const std::vector<float> start = random_slot(g, sh);
    for (size_t r = 1; r < nq; ++r)
        for (size_t i = 0; i < Dk; ++i) {
            qkv[r * C + (r - 1) * Dk + i] *= 1e-4f;
            qkv[r * C + Hk * Dk + (r - 1) * Dk + i] *= 1e-4f;
        }
    CpuBackend cpu;
    const DeltaRun r = run_delta(cpu, sh, qkv, in, start, 2, nq);
    std::vector<double> out, S;
    delta_reference(sh, qkv.data(), in.alpha.data(), in.b.data(), in.a.data(), in.dt_bias.data(), nq, start, 2, out, S);
    check_delta(sh, r.rows.data(), r.state.data(), out, S, nq, "small q and k heads");
}

// state_alloc zero-fills whole slots and refuses a shape with V heads that are no multiple of the K heads; state_copy copies one slot in every layer and leaves the others.
void check_state_storage(std::mt19937& g) {
    CpuBackend cpu;
    const StateShape sh = {2, 6, 12, 10};
    require(sh.channels() == 2 * 2 * 12 + 6 * 10 && sh.slot_floats() == 6 * 12 * 10 + 3 * sh.channels(), "state shape sizes");
    auto s = cpu.state_alloc(3, 4, sh);
    require(s->layers() == 3 && s->slots() == 4, "state storage counts");
    for (size_t l = 0; l < 3; ++l) {
        require(s->layer(l).size() == 4 * sh.slot_floats() * sizeof(float), "a state layer's size");
        const auto all = download(cpu, s->layer(l), 4 * sh.slot_floats());
        require(std::all_of(all.begin(), all.end(), [](float f) { return f == 0.0f; }), "state_alloc left a slot unzeroed");
    }
    std::vector<std::vector<std::vector<float>>> slots(3);
    for (size_t l = 0; l < 3; ++l)
        for (size_t k = 0; k < 4; ++k) {
            slots[l].push_back(random_slot(g, sh));
            write_slot(cpu, *s, l, k, slots[l][k]);
        }
    cpu.state_copy(*s, 3, 1);
    cpu.state_copy(*s, 2, 2);
    for (size_t l = 0; l < 3; ++l)
        for (size_t k = 0; k < 4; ++k)
            require(same_bits(read_slot(cpu, *s, l, k), slots[l][k == 3 ? 1 : k]), "state_copy moved the wrong slot");
    bool refused = false;
    try { cpu.state_copy(*s, 4, 0); } catch (const std::runtime_error&) { refused = true; }
    require(refused, "state_copy took a slot outside the storage");
    for (const StateShape& bad : {StateShape{2, 3, 4, 4}, StateShape{0, 2, 4, 4}, StateShape{2, 2, 0, 4}}) {
        refused = false;
        try { cpu.state_alloc(1, 1, bad); } catch (const std::runtime_error&) { refused = true; }
        require(refused, "state_alloc took an invalid shape");
    }
    // A storage made by hand whose buffer is a float short of its slots, or missing, is refused, so no op writes past a buffer.
    const size_t bytes = 4 * sh.slot_floats() * sizeof(float);
    for (int missing = 0; missing < 2; ++missing) {
        std::vector<BufferPtr> buffers = {zeros(cpu, bytes), missing ? nullptr : zeros(cpu, bytes - sizeof(float))};
        refused = false;
        try { (void)StateStorage(std::move(buffers), 4, sh); } catch (const std::runtime_error&) { refused = true; }
        require(refused, missing ? "a state storage without a buffer was taken" : "a state storage whose buffer is short of its slots was taken");
    }
}

// Malformed views are refused before any output or state is written.
void check_view_refusals(std::mt19937& g) {
    const StateShape sh = {1, 2, 4, 4};
    CpuBackend cpu;
    auto s = cpu.state_alloc(2, 3, sh);
    auto other = cpu.state_alloc(2, 3, StateShape{1, 1, 4, 4});
    const LinearInputs in = linear_inputs(g, sh, 4);
    BufferPtr x = upload(cpu, in.x), w = upload(cpu, in.w), alpha = upload(cpu, in.alpha), b = upload(cpu, in.b);
    BufferPtr a = upload(cpu, in.a), dt = upload(cpu, in.dt_bias);
    const std::vector<float> marker(4 * sh.channels(), 7.0f);
    BufferPtr u = upload(cpu, marker), o = upload(cpu, marker);
    const std::vector<std::vector<StateView>> bad = {
        {{s.get(), 0, 0, 0, 2}, {s.get(), 1, 0, 0, 2}},   // two views write one slot
        {{s.get(), 0, 1, 1, 2}, {s.get(), 1, 2, 0, 2}},   // one view writes the slot a later one reads
        {{s.get(), 1, 0, 0, 2}, {s.get(), 2, 1, 1, 2}},   // one view reads the slot a later one writes
        {{s.get(), 3, 3, 0, 4}},                          // a slot outside the storage
        {{s.get(), 0, 0, 0, 0}},                          // a view without rows
        {{nullptr, 0, 0, 0, 4}},                          // no storage
        {{s.get(), 0, 0, 0, 2}, {other.get(), 0, 0, 0, 2}},   // two shapes in one call
    };
    for (size_t k = 0; k < bad.size(); ++k)
        for (int op = 0; op < 2; ++op) {
            bool refused = false;
            try {
                if (op == 0) cpu.causal_conv_silu({u.get(), 0}, {x.get(), 0}, {w.get(), 0}, 0, bad[k].data(), bad[k].size());
                else cpu.gated_delta_rule({o.get(), 0}, {x.get(), 0}, {alpha.get(), 0}, {b.get(), 0}, {a.get(), 0}, {dt.get(), 0},
                                          0, bad[k].data(), bad[k].size());
            } catch (const std::runtime_error&) { refused = true; }
            require(refused, "malformed views " + std::to_string(k) + " were taken");
        }
    // A layer outside the storage.
    const StateView view = {s.get(), 0, 0, 0, 4};
    bool refused = false;
    try { cpu.causal_conv_silu({u.get(), 0}, {x.get(), 0}, {w.get(), 0}, 2, &view, 1); } catch (const std::runtime_error&) { refused = true; }
    require(refused, "a layer outside the storage was taken");
    require(same_bits(download(cpu, *u, marker.size()), marker) && same_bits(download(cpu, *o, marker.size()), marker),
            "a refused call wrote its output");
    for (size_t l = 0; l < 2; ++l)
        for (size_t k = 0; k < 3; ++k) {
            const auto slot = read_slot(cpu, *s, l, k);
            require(std::all_of(slot.begin(), slot.end(), [](float f) { return f == 0.0f; }), "a refused call wrote a state");
        }
}

// ---------------------------------------------------------------------------------------------------------------------
// The gated norm, the partial rope and the output gate.

// dst = RMSNorm(x; w) * silu(z) per head against the math, a head near 1e-4 holding eps, at every thread count and with each row alone, bit for bit.
size_t check_gated_norm(std::mt19937& g, size_t heads, size_t dim) {
    const size_t rows = 23, n = rows * heads * dim;
    auto x = uniform(g, n, -2.0f, 2.0f);
    const auto z = uniform(g, n, -6.0f, 6.0f), w = uniform(g, dim, 0.5f, 1.5f);
    for (size_t i = 0; i < dim; ++i) x[(5 * heads + 1) * dim + i] *= 1e-4f;
    const float eps = 1e-6f;
    CpuBackend cpu;
    BufferPtr xb = upload(cpu, x), zb = upload(cpu, z), wb = upload(cpu, w), yb = zeros(cpu, n * sizeof(float));
    std::vector<float> first;
    for (int t : kThreadCounts) {
        cpu.set_threads(t);
        cpu.gated_rms_norm({yb.get(), 0}, {xb.get(), 0}, {zb.get(), 0}, {wb.get(), 0}, rows, heads, dim, eps);
        const auto y = download(cpu, *yb, n);
        if (first.empty()) first = y;
        require(same_bits(y, first), "gated norm: the thread count changed a result");
    }
    for (size_t r = 0; r < rows; ++r)
        for (size_t h = 0; h < heads; ++h) {
            const size_t o = (r * heads + h) * dim;
            double ss = 0;
            for (size_t i = 0; i < dim; ++i) ss += (double)x[o + i] * x[o + i];
            const double inv = 1.0 / std::sqrt(ss / dim + eps);
            // The sum of dim squares, the root and three products in F32: (dim + 8) units of the result.
            for (size_t i = 0; i < dim; ++i) {
                const double ref = x[o + i] * inv * w[i] * silu(z[o + i]);
                check_close(first[o + i], ref, (dim + 8) * U * (std::fabs(ref) + 1e-6), worst.norm, "gated norm");
            }
        }
    // A row alone in its call computes what it computes beside the others, and in place over x.
    for (size_t r : {size_t(0), size_t(11), rows - 1}) {
        const size_t o = r * heads * dim;
        BufferPtr one = zeros(cpu, heads * dim * sizeof(float));
        cpu.gated_rms_norm({one.get(), 0}, {xb.get(), o}, {zb.get(), o}, {wb.get(), 0}, 1, heads, dim, eps);
        require(same_bits(download(cpu, *one, heads * dim), std::vector<float>(first.begin() + o, first.begin() + o + heads * dim)),
                "gated norm: a row alone differs from the row in a batch");
    }
    BufferPtr in_place = upload(cpu, x);
    cpu.gated_rms_norm({in_place.get(), 0}, {in_place.get(), 0}, {zb.get(), 0}, {wb.get(), 0}, rows, heads, dim, eps);
    require(same_bits(download(cpu, *in_place, n), first), "gated norm: in place differs");
    return rows * heads;
}

// norm_rope_partial reading q's heads between their gates, as attn_q holds them, and k in place, against the math with every pair rotated at the token's position, which the rope sections give for text (docs/QWEN35.md, Gated attention).
// A head near 1e-4 holds eps; rows alone and every thread count give the same bits; and with a full rotary width over contiguous heads it is norm_rope_rows.
size_t check_partial_rope(std::mt19937& g, size_t heads, size_t head_dim, size_t rope_dim, double base) {
    const size_t rows = 19, half = rope_dim / 2, positions = 64;
    std::vector<float> cos(positions * half), sin(positions * half);
    for (size_t p = 0; p < positions; ++p)
        for (size_t i = 0; i < half; ++i) {
            const double t = (double)p * std::pow(base, -2.0 * (double)i / (double)rope_dim);
            cos[p * half + i] = (float)std::cos(t);
            sin[p * half + i] = (float)std::sin(t);
        }
    std::vector<uint32_t> pos(rows);
    for (size_t r = 0; r < rows; ++r) pos[r] = (uint32_t)((r * 37 + 5) % positions);
    auto r_rows = uniform(g, rows * heads * 2 * head_dim, -2.0f, 2.0f);
    const auto w = uniform(g, head_dim, 0.5f, 1.5f);
    for (size_t i = 0; i < head_dim; ++i) r_rows[4 * heads * 2 * head_dim + 2 * head_dim + i] *= 1e-4f;
    const float eps = 1e-6f;
    CpuBackend cpu;
    BufferPtr src = upload(cpu, r_rows), wb = upload(cpu, w), cb = upload(cpu, cos), sb = upload(cpu, sin);
    BufferPtr dst = zeros(cpu, rows * heads * head_dim * sizeof(float));
    std::vector<float> first;
    for (int t : kThreadCounts) {
        cpu.set_threads(t);
        cpu.norm_rope_partial({dst.get(), 0}, {src.get(), 0}, rows, heads * 2 * head_dim, 2 * head_dim, heads, head_dim, rope_dim,
                              {wb.get(), 0}, eps, {cb.get(), 0}, {sb.get(), 0}, pos.data());
        const auto y = download(cpu, *dst, rows * heads * head_dim);
        if (first.empty()) first = y;
        require(same_bits(y, first), "partial rope: the thread count changed a result");
    }
    for (size_t r = 0; r < rows; ++r)
        for (size_t h = 0; h < heads; ++h) {
            const float* x = r_rows.data() + r * heads * 2 * head_dim + h * 2 * head_dim;
            double ss = 0;
            for (size_t i = 0; i < head_dim; ++i) ss += (double)x[i] * x[i];
            const double inv = 1.0 / std::sqrt(ss / head_dim + eps);
            std::vector<double> n(head_dim), ref(head_dim);
            for (size_t i = 0; i < head_dim; ++i) n[i] = ref[i] = x[i] * inv * w[i];
            for (size_t i = 0; i < half; ++i) {
                const double t = (double)pos[r] * std::pow(base, -2.0 * (double)i / (double)rope_dim);
                ref[i] = n[i] * std::cos(t) - n[i + half] * std::sin(t);
                ref[i + half] = n[i] * std::sin(t) + n[i + half] * std::cos(t);
            }
            // The norm as in the gated norm, then a rotation through F32 tables: (head_dim + 16) units of the head's largest value.
            double top = 0;
            for (double v : ref) top = std::max(top, std::fabs(v));
            for (size_t i = 0; i < head_dim; ++i)
                check_close(first[(r * heads + h) * head_dim + i], ref[i], (head_dim + 16) * U * top, worst.rope, "partial rope");
        }
    // A row alone, and k's contiguous heads in place.
    cpu.set_threads(3);
    for (size_t r : {size_t(0), size_t(7), rows - 1}) {
        BufferPtr one = zeros(cpu, heads * head_dim * sizeof(float));
        cpu.norm_rope_partial({one.get(), 0}, {src.get(), r * heads * 2 * head_dim}, 1, heads * 2 * head_dim, 2 * head_dim, heads,
                              head_dim, rope_dim, {wb.get(), 0}, eps, {cb.get(), 0}, {sb.get(), 0}, pos.data() + r);
        const size_t o = r * heads * head_dim;
        require(same_bits(download(cpu, *one, heads * head_dim), std::vector<float>(first.begin() + o, first.begin() + o + heads * head_dim)),
                "partial rope: a row alone differs from the row in a batch");
    }
    std::vector<float> k(rows * heads * head_dim);
    for (size_t r = 0; r < rows; ++r)
        for (size_t h = 0; h < heads; ++h)
            std::memcpy(&k[(r * heads + h) * head_dim], &r_rows[r * heads * 2 * head_dim + h * 2 * head_dim], head_dim * sizeof(float));
    BufferPtr kb = upload(cpu, k);
    cpu.norm_rope_partial({kb.get(), 0}, {kb.get(), 0}, rows, heads * head_dim, head_dim, heads, head_dim, rope_dim, {wb.get(), 0}, eps,
                          {cb.get(), 0}, {sb.get(), 0}, pos.data());
    require(same_bits(download(cpu, *kb, k.size()), first), "partial rope: k in place differs from q read between its gates");
    bool refused = false;
    try {
        cpu.norm_rope_partial({src.get(), 0}, {src.get(), 0}, rows, heads * 2 * head_dim, 2 * head_dim, heads, head_dim, rope_dim,
                              {wb.get(), 0}, eps, {cb.get(), 0}, {sb.get(), 0}, pos.data());
    } catch (const std::runtime_error&) { refused = true; }
    require(refused, "partial rope in place over heads that are not contiguous was taken");
    // The full rotary width over contiguous heads is norm_rope_rows, bit for bit.
    std::vector<float> fc(positions * head_dim / 2), fs(positions * head_dim / 2);
    for (size_t p = 0; p < positions; ++p)
        for (size_t i = 0; i < head_dim / 2; ++i) {
            const double t = (double)p * std::pow(base, -2.0 * (double)i / (double)head_dim);
            fc[p * head_dim / 2 + i] = (float)std::cos(t);
            fs[p * head_dim / 2 + i] = (float)std::sin(t);
        }
    BufferPtr fcb = upload(cpu, fc), fsb = upload(cpu, fs), full = upload(cpu, k), rows_op = upload(cpu, k);
    cpu.norm_rope_partial({full.get(), 0}, {full.get(), 0}, rows, heads * head_dim, head_dim, heads, head_dim, head_dim, {wb.get(), 0}, eps,
                          {fcb.get(), 0}, {fsb.get(), 0}, pos.data());
    cpu.norm_rope_rows({rows_op.get(), 0}, rows, heads * head_dim, heads, {wb.get(), 0}, eps, {fcb.get(), 0}, {fsb.get(), 0}, head_dim / 2,
                       pos.data());
    require(same_bits(download(cpu, *full, k.size()), download(cpu, *rows_op, k.size())), "partial rope at the full width differs from norm_rope_rows");
    return rows * heads;
}

// A pair rotated in the scalar tail gives the bits of the same pair rotated in the vector body, whose multiply-adds are explicit, so no compiler's contraction changes it (docs/QWEN35.md, Row classes).
// Rotary width 24 leaves pairs 8 to 11 to the tail; each repeats pair i - 8's values and table entries, and a weight of 1 keeps the norm's scale the same in both.
void check_rope_tail(std::mt19937& g) {
    const size_t head_dim = 24, half = 12, rows = 3, positions = 4;
    auto x = uniform(g, rows * head_dim, -2.0f, 2.0f), cos = uniform(g, positions * half, -1.0f, 1.0f), sin = uniform(g, positions * half, -1.0f, 1.0f);
    for (size_t i = 0; i < 4; ++i) {
        for (size_t r = 0; r < rows; ++r) {
            x[r * head_dim + 8 + i] = x[r * head_dim + i];
            x[r * head_dim + half + 8 + i] = x[r * head_dim + half + i];
        }
        for (size_t p = 0; p < positions; ++p) {
            cos[p * half + 8 + i] = cos[p * half + i];
            sin[p * half + 8 + i] = sin[p * half + i];
        }
    }
    const std::vector<float> w(head_dim, 1.0f);
    const std::vector<uint32_t> pos = {1, 3, 2};
    CpuBackend cpu;
    BufferPtr xb = upload(cpu, x), wb = upload(cpu, w), cb = upload(cpu, cos), sb = upload(cpu, sin);
    cpu.norm_rope_partial({xb.get(), 0}, {xb.get(), 0}, rows, head_dim, head_dim, 1, head_dim, head_dim, {wb.get(), 0}, 1e-6f, {cb.get(), 0},
                          {sb.get(), 0}, pos.data());
    const auto y = download(cpu, *xb, x.size());
    for (size_t r = 0; r < rows; ++r)
        for (size_t i = 0; i < 4; ++i)
            require(same_bits(y[r * head_dim + 8 + i], y[r * head_dim + i]) && same_bits(y[r * head_dim + half + 8 + i], y[r * head_dim + half + i]),
                    "rope: a pair in the scalar tail rotates otherwise than in the vector body");
}

// An RMSNorm element in the scalar tail gives the bits of the same element in the vector body: width 12 leaves elements 8 to 11 to the tail, each repeating element i - 8's value and weight, and the row's scale is one for the whole row.
void check_norm_tail(std::mt19937& g) {
    const size_t n = 12, rows = 16;
    auto x = uniform(g, rows * n, -2.0f, 2.0f), w = uniform(g, n, 0.5f, 1.5f);
    for (size_t i = 0; i < 4; ++i) {
        w[8 + i] = w[i];
        for (size_t r = 0; r < rows; ++r) x[r * n + 8 + i] = x[r * n + i];
    }
    CpuBackend cpu;
    BufferPtr xb = upload(cpu, x), wb = upload(cpu, w), yb = upload(cpu, std::vector<float>(x.size()));
    cpu.rms_norm_rows({yb.get(), 0}, {xb.get(), 0}, {wb.get(), 0}, rows, n, n, 1e-6f);
    const auto y = download(cpu, *yb, x.size());
    for (size_t r = 0; r < rows; ++r)
        for (size_t i = 0; i < 4; ++i)
            require(same_bits(y[r * n + 8 + i], y[r * n + i]), "rms_norm: an element in the scalar tail is scaled otherwise than in the vector body");
}

// sigmoid_mul as the output gate, each query head's gate read in place from attn_q's rows, and as a scale of one value per row, against the math; rows alone and thread counts bit for bit.
size_t check_sigmoid_mul(std::mt19937& g, size_t heads, size_t dim) {
    const size_t rows = 21;
    const auto q = uniform(g, rows * heads * 2 * dim, -8.0f, 8.0f), x = uniform(g, rows * heads * dim, -3.0f, 3.0f);
    const auto per_row = uniform(g, rows, -8.0f, 8.0f);
    CpuBackend cpu;
    BufferPtr qb = upload(cpu, q), xb = upload(cpu, x), pb = upload(cpu, per_row), yb = zeros(cpu, x.size() * sizeof(float));
    for (int mode = 0; mode < 2; ++mode) {
        // The output gate: head h's gate is attn_q's row at 2 dim h + dim; the per-row scale: heads = the width, dim = 1, head stride 0.
        const size_t h_ = mode ? heads * dim : heads, d_ = mode ? 1 : dim, gs = mode ? 1 : heads * 2 * dim, ghs = mode ? 0 : 2 * dim;
        const backend::CSlice gate = mode ? backend::CSlice{pb.get(), 0} : backend::CSlice{qb.get(), dim};
        auto gate_at = [&](size_t r, size_t h, size_t d) { return mode ? per_row[r] : q[r * gs + h * ghs + dim + d]; };
        std::vector<float> first;
        for (int t : kThreadCounts) {
            cpu.set_threads(t);
            cpu.sigmoid_mul({yb.get(), 0}, {xb.get(), 0}, gate, rows, h_, d_, gs, ghs);
            const auto y = download(cpu, *yb, x.size());
            if (first.empty()) first = y;
            require(same_bits(y, first), "sigmoid_mul: the thread count changed a result");
        }
        for (size_t r = 0; r < rows; ++r)
            for (size_t h = 0; h < h_; ++h)
                for (size_t d = 0; d < d_; ++d) {
                    const size_t i = r * h_ * d_ + h * d_ + d;
                    const double ref = x[i] * sigmoid(gate_at(r, h, d));
                    // An exponential, a sum, a quotient and a product in F32: 8 units of the result.
                    check_close(first[i], ref, 8 * U * std::fabs(ref) + 1e-30, worst.gate, mode ? "per-row scale" : "output gate");
                }
        for (size_t r : {size_t(0), size_t(9), rows - 1}) {
            BufferPtr one = zeros(cpu, heads * dim * sizeof(float));
            const backend::CSlice g1 = mode ? backend::CSlice{pb.get(), r} : backend::CSlice{qb.get(), r * gs + dim};
            cpu.sigmoid_mul({one.get(), 0}, {xb.get(), r * heads * dim}, g1, 1, h_, d_, gs, ghs);
            const size_t o = r * heads * dim;
            require(same_bits(download(cpu, *one, heads * dim), std::vector<float>(first.begin() + o, first.begin() + o + heads * dim)),
                    "sigmoid_mul: a row alone differs from the row in a batch");
        }
        BufferPtr in_place = upload(cpu, x);
        cpu.sigmoid_mul({in_place.get(), 0}, {in_place.get(), 0}, gate, rows, h_, d_, gs, ghs);
        require(same_bits(download(cpu, *in_place, x.size()), first), "sigmoid_mul: in place differs");
    }
    return 2 * rows;
}

// A backend without the ops, as the Vulkan backend is until it implements them, refuses each by its name; Backend's own forms are what such a backend runs.
void check_refusals() {
    CpuBackend cpu;
    auto s = cpu.state_alloc(1, 1, StateShape{1, 1, 4, 4});
    BufferPtr buf = zeros(cpu, 4096);
    const backend::Slice o = {buf.get(), 0};
    const StateView view = {s.get(), 0, 0, 0, 1};
    const uint32_t pos = 0;
    auto refused = [](const char* op, auto&& call) {
        try {
            call();
        } catch (const std::runtime_error& e) {
            require(std::string(e.what()).find(op) != std::string::npos, std::string("a refusal that does not name ") + op + ": " + e.what());
            return;
        }
        throw std::runtime_error(std::string("Backend's own ") + op + " ran");
    };
    backend::Backend& base = cpu;
    refused("causal_conv_silu", [&] { base.Backend::causal_conv_silu(o, o, o, 0, &view, 1); });
    refused("gated_delta_rule", [&] { base.Backend::gated_delta_rule(o, o, o, o, o, o, 0, &view, 1); });
    refused("gated_rms_norm", [&] { base.Backend::gated_rms_norm(o, o, o, o, 1, 1, 4, 1e-6f); });
    refused("norm_rope_partial", [&] { base.Backend::norm_rope_partial(o, o, 1, 8, 8, 1, 8, 4, o, 1e-6f, o, o, &pos); });
    refused("sigmoid_mul", [&] { base.Backend::sigmoid_mul(o, o, o, 1, 1, 4, 4, 0); });
}
}  // namespace

int main() {
    try {
        std::mt19937 g(35);
        // Hv = Hk and Hv = 3 Hk at the tiny fixtures' widths, which leave the 8- and 1-column tails; one wide enough to split the conv over threads and take the 32-column blocks; and the files' 128 by 128.
        const std::vector<StateShape> shapes = {{2, 2, 12, 10}, {2, 6, 12, 10}, {2, 4, 64, 40}, {2, 4, 128, 128}};
        size_t mixed = 0, grouped = 0, steps = 0;
        for (const StateShape& sh : shapes) {
            mixed += check_linear_mix(g, sh);
            grouped += check_linear_invariance(g, sh);
            steps += check_one_token(g, sh);
        }
        check_decay_flush(g);
        check_column_paths(g);
        check_l2_eps(g);
        check_state_storage(g);
        check_view_refusals(g);
        std::printf("linear attention: %zu rows of mixed sequences against the references, %zu runs of thread counts and groupings bit for bit, %zu one-token steps; decay flush, column blocks, L2 epsilon, state storage and view refusals\n",
                    mixed, grouped, steps);
        size_t norm = 0, rope = 0, gate = 0;
        norm += check_gated_norm(g, 2, 10);
        norm += check_gated_norm(g, 4, 128);
        rope += check_partial_rope(g, 4, 40, 8, 100.0);
        rope += check_partial_rope(g, 3, 256, 64, 1e7);
        check_rope_tail(g);
        check_norm_tail(g);
        gate += check_sigmoid_mul(g, 4, 40);
        gate += check_sigmoid_mul(g, 3, 256);
        check_refusals();
        std::printf("gated attention: %zu gated-norm heads, %zu partial-rope heads, %zu gated rows; rope and norm tails, refusals name each op\n", norm, rope, gate);
        std::printf("worst error as a fraction of its bound: conv %.3f, delta rule %.3f, gated norm %.3f, partial rope %.3f, sigmoid_mul %.3f\n",
                    worst.conv, worst.delta, worst.norm, worst.rope, worst.gate);
        return 0;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
