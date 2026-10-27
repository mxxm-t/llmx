#pragma once
// The routed step (Backend::moe_step) in tests: a CPU backend counting the steps a model takes, and a step's weights, inputs and outputs, run through a backend's op or its base sequence.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "backends/backend.hpp"
#include "backends/cpu/cpu_backend.hpp"
#include "quant/quant.hpp"
#include "row_classes.hpp"

namespace moe {

// A CPU backend counting routed steps and the routed ops a model calls outside one, which every routed layer leaves to its step.
struct CountingCpu : backend::CpuBackend {
    size_t steps = 0, outside = 0;
    bool inside = false;
    CountingCpu() { set_threads(1); }
    void moe_step(const backend::MoeStep& m, backend::CSlice h, backend::Slice y, size_t nin, size_t rows, backend::RowRuns runs = {},
                  backend::Dtype dtype = backend::Dtype::f16) override {
        ++steps;
        inside = true;
        try {
            backend::CpuBackend::moe_step(m, h, y, nin, rows, runs, dtype);
        } catch (...) {
            inside = false;
            throw;
        }
        inside = false;
    }
    void route_experts(backend::CSlice scores, size_t rows, size_t n_expert, size_t k, bool normalize, backend::Slice ids, backend::Slice weights) override {
        outside += inside ? 0 : 1;
        backend::CpuBackend::route_experts(scores, rows, n_expert, k, normalize, ids, weights);
    }
    void matmul_experts(std::initializer_list<backend::Projection> projections, backend::CSlice x, size_t nin, size_t rows, const Routing& routing,
                        backend::RowRuns runs = {}, backend::Dtype dtype = backend::Dtype::f16) override {
        outside += inside ? 0 : 1;
        backend::CpuBackend::matmul_experts(projections, x, nin, rows, routing, runs, dtype);
    }
    void matmul_experts_add(backend::WeightView weight, backend::CSlice x, backend::Slice y, size_t nin, size_t nout, size_t rows,
                            const Routing& routing, backend::RowRuns runs = {}, backend::Dtype dtype = backend::Dtype::f16) override {
        outside += inside ? 0 : 1;
        backend::CpuBackend::matmul_experts_add(weight, x, y, nin, nout, rows, routing, runs, dtype);
    }
};

// A routed layer's types: the experts' gate and up, their down, and the shared expert's gate and up and down, sh_gate 0 for a layer without one.
struct Types { uint32_t gate, down, sh_gate, sh_down; };
// The routed layers of Qwen3.8-Flash-Next UD-Q4_K_XL, Qwen3.6-35B-A3B Q4_K_M and Qwen3-30B-A3B Q8_0, the combinations a device may fuse.
const Types kFlash{quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q8_0};
const Types k35b{quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q6_K};
const Types k30b{quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q8_0, 0, 0};

// A step's widths: the residual E, experts, picks a token, an expert's hidden rows (the shared expert's too), and whether y starts from a zero row.
struct Shape { size_t E, n_expert, k, ff; Types t; bool zero; };

using Matrix = std::function<std::vector<uint8_t>(uint32_t type, size_t nin, size_t rows, uint32_t seed)>;

// A copy of host data in the backend's own storage, since the CPU adopts memory in place.
inline backend::BufferPtr copy_in(backend::Backend& b, const void* data, size_t bytes) {
    backend::BufferPtr buf = b.alloc(bytes, backend::Memory::device);
    b.write(*buf, 0, data, bytes);
    return buf;
}

struct Weights {
    Shape s;
    backend::BufferPtr router, gate, up, down, sh_gate, sh_up, sh_down, row_gate, zero;
};

inline Weights weights(backend::Backend& b, const Shape& s, const Matrix& matrix, uint32_t seed) {
    auto put = [&](uint32_t type, size_t nin, size_t rows, uint32_t salt) {
        const std::vector<uint8_t> w = matrix(type, nin, rows, seed * 16 + salt);
        return copy_in(b, w.data(), w.size());
    };
    Weights w{s, put(quant::GGML_TYPE_F32, s.E, s.n_expert, 1), put(s.t.gate, s.E, s.n_expert * s.ff, 2), put(s.t.gate, s.E, s.n_expert * s.ff, 3),
              put(s.t.down, s.ff, s.n_expert * s.E, 4), nullptr, nullptr, nullptr, nullptr, nullptr};
    if (s.t.sh_gate) {
        w.sh_gate = put(s.t.sh_gate, s.E, s.ff, 5);
        w.sh_up = put(s.t.sh_gate, s.E, s.ff, 6);
        w.sh_down = put(s.t.sh_down, s.ff, s.E, 7);
        w.row_gate = put(quant::GGML_TYPE_F32, s.E, 1, 8);
    }
    const std::vector<float> zeros(s.E, 0.0f);
    w.zero = copy_in(b, zeros.data(), zeros.size() * sizeof(float));
    return w;
}

// Rows of h, row i by i mod 6: random, zero, scaled by 1000 so later picks' weights underflow to 0, by 1e-30 and by 1e-20, whose products are subnormal, and one NaN value, which routes nowhere.
inline std::vector<float> inputs(size_t rows, size_t E, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> h(rows * E);
    for (size_t r = 0; r < rows; ++r)
        for (size_t i = 0; i < E; ++i) {
            const float v = d(rng);
            const size_t kind = r % 6;
            h[r * E + i] = kind == 1 ? 0.0f : kind == 2 ? v * 1000.0f : kind == 3 ? v * 1e-30f : kind == 4 ? v * 1e-20f : v;
        }
    for (size_t r = 5; r < rows; r += 6) h[r * E] = std::numeric_limits<float>::quiet_NaN();
    return h;
}

// The residual a step joins, with some values -0 so a sum's sign of zero shows.
inline std::vector<float> residual(size_t rows, size_t E, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> y(rows * E);
    for (size_t i = 0; i < y.size(); ++i) y[i] = i % 7 == 3 ? -0.0f : d(rng);
    return y;
}

struct Out {
    std::vector<float> y, scores, weights, g, u, act, row_gate;
    std::vector<uint32_t> ids;
};

// How a step runs: the backend's op, the base op's sequence, or the calls the routed layers made before the op, written out.
enum class How { op, base, calls };

// One step over `rows` rows of h, every output buffer starting as the same NaN pattern, so rows left unwritten compare too.
inline Out run(backend::Backend& b, const Weights& w, const std::vector<float>& h, const std::vector<float>& y0, size_t rows,
               const std::vector<backend::RowRun>& runs, backend::Dtype dtype, How how) {
    const Shape& s = w.s;
    const bool shared = s.t.sh_gate != 0;
    const size_t slot = rows * (s.k * s.ff + (shared ? s.ff : 0));
    auto poisoned = [&](size_t n) {
        const std::vector<uint32_t> bits(n, 0x7fc0beefu);
        return copy_in(b, bits.data(), n * sizeof(float));
    };
    const backend::BufferPtr hb = copy_in(b, h.data(), rows * s.E * sizeof(float)), yb = copy_in(b, y0.data(), rows * s.E * sizeof(float));
    const backend::BufferPtr scores = poisoned(rows * s.n_expert), ids = poisoned(rows * s.k), wts = poisoned(rows * s.k), sg = poisoned(rows);
    const backend::BufferPtr g = poisoned(slot), u = poisoned(slot), act = poisoned(slot);
    const size_t routed = rows * s.k * s.ff;
    std::vector<backend::RowRun> scratch;
    const backend::MoeShared sh{{{s.t.sh_gate, {w.sh_gate.get(), 0}}, {g.get(), routed}, s.ff},
                                {{s.t.sh_gate, {w.sh_up.get(), 0}}, {u.get(), routed}, s.ff},
                                {s.t.sh_down, {w.sh_down.get(), 0}}, {act.get(), routed},
                                {{quant::GGML_TYPE_F32, {w.row_gate.get(), 0}}, {sg.get(), 0}, 1}};
    const backend::MoeStep m{{{quant::GGML_TYPE_F32, {w.router.get(), 0}}, {scores.get(), 0}, s.n_expert}, s.k, true,
                             {ids.get(), 0}, {wts.get(), 0}, {{s.t.gate, {w.gate.get(), 0}}, {g.get(), 0}, s.ff},
                             {{s.t.gate, {w.up.get(), 0}}, {u.get(), 0}, s.ff}, {s.t.down, {w.down.get(), 0}},
                             {act.get(), 0}, shared ? &sh : nullptr, s.zero ? w.zero.get() : nullptr, &scratch};
    const backend::RowRuns rr{runs.data(), runs.size()};
    if (how == How::base) b.Backend::moe_step(m, {hb.get(), 0}, {yb.get(), 0}, s.E, rows, rr, dtype);
    else if (how == How::op) b.moe_step(m, {hb.get(), 0}, {yb.get(), 0}, s.E, rows, rr, dtype);
    else {
        // Today's calls of the three architectures written out, as they ran before the op existed: the residual cleared, the router in F32, the routing, the experts' SwiGLU and weighted down, then the shared expert's gate, SwiGLU, scale and down.
        const backend::CSlice h0{hb.get(), 0};
        const backend::Slice y{yb.get(), 0};
        if (m.zero) {
            b.copy(*yb, 0, *m.zero, 0, s.E * sizeof(float));
            for (size_t done = 1; done < rows;) {
                const size_t count = std::min(done, rows - done);
                b.copy(*yb, done * s.E * sizeof(float), *yb, 0, count * s.E * sizeof(float));
                done += count;
            }
        }
        b.matmul(m.router.weight, h0, m.router.out, s.E, s.n_expert, rows, rr, backend::Dtype::f32);
        b.route_experts(m.router.out, rows, s.n_expert, s.k, true, m.ids, m.weights);
        const backend::Backend::Routing routing{m.ids, m.weights, s.k, s.n_expert};
        b.matmul_experts({m.gate, m.up}, h0, s.E, rows, routing, rr, dtype);
        std::vector<backend::RowRun> entry_runs;
        for (const backend::RowRun& r : runs) entry_runs.push_back({r.end * s.k, r.extent});
        b.silu_mul(m.act, m.gate.out, m.up.out, routed, {entry_runs.data(), entry_runs.size()});
        b.matmul_experts_add(m.down, m.act, y, s.ff, s.E, rows, routing, rr, dtype);
        if (shared) {
            b.matmul(sh.row_gate.weight, h0, sh.row_gate.out, s.E, 1, rows, rr, dtype);
            b.matmul_group({sh.gate, sh.up}, h0, s.E, rows, rr, dtype);
            b.silu_mul(sh.act, sh.gate.out, sh.up.out, rows * s.ff, rr);
            b.sigmoid_mul(sh.act, sh.act, sh.row_gate.out, rows, s.ff, 1, 1, 0, rr);
            b.matmul_add(sh.down, sh.act, y, s.ff, s.E, rows, rr, dtype);
        }
    }
    Out o;
    auto read = [&](const backend::BufferPtr& buf, size_t n) {
        std::vector<float> v(n);
        b.read(*buf, 0, v.data(), n * sizeof(float));
        return v;
    };
    o.y = read(yb, rows * s.E);
    o.scores = read(scores, rows * s.n_expert);
    o.weights = read(wts, rows * s.k);
    o.g = read(g, slot);
    o.u = read(u, slot);
    o.act = read(act, slot);
    o.row_gate = read(sg, rows);
    o.ids.resize(rows * s.k);
    b.read(*ids, 0, o.ids.data(), o.ids.size() * sizeof(uint32_t));
    return o;
}

// The decode runs of `rows` generated tokens, one run of extent 1 each.
inline std::vector<backend::RowRun> decode_runs(size_t rows) {
    std::vector<backend::RowRun> runs;
    for (size_t i = 0; i < rows; ++i) runs.push_back({i + 1, 1});
    return runs;
}

// Whether two steps' outputs are the same bits, naming the first that differs in `what`; returns the values compared.
inline size_t same(const Out& a, const Out& b, std::string& what) {
    auto eq = [&](const void* x, const void* y, size_t bytes, const char* name) {
        if (std::memcmp(x, y, bytes) == 0) return true;
        what = name;
        return false;
    };
    if (!eq(a.y.data(), b.y.data(), a.y.size() * 4, "y") || !eq(a.scores.data(), b.scores.data(), a.scores.size() * 4, "scores") ||
        !eq(a.ids.data(), b.ids.data(), a.ids.size() * 4, "ids") || !eq(a.weights.data(), b.weights.data(), a.weights.size() * 4, "weights") ||
        !eq(a.g.data(), b.g.data(), a.g.size() * 4, "gate rows") || !eq(a.u.data(), b.u.data(), a.u.size() * 4, "up rows") ||
        !eq(a.act.data(), b.act.data(), a.act.size() * 4, "activations") || !eq(a.row_gate.data(), b.row_gate.data(), a.row_gate.size() * 4, "row gates"))
        return 0;
    return a.y.size() + a.scores.size() + a.ids.size() + a.weights.size() + a.g.size() + a.u.size() + a.act.size() + a.row_gate.size();
}

// Every pair of extents one class holds gives the same bits through the routed step, four rows each, every type combination, f16 and int8 where the backend runs it: a generated token's step is fused where the device fuses it.
inline size_t check_row_classes(backend::Backend& b, const Matrix& matrix) {
    const Shape shapes[] = {{256, 64, 2, 64, kFlash, true}, {256, 64, 2, 256, k35b, false}, {256, 64, 2, 64, k30b, false}};
    std::vector<backend::Dtype> dtypes = {backend::Dtype::f16};
    const std::vector<backend::Dtype> native = b.native_dtypes();
    if (std::find(native.begin(), native.end(), backend::Dtype::int8) != native.end()) dtypes.push_back(backend::Dtype::int8);
    const size_t rows = 4;
    size_t pairs = 0;
    for (const Shape& s : shapes) {
        const Weights w = weights(b, s, matrix, 61);
        const std::vector<float> h = inputs(rows, s.E, 62), y0 = residual(rows, s.E, 63);
        for (backend::Dtype dtype : dtypes) {
            std::vector<std::vector<float>> out;
            for (size_t e : row_classes::kExtents) {
                const std::vector<backend::RowRun> runs = {{rows, e}};
                const Out o = run(b, w, h, y0, rows, runs, dtype, How::op);
                std::vector<float> v = o.y;
                v.insert(v.end(), o.weights.begin(), o.weights.end());
                v.insert(v.end(), o.act.begin(), o.act.end());
                out.push_back(v);
            }
            pairs += row_classes::same_within_classes(b, out, std::string(backend::dtype_name(dtype)) + " a routed step");
        }
    }
    return pairs;
}

} // namespace moe
