// The CPU decode dots over quantized activations (backends/cpu/q8_dots.hpp), 8-bit or 16-bit by type, against a double-precision reference fed the same quantized activations, for every type they take.
// Also that a decode row computes the same alone and beside others, and grouped projections the same as separate ones, bit for bit.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "backends/cpu/cpu_backend.hpp"
#include "quant/quant.hpp"

namespace {
void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

// Packed rows of a type: the block quantizers from floats, the K-quants from a byte pattern with small half scales.
std::vector<uint8_t> packed(uint32_t type, size_t rows, size_t nin, std::mt19937& rng) {
    const quant::QuantType* qt = quant::Registry::instance().get(type);
    const size_t blocks = rows * nin / qt->block_size;
    std::vector<uint8_t> out(blocks * qt->type_size);
    if (qt->quantize) {
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::vector<float> f(rows * nin);
        for (float& v : f) v = u(rng);
        qt->quantize(f.data(), out.data(), blocks);
        return out;
    }
    for (uint8_t& b : out) b = uint8_t(rng());
    for (size_t b = 0; b < blocks; ++b) {
        uint8_t* p = out.data() + b * qt->type_size;
        if (type == gguf::GGML_TYPE_Q6_K) { p[208] = 0x00; p[209] = 0x14; }
        else { p[0] = 0x00; p[1] = 0x14; p[2] = 0x00; p[3] = 0x10; }
    }
    return out;
}

// The activations as the dots round them: per block of 32 the largest magnitude over `levels` (127 or 32767), and each value to the nearest step, ties to even.
void rounded(const float* x, size_t n, float levels, std::vector<double>& q, std::vector<double>& d) {
    q.assign(n, 0.0);
    d.assign(n / 32, 0.0);
    for (size_t b = 0; b < n / 32; ++b) {
        float top = 0.0f;
        for (size_t i = 0; i < 32; ++i) top = std::max(top, std::fabs(x[b * 32 + i]));
        const float step = top / levels, inv = top > 0.0f ? levels / top : 0.0f;
        d[b] = step;
        for (size_t i = 0; i < 32; ++i) q[b * 32 + i] = std::nearbyint(x[b * 32 + i] * inv);
    }
}

size_t check_type(uint32_t type, size_t nin, std::mt19937& rng) {
    const quant::QuantType* qt = quant::Registry::instance().get(type);
    const size_t rows = 37, cols = 3;
    const auto w = packed(type, rows, nin, rng);
    std::vector<float> wf(rows * nin);
    qt->dequantize(w.data(), wf.data(), rows * nin / qt->block_size);
    // Ordinary values, and one with a large outlier per block, which is where a block's rounding is coarsest.
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> x(cols * nin);
    for (size_t i = 0; i < x.size(); ++i) x[i] = u(rng) * (i % 97 == 0 ? 40.0f : 1.0f);

    backend::CpuBackend cpu;
    cpu.set_threads(4);
    const auto wb = cpu.adopt(w.data(), w.size());
    const auto xb = cpu.adopt(x.data(), x.size() * sizeof(float));
    std::vector<float> alone(rows * cols), together(rows * cols), grouped(2 * rows * cols);
    for (size_t c = 0; c < cols; ++c) {
        const auto xc = cpu.adopt(x.data() + c * nin, nin * sizeof(float));
        const auto yc = cpu.adopt(alone.data() + c * rows, rows * sizeof(float));
        cpu.matmul(type, {wb.get(), 0}, {xc.get(), 0}, {yc.get(), 0}, nin, rows, 1);
    }
    // Three generated tokens in one call, as a server's decode pass has them.
    const backend::RowRun decode[3] = {{1, 1}, {2, 1}, {3, 1}};
    const auto yt = cpu.adopt(together.data(), together.size() * sizeof(float));
    cpu.matmul(type, {wb.get(), 0}, {xb.get(), 0}, {yt.get(), 0}, nin, rows, cols, {decode, 3});
    require(std::memcmp(alone.data(), together.data(), alone.size() * sizeof(float)) == 0, "a decode row differs beside others");
    const auto yg = cpu.adopt(grouped.data(), grouped.size() * sizeof(float));
    cpu.matmul_group({{type, {wb.get(), 0}, {yg.get(), 0}, rows}, {type, {wb.get(), 0}, {yg.get(), rows * cols}, rows}},
                     {xb.get(), 0}, nin, cols, {decode, 3});
    for (size_t c = 0; c < cols; ++c)
        for (size_t o = 0; o < rows; ++o)
            require(grouped[c * rows + o] == together[c * rows + o] && grouped[rows * cols + c * rows + o] == together[c * rows + o],
                    "grouped projections differ from separate ones");

    // The same columns as a prompt's rows, through the prompt dots for the K-quants and the float path for the rest: alone, and beside a generated token, where the prompt's rows compute as alone and the token as a decode row.
    std::vector<float> prompt(rows * cols), mixed(rows * cols);
    const backend::RowRun as_prompt[1] = {{cols, 512}}, beside[2] = {{1, 1}, {cols, 512}};
    const auto yp = cpu.adopt(prompt.data(), prompt.size() * sizeof(float)), ym = cpu.adopt(mixed.data(), mixed.size() * sizeof(float));
    cpu.matmul(type, {wb.get(), 0}, {xb.get(), 0}, {yp.get(), 0}, nin, rows, cols, {as_prompt, 1});
    cpu.matmul(type, {wb.get(), 0}, {xb.get(), 0}, {ym.get(), 0}, nin, rows, cols, {beside, 2});
    require(std::memcmp(mixed.data(), alone.data(), rows * sizeof(float)) == 0, "a decode row differs beside a prompt");
    require(std::memcmp(mixed.data() + rows, prompt.data() + rows, (cols - 1) * rows * sizeof(float)) == 0,
            "a prompt's rows differ beside a generated token");

    std::vector<double> q, d;
    for (size_t c = 0; c < cols; ++c) {
        rounded(x.data() + c * nin, nin, backend::q8::reads16(type) ? 32767.0f : 127.0f, q, d);
        for (size_t o = 0; o < rows; ++o) {
            double ref = 0.0, mag = 0.0;
            for (size_t i = 0; i < nin; ++i) {
                const double t = (double)wf[o * nin + i] * q[i] * d[i / 32];
                ref += t;
                mag += std::fabs(t);
            }
            const bool kquant = type == gguf::GGML_TYPE_Q4_K || type == gguf::GGML_TYPE_Q5_K || type == gguf::GGML_TYPE_Q6_K;
            for (const float got : kquant ? std::vector<float>{alone[c * rows + o], prompt[c * rows + o]} : std::vector<float>{alone[c * rows + o]})
                require(std::isfinite(got) && std::fabs(got - ref) <= 1e-5 * mag + 1e-30,
                        "type " + std::to_string(type) + " nin " + std::to_string(nin) + ": " + std::to_string(got) +
                        " against " + std::to_string(ref));
        }
    }
    return rows * cols;
}
// A generated token's routed entries through the same dots: every entry equals a one-column matmul of its expert's matrix and its token's row, bit for bit, and the down projection their weighted sum.
size_t check_experts(uint32_t type, std::mt19937& rng) {
    const quant::QuantType* qt = quant::Registry::instance().get(type);
    const size_t n_expert = 3, k = 2, rows = 6, entries = rows * k, nin = 256, nout = 8;
    const size_t stride = nout * (nin / qt->block_size) * qt->type_size;
    const auto w = packed(type, n_expert * nout, nin, rng);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> x(rows * nin), x2(entries * nin), ids(entries), weights(entries, 0.5f);
    for (float& v : x) v = u(rng);
    for (float& v : x2) v = u(rng);
    for (size_t e = 0; e < entries; ++e) {
        const uint32_t id = (uint32_t)((e * 7 + e / k) % n_expert);
        std::memcpy(&ids[e], &id, sizeof(id));
    }
    backend::CpuBackend cpu;
    cpu.set_threads(4);
    const auto wb = cpu.adopt(w.data(), w.size());
    const auto xb = cpu.adopt(x.data(), x.size() * sizeof(float)), x2b = cpu.adopt(x2.data(), x2.size() * sizeof(float));
    const auto ib = cpu.adopt(ids.data(), ids.size() * sizeof(float)), gb = cpu.adopt(weights.data(), weights.size() * sizeof(float));
    const backend::Backend::Routing routing{{ib.get(), 0}, {gb.get(), 0}, k, n_expert};
    const backend::RowRun decode[6] = {{1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}, {6, 1}};
    const backend::RowRuns runs{decode, 6};
    std::vector<float> up(entries * nout), down(rows * nout, 0.0f);
    const auto ub = cpu.adopt(up.data(), up.size() * sizeof(float));
    cpu.matmul_experts({{type, {wb.get(), 0}, {ub.get(), 0}, nout}}, {xb.get(), 0}, nin, rows, routing, runs);
    for (size_t e = 0; e < entries; ++e) {
        uint32_t id;
        std::memcpy(&id, &ids[e], sizeof(id));
        std::vector<float> y(nout), yd(nout);
        const auto xe = cpu.adopt(x.data() + (e / k) * nin, nin * sizeof(float)), ye = cpu.adopt(y.data(), nout * sizeof(float));
        cpu.matmul(type, {wb.get(), id * stride / sizeof(float)}, {xe.get(), 0}, {ye.get(), 0}, nin, nout, 1);
        require(std::memcmp(y.data(), up.data() + e * nout, nout * sizeof(float)) == 0,
                "type " + std::to_string(type) + ": a routed entry differs from its expert's matmul");
        const auto xd = cpu.adopt(x2.data() + e * nin, nin * sizeof(float)), yb = cpu.adopt(yd.data(), nout * sizeof(float));
        cpu.matmul(type, {wb.get(), id * stride / sizeof(float)}, {xd.get(), 0}, {yb.get(), 0}, nin, nout, 1);
        for (size_t o = 0; o < nout; ++o) down[(e / k) * nout + o] += 0.5f * yd[o];
    }
    // The same rows as a prompt, whose entries go through their expert's rows several at a time: each prompt row computes the same beside generated tokens as in a call of its own, and each generated token as alone.
    const backend::RowRun prompt[1] = {{rows, 512}}, mixed[3] = {{2, 1}, {5, 512}, {6, 1}};
    std::vector<float> ups(entries * nout), upm(entries * nout);
    const auto usb = cpu.adopt(ups.data(), ups.size() * sizeof(float)), umb = cpu.adopt(upm.data(), upm.size() * sizeof(float));
    cpu.matmul_experts({{type, {wb.get(), 0}, {usb.get(), 0}, nout}}, {xb.get(), 0}, nin, rows, routing, {prompt, 1});
    cpu.matmul_experts({{type, {wb.get(), 0}, {umb.get(), 0}, nout}}, {xb.get(), 0}, nin, rows, routing, {mixed, 3});
    for (size_t e = 0; e < entries; ++e) {
        const size_t token = e / k;
        const std::vector<float>& want = token >= 2 && token < 5 ? ups : up;
        require(std::memcmp(want.data() + e * nout, upm.data() + e * nout, nout * sizeof(float)) == 0,
                "type " + std::to_string(type) + ": a routed entry differs beside other kinds of row");
    }
    // A prompt's products on the prompt dots stay within the quantized reference's bound of the decode dots'.
    for (size_t i = 0; i < entries * nout; ++i)
        require(std::fabs(ups[i] - up[i]) <= 1e-5f * (1.0f + std::fabs(up[i])) * 64.0f,
                "type " + std::to_string(type) + ": a prompt's routed entry strays from a generated token's");
    std::vector<float> y(rows * nout, 0.0f);
    const auto yb = cpu.adopt(y.data(), y.size() * sizeof(float));
    cpu.matmul_experts_add(type, {wb.get(), 0}, {x2b.get(), 0}, {yb.get(), 0}, nin, nout, rows, routing, runs);
    for (size_t i = 0; i < rows * nout; ++i)
        require(std::fabs(y[i] - down[i]) <= 1e-6f * (1.0f + std::fabs(down[i])),
                "type " + std::to_string(type) + ": a routed down projection differs from its experts' matmuls");
    return entries;
}
}  // namespace

int main() {
    try {
        quant::register_builtins();
        std::mt19937 rng(7);
        size_t n = 0;
        for (uint32_t type : {gguf::GGML_TYPE_Q8_0, gguf::GGML_TYPE_Q4_0, gguf::GGML_TYPE_Q4_1,
                              gguf::GGML_TYPE_Q4_K, gguf::GGML_TYPE_Q5_K, gguf::GGML_TYPE_Q6_K})
            for (size_t nin : {size_t(256), size_t(2048)}) n += check_type(type, nin, rng);
        size_t routed = 0;
        for (uint32_t type : {gguf::GGML_TYPE_Q8_0, gguf::GGML_TYPE_Q4_0, gguf::GGML_TYPE_Q4_1,
                              gguf::GGML_TYPE_Q4_K, gguf::GGML_TYPE_Q5_K, gguf::GGML_TYPE_Q6_K})
            routed += check_experts(type, rng);
        std::printf("q8 dots: %zu rows against the reference, alone, beside others and grouped; %zu routed entries against their experts\n",
                    n, routed);
        return 0;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
