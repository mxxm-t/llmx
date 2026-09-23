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
            const double got = alone[c * rows + o];
            require(std::isfinite(got) && std::fabs(got - ref) <= 1e-5 * mag + 1e-30,
                    "type " + std::to_string(type) + " nin " + std::to_string(nin) + ": " + std::to_string(got) +
                    " against " + std::to_string(ref));
        }
    }
    return rows * cols;
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
        std::printf("q8 dots: %zu rows against the reference, alone, beside others and grouped\n", n);
        return 0;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
