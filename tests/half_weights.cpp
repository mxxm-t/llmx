#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/cpu/cpu_backend.hpp"
#include "quant/quant.hpp"

namespace {

constexpr uint32_t kF16 = 1, kBF16 = 30;
constexpr float kGuard = 123456.0f;
size_t checked = 0;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

uint32_t bits(float value) {
    uint32_t out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

// The finite format value from its integer significand and power of two, without either production conversion helper.
float widened(uint32_t type, uint16_t raw) {
    const unsigned fraction_bits = type == kF16 ? 10 : 7, exponent_bits = type == kF16 ? 5 : 8;
    const unsigned mask = (1u << exponent_bits) - 1, bias = mask / 2;
    const unsigned exponent = (raw >> fraction_bits) & mask, fraction = raw & ((1u << fraction_bits) - 1);
    require(exponent != mask, "oracle called with a nonfinite encoding");
    const double magnitude = std::ldexp(double(fraction + (exponent ? 1u << fraction_bits : 0)),
                                        int(exponent ? exponent : 1) - int(bias) - int(fraction_bits));
    return std::copysign(float(magnitude), raw & 0x8000u ? -1.0f : 1.0f);
}

bool finite(uint32_t type, uint16_t raw) {
    const unsigned mask = type == kF16 ? 0x7c00u : 0x7f80u;
    return (raw & mask) != mask;
}

void put(uint8_t* dst, uint16_t raw) {
    dst[0] = uint8_t(raw);
    dst[1] = uint8_t(raw >> 8);
}

uint16_t weight_code(uint32_t type, size_t i) {
    const unsigned fraction_bits = type == kF16 ? 10 : 7, bias = type == kF16 ? 15 : 127;
    const unsigned fraction = unsigned(i * 73 + 19) & ((1u << fraction_bits) - 1);
    const unsigned exponent = i % 29 == 0 ? 0 : bias - 5 + unsigned(i % 10);
    return uint16_t((i % 3 == 0 ? 0x8000u : 0) | (exponent << fraction_bits) | (i % 31 == 0 ? 0 : fraction));
}

struct Matrix {
    std::vector<uint8_t> packed;
    std::vector<float> wide;
    backend::BufferPtr half, f32;

    Matrix(backend::CpuBackend& cpu, uint32_t t, size_t n, size_t m, size_t seed = 0)
        : packed(2 * n * m + 2, 0xa5), wide(n * m + 1, kGuard) {
        for (size_t i = 0; i < n * m; ++i) {
            const uint16_t raw = weight_code(t, i + seed);
            put(packed.data() + 1 + 2 * i, raw);
            wide[1 + i] = widened(t, raw);
        }
        half = cpu.adopt(packed.data() + 1, 2 * n * m);
        f32 = cpu.adopt(wide.data() + 1, n * m * sizeof(float));
    }
};

struct Output {
    std::vector<float> values;
    backend::BufferPtr buffer;

    Output(backend::CpuBackend& cpu, size_t n) : values(n + 2, kGuard), buffer(cpu.adopt(values.data(), values.size() * sizeof(float))) {}
    backend::Slice slice() { return {buffer.get(), 1}; }
    void reset() {
        for (size_t i = 1; i + 1 < values.size(); ++i) values[i] = float(int(i % 13) - 6) / 8.0f;
    }
};

void same(const Output& got, const Output& want, const std::string& at) {
    require(got.values.front() == kGuard && got.values.back() == kGuard && want.values.front() == kGuard && want.values.back() == kGuard,
            at + ": output guard changed");
    require(got.values.size() == want.values.size(), at + ": output size differs");
    for (size_t i = 1; i + 1 < got.values.size(); ++i) {
        if (bits(got.values[i]) != bits(want.values[i])) throw std::runtime_error(at + ": widened-F32 mismatch at " + std::to_string(i - 1));
        ++checked;
    }
}

std::vector<float> inputs(size_t n) {
    std::vector<float> out(n + 1, kGuard);
    for (size_t i = 1; i <= n; ++i) out[i] = float(int((i * 769 + 23) % 8191) - 4095) / 523.0f;
    return out;
}

// The decoder and embedding preserve every finite encoding's bits, including both zeros; width-one products also reach the complete range without overflow.
void all_finite(backend::CpuBackend& cpu, uint32_t type) {
    const quant::QuantType* q = quant::Registry::instance().get(type);
    require(q && q->dequantize && !q->quantize && q->block_size == 1 && q->type_size == 2, "half weights need a read-only scalar decoder");
    require(cpu.supports_type(type), "CPU does not advertise half-weight support");
    std::vector<uint16_t> codes;
    for (unsigned raw = 0; raw < 65536; ++raw)
        if (finite(type, uint16_t(raw))) codes.push_back(uint16_t(raw));
    const size_t n = codes.size();
    std::vector<uint8_t> packed(2 * n + 2, 0xa5);
    std::vector<float> expected(n), decoded(n + 2, kGuard);
    std::vector<uint32_t> ids(n);
    for (size_t i = 0; i < n; ++i) {
        put(packed.data() + 1 + 2 * i, codes[i]);
        expected[i] = widened(type, codes[i]);
        ids[i] = uint32_t(n - i - 1);
    }
    const auto original = packed;
    q->dequantize(packed.data() + 1, decoded.data() + 1, n);
    require(decoded.front() == kGuard && decoded.back() == kGuard, "half decoder crossed its output");
    for (size_t i = 0; i < n; ++i) {
        if (bits(decoded[i + 1]) != bits(expected[i])) throw std::runtime_error("half decoder differs at encoding " + std::to_string(codes[i]));
        ++checked;
    }
    q->dequantize(nullptr, nullptr, 0);
    const auto wb = cpu.adopt(packed.data() + 1, 2 * n), fb = cpu.adopt(expected.data(), n * sizeof(float));
    Output got(cpu, n), want(cpu, n);
    cpu.embed(got.slice(), type, {wb.get(), 0}, 1, n, ids.data(), n);
    cpu.embed(want.slice(), quant::GGML_TYPE_F32, {fb.get(), 0}, 1, n, ids.data(), n);
    same(got, want, "all finite embeddings");
    const float x[] = {1.0f, -1.0f, 0.0009765625f};
    const auto xb = cpu.adopt(x, sizeof(x));
    Output product(cpu, n * 3), reference(cpu, n * 3);
    cpu.matmul(type, {wb.get(), 0}, {xb.get(), 0}, product.slice(), 1, n, 3, {}, backend::Dtype::f32);
    cpu.matmul(quant::GGML_TYPE_F32, {fb.get(), 0}, {xb.get(), 0}, reference.slice(), 1, n, 3, {}, backend::Dtype::f32);
    same(product, reference, "all finite products");
    require(packed == original, "a decoder or product changed half-weight bytes");
}

// Odd widths and row counts cross the SIMD and four-row tails; one table serves both embedding and output-head calls.
void dense(backend::CpuBackend& cpu, uint32_t type, size_t width, size_t rows, size_t batch, backend::Dtype dtype) {
    Matrix w(cpu, type, width, rows);
    auto x = inputs(width * batch);
    const auto xb = cpu.adopt(x.data(), x.size() * sizeof(float));
    const auto original_x = x;
    Output got(cpu, rows * batch), want(cpu, rows * batch);
    const backend::RowRun decode[] = {{batch, 1}}, prompt[] = {{batch, std::max<size_t>(2, batch)}}, mixed[] = {{1, 1}, {batch, batch}};
    const std::string at = "dense type " + std::to_string(type) + " width " + std::to_string(width) + " batch " + std::to_string(batch);
    for (const backend::RowRuns runs : {backend::RowRuns{}, backend::RowRuns{decode, 1}, backend::RowRuns{prompt, 1},
                                       backend::RowRuns{mixed, batch > 1 ? size_t(2) : size_t(1)}}) {
        cpu.matmul(type, {w.half.get(), 0}, {xb.get(), 1}, got.slice(), width, rows, batch, runs, dtype);
        cpu.matmul(quant::GGML_TYPE_F32, {w.f32.get(), 0}, {xb.get(), 1}, want.slice(), width, rows, batch, runs, dtype);
        same(got, want, at + " matmul");
        cpu.matmul_logits(type, {w.half.get(), 0}, {xb.get(), 1}, got.slice(), width, rows, batch, runs, dtype);
        cpu.matmul_logits(quant::GGML_TYPE_F32, {w.f32.get(), 0}, {xb.get(), 1}, want.slice(), width, rows, batch, runs, dtype);
        same(got, want, at + " head");
        got.reset(); want.reset();
        cpu.matmul_add(type, {w.half.get(), 0}, {xb.get(), 1}, got.slice(), width, rows, batch, runs, dtype);
        cpu.matmul_add(quant::GGML_TYPE_F32, {w.f32.get(), 0}, {xb.get(), 1}, want.slice(), width, rows, batch, runs, dtype);
        same(got, want, at + " residual");
    }
    const uint32_t ids[] = {uint32_t(rows - 1), 0, uint32_t(rows / 2), uint32_t(rows - 1)};
    Output embedded(cpu, width * 4), expanded(cpu, width * 4);
    cpu.embed(embedded.slice(), type, {w.half.get(), 0}, width, rows, ids, 4);
    cpu.embed(expanded.slice(), quant::GGML_TYPE_F32, {w.f32.get(), 0}, width, rows, ids, 4);
    same(embedded, expanded, at + " embedding");
    require(x == original_x && w.packed.front() == 0xa5 && w.packed.back() == 0xa5, at + ": input or weight guard changed");
}

// Mixed F16/BF16/F32 groups and two chosen experts per token retain F32's entry and weighted-sum order.
void grouped(backend::CpuBackend& cpu, size_t width, size_t batch, backend::Dtype dtype) {
    constexpr size_t a_rows = 7, b_rows = 5, experts = 3, chosen = 2;
    Matrix a(cpu, kF16, width, a_rows * experts, 11), b(cpu, kBF16, width, b_rows * experts, 23);
    auto x = inputs(width * batch * chosen);
    const auto original_x = x;
    const auto xb = cpu.adopt(x.data(), x.size() * sizeof(float));
    std::vector<uint32_t> ids(batch * chosen);
    std::vector<float> gains(batch * chosen);
    for (size_t r = 0; r < batch; ++r) {
        ids[2 * r] = uint32_t((2 * r + 2) % experts);
        ids[2 * r + 1] = uint32_t((2 * r) % experts);
        gains[2 * r] = 0.25f; gains[2 * r + 1] = 0.75f;
    }
    const auto ib = cpu.adopt(ids.data(), ids.size() * sizeof(uint32_t)), gb = cpu.adopt(gains.data(), gains.size() * sizeof(float));
    const backend::Backend::Routing routing{{ib.get(), 0}, {gb.get(), 0}, chosen, experts};
    Output ag(cpu, a_rows * batch * chosen), bg(cpu, b_rows * batch * chosen), cg(cpu, a_rows * batch * chosen);
    Output aw(cpu, a_rows * batch * chosen), bw(cpu, b_rows * batch * chosen), cw(cpu, a_rows * batch * chosen);
    Output ad(cpu, a_rows * batch), bd(cpu, b_rows * batch), cd(cpu, a_rows * batch);
    Output ar(cpu, a_rows * batch), br(cpu, b_rows * batch), cr(cpu, a_rows * batch);
    const backend::RowRun decode[] = {{batch, 1}}, prompt[] = {{batch, std::max<size_t>(2, batch)}}, mixed[] = {{1, 1}, {batch, batch}};
    const std::string at = "mixed width " + std::to_string(width) + " batch " + std::to_string(batch);
    for (const backend::RowRuns runs : {backend::RowRuns{decode, 1}, backend::RowRuns{prompt, 1},
                                       backend::RowRuns{mixed, batch > 1 ? size_t(2) : size_t(1)}}) {
        cpu.matmul_group({{kF16, {a.half.get(), 0}, ad.slice(), a_rows}, {kBF16, {b.half.get(), 0}, bd.slice(), b_rows},
                          {quant::GGML_TYPE_F32, {a.f32.get(), 0}, cd.slice(), a_rows}}, {xb.get(), 1}, width, batch, runs, dtype);
        cpu.matmul_group({{quant::GGML_TYPE_F32, {a.f32.get(), 0}, ar.slice(), a_rows}, {quant::GGML_TYPE_F32, {b.f32.get(), 0}, br.slice(), b_rows},
                          {quant::GGML_TYPE_F32, {a.f32.get(), 0}, cr.slice(), a_rows}}, {xb.get(), 1}, width, batch, runs, dtype);
        same(ad, ar, at + " grouped F16"); same(bd, br, at + " grouped BF16"); same(cd, cr, at + " grouped F32");
        cpu.matmul_experts({{kF16, {a.half.get(), 0}, ag.slice(), a_rows}, {kBF16, {b.half.get(), 0}, bg.slice(), b_rows},
                            {quant::GGML_TYPE_F32, {a.f32.get(), 0}, cg.slice(), a_rows}}, {xb.get(), 1}, width, batch, routing, runs, dtype);
        cpu.matmul_experts({{quant::GGML_TYPE_F32, {a.f32.get(), 0}, aw.slice(), a_rows}, {quant::GGML_TYPE_F32, {b.f32.get(), 0}, bw.slice(), b_rows},
                            {quant::GGML_TYPE_F32, {a.f32.get(), 0}, cw.slice(), a_rows}}, {xb.get(), 1}, width, batch, routing, runs, dtype);
        same(ag, aw, at + " routed F16"); same(bg, bw, at + " routed BF16"); same(cg, cw, at + " routed F32");
        ad.reset(); ar.reset(); bd.reset(); br.reset();
        cpu.matmul_experts_add(kF16, {a.half.get(), 0}, {xb.get(), 1}, ad.slice(), width, a_rows, batch, routing, runs, dtype);
        cpu.matmul_experts_add(quant::GGML_TYPE_F32, {a.f32.get(), 0}, {xb.get(), 1}, ar.slice(), width, a_rows, batch, routing, runs, dtype);
        cpu.matmul_experts_add(kBF16, {b.half.get(), 0}, {xb.get(), 1}, bd.slice(), width, b_rows, batch, routing, runs, dtype);
        cpu.matmul_experts_add(quant::GGML_TYPE_F32, {b.f32.get(), 0}, {xb.get(), 1}, br.slice(), width, b_rows, batch, routing, runs, dtype);
        same(ad, ar, at + " routed residual F16"); same(bd, br, at + " routed residual BF16");
    }
    require(x == original_x, at + ": activation input changed");
}

} // namespace

int main() {
    const unsigned csr = _mm_getcsr();
    _mm_setcsr(csr & ~0x8040u);
    try {
        require(bits(widened(kF16, 0x3c00)) == 0x3f800000u && bits(widened(kBF16, 0x3f80)) == 0x3f800000u,
                "one has different encodings in F16 and BF16");
        require(bits(widened(kBF16, 0x3c00)) == 0x3c000000u && bits(widened(kF16, 0x3f80)) == 0x3ff00000u,
                "the two half formats were conflated");
        require(bits(widened(kF16, 1)) == 0x33800000u && bits(widened(kBF16, 1)) == 0x00010000u,
                "oracle smallest subnormal anchors");
        require(bits(widened(kF16, 0x8000)) == 0x80000000u && bits(widened(kBF16, 0x8000)) == 0x80000000u,
                "oracle negative zero anchors");
        backend::CpuBackend cpu;
        cpu.set_threads(1);
        all_finite(cpu, kF16);
        all_finite(cpu, kBF16);
        for (int threads : {1, 3}) {
            cpu.set_threads(threads);
            for (const auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16}) {
                for (uint32_t type : {kF16, kBF16})
                    for (size_t width : {1u, 7u, 8u, 9u, 31u, 32u, 33u, 65u, 257u, 4096u})
                        for (size_t batch : {1u, 2u, 3u, 7u}) dense(cpu, type, width, width >= 257 ? 65 : 7, batch, dtype);
                for (size_t width : {7u, 32u, 65u, 4096u})
                    for (size_t batch : {1u, 3u, 7u}) grouped(cpu, width, batch, dtype);
            }
        }
        _mm_setcsr(csr);
        std::cout << "half weights: " << checked << " exact finite widening and CPU projection values pass\n";
        return 0;
    } catch (const std::exception& e) {
        _mm_setcsr(csr);
        std::cerr << "half weights: " << e.what() << "\n";
        return 1;
    }
}
