#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "backends/cpu/cpu_backend.hpp"
#include "matrix_precision.hpp"
#include "row_classes.hpp"

namespace {

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

uint32_t bits(float value) {
    uint32_t out;
    std::memcpy(&out, &value, sizeof out);
    return out;
}

// Independent format values and mathematical binary16 widening, without the runtime's table or converter.
float weight(uint16_t h, unsigned code, int multiplier = 1) {
    constexpr int values[] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    const unsigned exponent = (h >> 10) & 31, fraction = h & 1023;
    double scale = exponent ? std::ldexp(double(1024 + fraction), int(exponent) - 25)
                            : std::ldexp(double(fraction), -24);
    if (h & 0x8000) scale = -scale;
    return float(scale * multiplier * values[code]);
}

void block(uint8_t* p, uint16_t h, unsigned shift) {
    p[0] = uint8_t(h); p[1] = uint8_t(h >> 8);
    for (unsigned i = 0; i < 16; ++i)
        p[2 + i] = uint8_t(((i + shift) & 15) | (((5 * i + 3 + shift) & 15) << 4));
}

unsigned code(const uint8_t* p, size_t i) {
    return (p[2 + i % 16] >> (i < 16 ? 0 : 4)) & 15;
}

// The XS fixture packs the logical scale fields independently of the production decoder.
void test_block(uint32_t type, uint8_t* p, uint16_t h, unsigned shift) {
    if (type == 20) { block(p, h, shift); return; }
    p[0] = uint8_t(h); p[1] = uint8_t(h >> 8);
    std::fill(p + 2, p + 8, uint8_t(0));
    for (unsigned g = 0; g < 8; ++g) {
        const unsigned scale = (shift + 9 * g) % 64;
        p[2 + g / 4] |= uint8_t((scale / 16) << (2 * (g % 4)));
        p[4 + g / 2] |= uint8_t((scale % 16) << (4 * (g % 2)));
        for (unsigned j = 0; j < 16; ++j)
            p[8 + g * 16 + j] = uint8_t(((j + shift + g) % 16) | (((5 * j + 3 + shift + g) % 16) << 4));
    }
}

float test_weight(uint32_t type, const uint8_t* p, uint16_t h, unsigned shift, size_t i) {
    if (type == 20) return weight(h, code(p, i));
    const unsigned packed = p[8 + (i / 32) * 16 + i % 16];
    const unsigned value = (packed >> (i % 32 < 16 ? 0 : 4)) & 15;
    return weight(h, value, int((shift + 9 * (i / 32)) % 64) - 32);
}

size_t xs_decoder() {
    const auto* qt = quant::Registry::instance().get(23);
    require(qt && qt->block_size == 256 && qt->type_size == 136 && qt->dequantize && !qt->quantize,
            "IQ4_XS read-only registry entry differs from the format");
    require(quant::row_bytes(23, 768, 3) == 1224, "IQ4_XS row size differs");
    qt->dequantize(nullptr, nullptr, 0);
    std::array<uint8_t, 137> storage{};
    std::array<float, 258> output{};
    size_t checked = 0;
    const auto check = [&](uint16_t h, unsigned shift) {
        test_block(23, storage.data() + 1, h, shift);
        const auto saved = storage;
        output.fill(123);
        qt->dequantize(storage.data() + 1, output.data() + 1, 1);
        require(storage == saved && output.front() == 123 && output.back() == 123, "IQ4_XS decode changed input or guards");
        for (size_t i = 0; i < 256; ++i) {
            require(bits(output[i + 1]) == bits(test_weight(23, storage.data() + 1, h, shift, i)),
                    "IQ4_XS half/subscale/code decode differs, including signed zero");
            ++checked;
        }
    };
    // Every finite half scale with all codes in every sub-block, including subscale zero.
    for (unsigned h = 0; h < 65536; ++h)
        if ((h & 0x7c00) != 0x7c00) check(uint16_t(h), 32);
    // Every six-bit scale in each sub-block, all codes at every nibble position, and both signs/edges of d.
    for (uint16_t h : {uint16_t(0), uint16_t(0x8000), uint16_t(1), uint16_t(0x8001), uint16_t(0x03ff),
                       uint16_t(0x0400), uint16_t(0x3c00), uint16_t(0xbc00), uint16_t(0x7bff), uint16_t(0xfbff)})
        for (unsigned shift = 0; shift < 64; ++shift) check(h, shift);
    return checked;
}

size_t decoder() {
    const auto* qt = quant::Registry::instance().get(20);
    require(qt && qt->block_size == 32 && qt->type_size == 18 && qt->dequantize && !qt->quantize,
            "IQ4_NL read-only registry entry differs from the format");
    require(quant::row_bytes(20, 96, 3) == 162, "IQ4_NL row sizing differs");
    qt->dequantize(nullptr, nullptr, 0);
    size_t checked = 0;
    std::array<uint8_t, 19> storage{};
    std::array<float, 34> output;
    for (unsigned h = 0; h < 65536; ++h) {
        if ((h & 0x7c00) == 0x7c00) continue;
        for (unsigned shift = 0; shift < 16; ++shift) {
            block(storage.data() + 1, uint16_t(h), shift);
            const auto saved = storage;
            output.fill(123.0f);
            qt->dequantize(storage.data() + 1, output.data() + 1, 1);
            require(storage == saved, "IQ4_NL decoder changed input");
            require(output.front() == 123 && output.back() == 123, "IQ4_NL decoder crossed output guards");
            for (size_t i = 0; i < 32; ++i) {
                require(bits(output[i + 1]) == bits(weight(uint16_t(h), code(storage.data() + 1, i))),
                        "IQ4_NL finite scale/code decode differs, including signed zero");
                ++checked;
            }
        }
    }
    return checked;
}

size_t products(uint32_t type, backend::CpuBackend& cpu, size_t nin, size_t nout, size_t cols) {
    constexpr size_t experts = 3, picks = 2;
    const size_t values = type == 23 ? 256 : 32, bytes = type == 23 ? 136 : 18;
    const size_t rb = nin / values * bytes, count = cols * nout;
    std::vector<uint8_t> packed(1 + experts * nout * rb);
    std::vector<float> widened(1 + experts * nout * nin), x(1 + cols * picks * nin);
    constexpr uint16_t scales[] = {0x2401, 0xb123, 1, 0x83ff, 0x3c00};
    for (size_t b = 0; b < experts * nout * nin / values; ++b) {
        uint8_t* p = packed.data() + 1 + b * bytes;
        const uint16_t h = scales[b % 5];
        test_block(type, p, h, unsigned(b));
        for (size_t i = 0; i < values; ++i) widened[1 + b * values + i] = test_weight(type, p, h, unsigned(b), i);
    }
    for (size_t i = 1; i < x.size(); ++i) x[i] = float(int((i * 769 + 23) % 32749) - 16374) / 523.0f;
    std::vector<float> a(count * picks + 2), b(a.size()), c(a.size()), d(a.size());
    std::vector<uint32_t> ids(cols * picks);
    std::vector<float> gains(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) { ids[i] = uint32_t((i * 2 + 1) % experts); gains[i] = i % 2 ? 0.25f : 0.75f; }
    const auto pw = cpu.adopt(packed.data() + 1, packed.size() - 1);
    const auto fw = cpu.adopt(widened.data() + 1, (widened.size() - 1) * sizeof(float));
    const auto xb = cpu.adopt(x.data(), x.size() * sizeof(float));
    const auto ab = cpu.adopt(a.data(), a.size() * sizeof(float)), bb = cpu.adopt(b.data(), b.size() * sizeof(float));
    const auto cb = cpu.adopt(c.data(), c.size() * sizeof(float)), db = cpu.adopt(d.data(), d.size() * sizeof(float));
    const auto ib = cpu.adopt(ids.data(), ids.size() * sizeof(uint32_t)), gb = cpu.adopt(gains.data(), gains.size() * sizeof(float));
    const backend::Backend::Routing routing{{ib.get(), 0}, {gb.get(), 0}, picks, experts};
    const backend::RowRun decode[] = {{cols, 1}}, prompt[] = {{cols, cols}}, mixed[] = {{1, 1}, {cols, cols}};
    size_t checked = 0;
    const auto reset = [&] { for (auto* out : {&a, &b, &c, &d}) std::fill(out->begin(), out->end(), 17.0f); };
    const auto check = [&](const std::vector<float>& actual, const std::vector<float>& expected, size_t n) {
        require(actual.front() == 17 && actual.back() == 17, "IQ4 product crossed guards");
        require(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0,
                "IQ4 product differs from independently widened F32 weights");
        for (size_t i = 1; i <= n; ++i) require(std::isfinite(actual[i]), "IQ4 ordinary product is not finite");
        checked += n;
    };
    for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8})
    for (backend::RowRuns runs : {backend::RowRuns{decode, 1}, backend::RowRuns{prompt, 1}, backend::RowRuns{mixed, 2}}) {
        reset();
        cpu.matmul({type, {pw.get(), 0}}, {xb.get(), 1}, {ab.get(), 1}, nin, nout, cols, runs, dtype);
        cpu.matmul({0, {fw.get(), 0}}, {xb.get(), 1}, {bb.get(), 1}, nin, nout, cols, runs, dtype);
        check(a, b, count);
        reset();
        cpu.matmul_add({type, {pw.get(), 0}}, {xb.get(), 1}, {ab.get(), 1}, nin, nout, cols, runs, dtype);
        cpu.matmul_add({0, {fw.get(), 0}}, {xb.get(), 1}, {bb.get(), 1}, nin, nout, cols, runs, dtype);
        check(a, b, count);
        reset();
        cpu.matmul_logits({type, {pw.get(), 0}}, {xb.get(), 1}, {ab.get(), 1}, nin, nout, cols, runs, dtype);
        cpu.matmul_logits({0, {fw.get(), 0}}, {xb.get(), 1}, {bb.get(), 1}, nin, nout, cols, runs, dtype);
        check(a, b, count);
        reset();
        cpu.matmul_group({{{type, {pw.get(), 0}}, {ab.get(), 1}, nout}, {{type, {pw.get(), 0}}, {cb.get(), 1}, nout}}, {xb.get(), 1}, nin, cols, runs, dtype);
        cpu.matmul_group({{{0, {fw.get(), 0}}, {bb.get(), 1}, nout}, {{0, {fw.get(), 0}}, {db.get(), 1}, nout}}, {xb.get(), 1}, nin, cols, runs, dtype);
        check(a, b, count); check(c, d, count);
        reset();
        cpu.matmul_experts({{{type, {pw.get(), 0}}, {ab.get(), 1}, nout}}, {xb.get(), 1}, nin, cols, routing, runs, dtype);
        cpu.matmul_experts({{{0, {fw.get(), 0}}, {bb.get(), 1}, nout}}, {xb.get(), 1}, nin, cols, routing, runs, dtype);
        check(a, b, count * picks);
        reset();
        cpu.matmul_experts_add({type, {pw.get(), 0}}, {xb.get(), 1}, {ab.get(), 1}, nin, nout, cols, routing, runs, dtype);
        cpu.matmul_experts_add({0, {fw.get(), 0}}, {xb.get(), 1}, {bb.get(), 1}, nin, nout, cols, routing, runs, dtype);
        check(a, b, count);
        const auto paths = testq::take_matrix_paths(cpu);
        require(paths == std::vector<std::string>{dtype == backend::Dtype::bf16 ? "bf16" : "f32"}, "IQ4 arithmetic witness differs");
    }
    const std::array<uint32_t, 4> tokens{uint32_t(nout - 1), 0, uint32_t(nout / 2), uint32_t(nout)};
    std::vector<float> embedding(4 * nin + 2, 17), reference(embedding);
    const auto eb = cpu.adopt(embedding.data(), embedding.size() * sizeof(float));
    const auto rbuff = cpu.adopt(reference.data(), reference.size() * sizeof(float));
    const auto tokensb = cpu.adopt(tokens.data(), sizeof tokens);
    cpu.embed({eb.get(), 1}, {type, {pw.get(), 0}}, nin, nout, tokens.data(), 3);
    cpu.embed({rbuff.get(), 1}, {0, {fw.get(), 0}}, nin, nout, tokens.data(), 3);
    check(embedding, reference, 3 * nin);
    cpu.embed_ids({eb.get(), 1}, {type, {pw.get(), 0}}, nin, nout, {tokensb.get(), 0}, tokens.size());
    cpu.embed_ids({rbuff.get(), 1}, {0, {fw.get(), 0}}, nin, nout, {tokensb.get(), 0}, tokens.size());
    check(embedding, reference, 4 * nin);
    return checked;
}

size_t ranges(uint32_t type, backend::CpuBackend& cpu) {
    const size_t width = type == 23 ? 256 : 32, size = type == 23 ? 136 : 18;
    std::array<uint8_t, 137> bytes{};
    std::array<float, 257> x{};
    std::array<float, 3> y{};
    const auto wb = cpu.adopt(bytes.data() + 1, size), xb = cpu.adopt(x.data(), sizeof x), yb = cpu.adopt(y.data(), sizeof y);
    size_t checked = 0;
    for (uint16_t h : {uint16_t(0), uint16_t(0x8000), uint16_t(1), uint16_t(0x8001), uint16_t(0x3c00), uint16_t(0x7bff)})
    for (unsigned shift = 0; shift < (type == 23 ? 64u : 16u); ++shift)
    for (float scale : {1e-30f, 1.0f, 1e30f, 1e36f}) {
        test_block(type, bytes.data() + 1, h, shift);
        for (size_t lane = 0; lane < width; ++lane) {
            x.fill(0); x[lane + 1] = scale;
            y.fill(17);
            const double want = double(test_weight(type, bytes.data() + 1, h, shift, lane)) * scale;
            if (std::fabs(want) > std::numeric_limits<float>::max()) continue;
            cpu.matmul({type, {wb.get(), 0}}, {xb.get(), 1}, {yb.get(), 1}, width, 1, 1);
            require(y.front() == 17 && y.back() == 17, "IQ4 range product crossed guards");
            const double bound = std::max(double(std::numeric_limits<float>::denorm_min()), std::fabs(want) * 2e-7);
            require(std::isfinite(y[1]) && std::fabs(double(y[1]) - want) <= bound,
                    "IQ4 scale-before-product range differs from double oracle");
            ++checked;
        }
    }
    return checked;
}

} // namespace

int main() {
    try {
        // Gradual underflow also applies to workers created below.
        _mm_setcsr(_mm_getcsr() & ~unsigned(0x8040));
        const size_t decoded = decoder(), xs_decoded = xs_decoder();
        backend::CpuBackend cpu;
        cpu.set_threads(1);
        require(cpu.supports_type(20) && cpu.supports_type(23), "CPU refuses an IQ4 type");
        const size_t bounded = ranges(20, cpu) + ranges(23, cpu);
        size_t checked = 0;
        for (int threads : {1, 3}) {
            cpu.set_threads(threads);
            for (uint32_t type : {20u, 23u})
            for (size_t nin : (type == 20 ? std::vector<size_t>{32, 96, 544, 4096} : std::vector<size_t>{256, 768, 4352, 4096}))
            for (size_t nout : {1u, 5u, 65u})
            for (size_t cols : {1u, 2u, 3u, 9u}) checked += products(type, cpu, nin, nout, cols);
        }
        const size_t classes = row_classes::check(cpu, {20, 23}, [](uint32_t type, size_t nin, size_t rows, uint32_t seed) {
            const size_t values = type == 23 ? 256 : 32, bytes = type == 23 ? 136 : 18;
            require((type == 20 || type == 23) && nin % values == 0, "invalid IQ4 row-class fixture");
            std::vector<uint8_t> packed(rows * (nin / values) * bytes);
            for (size_t b = 0; b < packed.size() / bytes; ++b) {
                seed = seed * 1664525u + 1013904223u;
                test_block(type, packed.data() + b * bytes, uint16_t(0x1400u | ((seed >> 16) & 0x83ffu)), seed & 63);
            }
            return packed;
        });
        std::cout << "IQ4_NL: " << decoded << " and IQ4_XS: " << xs_decoded << " exact decoded values, " << bounded << " range dots, "
                  << checked << " exact widened-F32 product/embedding values, " << classes << " exact row-class pairs passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
