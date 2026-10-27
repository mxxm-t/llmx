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
constexpr uint32_t type = 6;
size_t checked = 0;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
uint32_t bits(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }
// Mathematical binary16, independent of the runtime conversion helper.
float scale(uint16_t h) {
    unsigned e = (h >> 10) & 31, f = h & 1023;
    return std::copysign(float(std::ldexp(double(f + (e ? 1024 : 0)), int(e ? e : 1) - 25)),
                         h & 0x8000 ? -1.0f : 1.0f);
}
void block(uint8_t* p, uint16_t h, unsigned shift) {
    std::fill(p, p + 22, uint8_t(0)); p[0] = uint8_t(h); p[1] = uint8_t(h >> 8);
    for (unsigned i = 0; i < 32; ++i) {
        unsigned q = (i + shift) & 31;
        p[2 + i / 8] |= uint8_t((q >> 4) << (i % 8));
        p[6 + i % 16] |= uint8_t((q & 15) << (4 * (i / 16)));
    }
}
float value(uint16_t h, unsigned q) { return scale(h) * float(int(q) - 16); }
void decode() {
    const auto* qt = quant::Registry::instance().get(type);
    require(qt && !qt->quantize && qt->block_size == 32 && qt->type_size == 22,
            "Q5_0 read-only registry entry absent or incorrect");
    std::array<uint8_t, 45> raw{}; std::array<float, 66> out{};
    for (unsigned h = 0; h < 65536; ++h) {
        if ((h & 0x7c00) == 0x7c00) continue;
        block(raw.data() + 1, uint16_t(h), 0); block(raw.data() + 23, uint16_t(h), 17);
        out.fill(123); qt->dequantize(raw.data() + 1, out.data() + 1, 2);
        require(out.front() == 123 && out.back() == 123, "Q5_0 decoder crossed output guards");
        for (unsigned i = 0; i < 64; ++i) {
            require(bits(out[1 + i]) == bits(value(uint16_t(h), (i + (i >= 32 ? 17 : 0)) & 31)),
                    "Q5_0 decode differs from independent finite-scale/code oracle");
            ++checked;
        }
    }
}
void products(backend::CpuBackend& cpu, size_t nin, size_t cols) {
    constexpr size_t rows = 5, experts = 3, picks = 2;
    const size_t rb = nin / 32 * 22, count = cols * rows;
    std::vector<uint8_t> raw(1 + experts * rows * rb);
    std::vector<float> weights(experts * rows * nin), x(cols * picks * nin);
    for (size_t b = 0; b < weights.size() / 32; ++b) {
        const uint16_t h = uint16_t((b % 3 == 0 ? 0x8000 : 0) | (b % 7 == 0 ? 1 : 0x2401));
        block(raw.data() + 1 + b * 22, h, unsigned(b));
        for (unsigned j = 0; j < 32; ++j) weights[b * 32 + j] = value(h, unsigned(b + j) & 31);
    }
    for (size_t i = 0; i < x.size(); ++i) x[i] = float(int(i * 17 % 251) - 125) / 127.0f;
    std::vector<float> a(count * picks + 2), b(a.size());
    std::vector<uint32_t> ids(cols * picks); std::vector<float> gains(ids.size(), 0.5f);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = uint32_t(i % experts);
    auto w = cpu.adopt(raw.data() + 1, raw.size() - 1), in = cpu.adopt(x.data(), x.size() * 4);
    auto out = cpu.adopt(a.data(), a.size() * 4), other = cpu.adopt(b.data(), b.size() * 4);
    auto ib = cpu.adopt(ids.data(), ids.size() * 4), gb = cpu.adopt(gains.data(), gains.size() * 4);
    backend::Backend::Routing routing{{ib.get(), 0}, {gb.get(), 0}, picks, experts};
    const backend::RowRun decode[] = {{cols, 1}}, prompt[] = {{cols, cols}};
    for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8})
    for (auto runs : {backend::RowRuns{decode, 1}, backend::RowRuns{prompt, 1}})
    for (unsigned call = 0; call < 6; ++call) {
        std::fill(a.begin(), a.end(), 19.0f); std::fill(b.begin(), b.end(), 19.0f);
        testq::take_matrix_paths(cpu);
        switch (call) {
        case 0: cpu.matmul({type, {w.get(), 0}}, {in.get(), 0}, {out.get(), 1}, nin, rows, cols, runs, dtype); break;
        case 1: cpu.matmul_add({type, {w.get(), 0}}, {in.get(), 0}, {out.get(), 1}, nin, rows, cols, runs, dtype); break;
        case 2: cpu.matmul_logits({type, {w.get(), 0}}, {in.get(), 0}, {out.get(), 1}, nin, rows, cols, runs, dtype); break;
        case 3: cpu.matmul_group({{{type, {w.get(), 0}}, {out.get(), 1}, rows}, {{type, {w.get(), 0}}, {other.get(), 1}, rows}}, {in.get(), 0}, nin, cols, runs, dtype); break;
        case 4: cpu.matmul_experts({{{type, {w.get(), 0}}, {out.get(), 1}, rows}}, {in.get(), 0}, nin, cols, routing, runs, dtype); break;
        case 5: cpu.matmul_experts_add({type, {w.get(), 0}}, {in.get(), 0}, {out.get(), 1}, nin, rows, cols, routing, runs, dtype); break;
        }
        require(testq::take_matrix_paths(cpu) == std::vector<std::string>{dtype == backend::Dtype::bf16 ? "bf16" : "f32"},
                "Q5_0 wider CPU arithmetic witness incorrect");
        require(a.front() == 19 && a.back() == 19 && b.front() == 19 && b.back() == 19, "Q5_0 product crossed guards");
        const auto dot = [&](size_t expert, size_t row, size_t col, bool down) {
            double sum = 0, magnitude = 0;
            for (size_t k = 0; k < nin; ++k) {
                float v = x[(down ? col : col / picks) * nin + k];
                // Independent nearest-even BF16 rounding from the discarded low bits.
                if (dtype == backend::Dtype::bf16) {
                    uint32_t rawbits = bits(v); const uint32_t low = rawbits & 65535;
                    rawbits = (rawbits & 0xffff0000u) + (low > 32768 || (low == 32768 && (rawbits & 65536)) ? 65536 : 0);
                    std::memcpy(&v, &rawbits, 4);
                }
                const double product = double(weights[(expert * rows + row) * nin + k]) * v;
                sum += product; magnitude += std::fabs(product);
            }
            return std::array<double, 2>{sum, magnitude};
        };
        const size_t n = call == 4 ? cols * picks : cols;
        for (size_t c = 0; c < n; ++c) for (size_t r = 0; r < rows; ++r) {
            std::array<double, 2> reference{};
            if (call == 5) for (size_t p = 0; p < picks; ++p) {
                const auto term = dot(ids[c * picks + p], r, c * picks + p, true);
                reference[0] += 0.5 * term[0]; reference[1] += 0.5 * term[1];
            } else reference = dot(call == 4 ? ids[c] : 0, r, call == 4 ? c : c * picks, false);
            if (call == 1 || call == 5) { reference[0] += 19; reference[1] += 19; }
            const double want = reference[0];
            // Sum-of-magnitudes F32 accumulation bound also covers cancellation and routed adds.
            const double bound = 8 * nin * std::numeric_limits<float>::epsilon() * reference[1]
                               + std::numeric_limits<float>::denorm_min();
            require(std::isfinite(a[1 + c * rows + r]) && std::fabs(double(a[1 + c * rows + r]) - want) <= bound,
                    "Q5_0 product differs from independent double dot");
            if (call == 3) require(bits(a[1 + c * rows + r]) == bits(b[1 + c * rows + r]), "Q5_0 grouped projections differ");
            ++checked;
        }
    }
    std::vector<float> embedding(3 * nin + 2, 19); auto eb = cpu.adopt(embedding.data(), embedding.size() * 4);
    const uint32_t tokens[] = {4, 0, 2}; auto tb = cpu.adopt(tokens, sizeof tokens);
    for (bool resident : {false, true}) {
        if (resident) cpu.embed_ids({eb.get(), 1}, {type, {w.get(), 0}}, nin, rows, {tb.get(), 0}, 3);
        else cpu.embed({eb.get(), 1}, {type, {w.get(), 0}}, nin, rows, tokens, 3);
        require(embedding.front() == 19 && embedding.back() == 19, "Q5_0 embedding crossed guards");
        for (size_t i = 0; i < 3 * nin; ++i) {
            require(bits(embedding[1 + i]) == bits(weights[tokens[i / nin] * nin + i % nin]), "Q5_0 embedding differs bitwise"); ++checked;
        }
    }
}
void ranges(backend::CpuBackend& cpu) {
    std::array<uint8_t, 23> raw{}; std::array<float, 32> x{}; std::array<float, 3> y{};
    auto w = cpu.adopt(raw.data() + 1, 22), in = cpu.adopt(x.data(), sizeof x), out = cpu.adopt(y.data(), sizeof y);
    for (uint16_t h : {uint16_t(0), uint16_t(0x8000), uint16_t(1), uint16_t(0x8001), uint16_t(0x3c00), uint16_t(0x7bff)})
    for (float magnitude : {1e-30f, 1.0f, 1e30f, 1e36f}) {
        block(raw.data() + 1, h, 0);
        for (size_t i = 0; i < 32; ++i) {
            x.fill(0); x[i] = magnitude; y.fill(19);
            const double want = double(value(h, unsigned(i))) * magnitude;
            if (std::fabs(want) > std::numeric_limits<float>::max()) continue;
            cpu.matmul({type, {w.get(), 0}}, {in.get(), 0}, {out.get(), 1}, 32, 1, 1);
            const double bound = std::max(double(std::numeric_limits<float>::denorm_min()), std::fabs(want) * 2e-7);
            require(y.front() == 19 && y.back() == 19 && std::isfinite(y[1]) && std::fabs(double(y[1]) - want) <= bound,
                    "Q5_0 finite one-hot range differs from independent product"); ++checked;
        }
    }
}
}
int main() {
    try {
        _mm_setcsr(_mm_getcsr() & ~unsigned(0x8040)); decode();
        backend::CpuBackend cpu; cpu.set_threads(1); ranges(cpu);
        for (int threads : {1, 3}) { cpu.set_threads(threads);
            for (size_t nin : {32u, 64u, 96u, 544u}) for (size_t cols : {1u, 3u, 9u}) products(cpu, nin, cols);
        }
        const size_t classes = row_classes::check(cpu, {type}, [](uint32_t t, size_t nin, size_t rows, uint32_t seed) {
            require(t == type && nin % 32 == 0, "invalid Q5_0 row-class fixture");
            std::vector<uint8_t> raw(rows * (nin / 32) * 22);
            for (size_t b = 0; b < raw.size() / 22; ++b) {
                seed = seed * 1664525u + 1013904223u;
                block(raw.data() + b * 22, uint16_t(0x1400u | ((seed >> 16) & 0x83ffu)), seed & 31);
            }
            return raw;
        });
        std::cout << "Q5_0: " << checked << " independent decode/product/embedding values and " << classes << " exact row-class pairs passed; F32 wider and BF16 witnesses checked\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
