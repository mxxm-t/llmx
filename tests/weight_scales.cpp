#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "backends/cpu/cpu_backend.hpp"

namespace {
constexpr size_t width = 128, rows = 3, row_bytes = 72;
constexpr float guard = 12345.0f;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
uint32_t bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }

// Test-owned E2M1 arithmetic, with ordinary E4M3 scale 1 in every group.
float coefficient(unsigned code) {
    unsigned e = (code >> 1) & 3, m = code & 1;
    if (!(code & 7)) return 0;
    const float value = e ? std::ldexp(float(2 + m), int(e) - 2) : 0.5f;
    return code & 8 ? -value : value;
}
float expected(const uint8_t* data, size_t row, size_t i, float scale) {
    const uint8_t* block = data + row * row_bytes + (i / 64) * 36;
    const size_t lane = i % 16;
    const unsigned code = (block[4 + ((i % 64) / 16) * 8 + lane % 8] >> (4 * (lane / 8))) & 15;
    return coefficient(code) * scale;
}
std::vector<uint8_t> packed(size_t experts = 1) {
    std::vector<uint8_t> data(experts * rows * row_bytes);
    for (size_t b = 0; b < experts * rows * 2; ++b) {
        uint8_t* p = data.data() + b * 36;
        for (size_t g = 0; g < 4; ++g) p[g] = 56;
        for (size_t j = 0; j < 32; ++j) p[4 + j] = uint8_t(j * 73 + b * 11);
    }
    return data;
}
std::vector<float> read(backend::CpuBackend& b, const backend::Buffer& y, size_t count) {
    std::vector<float> out(count); b.read(y, 0, out.data(), count * sizeof(float)); return out;
}
void reset(backend::CpuBackend& b, backend::Buffer& y, size_t count, float value = guard) {
    std::vector<float> data(count, value); b.write(y, 0, data.data(), count * sizeof(float));
}

void dense(backend::CpuBackend& cpu) {
    const auto data = packed();
    auto w = cpu.adopt(data.data(), data.size());
    std::vector<float> input(width * width, 0);
    for (size_t i = 0; i < width; ++i) input[i * width + i] = 1;
    auto x = cpu.adopt(input.data(), input.size() * sizeof(float));
    auto y = cpu.alloc((width * rows + 2) * sizeof(float), backend::Memory::host_visible);
    for (float scale : {0.0f, -0.0f, 0.0001f, -0.0001f, std::ldexp(1.0f, -100), -std::ldexp(1.0f, 80)}) {
        const float scales[] = {guard, scale, guard};
        auto sb = cpu.adopt(scales, sizeof scales);
        const backend::WeightView weight{quant::GGML_TYPE_NVFP4, {w.get(), 0}, {sb.get(), 1}, 1};
        for (bool decode : {false, true}) {
            reset(cpu, *y, width * rows + 2);
            if (decode) {
                for (size_t i = 0; i < width; ++i)
                    cpu.matmul(weight, {x.get(), i * width}, {y.get(), 1 + i * rows}, width, rows, 1, {}, backend::Dtype::f32);
            } else cpu.matmul_logits(weight, {x.get(), 0}, {y.get(), 1}, width, rows, width, {}, backend::Dtype::f32);
            auto out = read(cpu, *y, width * rows + 2);
            require(out.front() == guard && out.back() == guard, "scaled dense guards");
            for (size_t i = 0; i < width; ++i) for (size_t r = 0; r < rows; ++r)
                require(out[1 + i * rows + r] == expected(data.data(), r, i, scale), "scaled dense one-hot product");
        }
        const uint32_t ids[] = {2, 0, 1};
        reset(cpu, *y, width * rows + 2);
        cpu.embed({y.get(), 1}, weight, width, rows, ids, 3);
        auto out = read(cpu, *y, width * rows + 2);
        for (size_t r = 0; r < rows; ++r) for (size_t i = 0; i < width; ++i)
            require(bits(out[1 + r * width + i]) == bits(expected(data.data(), ids[r], i, scale)), "scaled embedding signed zero");
        auto ib = cpu.adopt(ids, sizeof ids);
        cpu.embed_ids({y.get(), 1}, weight, width, rows, {ib.get(), 0}, 3);
        require(read(cpu, *y, width * rows + 2) == out, "scaled device-id embedding");
    }
    // An unscaled NVFP4 product overflows; the actual F32 weights and result fit.
    std::vector<float> huge(width, 0); huge[7] = 1e38f;
    auto hx = cpu.adopt(huge.data(), huge.size() * sizeof(float));
    const float scale = 0.0001f;
    auto sb = cpu.adopt(&scale, sizeof scale);
    const backend::WeightView weight{quant::GGML_TYPE_NVFP4, {w.get(), 0}, {sb.get(), 0}, 1};
    reset(cpu, *y, width * rows + 2, 0.25f);
    cpu.matmul_add(weight, {hx.get(), 0}, {y.get(), 1}, width, rows, 1, {}, backend::Dtype::f32);
    const auto out = read(cpu, *y, width * rows + 2);
    for (size_t r = 0; r < rows; ++r) {
        const float want = expected(data.data(), r, 7, scale) * huge[7] + 0.25f;
        require(std::isfinite(out[1 + r]) && std::fabs(double(out[1 + r]) - want) <= std::fabs(double(want)) * 2e-7,
                "scaled weight range repair or before-add order");
    }
}

void routed(backend::CpuBackend& cpu) {
    const auto data = packed(2);
    auto w = cpu.adopt(data.data(), data.size());
    const float scales[] = {0.125f, -0.25f};
    auto sb = cpu.adopt(scales, sizeof scales);
    const backend::WeightView weight{quant::GGML_TYPE_NVFP4, {w.get(), 0}, {sb.get(), 0}, 2};
    const uint32_t ids[] = {1, 0, 0, 1};
    const float gains[] = {0.25f, 0.75f, 0.5f, 0.5f};
    auto ib = cpu.adopt(ids, sizeof ids), gb = cpu.adopt(gains, sizeof gains);
    const backend::Backend::Routing routing{{ib.get(), 0}, {gb.get(), 0}, 2, 2};
    std::vector<float> input(4 * width, 0); input[7] = 1; input[width + 17] = 1;
    input[2 * width + 23] = 1; input[3 * width + 41] = 1;
    auto x = cpu.adopt(input.data(), input.size() * sizeof(float));
    auto y = cpu.alloc((4 * rows + 2) * sizeof(float), backend::Memory::host_visible);
    for (size_t extent : {1u, 3u}) {
        const backend::RowRun run{2, extent};
        reset(cpu, *y, 4 * rows + 2);
        cpu.matmul_experts({{weight, {y.get(), 1}, rows}}, {x.get(), 0}, width, 2, routing, {&run, 1}, backend::Dtype::f32);
        auto out = read(cpu, *y, 4 * rows + 2);
        for (size_t entry = 0; entry < 4; ++entry) for (size_t r = 0; r < rows; ++r) {
            const size_t lane = entry < 2 ? 7 : 17;
            require(out[1 + entry * rows + r] == expected(data.data(), ids[entry] * rows + r, lane, scales[ids[entry]]),
                    "per-expert scaled product");
        }
        reset(cpu, *y, 4 * rows + 2, 0.25f);
        cpu.matmul_experts_add(weight, {x.get(), 0}, {y.get(), 1}, width, rows, 2, routing, {&run, 1}, backend::Dtype::f32);
        out = read(cpu, *y, 4 * rows + 2);
        const size_t lanes[] = {7, 17, 23, 41};
        for (size_t token = 0; token < 2; ++token) for (size_t r = 0; r < rows; ++r) {
            float sum = 0;
            for (size_t pick = 0; pick < 2; ++pick) {
                const size_t e = token * 2 + pick;
                sum += gains[e] * expected(data.data(), ids[e] * rows + r, lanes[e], scales[ids[e]]);
            }
            require(out[1 + token * rows + r] == sum + 0.25f, "per-expert scaled weighted-add order");
        }
    }
}

void refusals(backend::CpuBackend& cpu) {
    const auto data = packed(); auto w = cpu.adopt(data.data(), data.size());
    const float scale = 1; auto sb = cpu.adopt(&scale, sizeof scale);
    std::vector<float> input(width, 1); auto x = cpu.adopt(input.data(), input.size() * sizeof(float));
    auto y = cpu.alloc(2 * rows * sizeof(float), backend::Memory::host_visible);
    const backend::WeightView good{quant::GGML_TYPE_NVFP4, {w.get(), 0}};
    const backend::WeightView bad[] = {
        {quant::GGML_TYPE_NVFP4, {w.get(), 0}, {}, 1},
        {quant::GGML_TYPE_NVFP4, {w.get(), 0}, {sb.get(), 0}, 0},
        {quant::GGML_TYPE_NVFP4, {w.get(), 0}, {sb.get(), 0}, 2},
        {quant::GGML_TYPE_NVFP4, {w.get(), 0}, {sb.get(), 1}, 1},
        {quant::GGML_TYPE_Q4_0, {w.get(), 0}, {sb.get(), 0}, 1},
    };
    for (const auto& weight : bad) {
        reset(cpu, *y, 2 * rows);
        bool refused = false;
        try { cpu.matmul_group({{good, {y.get(), 0}, rows}, {weight, {y.get(), rows}, rows}}, {x.get(), 0}, width, 1, {}, backend::Dtype::f32); }
        catch (const std::runtime_error&) { refused = true; }
        require(refused, "malformed scale descriptor accepted");
        const auto out = read(cpu, *y, 2 * rows);
        for (float value : out) require(value == guard, "malformed later scale wrote grouped output");
    }
}

// F8_E4M3 under its scale grid: the test's own E4M3 arithmetic, a 200-row matrix of 256 inputs (two row blocks, the second partial) at row shifts 0 and 7.
constexpr size_t fp8_rows = 200, fp8_in = 256;
float e4m3(uint8_t c) {
    const int e = (c >> 3) & 15, m = c & 7;
    const float v = e ? std::ldexp(float(8 + m), e - 10) : std::ldexp(float(m), -9);
    return c & 0x80 ? -v : v;
}
uint8_t fp8_code(size_t e, size_t r, size_t k) {
    const uint8_t c = uint8_t(r * 37 + k * 11 + e * 101);
    return (c & 0x7F) == 0x7F ? 0x3F : c;
}
std::vector<float> fp8_grid(unsigned shift, size_t experts = 1) {
    std::vector<float> g(experts * (((fp8_rows - 1) >> shift) + 1) * (fp8_in / 128));
    for (size_t i = 0; i < g.size(); ++i) g[i] = std::ldexp(float(int(i % 7) - 3) * 0.75f + 0.5f, -int(i % 5) - 6);
    return g;
}
float fp8_weight(const std::vector<float>& g, unsigned shift, size_t e, size_t r, size_t k) {
    const size_t per = (((fp8_rows - 1) >> shift) + 1) * (fp8_in / 128);
    return e4m3(fp8_code(e, r, k)) * g[e * per + (r >> shift) * (fp8_in / 128) + k / 128];
}

void fp8(backend::CpuBackend& cpu) {
    std::vector<uint8_t> codes(2 * fp8_rows * fp8_in);
    for (size_t e = 0; e < 2; ++e) for (size_t r = 0; r < fp8_rows; ++r) for (size_t k = 0; k < fp8_in; ++k) codes[(e * fp8_rows + r) * fp8_in + k] = fp8_code(e, r, k);
    auto w = cpu.adopt(codes.data(), codes.size());
    // Columns: one-hot at input i with 1 + 2^-12, which binary16 rounds to 1, so f16 and f32 differ by that factor exactly.
    const size_t lanes[] = {0, 77, 127, 128, 200, 255};
    const float bump = 1.0f + std::ldexp(1.0f, -12);
    std::vector<float> input(6 * fp8_in, 0);
    for (size_t c = 0; c < 6; ++c) input[c * fp8_in + lanes[c]] = bump;
    auto x = cpu.adopt(input.data(), input.size() * sizeof(float));
    auto y = cpu.alloc((6 * fp8_rows + 2) * sizeof(float), backend::Memory::host_visible);
    for (unsigned shift : {0u, 7u}) {
        const auto grid = fp8_grid(shift);
        auto gb = cpu.adopt(grid.data(), grid.size() * sizeof(float));
        const backend::WeightView weight{quant::GGML_TYPE_F8_E4M3, {w.get(), 0}, {gb.get(), 0}, grid.size(), shift};
        for (backend::Dtype dtype : {backend::Dtype::f32, backend::Dtype::f16}) {
            const float factor = dtype == backend::Dtype::f16 ? 1.0f : bump;
            for (bool decode : {false, true}) {
                reset(cpu, *y, 6 * fp8_rows + 2);
                if (decode) for (size_t c = 0; c < 6; ++c) cpu.matmul(weight, {x.get(), c * fp8_in}, {y.get(), 1 + c * fp8_rows}, fp8_in, fp8_rows, 1, {}, dtype);
                else cpu.matmul(weight, {x.get(), 0}, {y.get(), 1}, fp8_in, fp8_rows, 6, {}, dtype);
                const auto out = read(cpu, *y, 6 * fp8_rows + 2);
                require(out.front() == guard && out.back() == guard, "F8_E4M3 guards");
                for (size_t c = 0; c < 6; ++c) for (size_t r = 0; r < fp8_rows; ++r)
                    require(out[1 + c * fp8_rows + r] == e4m3(fp8_code(0, r, lanes[c])) * factor * grid[(r >> shift) * 2 + lanes[c] / 128],
                            "F8_E4M3 one-hot product: code times its grid scale, the input binary16 under f16");
            }
        }
        const uint32_t ids[] = {199, 0, 128};
        reset(cpu, *y, 3 * fp8_in + 2);
        cpu.embed({y.get(), 1}, weight, fp8_in, fp8_rows, ids, 3);
        const auto out = read(cpu, *y, 3 * fp8_in + 2);
        for (size_t i = 0; i < 3; ++i) for (size_t k = 0; k < fp8_in; ++k)
            require(bits(out[1 + i * fp8_in + k]) == bits(fp8_weight(grid, shift, 0, ids[i], k)), "F8_E4M3 embedding row under its grid");
    }
    // A dense column against a reference of exact 128-input partials, and the same column inside a batch of three keeps its bits.
    std::vector<float> dense(3 * fp8_in);
    for (size_t i = 0; i < dense.size(); ++i) dense[i] = std::sin(float(i) * 0.37f);
    auto dx = cpu.adopt(dense.data(), dense.size() * sizeof(float));
    const auto grid = fp8_grid(7);
    auto gb = cpu.adopt(grid.data(), grid.size() * sizeof(float));
    const backend::WeightView weight{quant::GGML_TYPE_F8_E4M3, {w.get(), 0}, {gb.get(), 0}, grid.size(), 7};
    reset(cpu, *y, 3 * fp8_rows + 2);
    cpu.matmul(weight, {dx.get(), 0}, {y.get(), 1}, fp8_in, fp8_rows, 3, {}, backend::Dtype::f32);
    const auto batch = read(cpu, *y, 3 * fp8_rows + 2);
    for (size_t c = 0; c < 3; ++c) {
        reset(cpu, *y, 3 * fp8_rows + 2);
        cpu.matmul(weight, {dx.get(), c * fp8_in}, {y.get(), 1}, fp8_in, fp8_rows, 1, {}, backend::Dtype::f32);
        const auto one = read(cpu, *y, 3 * fp8_rows + 2);
        for (size_t r = 0; r < fp8_rows; ++r) {
            require(bits(one[1 + r]) == bits(batch[1 + c * fp8_rows + r]), "F8_E4M3 row bits independent of the batch");
            double want = 0, size = 0;
            for (size_t k = 0; k < fp8_in; ++k) {
                want += double(fp8_weight(grid, 7, 0, r, k)) * dense[c * fp8_in + k];
                size += std::fabs(double(fp8_weight(grid, 7, 0, r, k)) * dense[c * fp8_in + k]);
            }
            require(std::fabs(double(one[1 + r]) - want) <= size * 1e-6, "F8_E4M3 dense product against exact partials");
        }
    }
    // Two experts, each with its own grid: one-hot entries read their expert's codes and scales.
    const auto grids = fp8_grid(0, 2);
    auto gg = cpu.adopt(grids.data(), grids.size() * sizeof(float));
    const backend::WeightView experts{quant::GGML_TYPE_F8_E4M3, {w.get(), 0}, {gg.get(), 0}, grids.size(), 0};
    const uint32_t picks[] = {1, 0, 0, 1};
    const float gains[] = {1, 1, 1, 1};
    auto ib = cpu.adopt(picks, sizeof picks), wb = cpu.adopt(gains, sizeof gains);
    const backend::Backend::Routing routing{{ib.get(), 0}, {wb.get(), 0}, 2, 2};
    for (size_t extent : {1u, 3u}) {
        const backend::RowRun run{2, extent};
        reset(cpu, *y, 4 * fp8_rows + 2);
        cpu.matmul_experts({{experts, {y.get(), 1}, fp8_rows}}, {x.get(), 0}, fp8_in, 2, routing, {&run, 1}, backend::Dtype::f32);
        const auto out = read(cpu, *y, 4 * fp8_rows + 2);
        for (size_t entry = 0; entry < 4; ++entry) for (size_t r = 0; r < fp8_rows; ++r)
            require(out[1 + entry * fp8_rows + r] == e4m3(fp8_code(picks[entry], r, lanes[entry / 2])) * bump * grids[picks[entry] * 400 + r * 2 + lanes[entry / 2] / 128],
                    "F8_E4M3 per-expert grid product");
    }
    // A grid of another count or row shift, or none, is refused before any output is written.
    const backend::WeightView bad[] = {
        {quant::GGML_TYPE_F8_E4M3, {w.get(), 0}, {gb.get(), 0}, grid.size() - 1, 7},
        {quant::GGML_TYPE_F8_E4M3, {w.get(), 0}, {gb.get(), 0}, grid.size(), 6},
        {quant::GGML_TYPE_F8_E4M3, {w.get(), 0}, {gb.get(), 0}, grid.size(), 0},
        {quant::GGML_TYPE_F8_E4M3, {w.get(), 0}},
    };
    for (const auto& view : bad) {
        reset(cpu, *y, fp8_rows);
        bool refused = false;
        try { cpu.matmul(view, {x.get(), 0}, {y.get(), 0}, fp8_in, fp8_rows, 1, {}, backend::Dtype::f32); }
        catch (const std::runtime_error&) { refused = true; }
        require(refused, "malformed F8_E4M3 grid accepted");
        for (float value : read(cpu, *y, fp8_rows)) require(value == guard, "malformed F8_E4M3 grid wrote output");
    }
}
}
int main() {
    try {
        for (int threads : {1, 3}) { backend::CpuBackend cpu; cpu.set_threads(threads); dense(cpu); routed(cpu); refusals(cpu); fp8(cpu); }
        std::cout << "weight scales: signed/zero/range dense+embedding, per-expert products/adds and pre-write refusals; F8_E4M3 grids at row shifts 0 and 7, f16 and f32 inputs, embedding, batch-invariant rows, per-expert grids and grid refusals [ok]\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
