#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "config.hpp"
#include "backends/cpu/cpu_backend.hpp"
#include "format/weight_scales.hpp"
#if LLMX_HAS_BACKEND_VULKAN
#include "backends/vulkan/vulkan_backend.hpp"
#endif

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

// A routed add rounds each pick's product before it joins the sum: -1 * a + b * b with b * b rounding to a gives 0, where a product fused into the add gives 2^-24.
void unfused(backend::CpuBackend& cpu) {
    const float a = 1.0f + std::ldexp(1.0f, -11), b = 1.0f + std::ldexp(1.0f, -12);
    std::vector<float> experts(2 * width, 0.0f), input(width, 0.0f);
    experts[0] = a;
    experts[width] = b;
    input[0] = 1.0f;
    auto w = cpu.adopt(experts.data(), experts.size() * sizeof(float));
    const backend::WeightView weight{quant::GGML_TYPE_F32, {w.get(), 0}};
    const uint32_t ids[] = {0, 1};
    const float gains[] = {-1.0f, b};
    auto ib = cpu.adopt(ids, sizeof ids), gb = cpu.adopt(gains, sizeof gains);
    const backend::Backend::Routing routing{{ib.get(), 0}, {gb.get(), 0}, 2, 2};
    std::vector<float> both(2 * width, 0.0f);
    both[0] = both[width] = 1.0f;
    auto x = cpu.adopt(both.data(), both.size() * sizeof(float));
    auto y = cpu.alloc(3 * sizeof(float), backend::Memory::host_visible);
    for (size_t extent : {1u, 2u}) {
        const backend::RowRun run{1, extent};
        reset(cpu, *y, 3, 0.0f);
        cpu.matmul_experts_add(weight, {x.get(), 0}, {y.get(), 1}, width, 1, 1, routing, {&run, 1}, backend::Dtype::f32);
        const auto out = read(cpu, *y, 3);
        require(out[0] == 0.0f && out[2] == 0.0f, "routed add stays in its row");
        require(bits(out[1]) == 0, "routed add rounds each product before the sum");
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

#if LLMX_HAS_BACKEND_VULKAN
// F8_E4M3 on the Vulkan device against the CPU on the same codes, grids and inputs: one-hot products equal bit for bit, dense ones within the F32 order bound.
struct Fp8Weight { size_t nin, rows, experts; unsigned shift; std::vector<uint8_t> codes; std::vector<float> grid; };
Fp8Weight fp8_weight(size_t nin, size_t rows, size_t experts, unsigned shift, unsigned seed) {
    Fp8Weight w{nin, rows, experts, shift, std::vector<uint8_t>(experts * rows * nin), std::vector<float>(gguf::fp8_grid_count(nin, rows, shift, experts))};
    for (size_t i = 0; i < w.codes.size(); ++i) {
        const uint8_t c = uint8_t(i * 37 + i / nin * 11 + i / 7 + seed * 101);
        w.codes[i] = (c & 0x7F) == 0x7F ? 0x3F : c;
    }
    for (size_t i = 0; i < w.grid.size(); ++i) w.grid[i] = std::ldexp(float(int((i + seed) % 7) - 3) * 0.75f + 0.5f, -int(i % 5) - 6);
    return w;
}
// Modes: 0 matmul, 1 matmul_add onto 0.25, 2 a group of `w` and `v` (v's outputs after w's), 3 routed (two picks a column), 4 routed weighted add onto 0.25.
std::vector<float> fp8_run(backend::Backend& b, const Fp8Weight& w, const Fp8Weight* v, int mode, backend::Dtype dtype, const std::vector<float>& x, size_t cols,
                           size_t extent = 0) {
    auto wb = b.adopt(w.codes.data(), w.codes.size()), gb = b.adopt(w.grid.data(), w.grid.size() * sizeof(float));
    auto vb = b.adopt(v ? v->codes.data() : w.codes.data(), v ? v->codes.size() : 4), hb = b.adopt(v ? v->grid.data() : w.grid.data(), v ? v->grid.size() * 4 : 4);
    auto xb = b.adopt(x.data(), x.size() * sizeof(float));
    const backend::WeightView wv{quant::GGML_TYPE_F8_E4M3, {wb.get(), 0}, {gb.get(), 0}, w.grid.size(), w.shift};
    const size_t picks = mode >= 3 ? 2 : 1, out_cols = mode == 3 ? cols * picks : cols, extra = v ? cols * v->rows : 0;
    std::vector<float> y(out_cols * w.rows + extra + 2, mode == 1 || mode == 4 ? 0.25f : guard);
    y.front() = y.back() = guard;
    auto yb = b.alloc(y.size() * sizeof(float), backend::Memory::host_visible);
    b.write(*yb, 0, y.data(), y.size() * sizeof(float));
    std::vector<uint32_t> ids(cols * picks);
    std::vector<float> gains(cols * picks);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = uint32_t((i / picks + i % picks) % w.experts), gains[i] = i % picks ? 0.75f : 0.25f;
    auto ib = b.adopt(ids.data(), ids.size() * sizeof(uint32_t)), pb = b.adopt(gains.data(), gains.size() * sizeof(float));
    const backend::RowRun run{cols, extent ? extent : cols};
    const backend::Backend::Routing routing{{ib.get(), 0}, {pb.get(), 0}, picks, w.experts};
    const backend::Slice out{yb.get(), 1};
    if (mode == 0) b.matmul(wv, {xb.get(), 0}, out, w.nin, w.rows, cols, {}, dtype);
    if (mode == 1) b.matmul_add(wv, {xb.get(), 0}, out, w.nin, w.rows, cols, {}, dtype);
    if (mode == 2) {
        const backend::WeightView vv{quant::GGML_TYPE_F8_E4M3, {vb.get(), 0}, {hb.get(), 0}, v->grid.size(), v->shift};
        b.matmul_group({{wv, out, w.rows}, {vv, {yb.get(), 1 + cols * w.rows}, v->rows}}, {xb.get(), 0}, w.nin, cols, {}, dtype);
    }
    if (mode == 3) b.matmul_experts({{wv, out, w.rows}}, {xb.get(), 0}, w.nin, cols, routing, {&run, 1}, dtype);
    if (mode == 4) b.matmul_experts_add(wv, {xb.get(), 0}, out, w.nin, w.rows, cols, routing, {&run, 1}, dtype);
    b.read(*yb, 0, y.data(), y.size() * sizeof(float));
    require(y.front() == guard && y.back() == guard, "F8_E4M3 device call crossed its guards");
    return y;
}
// Bit equality, or with `size` the F32 order bound: a relative 2e-5 of each output's sum of term magnitudes.
void fp8_compare(const std::vector<float>& cpu, const std::vector<float>& dev, const std::vector<float>& size, const char* what) {
    for (size_t i = 0; i < cpu.size(); ++i) {
        const bool same = bits(cpu[i]) == bits(dev[i]);
        if (size.empty() ? !same : !(same || std::fabs(double(cpu[i]) - dev[i]) <= 2e-5 * size[i]))
            throw std::runtime_error(std::string(what) + " differs at " + std::to_string(i) + ": cpu " + std::to_string(cpu[i]) + " device " + std::to_string(dev[i]));
    }
}
std::vector<float> one_hot(size_t nin, size_t cols, float value) {
    std::vector<float> x(nin * cols, 0.0f);
    for (size_t c = 0; c < cols; ++c) x[c * nin + (c * 53 + 7) % nin] = value;
    return x;
}
std::vector<float> dense_x(size_t nin, size_t cols) {
    std::vector<float> x(nin * cols);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i) * 0.37f) * (1.0f + float(i % 13));
    return x;
}
// Each output's sum of term magnitudes for a dense product of `w` and `x`, laid out as fp8_run's outputs.
std::vector<float> fp8_sizes(const Fp8Weight& w, const std::vector<float>& x, size_t cols) {
    std::vector<float> s(cols * w.rows + 2, 0.0f);
    for (size_t c = 0; c < cols; ++c)
        for (size_t r = 0; r < w.rows; ++r) {
            double sum = 0;
            for (size_t k = 0; k < w.nin; ++k)
                sum += std::fabs(double(e4m3(w.codes[r * w.nin + k])) * w.grid[(r >> w.shift) * (w.nin / 128) + k / 128] * x[c * w.nin + k]);
            s[1 + c * w.rows + r] = float(sum * 1.01 + 1e-30);
        }
    return s;
}

void fp8_vulkan(backend::Backend& dev) {
    backend::CpuBackend cpu;
    cpu.set_threads(3);
    const bool f16_inputs = dev.dtype_path(backend::Dtype::f16).find("F8_E4M3") != std::string::npos;
    std::cout << "F8_E4M3 Vulkan: " << backend::vulkan_device_name(dev) << (f16_inputs ? ", binary16 inputs" : ", F32 inputs under f16 (no half conversion controls)") << '\n';
    // All 256 codes, both zeros and both NaN codes among them, through the embedding, which scales each by one F32 product.
    {
        Fp8Weight w{256, 3, 1, 7, std::vector<uint8_t>(3 * 256), {1.0f, -std::ldexp(1.0f, -7)}};
        for (size_t i = 0; i < w.codes.size(); ++i) w.codes[i] = uint8_t(i);
        const uint32_t ids[] = {2, 0, 1};
        std::vector<std::vector<float>> got;
        for (backend::Backend* b : {static_cast<backend::Backend*>(&cpu), &dev}) {
            auto wb = b->adopt(w.codes.data(), w.codes.size()), gb = b->adopt(w.grid.data(), w.grid.size() * sizeof(float));
            std::vector<float> y(3 * 256 + 2, guard);
            auto yb = b->alloc(y.size() * sizeof(float), backend::Memory::host_visible);
            b->write(*yb, 0, y.data(), y.size() * sizeof(float));
            b->embed({yb.get(), 1}, {quant::GGML_TYPE_F8_E4M3, {wb.get(), 0}, {gb.get(), 0}, 2, 7}, 256, 3, ids, 3);
            b->read(*yb, 0, y.data(), y.size() * sizeof(float));
            got.push_back(y);
        }
        for (size_t i = 0; i < got[0].size(); ++i)
            require(bits(got[0][i]) == bits(got[1][i]) || (std::isnan(got[0][i]) && std::isnan(got[1][i]) && std::signbit(got[0][i]) == std::signbit(got[1][i])),
                    "F8_E4M3 device embedding of every code differs from the CPU");
    }
    const backend::Dtype dtypes[] = {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::int8, backend::Dtype::bf16};
    const auto cpu_dtype = [](backend::Dtype d) { return d == backend::Dtype::int8 ? backend::Dtype::f16 : d; };
    // One-hot columns of 1 + 2^-12, which binary16 rounds to 1: rows (1 and 3 columns) and tiles (64 and 130), 200 rows so a 128-row grid block and a tile are partial.
    for (unsigned shift : {0u, 7u}) {
        const Fp8Weight w = fp8_weight(256, 200, 1, shift, shift);
        for (backend::Dtype dtype : dtypes) {
            if (!f16_inputs && (dtype == backend::Dtype::f16 || dtype == backend::Dtype::int8)) continue;
            for (size_t cols : {1u, 3u, 64u, 130u}) {
                const auto x = one_hot(256, cols, 1.0f + std::ldexp(1.0f, -12));
                for (int mode : {0, 1})
                    fp8_compare(fp8_run(cpu, w, nullptr, mode, cpu_dtype(dtype), x, cols), fp8_run(dev, w, nullptr, mode, dtype, x, cols), {},
                                "F8_E4M3 one-hot product or residual add");
            }
        }
    }
    // Builds each device reading timed, without their column-count and grouped variants, and whether any reading was timed.
    std::vector<std::string> seen;
    bool timed = false;
    const auto ran = [&]() {
        std::vector<std::string> names;
        for (const auto& entry : backend::vulkan_kernel_times(dev)) {
            std::string n = entry.first;
            for (size_t at = n.find('_'); at != std::string::npos; at = n.find('_', at + 1))
                if (at + 1 < n.size() && std::isdigit((unsigned char)n[at + 1])) { n.resize(at); break; }
            for (const char* suffix : {"_grouped", "_narrow"})
                if (n.size() > std::strlen(suffix) && n.compare(n.size() - std::strlen(suffix), std::string::npos, suffix) == 0) n.resize(n.size() - std::strlen(suffix));
            names.push_back(n);
        }
        timed |= backend::vulkan_timed_dispatches(dev) != 0;
        seen.insert(seen.end(), names.begin(), names.end());
        return names;
    };
    const auto has = [](const std::vector<std::string>& names, const std::string& name) { return std::find(names.begin(), names.end(), name) != names.end(); };
    // The tall tile's edge on this device, as the backend's tile_rows_for gives it for 4096 rows 256 wide: its fewest 64-column groups, the last one partial.
    const backend::DeviceCaps caps = backend::vulkan_device_caps(dev);
    const backend::DeviceProfile profile = backend::vulkan_device_profile(dev);
    size_t tall_groups = 1;
    while (backend::tile_rows_for(caps, profile, 32, 64, 128, 4096, tall_groups, 256) != 128) ++tall_groups;   // kTileRowsSmall, Short and Tall
    const size_t tall_cols = 64 * (tall_groups - 1) + 1;
    // Dense products: row builds at 1 to 8 columns give each column the bits it has alone, and tiles, split and tall ones among them, stay inside the F32 order bound of the CPU.
    {
        const Fp8Weight w = fp8_weight(1024, 200, 1, 7, 3), c = fp8_weight(1024, 72, 1, 0, 4);
        const Fp8Weight deep = fp8_weight(4096, 64, 1, 7, 5), tall = fp8_weight(256, 4096, 1, 0, 6);
        const auto x = dense_x(1024, 130), dx = dense_x(4096, 64), tx = dense_x(256, tall_cols);
        const auto tsize = fp8_sizes(tall, tx, tall_cols);
        for (backend::Dtype dtype : dtypes) {
            if (!f16_inputs && (dtype == backend::Dtype::f16 || dtype == backend::Dtype::int8)) continue;
            for (size_t cols : {1u, 3u, 8u, 64u, 130u}) {
                const std::vector<float> xs(x.begin(), x.begin() + cols * 1024);
                const auto size = fp8_sizes(w, xs, cols);
                const auto d = fp8_run(dev, w, nullptr, 0, dtype, xs, cols);
                fp8_compare(fp8_run(cpu, w, nullptr, 0, cpu_dtype(dtype), xs, cols), d, size, "F8_E4M3 dense product");
                if (cols <= 8 && dtype != backend::Dtype::bf16)
                    for (size_t k = 0; k < cols; ++k) {
                        const std::vector<float> one(xs.begin() + k * 1024, xs.begin() + (k + 1) * 1024);
                        const auto alone = fp8_run(dev, w, nullptr, 0, dtype, one, 1);
                        for (size_t r = 0; r < w.rows; ++r)
                            require(bits(alone[1 + r]) == bits(d[1 + k * w.rows + r]), "F8_E4M3 device row bits depend on the column count");
                    }
                // A group of a block grid and a channel grid: each projection reads its own row shift.
                auto gsize = size;
                gsize.pop_back();
                const auto csize = fp8_sizes(c, xs, cols);
                gsize.insert(gsize.end(), csize.begin() + 1, csize.end());
                fp8_compare(fp8_run(cpu, w, &c, 2, cpu_dtype(dtype), xs, cols), fp8_run(dev, w, &c, 2, dtype, xs, cols), gsize,
                            "F8_E4M3 grouped product of a block grid and a channel grid");
            }
            // A deep, short call has few workgroups, so the tile splits the inputs into parts.
            fp8_compare(fp8_run(cpu, deep, nullptr, 0, cpu_dtype(dtype), dx, 64), fp8_run(dev, deep, nullptr, 0, dtype, dx, 64), fp8_sizes(deep, dx, 64),
                        "F8_E4M3 split tile");
            // A call at the tall tile's edge takes the dtype's tall build, and one column group fewer does not.
            const std::string tall_build = std::string("matmul_tile_fp8") + (dtype == backend::Dtype::bf16 ? "_bf16" : dtype != backend::Dtype::f32 ? "_f16" : "") + "_tall";
            ran();
            fp8_compare(fp8_run(cpu, tall, nullptr, 0, cpu_dtype(dtype), tx, tall_cols), fp8_run(dev, tall, nullptr, 0, dtype, tx, tall_cols), tsize, "F8_E4M3 tall tile");
            const auto at_edge = ran();
            if (timed) require(has(at_edge, tall_build), ("F8_E4M3 call at the tall edge did not take " + tall_build).c_str());
            if (tall_groups > 1) {
                const std::vector<float> below(tx.begin(), tx.begin() + 64 * (tall_groups - 1) * 256);
                ran();
                (void)fp8_run(dev, tall, nullptr, 0, dtype, below, 64 * (tall_groups - 1));
                for (const std::string& name : ran())
                    require(name.find("_tall") == std::string::npos, ("F8_E4M3 call below the tall edge took " + name).c_str());
            }
        }
    }
    // Routed calls over three experts with their own grids, as rows (extent 1) and as tiles (extent 130), plain and weighted onto a residual, at every dtype: routed int8 reaches the binary16 builds through its own plan.
    for (unsigned shift : {0u, 7u}) {
        const Fp8Weight w = fp8_weight(256, 200, 3, shift, 7 + shift);
        for (backend::Dtype dtype : dtypes) {
            if (!f16_inputs && (dtype == backend::Dtype::f16 || dtype == backend::Dtype::int8)) continue;
            for (size_t extent : {1u, 130u}) {
                const auto x = one_hot(256, 130, 1.0f), ax = one_hot(256, 260, 1.0f);
                fp8_compare(fp8_run(cpu, w, nullptr, 3, cpu_dtype(dtype), x, 130, extent), fp8_run(dev, w, nullptr, 3, dtype, x, 130, extent), {}, "F8_E4M3 routed product");
                fp8_compare(fp8_run(cpu, w, nullptr, 4, cpu_dtype(dtype), ax, 130, extent), fp8_run(dev, w, nullptr, 4, dtype, ax, 130, extent), {},
                            "F8_E4M3 routed weighted add");
            }
        }
    }
    // A grid of another count is refused.
    {
        Fp8Weight w = fp8_weight(256, 200, 1, 7, 1);
        w.grid.pop_back();
        bool refused = false;
        try { fp8_run(dev, w, nullptr, 0, backend::Dtype::f32, one_hot(256, 1, 1.0f), 1); }
        catch (const std::runtime_error&) { refused = true; }
        require(refused, "F8_E4M3 device accepted a malformed grid");
    }
    // Every build this device selects ran: the binary16-input ones only where the device takes binary16 inputs.
    ran();
    std::vector<std::string> builds = {"matmul_row_fp8", "matmul_tile_fp8", "matmul_tile_fp8_tall", "matmul_tile_fp8_bf16", "matmul_tile_fp8_bf16_tall"};
    if (f16_inputs) builds.insert(builds.end(), {"matmul_row_fp8_f16", "matmul_tile_fp8_f16", "matmul_tile_fp8_f16_tall"});
    std::string missing;
    for (const std::string& b : builds)
        if (!has(seen, b)) missing += " " + b;
    if (timed) require(missing.empty(), ("F8_E4M3 checks did not witness" + missing).c_str());
    std::cout << "F8_E4M3 Vulkan: every code through the embedding, one-hot and dense products, residual adds, grouped grids of both shifts, split and tall tiles, routed experts"
              << " and a grid refusal against the CPU at f32, f16, int8 and bf16 [" << (timed ? "ok, kernels witnessed]\n" : "ok, timestamps unavailable]\n");
}
#endif
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--vulkan") {
#if LLMX_HAS_BACKEND_VULKAN
        try { auto device = backend::make_vulkan_backend(0, true); fp8_vulkan(*device); return 0; }
        catch (const backend::VulkanUnavailable& e) { std::cout << "F8_E4M3 Vulkan unavailable: " << e.what() << '\n'; return 77; }
        catch (const std::exception& e) { std::cerr << "F8_E4M3 Vulkan: " << e.what() << '\n'; return 1; }
#else
        std::cout << "F8_E4M3 Vulkan backend not built\n"; return 77;
#endif
    }
    if (argc != 1) { std::cerr << "usage: llmx-weight-scales-test [--vulkan]\n"; return 1; }
    try {
        for (int threads : {1, 3}) { backend::CpuBackend cpu; cpu.set_threads(threads); dense(cpu); routed(cpu); unfused(cpu); refusals(cpu); fp8(cpu); }
        std::cout << "weight scales: signed/zero/range dense+embedding, per-expert products/adds, unfused routed adds and pre-write refusals; F8_E4M3 grids at row shifts 0 and 7, f16 and f32 inputs, embedding, batch-invariant rows, per-expert grids and grid refusals [ok]\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
