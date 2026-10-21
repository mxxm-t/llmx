#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "config.hpp"
#include "backends/backend.hpp"
#if LLMX_HAS_BACKEND_VULKAN
#include "backends/vulkan/vulkan_backend.hpp"
#endif
#include "quant/quant.hpp"

namespace testq {
// Q2/Q3 rows and eligible tiles share the profile's integer preference, independently of the completed arithmetic witness.
inline bool k23_integer(backend::Backend& b, backend::Dtype dtype) {
#if LLMX_HAS_BACKEND_VULKAN
    if (!b.is_cpu() && (dtype == backend::Dtype::f16 || dtype == backend::Dtype::int8))
        return backend::vulkan_device_profile(b).prefer_integer_dot;
#else
    (void)b; (void)dtype;
#endif
    return false;
}

// Reconstruct each 32-value block at 32767 levels, or 127 for int8, with half-away-from-zero rounding.
// Feeding these activations to the CPU separates matrix arithmetic from input quantization (AGENTS.md, Tests).
inline std::vector<float> row_activations(const std::vector<float>& x, int levels = 32767) {
    std::vector<float> out(x.size());
    const float top = float(levels);
    for (size_t b = 0; b + 32 <= x.size(); b += 32) {
        float amax = 0.0f;
        for (size_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[b + i]));
        const float d = amax / top, id = amax > 0.0f ? top / amax : 0.0f;
        for (size_t i = 0; i < 32; ++i) {
            const float r = x[b + i] * id;
            int q = (int)(std::copysign(std::floor(std::fabs(r) + 0.5f), r));
            q = std::max(-levels, std::min(levels, q));
            out[b + i] = (float)q * d;
        }
    }
    return out;
}

inline std::vector<std::string> take_matrix_paths(backend::Backend& device) {
    backend::MatrixPaths paths;
    device.swap_matrix_paths(paths);
    return paths.take();
}

// Opposite blocks cancel before a final small residual; the existing gamma_n magnitude contract still applies to every finite intermediate.
inline size_t check_k23_cancellation(backend::Backend& b, uint32_t type, const std::vector<uint8_t>& packed,
                                     const std::vector<float>& weights, const std::vector<float>& x) {
    if (weights.size() != x.size() || x.empty() || x.size() % 256) throw std::runtime_error("K23 cancellation fixture shape");
    const size_t nin = x.size();
    const auto reconstructed = row_activations(x);
    const auto w = b.adopt(packed.data(), packed.size()), input = b.adopt(x.data(), x.size() * sizeof(float));
    const auto output = b.alloc(3 * sizeof(float));
    const backend::RowRun row{1, 1};
    size_t checked = 0;
    for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::int8}) {
        const float guard[] = {19, 19, 19};
        b.write(*output, 0, guard, sizeof guard); take_matrix_paths(b);
        b.matmul(type, {w.get(), 0}, {input.get(), 0}, {output.get(), 1}, nin, 1, 1, {&row, 1}, dtype);
        const bool integer = k23_integer(b, dtype);
        if (take_matrix_paths(b) != std::vector<std::string>{integer ? "block-int16" : "f32"})
            throw std::runtime_error("K23 cancellation arithmetic witness");
        float got[3]; b.read(*output, 0, got, sizeof got);
        double want = 0, magnitude = 0;
        for (size_t i = 0; i < nin; ++i) {
            const double term = double(weights[i]) * (integer ? reconstructed : x)[i];
            want += term; magnitude += std::fabs(term);
        }
        const double nu = double(nin + 6) * 0x1p-24;
        const double bound = nu / (1 - nu) * magnitude + double(nin) * std::numeric_limits<float>::denorm_min();
        if (got[0] != guard[0] || got[2] != guard[2] || !std::isfinite(got[1]) || std::fabs(double(got[1]) - want) > bound)
            throw std::runtime_error("K23 long cancellation differs from independent reconstructed-input double bound");
        ++checked;
    }
    return checked;
}

// A unit weight selects a small input beside a peak. Its error is bounded by half a 16-bit reconstruction step, independently of the encoder.
inline size_t check_matrix_precision(backend::Backend& device) {
    const auto require = [](bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); };
    take_matrix_paths(device);
    size_t checked = 0;
    for (uint32_t type : {quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K}) {
        const bool kquant = type == quant::GGML_TYPE_Q4_K || type == quant::GGML_TYPE_Q5_K || type == quant::GGML_TYPE_Q6_K;
        for (size_t width : {size_t(256), size_t(3840), size_t(4096), size_t(4352)})
        for (float selected : {1.0f / 1024.0f, 1.006111f}) {
            constexpr size_t rows = 37, cols = 3, count = rows * cols;
            const size_t row_bytes = quant::row_bytes(type, width);
            const auto* qt = quant::Registry::instance().get(type);
            const size_t block_bytes = qt->type_size, block_values = qt->block_size;
            std::vector<uint8_t> weights(row_bytes * rows, 0);
            for (size_t o = 0; o < rows; ++o) {
                uint8_t* last = weights.data() + (o + 1) * row_bytes - block_bytes;
                if (type == quant::GGML_TYPE_Q4_0) {
                    std::fill(last + 2, last + block_bytes, uint8_t(0x88));
                    last[1] = 0x3c; last[3] = 0x89;
                } else if (type == quant::GGML_TYPE_Q4_1) {
                    last[1] = 0x3c; last[5] = 1;
                } else if (type == quant::GGML_TYPE_Q5_1) {
                    last[1] = 0x3c; last[9] = 1;
                } else if (type == quant::GGML_TYPE_Q6_K) {
                    std::fill(last + 128, last + 192, uint8_t(0xaa));
                    last[1] = 1; last[192] = 1; last[209] = 0x3c;
                } else {
                    last[1] = 0x3c; last[4] = 1;
                    last[(type == quant::GGML_TYPE_Q5_K ? 48 : 16) + 1] = 1;
                }
            }
            std::vector<float> x(width * cols), y(2 * count + 2), ids(cols, 0), gains(cols, 0.5f);
            for (size_t c = 0; c < cols; ++c) {
                x[c * width + width - block_values] = 2.0f;
                x[c * width + width - block_values + 1] = (c == 1 ? -1.0f : 1.0f) * selected;
            }
            const auto w = device.adopt(weights.data(), weights.size()), in = device.adopt(x.data(), x.size() * sizeof(float)),
                       out = device.alloc(y.size() * sizeof(float)), id = device.adopt(ids.data(), ids.size() * sizeof(float)),
                       gain = device.adopt(gains.data(), gains.size() * sizeof(float));
            const backend::Backend::Routing routing{{id.get(), 0}, {gain.get(), 0}, 1, 1};
            const backend::RowRun prompt[] = {{cols, 512}}, decode[] = {{cols, 1}}, mixed[] = {{1, 1}, {cols, 512}};
            for (int threads : {1, 4}) {
                if (!device.is_cpu() && threads != 1) continue;
                device.set_threads(threads);
                for (backend::RowRuns runs : {backend::RowRuns{prompt, 1}, {decode, 1}, {mixed, 2}}) {
                    for (int op = 0; op < 6; ++op) {
                        const bool additive = op == 3 || op == 5;
                        const float initial = additive ? 0.25f : 0.0f;
                        std::fill(y.begin(), y.end(), initial);
                        y.front() = y.back() = 19.0f;
                        device.write(*out, 0, y.data(), y.size() * sizeof(float));
                        switch (op) {
                        case 0: device.matmul(type, {w.get(), 0}, {in.get(), 0}, {out.get(), 1}, width, rows, cols, runs); break;
                        case 1: device.matmul_logits(type, {w.get(), 0}, {in.get(), 0}, {out.get(), 1}, width, rows, cols, runs); break;
                        case 2: device.matmul_group({{type, {w.get(), 0}, {out.get(), 1}, rows}, {type, {w.get(), 0}, {out.get(), count + 1}, rows}},
                                                 {in.get(), 0}, width, cols, runs); break;
                        case 3: device.matmul_add(type, {w.get(), 0}, {in.get(), 0}, {out.get(), 1}, width, rows, cols, runs); break;
                        case 4: device.matmul_experts({{type, {w.get(), 0}, {out.get(), 1}, rows}}, {in.get(), 0}, width, cols, routing, runs); break;
                        case 5: device.matmul_experts_add(type, {w.get(), 0}, {in.get(), 0}, {out.get(), 1}, width, rows, cols, routing, runs); break;
                        }
                        device.read(*out, 0, y.data(), y.size() * sizeof(float));
                        const size_t values = op == 2 ? 2 * count : count;
                        const double scale = op == 5 ? 0.5 : 1.0;
                        for (size_t i = 0; i < values; ++i) {
                            const size_t column = (i % count) / rows;
                            const double small = (column == 1 ? -1.0 : 1.0) * selected;
                            const double expected = initial + scale * small;
                            require(std::isfinite(y[i + 1]) && std::fabs(double(y[i + 1]) - expected) <= scale / 32767.0 + 3e-8,
                                    "matrix activation precision: type " + std::to_string(type) + " width " + std::to_string(width) +
                                    " op " + std::to_string(op) + " got " + std::to_string(y[i + 1]) + " expected " + std::to_string(expected));
                            if (device.is_cpu() && kquant) {
                                const bool integer = op >= 4 || width >= 4096 || runs.runs == decode || (runs.runs == mixed && column == 0);
                                const double value = integer ? std::nearbyint(small * 32767.0 / 2.0) * (2.0 / 32767.0) : small;
                                require(std::fabs(double(y[i + 1]) - (initial + scale * value)) < 2e-7,
                                        "CPU K arithmetic differs from its threshold witness");
                            }
                            ++checked;
                        }
                        require(y.front() == 19.0f && y.back() == 19.0f, "matrix precision output guard changed");
                        for (size_t i = values + 1; i + 1 < y.size(); ++i) require(y[i] == initial, "matrix precision wrote past output");
                        const auto paths = take_matrix_paths(device);
                        if (device.is_cpu() && kquant) {
                            std::vector<std::string> expected_paths;
                            if (op < 4 && width < 4096 && runs.runs != decode) expected_paths.push_back("f32");
                            if (op >= 4 || width >= 4096 || runs.runs != prompt) expected_paths.push_back("block-int16");
                            require(paths == expected_paths, "CPU K activation precision witness");
                        }
                        require(!paths.empty() && std::all_of(paths.begin(), paths.end(), [](const std::string& path) {
                            return path == "f32" || path == "block-int16";
                        }), "matrix precision used a narrower or unknown activation path");
                    }
                }
            }
        }
    }
    return checked;
}

} // namespace testq
