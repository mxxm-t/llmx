#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
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
// Feeding these activations to the CPU separates matrix arithmetic from input quantization (docs/TESTS.md).
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

// Shared wide-only K23 checks; the tree-overflow fixture must not run an explicit F32 backend dot.
inline size_t check_k23_wide_repair(backend::Backend& b, uint32_t type, const std::vector<uint8_t>& packed,
                                   const std::vector<float>& weights, const std::vector<float>& x,
                                   bool expect_repair, bool expect_tree_overflow = false) {
    if (weights.size() != x.size() || x.empty() || x.size() % 256) throw std::runtime_error("K23 wide fixture shape");
    const size_t nin = x.size();
    const auto reconstructed = row_activations(x);
    float peak = 0;
    for (size_t i = 0; i < 32; ++i) peak = std::max(peak, std::fabs(x[i]));
    const float scale = peak / 32767.0f;
    const bool flagged = scale != 0 && (scale < std::ldexp(1.0f, -80) || scale > std::ldexp(1.0f, 80));
    if (flagged != expect_repair) throw std::runtime_error("K23 wide fixture does not establish its scale-repair predicate");
    if (expect_tree_overflow) {
        if (flagged || nin != 65536 || scale != std::ldexp(1.0f, 79)) throw std::runtime_error("K23 tree repair fixture range");
        for (size_t i = 0; i < nin; ++i)
            if (weights[i] != weights[0] || reconstructed[i] != ((i / 32) % 2 ? -peak : peak))
                throw std::runtime_error("K23 tree repair fixture is not the independently reconstructed constant product");
        // Each lane takes every 64th 32-value block. Two ordered 16-value products make its block contribution.
        const float half = weights[0] * float(32767 * 16), block = scale * (half + half);
        std::vector<float> lane(64, 0.0f);
        for (size_t i = 0; i < nin / 32; ++i) lane[i % 64] += i % 2 ? -block : block;
        for (float v : lane) if (!std::isfinite(v)) throw std::runtime_error("K23 tree fixture has a nonfinite lane sum");
        const float lane_peak = std::fabs(lane[0]);
        for (size_t off = 32; off; off >>= 1) {
            const auto previous = lane;
            for (size_t i = 0; i < lane.size(); ++i) lane[i] = previous[i] + previous[i ^ off];
        }
        float serial = 0, serial_peak = 0;
        for (size_t i = 0; i < nin; ++i) {
            serial = std::fma(weights[i], reconstructed[i], serial);
            if (!std::isfinite(serial)) throw std::runtime_error("K23 tree fixture serial repair is not finite");
            serial_peak = std::max(serial_peak, std::fabs(serial));
        }
        if (std::isfinite(lane[0]) || !(serial_peak < std::numeric_limits<float>::max() / 64))
            throw std::runtime_error("K23 tree fixture lacks overflow or finite serial margin");
        std::cout << "K23 tree repair host proof type " << type << " scale " << scale << " lane " << lane_peak
                  << " tree nonfinite, serial peak " << serial_peak << " final " << serial << '\n';
    }
    // Five distinct rows: exact power-of-two scales, with two signs at the first two magnitudes.
    // Use upward factors for tiny scales, downward factors near the finite-half maximum.
    constexpr size_t rows = 5;
    const size_t blocks = nin / 256, bytes = packed.size() / blocks, scales = type == quant::GGML_TYPE_Q2_K ? 2 : 1;
    if (packed.size() != blocks * (type == quant::GGML_TYPE_Q2_K ? 84 : 110)) throw std::runtime_error("K23 wide packed row shape");
    bool down = false;
    for (size_t j = 0; j < blocks; ++j) for (size_t k = 0; k < scales; ++k)
        down = down || (packed[j * bytes + bytes - 2 * scales + 2 * k + 1] & 0x7c) >= 0x74;
    std::vector<uint8_t> repeated(rows * packed.size());
    std::vector<float> row_weights(rows * nin), decoded(rows * nin);
    for (size_t r = 0; r < rows; ++r) {
        std::copy(packed.begin(), packed.end(), repeated.begin() + r * packed.size());
        const int shifts = int(r / 2), direction = down ? -1 : 1;
        const float factor = std::ldexp(r % 2 ? -1.0f : 1.0f, direction * shifts);
        for (size_t i = 0; i < nin; ++i) row_weights[r * nin + i] = weights[i] * factor;
        for (size_t j = 0; j < blocks; ++j) for (size_t k = 0; k < scales; ++k) {
            uint8_t* h = repeated.data() + r * packed.size() + j * bytes + bytes - 2 * scales + 2 * k;
            unsigned word = unsigned(h[0]) | (unsigned(h[1]) << 8), magnitude = word & 0x7fff;
            for (int step = 0; step < shifts && magnitude; ++step) {
                if (down) {
                    if (magnitude < 0x800 && (magnitude & 1)) throw std::runtime_error("K23 row scale would round a half");
                    magnitude = magnitude < 0x800 ? magnitude / 2 : magnitude - 0x400;
                } else magnitude = magnitude < 0x400 ? magnitude * 2 : magnitude + 0x400;
            }
            if (magnitude >= 0x7c00) throw std::runtime_error("K23 row scale would overflow a half");
            word = ((word ^ (r % 2 ? 0x8000 : 0)) & 0x8000) | magnitude;
            h[0] = uint8_t(word); h[1] = uint8_t(word >> 8);
        }
    }
    quant::Registry::instance().get(type)->dequantize(repeated.data(), decoded.data(), rows * blocks);
    if (decoded != row_weights) throw std::runtime_error("K23 distinct packed rows differ from independent power-of-two weights");
    for (size_t r = 0; r < rows; ++r) for (size_t prev = 0; prev < r; ++prev)
        if (std::equal(row_weights.begin() + r * nin, row_weights.begin() + (r + 1) * nin, row_weights.begin() + prev * nin))
            throw std::runtime_error("K23 wide fixture weight rows are not distinct");
    size_t checked = 0;
#if LLMX_HAS_BACKEND_VULKAN
    if (!b.is_cpu() && k23_integer(b, backend::Dtype::f16)) {
        // K23 column chunks choose the narrowest available build, independently of output height.
        // Five rows leave a partial group in both the two-row (2/4/8 columns) and four-row (16) builds.
        constexpr size_t most = 17;
        constexpr float guard = 19;
        std::vector<float> inputs(most * nin);
        for (size_t c = 0; c < most; ++c)
            for (size_t i = 0; i < nin; ++i) inputs[c * nin + i] = c % 2 ? -x[i] : x[i];
        // A flagged lane selects per-row fallback repair; otherwise the tree fixture's nonfinite sum selects tree repair.
        const auto rw = b.adopt(repeated.data(), repeated.size()), rx = b.adopt(inputs.data(), inputs.size() * sizeof(float));
        const auto ry = b.alloc((most * rows + 2) * sizeof(float));
        std::vector<double> want(rows, 0), bound(rows);
        const double nu = double(nin + 6) * 0x1p-24;
        for (size_t r = 0; r < rows; ++r) {
            double magnitude = 0;
            for (size_t i = 0; i < nin; ++i) {
                const double term = double(row_weights[r * nin + i]) * reconstructed[i];
                want[r] += term; magnitude += std::fabs(term);
            }
            bound[r] = nu / (1 - nu) * magnitude + double(nin) * std::numeric_limits<float>::denorm_min();
        }
        auto product = [&](size_t col0, size_t cols) {
            std::vector<float> result(most * rows + 2, guard);
            b.write(*ry, 0, result.data(), result.size() * sizeof(float));
            take_matrix_paths(b);
            backend::vulkan_kernel_times(b);
            const backend::RowRun run{cols, 1};
            b.matmul({type, {rw.get(), 0}}, {rx.get(), col0 * nin}, {ry.get(), 1}, nin, rows, cols, {&run, 1}, backend::Dtype::f16);
            b.read(*ry, 0, result.data(), result.size() * sizeof(float));
            if (take_matrix_paths(b) != std::vector<std::string>{"block-int16"}) throw std::runtime_error("K23 wide cancellation arithmetic witness");
            if (result.front() != guard || !std::all_of(result.begin() + 1 + cols * rows, result.end(), [guard](float v) { return v == guard; }))
                throw std::runtime_error("K23 wide cancellation crossed output guards");
            for (size_t c = 0; c < cols; ++c) for (size_t r = 0; r < rows; ++r) {
                const float got = result[1 + c * rows + r];
                const double expected = (col0 + c) % 2 ? -want[r] : want[r];
                if (!std::isfinite(got) || std::fabs(double(got) - expected) > bound[r])
                    throw std::runtime_error("K23 wide cancellation differs from independent reconstructed-input double bound");
                ++checked;
            }
            return result;
        };
        const auto positive = product(0, 1), negative = product(1, 1);
        const std::string kernel = type == quant::GGML_TYPE_Q2_K ? "matmul_row_q2_dot_" : "matmul_row_q3_dot_";
        const auto profile = backend::vulkan_device_profile(b);
        std::vector<std::string> dispatched;
        std::vector<size_t> capacities{1};
        for (size_t n : {2u, 4u, 8u, 16u}) if (n <= profile.row_decode_cols) capacities.push_back(n);
        if (capacities.size() == 1) capacities.push_back(8); // The one-row wide fallback, as in column_builds.
        for (size_t cols : {2u, 3u, 4u, 5u, 8u, 9u, 16u, 17u}) {
            const auto got = product(0, cols);
            for (size_t c = 0; c < cols; ++c) {
                const auto& alone = c % 2 ? negative : positive;
                if (std::memcmp(got.data() + 1 + c * rows, alone.data() + 1, rows * sizeof(float)))
                    throw std::runtime_error("K23 repaired wide column differs from its one-column baseline");
            }
            // Match the actual whole widest chunk plus the narrowest remainder build, including 17=>16+1.
            std::vector<std::string> expected;
            const size_t widest = capacities.back(), whole = cols > widest ? cols / widest : 0, rest = cols - whole * widest;
            if (whole) expected.push_back(kernel + std::to_string(widest) + "col");
            if (rest) {
                const auto n = std::lower_bound(capacities.begin(), capacities.end(), rest);
                const std::string name = kernel + std::to_string(*n) + "col";
                if (std::find(expected.begin(), expected.end(), name) == expected.end()) expected.push_back(name);
            }
            const auto times = backend::vulkan_kernel_times(b);
            for (const auto& entry : times) if (entry.first.rfind(kernel, 0) == 0) {
                const auto found = std::find(expected.begin(), expected.end(), entry.first);
                if (found == expected.end()) throw std::runtime_error("K23 wide repair dispatched an unexpected column build");
                expected.erase(found);
                if (std::find(dispatched.begin(), dispatched.end(), entry.first) == dispatched.end()) dispatched.push_back(entry.first);
                std::cout << "K23 wide repair type " << type << " cols " << cols << " rows " << rows
                          << " scale " << scale << " flagged " << flagged << " timed " << entry.first << '\n';
            }
            if (times.empty())
                std::cout << "K23 wide repair type " << type << " cols " << cols << ": timing unavailable, numerical checks only\n";
            else if (!expected.empty()) throw std::runtime_error("K23 wide repair lacks its timed integer row dispatch");
        }
        // Query representations once. Timed names establish dispatch; a returned header adds actual rows/tree evidence.
        const auto representations = backend::vulkan_kernel_representations(b);
        for (const auto& name : dispatched) {
            bool metadata = false;
            const size_t cols = std::stoul(name.substr(kernel.size()));
            for (const auto& rep : representations)
                if (rep.first == name && rep.second.rfind("; row_build ", 0) == 0) {
                    const size_t build_rows = cols == 16 ? 4 : 2;
                    if (rep.second.rfind("; row_build cols=" + std::to_string(cols) + " rows=" + std::to_string(build_rows) + " ", 0) != 0)
                        throw std::runtime_error("K23 wide repair row-build metadata differs");
                    metadata = true;
                    std::cout << "K23 wide cancellation " << name << ' ' << rep.second.substr(0, rep.second.find('\n')) << '\n';
                }
            if (!metadata) std::cout << "K23 wide cancellation " << name << ": row-build metadata unavailable\n";
        }
    }
#else
    (void)b; (void)type; (void)packed;
#endif
    return checked;
}

// Opposite blocks cancel before a final small residual; the existing gamma_n magnitude contract still applies to every finite intermediate.
inline size_t check_k23_cancellation(backend::Backend& b, uint32_t type, const std::vector<uint8_t>& packed,
                                     const std::vector<float>& weights, const std::vector<float>& x, bool expect_repair = false) {
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
        b.matmul({type, {w.get(), 0}}, {input.get(), 0}, {output.get(), 1}, nin, 1, 1, {&row, 1}, dtype);
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
    return checked + check_k23_wide_repair(b, type, packed, weights, x, expect_repair);
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
                        case 0: device.matmul({type, {w.get(), 0}}, {in.get(), 0}, {out.get(), 1}, width, rows, cols, runs); break;
                        case 1: device.matmul_logits({type, {w.get(), 0}}, {in.get(), 0}, {out.get(), 1}, width, rows, cols, runs); break;
                        case 2: device.matmul_group({{{type, {w.get(), 0}}, {out.get(), 1}, rows}, {{type, {w.get(), 0}}, {out.get(), count + 1}, rows}},
                                                 {in.get(), 0}, width, cols, runs); break;
                        case 3: device.matmul_add({type, {w.get(), 0}}, {in.get(), 0}, {out.get(), 1}, width, rows, cols, runs); break;
                        case 4: device.matmul_experts({{{type, {w.get(), 0}}, {out.get(), 1}, rows}}, {in.get(), 0}, width, cols, routing, runs); break;
                        case 5: device.matmul_experts_add({type, {w.get(), 0}}, {in.get(), 0}, {out.get(), 1}, width, rows, cols, routing, runs); break;
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
