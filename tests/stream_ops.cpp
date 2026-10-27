// Qwen4Exp stream, residual and gate CPU operations against independent double references.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backends/devices.hpp"
#include "core/fp16.hpp"
#include "quant/quant.hpp"

namespace {
using backend::BufferPtr;
using backend::Backend;
using backend::CSlice;
backend::BackendPtr active_backend;

constexpr double U = 0x1p-24;
const int thread_counts[] = {1, 2, 3, 5, 8, 16};

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

void refuses(const std::string& message, const std::function<void()>& fn) {
    try { fn(); }
    catch (const std::runtime_error&) { return; }
    throw std::runtime_error(message);
}

BufferPtr zeros(Backend& cpu, size_t floats) {
    return cpu.alloc(std::max<size_t>(floats, 1) * sizeof(float), backend::Memory::device);
}

BufferPtr upload(Backend& cpu, const std::vector<float>& values) {
    BufferPtr b = zeros(cpu, values.size());
    if (!values.empty()) cpu.write(*b, 0, values.data(), values.size() * sizeof(float));
    return b;
}

std::vector<float> download(Backend& cpu, const backend::Buffer& b, size_t count, size_t offset = 0) {
    std::vector<float> values(count);
    if (count) cpu.read(b, offset * sizeof(float), values.data(), count * sizeof(float));
    return values;
}

std::vector<float> random_values(std::mt19937& rng, size_t count, float low, float high) {
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<float> values(count);
    for (float& value : values) value = dist(rng);
    return values;
}

BufferPtr upload_bytes(Backend& device, const std::vector<uint8_t>& values) {
    BufferPtr buffer = device.alloc(std::max<size_t>(values.size(), 1), backend::Memory::device);
    if (!values.empty()) device.write(*buffer, 0, values.data(), values.size());
    return buffer;
}

std::vector<float> row_twin(const std::vector<float>& values) {
    std::vector<float> rounded(values.size());
    for (size_t base = 0; base < values.size(); base += 32) {
        float amax = 0.0f;
        for (size_t j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(values[base + j]));
        const float d = amax / 32767.0f;
        const float id = amax > 0.0f ? 32767.0f / amax : 0.0f;
        for (size_t j = 0; j < 32; ++j) {
            const float scaled = values[base + j] * id;
            int q = static_cast<int>(std::copysign(std::floor(std::fabs(scaled) + 0.5f), scaled));
            q = std::max(-32767, std::min(32767, q));
            rounded[base + j] = float(q) * d;
        }
    }
    return rounded;
}

struct TwinRounding {
    float block_amax = 0.0f;
    float scaled = 0.0f;
    float step = 0.0f;
    double margin = 0.0;
    double guard = 0.0;
    int code = 0;
};

TwinRounding twin_rounding(const std::vector<float>& values, size_t index) {
    const size_t block = index & ~size_t(31);
    TwinRounding info;
    for (size_t j = 0; j < 32; ++j)
        info.block_amax = std::max(info.block_amax, std::fabs(values[block + j]));
    info.step = info.block_amax / 32767.0f;
    const float inverse = info.block_amax > 0.0f ? 32767.0f / info.block_amax : 0.0f;
    info.scaled = values[index] * inverse;
    const float magnitude = std::fabs(info.scaled);
    const float fraction = magnitude - std::floor(magnitude);
    info.margin = std::fabs(double(fraction) - 0.5);
    // Eight unit-roundoff steps cover the host/device divide and multiply; two more cover the half-away boundary addition.
    info.guard = 8.0 * U * magnitude + 2.0 * U;
    info.code = std::max(-32767, std::min(32767,
        int(std::copysign(std::floor(magnitude + 0.5f), info.scaled))));
    return info;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

void close_to(float got, double expected, double bound, const std::string& label) {
    if (!(std::fabs(double(got) - expected) <= bound)) {
        char text[256];
        std::snprintf(text, sizeof(text), "%s: got %.9g, expected %.9g, error %.3g > %.3g",
                      label.c_str(), got, expected, std::fabs(double(got) - expected), bound);
        throw std::runtime_error(text);
    }
}

void check_group_rms_norm(std::mt19937& rng) {
    constexpr size_t rows = 7, groups = 3, dim = 13;
    constexpr size_t width = groups * dim, count = rows * width;
    constexpr float eps = 1e-6f;
    const auto x = random_values(rng, count, -1.4f, 1.4f);
    const auto w = random_values(rng, width, 0.6f, 1.4f);
    Backend& cpu = *active_backend;
    auto xb = upload(cpu, x), wb = upload(cpu, w);
    auto guarded = upload(cpu, std::vector<float>(count + 2, 91.0f));
    std::vector<float> first;
    for (int threads : thread_counts) {
        if (!cpu.is_cpu() && threads != 1) break;
        cpu.set_threads(threads);
        cpu.group_rms_norm({guarded.get(), 1}, {xb.get(), 0}, {wb.get(), 0}, rows, groups, dim, eps);
        const auto y = download(cpu, *guarded, count + 2);
        require(y.front() == 91.0f && y.back() == 91.0f, "group RMSNorm changed an output guard");
        std::vector<float> body(y.begin() + 1, y.end() - 1);
        if (first.empty()) first = body;
        require(same_bits(body, first), "group RMSNorm changed with thread count");
    }
    for (size_t r = 0; r < rows; ++r)
        for (size_t g = 0; g < groups; ++g) {
            double square = 0;
            for (size_t d = 0; d < dim; ++d) {
                const double value = x[r * width + g * dim + d];
                square += value * value;
            }
            const double inv = 1.0 / std::sqrt(square / dim + eps);
            for (size_t d = 0; d < dim; ++d) {
                const size_t i = r * width + g * dim + d;
                const double expected = double(x[i]) * inv * w[g * dim + d];
                close_to(first[i], expected, (dim + 12) * U * std::fabs(expected) + 1e-30, "group RMSNorm");
            }
        }
    auto inplace = upload(cpu, x);
    cpu.group_rms_norm({inplace.get(), 0}, {inplace.get(), 0}, {wb.get(), 0}, rows, groups, dim, eps);
    require(same_bits(download(cpu, *inplace, count), first), "group RMSNorm exact in-place differs");

    auto short_src = upload(cpu, x), protected_dst = upload(cpu, std::vector<float>(count, 37.0f));
    refuses("group RMSNorm accepted a short source span", [&] {
        cpu.group_rms_norm({protected_dst.get(), 0}, {short_src.get(), 1}, {wb.get(), 0}, rows, groups, dim, eps);
    });
    const auto protected_values = download(cpu, *protected_dst, count);
    require(std::all_of(protected_values.begin(), protected_values.end(),
                        [](float value) { return value == 37.0f; }), "group RMSNorm wrote before refusing a short source");
    auto overflow_guard = upload(cpu, {19.0f});
    refuses("group RMSNorm accepted overflowing dimensions", [&] {
        cpu.group_rms_norm({overflow_guard.get(), 0}, {xb.get(), 0}, {wb.get(), 0},
                           std::numeric_limits<size_t>::max(), 2, 1, eps);
    });
    require(download(cpu, *overflow_guard, 1)[0] == 19.0f,
            "group RMSNorm wrote before refusing overflowing dimensions");
    auto overlap = upload(cpu, std::vector<float>(count + 1, 2.0f));
    refuses("group RMSNorm accepted a partial output overlap", [&] {
        cpu.group_rms_norm({overlap.get(), 1}, {overlap.get(), 0}, {wb.get(), 0}, rows, groups, dim, eps);
    });
}

void check_silu_scaled(std::mt19937& rng) {
    constexpr size_t rows = 19, width = 35, count = rows * width;
    constexpr float scale = 0.25f;
    const auto x = random_values(rng, count, -12.0f, 12.0f);
    Backend& cpu = *active_backend;
    auto xb = upload(cpu, x), yb = zeros(cpu, count);
    std::vector<float> first;
    for (int threads : thread_counts) {
        if (!cpu.is_cpu() && threads != 1) break;
        cpu.set_threads(threads);
        cpu.silu_scaled({yb.get(), 0}, {xb.get(), 0}, rows, width, scale);
        const auto y = download(cpu, *yb, count);
        if (first.empty()) first = y;
        require(same_bits(y, first), "scaled SiLU changed with thread count");
    }
    for (size_t i = 0; i < count; ++i) {
        const double z = double(x[i]) * scale;
        const double expected = z / (1.0 + std::exp(-z));
        close_to(first[i], expected, 8 * U * std::fabs(expected) + 1e-30, "scaled SiLU");
    }
    auto inplace = upload(cpu, x);
    cpu.silu_scaled({inplace.get(), 0}, {inplace.get(), 0}, rows, width, scale);
    require(same_bits(download(cpu, *inplace, count), first), "scaled SiLU exact in-place differs");
    auto overlap = upload(cpu, std::vector<float>(count + 1, 3.0f));
    refuses("scaled SiLU accepted a partial output overlap", [&] {
        cpu.silu_scaled({overlap.get(), 1}, {overlap.get(), 0}, rows, width, scale);
    });
    auto short_src = upload(cpu, x), protected_dst = upload(cpu, std::vector<float>(count, 29.0f));
    refuses("scaled SiLU accepted a short source span", [&] {
        cpu.silu_scaled({protected_dst.get(), 0}, {short_src.get(), 1}, rows, width, scale);
    });
    const auto protected_values = download(cpu, *protected_dst, count);
    require(std::all_of(protected_values.begin(), protected_values.end(),
                        [](float value) { return value == 29.0f; }), "scaled SiLU wrote before refusing a short source");
}

void check_mix_streams(std::mt19937& rng) {
    constexpr size_t rows = 9, streams = 4, dim = 13;
    constexpr size_t width = streams * dim, src_count = rows * width, out_count = rows * dim;
    const auto x = random_values(rng, src_count, -1.2f, 1.2f);
    const auto gates = random_values(rng, src_count, -7.0f, 7.0f);
    Backend& cpu = *active_backend;
    auto xb = upload(cpu, x), gb = upload(cpu, gates), yb = zeros(cpu, out_count);
    std::vector<float> first;
    for (int threads : thread_counts) {
        if (!cpu.is_cpu() && threads != 1) break;
        cpu.set_threads(threads);
        cpu.mix_streams({yb.get(), 0}, {xb.get(), 0}, {gb.get(), 0}, rows, streams, dim);
        const auto y = download(cpu, *yb, out_count);
        if (first.empty()) first = y;
        require(same_bits(y, first), "stream mix changed with thread count");
    }
    for (size_t r = 0; r < rows; ++r)
        for (size_t d = 0; d < dim; ++d) {
            double sum = 0, magnitude = 0;
            for (size_t s = 0; s < streams; ++s) {
                const size_t i = r * width + s * dim + d;
                const double term = 1.0 / (1.0 + std::exp(-double(gates[i]))) * x[i];
                sum += term;
                magnitude += std::fabs(term);
            }
            close_to(first[r * dim + d], sum / streams, (streams + 8) * U * magnitude / streams + 1e-30, "stream mix");
        }
    auto overlap = upload(cpu, std::vector<float>(src_count + 1, 5.0f));
    refuses("stream mix accepted output/input overlap", [&] {
        cpu.mix_streams({overlap.get(), 0}, {overlap.get(), 0}, {gb.get(), 0}, rows, streams, dim);
    });
    auto short_gate = upload(cpu, std::vector<float>(src_count - 1, 0.0f));
    auto protected_dst = upload(cpu, std::vector<float>(out_count, 31.0f));
    refuses("stream mix accepted a short gate span", [&] {
        cpu.mix_streams({protected_dst.get(), 0}, {xb.get(), 0}, {short_gate.get(), 0}, rows, streams, dim);
    });
    const auto protected_values = download(cpu, *protected_dst, out_count);
    require(std::all_of(protected_values.begin(), protected_values.end(),
                        [](float value) { return value == 31.0f; }), "stream mix wrote before refusing a short gate");
}

void check_inject_streams(std::mt19937& rng) {
    constexpr size_t rows = 11, streams = 3, dim = 13;
    constexpr size_t width = streams * dim, count = rows * width;
    const auto initial = random_values(rng, count, -1.0f, 1.0f);
    const auto block = random_values(rng, rows * dim, -0.6f, 0.6f);
    const auto gate = random_values(rng, rows * streams, -7.0f, 7.0f);
    Backend& cpu = *active_backend;
    auto bb = upload(cpu, block), gb = upload(cpu, gate);
    auto guarded = upload(cpu, std::vector<float>(count + 2, 43.0f));
    cpu.write(*guarded, sizeof(float), initial.data(), count * sizeof(float));
    std::vector<float> first;
    for (int threads : thread_counts) {
        if (!cpu.is_cpu() && threads != 1) break;
        cpu.set_threads(threads);
        cpu.write(*guarded, sizeof(float), initial.data(), count * sizeof(float));
        cpu.inject_streams({guarded.get(), 1}, {bb.get(), 0}, {gb.get(), 0}, rows, streams, dim);
        const auto y = download(cpu, *guarded, count + 2);
        require(y.front() == 43.0f && y.back() == 43.0f, "stream injection changed an output guard");
        std::vector<float> body(y.begin() + 1, y.end() - 1);
        if (first.empty()) first = body;
        require(same_bits(body, first), "stream injection changed with thread count");
    }
    for (size_t r = 0; r < rows; ++r)
        for (size_t s = 0; s < streams; ++s)
            for (size_t d = 0; d < dim; ++d) {
                const size_t i = r * width + s * dim + d;
                const double weight = 2.0 / (1.0 + std::exp(-double(gate[r * streams + s]) / streams));
                const double product = double(block[r * dim + d]) * weight;
                close_to(first[i], double(initial[i]) + product,
                         12 * U * (std::fabs(double(initial[i])) + std::fabs(product)) + 1e-30, "stream injection");
            }
    auto alias = upload(cpu, initial);
    cpu.inject_streams({alias.get(), 0}, {bb.get(), 0}, {gb.get(), 0}, rows, streams, dim);
    require(same_bits(download(cpu, *alias, count), first), "stream injection residual in-place differs");
    auto bad = upload(cpu, std::vector<float>(count, 27.0f));
    refuses("stream injection accepted block/output overlap", [&] {
        cpu.inject_streams({bad.get(), 0}, {bad.get(), 0}, {gb.get(), 0}, rows, streams, dim);
    });
    const auto bad_values = download(cpu, *bad, count);
    require(std::all_of(bad_values.begin(), bad_values.end(),
                        [](float value) { return value == 27.0f; }), "stream injection wrote before refusing overlap");
    auto short_gate = upload(cpu, std::vector<float>(rows * streams - 1, 0.0f));
    auto short_guard = upload(cpu, std::vector<float>(count, 33.0f));
    refuses("stream injection accepted a short gate span", [&] {
        cpu.inject_streams({short_guard.get(), 0}, {bb.get(), 0}, {short_gate.get(), 0}, rows, streams, dim);
    });
    const auto short_values = download(cpu, *short_guard, count);
    require(std::all_of(short_values.begin(), short_values.end(),
                        [](float value) { return value == 33.0f; }), "stream injection wrote before refusing a short gate");
}

void check_signed_sqrt_gate(std::mt19937& rng) {
    constexpr size_t rows = 8, groups = 3, dim = 13;
    constexpr size_t width = groups * dim, count = rows * width;
    auto key = random_values(rng, count, -1.0f, 1.0f);
    auto query = random_values(rng, count, -1.0f, 1.0f);
    const auto value = random_values(rng, rows * dim, -2.0f, 2.0f);
    for (size_t d = 0; d < dim; ++d) {
        query[d] = 0.0f;
        query[dim + d] = key[dim + d] * 1e-9f;
        query[2 * dim + d] = -key[2 * dim + d];
    }
    Backend& cpu = *active_backend;
    auto kb = upload(cpu, key), qb = upload(cpu, query), vb = upload(cpu, value);
    auto guarded = upload(cpu, std::vector<float>(count + 2, 53.0f));
    std::vector<float> first;
    for (int threads : thread_counts) {
        if (!cpu.is_cpu() && threads != 1) break;
        cpu.set_threads(threads);
        cpu.signed_sqrt_gate({guarded.get(), 1}, {kb.get(), 0}, {qb.get(), 0}, {vb.get(), 0}, rows, groups, dim);
        const auto y = download(cpu, *guarded, count + 2);
        require(y.front() == 53.0f && y.back() == 53.0f, "signed gate changed an output guard");
        std::vector<float> body(y.begin() + 1, y.end() - 1);
        if (first.empty()) first = body;
        require(same_bits(body, first), "signed gate changed with thread count");
    }
    for (size_t r = 0; r < rows; ++r)
        for (size_t g = 0; g < groups; ++g) {
            double dot = 0;
            for (size_t d = 0; d < dim; ++d)
                dot += double(key[r * width + g * dim + d]) * query[r * width + g * dim + d];
            const double score = dot / std::sqrt(double(dim));
            const double signed_root = (score > 0 ? 1.0 : score < 0 ? -1.0 : 0.0) * std::sqrt(std::max(std::fabs(score), 1e-6));
            const double gate = 1.0 / (1.0 + std::exp(-signed_root));
            for (size_t d = 0; d < dim; ++d) {
                const size_t i = r * width + g * dim + d;
                const double expected = double(value[r * dim + d]) * gate;
                close_to(first[i], expected, 64 * dim * U * std::max(1.0, std::fabs(expected)) + 1e-30, "signed square-root gate");
            }
        }
    auto alias = upload(cpu, key);
    refuses("signed gate accepted output/key overlap", [&] {
        cpu.signed_sqrt_gate({alias.get(), 0}, {alias.get(), 0}, {qb.get(), 0}, {vb.get(), 0}, rows, groups, dim);
    });
    require(same_bits(download(cpu, *alias, count), key), "signed gate modified aliased input before refusing");
    auto short_value = upload(cpu, std::vector<float>(rows * dim - 1, 0.0f));
    auto protected_dst = upload(cpu, std::vector<float>(count, 47.0f));
    refuses("signed gate accepted a short value span", [&] {
        cpu.signed_sqrt_gate({protected_dst.get(), 0}, {kb.get(), 0}, {qb.get(), 0},
                             {short_value.get(), 0}, rows, groups, dim);
    });
    const auto protected_values = download(cpu, *protected_dst, count);
    require(std::all_of(protected_values.begin(), protected_values.end(),
                        [](float x) { return x == 47.0f; }), "signed gate wrote before refusing a short value");
}

struct TwinConsumer {
    BufferPtr weights, output;
    std::vector<backend::RowRun> row_runs;
    std::vector<size_t> selected;
    size_t rows = 0, dim = 0;
    float decoded_weight = 0.0f;
    std::string label;

    void enqueue(Backend& device, CSlice activation) {
        if (!weights) return;
        // This is the producer's exact buffer and offset; callers enqueue immediately before reading it or dispatching anything else.
        // Q8_0 with F16 consumes the Vulkan xquant16 twin in its row kernel, which matches the producer's cached tag.
        device.matmul({quant::GGML_TYPE_Q8_0, {weights.get(), 0}}, activation, {output.get(), 0}, dim, 3,
                      rows, {row_runs.data(), row_runs.size()}, backend::Dtype::f16);
    }

    void check(Backend& device, const std::vector<float>& activation) {
        if (!weights) return;
        const auto rounded = row_twin(activation);
        const auto actual = download(device, *output, rows * 3);
        auto repacked_input = upload(device, activation);
        enqueue(device, {repacked_input.get(), 0});
        const auto repacked = download(device, *output, rows * 3);
        require(same_bits(actual, repacked), label + " producer twin differs from separately packed output");
        const size_t source_rows[] = {0, rows / 2, rows - 1};
        for (size_t o = 0; o < 3; ++o) {
            const size_t witness = source_rows[o] * dim + selected[o];
            require(std::fabs(double(activation[witness]) - rounded[witness]) > 1e-6,
                    label + " lost its selected 16-bit twin witness");
            for (size_t r = 0; r < rows; ++r) {
                const size_t index = r * dim + selected[o];
                const TwinRounding info = twin_rounding(activation, index);
                if (info.margin <= info.guard) {
                    char details[384];
                    std::snprintf(details, sizeof(details),
                                  "%s selected twin became rounding-ambiguous: row=%zu selected=%zu activation=%.9g block_amax=%.9g host_scaled=%.9g host_code=%d margin=%.9g guard=%.9g step=%.9g",
                                  label.c_str(), r, selected[o], activation[index], info.block_amax,
                                  info.scaled, info.code, info.margin, info.guard, info.step);
                    throw std::runtime_error(details);
                }
            }
        }
        for (size_t r = 0; r < rows; ++r)
            for (size_t o = 0; o < 3; ++o) {
                const size_t index = r * dim + selected[o];
                const double expected = double(decoded_weight) * rounded[index];
                const double bound = 8 * U * std::max(1.0, std::fabs(expected)) + 2e-6;
                // One nonzero Q8 weight per projection makes this a narrow witness of the producer's rounded twin.
                const double error = std::fabs(double(actual[r * 3 + o]) - expected);
                if (error > bound) {
                    const TwinRounding info = twin_rounding(activation, index);
                    char details[512];
                    std::snprintf(details, sizeof(details),
                                  "%s 16-bit activation-twin matmul: row=%zu selected=%zu flat=%zu activation=%.9g block_amax=%.9g host_scaled=%.9g host_code=%d margin=%.9g guard=%.9g step=%.9g got=%.9g expected=%.9g error=%.3g bound=%.3g",
                                  label.c_str(), r, selected[o], index, activation[index], info.block_amax,
                                  info.scaled, info.code, info.margin, info.guard, info.step,
                                  actual[r * 3 + o], expected, error, bound);
                    throw std::runtime_error(details);
                }
            }
    }
};

TwinConsumer prepare_twin_consumer(Backend& device, const std::vector<float>& expected_activation,
                                   size_t rows, size_t dim, const std::string& label) {
    TwinConsumer consumer;
    consumer.rows = rows;
    consumer.dim = dim;
    consumer.label = label;
    consumer.row_runs.resize(rows);
    for (size_t r = 0; r < rows; ++r) consumer.row_runs[r] = {r + 1, 1};
    if (device.is_cpu()) return consumer;
    const auto expected_twin = row_twin(expected_activation);
    constexpr size_t outputs = 3;
    const size_t source_rows[] = {0, rows / 2, rows - 1};
    consumer.selected.resize(outputs);
    for (size_t o = 0; o < outputs; ++o) {
        const size_t row = source_rows[o];
        double largest_rounding = 0.0;
        bool found = false;
        for (size_t d = 0; d < dim; ++d) {
            bool stable = true;
            for (size_t r = 0; r < rows; ++r) {
                const TwinRounding info = twin_rounding(expected_activation, r * dim + d);
                if (info.margin <= info.guard) {
                    stable = false;
                    break;
                }
            }
            const double error = std::fabs(double(expected_activation[row * dim + d]) - expected_twin[row * dim + d]);
            if (stable && error > largest_rounding) {
                largest_rounding = error;
                consumer.selected[o] = d;
                found = true;
            }
        }
        require(found && largest_rounding > 1e-6,
                label + " has no all-row-stable 16-bit twin witness above 1e-6");
    }
    std::vector<uint8_t> quantized(outputs * (dim / quant::Q8_0_BLOCK) * quant::Q8_0_TYPESIZE, 0);
    const size_t row_bytes = (dim / quant::Q8_0_BLOCK) * quant::Q8_0_TYPESIZE;
    const uint16_t scale_bits = f32_to_f16(1.0f / 127.0f);
    const float scale = f16_to_f32(scale_bits);
    consumer.decoded_weight = scale * 127.0f;
    for (size_t o = 0; o < outputs; ++o) {
        const size_t block_count = dim / 32;
        for (size_t b = 0; b < block_count; ++b) {
            uint8_t* block = quantized.data() + o * row_bytes + b * 34;
            block[0] = uint8_t(scale_bits);
            block[1] = uint8_t(scale_bits >> 8);
        }
        quantized[o * row_bytes + (consumer.selected[o] / 32) * 34 + 2 + consumer.selected[o] % 32] = 127;
    }
    consumer.weights = upload_bytes(device, quantized);
    consumer.output = zeros(device, rows * outputs);
    return consumer;
}

// A first producer pass selects unambiguous packing witnesses from its F32 values; a second queues the consumer immediately after production.
// Every F32 output is still checked against the independent mathematical oracle.
void check_twin_producers(std::mt19937& rng) {
    Backend& device = *active_backend;
    if (device.is_cpu()) {
        std::puts("Qwen4Exp activation-twin matmul: skipped on CPU (no Vulkan activation twin)");
    }
    for (size_t dim : {size_t(32), size_t(128), size_t(1056)}) {
        // tile_reads uses the largest profile cutoff, currently 96; bracket it and run a prompt-sized batch.
        for (const auto shape : {std::pair<size_t, size_t>{1, 1}, {1, 95}, {1, 96}, {96, 96}}) {
            const size_t rows = shape.first, extent = shape.second;
            const size_t count = rows * dim;
            const backend::RowRun run{rows, extent};
            const backend::RowRuns runs{&run, 1};
            const std::string suffix = " rows=" + std::to_string(rows) + " extent=" +
                                       std::to_string(extent) + " dim=" + std::to_string(dim);
            auto shared_out = zeros(device, 96 * 1056);

            const auto norm_x = random_values(rng, count, -1.4f, 1.4f);
            const auto norm_w = random_values(rng, dim, 0.6f, 1.4f);
            auto nx = upload(device, norm_x), nw = upload(device, norm_w);
            device.group_rms_norm({shared_out.get(), 0}, {nx.get(), 0}, {nw.get(), 0}, rows, 1, dim, 1e-6f, runs);
            const auto norm_witness = download(device, *shared_out, count);
            auto norm_consumer = prepare_twin_consumer(device, norm_witness, rows, dim, "group RMSNorm" + suffix);
            device.group_rms_norm({shared_out.get(), 0}, {nx.get(), 0}, {nw.get(), 0}, rows, 1, dim, 1e-6f, runs);
            norm_consumer.enqueue(device, {shared_out.get(), 0});
            auto norm_y = download(device, *shared_out, count);
            for (size_t r = 0; r < rows; ++r) {
                double square = 0;
                for (size_t d = 0; d < dim; ++d) square += double(norm_x[r * dim + d]) * norm_x[r * dim + d];
                const double inv = 1.0 / std::sqrt(square / dim + 1e-6);
                for (size_t d = 0; d < dim; ++d) {
                    const size_t i = r * dim + d;
                    const double expected = double(norm_x[i]) * inv * norm_w[d];
                    close_to(norm_y[i], expected, (dim + 12) * U * std::fabs(expected) + 1e-30, "wide group RMSNorm" + suffix);
                }
            }
            norm_consumer.check(device, norm_y);

            constexpr float silu_scale = 0.25f;
            const auto silu_x = random_values(rng, count, -12.0f, 12.0f);
            auto sx = upload(device, silu_x);
            std::vector<float> silu_reference(count);
            for (size_t i = 0; i < count; ++i) {
                const double x = double(silu_x[i]) * silu_scale;
                silu_reference[i] = float(x / (1.0 + std::exp(-x)));
            }
            device.silu_scaled({shared_out.get(), 0}, {sx.get(), 0}, rows, dim, silu_scale, runs);
            const auto silu_witness = download(device, *shared_out, count);
            auto silu_consumer = prepare_twin_consumer(device, silu_witness, rows, dim, "scaled SiLU" + suffix);
            device.silu_scaled({shared_out.get(), 0}, {sx.get(), 0}, rows, dim, silu_scale, runs);
            silu_consumer.enqueue(device, {shared_out.get(), 0});
            auto silu_y = download(device, *shared_out, count);
            for (size_t i = 0; i < count; ++i)
                close_to(silu_y[i], silu_reference[i], 8 * U * std::fabs(silu_reference[i]) + 1e-30,
                         "wide scaled SiLU" + suffix);
            silu_consumer.check(device, silu_y);

            constexpr size_t streams = 2;
            const auto mix_x = random_values(rng, rows * streams * dim, -1.2f, 1.2f);
            const auto mix_gate = random_values(rng, rows * streams * dim, -7.0f, 7.0f);
            auto mx = upload(device, mix_x), mg = upload(device, mix_gate);
            device.mix_streams({shared_out.get(), 0}, {mx.get(), 0}, {mg.get(), 0}, rows, streams, dim, runs);
            const auto mix_witness = download(device, *shared_out, count);
            auto mix_consumer = prepare_twin_consumer(device, mix_witness, rows, dim, "stream mix" + suffix);
            device.mix_streams({shared_out.get(), 0}, {mx.get(), 0}, {mg.get(), 0}, rows, streams, dim, runs);
            mix_consumer.enqueue(device, {shared_out.get(), 0});
            auto mix_y = download(device, *shared_out, count);
            for (size_t r = 0; r < rows; ++r)
                for (size_t d = 0; d < dim; ++d) {
                    double sum = 0, magnitude = 0;
                    for (size_t s = 0; s < streams; ++s) {
                        const size_t i = (r * streams + s) * dim + d;
                        const double term = mix_x[i] / (1.0 + std::exp(-double(mix_gate[i])));
                        sum += term;
                        magnitude += std::fabs(term);
                    }
                    close_to(mix_y[r * dim + d], sum / streams,
                             (streams + 8) * U * magnitude / streams + 1e-30, "wide stream mix" + suffix);
                }
            mix_consumer.check(device, mix_y);

            auto gate_x = random_values(rng, count, -1.0f, 1.0f);
            auto gate_q = random_values(rng, count, -1.0f, 1.0f);
            const auto gate_v = random_values(rng, count, -2.0f, 2.0f);
            if (rows >= 3) {
                for (size_t d = 0; d < dim; ++d) {
                    gate_q[d] = 0.0f;
                    gate_q[dim + d] = gate_x[dim + d] * 1e-9f;
                    gate_q[2 * dim + d] = -gate_x[2 * dim + d];
                }
            }
            auto gx = upload(device, gate_x), gq = upload(device, gate_q), gv = upload(device, gate_v);
            device.signed_sqrt_gate({shared_out.get(), 0}, {gx.get(), 0}, {gq.get(), 0}, {gv.get(), 0},
                                    rows, 1, dim, runs);
            const auto signed_witness = download(device, *shared_out, count);
            auto signed_consumer = prepare_twin_consumer(device, signed_witness, rows, dim, "signed gate" + suffix);
            device.signed_sqrt_gate({shared_out.get(), 0}, {gx.get(), 0}, {gq.get(), 0}, {gv.get(), 0},
                                    rows, 1, dim, runs);
            signed_consumer.enqueue(device, {shared_out.get(), 0});
            auto signed_y = download(device, *shared_out, count);
            for (size_t r = 0; r < rows; ++r) {
                double dot = 0;
                for (size_t d = 0; d < dim; ++d) dot += double(gate_x[r * dim + d]) * gate_q[r * dim + d];
                const double score = dot / std::sqrt(double(dim));
                const double sign = score > 0 ? 1.0 : score < 0 ? -1.0 : 0.0;
                const double gate = 1.0 / (1.0 + std::exp(-sign * std::sqrt(std::max(std::fabs(score), 1e-6))));
                for (size_t d = 0; d < dim; ++d) {
                    const double expected = double(gate_v[r * dim + d]) * gate;
                    close_to(signed_y[r * dim + d], expected,
                             64 * dim * U * std::max(1.0, std::fabs(expected)) + 1e-30,
                             "wide signed gate" + suffix);
                }
            }
            signed_consumer.check(device, signed_y);

            const size_t residual_count = rows * streams * dim;
            const auto residual = random_values(rng, residual_count, -0.8f, 0.8f);
            const auto block = random_values(rng, rows * dim, -0.5f, 0.5f);
            const auto injection_gate = random_values(rng, rows * streams, -6.0f, 6.0f);
            auto ib = upload(device, block), ig = upload(device, injection_gate), ir = upload(device, residual);
            device.inject_streams({ir.get(), 0}, {ib.get(), 0}, {ig.get(), 0}, rows, streams, dim);
            const auto injected = download(device, *ir, residual_count);
            for (size_t r = 0; r < rows; ++r)
                for (size_t s = 0; s < streams; ++s)
                    for (size_t d = 0; d < dim; ++d) {
                        const size_t i = (r * streams + s) * dim + d;
                        const double weight = 2.0 / (1.0 + std::exp(-double(injection_gate[r * streams + s]) / streams));
                        const double product = double(block[r * dim + d]) * weight;
                        const double expected = double(residual[i]) + product;
                        close_to(injected[i], expected,
                                 12 * U * (std::fabs(double(residual[i])) + std::fabs(product)) + 1e-30,
                                 "wide stream injection" + suffix);
                    }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string device = "cpu";
        if (argc == 3 && std::string(argv[1]) == "--device") device = argv[2];
        else if (argc != 1) throw std::runtime_error("usage: stream-ops-test [--device cpu|vulkan:N]");
        active_backend = backend::make_backend(device);
        std::mt19937 rng(3802027);
        check_group_rms_norm(rng);
        check_silu_scaled(rng);
        check_mix_streams(rng);
        check_inject_streams(rng);
        check_signed_sqrt_gate(rng);
        check_twin_producers(rng);
        std::puts("Qwen4Exp stream ops: grouped RMSNorm, scaled SiLU, stream mixing/injection and signed gate passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Qwen4Exp stream ops: FAIL: %s\n", error.what());
        return 1;
    }
}
