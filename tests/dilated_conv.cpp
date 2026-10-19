// Dilated causal depthwise convolution with a residual and recurrent raw-row tail.
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
#include <vector>

#include "backends/devices.hpp"

namespace {
using backend::Backend;
using backend::BufferPtr;
using backend::StateShape;
using backend::StateStorage;
using backend::StateView;
using backend::Slice;
using backend::CSlice;
constexpr double U = 0x1p-24;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

void refuses(const std::string& message, const std::function<void()>& fn) {
    try { fn(); }
    catch (const std::runtime_error&) { return; }
    throw std::runtime_error(message);
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

BufferPtr floats_on(Backend& device, const std::vector<float>& values) {
    BufferPtr buffer = device.alloc(std::max<size_t>(values.size(), 1) * sizeof(float), backend::Memory::device);
    if (!values.empty()) device.write(*buffer, 0, values.data(), values.size() * sizeof(float));
    return buffer;
}

std::vector<float> read_floats(Backend& device, const backend::Buffer& buffer, size_t count, size_t offset = 0) {
    std::vector<float> values(count);
    if (count) device.read(buffer, offset * sizeof(float), values.data(), count * sizeof(float));
    return values;
}

std::vector<float> random_values(std::mt19937& rng, size_t count, float low, float high) {
    std::uniform_real_distribution<float> distribution(low, high);
    std::vector<float> values(count);
    for (float& value : values) value = distribution(rng);
    return values;
}

struct Sequence {
    size_t length, src, dst, rows;
};

void seed_state(Backend& device, StateStorage& state, size_t channels, size_t history, size_t suffix,
                const std::vector<Sequence>& sequences, std::mt19937& rng, std::vector<float>& initial) {
    const size_t stride = state.slot_floats(0), core = state.shape().slot_floats();
    initial = random_values(rng, state.slots() * stride, -0.4f, 0.4f);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (size_t slot = 0; slot < state.slots(); ++slot) {
        const size_t tail = slot * stride + core;
        for (size_t i = 0; i < suffix; ++i) initial[tail + history * channels + i] = float(80 + slot + i);
    }
    for (const Sequence& sequence : sequences) {
        const size_t tail = sequence.src * stride + core;
        if (!sequence.length) {
            std::fill(initial.begin() + tail, initial.begin() + tail + history * channels, nan);
            continue;
        }
        std::fill(initial.begin() + tail, initial.begin() + tail + history * channels, 0.0f);
        const size_t carried = std::min(history, sequence.length);
        const auto values = random_values(rng, carried * channels, -0.7f, 0.7f);
        const size_t first = history - carried;
        std::copy(values.begin(), values.end(), initial.begin() + tail + first * channels);
    }
    device.write(state.layer(0), 0, initial.data(), initial.size() * sizeof(float));
}

std::vector<double> reference_rows(const std::vector<float>& x, const std::vector<float>& residual,
                                 const std::vector<float>& weights, const std::vector<float>& initial,
                                 size_t channels, size_t taps, size_t dilation, size_t history,
                                 size_t state_stride, size_t core,
                                 const std::vector<Sequence>& sequences) {
    size_t total_rows = 0;
    for (const Sequence& sequence : sequences) total_rows += sequence.rows;
    std::vector<double> output(total_rows * channels);
    size_t first = 0;
    for (const Sequence& sequence : sequences) {
        for (size_t row = 0; row < sequence.rows; ++row)
            for (size_t c = 0; c < channels; ++c) {
                double acc = 0.0;
                for (size_t tap = 0; tap < taps; ++tap) {
                    const size_t lag = (taps - 1 - tap) * dilation;
                    double sample = 0.0;
                    if (row >= lag) sample = x[(first + row - lag) * channels + c];
                    else {
                        const size_t before = lag - row;
                        if (before <= sequence.length)
                            sample = initial[sequence.src * state_stride + core + (history - before) * channels + c];
                    }
                    const double product = double(weights[c * taps + tap]) * sample;
                    acc += product;
                }
                const double activated = acc / (1.0 + std::exp(-acc));
                output[(first + row) * channels + c] = double(residual[(first + row) * channels + c]) + activated;
            }
        first += sequence.rows;
    }
    return output;
}

std::vector<float> expected_tail(const std::vector<float>& x, const std::vector<float>& initial,
                                 const Sequence& sequence, size_t first, size_t channels,
                                 size_t history, size_t suffix, size_t slot_stride) {
    const size_t core = slot_stride - history * channels - suffix;
    std::vector<float> expected(history * channels + suffix);
    const size_t total = sequence.length + sequence.rows;
    for (size_t j = 0; j < history; ++j)
        for (size_t c = 0; c < channels; ++c) {
            float value = 0.0f;
            if (total >= history - j) {
                const size_t position = total - (history - j);
                if (position >= sequence.length) value = x[(first + position - sequence.length) * channels + c];
                else {
                    const size_t before = sequence.length - position;
                    value = initial[sequence.src * slot_stride + core + (history - before) * channels + c];
                }
            }
            expected[j * channels + c] = value;
        }
    const size_t tail_base = sequence.dst * slot_stride + core;
    std::copy(initial.begin() + tail_base + history * channels,
              initial.begin() + tail_base + history * channels + suffix,
              expected.begin() + history * channels);
    return expected;
}

void run_case(Backend& device, std::mt19937& rng, size_t channels, size_t taps, size_t dilation) {
    const size_t history = (taps - 1) * dilation, suffix = 3;
    const std::vector<Sequence> sequences = {{0, 0, 0, 2}, {1, 1, 2, 1}, {8, 3, 4, 3}, {5, 5, 5, 1}};
    const size_t rows = 7, count = rows * channels, tail_floats = history * channels + suffix;
    const StateShape shape = {1, 1, 2, 2};
    const size_t layer_slots = shape.slot_floats() + tail_floats;
    const auto x = random_values(rng, count, -0.8f, 0.8f);
    const auto residual = random_values(rng, count, -0.5f, 0.5f);
    const auto weights = random_values(rng, channels * taps, -0.6f, 0.6f);
    auto fixture = device.state_alloc(1, 6, shape, {tail_floats});
    std::vector<float> initial;
    seed_state(device, *fixture, channels, history, suffix, sequences, rng, initial);
    const auto expected = reference_rows(x, residual, weights, initial, channels, taps, dilation, history,
                                         layer_slots, shape.slot_floats(), sequences);
    fixture.reset();

    std::vector<float> first_output, first_state;
    const int thread_counts[] = {1, 2, 3, 5, 8, 16};
    for (int threads : thread_counts) {
        if (!device.is_cpu() && threads != 1) break;
        device.set_threads(threads);
        auto state = device.state_alloc(1, 6, shape, {tail_floats});
        device.write(state->layer(0), 0, initial.data(), initial.size() * sizeof(float));
        auto xb = floats_on(device, x), rb = floats_on(device, residual), wb = floats_on(device, weights);
        auto guarded = floats_on(device, std::vector<float>(count + 2, 91.0f));
        std::vector<StateView> views;
        for (const Sequence& sequence : sequences) {
            views.push_back({state.get(), sequence.src, sequence.dst, sequence.length, sequence.rows});
        }
        device.dilated_conv_silu_add({guarded.get(), 1}, {xb.get(), 0}, {rb.get(), 0}, {wb.get(), 0},
                                     channels, taps, dilation, 0, views.data(), views.size());
        const auto guarded_result = read_floats(device, *guarded, count + 2);
        require(guarded_result.front() == 91.0f && guarded_result.back() == 91.0f, "dilated conv changed an output guard");
        std::vector<float> output(guarded_result.begin() + 1, guarded_result.end() - 1);
        for (size_t i = 0; i < count; ++i) {
            const double error = std::fabs(double(output[i]) - expected[i]);
            require(error <= 32 * U * std::max(1.0, std::fabs(expected[i])) + 2e-6,
                    "dilated conv differs from independent double oracle at " + std::to_string(i));
        }
        const auto state_after = read_floats(device, state->layer(0), 6 * layer_slots);
        size_t row_start = 0;
        std::vector<bool> written(6, false);
        for (const Sequence& sequence : sequences) written[sequence.dst] = true;
        for (size_t slot = 0; slot < 6; ++slot) {
            if (!written[slot]) {
                require(std::equal(initial.begin() + slot * layer_slots, initial.begin() + (slot + 1) * layer_slots,
                                   state_after.begin() + slot * layer_slots),
                        "dilated conv modified an unrelated state slot");
                continue;
            }
            require(std::equal(initial.begin() + slot * layer_slots,
                               initial.begin() + slot * layer_slots + shape.slot_floats(),
                               state_after.begin() + slot * layer_slots),
                    "dilated conv modified the GDN core prefix");
        }
        for (const Sequence& sequence : sequences) {
            const auto tail = expected_tail(x, initial, sequence, row_start, channels, history, suffix, layer_slots);
            const size_t tail_at = sequence.dst * layer_slots + shape.slot_floats();
            require(std::equal(tail.begin(), tail.end(), state_after.begin() + tail_at),
                    "dilated conv did not retain its exact raw history or preserved tail suffix");
            row_start += sequence.rows;
        }
        if (first_output.empty()) {
            first_output = output;
            first_state = state_after;
        } else {
            require(same_bits(output, first_output) && same_bits(state_after, first_state),
                    "dilated conv changed with CPU thread count");
        }
        device.set_threads(1);
    }
}

void check_in_place_residual(Backend& device) {
    constexpr size_t channels = 3, taps = 3, dilation = 2, history = (taps - 1) * dilation, rows = 2;
    const size_t tail_floats = history * channels;
    const StateShape shape = {1, 1, 2, 2};
    auto state = device.state_alloc(1, 2, shape, {tail_floats});
    std::vector<float> initial(2 * (shape.slot_floats() + tail_floats), 0.0f);
    for (size_t i = 0; i < tail_floats; ++i) initial[shape.slot_floats() + i] = float(int(i) - 4) / 9.0f;
    device.write(state->layer(0), 0, initial.data(), initial.size() * sizeof(float));
    const auto x = std::vector<float>{0.2f, -0.4f, 0.7f, -0.3f, 0.8f, -0.2f};
    const auto residual = std::vector<float>{0.1f, 0.4f, -0.5f, 0.8f, -0.1f, 0.2f};
    const auto weights = std::vector<float>{0.2f, -0.4f, 0.7f, -0.1f, 0.3f, 0.8f, 0.6f, -0.2f, 0.1f};
    auto xb = floats_on(device, x), rb = floats_on(device, residual), wb = floats_on(device, weights);
    const StateView view{state.get(), 0, 1, 1, rows};
    device.dilated_conv_silu_add({rb.get(), 0}, {xb.get(), 0}, {rb.get(), 0}, {wb.get(), 0},
                                 channels, taps, dilation, 0, &view, 1);
    const auto got = read_floats(device, *rb, rows * channels);
    for (size_t r = 0; r < rows; ++r)
        for (size_t c = 0; c < channels; ++c) {
            double sum = 0.0;
            for (size_t tap = 0; tap < taps; ++tap) {
                const size_t lag = (taps - 1 - tap) * dilation;
                double sample = 0.0;
                if (r >= lag) sample = x[(r - lag) * channels + c];
                else if (lag - r <= 1)
                    sample = initial[shape.slot_floats() + (history - (lag - r)) * channels + c];
                sum += double(weights[c * taps + tap]) * sample;
            }
            const double expected = double(residual[r * channels + c]) + sum / (1.0 + std::exp(-sum));
            require(std::fabs(double(got[r * channels + c]) - expected) <= 32 * U * std::max(1.0, std::fabs(expected)) + 2e-6,
                    "exact residual in-place dilated conv differs from its oracle");
        }
}

void check_long_slices(Backend& device, size_t channels, std::mt19937& rng) {
    constexpr size_t taps = 4, dilation = 3, history = (taps - 1) * dilation;
    constexpr size_t rows = 33, initial_length = 17, suffix = 7;
    const size_t count = rows * channels, tail_floats = history * channels + suffix;
    const StateShape shape = {1, 1, 2, 2};
    const size_t stride = shape.slot_floats() + tail_floats;
    const std::vector<Sequence> sequences = {{initial_length, 0, 1, rows}};
    const auto x = random_values(rng, count, -0.9f, 0.9f);
    const auto residual = random_values(rng, count, -0.7f, 0.7f);
    const auto weights = random_values(rng, channels * taps, -0.8f, 0.8f);
    auto initial_state = device.state_alloc(1, 2, shape, {tail_floats});
    std::vector<float> initial;
    seed_state(device, *initial_state, channels, history, suffix, sequences, rng, initial);
    initial_state.reset();
    const auto expected = reference_rows(x, residual, weights, initial, channels, taps, dilation, history,
                                         stride, shape.slot_floats(), sequences);
    auto xb = floats_on(device, x), rb = floats_on(device, residual), wb = floats_on(device, weights);

    std::vector<float> whole_output, whole_state;
    const int whole_threads = device.is_cpu() ? 4 : 1;
    device.set_threads(whole_threads);
    {
        auto state = device.state_alloc(1, 2, shape, {tail_floats});
        device.write(state->layer(0), 0, initial.data(), initial.size() * sizeof(float));
        auto out = floats_on(device, std::vector<float>(count, 99.0f));
        const StateView view{state.get(), 0, 1, initial_length, rows};
        device.dilated_conv_silu_add({out.get(), 0}, {xb.get(), 0}, {rb.get(), 0}, {wb.get(), 0},
                                     channels, taps, dilation, 0, &view, 1);
        whole_output = read_floats(device, *out, count);
        whole_state = read_floats(device, state->layer(0), 2 * stride);
    }
    for (size_t i = 0; i < count; ++i) {
        const double error = std::fabs(double(whole_output[i]) - expected[i]);
        require(error <= 32 * U * std::max(1.0, std::fabs(expected[i])) + 2e-6,
                "long dilated conv differs from independent double oracle at " + std::to_string(i));
    }
    const auto expected_final_tail = expected_tail(x, initial, sequences[0], 0, channels,
                                                  history, suffix, stride);
    require(std::equal(expected_final_tail.begin(), expected_final_tail.end(),
                       whole_state.begin() + stride + shape.slot_floats()),
            "long dilated conv did not retain its raw tail and suffix");

    for (size_t chunk : {size_t(1), size_t(2), size_t(3), size_t(16)}) {
        auto state = device.state_alloc(1, 2, shape, {tail_floats});
        device.write(state->layer(0), 0, initial.data(), initial.size() * sizeof(float));
        auto out = floats_on(device, std::vector<float>(count, 99.0f));
        size_t first = 0;
        while (first < rows) {
            const size_t nq = std::min(chunk, rows - first);
            const size_t src = first ? 1 : 0;
            const StateView view{state.get(), src, 1, initial_length + first, nq};
            device.dilated_conv_silu_add({out.get(), first * channels}, {xb.get(), first * channels},
                                         {rb.get(), first * channels}, {wb.get(), 0},
                                         channels, taps, dilation, 0, &view, 1);
            first += nq;
        }
        const auto sliced_output = read_floats(device, *out, count);
        const auto sliced_state = read_floats(device, state->layer(0), 2 * stride);
        require(same_bits(sliced_output, whole_output),
                "long dilated conv changed bits when sliced into chunks of " + std::to_string(chunk));
        require(same_bits(sliced_state, whole_state),
                "long dilated conv state changed bits when sliced into chunks of " + std::to_string(chunk));
    }
    device.set_threads(1);
}

void check_oldest_tap(Backend& device) {
    constexpr size_t channels = 128, taps = 4, dilation = 3, history = (taps - 1) * dilation;
    constexpr size_t suffix = 2;
    const size_t tail_floats = history * channels + suffix;
    const StateShape shape = {1, 1, 2, 2};
    auto state = device.state_alloc(1, 2, shape, {tail_floats});
    std::vector<float> initial(2 * (shape.slot_floats() + tail_floats), 0.0f);
    initial[shape.slot_floats()] = 1.0f;
    device.write(state->layer(0), 0, initial.data(), initial.size() * sizeof(float));
    auto x = floats_on(device, std::vector<float>(channels, 0.0f));
    auto residual = floats_on(device, std::vector<float>(channels, 0.0f));
    std::vector<float> weights(channels * taps, 0.0f);
    weights[0] = 1.0f;
    auto w = floats_on(device, weights);
    auto out = floats_on(device, std::vector<float>(channels, -1.0f));
    const StateView view{state.get(), 0, 1, history, 1};
    device.dilated_conv_silu_add({out.get(), 0}, {x.get(), 0}, {residual.get(), 0}, {w.get(), 0},
                                 channels, taps, dilation, 0, &view, 1);
    const auto got = read_floats(device, *out, channels);
    const double expected = 1.0 / (1.0 + std::exp(-1.0));
    require(std::fabs(double(got[0]) - expected) <= 2e-7,
            "dilated conv failed the oldest-tap impulse at the full dilation distance");
    require(std::all_of(got.begin() + 1, got.end(), [](float value) { return value == 0.0f; }),
            "oldest-tap impulse leaked into another channel");
}

void check_view_order(Backend& device, std::mt19937& rng) {
    constexpr size_t channels = 128, taps = 4, dilation = 3, history = (taps - 1) * dilation, suffix = 3;
    const size_t tail_floats = history * channels + suffix;
    const StateShape shape = {1, 1, 2, 2};
    const size_t stride = shape.slot_floats() + tail_floats;
    const std::vector<Sequence> sequences = {{14, 0, 1, 33}, {22, 2, 3, 17}};
    std::vector<std::vector<float>> xs, residuals;
    for (const Sequence& sequence : sequences) {
        xs.push_back(random_values(rng, sequence.rows * channels, -0.9f, 0.9f));
        residuals.push_back(random_values(rng, sequence.rows * channels, -0.5f, 0.5f));
    }
    const auto weights = random_values(rng, channels * taps, -0.7f, 0.7f);
    auto seed = device.state_alloc(1, 4, shape, {tail_floats});
    std::vector<float> initial;
    seed_state(device, *seed, channels, history, suffix, sequences, rng, initial);
    seed.reset();
    auto wb = floats_on(device, weights);

    std::vector<float> canonical_outputs[2], canonical_state;
    for (bool reverse : {false, true}) {
        auto state = device.state_alloc(1, 4, shape, {tail_floats});
        device.write(state->layer(0), 0, initial.data(), initial.size() * sizeof(float));
        std::vector<size_t> order = reverse ? std::vector<size_t>{1, 0} : std::vector<size_t>{0, 1};
        std::vector<float> x, residual;
        std::vector<StateView> views;
        for (size_t index : order) {
            x.insert(x.end(), xs[index].begin(), xs[index].end());
            residual.insert(residual.end(), residuals[index].begin(), residuals[index].end());
            const Sequence& sequence = sequences[index];
            views.push_back({state.get(), sequence.src, sequence.dst, sequence.length, sequence.rows});
        }
        auto xb = floats_on(device, x), rb = floats_on(device, residual);
        auto out = floats_on(device, std::vector<float>(x.size(), -17.0f));
        device.dilated_conv_silu_add({out.get(), 0}, {xb.get(), 0}, {rb.get(), 0}, {wb.get(), 0},
                                     channels, taps, dilation, 0, views.data(), views.size());
        const auto grouped = read_floats(device, *out, x.size());
        size_t offset = 0;
        for (size_t index : order) {
            const size_t count = sequences[index].rows * channels;
            std::vector<float> sequence_output(grouped.begin() + offset, grouped.begin() + offset + count);
            if (!reverse) canonical_outputs[index] = std::move(sequence_output);
            else require(same_bits(sequence_output, canonical_outputs[index]),
                         "dilated conv output changed when grouped views were reordered");
            offset += count;
        }
        const auto final_state = read_floats(device, state->layer(0), 4 * stride);
        if (!reverse) canonical_state = final_state;
        else require(same_bits(final_state, canonical_state),
                     "dilated conv state changed when grouped views were reordered");
    }
}

void check_refusals(Backend& device) {
    constexpr size_t channels = 3, taps = 3, dilation = 2, rows = 2, count = channels * rows;
    constexpr size_t history = (taps - 1) * dilation;
    const size_t required_tail = history * channels;
    const StateShape shape = {1, 1, 2, 2};
    auto state = device.state_alloc(1, 2, shape, {required_tail});
    std::vector<float> state_before(state->layer(0).size() / sizeof(float), 73.0f);
    device.write(state->layer(0), 0, state_before.data(), state_before.size() * sizeof(float));
    const StateView view{state.get(), 0, 1, 0, rows};
    const auto x = std::vector<float>(count, 0.25f), residual = std::vector<float>(count, 0.5f);
    const auto weights = std::vector<float>(channels * taps, -0.25f);
    auto xb = floats_on(device, x), rb = floats_on(device, residual), wb = floats_on(device, weights);
    auto protected_output = floats_on(device, std::vector<float>(count, 91.0f));
    const auto call = [&](backend::Slice out, backend::CSlice input, backend::CSlice res, backend::CSlice weight,
                          size_t n_taps, size_t n_dilation, size_t layer, const StateView* views, size_t n_views) {
        device.dilated_conv_silu_add(out, input, res, weight, channels, n_taps, n_dilation, layer, views, n_views);
    };
    const auto check_unchanged = [&](const std::string& name) {
        const auto output = read_floats(device, *protected_output, count);
        require(std::all_of(output.begin(), output.end(),
                            [](float value) { return value == 91.0f; }), name + " wrote output before refusing");
        require(same_bits(read_floats(device, state->layer(0), state_before.size()), state_before),
                name + " changed state before refusing");
    };
    refuses("dilated conv accepted zero taps", [&] { call({protected_output.get(), 0}, {xb.get(), 0}, {rb.get(), 0}, {wb.get(), 0}, 0, dilation, 0, &view, 1); });
    check_unchanged("zero-tap refusal");
    refuses("dilated conv accepted zero dilation", [&] { call({protected_output.get(), 0}, {xb.get(), 0}, {rb.get(), 0}, {wb.get(), 0}, taps, 0, 0, &view, 1); });
    check_unchanged("zero-dilation refusal");
    refuses("dilated conv accepted overflowing history size", [&] {
        call({protected_output.get(), 0}, {xb.get(), 0}, {rb.get(), 0}, {wb.get(), 0},
             std::numeric_limits<size_t>::max(), dilation, 0, &view, 1);
    });
    check_unchanged("overflowing-history refusal");
    auto short_state = device.state_alloc(1, 2, shape, {required_tail - 1});
    const StateView short_view{short_state.get(), 0, 1, 0, rows};
    refuses("dilated conv accepted a short tail", [&] {
        call({protected_output.get(), 0}, {xb.get(), 0}, {rb.get(), 0}, {wb.get(), 0}, taps, dilation, 0, &short_view, 1);
    });
    check_unchanged("short-tail refusal");
    const StateView bad_view{state.get(), std::numeric_limits<size_t>::max(), 1, 0, rows};
    refuses("dilated conv accepted an invalid slot", [&] {
        call({protected_output.get(), 0}, {xb.get(), 0}, {rb.get(), 0}, {wb.get(), 0}, taps, dilation, 0, &bad_view, 1);
    });
    check_unchanged("invalid-slot refusal");

    auto partial = floats_on(device, std::vector<float>(count + 1, 91.0f));
    refuses("dilated conv accepted partial output/residual overlap", [&] {
        call({partial.get(), 1}, {xb.get(), 0}, {partial.get(), 0}, {wb.get(), 0}, taps, dilation, 0, &view, 1);
    });
    const auto partial_values = read_floats(device, *partial, count + 1);
    require(std::all_of(partial_values.begin(), partial_values.end(),
                        [](float value) { return value == 91.0f; }), "partial-overlap refusal wrote before refusing");
    auto alias = floats_on(device, std::vector<float>(count, 91.0f));
    refuses("dilated conv accepted output/input overlap", [&] {
        call({alias.get(), 0}, {alias.get(), 0}, {rb.get(), 0}, {wb.get(), 0}, taps, dilation, 0, &view, 1);
    });
    const auto alias_values = read_floats(device, *alias, count);
    require(std::all_of(alias_values.begin(), alias_values.end(),
                        [](float value) { return value == 91.0f; }), "input-overlap refusal wrote before refusing");
    auto weight_alias = floats_on(device, std::vector<float>(std::max(count, channels * taps), 91.0f));
    refuses("dilated conv accepted output/weight overlap", [&] {
        call({weight_alias.get(), 0}, {xb.get(), 0}, {rb.get(), 0}, {weight_alias.get(), 0}, taps, dilation, 0, &view, 1);
    });
    const auto weight_values = read_floats(device, *weight_alias, std::max(count, channels * taps));
    require(std::all_of(weight_values.begin(), weight_values.end(),
                        [](float value) { return value == 91.0f; }), "weight-overlap refusal wrote before refusing");

    const CSlice state_as_input{&state->layer(0), 0};
    refuses("dilated conv accepted an operand overlapping its state", [&] {
        call({protected_output.get(), 0}, state_as_input, {rb.get(), 0}, {wb.get(), 0}, taps, dilation, 0, &view, 1);
    });
    check_unchanged("state-overlap refusal");
    refuses("dilated conv accepted output overlapping its state", [&] {
        call({&state->layer(0), 0}, {xb.get(), 0}, {rb.get(), 0}, {wb.get(), 0}, taps, dilation, 0, &view, 1);
    });
    check_unchanged("state-output-overlap refusal");
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string device_name = "cpu";
        if (argc == 3 && std::string(argv[1]) == "--device") device_name = argv[2];
        else if (argc != 1) throw std::runtime_error("usage: dilated-conv-test [--device cpu|vulkan:N]");
        auto device = backend::make_backend(device_name);
        std::mt19937 rng(20261014);
        run_case(*device, rng, 1, 4, 2);
        run_case(*device, rng, 5, 3, 1);
        run_case(*device, rng, 32, 1, 7);
        check_oldest_tap(*device);
        check_long_slices(*device, 128, rng);
        check_view_order(*device, rng);
        check_long_slices(*device, 10240, rng);
        check_in_place_residual(*device);
        check_refusals(*device);
        std::puts("Dilated causal conv: independent oracle, mixed views, state tails and thread paths passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Dilated causal conv: FAIL: %s\n", error.what());
        return 1;
    }
}
