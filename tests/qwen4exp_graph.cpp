// Qwen4Exp PLE and hyper-connection graph pieces against the frozen HF intermediates.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backends/devices.hpp"
#include "core/json.hpp"
#include "model/arch/qwen4exp.hpp"
#include "quant/quant.hpp"

namespace {
using backend::Backend;
using backend::BufferPtr;
using backend::CSlice;
using backend::Dtype;
using backend::Slice;
using infer::Weight;
using infer::Step;
using Json = jmini::Value;

size_t checks = 0;
void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

const Json& field(const Json& v, const std::string& key) {
    const Json* out = v.get(key);
    if (!out) throw std::runtime_error("reference: missing field " + key);
    return *out;
}

std::vector<float> floats(const Json& value) {
    std::vector<float> out;
    auto append = [&](auto&& self, const Json& v) -> void {
        if (v.isNumber()) out.push_back(float(v.asNumber()));
        else if (v.isArray()) for (const Json& x : v.asArray()) self(self, x);
        else throw std::runtime_error("reference: expected numeric array");
    };
    append(append, value);
    return out;
}

size_t integer(const Json& v, const std::string& key) {
    const double n = field(v, key).asNumber();
    if (n < 0 || n > double(std::numeric_limits<size_t>::max()) || std::floor(n) != n)
        throw std::runtime_error("reference: invalid integer " + key);
    return size_t(n);
}

float max_abs(const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "reference: vector width mismatch");
    float worst = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float error = std::fabs(actual[i] - expected[i]);
        if (!std::isfinite(error)) return std::numeric_limits<float>::infinity();
        worst = std::max(worst, error);
    }
    return worst;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

void bounded(const std::vector<float>& actual, const std::vector<float>& expected, float bound,
             const std::string& label, float& worst) {
    const float error = max_abs(actual, expected);
    worst = std::max(worst, error);
    require(error <= bound, label + " max abs error " + std::to_string(error) + " exceeds " + std::to_string(bound));
}

BufferPtr upload(Backend& b, const std::vector<float>& values, size_t guards = 0, float guard = 0.0f) {
    const size_t count = values.size() + guards * 2;
    BufferPtr p = b.alloc(std::max<size_t>(count, 1) * sizeof(float), backend::Memory::device);
    if (count) {
        std::vector<float> initialized(count, guard);
        std::copy(values.begin(), values.end(), initialized.begin() + guards);
        b.write(*p, 0, initialized.data(), count * sizeof(float));
    }
    return p;
}

BufferPtr guarded_output(Backend& b, size_t payload, float guard) {
    std::vector<float> initial(payload + 2, guard);
    BufferPtr p = b.alloc(initial.size() * sizeof(float), backend::Memory::device);
    b.write(*p, 0, initial.data(), initial.size() * sizeof(float));
    return p;
}

std::vector<float> download(Backend& b, const backend::Buffer& p, size_t count, size_t offset = 0) {
    std::vector<float> out(count);
    if (count) b.read(p, offset * sizeof(float), out.data(), count * sizeof(float));
    return out;
}

Weight weight(Backend& b, const Json& state, const std::string& name,
              const std::vector<size_t>& shape, size_t nin, size_t nout, bool norm = false) {
    const Json& t = field(state, name);
    const auto& got_shape = field(t, "shape").asArray();
    require(got_shape.size() == shape.size(), name + " tensor rank differs from the fixture");
    for (size_t i = 0; i < shape.size(); ++i)
        require(size_t(got_shape[i].asNumber()) == shape[i], name + " tensor shape differs from the fixture");
    std::vector<float> data = floats(field(t, "values"));
    require(data.size() == nin * nout, name + " tensor element count differs from its graph shape");
    if (norm) for (float& x : data) x += 1.0f;
    Weight out;
    out.type = quant::GGML_TYPE_F32;
    out.data = upload(b, data);
    out.nin = nin;
    out.nout = nout;
    return out;
}

struct FixtureCase {
    std::string name;
    std::vector<float> ple_input, embeddings, ple_output;
    std::vector<float> streams, mixed, injection_gates;
};

std::vector<FixtureCase> cases(const Json& root) {
    std::vector<FixtureCase> out;
    for (const Json& c : field(root, "cases").asArray()) {
        const Json& ple = field(c, "ple_intermediates");
        const Json& hc = field(c, "attention_hyper_connection");
        FixtureCase v;
        v.name = field(c, "name").asString();
        v.ple_input = floats(field(ple, "input"));
        v.embeddings = floats(field(ple, "ngram_embeddings"));
        v.ple_output = floats(field(ple, "output"));
        v.streams = floats(field(hc, "original_streams"));
        v.mixed = floats(field(hc, "mixed_input"));
        v.injection_gates = floats(field(hc, "injection_gates"));
        out.push_back(std::move(v));
    }
    return out;
}

infer::qwen4exp::Config config(const Json& root) {
    const Json& c = field(root, "config");
    infer::qwen4exp::Config out;
    out.n_embd = int(integer(c, "hidden_size"));
    out.hc_count = int(integer(c, "hc_count"));
    out.hc_low_rank = int(integer(c, "hc_lowrank"));
    out.rms_eps = float(field(c, "rms_norm_eps").asNumber());
    out.k_heads = int(integer(c, "linear_num_key_heads"));
    out.v_heads = int(integer(c, "linear_num_value_heads"));
    out.k_dim = int(integer(c, "linear_key_head_dim"));
    out.v_dim = int(integer(c, "linear_value_head_dim"));
    out.ple_ngram_size = int(integer(c, "ngram_size"));
    out.ple_heads_per_ngram = int(integer(c, "heads_per_ngram"));
    out.ple_conv_kernel = int(integer(c, "ple_conv_kernel_size"));
    out.ple_embedding_dim = int(integer(c, "ple_embed_dim"));
    out.ple_embedding_length = out.ple_embedding_dim / ((out.ple_ngram_size - 1) * out.ple_heads_per_ngram);
    require(out.n_embd == 32 && out.hc_count == 4 && out.hc_low_rank == 4 && out.ple_embedding_dim == 8,
            "fixture dimensions changed from the native graph case");
    require(out.ple_embedding_length * (out.ple_ngram_size - 1) * out.ple_heads_per_ngram == out.ple_embedding_dim,
            "fixture PLE embedding width does not factor into its n-gram heads");
    return out;
}

struct PleRun {
    std::vector<float> output, normed, gated, tail, state_prefix, replay;
};

infer::Step step(Backend& b, BufferPtr& x, size_t rows, const backend::RowRun& run, size_t x_offset = 0) {
    infer::Step s{b, nullptr, nullptr, Slice{x.get(), x_offset}, rows, backend::RowRuns{&run, 1},
                  nullptr, 0, nullptr, 0, 0, nullptr, nullptr, nullptr};
    s.dtype = Dtype::f32;
    return s;
}

void check_guards(Backend& b, const backend::Buffer& p, size_t payload, float guard, const std::string& label) {
    const auto all = download(b, p, payload + 2, 0);
    require(all.front() == guard && all.back() == guard, label + " changed an output guard");
}

PleRun run_ple(Backend& b, const infer::qwen4exp::Config& c, const Json& state,
               const FixtureCase& test, size_t chunk) {
    using namespace infer::qwen4exp;
    const size_t rows = 13, H = size_t(c.hc_count), E = size_t(c.n_embd), width = H * E;
    const size_t embdim = size_t(c.ple_embedding_dim), history = (size_t(c.ple_conv_kernel) - 1) * size_t(c.ple_ngram_size);
    const size_t tail_floats = history * width;
    const backend::StateShape shape{size_t(c.k_heads), size_t(c.v_heads), size_t(c.k_dim), size_t(c.v_dim)};
    Weight key = weight(b, state, "model.layers.0.ple.key_proj.weight", {width, embdim}, embdim, width);
    Weight value = weight(b, state, "model.layers.0.ple.value_proj.weight", {E, embdim}, embdim, E);
    Weight norm_key = weight(b, state, "model.layers.0.ple.norm_key.weight", {width}, width, 1, true);
    Weight norm_query = weight(b, state, "model.layers.0.ple.norm_query.weight", {width}, width, 1, true);
    Weight norm_conv = weight(b, state, "model.layers.0.ple.norm_conv.weight", {width}, width, 1, true);
    Weight conv = weight(b, state, "model.layers.0.ple.conv1d.weight", {width, 1, size_t(c.ple_conv_kernel)},
                         size_t(c.ple_conv_kernel), width);
    PleWeights weights{key, value, norm_key, norm_query, norm_conv, conv};

    BufferPtr x = upload(b, test.ple_input);
    BufferPtr embeddings = upload(b, test.embeddings);
    BufferPtr key_buf = b.alloc(rows * width * sizeof(float), backend::Memory::device);
    BufferPtr value_buf = b.alloc(rows * E * sizeof(float), backend::Memory::device);
    BufferPtr query_buf = b.alloc(rows * width * sizeof(float), backend::Memory::device);
    BufferPtr gated_buf = b.alloc(rows * width * sizeof(float), backend::Memory::device);
    BufferPtr normed_buf = b.alloc(rows * width * sizeof(float), backend::Memory::device);
    BufferPtr out_buf = guarded_output(b, rows * width, -123.0f);
    auto storage = b.state_alloc(1, 2, shape, {tail_floats});
    const size_t stride = storage->slot_floats(0), core = shape.slot_floats();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> initial(stride * 2);
    for (size_t i = 0; i < initial.size(); ++i) initial[i] = float(int(i % 13) - 6) * 0.003f;
    std::fill(initial.begin() + core, initial.begin() + stride, nan);
    std::fill(initial.begin() + stride + core, initial.end(), nan);
    b.write(storage->layer(0), 0, initial.data(), initial.size() * sizeof(float));
    std::vector<float> out_guard(rows * width + 2, -123.0f);
    b.write(*out_buf, 0, out_guard.data(), out_guard.size() * sizeof(float));

    const size_t step_rows = chunk ? chunk : rows;
    for (size_t first = 0; first < rows; first += step_rows) {
        const size_t n = std::min(step_rows, rows - first);
        backend::StateView view{storage.get(), 0, 0, first, n};
        const backend::RowRun run{n, rows};
        Step s = step(b, x, n, run, first * width);
        s.states = &view;
        s.n_views = 1;
        s.state_layer = 0;
        PleScratch part{{key_buf.get(), first * width}, {value_buf.get(), first * E},
                        {query_buf.get(), first * width}, {gated_buf.get(), first * width},
                        {normed_buf.get(), first * width}};
        ple(c, s, weights, CSlice{embeddings.get(), first * embdim}, Slice{out_buf.get(), 1 + first * width}, part);
    }
    PleRun result;
    result.output = download(b, *out_buf, rows * width, 1);
    result.normed = download(b, *normed_buf, rows * width);
    result.gated = download(b, *gated_buf, rows * width);
    const auto state_values = download(b, storage->layer(0), stride * 2);
    result.tail.assign(state_values.begin() + core, state_values.begin() + core + tail_floats);
    result.state_prefix.assign(state_values.begin(), state_values.begin() + core);
    result.state_prefix.insert(result.state_prefix.end(), state_values.begin() + stride, state_values.begin() + stride + core);
    require(same_bits(result.tail, std::vector<float>(result.normed.end() - (std::ptrdiff_t)tail_floats, result.normed.end())),
            "PLE did not retain its final normalized inputs as raw convolution history");
    require(same_bits(result.state_prefix, [&] {
        std::vector<float> expected;
        expected.insert(expected.end(), initial.begin(), initial.begin() + core);
        expected.insert(expected.end(), initial.begin() + stride, initial.begin() + stride + core);
        return expected;
    }()), "PLE changed the common recurrent prefix or an unused state slot");
    check_guards(b, *out_buf, rows * width, -123.0f, "PLE");

    BufferPtr replay_buf = guarded_output(b, rows * width, -123.0f);
    std::vector<float> replay_guard(rows * width + 2, -123.0f);
    b.write(*replay_buf, 0, replay_guard.data(), replay_guard.size() * sizeof(float));
    constexpr size_t accepted = 5;
    for (const auto [first, n] : {std::pair<size_t, size_t>{0, accepted}, {accepted, rows - accepted}}) {
        backend::StateView replay_view{storage.get(), 1, 1, first, n};
        const backend::RowRun replay_run{n, rows};
        Step replay_step = step(b, x, n, replay_run, first * width);
        replay_step.states = &replay_view;
        replay_step.n_views = 1;
        replay_step.state_layer = 0;
        PleScratch replay_tmp{{key_buf.get(), first * width}, {value_buf.get(), first * E},
                              {query_buf.get(), first * width}, {gated_buf.get(), first * width},
                              {normed_buf.get(), first * width}};
        ple_recur(c, replay_step, conv, Slice{replay_buf.get(), 1 + first * width}, replay_tmp);
    }
    result.replay = download(b, *replay_buf, rows * width, 1);
    require(same_bits(result.output, result.replay), "PLE recurrent replay differs from its original convolution output");
    check_guards(b, *replay_buf, rows * width, -123.0f, "PLE replay");
    return result;
}

struct MixRun { std::vector<float> mixed, injected; };

MixRun run_mix(Backend& b, const infer::qwen4exp::Config& c, const Json& state, const FixtureCase& test) {
    using namespace infer::qwen4exp;
    const size_t rows = 13, H = size_t(c.hc_count), E = size_t(c.n_embd), width = H * E, L = size_t(c.hc_low_rank);
    Weight norm = weight(b, state, "model.layers.1.attn_hyper_connection.hc_norm.weight", {width}, width, 1, true);
    Weight down = weight(b, state, "model.layers.1.attn_hyper_connection.input_mix_weight_down.weight", {L, width}, width, L);
    Weight up = weight(b, state, "model.layers.1.attn_hyper_connection.input_mix_weight_up.weight", {width, L}, L, width);
    Weight inject_weight = weight(b, state, "model.layers.1.attn_hyper_connection.block_inject_weight.weight", {H, width}, width, H);
    const float guard = -91.0f;
    BufferPtr x = upload(b, test.streams, 1, guard);
    BufferPtr out = guarded_output(b, rows * E, guard);
    BufferPtr normed = b.alloc(rows * width * sizeof(float), backend::Memory::device);
    BufferPtr low = b.alloc(rows * L * sizeof(float), backend::Memory::device);
    BufferPtr gates = b.alloc(rows * width * sizeof(float), backend::Memory::device);
    BufferPtr inj_gates = b.alloc(rows * H * sizeof(float), backend::Memory::device);
    const backend::RowRun run{rows, rows};
    Step s = step(b, x, rows, run, 1);
    MixScratch scratch{{normed.get(), 0}, {low.get(), 0}, {gates.get(), 0}};
    mix(c, s, MixWeights{norm, down, up}, Slice{out.get(), 1}, scratch, &inject_weight, Slice{inj_gates.get(), 0});
    MixRun result;
    result.mixed = download(b, *out, rows * E, 1);
    check_guards(b, *out, rows * E, guard, "HC mix");
    require(same_bits(download(b, *x, test.streams.size() + 2), [&] {
        std::vector<float> expected(test.streams.size() + 2, guard);
        std::copy(test.streams.begin(), test.streams.end(), expected.begin() + 1);
        return expected;
    }()), "mix changed its source residual");

    std::vector<float> block(rows * E);
    for (size_t i = 0; i < block.size(); ++i)
        block[i] = 0.025f * std::sin(float(i) * 0.17f) + 0.009f * std::cos(float(i) * 0.071f);
    BufferPtr block_buf = upload(b, block);
    inject(c, s, CSlice{block_buf.get(), 0}, CSlice{inj_gates.get(), 0});
    result.injected = download(b, *x, test.streams.size() + 2);
    return result;
}

std::vector<float> injected_reference(const FixtureCase& test, size_t streams, size_t dim) {
    const size_t rows = test.mixed.size() / dim;
    std::vector<float> out = test.streams;
    for (size_t r = 0; r < rows; ++r)
        for (size_t h = 0; h < streams; ++h)
            for (size_t d = 0; d < dim; ++d) {
                const size_t i = (r * streams + h) * dim + d;
                const size_t block_i = r * dim + d;
                const float block = 0.025f * std::sin(float(block_i) * 0.17f) + 0.009f * std::cos(float(block_i) * 0.071f);
                out[i] += block * test.injection_gates[r * streams + h];
            }
    return out;
}

Json read_reference(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open HF reference " + path);
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return jmini::parse(text);
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string reference = "tests/data/qwen4exp_reference.json", device = "cpu";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--reference" && i + 1 < argc) reference = argv[++i];
            else if (arg == "--device" && i + 1 < argc) device = argv[++i];
            else throw std::runtime_error("usage: llmx-qwen4exp-graph-test [--reference FILE] [--device cpu|vulkan:N]");
        }
        const Json root = read_reference(reference);
        const infer::qwen4exp::Config c = config(root);
        const Json& state = field(root, "state_dict");
        const std::vector<FixtureCase> tests = cases(root);
        require(tests.size() == 3, "HF reference must contain all three PLE cases");
        backend::BackendPtr device_backend = backend::make_backend(device);
        auto* cpu = dynamic_cast<backend::CpuBackend*>(device_backend.get());
        float ple_worst = 0.0f, mix_worst = 0.0f, inject_worst = 0.0f;
        for (const FixtureCase& test : tests) {
            require(test.ple_input.size() == 13 * size_t(c.hc_count) * size_t(c.n_embd) &&
                    test.embeddings.size() == 13 * size_t(c.ple_embedding_dim) &&
                    test.ple_output.size() == test.ple_input.size() &&
                    test.streams.size() == 13 * size_t(c.hc_count) * size_t(c.n_embd) &&
                    test.mixed.size() == 13 * size_t(c.n_embd) &&
                    test.injection_gates.size() == 13 * size_t(c.hc_count),
                    test.name + " PLE/HC fixture dimensions differ");
            const std::vector<int> threads = cpu ? std::vector<int>{1, 2} : std::vector<int>{0};
            std::vector<float> canonical;
            for (int thread_count : threads) {
                if (cpu) cpu->set_threads(thread_count);
                const std::string suffix = test.name + (cpu ? " CPU/" + std::to_string(thread_count) : " " + device);
                const PleRun full = run_ple(*device_backend, c, state, test, 0);
                bounded(full.output, test.ple_output, 2e-5f, suffix + " PLE whole", ple_worst);
                if (canonical.empty()) canonical = full.output;
                else require(same_bits(full.output, canonical), suffix + " PLE whole changed bits across CPU thread counts");
                for (size_t chunk : {size_t(1), size_t(3), size_t(4)}) {
                    const PleRun sliced = run_ple(*device_backend, c, state, test, chunk);
                    bounded(sliced.output, test.ple_output, 2e-5f, suffix + " PLE sliced", ple_worst);
                    require(same_bits(sliced.output, full.output), suffix + " PLE slicing changed bits");
                    require(same_bits(sliced.tail, full.tail), suffix + " PLE slicing changed the saved tail");
                }
            }
            std::vector<float> canonical_mix, canonical_inject;
            for (int thread_count : threads) {
                if (cpu) cpu->set_threads(thread_count);
                const std::string suffix = test.name + (cpu ? " CPU/" + std::to_string(thread_count) : " " + device);
                MixRun result = run_mix(*device_backend, c, state, test);
                bounded(result.mixed, test.mixed, 2e-5f, suffix + " HC mix", mix_worst);
                std::vector<float> injected(result.injected.begin() + 1, result.injected.end() - 1);
                const std::vector<float> expected = injected_reference(test, size_t(c.hc_count), size_t(c.n_embd));
                bounded(injected, expected, 2e-5f, suffix + " HC injection", inject_worst);
                require(result.injected.front() == -91.0f && result.injected.back() == -91.0f,
                        suffix + " HC injection changed an input guard");
                if (canonical_mix.empty()) {
                    canonical_mix = result.mixed;
                    canonical_inject = injected;
                } else {
                    require(same_bits(result.mixed, canonical_mix) && same_bits(injected, canonical_inject),
                            suffix + " HC graph changed bits across CPU thread counts");
                }
            }
        }
        std::cout << "qwen4exp-graph: " << checks << " checks passed; max abs PLE=" << ple_worst
                  << " mix=" << mix_worst << " inject=" << inject_worst << " on " << device << '\n';
    } catch (const std::exception& e) {
        std::cerr << "qwen4exp-graph: FAIL: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
