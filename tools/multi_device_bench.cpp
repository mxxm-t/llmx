// Each stage's device time of a model split by layers over Vulkan devices as the CLI places it, a decode pass and, when asked, a prefill pass, from GPU timestamps: the stage-time source of the layer split's and the tensor split's models (docs/STATUS-2026-09.md, layer split phase 3, step 0).
// `stages MODEL D... [options]` is its one mode.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "backends/vulkan/vulkan_backend.hpp"
#include "inference/load.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double ms_since(Clock::time_point t0) {
    return ms_between(t0, Clock::now());
}

struct StagesOptions {
    std::vector<size_t> rows{1, 8, 16, 32, 64};
    std::vector<size_t> prefill;   // prompt rows of the prefill passes timed, none unless asked
    size_t context = 512, steps = 16;
};

// Layers per device from a split's plan, which gives each device a line as LayerSplit::describe writes it; empty when a line does not read so.
std::vector<int> plan_layers(const std::string& plan, size_t devices) {
    std::vector<int> out;
    std::istringstream in(plan);
    for (std::string line; out.size() < devices && std::getline(in, line);) {
        const size_t at = line.find(": layers ");
        if (at == std::string::npos) {
            if (line.find(": no layers") == std::string::npos) return {};
            out.push_back(0);
            continue;
        }
        char* end = nullptr;
        const long first = std::strtol(line.c_str() + at + 9, &end, 10);
        if (*end != '-') return {};
        const long last = std::strtol(end + 1, nullptr, 10);
        out.push_back((int)(last - first + 1));
    }
    return out.size() == devices ? out : std::vector<int>{};
}

// Decode passes of each row count through the model split over `devices`, every sequence first given `context` tokens, and prefill passes of each prompt row count, each the slice that ends a prompt after `context` tokens, so it wants its last row's logits.
// Each device's time a pass is the sum of its dispatches' GPU timestamps.
// The last stage is set against the others, and where the plan gives the layer counts, what it takes beyond its layers at the others' time a layer is the head's.
int stages(const std::string& path, const std::vector<int>& devices, const StagesOptions& o) {
    std::vector<backend::BackendPtr> backends;
    infer::PlacementRequest request;
    for (int d : devices) {
        backends.push_back(backend::make_vulkan_backend(d, true));
        request.names.push_back("vulkan:" + std::to_string(d));
    }
    const size_t most = *std::max_element(o.rows.begin(), o.rows.end()), warm = 3;
    const size_t widest = o.prefill.empty() ? 0 : *std::max_element(o.prefill.begin(), o.prefill.end());
    request.decode_rows = most;
    request.histories = most;
    request.history_tokens = o.context + std::max(warm + o.steps, widest);
    const auto loaded = infer::load_model(path, backends, request);
    infer::Model& model = *loaded->model;
    std::printf("%s, %zu tokens of context, %zu passes timed after %zu:\n%s", path.c_str(), o.context, o.steps, warm, loaded->plan.c_str());
    const std::vector<int> layers = plan_layers(loaded->plan, devices.size());
    if (layers.empty()) std::printf("plan not read: the layer counts and the head's share are left out\n");
    // Fixed pseudo-random ids below 1000, or below a smaller vocabulary's size, the same in every run.
    const uint32_t vocab = (uint32_t)std::min<size_t>(1000, model.n_vocab());
    std::mt19937 rng(12345);
    auto ids = [&](size_t n) {
        std::vector<uint32_t> v(n);
        for (auto& t : v) t = (uint32_t)(rng() % vocab);
        return v;
    };
    const std::vector<uint32_t> prompt = ids(o.context), gen = ids(warm + o.steps), tail = ids(widest);
    const size_t n = devices.size(), ubatch = model.prefill_batch();
    infer::ExecContext ctx;
    // Tokens [from, to) of the prompt `whole` into q, in slices of the ubatch, each slice given the whole prompt's extent; the slice that ends the prompt wants its last row's logits.
    auto feed = [&](infer::Sequence& q, const std::vector<uint32_t>& whole, size_t from, size_t to) {
        for (size_t at = from; at < to; at += ubatch) {
            const size_t len = std::min(ubatch, to - at);
            infer::BatchEntry e{&q, whole.data() + at, len, at + len == whole.size()};
            e.extent = whole.size();
            model.forward(ctx, &e, 1);
        }
    };
    struct Reading {
        std::vector<double> device, dispatches;   // each device's time and dispatches a pass
        double host = 0;                          // a pass on the host, the stages one after another
    };
    // `o.steps` passes timed after `warm`, each after its `setup`, which is not timed.
    auto time_passes = [&](const auto& setup, const auto& pass) {
        Reading r{std::vector<double>(n, 0.0), std::vector<double>(n, 0.0), 0.0};
        for (size_t g = 0; g < warm + o.steps; ++g) {
            setup(g);
            for (auto& b : backends) backend::vulkan_kernel_times(*b);
            const auto t0 = Clock::now();
            pass(g);
            if (g < warm) continue;
            r.host += ms_since(t0) / (double)o.steps;
            for (size_t d = 0; d < n; ++d) {
                for (const auto& k : backend::vulkan_kernel_times(*backends[d])) r.device[d] += k.second / (double)o.steps;
                r.dispatches[d] += (double)backend::vulkan_timed_dispatches(*backends[d]) / (double)o.steps;
            }
        }
        if (std::all_of(r.dispatches.begin(), r.dispatches.end(), [](double v) { return v == 0; }))
            throw std::runtime_error("no device timestamps its dispatches");
        return r;
    };
    auto report = [&](const Reading& r) {
        auto runs_layers = [&](size_t d) { return layers.empty() ? r.dispatches[d] > 0 : layers[d] > 0; };
        size_t last = n;
        for (size_t d = 0; d < n; ++d)
            if (runs_layers(d)) last = d;
        double others = 0;
        size_t n_others = 0;
        int other_layers = 0;
        for (size_t d = 0; d < n; ++d) {
            std::printf("  %s", request.names[d].c_str());
            if (!layers.empty()) std::printf(", %d layers", layers[d]);
            std::printf(": %.3f ms of device time a pass over %.0f dispatches\n", r.device[d], r.dispatches[d]);
            if (d != last && runs_layers(d)) {
                others += r.device[d];
                ++n_others;
                if (!layers.empty()) other_layers += layers[d];
            }
        }
        if (!n_others || last == n) return;
        std::printf("  last stage: %+.1f%% against the other stages' mean", 100 * (r.device[last] / (others / (double)n_others) - 1));
        if (other_layers) {
            const double per = others / other_layers, head = r.device[last] - layers[last] * per;
            std::printf("; beyond its %d layers at their %.3f ms a layer it takes %.3f ms, %.1f%% of it", layers[last], per, head, 100 * head / r.device[last]);
        }
        std::printf("\n");
    };
    for (size_t rows : o.rows) {
        std::vector<infer::Sequence> seqs;
        for (size_t i = 0; i < rows; ++i) seqs.push_back(model.make_sequence());
        for (auto& q : seqs) feed(q, prompt, 0, prompt.size());
        ctx.logits(0);
        std::vector<infer::BatchEntry> batch;
        const Reading r = time_passes([](size_t) {}, [&](size_t g) {
            batch.clear();
            for (auto& q : seqs) batch.push_back(infer::BatchEntry{&q, &gen[g], 1, true});
            model.forward(ctx, batch.data(), batch.size());
            ctx.logits(0);
        });
        for (auto& q : seqs) model.reset(q);
        std::printf("\n%zu row%s a pass: %.2f ms a pass on the host, the stages one after another\n", rows, rows == 1 ? "" : "s", r.host);
        report(r);
    }
    for (size_t rows : o.prefill) {
        std::vector<uint32_t> whole = prompt;
        whole.insert(whole.end(), tail.begin(), tail.begin() + (std::ptrdiff_t)rows);
        infer::Sequence q = model.make_sequence();
        const Reading r = time_passes(
            [&](size_t) {
                model.reset(q);
                feed(q, whole, 0, o.context);
            },
            [&](size_t) {
                feed(q, whole, o.context, whole.size());
                ctx.logits(0);
            });
        model.reset(q);
        std::printf("\nprefill, %zu prompt rows a pass ending a prompt after %zu tokens: %.2f ms a pass on the host, the stages one after another\n", rows, o.context, r.host);
        report(r);
    }
    return 0;
}

const char* const kUsage =
    "usage: llmx-multi-device-bench stages MODEL D... [--rows R[,R...]] [--prefill R[,R...]] [--context N] [--steps N]\n";

struct Usage : std::runtime_error {
    using std::runtime_error::runtime_error;
};

double number(const std::string& s, const std::string& what) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || *end || !std::isfinite(v) || v < 0) throw Usage(what + ": " + s + " is not a number of zero or more");
    return v;
}

size_t count(const std::string& s, const std::string& what) {
    const double v = number(s, what);
    if (v != std::floor(v)) throw Usage(what + ": " + s + " is not a whole number");
    return (size_t)v;
}

std::vector<std::string> items(const std::string& s, const std::string& what) {
    std::vector<std::string> out;
    std::istringstream in(s);
    for (std::string item; std::getline(in, item, ',');) out.push_back(item);
    if (out.empty()) throw Usage(what + ": an empty list");
    return out;
}

// A mode's arguments: device indices, then `--name value` pairs among the mode's options.
struct Args {
    std::vector<int> devices;
    std::map<std::string, std::string> named;
    bool has(const std::string& k) const { return named.count(k) != 0; }
    const std::string& at(const std::string& k) const { return named.at(k); }
};

Args parse_args(int argc, char** argv, int from, const std::vector<std::string>& known) {
    Args a;
    int i = from;
    for (; i < argc && std::strncmp(argv[i], "--", 2) != 0; ++i) a.devices.push_back((int)count(argv[i], "device"));
    for (; i < argc; i += 2) {
        const std::string k = argv[i];
        if (std::find(known.begin(), known.end(), k) == known.end()) throw Usage(k + ": not an option of this mode");
        if (i + 1 >= argc) throw Usage(k + ": needs a value");
        a.named[k] = argv[i + 1];
    }
    return a;
}

std::vector<size_t> count_list(const Args& a, const std::string& k, std::vector<size_t> fallback) {
    if (!a.has(k)) return fallback;
    std::vector<size_t> out;
    for (const auto& item : items(a.at(k), k)) out.push_back(count(item, k));
    return out;
}

StagesOptions stages_options(const Args& a) {
    StagesOptions o;
    o.rows = count_list(a, "--rows", o.rows);
    for (size_t r : o.rows)
        if (!r) throw Usage("--rows: at least one row a pass");
    o.prefill = count_list(a, "--prefill", o.prefill);
    for (size_t r : o.prefill)
        if (!r || r > (size_t)infer::kDefaultUbatch) throw Usage("--prefill: 1 to " + std::to_string(infer::kDefaultUbatch) + " prompt rows a pass, the ubatch");
    if (a.has("--context")) o.context = count(a.at("--context"), "--context");
    if (a.has("--steps")) o.steps = count(a.at("--steps"), "--steps");
    if (!o.context || !o.steps) throw Usage("--context and --steps: at least one");
    return o;
}

} // namespace

int main(int argc, char** argv) {
    // Every line reaches a log as it is printed, so a long run shows how far it got.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "stages") {
            if (argc < 3 || !std::strncmp(argv[2], "--", 2)) throw Usage("stages: the model file first");
            const Args a = parse_args(argc, argv, 3, {"--rows", "--prefill", "--context", "--steps"});
            if (a.devices.empty()) throw Usage("stages: one device or more");
            return stages(argv[2], a.devices, stages_options(a));
        }
        throw Usage(mode.empty() ? "no mode given" : mode + ": not a mode");
    } catch (const Usage& e) {
        std::fprintf(stderr, "llmx-multi-device-bench: %s\n%s", e.what(), kUsage);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "llmx-multi-device-bench: %s\n", e.what());
        return 1;
    }
}
