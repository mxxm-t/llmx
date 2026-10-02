#pragma once
#include <algorithm>
#include <chrono>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "backends/backend.hpp"
#include "backends/cpu/cpu_backend.hpp"
#include "core/host_memory.hpp"
#include "model/weights.hpp"
#include "model/architecture.hpp"
#include "model/runtime.hpp"
#include "model/layer_split.hpp"

// Where a model runs: what it asks of each device's memory, counted from its plan, and the one place a model is placed over the backends its caller made.

namespace infer {

// What a model asks of the memory of the devices it runs on, for a split fitted to them (model/layer_split.hpp), counted from its plan.
// A layer lists the tensors its roles take in the file's order, each once and a product where a role reads it as a matrix, whatever its rank; a tensor no role takes costs nothing.
// The embedding is the embed part's table, the output the head's matrix, tied when that role took its alias, and the output norm the head's norm; a pass role of any other part and kind, or a second role for one of those fields, has no field to count it in and is the plan's error.
// A layer's cache is counted by its kind, KV for every position the options budget and a state for every slot they give; activations are the plan's arena slots, and a handoff row is a residual row.
inline Footprint footprint(const ModelWeights& weights, const ModelPlan& plan, const ModelOptions& options) {
    auto matrix = [&](size_t i, bool product) {
        const TensorView& t = weights.tensors[i];
        Matrix w{t.type, t.shape.empty() ? 0 : (size_t)t.shape[0], 1, t.bytes, product};
        for (size_t d = 1; d < t.shape.size(); ++d) w.rows *= (size_t)t.shape[d];
        return w;
    };
    Footprint fp;
    fp.layers.resize(plan.layers.size());
    for (size_t l = 0; l < plan.layers.size(); ++l) {
        std::vector<std::pair<size_t, bool>> taken;
        for (const Role& role : plan.layers[l].roles)
            if (role.tensor) taken.push_back({*role.tensor, role.kind == RoleKind::matrix});
        std::sort(taken.begin(), taken.end());
        for (size_t k = 0; k < taken.size(); ++k) {
            if (k && taken[k].first == taken[k - 1].first) {
                fp.layers[l].back().product = fp.layers[l].back().product || taken[k].second;
                continue;
            }
            fp.layers[l].push_back(matrix(taken[k].first, taken[k].second));
        }
    }
    std::vector<const Matrix*> counted;
    for (const Role& role : plan.pass) {
        Matrix* field = nullptr;
        if (role.part == Part::embed && role.kind == RoleKind::gather) field = &fp.embedding;
        else if (role.part == Part::head && role.kind == RoleKind::matrix) field = &fp.output;
        else if (role.part == Part::head && role.kind == RoleKind::norm) field = &fp.output_norm;
        if (!field || std::find(counted.begin(), counted.end(), field) != counted.end())
            throw std::logic_error("footprint: no field of its own for the pass role " + role.name);
        counted.push_back(field);
        if (!role.tensor) continue;
        *field = matrix(*role.tensor, field == &fp.output);
        if (field == &fp.output) fp.tied = role.aliased;
    }
    fp.logits_per_row = fp.output.rows * sizeof(float);
    for (const LayerPlan& layer : plan.layers)
        fp.cache.push_back(layer.cache == Cache::kv      ? kv_tokens(plan, options) * kv_bytes_per_position(plan, options)
                           : layer.cache == Cache::state ? backend::size_add(plan.state.layer_bytes(backend::size_add(backend::size_add(options.state_slots, options.checkpoint_slots), options.mark_slots)),
                                                                             backend::size_mul(backend::size_mul(options.mark_slots, options.mark_rows), backend::size_mul(saved_floats(layer), sizeof(float))))
                                                         : 0);
    for (size_t n : plan.tables) fp.tables += n * sizeof(float);
    fp.handoff_per_row = plan.residual * sizeof(float);
    for (size_t w : plan.slots) fp.activations_per_row += w * sizeof(float);
    return fp;
}

// The placement a layer split describes: each layer's mixer and feed-forward block on the device that runs it, the embedding and the head where the split put them.
inline Placement placement_for(const LayerSplit& split) {
    Placement p;
    for (size_t d = 0; d < split.stages.size(); ++d)
        for (int i = 0; i < split.stages[d].count; ++i) {
            p.mixer_device.push_back((int)d);
            p.ffn_device.push_back((int)d);
        }
    p.embed_device = split.embed_device;
    p.output_device = split.output_device;
    return p;
}

// How a caller wants a model placed over the backends it made (docs/MULTI-DEVICE.md).
struct PlacementRequest {
    std::optional<backend::Dtype> dtype; // empty selects auto
    std::vector<std::string> names;   // each backend's name, for the fit's messages and its description
    std::vector<int> shares;          // each backend's proportion of the layers; empty to fit them to the devices' free memory
    int cpu_moe = 0;                  // with one backend, the routed layers whose experts run on the CPU beside it, -1 for every one
    size_t stream_from = 0;           // with experts on the CPU, the prompt length from which they are copied to the device (Placement::stream_from)
    int ubatch = 0;                   // prompt tokens a pass takes, kDefaultUbatch when 0
    size_t decode_rows = 0;           // generated tokens a pass may carry beside a prompt's: a server's decoding requests
    size_t slots = 0;                 // passes the caller keeps in flight (Model::reserve_passes), each with a handoff buffer on every stage but the last, two at least, which a split's fit counts
    size_t logit_rows = 0;            // rows of logits the caller keeps (Model::reserve_passes), which a split's fit counts; 0 for one per row of a pass
    // Histories the caller holds at once and the tokens each reaches, when it knows them, as bench does its sequences; zero leaves the options' budget as it is.
    // Each history takes whole blocks, up to the model's context, so the budget grows to hold them all where it would not.
    size_t histories = 0, history_tokens = 0;
    // The KV budget fitted to what the devices hold beside everything else, at most the options' budget, and backed whole at load (fitted_kv); a server's.
    bool fit_kv = false;
    // With fit_kv, the options' checkpoint slots are the most the fit gives rather than a number it must hold (fitted_kv).
    bool fit_checkpoints = false;
};

// The run's activation policy and each device's implementation, resolved once before model construction.
struct DtypePlan {
    std::optional<backend::Dtype> requested;
    backend::Dtype declared = backend::Dtype::bf16, effective = backend::Dtype::f32;
    struct Device {
        std::string name, how, paths;
        backend::Dtype effective = backend::Dtype::f32;
    };
    std::vector<Device> devices;
    const char* requested_name() const { return requested ? backend::dtype_name(*requested) : "auto"; }
    // The one line the CLI shows, including a warning when any device emulates or widens the request.
    std::string describe() const {
        std::string s = std::string("dtype: ") + requested_name() + " -> " + backend::dtype_name(effective) + " (model declares " + backend::dtype_name(declared) + ")";
        for (const Device& d : devices) s += "; " + d.name + " " + d.how + ": " + d.paths;
        if (std::any_of(devices.begin(), devices.end(), [](const Device& d) { return d.how != "native"; }))
            s += " (warning: emulated or wider fallback)";
        return s + "\n";
    }
};
inline DtypePlan resolve_dtype(backend::Dtype declared, const std::vector<backend::BackendPtr>& backends, const std::vector<std::string>& names,
                               std::optional<backend::Dtype> requested = {}) {
    if (backends.empty()) throw std::runtime_error("dtype: no device");
    std::vector<std::vector<backend::Dtype>> supported;
    for (const auto& b : backends) {
        if (!b) throw std::runtime_error("dtype: missing backend");
        supported.push_back(b->native_dtypes());
    }
    const auto native = [&](backend::Dtype d) {
        return std::all_of(supported.begin(), supported.end(), [d](const std::vector<backend::Dtype>& n) {
            return std::find(n.begin(), n.end(), d) != n.end();
        });
    };
    DtypePlan plan;
    plan.declared = declared;
    plan.requested = requested;
    plan.effective = backend::Dtype::f32;
    if (requested) plan.effective = *requested;
    else if (declared != backend::Dtype::f32 && native(declared)) plan.effective = declared;
    else
        for (backend::Dtype d : supported.front())
            if (native(d)) { plan.effective = d; break; }
    for (size_t i = 0; i < backends.size(); ++i) {
        backend::Dtype d = plan.effective;
        std::string how = "native";
        if (std::find(supported[i].begin(), supported[i].end(), d) == supported[i].end()) {
            if (backends[i]->emulates_dtype(d)) how = "emulated";
            else { d = backend::Dtype::f32; how = "fallback"; }
        }
        plan.devices.push_back({i < names.size() ? names[i] : "device " + std::to_string(i), how, backends[i]->dtype_path(d), d});
    }
    if (std::all_of(plan.devices.begin(), plan.devices.end(), [](const DtypePlan::Device& d) { return d.effective == backend::Dtype::f32; }))
        plan.effective = backend::Dtype::f32;
    return plan;
}

// A placed model and its resolved execution policy and placement description.
struct PlacedModel {
    std::unique_ptr<Model> model;
    std::string plan;
    DtypePlan dtype;
    size_t checkpoint_kv_tokens = 0;   // the KV tokens the fitted checkpoint slots took from the budget (fitted_kv)
};

// Whether the placement of `request` over `backends` adds a CPU backend for experts on the CPU, which it does beside one backend that is not the CPU.
inline bool adds_host_for_experts(const std::vector<backend::BackendPtr>& backends, const PlacementRequest& request) {
    return request.cpu_moe && backends.size() == 1 && request.shares.empty() && !backends[0]->is_cpu();
}

// Whether a backend of that placement reads weights in place: one of `backends`, or the CPU backend it adds for experts.
inline bool host_reads_in_place(const std::vector<backend::BackendPtr>& backends, const PlacementRequest& request) {
    return adds_host_for_experts(backends, request) ||
           std::any_of(backends.begin(), backends.end(), [](const backend::BackendPtr& b) { return b && b->reads_in_place(); });
}

// Whether the request puts routed layer l's feed-forward block on the CPU beside its one device: the first `cpu_moe` routed layers, or every one.
inline bool ffn_on_host(const PlacementRequest& request, const std::vector<LayerPlan>& layers, size_t l) {
    if (!layers[l].routed) return false;
    size_t before = 0;
    for (size_t i = 0; i < l; ++i) before += layers[i].routed;
    return request.cpu_moe < 0 || before < (size_t)request.cpu_moe;
}

// How long a fit the devices' free memory falls short of waits for that memory to settle: a read every kSettleWait, until kSettleQuiet reads in a row find it no higher or kSettleReads reads have passed.
// An MI50 gave back an ended 27B server's memory over three seconds in steps, holding it level for more than two seconds between them, so the quiet reads span five seconds and the bound thirty.
inline constexpr std::chrono::milliseconds kSettleWait{250};
inline constexpr int kSettleQuiet = 20;
inline constexpr int kSettleReads = 120;

// Whether a fit of `request` over `backends` waits for their free memory to stay level before it reads it: several backends, one a device that reports its free memory, whose layers or KV budget follow that memory (no shares given, or a fitted budget), since there a fit that holds does not show a card has given back an ended process's memory, another taking the layers it would hold.
inline bool level_first(const std::vector<backend::BackendPtr>& backends, const PlacementRequest& request) {
    if (backends.size() < 2 || (!request.shares.empty() && !request.fit_kv)) return false;
    return std::any_of(backends.begin(), backends.end(), [](const backend::BackendPtr& b) { return b && !b->is_cpu() && b->memory_available(); });
}

// A process that has just ended gives a device its memory back over a few seconds, so a fit that `settled` says falls short is tried again each time the free memory the devices report rises, until kSettleQuiet reads in a row find it no higher or kSettleReads reads have passed; `budgets` holds the last read.
// With `level` (level_first) the reads go on until that memory has risen no further for kSettleQuiet reads, or kSettleReads have passed, and `settled` is asked of that reading alone.
template <class Settled>
inline void settle(std::vector<DeviceBudget>& budgets, const std::vector<backend::BackendPtr>& backends, const std::vector<std::string>& names, Settled settled,
                   bool level = false) {
    if (!level && settled()) return;
    for (int read = 0, steady = 0; read < kSettleReads && steady < kSettleQuiet; ++read) {
        std::this_thread::sleep_for(kSettleWait);
        std::vector<DeviceBudget> again = budgets_for(backends, names);
        bool rose = false;
        for (size_t d = 0; d < again.size(); ++d) rose = rose || again[d].bytes.value_or(0) > budgets[d].bytes.value_or(0);
        budgets = std::move(again);
        steady = rose ? 0 : steady + 1;
        if (!level && rose && settled()) return;
    }
    if (level) settled();
}

// The options with the KV budget fitted (PlacementRequest::fit_kv): the most whole blocks of the largest block size, at most the options' budget, at which the fit of model/layer_split.hpp places the model on these devices and the host, and backed whole at load.
// Beside a device with experts on the CPU, the device holds neither those layers' feed-forward blocks nor their experts; a device that cannot tell its free memory takes the options' budget.
// Refused when not one block fits beside the weights, the activations and the recurrent state slots.
// With fit_checkpoints a model that keeps a state takes as checkpoint slots the fewer of the options' and the most at which the placement still holds three quarters of the budget it holds without them, so they take at most a quarter of the KV room, each count tried through the fit itself once the devices' free memory has settled (docs/SPECULATIVE.md, section 2); `given_up`, when given, gets the KV tokens the checkpoints took from the budget, and `read`, the devices' budgets the fit settled on, so a split places its layers by the same reading.
inline ModelOptions fitted_kv(const ModelWeights& weights, const ModelPlan& plan, const std::vector<backend::BackendPtr>& backends,
                              const PlacementRequest& request, ModelOptions options, size_t* given_up = nullptr,
                              std::vector<DeviceBudget>* read = nullptr) {
    size_t block = 1;
    for (const auto& b : backends) {
        if (!b) throw std::runtime_error("inference: missing backend");
        block = std::max(block, b->kv_layout().block_tokens);
    }
    ModelPlan held = plan;
    if (adds_host_for_experts(backends, request))
        for (size_t l = 0; l < held.layers.size(); ++l)
            if (ffn_on_host(request, plan.layers, l))
                for (Role& role : held.layers[l].roles)
                    if (role.part == Part::ffn) role.tensor.reset();
    std::vector<DeviceBudget> budgets = budgets_for(backends, request.names);
    const size_t rows = (size_t)(request.ubatch > 0 ? request.ubatch : kDefaultUbatch) + request.decode_rows;
    const std::optional<size_t> logits = request.logit_rows ? std::optional<size_t>(request.logit_rows) : std::nullopt;
    std::string why;
    size_t kept = options.checkpoint_slots;
    const auto fits = [&](size_t tokens) {
        ModelOptions o = options;
        o.kv_tokens = tokens;
        o.checkpoint_slots = kept;
        try {
            split_layers(footprint(weights, held, o), budgets, rows, request.shares, core::host_memory_available(), std::max<size_t>(1, request.slots), logits);
            return true;
        } catch (const std::runtime_error& e) {
            why = e.what();
            return false;
        }
    };
    const size_t want = kv_tokens(plan, options);
    // The options' budget where it fits, else the most whole blocks that fit, by bisection, since each more block only adds to what the devices hold; 0 when none does.
    const auto most = [&] {
        if (fits(want)) return want;
        size_t lo = 0, hi = want / block;
        while (lo < hi) {
            const size_t mid = lo + (hi - lo + 1) / 2;
            if (fits(mid * block)) lo = mid;
            else hi = mid - 1;
        }
        return lo * block;
    };
    // The budget without checkpoint slots first, read again while the devices' free memory rises, since a card still taking back an ended process's memory would leave room for none.
    const bool choose = request.fit_checkpoints && kept &&
                        std::any_of(plan.layers.begin(), plan.layers.end(), [](const LayerPlan& l) { return l.cache == Cache::state; });
    const size_t most_kept = kept;
    if (choose) kept = 0;
    size_t tokens = 0;
    settle(budgets, backends, request.names, [&] { return (tokens = most()) >= want; }, level_first(backends, request));
    // Then the checkpoint slots, by bisection, since each more slot only adds to what a device holds: the most at which the placement holds three quarters of the blocks it holds without them, rounded up, so the slots take at most a quarter of the KV room; none where even one does not fit.
    const size_t bare = tokens;
    if (choose && tokens) {
        const size_t blocks = tokens / block, target = std::max<size_t>(1, blocks - blocks / 4) * block;
        size_t lo = 0, hi = most_kept;
        while (lo < hi) {
            kept = lo + (hi - lo) / 2 + (hi - lo) % 2;
            if (fits(target)) lo = kept;
            else hi = kept - 1;
        }
        kept = lo;
        if (kept) tokens = most();
    }
    if (given_up) *given_up = bare - tokens;
    if (read) *read = budgets;
    if (!tokens) {
        if (!fits(1))
            throw std::runtime_error("placement: the model does not fit the devices' free memory even without its KV, which stayed level for five seconds (" + why + ")");
        fits(block);
        throw std::runtime_error("placement: no room for one KV block of " + std::to_string(block) +
                                 " tokens beside the weights, the activations and the recurrent states (" + why + ")");
    }
    options.kv_tokens = tokens;
    options.kv_backed = true;
    options.checkpoint_slots = kept;
    return options;
}

// The model over one backend, over one with the first `cpu_moe` routed layers' experts on the CPU beside it, or split by layers over several, with the request's ubatch set.
// The CPU is device 0 of an experts placement, so the thread count the model reports is the host's; the mixers, the dense blocks, the embedding and the head stay on the device.
inline PlacedModel place_model(const ModelWeights& weights, std::vector<backend::BackendPtr> backends, const PlacementRequest& request,
                               ModelOptions options, const AdoptWeight& adopt = {}) {
    if (backends.empty()) throw std::runtime_error("placement: no device");
    if (request.stream_from && !request.cpu_moe)
        throw std::runtime_error("--moe-stream-from: only experts on the CPU are streamed; give --n-cpu-moe or --cpu-moe");
    const ModelPlan plan = plan_model(weights);
    // A model without routed layers has no experts to put on the CPU, so every placement refuses the flags, on the CPU as beside a device.
    const std::string experts_flag = request.cpu_moe < 0 ? "--cpu-moe" : "--n-cpu-moe";
    if (request.cpu_moe && std::none_of(plan.layers.begin(), plan.layers.end(), [](const LayerPlan& l) { return l.routed; }))
        throw std::runtime_error(experts_flag + ": the model has no expert layers");
    // A storage has the blocks the budget fills at its backend's block size, and each history takes whole ones, so the request's histories are counted in each backend's blocks.
    // Where any storage would fall short, the budget becomes what they take in the largest blocks, which every other size divides, so every storage holds them and the fit counts them.
    // No history holds more than the model's context, so one that asks for more is counted at the context: the pool does not grow for tokens no run can hold, and the run is refused where it passes the context.
    if (request.histories) {
        const size_t budget = kv_tokens(plan, options), tokens = std::min(request.history_tokens, plan.context_length);
        size_t held = 0;
        bool short_of = false;
        for (const auto& b : backends) {
            if (!b) throw std::runtime_error("inference: missing backend");
            const size_t bt = b->kv_layout().block_tokens;
            const size_t blocks = backend::size_mul(request.histories, backend::blocks_for(tokens, bt));
            short_of = short_of || blocks > backend::blocks_for(budget, bt);
            held = std::max(held, backend::size_mul(blocks, bt));
        }
        if (short_of) options.kv_tokens = held;
    }
    PlacedModel placed;
    // A fitted budget's reading of the devices is the one the split places its layers by, so both see the same settled memory.
    std::vector<DeviceBudget> budgets;
    if (request.fit_kv) options = fitted_kv(weights, plan, backends, request, options, &placed.checkpoint_kv_tokens, &budgets);
    backend::BackendPtr host;
    std::vector<backend::BackendPtr> all = backends;
    std::vector<std::string> names = request.names;
    if (adds_host_for_experts(backends, request)) {
        host = backend::make_cpu_backend();
        all.insert(all.begin(), host);
        names.insert(names.begin(), "cpu");
    }
    DtypePlan dtype = resolve_dtype(weights.declared_dtype, all, names, request.dtype);
    options.dtype = dtype.effective;
    options.device_dtypes.clear();
    for (const auto& d : dtype.devices) options.device_dtypes.push_back(d.effective);
    if (backends.size() > 1 || !request.shares.empty()) {
        if (request.cpu_moe)
            throw std::runtime_error(experts_flag + ": not with several devices; list the CPU as a device to give it layers");
        const size_t rows = (size_t)(request.ubatch > 0 ? request.ubatch : kDefaultUbatch) + request.decode_rows;
        const Footprint fp = footprint(weights, plan, options);
        std::optional<LayerSplit> split;
        std::exception_ptr refused;
        auto fit = [&] {
            try {
                split = split_layers(fp, budgets, rows, request.shares, core::host_memory_available(), request.slots,
                                     request.logit_rows ? std::optional<size_t>(request.logit_rows) : std::nullopt);
                return true;
            } catch (const std::runtime_error&) {
                refused = std::current_exception();
                return false;
            }
        };
        if (request.fit_kv) {
            fit();
        } else {
            budgets = budgets_for(backends, request.names);
            settle(budgets, backends, request.names, fit, level_first(backends, request));
        }
        if (!split) std::rethrow_exception(refused);
        placed.model = std::make_unique<Model>(weights, plan, std::move(backends), placement_for(*split), options, adopt);
        placed.plan = split->describe(budgets);
    } else if (!adds_host_for_experts(backends, request)) {
        placed.model = std::make_unique<Model>(weights, plan, std::move(backends), Placement{}, options, adopt);
    } else {
        const size_t n_layer = plan.layers.size();
        Placement place;
        place.mixer_device.assign(n_layer, 1);
        place.ffn_device.assign(n_layer, 1);
        place.embed_device = place.output_device = 1;
        place.stream_from = request.stream_from;
        for (size_t l = 0; l < n_layer; ++l)
            if (ffn_on_host(request, plan.layers, l)) place.ffn_device[l] = 0;
        std::vector<backend::BackendPtr> both{std::move(host), std::move(backends[0])};
        placed.model = std::make_unique<Model>(weights, plan, std::move(both), place, options, adopt);
    }
    placed.model->set_ubatch(request.ubatch);
    placed.dtype = std::move(dtype);
    return placed;
}

} // namespace infer
