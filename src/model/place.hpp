#pragma once
#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
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
#include "model/shard.hpp"

// Where a model runs: what it asks of each device's memory, counted from its plan, and the one place a model is placed over the backends its caller made.

namespace infer {

// What a model asks of its devices' memory, for a split fitted to them (model/layer_split.hpp), counted from its plan: each layer's tensors once, the embedding and head, the caches by kind, the activations and the handoff rows.
// For member `member` of a tensor group of `width` (docs/TENSOR-SPLIT.md, section 4.6), split tensors, caches and logits count the member's share (model/shard.hpp), and the arena, handoff rows and mark rows stay one device's.
inline Footprint footprint(const ModelWeights& weights, const ModelPlan& plan, const ModelOptions& options, size_t width = 1, std::optional<size_t> named = std::nullopt) {
    // With no member named, each part is the largest any member's: shards are uneven where a tensor's blocks do not split evenly (ShardSection::align), and every member is fitted to one budget.
    if (width > 1 && !named) {
        Footprint fp = footprint(weights, plan, options, width, 0);
        const auto larger = [](Matrix& a, const Matrix& b) {
            if (b.bytes > a.bytes) a = b;
        };
        for (size_t m = 1; m < width; ++m) {
            const Footprint o = footprint(weights, plan, options, width, m);
            for (size_t l = 0; l < fp.layers.size(); ++l)
                for (size_t i = 0; i < fp.layers[l].size(); ++i) larger(fp.layers[l][i], o.layers[l][i]);
            for (size_t l = 0; l < fp.lookups.size(); ++l) {
                fp.lookups[l].rows_per_token = std::max(fp.lookups[l].rows_per_token, o.lookups[l].rows_per_token);
                fp.lookups[l].row_bytes = std::max(fp.lookups[l].row_bytes, o.lookups[l].row_bytes);
                fp.lookups[l].io_bytes = std::max(fp.lookups[l].io_bytes, o.lookups[l].io_bytes);
            }
            larger(fp.embedding, o.embedding);
            larger(fp.output, o.output);
            larger(fp.output_norm, o.output_norm);
            for (size_t i = 0; i < fp.head_aux.size(); ++i) larger(fp.head_aux[i], o.head_aux[i]);
            larger(fp.drafter_embedding, o.drafter_embedding);
            for (size_t i = 0; i < fp.drafter.size(); ++i) larger(fp.drafter[i], o.drafter[i]);
            for (size_t l = 0; l < fp.cache.size(); ++l) fp.cache[l] = std::max(fp.cache[l], o.cache[l]);
            fp.drafter_cache = std::max(fp.drafter_cache, o.drafter_cache);
            fp.head_slice_per_row = std::max(fp.head_slice_per_row, o.head_slice_per_row);
        }
        return fp;
    }
    const size_t member = named.value_or(0);
    auto matrix = [&](size_t i, bool product, const Role* role = nullptr) {
        const TensorView& t = weights.tensors[i];
        const bool split = role && width > 1 && role->shard.axis != Axis::none;
        const std::vector<uint64_t> shape = split ? shard::shape(*role, t, width, member) : t.shape;
        Matrix w{t.type, shape.empty() ? 0 : (size_t)shape[0], 1, split ? shard::held(*role, t, width, member).bytes : t.bytes, product};
        w.bytes = backend::size_add(w.bytes, backend::size_mul(weight_scale_count(weights, i), sizeof(float)));
        for (size_t d = 1; d < shape.size(); ++d) w.rows *= (size_t)shape[d];
        return w;
    };
    Footprint fp;
    fp.layers.resize(plan.layers.size());
    fp.lookups.resize(plan.layers.size());
    for (size_t l = 0; l < plan.layers.size(); ++l) {
        std::vector<std::pair<size_t, const Role*>> taken;
        const LayerPlan& layer = plan.layers[l];
        const Role* lookup_role = nullptr;
        if (layer.lookup) {
            for (const Role& role : layer.roles)
                if (role.id == layer.lookup->role) { lookup_role = &role; break; }
            if (lookup_role && lookup_role->tensor) {
                const TensorView& source = weights.tensors[*lookup_role->tensor];
                if (source.row_source) {
                    fp.lookups[l] = LookupMemory{layer.lookup->rows, source.row_source->row_bytes(), source.row_source->host_bytes()};
                    const size_t scales = weight_scale_count(weights, *lookup_role->tensor);
                    if (scales) fp.layers[l].push_back({quant::GGML_TYPE_F32, scales, 1, scales * sizeof(float), false});
                }
            }
        }
        for (const Role& role : layer.roles)
            if (role.tensor && role.kind != RoleKind::source && !(lookup_role == &role && weights.tensors[*role.tensor].row_source))
                taken.push_back({*role.tensor, &role});
        std::stable_sort(taken.begin(), taken.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (size_t k = 0; k < taken.size(); ++k) {
            const bool product = taken[k].second->kind == RoleKind::matrix;
            if (k && taken[k].first == taken[k - 1].first) {
                fp.layers[l].back().product = fp.layers[l].back().product || product;
                continue;
            }
            fp.layers[l].push_back(matrix(taken[k].first, product, taken[k].second));
        }
    }
    std::vector<const Matrix*> counted;
    bool head_output = false, head_norm = false;
    for (size_t i = 0; i < plan.pass.size(); ++i) {
        const Role& role = plan.pass[i];
        for (size_t j = 0; j < i; ++j)
            if (plan.pass[j].id == role.id) throw std::logic_error("footprint: repeated pass role " + role.name);
        Matrix* field = nullptr;
        if (role.part == Part::embed && role.kind == RoleKind::gather) field = &fp.embedding;
        else if (role.part == Part::head && role.kind == RoleKind::matrix && !head_output) {
            // Model::head_rows uses the first head matrix as the vocabulary logits weight.
            head_output = true;
            field = &fp.output;
        } else if (role.part == Part::head && role.kind == RoleKind::norm && !head_norm) {
            head_norm = true;
            field = &fp.output_norm;
        } else if (role.part == Part::head && (role.kind == RoleKind::matrix || role.kind == RoleKind::norm)) {
            fp.head_aux.push_back(role.tensor ? matrix(*role.tensor, role.kind == RoleKind::matrix, &role) : Matrix{});
            continue;
        }
        if (!field || std::find(counted.begin(), counted.end(), field) != counted.end())
            throw std::logic_error("footprint: no field of its own for the pass role " + role.name);
        counted.push_back(field);
        if (!role.tensor) continue;
        *field = matrix(*role.tensor, field == &fp.output, &role);
        // A member's head is its shard, a buffer of its own beside the embedding's table or, where the architecture splits it, the member's rows of it, so a group counts both.
        if (field == &fp.output) fp.tied = role.aliased && width == 1;
    }
    // The host's logits rows hold the whole vocabulary; on a tensor group each member of the head's group keeps its slice of them too, and off one the head's device keeps argmax passes' rows.
    fp.logits_per_row = plan.vocab * sizeof(float);
    if (width > 1) fp.head_slice_per_row = fp.output.rows * sizeof(float);
    else {
        fp.argmax_per_row = (plan.vocab + 1) * sizeof(float);
        fp.argmax_rows = argmax_rows(options);
    }
    // A member's caches hold its heads, one device's plan with the member's KV heads and state.
    ModelPlan heads;
    if (width > 1) {
        heads.kv_heads = shard::kv_heads(plan, width);
        heads.head_dim = plan.head_dim;
        heads.state = shard::state(plan, width);
    }
    const ModelPlan& kept = width > 1 ? heads : plan;
    // An embedded drafter's weights but those the head holds already, its embedding apart, and its KV layer, carried rows and the rows a mark saves.
    // A member holds its shards of the drafter's block and head (blocks::draft_head), each a buffer of its own beside the pass's, so the head's matrix counts here once more, or where it is tied with the embedding table.
    if (plan.drafter) {
        std::vector<size_t> held;
        for (const Role& role : plan.pass)
            if (role.part == Part::head && role.tensor && (width == 1 || role.shard.axis == Axis::none)) held.push_back(*role.tensor);
        std::optional<size_t> table;
        for (const Role& role : plan.drafter->roles)
            if (role.kind == RoleKind::gather) table = role.tensor;
        std::vector<std::pair<std::pair<size_t, bool>, const Role*>> taken;
        for (const Role& role : plan.drafter->roles) {
            if (!role.tensor || std::find(held.begin(), held.end(), *role.tensor) != held.end()) continue;
            if (role.kind == RoleKind::gather) {
                fp.drafter_embedding = matrix(*role.tensor, false);
                continue;
            }
            if (role.tensor == table) continue;
            taken.push_back({{*role.tensor, role.kind == RoleKind::matrix}, &role});
        }
        std::sort(taken.begin(), taken.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        taken.erase(std::unique(taken.begin(), taken.end(), [](const auto& a, const auto& b) { return a.first == b.first; }), taken.end());
        for (const auto& t : taken) fp.drafter.push_back(matrix(t.first.first, t.first.second, t.second));
        const size_t slots = backend::size_add(backend::size_add(options.state_slots, options.checkpoint_slots), options.mark_slots);
        fp.drafter_cache = backend::size_add(backend::size_mul(kv_tokens(plan, options), kv_bytes_per_position(kept, options, plan.drafter->kv_aux_floats())),
                                             backend::size_mul(backend::size_add(slots, backend::size_mul(options.mark_slots, options.mark_rows)), plan.residual * sizeof(float)));
        // A chain's ids and a member's logits of its vocabulary rows for a mark's sequences at the most drafts a mark holds (Model::reserve_drafting).
        if (options.mark_slots && options.mark_rows > 1) {
            const size_t ids = backend::size_mul(backend::size_mul(options.mark_slots, options.mark_rows), sizeof(uint32_t));
            const size_t vocab = width > 1 ? fp.output.rows : plan.vocab;
            const size_t logits = backend::size_mul(backend::size_mul(options.mark_slots, options.mark_rows - 1), backend::size_mul(vocab, sizeof(float)));
            fp.drafter_cache = backend::size_add(fp.drafter_cache, backend::size_add(ids, logits));
        }
    }
    {
        const bool states = std::any_of(plan.layers.begin(), plan.layers.end(), [](const LayerPlan& l) { return l.cache == Cache::state; });
        fp.context_rows = std::max(states && options.mark_slots ? options.mark_rows : 0, plan.drafter && options.mark_slots ? options.mark_slots : 0);
    }
    // Runtime gives every state layer on one device the largest saved and rerun row of that device. Before placement, charge the model-wide maxima for each state layer.
    size_t saved_max = 0, rerun_max = 0;
    if (options.mark_slots)
        for (const LayerPlan& layer : plan.layers)
            if (layer.cache == Cache::state) {
                saved_max = std::max(saved_max, shard::saved_floats(layer, width));
                rerun_max = std::max(rerun_max, recur_floats(plan, layer));
            }
    const bool reads = std::any_of(plan.layers.begin(), plan.layers.end(), [](const LayerPlan& l) { return l.entry_source >= 0; });
    for (const LayerPlan& layer : plan.layers) {
        size_t bytes = 0;
        // An entries layer takes its slots and its pooled rows, and saves nothing for a mark.
        if (keeps_kv(layer)) bytes = backend::size_mul(kv_tokens(plan, options), kv_bytes_per_position(kept, options, layer.kv_aux_floats()));
        if (reads) {
            fp.reads.push_back(layer.entry_source);
            fp.pooled.push_back(layer.cache == Cache::entries ? bytes : 0);
        }
        if (keeps_slot(layer)) {
            const size_t slots = backend::size_add(backend::size_add(options.state_slots, options.checkpoint_slots), options.mark_slots);
            bytes = backend::size_add(bytes, backend::size_mul(backend::size_mul(slots, backend::size_add(kept.state.slot_floats(), layer.state_tail)), sizeof(float)));
        }
        if (layer.cache == Cache::state) {
            bytes = backend::size_add(bytes, backend::size_mul(backend::size_mul(options.mark_slots, options.mark_rows), backend::size_mul(saved_max, sizeof(float))));
            if (options.mark_slots) bytes = backend::size_add(bytes, backend::size_mul(options.mark_rows, backend::size_mul(rerun_max, sizeof(float))));
        }
        fp.cache.push_back(bytes);
        if (layer.token_history) {
            const size_t slots = backend::size_add(backend::size_add(options.state_slots, options.checkpoint_slots), options.mark_slots);
            const size_t ids = backend::size_add(backend::size_mul(slots, layer.token_history), backend::size_mul(options.mark_slots, options.mark_rows));
            fp.host_state = backend::size_add(fp.host_state, backend::size_mul(ids, sizeof(uint32_t)));
        }
    }
    // The tables and the position slots cover the positions the budget reaches, so a smaller budget leaves room for more rows.
    const size_t positions = covered_positions(plan, kv_tokens(plan, options));
    for (size_t t = 0; t < plan.tables.size(); ++t) fp.tables = backend::size_add(fp.tables, backend::size_mul(table_floats(plan, t, positions), sizeof(float)));
    fp.handoff_per_row = backend::size_mul(handoff_width(plan), sizeof(float));
    fp.widths = plan.widths;
    fp.widths.heads = (fp.widths.heads + width - 1) / width;
    fp.widths.context = kv_tokens(plan, options);
    fp.widths.vocab = width > 1 ? fp.output.rows : plan.vocab;
    fp.host_scratch_per_row = backend::size_mul(backend::size_mul(plan.index_width, sizeof(uint32_t)), width);
    for (size_t s = 0; s < plan.slots.size(); ++s) fp.activations_per_row += slot_floats(plan, s, positions) * sizeof(float);
    // A member of a group that runs routed experts keeps a zero row to clear its partial rows from.
    if (width > 1 && (std::any_of(plan.layers.begin(), plan.layers.end(), has_experts) || (plan.drafter && has_experts(*plan.drafter)))) fp.activations_per_row += sum_width(plan) * sizeof(float);
    return fp;
}

// The placement a layer split describes: each layer's mixer and feed-forward block on the device that runs it, and the embedding and head where the split put them.
// Over tensor groups of `width` devices each stage is a group, named by its first member (Placement::width).
inline Placement placement_for(const LayerSplit& split, size_t width = 1) {
    Placement p;
    for (size_t d = 0; d < split.stages.size(); ++d)
        for (int i = 0; i < split.stages[d].count; ++i) {
            p.mixer_device.push_back((int)(d * width));
            p.ffn_device.push_back((int)(d * width));
        }
    p.embed_device = split.embed_device * (int)width;
    p.output_device = split.output_device * (int)width;
    p.width = width;
    return p;
}

// What a tensor split's fit places over (docs/TENSOR-SPLIT.md, section 4.6): each group of `width` consecutive devices as one device, named by its members, with its least member's free memory and its members' host memory.
// `summed` is the floats in a row the group sums (sum_width).
inline std::vector<DeviceBudget> group_budgets(const std::vector<DeviceBudget>& devices, size_t width, size_t summed) {
    if (width == 1) return devices;
    std::vector<DeviceBudget> groups;
    for (size_t g = 0; g + width <= devices.size(); g += width) {
        DeviceBudget b = devices[g];
        for (size_t m = g + 1; m < g + width; ++m) {
            b.name += "+" + devices[m].name;
            b.bytes = b.bytes && devices[m].bytes ? std::optional<size_t>(std::min(*b.bytes, *devices[m].bytes)) : std::nullopt;
            b.host_side = backend::size_add(b.host_side, devices[m].host_side);
            b.scratch = std::max(b.scratch, devices[m].scratch);
        }
        // Each member prices its own scratch, and the group fits its most.
        std::vector<std::function<size_t(const backend::PassWidths&)>> members;
        for (size_t m = g; m < g + width; ++m)
            if (devices[m].priced) members.push_back(devices[m].priced);
        b.priced = members.empty() ? nullptr : std::function<size_t(const backend::PassWidths&)>([members](const backend::PassWidths& w) {
            size_t most = 0;
            for (const auto& p : members) most = std::max(most, p(w));
            return most;
        });
        for (size_t m = g; m < g + width; ++m)
            if (devices[m].collective) b.collective_per_row = std::max(b.collective_per_row, devices[m].collective(width, summed));
        groups.push_back(std::move(b));
    }
    return groups;
}

// Where a tensor group of `width` consecutive devices spans PCI roots though the listed devices could form every group under one, the order that does: the devices of each root together, roots and devices in the order given.
// Empty where no group spans roots, where a device's root is unknown, or where a root's devices are not whole groups (docs/TENSOR-SPLIT.md, section 4.5).
inline std::vector<size_t> one_root_order(const std::vector<std::string>& roots, size_t width) {
    if (width < 2 || roots.size() % width) return {};
    bool spans = false;
    std::vector<std::string> seen;
    for (size_t d = 0; d < roots.size(); ++d) {
        if (roots[d].empty()) return {};
        spans = spans || roots[d] != roots[d - d % width];
        if (std::find(seen.begin(), seen.end(), roots[d]) == seen.end()) seen.push_back(roots[d]);
    }
    if (!spans) return {};
    std::vector<size_t> order;
    for (const std::string& r : seen) {
        if ((size_t)std::count(roots.begin(), roots.end(), r) % width) return {};
        for (size_t d = 0; d < roots.size(); ++d)
            if (roots[d] == r) order.push_back(d);
    }
    return order;
}

enum class LookupStorage { automatic, device, file };

inline const char* lookup_storage_name(LookupStorage storage) {
    switch (storage) {
    case LookupStorage::automatic: return "auto";
    case LookupStorage::device: return "device";
    case LookupStorage::file: return "file";
    }
    throw std::invalid_argument("lookup storage: unknown policy");
}

inline LookupStorage lookup_storage_of(std::string_view name) {
    for (LookupStorage storage : {LookupStorage::automatic, LookupStorage::device, LookupStorage::file})
        if (lookup_storage_name(storage) == name) return storage;
    throw std::invalid_argument("--lookup-storage: expected auto, device or file");
}

// A loader's bounded source for one declared lookup tensor; placement chooses storage before the loader materializes any whole table.
struct LookupCandidate {
    size_t tensor;
    std::shared_ptr<const WeightRows> file;
};

// The tokens whose distinct rows a table's hot-row store holds at most: where LRU met the unbounded hit rate on the measured row trace (docs/CACHE-PLAN.md, section 9).
inline constexpr size_t kRowStoreTokens = 1500000;

// One host budget split between the lookup tables' hot rows and a server's evicted prefixes (docs/CACHE-PLAN.md, section 9).
struct HostShares {
    size_t rows = 0, donors = 0;
};

// The rows take first, at most half the budget and at most `rows_cap`, and the donors the rest; `given` is --host-cache-bytes, without which the budget is half the host memory `available`, none where that is unknown.
inline HostShares host_shares(std::optional<size_t> given, std::optional<size_t> available, size_t rows_cap) {
    const size_t budget = given ? *given : available ? *available / 2 : 0;
    HostShares s;
    s.rows = std::min(rows_cap, budget / 2);
    s.donors = budget - s.rows;
    return s;
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
    size_t draft_rows = 0;            // draft rows a pass may carry beside a prompt's and the generated tokens': a server's verifies
    size_t slots = 0;                 // passes the caller keeps in flight (Model::reserve_passes), each with a handoff buffer on every stage but the last, two at least, which a split's fit counts
    size_t logit_rows = 0;            // rows of logits the caller keeps (Model::reserve_passes), which a split's fit counts; 0 for one per row of a pass
    // Histories the caller holds at once and the tokens each reaches, when it knows them, as bench does its sequences; zero leaves the options' budget as it is.
    // Each history takes whole blocks, up to the model's context, so the budget grows to hold them all where it would not.
    size_t histories = 0, history_tokens = 0;
    // Every placement fits the KV budget to what the devices hold beside everything else, at most the options' budget, and backs it whole at load (fitted_kv).
    // With kv_auto the budget is every whole block the devices' free memory holds, the drafter and its first mark in the fit, rather than the options' (a server without --kv-tokens).
    bool kv_auto = false;
    // The options' checkpoint slots are the most the fit gives rather than a number it must hold (fitted_kv).
    bool fit_checkpoints = false;
    // The options' mark slots past the first are the most the fit gives, only in the room the KV budget and the checkpoint slots leave (fitted_kv).
    bool fit_marks = false;
    // The file's embedded drafter planned and loaded beside the model (Architecture::plan_drafter, docs/SPECULATIVE.md, section 7).
    bool drafter = false;
    // Devices each layer is split across (docs/TENSOR-SPLIT.md): consecutive backends form groups of this many, and the groups are the stages of the layer split; 1 is the layer split alone.
    size_t width = 1;
    LookupStorage lookup_storage = LookupStorage::automatic;
    // The host budget the lookup tables' hot rows share with a server's evicted prefixes (host_shares): a server's --host-cache-bytes, none to take it from the host memory available.
    std::optional<size_t> host_budget;
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
    // The one line the CLI shows, a widened device with the dtype it runs, including a warning when any device emulates or widens the request.
    std::string describe() const {
        std::string s = std::string("dtype: ") + requested_name() + " -> " + backend::dtype_name(effective) + " (model declares " + backend::dtype_name(declared) + ")";
        for (const Device& d : devices)
            s += "; " + d.name + " " + d.how + (d.how == "fallback" ? std::string(" to ") + backend::dtype_name(d.effective) : std::string()) + ": " + d.paths;
        if (std::any_of(devices.begin(), devices.end(), [](const Device& d) { return d.how != "native"; }))
            s += " (warning: emulated or wider fallback)";
        return s + "\n";
    }
};
// A device without the requested dtype takes a fast exact emulation, else the next wider dtype it has, with a warning: f16 for int8, f32 otherwise (docs/PRECISION.md, rule 2).
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
            else {
                const bool f16 = std::find(supported[i].begin(), supported[i].end(), backend::Dtype::f16) != supported[i].end();
                d = d == backend::Dtype::int8 && f16 ? backend::Dtype::f16 : backend::Dtype::f32;
                how = "fallback";
            }
        }
        plan.devices.push_back({i < names.size() ? names[i] : "device " + std::to_string(i), how, backends[i]->dtype_path(d), d});
    }
    // Where every device widened to one dtype, that is the run's.
    const backend::Dtype first = plan.devices.front().effective;
    if (std::all_of(plan.devices.begin(), plan.devices.end(), [first](const DtypePlan::Device& d) { return d.effective == first; }))
        plan.effective = first;
    return plan;
}

// A placed model and its resolved execution policy and placement description.
struct PlacedModel {
    std::unique_ptr<Model> model;
    std::string plan;
    DtypePlan dtype;
    size_t checkpoint_kv_tokens = 0;   // the KV tokens the fitted checkpoint slots took from the budget (fitted_kv)
    std::string kv_kept_context;       // why an automatic budget (PlacementRequest::kv_auto) kept the model's context, empty where it read the devices
};

// A placement without model buffers: the loader attaches the selected row sources before constructing it once.
struct PlannedModel {
    ModelPlan architecture;
    ModelOptions options;
    std::vector<backend::BackendPtr> backends;
    Placement placement;
    std::vector<LookupStorage> lookups;
    std::string plan;
    DtypePlan dtype;
    size_t checkpoint_kv_tokens = 0;
    std::string kv_kept_context;
};

// Whether the placement of `request` over `backends` adds a CPU backend for experts on the CPU, which it does beside one backend that is not the CPU.
inline bool adds_host_for_experts(const std::vector<backend::BackendPtr>& backends, const PlacementRequest& request) {
    return request.cpu_moe && backends.size() == 1 && request.shares.empty() && !backends[0]->is_cpu();
}

// The rows of a pass the fit counts: a ubatch of prompt rows, the generated tokens' and a verify's drafts.
inline size_t fitted_rows(const PlacementRequest& request) {
    return (size_t)(request.ubatch > 0 ? request.ubatch : kDefaultUbatch) + request.decode_rows + request.draft_rows;
}

// The rows of the replies' arena a server keeps beside its passes' above one pass in flight: its generated tokens' and a verify's drafts (Scheduler, Model::reserve_passes).
inline size_t reply_rows(const PlacementRequest& request) { return request.slots > 1 ? request.decode_rows + request.draft_rows : 0; }


// Each tensor group's backends and names put in the order its first backend gives (Backend::member_order), so every member index, shard and name after it follows the devices (docs/TENSOR-SPLIT.md, section 4.1).
inline void order_members(std::vector<backend::BackendPtr>& backends, PlacementRequest& request, size_t summed) {
    const size_t w = request.width;
    for (size_t g = 0; w > 1 && g + w <= backends.size(); g += w) {
        std::vector<backend::Backend*> members;
        for (size_t m = g; m < g + w; ++m) {
            if (!backends[m]) throw std::runtime_error("inference: missing backend");
            members.push_back(backends[m].get());
        }
        const std::vector<size_t> order = backends[g]->member_order(members, fitted_rows(request), summed);
        std::vector<char> seen(w, 0);
        for (size_t k : order) {
            if (k >= w || seen[k]) throw std::logic_error("placement: a member order that is not an order of the group's members");
            seen[k] = 1;
        }
        if (order.size() != w) throw std::logic_error("placement: a member order that is not an order of the group's members");
        std::vector<backend::BackendPtr> held(backends.begin() + g, backends.begin() + g + w);
        for (size_t k = 0; k < w; ++k) backends[g + k] = held[order[k]];
        if (request.names.size() == backends.size()) {
            const std::vector<std::string> names(request.names.begin() + g, request.names.begin() + g + w);
            for (size_t k = 0; k < w; ++k) request.names[g + k] = names[order[k]];
        }
    }
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

// Whether a fit of `request` over `backends` waits for their free memory to stay level before reading it: several backends, one reporting free memory, whose layers or automatic KV budget follow it (no shares given, or kv_auto).
// A fit that holds there does not show a card has given back an ended process's memory, another taking the layers it would hold.
inline bool level_first(const std::vector<backend::BackendPtr>& backends, const PlacementRequest& request) {
    if (backends.size() < 2 || (!request.shares.empty() && !request.kv_auto)) return false;
    return std::any_of(backends.begin(), backends.end(), [](const backend::BackendPtr& b) { return b && !b->is_cpu() && b->memory_available(); });
}

// A process that has just ended gives a device its memory back over a few seconds, so a fit `settled` says falls short is retried while the free memory rises, until kSettleQuiet reads find it no higher or kSettleReads have passed.
// With `level` (level_first) the reads go on until that memory has risen no further, and `settled` is asked of that reading alone; `budgets` holds the last read.
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

// The automatic budget's search bound (PlacementRequest::kv_auto): the larger of the context and the whole blocks `free` bytes hold at `per_token` bytes a token, as many as an int32 id names at most.
// The context alone where the free memory is unknown or a token costs no KV bytes (a plan of states alone); never a number whose product with a layer's bytes a position passes what free memory states.
inline size_t kv_search_bound(size_t context, std::optional<size_t> free, size_t per_token, size_t block) {
    if (!free || !per_token || !block) return context;
    const size_t blocks = std::min({*free / per_token / block, (size_t)std::numeric_limits<int32_t>::max(), SIZE_MAX / block});
    return std::max(context, blocks * block);
}

// The options with the KV budget fitted, as every placement fits it: the most whole blocks of the largest size, at most the options' budget, at which the fit of model/layer_split.hpp places the model, refused when not one block fits.
// Drafting never lowers it: the drafter, the marks (fit_marks) and then the checkpoint slots (fit_checkpoints, `given_up`, `read`) are fitted beside that budget as docs/SPECULATIVE.md, sections 2 and 3 give.
inline ModelOptions fitted_kv(const ModelWeights& weights, const ModelPlan& plan, const std::vector<backend::BackendPtr>& backends,
                              const PlacementRequest& request, ModelOptions options, size_t* given_up = nullptr,
                              std::vector<DeviceBudget>* read = nullptr, std::string* kept_context = nullptr,
                              size_t host_bytes = 0, const ModelWeights* resident = nullptr) {
    size_t block = 1;
    for (const auto& b : backends) {
        if (!b) throw std::runtime_error("inference: missing backend");
        block = std::max(block, b->kv_layout().block_tokens);
    }
    // What the devices hold of a plan: beside experts on the CPU, not those layers' feed-forward blocks.
    const auto held_of = [&](ModelPlan p) {
        if (adds_host_for_experts(backends, request))
            for (size_t l = 0; l < p.layers.size(); ++l)
                if (ffn_on_host(request, plan.layers, l))
                    for (Role& role : p.layers[l].roles)
                        if (role.part == Part::ffn) role.tensor.reset();
        return p;
    };
    // The no-drafter fit reads the plan without the embedded drafter, whose roles, cache and arena slots only drafting takes.
    ModelPlan undrafted = plan.drafter ? plan_model(weights) : ModelPlan{};
    if (plan.drafter) settle_cache(undrafted, options);
    const ModelPlan held = held_of(plan), bare_plan = plan.drafter ? held_of(undrafted) : held;
    const ModelPlan* fitting = &bare_plan;
    // A request over one device may leave it unnamed; a split names every device (budgets_for).
    std::vector<std::string> names = request.names;
    if (names.empty() && backends.size() == 1) names.push_back("device");
    std::vector<DeviceBudget> budgets = budgets_for(backends, names);
    const size_t rows = fitted_rows(request);
    const std::optional<size_t> logits = request.logit_rows ? std::optional<size_t>(request.logit_rows) : std::nullopt;
    std::string why;
    size_t kept = options.checkpoint_slots;
    const auto fits = [&](size_t tokens) {
        ModelOptions o = options;
        o.kv_tokens = tokens;
        o.checkpoint_slots = kept;
        const auto holds = [&](const ModelWeights& source) {
            try {
                Footprint fp = footprint(source, *fitting, o, request.width);
                fp.host_state = backend::size_add(fp.host_state, host_bytes);
                split_layers(fp, group_budgets(budgets, request.width, sum_width(plan)), rows, request.shares, core::host_memory_available(),
                             std::max<size_t>(1, request.slots), logits, reply_rows(request));
                return true;
            } catch (const std::runtime_error& e) {
                why = e.what();
                return false;
            }
        };
        return holds(weights) || (resident && holds(*resident));
    };
    size_t want = kv_tokens(plan, options);
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
    // The budget without checkpoint slots first, on a reading taken once the devices' free memory has settled, since a card still taking back an ended process's memory would leave room for none.
    const bool keeps_state = std::any_of(plan.layers.begin(), plan.layers.end(), [](const LayerPlan& l) { return keeps_slot(l); });
    const bool choose = request.fit_checkpoints && kept && keeps_state;
    const size_t most_marks = options.mark_slots;
    const bool marks = request.fit_marks && most_marks > 1 && keeps_state;
    // The no-drafter fit: neither the embedded drafter nor a mark, which only drafting takes, so drafting never lowers the budget (docs/SPECULATIVE.md, section 3); only automatic checkpoint slots give way to it, below.
    const bool drafter = plan.drafter.has_value();
    options.mark_slots = 0;
    const size_t most_kept = kept;
    if (choose) kept = 0;
    // The automatic budget reads the devices' free memory, and keeps the model's context where one's is the host's (Backend::memory_is_host), which the host tier shares.
    const bool host_memory = std::any_of(backends.begin(), backends.end(), [](const backend::BackendPtr& b) { return b->memory_is_host(); });
    const bool unknown = std::any_of(budgets.begin(), budgets.end(), [](const DeviceBudget& b) { return !b.bytes; });
    const bool fills = request.kv_auto && !host_memory && !unknown;
    if (request.kv_auto && !fills && kept_context) *kept_context = host_memory ? "a device's free memory is the host's, which the host tier shares" : "a device reports no free memory";
    // Whether the devices hold the whole request: the options' budget beside every checkpoint and mark asked for and the embedded drafter.
    // Only then is there nothing a later reading could add, so a reading short of it waits for the free memory to settle, as a card giving back an ended process's memory may hold the budget alone and not what goes beside it.
    const auto whole = [&] {
        // An automatic budget over devices asks for more than they hold, so its reading always waits for their free memory to stay level.
        if (fills) return false;
        const ModelPlan* was = fitting;
        const size_t was_kept = kept, was_marks = options.mark_slots;
        fitting = &held;
        kept = most_kept;
        options.mark_slots = most_marks;
        const bool ok = fits(want);
        fitting = was;
        kept = was_kept;
        options.mark_slots = was_marks;
        return ok;
    };
    settle(budgets, backends, names, whole, level_first(backends, request));
    // The automatic budget: searched up to what the settled free memory holds at a token's KV bytes, with the drafter and its first mark in the fit (docs/CACHE-PLAN.md, section 7), so it never leaves the drafter no room.
    if (fills) {
        if (drafter || most_marks) {
            fitting = &held;
            options.mark_slots = marks ? 1 : most_marks;
        }
        const auto kv_bytes = [&](size_t tokens) {
            ModelOptions o = options;
            o.kv_tokens = tokens;
            o.checkpoint_slots = kept;
            const Footprint fp = footprint(weights, *fitting, o, request.width);
            size_t bytes = fp.drafter_cache;
            for (size_t c : fp.cache) bytes = backend::size_add(bytes, c);
            return bytes;
        };
        std::optional<size_t> free = size_t(0);
        for (const DeviceBudget& b : budgets) free = free && b.bytes ? std::optional<size_t>(*free > SIZE_MAX - *b.bytes ? SIZE_MAX : *free + *b.bytes) : std::nullopt;
        const size_t per_token = (kv_bytes(2 * block) - kv_bytes(block) + block - 1) / block;
        want = kv_search_bound(plan.context_length, free, per_token, block);
        if (kept_context && !per_token) *kept_context = "the model keeps no KV that grows with the pool";
    }
    size_t tokens = most();
    // Where not one block fits beside the hot rows of tables read from their files, those tables hold none and read every row from the file, as the lookup choice in place_model would leave them.
    if (!tokens) {
        bool held_rows = false;
        for (const TensorView& t : weights.tensors)
            if (t.row_source && t.row_source->store_bytes()) {
                t.row_source->hold(0);
                held_rows = true;
            }
        if (held_rows) tokens = most();
    }
    // Then the checkpoint slots, by bisection, since each more slot only adds to what a device holds: the most at which the placement holds three quarters of the blocks it holds without them, none where even one does not fit.
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
    // Then the drafter and a mark (or every mark asked for where marks are not fitted) beside that budget, which stays as it is: automatic checkpoint slots give way, checkpoints asked for by number do not, else a refusal names what took the room.
    if (tokens && (drafter || most_marks)) {
        fitting = &held;
        options.mark_slots = marks ? 1 : most_marks;
        if (!fits(tokens) && choose) {
            size_t lo = 0, hi = kept;
            while (lo < hi) {
                kept = lo + (hi - lo + 1) / 2;
                if (fits(tokens)) lo = kept;
                else hi = kept - 1;
            }
            kept = lo;
        }
        if (!fits(tokens))
            throw std::runtime_error("placement: drafting needs room beyond the KV budget of " + std::to_string(tokens) + " tokens and " + std::to_string(kept) +
                                     " state checkpoints, which leave none for it (" + why + "); a smaller budget or fewer checkpoints leaves the room");
    }
    // Then the marks past the first, by bisection, the most at which the budget and the checkpoint slots still fit.
    if (marks && tokens) {
        size_t lo = 1, hi = most_marks;
        while (lo < hi) {
            options.mark_slots = lo + (hi - lo) / 2 + (hi - lo) % 2;
            if (fits(tokens)) lo = options.mark_slots;
            else hi = options.mark_slots - 1;
        }
        options.mark_slots = lo;
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

// Gives each table read from its file its share of the rows' budget (host_shares) by its bytes a token, at most the rows of kRowStoreTokens tokens and every row the table has.
inline void hold_rows(const ModelPlan& plan, const std::vector<LookupCandidate>& candidates, std::optional<size_t> given) {
    std::vector<size_t> per_token(candidates.size(), 0), cap(candidates.size(), 0);
    size_t caps = 0;
    long double weight = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        for (const LayerPlan& layer : plan.layers)
            if (layer.lookup)
                for (const Role& role : layer.roles)
                    if (role.id == layer.lookup->role && role.tensor == candidates[i].tensor) per_token[i] += layer.lookup->rows;
        const WeightRows& t = *candidates[i].file;
        cap[i] = WeightRows::store_bytes_for(std::min(t.rows(), backend::size_mul(per_token[i], kRowStoreTokens)), t.row_bytes());
        caps = backend::size_add(caps, cap[i]);
        weight += (long double)per_token[i] * (long double)t.row_bytes();
    }
    // A table whose part reaches its cap takes the cap, and what it leaves goes to the others by their bytes a token.
    size_t left = host_shares(given, core::host_memory_available(), caps).rows;
    std::vector<size_t> give(candidates.size(), 0);
    std::vector<char> capped(candidates.size(), 0);
    for (bool again = true; again && weight > 0;) {
        again = false;
        for (size_t i = 0; i < candidates.size(); ++i) {
            if (capped[i]) continue;
            give[i] = size_t((long double)left * (long double)per_token[i] * (long double)candidates[i].file->row_bytes() / weight);
            if (give[i] < cap[i]) continue;
            give[i] = cap[i];
            capped[i] = 1;
            left -= cap[i];
            weight -= (long double)per_token[i] * (long double)candidates[i].file->row_bytes();
            again = true;
            break;
        }
    }
    for (size_t i = 0; i < candidates.size(); ++i) candidates[i].file->hold(give[i]);
}

// The placement over one backend, over one with the first `cpu_moe` routed layers' experts on the CPU beside it, or split by layers over several, before model buffers exist.
// The CPU is device 0 of an experts placement, so the thread count the model reports is the host's; the mixers, the dense blocks, the embedding and the head stay on the device.
inline PlannedModel plan_placement(const ModelWeights& weights, std::vector<backend::BackendPtr> backends, PlacementRequest request,
                                   ModelOptions options, const std::vector<LookupCandidate>& candidates = {}) {
    if (backends.empty()) throw std::runtime_error("placement: no device");
    if (!request.width || backends.size() % request.width)
        throw std::runtime_error("placement: a tensor width of " + std::to_string(request.width) + " needs whole groups of devices, and " + std::to_string(backends.size()) + " were given");
    if (request.width > 1 && request.cpu_moe) throw std::runtime_error(std::string(request.cpu_moe < 0 ? "--cpu-moe" : "--n-cpu-moe") + ": not with a tensor split");
    if (request.stream_from && !request.cpu_moe)
        throw std::runtime_error("--moe-stream-from: only experts on the CPU are streamed; give --n-cpu-moe or --cpu-moe");
    ModelPlan plan = plan_model(weights, request.drafter);
    settle_cache(plan, options);
    std::optional<ModelWeights> bounded;
    if (!candidates.empty()) {
        bounded = weights;
    }
    for (size_t i = 0; i < candidates.size(); ++i) {
        const LookupCandidate& candidate = candidates[i];
        if (candidate.tensor >= weights.tensors.size() || !candidate.file)
            throw std::runtime_error("placement: invalid lookup source");
        for (size_t j = 0; j < i; ++j)
            if (candidates[j].tensor == candidate.tensor) throw std::runtime_error("placement: repeated lookup candidate");
        const TensorView& tensor = weights.tensors[candidate.tensor];
        bool declared = false;
        for (const LayerPlan& layer : plan.layers)
            for (const Role& role : layer.roles)
                if (role.tensor == candidate.tensor) {
                    if (!layer.lookup || !layer.lookup->rows || layer.lookup->role != role.id || role.kind != RoleKind::gather || role.part != Part::mixer ||
                        role.stream != Stream::none || role.shard.axis != Axis::none || role.in != candidate.file->columns() || role.out != candidate.file->rows())
                        throw std::runtime_error("placement: candidate is not a declared lookup tensor: " + tensor.name);
                    declared = true;
                }
        for (const Role& role : plan.pass)
            if (role.tensor == candidate.tensor) throw std::runtime_error("placement: lookup candidate also has a pass role: " + tensor.name);
        if (plan.drafter)
            for (const Role& role : plan.drafter->roles)
                if (role.tensor == candidate.tensor) throw std::runtime_error("placement: lookup candidate also has a drafter role: " + tensor.name);
        if (!declared || tensor.shape.size() != 2 || tensor.shape[0] != candidate.file->columns() || tensor.shape[1] != candidate.file->rows() ||
            tensor.type != candidate.file->type() || tensor.bytes != candidate.file->bytes())
            throw std::runtime_error("placement: lookup source does not match tensor: " + tensor.name);
        bounded->tensors[candidate.tensor].row_source = candidate.file;
    }
    // A table the request puts on the devices reads nothing from its file, so only the others hold rows.
    if (request.lookup_storage != LookupStorage::device) hold_rows(plan, candidates, request.host_budget);
    // A width the model's shards cannot take is refused by its name here, before the fit reads a member's shards, and so is a malformed split.
    shard::check_plan(plan, weights.tensors, request.width);
    check_split(backends.size() / request.width, request.shares);
    order_members(backends, request, sum_width(plan));
    // A model without routed layers has no experts to put on the CPU, so every placement refuses the flags, on the CPU as beside a device.
    const std::string experts_flag = request.cpu_moe < 0 ? "--cpu-moe" : "--n-cpu-moe";
    if (request.cpu_moe && std::none_of(plan.layers.begin(), plan.layers.end(), [](const LayerPlan& l) { return l.routed; }))
        throw std::runtime_error(experts_flag + ": the model has no expert layers");
    // A storage has the blocks the budget fills at its backend's block size and each history takes whole ones, so the request's histories are counted in each backend's blocks.
    // Where any storage would fall short the budget becomes what they take in the largest blocks, and no history counts beyond the model's context.
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
    PlannedModel placed;
    placed.architecture = plan;
    placed.lookups.assign(candidates.size(), LookupStorage::file);
    if (request.lookup_storage == LookupStorage::device)
        for (size_t i = 0; i < candidates.size(); ++i) {
            bounded->tensors[candidates[i].tensor].row_source.reset();
            placed.lookups[i] = LookupStorage::device;
        }
    std::optional<ModelWeights> resident;
    if (!candidates.empty() && request.lookup_storage == LookupStorage::automatic) {
        resident = weights;
        for (const LookupCandidate& candidate : candidates) resident->tensors[candidate.tensor].row_source.reset();
    }
    // A lookup's host admission counts the expert payload already retained beside the device as well.
    size_t host_experts = 0;
    if (!candidates.empty() && adds_host_for_experts(backends, request)) {
        std::vector<size_t> counted;
        for (size_t l = 0; l < plan.layers.size(); ++l)
            if (ffn_on_host(request, plan.layers, l))
                for (const Role& role : plan.layers[l].roles)
                    if (role.part == Part::ffn && role.tensor && std::find(counted.begin(), counted.end(), *role.tensor) == counted.end()) {
                        counted.push_back(*role.tensor);
                        host_experts = backend::size_add(host_experts, weights.tensors[*role.tensor].bytes);
                        host_experts = backend::size_add(host_experts, backend::size_mul(weight_scale_count(weights, *role.tensor), sizeof(float)));
                    }
        host_experts = backend::size_mul(backend::blocks_for(host_experts, core::page_size()), core::page_size());
    }
    // The fitted budget's reading of the devices is the one the split places its layers by, so both see the same settled memory.
    std::vector<DeviceBudget> budgets;
    options = fitted_kv(bounded ? *bounded : weights, plan, backends, request, options, &placed.checkpoint_kv_tokens, &budgets,
                        &placed.kv_kept_context, host_experts, resident ? &*resident : nullptr);
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
    // All choices use the baseline fit's KV budget and the same reading of free memory; a table read from its file costs only its row staging.
    std::optional<LayerSplit> selected;
    if (!candidates.empty()) {
        ModelPlan held = plan;
        if (adds_host_for_experts(backends, request))
            for (size_t l = 0; l < held.layers.size(); ++l)
                if (ffn_on_host(request, plan.layers, l))
                    for (Role& role : held.layers[l].roles)
                        if (role.part == Part::ffn) role.tensor.reset();
        const std::optional<size_t> host_room = core::host_memory_available();
        const auto lookup_footprint = [&] {
            ModelWeights choice = *bounded;
            for (size_t i = 0; i < candidates.size(); ++i)
                choice.tensors[candidates[i].tensor].row_source = placed.lookups[i] == LookupStorage::device ? nullptr : candidates[i].file;
            Footprint fp = footprint(choice, held, options, request.width);
            fp.host_state = backend::size_add(fp.host_state, host_experts);
            return fp;
        };
        std::string why;
        const auto fits = [&] {
            try {
                selected = split_layers(lookup_footprint(), group_budgets(budgets, request.width, sum_width(plan)), fitted_rows(request), request.shares,
                                        host_room, std::max<size_t>(1, request.slots), request.logit_rows ? std::optional<size_t>(request.logit_rows) : std::nullopt,
                                        reply_rows(request));
                return true;
            } catch (const std::runtime_error& e) {
                selected.reset();
                why = e.what();
                return false;
            }
        };
        // A host whose budget has no room for the tables' hot rows reads every row from the file, holding none; where the fit fails without them too, they were not why, and the tables keep them.
        const auto without_rows = [&] {
            std::vector<size_t> held(candidates.size());
            bool any = false;
            for (size_t i = 0; i < candidates.size(); ++i) {
                held[i] = candidates[i].file->store_bytes();
                any = any || held[i];
                candidates[i].file->hold(0);
            }
            if (!any) return false;
            if (fits()) return true;
            for (size_t i = 0; i < candidates.size(); ++i) candidates[i].file->hold(held[i]);
            return false;
        };
        // Host staging can be the limiting budget even where the resident table costs more device memory.
        const auto initial_fit = [&] {
            if (fits()) return true;
            if (request.lookup_storage == LookupStorage::automatic) {
                const auto previous = placed.lookups;
                placed.lookups.assign(candidates.size(), LookupStorage::device);
                if (fits()) return true;
                placed.lookups = previous;
            }
            return without_rows();
        };
        initial_fit();
        if (!selected)
            throw std::runtime_error("placement: lookup storage " + std::string(lookup_storage_name(request.lookup_storage)) +
                                     " does not fit beside the KV budget of " + std::to_string(kv_tokens(plan, options)) + " tokens (" + why + ")");
        if (request.lookup_storage == LookupStorage::automatic) {
            for (size_t i = 0; i < candidates.size(); ++i) {
                placed.lookups[i] = LookupStorage::device;
                if (fits()) continue;
                placed.lookups[i] = LookupStorage::file;
                if (!fits()) throw std::logic_error("placement: an unchanged bounded lookup fit failed");
            }
        } else {
            placed.lookups.assign(candidates.size(), request.lookup_storage);
            if (!fits() && !without_rows())
                throw std::runtime_error("placement: lookup storage " + std::string(lookup_storage_name(request.lookup_storage)) +
                                         " does not fit beside the KV budget of " + std::to_string(kv_tokens(plan, options)) + " tokens (" + why + ")");
        }
        // A table on the devices reads nothing from its file, so it holds no rows in host memory.
        for (size_t i = 0; i < candidates.size(); ++i)
            if (placed.lookups[i] == LookupStorage::device) candidates[i].file->hold(0);
    }
    if (backends.size() > 1 || !request.shares.empty()) {
        if (request.cpu_moe)
            throw std::runtime_error(experts_flag + ": not with several devices; list the CPU as a device to give it layers");
        const size_t rows = fitted_rows(request);
        const Footprint fp = footprint(weights, plan, options, request.width);
        std::optional<LayerSplit> split;
        std::exception_ptr refused;
        auto fit = [&] {
            try {
                split = split_layers(fp, group_budgets(budgets, request.width, sum_width(plan)), rows, request.shares, core::host_memory_available(), request.slots,
                                     request.logit_rows ? std::optional<size_t>(request.logit_rows) : std::nullopt, reply_rows(request));
                return true;
            } catch (const std::runtime_error&) {
                refused = std::current_exception();
                return false;
            }
        };
        if (selected) split = selected;
        else fit();
        if (!split) std::rethrow_exception(refused);
        placed.placement = placement_for(*split, request.width);
        placed.plan = split->describe(group_budgets(budgets, request.width, sum_width(plan)));
    } else if (!adds_host_for_experts(backends, request)) {
        placed.placement = Placement{};
    } else {
        const size_t n_layer = plan.layers.size();
        Placement place;
        place.mixer_device.assign(n_layer, 1);
        place.ffn_device.assign(n_layer, 1);
        place.embed_device = place.output_device = 1;
        place.stream_from = request.stream_from;
        for (size_t l = 0; l < n_layer; ++l)
            if (ffn_on_host(request, plan.layers, l)) place.ffn_device[l] = 0;
        placed.placement = std::move(place);
    }
    placed.backends = std::move(all);
    placed.options = options;
    placed.dtype = std::move(dtype);
    return placed;
}

inline PlacedModel construct_model(const ModelWeights& weights, PlannedModel planned, int ubatch, const AdoptWeight& adopt = {}) {
    PlacedModel placed;
    placed.model = std::make_unique<Model>(weights, std::move(planned.architecture), std::move(planned.backends), planned.placement, planned.options, adopt);
    placed.model->set_ubatch(ubatch);
    placed.plan = std::move(planned.plan);
    placed.dtype = std::move(planned.dtype);
    placed.checkpoint_kv_tokens = planned.checkpoint_kv_tokens;
    placed.kv_kept_context = std::move(planned.kv_kept_context);
    return placed;
}

// Each device that runs a role makes its scratch at the widths its fit priced, before the first pass (Backend::reserve_scratch).
inline void reserve_scratch(const ModelWeights& weights, const PlannedModel& planned, const PlacementRequest& request) {
    const Placement& pl = planned.placement;
    const size_t width = std::max<size_t>(1, pl.width), rows = fitted_rows(request);
    const Footprint fp = footprint(weights, planned.architecture, planned.options, width);
    std::vector<bool> used(planned.backends.size(), false);
    const auto use = [&](int first) {
        for (size_t m = 0; m < width && size_t(first) + m < used.size(); ++m) used[size_t(first) + m] = true;
    };
    for (int d : pl.mixer_device) use(d);
    for (int d : pl.ffn_device) use(d);
    use(pl.embed_device);
    use(pl.output_device);
    const size_t head = size_t(pl.output_device);
    for (size_t d = 0; d < planned.backends.size(); ++d)
        if (used[d] && planned.backends[d])
            planned.backends[d]->reserve_scratch(pass_widths(fp, rows, request.logit_rows ? request.logit_rows : rows, d >= head && d < head + width));
}

// How many times build_fitted builds again with fewer KV tokens after a build ran a device short of what its reading stated.
inline constexpr int kShortRebuilds = 2;

// The KV bytes one block of `block` tokens takes on each of `planned`'s backends: each layer's on its mixer's device, every member of a tensor group, and the drafter's where the head runs.
inline std::vector<size_t> kv_block_bytes(const ModelWeights& weights, const PlannedModel& planned, size_t block) {
    const size_t width = std::max<size_t>(1, planned.placement.width);
    ModelOptions one = planned.options, two = planned.options;
    one.kv_tokens = block;
    two.kv_tokens = 2 * block;
    const Footprint a = footprint(weights, planned.architecture, one, width), b = footprint(weights, planned.architecture, two, width);
    std::vector<size_t> bytes(planned.backends.size(), 0);
    const auto add = [&](int first, size_t n) {
        for (size_t m = 0; m < width && size_t(first) + m < bytes.size(); ++m) bytes[size_t(first) + m] = backend::size_add(bytes[size_t(first) + m], n);
    };
    const std::vector<int>& mixers = planned.placement.mixer_device;
    for (size_t l = 0; l < a.cache.size() && l < b.cache.size(); ++l) add(l < mixers.size() ? mixers[l] : 0, b.cache[l] - a.cache[l]);
    add(planned.placement.output_device, b.drafter_cache - a.drafter_cache);
    return bytes;
}

// The model `planned` built, `allocating` running before the first build and as each build is given up, so the readings after it see none of what it recorded.
// With an automatic budget, a build that runs a device short is given up and the same placement built again with fewer KV tokens (docs/src/model-place.md).
inline PlacedModel build_fitted(const ModelWeights& weights, PlannedModel planned, const PlacementRequest& request, const AdoptWeight& adopt = {},
                                const std::function<void()>& allocating = {}) {
    const std::vector<backend::BackendPtr> backends = planned.backends;
    const auto device = [](const backend::BackendPtr& b) { return b && !b->is_cpu() && !b->memory_is_host(); };
    size_t block = 1;
    for (const backend::BackendPtr& b : backends)
        if (b) block = std::max(block, b->kv_layout().block_tokens);
    const std::vector<size_t> per_block = request.kv_auto ? kv_block_bytes(weights, planned, block) : std::vector<size_t>{};
    if (allocating) allocating();
    for (int rebuild = 0;; ++rebuild) {
        const bool again = request.kv_auto && rebuild < kShortRebuilds;
        std::vector<std::optional<size_t>> before(backends.size());
        for (size_t d = 0; again && d < backends.size(); ++d)
            if (device(backends[d])) before[d] = backends[d]->memory_available();
        // What each device holding KV was short by: what it keeps free (Backend::memory_kept) that a built model with its scratch made left it below, then how far its reading fell once the build was given up.
        std::vector<size_t> short_by(backends.size(), 0);
        std::exception_ptr failed;
        try {
            PlacedModel placed = construct_model(weights, planned, request.ubatch, adopt);
            reserve_scratch(weights, planned, request);
            if (!again) return placed;
            for (size_t d = 0; d < backends.size(); ++d) {
                if (!device(backends[d]) || !per_block[d]) continue;
                const std::optional<size_t> free = backends[d]->memory_available();
                const size_t kept = backends[d]->memory_kept(free.value_or(0));
                if (free && *free < kept) short_by[d] = kept - *free;
            }
            if (std::all_of(short_by.begin(), short_by.end(), [](size_t n) { return n == 0; })) return placed;
        } catch (const std::bad_alloc&) {
            if (!again) throw;
            failed = std::current_exception();
        }
        if (allocating) allocating();
        for (const backend::BackendPtr& b : backends) b->sync();
        size_t cut = 0;
        for (size_t d = 0; d < backends.size(); ++d) {
            if (!device(backends[d]) || !before[d]) continue;
            const std::optional<size_t> after = backends[d]->memory_available();
            const size_t fell = after && *after < *before[d] ? *before[d] - *after : 0, need = std::max(fell, short_by[d]);
            if (need && per_block[d]) cut = std::max(cut, backend::blocks_for(need, per_block[d]));
        }
        const size_t tokens = kv_tokens(planned.architecture, planned.options) / block;
        if (!cut && failed) std::rethrow_exception(failed);
        if (!cut || cut >= tokens)
            throw std::runtime_error("inference: the KV budget of " + std::to_string(tokens * block) + " tokens ran a device short once built, and no smaller budget fits");
        planned.options.kv_tokens = (tokens - cut) * block;
    }
}

// The model fitted (plan_placement) and built (build_fitted); `fitted` runs before the first build and as each build is given up, dropping what it recorded.
inline PlacedModel place_model(const ModelWeights& weights, std::vector<backend::BackendPtr> backends, const PlacementRequest& request,
                               ModelOptions options, const AdoptWeight& adopt = {}, const std::function<void()>& fitted = {}) {
    return build_fitted(weights, plan_placement(weights, std::move(backends), request, options), request, adopt, fitted);
}

} // namespace infer
