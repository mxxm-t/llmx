#pragma once
#include <algorithm>
#include <memory>
#include <functional>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>
#include <optional>

#include "format/gguf.hpp"
#include "backends/backend.hpp"
#include "model/weights.hpp"
#include "model/architecture.hpp"
#include "model/kv_cache.hpp"
#include "model/layer_split.hpp"
#include "backends/cpu/cpu_backend.hpp"

// The model runtime: sequences, passes over batches of them, stages over devices, the activation arena and the crossings between devices, running an architecture's plan and parts (model/architecture.hpp).
// The compute primitives are delegated to a backend::Backend, so the same code runs on every backend, and the math of each part to the architecture, so the runtime names none.

namespace infer {

// The plan of a model's weights: their tensors indexed once, a repeated name refused there, the architecture's plan over them, and each role's tensor, its name's or else its alias's, which the fit, the experts placement and the model all read.
// A plan whose slot 0 is not the residual's width, or with a role id past its row of weights, is the architecture's error.
inline ModelPlan plan_model(const ModelWeights& weights) {
    const TensorIndex tensors(weights.tensors);
    ModelPlan plan = weights.arch->plan(tensors);
    if (plan.slots.empty() || plan.slots[0] != plan.residual)
        throw std::logic_error("inference: a plan whose slot 0 is not the residual");
    auto resolve = [&](Role& role) {
        if (role.id >= plan.role_ids) throw std::logic_error("inference: a plan role's id past its row of weights " + role.name);
        role.tensor = tensors.find(role.name);
        if (!role.tensor && !role.alias.empty()) {
            role.tensor = tensors.find(role.alias);
            role.aliased = role.tensor.has_value();
        }
    };
    for (Role& role : plan.pass) resolve(role);
    for (LayerPlan& layer : plan.layers)
        for (Role& role : layer.roles) resolve(role);
    return plan;
}

class Model;

// Where each tensor role runs, as an index into the model's backends, with empty meaning everything on device 0.
// Per role rather than per layer, so a layer's mixer and its feed-forward block can sit on different devices, as expert offload places them (docs/EXECUTION.md).
struct Placement {
    std::vector<int> mixer_device, ffn_device;
    int embed_device = 0, output_device = 0;
    // A routed layer with its feed-forward block on a host and its mixer on a device runs a prompt of at least this many tokens on the device, its experts copied there for each pass: past some length a prompt's expert products on the host cost more than moving the experts.
    // By the prompt's whole length (BatchEntry::extent), so every row a prompt computes takes one path however the prompt is sliced or batched; a server forks a donor's rows only where that path is the new prompt's (row_class).
    // Zero keeps every run on the host, and neither a generated token nor a one-token prompt, both of extent 1, streams, so 1 streams what 2 does: one row cannot pay for moving a layer's experts.
    size_t stream_from = 0;
};

// Choices made once at construction, before the caches are allocated: how each cache side is stored (backend.hpp KVType, the CLI's --cache-type-k and --cache-type-v), the same on every backend or refused.
// The runtime's default cache type is set here and nowhere else: the CLI changes a side only when its flag is given.
struct ModelOptions {
    backend::KVType kv_k = backend::KVType::f16;
    backend::KVType kv_v = backend::KVType::f16;
    // Tokens the KV pool holds in total, shared by every sequence; zero means one model context, which is what one conversation needs and what a server divides among its requests unless told otherwise.
    size_t kv_tokens = 0;
    // Sequences that may hold a recurrent state at once, for a model whose layers keep one: each state storage holds this many slots from load on and never grows.
    size_t state_slots = 1;
    // States kept at a position beside those (Model::checkpoint), each a slot more in every state storage (docs/SPECULATIVE.md, section 2).
    size_t checkpoint_slots = 0;
    // Marks (Model::mark), each a slot more in every state storage and room for the recurrent inputs of `mark_rows` rows of every state layer, the most a pass after a mark may take.
    size_t mark_slots = 0, mark_rows = 0;
    // The whole KV budget backed as the model is made rather than as passes write it, so no pass grows the cache (a server's fitted budget, PlacementRequest::fit_kv).
    bool kv_backed = false;
};

// One request's history in a model's cache, made by Model::make_sequence for that model's pools and block sizes: the committed length of each stage, a block table per KV storage, and per device the ticket of the last pass that touched it, which a release waits on rather than draining the device (docs/EXECUTION.md).
// Movable, not copyable; from Model::begin_pass until its end_pass or abort_pass it is in flight, when no other pass, reset or fork takes it and it must not move, since the pass holds its address.
class Sequence {
public:
    Sequence() = default;
    // The first stage's committed length.
    // The stages can disagree while a pass is part way through them, and the model continues a history from the first stage's (Model::history).
    size_t length() const { return stage_length(0); }
    bool in_flight() const { return in_flight_; }
private:
    friend class Model;
    // Stage s's committed length, which has one owner: the stage's KV sequence where its layers keep KV, else its own count.
    size_t stage_length(size_t s) const {
        if (storage_of_.empty()) return 0;
        return storage_of_[s] >= 0 ? kv_[(size_t)storage_of_[s]].length() : length_[s];
    }
    std::vector<int> storage_of_;         // per stage, its KV storage, or -1 where its layers keep none
    std::vector<size_t> length_;          // per stage, the count of a stage without KV
    std::vector<KVSequence> kv_;          // per KV storage
    StateSlot state_;                     // its live slot in every state storage, from its first pass to its reset
    Checkpoint kept_;                     // its state kept at a position, which a fork or a retract continues from; one at most, a newer replacing it
    static constexpr size_t kLive = SIZE_MAX;
    std::vector<size_t> from_;            // per stage, the slot its state is read from: the live slot (kLive) or a checkpoint's
    // A mark (Model::mark): its hold on a buffer and maybe a slot, the history at `pos`, whose state is read from `from` on each stage while one pass runs past it, and whether that pass has been planned.
    struct Mark {
        MarkHold hold;
        bool ran = false;
        size_t pos = 0;
        std::vector<size_t> from;
        bool held() const { return hold.held(); }
    };
    Mark mark_;
    std::vector<backend::Ticket> last_;
    const Model* owner_ = nullptr;
    bool in_flight_ = false;
};

// What one sequence contributes to a pass: `n` tokens appended to `seq`, and whether the logits after its last token are wanted.
// A prefill microbatch is one entry with many tokens, a decode batch is many entries with one, and the two mix freely.
// A sequence appears in a batch at most once.
struct BatchEntry {
    Sequence* seq;
    const uint32_t* ids;
    size_t n;
    bool want_logits;
    // The logits after every token of the entry rather than only its last, for scoring a text through the same batched passes a prompt takes; with want_logits.
    bool every_logits = false;
    // What a device chooses this entry's kernels by (backend::RowRun), and a streamed layer its path (Placement::stream_from): for a prompt's rows the position one past the prompt's last token, for generated tokens 1 however many the entry carries, as a paused request's resume recomputes them.
    // Zero takes the entry's own row count.
    // A prompt given its extent takes the same kernels and path whether it arrives in one pass or in slices, alone or beside other sequences, with or without a reused prefix.
    size_t extent = 0;
    // Keep the state after the entry's last token as its sequence's checkpoint (Model::checkpoint), at a position of whole blocks in every storage; the pass takes a checkpoint slot for it.
    bool keep = false;
};

// What a pass's stages read as they are recorded: its entries, its rows and their positions, the rows the head reads, and each storage's cache views once its stage has reserved them.
// A context keeps one per pass it has in flight: one for a pass run whole, one per stage while a prompt's chunks flow through the stages together, one per slot of a context reserved for passes (Model::reserve_passes).
struct Pass {
    std::vector<BatchEntry> entries;
    std::vector<size_t> start;                         // per entry, the history the pass found, which a failed pass returns to
    size_t rows = 0, want = 0;
    bool long_runs = false;                            // some entry takes its streamed layers on the device
    std::vector<uint32_t> ids, pos, pick;
    std::vector<std::vector<backend::KVView>> views;   // per storage, per entry
    std::vector<std::vector<backend::StateView>> states;   // per device, per entry, on a device whose layers keep a state
    std::vector<Checkpoint> kept;                      // per entry, the checkpoint a keep entry writes, its sequence's once the last stage commits it
    std::vector<backend::RowRun> runs, head_runs;      // the pass's rows and the head's, by entry
    size_t handoff = 0;                                // which of each device's handoff buffers its crossings use: a prompt chunk's parity, a reserved pass's slot
    size_t logits_base = 0;                            // the context's logits row its head writes first
    size_t at = 0;                                     // the device its residual left the last stage from
    backend::Ticket sent = 0;                          // the submission that copied it out, after the last stage the head's
    bool in_flight = false;                            // a slot's pass between begin_pass and end_pass or abort_pass
    size_t ran = 0;                                    // the stages run_pass_stage has recorded
};

// Where a context's passes run, as plain data Model fills: an activation arena per device, which each device's passes use in turn, host-visible handoff buffers on each device a crossing leaves, the host-visible logits rows on the output device, and the tickets of the submissions.
// A forward grows it to the largest pass seen, while Model::reserve_passes sizes it once for passes in flight, each slot with its own handoff buffers and logits rows, and replaces nothing after that.
struct ExecContext {
    // Row i of the logits the last forward produced, in entry order, valid until the next forward through this context.
    // The first read waits on the pass's ticket, so forward itself never blocks: a caller with two contexts submits the next pass before it reads this one.
    const float* logits(size_t i) {
        if (!logits_buf || i >= n_logits)
            throw std::out_of_range("inference: no such logits row");
        if (pending) { backend->wait(ticket); pending = false; }
        const void* p = logits_buf->host_ptr();
        if (!p) throw std::runtime_error("inference: logits are not host visible");
        return (const float*)p + i * width;
    }
    size_t n_logits = 0;
    size_t width = 0;
    backend::Ticket ticket = 0;
    backend::Backend* backend = nullptr;
    bool pending = false;

    struct Scratch {
        backend::BufferPtr arena;
        size_t rows = 0;
        std::vector<size_t> offset;            // bytes, per slot of the plan
    };
    std::vector<Scratch> scratch;              // per device
    backend::BufferPtr logits_buf;
    size_t logit_rows = 0;
    std::vector<std::vector<backend::BufferPtr>> handoff;   // per device
    size_t handoff_rows = 0;
    std::vector<Pass> passes;
    size_t slots = 0, pass_rows = 0;           // what reserve_passes froze it for: its pass slots and the rows a pass may take; zero slots while it grows
    std::vector<backend::RowRun> part_runs;    // a streamed layer's group of entries, rebased
    std::vector<backend::RowRun> entry_runs;   // the run list a part may rebuild (Step::scratch)
    std::vector<backend::Ticket> tickets;      // per device
};

// Prompt tokens a pass takes by default (Model::set_ubatch), and so the prompt rows a placement is fitted for.
inline constexpr int kDefaultUbatch = 512;

// The positions a model's caches are budgeted for: the options' tokens, else the whole context.
inline size_t kv_tokens(const ModelPlan& plan, const ModelOptions& options) {
    return options.kv_tokens ? options.kv_tokens : plan.context_length;
}

// The bytes one position of one layer's cache takes, key and value, at the options' cache types.
inline size_t kv_bytes_per_position(const ModelPlan& plan, const ModelOptions& options) {
    return plan.kv_heads * plan.head_dim * (backend::kv_elem_bytes(options.kv_k) + backend::kv_elem_bytes(options.kv_v));
}

class Model {
public:
    // Construct from a model's weights on one backend (defaults to the CPU backend), or over several with a placement of every role, each weight put on the backend that hosts it by `adopt`, planned here or given the plan plan_model made of these weights.
    // The views are not kept; the bytes a backend adopted in place must outlive the Model (Backend::adopt).
    explicit Model(const ModelWeights& weights,
                   backend::BackendPtr backend = backend::make_cpu_backend(),
                   ModelOptions options = ModelOptions{})
        : Model(weights, std::vector<backend::BackendPtr>{std::move(backend)}, Placement{}, options) {}
    Model(const ModelWeights& weights, std::vector<backend::BackendPtr> backends,
          Placement placement, ModelOptions options = ModelOptions{}, const AdoptWeight& adopt = {})
        : Model(weights, plan_model(weights), std::move(backends), std::move(placement), options, adopt) {}
    Model(const ModelWeights& weights, ModelPlan plan, std::vector<backend::BackendPtr> backends,
          Placement placement, ModelOptions options = ModelOptions{}, const AdoptWeight& adopt = {})
        : place_(std::move(placement)), options_(options), arch_(weights.arch), plan_(std::move(plan)) {
        if (backends.empty()) throw std::runtime_error("inference: missing backend");
        for (const auto& b : backends)
            if (!b) throw std::runtime_error("inference: missing backend");
        const size_t n_layer = plan_.layers.size();

        if (place_.mixer_device.empty() && place_.ffn_device.empty()) {
            place_.mixer_device.assign(n_layer, 0);
            place_.ffn_device.assign(n_layer, 0);
        }
        if (place_.mixer_device.size() != n_layer ||
            place_.ffn_device.size() != n_layer)
            throw std::runtime_error("inference: placement does not cover every layer");
        auto device_index = [&](int d) {
            if (d < 0 || (size_t)d >= backends.size())
                throw std::runtime_error("inference: placement names a device the model does not have");
            return (size_t)d;
        };
        device_index(place_.embed_device);
        device_index(place_.output_device);
        auto accepts = [&](const Role& role, size_t device) {
            return !role.tensor || backends[device]->supports_type(weights.tensors[*role.tensor].type);
        };
        auto require_type = [&](const Role& role, size_t device, const std::string& part) {
            if (!accepts(role, device)) {
                const TensorView& t = weights.tensors[*role.tensor];
                const quant::QuantType* q = quant::Registry::instance().get(t.type);
                const std::string type = q ? std::string(q->name) + " (" + std::to_string(t.type) + ")" : std::to_string(t.type);
                throw std::runtime_error("inference: " + part + " needs tensor " + t.name + " of type " + type +
                                         ", which the backend of device " + std::to_string(device) + " does not support");
            }
        };
        for (const Role& role : plan_.pass)
            require_type(role, device_of(role.part, 0), role.part == Part::embed ? "embedding" : "head");
        // Home weights must run on their assigned device before anything is adopted; a stream destination that lacks a weight type leaves the whole layer at home.
        stream_device_.assign(n_layer, -1);
        for (size_t l = 0; l < n_layer; ++l) {
            const LayerPlan& layer = plan_.layers[l];
            const size_t a = device_index(place_.mixer_device[l]), f = device_index(place_.ffn_device[l]);
            // Experts read in place on the host beside a mixer on a device that copies its weights.
            bool streamed = layer.routed && place_.stream_from && a != f && !backends[a]->reads_in_place() && backends[f]->reads_in_place();
            for (const Role& role : layer.roles) {
                require_type(role, device_of(role.part, l), "layer " + std::to_string(l) + "'s " +
                             (role.part == Part::mixer ? "mixer" : "feed-forward part"));
                if (role.stream != Stream::none && !accepts(role, a)) streamed = false;
            }
            // Every op a part issues beyond the common set runs on its device; a stream destination without one of the feed-forward part's leaves the layer at home, as a type does.
            for (const OpUse& u : layer.ops) {
                if (u.part == Part::mixer) {
                    if (!backends[a]->implements(u.op)) refuse_op(l, u);
                    continue;
                }
                if (!backends[f]->implements(u.op)) refuse_op(l, u);
                if (!backends[a]->implements(u.op)) streamed = false;
            }
            if (streamed) stream_device_[l] = (int)a;
        }
        devices_.reserve(backends.size());
        for (auto& b : backends) {
            devices_.push_back(std::make_unique<Device>());
            devices_.back()->b = std::move(b);
            devices_.back()->local_layer.assign(n_layer, -1);
        }
        devices_[(size_t)place_.embed_device]->used = true;
        devices_[(size_t)place_.output_device]->used = true;
        for (int l = 0; l < (int)n_layer; ++l) {
            Device& a = *devices_[device_index(place_.mixer_device[(size_t)l])];
            ++a.mixer_layers;
            if (plan_.layers[(size_t)l].cache == Cache::kv) {
                a.local_layer[(size_t)l] = a.kv_layers++;
                ++kv_layers_;
            } else if (plan_.layers[(size_t)l].cache == Cache::state) {
                a.local_layer[(size_t)l] = a.state_layers++;
                ++state_layers_;
            }
            a.used = true;
            devices_[device_index(place_.ffn_device[(size_t)l])]->used = true;
        }
        // A device's mixer layers are one run, so each storage is written by one stage, which reserves and commits it once a pass.
        for (int l = 0; l < (int)n_layer; ++l) {
            const size_t a = (size_t)place_.mixer_device[(size_t)l];
            if (!stages_.empty() && stages_.back().device == a) { stages_.back().end = l + 1; continue; }
            for (const Stage& st : stages_)
                if (st.device == a) throw std::runtime_error("inference: a device's attention layers must be consecutive");
            stages_.push_back(Stage{a, l, l + 1, {}});
        }
        for (size_t s = 0; s < stages_.size(); ++s) {
            Stage& st = stages_[s];
            auto touch = [&](size_t d) {
                if (std::find(st.touches.begin(), st.touches.end(), d) == st.touches.end()) st.touches.push_back(d);
            };
            touch(st.device);
            for (int l = st.first; l < st.end; ++l) touch((size_t)place_.ffn_device[(size_t)l]);
            if (s == 0) touch((size_t)place_.embed_device);
            if (s + 1 == stages_.size()) touch((size_t)place_.output_device);
        }
        // A prompt's chunks flow through the stages together when nothing crosses inside a stage: the embedding on the first stage's device, the head on the last's, every feed-forward block beside its mixer.
        pipelined_ = stages_.size() > 1 && place_.embed_device == (int)stages_.front().device &&
                     place_.output_device == (int)stages_.back().device;
        for (size_t s = 0; pipelined_ && s < stages_.size(); ++s) {
            for (int l = stages_[s].first; l < stages_[s].end; ++l)
                pipelined_ = pipelined_ && place_.ffn_device[(size_t)l] == (int)stages_[s].device;
        }
        // The residual leaves a device wherever the next role on its path (the embedding, each layer's mixer and feed-forward block, the head) sits on another.
        // A feed-forward block away from its mixer sends too, since a streamed layer's host rows cross back from it to the mixer's device.
        size_t at = (size_t)place_.embed_device;
        for (int l = 0; l < (int)n_layer; ++l) {
            const size_t a = (size_t)place_.mixer_device[(size_t)l], f = (size_t)place_.ffn_device[(size_t)l];
            if (a != at) devices_[at]->sends = true;
            if (f != a) devices_[a]->sends = devices_[f]->sends = true;
            at = f;
        }
        if ((size_t)place_.output_device != at) devices_[at]->sends = true;

        try {
            resolve_tensors(weights, adopt);

            // Each device whose mixer layers keep KV gets a storage for exactly those layers, with its own block size and pool.
            // Budget: the option's tokens, else the whole context; storage is backed on demand, so a short chat does not allocate it, unless the options back it whole.
            const size_t budget = kv_tokens(plan_, options_);
            for (auto& dp : devices_) {
                Device& d = *dp;
                if (!d.kv_layers) continue;
                // A shared prefix ends on a whole block of the largest size (kv_block_tokens), which is whole in every storage only when the sizes nest.
                for (const Device* other : storages_) {
                    const size_t a = d.b->kv_layout().block_tokens, b = other->b->kv_layout().block_tokens;
                    if (std::max(a, b) % std::min(a, b))
                        throw std::runtime_error("inference: cache blocks of " + std::to_string(a) + " and " + std::to_string(b) +
                                                 " tokens in one model; a split needs one size to divide the other");
                }
                d.storage = d.b->kv_alloc((size_t)d.kv_layers, plan_.kv_heads, plan_.head_dim,
                                          budget, options_.kv_k, options_.kv_v);
                if (options_.kv_backed) d.storage->back_all();
                d.pool.configure(d.storage->max_blocks());
                d.storage_index = (int)storages_.size();
                storages_.push_back(&d);
            }
            // Each device whose mixer layers keep a state holds every slot of theirs from now on, zeroed, so no pass allocates state.
            if (state_layers_) {
                const size_t slots = backend::size_add(backend::size_add(options_.state_slots, options_.checkpoint_slots), options_.mark_slots);
                for (size_t di = 0; di < devices_.size(); ++di) {
                    Device& d = *devices_[di];
                    if (!d.state_layers) continue;
                    d.states = d.b->state_alloc((size_t)d.state_layers, slots, plan_.state);
                    // Each mark's room for the recurrent inputs of its rows, per state layer of the device.
                    if (!options_.mark_slots) continue;
                    for (size_t l = 0; l < n_layer; ++l)
                        if (place_.mixer_device[l] == (int)di && plan_.layers[l].cache == Cache::state)
                            d.saved_floats = std::max(d.saved_floats, saved_floats(plan_.layers[l]));
                    if (!d.saved_floats) throw std::logic_error("inference: a state layer that saves no inputs for a mark");
                    d.saved = d.b->alloc(backend::size_mul(backend::size_mul(backend::size_mul(options_.mark_slots, (size_t)d.state_layers),
                                                                             backend::size_mul(options_.mark_rows, d.saved_floats)), sizeof(float)));
                }
                slots_.configure(options_.state_slots, options_.checkpoint_slots, options_.mark_slots);
            }
            seq_ = make_sequence();

            // The position tables, sized by the plan and filled once by the architecture.
            // Every device that runs a mixer reads them through adopted buffers, so the host vectors stay alive for the model's lifetime; on CPU that is the same memory.
            tables_.resize(plan_.tables.size());
            for (size_t t = 0; t < tables_.size(); ++t) tables_[t].assign(plan_.tables[t], 0.0f);
            arch_->fill_tables(tables_);
            for (auto& d : devices_)
                if (d->mixer_layers)
                    for (const std::vector<float>& t : tables_) d->tables.push_back(d->b->adopt(t.data(), t.size() * sizeof(float)));
        } catch (...) {
            // Constructor members still exist here, so pending uploads retire before unwinding releases them.
            retire();
            throw;
        }
    }

    ~Model() { retire(); }

    // Sequences hold the pools' addresses, so a Model is neither copied nor moved.
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // CPU worker counts, applied to every backend; a device backend ignores them.
    void set_threads(int n) { for (auto& d : devices_) d->b->set_threads(n); }

    // The cache pools a scheduler admits against, one per device whose mixer layers keep KV, each counted in its own blocks (docs/SERVER.md, docs/MULTI-DEVICE.md).
    size_t kv_pools() const { return storages_.size(); }
    size_t kv_pool_block_tokens(size_t s) const { return storages_.at(s)->b->kv_layout().block_tokens; }
    size_t kv_pool_blocks(size_t s) const { return storages_.at(s)->pool.max_blocks(); }
    // Tokens every pool can hold.
    size_t kv_tokens_total() const {
        size_t least = std::numeric_limits<size_t>::max();
        for (size_t s = 0; s < storages_.size(); ++s) least = std::min(least, kv_pool_blocks(s) * kv_pool_block_tokens(s));
        return storages_.empty() ? 0 : least;
    }
    // The largest block of any pool: a reusable prefix ends on a whole one, which is whole in every pool because the sizes nest.
    size_t kv_block_tokens() const {
        size_t largest = 1;
        for (const Device* d : storages_) largest = std::max(largest, d->b->kv_layout().block_tokens);
        return largest;
    }
    size_t n_vocab() const { return plan_.vocab; }
    size_t prefill_batch() const { return (size_t)ubatch_; }
    // The host's count wherever it sits among the devices; a device backend reports 0.
    int threads_available() const {
        int n = 0;
        for (const auto& d : devices_) n = std::max(n, d->b->threads_available());
        return n;
    }
    // 0 keeps the default.
    // Sets how a prompt is chunked; storage follows the passes actually run.
    void set_ubatch(int n) { if (n > 0) ubatch_ = n; }

    int n_tokens() const { return (int)seq_.length(); }
    int context_length() const { return (int)plan_.context_length; }

    // A second history holding the first `length` tokens of `src`, which must be whole blocks in every storage: every block below `length` is shared, read-only from now on, and the fork appends into fresh ones, so nothing is allocated or copied here.
    // The fork inherits the tickets of the passes that wrote what it shares.
    // A recurrent state exists only at the end of what it has read, so on a model that keeps one the fork takes src's checkpoint at `length`, whose state its first pass reads in place.
    Sequence fork(const Sequence& src, size_t length) {
        if (src.owner_ != this) throw std::runtime_error("inference: sequence of another model");
        if (src.in_flight_) throw std::logic_error("inference: a fork of a sequence in flight");
        if (src.mark_.held()) throw std::logic_error("inference: a fork of a marked sequence");
        if (state_layers_ && (!src.kept_.held() || src.kept_.pos() != length))
            throw std::logic_error("inference: a fork of a model whose layers keep a recurrent state takes its source's checkpoint");
        if (length > src.length()) throw std::logic_error("KV cache: a fork takes whole blocks of the history");
        Sequence f;
        f.storage_of_ = src.storage_of_;
        f.length_.assign(stages_.size(), length);
        f.kv_.reserve(storages_.size());
        for (const KVSequence& kv : src.kv_) f.kv_.push_back(kv.fork(length));
        f.last_ = src.last_;
        f.owner_ = this;
        f.from_.assign(stages_.size(), Sequence::kLive);
        if (state_layers_) {
            f.kept_ = src.kept_;
            std::fill(f.from_.begin(), f.from_.end(), f.kept_.slot());
        }
        return f;
    }

    // A fresh history over this model's cache: a table per KV storage, and a count for each stage without one.
    Sequence make_sequence() {
        Sequence s;
        for (const Stage& st : stages_) s.storage_of_.push_back(devices_[st.device]->storage_index);
        s.length_.assign(stages_.size(), 0);
        s.kv_.reserve(storages_.size());
        for (Device* d : storages_)
            s.kv_.emplace_back(&d->pool, d->b->kv_layout().block_tokens);
        s.last_.assign(devices_.size(), 0);
        s.from_.assign(stages_.size(), Sequence::kLive);
        s.owner_ = this;
        return s;
    }

    // One pass over every entry: each sequence's tokens at their own positions through their own history, the logits after each wanting entry's last token landing in the context in entry order.
    // Its stages run in a row, each submitting its own work and committing the blocks of the storage it writes; a failure anywhere returns every history to where the pass found it.
    void forward(ExecContext& ctx, const BatchEntry* entries, size_t n_entries) {
        if (ctx.slots) throw std::logic_error("inference: a context reserved for passes runs them through begin_pass");
        if (ctx.passes.empty()) ctx.passes.resize(1);
        Pass& p = ctx.passes[0];
        begin(ctx, p, entries, n_entries);
        try {
            for (size_t s = 0; s < stages_.size(); ++s) run_stage(ctx, p, s);
        } catch (...) {
            roll_back(p);
            throw;
        }
        finish(ctx, p);
    }

    // Passes in flight: a caller keeps several passes of different sequences in one context and runs their stages itself, so on a pipelined split every stage works on some pass while the host samples another (docs/MULTI-DEVICE.md, passes in flight).
    // Each pass's stages run in order, and passes interleave as the caller likes: each device runs the stages recorded on it in that order, and a pass keeps its own handoff buffer, logits rows and ticket.
    size_t stage_count() const { return stages_.size(); }
    // Whether passes may be in flight together: several stages, the embedding on the first stage's device, the head on the last's and every feed-forward block beside its mixer.
    bool pipelined() const { return pipelined_; }
    // Whether stage s runs on the host, whose backend computes as its work is recorded, so recording it holds the calling thread for the stage's whole time.
    bool stage_on_host(size_t s) const { return devices_[stages_.at(s).device]->b->is_cpu(); }
    // The backend stage s runs on, which a caller timing the stages reads its host and device times from.
    backend::Backend& stage_backend(size_t s) { return *devices_[stages_.at(s).device]->b; }

    // Size a fresh context once, before any pass, for `slots` passes in flight, which above one need a pipelined placement, of up to `rows` rows each, with their handoff buffers and `logit_rows` rows of logits the caller hands out (begin_pass's logits_base); a reservation that fails leaves the context fresh, so a smaller one may follow.
    // The context is frozen from then on: begin_pass refuses a pass that needs more before any work, nothing is replaced while passes are in flight, and forward refuses it.
    void reserve_passes(ExecContext& ctx, size_t slots, size_t rows, size_t logit_rows) {
        if (ctx.slots || !ctx.scratch.empty()) throw std::logic_error("inference: reserve_passes takes a fresh context, once");
        if (!slots || !rows) throw std::logic_error("inference: reserve_passes needs a slot and a row");
        if (slots > 1 && !pipelined_) throw std::logic_error("inference: passes in flight need a pipelined placement");
        ExecContext reserved;
        ensure(reserved, rows, logit_rows, handoffs(slots));
        reserved.passes.assign(slots, Pass{});
        reserved.slots = slots;
        reserved.pass_rows = rows;
        ctx = std::move(reserved);
    }

    // Plan a pass in `slot` of a reserved context, its entries' tokens copied here and its wanting rows written from logits row `logits_base` on, and put its sequences in flight until end_pass or abort_pass.
    // A sequence in flight or listed twice, a slot in use or beyond the reservation, and more rows or logits rows than reserved are refused, with nothing changed.
    void begin_pass(ExecContext& ctx, size_t slot, const BatchEntry* entries, size_t n_entries, size_t logits_base) {
        if (!ctx.slots) throw std::logic_error("inference: begin_pass needs a context reserved for passes");
        if (slot >= ctx.slots) throw std::logic_error("inference: a pass slot beyond the reservation");
        Pass& p = ctx.passes[slot];
        if (p.in_flight) throw std::logic_error("inference: a pass slot already in flight");
        begin(ctx, p, entries, n_entries, logits_base);
        for (size_t e = 0; e < n_entries; ++e) entries[e].seq->in_flight_ = true;
        p.handoff = slot;
        p.in_flight = true;
    }

    // Stage s of the pass in `slot`, which must be the stage after the last one run: its storage reserved, the residual embedded or received, its layers, the head or the handoff out, its submissions and the storage's commit.
    // A failure aborts the pass, as abort_pass does, before it is rethrown; other passes in flight go on.
    void run_pass_stage(ExecContext& ctx, size_t slot, size_t s) {
        Pass& p = in_flight(ctx, slot);
        if (s != p.ran || s >= stages_.size()) throw std::logic_error("inference: a pass's stages run in order, each once");
        try {
            run_stage(ctx, p, s);
        } catch (...) {
            roll_back(p);
            release(p);
            throw;
        }
        ++p.ran;
    }

    // Row i of the pass's wanting rows, in entry order, once its last stage has run; this waits on the pass's own ticket, never on a later pass's.
    const float* pass_logits(ExecContext& ctx, size_t slot, size_t i) {
        const Pass& p = in_flight(ctx, slot);
        if (p.ran < stages_.size()) throw std::logic_error("inference: the logits of a pass before its last stage");
        if (i >= p.want) throw std::out_of_range("inference: no such logits row");
        devices_[(size_t)place_.output_device]->b->wait(p.sent);
        const void* host = ctx.logits_buf->host_ptr();
        if (!host) throw std::runtime_error("inference: logits are not host visible");
        return (const float*)host + (p.logits_base + i) * ctx.width;
    }

    // The pass in `slot` is done once its last stage has run: its sequences leave flight with the tokens committed, and the slot, its handoff buffers and the logits rows it was given are free for the next pass.
    void end_pass(ExecContext& ctx, size_t slot) {
        Pass& p = in_flight(ctx, slot);
        if (p.ran < stages_.size()) throw std::logic_error("inference: a pass ends after its last stage, and abort_pass abandons one before");
        release(p);
    }

    // Abandon the pass in `slot`, run or not: every device drained, then only its entries' histories back to where it found them in every storage, and its sequences out of flight.
    void abort_pass(ExecContext& ctx, size_t slot) {
        Pass& p = in_flight(ctx, slot);
        roll_back(p);
        release(p);
    }

    // Start a new history.
    // Blocks and the state slot return to their pools; their storage is retained.
    // Every pass ends in a submit or, on failure, a sync, so the sequence's last tickets cover everything that could still be touching a block or a slot: this waits for those and no more.
    void reset(Sequence& s) {
        settle(s, "a reset");
        drop_mark(s);
        for (auto& kv : s.kv_) kv.reset();
        std::fill(s.length_.begin(), s.length_.end(), 0);
        std::fill(s.from_.begin(), s.from_.end(), Sequence::kLive);
        s.kept_.release();
        s.state_.release();
    }

    // The history back to at most `length`, the one call that shortens it (docs/SPECULATIVE.md, section 1), and the length it reached: `length` wherever the caches hold it, else, on a model whose layers keep a state, the sequence's checkpoint at or below it, else 0.
    // The caller computes the rest again, without sampling, as a resume does; blocks and a checkpoint past the length reached return to their pools.
    // Inside a mark it reaches `length` itself: past the mark its state is run again from the mark's over the kept rows' saved inputs (Architecture::recur), at the mark it is the mark's, and the mark goes; a retract that throws keeps the mark, so it may be called again.
    size_t retract(Sequence& s, size_t length) {
        settle(s, "a retract");
        if (s.mark_.held() && s.mark_.ran && length > s.mark_.pos && length < s.length()) {
            rerun(s, length - s.mark_.pos);
            drop_mark(s);
            for (size_t& n : s.length_) n = std::min(n, length);
            for (auto& kv : s.kv_) kv.truncate(length);
            return length;
        }
        if (length >= s.length()) {
            if (s.mark_.ran) drop_mark(s);
            else if (s.mark_.held()) restore_mark(s);
            return s.length();
        }
        return rewind(s, length);
    }

    // Keep the sequence's state at its current length while one pass runs past it, so a retract into that pass reaches any of its rows exactly (docs/SPECULATIVE.md, section 1): a verify of drafts marks its history first.
    // Nothing on a model that keeps no state, whose caches reach every length; on one that keeps a state, the live slot becomes the mark's and the pass writes a fresh one, saving its rows' recurrent inputs.
    // False when no mark is free, and then nothing is marked; a second mark is refused.
    bool mark(Sequence& s) {
        settle(s, "a mark");
        if (s.mark_.held()) throw std::logic_error("inference: a second mark");
        if (!state_layers_) return true;
        // Everything that may throw first, then the holds, which do not.
        const bool live = s.state_.held();
        std::vector<size_t> from = live ? std::vector<size_t>(stages_.size(), s.state_.slot()) : s.from_;
        if (!s.mark_.hold.take(slots_, live, live ? s.state_.slot() : 0)) return false;
        if (live) s.state_.forget();
        s.mark_.ran = false;
        s.mark_.pos = s.length();
        std::copy(from.begin(), from.end(), s.from_.begin());
        s.mark_.from.swap(from);
        return true;
    }

    // The sequence's state kept at its current length, between passes, as its checkpoint: a paused history keeps its state this way, its live slot becoming the checkpoint's with no copy.
    // False when the model keeps no state or no checkpoint slot is free; the older checkpoint it replaces goes first, so its slot serves.
    bool keep(Sequence& s) {
        settle(s, "a keep");
        if (s.mark_.held()) throw std::logic_error("inference: a keep of a marked sequence");
        if (!state_layers_) return false;
        if (s.kept_.held() && s.kept_.pos() == s.length()) return true;
        if (!s.state_.held()) return false;
        s.kept_.release();
        if (!slots_.keep_live(s.state_.slot())) return false;
        s.kept_ = Checkpoint(slots_, s.state_.slot(), s.length());
        std::fill(s.from_.begin(), s.from_.end(), s.state_.slot());
        s.state_.forget();
        return true;
    }

    // The position of the sequence's checkpoint, which a fork of it takes and a retract reaches.
    std::optional<size_t> checkpoint(const Sequence& s) const {
        if (!s.kept_.held()) return std::nullopt;
        return s.kept_.pos();
    }
    // Checkpoint slots in all, and those a keep can still take.
    size_t checkpoint_slots() const { return state_layers_ ? options_.checkpoint_slots : 0; }
    size_t checkpoints_free() const { return state_layers_ ? slots_.kept_available() : 0; }

    // The single-sequence entry points the CLI uses: one sequence and one context owned here, and one entry per pass.

    // Run one token through the model (prefill or continue).
    // Returns logits over the full vocabulary.
    std::vector<float> step(int token_id) {
        const uint32_t id = (uint32_t)token_id;
        const BatchEntry entry{&seq_, &id, 1, true};
        forward(ctx_, &entry, 1);
        return row(ctx_, 0);
    }

    // The logits after each of `n` tokens fed in one pass as generated tokens, as a verify of drafts feeds them (docs/SPECULATIVE.md, section 3): `n` rows of the vocabulary, each the bits single steps give, valid until the next pass.
    const float* step(const uint32_t* ids, size_t n) {
        BatchEntry entry{&seq_, ids, n, true};
        entry.every_logits = true;
        entry.extent = 1;
        forward(ctx_, &entry, 1);
        return ctx_.logits(0);
    }
    bool mark() { return mark(seq_); }

    // Process a whole prompt with matrix-matrix matmuls instead of one token at a time.
    // Each weight row is then reused across the batch, which is the difference between prefill being compute bound and paying the entire weight stream once per token.
    // Only the final token's logits are needed, so only the last pass asks for them.
    // A `keep_at` inside the prompt keeps the state there as the sequence's checkpoint, the chunk before it cut to end there.
    std::vector<float> prefill(const std::vector<uint32_t>& ids, size_t keep_at = 0) {
        if (ids.empty()) throw std::runtime_error("inference: empty prompt");
        // The prompt is one transaction across its microbatches: a failure in any of them restores the history from before the call.
        const size_t start = seq_.length();
        auto work = [&] {
            // Sized to the largest chunk this prompt will use, inside the scope, so a short prompt does not allocate scratch for a full ubatch.
            // Sized before any chunk runs, so nothing in flight loses its storage.
            ensure(ctx_, std::min((size_t)ubatch(), ids.size()), 1, handoffs(1));
            // Chunks of the ubatch, one cut short to end at `keep_at`.
            const size_t B = (size_t)ubatch(), cut = keep_at > start ? keep_at - start : 0;
            std::vector<std::pair<size_t, size_t>> parts;
            for (size_t i = 0; i < ids.size();) {
                size_t n = std::min(B, ids.size() - i);
                if (cut > i && cut < i + n) n = cut - i;
                parts.push_back({i, n});
                i += n;
            }
            const size_t chunks = parts.size();
            auto chunk = [&](size_t c) {
                const size_t i = parts[c].first, n = parts[c].second;
                BatchEntry entry{&seq_, ids.data() + i, n, i + n == ids.size()};
                entry.extent = start + ids.size();
                entry.keep = cut && i + n == cut;
                return entry;
            };
            if (!pipelined_) {
                for (size_t c = 0; c < chunks; ++c) {
                    const BatchEntry entry = chunk(c);
                    forward(ctx_, &entry, 1);
                }
                return;
            }
            // Step t runs stage s of chunk t - s, the first stage first, so every device has its next chunk queued before it finishes the one it runs; each device takes its chunks in order, which is what lets them share its arena.
            const size_t S = stages_.size();
            if (ctx_.passes.size() < S) ctx_.passes.resize(S);
            for (size_t t = 0; t + 1 < chunks + S; ++t)
                for (size_t s = 0; s < S; ++s) {
                    if (t < s || t - s >= chunks) continue;
                    const size_t c = t - s;
                    Pass& p = ctx_.passes[c % S];
                    if (s == 0) {
                        const BatchEntry entry = chunk(c);
                        begin(ctx_, p, &entry, 1);
                        p.handoff = c % 2;
                    }
                    run_stage(ctx_, p, s);
                }
            finish(ctx_, ctx_.passes[(chunks - 1) % S]);
        };
        try {
            scoped(0, work);
        } catch (...) {
            retire();
            rewind(seq_, start);
            throw;
        }
        return row(ctx_, 0);
    }

    // Every position's logits for a text from an empty history, through the batched passes prefill takes, handed to `each` as (position, logits) a microbatch at a time.
    // Scoring through this rather than step exercises the prompt path, whose kernels differ from the decode path's on a device (inference/perplexity.hpp).
    void score(const std::vector<uint32_t>& ids, const std::function<void(size_t, const float*)>& each) {
        if (ids.empty()) throw std::runtime_error("inference: empty text");
        reset();
        auto work = [&] {
            ensure(ctx_, std::min((size_t)ubatch(), ids.size()), std::min((size_t)ubatch(), ids.size()), handoffs(1));
            for (size_t i = 0; i < ids.size();) {
                const size_t B = std::min((size_t)ubatch(), ids.size() - i);
                BatchEntry entry{&seq_, ids.data() + i, B, true};
                entry.every_logits = true;
                entry.extent = ids.size();
                forward(ctx_, &entry, 1);
                for (size_t j = 0; j < B; ++j) each(i + j, ctx_.logits(j));
                i += B;
            }
        };
        try {
            scoped(0, work);
        } catch (...) {
            retire();
            reset();
            throw;
        }
    }

    void reset() { reset(seq_); }
    size_t retract(size_t length) { return retract(seq_, length); }

    // Allocated is what the backends back; used is the committed history.
    // The gap is the paging cost in memory (docs/KV-CACHE.md).
    size_t kv_allocated_bytes() const {
        size_t n = 0;
        for (Device* d : storages_) n += d->storage->allocated_bytes();
        return n;
    }
    size_t kv_peak_bytes() const {
        size_t n = 0;
        for (Device* d : storages_) n += d->storage->peak_bytes();
        return n;
    }
    size_t kv_used_bytes() const {
        return seq_.length() * kv_layers_ * kv_bytes_per_position(plan_, options_);
    }

    // Whether some layer keeps a recurrent state, which exists only at the end of what it has read: such a model forks and retracts only at a checkpoint, and a failed pass returns its entries to theirs.
    bool keeps_state() const { return state_layers_ > 0; }
    // The sequences that may hold a recurrent state at once (ModelOptions::state_slots), zero for a model whose layers keep none.
    size_t state_slots() const { return state_layers_ ? options_.state_slots : 0; }

    // The class of rows of this extent over the placement: each used device's (Backend::row_class) and whether they take a streamed layer on the device.
    // Rows of two extents of one class give the same bits, so a history may continue from rows computed at another extent only where the classes are equal (docs/SPECULATIVE.md, section 1).
    std::vector<size_t> row_class(size_t extent) const {
        std::vector<size_t> c;
        for (const auto& d : devices_)
            if (d->used) c.push_back(d->b->row_class(extent));
        c.push_back(streams(extent) ? 1 : 0);
        return c;
    }

private:
    // One backend and what the placement put on it.
    // A pool is not movable, because sequences hold its address, so devices live behind pointers.
    struct Device {
        backend::BackendPtr b;
        bool used = false;
        bool sends = false;                      // the residual leaves it, so it keeps handoff buffers
        int mixer_layers = 0;
        int kv_layers = 0;                       // its mixer layers whose cache is KV
        int state_layers = 0;                    // and those whose cache is a state
        int storage_index = -1;
        std::vector<int> local_layer;            // model layer -> layer in the storage of its cache
        std::unique_ptr<backend::KVStorage> storage;
        BlockPool pool;
        std::unique_ptr<backend::StateStorage> states;
        backend::BufferPtr saved;                // every mark's saved recurrent inputs, per mark, state layer, saved item and row
        size_t saved_floats = 0;                 // a row's saved inputs in one state layer
        std::vector<backend::BufferPtr> tables;  // the position tables, on a device that runs a mixer
    };

    // Consecutive layers whose mixer runs on one device, and every device a stage records work on, which it submits.
    struct Stage {
        size_t device;
        int first, end;
        std::vector<size_t> touches;
    };

    Placement place_;
    std::vector<std::unique_ptr<Device>> devices_;
    std::vector<Device*> storages_;              // the devices whose mixer layers keep KV
    size_t kv_layers_ = 0;                       // the model's layers that keep KV
    size_t state_layers_ = 0;                    // and those that keep a state
    SlotPool slots_;
    std::vector<Stage> stages_;
    bool pipelined_ = false;                     // a prompt's chunks flow through the stages together (prefill)
    int ubatch_ = kDefaultUbatch;
    ModelOptions options_;
    std::shared_ptr<const Architecture> arch_;
    ModelPlan plan_;
    std::vector<Weight> pass_;                  // the pass's roles by role id, on the embedding's and the head's devices
    std::vector<std::vector<Weight>> home_;     // per layer, its roles by role id, each on the device of its part
    // A routed layer run beside its mixer for a long prompt (Placement::stream_from): the device it runs on, or -1, and home_'s row with its copy roles adopted on that device and its window roles in that device's windows.
    std::vector<int> stream_device_;
    std::vector<std::vector<Weight>> stream_;
    std::vector<std::vector<backend::BufferPtr>> windows_;   // per device, a buffer per window role in role order, sized to the largest streamed layer's
    std::vector<std::vector<float>> tables_;
    Sequence seq_;
    ExecContext ctx_;

    size_t device_of(Part part, size_t l) const {
        if (part == Part::embed) return (size_t)place_.embed_device;
        if (part == Part::head) return (size_t)place_.output_device;
        return (size_t)(part == Part::mixer ? place_.mixer_device[l] : place_.ffn_device[l]);
    }

    // Resolve every role of the plan to a Weight, in plan order: the pass's roles, then each layer's followed by a streamed layer's copies.
    // Each role reads the tensor plan_model set, a role without one refused here, checked by the role's kind in the same step, so a resolved handle is well-formed by construction and the forward pass never looks a tensor up by name.
    // Each weight is put on the backend that runs its part, by the caller's hook when it gave one, and a tensor two roles take on one device is put there once, as a tied head beside the embedding reads the embedding's buffer.
    void resolve_tensors(const ModelWeights& weights, const AdoptWeight& adopt) {
        const size_t n_devices = devices_.size();
        std::vector<backend::BufferPtr> taken(weights.tensors.size() * n_devices);
        auto resolve = [&](const Role& role, size_t device) -> Weight {
            if (!role.tensor) throw TensorIndex::missing(role.alias.empty() ? role.name : role.alias);
            const size_t i = *role.tensor;
            const TensorView& t = weights.tensors[i];
            bool valid;
            if (role.kind == RoleKind::experts) {
                valid = t.shape.size() == 3 && t.shape[0] == role.in && t.shape[1] == role.out && t.shape[2] == role.experts;
            } else {
                const bool norm = role.kind == RoleKind::norm;
                valid = !t.shape.empty() && t.shape[0] == role.in;
                if (norm) {
                    valid = valid && t.type == quant::GGML_TYPE_F32;
                } else {
                    valid = valid && t.shape.size() >= 2 && t.shape[1] == role.out;
                    if (role.kind == RoleKind::table) valid = valid && t.type == quant::GGML_TYPE_F32;
                }
                for (size_t d = norm ? 1 : 2; d < t.shape.size(); ++d) valid = valid && t.shape[d] == 1;
            }
            if (!valid) throw std::runtime_error("inference: incompatible tensor layout " + t.name);
            backend::BufferPtr& buffer = taken[i * n_devices + device];
            if (!buffer) {
                backend::Backend& b = *devices_[device]->b;
                buffer = adopt ? adopt(i, b) : b.adopt(t.data, t.bytes);
            }
            return Weight{t.type, buffer, (size_t)role.in, (size_t)role.out};
        };
        pass_.assign(plan_.role_ids, Weight{});
        for (const Role& role : plan_.pass) pass_[role.id] = resolve(role, device_of(role.part, 0));
        const size_t n_layer = plan_.layers.size();
        home_.assign(n_layer, {});
        stream_.assign(n_layer, {});
        for (size_t l = 0; l < n_layer; ++l) {
            const LayerPlan& layer = plan_.layers[l];
            std::vector<Weight>& row = home_[l];
            row.assign(plan_.role_ids, Weight{});
            for (const Role& role : layer.roles) row[role.id] = resolve(role, device_of(role.part, l));
            if (stream_device_[l] < 0) continue;
            const size_t a = (size_t)stream_device_[l];
            stream_[l] = row;
            for (const Role& role : layer.roles)
                if (role.stream == Stream::copy) stream_[l][role.id] = resolve(role, a);
        }
        // The windows are allocated with the weights, so a pass never fails for want of one.
        std::vector<std::vector<size_t>> sizes(n_devices);
        for (size_t l = 0; l < n_layer; ++l) {
            if (stream_device_[l] < 0) continue;
            std::vector<size_t>& s = sizes[(size_t)stream_device_[l]];
            size_t k = 0;
            for (const Role& role : plan_.layers[l].roles) {
                if (role.stream != Stream::window) continue;
                if (k == s.size()) s.push_back(0);
                s[k] = std::max(s[k], home_[l][role.id].data->size());
                ++k;
            }
        }
        windows_.assign(n_devices, {});
        for (size_t d = 0; d < n_devices; ++d)
            for (size_t bytes : sizes[d]) windows_[d].push_back(devices_[d]->b->alloc(bytes));
        for (size_t l = 0; l < n_layer; ++l) {
            if (stream_device_[l] < 0) continue;
            size_t k = 0;
            for (const Role& role : plan_.layers[l].roles) {
                if (role.stream != Stream::window) continue;
                const Weight& home = home_[l][role.id];
                stream_[l][role.id] = Weight{home.type, windows_[(size_t)stream_device_[l]][k++], home.nin, home.nout};
            }
        }
    }

    // Physical prompt microbatch size, used to bound matrix width and scratch storage.
    int ubatch() const { return ubatch_; }

    [[noreturn]] static void refuse_op(size_t l, const OpUse& u) {
        throw std::runtime_error("inference: layer " + std::to_string(l) + "'s " + (u.part == Part::mixer ? "mixer" : "feed-forward part") +
                                 " needs " + backend::op_name(u.op) + ", which the backend of its device does not implement");
    }

    // The history a pass continues: the first stage's committed length, which a pipelined prompt's chunk commits first; outside a prompt every stage agrees.
    size_t history(const Sequence& s) const { return s.stage_length(0); }

    // A sequence out of flight whose passes have retired, before it gives blocks or slots back.
    void settle(Sequence& s, const char* what) {
        if (s.owner_ != this) throw std::runtime_error("inference: sequence of another model");
        if (s.in_flight_) throw std::logic_error(std::string("inference: ") + what + " of a sequence in flight");
        // A sequence moved from has no tickets, nothing of it being left to wait on.
        for (size_t d = 0; d < devices_.size() && d < s.last_.size(); ++d)
            if (devices_[d]->used) devices_[d]->b->wait(s.last_[d]);
    }

    // A history back to `length`, or on a model that keeps a state to its checkpoint at or below `length`, else 0, whose live state a pass may have written: every stage and storage at the length reached, which is returned, the blocks and a checkpoint beyond it returned.
    // The state is then the checkpoint's, or zero, so the live slot goes back too, and the next pass takes one: a donor parked at its checkpoint holds none of the slots admission counts on.
    size_t rewind(Sequence& s, size_t length) noexcept {
        // At or past a mark the state is the mark's, and the history goes back to its position; below it the mark is not needed.
        if (s.mark_.held() && length >= s.mark_.pos) {
            const size_t to = s.mark_.pos;
            restore_mark(s);
            for (size_t& n : s.length_) n = std::min(n, to);
            for (auto& kv : s.kv_) kv.truncate(to);
            return to;
        }
        drop_mark(s);
        size_t to = length;
        if (state_layers_) {
            if (s.kept_.held() && s.kept_.pos() > length) s.kept_.release();
            to = s.kept_.held() ? s.kept_.pos() : 0;
            std::fill(s.from_.begin(), s.from_.end(), s.kept_.held() ? s.kept_.slot() : Sequence::kLive);
            s.state_.release();
        }
        for (size_t& n : s.length_) n = std::min(n, to);
        for (auto& kv : s.kv_) kv.truncate(to);
        return to;
    }

    // Where row 0 of saved item `item` of local state layer `layer` of mark buffer `buffer` sits in device d's saved buffer, in floats; its rows follow one another.
    size_t saved_at(const Device& d, const LayerPlan& lp, size_t buffer, size_t layer, size_t item) const {
        size_t before = 0;
        for (size_t i = 0; i < item; ++i) before += lp.saved[i].width;
        return ((buffer * (size_t)d.state_layers + layer) * d.saved_floats + before) * options_.mark_rows;
    }

    // After state layer l's mixer on device `dev`: each marked entry's rows of the inputs its state's update read, copied into its mark's buffer.
    void save(ExecContext& ctx, const Pass& p, size_t dev, int l) {
        const Device& d = *devices_[dev];
        const LayerPlan& lp = plan_.layers[(size_t)l];
        const ExecContext::Scratch& sc = ctx.scratch[dev];
        for (size_t e = 0; e < p.entries.size(); ++e) {
            const Sequence::Mark& m = p.entries[e].seq->mark_;
            if (!m.held()) continue;
            const size_t r0 = e ? p.runs[e - 1].end : 0, n = p.entries[e].n;
            for (size_t i = 0; i < lp.saved.size(); ++i) {
                const Saved& v = lp.saved[i];
                d.b->copy(*d.saved, saved_at(d, lp, m.hold.buffer(), (size_t)d.local_layer[(size_t)l], i) * sizeof(float), *sc.arena,
                          sc.offset[v.slot] + (v.plane * p.rows + r0) * v.width * sizeof(float), n * v.width * sizeof(float));
            }
        }
    }

    // The state after the first `rows` rows of the pass past the sequence's mark, into its live slot: on each device that keeps a state, every state layer's saved inputs copied back into the model's own arena and its update run from the mark's state (Architecture::recur), in each device's order after that pass.
    // A failure drains every device and leaves the mark, which a retry reads again.
    void rerun(Sequence& s, size_t rows) {
        const Sequence::Mark& m = s.mark_;
        try {
            ensure(ctx_, rows, 0, handoffs(1));
            const backend::RowRun run{rows, 1};
            for (size_t st = 0; st < stages_.size(); ++st) {
                const size_t dev = stages_[st].device;
                Device& d = *devices_[dev];
                if (!d.states) continue;
                const ExecContext::Scratch& sc = ctx_.scratch[dev];
                const size_t src = m.from[st] == Sequence::kLive ? s.state_.slot() : m.from[st];
                const backend::StateView view{d.states.get(), src, s.state_.slot(), m.pos, rows};
                for (int l = stages_[st].first; l < stages_[st].end; ++l) {
                    const LayerPlan& lp = plan_.layers[(size_t)l];
                    if (lp.cache != Cache::state) continue;
                    const size_t layer = (size_t)d.local_layer[(size_t)l];
                    for (size_t i = 0; i < lp.saved.size(); ++i) {
                        const Saved& v = lp.saved[i];
                        d.b->copy(*sc.arena, sc.offset[v.slot] + v.plane * rows * v.width * sizeof(float), *d.saved,
                                  saved_at(d, lp, m.hold.buffer(), layer, i) * sizeof(float), rows * v.width * sizeof(float));
                    }
                    Step step = part(ctx_, dev, home_[(size_t)l].data(), lp.kind, 0, rows, {&run, 1});
                    step.states = &view;
                    step.n_views = 1;
                    step.state_layer = layer;
                    arch_->recur(step);
                }
                s.last_[dev] = d.b->submit();
            }
        } catch (...) {
            retire();
            throw;
        }
    }

    // The state at the mark's position as the sequence's again: the mark's slot its live slot where the mark took it from there, else read from where it was then; the live slot a pass wrote returned and the mark gone.
    void restore_mark(Sequence& s) noexcept {
        Sequence::Mark& m = s.mark_;
        if (!m.held()) return;
        if (m.hold.owns()) {
            s.state_.adopt(slots_, m.hold.give_slot());
            std::fill(s.from_.begin(), s.from_.end(), Sequence::kLive);
        } else {
            s.state_.release();
            std::copy(m.from.begin(), m.from.end(), s.from_.begin());
        }
        m.hold.release();
        m.ran = false;
    }

    // The mark gone with its slot, the state staying where the passes after it left it.
    void drop_mark(Sequence& s) noexcept {
        s.mark_.hold.release();
        s.mark_.ran = false;
    }

    // A pass's plan: its rows in entry order, their positions after each history, and the rows the head reads, from logits row `logits_base` on, with nothing reserved yet, since each stage reserves the blocks of the storage it writes.
    // A context reserved for passes is not grown: a pass that needs more rows or logits rows than it holds is refused here.
    void begin(ExecContext& ctx, Pass& p, const BatchEntry* entries, size_t n_entries, size_t logits_base = 0) {
        if (!entries || !n_entries) throw std::runtime_error("inference: empty batch");
        size_t rows = 0, want = 0;
        for (size_t e = 0; e < n_entries; ++e) {
            const BatchEntry& en = entries[e];
            if (!en.seq || en.seq->owner_ != this)
                throw std::runtime_error("inference: batch entry without a sequence of this model");
            if (en.seq->in_flight_) throw std::logic_error("inference: a sequence already in flight");
            if (!en.ids || !en.n)
                throw std::runtime_error("inference: batch entry without tokens");
            // The position tables cover [0, context_length); a row past them would read off the end.
            if (en.n > plan_.context_length ||
                history(*en.seq) > plan_.context_length - en.n)
                throw std::runtime_error("inference: context length exceeded (" +
                                         std::to_string(plan_.context_length) + " tokens)");
            rows += en.n;
            want += en.want_logits ? (en.every_logits ? en.n : 1) : 0;
        }
        // None is in flight, so the first entry found marked is listed twice; the marks come off again here.
        size_t marked = 0;
        for (; marked < n_entries && !entries[marked].seq->in_flight_; ++marked) entries[marked].seq->in_flight_ = true;
        for (size_t e = 0; e < marked; ++e) entries[e].seq->in_flight_ = false;
        if (marked < n_entries) throw std::logic_error("inference: a sequence listed twice in a pass");
        if (ctx.slots && (rows > ctx.pass_rows || want > ctx.logit_rows || logits_base > ctx.logit_rows - want))
            throw std::logic_error("inference: a pass beyond the rows or logits rows reserve_passes reserved");
        // Every entry holds a state slot from its first pass on, so a pass never runs short of one, and a keep entry a checkpoint slot; a pass that cannot take them all, is refused or fails to plan takes none.
        size_t fresh = 0, keeps = 0;
        for (size_t e = 0; state_layers_ && e < n_entries; ++e) {
            fresh += !entries[e].seq->state_.held();
            keeps += entries[e].keep;
        }
        if (fresh > slots_.available()) throw std::runtime_error("inference: every recurrent state slot is held");
        if (keeps > slots_.kept_available()) throw std::runtime_error("inference: every checkpoint slot is held");
        for (size_t e = 0; e < n_entries; ++e) {
            const Sequence::Mark& m = entries[e].seq->mark_;
            if (!m.held()) continue;
            if (m.ran) throw std::logic_error("inference: a marked sequence takes one pass before its retract");
            if (entries[e].n > options_.mark_rows) throw std::logic_error("inference: a pass after a mark beyond the rows a mark saves");
            if (entries[e].keep) throw std::logic_error("inference: a checkpoint in a pass after a mark");
        }
        for (size_t e = 0; keeps && e < n_entries; ++e)
            if (entries[e].keep && !whole_blocks(history(*entries[e].seq) + entries[e].n))
                throw std::logic_error("inference: a checkpoint at a position of whole blocks in every storage");
        if (!ctx.slots) ensure(ctx, rows, want, handoffs(1));
        p.entries.assign(entries, entries + n_entries);
        p.start.resize(n_entries);
        p.rows = rows;
        p.want = want;
        p.handoff = 0;
        p.logits_base = logits_base;
        p.ran = 0;
        p.ids.resize(rows);
        p.pos.resize(rows);
        p.pick.resize(want);
        p.runs.resize(n_entries);
        p.head_runs.clear();
        p.views.resize(storages_.size());
        for (auto& v : p.views) v.resize(n_entries);
        if (state_layers_) {
            p.states.resize(devices_.size());
            for (auto& v : p.states) v.resize(n_entries);
        }
        p.kept.clear();
        p.kept.resize(n_entries);
        size_t r = 0, w = 0;
        for (size_t e = 0; e < n_entries; ++e) {
            const BatchEntry& en = entries[e];
            const size_t len = history(*en.seq);
            p.start[e] = len;
            for (size_t b = 0; b < en.n; ++b) {
                p.ids[r + b] = en.ids[b];
                p.pos[r + b] = (uint32_t)(len + b);
            }
            const size_t extent = en.extent ? en.extent : en.n;
            p.runs[e] = backend::RowRun{r + en.n, extent};
            // The head reads one row per entry as a generated token's, or every row of a scored text as its prompt's.
            if (en.want_logits && en.every_logits) {
                for (size_t b = 0; b < en.n; ++b) p.pick[w++] = (uint32_t)(r + b);
                p.head_runs.push_back(backend::RowRun{w, extent});
            }
            r += en.n;
            if (en.want_logits && !en.every_logits) {
                p.pick[w++] = (uint32_t)(r - 1);
                p.head_runs.push_back(backend::RowRun{w, 1});
            }
        }
        p.long_runs = false;
        for (size_t e = 0; e < n_entries; ++e) p.long_runs = p.long_runs || streams(p, e);
        // Last, once nothing can fail: the checks above left a free slot for each.
        for (size_t e = 0; fresh && e < n_entries; ++e) entries[e].seq->state_.take(slots_);
        for (size_t e = 0; e < n_entries; ++e) entries[e].seq->mark_.ran = entries[e].seq->mark_.held();
        for (size_t e = 0; keeps && e < n_entries; ++e)
            if (entries[e].keep) p.kept[e] = Checkpoint(slots_, slots_.acquire_kept(), p.start[e] + entries[e].n);
    }

    // Whether `pos` is whole blocks in every KV storage, as a fork and a checkpoint need.
    bool whole_blocks(size_t pos) const {
        for (const Device* d : storages_)
            if (pos % d->b->kv_layout().block_tokens) return false;
        return true;
    }

    // Stage s of a pass: its storage's blocks reserved, the residual embedded or received from the stage before, its layers, then the head after the last stage or the residual sent on, its submissions, and the commit of its length and its storage.
    void run_stage(ExecContext& ctx, Pass& p, size_t s) {
        const Stage& st = stages_[s];
        Device& home = *devices_[st.device];
        const int storage = home.storage_index;
        for (size_t e = 0; storage >= 0 && e < p.entries.size(); ++e) {
            KVSequence& kv = p.entries[e].seq->kv_[(size_t)storage];
            kv.prepare(p.entries[e].n);
            p.views[(size_t)storage][e] = kv.view(home.storage.get());
            p.views[(size_t)storage][e].extent = p.runs[e].extent;
        }
        // A state is read from where the history this stage has committed left it, the live slot or a checkpoint, and written to the live slot, or for a keep entry to its checkpoint's.
        for (size_t e = 0; home.states && e < p.entries.size(); ++e) {
            const Sequence& q = *p.entries[e].seq;
            const size_t src = q.from_[s] == Sequence::kLive ? q.state_.slot() : q.from_[s];
            const size_t dst = p.kept[e].held() ? p.kept[e].slot() : q.state_.slot();
            p.states[st.device][e] = backend::StateView{home.states.get(), src, dst, q.stage_length(s), p.entries[e].n};
        }
        size_t cur = st.device;
        const backend::RowRuns all{p.runs.data(), p.runs.size()};
        if (s == 0) {
            cur = (size_t)place_.embed_device;
            arch_->embed(part(ctx, cur, pass_.data(), 0, 0, p.rows, all), p.ids.data());
        } else if (p.at != cur) {
            receive(ctx, p.at, p.handoff, p.sent, cur, 0, p.rows);
        }
        for (int l = st.first; l < st.end; l++) {
            if (st.device != cur) { cross(ctx, cur, st.device, 0, p.rows); cur = st.device; }
            arch_->mixer(mixer_part(ctx, p, cur, l));
            if (plan_.layers[(size_t)l].cache == Cache::state) save(ctx, p, cur, l);
            if (p.long_runs && stream_device_[(size_t)l] == (int)cur) {
                ffn_split(ctx, p, cur, l);
                continue;
            }
            const size_t f = (size_t)place_.ffn_device[(size_t)l];
            if (f != cur) { cross(ctx, cur, f, 0, p.rows); cur = f; }
            arch_->ffn(part(ctx, cur, home_[(size_t)l].data(), plan_.layers[(size_t)l].kind, 0, p.rows, all));
        }
        if (s + 1 < stages_.size()) {
            // A residual already where the next stage runs stays there.
            if (cur != stages_[s + 1].device) send(ctx, cur, p.handoff, 0, p.rows);
            p.at = cur;
        } else {
            const size_t o = (size_t)place_.output_device;
            if (o != cur) { cross(ctx, cur, o, 0, p.rows); cur = o; }
            if (p.want)
                arch_->head(HeadStep{part(ctx, cur, pass_.data(), 0, 0, p.rows, all), p.pick.data(), p.want,
                                     backend::RowRuns{p.head_runs.data(), p.head_runs.size()},
                                     {ctx.logits_buf.get(), p.logits_base * plan_.vocab}});
        }
        for (size_t d : st.touches) ctx.tickets[d] = devices_[d]->b->submit();
        p.sent = ctx.tickets[cur];
        for (size_t e = 0; e < p.entries.size(); ++e) {
            Sequence& q = *p.entries[e].seq;
            if (storage >= 0) q.kv_[(size_t)storage].commit();
            else q.length_[s] += p.entries[e].n;
            for (size_t d : st.touches) q.last_[d] = ctx.tickets[d];
            if (!state_layers_) continue;
            q.from_[s] = p.kept[e].held() ? p.kept[e].slot() : Sequence::kLive;
            // Once every stage holds it, the checkpoint is the sequence's, and the one it replaces goes: a later write of that slot is enqueued after every read of it on each device's stream.
            if (s + 1 == stages_.size() && p.kept[e].held()) q.kept_ = std::move(p.kept[e]);
        }
    }

    // After the last stage: where the logits are and the ticket that says they are ready, the pass's own head's.
    void finish(ExecContext& ctx, const Pass& p) {
        ctx.n_logits = p.want;
        ctx.backend = devices_[(size_t)place_.output_device]->b.get();
        ctx.ticket = p.sent;
        ctx.pending = p.want > 0;
    }

    // A failed pass: every device drained, then every entry's histories back to where the pass found them, or on a model that keeps a state, whose live state the pass may have written, to its checkpoint (rewind); the blocks and the checkpoint slots its stages reserved or wrote returned.
    void roll_back(Pass& p) noexcept {
        retire();
        for (size_t e = 0; e < p.entries.size(); ++e) rewind(*p.entries[e].seq, p.start[e]);
        p.kept.clear();
    }

    // The pass in a reserved context's slot, which must be in flight.
    static Pass& in_flight(ExecContext& ctx, size_t slot) {
        if (slot >= ctx.slots || !ctx.passes[slot].in_flight) throw std::logic_error("inference: no pass in flight in that slot");
        return ctx.passes[slot];
    }

    // A slot's pass leaves flight, and its sequences with it.
    static void release(Pass& p) noexcept {
        for (const BatchEntry& en : p.entries) en.seq->in_flight_ = false;
        p.in_flight = false;
    }

    // Handoff buffers on each device a crossing leaves (Device::sends) for `slots` passes in flight, by the rule the fit counts them with.
    size_t handoffs(size_t slots) const { return handoff_buffers(slots, pipelined_); }

    // The CPU prefill scope is per backend, so a prompt enters one on every device it runs on, nested.
    // A device backend's scope is the default and just runs the body.
    void scoped(size_t d, const std::function<void()>& work) {
        while (d < devices_.size() && !devices_[d]->used) ++d;
        if (d >= devices_.size()) { work(); return; }
        devices_[d]->b->run_prefill([&] { scoped(d + 1, work); });
    }

    // One backend allocation holds the activation slots of a pass, each aligned to 64 bytes.
    // Device allocators handle a few large blocks far better than many small ones, and resizing is one call.
    // The caller only publishes the result once this returns, so an allocation that throws leaves the previous arena intact.
    backend::BufferPtr alloc_arena(backend::Backend& b, const std::vector<size_t>& counts, std::vector<size_t>& offsets) const {
        size_t total = 0;
        offsets.resize(counts.size());
        for (size_t i = 0; i < counts.size(); ++i) {
            offsets[i] = total;
            const size_t bytes = counts[i] * sizeof(float);
            if (bytes / sizeof(float) != counts[i] || total > (size_t)-1 - bytes - 63)
                throw std::runtime_error("inference: activation arena size overflows");
            total = (total + bytes + 63) / 64 * 64;
        }
        return b.alloc(total);
    }

    // Storage for a pass of `rows` rows with `want` logits rows, on every device the placement uses, with `buffers` handoff buffers on each device a crossing leaves (handoffs): grown when a pass needs more rows than the context holds, never shrunk.
    // Each is allocated whole before it replaces what the context had.
    void ensure(ExecContext& ctx, size_t rows, size_t want, size_t buffers) {
        auto mul = [](size_t a, size_t b) {
            if (b && a > (size_t)-1 / b)
                throw std::runtime_error("inference: activation arena size overflows");
            return a * b;
        };
        ctx.scratch.resize(devices_.size());
        ctx.tickets.resize(devices_.size(), 0);
        // The run list a part rebuilds and a streamed layer's groups hold at most a run per entry, and so per row, so no part grows them.
        ctx.entry_runs.reserve(rows);
        ctx.part_runs.reserve(rows);
        for (size_t d = 0; d < devices_.size(); ++d) {
            ExecContext::Scratch& sc = ctx.scratch[d];
            if (!devices_[d]->used || (sc.arena && sc.rows >= rows)) continue;
            std::vector<size_t> counts(plan_.slots.size()), offsets;
            for (size_t i = 0; i < counts.size(); ++i) counts[i] = mul(rows, plan_.slots[i]);
            backend::BufferPtr arena = alloc_arena(*devices_[d]->b, counts, offsets);
            // A pass through this context may still run on the arena being replaced: nothing else waits for a pass that wanted no logits.
            if (sc.arena) devices_[d]->b->wait(ctx.tickets[d]);
            sc.arena = std::move(arena);
            sc.offset = std::move(offsets);
            sc.rows = rows;
        }
        // The host-visible buffers a crossing leaves each sending device through.
        size_t used = 0;
        for (const auto& d : devices_) used += d->used;
        if (used > 1 && ctx.handoff_rows < rows) {
            std::vector<std::vector<backend::BufferPtr>> handoff(devices_.size());
            const size_t bytes = mul(mul(rows, plan_.residual), sizeof(float));
            for (size_t d = 0; d < devices_.size(); ++d)
                for (size_t i = 0; devices_[d]->sends && i < buffers; ++i)
                    handoff[d].push_back(devices_[d]->b->alloc(bytes, backend::Memory::host_visible));
            for (size_t d = 0; d < ctx.handoff.size(); ++d)
                if (devices_[d]->used) devices_[d]->b->wait(ctx.tickets[d]);
            ctx.handoff = std::move(handoff);
            ctx.handoff_rows = rows;
        }
        if (want && (!ctx.logits_buf || ctx.logit_rows < want)) {
            // The head writes here and the host reads it in place once the pass has retired: the one point per pass that must be host visible, and the one wait per pass.
            backend::BufferPtr logits = devices_[(size_t)place_.output_device]->b->alloc(
                mul(mul(want, plan_.vocab), sizeof(float)), backend::Memory::host_visible);
            if (ctx.logits_buf) devices_[(size_t)place_.output_device]->b->wait(ctx.tickets[(size_t)place_.output_device]);
            ctx.logits_buf = std::move(logits);
            ctx.logit_rows = want;
            ctx.width = plan_.vocab;
        }
    }

    static backend::Slice slot(const ExecContext& ctx, size_t device, size_t i) {
        const ExecContext::Scratch& sc = ctx.scratch[device];
        return {sc.arena.get(), sc.offset[i] / sizeof(float)};
    }

    static std::vector<float> row(ExecContext& ctx, size_t i) {
        const float* p = ctx.logits(i);
        return std::vector<float>(p, p + ctx.width);
    }

    // The residual stream moves from one device's x slot to another's through host memory, `rows` rows from `base`; a few kilobytes on a decode token.
    // `send` copies them into the source's host-visible handoff buffer inside the source's own work, so they outlast the source moving on to its next pass, and the submission that carries the copy says when they are there.
    void send(ExecContext& ctx, size_t from, size_t handoff, size_t base, size_t rows) {
        const size_t E = plan_.residual;
        const backend::Slice x = slot(ctx, from, 0);
        devices_[from]->b->copy(*ctx.handoff[from][handoff], base * E * sizeof(float), *x.buffer,
                                (x.offset + base * E) * sizeof(float), rows * E * sizeof(float));
    }

    // `receive` waits for that submission and writes the rows into the destination's residual, enqueued there.
    void receive(ExecContext& ctx, size_t from, size_t handoff, backend::Ticket sent, size_t to, size_t base, size_t rows) {
        const size_t E = plan_.residual;
        devices_[from]->b->wait(sent);
        const backend::Slice x = slot(ctx, to, 0);
        const uint8_t* rows_out = (const uint8_t*)ctx.handoff[from][handoff]->host_ptr() + base * E * sizeof(float);
        devices_[to]->b->write(*x.buffer, (x.offset + base * E) * sizeof(float), rows_out, rows * E * sizeof(float));
    }

    // A crossing inside a stage, both halves at once.
    void cross(ExecContext& ctx, size_t from, size_t to, size_t base, size_t rows) {
        send(ctx, from, 0, base, rows);
        receive(ctx, from, 0, devices_[from]->b->submit(), to, base, rows);
    }

    // Whether rows of this extent take a streamed layer on the device (Placement::stream_from): a prompt long enough, two tokens at the least, never a generated token.
    bool streams(size_t extent) const { return place_.stream_from && extent >= std::max<size_t>(place_.stream_from, 2); }
    bool streams(const Pass& p, size_t e) const { return streams(p.runs[e].extent); }

    // A streamed layer in a pass with long runs: consecutive entries alike form a group, the long ones run where the residual is on the layer's streamed row, with each window role's bytes written into its window once, and the rest on the host through a crossing each way.
    // The residual ends where it started, on the layer's mixer device.
    void ffn_split(ExecContext& ctx, const Pass& p, size_t dev, int l) {
        const std::vector<Weight>& home = home_[(size_t)l];
        const std::vector<Weight>& streamed = stream_[(size_t)l];
        const size_t host = (size_t)place_.ffn_device[(size_t)l];
        const uint8_t kind = plan_.layers[(size_t)l].kind;
        bool copied = false;
        for (size_t e = 0, base = 0; e < p.runs.size();) {
            const bool on_device = streams(p, e);
            ctx.part_runs.clear();
            size_t end = base;
            for (; e < p.runs.size() && streams(p, e) == on_device; ++e) {
                end = p.runs[e].end;
                ctx.part_runs.push_back(backend::RowRun{end - base, p.runs[e].extent});
            }
            const backend::RowRuns runs{ctx.part_runs.data(), ctx.part_runs.size()};
            if (on_device) {
                if (!copied) {
                    backend::Backend& b = *devices_[dev]->b;
                    for (const Role& role : plan_.layers[(size_t)l].roles)
                        if (role.stream == Stream::window)
                            b.write(*streamed[role.id].data, 0, home[role.id].data->host_ptr(), home[role.id].data->size());
                    copied = true;
                }
                arch_->ffn(part(ctx, dev, streamed.data(), kind, base, end - base, runs));
            } else {
                cross(ctx, dev, host, base, end - base);
                arch_->ffn(part(ctx, host, home.data(), kind, base, end - base, runs));
                cross(ctx, host, dev, base, end - base);
            }
            base = end;
        }
    }

    // A call of a part on device `dev`: `rows` rows of the residual from row `base` with their runs, and the row of weights `w` by role id, the layer's kind with it.
    Step part(ExecContext& ctx, size_t dev, const Weight* w, uint8_t kind, size_t base, size_t rows, backend::RowRuns runs) const {
        const ExecContext::Scratch& sc = ctx.scratch[dev];
        const Device& d = *devices_[dev];
        return Step{*d.b, sc.arena.get(), sc.offset.data(), {sc.arena.get(), sc.offset[0] / sizeof(float) + base * plan_.residual},
                    rows, runs, w, kind, nullptr, 0, 0, nullptr, d.tables.data(), &ctx.entry_runs};
    }

    // Layer l's mixer over every row of the pass, with the cache views of its device's storage when the layer keeps KV, and the rows' positions.
    Step mixer_part(ExecContext& ctx, const Pass& p, size_t dev, int l) const {
        const Device& d = *devices_[dev];
        const LayerPlan& layer = plan_.layers[(size_t)l];
        Step s = part(ctx, dev, home_[(size_t)l].data(), layer.kind, 0, p.rows, {p.runs.data(), p.runs.size()});
        if (layer.cache == Cache::kv) {
            s.views = p.views[(size_t)d.storage_index].data();
            s.n_views = p.entries.size();
            s.kv_layer = (size_t)d.local_layer[(size_t)l];
        } else if (layer.cache == Cache::state) {
            s.states = p.states[dev].data();
            s.n_views = p.entries.size();
            s.state_layer = (size_t)d.local_layer[(size_t)l];
        }
        s.pos = p.pos.data();
        return s;
    }

    // A block returns to the pool only once the backend has retired every submission that touched it (docs/KV-CACHE.md).
    // Construction failures, failed passes and model teardown drain every used device with sync(), including work behind no ticket; reset() waits on tickets instead.
    void retire() noexcept {
        for (auto& d : devices_) if (d->used) d->b->sync();
    }
};

} // namespace infer
