#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <functional>
#include <exception>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>
#include <optional>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "core/host_memory.hpp"
#include "format/gguf.hpp"
#include "backends/backend.hpp"
#include "model/weights.hpp"
#include "model/architecture.hpp"
#include "model/kv_cache.hpp"
#include "model/layer_split.hpp"
#include "model/shard.hpp"
#include "backends/cpu/cpu_backend.hpp"

// The model runtime: sequences, passes over batches of them, stages over devices, the activation arena and the crossings between devices, running an architecture's plan and parts (model/architecture.hpp).
// The compute primitives are delegated to a backend::Backend, so the same code runs on every backend, and the math of each part to the architecture, so the runtime names none.

namespace infer {

// The plan of a model's weights: their tensors indexed once (a repeated name refused), the architecture's plan over them with its embedded drafter's when `drafter` asks (Architecture::plan_drafter), and each role's tensor by name or alias.
// The fit, the experts placement and the model all read it, and a plan whose slot 0 is not the residual's width, or with a role id past its row of weights, is the architecture's error.
inline ModelPlan plan_model(const ModelWeights& weights, bool drafter = false) {
    const TensorIndex tensors(weights.tensors);
    ModelPlan plan = weights.arch->plan(tensors);
    if (drafter) weights.arch->plan_drafter(tensors, plan);
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
    if (plan.drafter)
        for (Role& role : plan.drafter->roles) resolve(role);
    return plan;
}

class Model;

// Where each tensor role runs, as an index into the model's backends, with empty meaning everything on device 0.
// Per role rather than per layer, so a layer's mixer and its feed-forward block can sit on different devices, as expert offload places them (docs/EXECUTION.md).
struct Placement {
    std::vector<int> mixer_device, ffn_device;
    int embed_device = 0, output_device = 0;
    // A routed layer with its feed-forward block on a host and its mixer on a device runs a prompt of at least this many tokens on the device, its experts copied there for each pass.
    // It counts a prompt's whole length (BatchEntry::extent), so a prompt takes one path however it is sliced (row_class); zero keeps every run on the host, and 1 streams what 2 does, since an extent of 1 never streams.
    size_t stream_from = 0;
    // A tensor split (docs/TENSOR-SPLIT.md): each device named above is the first member of a group of `width` consecutive backends that run its roles together, each member its shard of every split role; 1 is one device a name.
    size_t width = 1;
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
    backend::Dtype dtype = backend::Dtype::f16;
    std::vector<backend::Dtype> device_dtypes; // empty applies dtype to every device
    // The whole KV budget backed as the model is made rather than as passes write it, so no pass grows the cache (the budget every placement fits, fitted_kv in model/place.hpp).
    bool kv_backed = false;
    // What the host can still give, in place of core::host_memory_available, for tests of the host pool under pressure.
    std::function<std::optional<size_t>()> host_memory;
};

// An image's rows in a history (docs/VISION.md, sections 4 and 5): its first row and its rows, the encoder's merged grid they fill row by row, its digest, and the encoder's rows, which a pass writes over their pad rows after the embedding.
// The rows, held until read and possibly only those from `from` on, and the digest, zero where no tier compares it, are what a fork, a host copy and the owner's giving it again are held to.
struct MediaRun {
    size_t first = 0, rows = 0;
    uint32_t grid_h = 0, grid_w = 0;
    std::array<uint8_t, 32> digest{};
    std::shared_ptr<const std::vector<float>> encoded;
    size_t from = 0;
};

// One request's history in a model's cache, made by Model::make_sequence: the committed length of each stage, a block table per KV storage, and per device the ticket of the last pass that touched it (docs/EXECUTION.md).
// Movable, not copyable; from Model::begin_pass until its passes end it is in flight, when no reset or fork takes it and it must not move, and its prompt may continue in later passes (Model::begin_pass).
class Sequence {
public:
    Sequence() = default;
    // The first stage's committed length.
    // The stages can disagree while a pass is part way through them, and the model continues a history from the first stage's (Model::history).
    size_t length() const { return stage_length(0); }
    bool in_flight() const { return flights_ > 0; }
    // Whether its history is still the one a restore's parts copy into, `tag` the restore's (Model::StageItem::restore).
    bool restoring(const void* tag) const { return deferred_ && deferred_.get() == tag; }
    // Whether a restore's copies into its history are still landing on the stages' threads; such a history is neither forked from by another nor kept as a donor.
    bool landing() const {
        if (!deferred_ || !deferred_->restore) return false;
        std::lock_guard<std::mutex> lk(deferred_->m);
        return deferred_->left > 0;
    }
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
    // Its device work handed to the stages' threads (Model::retract with a Defer): the parts still to run, each device's ticket as its part submits, a part's failure, and the mark's hold, which goes back once every part has run.
    struct Deferred {
        std::mutex m;
        std::condition_variable cv;
        size_t left = 0;
        std::vector<backend::Ticket> last;
        std::string error;
        MarkHold hold;
        bool restore = false;   // copies into the history from host memory, which a fork of it shares (Model::restore_host with a Defer)
    };
    std::shared_ptr<Deferred> deferred_;
    const Model* owner_ = nullptr;
    size_t flights_ = 0;                   // its passes in flight
    // Its images in history order, each with its rows uploaded to every embedding device that has read them (Model::add_media).
    struct Media {
        MediaRun run;
        std::vector<backend::BufferPtr> on;
    };
    std::vector<Media> media_;
};

class Model;

// A history's first `length` tokens copied to host memory (Model::save_host), restored and released only by the model that wrote it; per device it holds slabs with its KV blocks, the state slot if any, and the last copy's ticket.
// HostPending is a copy whose device parts run on the stages' threads (Model::save_host with a Defer): parts left, fresh slabs and tickets, and the first error, taken by Model::host_settle.
struct HostPending {
    std::mutex m;
    std::condition_variable cv;
    size_t left = 0;
    std::vector<size_t> wanted;                            // per device, the fresh slabs its part was to allocate
    std::vector<std::vector<backend::BufferPtr>> fresh;    // and those it did
    std::vector<backend::Ticket> tickets;
    std::string error;
};

// A copy whose parts are pending completes in place once they have run, which a reader holding it const may do (Model::host_settle), so its slabs, tickets and error are mutable.
struct HostHistory {
    const Model* owner = nullptr;
    size_t length = 0, bytes = 0, held = 0;
    std::vector<size_t> device_bytes;   // per device, the bytes of its runs, in the order its slabs hold them
    bool blocks = true;   // false for a state alone, at `length`, whose blocks a history on the devices holds (Model::fork with a state)
    size_t first = 0;     // the token its blocks begin at: past 0 for the blocks of a range alone (Model::save_host_blocks)
    bool state = true;    // false for blocks alone, with no state after them
    std::vector<MediaRun> media;   // the images below `length`, without their rows, which a restore carries
    mutable std::vector<std::vector<backend::BufferPtr>> slabs;
    mutable std::vector<backend::Ticket> tickets;
    mutable std::shared_ptr<HostPending> pending;
    mutable std::string error;    // a part that failed: the copy holds nothing a reader may take
};

// A history being restored from an image laid out as a whole host copy (Model::restore_into): its sequence, whose blocks and checkpoint slot are taken, and per device where each byte of that device's image goes, in the image's order.
// Its blocks hold rows from the start, so nothing reads or forks it until every copy into it has retired, which its caller holds to; Model::restored then gives its passes the copies' tickets.
struct RestoreInto {
    // `bytes` at `offset` of host memory, a fixed buffer, or a KV layer side resolved at each copy since growth replaces its buffers.
    struct Run {
        backend::Buffer* buffer = nullptr;
        const backend::BlockKVStorage* kv = nullptr;
        size_t layer = 0;
        unsigned side = 0;
        size_t offset = 0, bytes = 0;
        uint8_t* host = nullptr;
        backend::Buffer& target() const { return kv ? *(side == 0 ? kv->k_buffer(layer) : side == 1 ? kv->v_buffer(layer) : kv->auxiliary_buffer(layer)) : *buffer; }
    };
    Sequence seq;
    size_t length = 0;
    std::vector<std::vector<Run>> runs;
    std::vector<backend::Backend*> backends;   // per device, the backend its runs are copied on
    std::vector<size_t> stages;                // per device, the stage whose thread records on it (Model::StageItem)
};

// A stretch of a host history on one device: `bytes` at `offset` of that device's slabs taken end to end.
struct HostRange {
    size_t device = 0, offset = 0, bytes = 0;
};

// What one sequence contributes to a pass: `n` tokens appended to `seq`, and whether the logits after its last token are wanted.
// A prefill microbatch is one entry with many tokens and a decode batch many entries with one, mixing freely, and a sequence appears in a batch at most once.
struct BatchEntry {
    Sequence* seq;
    const uint32_t* ids;
    size_t n;
    bool want_logits;
    // The logits after every token of the entry rather than only its last, for scoring a text through the same batched passes a prompt takes; with want_logits.
    bool every_logits = false;
    // What a device chooses this entry's kernels by (backend::RowRun) and a streamed layer its path (Placement::stream_from): a prompt's rows the position one past its last token, generated tokens 1, zero the entry's own row count.
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
    bool image_rows = false;                           // some row is an image's, so each row holds the plan's three position streams (ModelPlan::image_streams)
    std::vector<std::pair<size_t, std::vector<uint32_t>>> group_pos;   // where a history holds an image, per group size of a pooled layer, the positions of the groups each entry completes
    std::vector<uint32_t> previous_tokens;             // the token-history layer's preceding IDs, a fixed-width row per entry
    std::vector<std::vector<backend::KVView>> views;   // per storage, per entry
    std::vector<std::vector<backend::StateView>> states;   // per device, per entry, on a device whose layers keep a state
    std::vector<Checkpoint> kept;                      // per entry, the checkpoint a keep entry writes, its sequence's once the last stage commits it
    std::vector<backend::RowRun> runs, head_runs;      // the pass's rows and the head's, by entry
    size_t handoff = 0;                                // which of each device's handoff buffers its crossings use: a prompt chunk's parity, a reserved pass's slot
    size_t logits_base = 0;                            // the context's logits row its head writes first
    size_t at = 0;                                     // the device its residual left the last stage from
    backend::Ticket sent = 0;                          // the submission that copied it out, after the last stage the head's
    size_t sent_at = 0;                                // the device `sent` is on
    mutable double lookup_ms = 0, sum_ms = 0;          // the last stage recorded: the host's waits for its lookup rows and in its tensor-group sums
    std::vector<backend::Ticket> sent_members;         // on a tensor split, after the last stage each member of the head's group's, whose copies of its logits rows the pass's logits wait for
    bool in_flight = false;                            // a slot's pass between begin_pass and end_pass or abort_pass
    std::vector<backend::Ticket> tickets;              // per device, its stage's submission there, which its histories take as their last
    size_t ran = 0;                                    // the stages run_pass_stage has recorded
};

// Where a context's passes run, as plain data Model fills: an activation arena per device, host-visible handoff buffers on each device a crossing leaves, the logits rows on the output device, and the tickets of the submissions.
// A forward grows it to the largest pass seen, while Model::reserve_passes sizes it once for passes in flight, each slot with its own handoff buffers and logits rows, and replaces nothing after that.
struct ExecContext {
    // Row i of the logits the last forward produced, in entry order, valid until the next forward through this context.
    // The first read waits on the pass's ticket, so forward itself never blocks: a caller with two contexts submits the next pass before it reads this one.
    const float* logits(size_t i) {
        if (!logits_buf || i >= n_logits)
            throw std::out_of_range("inference: no such logits row");
        if (pending) {
            backend->wait(ticket);
            for (const auto& [b, t] : member_waits) b->wait(t);
            backend->raise(ticket);
            for (const auto& [b, t] : member_waits) b->raise(t);
            pending = false;
        }
        const void* p = logits_host.data() ? logits_host.data() : logits_buf->host_ptr();
        if (!p) throw std::runtime_error("inference: logits are not host visible");
        return (const float*)p + i * width;
    }
    size_t n_logits = 0;
    size_t width = 0;
    backend::Ticket ticket = 0;
    backend::Backend* backend = nullptr;
    std::vector<std::pair<backend::Backend*, backend::Ticket>> member_waits;   // on a tensor split, the head group's other members' submissions the logits also wait for
    bool pending = false;

    struct Scratch {
        struct LookupScratch {
            std::unique_ptr<WeightRows::Scratch> host;
            std::vector<uint32_t> ids;
            backend::BufferPtr data;
            PreparedLookup view;
        };
        backend::BufferPtr arena;
        size_t rows = 0;
        std::vector<size_t> offset;            // bytes, per slot of the plan
        std::vector<backend::RowRun> entry_runs;   // the run list a part on this device may rebuild (Step::scratch): a device's own, since stages are recorded on a thread each
        std::vector<uint32_t> indices;            // host lookup scratch sized with the arena, consumed synchronously by backend gather calls
        std::vector<LookupScratch> lookups;
    };
    std::vector<Scratch> scratch;              // per device
    // On a tensor split, the logits rows in host memory every member of the head's group imports (logits_buf the first member's view of them, member_rows each member's), into which each copies its vocabulary slice of each row.
    // Declared before every buffer that wraps them, so the views go first: a device keeps imported memory for as long as its buffer lives.
    core::HostPages logits_host;
    backend::BufferPtr logits_buf;
    size_t logit_rows = 0;
    std::vector<std::vector<backend::BufferPtr>> handoff;   // per device
    size_t handoff_rows = 0;
    std::vector<Pass> passes;
    size_t slots = 0, pass_rows = 0;           // what reserve_passes froze it for: its pass slots and the rows a pass may take; zero slots while it grows
    std::vector<backend::RowRun> part_runs;    // a streamed layer's group of entries, rebased
    std::vector<backend::Ticket> tickets;      // per device
    std::vector<backend::CSlice> carry;        // per entry, the row an embedded drafter's first context row reads (DraftRowsStep::carry)
    // On a tensor split (Placement::width): per device, a group's collective on its first member, for the rows the arenas hold, and each member of the head's group its slice of the logits rows, which the first member gathers into logits_buf.
    std::vector<std::unique_ptr<backend::Collective>> collectives;
    size_t collective_rows = 0;
    // Per device of a group that runs routed experts, zero rows as many as its collective sums, which a member's partial rows are cleared from before a part that adds to them (clear_partial).
    std::vector<backend::BufferPtr> zeros;
    std::vector<backend::BufferPtr> member_logits;
    std::vector<backend::BufferPtr> member_rows;   // each member's view of logits_host
};

// Prompt tokens a pass takes by default (Model::set_ubatch), and so the prompt rows a placement is fitted for.
inline constexpr int kDefaultUbatch = 512;

// The positions a model's caches are budgeted for: the options' tokens, else the whole context.
inline size_t kv_tokens(const ModelPlan& plan, const ModelOptions& options) {
    return options.kv_tokens ? options.kv_tokens : plan.context_length;
}

// The bytes one layer keeps per position: K and V at the requested types, with any auxiliary row kept in F32.
inline size_t kv_bytes_per_position(const ModelPlan& plan, const ModelOptions& options, size_t auxiliary = 0) {
    return backend::size_add(backend::size_mul(backend::size_mul(plan.kv_heads, plan.head_dim),
                             backend::kv_elem_bytes(options.kv_k) + backend::kv_elem_bytes(options.kv_v)),
                             backend::size_mul(auxiliary, sizeof(float)));
}

class Model {
public:
    // The realized device topology, captured once after placement validation. Indices refer to this model's backend vector; tensor groups are consecutive members.
    struct Topology {
        enum class Mode { single, layer, tensor, staged_tensor };
        struct Device { size_t index = 0, member = 0; bool used = false; };
        struct Stage { size_t index = 0; int first_layer = 0, end_layer = 0; std::vector<size_t> members; };
        struct Lookup {
            size_t layer = 0, rows = 0, row_bytes = 0, bytes = 0;
            uint32_t type = 0;
            bool file_backed = false;
            std::vector<size_t> members;
        };
        struct Layer {
            size_t index = 0;
            std::vector<size_t> mixer_members, ffn_members, stream_members;
            bool routed = false;
            size_t stream_min_prompt_tokens = 0;
        };
        Mode mode = Mode::single;
        bool mixed_roles = false;
        size_t tensor_width = 1;
        std::vector<Device> devices;
        std::vector<Stage> stages;
        std::vector<size_t> embedding_members, head_members;
        std::vector<Layer> layers;
        std::vector<Lookup> lookups;
    };

    // Construct from a model's weights on one backend (the CPU by default) or over several with a placement of every role, each weight put on its host backend by `adopt`, planned here or given the plan plan_model made.
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
        // The reserve the host pool leaves the host is fixed here, from what the host has before this model takes its buffers, as the fit's is, and does not shrink as memory falls.
        host_reserve_ = backend::CpuBackend::host_reserve(host_available().value_or(0));
        for (const auto& b : backends)
            if (!b) throw std::runtime_error("inference: missing backend");
        if (!options_.device_dtypes.empty() && options_.device_dtypes.size() != backends.size())
            throw std::runtime_error("inference: dtype policy does not cover every device");
        const size_t n_layer = plan_.layers.size();
        for (const LayerPlan& layer : plan_.layers) {
            if (layer.lookup) {
                const auto role = std::find_if(layer.roles.begin(), layer.roles.end(), [&](const Role& r) { return r.id == layer.lookup->role; });
                if (!layer.lookup->rows || role == layer.roles.end() || role->id >= plan_.role_ids ||
                    role->kind != RoleKind::gather || role->part != Part::mixer)
                    throw std::logic_error("inference: lookup needs a mixer gather role and rows");
            }
            if (layer.kv_aux && layer.cache != Cache::kv)
                throw std::logic_error("inference: auxiliary KV rows need a KV layer");
            kv_aux_ = backend::size_add(kv_aux_, backend::size_mul(layer.kv_aux_floats(), sizeof(float)));
            if (!layer.token_history) continue;
            if (token_width_ || layer.cache != Cache::state || layer.token_padding >= plan_.vocab)
                throw std::logic_error("inference: token history needs one recurrent layer and a padding ID in the vocabulary");
            token_width_ = layer.token_history;
            token_padding_ = layer.token_padding;
        }
        if (plan_.drafter) kv_aux_ = backend::size_add(kv_aux_, backend::size_mul(plan_.drafter->kv_aux_floats(), sizeof(float)));

        if (place_.mixer_device.empty() && place_.ffn_device.empty()) {
            place_.mixer_device.assign(n_layer, 0);
            place_.ffn_device.assign(n_layer, 0);
        }
        if (place_.mixer_device.size() != n_layer ||
            place_.ffn_device.size() != n_layer)
            throw std::runtime_error("inference: placement does not cover every layer");
        // A tensor group is named by its first member, its other members the backends after it (Placement::width).
        width_ = place_.width;
        if (!width_ || backends.size() % width_)
            throw std::runtime_error("inference: a tensor width of " + std::to_string(width_) + " needs whole groups of devices, and " + std::to_string(backends.size()) + " were given");
        auto device_index = [&](int d) {
            if (d < 0 || (size_t)d >= backends.size())
                throw std::runtime_error("inference: placement names a device the model does not have");
            if ((size_t)d % width_) throw std::runtime_error("inference: placement names a device that is not the first of its tensor group");
            return (size_t)d;
        };
        device_index(place_.embed_device);
        device_index(place_.output_device);
        if (width_ > 1) {
            shard::check_plan(plan_, weights.tensors, width_);
            if (place_.stream_from) throw std::logic_error("inference: a tensor group streams no layer");
            // A group's members are one kind of device, whose weight types and ops its first member's checks below stand for, with a collective among them.
            for (size_t g = 0; g < backends.size(); g += width_) {
                std::vector<backend::Backend*> members;
                for (size_t m = g; m < g + width_; ++m) {
                    if (backends[m]->is_cpu() != backends[g]->is_cpu()) throw std::runtime_error("inference: a tensor group's devices are of one kind");
                    members.push_back(backends[m].get());
                }
                if (!backends[g]->join(members, 1, 1)) throw std::runtime_error("inference: the backends of a tensor group have no cross-device sum");
            }
        }
        std::vector<bool> used(weights.tensors.size(), false);
        auto type_name = [](uint32_t id) {
            const quant::StorageType* type = quant::storage_type(id);
            return type ? std::string(type->name) + " (" + std::to_string(id) + ")" : std::to_string(id);
        };
        auto accepts = [&](const Role& role, size_t device) {
            return !role.tensor || backends[device]->supports_type(weights.tensors[*role.tensor].type);
        };
        auto require_type = [&](const Role& role, size_t device, const std::string& part, bool lookup = false) {
            if (role.tensor && weights.tensors[*role.tensor].row_source && !lookup)
                throw std::runtime_error("inference: external row source needs a declared lookup " + role.name);
            if (role.tensor) {
                used[*role.tensor] = true;
                if (weight_scale_count(weights, *role.tensor)) used[*weights.tensors[*role.tensor].scale] = true;
            }
            if (!accepts(role, device)) {
                const TensorView& t = weights.tensors[*role.tensor];
                throw std::runtime_error("inference: " + part + " needs tensor " + t.name + " of type " + type_name(t.type) +
                                         ", which the backend of device " + std::to_string(device) + " does not support");
            }
        };
        for (const Role& role : plan_.pass)
            require_type(role, device_of(role.part, 0), role.part == Part::embed ? "embedding" : "head");
        for (const OpUse& u : plan_.pass_ops) {
            if (u.part != Part::embed && u.part != Part::head)
                throw std::runtime_error("inference: a pass op must belong to the embedding or head");
            if (!backends[device_of(u.part, 0)]->implements(u.op))
                throw std::runtime_error(std::string("inference: the ") + (u.part == Part::embed ? "embedding" : "head") +
                                         " needs " + backend::op_name(u.op) + ", which the backend of its device does not implement");
        }
        // An embedded drafter runs beside the head, every role and op of it on the head's device.
        if (plan_.drafter) {
            for (const Role& role : plan_.drafter->roles) require_type(role, device_of(Part::draft, 0), "embedded drafter");
            for (const OpUse& u : plan_.drafter->ops)
                if (!backends[device_of(Part::draft, 0)]->implements(u.op))
                    throw std::runtime_error(std::string("inference: the embedded drafter needs ") + backend::op_name(u.op) +
                                             ", which the backend of its device does not implement");
        }
        // Home weights must run on their assigned device before anything is adopted; a stream destination that lacks a weight type leaves the whole layer at home.
        stream_device_.assign(n_layer, -1);
        for (size_t l = 0; l < n_layer; ++l) {
            const LayerPlan& layer = plan_.layers[l];
            const size_t a = device_index(place_.mixer_device[l]), f = device_index(place_.ffn_device[l]);
            // Experts read in place on the host beside a mixer on a device that copies its weights.
            bool streamed = layer.routed && place_.stream_from && a != f && !backends[a]->reads_in_place() && backends[f]->reads_in_place();
            for (const Role& role : layer.roles) {
                if (role.tensor && weights.tensors[*role.tensor].row_source) {
                    const TensorView& t = weights.tensors[*role.tensor];
                    if (!layer.lookup || layer.lookup->role != role.id || !layer.lookup->rows ||
                        role.kind != RoleKind::gather || role.part != Part::mixer || role.stream != Stream::none ||
                        role.shard.axis != Axis::none || role.in != t.row_source->columns() || role.out != t.row_source->rows() ||
                        t.type != t.row_source->type() || t.bytes != t.row_source->bytes())
                        throw std::runtime_error("inference: incompatible external row source " + t.name);
                }
                require_type(role, device_of(role.part, l), "layer " + std::to_string(l) + "'s " +
                             (role.part == Part::mixer ? "mixer" : "feed-forward part"), layer.lookup && layer.lookup->role == role.id);
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
        // Metadata-only file access admits known storage layouts; model construction still refuses an unreadable unused tensor.
        for (size_t i = 0; i < weights.tensors.size(); ++i) {
            const TensorView& t = weights.tensors[i];
            if (!used[i] && std::none_of(backends.begin(), backends.end(), [&](const backend::BackendPtr& b) { return b->supports_type(t.type); }))
                throw std::runtime_error("inference: unused tensor " + t.name + " of type " + type_name(t.type) + " is not supported by any model backend");
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
        // The drafter's KV is one more layer of the head's device's storage, which the last stage reserves and commits with its own, so the head sits there.
        if (plan_.drafter) {
            if ((size_t)place_.output_device != stages_.back().device)
                throw std::runtime_error("inference: an embedded drafter runs on the last stage's device, where the head must be");
            if (!state_layers_) throw std::logic_error("inference: an embedded drafter carries its row in a state slot, which this model has none of");
            Device& o = *devices_[(size_t)place_.output_device];
            drafter_kv_ = (size_t)o.kv_layers++;
            ++kv_layers_;
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
        // A tensor group's other members hold the layers and caches its first member does and run what it runs, each on its shards; the residual leaves the group from its first member.
        if (width_ > 1) {
            bool beside = place_.embed_device == (int)stages_.front().device && place_.output_device == (int)stages_.back().device;
            for (size_t l = 0; l < n_layer; ++l) beside = beside && place_.ffn_device[l] == place_.mixer_device[l];
            if (!beside) throw std::runtime_error("inference: a tensor split runs the embedding on the first stage's group, the head on the last's and each feed-forward block beside its mixer");
            for (size_t d = 0; d < devices_.size(); ++d) {
                Device& m = *devices_[d];
                const Device& first = *devices_[d - d % width_];
                m.member = d % width_;
                if (!m.member) continue;
                m.used = first.used;
                m.mixer_layers = first.mixer_layers;
                m.kv_layers = first.kv_layers;
                m.state_layers = first.state_layers;
                m.local_layer = first.local_layer;
            }
        }
        try {
            resolve_tensors(weights, adopt);
            topology_ = make_topology();

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
                // A tensor group's members each keep their KV heads, in blocks of the one pool its first member holds, so one block table serves them all.
                std::vector<size_t> auxiliary((size_t)d.kv_layers, 0), pools((size_t)d.kv_layers, 0);
                for (size_t l = 0; l < plan_.layers.size(); ++l)
                    if (plan_.layers[l].cache == Cache::kv && d.local_layer[l] >= 0) {
                        auxiliary[(size_t)d.local_layer[l]] = plan_.layers[l].kv_aux;
                        pools[(size_t)d.local_layer[l]] = plan_.layers[l].kv_pool;
                    }
                if (plan_.drafter && (size_t)(&dp - devices_.data()) / width_ == (size_t)place_.output_device / width_) {
                    auxiliary[drafter_kv_] = plan_.drafter->kv_aux;
                    pools[drafter_kv_] = plan_.drafter->kv_pool;
                }
                d.storage = d.b->kv_alloc((size_t)d.kv_layers, shard::kv_heads(plan_, width_), plan_.head_dim,
                                          budget, options_.kv_k, options_.kv_v, auxiliary, pools);
                if (options_.kv_backed) d.storage->back_all();
                if (d.member) {
                    d.storage_index = devices_[(size_t)(&dp - devices_.data()) - d.member]->storage_index;
                    continue;
                }
                d.pool.configure(d.storage->max_blocks());
                d.storage_index = (int)storages_.size();
                storages_.push_back(&d);
            }
            // Each device whose mixer layers keep a state holds every slot of theirs from now on, zeroed, so no pass allocates state.
            // A tensor group's members each keep the state of their K and V heads and save a mark's inputs at their share of each width, under the slot ids the group shares.
            if (state_layers_) {
                const size_t slots = backend::size_add(backend::size_add(options_.state_slots, options_.checkpoint_slots), options_.mark_slots);
                if (token_width_) {
                    for (size_t s = 0; s < stages_.size(); ++s)
                        for (int l = stages_[s].first; l < stages_[s].end; ++l)
                            if (plan_.layers[(size_t)l].token_history) token_stage_ = s;
                    token_tails_.resize(backend::size_mul(slots, token_width_));
                    marked_tokens_.resize(backend::size_mul(options_.mark_slots, options_.mark_rows));
                }
                for (size_t di = 0; di < devices_.size(); ++di) {
                    Device& d = *devices_[di];
                    if (!d.state_layers) continue;
                    std::vector<size_t> tails((size_t)d.state_layers, 0);
                    for (size_t l = 0; l < n_layer; ++l)
                        if (place_.mixer_device[l] == (int)(di - d.member) && plan_.layers[l].cache == Cache::state)
                            tails[(size_t)d.local_layer[l]] = plan_.layers[l].state_tail;
                    d.states = d.b->state_alloc((size_t)d.state_layers, slots, shard::state(plan_, width_), tails);
                    // Each mark's room for the recurrent inputs of its rows, per state layer of the device.
                    if (!options_.mark_slots) continue;
                    for (size_t l = 0; l < n_layer; ++l)
                        if (place_.mixer_device[l] == (int)(di - d.member) && plan_.layers[l].cache == Cache::state)
                            d.saved_floats = std::max(d.saved_floats, shard::saved_floats(plan_.layers[l], width_));
                    if (!d.saved_floats) throw std::logic_error("inference: a state layer that saves no inputs for a mark");
                    // After every mark's inputs, each state layer's room for the slots its update writes, mark_rows rows of each, so a rerun reads the inputs where they were saved and runs the layers unordered.
                    d.rerun_base = backend::size_mul(backend::size_mul(options_.mark_slots, (size_t)d.state_layers), backend::size_mul(options_.mark_rows, d.saved_floats));
                    for (size_t l = 0; l < n_layer; ++l)
                        if (place_.mixer_device[l] == (int)(di - d.member) && plan_.layers[l].cache == Cache::state)
                            d.rerun_floats = std::max(d.rerun_floats, recur_floats(plan_, plan_.layers[l]));
                    d.saved = d.b->alloc(backend::size_mul(backend::size_add(d.rerun_base, backend::size_mul(backend::size_mul((size_t)d.state_layers, options_.mark_rows), d.rerun_floats)),
                                                           sizeof(float)));
                }
                slots_.configure(options_.state_slots, options_.checkpoint_slots, options_.mark_slots);
            }
            // The drafter's carried row, one a state slot on the head's device, a zero row a history of length 0 reads, and each mark's room for the normed rows of the pass after it.
            // Each member of the head's tensor group holds them, the rows being the same on every member.
            for (size_t m = 0; plan_.drafter && m < width_; ++m) {
                Device& o = *devices_[(size_t)place_.output_device + m];
                const size_t slots = backend::size_add(backend::size_add(options_.state_slots, options_.checkpoint_slots), options_.mark_slots);
                const size_t row = backend::size_mul(plan_.residual, sizeof(float));
                o.carry = o.b->alloc(backend::size_mul(slots, row));
                o.zero = o.b->alloc(row);
                if (options_.mark_slots) o.saved_h = o.b->alloc(backend::size_mul(backend::size_mul(options_.mark_slots, options_.mark_rows), row));
            }
            seq_ = make_sequence();

            // The architecture's tables can be read by any part, so every used device adopts them; CPU buffers borrow the shared host vectors for the model's lifetime.
            tables_.resize(plan_.tables.size());
            for (size_t t = 0; t < tables_.size(); ++t) tables_[t].assign(plan_.tables[t], 0.0f);
            arch_->fill_tables(tables_);
            for (auto& d : devices_)
                if (d->used)
                    for (const std::vector<float>& t : tables_) d->tables.push_back(d->b->adopt(t.data(), t.size() * sizeof(float)));
            // Each device of a placement over several waits while the others run their parts, which a device that waits on the host spends at an idle clock; a model made asks, and gives its request back when it goes.
            if (std::count_if(devices_.begin(), devices_.end(), [](const auto& d) { return d->used; }) > 1)
                for (auto& d : devices_) {
                    if (!d->used) continue;
                    d->b->hold_between_submissions(true);
                    d->holding = true;
                }
        } catch (...) {
            // Constructor members still exist here, so pending uploads retire before unwinding releases them.
            release_holds();
            retire();
            throw;
        }
    }

    ~Model() {
        release_holds();
        retire();
    }

    // Sequences hold the pools' addresses, so a Model is neither copied nor moved.
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // CPU worker counts, applied to every backend; a device backend ignores them.
    void set_threads(int n) { for (auto& d : devices_) d->b->set_threads(n); }

    // The cache pools a scheduler admits against, one per device whose mixer layers keep KV, each counted in its own blocks (docs/SERVER.md, docs/MULTI-DEVICE.md).
    size_t kv_pools() const { return storages_.size(); }
    size_t kv_pool_block_tokens(size_t s) const { return storages_.at(s)->b->kv_layout().block_tokens; }
    size_t kv_pool_blocks(size_t s) const { return storages_.at(s)->pool.max_blocks(); }
    // The blocks free in each pool now, and those sequence `q` holds there, shared ones and a pass's prepared ones included.
    std::vector<size_t> kv_free() const;
    std::vector<size_t> kv_held(const Sequence& q) const;
    // What dropping every sequence of `set` together frees, the one truth of sharing (docs/CACHE-PLAN.md, section 4): per pool the blocks no other sequence references, and the checkpoint slots none other holds.
    // `kept`, a fork still to be made of `kept_len` tokens of one of them, keeps the blocks it would share and its checkpoint.
    struct Freed {
        std::vector<size_t> blocks;
        size_t slots = 0;
    };
    Freed freed(const std::vector<const Sequence*>& set, const Sequence* kept = nullptr, size_t kept_len = 0) const;
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
    // The host bytes the lookup tables hold of their hot rows (WeightRows::store_bytes), each table once.
    size_t lookup_store_bytes() const {
        std::vector<const WeightRows*> seen;
        size_t bytes = 0;
        for (size_t l = 0; l < plan_.layers.size(); ++l) {
            if (!plan_.layers[l].lookup) continue;
            const WeightRows* source = home_[l][plan_.layers[l].lookup->role].row_source.get();
            if (!source || std::find(seen.begin(), seen.end(), source) != seen.end()) continue;
            seen.push_back(source);
            bytes = backend::size_add(bytes, source->store_bytes());
        }
        return bytes;
    }

    // Defined in model/history.hpp, the owner of a sequence's history (docs/SPECULATIVE.md, section 1).
    Sequence fork(const Sequence& src, size_t length);

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

    // Defined in model/passes.hpp, the owner of a pass and its stages.
    void forward(ExecContext& ctx, const BatchEntry* entries, size_t n_entries);

    // Passes in flight: a caller keeps several passes of different sequences in one context and runs their stages itself, so on a pipelined split every stage works while the host samples another (docs/MULTI-DEVICE.md, passes in flight).
    // Each pass's stages run in order, and passes interleave as the caller likes: each device runs the stages recorded on it in that order, and a pass keeps its own handoff buffer, logits rows and ticket.
    size_t stage_count() const { return stages_.size(); }
    // Whether passes may be in flight together: several stages, the embedding on the first stage's device, the head on the last's and every feed-forward block beside its mixer.
    bool pipelined() const { return pipelined_; }
    // Whether stage s runs on the host, whose backend computes as its work is recorded, so recording it holds the calling thread for the stage's whole time.
    bool stage_on_host(size_t s) const { return devices_[stages_.at(s).device]->b->is_cpu(); }
    // Whether stage s is a tensor group's, whose recording waits at every sum, each a submission its members wait on each other for.
    bool stage_grouped(size_t s) const { return stages_.at(s).device < devices_.size() && width_ > 1; }
    // Whether recording stage s waits on its devices: a tensor group's stage, and a device's on a pipelined split that receives the stage before it.
    // A caller that keeps passes in flight may record such a stage on a thread of that stage's (prepare_pass_stage, record_pass_stage, commit_pass_stage).
    bool stage_waits(size_t s) const { return stage_grouped(s) || (s > 0 && stages_.at(s).device < devices_.size() && pipelined_ && !stage_on_host(s)); }
    // External row preparation runs outside a caller's scheduler lock even with one pass on one device.
    bool stage_reads(size_t s) const {
        const Stage& stage = stages_.at(s);
        for (int l = stage.first; l < stage.end; ++l) {
            const auto& lookup = plan_.layers[size_t(l)].lookup;
            if (lookup && home_[size_t(l)][lookup->role].row_source) return true;
        }
        return false;
    }
    // The backend stage s runs on, which a caller timing the stages reads its host and device times from.
    backend::Backend& stage_backend(size_t s) { return *devices_[stages_.at(s).device]->b; }

    // Consume this model's completed matrix-path evidence, expert hosts included, in placement order.
    std::vector<std::vector<std::string>> take_matrix_paths() {
        std::vector<std::vector<std::string>> paths;
        for (auto& device : devices_) paths.push_back(device->matrix_paths.take());
        return paths;
    }

    // Defined in model/passes.hpp, the owner of a pass and its stages.
    void reserve_passes(ExecContext& ctx, size_t slots, size_t rows, size_t logit_rows);
    void begin_pass(ExecContext& ctx, size_t slot, const BatchEntry* entries, size_t n_entries, size_t logits_base);
    void run_pass_stage(ExecContext& ctx, size_t slot, size_t s, const std::function<void()>& before_undo = {});
    void prepare_pass_stage(ExecContext& ctx, size_t slot, size_t s);
    void record_pass_stage(ExecContext& ctx, size_t slot, size_t s);
    // That submission and its device's backend, which a caller timing the stage asks whether it has retired (docs/SERVER.md, The price table).
    std::pair<backend::Backend*, backend::Ticket> stage_sent(ExecContext& ctx, size_t slot) {
        const Pass& p = in_flight(ctx, slot);
        return {devices_[p.sent_at]->b.get(), p.sent};
    }
    // The last stage recorded of that pass: the milliseconds its recording waited for lookup rows and spent in tensor-group sums.
    std::pair<double, double> stage_host_waits(ExecContext& ctx, size_t slot) {
        const Pass& p = in_flight(ctx, slot);
        return {p.lookup_ms, p.sum_ms};
    }
    void commit_pass_stage(ExecContext& ctx, size_t slot);
    const float* pass_logits(ExecContext& ctx, size_t slot, size_t i);
    void end_pass(ExecContext& ctx, size_t slot);
    std::vector<size_t> abort_pass(ExecContext& ctx, size_t slot);

    // Defined in model/history.hpp, the owner of a sequence's history (docs/SPECULATIVE.md, section 1).
    void reset(Sequence& s);
    size_t retract(Sequence& s, size_t length);
    // One device's part of a history's work outside a pass, run by the caller on the thread that records that stage's passes; it throws as the work did, the history settling either way.
    // `restore`, where given, is the history a part copies into: its failure is reported with it, so its caller finds every holder of that history (Sequence::restoring).
    struct StageItem {
        size_t stage;
        std::function<void()> run;
        std::shared_ptr<const void> restore = {};
    };
    using Defer = std::function<void(StageItem)>;
    // Called with true before this thread waits for a deferred part on a stage's thread and with false after, so the caller lets that thread report without its own lock (the server's quiet).
    void on_stage_wait(std::function<void(bool)> f) { stage_wait_ = std::move(f); }
    size_t retract(Sequence& s, size_t length, const Defer& defer);
    bool mark(Sequence& s);
    bool can_draft(const Sequence& s) const;
    bool keep(Sequence& s);
    std::optional<size_t> checkpoint(const Sequence& s) const;
    size_t host_bytes(size_t length, bool blocks = true, size_t first = 0, bool state = true) const;
    std::string host_identity() const;
    void alloc_host(size_t length, HostHistory& out, size_t limit, bool blocks = true, size_t first = 0, bool state = true, std::vector<size_t>* deferred = nullptr,
                    const Defer* defer = nullptr);
    size_t host_fresh(size_t length) const;
    enum class HostFit { fits, limit, memory };
    HostFit host_copy_fits(size_t length, size_t limit, const std::vector<size_t>& freed) const;
    void save_host(Sequence& s, size_t length, HostHistory& out, size_t limit, bool blocks = true, const Defer& defer = {});
    void save_host_blocks(Sequence& s, size_t first, size_t length, HostHistory& out, size_t limit, const Defer& defer = {});
    bool host_settle(const HostHistory& h, bool wait) noexcept;
    bool release_settled(Sequence& s) noexcept;
    void wait_ticket(size_t device, backend::Ticket t) const noexcept;
    // Slabs for a copy as alloc_host takes them, with `defer` the fresh ones allocated on their devices' stages' threads and waited for, so the host can use them on return (the disk tier's reads).
    void take_host(size_t length, HostHistory& out, size_t limit, bool blocks, size_t first, bool state, const Defer& defer) {
        if (!defer) return alloc_host(length, out, limit, blocks, first, state);
        std::vector<size_t> fresh;
        alloc_host(length, out, limit, blocks, first, state, &fresh, &defer);
        grow_slabs(out, fresh, defer);
        wait_host(out);
        if (!out.error.empty()) {
            const std::string error = out.error;
            release_host(out);
            throw std::runtime_error("inference: host memory for the copy was not allocated (" + error + ")");
        }
    }
    // The fresh slabs alloc_host counted in `fresh` for `h`, allocated on their devices' stages' threads with nothing copied, as the idle growth takes them.
    void grow_slabs(HostHistory& h, const std::vector<size_t>& fresh, const Defer& defer) {
        copy_out(h, fresh, std::vector<std::vector<int32_t>>(devices_.size()), {}, 0, 0, false, defer);
    }
    std::vector<HostRange> host_ranges(const HostHistory& h, size_t first, size_t length) const;
    std::vector<HostRange> host_state_ranges(const HostHistory& h) const;
    size_t host_allocated() const;
    bool caches_on_devices() const;
    Sequence restore_host(HostHistory& h, const Defer& defer = {});
    Sequence fork(const Sequence& src, size_t length, HostHistory& state, const Defer& defer = {});
    RestoreInto restore_into(size_t length, const Defer& defer = {});
    HostHistory host_layout(size_t length) const;
    void restored(RestoreInto& r, const std::vector<backend::Ticket>& last);
    void release_host(HostHistory& h) noexcept;
    bool copies_retired(const std::vector<backend::Ticket>& tickets) const noexcept;
    bool copies_retired(const HostPending& p) const noexcept;
    void trim_host(size_t limit) noexcept;
    void trim_host(size_t limit, const Defer& defer);
    bool host_admits(size_t bytes, bool optional = false) const;
    bool host_short() const;
    size_t host_yield(const Defer& defer = {});
    void wait_host(const HostHistory& h) noexcept;
    // The bytes of each slab of a host history (HostHistory::slabs).
    static constexpr size_t host_slab_bytes() { return kHostSlab; }
    // One sequence's draft in a batch of drafts (draft).
    struct DraftAsk {
        Sequence* seq;
        uint32_t last;
        size_t k;
        std::vector<uint32_t>* out;
    };
    void draft(DraftAsk* asks, size_t n);
    // A batch of drafts in three parts, so a server records the chain on the head stage's thread (docs/src/model-history.md): begun and ended on the thread that owns the histories, recorded on the head's devices between.
    // Its sequences are in flight from draft_begin to draft_end, and its drafts are in its asks' `out` once it has ended recorded.
    struct Chain {
        std::vector<DraftAsk> asks;
        std::vector<size_t> order;             // the asks that draft, longest chain first
        size_t steps = 0;
        std::vector<uint32_t> last, carried;   // each row's last pick and the slot of the row its history carries
        std::vector<size_t> rows;              // each step's rows
        std::vector<uint32_t> pos;             // step m's positions from m * R
        std::vector<backend::KVView> views;    // step m's views on member g from (m * W + g) * R
        std::vector<KVSequence*> kv;
        size_t prepared = 0;                   // the KV sequences whose drafted rows are taken
        std::vector<backend::Ticket> sent;
        std::vector<uint32_t> ids;             // every member's drafted ids, read back as the chain is recorded
        bool recorded = false;
    };
    std::unique_ptr<Chain> draft_begin(const DraftAsk* asks, size_t n, bool behind = false);
    void draft_record(Chain& c);
    void draft_end(Chain& c);
    // Room for every mark's rerun and for a chain of every mark's sequence at the most drafts a mark holds, taken now, so neither allocates while a stage's thread records (Scheduler).
    void reserve_drafting();
    void draft(Sequence& s, uint32_t last, size_t k, std::vector<uint32_t>& out) {
        DraftAsk a{&s, last, k, &out};
        draft(&a, 1);
    }
    // The generated tokens every stage's products read each weight once for (Backend::decode_columns), what a pass's drafts ride on.
    size_t decode_columns() const {
        size_t n = SIZE_MAX;
        for (const auto& d : devices_) n = std::min(n, d->b->decode_columns());
        return n;
    }
    // The logits of draft row i of the last draft's sequence `r`, in the order it was asked for, valid until the next.
    const float* draft_logits(size_t i, size_t r = 0) const {
        if (draft_logits_.empty() || r >= draft_order_.size() || i >= draft_k_[r]) throw std::out_of_range("inference: no such draft row");
        return (const float*)draft_logits_[0]->host_ptr() + (i * draft_width_ + draft_order_[r]) * plan_.vocab;
    }
    // Checkpoint slots in all, and those a keep can still take.
    size_t checkpoint_slots() const { return state_layers_ ? options_.checkpoint_slots : 0; }
    // The marks sequences may hold at once; a model that keeps no state marks any number.
    size_t mark_slots() const { return state_layers_ ? options_.mark_slots : SIZE_MAX; }
    size_t checkpoints_free() const { return state_layers_ ? slots_.kept_available() : 0; }

    // Defined in model/history.hpp: an image given to a sequence, one it carries given again or one after every image it holds, its rows read when passes reach them.
    void add_media(Sequence& s, MediaRun run);
    // The same for the CLI's own sequence.
    void add_media(MediaRun run) { add_media(seq_, std::move(run)); }
    // Whether the model takes image rows: its architecture places them (ModelPlan::image_streams).
    bool takes_images() const { return !plan_.image_streams.empty(); }
    // The rows of a sequence's images, [first, end) in order, which a drafter does not draft from; the CLI's own sequence without one.
    std::vector<std::pair<size_t, size_t>> image_rows(const Sequence& s) const {
        std::vector<std::pair<size_t, size_t>> rows;
        for (const Sequence::Media& m : s.media_) rows.push_back({m.run.first, m.run.first + m.run.rows});
        return rows;
    }
    std::vector<std::pair<size_t, size_t>> image_rows() const { return image_rows(seq_); }
    // The width of an embedding row, which an encoder's rows must have.
    size_t embedding_width() const { return plan_.residual / plan_.embed_copies; }
    // The backend the embedding runs on, where an image is encoded.
    backend::Backend& embedding_backend() const { return *devices_[(size_t)place_.embed_device]->b; }

    // The single-sequence entry points the CLI uses: one sequence and one context owned here, and one entry per pass.

    // Run one token through the model (prefill or continue).
    // Returns logits over the full vocabulary, read in place and valid until the next pass.
    const float* step(int token_id) {
        const uint32_t id = (uint32_t)token_id;
        const BatchEntry entry{&seq_, &id, 1, true};
        forward(ctx_, &entry, 1);
        return ctx_.logits(0);
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
    void draft(uint32_t last, size_t k, std::vector<uint32_t>& out) { draft(seq_, last, k, out); }

    // Process a whole prompt with matrix-matrix matmuls instead of one token at a time, so each weight row is reused across the batch and only the last pass asks for the final token's logits.
    // A `keep_at` inside the prompt keeps the state there as the sequence's checkpoint, the chunk before it cut to end there; `media` are the prompt's images, counted from its first token.
    std::vector<float> prefill(const std::vector<uint32_t>& ids, size_t keep_at = 0, const std::vector<MediaRun>& media = {}) {
        if (ids.empty()) throw std::runtime_error("inference: empty prompt");
        // The prompt is one transaction across its microbatches: a failure in any of them restores the history from before the call.
        const size_t start = seq_.length();
        auto work = [&] {
            for (MediaRun run : media) {
                run.first += start;
                add_media(seq_, std::move(run));
            }
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
            run_chunks(ctx_, chunks, chunk);
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
    void score(const std::vector<uint32_t>& ids, const std::function<void(size_t, const float*)>& each, const std::vector<MediaRun>& media = {}) {
        if (ids.empty()) throw std::runtime_error("inference: empty text");
        reset();
        auto work = [&] {
            for (const MediaRun& run : media) add_media(seq_, run);
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
    bool keep() { return keep(seq_); }

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
        return seq_.length() * (kv_layers_ * kv_bytes_per_position(plan_, options_) + kv_aux_);
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

    const Topology& topology() const { return topology_; }

private:
    std::vector<size_t> group_members(int first) const {
        std::vector<size_t> members;
        for (size_t m = 0; m < width_; ++m) members.push_back((size_t)first + m);
        return members;
    }

    Topology make_topology() const {
        Topology t;
        t.tensor_width = width_;
        t.mode = width_ > 1 ? (stages_.size() > 1 ? Topology::Mode::staged_tensor : Topology::Mode::tensor)
                            : (stages_.size() > 1 ? Topology::Mode::layer : Topology::Mode::single);
        t.devices.reserve(devices_.size());
        for (size_t d = 0; d < devices_.size(); ++d)
            t.devices.push_back(Topology::Device{d, d % width_, devices_[d]->used});
        t.embedding_members = group_members(place_.embed_device);
        t.head_members = group_members(place_.output_device);
        t.stages.reserve(stages_.size());
        for (size_t s = 0; s < stages_.size(); ++s) {
            const Stage& stage = stages_[s];
            t.stages.push_back(Topology::Stage{s, stage.first, stage.end, group_members((int)stage.device)});
        }
        t.layers.reserve(plan_.layers.size());
        for (size_t l = 0; l < plan_.layers.size(); ++l) {
            Topology::Layer layer;
            layer.index = l;
            layer.mixer_members = group_members(place_.mixer_device[l]);
            layer.ffn_members = group_members(place_.ffn_device[l]);
            layer.routed = plan_.layers[l].routed;
            if (place_.mixer_device[l] != place_.ffn_device[l]) t.mixed_roles = true;
            if (stream_device_[l] >= 0) {
                layer.stream_members = group_members(stream_device_[l]);
                layer.stream_min_prompt_tokens = std::max<size_t>(place_.stream_from, 2);
            }
            if (plan_.layers[l].lookup) {
                const Weight& weight = home_[l][plan_.layers[l].lookup->role];
                const size_t rows = weight.row_source ? weight.row_source->rows() : weight.nout;
                const size_t row_bytes = weight.row_source ? weight.row_source->row_bytes() : quant::row_bytes(weight.type, weight.nin);
                const size_t bytes = weight.row_source ? weight.row_source->bytes() : backend::size_mul(rows, row_bytes);
                t.lookups.push_back(Topology::Lookup{l, rows, row_bytes, bytes, weight.type, bool(weight.row_source), layer.mixer_members});
            }
            t.layers.push_back(std::move(layer));
        }
        if (!stages_.empty()) {
            t.mixed_roles = t.mixed_roles || place_.embed_device != (int)stages_.front().device ||
                            place_.output_device != (int)stages_.back().device;
        }
        return t;
    }

    // One backend and what the placement put on it.
    // A pool is not movable, because sequences hold its address, so devices live behind pointers.
    struct Device {
        backend::BackendPtr b;
        backend::MatrixPaths matrix_paths;
        bool used = false;
        bool holding = false;                    // this model asked the backend to hold between submissions
        bool sends = false;                      // the residual leaves it, so it keeps handoff buffers
        size_t member = 0;                       // its place in its tensor group, 0 for the first member, which a placement names
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
        size_t rerun_base = 0;                   // where the rerun's rooms start in `saved`, in floats
        size_t rerun_floats = 0;                 // a row of one state layer's room (recur_floats)
        backend::BufferPtr carry, zero, saved_h; // on the head's device with an embedded drafter: its carried row a state slot, a zero row, and per mark and row the normed rows of the pass after it
        std::vector<backend::BufferPtr> tables;  // the position tables, on a device that runs a mixer
    };

    // Consecutive layers whose mixer runs on one device, and every device a stage records work on, which it submits.
    struct Stage {
        size_t device;
        int first, end;
        std::vector<size_t> touches;
    };

    Placement place_;
    Topology topology_;
    std::vector<std::unique_ptr<Device>> devices_;
    std::vector<Device*> storages_;              // the devices whose mixer layers keep KV
    size_t kv_layers_ = 0;                       // the model's layers that keep KV
    size_t kv_aux_ = 0;                          // auxiliary bytes per position over those layers
    size_t state_layers_ = 0;                    // and those that keep a state
    // Host-visible memory for copies of histories (Model::save_host), in slabs a released copy leaves per device for the next, since allocating and pinning host memory costs far more than copying into it.
    static constexpr size_t kHostSlab = size_t(64) << 20;
    std::vector<size_t> host_copy_bytes(size_t length, bool blocks, size_t first, bool state) const;
    void drain_recorded(const HostHistory& h, size_t submitted, size_t recorded) noexcept;
    void copy_out(HostHistory& h, const std::vector<size_t>& fresh, std::vector<std::vector<int32_t>> ids, std::vector<uint32_t> tail, size_t tail_at, size_t slot, bool state,
                  const Defer& defer);
    void free_slabs(std::vector<std::vector<backend::BufferPtr>> slabs, const Defer& defer);
    void copy_in(HostHistory& h, Sequence& s, std::vector<std::vector<int32_t>> ids, size_t slot, const Defer& defer);
    void read_tail(const HostHistory& h, size_t slot);
    void settled_whole(const HostHistory& h);
    std::optional<size_t> host_available() const { return options_.host_memory ? options_.host_memory() : core::host_memory_available(); }
    size_t host_reserve_ = 0;                                   // host memory the pool's fresh slabs leave free, fixed as the model is made
    // Growth nothing waits for keeps twice the reserve, so slabs given back at the reserve are not taken again while the host stays that short.
    static constexpr size_t kHostOptional = 2;
    std::vector<std::vector<backend::BufferPtr>> host_slabs_;   // per device, the idle slabs, each vector's capacity reserved for every slab of its device so a release does not allocate
    std::vector<size_t> host_allocated_;                        // per device, its slabs alive, idle or in a copy
    SlotPool slots_;
    size_t token_width_ = 0, token_stage_ = SIZE_MAX;
    uint32_t token_padding_ = 0;
    std::vector<uint32_t> token_tails_, marked_tokens_; // host IDs indexed by the existing state slot and mark buffer IDs
    std::vector<Stage> stages_;
    bool pipelined_ = false;                     // a prompt's chunks flow through the stages together (prefill)
    int ubatch_ = kDefaultUbatch;
    ModelOptions options_;
    std::shared_ptr<const Architecture> arch_;
    ModelPlan plan_;
    std::vector<Weight> pass_;                  // the pass's roles by role id, on the embedding's and the head's devices
    std::vector<std::vector<Weight>> home_;     // per layer, its roles by role id, each on the device of its part
    // A tensor group's other members' rows (Placement::width): per member past the first, the pass's and per layer the layer's, each weight that member's shard, and the packed copies of the shards the model adopted without a hook, which it keeps.
    size_t width_ = 1;
    std::vector<std::vector<Weight>> member_pass_, member_drafter_;
    std::vector<std::vector<std::vector<Weight>>> member_home_;
    std::vector<std::vector<uint8_t>> packed_;
    // A routed layer run beside its mixer for a long prompt (Placement::stream_from): the device it runs on, or -1, and home_'s row with its copy roles adopted on that device and its window roles in that device's windows.
    std::vector<int> stream_device_;
    std::vector<std::vector<Weight>> stream_;
    std::vector<std::vector<backend::BufferPtr>> windows_;   // per device, a buffer per window role in role order, sized to the largest streamed layer's
    std::vector<Weight> drafter_;                // an embedded drafter's roles by role id, on the head's device
    size_t drafter_kv_ = 0;                      // its KV layer in the head's device's storage
    // The last draft's ids, the last picks first, host visible, and its rows' logits per member of the head's group: the first member's for every row of every step, each other member's one step's rows, which only its argmax reads.
    std::vector<backend::BufferPtr> draft_ids_, draft_logits_;
    size_t draft_id_rows_ = 0, draft_rows_ = 0, draft_seqs_ = 0;   // the ids, the logits rows and a step's rows they hold
    std::vector<size_t> draft_order_, draft_k_;     // the last draft's sequences' places in its steps' rows and their chains' lengths, in the order they were asked for
    size_t draft_width_ = 0;                        // the last draft's sequences with a chain, the rows of its first step
    void draft_buffers(size_t seqs, size_t steps);
    std::vector<std::vector<float>> tables_;
    Sequence seq_;
    ExecContext ctx_;

    // The weights member m of the group a part runs on reads: the pass's, and layer l's.
    const Weight* pass_row(size_t m) const { return m ? member_pass_[m - 1].data() : pass_.data(); }
    const Weight* home_row(size_t m, size_t l) const { return m ? member_home_[m - 1][l].data() : home_[l].data(); }
    const Weight* drafter_row(size_t m) const { return m ? member_drafter_[m - 1].data() : drafter_.data(); }
    // The vocabulary rows member m of the head's group computes, its share of the head's matrix.
    size_t head_rows(size_t m) const {
        for (const Role& role : plan_.pass)
            if (role.part == Part::head && role.kind == RoleKind::matrix) return pass_row(m)[role.id].nout;
        throw std::logic_error("inference: a plan without a head");
    }

    size_t device_of(Part part, size_t l) const {
        if (part == Part::embed) return (size_t)place_.embed_device;
        if (part == Part::head || part == Part::draft) return (size_t)place_.output_device;
        return (size_t)(part == Part::mixer ? place_.mixer_device[l] : place_.ffn_device[l]);
    }

    // Resolve every role of the plan to a Weight, in plan order: the pass's roles, then each layer's followed by a streamed layer's copies.
    // Each role reads the tensor plan_model set (a role without one is refused) and is put on its part's backend by the caller's hook when given, once per device for a tensor two roles take, as a tied head beside the embedding.
    void resolve_tensors(const ModelWeights& weights, const AdoptWeight& adopt) {
        const size_t n_devices = devices_.size();
        std::vector<backend::BufferPtr> taken(weights.tensors.size() * n_devices);
        // A failed later adoption must drain while this local cache still owns earlier scale uploads.
        struct Pending {
            Model& model;
            int exceptions = std::uncaught_exceptions();
            ~Pending() { if (std::uncaught_exceptions() > exceptions) model.retire(); }
        } pending{*this};
        auto take = [&](size_t tensor, size_t device) {
            backend::BufferPtr& buffer = taken[tensor * n_devices + device];
            if (!buffer) {
                const TensorView& t = weights.tensors[tensor];
                backend::Backend& b = *devices_[device]->b;
                buffer = adopt ? adopt(tensor, b, nullptr) : b.adopt(t.data, t.bytes);
            }
            return buffer;
        };
        auto resolve = [&](const Role& role, size_t device, size_t member = 0) -> Weight {
            if (!role.tensor) throw TensorIndex::missing(role.alias.empty() ? role.name : role.alias);
            const size_t i = *role.tensor;
            const TensorView& t = weights.tensors[i];
            if (!role_fits(role, t.shape, t.type)) throw std::runtime_error("inference: incompatible tensor layout " + t.name);
            Weight weight{t.type, {}, size_t(role.in), size_t(role.out), t.row_source};
            weight.scale_count = weight_scale_count(weights, i);
            if (weight.scale_count) weight.scales = take(*t.scale, device + member);
            if (t.row_source) return weight;
            backend::Backend& b = *devices_[device + member]->b;
            // A member of a tensor group takes its shard of a split role, a buffer of its own even where another role reads the whole tensor there, as a tied head beside the embedding does.
            if (width_ > 1 && role.shard.axis != Axis::none) {
                const shard::Held held = shard::held(role, t, width_, member);
                backend::BufferPtr buffer;
                if (adopt) {
                    buffer = adopt(i, b, &held);
                } else {
                    packed_.emplace_back(held.bytes);
                    shard::pack(held, t.data, packed_.back().data());
                    buffer = b.adopt(packed_.back().data(), packed_.back().size());
                }
                const size_t n = (size_t)shard::share(role, width_, member);
                const bool rows = role.shard.axis == Axis::rows;
                weight.data = std::move(buffer);
                weight.nin = rows ? size_t(role.in) : n;
                weight.nout = rows ? n : size_t(role.out);
                return weight;
            }
            weight.data = take(i, device + member);
            return weight;
        };
        pass_.assign(plan_.role_ids, Weight{});
        for (const Role& role : plan_.pass) pass_[role.id] = resolve(role, device_of(role.part, 0));
        if (plan_.drafter) {
            drafter_.assign(plan_.role_ids, Weight{});
            for (const Role& role : plan_.drafter->roles) drafter_[role.id] = resolve(role, device_of(role.part, 0));
        }
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
        // Each other member of a tensor group gets its own rows, in the same order.
        member_pass_.assign(width_ - 1, std::vector<Weight>(plan_.role_ids));
        member_home_.assign(width_ - 1, std::vector<std::vector<Weight>>(n_layer, std::vector<Weight>(plan_.role_ids)));
        member_drafter_.assign(plan_.drafter ? width_ - 1 : 0, std::vector<Weight>(plan_.role_ids));
        for (size_t m = 1; m < width_; ++m) {
            for (const Role& role : plan_.pass) member_pass_[m - 1][role.id] = resolve(role, device_of(role.part, 0), m);
            for (size_t k = 0; plan_.drafter && k < plan_.drafter->roles.size(); ++k)
                member_drafter_[m - 1][plan_.drafter->roles[k].id] = resolve(plan_.drafter->roles[k], device_of(Part::draft, 0), m);
            for (size_t l = 0; l < n_layer; ++l)
                for (const Role& role : plan_.layers[l].roles) member_home_[m - 1][l][role.id] = resolve(role, device_of(role.part, l), m);
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
                Weight& weight = stream_[l][role.id];
                const size_t device = size_t(stream_device_[l]);
                weight.data = windows_[device][k++];
                if (weight.scale_count) weight.scales = take(*weights.tensors[*role.tensor].scale, device);
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

    // Defined in model/history.hpp, the owner of a sequence's history (docs/SPECULATIVE.md, section 1).
    void settle(Sequence& s, const char* what);
    size_t rewind(Sequence& s, size_t length, const Defer* defer = nullptr) noexcept;
    size_t saved_at(const Device& d, const LayerPlan& lp, size_t buffer, size_t layer, size_t item) const;
    void save(ExecContext& ctx, const Pass& p, size_t dev, int l);
    void save_h(ExecContext& ctx, const Pass& p);
    void rerun(Sequence& s, size_t rows, const Defer& defer = {});
    // A rerun's inputs: the rows to run, the mark's position, the slot each stage reads at the mark, the live slot written and the mark's buffer of saved inputs.
    struct Rerun {
        size_t rows, pos;
        std::vector<size_t> from;
        size_t live, buffer;
    };
    bool reruns_on(size_t dev) const;
    size_t stage_of_device(size_t dev) const;
    void rerun_on(size_t dev, const Rerun& a);
    std::function<void(bool)> stage_wait_;
    // Work outside a pass on the stages' threads (a deferred rerun's parts, a drafter's chain) takes the model's context one at a time.
    // It is the last lock taken: no other is taken and no other thread waited for while it is held.
    std::mutex context_m_;
    void restore_mark(Sequence& s) noexcept;
    void drop_mark(Sequence& s) noexcept;

    // Defined in model/passes.hpp, the owner of a pass and its stages.
    void begin(ExecContext& ctx, Pass& p, const BatchEntry* entries, size_t n_entries, size_t logits_base = 0, bool continues = false);
    void at_stage(const Pass& p, size_t s) const;
    void run_stage(ExecContext& ctx, Pass& p, size_t s);
    void prepare_stage(Pass& p, size_t s);
    void record_stage(ExecContext& ctx, Pass& p, size_t s);
    void run_chunks(ExecContext& ctx, size_t chunks, const std::function<BatchEntry(size_t)>& chunk);
    void single_prepare(Pass& p, size_t s);
    void start_lookup(ExecContext& ctx, const Pass& p, size_t group, int l);
    void finish_lookup(ExecContext& ctx, const Pass& p, size_t group, size_t width, int l);
    void single_record(ExecContext& ctx, Pass& p, size_t s);
    void group_prepare(Pass& p, size_t s);
    void group_record(ExecContext& ctx, Pass& p, size_t s);
    void clear_partial(ExecContext& ctx, size_t dev, backend::Slice partial, size_t rows);
    template <class Weights>
    void group_sum(ExecContext& ctx, size_t g, const std::vector<backend::Slice>& x, size_t rows, backend::RowRuns runs, uint8_t kind, const Weights& w);
    void stage_submit(ExecContext& ctx, Pass& p, size_t s, size_t cur);
    void stage_commit(Pass& p, size_t s);
    void prepare_tokens(Pass& p, size_t s) const;
    void advance_tokens(uint32_t* dst, const uint32_t* previous, size_t history, const uint32_t* ids, size_t n) const noexcept;
    void draft_context(ExecContext& ctx, Pass& p, size_t s);
    void finish(ExecContext& ctx, const Pass& p);
    void roll_back(Pass& p) noexcept;
    backend::BufferPtr alloc_arena(backend::Backend& b, const std::vector<size_t>& counts, std::vector<size_t>& offsets) const;
    void ensure(ExecContext& ctx, size_t rows, size_t want, size_t buffers);
    void send(ExecContext& ctx, size_t from, size_t handoff, size_t base, size_t rows);
    void receive(ExecContext& ctx, size_t from, size_t handoff, backend::Ticket sent, size_t to, size_t base, size_t rows);
    void cross(ExecContext& ctx, size_t from, size_t to, size_t base, size_t rows);
    void ffn_split(ExecContext& ctx, const Pass& p, size_t dev, int l);
    Step part(ExecContext& ctx, size_t dev, const Weight* w, uint8_t kind, size_t base, size_t rows, backend::RowRuns runs) const;
    Step mixer_part(ExecContext& ctx, const Pass& p, size_t dev, int l) const;
    void positions(const Sequence& s, size_t first, size_t n, bool streams, uint32_t* out) const;
    std::vector<Sequence::Media> carried(const Sequence& s, size_t length) const;
    void drop_media(Sequence& s, size_t length, const Defer* defer = nullptr) noexcept;
    void release_media(Sequence& s);
    // Images' device rows dropped once read, each with the device and the submission that last read it, freed once that submission has retired.
    struct Retiring {
        size_t device;
        backend::Ticket ticket;
        backend::BufferPtr rows;
    };
    std::vector<Retiring> retiring_;
    void write_media(ExecContext& ctx, const Pass& p, size_t dev);

    // Whether `pos` is whole blocks in every KV storage, as a fork and a checkpoint need.
    bool whole_blocks(size_t pos) const {
        for (const Device* d : storages_)
            if (pos % d->b->kv_layout().block_tokens) return false;
        return true;
    }

    // The pass in a reserved context's slot, which must be in flight.
    static Pass& in_flight(ExecContext& ctx, size_t slot) {
        if (slot >= ctx.slots || !ctx.passes[slot].in_flight) throw std::logic_error("inference: no pass in flight in that slot");
        return ctx.passes[slot];
    }

    // A slot's pass leaves flight, and its sequences with it.
    static void release(Pass& p) noexcept {
        for (const BatchEntry& en : p.entries) --en.seq->flights_;
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

    static backend::Slice slot(const ExecContext& ctx, size_t device, size_t i) {
        const ExecContext::Scratch& sc = ctx.scratch[device];
        return {sc.arena.get(), sc.offset[i] / sizeof(float)};
    }

    static std::vector<float> row(ExecContext& ctx, size_t i) {
        const float* p = ctx.logits(i);
        return std::vector<float>(p, p + ctx.width);
    }

    // Whether rows of this extent take a streamed layer on the device (Placement::stream_from): a prompt long enough, two tokens at the least, never a generated token.
    bool streams(size_t extent) const { return place_.stream_from && extent >= std::max<size_t>(place_.stream_from, 2); }
    bool streams(const Pass& p, size_t e) const { return streams(p.runs[e].extent); }

    // A block returns to the pool only once the backend has retired every submission that touched it (docs/KV-CACHE.md).
    // Construction failures, failed passes and model teardown drain every used device with sync(), including work behind no ticket; reset() waits on tickets instead.
    void retire() noexcept {
        for (auto& d : devices_) if (d->used) d->b->sync();
    }
    void release_holds() noexcept {
        for (auto& d : devices_)
            if (d->holding) {
                d->b->hold_between_submissions(false);
                d->holding = false;
            }
    }
};

} // namespace infer

#include "model/history.hpp"
#include "model/passes.hpp"
