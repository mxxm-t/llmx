#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/backend.hpp"
#include "model/weights.hpp"

// The contract between the runtime (model/runtime.hpp) and an architecture: the plan an architecture declares, which the runtime resolves, fits, places and streams from, and the parts of a pass it runs as backend ops.

namespace infer {

// The part of a pass a role runs with: a layer is a mixer part (attention) then a feed-forward part, which a placement may put on another device, and the embedding and the head are parts of the pass.
// An embedded drafter's roles (ModelPlan::drafter) run with the head, on its device.
enum class Part : uint8_t { embed, mixer, ffn, head, draft };

// How the model checks a role's tensor, and whether the fit counts it as a product.
enum class RoleKind : uint8_t {
    norm,     // F32, [in], trailing axes of one
    matrix,   // [in, out], trailing axes of one, read by a matrix product
    gather,   // checked as a matrix, its rows gathered by the embedding
    experts,  // exactly [in, out, experts], a routed stack
    table,    // F32, [in, out], trailing axes of one, read whole by an op rather than by a matrix product
};

// What a routed layer run beside its mixer for a long prompt (Placement::stream_from) does with a feed-forward role: nothing, a copy adopted on the mixer's device at load, or a copy written into that device's window in each pass that needs it.
enum class Stream : uint8_t { none, copy, window };

// The axis a role splits along over a tensor group's members (docs/TENSOR-SPLIT.md, section 4.2): none, its whole tensor on every member; its output rows, each member's product complete for its rows; or its input columns, each member's product a partial the group sums, a vector's elements being its columns.
enum class Axis : uint8_t { none, rows, columns };

// A run of the axis: `tiles` tiles of `units` units of `unit` rows or columns each, a member taking the same share of the units in every tile, so one tile is a contiguous split.
// `what` names the units in a refusal, and `replicate` lets a width that is a multiple of the units give each unit to width / units members, as KV heads are.
struct ShardSection {
    uint64_t units = 0, unit = 1, tiles = 1;
    const char* what = "rows";
    bool replicate = false;
};

// How a role splits: its axis and the sections that cover it, in order (model/shard.hpp).
struct Shard {
    Axis axis = Axis::none;
    std::vector<ShardSection> sections;
};

// A tensor the model reads: the id its resolved weight is indexed by, the part it runs with, how it is checked, its name and the name taken when that is absent (a tied head reads the embedding), its expected shape and what streaming does with it.
// plan_model sets `tensor` and `aliased`; `shard` is set by the architecture after the role is listed, and a role that keeps the default is whole on every member.
struct Role {
    uint16_t id;
    Part part;
    RoleKind kind;
    std::string name, alias;
    uint64_t in = 0, out = 1, experts = 0;
    Stream stream = Stream::none;
    std::optional<size_t> tensor = std::nullopt;   // the view it reads, its name's or else its alias's, absent when the file has neither
    bool aliased = false;                          // that view is its alias's
    Shard shard{};
};

// What a layer keeps for each sequence from one pass to the next: keys and values for every position it has read, a recurrent state of fixed size, or nothing.
enum class Cache : uint8_t { kv, state, none };

// An op a part issues beyond those every backend has, which the backend of the part's device must implement.
struct OpUse {
    Part part;
    backend::Op op;
};

// Rows of an arena slot that a state layer's mixer leaves for Architecture::recur: in slot `slot`, `width` floats a row, the `plane`-th block of the call's rows, so plane 1 starts Step::rows rows in.
// A slot's planes are consecutive entries of LayerPlan::saved, in plane order.
struct Saved {
    size_t slot, width, plane = 0;
};

// A decoder layer as the model runs it: the architecture's own kind for it, handed back on every call for the layer, whether its feed-forward part holds routed experts, what its mixer keeps between passes, its roles in the order they are adopted, the ops of its parts that some backends lack, and for a layer that keeps a state the inputs its state's update reads (Architecture::recur).
struct LayerPlan {
    uint8_t kind = 0;
    bool routed = false;
    Cache cache = Cache::kv;
    std::vector<Role> roles;
    std::vector<OpUse> ops;
    std::vector<Saved> saved;
    // A rerun (docs/SPECULATIVE.md, section 1) runs every state layer's update with no barrier between layers, which halves the cost of a partial rejection on a 27B, so it needs to know what an update writes and in what order.
    // The arena slots its state's update writes (Architecture::recur), which a rerun gives each layer a room of its own in, so no two layers write the same rows.
    std::vector<size_t> recur_writes;
    // The phases of its state's update, each reading what the ones before it in the same layer wrote, so a rerun runs one phase of every layer, then the next (Step::phase).
    int recur_phases = 1;
};

// What an architecture declares of a model: the size of every resolved row of weights, the vocabulary, the roles of the pass (the embedding's and the head's) and each layer's, the positions a sequence may reach, and what the arena, the caches and the tables take.
// Every role id is below role_ids, and slot 0 is as wide as the residual, which plan_model holds every plan to.
struct ModelPlan {
    size_t role_ids = 0;
    size_t vocab = 0;
    std::vector<Role> pass;
    std::vector<LayerPlan> layers;
    size_t context_length = 0;          // the tables cover these positions, a pass past them is refused, and the cache budget defaults to them
    size_t residual = 0;                // floats in a residual row: slot 0, a handoff row, a crossing
    std::vector<size_t> slots;          // floats one row takes in each arena slot; slot 0 is the residual
    size_t kv_heads = 0, head_dim = 0;  // K and V of every layer whose cache is KV: heads, and each head's width
    backend::StateShape state;          // the recurrent state of every layer whose cache is a state
    std::vector<size_t> tables;         // floats in each position table
    // An embedded drafter, planned only when a caller asks for one (Architecture::plan_drafter): its roles, all Part::draft, its ops and its cache, KV in the head's device's storage beside the layers' (docs/SPECULATIVE.md, section 7).
    // Its context rows leave the target's final-normed rows in arena slot `draft_h`, which the model carries and saves, and a draft step's residual rows are in arena slot `draft_x`, which a tensor group sums its members' partial rows into.
    std::optional<LayerPlan> drafter;
    size_t draft_h = 0, draft_x = 0;
};

// The floats a row of the slots a state layer's update writes take (LayerPlan::recur_writes), a layer's room in a rerun.
inline size_t recur_floats(const ModelPlan& plan, const LayerPlan& layer) {
    size_t n = 0;
    for (size_t slot : layer.recur_writes) n = backend::size_add(n, plan.slots.at(slot));
    return n;
}

// One call of an architecture's part: the backend of the device it runs on and that device's arena, the residual at the call's first row, the rows and their runs, the weights to read by role id, the layer's kind and cache views, the rows' positions, the position tables on that device, a run list the part may rebuild, which holds a run for every entry of the pass without allocating, and a state layer's views.
struct Step {
    backend::Backend& b;
    backend::Buffer* arena;
    const size_t* offsets;              // bytes, per slot
    backend::Slice x;
    size_t rows;
    backend::RowRuns runs;
    const Weight* w;                    // the layer's row, the pass's, or a streamed layer's copies and windows
    uint8_t kind;
    const backend::KVView* views;
    size_t n_views, kv_layer;           // views: one per entry, a KV layer's; n_views counts a state layer's too
    const uint32_t* pos;
    const backend::BufferPtr* tables;
    std::vector<backend::RowRun>* scratch;
    // A state layer's views, one per entry, each reading and writing its sequence's slot, and the layer's index in its device's state storage.
    const backend::StateView* states = nullptr;
    size_t state_layer = 0;
    backend::Dtype dtype = backend::Dtype::f16;
    // Which of the layer's update phases a call of recur runs (LayerPlan::recur_phases), every one when -1, as the mixer runs it.
    int phase = -1;
    // The tensor group the part runs on (docs/TENSOR-SPLIT.md, section 4.3): its width, each member's weights its shard (Role::shard) and its heads that share of the plan's, and where a part's last projection writes its partial rows, which the group then sums into every member's residual, in place of adding them to the residual itself (blocks::join).
    size_t width = 1;
    backend::Slice partial{};
    backend::Slice slot(size_t i) const { return {arena, offsets[i] / sizeof(float)}; }
};

// The KV heads a member of a tensor group of `width` keeps of `kv_heads`: its share, or one where the width is a multiple of them and each is replicated; none of none, which only a plan without KV layers has, a file's count being positive as it is read.
inline size_t kv_share(size_t kv_heads, size_t width) { return kv_heads >= width ? kv_heads / width : (kv_heads ? 1 : 0); }

// The head's call: the rows that want logits, their runs, and where their logits go.
struct HeadStep : Step {
    const uint32_t* pick;
    size_t want;
    backend::RowRuns head_runs;
    backend::Slice logits;
};

// An embedded drafter's call over every row of a pass, after its last stage (Architecture::draft_rows): the rows' token ids, and per entry the row its first row reads as the target's row before it, the sequence's carried row or a zero row; the cache views are the drafter's KV layer's.
struct DraftRowsStep : Step {
    const uint32_t* ids;
    const backend::CSlice* carry;
};

// One step of an embedded drafter's chains (Architecture::draft), a draft row for each of `rows` sequences, one cache view each: each token's row read from `id` on the device, the target's row before it at `prev`, row i of it or, with `prev_rows`, the row prev_rows[i] names, the drafted ids written to `next`, the drafter's output rows after its final norm left at `out`, the next step's `prev`, and the head's logits at `logits`.
// On a tensor group a step is three calls (Step::phase): 0 through the block's mixer, 1 its feed-forward block, 2 the final norm, the head and the argmax, the group summing the members' partial rows into the residual after each of the first two; -1 runs all of it, as one device does.
// Each member reads the head whole there, so every member finds the same id from the same rows with no exchange.
struct DraftStep : Step {
    backend::CSlice id, prev;
    const uint32_t* prev_rows = nullptr;
    backend::Slice next, out, logits;
};

// An architecture: its configuration, read from a file, the plan of that file's tensors, the values of its position tables, and the math of each part as backend ops.
// Immutable once read, so models built from one set of weights share it; the runtime calls each part once per layer per pass, or once per group of entries where a routed layer streams.
class Architecture {
public:
    virtual ~Architecture() = default;
    virtual ModelPlan plan(const TensorIndex& tensors) const = 0;
    // The tables come sized as the plan says.
    virtual void fill_tables(std::vector<std::vector<float>>& tables) const = 0;
    virtual void embed(const Step& s, const uint32_t* ids) const = 0;
    virtual void mixer(const Step& s) const = 0;
    // The ops of a state layer's mixer that update its state, from the rows LayerPlan::saved names in their slots: the mixer runs them, and a retract inside a mark runs them again over the rows it keeps (docs/SPECULATIVE.md, section 1).
    // A layer that keeps no state has none.
    // A rerun reads the saved rows in place, where a plane holds the rows the mark has room for, which it gives as Step::rows, the views giving the rows it runs.
    virtual void recur(const Step&) const {}
    virtual void ffn(const Step& s) const = 0;
    virtual void head(const HeadStep& s) const = 0;
    // An embedded drafter (docs/SPECULATIVE.md, section 7), which a file may carry: its plan, added to `plan` when a caller asks for one, refused where the architecture or the file has none.
    virtual void plan_drafter(const TensorIndex&, ModelPlan&) const { throw std::runtime_error("inference: this architecture has no embedded drafter"); }
    // The drafter's context rows of a pass: its cache's row for every row the pass feeds, from the token and the target's row before it.
    virtual void draft_rows(const DraftRowsStep&) const {}
    // One draft row, through the whole drafter and the head.
    virtual void draft(const DraftStep&) const {}
};

} // namespace infer
