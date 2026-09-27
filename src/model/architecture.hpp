#pragma once
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "backends/backend.hpp"
#include "model/weights.hpp"

// The contract between the runtime (model/runtime.hpp) and an architecture: the plan an architecture declares, which the runtime resolves, fits, places and streams from, and the parts of a pass it runs as backend ops.

namespace infer {

// The part of a pass a role runs with: a layer is a mixer part (attention) then a feed-forward part, which a placement may put on another device, and the embedding and the head are parts of the pass.
enum class Part : uint8_t { embed, mixer, ffn, head };

// How the model checks a role's tensor, and whether the fit counts it as a product.
enum class RoleKind : uint8_t {
    norm,     // F32, [in], trailing axes of one
    matrix,   // [in, out], trailing axes of one, read by a matrix product
    gather,   // checked as a matrix, its rows gathered by the embedding
    experts,  // exactly [in, out, experts], a routed stack
};

// What a routed layer run beside its mixer for a long prompt (Placement::stream_from) does with a feed-forward role: nothing, a copy adopted on the mixer's device at load, or a copy written into that device's window in each pass that needs it.
enum class Stream : uint8_t { none, copy, window };

// A tensor the model reads: the id its resolved weight is indexed by, the part it runs with, how it is checked, its name and the name taken when that is absent (a tied head reads the embedding), its expected shape and what streaming does with it.
// The architecture leaves the last two fields as they are; plan_model sets them.
struct Role {
    uint16_t id;
    Part part;
    RoleKind kind;
    std::string name, alias;
    uint64_t in = 0, out = 1, experts = 0;
    Stream stream = Stream::none;
    std::optional<size_t> tensor = std::nullopt;   // the view it reads, its name's or else its alias's, absent when the file has neither
    bool aliased = false;                          // that view is its alias's
};

// A decoder layer as the model runs it: the architecture's own kind for it, handed back on every call for the layer, whether its feed-forward part holds routed experts, and its roles in the order they are adopted.
struct LayerPlan {
    uint8_t kind = 0;
    bool routed = false;
    std::vector<Role> roles;
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
    size_t kv_heads = 0, head_dim = 0;  // K and V of every layer: heads, and each head's width
    std::vector<size_t> tables;         // floats in each position table
};

// One call of an architecture's part: the backend of the device it runs on and that device's arena, the residual at the call's first row, the rows and their runs, the weights to read by role id, the layer's kind and cache views, the rows' positions, the position tables on that device, and a run list the part may rebuild, which holds a run for every entry of the pass without allocating.
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
    size_t n_views, kv_layer;
    const uint32_t* pos;
    const backend::BufferPtr* tables;
    std::vector<backend::RowRun>* scratch;
    backend::Slice slot(size_t i) const { return {arena, offsets[i] / sizeof(float)}; }
};

// The head's call: the rows that want logits, their runs, and where their logits go.
struct HeadStep : Step {
    const uint32_t* pick;
    size_t want;
    backend::RowRuns head_runs;
    backend::Slice logits;
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
    virtual void ffn(const Step& s) const = 0;
    virtual void head(const HeadStep& s) const = 0;
};

} // namespace infer
