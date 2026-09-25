#pragma once
#include <algorithm>
#include <array>
#include <memory>
#include <functional>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <random>

#include "format/gguf.hpp"
#include "quant/quant.hpp"
#include "backends/backend.hpp"
#include "model/kv_cache.hpp"
#include "model/layer_split.hpp"
#include "backends/cpu/cpu_backend.hpp"
#include "core/host_memory.hpp"

// Qwen3-style transformer forward pass, from scratch: dense Qwen3 and its mixture-of-experts form, qwen3moe.
// The compute primitives (matmul, attention, RMSNorm, RoPE, expert routing) are delegated to a backend::Backend, so the same model code runs on every backend.
// Dense matrices use supported block quants or F32; normalization weights are F32.
// Tensor ne[0] is the input dimension, with each output row contiguous.
// Attention projection width is n_head*head_dim and need not equal n_embd.
// An absent output.weight ties the output projection to token_embd.weight.

namespace infer {

struct QwenConfig {
    int n_layer = 0;
    int n_embd = 0;
    int n_ff = 0;
    int n_head = 0;
    int n_head_kv = 0;
    int head_dim = 0;
    int context_length = 4096;
    float rope_theta = 10000.0f;
    float rms_eps = 1e-6f;
    // qwen3moe: experts per layer, experts each token takes, an expert's hidden width, and whether the chosen probabilities are renormalized to sum to one.
    // A layer is a mixture of experts when its router tensor is present, so dense and routed layers can mix.
    std::string arch = "qwen3";
    int n_expert = 0;
    int n_expert_used = 0;
    int n_ff_exp = 0;
    bool expert_norm = true;
};

inline QwenConfig load_config(const gguf::GGUFModel& m) {
    QwenConfig c;
    auto integer = [&](const std::string& k, int fallback = 0) -> int {
        const auto* v = m.find(k);
        if (!v) {
            if (fallback) return fallback;
            throw std::runtime_error("inference: missing metadata " + k);
        }
        uint64_t n;
        if (v->vtype == gguf::V_UINT32 || v->vtype == gguf::V_UINT64) {
            n = v->u;
        } else if (v->vtype == gguf::V_INT32 || v->vtype == gguf::V_INT64) {
            if (v->i <= 0) throw std::runtime_error("inference: invalid positive integer " + k);
            n = uint64_t(v->i);
        } else {
            throw std::runtime_error("inference: invalid integer type " + k);
        }
        if (!n || n > uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("inference: integer outside supported range " + k);
        return int(n);
    };
    auto real = [&](const std::string& k, float fallback) -> double {
        const auto* v = m.find(k);
        if (!v) return fallback;
        double n;
        if (v->vtype == gguf::V_FLOAT32) {
            float value;
            std::memcpy(&value, &v->fb, sizeof(value));
            n = value;
        } else if (v->vtype == gguf::V_FLOAT64) {
            n = v->f64;
        } else {
            throw std::runtime_error("inference: invalid floating-point type " + k);
        }
        if (!std::isfinite(n) || n <= 0 || n > std::numeric_limits<float>::max())
            throw std::runtime_error("inference: invalid positive float " + k);
        const float value = float(n);
        if (value == 0) throw std::runtime_error("inference: float underflow " + k);
        return n;
    };
    auto option = [&](const std::string& k, const std::string& supported) {
        const auto* v = m.find(k);
        if (v && (v->vtype != gguf::V_STRING || v->s != supported))
            throw std::runtime_error("inference: unsupported metadata " + k);
    };
    if (const auto* a = m.find("general.architecture")) {
        if (a->vtype != gguf::V_STRING || (a->s != "qwen3" && a->s != "qwen3moe"))
            throw std::runtime_error("inference: unsupported metadata general.architecture");
        c.arch = a->s;
    }
    const std::string p = c.arch + ".";
    const bool moe = c.arch == "qwen3moe";
    option(p + "tensor_data_layout", "reference");
    option(p + "rope.scaling.type", "none");
    if (real(p + "rope.scaling.factor", 1) != 1 ||
        real(p + "rope.scale_linear", 1) != 1)
        throw std::runtime_error("inference: scaled RoPE is unsupported");

    c.n_layer = integer(p + "block_count");
    c.n_embd = integer(p + "embedding_length");
    // A mixture-of-experts file needs the dense width only for its dense layers, if it has any.
    c.n_ff = moe && !m.find(p + "feed_forward_length") ? 0 : integer(p + "feed_forward_length");
    c.n_head = integer(p + "attention.head_count");
    c.n_head_kv = integer(p + "attention.head_count_kv", c.n_head);
    if (c.n_head % c.n_head_kv)
        throw std::runtime_error("inference: head count must be divisible by KV head count");
    if (m.find(p + "attention.key_length")) {
        c.head_dim = integer(p + "attention.key_length");
    } else {
        if (c.n_embd % c.n_head)
            throw std::runtime_error("inference: embedding width does not determine an integral head width");
        c.head_dim = c.n_embd / c.n_head;
    }
    if (c.head_dim <= 0 || c.head_dim % 2 ||
        c.n_head > std::numeric_limits<int>::max() / c.head_dim)
        throw std::runtime_error("inference: invalid attention projection dimensions");
    if (integer(p + "attention.value_length", c.head_dim) != c.head_dim ||
        integer(p + "rope.dimension_count", c.head_dim) != c.head_dim)
        throw std::runtime_error("inference: value and rotary widths must equal key width");
    c.context_length = integer(p + "context_length", c.context_length);
    c.rope_theta = float(real(p + "rope.freq_base", c.rope_theta));
    c.rms_eps = float(real(p + "attention.layer_norm_rms_epsilon", c.rms_eps));
    if (moe) {
        c.n_expert = integer(p + "expert_count");
        c.n_expert_used = integer(p + "expert_used_count");
        c.n_ff_exp = integer(p + "expert_feed_forward_length");
        if (c.n_expert_used > c.n_expert || c.n_expert_used > 256)
            throw std::runtime_error("inference: more experts per token than the layer has, or above 256");
        if (const auto* v = m.find(p + "expert_weights_norm")) {
            if (v->vtype != gguf::V_BOOL) throw std::runtime_error("inference: invalid boolean type " + p + "expert_weights_norm");
            c.expert_norm = v->b;
        }
        // Routing is a softmax over the scores; a sigmoid gate, a shared expert or a scaled mixture is another architecture's.
        if (const auto* g = m.find(p + "expert_gating_func"))
            if (g->vtype != gguf::V_UINT32 || g->u != 1) throw std::runtime_error("inference: unsupported expert gating function");
        if (m.find(p + "expert_shared_count") || m.find(p + "expert_shared_feed_forward_length"))
            throw std::runtime_error("inference: shared experts are unsupported");
        if (real(p + "expert_weights_scale", 1) != 1)
            throw std::runtime_error("inference: scaled expert weights are unsupported");
    }
    const uint64_t kv_width = uint64_t(c.n_head_kv) * c.head_dim;
    const auto max_floats = std::vector<float>().max_size();
    if (uint64_t(c.context_length) > max_floats / kv_width ||
        uint64_t(c.context_length) > max_floats / uint64_t(c.n_head))
        throw std::runtime_error("inference: context storage exceeds allocation limit");
    return c;
}

// A weight resolved once at load: type, storage and dimensions.
// Resolving per call meant rebuilding "blk.N." and hashing a tensor name for every projection of every layer of every token; the forward pass indexes layers_ instead.
// It is also what lets a device backend recognize a weight across calls, which is the prerequisite for residency (docs/DEVICE-EXECUTION.md).
struct Weight {
    uint32_t type = 0;
    // A handle, not a pointer: the backend decides where the bytes live.
    // The model never dereferences it: slice() below names a location, and only a backend turns that into an address.
    backend::BufferPtr data;
    size_t nin = 0, nout = 0;
    // Normalization weights are F32 by validation, so this is the whole row.
    backend::CSlice slice() const { return {data.get(), 0}; }
};

class Model;

struct LayerWeights {
    Weight attn_norm, attn_q_norm, attn_k_norm;
    Weight attn_q, attn_k, attn_v, attn_output;
    Weight ffn_norm, ffn_gate, ffn_up, ffn_down;
    // A routed layer's router and its stacked experts, whose nin and nout are one expert's.
    bool moe = false;
    Weight ffn_gate_inp, ffn_gate_exps, ffn_up_exps, ffn_down_exps;
    // A host-placed routed layer whose long prompt runs its attention device takes over (Placement::stream_from): the norm and router copied there, the experts copied into that device's window in each pass that needs them.
    int stream_device = -1;
    Weight stream_norm, stream_router;
};

// Where each tensor role runs, as an index into the model's backends.
// Per role rather than per layer, so a layer's attention and its feed-forward block can sit on different devices; that is what expert offload needs later (docs/EXECUTION.md).
// Empty means everything on device 0.
struct Placement {
    std::vector<int> attn_device, ffn_device;
    int embed_device = 0, output_device = 0;
    // A routed layer with its feed-forward block on a host and its attention on a device runs a prompt of at least this many new tokens on the device, its experts copied there for each pass: past some length a prompt's expert products on the host cost more than moving the experts.
    // By the tokens the request prefills (BatchEntry::fresh), so a short reply in a long conversation stays on the host, and every slice of one prompt takes the same path however it is batched. Zero keeps every run on the host, and a generated token never streams: one row cannot pay for moving a layer's experts.
    size_t stream_from = 0;
};

// Choices made once at construction, before the caches are allocated: how each cache side is stored (backend.hpp KVType, the CLI's --cache-type-k and --cache-type-v), the same on every backend or refused.
// The runtime's default cache type is set here and nowhere else: the CLI changes a side only when its flag is given.
struct ModelOptions {
    backend::KVType kv_k = backend::KVType::f16;
    backend::KVType kv_v = backend::KVType::f16;
    // Tokens the KV pool holds in total, shared by every sequence; zero means one model context, which is what one conversation needs and what a server divides among its requests unless told otherwise.
    size_t kv_tokens = 0;
};

// One request's history in a model's cache: a block table per storage and the committed length, and per device the ticket of the last pass that touched it, which is what a release waits on rather than draining the device (docs/EXECUTION.md).
// Made by Model::make_sequence so it is bound to that model's pools and block sizes.
// Movable, not copyable; the server keeps one per request.
class Sequence {
public:
    Sequence() = default;
    size_t length() const { return kv_.empty() ? 0 : kv_[0].length(); }
private:
    friend class Model;
    std::vector<KVSequence> kv_;
    std::vector<backend::Ticket> last_;
    const Model* owner_ = nullptr;
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
    // What a device chooses this entry's kernels by (backend::RowRun): for a prompt's rows the position one past the prompt's last token, for a generated token 1.
    // Zero takes the entry's own row count.
    // A prompt given its extent computes the same whether it arrives in one pass or in slices, alone or beside other sequences, with or without a reused prefix.
    size_t extent = 0;
    // The tokens the request prefills, its reused prefix excluded, the same for every slice of it: what a streamed layer follows (Placement::stream_from).
    // Zero takes the entry's own row count. A prompt computes the same in one pass or in slices, alone or beside other sequences; with a reused prefix its new tokens may take the host where one pass over the whole would take the device.
    size_t fresh = 0;
};

// What a pass's stages read as they are recorded: its entries, its rows and their positions, the rows the head reads, and each storage's cache views once its stage has reserved them.
// A context keeps one per pass it has in flight: one for a pass run whole, one per stage while a prompt's chunks flow through the stages together.
struct Pass {
    std::vector<BatchEntry> entries;
    std::vector<size_t> start;                         // per entry, the history the pass found, which a failed pass returns to
    size_t rows = 0, want = 0;
    bool long_runs = false;                            // some entry takes its streamed layers on the device
    std::vector<uint32_t> ids, pos, pick;
    std::vector<std::vector<backend::KVView>> views;   // per storage, per entry
    std::vector<backend::RowRun> runs, head_runs;      // the pass's rows and the head's, by entry
    std::vector<size_t> fresh;                         // per entry, the tokens its request prefills
    size_t parity = 0;                                 // which of each device's two handoff buffers its crossings use
    size_t at = 0;                                     // the device its residual left the last stage from
    backend::Ticket sent = 0;                          // the submission that copied it out
};

// Where a context's passes run: an activation arena per device, which each device's passes use in turn, a host-visible handoff buffer per device a crossing goes through (two on a pipelined split), the host-visible logits rows on the output device, and the tickets of the submissions.
// Storage is allocated by the first forward that needs it and grows to the largest pass seen.
// Two contexts are what let a scheduler keep one pass on the device while it reads another's logits; the CLI has one.
// Plain data that Model fills.
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

    static constexpr size_t kSlots = 12;
    struct Scratch {
        backend::BufferPtr arena;
        size_t rows = 0;
        size_t offset[kSlots] = {0};
    };
    std::vector<Scratch> scratch;              // per device
    backend::BufferPtr logits_buf;
    size_t logit_rows = 0;
    std::vector<std::array<backend::BufferPtr, 2>> handoff;   // per device
    size_t handoff_rows = 0;
    std::vector<Pass> passes;
    std::vector<backend::RowRun> part_runs;    // a streamed layer's group of entries, rebased
    std::vector<backend::Ticket> tickets;      // per device
};

// Prompt tokens a pass takes by default (Model::set_ubatch), and so the prompt rows a placement is fitted for.
inline constexpr int kDefaultUbatch = 512;

// The positions a model's caches are budgeted for: the options' tokens, else the whole context.
inline size_t kv_tokens(const QwenConfig& cfg, const ModelOptions& options) {
    return options.kv_tokens ? options.kv_tokens : (size_t)cfg.context_length;
}

// The bytes one position of one layer's cache takes, key and value, at the options' cache types.
inline size_t kv_bytes_per_position(const QwenConfig& cfg, const ModelOptions& options) {
    return (size_t)cfg.n_head_kv * (size_t)cfg.head_dim * (backend::kv_elem_bytes(options.kv_k) + backend::kv_elem_bytes(options.kv_v));
}

// Which layers route their feed-forward block through experts, found by their router tensor.
inline std::vector<bool> routed_layers(const gguf::GGUFModel& m, int n_layer) {
    std::vector<bool> routed((size_t)n_layer, false);
    const std::string suffix = ".ffn_gate_inp.weight";
    for (const auto& t : m.tensors) {
        if (t.name.compare(0, 4, "blk.") != 0 || t.name.size() <= suffix.size() ||
            t.name.compare(t.name.size() - suffix.size(), suffix.size(), suffix) != 0)
            continue;
        const size_t l = (size_t)std::strtoull(t.name.c_str() + 4, nullptr, 10);
        if (l < routed.size()) routed[l] = true;
    }
    return routed;
}

// The floats one row of a pass takes in each slot of an activation arena (ExecContext::Scratch), for the arena itself and for a split's fit.
// Slots: 0 x, 1 h, 2 q, 3 k, 4 v, 5 attn, 6 gate, 7 up, 8 ffn, 9 router scores, 10 expert ids, 11 expert weights.
// The feed-forward slots hold a dense layer's hidden rows or a routed layer's k expert rows per token, whichever is wider.
inline std::array<size_t, ExecContext::kSlots> slot_widths(const QwenConfig& cfg, bool dense) {
    const size_t q = (size_t)cfg.n_head * cfg.head_dim, kv = (size_t)cfg.n_head_kv * cfg.head_dim;
    const size_t ff = std::max(dense ? (size_t)cfg.n_ff : 0, (size_t)cfg.n_expert_used * (size_t)cfg.n_ff_exp);
    const size_t e = (size_t)cfg.n_embd, k = (size_t)cfg.n_expert_used;
    return {e, e, q, kv, kv, q, ff, ff, ff, (size_t)cfg.n_expert, k, k};
}

// A model of this architecture with the given shape and random weights, Q8_0 matrices and F32 norms, for timing the backend without a file (bench without --model).
inline gguf::GGUFModel synthetic_model(int n_layer, int n_embd, int n_ff, int n_head, int n_head_kv, int head_dim, int n_vocab, uint32_t seed) {
    gguf::GGUFModel m;
    auto u32 = [&](const std::string& k, uint64_t v) {
        gguf::MetaValue mv; mv.vtype = gguf::V_UINT32; mv.u = v;
        m.kv.emplace_back(k, mv);
    };
    u32("qwen3.block_count", (uint64_t)n_layer);
    u32("qwen3.embedding_length", (uint64_t)n_embd);
    u32("qwen3.feed_forward_length", (uint64_t)n_ff);
    u32("qwen3.attention.head_count", (uint64_t)n_head);
    u32("qwen3.attention.head_count_kv", (uint64_t)n_head_kv);
    u32("qwen3.attention.key_length", (uint64_t)head_dim);
    u32("qwen3.context_length", 2048);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // ne = [nin, nout]; f32 tensors are stored raw, others quantized to Q8_0.
    auto add_tensor = [&](const std::string& name, size_t nin, size_t nout, bool f32) {
        gguf::TensorInfo t;
        t.name = name;
        t.ne = { (uint64_t)nin, (uint64_t)nout };
        t.type = f32 ? gguf::GGML_TYPE_F32 : gguf::GGML_TYPE_Q8_0;
        t.offset = 0;
        if (f32) {
            std::vector<uint8_t> buf(nin * nout * 4);
            float* p = (float*)buf.data();
            for (size_t o = 0; o < nout; o++)
                for (size_t i = 0; i < nin; i++) *p++ = dist(rng);
            m.tensors.push_back(std::move(t));
            m.add_tensor_data(buf);
        } else {
            size_t nblocks = nin / gguf::Q8_0_BLOCK;
            std::vector<uint8_t> buf(nout * nblocks * gguf::Q8_0_TYPESIZE);
            std::vector<float> row(nin);
            for (size_t o = 0; o < nout; o++) {
                for (size_t i = 0; i < nin; i++) row[i] = dist(rng);
                quant::quantize_row_q8_0(row.data(), buf.data() + o * nblocks * gguf::Q8_0_TYPESIZE, nblocks);
            }
            m.tensors.push_back(std::move(t));
            m.add_tensor_data(buf);
        }
    };

    size_t kv_dim = (size_t)n_head_kv * head_dim;
    add_tensor("token_embd.weight", n_embd, n_vocab, false);
    add_tensor("output.weight", n_embd, n_vocab, false);
    add_tensor("output_norm.weight", n_embd, 1, true);
    for (int l = 0; l < n_layer; l++) {
        std::string pre = "blk." + std::to_string(l) + ".";
        add_tensor(pre + "attn_norm.weight", n_embd, 1, true);
        add_tensor(pre + "attn_q.weight", n_embd, n_embd, false);
        add_tensor(pre + "attn_k.weight", n_embd, kv_dim, false);
        add_tensor(pre + "attn_v.weight", n_embd, kv_dim, false);
        add_tensor(pre + "attn_output.weight", n_embd, n_embd, false);
        add_tensor(pre + "attn_q_norm.weight", head_dim, 1, true);
        add_tensor(pre + "attn_k_norm.weight", head_dim, 1, true);
        add_tensor(pre + "ffn_norm.weight", n_embd, 1, true);
        add_tensor(pre + "ffn_gate.weight", n_embd, n_ff, false);
        add_tensor(pre + "ffn_up.weight", n_embd, n_ff, false);
        add_tensor(pre + "ffn_down.weight", n_ff, n_embd, false);
    }
    return m;
}

// What this architecture asks of the memory of the devices it runs on, for a split fitted to them (model/layer_split.hpp).
// The cache is counted for every position the options budget; activations are the slots of ExecContext's arena.
inline Footprint footprint(const gguf::GGUFModel& m, const ModelOptions& options) {
    const QwenConfig cfg = load_config(m);
    Footprint fp;
    fp.layers.resize((size_t)cfg.n_layer);
    bool dense = false, output = false;
    // The roles a matrix product reads as its weights: a layer's projections and router, and the head; the embedding is gathered, norms are vectors and stacked experts are routed.
    // By role, not by rank, since a projection may carry trailing singleton axes.
    auto product = [](const std::string& name) {
        if (name == "output.weight") return true;
        const size_t dot = name.find('.', 4);
        if (name.compare(0, 4, "blk.") != 0 || dot == std::string::npos) return false;
        const std::string role = name.substr(dot + 1);
        for (const char* r : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "ffn_gate.weight",
                              "ffn_up.weight", "ffn_down.weight", "ffn_gate_inp.weight"})
            if (role == r) return true;
        return false;
    };
    for (const auto& t : m.tensors) {
        Matrix w{t.type, t.ne.empty() ? 0 : (size_t)t.ne[0], 1, (size_t)t.data_size(), product(t.name)};
        for (size_t d = 1; d < t.ne.size(); ++d) w.rows *= (size_t)t.ne[d];
        if (t.name == "token_embd.weight") fp.embedding = w;
        else if (t.name == "output.weight") { fp.output = w; output = true; }
        else if (t.name == "output_norm.weight") fp.output_norm = w;
        if (t.name.compare(0, 4, "blk.") != 0) continue;
        const size_t l = (size_t)std::strtoull(t.name.c_str() + 4, nullptr, 10);
        if (l < fp.layers.size()) fp.layers[l].push_back(w);
        dense = dense || t.name.find(".ffn_gate.weight") != std::string::npos;
    }
    fp.tied = !output;
    if (fp.tied) {
        fp.output = fp.embedding;
        fp.output.product = true;
    }
    fp.logits_per_row = fp.output.rows * sizeof(float);
    fp.cache_per_layer = kv_tokens(cfg, options) * kv_bytes_per_position(cfg, options);
    fp.tables = (size_t)cfg.context_length * (size_t)cfg.head_dim * sizeof(float);
    fp.handoff_per_row = (size_t)cfg.n_embd * sizeof(float);
    for (size_t w : slot_widths(cfg, dense)) fp.activations_per_row += w * sizeof(float);
    return fp;
}

// The placement a layer split describes: each layer's attention and feed-forward block on the device that runs it, the embedding and the head where the split put them.
inline Placement placement_for(const LayerSplit& split) {
    Placement p;
    for (size_t d = 0; d < split.stages.size(); ++d)
        for (int i = 0; i < split.stages[d].count; ++i) {
            p.attn_device.push_back((int)d);
            p.ffn_device.push_back((int)d);
        }
    p.embed_device = split.embed_device;
    p.output_device = split.output_device;
    return p;
}

class Model {
public:
    // Construct the model over a GGUF model on one backend (defaults to the CPU backend).
    // The model owns a reference to the model data, which must outlive the Model.
    explicit Model(const gguf::GGUFModel& m,
                   backend::BackendPtr backend = backend::make_cpu_backend(),
                   ModelOptions options = ModelOptions{})
        : Model(m, std::vector<backend::BackendPtr>{std::move(backend)}, Placement{}, options) {}

    // Construct over several backends with a placement of every role.
    Model(const gguf::GGUFModel& m, std::vector<backend::BackendPtr> backends,
          Placement placement, ModelOptions options = ModelOptions{})
        : m_(&m), place_(std::move(placement)), options_(options) {
        if (backends.empty()) throw std::runtime_error("inference: missing backend");
        for (const auto& b : backends)
            if (!b) throw std::runtime_error("inference: missing backend");
        quant::register_builtins(); // populate the quant registry (idempotent)
        cfg = load_config(m);
        // The attention projection width is n_head*head_dim, which only equals n_embd by coincidence on some models (Qwen3-8B: 32*128 == 4096).
        // Qwen3-0.6B/1.7B/4B have head_dim 128 with a smaller n_embd.
        q_dim_ = cfg.n_head * cfg.head_dim;

        if (place_.attn_device.empty() && place_.ffn_device.empty()) {
            place_.attn_device.assign((size_t)cfg.n_layer, 0);
            place_.ffn_device.assign((size_t)cfg.n_layer, 0);
        }
        if (place_.attn_device.size() != (size_t)cfg.n_layer ||
            place_.ffn_device.size() != (size_t)cfg.n_layer)
            throw std::runtime_error("inference: placement does not cover every layer");
        auto device_index = [&](int d) {
            if (d < 0 || (size_t)d >= backends.size())
                throw std::runtime_error("inference: placement names a device the model does not have");
            return (size_t)d;
        };
        devices_.reserve(backends.size());
        for (auto& b : backends) {
            devices_.push_back(std::make_unique<Device>());
            devices_.back()->b = std::move(b);
            devices_.back()->local_layer.assign((size_t)cfg.n_layer, -1);
        }
        device_index(place_.embed_device);
        device_index(place_.output_device);
        devices_[(size_t)place_.embed_device]->used = true;
        devices_[(size_t)place_.output_device]->used = true;
        for (int l = 0; l < cfg.n_layer; ++l) {
            Device& a = *devices_[device_index(place_.attn_device[(size_t)l])];
            a.local_layer[(size_t)l] = a.attn_layers++;
            a.used = true;
            devices_[device_index(place_.ffn_device[(size_t)l])]->used = true;
        }
        // A device's attention layers are one run, so each storage is written by one stage, which reserves and commits it once a pass.
        for (int l = 0; l < cfg.n_layer; ++l) {
            const size_t a = (size_t)place_.attn_device[(size_t)l];
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
        // A prompt's chunks flow through the stages together when nothing crosses inside a stage: the embedding on the first stage's device, the head on the last's, every feed-forward block beside its attention.
        pipelined_ = stages_.size() > 1 && place_.embed_device == (int)stages_.front().device &&
                     place_.output_device == (int)stages_.back().device;
        for (size_t s = 0; pipelined_ && s < stages_.size(); ++s) {
            for (int l = stages_[s].first; l < stages_[s].end; ++l)
                pipelined_ = pipelined_ && place_.ffn_device[(size_t)l] == (int)stages_[s].device;
        }

        if (m.offsets.size() != m.tensors.size())
            throw std::runtime_error("inference: tensor storage count mismatch");
        for (size_t i = 0; i < m.tensors.size(); i++) {
            const auto& t = m.tensors[i];
            if (!tindex_.emplace(t.name, i).second)
                throw std::runtime_error("inference: duplicate tensor " + t.name);
            if (t.ne.size() > 4)
                throw std::runtime_error("inference: invalid tensor rank " + t.name);
            const uint64_t bytes = t.data_size();
            if (m.offsets[i] % alignof(float) || m.offsets[i] > m.payload_size() ||
                bytes > m.payload_size() - m.offsets[i])
                throw std::runtime_error("inference: invalid tensor storage " + t.name);
        }

        // Tied embeddings: models without a separate output.weight reuse token_embd.weight as the output projection (same [n_embd, n_vocab] layout), so the head is just a matvec against the embedding matrix.
        out_name_ = tindex_.count("output.weight") ? "output.weight" : "token_embd.weight";
        host_reads_.assign(m_->tensors.size(), 0);
        copied_.assign(m_->tensors.size(), 0);
        try {
            resolve_tensors();
            // With experts on the host, the file stays mapped for them; the tensors the devices copied need not stay resident beside them.
            if (holds_payload_)
                for (size_t i = 0; i < copied_.size(); ++i)
                    if (copied_[i] && !host_reads_[i]) m_->drop_pages(i);

            // Each device that runs attention gets a storage for exactly its layers, with its own block size and pool.
            // Budget: the option's tokens, else the whole context; storage is backed on demand, so a short chat does not allocate it.
            const size_t budget = kv_tokens(cfg, options_);
            for (auto& dp : devices_) {
                Device& d = *dp;
                if (!d.attn_layers) continue;
                // A shared prefix ends on a whole block of the largest size (kv_block_tokens), which is whole in every storage only when the sizes nest.
                for (const Device* other : storages_) {
                    const size_t a = d.b->kv_layout().block_tokens, b = other->b->kv_layout().block_tokens;
                    if (std::max(a, b) % std::min(a, b))
                        throw std::runtime_error("inference: cache blocks of " + std::to_string(a) + " and " + std::to_string(b) +
                                                 " tokens in one model; a split needs one size to divide the other");
                }
                d.storage = d.b->kv_alloc((size_t)d.attn_layers, cfg.n_head_kv, cfg.head_dim,
                                          budget, options_.kv_k, options_.kv_v);
                d.pool.configure(d.storage->max_blocks());
                d.storage_index = (int)storages_.size();
                storages_.push_back(&d);
            }
            seq_ = make_sequence();

            // Precompute the RoPE cos/sin table for every position up to the context length.
            // Indexed as [pos*(head_dim/2) + i].
            // Every device that runs attention reads it through an adopted buffer, so the host vectors stay alive for the model's lifetime; on CPU that is the same memory.
            int half = cfg.head_dim / 2;
            rope_cos_.assign((size_t)cfg.context_length * half, 0.0f);
            rope_sin_.assign((size_t)cfg.context_length * half, 0.0f);
            for (int pos = 0; pos < cfg.context_length; pos++) {
                for (int i = 0; i < half; i++) {
                    float fre = std::pow(cfg.rope_theta, -2.0f * (float)i / (float)cfg.head_dim);
                    rope_cos_[(size_t)pos * half + i] = std::cos((float)pos * fre);
                    rope_sin_[(size_t)pos * half + i] = std::sin((float)pos * fre);
                }
            }
            for (Device* d : storages_) {
                d->rope_cos = d->b->adopt(rope_cos_.data(), rope_cos_.size() * sizeof(float));
                d->rope_sin = d->b->adopt(rope_sin_.data(), rope_sin_.size() * sizeof(float));
            }
        } catch (...) {
            // Constructor members still exist here, so pending uploads retire before unwinding releases them.
            retire();
            throw;
        }
    }

    ~Model() { retire(); }

    // Sequences hold the pools' addresses; moving the model would leave them pointing at the old ones.
    // Nothing moves a Model today.
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // CPU worker counts, applied to every backend; a device backend ignores them.
    void set_threads(int n) { for (auto& d : devices_) d->b->set_threads(n); }

    // The cache pools a scheduler admits against, one per device that runs attention, each counted in its own blocks (docs/SERVER.md, docs/MULTI-DEVICE.md).
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
    size_t n_vocab() const { return output_.nout; }
    const QwenConfig& config() const { return cfg; }
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
    int context_length() const { return cfg.context_length; }

    // A second history holding the first `length` tokens of `src`, which must be whole blocks in every storage: every block below `length` is shared, read-only from now on, and the fork appends into fresh ones, so nothing is allocated or copied here.
    // The fork inherits the tickets of the passes that wrote what it shares.
    Sequence fork(const Sequence& src, size_t length) {
        if (src.owner_ != this) throw std::runtime_error("inference: sequence of another model");
        Sequence f;
        f.kv_.reserve(storages_.size());
        for (const KVSequence& kv : src.kv_) f.kv_.push_back(kv.fork(length));
        f.last_ = src.last_;
        f.owner_ = this;
        return f;
    }

    // A fresh history over this model's cache: one table per storage.
    Sequence make_sequence() {
        Sequence s;
        s.kv_.reserve(storages_.size());
        for (Device* d : storages_)
            s.kv_.emplace_back(&d->pool, d->b->kv_layout().block_tokens);
        s.last_.assign(devices_.size(), 0);
        s.owner_ = this;
        return s;
    }

    // One pass over every entry: each sequence's tokens at their own positions through their own history, the logits after each wanting entry's last token landing in the context in entry order.
    // Its stages run in a row, each submitting its own work and committing the blocks of the storage it writes; a failure anywhere returns every history to where the pass found it.
    void forward(ExecContext& ctx, const BatchEntry* entries, size_t n_entries) {
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

    // Start a new history.
    // Blocks return to every pool; their storage is retained.
    // Every pass ends in a submit or, on failure, a sync, so the sequence's last tickets cover everything that could still be touching a block: this waits for those and no more.
    void reset(Sequence& s) {
        if (s.owner_ != this)
            throw std::runtime_error("inference: sequence of another model");
        for (size_t d = 0; d < devices_.size(); ++d)
            if (devices_[d]->used) devices_[d]->b->wait(s.last_[d]);
        for (auto& kv : s.kv_) kv.reset();
    }

    // The single-sequence entry points the CLI uses: one sequence and one context owned here, and one entry per pass.

    // Run one token through the model (prefill or continue).
    // Returns logits over the full vocabulary.
    std::vector<float> step(int token_id) {
        const uint32_t id = (uint32_t)token_id;
        const BatchEntry entry{&seq_, &id, 1, true};
        forward(ctx_, &entry, 1);
        return row(ctx_, 0);
    }

    // Process a whole prompt with matrix-matrix matmuls instead of one token at a time.
    // Each weight row is then reused across the batch, which is the difference between prefill being compute bound and paying the entire weight stream once per token.
    // Only the final token's logits are needed, so only the last pass asks for them.
    std::vector<float> prefill(const std::vector<uint32_t>& ids) {
        if (ids.empty()) throw std::runtime_error("inference: empty prompt");
        // The prompt is one transaction across its microbatches: a failure in any of them restores the history from before the call.
        const size_t start = seq_.length();
        auto work = [&] {
            // Sized to the largest chunk this prompt will use, inside the scope, so a short prompt does not allocate scratch for a full ubatch (at n_ff 12288 a 512-wide gate/up/ffn is about 25 MB each).
            // Sized before any chunk runs, so nothing in flight loses its storage.
            ensure(ctx_, std::min((size_t)ubatch(), ids.size()), 1);
            const size_t B = (size_t)ubatch(), chunks = (ids.size() + B - 1) / B;
            auto chunk = [&](size_t c) {
                const size_t i = c * B, n = std::min(B, ids.size() - i);
                BatchEntry entry{&seq_, ids.data() + i, n, i + n == ids.size()};
                entry.extent = start + ids.size();
                entry.fresh = ids.size();
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
                        p.parity = c % 2;
                    }
                    run_stage(ctx_, p, s);
                }
            finish(ctx_, ctx_.passes[(chunks - 1) % S]);
        };
        try {
            scoped(0, work);
        } catch (...) {
            retire();
            for (auto& kv : seq_.kv_) kv.truncate(start);
            throw;
        }
        return row(ctx_, 0);
    }

    // Every position's logits for a text from an empty history, through the batched passes prefill takes, handed to `each` as (position, logits) a microbatch at a time. Scoring through this rather than step exercises the prompt path, whose kernels differ from the decode path's on a device (inference/perplexity.hpp).
    void score(const std::vector<uint32_t>& ids, const std::function<void(size_t, const float*)>& each) {
        if (ids.empty()) throw std::runtime_error("inference: empty text");
        reset();
        auto work = [&] {
            ensure(ctx_, std::min((size_t)ubatch(), ids.size()), std::min((size_t)ubatch(), ids.size()));
            for (size_t i = 0; i < ids.size();) {
                const size_t B = std::min((size_t)ubatch(), ids.size() - i);
                BatchEntry entry{&seq_, ids.data() + i, B, true};
                entry.every_logits = true;
                entry.extent = ids.size();
                entry.fresh = ids.size();
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
        return seq_.length() * (size_t)cfg.n_layer * kv_bytes_per_position(cfg, options_);
    }

    // Whether any weight still reads the GGUF model's tensor bytes in place. A host backend adopts by aliasing them; a device backend copies them into its own memory, and a model on device backends alone then holds every weight twice unless its owner releases the host copy (GGUFModel::release_payload).
    bool holds_payload() const { return holds_payload_; }

private:
    // One backend and what the placement put on it.
    // A pool is not movable, because sequences hold its address, so devices live behind pointers.
    struct Device {
        backend::BackendPtr b;
        bool used = false;
        int attn_layers = 0;
        int storage_index = -1;
        std::vector<int> local_layer;            // model layer -> layer in storage
        std::unique_ptr<backend::KVStorage> storage;
        BlockPool pool;
        backend::BufferPtr rope_cos, rope_sin;
    };

    // Consecutive layers whose attention runs on one device, and every device a stage records work on, which it submits.
    struct Stage {
        size_t device;
        int first, end;
        std::vector<size_t> touches;
    };

    const gguf::GGUFModel* m_;
    bool holds_payload_ = false;   // some weight reads the GGUF model's bytes in place
    std::vector<char> host_reads_, copied_;   // per tensor: a host reads it in place, a device copied it
    Placement place_;
    std::vector<std::unique_ptr<Device>> devices_;
    std::vector<Device*> storages_;              // the devices that run attention
    std::vector<Stage> stages_;
    bool pipelined_ = false;                     // a prompt's chunks flow through the stages together (prefill)
    QwenConfig cfg;
    int q_dim_ = 0;
    int ubatch_ = kDefaultUbatch;
    ModelOptions options_;
    bool any_dense_ = false;   // some layer has a dense feed-forward block
    // Per device, what a streamed layer's experts are copied into: one buffer per projection, sized to the largest streamed layer's.
    struct Window { backend::BufferPtr gate, up, down; };
    std::vector<Window> windows_;
    std::string out_name_;
    std::unordered_map<std::string, size_t> tindex_;
    std::vector<LayerWeights> layers_;
    Weight token_embd_, output_norm_, output_;
    std::vector<float> rope_cos_, rope_sin_;
    Sequence seq_;
    ExecContext ctx_;

    const gguf::TensorInfo& tensor(const std::string& name) const {
        auto it = tindex_.find(name);
        if (it == tindex_.end()) throw std::runtime_error("inference: missing tensor " + name);
        return m_->tensors[it->second];
    }

    // Validate every tensor this architecture needs and resolve it to a Weight in the same pass, so a resolved handle is well-formed by construction and the forward pass never looks a tensor up by name.
    // Each weight is adopted by the backend that hosts its role.
    void resolve_tensors() {
        const auto& embedding = tensor("token_embd.weight");
        if (embedding.ne.size() < 2 || !embedding.ne[1] ||
            embedding.ne[1] > uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("inference: invalid vocabulary dimension");
        const uint64_t vocab = embedding.ne[1];
        auto check = [&](size_t device, const std::string& name, uint64_t input,
                         uint64_t output, bool norm = false) -> Weight {
            const auto& t = tensor(name);
            bool valid = !t.ne.empty() && t.ne[0] == input;
            if (norm) {
                valid = valid && t.type == gguf::GGML_TYPE_F32;
            } else {
                valid = valid && t.ne.size() >= 2 && t.ne[1] == output;
            }
            for (size_t d = norm ? 1 : 2; d < t.ne.size(); ++d) valid = valid && t.ne[d] == 1;
            if (!valid) throw std::runtime_error("inference: incompatible tensor layout " + name);
            // adopt, not copy: the payload is already resident and the GGUF model outlives this one by contract.
            const size_t i = tindex_.at(t.name);
            backend::BufferPtr buf = devices_[device]->b->adopt(m_->tensor_data(i), m_->tensor_bytes(i));
            note_reader(i, buf);
            return Weight{t.type, std::move(buf), (size_t)input, (size_t)output};
        };
        const size_t ed = (size_t)place_.embed_device, od = (size_t)place_.output_device;
        token_embd_ = check(ed, "token_embd.weight", cfg.n_embd, vocab);
        // A tied head on the embedding's own device reads the buffer adopted for the embedding rather than a second copy.
        output_ = out_name_ == "token_embd.weight" && ed == od ? token_embd_ : check(od, out_name_, cfg.n_embd, vocab);
        output_norm_ = check(od, "output_norm.weight", cfg.n_embd, 1, true);
        const uint64_t kv_width = uint64_t(cfg.n_head_kv) * cfg.head_dim;
        layers_.resize(cfg.n_layer);
        for (int l = 0; l < cfg.n_layer; ++l) {
            const std::string pre = "blk." + std::to_string(l) + ".";
            const size_t a = (size_t)place_.attn_device[(size_t)l];
            const size_t f = (size_t)place_.ffn_device[(size_t)l];
            LayerWeights& w = layers_[l];
            w.attn_norm   = check(a, pre + "attn_norm.weight", cfg.n_embd, 1, true);
            w.attn_q_norm = check(a, pre + "attn_q_norm.weight", cfg.head_dim, 1, true);
            w.attn_k_norm = check(a, pre + "attn_k_norm.weight", cfg.head_dim, 1, true);
            w.attn_q      = check(a, pre + "attn_q.weight", cfg.n_embd, q_dim_);
            w.attn_k      = check(a, pre + "attn_k.weight", cfg.n_embd, kv_width);
            w.attn_v      = check(a, pre + "attn_v.weight", cfg.n_embd, kv_width);
            w.attn_output = check(a, pre + "attn_output.weight", q_dim_, cfg.n_embd);
            w.ffn_norm    = check(f, pre + "ffn_norm.weight", cfg.n_embd, 1, true);
            w.moe = tindex_.count(pre + "ffn_gate_inp.weight") != 0;
            if (w.moe) {
                if (!cfg.n_expert) throw std::runtime_error("inference: expert tensors in a dense architecture " + pre);
                w.ffn_gate_inp  = check(f, pre + "ffn_gate_inp.weight", cfg.n_embd, cfg.n_expert);
                w.ffn_gate_exps = experts(f, pre + "ffn_gate_exps.weight", cfg.n_embd, cfg.n_ff_exp);
                w.ffn_up_exps   = experts(f, pre + "ffn_up_exps.weight", cfg.n_embd, cfg.n_ff_exp);
                w.ffn_down_exps = experts(f, pre + "ffn_down_exps.weight", cfg.n_ff_exp, cfg.n_embd);
                // Experts read in place on the host beside attention on a device that copies its weights.
                if (place_.stream_from && a != f && !devices_[a]->b->reads_in_place() && devices_[f]->b->reads_in_place()) {
                    w.stream_device = (int)a;
                    w.stream_norm = check(a, pre + "ffn_norm.weight", cfg.n_embd, 1, true);
                    w.stream_router = check(a, pre + "ffn_gate_inp.weight", cfg.n_embd, cfg.n_expert);
                }
                continue;
            }
            if (!cfg.n_ff) throw std::runtime_error("inference: dense layer without a feed-forward width " + pre);
            any_dense_ = true;
            w.ffn_gate    = check(f, pre + "ffn_gate.weight", cfg.n_embd, cfg.n_ff);
            w.ffn_up      = check(f, pre + "ffn_up.weight", cfg.n_embd, cfg.n_ff);
            w.ffn_down    = check(f, pre + "ffn_down.weight", cfg.n_ff, cfg.n_embd);
        }
        // The windows are allocated with the weights, so a pass never fails for want of one.
        windows_.resize(devices_.size());
        std::vector<size_t> sizes(devices_.size() * 3, 0);
        for (const LayerWeights& w : layers_) {
            if (w.stream_device < 0) continue;
            size_t* s = &sizes[(size_t)w.stream_device * 3];
            s[0] = std::max(s[0], w.ffn_gate_exps.data->size());
            s[1] = std::max(s[1], w.ffn_up_exps.data->size());
            s[2] = std::max(s[2], w.ffn_down_exps.data->size());
        }
        for (size_t d = 0; d < devices_.size(); ++d) {
            const size_t* s = &sizes[d * 3];
            if (!s[0]) continue;
            backend::Backend& b = *devices_[d]->b;
            windows_[d].gate = b.alloc(s[0]);
            windows_[d].up = b.alloc(s[1]);
            windows_[d].down = b.alloc(s[2]);
        }
    }

    // A stacked expert tensor: n_expert matrices of `output` rows of `input` values, the whole tensor adopted as one buffer.
    Weight experts(size_t device, const std::string& name, uint64_t input, uint64_t output) {
        const auto& t = tensor(name);
        if (t.ne.size() != 3 || t.ne[0] != input || t.ne[1] != output || t.ne[2] != (uint64_t)cfg.n_expert)
            throw std::runtime_error("inference: incompatible tensor layout " + name);
        const size_t i = tindex_.at(t.name);
        backend::BufferPtr buf = devices_[device]->b->adopt(m_->tensor_data(i), m_->tensor_bytes(i));
        note_reader(i, buf);
        return Weight{t.type, std::move(buf), (size_t)input, (size_t)output};
    }

    // Whether each tensor is read in place by a host and whether a device copied it, so the pages of a tensor only devices hold can leave the host's working set once every weight is resolved.
    void note_reader(size_t i, const backend::BufferPtr& buf) {
        const uint8_t* hp = static_cast<const uint8_t*>(buf->host_ptr());
        if (hp && m_->holds(hp)) {
            holds_payload_ = true;
            host_reads_[i] = 1;
        } else {
            copied_[i] = 1;
        }
    }

    // Physical prompt microbatch size, used to bound matrix width and scratch storage.
    int ubatch() const { return ubatch_; }

    // The history a pass continues: the committed length of the first stage's storage, which a pipelined prompt's chunk commits first; outside a prompt every storage agrees.
    size_t history(const Sequence& s) const {
        return s.kv_[(size_t)devices_[stages_.front().device]->storage_index].length();
    }

    // A pass's plan: its rows in entry order, their positions after each history, and the rows the head reads.
    // Nothing is reserved yet; each stage reserves the blocks of the storage it writes.
    void begin(ExecContext& ctx, Pass& p, const BatchEntry* entries, size_t n_entries) {
        if (!entries || !n_entries) throw std::runtime_error("inference: empty batch");
        size_t rows = 0, want = 0;
        for (size_t e = 0; e < n_entries; ++e) {
            const BatchEntry& en = entries[e];
            if (!en.seq || en.seq->owner_ != this)
                throw std::runtime_error("inference: batch entry without a sequence of this model");
            if (!en.ids || !en.n)
                throw std::runtime_error("inference: batch entry without tokens");
            // The RoPE table is precomputed for [0, context_length); a row past it would read off the end.
            if (en.n > (size_t)cfg.context_length ||
                history(*en.seq) > (size_t)cfg.context_length - en.n)
                throw std::runtime_error("inference: context length exceeded (" +
                                         std::to_string(cfg.context_length) + " tokens)");
            rows += en.n;
            want += en.want_logits ? (en.every_logits ? en.n : 1) : 0;
        }
        ensure(ctx, rows, want);
        p.entries.assign(entries, entries + n_entries);
        p.start.resize(n_entries);
        p.rows = rows;
        p.want = want;
        p.parity = 0;
        p.ids.resize(rows);
        p.pos.resize(rows);
        p.pick.resize(want);
        p.runs.resize(n_entries);
        p.fresh.resize(n_entries);
        p.head_runs.clear();
        p.views.resize(storages_.size());
        for (auto& v : p.views) v.resize(n_entries);
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
            p.fresh[e] = en.fresh ? en.fresh : en.n;
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
    }

    // Stage s of a pass: its storage's blocks reserved, the residual embedded or received from the stage before, its layers, then the head after the last stage or the residual sent on, its submissions, and the storage's commit.
    // A sequence listed twice fails at the first reservation, since its second finds the first still pending.
    void run_stage(ExecContext& ctx, Pass& p, size_t s) {
        const Stage& st = stages_[s];
        Device& home = *devices_[st.device];
        const size_t storage = (size_t)home.storage_index;
        for (size_t e = 0; e < p.entries.size(); ++e) {
            KVSequence& kv = p.entries[e].seq->kv_[storage];
            kv.prepare(p.entries[e].n);
            p.views[storage][e] = kv.view(home.storage.get());
            p.views[storage][e].extent = p.runs[e].extent;
        }
        size_t cur = st.device;
        if (s == 0) {
            cur = (size_t)place_.embed_device;
            devices_[cur]->b->embed(slot(ctx, cur, 0), token_embd_.type, token_embd_.slice(),
                                    token_embd_.nin, token_embd_.nout, p.ids.data(), p.rows);
        } else if (p.at != cur) {
            receive(ctx, p.at, p.parity, p.sent, cur, 0, p.rows);
        }
        const backend::RowRuns all{p.runs.data(), p.runs.size()};
        for (int l = st.first; l < st.end; l++) {
            if (st.device != cur) { cross(ctx, cur, st.device, 0, p.rows); cur = st.device; }
            attention_half(ctx, p, cur, l, p.rows, p.entries.size());
            if (p.long_runs && layers_[(size_t)l].stream_device == (int)cur) {
                ffn_split(ctx, p, cur, l);
                continue;
            }
            const size_t f = (size_t)place_.ffn_device[(size_t)l];
            if (f != cur) { cross(ctx, cur, f, 0, p.rows); cur = f; }
            ffn_half(ctx, cur, l, 0, p.rows, all, false);
        }
        if (s + 1 < stages_.size()) {
            // A residual already where the next stage runs stays there.
            if (cur != stages_[s + 1].device) send(ctx, cur, p.parity, 0, p.rows);
            p.at = cur;
        } else {
            const size_t o = (size_t)place_.output_device;
            if (o != cur) { cross(ctx, cur, o, 0, p.rows); cur = o; }
            if (p.want) {
                // The rows that want logits are not contiguous once entries mix, so they are compacted first and the head runs once over exactly those rows.
                const size_t E = (size_t)cfg.n_embd;
                backend::Backend& b = *devices_[cur]->b;
                b.gather_rows(slot(ctx, cur, 1), slot(ctx, cur, 0), E, p.pick.data(), p.want);
                b.rms_norm_rows(slot(ctx, cur, 1), slot(ctx, cur, 1), output_norm_.slice(),
                                p.want, E, E, cfg.rms_eps);
                b.matmul_logits(output_.type, output_.slice(), slot(ctx, cur, 1),
                                {ctx.logits_buf.get(), 0}, output_.nin, output_.nout, p.want,
                                backend::RowRuns{p.head_runs.data(), p.head_runs.size()});
            }
        }
        for (size_t d : st.touches) ctx.tickets[d] = devices_[d]->b->submit();
        p.sent = ctx.tickets[cur];
        for (const BatchEntry& en : p.entries) {
            en.seq->kv_[storage].commit();
            for (size_t d : st.touches) en.seq->last_[d] = ctx.tickets[d];
        }
    }

    // After the last stage: where the logits are and the ticket that says they are ready.
    void finish(ExecContext& ctx, const Pass& p) {
        ctx.n_logits = p.want;
        ctx.backend = devices_[(size_t)place_.output_device]->b.get();
        ctx.ticket = ctx.tickets[(size_t)place_.output_device];
        ctx.pending = p.want > 0;
    }

    // A failed pass: every device drained, then every entry's histories back to where the pass found them, the blocks its stages reserved or committed returned.
    void roll_back(const Pass& p) noexcept {
        retire();
        for (size_t e = 0; e < p.entries.size(); ++e)
            for (auto& kv : p.entries[e].seq->kv_) kv.truncate(p.start[e]);
    }

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
    backend::BufferPtr alloc_arena(backend::Backend& b,
                                   const size_t (&counts)[ExecContext::kSlots],
                                   size_t (&offsets)[ExecContext::kSlots]) const {
        size_t total = 0;
        for (size_t i = 0; i < ExecContext::kSlots; ++i) {
            offsets[i] = total;
            const size_t bytes = counts[i] * sizeof(float);
            if (bytes / sizeof(float) != counts[i] || total > (size_t)-1 - bytes - 63)
                throw std::runtime_error("inference: activation arena size overflows");
            total = (total + bytes + 63) / 64 * 64;
        }
        return b.alloc(total);
    }

    // Storage for a pass of `rows` rows with `want` logits rows, on every device the placement uses: grown when a pass needs more than the context holds, never shrunk.
    // Each is allocated whole before it replaces what the context had.
    void ensure(ExecContext& ctx, size_t rows, size_t want) {
        auto mul = [](size_t a, size_t b) {
            if (b && a > (size_t)-1 / b)
                throw std::runtime_error("inference: activation arena size overflows");
            return a * b;
        };
        ctx.scratch.resize(devices_.size());
        ctx.tickets.resize(devices_.size(), 0);
        for (size_t d = 0; d < devices_.size(); ++d) {
            ExecContext::Scratch& sc = ctx.scratch[d];
            if (!devices_[d]->used || (sc.arena && sc.rows >= rows)) continue;
            const auto widths = slot_widths(cfg, any_dense_);
            size_t counts[ExecContext::kSlots];
            for (size_t i = 0; i < ExecContext::kSlots; ++i) counts[i] = mul(rows, widths[i]);
            size_t offsets[ExecContext::kSlots];
            backend::BufferPtr arena = alloc_arena(*devices_[d]->b, counts, offsets);
            // A pass through this context may still run on the arena being replaced: nothing else waits for a pass that wanted no logits.
            if (sc.arena) devices_[d]->b->wait(ctx.tickets[d]);
            sc.arena = std::move(arena);
            std::copy(offsets, offsets + ExecContext::kSlots, sc.offset);
            sc.rows = rows;
        }
        // A host-visible buffer per used device, which a crossing leaves through, when more than one device is used: two on a pipelined split, so a chunk goes out through one while the chunk before it still waits in the other.
        size_t used = 0;
        for (const auto& d : devices_) used += d->used;
        if (used > 1 && ctx.handoff_rows < rows) {
            std::vector<std::array<backend::BufferPtr, 2>> handoff(devices_.size());
            const size_t bytes = mul(mul(rows, (size_t)cfg.n_embd), sizeof(float));
            for (size_t d = 0; d < devices_.size(); ++d)
                for (size_t i = 0; devices_[d]->used && i < (pipelined_ ? 2u : 1u); ++i)
                    handoff[d][i] = devices_[d]->b->alloc(bytes, backend::Memory::host_visible);
            for (size_t d = 0; d < ctx.handoff.size(); ++d)
                if (devices_[d]->used) devices_[d]->b->wait(ctx.tickets[d]);
            ctx.handoff = std::move(handoff);
            ctx.handoff_rows = rows;
        }
        if (want && (!ctx.logits_buf || ctx.logit_rows < want)) {
            // The head writes here and the host reads it in place once the pass has retired: the one point per pass that must be host visible, and the one wait per pass.
            backend::BufferPtr logits = devices_[(size_t)place_.output_device]->b->alloc(
                mul(mul(want, output_.nout), sizeof(float)), backend::Memory::host_visible);
            if (ctx.logits_buf) devices_[(size_t)place_.output_device]->b->wait(ctx.tickets[(size_t)place_.output_device]);
            ctx.logits_buf = std::move(logits);
            ctx.logit_rows = want;
            ctx.width = output_.nout;
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
    void send(ExecContext& ctx, size_t from, size_t parity, size_t base, size_t rows) {
        const size_t E = (size_t)cfg.n_embd;
        const backend::Slice x = slot(ctx, from, 0);
        devices_[from]->b->copy(*ctx.handoff[from][parity], base * E * sizeof(float), *x.buffer,
                                (x.offset + base * E) * sizeof(float), rows * E * sizeof(float));
    }

    // `receive` waits for that submission and writes the rows into the destination's residual, enqueued there.
    void receive(ExecContext& ctx, size_t from, size_t parity, backend::Ticket sent, size_t to, size_t base, size_t rows) {
        const size_t E = (size_t)cfg.n_embd;
        devices_[from]->b->wait(sent);
        const backend::Slice x = slot(ctx, to, 0);
        const uint8_t* rows_out = (const uint8_t*)ctx.handoff[from][parity]->host_ptr() + base * E * sizeof(float);
        devices_[to]->b->write(*x.buffer, (x.offset + base * E) * sizeof(float), rows_out, rows * E * sizeof(float));
    }

    // A crossing inside a stage, both halves at once.
    void cross(ExecContext& ctx, size_t from, size_t to, size_t base, size_t rows) {
        send(ctx, from, 0, base, rows);
        receive(ctx, from, 0, devices_[from]->b->submit(), to, base, rows);
    }

    // Whether entry e of a pass takes a streamed layer on the device (Placement::stream_from): a prompt of enough new tokens, never a generated token.
    bool streams(const Pass& p, size_t e) const {
        return place_.stream_from && p.runs[e].extent > 1 && p.fresh[e] >= std::max<size_t>(place_.stream_from, 2);
    }

    // A streamed layer in a pass with long runs: consecutive entries alike form a group, the long ones run where the residual is, with the experts copied into the window once, and the rest on the host through a crossing each way.
    // The residual ends where it started, on the layer's attention device.
    void ffn_split(ExecContext& ctx, const Pass& p, size_t dev, int l) {
        const LayerWeights& w = layers_[(size_t)l];
        const size_t host = (size_t)place_.ffn_device[(size_t)l];
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
                    const Window& win = windows_[dev];
                    b.write(*win.gate, 0, w.ffn_gate_exps.data->host_ptr(), w.ffn_gate_exps.data->size());
                    b.write(*win.up, 0, w.ffn_up_exps.data->host_ptr(), w.ffn_up_exps.data->size());
                    b.write(*win.down, 0, w.ffn_down_exps.data->host_ptr(), w.ffn_down_exps.data->size());
                    copied = true;
                }
                ffn_half(ctx, dev, l, base, end - base, runs, true);
            } else {
                cross(ctx, dev, host, base, end - base);
                ffn_half(ctx, host, l, base, end - base, runs, false);
                cross(ctx, host, dev, base, end - base);
            }
            base = end;
        }
    }

    // The slots are those of slot_widths.
    void attention_half(ExecContext& ctx, const Pass& p, size_t dev, int l, size_t rows, size_t n_views) {
        Device& d = *devices_[dev];
        backend::Backend& b = *d.b;
        const LayerWeights& w = layers_[(size_t)l];
        const size_t E = (size_t)cfg.n_embd, half = (size_t)cfg.head_dim / 2;
        const size_t KV = (size_t)cfg.n_head_kv * cfg.head_dim;
        const backend::Slice x = slot(ctx, dev, 0), h = slot(ctx, dev, 1), q = slot(ctx, dev, 2),
                             k = slot(ctx, dev, 3), v = slot(ctx, dev, 4), attn = slot(ctx, dev, 5);
        const size_t layer = (size_t)d.local_layer[(size_t)l];
        const std::vector<backend::KVView>& views = p.views[(size_t)d.storage_index];

        b.rms_norm_rows(h, x, w.attn_norm.slice(), rows, E, E, cfg.rms_eps);

        const backend::RowRuns runs{p.runs.data(), p.runs.size()};
        b.matmul_group({projection(w.attn_q, q),
                        projection(w.attn_k, k),
                        projection(w.attn_v, v)}, h, E, rows, runs);

        const backend::Backend::RopeArgs rope{{d.rope_cos.get(), 0}, {d.rope_sin.get(), 0},
                                              half, p.pos.data(), cfg.rms_eps};
        b.norm_rope_kv(q, (size_t)q_dim_, cfg.n_head, w.attn_q_norm.slice(),
                       k, v, KV, cfg.n_head_kv, w.attn_k_norm.slice(),
                       rope, rows, layer, views.data(), n_views);
        b.attention(q, layer, views.data(), n_views, attn,
                    cfg.n_head, cfg.n_head_kv, cfg.head_dim);

        b.matmul_add(w.attn_output.type, w.attn_output.slice(), attn, x,
                     w.attn_output.nin, w.attn_output.nout, rows, runs);
    }

    // `rows` rows of the residual from `base`, through the scratch slots from their start; a streamed layer's on its attention device reads the copies there.
    void ffn_half(ExecContext& ctx, size_t dev, int l, size_t base, size_t rows, backend::RowRuns runs, bool streamed) {
        backend::Backend& b = *devices_[dev]->b;
        const LayerWeights& w = layers_[(size_t)l];
        const size_t E = (size_t)cfg.n_embd;
        backend::Slice x = slot(ctx, dev, 0);
        x.offset += base * E;
        const backend::Slice h = slot(ctx, dev, 1), gate = slot(ctx, dev, 6), up = slot(ctx, dev, 7), ffn = slot(ctx, dev, 8);

        b.rms_norm_rows(h, x, (streamed ? w.stream_norm : w.ffn_norm).slice(), rows, E, E, cfg.rms_eps);

        if (w.moe) {
            const backend::Slice scores = slot(ctx, dev, 9), ids = slot(ctx, dev, 10), weights = slot(ctx, dev, 11);
            const size_t k = (size_t)cfg.n_expert_used, n_expert = (size_t)cfg.n_expert, ff = (size_t)cfg.n_ff_exp;
            const Weight& router = streamed ? w.stream_router : w.ffn_gate_inp;
            const Window* win = streamed ? &windows_[dev] : nullptr;
            b.matmul(router.type, router.slice(), h, scores, E, n_expert, rows, runs);
            b.route_experts(scores, rows, n_expert, k, cfg.expert_norm, ids, weights);
            const backend::Backend::Routing routing{ids, weights, k, n_expert};
            b.matmul_experts({projection(w.ffn_gate_exps, gate, win ? win->gate.get() : nullptr),
                              projection(w.ffn_up_exps, up, win ? win->up.get() : nullptr)}, h, E, rows, routing, runs);
            b.silu_mul(ffn, gate, up, rows * k * ff);
            b.matmul_experts_add(w.ffn_down_exps.type, {win ? win->down.get() : w.ffn_down_exps.data.get(), 0}, ffn, x,
                                 ff, E, rows, routing, runs);
            return;
        }
        b.matmul_group({projection(w.ffn_gate, gate),
                        projection(w.ffn_up, up)}, h, E, rows, runs);
        b.silu_mul(ffn, gate, up, rows * (size_t)cfg.n_ff);
        b.matmul_add(w.ffn_down.type, w.ffn_down.slice(), ffn, x,
                     w.ffn_down.nin, w.ffn_down.nout, rows, runs);
    }

    // A block returns to the pool only once the backend has retired every submission that touched it (docs/KV-CACHE.md).
    // Construction failures, failed passes and model teardown drain every used device with sync(), including work behind no ticket; reset() waits on tickets instead.
    void retire() noexcept {
        for (auto& d : devices_) if (d->used) d->b->sync();
    }

    // The buffer is passed by raw pointer, not by handle: three projections per layer per token is nearly two hundred refcount pairs a token if a shared pointer is copied here instead.
    static backend::Projection projection(const Weight& w, backend::Slice out, const backend::Buffer* copy = nullptr) {
        return {w.type, {copy ? copy : w.data.get(), 0}, out, w.nout};
    }
};

// How a caller wants a model placed over the backends it made (docs/MULTI-DEVICE.md).
struct PlacementRequest {
    std::vector<std::string> names;   // each backend's name, for the fit's messages and its description
    std::vector<int> shares;          // each backend's proportion of the layers; empty to fit them to the devices' free memory
    int cpu_moe = 0;                  // with one backend, the routed layers whose experts run on the CPU beside it, -1 for every one
    size_t stream_from = 0;           // with experts on the CPU, the prompt length from which they are copied to the device (Placement::stream_from)
    int ubatch = 0;                   // prompt tokens a pass takes, kDefaultUbatch when 0
    size_t decode_rows = 0;           // generated tokens a pass may carry beside a prompt's: a server's decoding requests
};

// A placed model and, when it was split, what each device was given (LayerSplit::describe).
struct PlacedModel {
    std::unique_ptr<Model> model;
    std::string plan;
};

// The model over one backend, over one with the first `cpu_moe` routed layers' experts on the CPU beside it, or split by layers over several, with the request's ubatch set.
// The CPU is device 0 of an experts placement, so the thread count the model reports is the host's; attention, the dense blocks, the embedding and the head stay on the device.
inline PlacedModel place_model(const gguf::GGUFModel& m, std::vector<backend::BackendPtr> backends, const PlacementRequest& request,
                               const ModelOptions& options) {
    if (backends.empty()) throw std::runtime_error("placement: no device");
    PlacedModel placed;
    if (backends.size() > 1 || !request.shares.empty()) {
        if (request.cpu_moe)
            throw std::runtime_error("--n-cpu-moe and --cpu-moe: not with several devices; list the CPU as a device to give it layers");
        const std::vector<DeviceBudget> budgets = budgets_for(backends, request.names);
        const size_t rows = (size_t)(request.ubatch > 0 ? request.ubatch : kDefaultUbatch) + request.decode_rows;
        const LayerSplit split = split_layers(footprint(m, options), budgets, rows, request.shares, core::host_memory_available());
        placed = {std::make_unique<Model>(m, std::move(backends), placement_for(split), options), split.describe(budgets)};
    } else if (!request.cpu_moe || backends[0]->reads_in_place()) {
        placed.model = std::make_unique<Model>(m, std::move(backends[0]), options);
    } else {
        const QwenConfig cfg = load_config(m);
        Placement place;
        place.attn_device.assign((size_t)cfg.n_layer, 1);
        place.ffn_device.assign((size_t)cfg.n_layer, 1);
        place.embed_device = place.output_device = 1;
        place.stream_from = request.stream_from;
        const std::vector<bool> routed = routed_layers(m, cfg.n_layer);
        int seen = 0;
        for (int l = 0; l < cfg.n_layer; ++l) {
            if (!routed[(size_t)l]) continue;
            if (request.cpu_moe < 0 || seen < request.cpu_moe) place.ffn_device[(size_t)l] = 0;
            ++seen;
        }
        if (!seen) throw std::runtime_error("--n-cpu-moe: the model has no expert layers");
        std::vector<backend::BackendPtr> both{backend::make_cpu_backend(), std::move(backends[0])};
        placed.model = std::make_unique<Model>(m, std::move(both), place, options);
    }
    placed.model->set_ubatch(request.ubatch);
    return placed;
}

} // namespace infer
