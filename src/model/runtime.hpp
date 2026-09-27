#pragma once
#include <algorithm>
#include <memory>
#include <functional>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <optional>
#include <random>

#include "format/gguf.hpp"
#include "quant/quant.hpp"
#include "backends/backend.hpp"
#include "model/weights.hpp"
#include "model/architecture.hpp"
#include "model/kv_cache.hpp"
#include "model/layer_split.hpp"
#include "backends/cpu/cpu_backend.hpp"
#include "core/host_memory.hpp"

// The model runtime, which runs an architecture's plan and parts (model/architecture.hpp), and Qwen3's architecture: dense Qwen3 and its mixture-of-experts form, qwen3moe.
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

namespace qwen3 {

// Qwen3's roles, by the id a row of resolved weights is indexed by, and its layer kinds.
enum Role : uint16_t {
    token_embd, output, output_norm,
    attn_norm, attn_q_norm, attn_k_norm, attn_q, attn_k, attn_v, attn_output,
    ffn_norm, ffn_gate, ffn_up, ffn_down, ffn_gate_inp, ffn_gate_exps, ffn_up_exps, ffn_down_exps,
};
enum Kind : uint8_t { dense, routed };

// The floats one row of a pass takes in each slot of an activation arena (ExecContext::Scratch), which the plan gives the arena and the fit.
// Slots: 0 x, 1 h, 2 q, 3 k, 4 v, 5 attn, 6 gate, 7 up, 8 ffn, 9 router scores, 10 expert ids, 11 expert weights.
// The feed-forward slots hold a dense layer's hidden rows or a routed layer's k expert rows per token, whichever is wider.
inline std::vector<size_t> slot_widths(const QwenConfig& cfg, bool dense) {
    const size_t q = (size_t)cfg.n_head * cfg.head_dim, kv = (size_t)cfg.n_head_kv * cfg.head_dim;
    const size_t ff = std::max(dense ? (size_t)cfg.n_ff : 0, (size_t)cfg.n_expert_used * (size_t)cfg.n_ff_exp);
    const size_t e = (size_t)cfg.n_embd, k = (size_t)cfg.n_expert_used;
    return {e, e, q, kv, kv, q, ff, ff, ff, (size_t)cfg.n_expert, k, k};
}

// Qwen3 and qwen3moe under one configuration: the plan of a file's tensors, the rope tables, and the embedding, attention, feed-forward block and head as backend ops.
class Qwen3 final : public Architecture {
public:
    explicit Qwen3(const QwenConfig& cfg) : cfg_(cfg) {}

    // The embedding gives the vocabulary, and a layer with a router is routed, which a dense architecture refuses, as a dense layer is refused without the dense width.
    // A layer's roles are its attention's, its feed-forward norm, then its router and expert stacks or its three dense matrices; a routed layer copies its norm and router beside its mixer and writes its stacks into a window.
    // The arena's feed-forward slots are as wide as a dense layer's when some layer is dense, and the two rope tables cover the context at half a head each.
    ModelPlan plan(const TensorIndex& tensors) const override {
        const QwenConfig& cfg = cfg_;
        const TensorView& embedding = tensors.view(tensors.at("token_embd.weight"));
        if (embedding.shape.size() < 2 || !embedding.shape[1] ||
            embedding.shape[1] > uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("inference: invalid vocabulary dimension");
        ModelPlan p;
        p.role_ids = ffn_down_exps + 1;
        p.vocab = embedding.shape[1];
        const uint64_t E = (uint64_t)cfg.n_embd, D = (uint64_t)cfg.head_dim;
        const uint64_t Q = (uint64_t)cfg.n_head * D, KV = (uint64_t)cfg.n_head_kv * D;
        p.pass = {{token_embd, Part::embed, RoleKind::gather, "token_embd.weight", "", E, p.vocab},
                  {output, Part::head, RoleKind::matrix, "output.weight", "token_embd.weight", E, p.vocab},
                  {output_norm, Part::head, RoleKind::norm, "output_norm.weight", "", E}};
        p.layers.resize((size_t)cfg.n_layer);
        bool any_dense = false;
        for (int l = 0; l < cfg.n_layer; ++l) {
            const std::string pre = "blk." + std::to_string(l) + ".";
            LayerPlan& layer = p.layers[(size_t)l];
            layer.routed = tensors.find(pre + "ffn_gate_inp.weight").has_value();
            if (layer.routed && !cfg.n_expert) throw std::runtime_error("inference: expert tensors in a dense architecture " + pre);
            if (!layer.routed && !cfg.n_ff) throw std::runtime_error("inference: dense layer without a feed-forward width " + pre);
            layer.kind = layer.routed ? routed : dense;
            const Stream copy = layer.routed ? Stream::copy : Stream::none;
            layer.roles = {{attn_norm, Part::mixer, RoleKind::norm, pre + "attn_norm.weight", "", E},
                           {attn_q_norm, Part::mixer, RoleKind::norm, pre + "attn_q_norm.weight", "", D},
                           {attn_k_norm, Part::mixer, RoleKind::norm, pre + "attn_k_norm.weight", "", D},
                           {attn_q, Part::mixer, RoleKind::matrix, pre + "attn_q.weight", "", E, Q},
                           {attn_k, Part::mixer, RoleKind::matrix, pre + "attn_k.weight", "", E, KV},
                           {attn_v, Part::mixer, RoleKind::matrix, pre + "attn_v.weight", "", E, KV},
                           {attn_output, Part::mixer, RoleKind::matrix, pre + "attn_output.weight", "", Q, E},
                           {ffn_norm, Part::ffn, RoleKind::norm, pre + "ffn_norm.weight", "", E, 1, 0, copy}};
            if (layer.routed) {
                const uint64_t X = (uint64_t)cfg.n_expert, F = (uint64_t)cfg.n_ff_exp;
                layer.roles.push_back({ffn_gate_inp, Part::ffn, RoleKind::matrix, pre + "ffn_gate_inp.weight", "", E, X, 0, Stream::copy});
                layer.roles.push_back({ffn_gate_exps, Part::ffn, RoleKind::experts, pre + "ffn_gate_exps.weight", "", E, F, X, Stream::window});
                layer.roles.push_back({ffn_up_exps, Part::ffn, RoleKind::experts, pre + "ffn_up_exps.weight", "", E, F, X, Stream::window});
                layer.roles.push_back({ffn_down_exps, Part::ffn, RoleKind::experts, pre + "ffn_down_exps.weight", "", F, E, X, Stream::window});
            } else {
                const uint64_t F = (uint64_t)cfg.n_ff;
                layer.roles.push_back({ffn_gate, Part::ffn, RoleKind::matrix, pre + "ffn_gate.weight", "", E, F});
                layer.roles.push_back({ffn_up, Part::ffn, RoleKind::matrix, pre + "ffn_up.weight", "", E, F});
                layer.roles.push_back({ffn_down, Part::ffn, RoleKind::matrix, pre + "ffn_down.weight", "", F, E});
            }
            any_dense = any_dense || !layer.routed;
        }
        p.context_length = (size_t)cfg.context_length;
        p.slots = slot_widths(cfg, any_dense);
        p.residual = (size_t)cfg.n_embd;
        p.kv_heads = (size_t)cfg.n_head_kv;
        p.head_dim = (size_t)cfg.head_dim;
        p.tables.assign(2, (size_t)cfg.context_length * (size_t)(cfg.head_dim / 2));
        return p;
    }

    // The RoPE cos and sin tables for every position up to the context length, indexed as [pos*(head_dim/2) + i].
    void fill_tables(std::vector<std::vector<float>>& tables) const override {
        std::vector<float>& rope_cos = tables[0];
        std::vector<float>& rope_sin = tables[1];
        int half = cfg_.head_dim / 2;
        for (int pos = 0; pos < cfg_.context_length; pos++) {
            for (int i = 0; i < half; i++) {
                float fre = std::pow(cfg_.rope_theta, -2.0f * (float)i / (float)cfg_.head_dim);
                rope_cos[(size_t)pos * half + i] = std::cos((float)pos * fre);
                rope_sin[(size_t)pos * half + i] = std::sin((float)pos * fre);
            }
        }
    }

    void embed(const Step& s, const uint32_t* ids) const override {
        const Weight& embedding = s.w[token_embd];
        s.b.embed(s.x, embedding.type, embedding.slice(), embedding.nin, embedding.nout, ids, s.rows);
    }

    // The slots are those of slot_widths.
    void mixer(const Step& s) const override {
        backend::Backend& b = s.b;
        const Weight* w = s.w;
        const size_t E = (size_t)cfg_.n_embd, half = (size_t)cfg_.head_dim / 2;
        const size_t Q = (size_t)cfg_.n_head * cfg_.head_dim, KV = (size_t)cfg_.n_head_kv * cfg_.head_dim;
        const backend::Slice x = s.x, h = s.slot(1), q = s.slot(2), k = s.slot(3), v = s.slot(4), attn = s.slot(5);

        b.rms_norm_rows(h, x, w[attn_norm].slice(), s.rows, E, E, cfg_.rms_eps, s.runs);

        b.matmul_group({projection(w[attn_q], q),
                        projection(w[attn_k], k),
                        projection(w[attn_v], v)}, h, E, s.rows, s.runs);

        const backend::Backend::RopeArgs rope{{s.tables[0].get(), 0}, {s.tables[1].get(), 0},
                                              half, s.pos, cfg_.rms_eps};
        b.norm_rope_kv(q, Q, cfg_.n_head, w[attn_q_norm].slice(),
                       k, v, KV, cfg_.n_head_kv, w[attn_k_norm].slice(),
                       rope, s.rows, s.kv_layer, s.views, s.n_views);
        b.attention(q, s.kv_layer, s.views, s.n_views, attn,
                    cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim);

        b.matmul_add(w[attn_output].type, w[attn_output].slice(), attn, x,
                     w[attn_output].nin, w[attn_output].nout, s.rows, s.runs);
    }

    // The rows of the residual from s.x, through the scratch slots from their start, reading whichever row of weights the runtime hands it: the layer's own, or on its attention device a streamed layer's copies and windows.
    void ffn(const Step& s) const override {
        backend::Backend& b = s.b;
        const Weight* w = s.w;
        const size_t E = (size_t)cfg_.n_embd;
        const backend::Slice x = s.x, h = s.slot(1), gate = s.slot(6), up = s.slot(7), ffn = s.slot(8);

        b.rms_norm_rows(h, x, w[ffn_norm].slice(), s.rows, E, E, cfg_.rms_eps, s.runs);

        if (s.kind == routed) {
            const backend::Slice scores = s.slot(9), ids = s.slot(10), weights = s.slot(11);
            const size_t k = (size_t)cfg_.n_expert_used, n_expert = (size_t)cfg_.n_expert, ff = (size_t)cfg_.n_ff_exp;
            const Weight& router = w[ffn_gate_inp];
            b.matmul(router.type, router.slice(), h, scores, E, n_expert, s.rows, s.runs);
            b.route_experts(scores, s.rows, n_expert, k, cfg_.expert_norm, ids, weights);
            const backend::Backend::Routing routing{ids, weights, k, n_expert};
            b.matmul_experts({projection(w[ffn_gate_exps], gate),
                              projection(w[ffn_up_exps], up)}, h, E, s.rows, routing, s.runs);
            // The routed down projection reads the SiLU's output as k entries a token row, each of its token's prompt.
            std::vector<backend::RowRun>& entry_runs = *s.scratch;
            entry_runs.clear();
            for (size_t i = 0; i < s.runs.n; ++i) entry_runs.push_back(backend::RowRun{s.runs.runs[i].end * k, s.runs.runs[i].extent});
            b.silu_mul(ffn, gate, up, s.rows * k * ff, {entry_runs.data(), entry_runs.size()});
            b.matmul_experts_add(w[ffn_down_exps].type, w[ffn_down_exps].slice(), ffn, x,
                                 ff, E, s.rows, routing, s.runs);
            return;
        }
        b.matmul_group({projection(w[ffn_gate], gate),
                        projection(w[ffn_up], up)}, h, E, s.rows, s.runs);
        b.silu_mul(ffn, gate, up, s.rows * (size_t)cfg_.n_ff, s.runs);
        b.matmul_add(w[ffn_down].type, w[ffn_down].slice(), ffn, x,
                     w[ffn_down].nin, w[ffn_down].nout, s.rows, s.runs);
    }

    // The rows that want logits are not contiguous once entries mix, so they are compacted first and the head runs once over exactly those rows.
    void head(const HeadStep& s) const override {
        const size_t E = (size_t)cfg_.n_embd;
        const Weight& norm = s.w[output_norm];
        const Weight& head = s.w[output];
        s.b.gather_rows(s.slot(1), s.x, E, s.pick, s.want);
        s.b.rms_norm_rows(s.slot(1), s.slot(1), norm.slice(), s.want, E, E, cfg_.rms_eps);
        s.b.matmul_logits(head.type, head.slice(), s.slot(1), s.logits, head.nin, head.nout, s.want, s.head_runs);
    }

private:
    QwenConfig cfg_;

    // The buffer is passed by raw pointer, not by handle, so building a projection copies no shared pointer on the per-token path.
    static backend::Projection projection(const Weight& w, backend::Slice out) {
        return {w.type, {w.data.get(), 0}, out, w.nout};
    }
};

} // namespace qwen3

// A GGUF model's weights: its architecture, whose configuration is read once, and a view of every tensor, whose data is null while the tensor's file is not mapped (gguf::map_payload).
// A tensor table whose storage count does not match its tensors, with a rank above four, or an offset or extent outside the payload is refused, which includes a payload its owner released; read_gguf has refused duplicate names already.
inline ModelWeights gguf_weights(const gguf::GGUFModel& m) {
    ModelWeights w;
    w.arch = std::make_shared<const qwen3::Qwen3>(load_config(m));
    if (m.offsets.size() != m.tensors.size())
        throw std::runtime_error("inference: tensor storage count mismatch");
    w.tensors.reserve(m.tensors.size());
    for (size_t i = 0; i < m.tensors.size(); i++) {
        const auto& t = m.tensors[i];
        if (t.ne.size() > 4)
            throw std::runtime_error("inference: invalid tensor rank " + t.name);
        const uint64_t bytes = t.data_size();
        if (m.offsets[i] % alignof(float) || m.offsets[i] > m.payload_size() ||
            bytes > m.payload_size() - m.offsets[i])
            throw std::runtime_error("inference: invalid tensor storage " + t.name);
        w.tensors.push_back({t.name, t.ne, t.type, m.tensor_data(i), (size_t)bytes});
    }
    return w;
}

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
// Per role rather than per layer, so a layer's attention and its feed-forward block can sit on different devices, as expert offload places them (docs/EXECUTION.md).
struct Placement {
    std::vector<int> attn_device, ffn_device;
    int embed_device = 0, output_device = 0;
    // A routed layer with its feed-forward block on a host and its attention on a device runs a prompt of at least this many tokens on the device, its experts copied there for each pass: past some length a prompt's expert products on the host cost more than moving the experts.
    // By the prompt's whole length (BatchEntry::extent), so every row a prompt computes takes one path however the prompt is sliced or batched; rows a server forks from a donor keep the path they were computed on, the donor prompt's for its prompt rows and the host for its generated rows (docs/SERVER.md, Open gaps).
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
};

// One request's history in a model's cache, made by Model::make_sequence for that model's pools and block sizes: a block table per storage, the committed length, and per device the ticket of the last pass that touched it, which a release waits on rather than draining the device (docs/EXECUTION.md).
// Movable, not copyable; from Model::begin_pass until its end_pass or abort_pass it is in flight, when no other pass, reset or fork takes it and it must not move, since the pass holds its address.
class Sequence {
public:
    Sequence() = default;
    // Storage 0's committed length.
    // The storages can disagree while a pass is part way through its stages, and the model continues a history from the first stage's storage (Model::history).
    size_t length() const { return kv_.empty() ? 0 : kv_[0].length(); }
    bool in_flight() const { return in_flight_; }
private:
    friend class Model;
    std::vector<KVSequence> kv_;
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
        t.type = f32 ? quant::GGML_TYPE_F32 : quant::GGML_TYPE_Q8_0;
        t.offset = 0;
        if (f32) {
            std::vector<uint8_t> buf(nin * nout * 4);
            float* p = (float*)buf.data();
            for (size_t o = 0; o < nout; o++)
                for (size_t i = 0; i < nin; i++) *p++ = dist(rng);
            m.tensors.push_back(std::move(t));
            m.add_tensor_data(buf);
        } else {
            size_t nblocks = nin / quant::Q8_0_BLOCK;
            std::vector<uint8_t> buf(nout * nblocks * quant::Q8_0_TYPESIZE);
            std::vector<float> row(nin);
            for (size_t o = 0; o < nout; o++) {
                for (size_t i = 0; i < nin; i++) row[i] = dist(rng);
                quant::quantize_row_q8_0(row.data(), buf.data() + o * nblocks * quant::Q8_0_TYPESIZE, nblocks);
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

// What a model asks of the memory of the devices it runs on, for a split fitted to them (model/layer_split.hpp), counted from its plan.
// A layer lists the tensors its roles take in the file's order, each once and a product where a role reads it as a matrix, whatever its rank; a tensor no role takes costs nothing.
// The embedding is the embed part's table, the output the head's matrix, tied when that role took its alias, and the output norm the head's norm; a pass role of any other part and kind, or a second role for one of those fields, has no field to count it in and is the plan's error.
// The cache is counted for every position the options budget; activations are the plan's arena slots, and a handoff row is a residual row.
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
    fp.cache_per_layer = kv_tokens(plan, options) * kv_bytes_per_position(plan, options);
    for (size_t n : plan.tables) fp.tables += n * sizeof(float);
    fp.handoff_per_row = plan.residual * sizeof(float);
    for (size_t w : plan.slots) fp.activations_per_row += w * sizeof(float);
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

        if (place_.attn_device.empty() && place_.ffn_device.empty()) {
            place_.attn_device.assign(n_layer, 0);
            place_.ffn_device.assign(n_layer, 0);
        }
        if (place_.attn_device.size() != n_layer ||
            place_.ffn_device.size() != n_layer)
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
            devices_.back()->local_layer.assign(n_layer, -1);
        }
        device_index(place_.embed_device);
        device_index(place_.output_device);
        devices_[(size_t)place_.embed_device]->used = true;
        devices_[(size_t)place_.output_device]->used = true;
        for (int l = 0; l < (int)n_layer; ++l) {
            Device& a = *devices_[device_index(place_.attn_device[(size_t)l])];
            a.local_layer[(size_t)l] = a.attn_layers++;
            a.used = true;
            devices_[device_index(place_.ffn_device[(size_t)l])]->used = true;
        }
        // A device's attention layers are one run, so each storage is written by one stage, which reserves and commits it once a pass.
        for (int l = 0; l < (int)n_layer; ++l) {
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
        // The residual leaves a device wherever the next role on its path (the embedding, each layer's attention and feed-forward block, the head) sits on another.
        // A feed-forward block away from its attention sends too, since a streamed layer's host rows cross back from it to the attention's device.
        size_t at = (size_t)place_.embed_device;
        for (int l = 0; l < (int)n_layer; ++l) {
            const size_t a = (size_t)place_.attn_device[(size_t)l], f = (size_t)place_.ffn_device[(size_t)l];
            if (a != at) devices_[at]->sends = true;
            if (f != a) devices_[a]->sends = devices_[f]->sends = true;
            at = f;
        }
        if ((size_t)place_.output_device != at) devices_[at]->sends = true;

        try {
            resolve_tensors(weights, adopt);

            // Each device that runs attention gets a storage for exactly its layers, with its own block size and pool.
            // Budget: the option's tokens, else the whole context; storage is backed on demand, so a short chat does not allocate it.
            const size_t budget = kv_tokens(plan_, options_);
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
                d.storage = d.b->kv_alloc((size_t)d.attn_layers, plan_.kv_heads, plan_.head_dim,
                                          budget, options_.kv_k, options_.kv_v);
                d.pool.configure(d.storage->max_blocks());
                d.storage_index = (int)storages_.size();
                storages_.push_back(&d);
            }
            seq_ = make_sequence();

            // The position tables, sized by the plan and filled once by the architecture.
            // Every device that runs attention reads them through adopted buffers, so the host vectors stay alive for the model's lifetime; on CPU that is the same memory.
            tables_.resize(plan_.tables.size());
            for (size_t t = 0; t < tables_.size(); ++t) tables_[t].assign(plan_.tables[t], 0.0f);
            arch_->fill_tables(tables_);
            for (Device* d : storages_)
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
    Sequence fork(const Sequence& src, size_t length) {
        if (src.owner_ != this) throw std::runtime_error("inference: sequence of another model");
        if (src.in_flight_) throw std::logic_error("inference: a fork of a sequence in flight");
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
    // Whether passes may be in flight together: several stages, the embedding on the first stage's device, the head on the last's and every feed-forward block beside its attention.
    bool pipelined() const { return pipelined_; }

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
        for (size_t e = 0; e < n_entries; ++e) {
            if (!entries[e].seq->in_flight_) { entries[e].seq->in_flight_ = true; continue; }
            while (e--) entries[e].seq->in_flight_ = false;
            throw std::logic_error("inference: a sequence listed twice in a pass");
        }
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
    // Blocks return to every pool; their storage is retained.
    // Every pass ends in a submit or, on failure, a sync, so the sequence's last tickets cover everything that could still be touching a block: this waits for those and no more.
    void reset(Sequence& s) {
        if (s.owner_ != this)
            throw std::runtime_error("inference: sequence of another model");
        if (s.in_flight_) throw std::logic_error("inference: a reset of a sequence in flight");
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
            // Sized to the largest chunk this prompt will use, inside the scope, so a short prompt does not allocate scratch for a full ubatch.
            // Sized before any chunk runs, so nothing in flight loses its storage.
            ensure(ctx_, std::min((size_t)ubatch(), ids.size()), 1, handoffs(1));
            const size_t B = (size_t)ubatch(), chunks = (ids.size() + B - 1) / B;
            auto chunk = [&](size_t c) {
                const size_t i = c * B, n = std::min(B, ids.size() - i);
                BatchEntry entry{&seq_, ids.data() + i, n, i + n == ids.size()};
                entry.extent = start + ids.size();
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
            for (auto& kv : seq_.kv_) kv.truncate(start);
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
        return seq_.length() * plan_.layers.size() * kv_bytes_per_position(plan_, options_);
    }

private:
    // One backend and what the placement put on it.
    // A pool is not movable, because sequences hold its address, so devices live behind pointers.
    struct Device {
        backend::BackendPtr b;
        bool used = false;
        bool sends = false;                      // the residual leaves it, so it keeps handoff buffers
        int attn_layers = 0;
        int storage_index = -1;
        std::vector<int> local_layer;            // model layer -> layer in storage
        std::unique_ptr<backend::KVStorage> storage;
        BlockPool pool;
        std::vector<backend::BufferPtr> tables;  // the position tables, on a device that runs attention
    };

    // Consecutive layers whose attention runs on one device, and every device a stage records work on, which it submits.
    struct Stage {
        size_t device;
        int first, end;
        std::vector<size_t> touches;
    };

    Placement place_;
    std::vector<std::unique_ptr<Device>> devices_;
    std::vector<Device*> storages_;              // the devices that run attention
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
        auto device_of = [&](Part part, size_t l) -> size_t {
            if (part == Part::embed) return (size_t)place_.embed_device;
            if (part == Part::head) return (size_t)place_.output_device;
            return (size_t)(part == Part::mixer ? place_.attn_device[l] : place_.ffn_device[l]);
        };
        pass_.assign(plan_.role_ids, Weight{});
        for (const Role& role : plan_.pass) pass_[role.id] = resolve(role, device_of(role.part, 0));
        const size_t n_layer = plan_.layers.size();
        home_.assign(n_layer, {});
        stream_.assign(n_layer, {});
        stream_device_.assign(n_layer, -1);
        for (size_t l = 0; l < n_layer; ++l) {
            const LayerPlan& layer = plan_.layers[l];
            std::vector<Weight>& row = home_[l];
            row.assign(plan_.role_ids, Weight{});
            for (const Role& role : layer.roles) row[role.id] = resolve(role, device_of(role.part, l));
            // Experts read in place on the host beside a mixer on a device that copies its weights.
            const size_t a = (size_t)place_.attn_device[l], f = (size_t)place_.ffn_device[l];
            if (!layer.routed || !place_.stream_from || a == f || devices_[a]->b->reads_in_place() || !devices_[f]->b->reads_in_place())
                continue;
            stream_device_[l] = (int)a;
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

    // The history a pass continues: the committed length of the first stage's storage, which a pipelined prompt's chunk commits first; outside a prompt every storage agrees.
    size_t history(const Sequence& s) const {
        return s.kv_[(size_t)devices_[stages_.front().device]->storage_index].length();
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
        if (!ctx.slots)
            ensure(ctx, rows, want, handoffs(1));
        else if (rows > ctx.pass_rows || want > ctx.logit_rows || logits_base > ctx.logit_rows - want)
            throw std::logic_error("inference: a pass beyond the rows or logits rows reserve_passes reserved");
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
        for (const BatchEntry& en : p.entries) {
            en.seq->kv_[storage].commit();
            for (size_t d : st.touches) en.seq->last_[d] = ctx.tickets[d];
        }
    }

    // After the last stage: where the logits are and the ticket that says they are ready, the pass's own head's.
    void finish(ExecContext& ctx, const Pass& p) {
        ctx.n_logits = p.want;
        ctx.backend = devices_[(size_t)place_.output_device]->b.get();
        ctx.ticket = p.sent;
        ctx.pending = p.want > 0;
    }

    // A failed pass: every device drained, then every entry's histories back to where the pass found them, the blocks its stages reserved or committed returned.
    void roll_back(const Pass& p) noexcept {
        retire();
        for (size_t e = 0; e < p.entries.size(); ++e)
            for (auto& kv : p.entries[e].seq->kv_) kv.truncate(p.start[e]);
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

    // Whether entry e of a pass takes a streamed layer on the device (Placement::stream_from): a prompt long enough, two tokens at the least, never a generated token.
    bool streams(const Pass& p, size_t e) const {
        return place_.stream_from && p.runs[e].extent >= std::max<size_t>(place_.stream_from, 2);
    }

    // A streamed layer in a pass with long runs: consecutive entries alike form a group, the long ones run where the residual is on the layer's streamed row, with each window role's bytes written into its window once, and the rest on the host through a crossing each way.
    // The residual ends where it started, on the layer's attention device.
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

    // Layer l's mixer over every row of the pass, with the cache views of its device's storage and the rows' positions.
    Step mixer_part(ExecContext& ctx, const Pass& p, size_t dev, int l) const {
        const Device& d = *devices_[dev];
        Step s = part(ctx, dev, home_[(size_t)l].data(), plan_.layers[(size_t)l].kind, 0, p.rows, {p.runs.data(), p.runs.size()});
        s.views = p.views[(size_t)d.storage_index].data();
        s.n_views = p.entries.size();
        s.kv_layer = (size_t)d.local_layer[(size_t)l];
        s.pos = p.pos.data();
        return s;
    }

    // A block returns to the pool only once the backend has retired every submission that touched it (docs/KV-CACHE.md).
    // Construction failures, failed passes and model teardown drain every used device with sync(), including work behind no ticket; reset() waits on tickets instead.
    void retire() noexcept {
        for (auto& d : devices_) if (d->used) d->b->sync();
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
    size_t slots = 0;                 // passes the caller keeps in flight (Model::reserve_passes), each with a handoff buffer on every stage but the last, two at least, which a split's fit counts
    // Histories the caller holds at once and the tokens each reaches, when it knows them, as bench does its sequences; zero leaves the options' budget as it is.
    // Each history takes whole blocks, up to the model's context, so the budget grows to hold them all where it would not.
    size_t histories = 0, history_tokens = 0;
};

// A placed model and, when it was split, what each device was given (LayerSplit::describe).
struct PlacedModel {
    std::unique_ptr<Model> model;
    std::string plan;
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

// The model over one backend, over one with the first `cpu_moe` routed layers' experts on the CPU beside it, or split by layers over several, with the request's ubatch set.
// The CPU is device 0 of an experts placement, so the thread count the model reports is the host's; attention, the dense blocks, the embedding and the head stay on the device.
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
    if (backends.size() > 1 || !request.shares.empty()) {
        if (request.cpu_moe)
            throw std::runtime_error(experts_flag + ": not with several devices; list the CPU as a device to give it layers");
        const std::vector<DeviceBudget> budgets = budgets_for(backends, request.names);
        const size_t rows = (size_t)(request.ubatch > 0 ? request.ubatch : kDefaultUbatch) + request.decode_rows;
        const LayerSplit split = split_layers(footprint(weights, plan, options), budgets, rows, request.shares, core::host_memory_available(), request.slots);
        placed = {std::make_unique<Model>(weights, plan, std::move(backends), placement_for(split), options, adopt), split.describe(budgets)};
    } else if (!adds_host_for_experts(backends, request)) {
        placed.model = std::make_unique<Model>(weights, plan, std::move(backends), Placement{}, options, adopt);
    } else {
        const size_t n_layer = plan.layers.size();
        Placement place;
        place.attn_device.assign(n_layer, 1);
        place.ffn_device.assign(n_layer, 1);
        place.embed_device = place.output_device = 1;
        place.stream_from = request.stream_from;
        int seen = 0;
        for (size_t l = 0; l < n_layer; ++l) {
            if (!plan.layers[l].routed) continue;
            if (request.cpu_moe < 0 || seen < request.cpu_moe) place.ffn_device[l] = 0;
            ++seen;
        }
        std::vector<backend::BackendPtr> both{backend::make_cpu_backend(), std::move(backends[0])};
        placed.model = std::make_unique<Model>(weights, plan, std::move(both), place, options, adopt);
    }
    placed.model->set_ubatch(request.ubatch);
    return placed;
}

} // namespace infer
