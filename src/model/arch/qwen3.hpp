#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "format/gguf.hpp"
#include "quant/quant.hpp"
#include "backends/backend.hpp"
#include "model/weights.hpp"
#include "model/architecture.hpp"
#include "model/arch/metadata.hpp"
#include "model/arch/blocks.hpp"

// Qwen3's architecture, from scratch: dense Qwen3 and its mixture-of-experts form, qwen3moe, which the registry (model/arch/registry.hpp) reaches through open_dense and open_routed.
// Dense matrices use supported block quants or F32; normalization weights are F32.
// Tensor ne[0] is the input dimension, with each output row contiguous.
// Attention projection width is n_head*head_dim and need not equal n_embd.
// An absent output.weight ties the output projection to token_embd.weight.

namespace infer::qwen3 {

struct Config {
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
    int n_expert = 0;
    int n_expert_used = 0;
    int n_ff_exp = 0;
    bool expert_norm = true;
};

// The rules a configuration keeps whichever file it came from: whole query heads for each KV head, an even head width, projections and a cache that fit an allocation, and at most 256 experts per token, no more than a layer has.
inline void check_config(const Config& c) {
    if (c.n_head % c.n_head_kv)
        throw std::runtime_error("inference: head count must be divisible by KV head count");
    if (c.head_dim <= 0 || c.head_dim % 2 ||
        c.n_head > std::numeric_limits<int>::max() / c.head_dim)
        throw std::runtime_error("inference: invalid attention projection dimensions");
    if (c.n_expert_used > c.n_expert || c.n_expert_used > 256)
        throw std::runtime_error("inference: more experts per token than the layer has, or above 256");
    const uint64_t kv_width = uint64_t(c.n_head_kv) * c.head_dim;
    const auto max_floats = std::vector<float>().max_size();
    if (uint64_t(c.context_length) > max_floats / kv_width ||
        uint64_t(c.context_length) > max_floats / uint64_t(c.n_head))
        throw std::runtime_error("inference: context storage exceeds allocation limit");
}

// A GGUF file's configuration, its keys read under `p`, the prefix the registry names: qwen3's, or qwen3moe's with `moe`, which adds the experts' keys and lets a file without the dense width have no dense layer.
inline Config read_config(const gguf::GGUFModel& m, const std::string& p, bool moe) {
    using metadata::integer;
    using metadata::option;
    using metadata::real;
    Config c;
    option(m, p + "tensor_data_layout", "reference");
    option(m, p + "rope.scaling.type", "none");
    if (real(m, p + "rope.scaling.factor", 1) != 1 ||
        real(m, p + "rope.scale_linear", 1) != 1)
        throw std::runtime_error("inference: scaled RoPE is unsupported");

    c.n_layer = integer(m, p + "block_count");
    c.n_embd = integer(m, p + "embedding_length");
    // A mixture-of-experts file needs the dense width only for its dense layers, if it has any.
    c.n_ff = moe && !m.find(p + "feed_forward_length") ? 0 : integer(m, p + "feed_forward_length");
    c.n_head = integer(m, p + "attention.head_count");
    c.n_head_kv = integer(m, p + "attention.head_count_kv", c.n_head);
    if (m.find(p + "attention.key_length")) {
        c.head_dim = integer(m, p + "attention.key_length");
    } else {
        if (c.n_embd % c.n_head)
            throw std::runtime_error("inference: embedding width does not determine an integral head width");
        c.head_dim = c.n_embd / c.n_head;
    }
    if (integer(m, p + "attention.value_length", c.head_dim) != c.head_dim ||
        integer(m, p + "rope.dimension_count", c.head_dim) != c.head_dim)
        throw std::runtime_error("inference: value and rotary widths must equal key width");
    c.context_length = integer(m, p + "context_length", c.context_length);
    c.rope_theta = float(real(m, p + "rope.freq_base", c.rope_theta));
    c.rms_eps = float(real(m, p + "attention.layer_norm_rms_epsilon", c.rms_eps));
    if (moe) {
        const metadata::Experts e = metadata::experts(m, p);
        c.n_expert = e.count, c.n_expert_used = e.used, c.n_ff_exp = e.ff, c.expert_norm = e.norm;
        // A shared expert is another architecture's.
        if (m.find(p + "expert_shared_count") || m.find(p + "expert_shared_feed_forward_length"))
            throw std::runtime_error("inference: shared experts are unsupported");
    }
    check_config(c);
    return c;
}

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
inline std::vector<size_t> slot_widths(const Config& cfg, bool dense) {
    const size_t q = (size_t)cfg.n_head * cfg.head_dim, kv = (size_t)cfg.n_head_kv * cfg.head_dim;
    const size_t ff = std::max(dense ? (size_t)cfg.n_ff : 0, (size_t)cfg.n_expert_used * (size_t)cfg.n_ff_exp);
    const size_t e = (size_t)cfg.n_embd, k = (size_t)cfg.n_expert_used;
    return {e, e, q, kv, kv, q, ff, ff, ff, (size_t)cfg.n_expert, k, k};
}

// Qwen3 and qwen3moe under one configuration: the plan of a file's tensors, the rope tables, and the embedding, attention, feed-forward block and head as backend ops.
class Qwen3 final : public Architecture {
public:
    explicit Qwen3(const Config& cfg) : cfg_(cfg) {}

    // The embedding gives the vocabulary, and a layer with a router is routed, which a dense architecture refuses, as a dense layer is refused without the dense width.
    // A layer's roles are its attention's, its feed-forward norm, then its router and expert stacks or its three dense matrices; a routed layer copies its norm and router beside its mixer and writes its stacks into a window.
    // The arena's feed-forward slots are as wide as a dense layer's when some layer is dense, and the two rope tables cover the context at half a head each.
    ModelPlan plan(const TensorIndex& tensors) const override {
        const Config& cfg = cfg_;
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
                blocks::routed_ops(layer, tensors, ffn_gate_exps, ffn_up_exps);
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

    void embed(const Step& s, const uint32_t* ids) const override { blocks::embed(s, s.w[token_embd], ids); }

    // The slots are those of slot_widths.
    void mixer(const Step& s) const override {
        backend::Backend& b = s.b;
        const Weight* w = s.w;
        const size_t E = (size_t)cfg_.n_embd, half = (size_t)cfg_.head_dim / 2;
        const size_t Q = (size_t)cfg_.n_head * cfg_.head_dim, KV = (size_t)cfg_.n_head_kv * cfg_.head_dim;
        const backend::Slice x = s.x, h = s.slot(1), q = s.slot(2), k = s.slot(3), v = s.slot(4), attn = s.slot(5);

        b.rms_norm_rows(h, x, w[attn_norm].slice(), s.rows, E, E, cfg_.rms_eps, s.runs);

        b.matmul_group({blocks::projection(w[attn_q], q),
                        blocks::projection(w[attn_k], k),
                        blocks::projection(w[attn_v], v)}, h, E, s.rows, s.runs, s.dtype);

        const backend::Backend::RopeArgs rope{{s.tables[0].get(), 0}, {s.tables[1].get(), 0},
                                              half, s.pos, cfg_.rms_eps};
        b.norm_rope_kv(q, Q, cfg_.n_head, w[attn_q_norm].slice(),
                       k, v, KV, cfg_.n_head_kv, w[attn_k_norm].slice(),
                       rope, s.rows, s.kv_layer, s.views, s.n_views);
        b.attention(q, s.kv_layer, s.views, s.n_views, attn,
                    cfg_.n_head, cfg_.n_head_kv, cfg_.head_dim);

        b.matmul_add(w[attn_output].type, w[attn_output].slice(), attn, x,
                     w[attn_output].nin, w[attn_output].nout, s.rows, s.runs, s.dtype);
    }

    // The rows of the residual from s.x, through the scratch slots from their start, reading whichever row of weights the runtime hands it: the layer's own, or on its mixer device a streamed layer's copies and windows.
    void ffn(const Step& s) const override {
        backend::Backend& b = s.b;
        const Weight* w = s.w;
        const size_t E = (size_t)cfg_.n_embd;
        const backend::Slice x = s.x, h = s.slot(1), gate = s.slot(6), up = s.slot(7), ffn = s.slot(8);

        b.rms_norm_rows(h, x, w[ffn_norm].slice(), s.rows, E, E, cfg_.rms_eps, s.runs);

        if (s.kind == routed) {
            blocks::routed_experts(s, w[ffn_gate_inp], w[ffn_gate_exps], w[ffn_up_exps], w[ffn_down_exps], (size_t)cfg_.n_expert_used, cfg_.expert_norm,
                                   h, gate, up, ffn, s.slot(9), s.slot(10), s.slot(11));
            return;
        }
        blocks::swiglu(s, w[ffn_gate], w[ffn_up], w[ffn_down], h, gate, up, ffn);
    }

    void head(const HeadStep& s) const override { blocks::head(s, s.w[output_norm], s.w[output], cfg_.rms_eps, s.slot(1)); }

private:
    Config cfg_;
};

// The registry's readers: a qwen3 file's architecture, and a qwen3moe file's.
inline std::shared_ptr<const Architecture> open_dense(const gguf::GGUFModel& m, const std::string& prefix) {
    return std::make_shared<const Qwen3>(read_config(m, prefix, false));
}
inline std::shared_ptr<const Architecture> open_routed(const gguf::GGUFModel& m, const std::string& prefix) {
    return std::make_shared<const Qwen3>(read_config(m, prefix, true));
}

// A model of this architecture with the given shape and random weights, Q8_0 matrices and F32 norms, for timing the backend without a file (bench without --model).
// It is written as a file of the dense entry `name`, the registry's, which names general.architecture and prefixes every key.
inline gguf::GGUFModel synthetic_model(const std::string& name, int n_layer, int n_embd, int n_ff, int n_head, int n_head_kv, int head_dim, int n_vocab, uint32_t seed) {
    gguf::GGUFModel m;
    gguf::MetaValue arch; arch.vtype = gguf::V_STRING; arch.s = name;
    m.kv.emplace_back("general.architecture", arch);
    auto u32 = [&](const std::string& k, uint64_t v) {
        gguf::MetaValue mv; mv.vtype = gguf::V_UINT32; mv.u = v;
        m.kv.emplace_back(name + "." + k, mv);
    };
    u32("block_count", (uint64_t)n_layer);
    u32("embedding_length", (uint64_t)n_embd);
    u32("feed_forward_length", (uint64_t)n_ff);
    u32("attention.head_count", (uint64_t)n_head);
    u32("attention.head_count_kv", (uint64_t)n_head_kv);
    u32("attention.key_length", (uint64_t)head_dim);
    u32("context_length", 2048);

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

} // namespace infer::qwen3
