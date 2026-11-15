#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "format/gguf.hpp"
#include "backends/backend.hpp"
#include "model/weights.hpp"
#include "model/architecture.hpp"
#include "model/arch/metadata.hpp"
#include "model/arch/blocks.hpp"

// MiniMax-M3.1's text model (docs/MINIMAX.md): GQA attention with block-sparse selection by an index on every layer, dense lead layers, then sigmoid-routed experts with one shared expert.
// The registry reaches it as minimax-m3; its plan is refused by name until the CPU math of docs/MINIMAX.md, step 5, runs.

namespace infer::minimax_m3 {

struct Config {
    int n_layer = 0, n_dense = 0;
    int n_embd = 0, n_ff = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0, rope_dim = 0;
    int context_length = 0;
    float rope_theta = 0.0f, rms_eps = 0.0f;
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0;
    float expert_scale = 1.0f;
    // The index: one head per KV head, its key width, the blocks of `index_block` positions a query keeps with its own counted.
    int index_heads = 0, index_dim = 0, index_top_k = 0, index_block = 0;
};

// The rules a configuration keeps whichever file it came from.
inline void check_config(const Config& c) {
    if (c.n_dense >= c.n_layer)
        throw std::runtime_error("inference: minimax-m3 needs a routed layer after its dense ones");
    if (c.n_head % c.n_head_kv)
        throw std::runtime_error("inference: head count must be divisible by KV head count");
    if (c.head_dim % 2 || c.rope_dim % 2 || c.rope_dim > c.head_dim || c.rope_dim > c.index_dim || c.index_dim % 2)
        throw std::runtime_error("inference: invalid rotary width, which must be even and at most the head and index widths");
    if (c.n_head > std::numeric_limits<int>::max() / c.head_dim || c.index_heads > std::numeric_limits<int>::max() / c.index_dim)
        throw std::runtime_error("inference: invalid attention projection dimensions");
    if (c.n_expert_used > c.n_expert || c.n_expert_used > 256)
        throw std::runtime_error("inference: more experts per token than the layer has, or above 256");
    const uint64_t row = uint64_t(c.n_head_kv) * uint64_t(c.head_dim) * 2 + uint64_t(c.index_dim);
    if (uint64_t(c.context_length) > std::vector<float>().max_size() / row)
        throw std::runtime_error("inference: context storage exceeds allocation limit");
}

// A file's configuration under `p`; every key that changes the math is required and held to the one value the module computes.
inline Config read_config(const gguf::GGUFModel& m, const std::string& p) {
    using metadata::integer;
    using metadata::real;
    const auto unsupported = [&](bool bad, const std::string& k) {
        if (bad) throw std::runtime_error("inference: unsupported metadata " + p + k);
    };
    const auto required = [&](const std::string& k) {
        if (!m.find(p + k)) throw std::runtime_error("inference: missing metadata " + p + k);
    };
    metadata::option(m, p + "tensor_data_layout", "reference");
    metadata::option(m, p + "rope.scaling.type", "none");
    if (real(m, p + "rope.scaling.factor", 1) != 1)
        throw std::runtime_error("inference: scaled RoPE is unsupported");

    Config c;
    c.n_layer = integer(m, p + "block_count");
    required("leading_dense_block_count");
    c.n_dense = metadata::count(m, p + "leading_dense_block_count", 0);
    c.n_embd = integer(m, p + "embedding_length");
    c.n_ff = integer(m, p + "feed_forward_length");
    c.n_head = integer(m, p + "attention.head_count");
    c.n_head_kv = integer(m, p + "attention.head_count_kv");
    c.head_dim = integer(m, p + "attention.key_length");
    unsupported(integer(m, p + "attention.value_length") != c.head_dim, "attention.value_length");
    c.rope_dim = integer(m, p + "rope.dimension_count");
    c.context_length = integer(m, p + "context_length");
    required("rope.freq_base");
    c.rope_theta = float(real(m, p + "rope.freq_base", 1));
    required("attention.layer_norm_rms_epsilon");
    c.rms_eps = float(real(m, p + "attention.layer_norm_rms_epsilon", 1));

    c.n_expert = integer(m, p + "expert_count");
    c.n_expert_used = integer(m, p + "expert_used_count");
    c.n_ff_exp = integer(m, p + "expert_feed_forward_length");
    unsupported(integer(m, p + "expert_shared_count") != 1, "expert_shared_count");
    unsupported(integer(m, p + "expert_shared_feed_forward_length", c.n_ff_exp) != c.n_ff_exp, "expert_shared_feed_forward_length");
    required("expert_gating_func");
    metadata::choice(m, p + "expert_gating_func", 2, "inference: unsupported metadata " + p + "expert_gating_func");
    required("expert_weights_norm");
    unsupported(!metadata::boolean(m, p + "expert_weights_norm", true), "expert_weights_norm");
    required("expert_weights_scale");
    c.expert_scale = float(real(m, p + "expert_weights_scale", 1));

    c.index_heads = integer(m, p + "attention.indexer.head_count");
    unsupported(c.index_heads != c.n_head_kv, "attention.indexer.head_count");
    c.index_dim = integer(m, p + "attention.indexer.key_length");
    c.index_top_k = integer(m, p + "attention.indexer.top_k");
    c.index_block = integer(m, p + "attention.indexer.block_size");
    unsupported(integer(m, p + "attention.indexer.local_blocks") != 1, "attention.indexer.local_blocks");
    check_config(c);
    return c;
}

enum Role : uint16_t {
    token_embd, output, output_norm,
    attn_norm, attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm, attn_output,
    index_q, index_k, index_q_norm, index_k_norm,
    ffn_norm, ffn_gate, ffn_up, ffn_down,
    ffn_gate_inp, exp_probs_b, ffn_gate_exps, ffn_up_exps, ffn_down_exps, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp,
    role_count
};
enum Kind : uint8_t { dense, routed };

// Arena slots: 0 x, 1 h, 2 q, 3 k, 4 v, 5 attn, 6 gate, 7 up, 8 ffn, 9 router scores, 10 expert ids, 11 expert weights, 12 index queries, 13 index keys.
// The feed-forward slots hold a dense layer's hidden rows or a routed layer's k expert rows and the shared expert's after them, whichever is wider.
inline std::vector<size_t> slot_widths(const Config& c) {
    const size_t E = size_t(c.n_embd), Q = size_t(c.n_head) * size_t(c.head_dim), KV = size_t(c.n_head_kv) * size_t(c.head_dim);
    const size_t F = std::max(size_t(c.n_ff), size_t(c.n_expert_used + 1) * size_t(c.n_ff_exp)), k = size_t(c.n_expert_used);
    return {E, E, Q, KV, KV, Q, F, F, F, size_t(c.n_expert), k, k, size_t(c.index_heads) * size_t(c.index_dim), size_t(c.index_dim)};
}

class MinimaxM3 final : public Architecture {
public:
    explicit MinimaxM3(const Config& cfg) : cfg_(cfg) {}

    // The plan of a file's tensors, which plan() refuses by name until the module's math runs.
    ModelPlan layout(const TensorIndex& tensors) const {
        const Config& c = cfg_;
        const TensorView& embedding = tensors.view(tensors.at("token_embd.weight"));
        if (embedding.shape.size() < 2 || !embedding.shape[1] || embedding.shape[1] > uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("inference: invalid vocabulary dimension");
        const uint64_t E = uint64_t(c.n_embd), D = uint64_t(c.head_dim), Q = uint64_t(c.n_head) * D, KV = uint64_t(c.n_head_kv) * D;
        const uint64_t I = uint64_t(c.index_dim), IQ = uint64_t(c.index_heads) * I;
        ModelPlan p;
        p.role_ids = role_count;
        p.vocab = embedding.shape[1];
        p.pass = {{token_embd, Part::embed, RoleKind::gather, "token_embd.weight", "", E, p.vocab},
                  {output, Part::head, RoleKind::matrix, "output.weight", "", E, p.vocab},
                  {output_norm, Part::head, RoleKind::norm, "output_norm.weight", "", E}};
        p.layers.resize(size_t(c.n_layer));
        for (int l = 0; l < c.n_layer; ++l) {
            const std::string pre = "blk." + std::to_string(l) + ".";
            LayerPlan& layer = p.layers[size_t(l)];
            layer.routed = l >= c.n_dense;
            layer.kind = layer.routed ? routed : dense;
            for (const char* name : {layer.routed ? "ffn_gate.weight" : "ffn_gate_inp.weight", layer.routed ? "ffn_up.weight" : "ffn_gate_exps.weight"})
                if (tensors.find(pre + name)) throw std::runtime_error("inference: a tensor its layer's kind does not read " + pre + name);
            layer.cache = Cache::kv;
            layer.kv_aux = size_t(I);
            const Stream copy = layer.routed ? Stream::copy : Stream::none;
            layer.roles = {{attn_norm, Part::mixer, RoleKind::norm, pre + "attn_norm.weight", "", E},
                           {attn_q, Part::mixer, RoleKind::matrix, pre + "attn_q.weight", "", E, Q},
                           {attn_k, Part::mixer, RoleKind::matrix, pre + "attn_k.weight", "", E, KV},
                           {attn_v, Part::mixer, RoleKind::matrix, pre + "attn_v.weight", "", E, KV},
                           {attn_q_norm, Part::mixer, RoleKind::norm, pre + "attn_q_norm.weight", "", D},
                           {attn_k_norm, Part::mixer, RoleKind::norm, pre + "attn_k_norm.weight", "", D},
                           {attn_output, Part::mixer, RoleKind::matrix, pre + "attn_output.weight", "", Q, E},
                           {index_q, Part::mixer, RoleKind::matrix, pre + "indexer.q_proj.weight", "", E, IQ},
                           {index_k, Part::mixer, RoleKind::matrix, pre + "indexer.k_proj.weight", "", E, I},
                           {index_q_norm, Part::mixer, RoleKind::norm, pre + "indexer.q_norm.weight", "", I},
                           {index_k_norm, Part::mixer, RoleKind::norm, pre + "indexer.k_norm.weight", "", I},
                           {ffn_norm, Part::ffn, RoleKind::norm, pre + "ffn_norm.weight", "", E, 1, 0, copy}};
            layer.ops.push_back({Part::mixer, backend::Op::norm_rope_partial});
            layer.ops.push_back({Part::mixer, backend::Op::indexer_scores});
            layer.ops.push_back({Part::mixer, backend::Op::raw_index_scores});
            layer.ops.push_back({Part::mixer, backend::Op::candidate_blocks});
            layer.ops.push_back({Part::mixer, backend::Op::sparse_attention});
            layer.ops.push_back({Part::mixer, backend::Op::sparse_attention_blocks});
            layer.ops.push_back({Part::ffn, backend::Op::silu_mul_clamp});
            layer.ops.push_back({Part::ffn, backend::Op::glu_quick_gelu});
            if (layer.routed) {
                const uint64_t X = uint64_t(c.n_expert), F = uint64_t(c.n_ff_exp);
                layer.roles.insert(layer.roles.end(), {
                    {ffn_gate_inp, Part::ffn, RoleKind::matrix, pre + "ffn_gate_inp.weight", "", E, X, 0, Stream::copy},
                    {exp_probs_b, Part::ffn, RoleKind::norm, pre + "exp_probs_b.bias", "", X, 1, 0, Stream::copy},
                    {ffn_gate_exps, Part::ffn, RoleKind::experts, pre + "ffn_gate_exps.weight", "", E, F, X, Stream::window},
                    {ffn_up_exps, Part::ffn, RoleKind::experts, pre + "ffn_up_exps.weight", "", E, F, X, Stream::window},
                    {ffn_down_exps, Part::ffn, RoleKind::experts, pre + "ffn_down_exps.weight", "", F, E, X, Stream::window},
                    {ffn_gate_shexp, Part::ffn, RoleKind::matrix, pre + "ffn_gate_shexp.weight", "", E, F, 0, Stream::copy},
                    {ffn_up_shexp, Part::ffn, RoleKind::matrix, pre + "ffn_up_shexp.weight", "", E, F, 0, Stream::copy},
                    {ffn_down_shexp, Part::ffn, RoleKind::matrix, pre + "ffn_down_shexp.weight", "", F, E, 0, Stream::copy}});
                blocks::routed_ops(layer, tensors, ffn_gate_exps, ffn_up_exps);
                layer.ops.push_back({Part::ffn, backend::Op::sigmoid_routing});
            } else {
                const uint64_t F = uint64_t(c.n_ff);
                layer.roles.insert(layer.roles.end(), {
                    {ffn_gate, Part::ffn, RoleKind::matrix, pre + "ffn_gate.weight", "", E, F},
                    {ffn_up, Part::ffn, RoleKind::matrix, pre + "ffn_up.weight", "", E, F},
                    {ffn_down, Part::ffn, RoleKind::matrix, pre + "ffn_down.weight", "", F, E}});
            }
        }
        p.context_length = size_t(c.context_length);
        p.slots = slot_widths(c);
        p.residual = size_t(c.n_embd);
        p.kv_heads = size_t(c.n_head_kv);
        p.head_dim = size_t(c.head_dim);
        {
            backend::PassWidths& w = p.widths;
            // The dense block's widths count only where a dense layer exists; the shared expert is as wide as a routed one.
            const size_t e = size_t(c.n_embd), q = size_t(Q), f = c.n_dense > 0 ? size_t(c.n_ff) : 0, fe = size_t(c.n_ff_exp);
            w.heads = size_t(c.n_head);
            w.head_dim = size_t(c.head_dim);
            w.dense_attention = true;
            w.routed = size_t(c.n_expert_used);
            w.expert_in = size_t(c.n_ff_exp);
            w.expert_out = e;
            w.shared_ff = size_t(c.n_ff_exp);
            w.product_in = std::max({e, q, f, fe});
            w.product_out = std::max({q + 2 * size_t(KV), size_t(IQ + I), e, 2 * f, 2 * fe});
        }
        p.tables.assign(2, 0);
        p.table_steps.assign(2, size_t(c.rope_dim / 2));
        return p;
    }

    ModelPlan plan(const TensorIndex& tensors) const override {
        layout(tensors);
        not_yet();
    }

    // NeoX RoPE over the first rope_dim values of a head, its frequencies over that width: cos and sin at [pos * rope_dim / 2 + i].
    void fill_tables(std::vector<std::vector<float>>& tables, const std::vector<TensorView>&) const override {
        const size_t half = size_t(cfg_.rope_dim / 2);
        for (size_t i = 0; i < half; ++i) {
            const float freq = 1.0f / std::pow(cfg_.rope_theta, 2.0f * float(i) / float(cfg_.rope_dim));
            for (size_t pos = 0; pos < tables[0].size() / half; ++pos) {
                tables[0][pos * half + i] = std::cos(float(pos) * freq);
                tables[1][pos * half + i] = std::sin(float(pos) * freq);
            }
        }
    }

    void embed(const Step& s, const uint32_t* ids) const override { blocks::embed(s, s.w[token_embd], ids); }
    void mixer(const Step&) const override { not_yet(); }
    void ffn(const Step&) const override { not_yet(); }
    void head(const HeadStep& s) const override { blocks::head(s, s.w[output_norm], s.w[output], cfg_.rms_eps, s.slot(1)); }


private:
    [[noreturn]] static void not_yet() { throw std::runtime_error("inference: minimax-m3 does not run yet; its plan is in docs/MINIMAX.md"); }
    Config cfg_;
};

inline std::shared_ptr<const Architecture> open(const gguf::GGUFModel& m, const std::string& prefix) {
    return std::make_shared<const MinimaxM3>(read_config(m, prefix));
}

} // namespace infer::minimax_m3
