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
#include "model/rope.hpp"

// MiniMax-M3.1's text model (docs/MINIMAX.md): GQA attention with block-sparse selection by an index on every layer, dense lead layers, then sigmoid-routed experts with one shared expert.
// The registry reaches it as minimax-m3; a backend lacking one of its ops refuses the model at load by the op's name.

namespace infer::minimax_m3 {

struct Config {
    int n_layer = 0, n_dense = 0;
    int n_embd = 0, n_ff = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0, rope_dim = 0;
    int context_length = 0;
    float rope_theta = 0.0f, rms_eps = 0.0f;
    rope::Policy rope_policy;  // the rope the model runs (model/rope.hpp), from the file and the command line
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
inline Config read_config(const gguf::GGUFModel& m, const std::string& p, const rope::Request& request = {}) {
    using metadata::integer;
    using metadata::real;
    const auto unsupported = [&](bool bad, const std::string& k) {
        if (bad) throw std::runtime_error("inference: unsupported metadata " + p + k);
    };
    const auto required = [&](const std::string& k) {
        if (!m.find(p + k)) throw std::runtime_error("inference: missing metadata " + p + k);
    };
    metadata::option(m, p + "tensor_data_layout", "reference");

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
    c.rope_policy = rope::resolve(m, p, {rope::Base::reciprocal_power, c.rope_theta, size_t(c.rope_dim), size_t(c.context_length)}, request);
    // YaRN from the command line sets its own context, held to the same limits.
    if (c.rope_policy.context != size_t(c.context_length)) {
        c.context_length = int(c.rope_policy.context);
        check_config(c);
    }
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

// Arena slots after slot 0, the residual.
enum Slot : size_t { h_rows = 1, q_rows, k_rows, v_rows, attn_rows, gate_rows, up_rows, act_rows, router_scores, expert_ids, expert_weights, index_q_rows, index_k_rows, block_scores, block_lens, block_ids };

// The feed-forward slots hold a dense layer's hidden rows or a routed layer's k expert rows and the shared expert's after them, whichever is wider.
// A row's block scores are each index head's over the context's blocks, its block lengths one per index head, and its kept block ids top_k per index head.
inline std::vector<size_t> slot_widths(const Config& c) {
    const size_t E = size_t(c.n_embd), Q = size_t(c.n_head) * size_t(c.head_dim), KV = size_t(c.n_head_kv) * size_t(c.head_dim);
    const size_t F = std::max(size_t(c.n_ff), size_t(c.n_expert_used + 1) * size_t(c.n_ff_exp)), k = size_t(c.n_expert_used);
    const size_t IH = size_t(c.index_heads), blocks = (size_t(c.context_length) + size_t(c.index_block) - 1) / size_t(c.index_block);
    return {E, E, Q, KV, KV, Q, F, F, F, size_t(c.n_expert), k, k, IH * size_t(c.index_dim), size_t(c.index_dim), IH * blocks, IH, IH * size_t(c.index_top_k)};
}

class MinimaxM3 final : public Architecture {
public:
    explicit MinimaxM3(const Config& cfg) : cfg_(cfg) {
        cfg_.rope_policy = rope::bind(cfg_.rope_policy, {rope::Base::reciprocal_power, cfg_.rope_theta, size_t(cfg_.rope_dim), size_t(cfg_.context_length)});
    }

    ModelPlan plan(const TensorIndex& tensors) const override {
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
        p.index_width = size_t(c.index_heads);
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

    // NeoX RoPE over the first rope_dim values of a head, its frequencies over that width: cos and sin at [pos * rope_dim / 2 + i].
    void fill_tables(std::vector<std::vector<float>>& tables, const std::vector<TensorView>&) const override {
        rope::fill(cfg_.rope_policy, tables[0], tables[1]);
    }
    const rope::Policy& rope_policy() const override { return cfg_.rope_policy; }

    void embed(const Step& s, const uint32_t* ids) const override { blocks::embed(s, s.w[token_embd], ids); }

    // Each index head of a row keeps its blocks up to the row's own, whose count candidate_blocks reads as the row's length, the same in every layer.
    void pass_values(const Step& s) const override {
        const size_t IH = size_t(cfg_.index_heads), B = size_t(cfg_.index_block);
        if (!s.indices || s.indices_size < backend::size_mul(s.rows, IH)) throw std::runtime_error("inference: the minimax-m3 block lengths' scratch is too small");
        for (size_t r = 0; r < s.rows; ++r)
            for (size_t i = 0; i < IH; ++i) s.indices[r * IH + i] = uint32_t(size_t(s.pos[r]) / B + 1);
        s.b.write(*s.arena, s.offsets[block_lens], s.indices, s.rows * IH * sizeof(uint32_t));
    }

    // Attention over each KV head's kept blocks (docs/MINIMAX.md, The forward pass): the index keys stored beside K and V, each index head's block maxima and its top_k blocks with the row's own pinned, then GQA over their positions.
    void mixer(const Step& s) const override {
        backend::Backend& b = s.b;
        const Weight* w = s.w;
        const Config& c = cfg_;
        const size_t E = size_t(c.n_embd), D = size_t(c.head_dim), H = size_t(c.n_head), HK = size_t(c.n_head_kv), R = size_t(c.rope_dim);
        const size_t I = size_t(c.index_dim), IH = size_t(c.index_heads), B = size_t(c.index_block), K = size_t(c.index_top_k);
        const backend::CSlice cos{s.tables[0].get(), 0}, sin{s.tables[1].get(), 0};
        b.rms_norm_rows(s.slot(h_rows), s.x, w[attn_norm].slice(), s.rows, E, E, c.rms_eps, s.runs);
        b.matmul_group({blocks::projection(w[attn_q], s.slot(q_rows)), blocks::projection(w[attn_k], s.slot(k_rows)), blocks::projection(w[attn_v], s.slot(v_rows)),
                        blocks::projection(w[index_q], s.slot(index_q_rows)), blocks::projection(w[index_k], s.slot(index_k_rows))},
                       s.slot(h_rows), E, s.rows, s.runs, s.dtype);
        b.norm_rope_partial(s.slot(q_rows), s.slot(q_rows), s.rows, H * D, D, H, D, R, w[attn_q_norm].slice(), c.rms_eps, cos, sin, s.pos, backend::RopeForm{});
        b.norm_rope_partial(s.slot(k_rows), s.slot(k_rows), s.rows, HK * D, D, HK, D, R, w[attn_k_norm].slice(), c.rms_eps, cos, sin, s.pos, backend::RopeForm{});
        b.norm_rope_partial(s.slot(index_q_rows), s.slot(index_q_rows), s.rows, IH * I, I, IH, I, R, w[index_q_norm].slice(), c.rms_eps, cos, sin, s.pos,
                            backend::RopeForm{});
        b.norm_rope_partial(s.slot(index_k_rows), s.slot(index_k_rows), s.rows, I, I, 1, I, R, w[index_k_norm].slice(), c.rms_eps, cos, sin, s.pos,
                            backend::RopeForm{});
        b.kv_write(s.kv_layer, s.views, s.n_views, s.slot(k_rows), s.slot(v_rows), s.slot(index_k_rows));
        size_t n = 0;
        for (size_t e = 0; e < s.n_views; ++e) n = std::max(n, (s.views[e].length + s.views[e].nq + B - 1) / B);
        b.indexer_scores(s.slot(block_scores), s.slot(index_q_rows), s.kv_layer, s.views, s.n_views, {}, 1.0f, IH, I, backend::ModelFormat::f32, n, {}, 0, B,
                         backend::IndexForm::raw_blocks);
        b.candidate_blocks(s.slot(block_ids), s.slot(block_scores), s.slot(block_lens), s.rows * IH, n, 1, K);
        b.sparse_attention(s.slot(q_rows), s.kv_layer, s.views, s.n_views, s.slot(block_ids), K, s.slot(attn_rows), int(H), int(HK), int(D), IH, B);
        blocks::join(s, w[attn_output], s.slot(attn_rows));
    }

    // A dense layer's GLU block, or the sigmoid-routed experts and the shared expert, all in the quick_gelu form under the architecture's limit.
    void ffn(const Step& s) const override {
        const Weight* w = s.w;
        const size_t E = size_t(cfg_.n_embd);
        s.b.rms_norm_rows(s.slot(h_rows), s.x, w[ffn_norm].slice(), s.rows, E, E, cfg_.rms_eps, s.runs);
        if (s.kind == dense) {
            blocks::swiglu(s, w[ffn_gate], w[ffn_up], w[ffn_down], s.slot(h_rows), s.slot(gate_rows), s.slot(up_rows), s.slot(act_rows), glu_limit, backend::GluForm::quick_gelu);
            return;
        }
        const blocks::SharedExpert shared{w[ffn_gate_shexp], w[ffn_up_shexp], w[ffn_down_shexp], nullptr, glu_limit, backend::GluForm::quick_gelu};
        blocks::moe_step(s, w[ffn_gate_inp], w[ffn_gate_exps], w[ffn_up_exps], w[ffn_down_exps], size_t(cfg_.n_expert_used),
                         backend::RouteRule::sigmoid(w[exp_probs_b].slice(), cfg_.expert_scale, true), s.slot(h_rows), s.slot(gate_rows), s.slot(up_rows), s.slot(act_rows),
                         s.slot(router_scores), s.slot(expert_ids), s.slot(expert_weights), &shared, {}, nullptr, glu_limit, backend::GluForm::quick_gelu);
    }

    void head(const HeadStep& s) const override { blocks::head(s, s.w[output_norm], s.w[output], cfg_.rms_eps, s.slot(h_rows)); }

private:
    // The GLU's clamp, which the GGUF does not carry (docs/MINIMAX.md, The model).
    static constexpr float glu_limit = 7.0f;
    Config cfg_;
};

inline std::shared_ptr<const Architecture> open(const gguf::GGUFModel& m, const std::string& prefix, const rope::Request& request) {
    return std::make_shared<const MinimaxM3>(read_config(m, prefix, request));
}

} // namespace infer::minimax_m3
