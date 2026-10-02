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

// Qwen 3.5, 3.6 and 3.8, the qwen35 and qwen35moe architectures (docs/QWEN35.md), which the registry (model/arch/registry.hpp) reaches through open_dense and open_routed: linear-attention layers, a gated delta net carrying a recurrent state, with a gated full-attention layer at every interval, each followed by Qwen3's dense feed-forward block, or on qwen35moe by routed experts beside a gated shared expert.
// Weights are read as the converter stores them (docs/QWEN35.md, GGUF conventions): the norms but ssm_norm hold 1 + w, ssm_a holds -exp(A_log), attn_q holds each head's q and gate side by side, and V head j reads K head j mod Hk.

namespace infer::qwen35 {

struct Config {
    int n_layer = 0;              // decoder layers: the blocks less the MTP blocks after them, which the model does not run
    int n_nextn = 0;
    int n_embd = 0, n_ff = 0;
    int n_head = 0, n_head_kv = 0, head_dim = 0;
    int rope_dim = 0;             // the leading dims of each head that rotate
    int context_length = 0;
    float rope_theta = 10000.0f;
    float rms_eps = 1e-6f;
    int k_heads = 0, v_heads = 0, k_dim = 0, v_dim = 0;   // the linear attention's heads and widths
    std::vector<uint8_t> full;    // per decoder layer, whether it is a full-attention layer
    // qwen35moe: experts per layer, experts each token takes, a routed expert's and the shared expert's hidden widths, and whether the chosen probabilities are renormalized to sum to one.
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;
    bool expert_norm = true;
};

// The rules a configuration keeps whichever file it came from: whole query heads per KV head and V heads per K head, a rotary width that is even and inside a head, and projections and a cache that fit an allocation.
inline void check_config(const Config& c) {
    if (c.n_layer <= 0) throw std::runtime_error("inference: a model without decoder layers");
    if (c.n_head % c.n_head_kv) throw std::runtime_error("inference: head count must be divisible by KV head count");
    if (c.v_heads % c.k_heads) throw std::runtime_error("inference: linear-attention V heads must be a multiple of its K heads");
    if (c.rope_dim <= 0 || c.rope_dim % 2 || c.rope_dim > c.head_dim)
        throw std::runtime_error("inference: invalid rotary width, which must be even and at most the head width");
    if (c.n_expert_used > c.n_expert || c.n_expert_used > 256)
        throw std::runtime_error("inference: more experts per token than the layer has, or above 256");
    const uint64_t q = uint64_t(c.n_head) * 2 * uint64_t(c.head_dim), channels = 2 * uint64_t(c.k_heads) * uint64_t(c.k_dim);
    const uint64_t v = uint64_t(c.v_heads) * uint64_t(c.v_dim);
    if (q > uint64_t(std::numeric_limits<int>::max()) || channels + v > uint64_t(std::numeric_limits<int>::max()) ||
        uint64_t(c.k_dim) * uint64_t(c.v_dim) > uint64_t(std::numeric_limits<int>::max()))
        throw std::runtime_error("inference: invalid attention projection dimensions");
    const uint64_t kv_width = uint64_t(c.n_head_kv) * uint64_t(c.head_dim);
    const auto max_floats = std::vector<float>().max_size();
    if (uint64_t(c.context_length) > max_floats / kv_width || uint64_t(c.context_length) > max_floats / uint64_t(c.rope_dim))
        throw std::runtime_error("inference: context storage exceeds allocation limit");
}

// A qwen35 file's configuration, its keys read under `p`, the prefix the registry names; with `moe`, a qwen35moe file's, which adds the experts' keys and has no dense width.
inline Config read_config(const gguf::GGUFModel& m, const std::string& p, bool moe) {
    using metadata::integer;
    using metadata::real;
    Config c;
    metadata::option(m, p + "tensor_data_layout", "reference");
    metadata::option(m, p + "rope.scaling.type", "none");
    if (real(m, p + "rope.scaling.factor", 1) != 1 || real(m, p + "rope.scale_linear", 1) != 1)
        throw std::runtime_error("inference: scaled RoPE is unsupported");

    const int blocks = integer(m, p + "block_count");
    c.n_nextn = metadata::count(m, p + "nextn_predict_layers", 0);
    // An MTP block runs only as a drafter, which reads one.
    if (c.n_nextn > 1 || c.n_nextn >= blocks) throw std::runtime_error("inference: unsupported metadata " + p + "nextn_predict_layers");
    c.n_layer = blocks - c.n_nextn;
    c.n_embd = integer(m, p + "embedding_length");
    if (moe) {
        const metadata::Experts e = metadata::experts(m, p);
        c.n_expert = e.count, c.n_expert_used = e.used, c.n_ff_exp = e.ff, c.expert_norm = e.norm;
        c.n_ff_shexp = integer(m, p + "expert_shared_feed_forward_length");
    } else {
        c.n_ff = integer(m, p + "feed_forward_length");
    }
    c.n_head = integer(m, p + "attention.head_count");
    c.n_head_kv = integer(m, p + "attention.head_count_kv", c.n_head);
    c.head_dim = integer(m, p + "attention.key_length");
    if (integer(m, p + "attention.value_length", c.head_dim) != c.head_dim)
        throw std::runtime_error("inference: value width must equal key width");
    c.rope_dim = integer(m, p + "rope.dimension_count", c.head_dim);
    // For text every position stream holds the same position, so the sections give each rotated pair its plain rope angle; they must cover the rotated dims.
    const std::vector<int> sections = metadata::counts(m, p + "rope.dimension_sections");
    if (!sections.empty()) {
        int64_t sum = 0;
        for (int s : sections) sum += s;
        if (2 * sum != c.rope_dim) throw std::runtime_error("inference: " + p + "rope.dimension_sections does not cover the rotary width");
    }
    c.context_length = integer(m, p + "context_length");
    c.rope_theta = float(real(m, p + "rope.freq_base", c.rope_theta));
    c.rms_eps = float(real(m, p + "attention.layer_norm_rms_epsilon", c.rms_eps));

    if (integer(m, p + "ssm.conv_kernel") != int(backend::kConvTaps))
        throw std::runtime_error("inference: unsupported metadata " + p + "ssm.conv_kernel");
    c.k_dim = integer(m, p + "ssm.state_size");
    c.k_heads = integer(m, p + "ssm.group_count");
    c.v_heads = integer(m, p + "ssm.time_step_rank");
    const int inner = integer(m, p + "ssm.inner_size");
    if (inner % c.v_heads) throw std::runtime_error("inference: " + p + "ssm.inner_size is not a whole number of V heads");
    c.v_dim = inner / c.v_heads;

    // A layer's kind comes from the per-block array when a file has one, whose MTP blocks are full attention, else from the interval.
    const std::vector<bool> recurrent = metadata::booleans(m, p + "attention.recurrent_layers");
    c.full.resize((size_t)c.n_layer);
    if (m.find(p + "attention.recurrent_layers")) {
        if (recurrent.size() != (size_t)blocks) throw std::runtime_error("inference: unsupported metadata " + p + "attention.recurrent_layers");
        for (int l = c.n_layer; l < blocks; ++l)
            if (recurrent[(size_t)l]) throw std::runtime_error("inference: unsupported metadata " + p + "attention.recurrent_layers");
        for (int l = 0; l < c.n_layer; ++l) c.full[(size_t)l] = !recurrent[(size_t)l];
    } else {
        const int interval = integer(m, p + "full_attention_interval");
        for (int l = 0; l < c.n_layer; ++l) c.full[(size_t)l] = (l + 1) % interval == 0;
    }
    check_config(c);
    return c;
}

// qwen35's roles, by the id a row of resolved weights is indexed by, and its layer kinds.
enum Role : uint16_t {
    token_embd, output, output_norm,
    attn_norm, post_attention_norm,
    attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm, attn_output,
    attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_conv1d, ssm_a, ssm_dt, ssm_norm, ssm_out,
    ffn_gate, ffn_up, ffn_down,
    ffn_gate_inp, ffn_gate_exps, ffn_up_exps, ffn_down_exps, ffn_gate_inp_shexp, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp,
};
enum Kind : uint8_t { linear, full };

// The floats one row of a pass takes in each arena slot, the widest use either layer kind makes of it, since a layer uses only its own.
// Slots: 0 x, 1 h, 2 attn_q's rows or the raw qkv rows, 3 k or the conv's output, 4 v or z, 5 the normed and rotated q or alpha then beta, 6 the attention's or the recurrence's output, gated, 7 gate, 8 up, 9 ffn; on qwen35moe also 10 router scores, 11 expert ids, 12 expert weights, 13 the shared expert's gate.
// On qwen35moe the feed-forward slots hold a token's k routed expert rows or the shared expert's row, whichever is wider.
inline std::vector<size_t> slot_widths(const Config& c) {
    const size_t e = (size_t)c.n_embd, d = (size_t)c.head_dim;
    const size_t q = (size_t)c.n_head * d, kv = (size_t)c.n_head_kv * d;
    const size_t v = (size_t)c.v_heads * (size_t)c.v_dim, channels = 2 * (size_t)c.k_heads * (size_t)c.k_dim + v;
    const size_t k = (size_t)c.n_expert_used, ff = std::max({(size_t)c.n_ff, k * (size_t)c.n_ff_exp, (size_t)c.n_ff_shexp});
    std::vector<size_t> w = {e, e, std::max(2 * q, channels), std::max(kv, channels), std::max(kv, v), std::max(q, 2 * (size_t)c.v_heads), std::max(q, v), ff, ff, ff};
    if (c.n_expert) w.insert(w.end(), {(size_t)c.n_expert, k, k, 1});
    return w;
}

class Qwen35 final : public Architecture {
public:
    explicit Qwen35(const Config& cfg) : cfg_(cfg) {}

    // The embedding gives the vocabulary. A layer's roles are its norm, then its mixer's by its kind, then the feed-forward norm and the dense block, or on qwen35moe the router, the expert stacks and the shared expert with its gate; a full-attention layer keeps KV and a linear-attention layer a state.
    // A layer holding the other kind's tensors, or a router in a dense architecture, is refused, and blocks past the decoder layers are not read.
    // A routed layer run beside its mixer copies its norm, router and shared expert there and writes its stacks into a window.
    ModelPlan plan(const TensorIndex& tensors) const override {
        const Config& c = cfg_;
        const TensorView& embedding = tensors.view(tensors.at("token_embd.weight"));
        if (embedding.shape.size() < 2 || !embedding.shape[1] || embedding.shape[1] > uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("inference: invalid vocabulary dimension");
        ModelPlan p;
        p.role_ids = ffn_down_shexp + 1;
        p.vocab = embedding.shape[1];
        const uint64_t E = (uint64_t)c.n_embd, D = (uint64_t)c.head_dim, F = (uint64_t)c.n_ff;
        const uint64_t Q = (uint64_t)c.n_head * D, KV = (uint64_t)c.n_head_kv * D;
        const uint64_t Hv = (uint64_t)c.v_heads, V = Hv * (uint64_t)c.v_dim, C = 2 * (uint64_t)c.k_heads * (uint64_t)c.k_dim + V;
        p.pass = {{token_embd, Part::embed, RoleKind::gather, "token_embd.weight", "", E, p.vocab},
                  {output, Part::head, RoleKind::matrix, "output.weight", "token_embd.weight", E, p.vocab},
                  {output_norm, Part::head, RoleKind::norm, "output_norm.weight", "", E}};
        p.layers.resize((size_t)c.n_layer);
        for (int l = 0; l < c.n_layer; ++l) {
            const std::string pre = "blk." + std::to_string(l) + ".";
            LayerPlan& layer = p.layers[(size_t)l];
            if (!c.n_expert && tensors.find(pre + "ffn_gate_inp.weight")) throw std::runtime_error("inference: expert tensors in a dense architecture " + pre);
            layer.roles = {{attn_norm, Part::mixer, RoleKind::norm, pre + "attn_norm.weight", "", E}};
            if (c.full[(size_t)l]) {
                if (tensors.find(pre + "attn_qkv.weight"))
                    throw std::runtime_error("inference: a full-attention layer holds the linear-attention tensor " + pre + "attn_qkv.weight");
                layer.kind = full;
                layer.cache = Cache::kv;
                layer.roles.insert(layer.roles.end(), {{attn_q, Part::mixer, RoleKind::matrix, pre + "attn_q.weight", "", E, 2 * Q},
                                                       {attn_k, Part::mixer, RoleKind::matrix, pre + "attn_k.weight", "", E, KV},
                                                       {attn_v, Part::mixer, RoleKind::matrix, pre + "attn_v.weight", "", E, KV},
                                                       {attn_q_norm, Part::mixer, RoleKind::norm, pre + "attn_q_norm.weight", "", D},
                                                       {attn_k_norm, Part::mixer, RoleKind::norm, pre + "attn_k_norm.weight", "", D},
                                                       {attn_output, Part::mixer, RoleKind::matrix, pre + "attn_output.weight", "", Q, E}});
                layer.ops = {{Part::mixer, backend::Op::norm_rope_partial}, {Part::mixer, backend::Op::sigmoid_mul}};
            } else {
                if (tensors.find(pre + "attn_q.weight"))
                    throw std::runtime_error("inference: a linear-attention layer holds the full-attention tensor " + pre + "attn_q.weight");
                layer.kind = linear;
                layer.cache = Cache::state;
                layer.roles.insert(layer.roles.end(), {{attn_qkv, Part::mixer, RoleKind::matrix, pre + "attn_qkv.weight", "", E, C},
                                                       {attn_gate, Part::mixer, RoleKind::matrix, pre + "attn_gate.weight", "", E, V},
                                                       {ssm_alpha, Part::mixer, RoleKind::matrix, pre + "ssm_alpha.weight", "", E, Hv},
                                                       {ssm_beta, Part::mixer, RoleKind::matrix, pre + "ssm_beta.weight", "", E, Hv},
                                                       {ssm_conv1d, Part::mixer, RoleKind::table, pre + "ssm_conv1d.weight", "", backend::kConvTaps, C},
                                                       {ssm_a, Part::mixer, RoleKind::norm, pre + "ssm_a", "", Hv},
                                                       {ssm_dt, Part::mixer, RoleKind::norm, pre + "ssm_dt.bias", "", Hv},
                                                       {ssm_norm, Part::mixer, RoleKind::norm, pre + "ssm_norm.weight", "", (uint64_t)c.v_dim},
                                                       {ssm_out, Part::mixer, RoleKind::matrix, pre + "ssm_out.weight", "", V, E}});
                layer.ops = {{Part::mixer, backend::Op::causal_conv_silu}, {Part::mixer, backend::Op::gated_delta_rule},
                             {Part::mixer, backend::Op::gated_rms_norm}};
                layer.saved = {{2, (size_t)C}, {5, (size_t)Hv}, {5, (size_t)Hv, 1}};
            }
            if (c.n_expert) {
                const uint64_t X = (uint64_t)c.n_expert, Fe = (uint64_t)c.n_ff_exp, Fs = (uint64_t)c.n_ff_shexp;
                layer.routed = true;
                // The shared expert's gate is a vector of E weights, one dot product a row, checked as an F32 row.
                layer.roles.insert(layer.roles.end(), {{post_attention_norm, Part::ffn, RoleKind::norm, pre + "post_attention_norm.weight", "", E, 1, 0, Stream::copy},
                                                       {ffn_gate_inp, Part::ffn, RoleKind::matrix, pre + "ffn_gate_inp.weight", "", E, X, 0, Stream::copy},
                                                       {ffn_gate_exps, Part::ffn, RoleKind::experts, pre + "ffn_gate_exps.weight", "", E, Fe, X, Stream::window},
                                                       {ffn_up_exps, Part::ffn, RoleKind::experts, pre + "ffn_up_exps.weight", "", E, Fe, X, Stream::window},
                                                       {ffn_down_exps, Part::ffn, RoleKind::experts, pre + "ffn_down_exps.weight", "", Fe, E, X, Stream::window},
                                                       {ffn_gate_inp_shexp, Part::ffn, RoleKind::norm, pre + "ffn_gate_inp_shexp.weight", "", E, 1, 0, Stream::copy},
                                                       {ffn_gate_shexp, Part::ffn, RoleKind::matrix, pre + "ffn_gate_shexp.weight", "", E, Fs, 0, Stream::copy},
                                                       {ffn_up_shexp, Part::ffn, RoleKind::matrix, pre + "ffn_up_shexp.weight", "", E, Fs, 0, Stream::copy},
                                                       {ffn_down_shexp, Part::ffn, RoleKind::matrix, pre + "ffn_down_shexp.weight", "", Fs, E, 0, Stream::copy}});
                layer.ops.push_back({Part::ffn, backend::Op::sigmoid_mul});
                blocks::routed_ops(layer, tensors, ffn_gate_exps, ffn_up_exps);
            } else {
                layer.roles.insert(layer.roles.end(), {{post_attention_norm, Part::ffn, RoleKind::norm, pre + "post_attention_norm.weight", "", E},
                                                       {ffn_gate, Part::ffn, RoleKind::matrix, pre + "ffn_gate.weight", "", E, F},
                                                       {ffn_up, Part::ffn, RoleKind::matrix, pre + "ffn_up.weight", "", E, F},
                                                       {ffn_down, Part::ffn, RoleKind::matrix, pre + "ffn_down.weight", "", F, E}});
            }
        }
        p.context_length = (size_t)c.context_length;
        p.slots = slot_widths(c);
        p.residual = (size_t)c.n_embd;
        p.kv_heads = (size_t)c.n_head_kv;
        p.head_dim = (size_t)c.head_dim;
        p.state = backend::StateShape{(size_t)c.k_heads, (size_t)c.v_heads, (size_t)c.k_dim, (size_t)c.v_dim};
        p.tables.assign(2, (size_t)c.context_length * (size_t)(c.rope_dim / 2));
        return p;
    }

    // The rope's cos and sin for every position of the context over the rotated pairs only, indexed as [pos * (rope_dim / 2) + i].
    void fill_tables(std::vector<std::vector<float>>& tables) const override {
        const size_t half = (size_t)cfg_.rope_dim / 2;
        std::vector<float> freq(half);
        for (size_t i = 0; i < half; ++i) freq[i] = std::pow(cfg_.rope_theta, -2.0f * (float)i / (float)cfg_.rope_dim);
        for (size_t pos = 0; pos < (size_t)cfg_.context_length; ++pos)
            for (size_t i = 0; i < half; ++i) {
                tables[0][pos * half + i] = std::cos((float)pos * freq[i]);
                tables[1][pos * half + i] = std::sin((float)pos * freq[i]);
            }
    }

    void embed(const Step& s, const uint32_t* ids) const override { blocks::embed(s, s.w[token_embd], ids); }

    // The slots are those of slot_widths.
    void mixer(const Step& s) const override {
        backend::Backend& b = s.b;
        const Weight* w = s.w;
        const size_t E = (size_t)cfg_.n_embd;
        const backend::Slice x = s.x, h = s.slot(1);
        b.rms_norm_rows(h, x, w[attn_norm].slice(), s.rows, E, E, cfg_.rms_eps, s.runs);
        if (s.kind == full) {
            full_attention(s, h);
        } else {
            linear_attention(s, h);
        }
    }

    // The MoE FFN (docs/QWEN35.md): the routed sum joins the residual, then the shared expert's down projection reads its SwiGLU scaled by sigmoid(gate . h), one value a row, and joins it too.
    void ffn(const Step& s) const override {
        backend::Backend& b = s.b;
        const Weight* w = s.w;
        const size_t E = (size_t)cfg_.n_embd;
        const backend::Slice h = s.slot(1), g = s.slot(7), u = s.slot(8), act = s.slot(9);
        b.rms_norm_rows(h, s.x, w[post_attention_norm].slice(), s.rows, E, E, cfg_.rms_eps, s.runs);
        if (!cfg_.n_expert) {
            blocks::swiglu(s, w[ffn_gate], w[ffn_up], w[ffn_down], h, g, u, act);
            return;
        }
        blocks::routed_experts(s, w[ffn_gate_inp], w[ffn_gate_exps], w[ffn_up_exps], w[ffn_down_exps], (size_t)cfg_.n_expert_used, cfg_.expert_norm,
                               h, g, u, act, s.slot(10), s.slot(11), s.slot(12));
        const backend::Slice sg = s.slot(13);
        const Weight& gate = w[ffn_gate_inp_shexp];
        b.matmul(gate.type, gate.slice(), h, sg, E, 1, s.rows, s.runs, s.dtype);
        blocks::swiglu(s, w[ffn_gate_shexp], w[ffn_up_shexp], w[ffn_down_shexp], h, g, u, act, &sg);
    }

    void head(const HeadStep& s) const override { blocks::head(s, s.w[output_norm], s.w[output], cfg_.rms_eps, s.slot(1)); }

    // A linear-attention layer's state update: the causal conv over the raw rows (slot 2) into the conv's output (slot 3), and the recurrence over it with alpha and beta (slot 5) into the recurrence's output (slot 6), each reading the views' src slots and writing their dst slots.
    void recur(const Step& s) const override {
        backend::Backend& b = s.b;
        const Weight* w = s.w;
        const size_t Hv = (size_t)cfg_.v_heads;
        const backend::Slice raw = s.slot(2), u = s.slot(3), alpha = s.slot(5), o = s.slot(6);
        const backend::Slice beta{alpha.buffer, alpha.offset + s.rows * Hv};
        b.causal_conv_silu(u, raw, w[ssm_conv1d].slice(), s.state_layer, s.states, s.n_views);
        b.gated_delta_rule(o, u, alpha, beta, w[ssm_a].slice(), w[ssm_dt].slice(), s.state_layer, s.states, s.n_views);
    }

private:
    Config cfg_;

    // Gated attention (docs/QWEN35.md): attn_q gives each head's q and gate side by side; q and k are normed per head and their leading rope_dim dims rotated, the KV cache takes k and v, and the attention's output is gated by sigmoid(gate) before the output projection joins the residual.
    void full_attention(const Step& s, backend::Slice h) const {
        backend::Backend& b = s.b;
        const Weight* w = s.w;
        const size_t E = (size_t)cfg_.n_embd, D = (size_t)cfg_.head_dim, Hq = (size_t)cfg_.n_head, Hkv = (size_t)cfg_.n_head_kv;
        const backend::Slice r = s.slot(2), k = s.slot(3), v = s.slot(4), q = s.slot(5), o = s.slot(6);
        b.matmul_group({blocks::projection(w[attn_q], r), blocks::projection(w[attn_k], k), blocks::projection(w[attn_v], v)}, h, E, s.rows, s.runs, s.dtype);
        const backend::CSlice cos{s.tables[0].get(), 0}, sin{s.tables[1].get(), 0};
        const size_t R = (size_t)cfg_.rope_dim;
        b.norm_rope_partial(q, r, s.rows, 2 * Hq * D, 2 * D, Hq, D, R, w[attn_q_norm].slice(), cfg_.rms_eps, cos, sin, s.pos);
        b.norm_rope_partial(k, k, s.rows, Hkv * D, D, Hkv, D, R, w[attn_k_norm].slice(), cfg_.rms_eps, cos, sin, s.pos);
        b.kv_write(s.kv_layer, s.views, s.n_views, k, v);
        b.attention(q, s.kv_layer, s.views, s.n_views, o, (int)Hq, (int)Hkv, (int)D);
        b.sigmoid_mul(o, o, backend::CSlice{r.buffer, r.offset + D}, s.rows, Hq, D, 2 * Hq * D, 2 * D, s.runs);
        b.matmul_add(w[attn_output].type, w[attn_output].slice(), o, s.x, w[attn_output].nin, w[attn_output].nout, s.rows, s.runs, s.dtype);
    }

    // The gated delta net (docs/QWEN35.md, Linear attention): the raw q, k and v rows, z, alpha and beta; the state's update (recur), the causal conv over the raw rows and the state's carried ones and the recurrence from the sequence's state; the gated norm by z; and the output projection joining the residual.
    void linear_attention(const Step& s, backend::Slice h) const {
        backend::Backend& b = s.b;
        const Weight* w = s.w;
        const size_t E = (size_t)cfg_.n_embd, Hv = (size_t)cfg_.v_heads;
        const backend::Slice raw = s.slot(2), z = s.slot(4), alpha = s.slot(5), o = s.slot(6);
        const backend::Slice beta{alpha.buffer, alpha.offset + s.rows * Hv};
        b.matmul(w[attn_qkv].type, w[attn_qkv].slice(), h, raw, E, w[attn_qkv].nout, s.rows, s.runs, s.dtype);
        b.matmul_group({blocks::projection(w[attn_gate], z), blocks::projection(w[ssm_alpha], alpha), blocks::projection(w[ssm_beta], beta)},
                       h, E, s.rows, s.runs, s.dtype);
        recur(s);
        b.gated_rms_norm(o, o, z, w[ssm_norm].slice(), s.rows, Hv, (size_t)cfg_.v_dim, cfg_.rms_eps, s.runs);
        b.matmul_add(w[ssm_out].type, w[ssm_out].slice(), o, s.x, w[ssm_out].nin, w[ssm_out].nout, s.rows, s.runs, s.dtype);
    }
};

// The registry's readers: a qwen35 file's architecture, and a qwen35moe file's.
inline std::shared_ptr<const Architecture> open_dense(const gguf::GGUFModel& m, const std::string& prefix) {
    return std::make_shared<const Qwen35>(read_config(m, prefix, false));
}
inline std::shared_ptr<const Architecture> open_routed(const gguf::GGUFModel& m, const std::string& prefix) {
    return std::make_shared<const Qwen35>(read_config(m, prefix, true));
}

} // namespace infer::qwen35
