#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "format/gguf.hpp"
#include "backends/backend.hpp"
#include "model/weights.hpp"
#include "model/architecture.hpp"
#include "model/arch/metadata.hpp"
#include "model/rope.hpp"
#include "model/arch/blocks.hpp"
#include "model/arch/dflash.hpp"
#include "model/arch/hf_config.hpp"

// Qwen 3.5, 3.6 and 3.8 (docs/QWEN35.md): linear-attention layers alternate with a gated full-attention layer at every interval, each followed by a dense feed-forward block or, on qwen35moe, routed experts and a gated shared expert.
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
    rope::Policy rope_policy;     // the rope the model runs (model/rope.hpp), from the file and the command line
    float rms_eps = 1e-6f;
    int k_heads = 0, v_heads = 0, k_dim = 0, v_dim = 0;   // the linear attention's heads and widths
    std::vector<uint8_t> full;    // per decoder layer, whether it is a full-attention layer
    std::vector<uint8_t> image_streams;   // per rotated pair, the position an image row turns it by (ModelPlan::image_streams); empty without rope sections
    // qwen35moe: experts per layer, experts each token takes, a routed expert's and the shared expert's hidden widths, and whether the chosen probabilities are renormalized to sum to one.
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;
    bool expert_norm = true;
    // A DFlash sidecar joined to the file (model/arch/dflash.hpp), which drafts in place of its MTP block.
    std::optional<dflash::Config> dflash;
    rope::Policy draft_rope;   // its rope over its whole head, YaRN where --yarn asks it of the target
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
inline Config read_config(const gguf::GGUFModel& m, const std::string& p, bool moe, const rope::Request& request = {}) {
    using metadata::integer;
    using metadata::real;
    Config c;
    metadata::option(m, p + "tensor_data_layout", "reference");

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
        if (sections.size() >= 3) c.image_streams = blocks::interleaved_streams(sections, (size_t)c.rope_dim / 2);
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
    if (m.find(p + "dflash.block_count")) c.dflash = dflash::read_config(m, p);
    c.rope_policy = rope::resolve(m, p, {rope::Base::negative_power, c.rope_theta, size_t(c.rope_dim), size_t(c.context_length)}, request);
    // --yarn scales a DFlash sidecar's rope by the target's factor over the target's original length, at most its context, so both cover the one extended context.
    if (c.dflash) {
        rope::Request draft;
        if (request.factor) draft = {request.factor, std::min(c.rope_policy.original, size_t(c.context_length))};
        c.draft_rope = rope::resolve(m, p + "dflash.", {rope::Base::negative_power, c.dflash->rope_theta, size_t(c.dflash->head_dim), size_t(c.context_length)}, draft);
    }
    // YaRN from the command line sets its own context, held to the same limits.
    if (c.rope_policy.context != size_t(c.context_length)) {
        c.context_length = int(c.rope_policy.context);
        check_config(c);
    }
    return c;
}

// The keys of a directory's text_config no single value gives (hf::Table::derive): the blocks with the MTP layer, the V side's width, the layer kinds and the rotary keys.
inline void hf_derive(const jmini::Value& c, hf::Meta& kv, const std::string& p) {
    const auto whole = [&](const jmini::Value* v, const std::string& key) -> uint64_t {
        if (!v || !v->isNumber() || !v->whole || v->num < 0 || v->num > 4294967295.0) throw std::runtime_error("config.json " + key + " is not a whole number");
        return uint64_t(v->num);
    };
    const auto uint = [&](const std::string& key, uint64_t value) {
        gguf::MetaValue v;
        v.vtype = gguf::V_UINT32;
        v.u = value;
        kv.emplace_back(p + key, v);
    };
    const uint64_t layers = whole(c.get("num_hidden_layers"), "num_hidden_layers");
    const jmini::Value* mtp_key = c.get("mtp_num_hidden_layers");
    const uint64_t mtp = mtp_key ? whole(mtp_key, "mtp_num_hidden_layers") : 0;
    if (!layers || mtp > 1) throw std::runtime_error("config.json num_hidden_layers or mtp_num_hidden_layers is not one llmx reads (an MTP layer at most)");
    uint("block_count", layers + mtp);
    if (mtp) uint("nextn_predict_layers", mtp);
    uint("ssm.inner_size", whole(c.get("linear_value_head_dim"), "linear_value_head_dim") * whole(c.get("linear_num_value_heads"), "linear_num_value_heads"));
    // The layer kinds follow the interval, as read_config reads them from it; without full_attention_interval, the first full-attention layer gives it.
    const jmini::Value* kinds = c.get("layer_types");
    if (!kinds || !kinds->isArray() || kinds->arr.size() != layers) throw std::runtime_error("config.json layer_types does not give every layer");
    uint64_t interval = 0;
    if (const jmini::Value* given = c.get("full_attention_interval")) interval = whole(given, "full_attention_interval");
    for (uint64_t l = 0; !interval && l < layers; ++l)
        if (kinds->arr[l].isString() && kinds->arr[l].str == "full_attention") interval = l + 1;
    if (!interval) throw std::runtime_error("config.json layer_types holds no full-attention layer");
    for (uint64_t l = 0; l < layers; ++l)
        if (!kinds->arr[l].isString() || kinds->arr[l].str != ((l + 1) % interval == 0 ? "full_attention" : "linear_attention"))
            throw std::runtime_error("config.json layer_types does not follow full_attention_interval");
    uint("full_attention_interval", interval);
    // rope_parameters: the interleaved sections of the default rope over the leading partial_rotary_factor of each head.
    const jmini::Value* rope = c.get("rope_parameters");
    if (!rope || !rope->isObject()) throw std::runtime_error("config.json gives no rope_parameters");
    for (const auto& [key, value] : rope->obj) {
        const std::string text = hf::detail::text(value);
        if ((key == "rope_type" && text == "\"default\"") || (key == "mrope_interleaved" && text == "true") || key == "rope_theta" || key == "partial_rotary_factor" ||
            key == "mrope_section")
            continue;
        throw std::runtime_error("config.json rope_parameters " + key + " = " + text + " is not supported");
    }
    const jmini::Value* theta = rope->get("rope_theta");
    const jmini::Value* factor = rope->get("partial_rotary_factor");
    const jmini::Value* sections = rope->get("mrope_section");
    if (!theta || !factor || !factor->isNumber() || !sections || !sections->isArray() || sections->arr.size() != 3 || !rope->get("mrope_interleaved"))
        throw std::runtime_error("config.json rope_parameters lacks rope_theta, partial_rotary_factor, mrope_section or mrope_interleaved");
    // Qwen 3.6 repeats the factor beside rope_parameters, which HF's rope reads.
    if (const jmini::Value* outer = c.get("partial_rotary_factor"); outer && (!outer->isNumber() || outer->num != factor->num))
        throw std::runtime_error("config.json partial_rotary_factor differs from rope_parameters'");
    kv.emplace_back(p + "rope.freq_base", hf::detail::meta(*theta, hf::Rule::real, "rope_parameters rope_theta"));
    const double rotated = double(whole(c.get("head_dim"), "head_dim")) * factor->num;
    if (!(rotated > 0) || rotated != double(uint64_t(rotated))) throw std::runtime_error("config.json partial_rotary_factor does not give whole rotated dims");
    uint("rope.dimension_count", uint64_t(rotated));
    // The converter writes four sections, the fourth 0.
    gguf::MetaValue list;
    list.vtype = gguf::V_ARRAY;
    list.u = gguf::V_INT32;
    for (size_t k = 0; k < 4; ++k) {
        gguf::MetaValue e;
        e.vtype = gguf::V_INT32;
        e.i = k < 3 ? int64_t(whole(&sections->arr[k], "rope_parameters mrope_section")) : 0;
        list.arr.push_back(e);
    }
    kv.emplace_back(p + "rope.dimension_sections", list);
}

// The table of a Qwen 3.5 or 3.6 dense directory (hf::Table), whose text model's keys sit in text_config; the vision tower is dropped, since images come from an mmproj file.
inline const hf::Table& hf_table() {
    using hf::Rule;
    using hf::VHeads;
    static const hf::Table table = [] {
        hf::Table t{"qwen3_5", "Qwen3_5ForConditionalGeneration",
            {{"hidden_size", "embedding_length", Rule::integer, nullptr},
             {"intermediate_size", "feed_forward_length", Rule::integer, nullptr},
             {"max_position_embeddings", "context_length", Rule::integer, nullptr},
             {"num_attention_heads", "attention.head_count", Rule::integer, nullptr},
             {"num_key_value_heads", "attention.head_count_kv", Rule::integer, nullptr},
             {"head_dim", "attention.key_length", Rule::integer, nullptr},
             {"head_dim", "attention.value_length", Rule::integer, nullptr},
             {"rms_norm_eps", "attention.layer_norm_rms_epsilon", Rule::real, nullptr},
             {"linear_conv_kernel_dim", "ssm.conv_kernel", Rule::integer, nullptr},
             {"linear_key_head_dim", "ssm.state_size", Rule::integer, nullptr},
             {"linear_num_key_heads", "ssm.group_count", Rule::integer, nullptr},
             {"linear_num_value_heads", "ssm.time_step_rank", Rule::integer, nullptr},
             {"model_type", nullptr, Rule::check, "\"qwen3_5_text\""},
             {"attention_bias", nullptr, Rule::check, "false"},
             {"attn_output_gate", nullptr, Rule::check, "true"},
             {"output_gate_type", nullptr, Rule::check, "\"swish\""},
             {"hidden_act", nullptr, Rule::check, "\"silu\""},
             {"mamba_ssm_dtype", nullptr, Rule::check, "\"float32\""},
             {"mtp_use_dedicated_embeddings", nullptr, Rule::check, "false"},
             {"tie_word_embeddings", nullptr, Rule::check, "true|false"},
             {"dtype", nullptr, Rule::check, "\"bfloat16\"|\"float16\"|\"float32\""},
             // Read by hf_derive.
             {"num_hidden_layers", nullptr, Rule::check, nullptr},
             {"mtp_num_hidden_layers", nullptr, Rule::check, nullptr},
             {"linear_value_head_dim", nullptr, Rule::check, nullptr},
             {"full_attention_interval", nullptr, Rule::check, nullptr},
             {"layer_types", nullptr, Rule::check, nullptr},
             {"rope_parameters", nullptr, Rule::check, nullptr},
             {"partial_rotary_factor", nullptr, Rule::check, nullptr},
             // A dense model has no expert layers to leave out; the rest are read elsewhere (vocab_size against the embedding, the token ids beside the tokenizer's) or have no effect on inference.
             {"mlp_only_layers", nullptr, Rule::check, nullptr},
             {"vocab_size", nullptr, Rule::check, nullptr},
             {"bos_token_id", nullptr, Rule::check, nullptr},
             {"eos_token_id", nullptr, Rule::check, nullptr},
             {"pad_token_id", nullptr, Rule::check, nullptr},
             {"attention_dropout", nullptr, Rule::check, nullptr},
             {"initializer_range", nullptr, Rule::check, nullptr},
             {"use_cache", nullptr, Rule::check, nullptr}},
            {},
            "token_embd.weight", "output.weight"};
        t.config = "text_config";
        t.outer = {{"tie_word_embeddings", nullptr, Rule::check, "true|false"},
                   {"transformers_version", nullptr, Rule::check, nullptr},
                   {"vision_config", nullptr, Rule::check, nullptr},
                   {"image_token_id", nullptr, Rule::check, nullptr},
                   {"video_token_id", nullptr, Rule::check, nullptr},
                   {"vision_start_token_id", nullptr, Rule::check, nullptr},
                   {"vision_end_token_id", nullptr, Rule::check, nullptr},
                   {"language_model_only", nullptr, Rule::check, "false"}};
        t.derive = hf_derive;
        t.k_heads = "ssm.group_count";
        t.v_heads = "ssm.time_step_rank";
        t.k_width = "ssm.state_size";
        // A full-attention layer's and every layer's tensors, which the MTP layer holds too under blk.{L}.
        const std::vector<hf::Name> full = {
            {"self_attn.q_proj.weight", "attn_q.weight"},
            {"self_attn.k_proj.weight", "attn_k.weight"},
            {"self_attn.v_proj.weight", "attn_v.weight"},
            {"self_attn.o_proj.weight", "attn_output.weight"},
            {"self_attn.q_norm.weight", "attn_q_norm.weight", hf::plus_one},
            {"self_attn.k_norm.weight", "attn_k_norm.weight", hf::plus_one},
            {"input_layernorm.weight", "attn_norm.weight", hf::plus_one},
            {"post_attention_layernorm.weight", "post_attention_norm.weight", hf::plus_one},
            {"mlp.gate_proj.weight", "ffn_gate.weight"},
            {"mlp.up_proj.weight", "ffn_up.weight"},
            {"mlp.down_proj.weight", "ffn_down.weight"}};
        const std::vector<hf::Name> linear = {
            {"linear_attn.in_proj_qkv.weight", "attn_qkv.weight", hf::none, VHeads::rows_after_qk},
            {"linear_attn.in_proj_z.weight", "attn_gate.weight", hf::none, VHeads::rows},
            {"linear_attn.in_proj_b.weight", "ssm_beta.weight", hf::none, VHeads::rows},
            {"linear_attn.in_proj_a.weight", "ssm_alpha.weight", hf::none, VHeads::rows},
            {"linear_attn.conv1d.weight", "ssm_conv1d.weight", hf::widen, VHeads::rows_after_qk},
            {"linear_attn.dt_bias", "ssm_dt.bias", hf::none, VHeads::rows},
            {"linear_attn.A_log", "ssm_a", hf::neg_exp, VHeads::rows},
            {"linear_attn.norm.weight", "ssm_norm.weight"},
            {"linear_attn.out_proj.weight", "ssm_out.weight", hf::none, VHeads::columns}};
        t.names = {{"model.language_model.embed_tokens.weight", "token_embd.weight"},
                   {"model.language_model.norm.weight", "output_norm.weight", hf::plus_one},
                   {"lm_head.weight", "output.weight"},
                   {"mtp.fc.weight", "blk.{L}.nextn.eh_proj.weight"},
                   {"mtp.pre_fc_norm_embedding.weight", "blk.{L}.nextn.enorm.weight", hf::plus_one},
                   {"mtp.pre_fc_norm_hidden.weight", "blk.{L}.nextn.hnorm.weight", hf::plus_one},
                   {"mtp.norm.weight", "blk.{L}.nextn.shared_head_norm.weight", hf::plus_one},
                   {"model.visual.*", ""}};
        for (const auto* part : {&full, &linear})
            for (const hf::Name& n : *part) t.names.push_back({"model.language_model.layers.{N}." + n.hf, "blk.{N}." + n.gguf, n.transform, n.heads});
        for (const hf::Name& n : full) t.names.push_back({"mtp.layers.0." + n.hf, "blk.{L}." + n.gguf, n.transform, n.heads});
        return t;
    }();
    return table;
}

// qwen35's roles, by the id a row of resolved weights is indexed by, and its layer kinds.
enum Role : uint16_t {
    token_embd, output, output_norm,
    attn_norm, post_attention_norm,
    attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm, attn_output,
    attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_conv1d, ssm_a, ssm_dt, ssm_norm, ssm_out,
    ffn_gate, ffn_up, ffn_down,
    ffn_gate_inp, ffn_gate_exps, ffn_up_exps, ffn_down_exps, ffn_gate_inp_shexp, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp,
    nextn_eh_proj, nextn_enorm, nextn_hnorm, nextn_shared_head_norm,
    dflash_fc, dflash_hidden_norm, dflash_output_norm, dflash_codebook, dflash_gate, dflash_gate_bias, dflash_pred_gate, dflash_prev, dflash_layers,
};
// A DFlash drafter layer's roles, the layer's block of ids from dflash_layers on (dflash_role).
enum DFlashRole : uint16_t { df_norm, df_q, df_k, df_v, df_q_norm, df_k_norm, df_output, df_ffn_norm, df_gate, df_up, df_down, df_read, df_write, df_roles };
inline uint16_t dflash_role(size_t layer, DFlashRole r) { return uint16_t(dflash_layers + layer * df_roles + r); }
// A layer's kind, with writes_tap set on a layer whose entering residual a DFlash drafter reads.
enum Kind : uint8_t { linear, full, writes_tap = 128 };

// The floats one row of a pass takes in each arena slot, the widest use either layer kind makes of it; on qwen35moe the feed-forward slots hold k routed expert rows and then the shared expert's.
// Slots 0 to 9 are x, h, the mixer's working rows, the gates and the dense feed-forward rows, and on qwen35moe 10 to 13 hold router scores, expert ids, expert weights and the shared expert's gate (docs/src/model-arch-qwen35.md, Arena slots).
inline std::vector<size_t> slot_widths(const Config& c) {
    const size_t e = (size_t)c.n_embd, d = (size_t)c.head_dim;
    const size_t q = (size_t)c.n_head * d, kv = (size_t)c.n_head_kv * d;
    const size_t v = (size_t)c.v_heads * (size_t)c.v_dim, channels = 2 * (size_t)c.k_heads * (size_t)c.k_dim + v;
    const size_t k = (size_t)c.n_expert_used, ff = std::max((size_t)c.n_ff, k * (size_t)c.n_ff_exp + (size_t)c.n_ff_shexp);
    std::vector<size_t> w = {e, e, std::max(2 * q, channels), std::max(kv, channels), std::max(kv, v), std::max(q, 2 * (size_t)c.v_heads), std::max(q, v), ff, ff, ff};
    if (c.n_expert) w.insert(w.end(), {(size_t)c.n_expert, k, k, 1});
    return w;
}

class Qwen35 final : public Architecture {
public:
    explicit Qwen35(const Config& cfg) : cfg_(cfg) {
        cfg_.rope_policy = rope::bind(cfg_.rope_policy, {rope::Base::negative_power, cfg_.rope_theta, size_t(cfg_.rope_dim), size_t(cfg_.context_length)});
        if (cfg_.dflash)
            draft_rope_ = rope::bind(cfg_.draft_rope, {rope::Base::negative_power, cfg_.dflash->rope_theta, size_t(cfg_.dflash->head_dim), size_t(cfg_.context_length)});
    }

    // A layer's roles are its norm, then its mixer's by its kind, then the feed-forward norm and the dense block or, on qwen35moe, the router, expert stacks and shared expert with its gate.
    // A full-attention layer keeps KV and a linear-attention layer a state; a layer holding the other kind's tensors, or a router in a dense architecture, is refused, and blocks past the decoder layers are not read.
    ModelPlan plan(const TensorIndex& tensors) const override {
        const Config& c = cfg_;
        const TensorView& embedding = tensors.view(tensors.at("token_embd.weight"));
        if (embedding.shape.size() < 2 || !embedding.shape[1] || embedding.shape[1] > uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("inference: invalid vocabulary dimension");
        ModelPlan p;
        p.role_ids = ffn_down_shexp + 1;
        p.vocab = embedding.shape[1];
        const uint64_t E = (uint64_t)c.n_embd;
        const uint64_t Hv = (uint64_t)c.v_heads, V = Hv * (uint64_t)c.v_dim, C = 2 * (uint64_t)c.k_heads * (uint64_t)c.k_dim + V;
        p.pass = {{token_embd, Part::embed, RoleKind::gather, "token_embd.weight", "", E, p.vocab},
                  {output, Part::head, RoleKind::matrix, "output.weight", "token_embd.weight", E, p.vocab},
                  {output_norm, Part::head, RoleKind::norm, "output_norm.weight", "", E}};
        blocks::shard(p.pass, output, Axis::rows, {{p.vocab, 1, 1, "vocabulary rows"}});
        p.layers.resize((size_t)c.n_layer);
        for (int l = 0; l < c.n_layer; ++l) {
            const std::string pre = "blk." + std::to_string(l) + ".";
            LayerPlan& layer = p.layers[(size_t)l];
            if (!c.n_expert && tensors.find(pre + "ffn_gate_inp.weight")) throw std::runtime_error("inference: expert tensors in a dense architecture " + pre);
            layer.roles = {{attn_norm, Part::mixer, RoleKind::norm, pre + "attn_norm.weight", "", E}};
            if (c.full[(size_t)l]) {
                if (tensors.find(pre + "attn_qkv.weight"))
                    throw std::runtime_error("inference: a full-attention layer holds the linear-attention tensor " + pre + "attn_qkv.weight");
                full_layer(layer, pre, Part::mixer);
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
                shard_linear(layer);
                layer.saved = {{2, (size_t)C}, {5, (size_t)Hv}, {5, (size_t)Hv, 1}};
                // The conv's output and the recurrence's, the conv first, since the recurrence reads its output.
                layer.recur_writes = {3, 6};
                layer.recur_phases = 2;
            }
            ffn_roles(layer, tensors, pre, Part::ffn);
        }
        p.context_length = (size_t)c.context_length;
        p.slots = slot_widths(c);
        p.residual = (size_t)c.n_embd;
        p.kv_heads = (size_t)c.n_head_kv;
        p.head_dim = (size_t)c.head_dim;
        p.state = backend::StateShape{(size_t)c.k_heads, (size_t)c.v_heads, (size_t)c.k_dim, (size_t)c.v_dim};
        {
            backend::PassWidths& w = p.widths;
            const size_t e = (size_t)c.n_embd, q = (size_t)c.n_head * c.head_dim, kv = (size_t)c.n_head_kv * c.head_dim;
            const size_t inner = (size_t)c.v_heads * c.v_dim, channels = 2 * (size_t)c.k_heads * c.k_dim + inner;
            w.heads = (size_t)c.n_head;
            w.head_dim = (size_t)c.head_dim;
            w.dense_attention = true;
            if (c.n_expert) {
                w.routed = (size_t)c.n_expert_used;
                w.expert_in = (size_t)c.n_ff_exp;
                w.expert_out = e;
                w.shared_ff = (size_t)c.n_ff_shexp;
            }
            w.product_in = std::max({e, q, inner, (size_t)c.n_ff, (size_t)c.n_ff_shexp});
            w.product_out = std::max({2 * q + 2 * kv, channels + inner + 2 * (size_t)c.v_heads, e, 2 * (size_t)c.n_ff, 2 * (size_t)c.n_ff_shexp});
        }
        p.tables.assign(2, 0);
        p.table_steps.assign(2, (size_t)(c.rope_dim / 2));
        p.image_streams = c.image_streams;
        return p;
    }

    // The rope's cos and sin for every position the tables cover over the rotated pairs only, indexed as [pos * (rope_dim / 2) + i].
    // A DFlash drafter's own rope follows in tables 2 and 3 (plan_drafter).
    void fill_tables(std::vector<std::vector<float>>& tables, const std::vector<TensorView>&) const override {
        rope::fill(cfg_.rope_policy, tables[0], tables[1]);
        if (tables.size() > 2) rope::fill(draft_rope_, tables[2], tables[3]);
    }

    void embed(const Step& s, const uint32_t* ids) const override { blocks::embed(s, s.w[token_embd], ids); }

    // The slots are those of slot_widths.
    void mixer(const Step& s) const override {
        backend::Backend& b = s.b;
        const Weight* w = s.w;
        const size_t E = (size_t)cfg_.n_embd;
        const backend::Slice x = s.x, h = s.slot(1);
        if (s.kind & writes_tap) tap(s);
        b.rms_norm_rows(h, x, w[attn_norm].slice(), s.rows, E, E, cfg_.rms_eps, s.runs);
        if ((s.kind & ~writes_tap) == full) {
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
        const blocks::SharedExpert shared{w[ffn_gate_shexp], w[ffn_up_shexp], w[ffn_down_shexp], &w[ffn_gate_inp_shexp]};
        blocks::moe_step(s, w[ffn_gate_inp], w[ffn_gate_exps], w[ffn_up_exps], w[ffn_down_exps], (size_t)cfg_.n_expert_used, backend::RouteRule::softmax(cfg_.expert_norm),
                         h, g, u, act, s.slot(10), s.slot(11), s.slot(12), &shared, s.slot(13));
    }

    void head(const HeadStep& s) const override { blocks::head(s, s.w[output_norm], s.w[output], cfg_.rms_eps, s.slot(1)); }

    // The MTP block after the decoder layers (docs/SPECULATIVE.md, section 7): the target's embedding, final norm and head, the block's input norms and eh_proj, and a full-attention layer with its feed-forward block, all on the head's device.
    // Its cache is KV, and four more arena slots hold the target's normed rows (draft_h), the block's input rows, their interleave, and eh_proj's output, which is the block's residual.
    void plan_drafter(const TensorIndex& tensors, ModelPlan& p) const override {
        const Config& c = cfg_;
        if (c.dflash) return plan_dflash(tensors, p);
        if (!c.n_nextn) throw std::runtime_error("inference: the file carries no MTP block for an embedded drafter");
        const uint64_t E = (uint64_t)c.n_embd;
        const std::string pre = "blk." + std::to_string(c.n_layer) + ".";
        if (tensors.find(pre + "attn_qkv.weight"))
            throw std::runtime_error("inference: the MTP block holds the linear-attention tensor " + pre + "attn_qkv.weight");
        LayerPlan d;
        d.roles = {{token_embd, Part::draft, RoleKind::gather, "token_embd.weight", "", E, p.vocab},
                   {output, Part::draft, RoleKind::matrix, "output.weight", "token_embd.weight", E, p.vocab},
                   {output_norm, Part::draft, RoleKind::norm, "output_norm.weight", "", E},
                   {nextn_enorm, Part::draft, RoleKind::norm, pre + "nextn.enorm.weight", "", E},
                   {nextn_hnorm, Part::draft, RoleKind::norm, pre + "nextn.hnorm.weight", "", E},
                   {nextn_eh_proj, Part::draft, RoleKind::matrix, pre + "nextn.eh_proj.weight", "", 2 * E, E},
                   {nextn_shared_head_norm, Part::draft, RoleKind::norm, pre + "nextn.shared_head_norm.weight", "", E},
                   {attn_norm, Part::draft, RoleKind::norm, pre + "attn_norm.weight", "", E}};
        full_layer(d, pre, Part::draft);
        ffn_roles(d, tensors, pre, Part::draft);
        for (infer::Role& r : d.roles) r.stream = Stream::none;
        d.routed = false;
        blocks::shard(d.roles, output, Axis::rows, {{p.vocab, 1, 1, "vocabulary rows"}});
        d.ops.push_back({Part::draft, backend::Op::argmax_rows});
        d.ops.push_back({Part::draft, backend::Op::embed_ids});
        p.role_ids = nextn_shared_head_norm + 1;
        p.draft_h = p.slots.size();
        p.draft_x = p.draft_h + 3;
        p.slots.insert(p.slots.end(), {(size_t)E, 2 * (size_t)E, 2 * (size_t)E, (size_t)E});
        p.drafter = std::move(d);
    }

    // Every row of the pass in the MTP layer's cache: the final norm over every row into draft_h, the row before each, the block's input (blocks::nextn_input), then attn_norm, K and V, the k norm and the partial rope, written to the cache.
    // q, attention, the feed-forward block and the head are not run for these rows, since only their K and V are ever read; on a tensor group each member writes the K and V of its own KV heads from the same rows, with no sum.
    void draft_rows(const DraftRowsStep& s) const override {
        if (cfg_.dflash) return dflash_rows(s);
        backend::Backend& b = s.b;
        const Weight* w = s.w;
        const size_t E = (size_t)cfg_.n_embd, D = (size_t)cfg_.head_dim, Hkv = kv_share((size_t)cfg_.n_head_kv, s.width);
        const size_t base = draft_slot();
        const backend::Slice hn = s.slot(base), pair = s.slot(base + 1), u = s.slot(base + 3), h = s.slot(1), k = s.slot(3), v = s.slot(4);
        b.rms_norm_rows(hn, s.x, w[output_norm].slice(), s.rows, E, E, cfg_.rms_eps);
        b.embed(pair, w[token_embd].operand(), w[token_embd].nin, w[token_embd].nout, s.ids, s.rows);
        const size_t prev = pair.offset + s.rows * E;
        for (size_t e = 0, r0 = 0; e < s.runs.n; r0 = s.runs.runs[e++].end) {
            const size_t r1 = s.runs.runs[e].end;
            b.copy(*pair.buffer, (prev + r0 * E) * sizeof(float), *s.carry[e].buffer, s.carry[e].offset * sizeof(float), E * sizeof(float));
            if (r1 - r0 > 1)
                b.copy(*pair.buffer, (prev + (r0 + 1) * E) * sizeof(float), *hn.buffer, (hn.offset + r0 * E) * sizeof(float), (r1 - r0 - 1) * E * sizeof(float));
        }
        blocks::nextn_input(s, w[nextn_enorm], w[nextn_hnorm], w[nextn_eh_proj], pair, s.slot(base + 2), u, cfg_.rms_eps);
        b.rms_norm_rows(h, u, w[attn_norm].slice(), s.rows, E, E, cfg_.rms_eps, s.runs);
        b.matmul_group({blocks::projection(w[attn_k], k), blocks::projection(w[attn_v], v)}, h, E, s.rows, s.runs, s.dtype);
        const backend::CSlice cos{s.tables[0].get(), 0}, sin{s.tables[1].get(), 0};
        b.norm_rope_partial(k, k, s.rows, Hkv * D, D, Hkv, D, (size_t)cfg_.rope_dim, w[attn_k_norm].slice(), cfg_.rms_eps, cos, sin, s.pos, backend::RopeForm{}, s.streams);
        b.kv_write(s.kv_layer, s.views, s.n_views, k, v);
    }

    // A draft row a sequence: each token's row read from its id and the row before it as the block's input, then the block's layers, shared_head_norm into `out`, the target's head into `logits` and its argmax into the next id.
    void draft(const DraftStep& s) const override {
        if (cfg_.dflash) return dflash_block(s);
        backend::Backend& b = s.b;
        const Weight* w = s.w;
        const size_t E = (size_t)cfg_.n_embd, n = s.rows;
        const size_t base = draft_slot();
        const backend::Slice pair = s.slot(base + 1), x = s.slot(base + 3);
        Step layer = s;
        layer.x = x;
        layer.kind = full;
        if (s.phase <= 0) {
            b.embed_ids(pair, w[token_embd].operand(), w[token_embd].nin, w[token_embd].nout, s.id, n);
            const backend::Slice prev{pair.buffer, pair.offset + n * E};
            if (s.prev_rows) b.gather_rows(prev, s.prev, E, s.prev_rows, n);
            else b.copy(*pair.buffer, prev.offset * sizeof(float), *s.prev.buffer, s.prev.offset * sizeof(float), n * E * sizeof(float));
            blocks::nextn_input(s, w[nextn_enorm], w[nextn_hnorm], w[nextn_eh_proj], pair, s.slot(base + 2), x, cfg_.rms_eps);
            mixer(layer);
        }
        if (s.phase < 0 || s.phase == 1) ffn(layer);
        if (s.phase < 0 || s.phase == 2) {
            b.rms_norm_rows(s.out, x, w[nextn_shared_head_norm].slice(), n, E, E, cfg_.rms_eps);
            blocks::draft_head(s, w[output], s.out, E);
        }
    }

    // A linear-attention layer's state update: phase 0 the causal conv over the raw rows (slot 2) into slot 3, phase 1 the recurrence over it with alpha and beta (slot 5) into slot 6, each reading the views' src slots and writing their dst slots.
    void recur(const Step& s) const override {
        backend::Backend& b = s.b;
        const Weight* w = s.w;
        const size_t Hv = (size_t)cfg_.v_heads / s.width;
        const backend::Slice raw = s.slot(2), u = s.slot(3), alpha = s.slot(5), o = s.slot(6);
        const backend::Slice beta{alpha.buffer, alpha.offset + s.rows * Hv};
        if (s.phase != 1) b.causal_conv_silu(u, raw, w[ssm_conv1d].slice(), s.state_layer, s.states, s.n_views);
        if (s.phase != 0) b.gated_delta_rule(o, u, alpha, beta, w[ssm_a].slice(), w[ssm_dt].slice(), s.state_layer, s.states, s.n_views);
    }

    const rope::Policy& rope_policy() const override { return cfg_.rope_policy; }

private:
    Config cfg_;
    rope::Policy draft_rope_;   // a DFlash drafter's rope over its whole head (tables 2 and 3)

    // The DFlash drafter's arena slots after slot_widths': the taps, one plane of a pass's rows each, their rows side by side, and fc's normed output, the context row its layers project.
    size_t tap_slot() const { return draft_slot(); }

    // A tap: the residual entering the layer into its plane of the taps' slot, the planes holding the pass's rows one after another.
    void tap(const Step& s) const {
        const std::vector<int>& taps = cfg_.dflash->taps;
        const size_t i = size_t(std::find(taps.begin(), taps.end(), s.layer) - taps.begin()), E = (size_t)cfg_.n_embd;
        if (i == taps.size()) throw std::logic_error("inference: a qwen35 tap on a layer the drafter does not read");
        const backend::Slice plane = s.plane(tap_slot(), i, E);
        s.b.copy(*plane.buffer, plane.offset * sizeof(float), *s.x.buffer, s.x.offset * sizeof(float), s.rows * E * sizeof(float));
    }

    // A DFlash sidecar (docs/SPECULATIVE.md, section 11): fc and its norm over the taps, the drafter's layers each in a KV layer of its own, its final norm, the code bridge and the target's embedding and head.
    void plan_dflash(const TensorIndex& tensors, ModelPlan& p) const {
        const dflash::Config& d = *cfg_.dflash;
        const uint64_t E = (uint64_t)cfg_.n_embd, T = d.taps.size(), D = (uint64_t)d.head_dim, Q = (uint64_t)d.heads * D, KV = (uint64_t)d.kv_heads * D, F = (uint64_t)d.ff;
        const auto shape = [&](const std::string& name, size_t axis) { return (uint64_t)tensors.view(tensors.at(name)).shape.at(axis); };
        const uint64_t C = shape("dflash.dspine_codebook.weight", 0), R = shape("dflash.dspine_gate.bias", 0);
        if (uint64_t(d.top_k) > p.vocab) throw std::runtime_error("inference: a DFlash drafter's selector_top_k exceeds the vocabulary");
        LayerPlan dl;
        dl.kind = full;
        dl.cache = Cache::kv;
        dl.roles = {{token_embd, Part::draft, RoleKind::gather, "token_embd.weight", "", E, p.vocab},
                    {output, Part::draft, RoleKind::matrix, "output.weight", "token_embd.weight", E, p.vocab},
                    {dflash_fc, Part::draft, RoleKind::matrix, "dflash.fc.weight", "", T * E, E},
                    {dflash_hidden_norm, Part::draft, RoleKind::norm, "dflash.enc.output_norm.weight", "", E},
                    {dflash_output_norm, Part::draft, RoleKind::norm, "dflash.output_norm.weight", "", E},
                    {dflash_codebook, Part::draft, RoleKind::gather, "dflash.dspine_codebook.weight", "", C, p.vocab},
                    {dflash_gate, Part::draft, RoleKind::matrix, "dflash.dspine_gate.weight", "", C, R},
                    {dflash_gate_bias, Part::draft, RoleKind::norm, "dflash.dspine_gate.bias", "", R},
                    {dflash_pred_gate, Part::draft, RoleKind::matrix, "dflash.dspine_pred_gate.weight", "", C, R},
                    {dflash_prev, Part::draft, RoleKind::matrix, "dflash.dspine_prev.weight", "", C, R}};
        for (size_t l = 0; l < size_t(d.layers); ++l) {
            const std::string pre = "dflash.blk." + std::to_string(l) + ".";
            const auto role = [&](DFlashRole r, RoleKind kind, const char* name, uint64_t in, uint64_t out = 1) {
                dl.roles.push_back({dflash_role(l, r), Part::draft, kind, pre + name, "", in, out});
            };
            role(df_norm, RoleKind::norm, "attn_norm.weight", E);
            role(df_q, RoleKind::matrix, "attn_q.weight", E, Q);
            role(df_k, RoleKind::matrix, "attn_k.weight", E, KV);
            role(df_v, RoleKind::matrix, "attn_v.weight", E, KV);
            role(df_q_norm, RoleKind::norm, "attn_q_norm.weight", D);
            role(df_k_norm, RoleKind::norm, "attn_k_norm.weight", D);
            role(df_output, RoleKind::matrix, "attn_output.weight", Q, E);
            role(df_ffn_norm, RoleKind::norm, "ffn_norm.weight", E);
            role(df_gate, RoleKind::matrix, "ffn_gate.weight", E, F);
            role(df_up, RoleKind::matrix, "ffn_up.weight", E, F);
            role(df_down, RoleKind::matrix, "ffn_down.weight", F, E);
            role(df_read, RoleKind::matrix, "dspine_read.weight", E, C);
            role(df_write, RoleKind::matrix, "dspine_write.weight", R, E);
        }
        for (infer::Role& r : dl.roles) r.stream = Stream::none;
        using backend::Op;
        for (Op op : {Op::norm_rope_partial, Op::attention_window, Op::embed_ids, Op::unit_rows, Op::sigmoid_gate_rows, Op::topk_rows, Op::matmul_at_ids, Op::chain_pick})
            dl.ops.push_back({Part::draft, op});
        // A tap layer's mixer writes its plane, which crosses with the residual from the first tap on (ModelPlan::Carried).
        for (int t : d.taps) p.layers[size_t(t)].kind |= writes_tap;
        p.carried.push_back({tap_slot(), d.taps.front() + 1});
        // Then the block's: its codes, gates, messages, row norms and last raw rows, and the tail's refined rows, scores and candidates, R * K tail rows in R * block_size rows' room.
        const size_t f = (size_t(d.top_k) + size_t(d.block) - 1) / size_t(d.block), two = std::max<size_t>(2, f), K = size_t(d.top_k);
        p.slots.insert(p.slots.end(), {size_t(T * E), size_t(T * E), size_t(E), two * C, 2 * R, two * R, two * R, f * R, 1, size_t(E), f * E, K * K, K, 1});
        for (const auto& [slot, width] : std::initializer_list<std::pair<size_t, size_t>>{{2, size_t(Q)}, {3, size_t(KV)}, {4, size_t(KV)}, {6, size_t(Q)}, {7, size_t(F)}, {8, size_t(F)}, {9, size_t(F)}})
            p.slots[slot] = std::max(p.slots[slot], width);
        p.index_width = std::max({p.index_width, size_t(T), 2 * f});
        p.tables.resize(4, 0);
        p.table_steps.resize(4, size_t(D / 2));
        p.role_ids = dflash_role(size_t(d.layers), df_norm);
        p.draft_kv_layers = size_t(d.layers);
        if (d.kv_heads != cfg_.n_head_kv || d.head_dim != cfg_.head_dim) p.draft_kv = {size_t(d.kv_heads), size_t(d.head_dim)};
        p.draft_window = size_t(d.window);
        p.draft_block = size_t(d.block) - 1;
        p.draft_anchor = 1;
        p.drafter = std::move(dl);
    }

    // The context rows of a pass: the taps of each row side by side through fc and its norm, then each layer's K and V of them, normed and rotated over the whole head by the drafter's own rope, in that layer's cache.
    void dflash_rows(const DraftRowsStep& s) const {
        const dflash::Config& d = *cfg_.dflash;
        const Weight* w = s.w;
        const size_t E = (size_t)cfg_.n_embd, T = d.taps.size(), D = (size_t)d.head_dim, Hkv = (size_t)d.kv_heads, rows = s.rows;
        if (s.width > 1) throw std::logic_error("inference: a DFlash drafter runs whole on a tensor group's first member (Model::draft_members)");
        if (!s.indices || s.indices_size < backend::size_mul(rows, T)) throw std::runtime_error("inference: the DFlash context rows' index scratch is too small");
        for (size_t r = 0; r < rows; ++r)
            for (size_t t = 0; t < T; ++t) s.indices[r * T + t] = uint32_t(t * rows + r);
        const backend::Slice side = s.slot(tap_slot() + 1), ctx = s.slot(tap_slot() + 2), k = s.slot(3), v = s.slot(4);
        s.b.gather_rows(side, s.slot(tap_slot()), E, s.indices, rows * T);
        s.b.matmul(w[dflash_fc].operand(), side, ctx, T * E, E, rows, s.runs, s.dtype);
        s.b.rms_norm_rows(ctx, ctx, w[dflash_hidden_norm].slice(), rows, E, E, cfg_.rms_eps, s.runs);
        const backend::CSlice cos{s.tables[2].get(), 0}, sin{s.tables[3].get(), 0};
        for (size_t l = 0; l < size_t(d.layers); ++l) {
            s.b.matmul_group({blocks::projection(w[dflash_role(l, df_k)], k), blocks::projection(w[dflash_role(l, df_v)], v)}, ctx, E, rows, s.runs, s.dtype);
            s.b.norm_rope_partial(k, k, rows, Hkv * D, D, Hkv, D, D, w[dflash_role(l, df_k_norm)].slice(), cfg_.rms_eps, cos, sin, s.pos, backend::RopeForm{}, s.streams);
            s.b.kv_write(s.kv_layer + l, s.views, s.n_views, k, v);
        }
    }

    // A DSpine block of each sequence (tools/dspine_reference.py, block): its last pick then mask rows at the next positions through the drafter's layers, each layer's code bridge passing row r - 1's code to row r.
    // The target's head over the final-normed rows past the first gives each draft's logits and its top_k candidates, and the refinement tail scores each candidate after each of the step before's, which chain_pick walks.
    void dflash_block(const DraftStep& s) const {
        const dflash::Config& d = *cfg_.dflash;
        backend::Backend& b = s.b;
        const Weight* w = s.w;
        const auto lw = [&](size_t l, DFlashRole r) -> const Weight& { return w[dflash_role(l, r)]; };
        const size_t E = (size_t)cfg_.n_embd, D = (size_t)d.head_dim, H = (size_t)d.heads, Hkv = (size_t)d.kv_heads, L = (size_t)d.layers;
        const size_t B = s.ids, A = B + 1, K = (size_t)d.top_k, R = s.rows, n = R * A;
        const Weight &embd = w[token_embd], &head = w[output], &codebook = w[dflash_codebook];
        const size_t C = codebook.nin, G = w[dflash_gate].nout, V = head.nout;
        const float eps = cfg_.rms_eps, gain = std::sqrt(float(C));
        if (s.width > 1) throw std::logic_error("inference: a DFlash drafter runs whole on a tensor group's first member (Model::draft_members)");
        if (s.stride != R || !B || A > (size_t)d.block || s.phase >= 0) throw std::logic_error("inference: a DFlash draft is one call over its sequences' blocks");
        if (!s.indices || s.indices_size < 2 * std::max(n, R * K)) throw std::runtime_error("inference: the DFlash draft's index scratch is too small");
        const size_t base = tap_slot();
        const backend::Slice h = s.slot(1), q = s.slot(2), k = s.slot(3), v = s.slot(4), att = s.slot(6), codes = s.slot(base + 3), qg = s.slot(base + 4),
                             sg = s.slot(base + 5), vg = s.slot(base + 6), msg = s.slot(base + 7), rms = s.slot(base + 8), raw = s.slot(base + 9),
                             refined = s.slot(base + 10), score = s.slot(base + 11), cand = s.slot(base + 12), lens = s.slot(base + 13);
        const auto at = [](backend::Slice x, size_t floats) { return backend::Slice{x.buffer, x.offset + floats}; };
        uint32_t *first = s.indices, *second = s.indices + std::max(n, R * K);
        const backend::RowRun run{n, 1}, coded{n + R, 1};
        Step x = s;
        x.rows = n;
        x.runs = {&run, 1};
        b.embed_ids(h, embd.operand(), E, embd.nout, s.id, R);
        b.embed(at(h, R * E), embd.operand(), E, embd.nout, &d.mask, 1);
        for (size_t i = 0; i < n; ++i) first[i] = uint32_t(i % A ? R : i / A);
        b.gather_rows(s.x, h, E, first, n);
        b.rms_norm_rows(h, s.x, lw(0, df_norm).slice(), n, E, E, eps, x.runs);
        const backend::CSlice cos{s.tables[2].get(), 0}, sin{s.tables[3].get(), 0};
        for (size_t l = 0; l < L; ++l) {
            b.matmul_group({blocks::projection(lw(l, df_q), q), blocks::projection(lw(l, df_k), k), blocks::projection(lw(l, df_v), v)}, h, E, n, x.runs, s.dtype);
            b.norm_rope_partial(q, q, n, H * D, D, H, D, D, lw(l, df_q_norm).slice(), eps, cos, sin, s.pos, backend::RopeForm{}, s.streams);
            b.norm_rope_partial(k, k, n, Hkv * D, D, Hkv, D, D, lw(l, df_k_norm).slice(), eps, cos, sin, s.pos, backend::RopeForm{}, s.streams);
            // Every row's K and V are written before any row attends, so no row reads a stale row of an earlier block at its position.
            b.kv_write(s.kv_layer + l, s.views, s.n_views, k, v);
            b.attention(q, s.kv_layer + l, s.views, s.n_views, att, int(H), int(Hkv), int(D));
            b.matmul_add(lw(l, df_output).operand(), att, s.x, H * D, E, n, x.runs, s.dtype);
            b.rms_norm_rows(h, s.x, lw(l, df_ffn_norm).slice(), n, E, E, eps, x.runs);
            blocks::swiglu(x, lw(l, df_gate), lw(l, df_up), lw(l, df_down), h, s.slot(7), s.slot(8), s.slot(9));
            // The code bridge: each row's unit code, the anchor's from the codebook after the block's rows, and row r's message from row r - 1's code, none for an anchor row.
            b.unit_rows(h, rms, s.x, n, E, 1.0f, eps, backend::RowScale::rms);
            b.matmul(lw(l, df_read).operand(), h, codes, E, C, n, x.runs, s.dtype);
            b.unit_rows(codes, {}, codes, n, C, gain, eps, backend::RowScale::l2);
            b.embed_ids(at(codes, n * C), codebook.operand(), C, codebook.nout, s.id, R);
            b.unit_rows(at(codes, n * C), {}, at(codes, n * C), R, C, gain, 0.0f, backend::RowScale::none);
            b.matmul_group({blocks::projection(w[dflash_gate], qg), blocks::projection(w[dflash_pred_gate], sg), blocks::projection(w[dflash_prev], vg)}, codes, C, n + R,
                           {&coded, 1}, s.dtype);
            for (size_t r = 0; r < n; ++r) {
                first[r] = uint32_t(r);
                second[r] = r % A == 0 ? UINT32_MAX : uint32_t(r % A == 1 ? n + r / A : r - 1);
            }
            b.sigmoid_gate_rows(msg, qg, sg, vg, w[dflash_gate_bias].slice(), rms, n, G, first, second);
            if (l + 1 == L) b.copy(*raw.buffer, raw.offset * sizeof(float), *s.x.buffer, s.x.offset * sizeof(float), n * E * sizeof(float));
            b.matmul_add(lw(l, df_write).operand(), msg, s.x, G, E, n, x.runs, s.dtype);
            if (l + 1 < L) b.rms_norm_rows(h, s.x, lw(l + 1, df_norm).slice(), n, E, E, eps, x.runs);
        }
        // Draft j of sequence i from row i * A + j + 1, in row j * R + i of the logits: the head over its final-normed row, and its top_k candidates.
        for (size_t j = 0; j < B; ++j)
            for (size_t i = 0; i < R; ++i) first[j * R + i] = uint32_t(i * A + j + 1);
        const backend::RowRun drafts{R * B, 1};
        b.gather_rows(h, s.x, E, first, R * B);
        b.rms_norm_rows(h, h, w[dflash_output_norm].slice(), R * B, E, E, eps, {&drafts, 1});
        b.matmul_logits(head.operand(), h, s.logits, E, V, R * B, {&drafts, 1}, s.dtype);
        const std::vector<uint32_t> whole(R * B, uint32_t(V));
        b.write(*lens.buffer, lens.offset * sizeof(float), whole.data(), whole.size() * sizeof(uint32_t));
        b.topk_rows(cand, s.logits, lens, R * B, V, K, 0);
        // The tail: step t scores the candidates of draft t after each of the step before's, the last layer's raw row and gate of row t + 1 under each one's code; step 0 follows the last pick alone.
        for (size_t t = 0; t < B; ++t) {
            const size_t per = t ? K : 1, rows = R * per;
            const backend::RowRun tail{rows, 1};
            b.embed_ids(codes, codebook.operand(), C, codebook.nout, t ? backend::CSlice{cand.buffer, cand.offset + (t - 1) * R * K} : s.id, rows);
            b.unit_rows(codes, {}, codes, rows, C, gain, 0.0f, backend::RowScale::none);
            b.matmul_group({blocks::projection(w[dflash_pred_gate], sg), blocks::projection(w[dflash_prev], vg)}, codes, C, rows, {&tail, 1}, s.dtype);
            for (size_t r = 0; r < rows; ++r) {
                first[r] = uint32_t(r / per * A + t + 1);
                second[r] = uint32_t(r);
            }
            b.sigmoid_gate_rows(msg, qg, sg, vg, w[dflash_gate_bias].slice(), rms, rows, G, first, second);
            b.gather_rows(refined, raw, E, first, rows);
            b.matmul_add(lw(L - 1, df_write).operand(), msg, refined, G, E, rows, {&tail, 1}, s.dtype);
            b.matmul_at_ids(at(score, t * R * K * K), refined, w[dflash_output_norm].slice(), head.operand(), E, V, backend::CSlice{cand.buffer, cand.offset + t * R * K}, R, K, per);
        }
        b.chain_pick(s.next, score, cand, R, B, K);
    }

    // A full-attention layer's mixer roles under `pre`, run with `part`: its kind, its KV cache and its ops.
    void full_layer(LayerPlan& layer, const std::string& pre, Part part) const {
        const uint64_t E = (uint64_t)cfg_.n_embd, D = (uint64_t)cfg_.head_dim, Q = (uint64_t)cfg_.n_head * D, KV = (uint64_t)cfg_.n_head_kv * D;
        layer.kind = full;
        layer.cache = Cache::kv;
        layer.roles.insert(layer.roles.end(), {{attn_q, part, RoleKind::matrix, pre + "attn_q.weight", "", E, 2 * Q},
                                               {attn_k, part, RoleKind::matrix, pre + "attn_k.weight", "", E, KV},
                                               {attn_v, part, RoleKind::matrix, pre + "attn_v.weight", "", E, KV},
                                               {attn_q_norm, part, RoleKind::norm, pre + "attn_q_norm.weight", "", D},
                                               {attn_k_norm, part, RoleKind::norm, pre + "attn_k_norm.weight", "", D},
                                               {attn_output, part, RoleKind::matrix, pre + "attn_output.weight", "", Q, E}});
        layer.ops.push_back({part, backend::Op::norm_rope_partial});
        layer.ops.push_back({part, backend::Op::sigmoid_mul});
        // A tensor group splits it by heads, a q head with its gate, each member's q heads reading its KV heads.
        const ShardSection heads{(uint64_t)cfg_.n_head, 2 * D, 1, "heads"}, kv_heads{(uint64_t)cfg_.n_head_kv, D, 1, "KV heads", true};
        blocks::shard(layer.roles, attn_q, Axis::rows, {heads});
        blocks::shard(layer.roles, attn_k, Axis::rows, {kv_heads});
        blocks::shard(layer.roles, attn_v, Axis::rows, {kv_heads});
        blocks::shard(layer.roles, attn_output, Axis::columns, {{(uint64_t)cfg_.n_head, D, 1, "heads"}});
    }

    // A tensor group splits a linear-attention layer by K heads: each member takes its K heads' q and k rows and the V heads that read them (V head j reads K head j mod Hk).
    // attn_qkv's sections and the conv's channels follow, as do z, alpha, beta, the decay, the time step and ssm_out's columns by V head.
    void shard_linear(LayerPlan& layer) const {
        const uint64_t Hk = (uint64_t)cfg_.k_heads, tiles = (uint64_t)(cfg_.v_heads / cfg_.k_heads), Kd = (uint64_t)cfg_.k_dim, Vd = (uint64_t)cfg_.v_dim;
        const ShardSection k{Hk, Kd, 1, "K heads"}, v{Hk, Vd, tiles, "V heads of each tile"}, one{Hk, 1, tiles, "V heads of each tile"};
        blocks::shard(layer.roles, attn_qkv, Axis::rows, {k, k, v});
        blocks::shard(layer.roles, ssm_conv1d, Axis::rows, {k, k, v});
        blocks::shard(layer.roles, attn_gate, Axis::rows, {v});
        blocks::shard(layer.roles, ssm_alpha, Axis::rows, {one});
        blocks::shard(layer.roles, ssm_beta, Axis::rows, {one});
        blocks::shard(layer.roles, ssm_a, Axis::columns, {one});
        blocks::shard(layer.roles, ssm_dt, Axis::columns, {one});
        blocks::shard(layer.roles, ssm_out, Axis::columns, {v});
    }

    // A layer's feed-forward roles under `pre`, run with `part`: Qwen3's dense block, or on qwen35moe the router, the expert stacks and the shared expert with its gate.
    // A routed layer run beside its mixer copies the norm, router and shared expert and writes its stacks into a window.
    void ffn_roles(LayerPlan& layer, const TensorIndex& tensors, const std::string& pre, Part part) const {
        const Config& c = cfg_;
        const uint64_t E = (uint64_t)c.n_embd, F = (uint64_t)c.n_ff;
        if (!c.n_expert) {
            layer.roles.insert(layer.roles.end(), {{post_attention_norm, part, RoleKind::norm, pre + "post_attention_norm.weight", "", E},
                                                   {ffn_gate, part, RoleKind::matrix, pre + "ffn_gate.weight", "", E, F},
                                                   {ffn_up, part, RoleKind::matrix, pre + "ffn_up.weight", "", E, F},
                                                   {ffn_down, part, RoleKind::matrix, pre + "ffn_down.weight", "", F, E}});
            blocks::shard_swiglu(layer.roles, ffn_gate, ffn_up, ffn_down, F);
            return;
        }
        const uint64_t X = (uint64_t)c.n_expert, Fe = (uint64_t)c.n_ff_exp, Fs = (uint64_t)c.n_ff_shexp;
        layer.routed = true;
        // The shared expert's gate is a vector of E weights, one dot product a row, checked as an F32 row.
        layer.roles.insert(layer.roles.end(), {{post_attention_norm, part, RoleKind::norm, pre + "post_attention_norm.weight", "", E, 1, 0, Stream::copy},
                                               {ffn_gate_inp, part, RoleKind::matrix, pre + "ffn_gate_inp.weight", "", E, X, 0, Stream::copy},
                                               {ffn_gate_exps, part, RoleKind::experts, pre + "ffn_gate_exps.weight", "", E, Fe, X, Stream::window},
                                               {ffn_up_exps, part, RoleKind::experts, pre + "ffn_up_exps.weight", "", E, Fe, X, Stream::window},
                                               {ffn_down_exps, part, RoleKind::experts, pre + "ffn_down_exps.weight", "", Fe, E, X, Stream::window},
                                               {ffn_gate_inp_shexp, part, RoleKind::norm, pre + "ffn_gate_inp_shexp.weight", "", E, 1, 0, Stream::copy},
                                               {ffn_gate_shexp, part, RoleKind::matrix, pre + "ffn_gate_shexp.weight", "", E, Fs, 0, Stream::copy},
                                               {ffn_up_shexp, part, RoleKind::matrix, pre + "ffn_up_shexp.weight", "", E, Fs, 0, Stream::copy},
                                               {ffn_down_shexp, part, RoleKind::matrix, pre + "ffn_down_shexp.weight", "", Fs, E, 0, Stream::copy}});
        layer.ops.push_back({part, backend::Op::sigmoid_mul});
        blocks::routed_ops(layer, tensors, ffn_gate_exps, ffn_up_exps);
        // A tensor group splits each routed expert's hidden rows and the shared expert's, one more expert, each covered by whole blocks of its down projection; the router and the shared expert's gate vector are whole on every member.
        blocks::shard_experts(layer.roles, tensors, ffn_gate_exps, ffn_up_exps, ffn_down_exps, Fe, X);
        blocks::shard_experts(layer.roles, tensors, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp, Fs, 1);
    }

    // The first of the drafter's four arena slots (plan_drafter), after those of slot_widths.
    size_t draft_slot() const { return slot_widths(cfg_).size(); }

    // Gated attention (docs/QWEN35.md): attn_q gives each head's q and gate side by side; q and k are normed per head and rotated over rope_dim, and the output is gated by sigmoid(gate) before the projection joins the residual.
    void full_attention(const Step& s, backend::Slice h) const {
        backend::Backend& b = s.b;
        const Weight* w = s.w;
        // On a tensor group each member runs its share of the heads, a KV head replicated where the width is a multiple of them.
        const size_t E = (size_t)cfg_.n_embd, D = (size_t)cfg_.head_dim, Hq = (size_t)cfg_.n_head / s.width;
        const size_t Hkv = kv_share((size_t)cfg_.n_head_kv, s.width);
        const backend::Slice r = s.slot(2), k = s.slot(3), v = s.slot(4), q = s.slot(5), o = s.slot(6);
        b.matmul_group({blocks::projection(w[attn_q], r), blocks::projection(w[attn_k], k), blocks::projection(w[attn_v], v)}, h, E, s.rows, s.runs, s.dtype);
        const backend::CSlice cos{s.tables[0].get(), 0}, sin{s.tables[1].get(), 0};
        const backend::Backend::GatedLayer in{q, r, k, v, (size_t)cfg_.rope_dim, w[attn_q_norm].slice(), w[attn_k_norm].slice(), cfg_.rms_eps, cos, sin, s.pos, s.streams};
        b.gated_attention_layer(in, s.kv_layer, s.views, s.n_views, o, (int)Hq, (int)Hkv, (int)D, {backend::CSlice{r.buffer, r.offset + D}, s.rows, 2 * Hq * D, 2 * D, s.runs});
        blocks::join(s, w[attn_output], o);
    }

    // The gated delta net (docs/QWEN35.md, Linear attention): one delta_step over slots 2 (qkv), 4 (z), 5 (alpha, beta), 3 (conv) and 6 (out), then the output projection.
    // The step leaves slots 2 and 5 as recur's phases read them, so a retract's rerun is recur over them.
    void linear_attention(const Step& s, backend::Slice h) const {
        const Weight* w = s.w;
        // On a tensor group each member runs its K heads and the V heads that read them, over its own state.
        const size_t E = (size_t)cfg_.n_embd, Hv = (size_t)cfg_.v_heads / s.width;
        const backend::Slice alpha = s.slot(5), beta{alpha.buffer, alpha.offset + s.rows * Hv}, o = s.slot(6);
        const backend::DeltaStep d{blocks::projection(w[attn_qkv], s.slot(2)), blocks::projection(w[attn_gate], s.slot(4)), blocks::projection(w[ssm_alpha], alpha),
                                   blocks::projection(w[ssm_beta], beta), w[ssm_conv1d].slice(), w[ssm_a].slice(), w[ssm_dt].slice(), w[ssm_norm].slice(), s.slot(3),
                                   o, cfg_.rms_eps, backend::NormGate::silu};
        s.b.delta_step(d, h, E, s.rows, s.state_layer, s.states, s.n_views, s.runs, s.dtype);
        blocks::join(s, w[ssm_out], o);
    }
};

// The registry's readers: a qwen35 file's architecture, and a qwen35moe file's.
inline std::shared_ptr<const Architecture> open_dense(const gguf::GGUFModel& m, const std::string& prefix, const rope::Request& request) {
    return std::make_shared<const Qwen35>(read_config(m, prefix, false, request));
}
inline std::shared_ptr<const Architecture> open_routed(const gguf::GGUFModel& m, const std::string& prefix, const rope::Request& request) {
    return std::make_shared<const Qwen35>(read_config(m, prefix, true, request));
}

} // namespace infer::qwen35
