#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <set>
#include <string>
#include <vector>

#include "format/gguf.hpp"
#include "model/arch/metadata.hpp"
#include "model/arch/blocks.hpp"

// Qwen4Exp text configuration, integer n-gram selection and layer graph pieces.
namespace infer::qwen4exp {

enum LayerKind : uint8_t { linear_attention, qwen_sparse_attention };

struct Config {
    int n_layer = 0, n_nextn = 0, n_embd = 0;
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;
    bool expert_norm = true;
    int n_head = 0, n_head_kv = 0, head_dim = 0, rope_dim = 0, context_length = 0;
    float rope_theta = 10000000.0f, rms_eps = 1e-6f;
    int k_heads = 0, v_heads = 0, k_dim = 0, v_dim = 0;
    std::vector<int> rope_sections;
    std::vector<LayerKind> layers;
    std::vector<int> compress_ratios;
    int indexer_n_heads = 0, indexer_kv_heads = 1, indexer_head_dim = 0;
    int indexer_budget = 0, indexer_compress_ratio = 0;
    int nextn_compress_ratio = 0;
    int hc_count = 0, hc_low_rank = 0;
    bool has_ple = false;
    int ple_layer = -1, ple_ngram_size = 0, ple_heads_per_ngram = 0;
    int ple_conv_kernel = 0, ple_embedding_length = 0, ple_embedding_dim = 0, ple_eos_token_id = 0;
    std::vector<uint64_t> ple_layer_multipliers, ple_head_offsets, ple_head_vocab_sizes;
};

struct PleWeights {
    const Weight &key, &value, &norm_key, &norm_query, &norm_conv, &conv;
};

// Separate arena rows retained until the layer's state inputs have been saved for a marked pass.
struct PleScratch {
    backend::Slice key, value, query, gated, normed;
};

// Replay just the PLE state update from its saved normalized and gated values.
inline void ple_recur(const Config& c, const Step& s, const Weight& conv, backend::Slice out, const PleScratch& tmp) {
    s.b.dilated_conv_silu_add(out, tmp.normed, tmp.gated, conv.slice(), size_t(c.hc_count) * size_t(c.n_embd),
                             size_t(c.ple_conv_kernel), size_t(c.ple_ngram_size), s.state_layer, s.states, s.n_views);
}

// The PLE contribution before its caller adds it to the stream residual; embeddings are the ordered, concatenated n-gram rows.
inline void ple(const Config& c, const Step& s, const PleWeights& w, backend::CSlice embeddings,
                backend::Slice out, const PleScratch& tmp) {
    const size_t E = size_t(c.n_embd), H = size_t(c.hc_count);
    s.b.matmul_group({blocks::projection(w.key, tmp.key), blocks::projection(w.value, tmp.value)},
                     embeddings, size_t(c.ple_embedding_dim), s.rows, s.runs, s.dtype);
    s.b.group_rms_norm(tmp.key, tmp.key, w.norm_key.slice(), s.rows, H, E, c.rms_eps, s.runs);
    s.b.group_rms_norm(tmp.query, s.x, w.norm_query.slice(), s.rows, H, E, c.rms_eps, s.runs);
    s.b.signed_sqrt_gate(tmp.gated, tmp.key, tmp.query, tmp.value, s.rows, H, E, s.runs);
    s.b.group_rms_norm(tmp.normed, tmp.gated, w.norm_conv.slice(), s.rows, H, E, c.rms_eps, s.runs);
    ple_recur(c, s, w.conv, out, tmp);
}

struct MixWeights {
    const Weight &norm, &down, &up;
};

struct MixScratch {
    backend::Slice normed, low, gates;
};

// Learned mixing of the normalized streams, shared by the attention, expert and final-head inputs.
inline void mix(const Config& c, const Step& s, const MixWeights& w, backend::Slice out, const MixScratch& tmp) {
    const size_t E = size_t(c.n_embd), H = size_t(c.hc_count), L = size_t(c.hc_low_rank);
    s.b.group_rms_norm(tmp.normed, s.x, w.norm.slice(), s.rows, H, E, c.rms_eps, s.runs);
    s.b.matmul(w.down.operand(), tmp.normed, tmp.low, H * E, L, s.rows, s.runs, s.dtype);
    s.b.silu_scaled(tmp.low, tmp.low, s.rows, L, 1.0f / float(H), s.runs);
    s.b.matmul(w.up.operand(), tmp.low, tmp.gates, L, H * E, s.rows, s.runs, s.dtype);
    s.b.mix_streams(out, tmp.normed, tmp.gates, s.rows, H, E, s.runs);
}

// A part's block added into the residual streams, weighted by gates from the normed streams.
// On a group the block is the member's partial rows: the gates are made here, and the inject waits for the group's sum (Qwen4Exp::join).
inline void inject(const Config& c, const Step& s, const Weight& w, backend::CSlice normed,
                   backend::CSlice block, backend::Slice gates) {
    const size_t E = size_t(c.n_embd), H = size_t(c.hc_count);
    s.b.matmul(w.operand(), normed, gates, H * E, H, s.rows, s.runs, s.dtype);
    if (s.width == 1) s.b.inject_streams(s.x, block, gates, s.rows, H, E);
}

inline std::vector<uint64_t> uint64_array(const gguf::GGUFModel& m, const std::string& key, size_t expected) {
    const gguf::MetaValue* value = m.find(key);
    if (!value) throw std::runtime_error("inference: missing metadata " + key);
    if (value->vtype != gguf::V_ARRAY || value->u != gguf::V_UINT64)
        throw std::runtime_error("inference: invalid uint64 array type " + key);
    if (value->arr.size() != expected)
        throw std::runtime_error("inference: invalid uint64 array length " + key);
    std::vector<uint64_t> result;
    if (value->arr.size() > result.max_size())
        throw std::runtime_error("inference: metadata array exceeds allocation limit " + key);
    result.reserve(value->arr.size());
    for (const gguf::MetaValue& item : value->arr) {
        if (item.vtype != gguf::V_UINT64)
            throw std::runtime_error("inference: invalid uint64 array element " + key);
        result.push_back(item.u);
    }
    return result;
}

inline void check_config(const Config& c) {
    if (c.n_layer <= 0) throw std::runtime_error("inference: a model without decoder layers");
    if (c.n_nextn < 0 || c.n_nextn > 1)
        throw std::runtime_error("inference: unsupported number of Qwen4Exp MTP blocks");
    if (c.nextn_compress_ratio < 0 || c.nextn_compress_ratio == 1)
        throw std::runtime_error("inference: MTP compression ratio must be zero or greater than one");
    if (c.n_embd <= 0 || c.n_expert <= 0 || c.n_expert_used <= 0 || c.n_ff_exp <= 0 || c.n_ff_shexp <= 0 ||
        c.n_head <= 0 || c.n_head_kv <= 0 || c.head_dim <= 0 || c.context_length <= 0 ||
        c.k_heads <= 0 || c.v_heads <= 0 || c.k_dim <= 0 || c.v_dim <= 0)
        throw std::runtime_error("inference: invalid Qwen4Exp dimensions");
    if (!std::isfinite(c.rope_theta) || c.rope_theta <= 0 || !std::isfinite(c.rms_eps) || c.rms_eps <= 0)
        throw std::runtime_error("inference: invalid Qwen4Exp floating-point configuration");
    if (c.n_expert_used > c.n_expert)
        throw std::runtime_error("inference: more experts per token than the layer has");
    if (c.layers.size() != size_t(c.n_layer) || c.compress_ratios.size() != size_t(c.n_layer))
        throw std::runtime_error("inference: invalid Qwen4Exp layer metadata length");
    if (c.n_head % c.n_head_kv)
        throw std::runtime_error("inference: head count must be divisible by KV head count");
    if (c.v_heads % c.k_heads)
        throw std::runtime_error("inference: linear-attention V heads must be a multiple of its K heads");
    if (c.rope_dim <= 0 || c.rope_dim % 2 || c.rope_dim > c.head_dim)
        throw std::runtime_error("inference: invalid rotary width, which must be even and at most the head width");
    if (c.rope_sections.size() != 4)
        throw std::runtime_error("inference: qwen4exp.rope.dimension_sections must have four entries");
    {
        uint64_t sum = 0;
        for (int section : c.rope_sections) {
            if (section < 0 || uint64_t(section) > uint64_t(c.rope_dim / 2) - sum)
                throw std::runtime_error("inference: qwen4exp.rope.dimension_sections does not cover the rotary width");
            sum += uint64_t(section);
        }
        if (sum * 2 != uint64_t(c.rope_dim))
            throw std::runtime_error("inference: qwen4exp.rope.dimension_sections does not cover the rotary width");
    }
    if (c.hc_count <= 1) throw std::runtime_error("inference: qwen4exp.hyper_connection.count must be greater than one");
    if (c.hc_low_rank <= 0) throw std::runtime_error("inference: invalid hyper-connection low rank");
    if (uint64_t(c.hc_count) * uint64_t(c.n_embd) > uint64_t(std::numeric_limits<int>::max()) ||
        uint64_t(c.hc_count) * uint64_t(c.hc_low_rank) > uint64_t(std::numeric_limits<int>::max()))
        throw std::runtime_error("inference: invalid hyper-connection dimensions");

    bool has_qsa = false;
    uint64_t qsa_ratio = 0;
    for (size_t layer = 0; layer < c.layers.size(); ++layer) {
        const int ratio = c.compress_ratios[layer];
        if (c.layers[layer] != linear_attention && c.layers[layer] != qwen_sparse_attention)
            throw std::runtime_error("inference: invalid Qwen4Exp layer kind");
        if (c.layers[layer] == linear_attention) {
            if (ratio != 0) throw std::runtime_error("inference: compression ratio on a linear-attention layer");
        } else {
            has_qsa = true;
            if (ratio <= 1) throw std::runtime_error("inference: QSA compression ratio must be greater than one");
            if (qsa_ratio && qsa_ratio != uint64_t(ratio))
                throw std::runtime_error("inference: QSA layers must share one compression ratio");
            qsa_ratio = uint64_t(ratio);
        }
    }
    if (has_qsa) {
        if (c.indexer_n_heads <= 0 || c.indexer_kv_heads != 1 || c.indexer_head_dim <= 0 || c.indexer_budget <= 0)
            throw std::runtime_error("inference: invalid QSA indexer dimensions");
        if (c.indexer_head_dim < c.rope_dim)
            throw std::runtime_error("inference: indexer head width is smaller than the rotary width");
        if (uint64_t(c.indexer_n_heads) * uint64_t(c.indexer_head_dim) > uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("inference: QSA indexer projection dimensions exceed the supported range");
        if (c.indexer_budget % int(qsa_ratio))
            throw std::runtime_error("inference: QSA compression ratio must divide the indexer budget");
        if (c.indexer_compress_ratio != int(qsa_ratio))
            throw std::runtime_error("inference: QSA compression ratio disagrees with the layer metadata");
    }
    if (c.nextn_compress_ratio) {
        if (!c.n_nextn) throw std::runtime_error("inference: MTP compression metadata without an MTP block");
        if (c.indexer_n_heads <= 0 || c.indexer_kv_heads != 1 || c.indexer_head_dim < c.rope_dim || c.indexer_budget <= 0)
            throw std::runtime_error("inference: invalid MTP QSA indexer dimensions");
        if (uint64_t(c.indexer_n_heads) * uint64_t(c.indexer_head_dim) > uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("inference: MTP QSA indexer projection dimensions exceed the supported range");
        if (c.indexer_budget % c.nextn_compress_ratio)
            throw std::runtime_error("inference: MTP compression ratio must divide the indexer budget");
    }

    if (c.has_ple) {
        if (c.ple_layer < 0 || c.ple_layer >= c.n_layer || c.layers[size_t(c.ple_layer)] != linear_attention)
            throw std::runtime_error("inference: PLE layer must be an in-range linear-attention layer");
        if (c.ple_ngram_size < 2 || c.ple_heads_per_ngram <= 0)
            throw std::runtime_error("inference: PLE n-gram size is out of range");
        const uint64_t heads = uint64_t(c.ple_ngram_size - 1) * uint64_t(c.ple_heads_per_ngram);
        if (!heads || heads > uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("inference: PLE head count is out of range");
        if (c.ple_conv_kernel != 4 || c.ple_embedding_length <= 0)
            throw std::runtime_error("inference: unsupported PLE convolution or embedding width");
        if (c.ple_eos_token_id < 0)
            throw std::runtime_error("inference: invalid PLE EOS token id");
        if (heads * uint64_t(c.ple_embedding_length) > uint64_t(std::numeric_limits<int>::max()) ||
            c.ple_embedding_dim != int(heads * uint64_t(c.ple_embedding_length)))
            throw std::runtime_error("inference: PLE embedding dimensions exceed the supported range");
        if (c.ple_layer_multipliers.size() != size_t(c.ple_ngram_size) ||
            c.ple_head_offsets.size() != size_t(heads) || c.ple_head_vocab_sizes.size() != size_t(heads))
            throw std::runtime_error("inference: PLE integer array length mismatch");
        for (uint64_t multiplier : c.ple_layer_multipliers)
            if (!multiplier || !(multiplier & 1) || multiplier > uint64_t(std::numeric_limits<int64_t>::max()))
                throw std::runtime_error("inference: invalid PLE layer multiplier");
        for (size_t head = 0; head < size_t(heads); ++head) {
            const uint64_t offset = c.ple_head_offsets[head], size = c.ple_head_vocab_sizes[head];
            if (!size || offset > uint64_t(std::numeric_limits<int32_t>::max()) ||
                size > uint64_t(std::numeric_limits<int32_t>::max()) ||
                size - 1 > uint64_t(std::numeric_limits<int32_t>::max()) - offset)
                throw std::runtime_error("inference: PLE head range does not fit the int32 row index");
        }
    } else if (c.ple_layer != -1 || c.ple_ngram_size || c.ple_heads_per_ngram || c.ple_conv_kernel ||
               c.ple_embedding_length || c.ple_eos_token_id || !c.ple_layer_multipliers.empty() ||
               !c.ple_head_offsets.empty() || !c.ple_head_vocab_sizes.empty()) {
        throw std::runtime_error("inference: PLE metadata without qwen4exp.ple.layers");
    }

    const uint64_t q = uint64_t(c.n_head) * 2 * uint64_t(c.head_dim);
    const uint64_t channels = 2 * uint64_t(c.k_heads) * uint64_t(c.k_dim) + uint64_t(c.v_heads) * uint64_t(c.v_dim);
    const uint64_t kv_width = uint64_t(c.n_head_kv) * uint64_t(c.head_dim);
    const uint64_t max_floats = std::vector<float>().max_size();
    if (q > uint64_t(std::numeric_limits<int>::max()) || channels > uint64_t(std::numeric_limits<int>::max()) ||
        uint64_t(c.k_dim) * uint64_t(c.v_dim) > uint64_t(std::numeric_limits<int>::max()) ||
        uint64_t(c.context_length) > max_floats / kv_width ||
        uint64_t(c.context_length) > max_floats / uint64_t(c.rope_dim))
        throw std::runtime_error("inference: Qwen4Exp dimensions or context exceed allocation limit");
    if (has_qsa && uint64_t(c.context_length) > max_floats / uint64_t(c.indexer_head_dim))
        throw std::runtime_error("inference: QSA indexer cache exceeds allocation limit");
}

// Select the PLE table rows for a token chunk using the pinned HF n-gram hash.
// previous is the oldest-first tail immediately before current; output is token-major.
inline size_t ple_rows(const Config& c, uint32_t vocab_size,
                      const uint32_t* previous, size_t previous_count,
                      const uint32_t* current, size_t current_count,
                      int32_t* output, size_t output_capacity) {
    if (!c.has_ple) throw std::runtime_error("inference: PLE rows requested without PLE metadata");
    if (!vocab_size || c.ple_eos_token_id < 0 || uint32_t(c.ple_eos_token_id) >= vocab_size)
        throw std::runtime_error("inference: PLE token id is outside the vocabulary");
    if (c.ple_ngram_size < 2 || c.ple_heads_per_ngram <= 0 ||
        size_t(c.ple_ngram_size) != c.ple_layer_multipliers.size())
        throw std::runtime_error("inference: invalid PLE n-gram metadata");
    const size_t max_previous = size_t(c.ple_ngram_size - 1);
    if (previous_count > max_previous || (previous_count && !previous) || (current_count && !current))
        throw std::runtime_error("inference: invalid PLE token history");
    const size_t heads = size_t(c.ple_ngram_size - 1) * size_t(c.ple_heads_per_ngram);
    if (c.ple_head_offsets.size() != heads || c.ple_head_vocab_sizes.size() != heads)
        throw std::runtime_error("inference: invalid PLE head metadata");
    if (current_count && heads > std::numeric_limits<size_t>::max() / current_count)
        throw std::runtime_error("inference: PLE output size overflow");
    const size_t required = current_count * heads;
    if (required > output_capacity || (required && !output))
        throw std::runtime_error("inference: PLE output buffer is too small");

    const uint64_t largest_token = uint64_t(vocab_size - 1);
    for (uint64_t multiplier : c.ple_layer_multipliers) {
        if (multiplier > uint64_t(std::numeric_limits<int64_t>::max()) ||
            (largest_token && multiplier > uint64_t(std::numeric_limits<int64_t>::max()) / largest_token))
            throw std::runtime_error("inference: PLE token hash product exceeds int64 range");
    }
    for (size_t i = 0; i < previous_count; ++i)
        if (previous[i] >= vocab_size)
            throw std::runtime_error("inference: PLE token id is outside the vocabulary");
    for (size_t i = 0; i < current_count; ++i)
        if (current[i] >= vocab_size)
            throw std::runtime_error("inference: PLE token id is outside the vocabulary");

    const size_t ngram = size_t(c.ple_ngram_size);
    const size_t per_ngram = size_t(c.ple_heads_per_ngram);
    const size_t history_count = previous_count;
    for (size_t i = 0; i < current_count; ++i) {
        auto token_before = [&](size_t distance) -> uint32_t {
            const size_t prefix_count = history_count + i;
            if (distance > prefix_count) return uint32_t(c.ple_eos_token_id);
            const size_t index = prefix_count - distance;
            return index < history_count ? previous[index] : current[index - history_count];
        };
        uint64_t mixed = uint64_t(current[i]) * c.ple_layer_multipliers[0];
        bool cut = false;
        for (size_t s = 1; s < ngram; ++s) {
            const uint32_t token = cut ? uint32_t(c.ple_eos_token_id) : token_before(s);
            cut = cut || token == uint32_t(c.ple_eos_token_id);
            mixed ^= uint64_t(token) * c.ple_layer_multipliers[s];
            const size_t first_head = (s - 1) * per_ngram;
            for (size_t g = 0; g < per_ngram; ++g) {
                const size_t head = first_head + g;
                output[i * heads + head] = int32_t(
                    mixed % c.ple_head_vocab_sizes[head] + c.ple_head_offsets[head]);
            }
        }
    }
    return required;
}

inline Config read_config(const gguf::GGUFModel& m, const std::string& p) {
    using metadata::integer;
    using metadata::real;
    Config c;
    metadata::option(m, p + "tensor_data_layout", "reference");
    metadata::option(m, p + "rope.scaling.type", "none");
    if (real(m, p + "rope.scaling.factor", 1) != 1 || real(m, p + "rope.scale_linear", 1) != 1)
        throw std::runtime_error("inference: scaled RoPE is unsupported");

    const int blocks = integer(m, p + "block_count");
    c.n_nextn = metadata::count(m, p + "nextn_predict_layers", 0);
    if (c.n_nextn > 1 || c.n_nextn >= blocks)
        throw std::runtime_error("inference: unsupported metadata " + p + "nextn_predict_layers");
    c.n_layer = blocks - c.n_nextn;
    if (size_t(blocks) > std::vector<LayerKind>().max_size())
        throw std::runtime_error("inference: Qwen4Exp layer metadata exceeds allocation limit");
    c.n_embd = integer(m, p + "embedding_length");
    c.n_expert = integer(m, p + "expert_count");
    c.n_expert_used = integer(m, p + "expert_used_count");
    c.n_ff_exp = integer(m, p + "expert_feed_forward_length");
    c.n_ff_shexp = integer(m, p + "expert_shared_feed_forward_length");
    c.expert_norm = metadata::boolean(m, p + "expert_weights_norm", true);
    metadata::choice(m, p + "expert_gating_func", 1, "inference: unsupported expert gating function");
    if (real(m, p + "expert_weights_scale", 1) != 1)
        throw std::runtime_error("inference: scaled expert weights are unsupported");
    c.n_head = integer(m, p + "attention.head_count");
    c.n_head_kv = integer(m, p + "attention.head_count_kv", c.n_head);
    c.head_dim = integer(m, p + "attention.key_length");
    if (integer(m, p + "attention.value_length", c.head_dim) != c.head_dim)
        throw std::runtime_error("inference: value width must equal key width");
    c.rope_dim = integer(m, p + "rope.dimension_count", c.head_dim);
    c.context_length = integer(m, p + "context_length");
    c.rope_theta = float(real(m, p + "rope.freq_base", c.rope_theta));
    c.rms_eps = float(real(m, p + "attention.layer_norm_rms_epsilon", c.rms_eps));
    const gguf::MetaValue* sections = m.find(p + "rope.dimension_sections");
    if (!sections) throw std::runtime_error("inference: missing metadata " + p + "rope.dimension_sections");
    if (sections->vtype != gguf::V_ARRAY || sections->arr.size() != 4)
        throw std::runtime_error("inference: " + p + "rope.dimension_sections must have four entries");
    c.rope_sections = metadata::counts(m, p + "rope.dimension_sections");

    if (integer(m, p + "ssm.conv_kernel") != 4)
        throw std::runtime_error("inference: unsupported metadata " + p + "ssm.conv_kernel");
    c.k_dim = integer(m, p + "ssm.state_size");
    c.k_heads = integer(m, p + "ssm.group_count");
    c.v_heads = integer(m, p + "ssm.time_step_rank");
    const int inner = integer(m, p + "ssm.inner_size");
    if (inner % c.v_heads) throw std::runtime_error("inference: " + p + "ssm.inner_size is not a whole number of V heads");
    c.v_dim = inner / c.v_heads;

    c.hc_count = integer(m, p + "hyper_connection.count");
    c.hc_low_rank = integer(m, p + "hyper_connection.low_rank");
    const gguf::MetaValue* recurrent = m.find(p + "attention.recurrent_layers");
    if (recurrent) {
        if (recurrent->vtype != gguf::V_ARRAY || recurrent->arr.size() != size_t(blocks))
            throw std::runtime_error("inference: unsupported metadata " + p + "attention.recurrent_layers");
        c.layers.reserve(size_t(c.n_layer));
        for (int layer = 0; layer < blocks; ++layer) {
            const gguf::MetaValue& item = recurrent->arr[size_t(layer)];
            if (item.vtype != gguf::V_BOOL)
                throw std::runtime_error("inference: invalid boolean type " + p + "attention.recurrent_layers");
            if (layer >= c.n_layer && item.b)
                throw std::runtime_error("inference: unsupported metadata " + p + "attention.recurrent_layers");
            if (layer < c.n_layer)
                c.layers.push_back(item.b ? linear_attention : qwen_sparse_attention);
        }
    } else {
        const int interval = metadata::count(m, p + "full_attention_interval", 4);
        if (!interval) throw std::runtime_error("inference: invalid positive integer " + p + "full_attention_interval");
        c.layers.reserve(size_t(c.n_layer));
        for (int layer = 0; layer < c.n_layer; ++layer)
            c.layers.push_back(((layer + 1) % interval) ? linear_attention : qwen_sparse_attention);
    }
    std::vector<int> all_compress_ratios;
    const gguf::MetaValue* compress_ratios = m.find(p + "attention.compress_ratios");
    if (compress_ratios) {
        if (compress_ratios->vtype != gguf::V_ARRAY || compress_ratios->arr.size() < size_t(c.n_layer) ||
            compress_ratios->arr.size() > size_t(blocks))
            throw std::runtime_error("inference: unsupported metadata " + p + "attention.compress_ratios");
        all_compress_ratios.reserve(compress_ratios->arr.size());
        for (const gguf::MetaValue& item : compress_ratios->arr)
            all_compress_ratios.push_back(metadata::count_value(item, p + "attention.compress_ratios"));
    } else {
        all_compress_ratios.resize(size_t(blocks), 0);
    }
    c.compress_ratios.assign(all_compress_ratios.begin(), all_compress_ratios.begin() + c.n_layer);

    const bool has_indexer_heads = m.find(p + "attention.indexer.head_count") != nullptr;
    const bool has_indexer_width = m.find(p + "attention.indexer.key_length") != nullptr;
    const bool has_indexer_budget = m.find(p + "attention.indexer.top_k") != nullptr;
    if (has_indexer_heads || has_indexer_width || has_indexer_budget) {
        if (!(has_indexer_heads && has_indexer_width && has_indexer_budget))
            throw std::runtime_error("inference: QSA indexer metadata is incomplete");
        c.indexer_n_heads = integer(m, p + "attention.indexer.head_count");
        c.indexer_head_dim = integer(m, p + "attention.indexer.key_length");
        c.indexer_budget = integer(m, p + "attention.indexer.top_k");
    }
    for (int ratio : c.compress_ratios) {
        if (ratio > 0) {
            if (c.indexer_compress_ratio && c.indexer_compress_ratio != ratio)
                throw std::runtime_error("inference: QSA layers must share one compression ratio");
            c.indexer_compress_ratio = ratio;
        }
    }
    if (c.n_nextn && all_compress_ratios.size() > size_t(c.n_layer))
        c.nextn_compress_ratio = all_compress_ratios[size_t(c.n_layer)];

    if (const gguf::MetaValue* values = m.find(p + "ple.layers"))
        if (values->vtype != gguf::V_ARRAY || values->arr.size() > 1)
            throw std::runtime_error("inference: only one PLE layer is supported");
    const std::vector<int> ple_layers = metadata::counts(m, p + "ple.layers");
    if (!ple_layers.empty()) {
        if (ple_layers.size() != 1) throw std::runtime_error("inference: only one PLE layer is supported");
        c.has_ple = true;
        c.ple_layer = ple_layers[0];
        c.ple_ngram_size = integer(m, p + "ple.ngram_size");
        c.ple_heads_per_ngram = integer(m, p + "ple.heads_per_ngram");
        c.ple_conv_kernel = integer(m, p + "ple.conv_kernel");
        c.ple_embedding_length = integer(m, p + "embedding_length_per_layer_input");
        if (!m.find(p + "ple.eos_token_id"))
            throw std::runtime_error("inference: missing metadata " + p + "ple.eos_token_id");
        c.ple_eos_token_id = metadata::count(m, p + "ple.eos_token_id", 0);
        if (c.ple_ngram_size < 2 || c.ple_heads_per_ngram <= 0 ||
            uint64_t(c.ple_ngram_size - 1) * uint64_t(c.ple_heads_per_ngram) > uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("inference: PLE n-gram/head dimensions are out of range");
        const uint64_t ple_head_count = uint64_t(c.ple_ngram_size - 1) * uint64_t(c.ple_heads_per_ngram);
        const size_t ple_heads = size_t(ple_head_count);
        if (uint64_t(c.ple_embedding_length) * ple_head_count > uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("inference: PLE embedding dimensions exceed the supported range");
        c.ple_embedding_dim = int(uint64_t(c.ple_embedding_length) * ple_head_count);
        c.ple_layer_multipliers = uint64_array(m, p + "ple.layer_multipliers", size_t(c.ple_ngram_size));
        c.ple_head_offsets = uint64_array(m, p + "ple.head_offsets", ple_heads);
        c.ple_head_vocab_sizes = uint64_array(m, p + "ple.head_vocab_sizes", ple_heads);
    } else if (m.find(p + "ple.ngram_size") || m.find(p + "ple.heads_per_ngram") ||
               m.find(p + "ple.conv_kernel") || m.find(p + "ple.eos_token_id") ||
               m.find(p + "ple.layer_multipliers") || m.find(p + "ple.head_offsets") ||
               m.find(p + "ple.head_vocab_sizes") || m.find(p + "embedding_length_per_layer_input")) {
        throw std::runtime_error("inference: PLE metadata without qwen4exp.ple.layers");
    }
    check_config(c);
    return c;
}

// Released MTP-only exports keep private globals or explicitly borrow the target's, and omit PLE which their graph never reads.
// Normalize headers only after compatibility is proven; tensor order, offsets, file spans and payload owners are untouched.
inline void normalize_drafter(const gguf::GGUFModel& target, gguf::GGUFModel& side) {
    const std::string p = "qwen4exp.";
    const int blocks = metadata::integer(target, p + "block_count");
    const int target_nextn = metadata::count(target, p + "nextn_predict_layers", 0), layers = blocks - target_nextn;
    const auto* count = side.find(p + "nextn_predict_layers");
    if (!count || !metadata::count(side, p + "nextn_predict_layers", 0)) return;
    bool has_block = false, global_embedding = false, global_output = false, normalized_embedding = false, normalized_output = false;
    const std::string pre = "blk." + std::to_string(layers) + ".";
    for (const auto& tensor : side.tensors) {
        if (const auto block = gguf::block_of(tensor.name)) {
            if (*block < size_t(layers)) return; // A complete ordinary draft model, possibly with its own MTP.
            has_block = true;
        }
        global_embedding = global_embedding || tensor.name == "token_embd.weight";
        global_output = global_output || tensor.name == "output.weight";
        normalized_embedding = normalized_embedding || tensor.name == pre + "nextn.token_embd.weight";
        normalized_output = normalized_output || tensor.name == pre + "nextn.output.weight";
    }
    if (!has_block) return;
    const auto refuse = [](const std::string& text) { return std::runtime_error("inference: Qwen4Exp MTP sidecar " + text); };
    const auto* shared_value = side.find(p + "nextn_shared_target_tensors");
    const bool shared = metadata::boolean(side, p + "nextn_shared_target_tensors", false);
    // The established llmx blocks-only pack format already shares the target and carries its exact metadata.
    if (!shared_value && !global_embedding && !global_output && !normalized_embedding && !normalized_output) return;
    if (target_nextn) throw refuse("target already carries an MTP block");
    if (metadata::count(side, p + "nextn_predict_layers", 0) != 1 || metadata::integer(side, p + "block_count") != blocks + 1)
        throw refuse("block_count and nextn_predict_layers must describe one block after the target");
    if (shared && (global_embedding || global_output || normalized_embedding || normalized_output))
        throw refuse("nextn_shared_target_tensors conflicts with private embedding or output tensors");
    if (!shared && !((global_embedding && global_output && !normalized_embedding && !normalized_output) ||
                     (normalized_embedding && normalized_output && !global_embedding && !global_output)))
        throw refuse("requires both private token_embd.weight and output.weight, or nextn_shared_target_tensors true");
    const std::set<std::string> roles{
        "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_q_norm.weight", "attn_k_norm.weight",
        "hc_attn_norm.weight", "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
        "hc_ffn_norm.weight", "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
        "ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
        "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
        "indexer.q_proj.weight", "indexer.k_proj.weight", "indexer.q_norm.weight", "indexer.k_norm.weight",
        "nextn.eh_proj.weight", "nextn.fc_hidden.weight", "nextn.enorm.weight", "nextn.hnorm.weight",
        "nextn.hc_head_norm.weight", "nextn.hc_head_down.weight", "nextn.hc_head_up.weight",
        "nextn.token_embd.weight", "nextn.output.weight"};
    uint64_t vocab = 0;
    for (const auto& tensor : target.tensors)
        if (tensor.name == "token_embd.weight" && tensor.ne.size() >= 2) vocab = tensor.ne[1];
    const uint64_t width = uint64_t(metadata::integer(target, p + "embedding_length"));
    for (const auto& tensor : side.tensors) {
        if (tensor.name == "token_embd.weight" || tensor.name == "output.weight" ||
            tensor.name == pre + "nextn.token_embd.weight" || tensor.name == pre + "nextn.output.weight") {
            if (tensor.ne.size() < 2 || tensor.ne[0] != width || tensor.ne[1] != vocab || !vocab ||
                std::any_of(tensor.ne.begin() + 2, tensor.ne.end(), [](uint64_t n) { return n != 1; }))
                throw refuse("invalid private global shape " + tensor.name);
        } else if (tensor.name.compare(0, pre.size(), pre) || !roles.count(tensor.name.substr(pre.size()))) {
            throw refuse("unsupported tensor " + tensor.name);
        }
    }
    auto normalized = side.kv;
    normalized.erase(std::remove_if(normalized.begin(), normalized.end(), [&](const auto& item) { return item.first == p + "nextn_shared_target_tensors"; }), normalized.end());
    const auto is_ple = [&](const std::string& key) { return key.compare(0, (p + "ple.").size(), p + "ple.") == 0 || key == p + "embedding_length_per_layer_input"; };
    const bool side_ple = std::any_of(side.kv.begin(), side.kv.end(), [&](const auto& item) { return is_ple(item.first); });
    if (!side_ple) for (const auto& item : target.kv) if (is_ple(item.first)) normalized.push_back(item);
    auto find = [&](const std::string& key) -> gguf::MetaValue* {
        for (auto& item : normalized) if (item.first == key) return &item.second;
        return nullptr;
    };
    const auto* target_ratios = target.find(p + "attention.compress_ratios");
    auto* side_ratios = find(p + "attention.compress_ratios");
    if (target_ratios && side_ratios && target_ratios->vtype == gguf::V_ARRAY && side_ratios->vtype == gguf::V_ARRAY &&
        target_ratios->arr.size() == size_t(blocks) && side_ratios->arr.size() == size_t(blocks + 1)) {
        bool legacy_zero_prefix = true;
        for (size_t i = 0; i < size_t(blocks); ++i)
            legacy_zero_prefix = legacy_zero_prefix && metadata::count_value(side_ratios->arr[i], p + "attention.compress_ratios") == 0;
        const int tail = metadata::count_value(side_ratios->arr.back(), p + "attention.compress_ratios");
        if (legacy_zero_prefix && tail > 1)
            std::copy(target_ratios->arr.begin(), target_ratios->arr.end(), side_ratios->arr.begin());
    }
    const auto check = [&](const std::string& key) {
        if (key.compare(0, p.size(), p) || key == p + "block_count" || key == p + "nextn_predict_layers") return;
        const auto* a = target.find(key); const auto* b = find(key);
        if (a && b && gguf::detail::equal_value(*a, *b)) return;
        if (a && b && a->vtype == gguf::V_ARRAY && b->vtype == gguf::V_ARRAY && a->arr.size() == size_t(blocks) && b->arr.size() == size_t(blocks + 1)) {
            bool same = true;
            for (size_t i = 0; i < a->arr.size(); ++i) same = same && gguf::detail::equal_value(a->arr[i], b->arr[i]);
            if (same) return;
        }
        throw refuse("incompatible metadata " + key);
    };
    for (const auto& item : target.kv) check(item.first);
    for (const auto& item : normalized) check(item.first);
    // Validate the resulting model geometry before committing any header edits.
    gguf::GGUFModel header;
    header.kv = normalized;
    (void)read_config(header, p);
    std::vector<std::string> names;
    names.reserve(side.tensors.size());
    for (const auto& tensor : side.tensors)
        names.push_back(tensor.name == "token_embd.weight" ? pre + "nextn.token_embd.weight" :
                        tensor.name == "output.weight" ? pre + "nextn.output.weight" : tensor.name);
    side.kv.swap(normalized);
    for (size_t i = 0; i < names.size(); ++i) side.tensors[i].name.swap(names[i]);
}

// Role ids are local to this architecture.
// The PLE table belongs to its one mixer, while the embedding and final hyper-connection belong to the pass.
enum Role : uint16_t {
    token_embd, output, output_hc_norm, output_hc_down, output_hc_up,
    hc_attn_norm, hc_attn_down, hc_attn_up, hc_attn_inject,
    hc_ffn_norm, hc_ffn_down, hc_ffn_up, hc_ffn_inject,
    attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm, attn_output,
    index_q, index_k, index_q_norm, index_k_norm,
    attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_conv1d, ssm_a, ssm_dt, ssm_norm, ssm_out,
    ple_table, ple_key, ple_value, ple_norm_key, ple_norm_query, ple_norm_conv, ple_conv,
    ffn_gate_inp, ffn_gate_exps, ffn_up_exps, ffn_down_exps,
    ffn_gate_inp_shexp, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp,
    nextn_enorm, nextn_hnorm, nextn_eh_proj, nextn_fc_hidden,
    role_count
};

// Saved recurrent inputs have dedicated slots: neither hyper-connection nor expert execution overwrites them before the runtime saves a marked pass.
enum Slot : size_t {
    residual, mixed, raw, conv, value_gate, alpha_beta, recurrent,
    expert_gate, expert_up, expert_act, scores, expert_ids, expert_weights, shared_gate,
    hc_normed, hc_low, hc_gates, block_output, query, index_query, index_key, selected,
    ple_embeddings, ple_keys, ple_values, ple_queries, ple_gated, ple_normed, ple_output
};

inline std::vector<size_t> slot_widths(const Config& c) {
    const size_t E = size_t(c.n_embd), R = size_t(c.hc_count) * E;
    const size_t Q = size_t(c.n_head) * size_t(c.head_dim), KV = size_t(c.n_head_kv) * size_t(c.head_dim);
    const size_t V = size_t(c.v_heads) * size_t(c.v_dim), C = 2 * size_t(c.k_heads) * size_t(c.k_dim) + V;
    const size_t F = size_t(c.n_expert_used) * size_t(c.n_ff_exp) + size_t(c.n_ff_shexp);
    return {R, E, std::max(C, 2 * Q), std::max(C, KV), std::max(V, KV), 2 * size_t(c.v_heads), std::max(V, Q),
            F, F, F, size_t(c.n_expert), size_t(c.n_expert_used), size_t(c.n_expert_used), 1,
            R, size_t(c.hc_low_rank), R, E, Q,
            std::max(size_t(1), size_t(c.indexer_n_heads) * size_t(c.indexer_head_dim)),
            std::max(size_t(1), size_t(c.indexer_head_dim)),
            std::max(size_t(1), size_t(c.indexer_budget) + size_t(std::max({c.indexer_compress_ratio, c.nextn_compress_ratio, 1}) - 1)),
            std::max(size_t(1), size_t(c.ple_embedding_dim)), R, E, R, R, R, R};
}

class Qwen4Exp final : public Architecture {
public:
    explicit Qwen4Exp(const Config& cfg, uint32_t vocab) : cfg_(cfg), vocab_(vocab) {
        check_config(cfg_);
        if (!vocab_ || vocab_ > uint32_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("inference: invalid vocabulary dimension");
        if (cfg_.n_expert_used > 256)
            throw std::runtime_error("inference: more than 256 experts per token are unsupported");
        if (cfg_.has_ple) ple_rows(cfg_, vocab_, nullptr, 0, nullptr, 0, nullptr, 0);
    }

    ModelPlan plan(const TensorIndex& tensors) const override {
        const Config& c = cfg_;
        const TensorView& embedding = tensors.view(tensors.at("token_embd.weight"));
        if (embedding.shape.size() < 2 || embedding.shape[1] != vocab_)
            throw std::runtime_error("inference: invalid vocabulary dimension");
        const uint64_t E = uint64_t(c.n_embd), R = uint64_t(c.hc_count) * E, L = uint64_t(c.hc_low_rank);
        ModelPlan p;
        p.role_ids = role_count;
        p.vocab = vocab_;
        p.pass = {{token_embd, Part::embed, RoleKind::gather, "token_embd.weight", "", E, p.vocab},
                  {output, Part::head, RoleKind::matrix, "output.weight", "token_embd.weight", E, p.vocab},
                  {output_hc_norm, Part::head, RoleKind::norm, "output_hc_norm.weight", "", R},
                  {output_hc_down, Part::head, RoleKind::matrix, "output_hc_down.weight", "", R, L},
                  {output_hc_up, Part::head, RoleKind::matrix, "output_hc_up.weight", "", L, R}};
        p.pass_ops = {{Part::head, backend::Op::group_rms_norm}, {Part::head, backend::Op::silu_scaled}, {Part::head, backend::Op::mix_streams}};
        blocks::shard(p.pass, output, Axis::rows, {{p.vocab, 1, 1, "vocabulary rows"}});
        p.layers.resize(size_t(c.n_layer));
        for (int l = 0; l < c.n_layer; ++l) {
            LayerPlan& layer = p.layers[size_t(l)];
            const std::string pre = "blk." + std::to_string(l) + ".";
            const bool has_ple = c.has_ple && l == c.ple_layer;
            layer.kind = c.layers[size_t(l)] == qwen_sparse_attention ? qsa_kind : (has_ple ? ple_kind : linear_kind);
            hyper_roles(layer, pre, Part::mixer, hc_attn_norm, hc_attn_down, hc_attn_up, hc_attn_inject, "hc_attn", Stream::none);
            if (c.layers[size_t(l)] == qwen_sparse_attention) {
                if (tensors.find(pre + "attn_qkv.weight"))
                    throw std::runtime_error("inference: a QSA layer holds the linear-attention tensor " + pre + "attn_qkv.weight");
                qsa_roles(layer, pre, size_t(c.indexer_compress_ratio));
            } else {
                if (tensors.find(pre + "attn_q.weight"))
                    throw std::runtime_error("inference: a linear-attention layer holds the QSA tensor " + pre + "attn_q.weight");
                linear_roles(layer, pre);
                if (has_ple) ple_roles(layer, tensors, pre);
            }
            if (!has_ple && tensors.find(pre + "ple_key.weight"))
                throw std::runtime_error("inference: PLE tensors on an undeclared layer " + pre);
            hyper_roles(layer, pre, Part::ffn, hc_ffn_norm, hc_ffn_down, hc_ffn_up, hc_ffn_inject, "hc_ffn", Stream::copy);
            expert_roles(layer, tensors, pre);
        }
        p.context_length = size_t(c.context_length);
        p.residual = size_t(R);
        // A group sums a part's block, and each member injects the sum into its streams (join).
        p.partial = size_t(c.n_embd);
        p.slots = slot_widths(c);
        p.index_width = std::max(size_t(c.hc_count), c.ple_head_offsets.size());
        p.kv_heads = size_t(c.n_head_kv);
        p.head_dim = size_t(c.head_dim);
        p.state = {size_t(c.k_heads), size_t(c.v_heads), size_t(c.k_dim), size_t(c.v_dim)};
        p.tables = {size_t(c.context_length) * size_t(c.rope_dim / 2),
                    size_t(c.context_length) * size_t(c.rope_dim / 2), size_t(R)};
        // Images take Qwen 3.5's interleaved positions, a pooled group turning at its first row's (Step::group_pos), and an embedding row fills every residual stream.
        p.image_streams = blocks::interleaved_streams(c.rope_sections, size_t(c.rope_dim / 2));
        p.embed_copies = size_t(c.hc_count);
        return p;
    }

    void fill_tables(std::vector<std::vector<float>>& tables) const override {
        const size_t half = size_t(cfg_.rope_dim / 2);
        for (size_t i = 0; i < half; ++i) {
            const float freq = 1.0f / std::pow(cfg_.rope_theta, 2.0f * float(i) / float(cfg_.rope_dim));
            for (size_t pos = 0; pos < size_t(cfg_.context_length); ++pos) {
                tables[0][pos * half + i] = std::cos(float(pos) * freq);
                tables[1][pos * half + i] = std::sin(float(pos) * freq);
            }
        }
        std::fill(tables[2].begin(), tables[2].end(), 0.0f);
    }

    void embed(const Step& s, const uint32_t* ids) const override {
        const size_t H = size_t(cfg_.hc_count), count = backend::size_mul(s.rows, H);
        require_indices(s, count);
        for (size_t r = 0; r < s.rows; ++r)
            for (size_t h = 0; h < H; ++h) s.indices[r * H + h] = ids[r];
        const Weight& table = s.w[token_embd];
        s.b.embed(s.x, table.operand(), table.nin, table.nout, s.indices, count);
    }

    void lookup_ids(const Step& s) const override {
        const size_t heads = cfg_.ple_head_offsets.size(), count = backend::size_mul(s.rows, heads);
        require_indices(s, count);
        if (!s.tokens || !s.states || s.previous_count != size_t(cfg_.ple_ngram_size - 1) || !s.previous_tokens)
            throw std::runtime_error("inference: Qwen4Exp PLE token history is missing");
        size_t first = 0;
        for (size_t v = 0; v < s.n_views; ++v) {
            const size_t n = s.states[v].nq;
            if (n > s.rows - first) throw std::runtime_error("inference: Qwen4Exp PLE views exceed the pass");
            ple_rows(cfg_, vocab_, s.previous_tokens + v * s.previous_count, s.previous_count,
                     s.tokens + first, n, reinterpret_cast<int32_t*>(s.indices + first * heads), n * heads);
            first += n;
        }
        if (first != s.rows) throw std::runtime_error("inference: Qwen4Exp PLE views do not cover the pass");
    }

    void mixer(const Step& s) const override {
        if (s.kind == ple_kind) apply_ple(s);
        mix(cfg_, s, {s.w[hc_attn_norm], s.w[hc_attn_down], s.w[hc_attn_up]}, s.slot(mixed), mix_scratch(s));
        Step block = s;
        block.x = s.width > 1 ? s.partial : s.slot(block_output);
        if (s.kind == qsa_kind || s.kind == mtp_qsa_kind || s.kind == dense_kind) sparse_attention(block);
        else linear_attention(block);
        inject(cfg_, s, s.w[hc_attn_inject], s.slot(hc_normed), block.x, s.slot(hc_gates));
    }

    void ffn(const Step& s) const override {
        mix(cfg_, s, {s.w[hc_ffn_norm], s.w[hc_ffn_down], s.w[hc_ffn_up]}, s.slot(mixed), mix_scratch(s));
        Step block = s;
        block.x = s.width > 1 ? s.partial : s.slot(block_output);
        // The expert products accumulate from zero into the block, a member's partial rows on a group.
        block.width = 1;
        const Weight* w = s.w;
        const blocks::SharedExpert shared{w[ffn_gate_shexp], w[ffn_up_shexp], w[ffn_down_shexp], &w[ffn_gate_inp_shexp]};
        blocks::moe_step(block, w[ffn_gate_inp], w[ffn_gate_exps], w[ffn_up_exps], w[ffn_down_exps], size_t(cfg_.n_expert_used), cfg_.expert_norm,
                         s.slot(mixed), s.slot(expert_gate), s.slot(expert_up), s.slot(expert_act), s.slot(scores), s.slot(expert_ids), s.slot(expert_weights),
                         &shared, s.slot(shared_gate), s.tables[2].get());
        inject(cfg_, s, w[hc_ffn_inject], s.slot(hc_normed), block.x, s.slot(hc_gates));
    }

    // On a group, the summed block of the part just run, injected with the gates that part left in hc_gates.
    void join(const Step& s) const override {
        s.b.inject_streams(s.x, s.partial, s.slot(hc_gates), s.rows, size_t(cfg_.hc_count), size_t(cfg_.n_embd));
    }

    void head(const HeadStep& s) const override {
        Step compact = s;
        compact.x = s.slot(ple_queries);
        compact.rows = s.want;
        compact.runs = s.head_runs;
        s.b.gather_rows(compact.x, s.x, size_t(cfg_.hc_count) * size_t(cfg_.n_embd), s.pick, s.want);
        mix(cfg_, compact, {s.w[output_hc_norm], s.w[output_hc_down], s.w[output_hc_up]}, s.slot(mixed), mix_scratch(s));
        const Weight& out = s.w[output];
        s.b.matmul_logits(out.operand(), s.slot(mixed), s.logits, out.nin, out.nout, s.want, s.head_runs, s.dtype);
    }

    // The MTP block uses the target's raw HC residual, its own per-stream input norms and its own HC head.
    void plan_drafter(const TensorIndex& tensors, ModelPlan& p) const override {
        const Config& c = cfg_;
        if (!c.n_nextn) throw std::runtime_error("inference: the file carries no MTP block for an embedded drafter");
        const uint64_t E = uint64_t(c.n_embd), R = E * uint64_t(c.hc_count), L = uint64_t(c.hc_low_rank);
        const std::string pre = "blk." + std::to_string(c.n_layer) + ".";
        if (tensors.find(pre + "attn_qkv.weight"))
            throw std::runtime_error("inference: the MTP block holds the linear-attention tensor " + pre + "attn_qkv.weight");
        const TensorView& projection = tensors.view(tensors.at(pre + "nextn.eh_proj.weight"));
        const bool split = !projection.shape.empty() && projection.shape[0] == E;
        if (!split && tensors.find(pre + "nextn.fc_hidden.weight"))
            throw std::runtime_error("inference: fused MTP projection also holds " + pre + "nextn.fc_hidden.weight");
        LayerPlan d;
        d.kind = c.nextn_compress_ratio ? mtp_qsa_kind : dense_kind;
        const bool own_embedding = tensors.find(pre + "nextn.token_embd.weight").has_value();
        if (own_embedding != tensors.find(pre + "nextn.output.weight").has_value())
            throw std::runtime_error("inference: MTP private embedding and output must both be present");
        d.roles = {{token_embd, Part::draft, RoleKind::gather, own_embedding ? pre + "nextn.token_embd.weight" : "token_embd.weight", "", E, p.vocab},
                   {output, Part::draft, RoleKind::matrix, own_embedding ? pre + "nextn.output.weight" : "output.weight", own_embedding ? "" : "token_embd.weight", E, p.vocab},
                   {nextn_enorm, Part::draft, RoleKind::norm, pre + "nextn.enorm.weight", "", E},
                   {nextn_hnorm, Part::draft, RoleKind::norm, pre + "nextn.hnorm.weight", "", R},
                   {nextn_eh_proj, Part::draft, RoleKind::matrix, pre + "nextn.eh_proj.weight", "", split ? E : 2 * E, E},
                   {output_hc_norm, Part::draft, RoleKind::norm, pre + "nextn.hc_head_norm.weight", "", R},
                   {output_hc_down, Part::draft, RoleKind::matrix, pre + "nextn.hc_head_down.weight", "", R, L},
                   {output_hc_up, Part::draft, RoleKind::matrix, pre + "nextn.hc_head_up.weight", "", L, R}};
        if (split) d.roles.push_back({nextn_fc_hidden, Part::draft, RoleKind::matrix, pre + "nextn.fc_hidden.weight", "", E, E});
        hyper_roles(d, pre, Part::draft, hc_attn_norm, hc_attn_down, hc_attn_up, hc_attn_inject, "hc_attn", Stream::none);
        qsa_roles(d, pre, size_t(c.nextn_compress_ratio));
        hyper_roles(d, pre, Part::draft, hc_ffn_norm, hc_ffn_down, hc_ffn_up, hc_ffn_inject, "hc_ffn", Stream::none);
        expert_roles(d, tensors, pre);
        for (infer::Role& r : d.roles) { r.part = Part::draft; r.stream = Stream::none; }
        for (OpUse& op : d.ops) op.part = Part::draft;
        d.ops.push_back({Part::draft, backend::Op::argmax_rows});
        d.ops.push_back({Part::draft, backend::Op::embed_ids});
        p.draft_h = p.slots.size();
        p.draft_x = p.draft_h + 3;
        p.slots.insert(p.slots.end(), {size_t(R), 2 * size_t(R), 2 * size_t(R), size_t(R), size_t(E)});
        p.index_width = std::max(p.index_width, 2 * size_t(c.hc_count));
        p.drafter = std::move(d);
    }

    void draft_rows(const DraftRowsStep& s) const override {
        const size_t E = size_t(cfg_.n_embd), R = E * size_t(cfg_.hc_count), base = draft_slot();
        const backend::Slice h = s.slot(base), prev{s.slot(base + 1).buffer, s.slot(base + 1).offset + s.rows * R};
        s.b.copy(*h.buffer, h.offset * sizeof(float), *s.x.buffer, s.x.offset * sizeof(float), s.rows * R * sizeof(float));
        for (size_t e = 0, first = 0; e < s.runs.n; ++e) {
            const size_t end = s.runs.runs[e].end;
            s.b.copy(*prev.buffer, (prev.offset + first * R) * sizeof(float), *s.carry[e].buffer, s.carry[e].offset * sizeof(float), R * sizeof(float));
            if (end > first + 1)
                s.b.copy(*prev.buffer, (prev.offset + (first + 1) * R) * sizeof(float), *h.buffer, (h.offset + first * R) * sizeof(float), (end - first - 1) * R * sizeof(float));
            first = end;
        }
        s.b.embed(s.slot(base + 4), s.w[token_embd].operand(), E, vocab_, s.ids, s.rows);
        draft_input(s);
        Step layer = s;
        layer.x = s.slot(base + 3);
        mix(cfg_, layer, {s.w[hc_attn_norm], s.w[hc_attn_down], s.w[hc_attn_up]}, s.slot(mixed), mix_scratch(s));
        attention_cache(layer);
    }

    void draft(const DraftStep& s) const override {
        const size_t R = size_t(cfg_.n_embd) * size_t(cfg_.hc_count), base = draft_slot();
        Step layer = s;
        layer.x = s.slot(base + 3);
        if (s.phase <= 0) {
            const backend::Slice prev{s.slot(base + 1).buffer, s.slot(base + 1).offset + s.rows * R};
            if (s.prev_rows) s.b.gather_rows(prev, s.prev, R, s.prev_rows, s.rows);
            else s.b.copy(*prev.buffer, prev.offset * sizeof(float), *s.prev.buffer, s.prev.offset * sizeof(float), s.rows * R * sizeof(float));
            s.b.embed_ids(s.slot(base + 4), s.w[token_embd].operand(), size_t(cfg_.n_embd), vocab_, s.id, s.rows);
            draft_input(s);
            mixer(layer);
        }
        if (s.phase < 0 || s.phase == 1) ffn(layer);
        if (s.phase < 0 || s.phase == 2) {
            s.b.copy(*s.out.buffer, s.out.offset * sizeof(float), *layer.x.buffer, layer.x.offset * sizeof(float), s.rows * R * sizeof(float));
            mix(cfg_, layer, {s.w[output_hc_norm], s.w[output_hc_down], s.w[output_hc_up]}, s.slot(mixed), mix_scratch(s));
            s.b.matmul_logits(s.w[output].operand(), s.slot(mixed), s.logits, size_t(cfg_.n_embd), vocab_, s.rows, s.runs, s.dtype);
            s.b.argmax_rows(s.next, s.logits, s.rows, vocab_, s.id);
        }
    }

    // Only the accepted rows' two convolutions and GDN recurrence are replayed.
    // Projections, lookup, stream mixing, attention and experts are not rerun.
    void recur(const Step& s) const override {
        if (s.phase != 1) {
            s.b.causal_conv_silu(s.slot(conv), s.slot(raw), s.w[ssm_conv1d].slice(), s.state_layer, s.states, s.n_views);
            if (s.kind == ple_kind) ple_recur(cfg_, s, s.w[ple_conv], s.slot(ple_output), ple_scratch(s));
        }
        if (s.phase != 0) {
            const backend::Slice alpha = s.slot(alpha_beta);
            const backend::Slice beta{alpha.buffer, alpha.offset + s.rows * s.w[ssm_alpha].nout};
            s.b.gated_delta_rule(s.slot(recurrent), s.slot(conv), alpha, beta, s.w[ssm_a].slice(), s.w[ssm_dt].slice(),
                                s.state_layer, s.states, s.n_views);
        }
    }

private:
    static constexpr uint8_t linear_kind = 0, qsa_kind = 1, ple_kind = 2, dense_kind = 3, mtp_qsa_kind = 4;
    Config cfg_;
    uint32_t vocab_;

    static constexpr size_t draft_slot() { return ple_output + 1; }

    // Normalize the token once and each raw hidden stream separately, then project each [token; hidden] pair with one shared weight.
    void draft_input(const Step& s) const {
        const size_t E = size_t(cfg_.n_embd), H = size_t(cfg_.hc_count), R = E * H, base = draft_slot(), n = s.rows;
        const backend::Slice pair = s.slot(base + 1), side = s.slot(base + 2), out = s.slot(base + 3), embedding = s.slot(base + 4);
        const backend::Slice prev{pair.buffer, pair.offset + n * R};
        s.b.rms_norm_rows(embedding, embedding, s.w[nextn_enorm].slice(), n, E, E, cfg_.rms_eps, s.runs);
        s.b.group_rms_norm(prev, prev, s.w[nextn_hnorm].slice(), n, H, E, cfg_.rms_eps, s.runs);
        require_indices(s, 2 * n * H);
        for (size_t row = 0; row < n * H; ++row) s.indices[row] = uint32_t(row / H);
        s.b.gather_rows(pair, embedding, E, s.indices, n * H);
        auto& runs = *s.scratch;
        runs.clear();
        for (size_t i = 0; i < s.runs.n; ++i) runs.push_back({s.runs.runs[i].end * H, s.runs.runs[i].extent});
        const backend::RowRuns streams{runs.data(), runs.size()};
        const Weight& projection = s.w[nextn_eh_proj];
        if (projection.nin == E) {
            s.b.matmul(projection.operand(), pair, out, E, E, n * H, streams, s.dtype);
            const Weight& hidden = s.w[nextn_fc_hidden];
            s.b.matmul(hidden.operand(), prev, side, E, E, n * H, streams, s.dtype);
            s.b.add(out, side, n * R);
        } else {
            for (size_t row = 0; row < n * H; ++row) {
                s.indices[2 * row] = uint32_t(row);
                s.indices[2 * row + 1] = uint32_t(n * H + row);
            }
            s.b.gather_rows(side, pair, E, s.indices, 2 * n * H);
            s.b.matmul(projection.operand(), side, out, 2 * E, E, n * H, streams, s.dtype);
        }
    }

    static MixScratch mix_scratch(const Step& s) { return {s.slot(hc_normed), s.slot(hc_low), s.slot(hc_gates)}; }
    static PleScratch ple_scratch(const Step& s) {
        return {s.slot(ple_keys), s.slot(ple_values), s.slot(ple_queries), s.slot(ple_gated), s.slot(ple_normed)};
    }
    static void require_indices(const Step& s, size_t count) {
        if (count > s.indices_size || (count && !s.indices))
            throw std::runtime_error("inference: Qwen4Exp index scratch is too small");
    }

    void hyper_roles(LayerPlan& layer, const std::string& pre, Part part, uint16_t norm, uint16_t down,
                     uint16_t up, uint16_t injection, const std::string& stem, Stream stream) const {
        const uint64_t E = uint64_t(cfg_.n_embd), R = uint64_t(cfg_.hc_count) * E, L = uint64_t(cfg_.hc_low_rank);
        layer.roles.insert(layer.roles.end(), {
            {norm, part, RoleKind::norm, pre + stem + "_norm.weight", "", R, 1, 0, stream},
            {down, part, RoleKind::matrix, pre + stem + "_down.weight", "", R, L, 0, stream},
            {up, part, RoleKind::matrix, pre + stem + "_up.weight", "", L, R, 0, stream},
            {injection, part, RoleKind::matrix, pre + stem + "_inject.weight", "", R, uint64_t(cfg_.hc_count), 0, stream}});
        for (backend::Op op : {backend::Op::group_rms_norm, backend::Op::silu_scaled, backend::Op::mix_streams, backend::Op::inject_streams})
            layer.ops.push_back({part, op});
    }

    void linear_roles(LayerPlan& layer, const std::string& pre) const {
        const uint64_t E = uint64_t(cfg_.n_embd), Hv = uint64_t(cfg_.v_heads), V = Hv * uint64_t(cfg_.v_dim);
        const uint64_t C = 2 * uint64_t(cfg_.k_heads) * uint64_t(cfg_.k_dim) + V;
        layer.cache = Cache::state;
        layer.roles.insert(layer.roles.end(), {
            {attn_qkv, Part::mixer, RoleKind::matrix, pre + "attn_qkv.weight", "", E, C},
            {attn_gate, Part::mixer, RoleKind::matrix, pre + "attn_gate.weight", "", E, V},
            {ssm_alpha, Part::mixer, RoleKind::matrix, pre + "ssm_alpha.weight", "", E, Hv},
            {ssm_beta, Part::mixer, RoleKind::matrix, pre + "ssm_beta.weight", "", E, Hv},
            {ssm_conv1d, Part::mixer, RoleKind::table, pre + "ssm_conv1d.weight", "", backend::kConvTaps, C},
            {ssm_a, Part::mixer, RoleKind::norm, pre + "ssm_a", "", Hv},
            {ssm_dt, Part::mixer, RoleKind::norm, pre + "ssm_dt.bias", "", Hv},
            {ssm_norm, Part::mixer, RoleKind::norm, pre + "ssm_norm.weight", "", uint64_t(cfg_.v_dim)},
            {ssm_out, Part::mixer, RoleKind::matrix, pre + "ssm_out.weight", "", V, E}});
        for (backend::Op op : {backend::Op::causal_conv_silu, backend::Op::gated_delta_rule, backend::Op::gated_rms_norm})
            layer.ops.push_back({Part::mixer, op});
        layer.saved = {{raw, size_t(C)}, {alpha_beta, size_t(Hv)}, {alpha_beta, size_t(Hv), 1}};
        layer.recur_writes = {conv, recurrent};
        layer.recur_phases = 2;
        const uint64_t Hk = uint64_t(cfg_.k_heads), tiles = Hv / Hk;
        const ShardSection k{Hk, uint64_t(cfg_.k_dim), 1, "K heads"};
        const ShardSection v{Hk, uint64_t(cfg_.v_dim), tiles, "V heads of each tile"};
        const ShardSection one{Hk, 1, tiles, "V heads of each tile"};
        blocks::shard(layer.roles, attn_qkv, Axis::rows, {k, k, v});
        blocks::shard(layer.roles, ssm_conv1d, Axis::rows, {k, k, v});
        blocks::shard(layer.roles, attn_gate, Axis::rows, {v});
        blocks::shard(layer.roles, ssm_alpha, Axis::rows, {one});
        blocks::shard(layer.roles, ssm_beta, Axis::rows, {one});
        blocks::shard(layer.roles, ssm_a, Axis::columns, {one});
        blocks::shard(layer.roles, ssm_dt, Axis::columns, {one});
        blocks::shard(layer.roles, ssm_out, Axis::columns, {v});
    }

    void ple_roles(LayerPlan& layer, const TensorIndex& tensors, const std::string& pre) const {
        const uint64_t E = uint64_t(cfg_.n_embd), R = uint64_t(cfg_.hc_count) * E, D = uint64_t(cfg_.ple_embedding_dim);
        const TensorView& table = tensors.view(tensors.at("per_layer_token_embd.weight"));
        if (table.shape.size() < 2 || !table.shape[1] || table.shape[1] > uint64_t(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error("inference: invalid PLE table row count");
        for (size_t h = 0; h < cfg_.ple_head_offsets.size(); ++h)
            if (cfg_.ple_head_offsets[h] + cfg_.ple_head_vocab_sizes[h] > table.shape[1])
                throw std::runtime_error("inference: PLE head range exceeds the embedding table");
        layer.roles.insert(layer.roles.end(), {
            {ple_table, Part::mixer, RoleKind::gather, "per_layer_token_embd.weight", "", uint64_t(cfg_.ple_embedding_length), table.shape[1]},
            {ple_key, Part::mixer, RoleKind::matrix, pre + "ple_key.weight", "", D, R},
            {ple_value, Part::mixer, RoleKind::matrix, pre + "ple_value.weight", "", D, E},
            {ple_norm_key, Part::mixer, RoleKind::norm, pre + "ple_norm_key.weight", "", R},
            {ple_norm_query, Part::mixer, RoleKind::norm, pre + "ple_norm_query.weight", "", R},
            {ple_norm_conv, Part::mixer, RoleKind::norm, pre + "ple_norm_conv.weight", "", R},
            {ple_conv, Part::mixer, RoleKind::table, pre + "ple_conv1d.weight", "", uint64_t(cfg_.ple_conv_kernel), R}});
        layer.ops.push_back({Part::mixer, backend::Op::signed_sqrt_gate});
        layer.ops.push_back({Part::mixer, backend::Op::dilated_conv_silu_add});
        layer.state_tail = backend::size_mul(backend::size_mul(size_t(cfg_.ple_conv_kernel - 1), size_t(cfg_.ple_ngram_size)), size_t(R));
        layer.token_history = size_t(cfg_.ple_ngram_size - 1);
        layer.token_padding = uint32_t(cfg_.ple_eos_token_id);
        layer.lookup = Lookup{ple_table, cfg_.ple_head_offsets.size()};
        layer.saved.insert(layer.saved.end(), {{ple_normed, size_t(R), 0, true}, {ple_gated, size_t(R), 0, true}});
        layer.recur_writes.push_back(ple_output);
    }

    // A QSA layer, whose index keys and the pooled rows of their groups of `compression` the KV's auxiliary blocks keep; 0 is the MTP block's dense attention.
    void qsa_roles(LayerPlan& layer, const std::string& pre, size_t compression) const {
        const bool sparse = compression != 0;
        const uint64_t E = uint64_t(cfg_.n_embd), D = uint64_t(cfg_.head_dim), Q = uint64_t(cfg_.n_head) * D;
        const uint64_t KV = uint64_t(cfg_.n_head_kv) * D, I = uint64_t(cfg_.indexer_head_dim), IQ = uint64_t(cfg_.indexer_n_heads) * I;
        layer.cache = Cache::kv;
        layer.kv_aux = sparse ? size_t(I) : 0;
        layer.kv_pool = compression;
        layer.roles.insert(layer.roles.end(), {
            {attn_q, Part::mixer, RoleKind::matrix, pre + "attn_q.weight", "", E, 2 * Q},
            {attn_k, Part::mixer, RoleKind::matrix, pre + "attn_k.weight", "", E, KV},
            {attn_v, Part::mixer, RoleKind::matrix, pre + "attn_v.weight", "", E, KV},
            {attn_q_norm, Part::mixer, RoleKind::norm, pre + "attn_q_norm.weight", "", D},
            {attn_k_norm, Part::mixer, RoleKind::norm, pre + "attn_k_norm.weight", "", D},
            {attn_output, Part::mixer, RoleKind::matrix, pre + "attn_output.weight", "", Q, E}});
        for (backend::Op op : {backend::Op::norm_rope_partial, backend::Op::sigmoid_mul})
            layer.ops.push_back({Part::mixer, op});
        if (sparse) {
            layer.roles.insert(layer.roles.end(), {
            {index_q, Part::mixer, RoleKind::matrix, pre + "indexer.q_proj.weight", "", E, IQ},
            {index_k, Part::mixer, RoleKind::matrix, pre + "indexer.k_proj.weight", "", E, I},
            {index_q_norm, Part::mixer, RoleKind::norm, pre + "indexer.q_norm.weight", "", I},
            {index_k_norm, Part::mixer, RoleKind::norm, pre + "indexer.k_norm.weight", "", I}});
            for (backend::Op op : {backend::Op::select_kv, backend::Op::sparse_attention})
                layer.ops.push_back({Part::mixer, op});
        }
        const ShardSection heads{uint64_t(cfg_.n_head), 2 * D, 1, "heads"};
        const ShardSection kv{uint64_t(cfg_.n_head_kv), D, 1, "KV heads", true};
        blocks::shard(layer.roles, attn_q, Axis::rows, {heads});
        blocks::shard(layer.roles, attn_k, Axis::rows, {kv});
        blocks::shard(layer.roles, attn_v, Axis::rows, {kv});
        blocks::shard(layer.roles, attn_output, Axis::columns, {{uint64_t(cfg_.n_head), D, 1, "heads"}});
    }

    void expert_roles(LayerPlan& layer, const TensorIndex& tensors, const std::string& pre) const {
        const uint64_t E = uint64_t(cfg_.n_embd), X = uint64_t(cfg_.n_expert), F = uint64_t(cfg_.n_ff_exp), S = uint64_t(cfg_.n_ff_shexp);
        layer.routed = true;
        layer.roles.insert(layer.roles.end(), {
            {ffn_gate_inp, Part::ffn, RoleKind::matrix, pre + "ffn_gate_inp.weight", "", E, X, 0, Stream::copy},
            {ffn_gate_exps, Part::ffn, RoleKind::experts, pre + "ffn_gate_exps.weight", "", E, F, X, Stream::window},
            {ffn_up_exps, Part::ffn, RoleKind::experts, pre + "ffn_up_exps.weight", "", E, F, X, Stream::window},
            {ffn_down_exps, Part::ffn, RoleKind::experts, pre + "ffn_down_exps.weight", "", F, E, X, Stream::window},
            {ffn_gate_inp_shexp, Part::ffn, RoleKind::norm, pre + "ffn_gate_inp_shexp.weight", "", E, 1, 0, Stream::copy},
            {ffn_gate_shexp, Part::ffn, RoleKind::matrix, pre + "ffn_gate_shexp.weight", "", E, S, 0, Stream::copy},
            {ffn_up_shexp, Part::ffn, RoleKind::matrix, pre + "ffn_up_shexp.weight", "", E, S, 0, Stream::copy},
            {ffn_down_shexp, Part::ffn, RoleKind::matrix, pre + "ffn_down_shexp.weight", "", S, E, 0, Stream::copy}});
        layer.ops.push_back({Part::ffn, backend::Op::sigmoid_mul});
        blocks::routed_ops(layer, tensors, ffn_gate_exps, ffn_up_exps);
        blocks::shard_experts(layer.roles, tensors, ffn_gate_exps, ffn_up_exps, ffn_down_exps, F, X);
        blocks::shard_experts(layer.roles, tensors, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp, S, 1);
    }

    void apply_ple(const Step& s) const {
        const size_t heads = cfg_.ple_head_offsets.size(), count = backend::size_mul(s.rows, heads);
        const Weight* w = s.w;
        const Weight& table = w[ple_table];
        if (s.lookup) {
            if (!s.lookup->data.buffer || !s.lookup->rows || (count && !s.lookup->ids))
                throw std::runtime_error("inference: Qwen4Exp prepared lookup is incomplete");
            auto weight = table.operand();
            weight.data = s.lookup->data;
            s.b.embed(s.slot(ple_embeddings), weight, table.nin, s.lookup->rows, s.lookup->ids, count);
        } else {
            lookup_ids(s);
            s.b.embed(s.slot(ple_embeddings), table.operand(), table.nin, table.nout, s.indices, count);
        }
        ple(cfg_, s, {w[ple_key], w[ple_value], w[ple_norm_key], w[ple_norm_query], w[ple_norm_conv], w[ple_conv]},
            s.slot(ple_embeddings), s.slot(ple_output), ple_scratch(s));
        s.b.add(s.x, s.slot(ple_output), s.rows * size_t(cfg_.hc_count) * size_t(cfg_.n_embd));
    }

    void linear_attention(const Step& s) const {
        const Weight* w = s.w;
        const size_t E = size_t(cfg_.n_embd), Hv = w[ssm_alpha].nout;
        const backend::Slice alpha = s.slot(alpha_beta), beta{alpha.buffer, alpha.offset + s.rows * Hv};
        s.b.matmul(w[attn_qkv].operand(), s.slot(mixed), s.slot(raw), E, w[attn_qkv].nout, s.rows, s.runs, s.dtype);
        s.b.matmul_group({blocks::projection(w[attn_gate], s.slot(value_gate)), blocks::projection(w[ssm_alpha], alpha), blocks::projection(w[ssm_beta], beta)},
                         s.slot(mixed), E, s.rows, s.runs, s.dtype);
        // PLE has already updated its separate tail.
        // Calling only the GDN phases here avoids advancing that convolution twice in a forward.
        s.b.causal_conv_silu(s.slot(conv), s.slot(raw), w[ssm_conv1d].slice(), s.state_layer, s.states, s.n_views);
        s.b.gated_delta_rule(s.slot(recurrent), s.slot(conv), alpha, beta, w[ssm_a].slice(), w[ssm_dt].slice(),
                            s.state_layer, s.states, s.n_views);
        s.b.gated_rms_norm(s.slot(recurrent), s.slot(recurrent), s.slot(value_gate), w[ssm_norm].slice(),
                          s.rows, Hv, size_t(cfg_.v_dim), cfg_.rms_eps, s.runs, backend::NormGate::sigmoid);
        s.b.matmul(w[ssm_out].operand(), s.slot(recurrent), s.x, w[ssm_out].nin, E, s.rows, s.runs, s.dtype);
    }

    // Context rows need only the projections that the next draft reads from its cache.
    void attention_cache(const Step& s) const {
        const Weight* w = s.w;
        const size_t E = size_t(cfg_.n_embd), D = size_t(cfg_.head_dim), HK = w[attn_k].nout / D;
        s.b.matmul_group({blocks::projection(w[attn_k], s.slot(conv)), blocks::projection(w[attn_v], s.slot(value_gate))},
                         s.slot(mixed), E, s.rows, s.runs, s.dtype);
        const backend::CSlice cos{s.tables[0].get(), 0}, sin{s.tables[1].get(), 0};
        s.b.norm_rope_partial(s.slot(conv), s.slot(conv), s.rows, HK * D, D, HK, D, size_t(cfg_.rope_dim),
                              w[attn_k_norm].slice(), cfg_.rms_eps, cos, sin, s.pos);
        backend::CSlice auxiliary;
        if (s.kind == mtp_qsa_kind) {
            s.b.matmul(w[index_k].operand(), s.slot(mixed), s.slot(index_key), E, size_t(cfg_.indexer_head_dim), s.rows, s.runs, s.dtype);
            auxiliary = s.slot(index_key);
        }
        s.b.kv_write(s.kv_layer, s.views, s.n_views, s.slot(conv), s.slot(value_gate), auxiliary);
        // The groups these rows complete get their pooled rows now, as a selection pools its own rows' groups, so the chain's selections read every complete group.
        if (s.kind == mtp_qsa_kind) {
            const size_t I = size_t(cfg_.indexer_head_dim), IH = size_t(cfg_.indexer_n_heads);
            const backend::Backend::IndexArgs args{IH, I, size_t(cfg_.rope_dim), size_t(cfg_.nextn_compress_ratio), size_t(cfg_.indexer_budget), cfg_.rms_eps};
            s.b.pool_kv(s.kv_layer, s.views, s.n_views, args, w[index_k_norm].slice(), cos, sin, s.group_pos);
        }
    }

    void sparse_attention(const Step& s) const {
        const Weight* w = s.w;
        const size_t E = size_t(cfg_.n_embd), D = size_t(cfg_.head_dim), H = w[attn_q].nout / (2 * D), HK = w[attn_k].nout / D;
        const size_t I = size_t(cfg_.indexer_head_dim), IH = size_t(cfg_.indexer_n_heads), R = size_t(cfg_.rope_dim);
        const backend::CSlice cos{s.tables[0].get(), 0}, sin{s.tables[1].get(), 0};
        s.b.matmul_group({blocks::projection(w[attn_q], s.slot(raw)), blocks::projection(w[attn_k], s.slot(conv)),
                          blocks::projection(w[attn_v], s.slot(value_gate))}, s.slot(mixed), E, s.rows, s.runs, s.dtype);
        if (s.kind != dense_kind)
            s.b.matmul_group({blocks::projection(w[index_q], s.slot(index_query)),
                              blocks::projection(w[index_k], s.slot(index_key))}, s.slot(mixed), E, s.rows, s.runs, s.dtype);
        s.b.norm_rope_partial(s.slot(query), s.slot(raw), s.rows, 2 * H * D, 2 * D, H, D, R,
                              w[attn_q_norm].slice(), cfg_.rms_eps, cos, sin, s.pos, s.streams);
        s.b.norm_rope_partial(s.slot(conv), s.slot(conv), s.rows, HK * D, D, HK, D, R,
                              w[attn_k_norm].slice(), cfg_.rms_eps, cos, sin, s.pos, s.streams);
        if (s.kind == dense_kind) {
            s.b.kv_write(s.kv_layer, s.views, s.n_views, s.slot(conv), s.slot(value_gate));
            s.b.attention(s.slot(query), s.kv_layer, s.views, s.n_views, s.slot(recurrent), int(H), int(HK), int(D));
        } else {
            s.b.norm_rope_partial(s.slot(index_query), s.slot(index_query), s.rows, IH * I, I, IH, I, R,
                                  w[index_q_norm].slice(), cfg_.rms_eps, cos, sin, s.pos, s.streams);
            s.b.kv_write(s.kv_layer, s.views, s.n_views, s.slot(conv), s.slot(value_gate), s.slot(index_key));
            const size_t compression = s.kind == mtp_qsa_kind ? size_t(cfg_.nextn_compress_ratio) : size_t(cfg_.indexer_compress_ratio);
            const backend::Backend::IndexArgs args{IH, I, R, compression, size_t(cfg_.indexer_budget), cfg_.rms_eps};
            s.b.select_kv(s.slot(index_query), s.kv_layer, s.views, s.n_views, s.slot(selected), args, w[index_k_norm].slice(), cos, sin, s.group_pos);
            const size_t stride = size_t(cfg_.indexer_budget) + compression - 1;
            s.b.sparse_attention(s.slot(query), s.kv_layer, s.views, s.n_views, s.slot(selected), stride, s.slot(recurrent), int(H), int(HK), int(D));
        }
        const backend::Slice qraw = s.slot(raw);
        s.b.sigmoid_mul(s.slot(recurrent), s.slot(recurrent), {qraw.buffer, qraw.offset + D}, s.rows, H, D, 2 * H * D, 2 * D, s.runs);
        s.b.matmul(w[attn_output].operand(), s.slot(recurrent), s.x, H * D, E, s.rows, s.runs, s.dtype);
    }
};

inline std::shared_ptr<const Architecture> open(const gguf::GGUFModel& m, const std::string& prefix) {
    for (const gguf::TensorInfo& tensor : m.tensors)
        if (tensor.name == "token_embd.weight") {
            if (tensor.ne.size() < 2 || !tensor.ne[1] || tensor.ne[1] > uint64_t(std::numeric_limits<int>::max()))
                throw std::runtime_error("inference: invalid vocabulary dimension");
            return std::make_shared<const Qwen4Exp>(read_config(m, prefix), uint32_t(tensor.ne[1]));
        }
    throw TensorIndex::missing("token_embd.weight");
}

} // namespace infer::qwen4exp
