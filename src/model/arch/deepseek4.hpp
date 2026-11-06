#pragma once
#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/bf16.hpp"
#include "format/gguf.hpp"
#include "model/arch/blocks.hpp"
#include "model/arch/metadata.hpp"

// DeepSeek-V4.1: the configuration, its refusals, the plan of the fork layout's tensors and caches, and the math on the CPU (docs/DEEPSEEK4.md).
namespace infer::deepseek4 {

// One block's place among the shared caches, as model.py at 1bae0e10 builds it, its ratio 0 for the window alone.
// It may compress the KV others read, run an indexer, own the index keys, pick or read the candidates and run Engram, and `source` is the KV source it reads, -1 for none.
struct Layer {
    int ratio = 0;
    bool kv_source = false, index_source = false, owns_keys = false;
    bool candidate_source = false, uses_candidates = false, engram = false;
    int source = -1;
};

struct Config {
    std::string name;
    uint32_t vocab = 0;
    int n_layer = 0, context_length = 0, n_embd = 0, n_head = 0, head_dim = 0, rope_dim = 0;
    int q_lora_rank = 0, window = 0, out_groups = 0, out_rank = 0;
    float rope_theta = 0, compress_rope_theta = 0, rope_factor = 0, beta_fast = 0, beta_slow = 0, rms_eps = 0;
    int original_context = 0;
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0;
    float expert_scale = 0;
    std::vector<float> swiglu_exp, swiglu_shexp;  // each block's clamp, 0 for none
    int index_heads = 0, index_head_dim = 0, index_top_k = 0;
    int hc_count = 0, hc_iterations = 0;
    float hc_eps = 0;
    std::vector<int> compress_ratios, kv_sources, index_sources;
    int candidate_source = -1, candidate_block = 0, candidate_top_k = 0;
    std::vector<int> engram_layers;
    int engram_heads = 0, engram_head_dim = 0, engram_ngram = 0, engram_pad = 0, engram_compressed = 0;
    std::vector<uint64_t> engram_rows, engram_primes, engram_offsets, engram_multipliers;
    std::vector<uint32_t> token_map;
};

// Ascending distinct block ids, each below the block count.
inline bool layer_ids(const std::vector<int>& ids, int n_layer) {
    for (size_t i = 0; i < ids.size(); ++i)
        if (ids[i] < 0 || ids[i] >= n_layer || (i && ids[i] <= ids[i - 1])) return false;
    return true;
}

inline bool listed(const std::vector<int>& ids, int layer) { return std::find(ids.begin(), ids.end(), layer) != ids.end(); }

// The rules a configuration keeps whatever file it came from: the source layers model.py reads its caches through, the candidate blocks, Engram's arrays against its geometry, and sizes that fit an allocation.
inline void check_config(const Config& c) {
    if (c.n_layer <= 0) throw std::runtime_error("inference: a model without decoder layers");
    if (c.n_embd <= 0 || c.n_head <= 0 || c.head_dim <= 0 || c.q_lora_rank <= 0 || c.window <= 0 || c.out_groups <= 0 ||
        c.out_rank <= 0 || c.n_expert <= 0 || c.n_expert_used <= 0 || c.n_ff_exp <= 0 || c.index_heads <= 0 ||
        c.index_head_dim <= 0 || c.index_top_k <= 0 || c.hc_count <= 0 || c.hc_iterations <= 0 || c.context_length <= 0 ||
        c.original_context <= 0 || !c.vocab)
        throw std::runtime_error("inference: invalid DeepSeek-V4.1 dimensions");
    for (float f : {c.rope_theta, c.compress_rope_theta, c.rope_factor, c.beta_fast, c.beta_slow, c.rms_eps, c.hc_eps, c.expert_scale})
        if (!std::isfinite(f) || f <= 0) throw std::runtime_error("inference: invalid DeepSeek-V4.1 floating-point configuration");
    if (c.n_expert_used > c.n_expert) throw std::runtime_error("inference: more experts per token than the layer has");
    if (c.rope_dim <= 0 || c.rope_dim % 2 || c.rope_dim > c.head_dim || c.rope_dim > c.index_head_dim)
        throw std::runtime_error("inference: invalid rotary width, which must be even and at most the head and indexer head widths");
    if ((uint64_t(c.n_head) * uint64_t(c.head_dim)) % uint64_t(c.out_groups))
        throw std::runtime_error("inference: the output groups do not divide the attention heads' width");
    if (c.compress_ratios.size() != size_t(c.n_layer) || c.swiglu_exp.size() != size_t(c.n_layer) || c.swiglu_shexp.size() != size_t(c.n_layer))
        throw std::runtime_error("inference: invalid DeepSeek-V4.1 layer metadata length");
    for (size_t l = 0; l < size_t(c.n_layer); ++l)
        if (c.compress_ratios[l] < 0 || !std::isfinite(c.swiglu_exp[l]) || c.swiglu_exp[l] < 0 ||
            !std::isfinite(c.swiglu_shexp[l]) || c.swiglu_shexp[l] < 0)
            throw std::runtime_error("inference: invalid compression ratio or SwiGLU clamp of layer " + std::to_string(l));

    if (!layer_ids(c.kv_sources, c.n_layer)) throw std::runtime_error("inference: the KV source layers must be ascending distinct layers of the model");
    if (!layer_ids(c.index_sources, c.n_layer)) throw std::runtime_error("inference: the index source layers must be ascending distinct layers of the model");
    if (!layer_ids(c.engram_layers, c.n_layer)) throw std::runtime_error("inference: the Engram layers must be ascending distinct layers of the model");
    int source = -1;
    for (int l = 0; l < c.n_layer; ++l) {
        const int ratio = c.compress_ratios[size_t(l)];
        const bool kv = listed(c.kv_sources, l), index = listed(c.index_sources, l);
        if (kv) source = l;
        if ((kv || index) && !ratio) throw std::runtime_error("inference: source layer " + std::to_string(l) + " has no compression ratio");
        // A reader takes the compressed cache of the latest KV source at its own ratio (model.py:740-747 at e5decd76), and the index keys exist only on KV sources (model.py:500 at e5decd76).
        if (ratio && (source < 0 || c.compress_ratios[size_t(source)] != ratio))
            throw std::runtime_error("inference: compressed layer " + std::to_string(l) + " has no KV source of its ratio at or before it");
        if (kv && !index) throw std::runtime_error("inference: KV source layer " + std::to_string(l) + " is not an index source");
    }
    if (c.candidate_source >= 0) {
        if (c.candidate_source >= c.n_layer || !listed(c.index_sources, c.candidate_source))
            throw std::runtime_error("inference: the candidate source layer is not an index source");
        if (c.candidate_block <= 0 || c.candidate_top_k <= 0)
            throw std::runtime_error("inference: candidate blocks need a positive block size and count");
        // Every later index source masks its scores with the source's candidates over the source's compressed length (model.py:570-576 at e5decd76).
        for (int l : c.index_sources)
            if (l > c.candidate_source && c.compress_ratios[size_t(l)] != c.compress_ratios[size_t(c.candidate_source)])
                throw std::runtime_error("inference: index source layer " + std::to_string(l) + " reads candidates of another ratio");
    } else if (c.candidate_block || c.candidate_top_k) {
        throw std::runtime_error("inference: candidate blocks without a candidate source layer");
    }

    const size_t layers = c.engram_layers.size();
    if (layers) {
        if (c.engram_ngram < 2 || c.engram_heads <= 0 || c.engram_head_dim <= 0 || c.engram_compressed <= 0)
            throw std::runtime_error("inference: invalid Engram dimensions");
        if (c.engram_pad < 0 || uint32_t(c.engram_pad) >= c.vocab) throw std::runtime_error("inference: the Engram pad id is outside the vocabulary");
    }
    const uint64_t heads = layers ? uint64_t(c.engram_ngram - 1) * uint64_t(c.engram_heads) : 0;
    if (heads > uint64_t(std::numeric_limits<int>::max()) || heads * uint64_t(c.engram_head_dim) > uint64_t(std::numeric_limits<int>::max()))
        throw std::runtime_error("inference: Engram dimensions exceed the supported range");
    if (c.engram_rows.size() != layers || c.engram_multipliers.size() != layers * size_t(std::max(c.engram_ngram, 0)) ||
        c.engram_primes.size() != layers * heads || c.engram_offsets.size() != layers * heads)
        throw std::runtime_error("inference: Engram array lengths do not match its layers and heads");
    for (size_t l = 0; l < layers; ++l) {
        if (!c.engram_rows[l] || c.engram_rows[l] > uint64_t(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error("inference: invalid Engram table rows");
        if (c.engram_offsets[l * heads]) throw std::runtime_error("inference: an Engram layer's offsets do not start at 0");
        for (size_t h = l * heads; h < (l + 1) * heads; ++h)
            if (!c.engram_primes[h] || c.engram_offsets[h] > c.engram_rows[l] || c.engram_primes[h] > c.engram_rows[l] - c.engram_offsets[h])
                throw std::runtime_error("inference: an Engram head's rows lie outside its table");
    }
    if (layers && c.token_map.size() != c.vocab) throw std::runtime_error("inference: the Engram token map does not cover the vocabulary");
    for (uint32_t id : c.token_map)
        if (id >= uint32_t(c.engram_compressed)) throw std::runtime_error("inference: an Engram token map id is outside the compressed vocabulary");

    const uint64_t max_floats = std::vector<float>().max_size(), residual = uint64_t(c.hc_count) * uint64_t(c.n_embd);
    if (residual > uint64_t(std::numeric_limits<int>::max()) || uint64_t(c.context_length) > max_floats / uint64_t(c.rope_dim))
        throw std::runtime_error("inference: DeepSeek-V4.1 dimensions or context exceed allocation limit");
}

// Each block's place, from the source lists and ratios check_config has held to model.py's rules.
inline std::vector<Layer> layer_table(const Config& c) {
    std::vector<Layer> out(size_t(c.n_layer));
    int source = -1;
    for (int l = 0; l < c.n_layer; ++l) {
        Layer& t = out[size_t(l)];
        t.ratio = c.compress_ratios[size_t(l)];
        t.kv_source = listed(c.kv_sources, l);
        t.index_source = listed(c.index_sources, l);
        t.owns_keys = t.kv_source && t.index_source;
        t.candidate_source = l == c.candidate_source;
        t.uses_candidates = t.index_source && c.candidate_source >= 0 && c.candidate_source < l;
        t.engram = listed(c.engram_layers, l);
        if (t.kv_source) source = l;
        t.source = t.ratio ? source : -1;
    }
    return out;
}

// Engram layer `i`'s hash: its multipliers, offsets and primes, the token map into the compressed ids, and the pad id filling before the start, which cuts nothing in text (docs/DEEPSEEK4.md, section 4.4).
inline blocks::NgramHash engram_hash(const Config& c, size_t i) {
    const size_t n = size_t(c.engram_ngram), heads = (n - 1) * size_t(c.engram_heads);
    blocks::NgramHash h;
    h.ngram_size = c.engram_ngram;
    h.heads_per_ngram = c.engram_heads;
    h.fill_id = c.engram_pad;
    h.multipliers.assign(c.engram_multipliers.begin() + std::ptrdiff_t(i * n), c.engram_multipliers.begin() + std::ptrdiff_t((i + 1) * n));
    h.head_offsets.assign(c.engram_offsets.begin() + std::ptrdiff_t(i * heads), c.engram_offsets.begin() + std::ptrdiff_t((i + 1) * heads));
    h.head_vocab_sizes.assign(c.engram_primes.begin() + std::ptrdiff_t(i * heads), c.engram_primes.begin() + std::ptrdiff_t((i + 1) * heads));
    h.token_map = c.token_map;
    h.mapped_vocab = uint32_t(c.engram_compressed);
    h.fill_cuts = false;
    return h;
}

// An array key the reader requires, present before metadata.hpp reads its entries.
inline const gguf::MetaValue& required(const gguf::GGUFModel& m, const std::string& k) {
    const gguf::MetaValue* v = m.find(k);
    if (!v) throw std::runtime_error("inference: missing metadata " + k);
    return *v;
}

inline void supported(bool ok, const std::string& k) {
    if (!ok) throw std::runtime_error("inference: unsupported metadata " + k);
}

// The fork layout's keys under `p`; the upstream layout, which names its Engram keys `ple.*`, is refused by its key until it is read (docs/DEEPSEEK4.md, step 16).
inline Config read_config(const gguf::GGUFModel& m, const std::string& p, uint32_t vocab) {
    using metadata::integer;
    using metadata::real;
    const bool fork = m.find(p + "engram.layer_ids"), upstream = m.find(p + "ple.layers");
    if (fork && upstream) throw std::runtime_error("inference: unsupported metadata " + p + "ple.layers beside " + p + "engram.layer_ids");
    supported(!upstream, p + "ple.layers");
    Config c;
    c.name = p.substr(0, p.size() - 1);
    c.vocab = vocab;
    c.n_layer = integer(m, p + "block_count");
    c.context_length = integer(m, p + "context_length");
    c.n_embd = integer(m, p + "embedding_length");
    c.n_head = integer(m, p + "attention.head_count");
    supported(integer(m, p + "attention.head_count_kv") == 1, p + "attention.head_count_kv");
    c.head_dim = integer(m, p + "attention.key_length");
    supported(integer(m, p + "attention.value_length", c.head_dim) == c.head_dim, p + "attention.value_length");
    c.rope_dim = integer(m, p + "rope.dimension_count");
    c.q_lora_rank = integer(m, p + "attention.q_lora_rank");
    c.window = integer(m, p + "attention.sliding_window");
    c.out_groups = integer(m, p + "attention.output_group_count");
    c.out_rank = integer(m, p + "attention.output_lora_rank");

    required(m, p + "rope.scaling.type");
    metadata::option(m, p + "rope.scaling.type", "yarn");
    const auto real_key = [&](const std::string& k) {
        required(m, p + k);
        return float(real(m, p + k, 1));
    };
    c.rope_theta = real_key("rope.freq_base");
    c.compress_rope_theta = real_key("attention.compress_rope_freq_base");
    c.rope_factor = real_key("rope.scaling.factor");
    c.original_context = integer(m, p + "rope.scaling.original_context_length");
    c.beta_fast = real_key("rope.scaling.yarn_beta_fast");
    c.beta_slow = real_key("rope.scaling.yarn_beta_slow");
    c.rms_eps = real_key("attention.layer_norm_rms_epsilon");

    c.n_expert = integer(m, p + "expert_count");
    c.n_expert_used = integer(m, p + "expert_used_count");
    c.n_ff_exp = integer(m, p + "expert_feed_forward_length");
    supported(integer(m, p + "expert_shared_count") == 1, p + "expert_shared_count");
    required(m, p + "expert_gating_func");
    metadata::choice(m, p + "expert_gating_func", 4, "inference: unsupported metadata " + p + "expert_gating_func");
    c.expert_scale = real_key("expert_weights_scale");
    required(m, p + "expert_weights_norm");
    supported(metadata::boolean(m, p + "expert_weights_norm", true), p + "expert_weights_norm");
    supported(metadata::count(m, p + "hash_layer_count", 0) == 0, p + "hash_layer_count");
    for (auto [key, out] : {std::pair{"swiglu_clamp_exp", &c.swiglu_exp}, std::pair{"swiglu_clamp_shexp", &c.swiglu_shexp}}) {
        required(m, p + key);
        for (double v : metadata::reals(m, p + key)) out->push_back(float(v));
        supported(out->size() == size_t(c.n_layer), p + key);
    }

    c.index_heads = integer(m, p + "attention.indexer.head_count");
    c.index_head_dim = integer(m, p + "attention.indexer.key_length");
    c.index_top_k = integer(m, p + "attention.indexer.top_k");
    // The ratios may go on past the blocks with the drafter's stages, which the 1017 file's 74 keys do not count, so a tail of zeros of any length is taken.
    required(m, p + "attention.compress_ratios");
    c.compress_ratios = metadata::counts(m, p + "attention.compress_ratios");
    supported(c.compress_ratios.size() >= size_t(c.n_layer) &&
                  std::all_of(c.compress_ratios.begin() + c.n_layer, c.compress_ratios.end(), [](int r) { return r == 0; }),
              p + "attention.compress_ratios");
    c.compress_ratios.resize(size_t(c.n_layer));
    required(m, p + "attention.kv_source_layer_ids");
    c.kv_sources = metadata::counts(m, p + "attention.kv_source_layer_ids");
    required(m, p + "attention.index_source_layer_ids");
    c.index_sources = metadata::counts(m, p + "attention.index_source_layer_ids");
    if (m.find(p + "attention.candidate_source_layer_id")) {
        c.candidate_source = metadata::count(m, p + "attention.candidate_source_layer_id", 0);
        c.candidate_block = integer(m, p + "attention.candidate_block_size");
        c.candidate_top_k = integer(m, p + "attention.candidate_topk_blocks");
    } else if (m.find(p + "attention.candidate_block_size") || m.find(p + "attention.candidate_topk_blocks")) {
        throw std::runtime_error("inference: candidate metadata without " + p + "attention.candidate_source_layer_id");
    }

    c.hc_count = integer(m, p + "hyper_connection.count");
    c.hc_iterations = integer(m, p + "hyper_connection.sinkhorn_iterations");
    c.hc_eps = real_key("hyper_connection.epsilon");

    required(m, p + "engram.layer_ids");
    c.engram_layers = metadata::counts(m, p + "engram.layer_ids");
    c.engram_head_dim = integer(m, p + "engram.head_dim");
    c.engram_heads = integer(m, p + "engram.n_heads");
    c.engram_ngram = integer(m, p + "engram.max_ngram_size");
    required(m, p + "engram.pad_token_id");
    c.engram_pad = metadata::count(m, p + "engram.pad_token_id", 0);
    c.engram_compressed = integer(m, p + "engram.compressed_vocab_size");
    c.engram_rows = metadata::uint64s(m, p + "engram.num_embeddings");
    c.engram_primes = metadata::uint64s(m, p + "engram.primes");
    c.engram_offsets = metadata::uint64s(m, p + "engram.offsets");
    c.engram_multipliers = metadata::uint64s(m, p + "engram.multipliers");
    required(m, p + "engram.token_map");
    for (int id : metadata::counts(m, p + "engram.token_map")) c.token_map.push_back(uint32_t(id));
    check_config(c);
    return c;
}

// Engram's stream norm weight, engram_q times engram_k from BF16 or BF16-valued F32 views, exact in F32 (docs/DEEPSEEK4.md, section 4.4).
inline void engram_norm_weight(const TensorView& q, const TensorView& k, float* out, size_t n) {
    const auto value = [](const TensorView& t, size_t i) {
        if (t.type == quant::GGML_TYPE_BF16) return bf16_to_f32(reinterpret_cast<const uint16_t*>(t.data)[i]);
        float v;
        std::memcpy(&v, t.data + i * sizeof(float), sizeof(float));
        return v;
    };
    for (const TensorView* t : {&q, &k}) {
        const size_t width = t->type == quant::GGML_TYPE_BF16 ? sizeof(uint16_t) : t->type == quant::GGML_TYPE_F32 ? sizeof(float) : 0;
        if (!width || t->bytes != n * width || !t->data)
            throw std::runtime_error("inference: unsupported tensor type " + t->name + ", which the Engram norm reads as BF16 or F32");
    }
    for (size_t i = 0; i < n; ++i) out[i] = value(q, i) * value(k, i);
}

// Role ids are local to this architecture.
enum Role : uint16_t {
    token_embd, output_norm, output,
    attn_norm, hc_attn_fn, hc_attn_base, hc_attn_scale, attn_sinks,
    attn_q_a, attn_q_a_norm, attn_q_b, attn_kv, attn_kv_a_norm, attn_output_a, attn_output_b,
    compressor_kv, compressor_gate, compressor_norm, index_q_b, index_proj, index_kv, index_norm,
    engram_embed, engram_key, engram_value, engram_q, engram_k,
    ffn_norm, hc_ffn_fn, hc_ffn_base, hc_ffn_scale,
    ffn_gate_inp, exp_probs_b, ffn_gate_exps, ffn_up_exps, ffn_down_exps, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp,
    role_count
};

// A layer's kind, the flags of its Layer.
enum Kind : uint8_t {
    reads_compressed = 1, compresses_kv = 2, runs_indexer = 4, owns_index_keys = 8, picks_candidates = 16, reads_candidates = 32, runs_engram = 64
};

// The position tables, the window rope's and the compressed YaRN rope's (model.py:681-697 at e5decd76), then the ones, the first block's one-hot pre, the group ids and the zero row the FFN's sums start from.
// Each Engram layer's norm weight comes last.
enum : size_t { rope_cos, rope_sin, compressed_cos, compressed_sin, ones, first_pre, groups, zeros, engram_norms };

// The arena's slots, the streams and each sublayer's pre first and the compressor's, indexer's and candidates' rows last (docs/DEEPSEEK4.md, section 4.3).
// An index source's picks and the candidate source's blocks stay in their slots for the layers after it that read them.
enum Slot : size_t {
    residual, pre_attn, pre_ffn, mixes, post, comb, collapsed, head_x, head_pre, q_lora, query, window_kv, attn, window_ids, group_ids, grouped, block_out,
    scores, picks, pick_weights, gates, ups, acts, engram_rows, engram_keys, engram_value_row, engram_streams, engram_gated,
    compress_kv, compress_score, latent, index_keys, pooled, index_query, index_weights, index_scores, index_lens, index_picks, candidate_ids
};

inline std::vector<size_t> slot_widths(const Config& c) {
    const size_t E = size_t(c.n_embd), HC = size_t(c.hc_count), HD = size_t(c.n_head) * size_t(c.head_dim), G = size_t(c.out_groups);
    const size_t k = size_t(c.n_expert_used), FF = (k + 1) * size_t(c.n_ff_exp);
    const bool engram = !c.engram_layers.empty();
    const size_t lookups = engram ? size_t(c.engram_ngram - 1) * size_t(c.engram_heads) : 0, ER = engram ? HC * E : 0;
    const size_t D = size_t(c.head_dim), ID = size_t(c.index_head_dim), IH = size_t(c.index_heads);
    int least = 0;
    for (int r : c.compress_ratios)
        if (r && (!least || r < least)) least = r;
    const size_t groups = least ? size_t(c.context_length / least) : 0;
    return {HC * E, HC, HC, (2 + HC) * HC, HC, HC * HC, E, HC * E, HC, size_t(c.q_lora_rank), HD, size_t(c.head_dim), HD, size_t(c.window), G,
            G * size_t(c.out_rank), E, size_t(c.n_expert), k, k, FF, FF, FF, lookups * size_t(c.engram_head_dim), ER, engram ? E : 0, ER, ER,
            D, D, D, ID, D + ID, IH * ID, IH, groups, 1, size_t(c.index_top_k), size_t(c.candidate_source >= 0 ? c.candidate_top_k : 0)};
}

// One sublayer's coefficients from the streams x entering it: the mixes as an F32 product of hc_fn, then hc_coeffs' rsqrt over the flattened streams, sigmoids and Sinkhorn rounds (model.py:948-957 at e5decd76).
inline void hc_mix(backend::Backend& b, const Config& c, const Weight& fn, const Weight& scale, const Weight& base, backend::CSlice x,
                   backend::Slice mixes, backend::Slice pre, backend::Slice post, backend::Slice comb, size_t rows, backend::RowRuns runs) {
    const size_t HC = size_t(c.hc_count), E = size_t(c.n_embd);
    b.matmul(fn.operand(), x, mixes, fn.nin, fn.nout, rows, runs, backend::Dtype::f32);
    b.hc_coeffs(pre, post, comb, mixes, x, scale.slice(), base.slice(), rows, HC, E, c.rms_eps, c.hc_eps, size_t(c.hc_iterations));
}

// The rotary angle of each adjacent pair at position 1, in double: 1 / base^(2i / dim), and with original_context above 0 YaRN's ramp between the dims that turn beta_fast and beta_slow times over it (model.py:369-389).
inline std::vector<double> rope_frequencies(int dim, double base, int original_context, double factor, double beta_fast, double beta_slow) {
    std::vector<double> f(size_t(dim / 2));
    for (size_t i = 0; i < f.size(); ++i) f[i] = 1.0 / std::pow(base, double(2 * i) / double(dim));
    if (original_context <= 0) return f;
    const double pi = 3.14159265358979323846;
    const auto corrected = [&](double turns) { return double(dim) * std::log(double(original_context) / (turns * 2 * pi)) / (2 * std::log(base)); };
    const double low = std::max(std::floor(corrected(beta_fast)), 0.0), high = std::min(std::ceil(corrected(beta_slow)), double(dim - 1));
    for (size_t i = 0; i < f.size(); ++i) {
        const double ramp = std::clamp((double(i) - low) / std::max(high - low, 1e-3), 0.0, 1.0), smooth = 1 - ramp;
        f[i] = f[i] / factor * (1 - smooth) + f[i] * smooth;
    }
    return f;
}

class DeepSeek41 final : public Architecture {
public:
    explicit DeepSeek41(const Config& cfg) : cfg_(cfg) {
        check_config(cfg_);
        layers_ = layer_table(cfg_);
        for (size_t i = 0; i < cfg_.engram_layers.size(); ++i) {
            hashes_.push_back(engram_hash(cfg_, i));
            blocks::ngram_rows(hashes_.back(), cfg_.vocab, nullptr, 0, nullptr, 0, nullptr, 0);
        }
    }

    // The plan of the file's tensors as the math will read them: each block's roles by its place among the sources, the residual of hc_count streams, slot 0 alone, each layer's entries, and the position tables' sizes.
    // A tensor that its layer's kind does not read is refused, and every role's tensor must exist.
    ModelPlan plan(const TensorIndex& tensors) const override {
        const Config& c = cfg_;
        const uint64_t E = uint64_t(c.n_embd), HC = uint64_t(c.hc_count), R = HC * E, MIX = (2 + HC) * HC;
        const uint64_t D = uint64_t(c.head_dim), Q = uint64_t(c.q_lora_rank), HD = uint64_t(c.n_head) * D;
        const uint64_t G = uint64_t(c.out_groups), GR = G * uint64_t(c.out_rank), ID = uint64_t(c.index_head_dim);
        const uint64_t X = uint64_t(c.n_expert), F = uint64_t(c.n_ff_exp), W = uint64_t(c.window);
        const uint64_t lookups = uint64_t(c.engram_ngram > 0 ? c.engram_ngram - 1 : 0) * uint64_t(c.engram_heads);
        ModelPlan p;
        p.role_ids = role_count;
        p.unserved = "deepseek41 is not served until its checkpoints, pause and prefix reuse are built (docs/DEEPSEEK4.md, step 11)";
        p.vocab = c.vocab;
        p.pass_ops = {{Part::head, backend::Op::hc_collapse_norm}};
        p.pass = {{token_embd, Part::embed, RoleKind::gather, "token_embd.weight", "", E, p.vocab},
                  {output_norm, Part::head, RoleKind::norm, "output_norm.weight", "", E},
                  {output, Part::head, RoleKind::matrix, "output.weight", "", E, p.vocab}};
        for (const infer::Role& role : p.pass) tensors.at(role.name);
        p.layers.resize(size_t(c.n_layer));
        for (int l = 0; l < c.n_layer; ++l) {
            const Layer& t = layers_[size_t(l)];
            LayerPlan& layer = p.layers[size_t(l)];
            const std::string pre = "blk." + std::to_string(l) + ".";
            layer.kind = uint8_t((t.ratio ? reads_compressed : 0) | (t.kv_source ? compresses_kv : 0) | (t.index_source ? runs_indexer : 0) |
                                 (t.owns_keys ? owns_index_keys : 0) | (t.candidate_source ? picks_candidates : 0) |
                                 (t.uses_candidates ? reads_candidates : 0) | (t.engram ? runs_engram : 0));
            // Every layer keeps entries (docs/DEEPSEEK4.md, section 4.3): a slot of its window ring, window + H rows of D with H the window, and on a ratio-2 source its open group's kv and score rows, r + H of each.
            // A KV source's pooled row is its entry, then the index key derived from it, so both keep one length, one block table and one retract.
            layer.cache = Cache::entries;
            layer.headroom = W;
            layer.state_tail = size_t((W + W) * D + (t.kv_source && t.ratio > 1 ? (uint64_t(t.ratio) + W) * 2 * D : 0));
            // A compressed layer that is no source reads its entries and index keys from the one pooled row of its KV source (LayerPlan::entry_source).
            if (t.kv_source) {
                layer.kv_pool = size_t(t.ratio);
                layer.kv_pool_width = size_t(D + (t.owns_keys ? ID : 0));
            } else if (t.ratio) {
                layer.entry_source = t.source;
            }
            if (t.engram) {
                layer.token_history = size_t(c.engram_ngram - 1);
                layer.token_padding = uint32_t(c.engram_pad);
            }
            layer.routed = true;
            const auto add = [&](uint16_t id, Part part, RoleKind kind, const char* name, uint64_t in, uint64_t out = 1, uint64_t experts = 0,
                                 Stream stream = Stream::none) {
                layer.roles.push_back({id, part, kind, pre + name, "", in, out, experts, stream});
                tensors.at(layer.roles.back().name);
            };
            const auto absent = [&](bool read, std::initializer_list<const char*> names) {
                if (read) return;
                for (const char* name : names)
                    if (tensors.find(pre + name)) throw std::runtime_error("inference: a tensor its layer's kind does not read " + pre + name);
            };
            add(attn_norm, Part::mixer, RoleKind::norm, "attn_norm.weight", E);
            add(hc_attn_fn, Part::mixer, RoleKind::table, "hc_attn_fn.weight", R, MIX);
            add(hc_attn_base, Part::mixer, RoleKind::norm, "hc_attn_base.weight", MIX);
            add(hc_attn_scale, Part::mixer, RoleKind::norm, "hc_attn_scale.weight", 3);
            add(attn_sinks, Part::mixer, RoleKind::norm, "attn_sinks.weight", uint64_t(c.n_head));
            add(attn_q_a, Part::mixer, RoleKind::matrix, "attn_q_a.weight", E, Q);
            add(attn_q_a_norm, Part::mixer, RoleKind::norm, "attn_q_a_norm.weight", Q);
            add(attn_q_b, Part::mixer, RoleKind::matrix, "attn_q_b.weight", Q, HD);
            add(attn_kv, Part::mixer, RoleKind::matrix, "attn_kv.weight", E, D);
            add(attn_kv_a_norm, Part::mixer, RoleKind::norm, "attn_kv_a_norm.weight", D);
            add(attn_output_a, Part::mixer, RoleKind::matrix, "attn_output_a.weight", HD / G, GR);
            add(attn_output_b, Part::mixer, RoleKind::matrix, "attn_output_b.weight", GR, E);
            absent(t.kv_source, {"attn_compressor_kv.weight", "attn_compressor_norm.weight"});
            absent(t.kv_source && t.ratio > 1, {"attn_compressor_gate.weight"});
            if (t.kv_source) {
                add(compressor_kv, Part::mixer, RoleKind::matrix, "attn_compressor_kv.weight", E, D);
                if (t.ratio > 1) add(compressor_gate, Part::mixer, RoleKind::matrix, "attn_compressor_gate.weight", E, D);
                add(compressor_norm, Part::mixer, RoleKind::norm, "attn_compressor_norm.weight", D);
            }
            absent(t.index_source, {"indexer.attn_q_b.weight", "indexer.proj.weight"});
            absent(t.owns_keys, {"indexer_compressor_kv.weight", "indexer_compressor_norm.weight"});
            if (t.index_source) {
                add(index_q_b, Part::mixer, RoleKind::matrix, "indexer.attn_q_b.weight", Q, uint64_t(c.index_heads) * ID);
                add(index_proj, Part::mixer, RoleKind::matrix, "indexer.proj.weight", E, uint64_t(c.index_heads));
                if (t.owns_keys) {
                    add(index_kv, Part::mixer, RoleKind::matrix, "indexer_compressor_kv.weight", D, ID);
                    add(index_norm, Part::mixer, RoleKind::norm, "indexer_compressor_norm.weight", ID);
                }
            }
            absent(t.engram, {"engram_embed.weight", "engram_wkv.weight", "engram_q.weight", "engram_k.weight"});
            if (t.engram) {
                const size_t i = size_t(std::find(c.engram_layers.begin(), c.engram_layers.end(), l) - c.engram_layers.begin());
                const TensorView& table = tensors.view(tensors.at(pre + "engram_embed.weight"));
                if (table.shape.size() < 2 || table.shape[1] != c.engram_rows[i])
                    throw std::runtime_error("inference: " + pre + "engram_embed.weight does not hold " + c.name + ".engram.num_embeddings rows");
                // engram_wkv's first R rows are the streams' keys and its last E the value, two roles over runs of the one tensor (model.py:353-354 at e5decd76).
                add(engram_embed, Part::mixer, RoleKind::gather, "engram_embed.weight", uint64_t(c.engram_head_dim), c.engram_rows[i]);
                add(engram_key, Part::mixer, RoleKind::matrix, "engram_wkv.weight", lookups * uint64_t(c.engram_head_dim), R);
                layer.roles.back().tensor_rows = R + E;
                add(engram_value, Part::mixer, RoleKind::matrix, "engram_wkv.weight", lookups * uint64_t(c.engram_head_dim), E);
                layer.roles.back().first_row = R;
                layer.roles.back().tensor_rows = R + E;
                add(engram_q, Part::mixer, RoleKind::source, "engram_q.weight", E, HC);
                add(engram_k, Part::mixer, RoleKind::source, "engram_k.weight", E, HC);
                layer.lookup = Lookup{engram_embed, size_t(lookups)};
            }
            add(ffn_norm, Part::ffn, RoleKind::norm, "ffn_norm.weight", E, 1, 0, Stream::copy);
            add(hc_ffn_fn, Part::ffn, RoleKind::table, "hc_ffn_fn.weight", R, MIX, 0, Stream::copy);
            add(hc_ffn_base, Part::ffn, RoleKind::norm, "hc_ffn_base.weight", MIX, 1, 0, Stream::copy);
            add(hc_ffn_scale, Part::ffn, RoleKind::norm, "hc_ffn_scale.weight", 3, 1, 0, Stream::copy);
            add(ffn_gate_inp, Part::ffn, RoleKind::matrix, "ffn_gate_inp.weight", E, X, 0, Stream::copy);
            add(exp_probs_b, Part::ffn, RoleKind::norm, "exp_probs_b.bias", X, 1, 0, Stream::copy);
            add(ffn_gate_exps, Part::ffn, RoleKind::experts, "ffn_gate_exps.weight", E, F, X, Stream::window);
            add(ffn_up_exps, Part::ffn, RoleKind::experts, "ffn_up_exps.weight", E, F, X, Stream::window);
            add(ffn_down_exps, Part::ffn, RoleKind::experts, "ffn_down_exps.weight", F, E, X, Stream::window);
            add(ffn_gate_shexp, Part::ffn, RoleKind::matrix, "ffn_gate_shexp.weight", E, F, 0, Stream::copy);
            add(ffn_up_shexp, Part::ffn, RoleKind::matrix, "ffn_up_shexp.weight", E, F, 0, Stream::copy);
            add(ffn_down_shexp, Part::ffn, RoleKind::matrix, "ffn_down_shexp.weight", F, E, 0, Stream::copy);
            // The ops and modes each part issues beyond the common set, so a device without one refuses the layer at load and a stream destination without the FFN's keeps the layer at home.
            using backend::Op;
            const auto ops = [&](Part part, std::initializer_list<Op> list) {
                for (Op op : list) layer.ops.push_back({part, op});
            };
            ops(Part::mixer, {Op::hc_coeffs, Op::hc_collapse_norm, Op::hc_expand, Op::norm_rope_partial, Op::rope_forms, Op::model_format_qdq,
                              Op::sparse_attention_sink, Op::routed_experts});
            if (t.ratio) ops(Part::mixer, {Op::pool_write});
            if (t.kv_source) ops(Part::mixer, {Op::compress_pool});
            if (t.index_source) ops(Part::mixer, {Op::indexer_scores, Op::topk_rows});
            if (t.candidate_source) ops(Part::mixer, {Op::candidate_blocks});
            if (t.engram) ops(Part::mixer, {Op::group_rms_norm, Op::signed_sqrt_gate});
            ops(Part::ffn, {Op::hc_coeffs, Op::hc_collapse_norm, Op::hc_expand, Op::sqrt_softplus_routing});
            if (c.swiglu_exp[size_t(l)] != 0.0f || c.swiglu_shexp[size_t(l)] != 0.0f) ops(Part::ffn, {Op::silu_mul_clamp});
            blocks::routed_ops(layer, tensors, ffn_gate_exps, ffn_up_exps);
        }
        p.context_length = size_t(c.context_length);
        p.residual = size_t(R);
        p.slots = slot_widths(c);
        // What a later layer reads crosses with the residual (docs/DEEPSEEK4.md, section 4.3): both pres, a source's new pooled rows for its readers' replicas, the latest index source's picks a reader reuses and the candidate blocks.
        bool readers = false, reuses = false;
        for (const Layer& t : layers_) {
            readers = readers || (t.ratio && !t.kv_source);
            reuses = reuses || (t.ratio && !t.index_source);
        }
        p.carried = {pre_attn, pre_ffn};
        if (readers) p.carried.push_back(pooled);
        if (reuses) p.carried.push_back(index_picks);
        if (c.candidate_source >= 0) p.carried.push_back(candidate_ids);
        p.index_width = std::max({size_t(HC), size_t(lookups), size_t(W)});
        p.tables.assign(engram_norms + c.engram_layers.size(), size_t(c.context_length) * size_t(c.rope_dim / 2));
        for (size_t i = ones; i < p.tables.size(); ++i) p.tables[i] = i == groups ? size_t(G) : i == first_pre ? size_t(HC) : i == zeros ? size_t(E) : size_t(R);
        p.embed_copies = size_t(HC);
        return p;
    }

    // Each angle in double, rounded once; the sources are each Engram layer's engram_q then engram_k, as the plan lists them.
    void fill_tables(std::vector<std::vector<float>>& tables, const std::vector<TensorView>& sources) const override {
        const Config& c = cfg_;
        const size_t half = size_t(c.rope_dim / 2);
        const std::vector<double> window = rope_frequencies(c.rope_dim, c.rope_theta, 0, 1, 0, 0);
        const std::vector<double> compressed = rope_frequencies(c.rope_dim, c.compress_rope_theta, c.original_context, c.rope_factor, c.beta_fast, c.beta_slow);
        for (size_t pos = 0; pos < size_t(c.context_length); ++pos)
            for (size_t i = 0; i < half; ++i) {
                tables[rope_cos][pos * half + i] = float(std::cos(double(pos) * window[i]));
                tables[rope_sin][pos * half + i] = float(std::sin(double(pos) * window[i]));
                tables[compressed_cos][pos * half + i] = float(std::cos(double(pos) * compressed[i]));
                tables[compressed_sin][pos * half + i] = float(std::sin(double(pos) * compressed[i]));
            }
        std::fill(tables[first_pre].begin(), tables[first_pre].end(), 0.0f);
        tables[first_pre][0] = 1.0f;
        for (uint32_t g = 0; g < uint32_t(c.out_groups); ++g) std::memcpy(&tables[groups][g], &g, sizeof(g));
        std::fill(tables[ones].begin(), tables[ones].end(), 1.0f);
        std::fill(tables[zeros].begin(), tables[zeros].end(), 0.0f);
        if (c.engram_layers.empty()) return;
        if (sources.size() != 2 * c.engram_layers.size()) throw std::logic_error("inference: the Engram norm sources are not two a layer");
        for (size_t l = 0; l < c.engram_layers.size(); ++l) {
            engram_norm_weight(sources[2 * l], sources[2 * l + 1], tables[engram_norms + l].data(), tables[engram_norms + l].size());
        }
    }
    // The token's row in every stream, and the first block's one-hot pre (model.py:1262-1264 at e5decd76).
    void embed(const Step& s, const uint32_t* ids) const override {
        const size_t H = size_t(cfg_.hc_count), count = backend::size_mul(s.rows, H);
        if (!s.indices || s.indices_size < count) throw std::runtime_error("inference: the DeepSeek-V4.1 embedding's index scratch is too small");
        for (size_t r = 0; r < s.rows; ++r)
            for (size_t h = 0; h < H; ++h) s.indices[r * H + h] = ids[r];
        const Weight& table = s.w[token_embd];
        s.b.embed(s.x, table.operand(), table.nin, table.nout, s.indices, count);
        std::fill(s.indices, s.indices + s.rows, 0u);
        s.b.gather_rows(s.slot(pre_ffn), {s.tables[first_pre].get(), 0}, H, s.indices, s.rows);
    }

    // An Engram layer's table rows for each token: the n-gram hash of the token and the history before it, view by view.
    void lookup_ids(const Step& s) const override {
        const blocks::NgramHash& h = hashes_[engram_index(s)];
        const size_t heads = h.head_offsets.size(), count = backend::size_mul(s.rows, heads);
        if (!s.indices || s.indices_size < count) throw std::runtime_error("inference: the DeepSeek-V4.1 Engram's index scratch is too small");
        if (!s.tokens || !s.states || !s.previous_tokens || s.previous_count != size_t(h.ngram_size - 1))
            throw std::runtime_error("inference: the DeepSeek-V4.1 Engram's token history is missing");
        size_t first = 0;
        for (size_t v = 0; v < s.n_views; ++v) {
            const size_t n = s.states[v].nq;
            if (n > s.rows - first) throw std::runtime_error("inference: the DeepSeek-V4.1 Engram's views exceed the pass");
            blocks::ngram_rows(h, cfg_.vocab, s.previous_tokens + v * s.previous_count, s.previous_count, s.tokens + first, n,
                               reinterpret_cast<int32_t*>(s.indices + first * heads), n * heads);
            first += n;
        }
        if (first != s.rows) throw std::runtime_error("inference: the DeepSeek-V4.1 Engram's views do not cover the pass");
    }

    // The attention's coefficients from the streams, its input collapsed with the pre the block before left, and its output expanded back (model.py:982-986 at e5decd76).
    // An Engram layer first adds its gated value to the streams (model.py:1265-1266).
    void mixer(const Step& s) const override {
        const Weight* w = s.w;
        if (s.kind & runs_engram) engram(s);
        hc_mix(s.b, cfg_, w[hc_attn_fn], w[hc_attn_scale], w[hc_attn_base], s.x, s.slot(mixes), s.slot(pre_attn), s.slot(post), s.slot(comb), s.rows, s.runs);
        s.b.hc_collapse_norm(s.slot(collapsed), s.x, s.slot(pre_ffn), w[attn_norm].slice(), s.rows, size_t(cfg_.hc_count), size_t(cfg_.n_embd), cfg_.rms_eps);
        // The first reader on a device after its source's device stores the carried pooled rows into its replica, as the source stored them (LayerPlan::entry_source).
        if (s.appends) s.b.pool_write(s.kv_layer, s.views, s.n_views, s.slot(pooled));
        window_attention(s);
        s.b.hc_expand(s.x, s.slot(block_out), s.x, s.slot(post), s.slot(comb), s.rows, size_t(cfg_.hc_count), size_t(cfg_.n_embd));
    }

    // The FFN's coefficients, its input collapsed with the attention's pre, and its output expanded back; its pre is the next block's (model.py:988-993 at e5decd76).
    // The MoE between them (model.py:794-899): k experts picked by sqrt(softplus) score plus bias and weighted by the score, each SwiGLU clamped by the block's limit, and the shared expert, summed into block_out from zero.
    void ffn(const Step& s) const override {
        const Weight* w = s.w;
        hc_mix(s.b, cfg_, w[hc_ffn_fn], w[hc_ffn_scale], w[hc_ffn_base], s.x, s.slot(mixes), s.slot(pre_ffn), s.slot(post), s.slot(comb), s.rows, s.runs);
        s.b.hc_collapse_norm(s.slot(collapsed), s.x, s.slot(pre_attn), w[ffn_norm].slice(), s.rows, size_t(cfg_.hc_count), size_t(cfg_.n_embd), cfg_.rms_eps);
        if (s.layer < 0 || size_t(s.layer) >= cfg_.swiglu_exp.size()) throw std::logic_error("inference: a DeepSeek-V4.1 FFN without its layer");
        const blocks::SharedExpert shared{w[ffn_gate_shexp], w[ffn_up_shexp], w[ffn_down_shexp], nullptr, cfg_.swiglu_shexp[size_t(s.layer)]};
        Step out = s;
        out.x = s.slot(block_out);
        out.width = 1;
        blocks::moe_step(out, w[ffn_gate_inp], w[ffn_gate_exps], w[ffn_up_exps], w[ffn_down_exps], size_t(cfg_.n_expert_used),
                         backend::RouteRule::sqrt_softplus(w[exp_probs_b].slice(), cfg_.expert_scale, true), s.slot(collapsed), s.slot(gates), s.slot(ups),
                         s.slot(acts), s.slot(scores), s.slot(picks), s.slot(pick_weights), &shared, {}, s.tables[zeros].get(), cfg_.swiglu_exp[size_t(s.layer)]);
        s.b.hc_expand(s.x, s.slot(block_out), s.x, s.slot(post), s.slot(comb), s.rows, size_t(cfg_.hc_count), size_t(cfg_.n_embd));
    }

    // The wanted rows' streams collapsed with the last block's pre, normed, then the logits (model.py:1271-1272 at e5decd76).
    void head(const HeadStep& s) const override {
        const size_t H = size_t(cfg_.hc_count), E = size_t(cfg_.n_embd);
        s.b.gather_rows(s.slot(head_x), s.x, H * E, s.pick, s.want);
        s.b.gather_rows(s.slot(head_pre), s.slot(pre_ffn), H, s.pick, s.want);
        s.b.hc_collapse_norm(s.slot(collapsed), s.slot(head_x), s.slot(head_pre), s.w[output_norm].slice(), s.want, H, E, cfg_.rms_eps);
        const Weight& out = s.w[output];
        s.b.matmul_logits(out.operand(), s.slot(collapsed), s.logits, out.nin, out.nout, s.want, s.head_runs, s.dtype);
    }

private:
    size_t engram_index(const Step& s) const {
        const auto at = std::find(cfg_.engram_layers.begin(), cfg_.engram_layers.end(), s.layer);
        if (at == cfg_.engram_layers.end()) throw std::logic_error("inference: a DeepSeek-V4.1 Engram step without its layer");
        return size_t(at - cfg_.engram_layers.begin());
    }

    // Engram (model.py:350-365 at e5decd76) through the shared n-gram gate: the rows projected to each stream's key and the value, the keys normed with ones and the streams with engram_q times engram_k, a zero dot keeping its sign.
    // The gated value is then added to every stream.
    void engram(const Step& s) const {
        const size_t i = engram_index(s), H = size_t(cfg_.hc_count), E = size_t(cfg_.n_embd), R = H * E, heads = hashes_[i].head_offsets.size();
        const size_t count = backend::size_mul(s.rows, heads);
        const Weight& table = s.w[engram_embed];
        if (s.lookup) {
            if (!s.lookup->data.buffer || !s.lookup->rows || (count && !s.lookup->ids))
                throw std::runtime_error("inference: the DeepSeek-V4.1 Engram's prepared lookup is incomplete");
            backend::WeightView rows = table.operand();
            rows.data = s.lookup->data;
            s.b.embed(s.slot(engram_rows), rows, table.nin, s.lookup->rows, s.lookup->ids, count);
        } else {
            lookup_ids(s);
            s.b.embed(s.slot(engram_rows), table.operand(), table.nin, table.nout, s.indices, count);
        }
        const Weight keys{0, s.tables[ones], R, 1}, streams{0, s.tables[engram_norms + i], R, 1};
        blocks::ngram_gate(s, {s.w[engram_key], s.w[engram_value], keys, streams}, s.slot(engram_rows), heads * table.nin, E, H, cfg_.rms_eps,
                           s.slot(engram_gated), {s.slot(engram_keys), s.slot(engram_value_row), s.slot(engram_streams)}, backend::ZeroSign::kept);
        s.b.add(s.x, s.slot(engram_gated), backend::size_mul(s.rows, R));
    }

    // A layer's attention from the collapsed row to block_out (model.py:766-791 at e5decd76), a compressed layer's picked entries after the window's rows under one softmax.
    // An entry's rows go into its ring of 2 x window slots a window at a time before they attend, so no row overwrites a position a later row still reads.
    void window_attention(const Step& s) const {
        const Config& c = cfg_;
        const Weight* w = s.w;
        const size_t D = size_t(c.head_dim), H = size_t(c.n_head), HD = H * D, Q = size_t(c.q_lora_rank), rd = size_t(c.rope_dim), W = size_t(c.window);
        const size_t ring = 2 * W, G = size_t(c.out_groups), rows = s.rows;
        const bool squeezed = s.kind & reads_compressed;
        const backend::CSlice cos{s.tables[squeezed ? compressed_cos : rope_cos].get(), 0}, sin{s.tables[squeezed ? compressed_sin : rope_sin].get(), 0};
        const backend::RopeForm turn{true, D - rd, false}, back{true, D - rd, true};
        s.b.matmul(w[attn_q_a].operand(), s.slot(collapsed), s.slot(q_lora), w[attn_q_a].nin, w[attn_q_a].nout, rows, s.runs, s.dtype);
        s.b.rms_norm_rows(s.slot(q_lora), s.slot(q_lora), w[attn_q_a_norm].slice(), rows, Q, Q, c.rms_eps, s.runs);
        s.b.matmul(w[attn_q_b].operand(), s.slot(q_lora), s.slot(query), w[attn_q_b].nin, w[attn_q_b].nout, rows, s.runs, s.dtype);
        s.b.norm_rope_partial(s.slot(query), s.slot(query), rows, HD, D, H, D, rd, {}, c.rms_eps, cos, sin, s.pos, turn);
        s.b.matmul(w[attn_kv].operand(), s.slot(collapsed), s.slot(window_kv), w[attn_kv].nin, w[attn_kv].nout, rows, s.runs, s.dtype);
        s.b.norm_rope_partial(s.slot(window_kv), s.slot(window_kv), rows, D, D, 1, D, rd, w[attn_kv_a_norm].slice(), c.rms_eps, cos, sin, s.pos, turn);
        s.b.model_format_qdq(s.slot(window_kv), backend::size_mul(rows, D), backend::ModelFormat::fp8_power);

        // The kv rows are the layer's whole state buffer, a slot's ring starting on a row; each row reads its window's positions oldest first, none before 0.
        if (!s.states || !s.n_views) throw std::logic_error("inference: a DeepSeek-V4.1 window without state views");
        backend::StateStorage& store = *s.states[0].storage;
        const size_t slot_floats = store.slot_floats(s.state_layer), prefix = store.shape().slot_floats();
        if (slot_floats % D || prefix % D) throw std::logic_error("inference: a DeepSeek-V4.1 window ring that starts inside a row");
        backend::Buffer& kv = store.layer(s.state_layer);
        const size_t kv_rows = backend::size_mul(store.slots(), slot_floats / D);
        if (!s.indices || s.indices_size < backend::size_mul(rows, W)) throw std::runtime_error("inference: the DeepSeek-V4.1 window's index scratch is too small");
        size_t most = 0;
        for (size_t e = 0, row = 0; e < s.n_views; row += s.states[e].nq, ++e) {
            const backend::StateView& v = s.states[e];
            most = std::max(most, v.nq);
            const size_t base = (v.dst * slot_floats + prefix) / D;
            for (size_t i = 0; i < v.nq; ++i) {
                const uint64_t q = s.pos[row + i];
                for (size_t t = 0; t < W; ++t) {
                    const uint64_t behind = W - 1 - t;
                    s.indices[(row + i) * W + t] = q < behind ? UINT32_MAX : uint32_t(base + (q - behind) % ring);
                }
            }
            if (v.length && v.src != v.dst)
                s.b.copy(kv, (v.dst * slot_floats + prefix) * sizeof(float), kv, (v.src * slot_floats + prefix) * sizeof(float), (slot_floats - prefix) * sizeof(float));
        }
        s.b.write(*s.arena, s.offsets[window_ids], s.indices, backend::size_mul(rows, W) * sizeof(uint32_t));
        const size_t picked = squeezed ? compressed(s, cos, sin) : 0;
        const float scale = float(1.0 / std::sqrt(double(D)));
        std::vector<backend::KVView> run;
        const auto attend = [&](size_t first, size_t end) {
            if (first == end) return;
            const backend::Slice out = s.slot(attn), qs = s.slot(query), ids = s.slot(window_ids), entries = s.slot(index_picks);
            s.b.sparse_attention_sink({out.buffer, out.offset + first * HD}, {qs.buffer, qs.offset + first * HD}, {&kv, 0}, {ids.buffer, ids.offset + first * W},
                                      w[attn_sinks].slice(), end - first, H, D, kv_rows, W, scale, s.kv_layer, squeezed ? run.data() : nullptr,
                                      squeezed ? run.size() : 0, picked ? backend::CSlice{entries.buffer, entries.offset + first * picked} : backend::CSlice{}, picked);
            run.clear();
        };
        // Rows of one chunk that sit side by side attend in one call, so a pass whose entries are each at most a window attends once.
        for (size_t chunk = 0; chunk < most; chunk += W) {
            size_t first = 0, end = 0;
            for (size_t e = 0, row = 0; e < s.n_views; row += s.states[e].nq, ++e) {
                const backend::StateView& v = s.states[e];
                if (chunk >= v.nq) continue;
                const size_t n = std::min(W, v.nq - chunk), from = row + chunk;
                for (size_t i = 0; i < n;) {
                    const size_t at = size_t(s.pos[from + i] % ring), run = std::min(n - i, ring - at);
                    s.b.copy(kv, (v.dst * slot_floats + prefix + at * D) * sizeof(float), *s.arena, s.offsets[window_kv] + (from + i) * D * sizeof(float),
                             run * D * sizeof(float));
                    i += run;
                }
                if (from != end) {
                    attend(first, end);
                    first = from;
                }
                end = from + n;
                if (squeezed) {
                    backend::KVView part = s.views[e];
                    part.length += chunk;
                    part.nq = n;
                    run.push_back(part);
                }
            }
            attend(first, end);
        }
        s.b.norm_rope_partial(s.slot(attn), s.slot(attn), rows, HD, D, H, D, rd, {}, c.rms_eps, cos, sin, s.pos, back);
        std::fill(s.indices, s.indices + rows, 0u);
        s.b.gather_rows(s.slot(group_ids), {s.tables[groups].get(), 0}, G, s.indices, rows);
        s.b.matmul_grouped(backend::Projection(w[attn_output_a].operand(), s.slot(grouped), size_t(c.out_rank)), s.slot(attn), HD / G, rows, G, s.slot(group_ids),
                           s.runs, s.dtype, *s.scratch);
        s.b.matmul(w[attn_output_b].operand(), s.slot(grouped), s.slot(block_out), w[attn_output_b].nin, w[attn_output_b].nout, rows, s.runs, s.dtype);
    }

    // A compressed layer's entries and picks before its attention reads them (model.py:729-764 at e5decd76), returning the picks each row reads.
    // A KV source stores each group its rows complete as one pooled row, the entry then its index key; an index source then scores the visible entries and keeps the best.
    size_t compressed(const Step& s, backend::CSlice cos, backend::CSlice sin) const {
        const Config& c = cfg_;
        const Weight* w = s.w;
        const size_t D = size_t(c.head_dim), ID = size_t(c.index_head_dim), IH = size_t(c.index_heads), rd = size_t(c.rope_dim), W = size_t(c.window);
        const size_t ratio = size_t(c.compress_ratios.at(size_t(s.layer))), rows = s.rows;
        if (!s.views) throw std::logic_error("inference: a DeepSeek-V4.1 compressed layer without KV views");
        if (s.kind & compresses_kv) {
            if (!ratio || W % ratio) throw std::runtime_error("inference: a DeepSeek-V4.1 compression ratio that does not divide the window");
            const backend::Slice latents = s.slot(latent);
            std::vector<uint32_t> firsts;
            if (ratio == 1) {
                // A group of one row is the row's normed projection, in the activations' precision as the checkpoint keeps it (model.py:458-461 at e5decd76).
                s.b.matmul(w[compressor_kv].operand(), s.slot(collapsed), latents, w[compressor_kv].nin, w[compressor_kv].nout, rows, s.runs, s.dtype);
                s.b.rms_norm_rows(latents, latents, w[compressor_norm].slice(), rows, D, D, c.rms_eps, s.runs);
                firsts.assign(s.pos, s.pos + rows);
            } else {
                s.b.matmul(w[compressor_kv].operand(), s.slot(collapsed), s.slot(compress_kv), w[compressor_kv].nin, w[compressor_kv].nout, rows, s.runs, backend::Dtype::f32);
                s.b.matmul(w[compressor_gate].operand(), s.slot(collapsed), s.slot(compress_score), w[compressor_gate].nin, w[compressor_gate].nout, rows, s.runs,
                           backend::Dtype::f32);
                // The open group's kv and score rows follow the window ring in the slot, ratio + window of each at position mod ratio + window.
                backend::StateStorage& store = *s.states[0].storage;
                backend::Buffer& state = store.layer(s.state_layer);
                const size_t slot_floats = store.slot_floats(s.state_layer), prefix = store.shape().slot_floats() + 2 * W * D, ring = ratio + W;
                for (size_t e = 0, row = 0; e < s.n_views; row += s.states[e].nq, ++e) {
                    const size_t length = s.views[e].length, nq = s.states[e].nq, kv_at = s.states[e].dst * slot_floats + prefix, score_at = kv_at + ring * D;
                    for (size_t chunk = 0; chunk < nq; chunk += W) {
                        const size_t n = std::min(W, nq - chunk), from = row + chunk, start = length + chunk;
                        for (size_t i = 0; i < n;) {
                            const size_t at = (start + i) % ring, part = std::min(n - i, ring - at);
                            for (const auto& [slot, to] : {std::pair{compress_kv, kv_at}, std::pair{compress_score, score_at}})
                                s.b.copy(state, (to + at * D) * sizeof(float), *s.arena, s.offsets[slot] + (from + i) * D * sizeof(float), part * D * sizeof(float));
                            i += part;
                        }
                        // A group never wraps the ring, which the window holds whole groups of, so the groups up to the ring's end pool in one call.
                        for (size_t g = start / ratio, end = (start + n) / ratio; g < end;) {
                            const size_t at = g * ratio % ring, count = std::min(end - g, (ring - at) / ratio);
                            s.b.compress_pool({latents.buffer, latents.offset + firsts.size() * D}, {&state, kv_at + at * D}, {&state, score_at + at * D},
                                              w[compressor_norm].slice(), count, ratio, D, c.rms_eps);
                            for (size_t j = 0; j < count; ++j) firsts.push_back(uint32_t((g + j) * ratio));
                            g += count;
                        }
                    }
                }
            }
            const size_t groups = firsts.size();
            if (groups) {
                const backend::RopeForm turn{true, D - rd, false}, key_turn{true, ID - rd, false};
                const Weight& wk = w[index_kv];
                s.b.matmul(wk.operand(), s.slot(latent), s.slot(index_keys), wk.nin, wk.nout, groups, {}, s.dtype);
                s.b.norm_rope_partial(s.slot(index_keys), s.slot(index_keys), groups, ID, ID, 1, ID, rd, w[index_norm].slice(), c.rms_eps, cos, sin, firsts.data(),
                                      key_turn);
                s.b.model_format_qdq(s.slot(index_keys), backend::size_mul(groups, ID), backend::ModelFormat::fp4_power);
                s.b.norm_rope_partial(latents, latents, groups, D, D, 1, D, rd, {}, c.rms_eps, cos, sin, firsts.data(), turn);
                s.b.model_format_qdq(latents, backend::size_mul(groups, D), backend::ModelFormat::fp4_e4m3);
                // A pooled row is the entry, then the index key derived from it (LayerPlan::kv_pool_width).
                const size_t width = D + ID;
                for (size_t g = 0; g < groups; ++g) {
                    s.b.copy(*s.arena, s.offsets[pooled] + g * width * sizeof(float), *s.arena, s.offsets[latent] + g * D * sizeof(float), D * sizeof(float));
                    s.b.copy(*s.arena, s.offsets[pooled] + (g * width + D) * sizeof(float), *s.arena, s.offsets[index_keys] + g * ID * sizeof(float), ID * sizeof(float));
                }
            }
            s.b.pool_write(s.kv_layer, s.views, s.n_views, s.slot(pooled));
        }
        size_t visible = 0;
        for (size_t e = 0; e < s.n_views; ++e) visible = std::max(visible, (s.views[e].length + s.views[e].nq) / ratio);
        const size_t k = std::min(size_t(c.index_top_k), visible);
        if (!(s.kind & runs_indexer) || !k) return k;
        // The indexer (model.py:552-581): the query from the attention's low-rank row turned at each row's position in the index keys' FP4, and each head's weight scaled by ID^-1/2 heads^-1/2.
        const Weight &wq = w[index_q_b], &wp = w[index_proj];
        s.b.matmul(wq.operand(), s.slot(q_lora), s.slot(index_query), wq.nin, wq.nout, rows, s.runs, s.dtype);
        s.b.norm_rope_partial(s.slot(index_query), s.slot(index_query), rows, IH * ID, ID, IH, ID, rd, {}, c.rms_eps, cos, sin, s.pos,
                              backend::RopeForm{true, ID - rd, false});
        s.b.model_format_qdq(s.slot(index_query), backend::size_mul(rows, IH * ID), backend::ModelFormat::fp4_power);
        s.b.matmul(wp.operand(), s.slot(collapsed), s.slot(index_weights), wp.nin, wp.nout, rows, s.runs, s.dtype);
        const float weight_scale = float(std::pow(double(ID), -0.5) * std::pow(double(IH), -0.5));
        // A reader of the candidate source's blocks scores only inside them; the source keeps the best blocks of its own scores (model.py:570-611).
        const bool masked = s.kind & reads_candidates;
        const size_t blocks = size_t(c.candidate_top_k), block = size_t(c.candidate_block);
        s.b.indexer_scores(s.slot(index_scores), s.slot(index_query), s.kv_layer, s.views, s.n_views, s.slot(index_weights), weight_scale, IH, ID, visible,
                           masked ? backend::CSlice(s.slot(candidate_ids)) : backend::CSlice{}, masked ? blocks : 0, masked ? block : 0);
        for (size_t r = 0; r < rows; ++r) s.indices[r] = uint32_t((size_t(s.pos[r]) + 1) / ratio);
        s.b.write(*s.arena, s.offsets[index_lens], s.indices, rows * sizeof(uint32_t));
        if (s.kind & picks_candidates) s.b.candidate_blocks(s.slot(candidate_ids), s.slot(index_scores), s.slot(index_lens), rows, visible, block, blocks);
        s.b.topk_rows(s.slot(index_picks), s.slot(index_scores), s.slot(index_lens), rows, visible, k, 0);
        return k;
    }

    Config cfg_;
    std::vector<Layer> layers_;
    std::vector<blocks::NgramHash> hashes_;
};

inline std::shared_ptr<const Architecture> open_v41(const gguf::GGUFModel& m, const std::string& prefix) {
    for (const gguf::TensorInfo& tensor : m.tensors)
        if (tensor.name == "token_embd.weight") {
            if (tensor.ne.size() < 2 || !tensor.ne[1] || tensor.ne[1] > uint64_t(std::numeric_limits<int>::max()))
                throw std::runtime_error("inference: invalid vocabulary dimension");
            return std::make_shared<const DeepSeek41>(read_config(m, prefix, uint32_t(tensor.ne[1])));
        }
    throw TensorIndex::missing("token_embd.weight");
}

} // namespace infer::deepseek4
