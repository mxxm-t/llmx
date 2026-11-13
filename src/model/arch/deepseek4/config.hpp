#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/bf16.hpp"
#include "format/gguf.hpp"
#include "model/arch/blocks.hpp"
#include "model/arch/metadata.hpp"

// DeepSeek-V4.1's configuration: its keys read from a file, the rules it is held to, each block's place among the shared caches and the rope's frequencies (docs/DEEPSEEK4.md).
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
    std::vector<float> swiglu_exp, swiglu_shexp;  // each block's clamp and each drafter stage's after them, 0 for none
    int index_heads = 0, index_head_dim = 0, index_top_k = 0;
    int hc_count = 0, hc_iterations = 0;
    float hc_eps = 0;
    std::vector<int> compress_ratios, kv_sources, index_sources;
    int candidate_source = -1, candidate_block = 0, candidate_top_k = 0;
    std::vector<int> engram_layers;
    int engram_heads = 0, engram_head_dim = 0, engram_ngram = 0, engram_pad = 0, engram_compressed = 0;
    std::vector<uint64_t> engram_rows, engram_primes, engram_offsets, engram_multipliers;
    std::vector<uint32_t> token_map;
    // The DSpark drafter's stages after the blocks (docs/DEEPSEEK4.md, section 4.7), none where the file carries none.
    // Their expert counts, the rows of a draft's block, the token its rows past the first hold, and the layers whose entering streams it reads (taps).
    int n_nextn = 0, draft_experts = 0, draft_used = 0, draft_block = 0;
    uint32_t noise_id = 0;
    std::vector<int> taps;
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
    const size_t blocks = size_t(c.n_layer) + size_t(std::max(c.n_nextn, 0));
    if (c.compress_ratios.size() != size_t(c.n_layer) || c.swiglu_exp.size() != blocks || c.swiglu_shexp.size() != blocks)
        throw std::runtime_error("inference: invalid DeepSeek-V4.1 layer metadata length");
    for (size_t l = 0; l < blocks; ++l)
        if ((l < size_t(c.n_layer) && c.compress_ratios[l] < 0) || !std::isfinite(c.swiglu_exp[l]) || c.swiglu_exp[l] < 0 ||
            !std::isfinite(c.swiglu_shexp[l]) || c.swiglu_shexp[l] < 0)
            throw std::runtime_error("inference: invalid compression ratio or SwiGLU clamp of layer " + std::to_string(l));
    if (c.n_nextn < 0) throw std::runtime_error("inference: invalid DeepSeek-V4.1 drafter stage count");
    if (c.n_nextn) {
        if (c.taps.empty() || !layer_ids(c.taps, c.n_layer)) throw std::runtime_error("inference: the DSpark taps must be ascending distinct layers of the model");
        if (c.draft_block <= 0 || c.draft_block > c.window) throw std::runtime_error("inference: the DSpark block must hold 1 to window rows");
        if (c.draft_used <= 0 || c.draft_used > c.draft_experts) throw std::runtime_error("inference: invalid DSpark expert counts");
        if (c.noise_id >= c.vocab) throw std::runtime_error("inference: the DSpark noise token is outside the vocabulary");
    }

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
    // Drafter stages a sidecar joined after the blocks (deepseek4::normalize_drafter) are counted by nextn_predict_layers.
    c.n_nextn = metadata::count(m, p + "nextn_predict_layers", 0);
    c.n_layer = integer(m, p + "block_count") - c.n_nextn;
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
        supported(out->size() == size_t(c.n_layer + c.n_nextn), p + key);
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
    if (c.n_nextn) {
        c.draft_experts = integer(m, p + "nextn.expert_count");
        c.draft_used = integer(m, p + "nextn.expert_used_count");
        c.draft_block = integer(m, p + "nextn.block_size");
        required(m, p + "nextn.noise_token_id");
        c.noise_id = uint32_t(metadata::count(m, p + "nextn.noise_token_id", 0));
        required(m, p + "nextn.target_layers");
        c.taps = metadata::counts(m, p + "nextn.target_layers");
    }
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

} // namespace infer::deepseek4
