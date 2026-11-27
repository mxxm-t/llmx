#pragma once
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "format/gguf.hpp"
#include "model/arch/metadata.hpp"

// Inkling-2-Mini's configuration (`inkling2_loop`): its keys read from a file, the rules it is held to and the execution layers its loop runs (docs/INKLING2.md).
namespace infer::inkling2 {

struct Config {
    int n_layer = 0, context_length = 0, n_embd = 0, n_ff = 0, n_expert = 0, n_expert_used = 0;
    int vocab = 0, vocab_unpadded = 0;
    std::vector<int> heads, kv_heads;   // per physical layer
    std::vector<bool> sliding;          // per physical layer: a sliding-window layer, else full attention
    int head_dim = 0, head_dim_swa = 0;
    int loop_count = 0, loop_start = 0, loop_end = 0, gate_rank = 0;
    int block_size = 0, window = 0, rel_length = 0, rel_extent = 0, log_scale_floor = 0, conv_taps = 0;
    float rms_eps = 0, expert_scale = 0, gate_softcap = 0, up_softcap = 0, logit_softcap = 0, qk_clip = 0, rel_clip = 0, log_scale_alpha = 0;
};

// The rules a configuration is held to, each refused by its text.
inline void check_config(const Config& c) {
    const size_t n = size_t(c.n_layer);
    if (c.heads.size() != n || c.kv_heads.size() != n || c.sliding.size() != n)
        throw std::runtime_error("inference: the Inkling-2 per-layer head counts and window pattern do not cover its " + std::to_string(n) + " layers");
    for (size_t l = 0; l < n; ++l)
        if (c.heads[l] <= 0 || c.kv_heads[l] <= 0 || c.heads[l] % c.kv_heads[l])
            throw std::runtime_error("inference: layer " + std::to_string(l) + "'s query heads are not whole groups of its KV heads");
    if (c.vocab_unpadded > c.vocab) throw std::runtime_error("inference: the unpadded vocabulary is larger than the padded one");
    if (c.n_expert_used > c.n_expert) throw std::runtime_error("inference: more experts per token than the layer has");
    if (!(c.loop_start < c.loop_end && c.loop_end <= c.n_layer))
        throw std::runtime_error("inference: the loop's layers [" + std::to_string(c.loop_start) + ", " + std::to_string(c.loop_end) + ") are not layers of the model");
    // The residual bundle holds the block states of at most four blocks beside the hidden row (docs/INKLING2.md, section 3).
    if (c.loop_start % c.block_size || c.loop_end % c.block_size || c.loop_end - c.loop_start < 2 * c.block_size || c.n_layer > 4 * c.block_size)
        throw std::runtime_error("inference: the loop's layers do not start and end on attention-residual blocks of at least two, or the model has more than four blocks");
}

// The keys under `inkling2_loop.`, every one required; values the model assumes are refused when they differ.
inline Config read_config(const gguf::GGUFModel& m) {
    const std::string p = "inkling2_loop.";
    using metadata::integer;
    const auto real = [&](const char* k) {
        if (!m.find(p + k)) throw std::runtime_error("inference: missing metadata " + p + k);
        return float(metadata::real(m, p + k, 1));
    };
    const auto supported = [&](bool ok, const char* k) {
        if (!ok) throw std::runtime_error("inference: unsupported metadata " + p + k);
    };
    Config c;
    c.n_layer = integer(m, p + "block_count");
    c.context_length = integer(m, p + "context_length");
    c.n_embd = integer(m, p + "embedding_length");
    c.n_ff = integer(m, p + "feed_forward_length");
    supported(integer(m, p + "expert_feed_forward_length") == c.n_ff, "expert_feed_forward_length");
    c.n_expert = integer(m, p + "expert_count");
    c.n_expert_used = integer(m, p + "expert_used_count");
    c.vocab = integer(m, p + "vocab_size");
    c.vocab_unpadded = integer(m, p + "vocab_size_unpadded");
    for (const char* k : {"attention.head_count", "attention.head_count_kv", "attention.sliding_window_pattern"})
        if (!m.find(p + k)) throw std::runtime_error("inference: missing metadata " + p + k);
    c.heads = metadata::counts(m, p + "attention.head_count");
    c.kv_heads = metadata::counts(m, p + "attention.head_count_kv");
    c.sliding = metadata::booleans(m, p + "attention.sliding_window_pattern");
    c.head_dim = integer(m, p + "attention.key_length");
    supported(integer(m, p + "attention.value_length") == c.head_dim, "attention.value_length");
    c.head_dim_swa = integer(m, p + "attention.key_length_swa");
    supported(integer(m, p + "attention.value_length_swa") == c.head_dim_swa, "attention.value_length_swa");
    c.loop_count = integer(m, p + "loop.count");
    c.loop_start = metadata::count(m, p + "loop.start", -1);
    if (c.loop_start < 0) throw std::runtime_error("inference: missing metadata " + p + "loop.start");
    c.loop_end = integer(m, p + "loop.end");
    c.gate_rank = integer(m, p + "loop.gate_rank");
    if (!m.find(p + "loop.varying")) throw std::runtime_error("inference: missing metadata " + p + "loop.varying");
    supported(!metadata::boolean(m, p + "loop.varying", false), "loop.varying");
    c.block_size = integer(m, p + "attn_res.block_size");
    c.window = integer(m, p + "attention.sliding_window");
    c.rel_length = integer(m, p + "attention.relative_length");
    c.rel_extent = integer(m, p + "attention.relative_extent");
    c.log_scale_floor = integer(m, p + "attention.log_scale_floor");
    // No rotary positions: attention places tokens by its relative bias and short convolutions alone.
    if (!m.find(p + "rope.dimension_count")) throw std::runtime_error("inference: missing metadata " + p + "rope.dimension_count");
    supported(metadata::count(m, p + "rope.dimension_count", 0) == 0, "rope.dimension_count");
    // Four taps, so the convolution residual's divisors are the square root of five and three.
    c.conv_taps = integer(m, p + "shortconv.l_cache");
    supported(c.conv_taps == 4, "shortconv.l_cache");
    c.rms_eps = real("attention.layer_norm_rms_epsilon");
    c.expert_scale = real("expert_weights_scale");
    c.gate_softcap = real("activation.situ_beta");
    c.up_softcap = real("activation.situ_linear_beta");
    c.logit_softcap = real("final_logit_softcapping");
    c.qk_clip = real("attention.qk_clip");
    c.rel_clip = real("attention.relative_clip");
    c.log_scale_alpha = real("attention.log_scale_alpha");
    check_config(c);
    return c;
}

// The physical layer each execution layer runs, in order: those before the loop, the loop `loop_count` times, then the rest.
inline std::vector<int> execution_layers(const Config& c) {
    std::vector<int> out;
    for (int l = 0; l < c.loop_start; ++l) out.push_back(l);
    for (int pass = 0; pass < c.loop_count; ++pass)
        for (int l = c.loop_start; l < c.loop_end; ++l) out.push_back(l);
    for (int l = c.loop_end; l < c.n_layer; ++l) out.push_back(l);
    return out;
}

} // namespace infer::inkling2
