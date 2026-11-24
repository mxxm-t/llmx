#pragma once
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "format/gguf.hpp"
#include "model/arch/metadata.hpp"

// DFlash drafters (docs/SPECULATIVE.md, section 11): a `dflash` sidecar drafting a block of tokens from a few target layers' rows, its checks and its configuration.
// DSpark rewrites its sidecar into DeepSeek-V4.1's blocks; a DSpine sidecar keeps its `dflash.*` names beside its target.
namespace infer::dflash {

// Whether `side` is a DFlash sidecar.
inline bool sidecar(const gguf::GGUFModel& side) {
    const auto* arch = side.find("general.architecture");
    return arch && arch->vtype == gguf::V_STRING && arch->s == "dflash";
}

// The checks every DFlash sidecar beside target keys under `p` passes: each `dflash.` key but those in `own` equal to the target's, and a mask token for the block's rows past the first, which it returns.
inline const gguf::MetaValue& check_sidecar(const gguf::GGUFModel& target, const gguf::GGUFModel& side, const std::string& p, const std::set<std::string>& own,
                                            const std::string& name) {
    const std::string q = "dflash.";
    for (const auto& item : side.kv) {
        if (item.first.compare(0, q.size(), q) || own.count(item.first.substr(q.size()))) continue;
        const auto* t = target.find(p + item.first.substr(q.size()));
        if (!t || !gguf::detail::equal_value(*t, item.second)) throw std::runtime_error("inference: " + name + " sidecar metadata " + item.first + " differs from the target's");
    }
    const auto* mask = side.find("tokenizer.ggml.mask_token_id");
    if (!mask) throw std::runtime_error("inference: " + name + " sidecar names no tokenizer.ggml.mask_token_id for its block's rows");
    return *mask;
}

// A sidecar's configuration as a joined file holds it under `p` (normalize_beside): its layers, block, attention window and candidates a row, the taps it reads, its heads, rope and feed-forward width, and the mask token.
// A DFlash2 sidecar also gives non-causal attention, its convolution's taps and channels a coefficient, and its selector's rank; a DSpine sidecar leaves them 0.
struct Config {
    int layers = 0, block = 0, window = 0, top_k = 0;
    bool causal = true;
    int conv_taps = 0, conv_group = 0, selector_rank = 0;
    std::vector<int> taps;
    int heads = 0, kv_heads = 0, head_dim = 0, ff = 0;
    float rope_theta = 0.0f, rms_eps = 0.0f;
    uint32_t mask = 0;
};

// The joined configuration, refused where it asks for what the drafter does not run: attention not over a window on every layer, non-causal attention without DFlash2's convolution and selector, or a rope over part of a head.
inline Config read_config(const gguf::GGUFModel& m, const std::string& p) {
    using metadata::integer;
    const std::string q = p + "dflash.";
    const auto refuse = [&](const std::string& key) { return std::runtime_error("inference: unsupported metadata " + q + key); };
    Config c;
    c.layers = integer(m, q + "block_count");
    c.block = integer(m, q + "block_size");
    c.window = integer(m, q + "attention.sliding_window");
    c.top_k = integer(m, q + "selector_top_k");
    c.taps = metadata::counts(m, q + "target_layers");
    c.heads = integer(m, q + "attention.head_count");
    c.kv_heads = integer(m, q + "attention.head_count_kv");
    c.head_dim = integer(m, q + "attention.key_length");
    c.ff = integer(m, q + "feed_forward_length");
    c.rope_theta = float(metadata::real(m, q + "rope.freq_base", 10000.0f));
    c.rms_eps = float(metadata::real(m, p + "attention.layer_norm_rms_epsilon", 1e-6f));
    c.mask = uint32_t(metadata::count(m, q + "mask_token_id", 0));
    if (integer(m, q + "attention.value_length", c.head_dim) != c.head_dim) throw refuse("attention.value_length");
    if (c.heads % c.kv_heads || c.head_dim % 2) throw refuse("attention.head_count_kv");
    if (c.block < 2) throw refuse("block_size");
    if (c.top_k < 1 || c.top_k > 256) throw refuse("selector_top_k");
    c.causal = metadata::boolean(m, q + "attention.causal", false);
    c.conv_taps = int(metadata::count(m, q + "conv_kernel_size", 0));
    c.conv_group = int(metadata::count(m, q + "conv_group_size", 0));
    c.selector_rank = int(metadata::count(m, q + "selector_rank", 0));
    const bool dflash2 = c.conv_taps || c.conv_group || c.selector_rank;
    if (dflash2 && (c.conv_taps < 1 || c.conv_group < 1 || c.selector_rank < 1)) throw refuse("conv_kernel_size");
    if (c.causal == dflash2) throw refuse("attention.causal");
    const std::vector<bool> windowed = metadata::booleans(m, q + "attention.sliding_window_pattern");
    if (windowed.size() != size_t(c.layers)) throw refuse("attention.sliding_window_pattern");
    for (bool w : windowed)
        if (!w) throw refuse("attention.sliding_window_pattern");
    const std::vector<int> sections = metadata::counts(m, q + "rope.dimension_sections");
    for (size_t i = 0; i < sections.size(); ++i)
        if (sections[i] != (i ? 0 : c.head_dim / 2)) throw refuse("rope.dimension_sections");
    return c;
}

// A DFlash sidecar beside its target, renamed under `dflash.*` names so pairing joins it as the target's drafter (docs/src/model-arch-dflash.md).
// Its width, context and norm epsilon must be the target's, and its taps the target's decoder layers.
inline void normalize_beside(const gguf::GGUFModel& target, gguf::GGUFModel& side) {
    if (!sidecar(side)) return;
    const auto* arch = target.find("general.architecture");
    const std::string p = arch->s + ".", q = "dflash.";
    const auto refuse = [](const std::string& text) { return std::runtime_error("inference: DFlash sidecar " + text); };
    if (target.find(p + q + "block_count")) throw refuse("beside a model that carries a DFlash drafter already");
    const std::set<std::string> own{"block_count", "feed_forward_length", "attention.head_count", "attention.head_count_kv", "attention.key_length",
                                    "attention.value_length", "rope.freq_base", "rope.dimension_sections", "block_size", "target_layers",
                                    "attention.sliding_window", "attention.sliding_window_pattern", "attention.causal", "selector_top_k",
                                    "conv_kernel_size", "conv_group_size", "selector_rank"};
    const gguf::MetaValue mask = check_sidecar(target, side, p, own, "DFlash");
    const int layers = metadata::integer(target, p + "block_count") - metadata::count(target, p + "nextn_predict_layers", 0);
    const std::vector<int> taps = metadata::counts(side, q + "target_layers");
    for (size_t i = 0; i < taps.size(); ++i)
        if (taps[i] < 0 || taps[i] >= layers || (i && taps[i] <= taps[i - 1])) throw refuse("target_layers must be ascending distinct layers of the model");
    if (taps.empty()) throw refuse("names no target_layers");
    if (side.tensors.empty()) throw refuse("holds no tensors");

    std::vector<std::pair<std::string, gguf::MetaValue>> kv;
    for (const auto& item : side.kv) {
        if (!item.first.compare(0, q.size(), q)) kv.push_back({p + item.first, item.second});
        else if (item.first.compare(0, p.size(), p) && item.first != "general.architecture") kv.push_back(item);
    }
    kv.push_back({"general.architecture", *arch});
    kv.push_back({p + q + "mask_token_id", mask});
    side.kv.swap(kv);
    for (auto& tensor : side.tensors) tensor.name = q + tensor.name;
}

} // namespace infer::dflash
