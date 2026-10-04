#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "format/gguf.hpp"
#include "model/arch/metadata.hpp"
#include "model/arch/registry.hpp"

// A drafter file beside a model (docs/SPECULATIVE.md, step 6): the one check that it pairs with its target, on the two files' headers before any byte is read, and the joining of a drafter block to the target it pairs with.

namespace infer {
namespace spec {

// What a drafter file is for its target: its MTP blocks, as converters write them after the target's layers; a DFlash drafter; or a model of its own drafting by its own forward pass.
enum class DrafterKind { mtp, dflash, model };

namespace detail {

// A value as a refusal names it: a number or a string as it is, an array by its length.
inline std::string shown(const gguf::MetaValue* v) {
    if (!v) return "none";
    switch (v->vtype) {
        case gguf::V_STRING: return "'" + v->s + "'";
        case gguf::V_ARRAY: return std::to_string(v->arr.size()) + " entries";
        case gguf::V_INT8: case gguf::V_INT16: case gguf::V_INT32: case gguf::V_INT64: return std::to_string(v->i);
        case gguf::V_BOOL: return v->b ? "true" : "false";
        case gguf::V_FLOAT32: case gguf::V_FLOAT64: return "a real";
        default: return std::to_string(v->u);
    }
}

inline bool same(const gguf::MetaValue* a, const gguf::MetaValue* b) {
    return a && b ? gguf::detail::equal_value(*a, *b) : a == b;
}

inline std::string architecture(const gguf::GGUFModel& m) {
    const auto* a = m.find("general.architecture");
    return a && a->vtype == gguf::V_STRING ? a->s : std::string();
}

} // namespace detail

// The kind of drafter the file `drafter` (read from `drafter_path`) is for the model `target` (from `target_path`), or a refusal naming both files and what differs.
// Every kind shares the target's tokenizer: its model, pre-tokenizer, tokens, token types, merges and end of text.
// A file of the target's architecture whose tensors all sit in blocks past the target's layers is its MTP blocks: its metadata under the architecture's prefix must be the target's but for its block count, which counts them, its MTP count, and per-block arrays that go on past the target's, and the target must carry no MTP block of its own.
// A DFlash file must draft for an architecture whose entry allows it, at the target's hidden size, with its taps strictly increasing within the target's layers, a block of at least one and its mask a control or user token of the target.
// Any other file is a draft model, of an architecture llmx runs.
inline DrafterKind pair(const gguf::GGUFModel& target, const std::string& target_path, const gguf::GGUFModel& drafter, const std::string& drafter_path) {
    const auto refuse = [&](const std::string& what) {
        return std::runtime_error("inference: the drafter " + drafter_path + " does not pair with " + target_path + ": " + what);
    };
    const auto differs = [&](const std::string& key, const gguf::MetaValue* t, const gguf::MetaValue* d) {
        return refuse(key + " is " + detail::shown(d) + " where the model has " + detail::shown(t));
    };
    for (const char* key : {"tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges",
                            "tokenizer.ggml.eos_token_id"}) {
        const auto *t = target.find(key), *d = drafter.find(key);
        if (!detail::same(t, d)) throw differs(key, t, d);
    }
    const ArchEntry& entry = architecture_of(target);
    const std::string prefix = std::string(entry.name) + ".";
    const int blocks = metadata::integer(target, prefix + "block_count");
    const int nextn = metadata::count(target, prefix + "nextn_predict_layers", 0);
    const int layers = blocks - nextn;
    const std::string kind = detail::architecture(drafter);

    if (kind == "dflash") {
        if (!entry.dflash) throw refuse("a DFlash drafter drafts for no model of the architecture '" + std::string(entry.name) + "'");
        const auto *t = target.find(prefix + "embedding_length"), *d = drafter.find("dflash.embedding_length");
        if (!detail::same(t, d)) throw differs("dflash.embedding_length", t, d);
        const std::vector<int> taps = metadata::counts(drafter, "dflash.target_layers");
        if (taps.empty()) throw refuse("dflash.target_layers names no layer");
        for (size_t i = 0; i < taps.size(); ++i)
            if (taps[i] > layers || (i && taps[i] <= taps[i - 1]))
                throw refuse("dflash.target_layers is not strictly increasing within the model's " + std::to_string(layers) + " layers");
        metadata::integer(drafter, "dflash.block_size");
        const auto* mask = drafter.find("tokenizer.ggml.mask_token_id");
        const auto* types = target.find("tokenizer.ggml.token_type");
        const int id = mask ? metadata::count(drafter, "tokenizer.ggml.mask_token_id", 0) : -1;
        const bool special = types && id >= 0 && size_t(id) < types->arr.size() && (types->arr[size_t(id)].i == 3 || types->arr[size_t(id)].i == 4);
        if (!special) throw refuse("tokenizer.ggml.mask_token_id " + detail::shown(mask) + " is not a control or user token of the model");
        return DrafterKind::dflash;
    }

    bool blocks_only = !drafter.tensors.empty();
    for (const auto& t : drafter.tensors) {
        const auto b = gguf::block_of(t.name);
        if (!b || *b < size_t(layers)) blocks_only = false;
    }
    if (kind == entry.name && blocks_only) {
        const int extra = metadata::count(drafter, prefix + "nextn_predict_layers", 0);
        if (!extra) throw refuse(prefix + "nextn_predict_layers is 0, so its blocks are not MTP blocks");
        if (nextn) throw refuse("the model carries an MTP block of its own (" + prefix + "nextn_predict_layers " + std::to_string(nextn) + ")");
        const int total = metadata::integer(drafter, prefix + "block_count");
        if (total != blocks + extra)
            throw refuse(prefix + "block_count is " + std::to_string(total) + " where the model's " + std::to_string(blocks) + " layers and " +
                         std::to_string(extra) + " MTP block make " + std::to_string(blocks + extra));
        for (const auto& t : drafter.tensors)
            if (*gguf::block_of(t.name) >= size_t(total)) throw refuse("tensor " + t.name + " is past its " + std::to_string(total) + " blocks");
        // Every key under the prefix the same in both, but the counts and the per-block arrays the MTP blocks extend.
        const auto check = [&](const std::string& key) {
            if (key.compare(0, prefix.size(), prefix) || key == prefix + "block_count" || key == prefix + "nextn_predict_layers") return;
            const auto *t = target.find(key), *d = drafter.find(key);
            if (detail::same(t, d)) return;
            if (t && d && t->vtype == gguf::V_ARRAY && d->vtype == gguf::V_ARRAY && t->arr.size() == size_t(blocks) && d->arr.size() == size_t(total)) {
                bool prefix_equal = true;
                for (size_t i = 0; i < t->arr.size(); ++i) prefix_equal = prefix_equal && gguf::detail::equal_value(t->arr[i], d->arr[i]);
                if (prefix_equal) return;
            }
            throw differs(key, t, d);
        };
        for (const auto& kv : target.kv) check(kv.first);
        for (const auto& kv : drafter.kv) check(kv.first);
        return DrafterKind::mtp;
    }
    try {
        architecture_of(drafter);
    } catch (const std::runtime_error&) {
        throw refuse("its architecture '" + kind + "' is not one llmx runs, nor a drafter's blocks of the model's '" + std::string(entry.name) + "'");
    }
    return DrafterKind::model;
}

// The MTP blocks of `drafter`, which pair with `target` (pair), joined to it as if its file carried them: the drafter's metadata under the architecture's prefix taken over the target's, and its tensors appended as a file after the target's.
inline void join_blocks(gguf::GGUFModel& target, gguf::GGUFModel&& drafter) {
    const std::string prefix = std::string(architecture_of(target).name) + ".";
    for (const auto& kv : drafter.kv) {
        if (kv.first.compare(0, prefix.size(), prefix)) continue;
        bool replaced = false;
        for (auto& own : target.kv)
            if (own.first == kv.first) {
                own.second = kv.second;
                replaced = true;
            }
        if (!replaced) target.kv.push_back(kv);
    }
    gguf::append(target, std::move(drafter));
}

} // namespace spec
} // namespace infer
