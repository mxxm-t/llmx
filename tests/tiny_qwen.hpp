#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>
#include "format/gguf.hpp"

// A Qwen3-shaped F32 model with deterministic weights, 8 wide with two query heads over one KV head, for the native tests that run a forward pass.
// Without `tied` it has an output projection of its own.
inline gguf::GGUFModel tiny_qwen(int layers, uint64_t context, bool tied) {
    gguf::GGUFModel m;
    for (const auto& kv : std::vector<std::pair<std::string, uint64_t>>{
            {"block_count", uint64_t(layers)}, {"embedding_length", 8}, {"feed_forward_length", 12},
            {"attention.head_count", 2}, {"attention.head_count_kv", 1},
            {"attention.key_length", 4}, {"context_length", context}}) {
        gguf::MetaValue v; v.vtype = gguf::V_UINT32; v.u = kv.second;
        m.kv.push_back({"qwen3." + kv.first, v});
    }
    auto add = [&](const std::string& name, std::vector<uint64_t> shape, bool norm = false) {
        size_t count = 1;
        for (uint64_t d : shape) count *= size_t(d);
        const size_t offset = m.blob.size();
        m.blob.resize(offset + count * sizeof(float));
        for (size_t i = 0; i < count; ++i) {
            const float v = norm ? 1.0f : float(int((i * 17 + m.tensors.size() * 3) % 29) - 14) / 64.0f;
            std::memcpy(m.blob.data() + offset + i * sizeof(float), &v, sizeof(v));
        }
        m.tensors.push_back({name, std::move(shape), gguf::GGML_TYPE_F32, 0});
        m.offsets.push_back(offset);
    };
    add("token_embd.weight", {8, 16});
    add("output_norm.weight", {8}, true);
    for (int l = 0; l < layers; ++l) {
        const std::string pre = "blk." + std::to_string(l) + ".";
        for (const char* name : {"attn_norm", "ffn_norm"}) add(pre + name + ".weight", {8}, true);
        for (const char* name : {"attn_q_norm", "attn_k_norm"}) add(pre + name + ".weight", {4}, true);
        add(pre + "attn_q.weight", {8, 8});
        add(pre + "attn_k.weight", {8, 4});
        add(pre + "attn_v.weight", {8, 4});
        add(pre + "attn_output.weight", {8, 8});
        add(pre + "ffn_gate.weight", {8, 12});
        add(pre + "ffn_up.weight", {8, 12});
        add(pre + "ffn_down.weight", {12, 8});
    }
    if (!tied) add("output.weight", {8, 16});
    return m;
}
