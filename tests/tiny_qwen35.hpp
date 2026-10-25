#pragma once
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include "format/gguf.hpp"
#include "quant/quant.hpp"

// A tiny qwen35 model built in memory, which arch-qwen35 and image-prompt share.
namespace tiny35 {

constexpr uint64_t E = 8, F = 12, HQ = 2, HKV = 1, D = 8, HK = 1, HV = 2, DK = 4, DV = 4, VOCAB = 32, CTX = 64;
constexpr uint64_t C = 2 * HK * DK + HV * DV;

// A tensor of `shape` appended to `m`, F32 from a fixed pattern of its index and the tensors before it, times `scale` plus `offset`.
inline void add_tensor(gguf::GGUFModel& m, const std::string& name, std::vector<uint64_t> shape, float scale = 1.0f, float offset = 0.0f) {
    size_t count = 1;
    for (uint64_t d : shape) count *= size_t(d);
    const size_t at = m.blob.size();
    m.blob.resize(at + count * sizeof(float));
    for (size_t i = 0; i < count; ++i) {
        const float v = offset + scale * float(int((i * 17 + m.tensors.size() * 7) % 29) - 14) / 64.0f;
        std::memcpy(m.blob.data() + at + i * sizeof(float), &v, sizeof(v));
    }
    m.tensors.push_back({name, std::move(shape), quant::GGML_TYPE_F32, 0});
    m.offsets.push_back(at);
}


// Four layers, linear then full attention twice, 8 wide over a vocabulary of 32, with a rotary width of 4 of each head's 8; `edit` changes the metadata and tensors before the model is read.
inline gguf::GGUFModel tiny(const std::function<void(gguf::GGUFModel&)>& edit = {}) {
    gguf::GGUFModel m;
    auto meta = [&](const std::string& key, uint64_t value) {
        gguf::MetaValue v;
        v.vtype = gguf::V_UINT32;
        v.u = value;
        m.kv.push_back({"qwen35." + key, v});
    };
    gguf::MetaValue arch;
    arch.vtype = gguf::V_STRING;
    arch.s = "qwen35";
    m.kv.push_back({"general.architecture", arch});
    for (const auto& kv : std::vector<std::pair<std::string, uint64_t>>{
             {"block_count", 4}, {"embedding_length", E}, {"feed_forward_length", F}, {"attention.head_count", HQ},
             {"attention.head_count_kv", HKV}, {"attention.key_length", D}, {"attention.value_length", D}, {"rope.dimension_count", 4},
             {"context_length", CTX}, {"ssm.conv_kernel", 4}, {"ssm.state_size", DK}, {"ssm.group_count", HK},
             {"ssm.time_step_rank", HV}, {"ssm.inner_size", HV * DV}, {"full_attention_interval", 2}})
        meta(kv.first, kv.second);
    gguf::MetaValue sections;
    sections.vtype = gguf::V_ARRAY;
    sections.u = gguf::V_INT32;
    for (int s : {1, 1, 0, 0}) {
        gguf::MetaValue e;
        e.vtype = gguf::V_INT32;
        e.i = s;
        sections.arr.push_back(e);
    }
    m.kv.push_back({"qwen35.rope.dimension_sections", sections});
    auto add = [&](const std::string& name, std::vector<uint64_t> shape, float scale = 1.0f, float offset = 0.0f) {
        add_tensor(m, name, std::move(shape), scale, offset);
    };
    add("token_embd.weight", {E, VOCAB});
    add("output_norm.weight", {E}, 0.1f, 1.0f);
    for (int l = 0; l < 4; ++l) {
        const std::string pre = "blk." + std::to_string(l) + ".";
        add(pre + "attn_norm.weight", {E}, 0.1f, 1.0f);
        add(pre + "post_attention_norm.weight", {E}, 0.1f, 1.0f);
        if (l % 2) {
            add(pre + "attn_q.weight", {E, 2 * HQ * D});
            add(pre + "attn_k.weight", {E, HKV * D});
            add(pre + "attn_v.weight", {E, HKV * D});
            add(pre + "attn_q_norm.weight", {D}, 0.1f, 1.0f);
            add(pre + "attn_k_norm.weight", {D}, 0.1f, 1.0f);
            add(pre + "attn_output.weight", {HQ * D, E});
        } else {
            add(pre + "attn_qkv.weight", {E, C});
            add(pre + "attn_gate.weight", {E, HV * DV});
            add(pre + "ssm_alpha.weight", {E, HV});
            add(pre + "ssm_beta.weight", {E, HV}, 4.0f);
            add(pre + "ssm_conv1d.weight", {4, C}, 4.0f);
            add(pre + "ssm_a", {HV}, 4.0f, -1.0f);
            add(pre + "ssm_dt.bias", {HV});
            add(pre + "ssm_norm.weight", {DV}, 0.1f, 1.0f);
            add(pre + "ssm_out.weight", {HV * DV, E});
        }
        add(pre + "ffn_gate.weight", {E, F});
        add(pre + "ffn_up.weight", {E, F});
        add(pre + "ffn_down.weight", {F, E});
    }
    if (edit) edit(m);
    return m;
}

}  // namespace tiny35
