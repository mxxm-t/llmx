#pragma once
// Deterministic Flash weights shared by the model MTP oracle and server integration checks.
#include <algorithm>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "model/arch/qwen4exp.hpp"
#include "model/runtime.hpp"

namespace flash_mtp {
using Vec = std::vector<double>;
inline constexpr size_t E = 4, H = 2, R = E * H, V = 8;
struct Observed final : infer::Architecture {
    std::shared_ptr<const infer::Architecture> base;
    mutable std::vector<Vec> hidden;
    bool capture = false;
    explicit Observed(std::shared_ptr<const infer::Architecture> a) : base(std::move(a)) {}
    infer::ModelPlan plan(const infer::TensorIndex& t) const override { return base->plan(t); }
    void plan_drafter(const infer::TensorIndex& t, infer::ModelPlan& p) const override { base->plan_drafter(t, p); }
    void fill_tables(std::vector<std::vector<float>>& t) const override { base->fill_tables(t); }
    void embed(const infer::Step& s, const uint32_t* ids) const override { base->embed(s, ids); }
    void mixer(const infer::Step& s) const override { base->mixer(s); }
    void ffn(const infer::Step& s) const override { base->ffn(s); }
    void head(const infer::HeadStep& s) const override { base->head(s); }
    void recur(const infer::Step& s) const override { base->recur(s); }
    void draft_rows(const infer::DraftRowsStep& s) const override {
        if (capture) {
            std::vector<float> rows(s.rows * R);
            s.b.read(*s.x.buffer, s.x.offset * sizeof(float), rows.data(), rows.size() * sizeof(float));
            for (size_t i = 0; i < s.rows; ++i) hidden.emplace_back(rows.begin() + i * R, rows.begin() + (i + 1) * R);
        }
        base->draft_rows(s);
    }
    void draft(const infer::DraftStep& s) const override { base->draft(s); }
};

struct Fixture {
    infer::qwen4exp::Config c;
    std::shared_ptr<Observed> observed;
    infer::ModelWeights weights;
    std::map<std::string, std::vector<float>> data;
    Fixture(bool split, int compression, bool group_capable = false, bool ple = false,
            size_t trunk_layers = 2, size_t group_width = 2) {
        using namespace infer::qwen4exp;
        if (trunk_layers != 2 && trunk_layers != 3) throw std::runtime_error("Flash fixture supports two or three trunk layers");
        c.n_layer = int(trunk_layers); c.n_nextn = 1; c.n_embd = E;
        c.n_expert = 2; c.n_expert_used = 1; c.n_ff_exp = 4; c.n_ff_shexp = 4;
        c.n_head = 1; c.n_head_kv = 1; c.head_dim = E; c.rope_dim = 2; c.context_length = 64;
        c.k_heads = 1; c.v_heads = 2; c.k_dim = 2; c.v_dim = 2;
        if (group_capable) {
            if (group_width != 2 && group_width != 4)
                throw std::runtime_error("MTP group fixture width must be 2 or 4");
            c.n_head = c.n_head_kv = c.k_heads = int(group_width);
            c.v_heads = int(group_width); c.v_dim = 4 / int(group_width);
        }
        c.layers = trunk_layers == 2 ? std::vector<LayerKind>{linear_attention, qwen_sparse_attention}
                                     : std::vector<LayerKind>{linear_attention, qwen_sparse_attention, linear_attention};
        c.compress_ratios = trunk_layers == 2 ? std::vector<int>{0, 2} : std::vector<int>{0, 2, 0};
        c.rope_sections = {1, 0, 0, 0}; c.rope_theta = 10000.0f;
        c.indexer_n_heads = 1; c.indexer_head_dim = E; c.indexer_budget = 4; c.indexer_compress_ratio = 2;
        c.nextn_compress_ratio = compression; c.hc_count = H; c.hc_low_rank = 2;
        if (ple) {
            c.has_ple = true; c.ple_layer = 0; c.ple_ngram_size = 3; c.ple_heads_per_ngram = 1;
            c.ple_conv_kernel = 4; c.ple_embedding_length = 1; c.ple_embedding_dim = 2; c.ple_eos_token_id = 0;
            c.ple_layer_multipliers = {1, 3, 5}; c.ple_head_offsets = {0, V}; c.ple_head_vocab_sizes = {V, V};
        }
        observed = std::make_shared<Observed>(std::make_shared<Qwen4Exp>(c, uint32_t(V)));
        weights.arch = observed;
        weights.tensors = {{"token_embd.weight", {E, V}},
                           {"blk." + std::to_string(c.n_layer) + ".nextn.eh_proj.weight", {split ? E : 2 * E, E}}};
        if (ple) weights.tensors.push_back({"per_layer_token_embd.weight", {size_t(c.ple_embedding_length), size_t(2 * V)}});
        infer::TensorIndex seed(weights.tensors);
        auto p = observed->plan(seed);
        observed->plan_drafter(seed, p);
        std::map<std::string, infer::Role> roles;
        for (const auto& role : p.pass) roles.emplace(role.name, role);
        for (const auto& layer : p.layers) for (const auto& role : layer.roles) roles.emplace(role.name, role);
        for (const auto& role : p.drafter->roles) roles.emplace(role.name, role);
        weights.tensors.clear();
        for (const auto& item : roles) {
            const auto& role = item.second;
            const size_t count = size_t(role.in * role.out * std::max<uint64_t>(role.experts, 1));
            auto& values = data[item.first];
            values.resize(count);
            uint32_t hash = 2166136261u;
            for (unsigned char ch : item.first) hash = (hash ^ ch) * 16777619u;
            for (size_t i = 0; i < count; ++i) {
                hash = hash * 1664525u + 1013904223u;
                values[i] = float(int((hash >> 16) % 101) - 50) / 128.0f;
                if (role.kind == infer::RoleKind::norm) values[i] += 1.0f;
            }
            std::vector<uint64_t> shape{role.in};
            if (role.out != 1 || role.kind != infer::RoleKind::norm) shape.push_back(role.out);
            if (role.experts) shape.push_back(role.experts);
            weights.tensors.push_back({item.first, shape, quant::GGML_TYPE_F32, reinterpret_cast<const uint8_t*>(values.data()), count * sizeof(float)});
        }
    }
};

inline gguf::MetaValue integer(uint64_t n) { gguf::MetaValue v; v.vtype = gguf::V_UINT32; v.u = n; return v; }
inline gguf::MetaValue string(const std::string& s) { gguf::MetaValue v; v.vtype = gguf::V_STRING; v.s = s; return v; }
inline gguf::MetaValue array(const std::vector<uint64_t>& items) {
    gguf::MetaValue v; v.vtype = gguf::V_ARRAY; v.u = gguf::V_UINT32;
    for (auto n : items) v.arr.push_back(integer(n));
    return v;
}
inline gguf::MetaValue array64(const std::vector<uint64_t>& items) {
    gguf::MetaValue v; v.vtype = gguf::V_ARRAY; v.u = gguf::V_UINT64;
    for (auto n : items) { gguf::MetaValue item; item.vtype = gguf::V_UINT64; item.u = n; v.arr.push_back(std::move(item)); }
    return v;
}

inline std::pair<gguf::GGUFModel, gguf::GGUFModel> headers(const Fixture& f, int layout) {
    gguf::GGUFModel target, side;
    auto& kv = target.kv;
    kv.push_back({"general.architecture", string("qwen4exp")});
    for (const auto& item : std::vector<std::pair<std::string, uint64_t>>{
        {"block_count", uint64_t(f.c.n_layer)}, {"embedding_length", E}, {"expert_count", 2}, {"expert_used_count", 1},
        {"expert_feed_forward_length", E}, {"expert_shared_feed_forward_length", E},
        {"attention.head_count", uint64_t(f.c.n_head)}, {"attention.head_count_kv", uint64_t(f.c.n_head_kv)}, {"attention.key_length", E},
        {"attention.value_length", E}, {"rope.dimension_count", 2}, {"context_length", uint64_t(f.c.context_length)},
        {"ssm.conv_kernel", 4}, {"ssm.state_size", 2}, {"ssm.group_count", uint64_t(f.c.k_heads)}, {"ssm.time_step_rank", 2}, {"ssm.inner_size", 4},
        {"hyper_connection.count", H}, {"hyper_connection.low_rank", 2}, {"attention.indexer.head_count", 1},
        {"attention.indexer.key_length", E}, {"attention.indexer.top_k", 4}})
        kv.push_back({"qwen4exp." + item.first, integer(item.second)});
    kv.push_back({"qwen4exp.rope.dimension_sections", array({1, 0, 0, 0})});
    std::vector<uint64_t> ratios;
    for (int ratio : f.c.compress_ratios) ratios.push_back(uint64_t(ratio));
    kv.push_back({"qwen4exp.attention.compress_ratios", array(ratios)});
    kv.push_back({"qwen4exp.full_attention_interval", integer(2)});
    gguf::MetaValue theta; theta.vtype = gguf::V_FLOAT32; std::memcpy(&theta.fb, &f.c.rope_theta, sizeof(float));
    kv.push_back({"qwen4exp.rope.freq_base", theta});
    if (f.c.has_ple) {
        kv.push_back({"qwen4exp.ple.layers", array({uint64_t(f.c.ple_layer)})});
        kv.push_back({"qwen4exp.ple.ngram_size", integer(uint64_t(f.c.ple_ngram_size))});
        kv.push_back({"qwen4exp.ple.heads_per_ngram", integer(uint64_t(f.c.ple_heads_per_ngram))});
        kv.push_back({"qwen4exp.ple.conv_kernel", integer(uint64_t(f.c.ple_conv_kernel))});
        kv.push_back({"qwen4exp.embedding_length_per_layer_input", integer(uint64_t(f.c.ple_embedding_length))});
        kv.push_back({"qwen4exp.ple.eos_token_id", integer(uint64_t(f.c.ple_eos_token_id))});
        kv.push_back({"qwen4exp.ple.layer_multipliers", array64(f.c.ple_layer_multipliers)});
        kv.push_back({"qwen4exp.ple.head_offsets", array64(f.c.ple_head_offsets)});
        kv.push_back({"qwen4exp.ple.head_vocab_sizes", array64(f.c.ple_head_vocab_sizes)});
    }
    kv.push_back({"tokenizer.ggml.model", string("gpt2")});
    kv.push_back({"tokenizer.ggml.pre", string("qwen2")});
    gguf::MetaValue tokens; tokens.vtype = gguf::V_ARRAY; tokens.u = gguf::V_STRING;
    for (size_t i = 0; i < V; ++i) tokens.arr.push_back(string(std::string(1, char('a' + i))));
    kv.push_back({"tokenizer.ggml.tokens", tokens});
    side.kv = target.kv;
    for (auto& item : side.kv) {
        if (item.first == "qwen4exp.block_count") item.second = integer(uint64_t(f.c.n_layer + 1));
        if (item.first == "qwen4exp.attention.compress_ratios") {
            std::vector<uint64_t> with_mtp = ratios;
            with_mtp.push_back(uint64_t(f.c.nextn_compress_ratio));
            item.second = array(with_mtp);
        }
    }
    side.kv.push_back({"qwen4exp.nextn_predict_layers", integer(1)});
    if (layout == 1) {
        gguf::MetaValue shared; shared.vtype = gguf::V_BOOL; shared.b = true;
        side.kv.push_back({"qwen4exp.nextn_shared_target_tensors", shared});
    }
    if (layout && f.c.nextn_compress_ratio) {
        std::vector<uint64_t> legacy(size_t(f.c.n_layer), 0);
        legacy.push_back(uint64_t(f.c.nextn_compress_ratio));
        for (auto& item : side.kv)
            if (item.first == "qwen4exp.attention.compress_ratios") item.second = array(legacy);
    }
    for (const auto& tensor : f.weights.tensors) {
        gguf::TensorInfo info; info.name = tensor.name; info.ne = tensor.shape; info.type = tensor.type;
        (tensor.name.find("blk." + std::to_string(f.c.n_layer) + ".") == 0 ? side : target).tensors.push_back(info);
    }
    return {std::move(target), std::move(side)};
}

inline std::map<std::string, std::vector<float>> private_globals(const Fixture& f, gguf::GGUFModel& side) {
    std::map<std::string, std::vector<float>> data;
    for (const std::string name : {"token_embd.weight", "output.weight"}) {
        auto& values = data[name]; values = f.data.at(name);
        for (float& value : values) value = -1.25f * value + 0.0625f;
        side.tensors.push_back({name, {E, V}, quant::GGML_TYPE_F32, 0});
    }
    return data;
}

// Owns only its uniquely named temporary fixture directory.
struct Files {
    std::filesystem::path directory;
    std::string target_path, side_path;
    Files(const Fixture& f, gguf::GGUFModel target, gguf::GGUFModel side,
          const std::map<std::string, std::vector<float>>& private_data = {}) {
        directory = std::filesystem::temp_directory_path() / ("llmx-flash-mtp-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(directory);
        target_path = (directory / "model.gguf").string();
        side_path = (directory / "mtp.gguf").string();
        try {
            for (auto* file : {&target, &side})
                gguf::write_gguf(*file, file == &target ? target_path : side_path, [&](size_t i) {
                    const auto& name = file->tensors[i].name;
                    const auto& values = file == &side && private_data.count(name) ? private_data.at(name) : f.data.at(name);
                    return reinterpret_cast<const uint8_t*>(values.data());
                });
        } catch (...) { clear(); throw; }
    }
    Files(const Files&) = delete;
    Files& operator=(const Files&) = delete;
    void clear() { std::error_code ec; std::filesystem::remove_all(directory, ec); }
    ~Files() { clear(); }
};
} // namespace flash_mtp
