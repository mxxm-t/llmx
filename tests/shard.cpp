#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/runtime.hpp"
#include "model/place.hpp"
#include "model/shard.hpp"
#include "model/arch/registry.hpp"

// The tensor split's shards (docs/TENSOR-SPLIT.md, section 4.2, model/shard.hpp): spans by hand, every member's packed bytes against an oracle written here for every storage type, and a member's footprint against hand counts.

namespace {

size_t checks = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

// A model's views over random bytes: the tensors a plan reads, each `type` where it is a matrix and F32 otherwise.
struct Views {
    std::vector<infer::TensorView> views;
    std::vector<std::vector<uint8_t>> bytes;
    void add(const std::string& name, std::vector<uint64_t> shape, uint32_t type, std::mt19937& rng) {
        size_t rows = 1;
        for (size_t d = 1; d < shape.size(); ++d) rows *= size_t(shape[d]);
        bytes.emplace_back(quant::row_bytes(type, size_t(shape[0]), rows));
        for (uint8_t& b : bytes.back()) b = uint8_t(rng());
        views.push_back({name, std::move(shape), type, nullptr, 0});
    }
    // Points every view at its bytes, once no more are added.
    void bind() {
        for (size_t i = 0; i < views.size(); ++i) {
            views[i].data = bytes[i].data();
            views[i].bytes = bytes[i].size();
        }
    }
};

// A dense qwen3 of two layers whose every split falls on whole 256-value blocks at widths 2 and 4, KV heads replicated at 4: 8 heads of 128 over 2 KV heads, a hidden width of 1024 and a vocabulary of 64.
infer::ModelWeights qwen3_model(uint32_t type, Views& v, bool tied) {
    infer::qwen3::Config c;
    c.n_layer = 2, c.n_embd = 256, c.n_ff = 1024, c.n_head = 8, c.n_head_kv = 2, c.head_dim = 128, c.context_length = 256;
    std::mt19937 rng(7 + type);
    const uint64_t E = 256, Q = 1024, KV = 256, F = 1024;
    v.add("token_embd.weight", {E, 64}, type, rng);
    if (!tied) v.add("output.weight", {E, 64}, type, rng);
    v.add("output_norm.weight", {E}, quant::GGML_TYPE_F32, rng);
    for (int l = 0; l < c.n_layer; ++l) {
        const std::string pre = "blk." + std::to_string(l) + ".";
        v.add(pre + "attn_norm.weight", {E}, quant::GGML_TYPE_F32, rng);
        v.add(pre + "attn_q_norm.weight", {128}, quant::GGML_TYPE_F32, rng);
        v.add(pre + "attn_k_norm.weight", {128}, quant::GGML_TYPE_F32, rng);
        v.add(pre + "attn_q.weight", {E, Q}, type, rng);
        v.add(pre + "attn_k.weight", {E, KV}, type, rng);
        v.add(pre + "attn_v.weight", {E, KV}, type, rng);
        v.add(pre + "attn_output.weight", {Q, E}, type, rng);
        v.add(pre + "ffn_norm.weight", {E}, quant::GGML_TYPE_F32, rng);
        v.add(pre + "ffn_gate.weight", {E, F}, type, rng);
        v.add(pre + "ffn_up.weight", {E, F}, type, rng);
        v.add(pre + "ffn_down.weight", {F, E}, type, rng);
    }
    v.bind();
    return {std::make_shared<const infer::qwen3::Qwen3>(c), v.views};
}

// A dense qwen35 of a linear and a full-attention layer: 8 K heads of 64 and 24 V heads of 128 in three tiles, 4 heads of 256 over 4 KV heads with their gates, the rest as qwen3_model.
// With `routed`, as qwen35moe: each dense block replaced by 4 experts of 512, 2 a token, and a shared expert of 512, the widths of Qwen3.6-35B-A3B.
infer::ModelWeights qwen35_model(uint32_t type, Views& v, bool routed = false) {
    infer::qwen35::Config c;
    c.n_layer = 2, c.n_embd = 256, c.n_ff = routed ? 0 : 1024, c.n_head = 4, c.n_head_kv = 4, c.head_dim = 256, c.rope_dim = 64, c.context_length = 256;
    c.k_heads = 8, c.v_heads = 24, c.k_dim = 64, c.v_dim = 128;
    c.full = {0, 1};
    if (routed) c.n_expert = 4, c.n_expert_used = 2, c.n_ff_exp = 512, c.n_ff_shexp = 512;
    std::mt19937 rng(11 + type);
    const uint64_t E = 256, F = 1024, V = 24 * 128, C = 2 * 8 * 64 + V;
    v.add("token_embd.weight", {E, 64}, type, rng);
    v.add("output.weight", {E, 64}, type, rng);
    v.add("output_norm.weight", {E}, quant::GGML_TYPE_F32, rng);
    for (int l = 0; l < 2; ++l) {
        const std::string pre = "blk." + std::to_string(l) + ".";
        v.add(pre + "attn_norm.weight", {E}, quant::GGML_TYPE_F32, rng);
        v.add(pre + "post_attention_norm.weight", {E}, quant::GGML_TYPE_F32, rng);
        if (l) {
            v.add(pre + "attn_q.weight", {E, 2 * 4 * 256}, type, rng);
            v.add(pre + "attn_k.weight", {E, 4 * 256}, type, rng);
            v.add(pre + "attn_v.weight", {E, 4 * 256}, type, rng);
            v.add(pre + "attn_q_norm.weight", {256}, quant::GGML_TYPE_F32, rng);
            v.add(pre + "attn_k_norm.weight", {256}, quant::GGML_TYPE_F32, rng);
            v.add(pre + "attn_output.weight", {4 * 256, E}, type, rng);
        } else {
            v.add(pre + "attn_qkv.weight", {E, C}, type, rng);
            v.add(pre + "attn_gate.weight", {E, V}, type, rng);
            v.add(pre + "ssm_alpha.weight", {E, 24}, type, rng);
            v.add(pre + "ssm_beta.weight", {E, 24}, type, rng);
            v.add(pre + "ssm_conv1d.weight", {4, C}, quant::GGML_TYPE_F32, rng);
            v.add(pre + "ssm_a", {24}, quant::GGML_TYPE_F32, rng);
            v.add(pre + "ssm_dt.bias", {24}, quant::GGML_TYPE_F32, rng);
            v.add(pre + "ssm_norm.weight", {128}, quant::GGML_TYPE_F32, rng);
            v.add(pre + "ssm_out.weight", {V, E}, type, rng);
        }
        if (routed) {
            v.add(pre + "ffn_gate_inp.weight", {E, 4}, quant::GGML_TYPE_F32, rng);
            v.add(pre + "ffn_gate_exps.weight", {E, 512, 4}, type, rng);
            v.add(pre + "ffn_up_exps.weight", {E, 512, 4}, type, rng);
            v.add(pre + "ffn_down_exps.weight", {512, E, 4}, type, rng);
            v.add(pre + "ffn_gate_inp_shexp.weight", {E}, quant::GGML_TYPE_F32, rng);
            v.add(pre + "ffn_gate_shexp.weight", {E, 512}, type, rng);
            v.add(pre + "ffn_up_shexp.weight", {E, 512}, type, rng);
            v.add(pre + "ffn_down_shexp.weight", {512, E}, type, rng);
            continue;
        }
        v.add(pre + "ffn_gate.weight", {E, F}, type, rng);
        v.add(pre + "ffn_up.weight", {E, F}, type, rng);
        v.add(pre + "ffn_down.weight", {F, E}, type, rng);
    }
    v.bind();
    return {std::make_shared<const infer::qwen35::Qwen35>(c), v.views};
}

// The members of `width` that hold position `i` of `role`'s axis, from the declaration's meaning alone: the section and tile it lies in, its unit there, and that unit's owners, whether shared among the members or replicated.
std::vector<size_t> owners(const infer::Role& role, uint64_t i, size_t width) {
    if (role.shard.axis == infer::Axis::none || width == 1) {
        std::vector<size_t> all(width);
        for (size_t m = 0; m < width; ++m) all[m] = m;
        return all;
    }
    for (const infer::ShardSection& s : role.shard.sections) {
        const uint64_t tile = s.units * s.unit;
        if (i >= s.tiles * tile) {
            i -= s.tiles * tile;
            continue;
        }
        const uint64_t unit = (i % tile) / s.unit;
        std::vector<size_t> out;
        if (s.units % width == 0) out.push_back(size_t(unit / (s.units / width)));
        else
            for (size_t m = 0; m < width; ++m)
                if (m / (width / s.units) == unit) out.push_back(m);
        return out;
    }
    throw std::runtime_error("a position past the axis of " + role.name);
}

bool owns(const infer::Role& role, uint64_t i, size_t width, size_t member) {
    for (size_t m : owners(role, i, width))
        if (m == member) return true;
    return false;
}

// Member `member`'s bytes of `role`'s tensor, gathered here: whole rows it owns in order on the row axis, and on the column axis each row's blocks it owns, a block owned where its first column is.
std::vector<uint8_t> expected(const infer::Role& role, const infer::TensorView& t, size_t width, size_t member) {
    const quant::StorageType* type = quant::storage_type(t.type);
    const size_t nin = size_t(t.shape[0]), row = quant::row_bytes(t.type, nin), rows = t.bytes / row;
    std::vector<uint8_t> out;
    if (role.shard.axis != infer::Axis::columns) {
        for (size_t r = 0; r < rows; ++r)
            if (role.shard.axis == infer::Axis::none || owns(role, r, width, member)) out.insert(out.end(), t.data + r * row, t.data + (r + 1) * row);
        return out;
    }
    for (size_t r = 0; r < rows; ++r)
        for (size_t b = 0; b < nin / type->block_size; ++b)
            if (owns(role, b * type->block_size, width, member)) {
                const uint8_t* at = t.data + r * row + b * type->type_size;
                out.insert(out.end(), at, at + type->type_size);
            }
    return out;
}

// The plan passes the checks at `width`, and each split role, every member: its runs pack the oracle's bytes, its shape counts them, and unless replicated the members' bytes add up to the tensor's.
void reassemble(const infer::ModelPlan& plan, const std::vector<infer::TensorView>& views, size_t width, const std::string& label) {
    infer::shard::check_plan(plan, views, width);
    auto each = [&](const infer::Role& role) {
        if (!role.tensor) return;
        const infer::TensorView& t = views[*role.tensor];
        size_t total = 0;
        bool replicated = false;
        for (const infer::ShardSection& s : role.shard.sections) replicated = replicated || s.units % width != 0;
        for (size_t m = 0; m < width; ++m) {
            const std::vector<infer::shard::Run> runs = infer::shard::runs(role, t, width, m);
            std::vector<uint8_t> packed(infer::shard::bytes(runs));
            infer::shard::pack(runs, t.data, packed.data());
            require(packed == expected(role, t, width, m), label + ": member " + std::to_string(m) + "'s bytes of " + role.name + " are not its shard");
            const std::vector<uint64_t> shape = infer::shard::shape(role, t, width, m);
            size_t rows = 1;
            for (size_t d = 1; d < shape.size(); ++d) rows *= size_t(shape[d]);
            require(quant::row_bytes(t.type, size_t(shape[0]), rows) == packed.size(), label + ": member " + std::to_string(m) + "'s shape of " + role.name + " does not hold its bytes");
            for (size_t k = 1; k < runs.size(); ++k)
                require(runs[k].from >= runs[k - 1].from + runs[k - 1].bytes && runs[k].to == runs[k - 1].to + runs[k - 1].bytes,
                        label + ": the runs of " + role.name + " are not in the tensor's order, packed back to back");
            total += packed.size();
        }
        if (role.shard.axis != infer::Axis::none && !replicated) require(total == t.bytes, label + ": the members of " + role.name + " do not hold each byte once");
    };
    for (const infer::Role& r : plan.pass) each(r);
    for (const infer::LayerPlan& layer : plan.layers)
        for (const infer::Role& r : layer.roles) each(r);
}

const infer::Role& role_named(const infer::ModelPlan& plan, const std::string& name) {
    for (const infer::Role& r : plan.pass)
        if (r.name == name) return r;
    for (const infer::LayerPlan& layer : plan.layers)
        for (const infer::Role& r : layer.roles)
            if (r.name == name) return r;
    throw std::runtime_error("no role " + name);
}

std::string spans_text(const std::vector<infer::shard::Span>& spans) {
    std::string s;
    for (const auto& x : spans) s += "[" + std::to_string(x.begin) + "+" + std::to_string(x.count) + "]";
    return s;
}

// A qwen35 linear layer's spans written out: attn_qkv's q and k rows by K head and its V rows from each tile, ssm_out's columns the same, and the decay's elements by V head.
void spans_by_hand() {
    Views v;
    const infer::ModelWeights w = qwen35_model(quant::GGML_TYPE_F32, v);
    const infer::ModelPlan plan = infer::plan_model(w);
    const infer::Role qkv = role_named(plan, "blk.0.attn_qkv.weight");
    // q rows 0..512, k 512..1024, V heads of 128 rows from 1024, tile t's head h at 1024 + (8 t + h) 128; member 1 of 2 takes K heads 4 to 7.
    require(spans_text(infer::shard::spans(qkv, 2, 1)) == "[256+256][768+256][1536+512][2560+512][3584+512]", "attn_qkv's spans at width 2: " + spans_text(infer::shard::spans(qkv, 2, 1)));
    require(spans_text(infer::shard::spans(qkv, 4, 0)) == "[0+128][512+128][1024+256][2048+256][3072+256]", "attn_qkv's spans at width 4: " + spans_text(infer::shard::spans(qkv, 4, 0)));
    require(spans_text(infer::shard::spans(role_named(plan, "blk.0.ssm_out.weight"), 4, 3)) == "[768+256][1792+256][2816+256]", "ssm_out's columns at width 4");
    require(spans_text(infer::shard::spans(role_named(plan, "blk.0.ssm_a"), 2, 0)) == "[0+4][8+4][16+4]", "the decay's V heads at width 2");
    require(spans_text(infer::shard::spans(role_named(plan, "blk.0.ssm_conv1d.weight"), 2, 0)) == "[0+256][512+256][1024+512][2048+512][3072+512]", "the conv's channels at width 2");
    {
        const infer::Role norm = role_named(plan, "blk.0.ssm_norm.weight");
        const std::vector<infer::shard::Run> runs = infer::shard::runs(norm, w.tensors[*norm.tensor], 4, 2);
        require(runs.size() == 1 && runs[0].from == 0 && runs[0].bytes == 128 * 4 && runs[0].to == 0, "ssm_norm is whole on every member");
    }
    // A full-attention layer's q heads of 256 rows with their gates, and its heads' 256 columns of attn_output.
    require(spans_text(infer::shard::spans(role_named(plan, "blk.1.attn_q.weight"), 4, 1)) == "[512+512]", "the gated q head at width 4");
    require(spans_text(infer::shard::spans(role_named(plan, "blk.1.attn_output.weight"), 2, 1)) == "[512+512]", "attn_output's columns at width 2");
    // qwen3's two KV heads over four members: members 0 and 1 hold head 0, 2 and 3 head 1.
    Views v3;
    const infer::ModelPlan dense = infer::plan_model(qwen3_model(quant::GGML_TYPE_F32, v3, false));
    for (size_t m = 0; m < 4; ++m)
        require(spans_text(infer::shard::spans(role_named(dense, "blk.0.attn_k.weight"), 4, m)) == (m < 2 ? "[0+128]" : "[128+128]"), "a replicated KV head at width 4");
    require(spans_text(infer::shard::spans(role_named(dense, "output.weight"), 4, 3)) == "[48+16]", "the head's vocabulary rows at width 4");
    require(spans_text(infer::shard::spans(role_named(dense, "blk.0.ffn_down.weight"), 2, 0)) == "[0+512]", "ffn_down's columns at width 2");
    require(spans_text(infer::shard::spans(role_named(dense, "blk.0.ffn_down.weight"), 1, 0)) == "[0+1024]", "the whole axis at width 1");
}

// A routed qwen3 layer whose experts' hidden width is three 256-value blocks, 768, as Qwen3-30B-A3B's is: 4 experts, the stacks in `type` and the rest F32.
infer::ModelWeights routed_model(uint32_t type, Views& v) {
    infer::qwen3::Config c;
    c.n_layer = 1, c.n_embd = 256, c.n_ff = 0, c.n_head = 8, c.n_head_kv = 2, c.head_dim = 128, c.context_length = 256;
    c.n_expert = 4, c.n_expert_used = 2, c.n_ff_exp = 768;
    std::mt19937 rng(11 + type);
    const uint32_t f32 = quant::GGML_TYPE_F32;
    const uint64_t E = 256, Q = 1024, KV = 256, F = 768, X = 4;
    v.add("token_embd.weight", {E, 64}, f32, rng);
    v.add("output_norm.weight", {E}, f32, rng);
    for (const char* n : {"attn_norm", "ffn_norm"}) v.add(std::string("blk.0.") + n + ".weight", {E}, f32, rng);
    for (const char* n : {"attn_q_norm", "attn_k_norm"}) v.add(std::string("blk.0.") + n + ".weight", {128}, f32, rng);
    v.add("blk.0.attn_q.weight", {E, Q}, f32, rng);
    v.add("blk.0.attn_k.weight", {E, KV}, f32, rng);
    v.add("blk.0.attn_v.weight", {E, KV}, f32, rng);
    v.add("blk.0.attn_output.weight", {Q, E}, f32, rng);
    v.add("blk.0.ffn_gate_inp.weight", {E, X}, f32, rng);
    v.add("blk.0.ffn_gate_exps.weight", {E, F, X}, type, rng);
    v.add("blk.0.ffn_up_exps.weight", {E, F, X}, type, rng);
    v.add("blk.0.ffn_down_exps.weight", {F, E, X}, type, rng);
    v.bind();
    return {std::make_shared<const infer::qwen3::Qwen3>(c), v.views};
}

// Covering blocks (ShardSection::align): where an expert's hidden width over the group's width is not whole quant blocks of its down stack, a member holds the whole blocks that cover its even share.
// Its down shard is the file's bytes of every covered column; its gate and up shards hold a row a covered column, the file's where the member owns the column and zero bytes elsewhere; over the members each row is owned once.
void covers() {
    for (uint32_t type : {(uint32_t)quant::GGML_TYPE_Q4_K, (uint32_t)quant::GGML_TYPE_Q6_K, (uint32_t)quant::GGML_TYPE_Q8_0}) {
        Views v;
        const infer::ModelWeights w = routed_model(type, v);
        const infer::ModelPlan plan = infer::plan_model(w);
        const size_t block = quant::storage_type(type)->block_size, F = 768, X = 4, E = 256;
        const infer::Role down = role_named(plan, "blk.0.ffn_down_exps.weight");
        if (block == 256) {
            require(spans_text(infer::shard::spans(down, 2, 0)) == "[0+512]" && spans_text(infer::shard::spans(down, 2, 1)) == "[256+512]", "the down stack's covers at width 2");
            require(spans_text(infer::shard::spans(down, 4, 0)) == "[0+256]" && spans_text(infer::shard::spans(down, 4, 1)) == "[0+512]" &&
                    spans_text(infer::shard::spans(down, 4, 2)) == "[256+512]" && spans_text(infer::shard::spans(down, 4, 3)) == "[512+256]", "the down stack's covers at width 4");
            require(spans_text(infer::shard::spans(role_named(plan, "blk.0.ffn_gate_exps.weight"), 2, 1)) == "[256+512][1024+512][1792+512][2560+512]", "the gate stack's covered rows at width 2");
        }
        for (size_t width : {size_t(2), size_t(4)}) {
            infer::shard::check_plan(plan, w.tensors, width);
            std::vector<size_t> owners(F, 0);
            for (size_t m = 0; m < width; ++m) {
                const size_t own0 = m * F / width, own1 = (m + 1) * F / width, c0 = own0 / block * block, c1 = (own1 + block - 1) / block * block;
                for (size_t r = own0; r < own1; ++r) ++owners[r];
                const std::string at = std::string(quant::storage_type(type)->name) + " at width " + std::to_string(width) + ", member " + std::to_string(m);
                for (const char* name : {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"}) {
                    const infer::Role role = role_named(plan, name);
                    const infer::TensorView& t = w.tensors[*role.tensor];
                    const std::vector<infer::shard::Run> runs = infer::shard::runs(role, t, width, m);
                    const size_t row = quant::row_bytes(type, E);
                    std::vector<uint8_t> packed(infer::shard::bytes(runs), 0xa5);
                    infer::shard::pack(runs, t.data, packed.data());
                    require(packed.size() == X * (c1 - c0) * row && infer::shard::share(role, width, m) == c1 - c0 && infer::shard::shape(role, t, width, m)[1] == c1 - c0, at + ": the covered rows of " + name);
                    bool ok = true;
                    for (size_t x = 0; x < X; ++x)
                        for (size_t r = c0; r < c1; ++r) {
                            const uint8_t* got = packed.data() + (x * (c1 - c0) + r - c0) * row;
                            const bool owned = r >= own0 && r < own1;
                            for (size_t b = 0; b < row; ++b) ok = ok && got[b] == (owned ? t.data[(x * F + r) * row + b] : 0);
                        }
                    require(ok, at + ": " + name + " does not hold the file's rows where it owns a column and zero rows elsewhere");
                    for (size_t k = 1; k < runs.size(); ++k)
                        require(runs[k - 1].zero ? runs[k].zero : (runs[k].zero || runs[k].from >= runs[k - 1].from + runs[k - 1].bytes), at + ": the file's runs of " + name + " are not in the tensor's order before its zero runs");
                }
                const infer::TensorView& t = w.tensors[*down.tensor];
                const std::vector<infer::shard::Run> runs = infer::shard::runs(down, t, width, m);
                std::vector<uint8_t> packed(infer::shard::bytes(runs));
                infer::shard::pack(runs, t.data, packed.data());
                const size_t row = quant::row_bytes(type, F), part = quant::row_bytes(type, c1 - c0), skip = quant::row_bytes(type, c0);
                bool ok = packed.size() == X * E * part && infer::shard::shape(down, t, width, m)[0] == c1 - c0;
                for (size_t r = 0; ok && r < X * E; ++r) ok = !std::memcmp(packed.data() + r * part, t.data + r * row + skip, part);
                require(ok, at + ": the down stack does not hold the file's bytes of every covered column");
            }
            require(std::all_of(owners.begin(), owners.end(), [](size_t n) { return n == 1; }), "an expert's hidden row is not owned by exactly one member");
        }
    }
    // A shared expert is covered as one more expert: 512 wide in Q4_K at width 4, a member owns 128 of its columns and holds the block of 256 around them, its gate rows the file's where it owns the column and zero elsewhere.
    Views v;
    const uint32_t type = quant::GGML_TYPE_Q4_K;
    const infer::ModelWeights w = qwen35_model(type, v, true);
    const infer::ModelPlan plan = infer::plan_model(w);
    infer::shard::check_plan(plan, w.tensors, 4);
    const infer::Role down = role_named(plan, "blk.0.ffn_down_shexp.weight"), gate = role_named(plan, "blk.0.ffn_gate_shexp.weight");
    const infer::TensorView& t = w.tensors[*gate.tensor];
    const size_t row = quant::row_bytes(type, 256);
    for (size_t m = 0; m < 4; ++m) {
        const size_t c0 = m / 2 * 256, own0 = m * 128;
        require(spans_text(infer::shard::spans(down, 4, m)) == "[" + std::to_string(c0) + "+256]" && infer::shard::shape(down, w.tensors[*down.tensor], 4, m)[0] == 256, "a shared expert's down projection is not covered by one block at width 4");
        const std::vector<infer::shard::Run> runs = infer::shard::runs(gate, t, 4, m);
        std::vector<uint8_t> packed(infer::shard::bytes(runs), 0xa5);
        infer::shard::pack(runs, t.data, packed.data());
        bool ok = packed.size() == 256 * row;
        for (size_t r = 0; ok && r < 256; ++r)
            for (size_t b = 0; ok && b < row; ++b) ok = packed[r * row + b] == (c0 + r >= own0 && c0 + r < own0 + 128 ? t.data[(c0 + r) * row + b] : 0);
        require(ok, "a shared expert's gate rows at width 4, member " + std::to_string(m));
    }
}

// Every storage type a matrix may have, at widths 1, 2 and 4, on both architectures; the K-quants and every other 256-value block on the columns too.
void every_type() {
    size_t types = 0;
    for (uint32_t id = 0; id < 64; ++id) {
        if (!quant::storage_type(id)) continue;
        ++types;
        for (size_t width : {size_t(1), size_t(2), size_t(4)}) {
            const std::string at = std::string(quant::storage_type(id)->name) + " at width " + std::to_string(width);
            for (bool tied : {false, true}) {
                Views v;
                const infer::ModelWeights w = qwen3_model(id, v, tied);
                reassemble(infer::plan_model(w), w.tensors, width, "qwen3 " + at);
            }
            Views v;
            const infer::ModelWeights w = qwen35_model(id, v);
            reassemble(infer::plan_model(w), w.tensors, width, "qwen35 " + at);
        }
    }
    require(types >= 30, "the storage types were not all reached");
}

// A member's footprint against counts made here: its rows of every split matrix at Q8_0, the whole norms and embedding, its vocabulary rows' logits, its KV heads' cache and its heads' state; at width 1 the footprint of one device.
void footprints() {
    infer::ModelOptions o;
    o.kv_tokens = 256;
    const size_t q8 = quant::row_bytes(quant::GGML_TYPE_Q8_0, 256);
    {
        Views v;
        const infer::ModelWeights w = qwen3_model(quant::GGML_TYPE_Q8_0, v, false);
        const infer::ModelPlan plan = infer::plan_model(w);
        const infer::Footprint one = infer::footprint(w, plan, o), whole = infer::footprint(w, plan, o, 1, 0);
        require(one.layers[0].size() == whole.layers[0].size() && one.cache == whole.cache && one.output.bytes == whole.output.bytes, "width 1 is not one device's footprint");
        for (size_t width : {size_t(2), size_t(4)})
            for (size_t m = 0; m < width; ++m) {
                const infer::Footprint fp = infer::footprint(w, plan, o, width, m);
                size_t layer = 0;
                for (const infer::Matrix& x : fp.layers[0]) layer += x.bytes;
                // Norms 256 + 128 + 128 + 256 floats; q 1024 / W rows, k and v 256 / W rows (one 128-row head when replicated at 4), attn_output 256 rows of 1024 / W columns, gate and up 1024 / W rows, down 256 rows of 1024 / W columns.
                const size_t kv_rows = width == 4 ? 128 : 256 / width;
                const size_t hand = 768 * 4 + (1024 / width) * q8 + 2 * kv_rows * q8 + 256 * quant::row_bytes(quant::GGML_TYPE_Q8_0, 1024 / width) +
                                    2 * (1024 / width) * q8 + 256 * quant::row_bytes(quant::GGML_TYPE_Q8_0, 1024 / width);
                require(layer == hand, "a qwen3 member's layer bytes at width " + std::to_string(width));
                require(fp.output.bytes == (64 / width) * q8 && fp.output.rows == 64 / width && fp.logits_per_row == 64 * 4 && fp.head_slice_per_row == (width > 1 ? (64 / width) * 4 : 0), "a member's head rows, beside the host's whole logits rows");
                require(fp.embedding.bytes == 64 * q8, "the embedding is whole on every member");
                const size_t kv_heads = width == 4 ? 1 : 2 / width;
                require(fp.cache[0] == 256 * kv_heads * 128 * 2 * 2, "a member's KV cache at width " + std::to_string(width));
            }
    }
    {
        // A tied head on one device is the embedding's buffer; a member's head is a shard of its own beside the whole table, so a group counts both.
        Views v;
        const infer::ModelWeights w = qwen3_model(quant::GGML_TYPE_Q8_0, v, true);
        const infer::ModelPlan plan = infer::plan_model(w);
        const infer::Footprint one = infer::footprint(w, plan, o), half = infer::footprint(w, plan, o, 2, 0);
        require(one.tied && one.output.bytes == 64 * q8, "a tied head on one device is not the embedding's buffer");
        require(!half.tied && half.embedding.bytes == 64 * q8 && half.output.bytes == 32 * q8, "a member's tied head is not counted as its shard beside the whole embedding");
    }
    {
        Views v;
        const infer::ModelWeights w = qwen35_model(quant::GGML_TYPE_Q8_0, v);
        const infer::ModelPlan plan = infer::plan_model(w);
        for (size_t width : {size_t(2), size_t(4)}) {
            const infer::Footprint fp = infer::footprint(w, plan, o, width, 0);
            // The state of one slot: each of the member's 24 / W V heads a 64 by 128 matrix, then three carried rows of its channels.
            const size_t channels = (2 * 8 * 64 + 24 * 128) / width;
            require(fp.cache[0] == ((24 / width) * 64 * 128 + 3 * channels) * 4, "a qwen35 member's state at width " + std::to_string(width));
            require(fp.cache[1] == 256 * (4 / width) * 256 * 2 * 2, "a qwen35 member's KV cache at width " + std::to_string(width));
            // With a mark of 17 rows: one more slot, each row's saved inputs at the member's share, its channels and the alpha and beta of its V heads, and the rerun's room, which is not split.
            infer::ModelOptions marked = o;
            marked.mark_slots = 1;
            marked.mark_rows = 17;
            require(infer::footprint(w, plan, marked, width, 0).cache[0] ==
                        2 * fp.cache[0] + 17 * (channels + 2 * (24 / width)) * 4 + 17 * infer::recur_floats(plan, plan.layers[0]) * 4,
                    "a qwen35 member's state with a mark at width " + std::to_string(width));
            size_t layer = 0;
            for (const infer::Matrix& x : fp.layers[0]) layer += x.bytes;
            // Norms 2 x 256 floats, ssm_norm 128, the conv's channels of 4 floats, the decay and time step 24 / W each; attn_qkv's channels, z's V rows, alpha's and beta's V heads and the dense matrices by rows or columns.
            const size_t V = 24 * 128 / width;
            const size_t hand = (512 + 128 + 4 * channels + 2 * (24 / width)) * 4 + channels * q8 + V * q8 + 2 * (24 / width) * q8 +
                                2 * (1024 / width) * q8 + 256 * quant::row_bytes(quant::GGML_TYPE_Q8_0, 1024 / width) + 256 * quant::row_bytes(quant::GGML_TYPE_Q8_0, V);
            require(layer == hand, "a qwen35 member's linear layer bytes at width " + std::to_string(width));
        }
    }
}

} // namespace

int main() {
    try {
        spans_by_hand();
        every_type();
        footprints();
        covers();
        std::cout << "shard: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "shard: FAILED: " << e.what() << "\n";
        return 1;
    }
}
