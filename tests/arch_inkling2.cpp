// The inkling2_loop configuration and plan (docs/INKLING2.md): the fixture's keys read, every key required, the values the model assumes held and the loop's rules refused.
// The execution layers the loop runs, their roles, caches and tables.
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/arch/inkling2/model.hpp"
#include "model/arch/registry.hpp"

namespace {
int failures = 0;

void require(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

std::string refusal(const std::function<void()>& f) {
    try { f(); } catch (const std::exception& e) { return e.what(); }
    return {};
}

gguf::MetaValue u32(uint64_t v) {
    gguf::MetaValue m;
    m.vtype = gguf::V_UINT32;
    m.u = v;
    return m;
}

gguf::MetaValue f32(float v) {
    gguf::MetaValue m;
    m.vtype = gguf::V_FLOAT32;
    std::memcpy(&m.fb, &v, sizeof v);
    m.f64 = v;
    return m;
}

gguf::MetaValue boolean(bool v) {
    gguf::MetaValue m;
    m.vtype = gguf::V_BOOL;
    m.b = v;
    return m;
}

template <typename T>
gguf::MetaValue array(uint32_t type, const std::vector<T>& values, gguf::MetaValue (*make)(T)) {
    gguf::MetaValue m;
    m.vtype = gguf::V_ARRAY;
    m.u = type;
    for (const T& v : values) m.arr.push_back(make(v));
    return m;
}

gguf::MetaValue u32_of(uint64_t v) { return u32(v); }
gguf::MetaValue bool_of(bool v) { return boolean(v); }

// The keys tests/fixture_inkling2.py writes, at its values: 12 layers, every third full attention, the loop over layers 3 to 8 three times.
std::vector<std::pair<std::string, gguf::MetaValue>> fixture_keys() {
    const std::string p = "inkling2_loop.";
    std::vector<uint64_t> heads, kv;
    std::vector<bool> sliding;
    for (int l = 0; l < 12; ++l) {
        const bool global = l % 3 == 2;
        heads.push_back(6);
        kv.push_back(2);
        sliding.push_back(!global);
    }
    return {
        {p + "block_count", u32(12)}, {p + "context_length", u32(4096)}, {p + "embedding_length", u32(20)}, {p + "feed_forward_length", u32(8)},
        {p + "expert_feed_forward_length", u32(8)}, {p + "expert_count", u32(6)}, {p + "expert_used_count", u32(2)},
        {p + "vocab_size", u32(300)}, {p + "vocab_size_unpadded", u32(288)},
        {p + "attention.head_count", array<uint64_t>(gguf::V_UINT32, heads, u32_of)},
        {p + "attention.head_count_kv", array<uint64_t>(gguf::V_UINT32, kv, u32_of)},
        {p + "attention.sliding_window_pattern", array<bool>(gguf::V_BOOL, sliding, bool_of)},
        {p + "attention.key_length", u32(4)}, {p + "attention.value_length", u32(4)},
        {p + "attention.key_length_swa", u32(4)}, {p + "attention.value_length_swa", u32(4)},
        {p + "loop.count", u32(3)}, {p + "loop.start", u32(3)}, {p + "loop.end", u32(9)}, {p + "loop.gate_rank", u32(3)}, {p + "loop.varying", boolean(false)},
        {p + "attn_res.block_size", u32(3)}, {p + "attention.sliding_window", u32(5)}, {p + "attention.relative_length", u32(3)},
        {p + "attention.relative_extent", u32(7)}, {p + "attention.log_scale_floor", u32(8)}, {p + "rope.dimension_count", u32(0)},
        {p + "shortconv.l_cache", u32(4)}, {p + "attention.layer_norm_rms_epsilon", f32(1e-6f)}, {p + "expert_weights_scale", f32(8)},
        {p + "activation.situ_beta", f32(4)}, {p + "activation.situ_linear_beta", f32(25)}, {p + "final_logit_softcapping", f32(30)},
        {p + "attention.qk_clip", f32(1.5f)}, {p + "attention.relative_clip", f32(0.75f)}, {p + "attention.log_scale_alpha", f32(0.1f)},
    };
}

gguf::GGUFModel file(const std::vector<std::pair<std::string, gguf::MetaValue>>& keys) {
    gguf::GGUFModel m;
    for (const auto& [k, v] : keys) m.kv.push_back({k, v});
    return m;
}

// The keys with `key` set to `value`.
gguf::GGUFModel with(const std::string& key, const gguf::MetaValue& value) {
    auto keys = fixture_keys();
    for (auto& kv : keys)
        if (kv.first == "inkling2_loop." + key) kv.second = value;
    return file(keys);
}

void reads_fixture() {
    const infer::inkling2::Config c = infer::inkling2::read_config(file(fixture_keys()));
    require(c.n_layer == 12 && c.loop_start == 3 && c.loop_end == 9 && c.loop_count == 3 && c.block_size == 3 && c.vocab_unpadded == 288 &&
                c.heads[2] == 6 && c.kv_heads[2] == 2 && !c.sliding[2] && c.sliding[0] && c.window == 5 && c.rel_extent == 7,
            "the fixture's keys were not read as written");
    const std::vector<int> run = infer::inkling2::execution_layers(c);
    std::vector<int> want{0, 1, 2};
    for (int pass = 0; pass < 3; ++pass)
        for (int l = 3; l < 9; ++l) want.push_back(l);
    for (int l = 9; l < 12; ++l) want.push_back(l);
    require(run == want, "the execution layers are not the layers before the loop, the loop three times and the rest");
}

void every_key_required() {
    for (size_t i = 0; i < fixture_keys().size(); ++i) {
        auto keys = fixture_keys();
        const std::string key = keys[i].first;
        keys.erase(keys.begin() + long(i));
        const std::string why = refusal([&] { infer::inkling2::read_config(file(keys)); });
        require(why == "inference: missing metadata " + key, key + " removed gave \"" + why + "\"");
    }
}

void held_values() {
    const std::string p = "inference: unsupported metadata inkling2_loop.";
    const auto refused = [&](const std::string& key, const gguf::MetaValue& v, const std::string& text) {
        const std::string why = refusal([&] { infer::inkling2::read_config(with(key, v)); });
        require(why == text, key + " changed gave \"" + why + "\", not \"" + text + "\"");
    };
    refused("loop.varying", boolean(true), p + "loop.varying");
    refused("rope.dimension_count", u32(8), p + "rope.dimension_count");
    refused("shortconv.l_cache", u32(3), p + "shortconv.l_cache");
    refused("expert_feed_forward_length", u32(16), p + "expert_feed_forward_length");
    refused("attention.value_length", u32(8), p + "attention.value_length");
    refused("attention.value_length_swa", u32(8), p + "attention.value_length_swa");
    refused("vocab_size_unpadded", u32(301), "inference: the unpadded vocabulary is larger than the padded one");
    refused("expert_used_count", u32(7), "inference: more experts per token than the layer has");
    refused("loop.end", u32(13), "inference: the loop's layers [3, 13) are not layers of the model");
    const std::string blocks = "inference: the loop's layers do not start and end on attention-residual blocks of at least two, or the model has more than four blocks";
    refused("loop.start", u32(4), blocks);
    refused("loop.end", u32(6), blocks);
    refused("attn_res.block_size", u32(2), blocks);
    refused("attention.head_count_kv", array<uint64_t>(gguf::V_UINT32, std::vector<uint64_t>(12, 4), u32_of), "inference: layer 0's query heads are not whole groups of its KV heads");
    refused("attention.sliding_window_pattern", array<bool>(gguf::V_BOOL, std::vector<bool>(11, true), bool_of),
            "inference: the Inkling-2 per-layer head counts and window pattern do not cover its 12 layers");
}
// The fixture's tensors as tests/fixture_inkling2.py writes them, shapes only: E 20, six heads of four over two KV heads, R 3, six experts of 8, gate rank 3.
std::vector<infer::TensorView> fixture_tensors(const std::vector<std::string>& extra = {}, const std::string& without = "") {
    std::vector<std::pair<std::string, std::vector<uint64_t>>> t = {{"token_embd.weight", {20, 300}}, {"token_embd_norm.weight", {20}}, {"output_norm.weight", {20}},
                                                                    {"output_res_query.weight", {20}}, {"output.weight", {20, 300}}};
    for (int l = 0; l < 12; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const bool glob = l % 3 == 2, loop = l >= 3 && l < 9;
        for (const char* n : {"ffn_res_query", "ffn_norm", "exp_probs_b"}) t.push_back({p + n + (std::string(n) == "exp_probs_b" ? ".bias" : ".weight"), {std::string(n) == "exp_probs_b" ? 6u : 20u}});
        t.push_back({p + "attn_q.weight", {20, 24}});
        t.push_back({p + "attn_gate.weight", {20, 24}});
        t.push_back({p + "attn_k.weight", {20, 8}});
        t.push_back({p + "attn_v.weight", {20, 8}});
        t.push_back({p + "attn_output.weight", {24, 20}});
        t.push_back({p + "attn_sink_proj.weight", {20, 6}});
        t.push_back({p + "attn_rel.weight", {20, 18}});
        t.push_back({p + "attn_q_clip.weight", {4, 6}});
        t.push_back({p + "attn_rel_clip.weight", {3, 6}});
        t.push_back({p + "attn_rel_bias.weight", {3, glob ? 7u : 5u}});
        t.push_back({p + "attn_k_conv.weight", {4, 8}});
        t.push_back({p + "attn_v_conv.weight", {4, 8}});
        t.push_back({p + "attn_post_conv.weight", {4, 20}});
        t.push_back({p + "ffn_post_conv.weight", {4, 20}});
        t.push_back({p + "ffn_gate_inp.weight", {20, 6}});
        t.push_back({p + "ffn_gate_up_exps.weight", {20, 16, 6}});
        t.push_back({p + "ffn_down_exps.weight", {8, 20, 6}});
        if (l > 0) t.push_back({p + "attn_res_query.weight", glob ? std::vector<uint64_t>{20, 3} : std::vector<uint64_t>{20}});
        if (glob) {
            for (const char* n : {"attn_q_inp_norm", "attn_k_inp_norm", "attn_v_inp_norm"}) t.push_back({p + n + ".weight", {20}});
            t.push_back({p + "attn_k_conv_proj.weight", {20, 8}});
            t.push_back({p + "attn_v_conv_proj.weight", {20, 8}});
        } else {
            t.push_back({p + "attn_norm.weight", {20}});
        }
        if (loop) {
            t.push_back({p + "loop_msg_query.weight", {20, 4}});
            for (const char* g : {"q", "k", "v", "ffn"}) {
                t.push_back({p + "loop_" + g + "_ctrl_down.weight", {40, 3}});
                t.push_back({p + "loop_" + g + "_ctrl_forget.weight", {3, 20}});
                t.push_back({p + "loop_" + g + "_ctrl_gain.weight", {3, 20}});
            }
        }
    }
    for (const std::string& name : extra) t.push_back({name, {20}});
    std::vector<infer::TensorView> views;
    for (const auto& [name, shape] : t)
        if (name != without) views.push_back({name, shape, quant::GGML_TYPE_F32, nullptr, 0});
    return views;
}

void plans_fixture() {
    const infer::inkling2::Inkling2 arch(infer::inkling2::read_config(file(fixture_keys())), {});
    const std::vector<infer::TensorView> views = fixture_tensors();
    const infer::TensorIndex index(views);
    const infer::ModelPlan p = arch.layout(index);
    require(p.layers.size() == 24 && p.vocab == 288 && p.residual == 120 && p.slots.size() == size_t(infer::inkling2::head_rows) + 1 && p.slots.front() == 120 && p.slots.back() == 120 && p.role_ids == infer::inkling2::role_count,
            "the plan's layers, vocabulary or residual are not the fixture's");
    const infer::Role& head = p.pass.back();
    require(head.id == infer::inkling2::output && head.out == 288 && head.tensor_rows == 300, "the head does not read the unpadded rows of the padded output");
    for (const infer::Role& role : p.pass) require(infer::role_fits(role, views[index.at(role.name)].shape, quant::GGML_TYPE_F32), "pass role does not fit " + role.name);
    for (size_t e = 0; e < p.layers.size(); ++e)
        for (const infer::Role& role : p.layers[e].roles)
            require(infer::role_fits(role, views[index.at(role.name)].shape, quant::GGML_TYPE_F32), "layer " + std::to_string(e) + " role does not fit " + role.name);
    using namespace infer::inkling2;
    // Execution layer 9 is the loop's second pass over layer 3, 23 the last full-attention layer.
    require(p.layers[3].kind == (block_boundary | opens_pass | reads_block_query | in_loop) && p.layers[9].kind == (block_boundary | opens_pass | reads_block_query | in_loop | second_pass) &&
                p.layers[17].kind == (global | reads_block_query | in_loop | third_pass) && p.layers[0].kind == block_boundary && p.layers[23].kind == (global | reads_block_query),
            "the execution layers' kinds are not the fixture's");
    require(p.layers[9].roles.front().name == "blk.3.attn_norm.weight" && p.layers[9].roles.size() == p.layers[3].roles.size(), "a pass of the loop does not repeat its layer's roles");
    require(p.layers[0].roles.size() == 22 && p.layers[2].roles.size() == 27 && p.layers[3].roles.size() == 36, "a layer's role count is not its kind's");
    // A sliding layer keeps its K and V ring and its convolution rings, (4 + 5) x 2 x 8 + (3 + 5) x (16 + 40), the headroom the window below 64; a full-attention layer the rings and a pooled K and V row.
    require(p.layers[0].cache == infer::Cache::entries && p.layers[0].state_tail == 592 && !p.layers[0].kv_pool && p.layers[0].headroom == 5,
            "a sliding layer's entries are not its rings");
    require(p.layers[2].kv_pool == 1 && p.layers[2].kv_pool_width == 16 && p.layers[2].state_tail == 448, "a full-attention layer's entries are not its rings and pooled rows");
    // Every layer declares its F32 widths as its native format and hands its feed-forward part the slot that holds that part's ring.
    for (const infer::LayerPlan& l : p.layers)
        require(l.native && l.native->pool_width == l.kv_pool_width && l.native->state_tail == l.state_tail && l.ffn_reads_state,
                "a layer's native widths are not its F32 widths or its feed-forward part does not read its slot");
    // The relative bias is a source the tables hold transposed: the zero and ones rows, then R x extent for each physical layer, 3 x 5 sliding and 3 x 7 full.
    require(p.tables.size() == 14 && p.tables[0] == 20 && p.tables[1] == 40 && p.tables[2] == 15 && p.tables[4] == 21, "the plan's tables are not the fixture's");
    require(arch.plan(index).layers.size() == 24, "the plan is not the layout");
}

void refuses_tensors() {
    const infer::inkling2::Inkling2 arch(infer::inkling2::read_config(file(fixture_keys())), {});
    const auto refused = [&](const std::vector<infer::TensorView>& views, const std::string& want) {
        const infer::TensorIndex index(views);
        const std::string got = refusal([&] { arch.layout(index); });
        require(got == want, "expected \"" + want + "\", got \"" + got + "\"");
    };
    const std::string kind = "inference: a tensor its layer's kind does not read ";
    refused(fixture_tensors({"blk.0.attn_k_conv_proj.weight"}), kind + "blk.0.attn_k_conv_proj.weight");
    refused(fixture_tensors({"blk.2.attn_norm.weight"}), kind + "blk.2.attn_norm.weight");
    refused(fixture_tensors({"blk.0.attn_res_query.weight"}), kind + "blk.0.attn_res_query.weight");
    refused(fixture_tensors({"blk.9.loop_ffn_ctrl_gain.weight"}), kind + "blk.9.loop_ffn_ctrl_gain.weight");
    refused(fixture_tensors({}, "blk.5.ffn_post_conv.weight"), "inference: missing tensor blk.5.ffn_post_conv.weight");
    refused(fixture_tensors({}, "output_res_query.weight"), "inference: missing tensor output_res_query.weight");
    gguf::GGUFModel m = file(fixture_keys());
    gguf::MetaValue name;
    name.vtype = gguf::V_STRING;
    name.s = "inkling2_loop";
    m.kv.push_back({"general.architecture", name});
    require(std::string(infer::architecture_of(m).name) == "inkling2_loop", "the registry does not name inkling2_loop");
}
} // namespace

int main() {
    try {
        reads_fixture();
        every_key_required();
        held_values();
        plans_fixture();
        refuses_tensors();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
    std::cout << (failures ? "arch-inkling2: FAIL\n" : "arch-inkling2: PASS\n");
    return failures ? 1 : 0;
}
