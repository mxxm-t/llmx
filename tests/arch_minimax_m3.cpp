// The minimax-m3 module (model/arch/minimax_m3.hpp) on a tiny file in memory: the configuration it reads, every required key and held value refused with its text, the plan of each layer kind, and the refusal of the architecture by name.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "format/gguf.hpp"
#include "model/arch/minimax_m3.hpp"
#include "model/arch/registry.hpp"
#include "model/runtime.hpp"

namespace {

int checks = 0;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
    ++checks;
}

void refuses(const std::string& what, const std::string& text, const std::function<void()>& run) {
    try {
        run();
    } catch (const std::exception& e) {
        require(std::string(e.what()) == text, what + ": refused with \"" + e.what() + "\", not \"" + text + "\"");
        return;
    }
    throw std::runtime_error(what + ": not refused");
}

const std::string P = "minimax-m3.";

// The tiny shape: three layers, the first dense, 8 query heads over 2 KV heads 16 wide, 8 rotary, 2 index heads 16 wide, 4 experts take 2.
constexpr uint64_t L = 3, LD = 1, E = 64, F = 96, HQ = 8, HKV = 2, D = 16, R = 8, DI = 16, X = 4, K = 2, FE = 32, V = 32;

gguf::MetaValue u32(uint64_t n) {
    gguf::MetaValue v;
    v.vtype = gguf::V_UINT32;
    v.u = n;
    return v;
}

gguf::MetaValue f32(float x) {
    gguf::MetaValue v;
    v.vtype = gguf::V_FLOAT32;
    std::memcpy(&v.fb, &x, sizeof(x));
    return v;
}

gguf::MetaValue flag(bool b) {
    gguf::MetaValue v;
    v.vtype = gguf::V_BOOL;
    v.b = b;
    return v;
}

void put(gguf::GGUFModel& m, const std::string& key, gguf::MetaValue v) {
    for (auto& kv : m.kv)
        if (kv.first == key) { kv.second = std::move(v); return; }
    m.kv.push_back({key, std::move(v)});
}

void drop(gguf::GGUFModel& m, const std::string& key) {
    m.kv.erase(std::remove_if(m.kv.begin(), m.kv.end(), [&](const auto& kv) { return kv.first == key; }), m.kv.end());
}

void add_tensor(gguf::GGUFModel& m, const std::string& name, std::vector<uint64_t> ne) {
    gguf::TensorInfo t{name, std::move(ne), quant::GGML_TYPE_F32, 0};
    m.tensors.push_back(t);
    m.add_tensor_data(std::vector<uint8_t>(size_t(t.data_size())));
}

// The keys the file of docs/MINIMAX.md carries, at the tiny shape.
const std::vector<std::pair<std::string, gguf::MetaValue>> keys = {
    {"block_count", u32(L)}, {"leading_dense_block_count", u32(LD)}, {"context_length", u32(4096)},
    {"embedding_length", u32(E)}, {"feed_forward_length", u32(F)},
    {"attention.head_count", u32(HQ)}, {"attention.head_count_kv", u32(HKV)},
    {"attention.key_length", u32(D)}, {"attention.value_length", u32(D)}, {"rope.dimension_count", u32(R)},
    {"rope.freq_base", f32(1e7f)}, {"attention.layer_norm_rms_epsilon", f32(1e-6f)},
    {"expert_count", u32(X)}, {"expert_used_count", u32(K)}, {"expert_feed_forward_length", u32(FE)},
    {"expert_shared_count", u32(1)}, {"expert_gating_func", u32(2)}, {"expert_weights_norm", flag(true)}, {"expert_weights_scale", f32(2.0f)},
    {"attention.indexer.head_count", u32(HKV)}, {"attention.indexer.key_length", u32(DI)}, {"attention.indexer.top_k", u32(3)},
    {"attention.indexer.block_size", u32(4)}, {"attention.indexer.local_blocks", u32(1)}};

gguf::GGUFModel tiny(const std::function<void(gguf::GGUFModel&)>& edit = {}) {
    gguf::GGUFModel m;
    gguf::MetaValue arch;
    arch.vtype = gguf::V_STRING;
    arch.s = "minimax-m3";
    m.kv.push_back({"general.architecture", arch});
    for (const auto& [k, v] : keys) m.kv.push_back({P + k, v});
    add_tensor(m, "token_embd.weight", {E, V});
    add_tensor(m, "output.weight", {E, V});
    add_tensor(m, "output_norm.weight", {E});
    for (uint64_t l = 0; l < L; ++l) {
        const std::string pre = "blk." + std::to_string(l) + ".";
        for (const char* n : {"attn_norm.weight", "ffn_norm.weight"}) add_tensor(m, pre + n, {E});
        add_tensor(m, pre + "attn_q.weight", {E, HQ * D});
        add_tensor(m, pre + "attn_k.weight", {E, HKV * D});
        add_tensor(m, pre + "attn_v.weight", {E, HKV * D});
        add_tensor(m, pre + "attn_output.weight", {HQ * D, E});
        add_tensor(m, pre + "attn_q_norm.weight", {D});
        add_tensor(m, pre + "attn_k_norm.weight", {D});
        add_tensor(m, pre + "indexer.q_proj.weight", {E, HKV * DI});
        add_tensor(m, pre + "indexer.k_proj.weight", {E, DI});
        add_tensor(m, pre + "indexer.q_norm.weight", {DI});
        add_tensor(m, pre + "indexer.k_norm.weight", {DI});
        if (l < LD) {
            add_tensor(m, pre + "ffn_gate.weight", {E, F});
            add_tensor(m, pre + "ffn_up.weight", {E, F});
            add_tensor(m, pre + "ffn_down.weight", {F, E});
        } else {
            add_tensor(m, pre + "ffn_gate_inp.weight", {E, X});
            add_tensor(m, pre + "exp_probs_b.bias", {X});
            add_tensor(m, pre + "ffn_gate_exps.weight", {E, FE, X});
            add_tensor(m, pre + "ffn_up_exps.weight", {E, FE, X});
            add_tensor(m, pre + "ffn_down_exps.weight", {FE, E, X});
            add_tensor(m, pre + "ffn_gate_shexp.weight", {E, FE});
            add_tensor(m, pre + "ffn_up_shexp.weight", {E, FE});
            add_tensor(m, pre + "ffn_down_shexp.weight", {FE, E});
        }
    }
    if (edit) edit(m);
    return m;
}

infer::minimax_m3::Config config_of(const gguf::GGUFModel& m) { return infer::minimax_m3::read_config(m, P); }

const infer::minimax_m3::MinimaxM3& module_of(const infer::ModelWeights& w) {
    const auto* arch = dynamic_cast<const infer::minimax_m3::MinimaxM3*>(w.arch.get());
    require(arch != nullptr, "the registry's minimax-m3 entry reads a MinimaxM3");
    return *arch;
}

void reads() {
    const infer::minimax_m3::Config c = config_of(tiny());
    require(c.n_layer == int(L) && c.n_dense == int(LD) && c.n_embd == int(E) && c.n_ff == int(F), "layers and widths");
    require(c.n_head == int(HQ) && c.n_head_kv == int(HKV) && c.head_dim == int(D) && c.rope_dim == int(R), "attention shape");
    require(c.rope_theta == 1e7f && c.rms_eps == 1e-6f && c.context_length == 4096, "rope base, epsilon and context");
    require(c.n_expert == int(X) && c.n_expert_used == int(K) && c.n_ff_exp == int(FE) && c.expert_scale == 2.0f, "experts");
    require(c.index_heads == int(HKV) && c.index_dim == int(DI) && c.index_top_k == 3 && c.index_block == 4, "index");
    require(config_of(tiny([](gguf::GGUFModel& m) { put(m, P + "leading_dense_block_count", u32(0)); })).n_dense == 0, "no dense lead");
}

// Every key the module reads, removed in turn, is refused as missing, but for the shared expert width, which takes the routed one.
void removed_keys() {
    for (const auto& kv : keys) {
        const std::string k = kv.first;
        refuses("without " + k, "inference: missing metadata " + P + k, [&] { config_of(tiny([&](gguf::GGUFModel& m) { drop(m, P + k); })); });
    }
    config_of(tiny([](gguf::GGUFModel& m) { put(m, P + "expert_shared_feed_forward_length", u32(FE)); }));
    require(true, "a shared expert as wide as the routed ones");
}

// Each value the module does not compute is refused with its text.
void held_values() {
    const std::string unsupported = "inference: unsupported metadata " + P;
    const std::pair<std::string, gguf::MetaValue> held[] = {
        {"expert_gating_func", u32(1)}, {"expert_weights_norm", flag(false)}, {"expert_shared_count", u32(2)},
        {"expert_shared_feed_forward_length", u32(2 * FE)}, {"attention.indexer.head_count", u32(1)},
        {"attention.indexer.local_blocks", u32(2)}, {"attention.value_length", u32(2 * D)}};
    for (const auto& kv : held)
        refuses(kv.first, unsupported + kv.first, [&] { config_of(tiny([&](gguf::GGUFModel& m) { put(m, P + kv.first, kv.second); })); });
    refuses("a layout", "inference: unsupported metadata " + P + "tensor_data_layout", [] {
        config_of(tiny([](gguf::GGUFModel& m) {
            gguf::MetaValue s;
            s.vtype = gguf::V_STRING;
            s.s = "other";
            put(m, P + "tensor_data_layout", s);
        }));
    });
    refuses("scaled RoPE", "inference: scaled RoPE is unsupported", [] { config_of(tiny([](gguf::GGUFModel& m) { put(m, P + "rope.scaling.factor", f32(4)); })); });
    refuses("no routed layer", "inference: minimax-m3 needs a routed layer after its dense ones",
            [] { config_of(tiny([](gguf::GGUFModel& m) { put(m, P + "leading_dense_block_count", u32(L)); })); });
    refuses("ragged GQA", "inference: head count must be divisible by KV head count", [] {
        config_of(tiny([](gguf::GGUFModel& m) {
            put(m, P + "attention.head_count_kv", u32(3));
            put(m, P + "attention.indexer.head_count", u32(3));
        }));
    });
    const std::string rotary = "inference: invalid rotary width, which must be even and at most the head and index widths";
    for (uint64_t r : {uint64_t(7), 2 * D})
        refuses("rotary width " + std::to_string(r), rotary, [&] { config_of(tiny([&](gguf::GGUFModel& m) { put(m, P + "rope.dimension_count", u32(r)); })); });
    refuses("an index narrower than the rotation", rotary,
            [] { config_of(tiny([](gguf::GGUFModel& m) { put(m, P + "attention.indexer.key_length", u32(R / 2)); })); });
    refuses("too many experts a token", "inference: more experts per token than the layer has, or above 256",
            [] { config_of(tiny([](gguf::GGUFModel& m) { put(m, P + "expert_used_count", u32(X + 1)); })); });
}

// Each layer's roles fit the file's tensors, the dense lead runs the dense block and the rest the routed one with the shared expert, and every layer keeps its index key per position.
void plan() {
    const gguf::GGUFModel m = tiny();
    const infer::ModelWeights w = infer::gguf_weights(m);
    const infer::TensorIndex index(w.tensors);
    const infer::ModelPlan p = module_of(w).plan(index);
    size_t roles = p.pass.size();
    for (const infer::Role& role : p.pass)
        require(infer::role_fits(role, index.view(index.at(role.name)).shape, index.view(index.at(role.name)).type), role.name + " fits its role");
    require(p.layers.size() == L && p.vocab == V && p.role_ids == infer::minimax_m3::role_count, "layers, vocabulary and roles");
    for (size_t l = 0; l < L; ++l) {
        const infer::LayerPlan& layer = p.layers[l];
        const bool routed = l >= LD;
        require(layer.routed == routed && layer.kind == (routed ? infer::minimax_m3::routed : infer::minimax_m3::dense), "layer " + std::to_string(l) + "'s kind");
        require(layer.cache == infer::Cache::kv && layer.kv_aux == DI && !layer.kv_pool, "layer " + std::to_string(l) + " keeps its index key a position");
        require(layer.roles.size() == (routed ? 20u : 15u), "layer " + std::to_string(l) + "'s role count");
        for (const infer::Role& role : layer.roles)
            require(infer::role_fits(role, index.view(index.at(role.name)).shape, index.view(index.at(role.name)).type), role.name + " fits its role");
        roles += layer.roles.size();
        const bool rope = std::any_of(layer.ops.begin(), layer.ops.end(), [](const infer::OpUse& u) { return u.op == backend::Op::norm_rope_partial; });
        const auto mixer = [&](backend::Op op) { return std::any_of(layer.ops.begin(), layer.ops.end(), [&](const infer::OpUse& u) { return u.op == op && u.part == infer::Part::mixer; }); };
        const auto declares = [&](backend::Op op) { return std::any_of(layer.ops.begin(), layer.ops.end(), [&](const infer::OpUse& u) { return u.op == op && u.part == infer::Part::ffn; }); };
        require(rope && declares(backend::Op::routed_experts) == routed && declares(backend::Op::sigmoid_routing) == routed &&
                declares(backend::Op::silu_mul_clamp) && declares(backend::Op::glu_quick_gelu) && mixer(backend::Op::indexer_scores) &&
                mixer(backend::Op::raw_index_scores) && mixer(backend::Op::candidate_blocks) &&
                mixer(backend::Op::sparse_attention) && mixer(backend::Op::sparse_attention_blocks), "layer " + std::to_string(l) + "'s ops");
    }
    require(roles == w.tensors.size(), "the plan reads every tensor of the file");
    // The fixture's context of 4096 holds 1024 blocks of 4, and a row keeps 3 of them an index head.
    require(p.slots.size() == 17 && p.slots[8] == std::max(F, (K + 1) * FE) && p.slots[12] == HKV * DI && p.slots[13] == DI && p.slots[14] == HKV * 1024 &&
            p.slots[15] == HKV && p.slots[16] == HKV * 3 && p.index_width == HKV, "arena slots");
    require(p.widths.shared_ff == FE && p.widths.routed == K && p.table_steps[0] == R / 2, "pass widths and rope tables");
}

// The products' widths count the shared expert's, and the dense block's only where a dense layer exists.
void widths() {
    const auto of = [](const std::function<void(gguf::GGUFModel&)>& edit) {
        const gguf::GGUFModel m = tiny(edit);
        const infer::ModelWeights w = infer::gguf_weights(m);
        return module_of(w).plan(infer::TensorIndex(w.tensors)).widths;
    };
    const uint64_t wide = 4 * HQ * D;
    const backend::PassWidths a = of([&](gguf::GGUFModel& m) { put(m, P + "expert_feed_forward_length", u32(wide)); });
    require(a.product_in == wide && a.product_out == 2 * wide, "the shared expert's product widths");
    const backend::PassWidths b = of([&](gguf::GGUFModel& m) {
        put(m, P + "leading_dense_block_count", u32(0));
        put(m, P + "feed_forward_length", u32(wide));
        for (const char* n : {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) {
            const auto t = std::find_if(m.tensors.begin(), m.tensors.end(), [&](const gguf::TensorInfo& i) { return i.name == std::string("blk.0.") + n; });
            m.offsets.erase(m.offsets.begin() + (t - m.tensors.begin()));
            m.tensors.erase(t);
        }
    });
    require(b.product_in == HQ * D && b.product_out == HQ * D + 2 * HKV * D, "no dense block's widths without a dense layer");
}

// A tensor the layer's kind does not read is refused, and a dense layer's plan names its index projections, so plan_model refuses a file without them.
void tensors() {
    for (const auto& [layer, name] : {std::pair{0, "ffn_gate_inp.weight"}, std::pair{0, "ffn_gate_exps.weight"}, std::pair{1, "ffn_gate.weight"}, std::pair{2, "ffn_up.weight"}}) {
        const std::string full = "blk." + std::to_string(layer) + "." + name;
        const gguf::GGUFModel m = tiny([&](gguf::GGUFModel& f) { add_tensor(f, full, {E, F}); });
        const infer::ModelWeights w = infer::gguf_weights(m);
        refuses(full, "inference: a tensor its layer's kind does not read " + full, [&] { module_of(w).plan(infer::TensorIndex(w.tensors)); });
    }
    const gguf::GGUFModel m = tiny();
    const infer::ModelWeights w = infer::gguf_weights(m);
    const infer::ModelPlan p = module_of(w).plan(infer::TensorIndex(w.tensors));
    const auto& roles = p.layers[0].roles;
    require(std::any_of(roles.begin(), roles.end(), [](const infer::Role& r) { return r.name == "blk.0.indexer.k_proj.weight"; }), "a dense layer reads its index");
}

// The registry names the module.
void named() {
    require(std::string(infer::architecture_of(tiny()).name) == "minimax-m3", "the registry's entry");
}

} // namespace

int main() {
    try {
        reads();
        removed_keys();
        held_values();
        plan();
        widths();
        tensors();
        named();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "arch-minimax-m3: " << checks << " checks passed" << std::endl;
    return 0;
}
