// The qwen35 module (model/arch/qwen35.hpp) on tiny models built in memory: every refused key and tensor with its text, the plan of each layer kind, the footprint, and the runtime's rules for a model whose layers keep a recurrent state.
// The math is held to HF by the suite's qwen35 component; here a split and slices are held to one CPU bit for bit, which the per-token recurrence gives.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <array>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "model/runtime.hpp"
#include "model/place.hpp"
#include "model/arch/registry.hpp"
#include "tiny_qwen.hpp"

// Fails the calling thread's Nth allocation after arming, once, so a case can fail each allocation of one call in turn.
static thread_local size_t fail_allocation = 0;

void* operator new(std::size_t n) {
    if (fail_allocation && --fail_allocation == 0) throw std::bad_alloc();
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
#if defined(__GNUC__) && !defined(__clang__)
// GCC takes the free below for a mismatch with operator new, though the replaced new above allocates with malloc.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace {

int checked = 0;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
    ++checked;
}

// `run` must throw a runtime_error or logic_error whose text holds `text`.
void refuses(const std::string& what, const std::string& text, const std::function<void()>& run) {
    try {
        run();
    } catch (const std::exception& e) {
        require(std::string(e.what()).find(text) != std::string::npos, what + ": refused with \"" + e.what() + "\", not \"" + text + "\"");
        return;
    }
    throw std::runtime_error(what + ": not refused");
}

constexpr uint64_t E = 8, F = 12, HQ = 2, HKV = 1, D = 8, HK = 1, HV = 2, DK = 4, DV = 4, VOCAB = 32, CONTEXT = 64;
constexpr uint64_t C = 2 * HK * DK + HV * DV;

// A tensor of `shape` appended to `m`, F32 from a fixed pattern of its index and the tensors before it, times `scale` plus `offset`.
void add_tensor(gguf::GGUFModel& m, const std::string& name, std::vector<uint64_t> shape, float scale = 1.0f, float offset = 0.0f) {
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
gguf::GGUFModel tiny(const std::function<void(gguf::GGUFModel&)>& edit = {}) {
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
             {"context_length", CONTEXT}, {"ssm.conv_kernel", 4}, {"ssm.state_size", DK}, {"ssm.group_count", HK},
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

gguf::MetaValue* meta(gguf::GGUFModel& m, const std::string& key) {
    for (auto& kv : m.kv)
        if (kv.first == "qwen35." + key) return &kv.second;
    gguf::MetaValue v;
    v.vtype = gguf::V_UINT32;
    m.kv.push_back({"qwen35." + key, v});
    return &m.kv.back().second;
}

void set(gguf::GGUFModel& m, const std::string& key, uint64_t value) {
    gguf::MetaValue* v = meta(m, key);
    v->vtype = gguf::V_UINT32;
    v->u = value;
}

void drop(gguf::GGUFModel& m, const std::string& key) {
    for (size_t i = 0; i < m.kv.size(); ++i)
        if (m.kv[i].first == "qwen35." + key) m.kv.erase(m.kv.begin() + (long)i);
}

void booleans(gguf::GGUFModel& m, const std::string& key, const std::vector<bool>& values) {
    gguf::MetaValue* v = meta(m, key);
    v->vtype = gguf::V_ARRAY;
    v->u = gguf::V_BOOL;
    v->arr.clear();
    for (bool b : values) {
        gguf::MetaValue e;
        e.vtype = gguf::V_BOOL;
        e.b = b;
        v->arr.push_back(e);
    }
}

// Tensor `name` renamed to `to`, which leaves a layer without it or holding another kind's.
void rename(gguf::GGUFModel& m, const std::string& name, const std::string& to) {
    for (auto& t : m.tensors)
        if (t.name == name) t.name = to;
}

// The tiny model as a qwen35moe file: every key under the qwen35moe prefix, no dense width, and each layer's dense block replaced by X = 4 experts of FE, K = 2 a token, and a shared expert of FS with its gate.
constexpr uint64_t X = 4, K = 2, FE = 5, FS = 7;
gguf::GGUFModel routed(gguf::GGUFModel m) {
    for (auto& kv : m.kv) {
        if (kv.first.rfind("qwen35.", 0) == 0) kv.first = "qwen35moe." + kv.first.substr(7);
        if (kv.first == "general.architecture") kv.second.s = "qwen35moe";
    }
    for (size_t i = 0; i < m.kv.size(); ++i)
        if (m.kv[i].first == "qwen35moe.feed_forward_length") m.kv.erase(m.kv.begin() + (long)i);
    for (const auto& kv : std::vector<std::pair<std::string, uint64_t>>{
             {"expert_count", X}, {"expert_used_count", K}, {"expert_feed_forward_length", FE}, {"expert_shared_feed_forward_length", FS}}) {
        gguf::MetaValue v;
        v.vtype = gguf::V_UINT32;
        v.u = kv.second;
        m.kv.push_back({"qwen35moe." + kv.first, v});
    }
    auto add = [&](const std::string& name, std::vector<uint64_t> shape, float scale) {
        size_t count = 1;
        for (uint64_t d : shape) count *= size_t(d);
        const size_t at = m.blob.size();
        m.blob.resize(at + count * sizeof(float));
        for (size_t i = 0; i < count; ++i) {
            const float v = scale * float(int((i * 13 + m.tensors.size() * 5) % 23) - 11) / 64.0f;
            std::memcpy(m.blob.data() + at + i * sizeof(float), &v, sizeof(v));
        }
        m.tensors.push_back({name, std::move(shape), quant::GGML_TYPE_F32, 0});
        m.offsets.push_back(at);
    };
    for (int l = 0; l < 4; ++l) {
        const std::string pre = "blk." + std::to_string(l) + ".";
        for (const char* dense : {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) rename(m, pre + dense, "unused." + pre + dense);
        add(pre + "ffn_gate_inp.weight", {E, X}, 4.0f);
        add(pre + "ffn_gate_exps.weight", {E, FE, X}, 1.0f);
        add(pre + "ffn_up_exps.weight", {E, FE, X}, 1.0f);
        add(pre + "ffn_down_exps.weight", {FE, E, X}, 1.0f);
        add(pre + "ffn_gate_inp_shexp.weight", {E}, 2.0f);
        add(pre + "ffn_gate_shexp.weight", {E, FS}, 1.0f);
        add(pre + "ffn_up_shexp.weight", {E, FS}, 1.0f);
        add(pre + "ffn_down_shexp.weight", {FS, E}, 1.0f);
    }
    return m;
}

bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float));
}

// Every refused key with its text.
void configuration() {
    auto read = [](const gguf::GGUFModel& m) { infer::gguf_weights(m); };
    refuses("two MTP blocks", "unsupported metadata qwen35.nextn_predict_layers", [&] { read(tiny([](gguf::GGUFModel& m) { set(m, "nextn_predict_layers", 2); })); });
    refuses("MTP blocks for every block", "unsupported metadata qwen35.nextn_predict_layers", [&] {
        read(tiny([](gguf::GGUFModel& m) { set(m, "block_count", 1); set(m, "nextn_predict_layers", 1); }));
    });
    refuses("a value width apart", "value width must equal key width", [&] { read(tiny([](gguf::GGUFModel& m) { set(m, "attention.value_length", 4); })); });
    refuses("an odd rotary width", "invalid rotary width", [&] {
        read(tiny([](gguf::GGUFModel& m) { set(m, "rope.dimension_count", 3); drop(m, "rope.dimension_sections"); }));
    });
    refuses("a rotary width past the head", "invalid rotary width", [&] {
        read(tiny([](gguf::GGUFModel& m) { set(m, "rope.dimension_count", 10); drop(m, "rope.dimension_sections"); }));
    });
    refuses("sections apart from the rotary width", "rope.dimension_sections does not cover the rotary width",
            [&] { read(tiny([](gguf::GGUFModel& m) { set(m, "rope.dimension_count", 6); })); });
    refuses("a conv of three taps", "unsupported metadata qwen35.ssm.conv_kernel", [&] { read(tiny([](gguf::GGUFModel& m) { set(m, "ssm.conv_kernel", 3); })); });
    refuses("V heads apart from the K heads", "V heads must be a multiple of its K heads", [&] {
        read(tiny([](gguf::GGUFModel& m) { set(m, "ssm.group_count", 2); set(m, "ssm.time_step_rank", 3); set(m, "ssm.inner_size", 12); }));
    });
    refuses("an inner size of part of a head", "ssm.inner_size is not a whole number of V heads", [&] { read(tiny([](gguf::GGUFModel& m) { set(m, "ssm.inner_size", 9); })); });
    refuses("KV heads apart from the query heads", "head count must be divisible by KV head count", [&] {
        read(tiny([](gguf::GGUFModel& m) { set(m, "attention.head_count_kv", 3); }));
    });
    refuses("no layer kinds", "missing metadata qwen35.full_attention_interval", [&] { read(tiny([](gguf::GGUFModel& m) { drop(m, "full_attention_interval"); })); });
    refuses("recurrent layers of another length", "unsupported metadata qwen35.attention.recurrent_layers", [&] {
        read(tiny([](gguf::GGUFModel& m) { booleans(m, "attention.recurrent_layers", {true, false, true}); }));
    });
    refuses("a recurrent MTP block", "unsupported metadata qwen35.attention.recurrent_layers", [&] {
        read(tiny([](gguf::GGUFModel& m) {
            set(m, "block_count", 5);
            set(m, "nextn_predict_layers", 1);
            booleans(m, "attention.recurrent_layers", {true, false, true, false, true});
        }));
    });
    refuses("scaled rope", "scaled RoPE is unsupported", [&] {
        read(tiny([](gguf::GGUFModel& m) {
            gguf::MetaValue* v = meta(m, "rope.scaling.factor");
            v->vtype = gguf::V_FLOAT32;
            const float two = 2.0f;
            std::memcpy(&v->fb, &two, sizeof two);
        }));
    });
}

// The plan of each layer kind, its caches, ops and slots, and every refused tensor with its text.
void plan() {
    const gguf::GGUFModel m = tiny();
    const infer::ModelWeights w = infer::gguf_weights(m);
    const infer::ModelPlan p = infer::plan_model(w);
    require(p.layers.size() == 4 && p.vocab == VOCAB && p.residual == E && p.context_length == CONTEXT, "the plan's shape");
    for (size_t l = 0; l < 4; ++l) {
        const bool full = l % 2;
        const infer::LayerPlan& layer = p.layers[l];
        require(layer.cache == (full ? infer::Cache::kv : infer::Cache::state) && !layer.routed, "layer " + std::to_string(l) + "'s cache");
        require(layer.roles.size() == (full ? 11u : 14u), "layer " + std::to_string(l) + "'s roles");
        require(layer.ops.size() == (full ? 2u : 3u) && layer.ops[0].part == infer::Part::mixer, "layer " + std::to_string(l) + "'s ops");
        for (const infer::Role& role : layer.roles)
            if (role.name.find("ssm_conv1d") != std::string::npos) require(role.kind == infer::RoleKind::table && role.in == 4 && role.out == C, "the conv's taps as a table");
    }
    require(p.state.k_heads == HK && p.state.v_heads == HV && p.state.k_dim == DK && p.state.v_dim == DV, "the state's shape");
    require(p.kv_heads == HKV && p.head_dim == D, "the KV geometry");
    require(p.tables.size() == 2 && p.tables[0] == CONTEXT * 2, "rope tables of rope_dim / 2 a position");
    require(p.slots.size() == 10 && p.slots[0] == E && p.slots[2] == 2 * HQ * D && p.slots[3] == C && p.slots[5] == HQ * D, "the arena's slots");
    // Without the interval the layers' kinds come from the recurrent layers' array, which names each block, the MTP block's false, and leaves that block unplanned.
    const gguf::GGUFModel mtp = tiny([](gguf::GGUFModel& f) {
        set(f, "block_count", 5);
        set(f, "nextn_predict_layers", 1);
        drop(f, "full_attention_interval");
        booleans(f, "attention.recurrent_layers", {true, false, true, false, false});
    });
    const infer::ModelPlan q = infer::plan_model(infer::gguf_weights(mtp));
    require(q.layers.size() == 4 && q.layers[0].cache == infer::Cache::state && q.layers[1].cache == infer::Cache::kv &&
                q.layers[2].cache == infer::Cache::state && q.layers[3].cache == infer::Cache::kv,
            "the recurrent layers' array");

    auto planned = [](const gguf::GGUFModel& f) { infer::plan_model(infer::gguf_weights(f)); };
    refuses("a linear layer's tensor in a full layer", "a full-attention layer holds the linear-attention tensor blk.1.attn_qkv.weight",
            [&] { planned(tiny([](gguf::GGUFModel& f) { rename(f, "blk.0.attn_qkv.weight", "blk.1.attn_qkv.weight"); })); });
    refuses("a full layer's tensor in a linear layer", "a linear-attention layer holds the full-attention tensor blk.0.attn_q.weight",
            [&] { planned(tiny([](gguf::GGUFModel& f) { rename(f, "blk.1.attn_q.weight", "blk.0.attn_q.weight"); })); });
    refuses("a router", "expert tensors in a dense architecture blk.2.",
            [&] { planned(tiny([](gguf::GGUFModel& f) { rename(f, "blk.0.ffn_gate.weight", "blk.2.ffn_gate_inp.weight"); })); });
    auto built = [](const gguf::GGUFModel& f) { infer::Model model(infer::gguf_weights(f)); };
    refuses("no conv taps", "missing tensor blk.2.ssm_conv1d.weight", [&] { built(tiny([](gguf::GGUFModel& f) { rename(f, "blk.2.ssm_conv1d.weight", "x"); })); });
    refuses("conv taps with HF's middle axis", "incompatible tensor layout blk.0.ssm_conv1d.weight", [&] {
        built(tiny([](gguf::GGUFModel& f) {
            for (auto& t : f.tensors)
                if (t.name == "blk.0.ssm_conv1d.weight") t.ne = {4, 1, C};
        }));
    });
    refuses("a decay of the wrong width", "incompatible tensor layout blk.2.ssm_a", [&] {
        built(tiny([](gguf::GGUFModel& f) {
            for (auto& t : f.tensors)
                if (t.name == "blk.2.ssm_a") t.ne = {1};
        }));
    });
}

// Each layer's cache in the footprint: the KV layers' by the budget, the state layers' by the slots, and the conv's taps no product.
void footprint() {
    const gguf::GGUFModel m = tiny();
    const infer::ModelWeights w = infer::gguf_weights(m);
    const infer::ModelPlan p = infer::plan_model(w);
    infer::ModelOptions options;
    options.state_slots = 3;
    options.kv_k = options.kv_v = backend::KVType::f32;
    const infer::Footprint fp = infer::footprint(w, p, options);
    const size_t kv = CONTEXT * HKV * D * 2 * sizeof(float), state = 3 * (HV * DK * DV + 3 * C) * sizeof(float);
    require(fp.cache.size() == 4 && fp.cache[0] == state && fp.cache[1] == kv && fp.cache[2] == state && fp.cache[3] == kv, "the footprint's caches");
    bool conv_product = false;
    for (const infer::Matrix& t : fp.layers[0])
        if (t.nin == 4 && t.rows == C) conv_product = t.product;
    require(!conv_product, "the conv's taps counted as a product");
}

// A CPU backend without the linear attention's ops, as a device backend is until it has them, and one whose head fails once when armed.
struct Lacking : backend::CpuBackend {
    bool implements(backend::Op op) const override { return op != backend::Op::causal_conv_silu; }
};
struct FailingHead : backend::CpuBackend {
    bool fail = false;
    void matmul_logits(uint32_t type, backend::CSlice data, backend::CSlice x, backend::Slice y, size_t nin, size_t nout, size_t nbatch,
                       backend::RowRuns runs = {}, backend::Dtype dtype = backend::Dtype::f16) override {
        if (fail) {
            fail = false;
            throw std::runtime_error("injected");
        }
        backend::CpuBackend::matmul_logits(type, data, x, y, nin, nout, nbatch, runs, dtype);
    }
};

// The runtime's rules for a model that keeps a state: the op check, no fork without a checkpoint, the slots, and a failed pass going back to the start, where the state reads as zero, when it has no checkpoint.
void state_rules() {
    const gguf::GGUFModel m = tiny();
    const infer::ModelWeights w = infer::gguf_weights(m);
    refuses("a backend without the conv", "inference: layer 0's mixer needs causal_conv_silu, which the backend of its device does not implement",
            [&] { infer::Model model(w, std::make_shared<Lacking>()); });
    infer::Model model(w);
    require(model.keeps_state(), "a qwen35 model keeps a state");
    {
        const gguf::GGUFModel dense = tiny_qwen(1, 16, true);
        infer::Model other(infer::gguf_weights(dense));
        require(!other.keeps_state(), "a Qwen3 model keeps no state");
    }
    const std::vector<uint32_t> ids = {3, 1, 4, 1, 5, 9, 2, 6};
    const std::vector<float> want = model.prefill(ids);
    infer::Sequence a = model.make_sequence();
    infer::ExecContext ctx;
    const infer::BatchEntry first{&a, ids.data(), 4, false};
    // One slot, held by the model's own sequence since its prompt, so a second sequence finds none until that one is reset.
    refuses("a second slot", "every recurrent state slot is held", [&] { model.forward(ctx, &first, 1); });
    model.reset();
    model.forward(ctx, &first, 1);
    refuses("a fork", "a fork of a model whose layers keep a recurrent state takes its source's checkpoint", [&] { model.fork(a, 0); });
    const infer::BatchEntry rest{&a, ids.data() + 4, 4, true};
    model.forward(ctx, &rest, 1);
    require(!std::memcmp(ctx.logits(0), want.data(), VOCAB * sizeof(float)), "a sequence in two passes differs from the prompt in one");
    model.reset(a);

    // A failed pass on a history whose state it may have written, with no checkpoint, goes back to the start; a failure from length 0 leaves a zero state behind, which is no loss.
    auto failing = std::make_shared<FailingHead>();
    infer::Model broken(w, failing);
    failing->fail = true;
    refuses("the first pass failing", "injected", [&] { broken.prefill(ids); });
    require(broken.n_tokens() == 0, "a failed first prompt left a history");
    require(same(broken.prefill(ids), want), "a sequence after a failed first pass differs");
    failing->fail = true;
    refuses("a step failing", "injected", [&] { broken.step(7); });
    require(broken.n_tokens() == 0, "a failed step without a checkpoint kept a history");
    require(same(broken.prefill(ids), want), "a sequence after a failed step differs from a fresh one");
}

// State checkpoints (docs/SPECULATIVE.md, section 1), on a model of a 512-token context whose prompts pass a CPU block: a keep changes no logits; a retract reaches the checkpoint and a fork reads it in place, each continuing with the bits of the history never stopped; keep() turns the live state into the checkpoint; a failed pass goes back to the checkpoint; the slots are counted and given back.
void checkpoints() {
    const gguf::GGUFModel m = tiny([](gguf::GGUFModel& g) { set(g, "context_length", 512); });
    const infer::ModelWeights w = infer::gguf_weights(m);
    std::vector<uint32_t> ids(200);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (uint32_t)((i * 7 + 3) % VOCAB);
    const std::vector<uint32_t> head(ids.begin(), ids.begin() + 128), tail(ids.begin() + 128, ids.end());
    infer::ModelOptions options;
    options.state_slots = 2;
    options.checkpoint_slots = 1;
    infer::Model fresh(w, backend::make_cpu_backend(), options);
    const std::vector<float> want = fresh.prefill(ids);
    std::vector<float> steps;
    for (uint32_t t : {9u, 14u}) {
        const std::vector<float> next = fresh.step((int)t);
        steps.insert(steps.end(), next.begin(), next.end());
    }

    infer::Model model(w, backend::make_cpu_backend(), options);
    require(model.checkpoint_slots() == 1 && model.checkpoints_free() == 1, "the checkpoint slots");
    require(same(model.prefill(ids, 128), want), "a prompt keeping its state at 128 differs");
    {
        infer::Sequence z = model.make_sequence();
        require(!model.checkpoint(z), "a fresh sequence holds a checkpoint");
    }
    require(model.checkpoints_free() == 0, "a kept state took no slot");
    require(model.retract(150) == 128 && model.n_tokens() == 128, "a retract inside the prompt did not reach its checkpoint");
    require(same(model.prefill(tail), want), "the prompt continued from its checkpoint differs");
    std::vector<float> got;
    for (uint32_t t : {9u, 14u}) {
        const std::vector<float> next = model.step((int)t);
        got.insert(got.end(), next.begin(), next.end());
    }
    require(same(got, steps), "decode after a checkpoint's continuation differs");
    model.reset();
    require(model.checkpoints_free() == 1, "a reset kept its checkpoint's slot");

    // Explicit sequences: a keep entry, a fork at its checkpoint read in place beside its source's continuation, and a fork where there is none refused.
    infer::Sequence a = model.make_sequence();
    infer::ExecContext ctx;
    infer::BatchEntry keep{&a, head.data(), head.size(), false};
    keep.keep = true;
    keep.extent = ids.size();
    model.forward(ctx, &keep, 1);
    require(model.checkpoint(a) == std::optional<size_t>(128), "a keep entry left no checkpoint at its end");
    refuses("a fork past the checkpoint", "takes its source's checkpoint", [&] { model.fork(a, 0); });
    infer::Sequence f = model.fork(a, 128);
    infer::BatchEntry both[] = {{&a, tail.data(), tail.size(), true}, {&f, tail.data(), tail.size(), true}};
    both[0].extent = both[1].extent = ids.size();
    model.forward(ctx, both, 2);
    require(!std::memcmp(ctx.logits(0), want.data(), VOCAB * sizeof(float)) && !std::memcmp(ctx.logits(1), want.data(), VOCAB * sizeof(float)),
            "a source and its fork continued from a checkpoint differ from the prompt in one");
    model.reset(f);
    require(model.checkpoints_free() == 0, "a fork's end gave its source's checkpoint slot back");

    // keep() between passes: the live state at 200 becomes the checkpoint, replacing the one at 128 in the one slot, and the history goes on from it.
    require(model.keep(a) && model.checkpoint(a) == std::optional<size_t>(200), "the live state was not kept");
    const uint32_t t9 = 9, t14 = 14;
    infer::BatchEntry s1{&a, &t9, 1, true}, s2{&a, &t14, 1, true};
    model.forward(ctx, &s1, 1);
    std::vector<float> kept(ctx.logits(0), ctx.logits(0) + VOCAB);
    model.forward(ctx, &s2, 1);
    kept.insert(kept.end(), ctx.logits(0), ctx.logits(0) + VOCAB);
    require(same(kept, steps), "decode after keeping the live state differs");
    require(model.retract(a, 201) == 200 && a.length() == 200, "a retract past the kept state did not reach it");
    model.reset(a);

    // A failed pass goes back to the checkpoint, and the history continues from there with the bits of one never failed.
    auto failing = std::make_shared<FailingHead>();
    infer::Model broken(w, failing, options);
    require(same(broken.prefill(ids, 128), want), "a prompt keeping its state differs on the failing backend");
    failing->fail = true;
    refuses("a step failing", "injected", [&] { broken.step(9); });
    require(broken.n_tokens() == 128, "a failed step did not go back to the checkpoint");
    require(same(broken.prefill(tail), want), "the history after a failed step differs");
}

// A hybrid history copied to host memory and back (Model::save_host, Model::restore_host): its 128 tokens and its checkpoint's state there, restored once the one checkpoint slot is free into a slot of its own, continue the prompt with the bits of the prompt read in one; a copy where no checkpoint is refused, and a restore with no checkpoint slot free throws and holds nothing.
void host_round_trip() {
    const gguf::GGUFModel m = tiny([](gguf::GGUFModel& g) { set(g, "context_length", 512); });
    const infer::ModelWeights w = infer::gguf_weights(m);
    std::vector<uint32_t> ids(200);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (uint32_t)((i * 7 + 3) % VOCAB);
    const std::vector<uint32_t> head(ids.begin(), ids.begin() + 128), tail(ids.begin() + 128, ids.end());
    infer::ModelOptions options;
    options.state_slots = 2;
    options.checkpoint_slots = 1;
    infer::Model fresh(w, backend::make_cpu_backend(), options);
    const std::vector<float> want = fresh.prefill(ids);
    infer::Model model(w, backend::make_cpu_backend(), options);
    infer::Sequence a = model.make_sequence();
    infer::ExecContext ctx;
    infer::BatchEntry keep{&a, head.data(), head.size(), false};
    keep.extent = ids.size();
    model.forward(ctx, &keep, 1);
    infer::HostHistory h;
    const size_t any = std::numeric_limits<size_t>::max();
    refuses("a copy without a checkpoint", "takes its checkpoint", [&] { model.save_host(a, 128, h, any); });
    model.reset(a);
    keep.keep = true;
    model.forward(ctx, &keep, 1);
    model.save_host(a, 128, h, any);
    require(h.length == 128 && h.held == model.host_bytes(128) && h.bytes > 0 && h.slabs.size() == 1 && !h.slabs[0].empty(), "a hybrid copy to host memory holds other bytes");
    refuses("a restore with every checkpoint slot held", "every checkpoint slot is held", [&] { model.restore_host(h); });
    require(model.checkpoints_free() == 0, "a refused restore took a slot");
    model.reset(a);
    infer::Sequence r = model.restore_host(h);
    require(model.checkpoint(r) == std::optional<size_t>(128) && r.length() == 128, "a restored hybrid history's checkpoint");
    infer::BatchEntry rest{&r, tail.data(), tail.size(), true};
    rest.extent = ids.size();
    model.forward(ctx, &rest, 1);
    require(!std::memcmp(ctx.logits(0), want.data(), VOCAB * sizeof(float)), "a hybrid history restored from host memory differs from the prompt in one");
    model.release_host(h);
    model.reset(r);
    require(model.checkpoints_free() == 1, "a restored history's reset kept its checkpoint slot");
}

// A state alone copied to host memory (Model::save_host without blocks) at a history's checkpoint at 128, which the history then replaces with one at 200: a fork of the history at 128 with that state (Model::fork with a state) continues the prompt with the bits of the prompt read in one, beside the history's own continuation in one pass.
// The state takes only its slot's bytes; a fork with a state at another length, with a whole history's copy or with every checkpoint slot held, and a restore of a state alone, are refused and hold nothing.
void host_state_fork() {
    const gguf::GGUFModel m = tiny([](gguf::GGUFModel& g) { set(g, "context_length", 512); });
    const infer::ModelWeights w = infer::gguf_weights(m);
    std::vector<uint32_t> ids(200);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (uint32_t)((i * 7 + 3) % VOCAB);
    const std::vector<uint32_t> head(ids.begin(), ids.begin() + 128), tail(ids.begin() + 128, ids.end());
    std::vector<uint32_t> more(40);
    for (size_t i = 0; i < more.size(); ++i) more[i] = (uint32_t)((i * 5 + 1) % VOCAB);
    infer::ModelOptions options;
    options.state_slots = 2;
    options.checkpoint_slots = 2;
    infer::Model fresh(w, backend::make_cpu_backend(), options);
    const std::vector<float> want = fresh.prefill(ids);
    std::vector<uint32_t> longer = ids;
    longer.insert(longer.end(), more.begin(), more.end());
    infer::Model model(w, backend::make_cpu_backend(), options);
    infer::ExecContext ctx;
    std::vector<float> want_longer;
    {
        infer::Sequence b = model.make_sequence();
        infer::BatchEntry first{&b, ids.data(), ids.size(), false}, second{&b, more.data(), more.size(), true};
        first.extent = ids.size();
        second.extent = longer.size();
        model.forward(ctx, &first, 1);
        model.forward(ctx, &second, 1);
        want_longer.assign(ctx.logits(0), ctx.logits(0) + VOCAB);
        model.reset(b);
    }
    infer::Sequence a = model.make_sequence();
    infer::BatchEntry keep{&a, head.data(), head.size(), false};
    keep.keep = true;
    keep.extent = ids.size();
    model.forward(ctx, &keep, 1);
    infer::HostHistory st, whole;
    const size_t any = std::numeric_limits<size_t>::max();
    model.save_host(a, 128, st, any, false);
    model.save_host(a, 128, whole, any);
    require(!st.blocks && st.length == 128 && st.held == model.host_bytes(128, false) && st.held <= whole.held && st.bytes < whole.bytes,
            "a state alone in host memory holds other bytes");
    infer::BatchEntry rest{&a, tail.data(), tail.size(), false};
    rest.extent = ids.size();
    model.forward(ctx, &rest, 1);
    require(model.keep(a) && model.checkpoint(a) == std::optional<size_t>(200), "the history's state at 200 was not kept");
    refuses("a fork with a state at another length", "a state alone at its length", [&] { model.fork(a, 0, st); });
    refuses("a fork with a whole history's copy", "a state alone at its length", [&] { model.fork(a, 128, whole); });
    refuses("a restore of a state alone", "another layout", [&] { model.restore_host(st); });
    infer::Sequence other = model.make_sequence();
    infer::BatchEntry other_keep{&other, head.data(), head.size(), false};
    other_keep.keep = true;
    other_keep.extent = ids.size();
    model.forward(ctx, &other_keep, 1);
    require(model.checkpoints_free() == 0, "two checkpoints left a slot free");
    refuses("a fork with a state with every checkpoint slot held", "every checkpoint slot is held", [&] { model.fork(a, 128, st); });
    model.reset(other);
    infer::Sequence f = model.fork(a, 128, st);
    require(model.checkpoint(f) == std::optional<size_t>(128) && f.length() == 128, "a fork with a state's checkpoint");
    infer::BatchEntry both[] = {{&f, tail.data(), tail.size(), true}, {&a, more.data(), more.size(), true}};
    both[0].extent = ids.size();
    both[1].extent = longer.size();
    model.forward(ctx, both, 2);
    require(!std::memcmp(ctx.logits(0), want.data(), VOCAB * sizeof(float)), "a fork with a state from host memory differs from the prompt in one");
    require(!std::memcmp(ctx.logits(1), want_longer.data(), VOCAB * sizeof(float)), "the history beside its fork with a state differs from its prompt in one");
    model.release_host(st);
    model.release_host(whole);
    model.reset(f);
    model.reset(a);
    require(model.checkpoints_free() == 2, "a fork with a state kept its checkpoint slot after its reset");
}

// A device reporting `room` bytes free that keeps copies of what it adopts, so the fit charges it the weights its layers take; its first `rise_after` reads report one byte, as a card still taking back an ended process's memory does.
// With `device` it says it is not the CPU, as a card does.
struct Room : backend::CpuBackend {
    size_t room = 0;
    mutable int rise_after = 0;
    bool device = false;
    std::optional<size_t> memory_available() const override { return rise_after-- > 0 ? 1 : room; }
    bool reads_in_place() const override { return false; }
    bool is_cpu() const override { return !device; }
};

// The automatic checkpoint count is tried through the fit itself (infer::fitted_kv): on a split whose first stage holds only a linear layer's states, on a device with room for its live states and not one checkpoint more, the fit gives no checkpoint slot and the model is placed; on roomy devices it gives the slots asked for. Slot counts whose sum a size cannot hold are refused.
void checkpoint_fit() {
    const gguf::GGUFModel m = tiny([](gguf::GGUFModel& g) { set(g, "context_length", 512); });
    const infer::ModelWeights w = infer::gguf_weights(m);
    const infer::ModelPlan plan = infer::plan_model(w);
    const auto devices = [](size_t first) {
        auto a = std::make_shared<Room>(), b = std::make_shared<Room>();
        a->room = first;
        b->room = size_t(1) << 30;
        return std::vector<backend::BackendPtr>{a, b};
    };
    infer::PlacementRequest request;
    request.names = {"device 0", "device 1"};
    request.shares = {1, 3};
    request.fit_kv = request.fit_checkpoints = true;
    infer::ModelOptions options;
    options.state_slots = 2;
    options.checkpoint_slots = 4;
    const auto holds = [&](size_t room, size_t kept) {
        infer::ModelOptions o = options;
        o.checkpoint_slots = kept;
        try {
            infer::split_layers(infer::footprint(w, plan, o), infer::budgets_for(devices(room), request.names), infer::kDefaultUbatch, request.shares,
                                core::host_memory_available());
            return true;
        } catch (const std::runtime_error&) {
            return false;
        }
    };
    size_t lo = 1, hi = size_t(1) << 30;
    while (hi - lo > 1) {
        const size_t mid = lo + (hi - lo) / 2;
        (holds(mid, 0) ? hi : lo) = mid;
    }
    require(!holds(hi, 1), "the first device holds a checkpoint more at the least room it needs without one");
    // The KV tokens the checkpoints took (PlacedModel::checkpoint_kv_tokens) are none where they took none or the budget stays whole.
    const infer::PlacedModel none = infer::place_model(w, devices(hi), request, options);
    require(none.model->checkpoint_slots() == 0 && none.checkpoint_kv_tokens == 0, "a checkpoint count the first device cannot hold was taken");
    const infer::PlacedModel roomy = infer::place_model(w, devices(size_t(1) << 30), request, options);
    require(roomy.model->checkpoint_slots() == 4 && roomy.checkpoint_kv_tokens == 0, "roomy devices did not take the checkpoint slots asked for");
    // On one device with room for the whole budget and no more, the checkpoints take at most a quarter of it, in whole blocks of 128: of 512 tokens all four slots asked for, each far smaller than a quarter of the budget's bytes, with 384 tokens left, and of 384 none, since a block is a third; asked for as many as a size holds, the search ends with a count that fits.
    infer::PlacementRequest one = request;
    one.names = {"device 0"};
    one.shares.clear();
    const auto alone = [&](size_t room) {
        auto d = std::make_shared<Room>();
        d->room = room;
        return std::vector<backend::BackendPtr>{d};
    };
    for (const size_t context : {size_t(512), size_t(384)}) {
        const gguf::GGUFModel mc = tiny([&](gguf::GGUFModel& g) { set(g, "context_length", context); });
        const infer::ModelWeights wc = infer::gguf_weights(mc);
        const infer::ModelPlan pc = infer::plan_model(wc);
        const auto whole = [&](size_t room) {
            infer::ModelOptions o = options;
            o.checkpoint_slots = 0;
            try {
                infer::split_layers(infer::footprint(wc, pc, o), infer::budgets_for(alone(room), one.names), infer::kDefaultUbatch, {},
                                    core::host_memory_available());
                return true;
            } catch (const std::runtime_error&) {
                return false;
            }
        };
        lo = 1, hi = size_t(1) << 30;
        while (hi - lo > 1) {
            const size_t mid = lo + (hi - lo) / 2;
            (whole(mid) ? hi : lo) = mid;
        }
        const infer::PlacedModel placed = infer::place_model(wc, alone(hi), one, options);
        const auto& tight = placed.model;
        const size_t want_slots = context == 512 ? 4 : 0, want_tokens = 384;
        require(tight->checkpoint_slots() == want_slots && tight->kv_tokens_total() == want_tokens && placed.checkpoint_kv_tokens == context - want_tokens,
                "a device holding the whole " + std::to_string(context) + "-token budget and no more took " + std::to_string(tight->checkpoint_slots()) +
                    " checkpoint slots beside " + std::to_string(tight->kv_tokens_total()) + " KV tokens");
        if (context == 512) {
            infer::ModelOptions most = options;
            most.checkpoint_slots = std::numeric_limits<size_t>::max();
            const infer::PlacedModel placed_all = infer::place_model(wc, alone(hi), one, most);
            const auto& all = placed_all.model;
            require(all->checkpoint_slots() >= 4 && all->checkpoint_slots() < 1000 && all->kv_tokens_total() == 384 && placed_all.checkpoint_kv_tokens == 128,
                    "asked for as many checkpoints as a size holds, the fit took " + std::to_string(all->checkpoint_slots()) + " beside " +
                        std::to_string(all->kv_tokens_total()) + " KV tokens");
        }
    }
    // Marks past the first take only the room the budget leaves (PlacementRequest::fit_marks): of six asked for, one on a device that holds the whole budget with one and no more, three where it holds three, all six on a roomy one, the 512-token budget whole each time; one mark more than the device holds is not taken.
    {
        infer::PlacementRequest marked = one;
        marked.fit_checkpoints = false;
        marked.fit_marks = true;
        infer::ModelOptions mo = options;
        mo.checkpoint_slots = 0;
        mo.mark_slots = 6;
        mo.mark_rows = 4;
        const auto least = [&](size_t marks) {
            infer::ModelOptions o = mo;
            o.mark_slots = marks;
            const auto holds = [&](size_t room) {
                try {
                    infer::split_layers(infer::footprint(w, plan, o), infer::budgets_for(alone(room), one.names), infer::kDefaultUbatch, {},
                                        core::host_memory_available());
                    return true;
                } catch (const std::runtime_error&) {
                    return false;
                }
            };
            size_t l = 1, h = size_t(1) << 30;
            while (h - l > 1) {
                const size_t mid = l + (h - l) / 2;
                (holds(mid) ? h : l) = mid;
            }
            return h;
        };
        for (const auto& [room, want] : {std::pair<size_t, size_t>{least(1), 1}, {least(3), 3}, {least(4) - 1, 3}, {size_t(1) << 30, 6}}) {
            const infer::PlacedModel placed = infer::place_model(w, alone(room), marked, mo);
            require(placed.model->mark_slots() == want && placed.model->kv_tokens_total() == 512,
                    "a device of " + std::to_string(room) + " bytes took " + std::to_string(placed.model->mark_slots()) + " mark slots beside " +
                        std::to_string(placed.model->kv_tokens_total()) + " KV tokens, where " + std::to_string(want) + " fit beside the whole budget");
        }
    }
    // A device whose free memory is still coming back when the fit first reads it, as a server restarted on the card its predecessor held finds it: once the memory has settled it holds the whole budget and the checkpoints asked for, so it gets both.
    {
        auto d = std::make_shared<Room>();
        d->room = size_t(1) << 30;
        d->rise_after = 2;
        const infer::PlacedModel placed_settled = infer::place_model(w, std::vector<backend::BackendPtr>{d}, one, options);
        const auto& settled = placed_settled.model;
        require(settled->checkpoint_slots() == 4 && settled->kv_tokens_total() == 512 && placed_settled.checkpoint_kv_tokens == 0,
                "a device whose memory settled after the first reads took " + std::to_string(settled->checkpoint_slots()) + " checkpoint slots beside " +
                    std::to_string(settled->kv_tokens_total()) + " KV tokens");
    }
    // Two cards fitted to their free memory, the first's coming back in a step some two seconds after the fit first reads it, as when a split server restarts on the cards its predecessor held: the split, the KV budget and the checkpoints are those of idle cards, not the second card holding every layer.
    {
        const auto cards = [](int rise) {
            auto a = std::make_shared<Room>(), b = std::make_shared<Room>();
            a->room = b->room = size_t(1) << 30;
            a->device = b->device = true;
            a->rise_after = rise;
            return std::vector<backend::BackendPtr>{a, b};
        };
        infer::PlacementRequest fitted = request;
        fitted.shares.clear();
        const infer::PlacedModel idle = infer::place_model(w, cards(0), fitted, options);
        const infer::PlacedModel late = infer::place_model(w, cards(8), fitted, options);
        require(late.plan == idle.plan && late.model->checkpoint_slots() == idle.model->checkpoint_slots() &&
                    late.model->kv_tokens_total() == idle.model->kv_tokens_total(),
                "a split whose first card's memory came back late was placed as\n" + late.plan + "where idle cards give\n" + idle.plan);
    }
    infer::ModelOptions wrapped;
    wrapped.state_slots = std::numeric_limits<size_t>::max();
    wrapped.checkpoint_slots = 2;
    refuses("state slots past a size", "size overflows", [&] { infer::Model model(w, backend::make_cpu_backend(), wrapped); });
}

// Runs a pass through every stage of a context reserved for passes and returns its first logits row, or nothing when it wants none.
std::vector<float> run_pass(infer::Model& model, infer::ExecContext& ctx, const infer::BatchEntry* entries, size_t n) {
    model.begin_pass(ctx, 0, entries, n, 0);
    for (size_t s = 0; s < model.stage_count(); ++s) model.run_pass_stage(ctx, 0, s);
    std::vector<float> out;
    if (entries[0].want_logits) out.assign(model.pass_logits(ctx, 0, 0), model.pass_logits(ctx, 0, 0) + VOCAB);
    model.end_pass(ctx, 0);
    return out;
}

// A refused pass takes no state slot: a fresh sequence in a pass refused for its rows, its logits rows, a sequence listed twice or too few slots for all its fresh sequences leaves every slot it did not hold free for the next pass.
// A sequence with a history keeps its slot and its state through a refused pass and continues with the bytes of one never refused.
void refused_passes_take_no_slot() {
    const gguf::GGUFModel m = tiny();
    const infer::ModelWeights w = infer::gguf_weights(m);
    const uint32_t ids[] = {3, 1, 4, 1, 5};
    infer::ModelOptions options;
    options.kv_tokens = 3 * 128;
    {
        options.state_slots = 1;
        infer::Model model(w, backend::make_cpu_backend(), options);
        infer::Sequence a = model.make_sequence(), b = model.make_sequence();
        infer::ExecContext ctx;
        model.reserve_passes(ctx, 1, 2, 1);
        const infer::BatchEntry retry{&b, ids, 1, true};
        const infer::BatchEntry rows{&a, ids, 3, true};
        refuses("a pass of more rows than reserved", "a pass beyond the rows or logits rows reserve_passes reserved", [&] { model.begin_pass(ctx, 0, &rows, 1, 0); });
        run_pass(model, ctx, &retry, 1);
        model.reset(b);
        const infer::BatchEntry logits{&a, ids, 2, true, true};
        refuses("a pass of more logits rows than reserved", "a pass beyond the rows or logits rows reserve_passes reserved", [&] { model.begin_pass(ctx, 0, &logits, 1, 0); });
        run_pass(model, ctx, &retry, 1);
        model.reset(b);
        const infer::BatchEntry twice[] = {{&a, ids, 1, false}, {&a, ids + 1, 1, true}};
        refuses("a sequence listed twice in a pass", "a sequence listed twice in a pass", [&] { model.begin_pass(ctx, 0, twice, 2, 0); });
        run_pass(model, ctx, &retry, 1);
        model.reset(b);
        infer::ExecContext whole;
        refuses("a sequence listed twice in a forward", "a sequence listed twice in a pass", [&] { model.forward(whole, twice, 2); });
        model.forward(whole, &retry, 1);
        model.reset(b);
    }
    options.state_slots = 2;
    infer::Model model(w, backend::make_cpu_backend(), options);
    infer::Model reference(w, backend::make_cpu_backend(), options);
    reference.prefill({ids, ids + 2});
    const std::vector<float> want = reference.step((int)ids[2]);
    infer::Sequence h = model.make_sequence(), a = model.make_sequence(), b = model.make_sequence();
    infer::ExecContext ctx;
    model.reserve_passes(ctx, 1, 3, 3);
    const infer::BatchEntry history{&h, ids, 2, false};
    run_pass(model, ctx, &history, 1);
    // One slot is left for two fresh sequences, so the pass is refused whichever of them comes first, and the history keeps its slot through it.
    const infer::BatchEntry short_of_slots[] = {{&h, ids + 2, 1, true}, {&a, ids, 1, true}, {&b, ids, 1, true}};
    refuses("two fresh sequences and one slot", "every recurrent state slot is held", [&] { model.begin_pass(ctx, 0, short_of_slots, 3, 0); });
    const infer::BatchEntry too_long[] = {{&h, ids + 2, 1, true}, {&a, ids, 3, true}};
    refuses("a history beside a pass of more rows than reserved", "a pass beyond the rows or logits rows reserve_passes reserved", [&] { model.begin_pass(ctx, 0, too_long, 2, 0); });
    require(h.length() == 2 && a.length() == 0 && b.length() == 0, "a refused pass changed a history");
    const infer::BatchEntry fresh{&b, ids, 2, true};
    run_pass(model, ctx, &fresh, 1);
    const infer::BatchEntry next{&h, ids + 2, 1, true};
    require(same(run_pass(model, ctx, &next, 1), want), "a history differs after the passes refused beside it");
    const infer::BatchEntry none_left{&a, ids, 1, true};
    refuses("a third sequence on two slots", "every recurrent state slot is held", [&] { model.begin_pass(ctx, 0, &none_left, 1, 0); });
    model.reset(b);
    run_pass(model, ctx, &none_left, 1);
    model.reset(a);
    model.reset(h);
}

// A pass accepted by every check whose planning then fails to allocate takes no slot either: each allocation of begin_pass fails in turn, on a model of one slot with a fresh reservation, and a valid pass of another fresh sequence must follow.
void failed_admission_takes_no_slot() {
    const gguf::GGUFModel m = tiny();
    const infer::ModelWeights w = infer::gguf_weights(m);
    const uint32_t ids[] = {3, 1, 4};
    infer::ModelOptions options;
    options.kv_tokens = 2 * 128;
    options.state_slots = 1;
    size_t failed = 0;
    for (size_t k = 1;; ++k) {
        infer::Model model(w, backend::make_cpu_backend(), options);
        infer::Sequence a = model.make_sequence(), b = model.make_sequence();
        infer::ExecContext ctx;
        model.reserve_passes(ctx, 1, 3, 1);
        const infer::BatchEntry entry{&a, ids, 3, true};
        bool threw = false;
        fail_allocation = k;
        try {
            model.begin_pass(ctx, 0, &entry, 1, 0);
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        fail_allocation = 0;
        if (!threw) {
            model.abort_pass(ctx, 0);
            break;
        }
        ++failed;
        require(a.length() == 0, "a pass whose planning failed changed a history");
        const infer::BatchEntry retry{&b, ids, 1, true};
        run_pass(model, ctx, &retry, 1);
        model.reset(b);
    }
    require(failed > 0, "begin_pass allocated nothing on a fresh reservation");
}

// A mark's holds are returned once, whatever becomes of its sequence: on a model of one live slot and one mark, a marked sequence destroyed, at length 0 and holding the live slot, leaves both for the next; a marked sequence moved, by construction and by assignment, takes its mark with it, so a reset of the one moved from releases nothing; and a mark whose allocations fail in turn changes nothing, the sequence then stepping and marking as before.
void marks() {
    // The weights are read in place, so the file outlives the model.
    const gguf::GGUFModel file = tiny();
    const infer::ModelWeights w = infer::gguf_weights(file);
    const uint32_t ids[] = {3, 1, 4, 1, 5};
    infer::ModelOptions options;
    options.state_slots = 1;
    options.mark_slots = 1;
    options.mark_rows = 4;
    infer::Model model(w, backend::make_cpu_backend(), options);
    infer::ExecContext ctx;
    const auto pass = [&](infer::Sequence& s, size_t n) {
        const infer::BatchEntry e{&s, ids, n, true};
        model.forward(ctx, &e, 1);
    };
    {
        infer::Sequence fresh = model.make_sequence();
        require(model.mark(fresh), "a fresh sequence's mark was refused");
    }
    {
        infer::Sequence live = model.make_sequence();
        pass(live, 3);
        require(model.mark(live), "the mark of a sequence holding the live slot was refused");
    }
    infer::Sequence a = model.make_sequence();
    pass(a, 3);
    require(model.mark(a), "a destroyed sequence kept its mark or the live slot");
    pass(a, 2);
    infer::Sequence b(std::move(a));
    model.reset(a);
    require(model.retract(b, 4) == 4, "a mark moved by construction did not reach its length");
    require(model.mark(b), "a mark was not returned by the retract after a move");
    infer::Sequence c = model.make_sequence();
    c = std::move(b);
    model.reset(b);
    pass(c, 1);
    require(model.retract(c, 4) == 4, "a mark moved by assignment did not reach the mark");
    // A marked sequence assigned over returns its mark, so another sequence takes it.
    require(model.mark(c), "a mark after a retract to the mark was refused");
    c = model.make_sequence();
    infer::Sequence after = model.make_sequence();
    require(model.mark(after), "a sequence assigned over kept its mark");
    model.reset(after);
    model.reset(c);
    // Each allocation of a mark failed in turn: nothing taken or changed, so the history steps and is marked afterwards.
    size_t failed = 0;
    for (size_t k = 1;; ++k) {
        infer::Sequence s = model.make_sequence();
        pass(s, 3);
        bool threw = false, marked = false;
        fail_allocation = k;
        try {
            marked = model.mark(s);
        } catch (const std::bad_alloc&) {
            threw = true;
        }
        fail_allocation = 0;
        if (!threw) {
            require(marked && model.retract(s, 3) == 3, "a mark after failed allocations was refused");
            break;
        }
        ++failed;
        require(s.length() == 3, "a mark whose allocation failed changed the history");
        pass(s, 1);
        require(model.mark(s), "a mark whose allocation failed kept a hold");
        model.reset(s);
    }
    require(failed > 0, "a mark allocated nothing");
}

// The tiny model with an MTP block after its four layers, blk.4: a full-attention layer with its feed-forward block, and the block's input norms, eh_proj and final norm; a context of 512, so a prompt passes a CPU block.
gguf::GGUFModel tiny_mtp(const std::function<void(gguf::GGUFModel&)>& edit = {}) {
    return tiny([&](gguf::GGUFModel& m) {
        set(m, "block_count", 5);
        set(m, "nextn_predict_layers", 1);
        set(m, "context_length", 512);
        const std::string pre = "blk.4.";
        add_tensor(m, pre + "attn_norm.weight", {E}, 0.1f, 1.0f);
        add_tensor(m, pre + "post_attention_norm.weight", {E}, 0.1f, 1.0f);
        add_tensor(m, pre + "attn_q.weight", {E, 2 * HQ * D});
        add_tensor(m, pre + "attn_k.weight", {E, HKV * D});
        add_tensor(m, pre + "attn_v.weight", {E, HKV * D});
        add_tensor(m, pre + "attn_q_norm.weight", {D}, 0.1f, 1.0f);
        add_tensor(m, pre + "attn_k_norm.weight", {D}, 0.1f, 1.0f);
        add_tensor(m, pre + "attn_output.weight", {HQ * D, E});
        add_tensor(m, pre + "ffn_gate.weight", {E, F});
        add_tensor(m, pre + "ffn_up.weight", {E, F});
        add_tensor(m, pre + "ffn_down.weight", {F, E});
        add_tensor(m, pre + "nextn.eh_proj.weight", {2 * E, E});
        add_tensor(m, pre + "nextn.enorm.weight", {E}, 0.1f, 1.0f);
        add_tensor(m, pre + "nextn.hnorm.weight", {E}, 0.1f, 1.0f);
        add_tensor(m, pre + "nextn.shared_head_norm.weight", {E}, 0.1f, 1.0f);
        if (edit) edit(m);
    });
}

// A draft's ids and every draft row's logits.
struct Drafted {
    std::vector<uint32_t> ids;
    std::vector<float> logits;
    bool operator==(const Drafted& o) const {
        return ids == o.ids && logits.size() == o.logits.size() && (logits.empty() || !std::memcmp(logits.data(), o.logits.data(), logits.size() * sizeof(float)));
    }
};
Drafted drafted(infer::Model& model, infer::Sequence& s, uint32_t last, size_t k) {
    Drafted d;
    model.draft(s, last, k, d.ids);
    for (size_t m = 0; m < d.ids.size(); ++m) d.logits.insert(d.logits.end(), model.draft_logits(m), model.draft_logits(m) + VOCAB);
    return d;
}

// A CPU backend whose drafter argmax marks its `invalid`-th row, counted over the backend's life, as an id past the vocabulary.
struct InvalidDraft : backend::CpuBackend {
    size_t calls = 0, invalid = 2;
    void argmax_rows(backend::Slice ids, backend::CSlice logits, size_t rows, size_t n, backend::CSlice after) override {
        backend::CpuBackend::argmax_rows(ids, logits, rows, n, after);
        if (++calls != invalid) return;
        const uint32_t past = (uint32_t)n;
        std::memcpy((float*)const_cast<void*>(ids.buffer->host_ptr()) + ids.offset, &past, sizeof(past));
    }
};
// A CPU backend that requires every product to take the model's activation dtype, and counts the heads' products.
struct DtypeWitness : backend::CpuBackend {
    backend::Dtype selected;
    size_t heads = 0;
    explicit DtypeWitness(backend::Dtype dtype) : selected(dtype) { set_threads(1); }
    void seen(backend::Dtype dtype, const char* what) const { require(dtype == selected, std::string(what) + " lost the model's activation dtype"); }
    void matmul(uint32_t type, backend::CSlice w, backend::CSlice x, backend::Slice y, size_t nin, size_t nout, size_t rows, backend::RowRuns runs = {},
                backend::Dtype dtype = backend::Dtype::f16) override {
        seen(dtype, "a product");
        backend::CpuBackend::matmul(type, w, x, y, nin, nout, rows, runs, dtype);
    }
    void matmul_add(uint32_t type, backend::CSlice w, backend::CSlice x, backend::Slice y, size_t nin, size_t nout, size_t rows, backend::RowRuns runs = {},
                    backend::Dtype dtype = backend::Dtype::f16) override {
        seen(dtype, "a residual product");
        backend::CpuBackend::matmul_add(type, w, x, y, nin, nout, rows, runs, dtype);
    }
    void matmul_group(std::initializer_list<backend::Projection> projections, backend::CSlice x, size_t nin, size_t rows, backend::RowRuns runs = {},
                      backend::Dtype dtype = backend::Dtype::f16) override {
        seen(dtype, "a grouped product");
        backend::CpuBackend::matmul_group(projections, x, nin, rows, runs, dtype);
    }
    void matmul_logits(uint32_t type, backend::CSlice w, backend::CSlice x, backend::Slice y, size_t nin, size_t nout, size_t rows, backend::RowRuns runs = {},
                       backend::Dtype dtype = backend::Dtype::f16) override {
        seen(dtype, "a head's product");
        ++heads;
        backend::CpuBackend::matmul_logits(type, w, x, y, nin, nout, rows, runs, dtype);
    }
};

// A CPU backend without the drafter's argmax.
struct NoArgmax : backend::CpuBackend {
    bool implements(backend::Op op) const override { return op != backend::Op::argmax_rows; }
};

// The embedded drafter (docs/SPECULATIVE.md, section 7) on the tiny MTP model: loaded, it leaves the logits as they were; its drafts and draft logits are those of a fresh sequence fed the same tokens in the same row classes, bit for bit, for a history in slices, beside another sequence, retracted at every position of a verify, forked and retracted at a checkpoint, reset, continued after a failed pass and split over four CPU stages; a first pass reads a zero carried row whatever its slot holds; drafting ends at the first invalid draft; a draft whose blocks run out takes none and changes nothing; and its refusals.
void drafts() {
    const gguf::GGUFModel file = tiny_mtp();
    const infer::ModelWeights w = infer::gguf_weights(file);
    const infer::ModelPlan with = infer::plan_model(w, true);
    require(with.drafter && with.drafter->roles.size() == 18 && with.slots.size() == 14 && with.draft_h == 10, "the drafter's plan");
    refuses("a drafter of a file without an MTP block", "the file carries no MTP block for an embedded drafter",
            [&] { const gguf::GGUFModel plain = tiny(); infer::plan_model(infer::gguf_weights(plain), true); });
    infer::ModelOptions options;
    options.state_slots = 3;
    options.checkpoint_slots = 2;
    options.mark_slots = 1;
    options.mark_rows = 5;
    auto make = [&](backend::BackendPtr b, infer::ModelOptions o) { return std::make_unique<infer::Model>(w, with, std::vector<backend::BackendPtr>{std::move(b)}, infer::Placement{}, o); };
    refuses("a backend without argmax_rows", "the embedded drafter needs argmax_rows, which the backend of its device does not implement",
            [&] { make(std::make_shared<NoArgmax>(), options); });
    std::vector<uint32_t> prompt(140);
    for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = (uint32_t)((i * 11 + 5) % VOCAB);
    const std::vector<uint32_t> steps = {9, 14, 3, 27};
    const uint32_t last = 7;
    const size_t depth = 4;

    // The drafter's products take the activation dtype the model was asked for, as the target's do: its context rows, each draft step's layer and its head.
    for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16}) {
        auto witness = std::make_shared<DtypeWitness>(dtype);
        infer::ModelOptions o = options;
        o.dtype = dtype;
        auto model = make(witness, o);
        model->prefill(prompt);
        const size_t before = witness->heads;
        std::vector<uint32_t> out;
        model->draft(last, depth, out);
        require(witness->heads == before + depth, "the drafter's head was not reached once a draft");
    }

    // Loaded, the drafter leaves every logit of the prompt and of decode as it was.
    {
        infer::Model plain(w, backend::make_cpu_backend(), options);
        auto drafting = make(backend::make_cpu_backend(), options);
        require(same(drafting->prefill(prompt), plain.prefill(prompt)), "a drafter changed the prompt's logits");
        for (uint32_t s : steps) require(same(drafting->step((int)s), plain.step((int)s)), "a drafter changed a decode step's logits");
        refuses("a draft without a drafter", "a draft without an embedded drafter", [&] { std::vector<uint32_t> out; plain.draft(last, 1, out); });
    }
    auto model = make(backend::make_cpu_backend(), options);
    infer::ExecContext ctx;
    // Each pass one entry: a prompt's rows at the prompt's extent, generated tokens at extent 1.
    auto feed = [&](infer::Model& m, infer::Sequence& s, const uint32_t* ids, size_t n, size_t extent, bool keep = false) {
        infer::BatchEntry e{&s, ids, n, false};
        e.extent = extent;
        e.keep = keep;
        m.forward(ctx, &e, 1);
    };
    auto history = [&](infer::Model& m, infer::Sequence& s, size_t generated) {
        feed(m, s, prompt.data(), prompt.size(), prompt.size());
        for (size_t i = 0; i < generated; ++i) feed(m, s, &steps[i], 1, 1);
    };
    {
        infer::Sequence e = model->make_sequence();
        refuses("a draft of an empty history", "a draft of a history no pass has fed", [&] { drafted(*model, e, last, 1); });
    }
    infer::Sequence a = model->make_sequence();
    history(*model, a, 2);
    const Drafted want = drafted(*model, a, last, depth);
    require(want.ids.size() == depth && a.length() == prompt.size() + 2, "a draft took fewer than its drafts or changed the history");
    require(drafted(*model, a, last, depth) == want, "a second draft of one history differs");
    model->reset(a);

    // The prompt in slices of 1, 3 and the rest, at its extent, then the steps.
    {
        infer::Sequence s = model->make_sequence();
        size_t at = 0;
        for (size_t n : {size_t(1), size_t(3), prompt.size() - 4}) {
            feed(*model, s, prompt.data() + at, n, prompt.size());
            at += n;
        }
        for (size_t i = 0; i < 2; ++i) feed(*model, s, &steps[i], 1, 1);
        require(drafted(*model, s, last, depth) == want, "a history in slices drafts otherwise");
        model->reset(s);
    }
    // Beside another sequence in every pass.
    {
        infer::Sequence s = model->make_sequence(), o = model->make_sequence();
        infer::BatchEntry both[] = {{&s, prompt.data(), prompt.size(), false}, {&o, prompt.data() + 10, 30, false}};
        both[0].extent = prompt.size();
        both[1].extent = 30;
        model->forward(ctx, both, 2);
        for (size_t i = 0; i < 2; ++i) {
            infer::BatchEntry pair[] = {{&o, &steps[3 - i], 1, false}, {&s, &steps[i], 1, false}};
            pair[0].extent = pair[1].extent = 1;
            model->forward(ctx, pair, 2);
        }
        require(drafted(*model, s, last, depth) == want, "a history beside another sequence drafts otherwise");
        model->reset(s);
        model->reset(o);
    }
    // Drafts of several histories in one batch, one a block each, chains of every length and one of none, asked in no order of length: each the drafts and every row's logits of its history alone.
    {
        infer::Sequence s[3] = {model->make_sequence(), model->make_sequence(), model->make_sequence()};
        feed(*model, s[0], prompt.data(), 40, 40);
        feed(*model, s[0], &steps[0], 1, 1);
        feed(*model, s[1], prompt.data(), 30, 30);
        feed(*model, s[2], prompt.data() + 60, 50, 50);
        const uint32_t lasts[3] = {last, 13, 3};
        for (const auto& ks : {std::array<size_t, 3>{1, depth, 0}, std::array<size_t, 3>{depth - 1, 2, depth}}) {
            Drafted alone[3];
            for (size_t i = 0; i < 3; ++i) alone[i] = drafted(*model, s[i], lasts[i], ks[i]);
            std::vector<uint32_t> out[3];
            infer::Model::DraftAsk asks[3];
            for (size_t i = 0; i < 3; ++i) asks[i] = {&s[i], lasts[i], ks[i], &out[i]};
            model->draft(asks, 3);
            for (size_t i = 0; i < 3; ++i) {
                Drafted d;
                d.ids = out[i];
                for (size_t m = 0; m < d.ids.size(); ++m) d.logits.insert(d.logits.end(), model->draft_logits(m, i), model->draft_logits(m, i) + VOCAB);
                require(d == alone[i] && d.ids.size() == ks[i], "a history drafted beside others drafts otherwise");
            }
        }
        for (auto& q : s) model->reset(q);
    }
    // A verify after a mark retracted to every position of it, as a round keeps n of its rows: the drafts are those of the history fed the kept tokens one at a time.
    const std::vector<uint32_t> verify = {11, 22, 5, 6, 17};
    for (size_t kept = 0; kept <= verify.size(); ++kept) {
        infer::Sequence s = model->make_sequence();
        history(*model, s, 2);
        require(model->mark(s), "a mark was refused");
        infer::BatchEntry e{&s, verify.data(), verify.size(), true};
        e.every_logits = true;
        e.extent = 1;
        model->forward(ctx, &e, 1);
        const size_t to = prompt.size() + 2 + kept;
        require(model->retract(s, to) == to, "a retract inside the verify missed its length");
        infer::Sequence f = model->make_sequence();
        history(*model, f, 2);
        for (size_t i = 0; i < kept; ++i) feed(*model, f, &verify[i], 1, 1);
        const Drafted fresh = drafted(*model, f, last, depth);
        require(drafted(*model, s, last, depth) == fresh, "a history retracted to " + std::to_string(kept) + " kept rows drafts otherwise");
        model->reset(s);
        model->reset(f);
    }
    // A checkpoint at a CPU block's end: a fork of it and a retract to it continue as the history never stopped.
    {
        infer::Sequence s = model->make_sequence();
        feed(*model, s, prompt.data(), 128, prompt.size(), true);
        feed(*model, s, prompt.data() + 128, prompt.size() - 128, prompt.size());
        for (size_t i = 0; i < 2; ++i) feed(*model, s, &steps[i], 1, 1);
        require(drafted(*model, s, last, depth) == want, "a history keeping a checkpoint drafts otherwise");
        infer::Sequence f = model->fork(s, 128);
        feed(*model, f, prompt.data() + 128, prompt.size() - 128, prompt.size());
        for (size_t i = 0; i < 2; ++i) feed(*model, f, &steps[i], 1, 1);
        require(drafted(*model, f, last, depth) == want, "a fork at a checkpoint drafts otherwise");
        require(model->retract(s, 135) == 128, "a retract past the checkpoint did not reach it");
        feed(*model, s, prompt.data() + 128, prompt.size() - 128, prompt.size());
        for (size_t i = 0; i < 2; ++i) feed(*model, s, &steps[i], 1, 1);
        require(drafted(*model, s, last, depth) == want, "a history retracted to its checkpoint drafts otherwise");
        model->reset(s);
        model->reset(f);
        history(*model, s, 2);
        require(drafted(*model, s, last, depth) == want, "a reset history drafts otherwise");
        model->reset(s);
    }
    // The host tier: a history copied to host memory at its checkpoint and promoted back into the other checkpoint slot drafts as one never evicted, the drafter's carried row going with the slot.
    {
        infer::Sequence s = model->make_sequence();
        feed(*model, s, prompt.data(), 128, prompt.size(), true);
        infer::HostHistory host;
        model->save_host(s, 128, host, SIZE_MAX);
        infer::Sequence r = model->restore_host(host);
        feed(*model, r, prompt.data() + 128, prompt.size() - 128, prompt.size());
        for (size_t i = 0; i < 2; ++i) feed(*model, r, &steps[i], 1, 1);
        require(drafted(*model, r, last, depth) == want, "a history promoted back from host memory drafts otherwise");
        model->reset(r);
        model->reset(s);
        model->release_host(host);
    }
    // A message boundary's state alone in host memory (Model::save_host without blocks) at the checkpoint at 128, the history going on past it: a fork of the history at 128 with that state, in a checkpoint slot of its own, drafts as one never evicted, the drafter's carried row coming back with the state.
    {
        infer::Sequence s = model->make_sequence();
        feed(*model, s, prompt.data(), 128, prompt.size(), true);
        infer::HostHistory state;
        model->save_host(s, 128, state, SIZE_MAX, false);
        feed(*model, s, prompt.data() + 128, prompt.size() - 128, prompt.size());
        infer::Sequence f = model->fork(s, 128, state);
        feed(*model, f, prompt.data() + 128, prompt.size() - 128, prompt.size());
        for (size_t i = 0; i < 2; ++i) feed(*model, f, &steps[i], 1, 1);
        require(drafted(*model, f, last, depth) == want, "a fork from a state alone in host memory drafts otherwise");
        model->reset(f);
        model->reset(s);
        model->release_host(state);
    }
    // A failed pass goes back to the checkpoint, and the history continues with the drafts of one never failed.
    {
        auto failing = std::make_shared<FailingHead>();
        auto broken = make(failing, options);
        infer::Sequence s = broken->make_sequence();
        feed(*broken, s, prompt.data(), 128, prompt.size(), true);
        feed(*broken, s, prompt.data() + 128, prompt.size() - 128, prompt.size());
        failing->fail = true;
        infer::BatchEntry e{&s, &steps[0], 1, true};
        e.extent = 1;
        refuses("a step failing", "injected", [&] { broken->forward(ctx, &e, 1); });
        require(s.length() == 128, "a failed step did not go back to the checkpoint");
        feed(*broken, s, prompt.data() + 128, prompt.size() - 128, prompt.size());
        for (size_t i = 0; i < 2; ++i) feed(*broken, s, &steps[i], 1, 1);
        require(drafted(*broken, s, last, depth) == want, "a history after a failed pass drafts otherwise");
    }
    // A slot whose carried row is NaN, left by a sequence whose token's embedding is NaN: the next sequence in that slot reads a zero carried row in its first pass, and drafts as a sequence in a clean slot does.
    {
        const uint32_t poison = VOCAB - 1;
        const gguf::GGUFModel tainted = tiny_mtp([&](gguf::GGUFModel& f) {
            add_tensor(f, "output.weight", {E, VOCAB});
            const float nan = std::numeric_limits<float>::quiet_NaN();
            for (size_t i = 0; i < E; ++i) std::memcpy(f.blob.data() + (poison * E + i) * sizeof(float), &nan, sizeof(nan));
        });
        const infer::ModelWeights tw = infer::gguf_weights(tainted);
        infer::Model m(tw, infer::plan_model(tw, true), {backend::make_cpu_backend()}, infer::Placement{}, options);
        const std::vector<uint32_t> clean = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        auto run = [&](infer::Sequence& s) {
            feed(m, s, clean.data(), clean.size(), clean.size());
            feed(m, s, &steps[0], 1, 1);
        };
        infer::Sequence f = m.make_sequence();
        run(f);
        const Drafted first = drafted(m, f, last, depth);
        m.reset(f);
        infer::Sequence nan = m.make_sequence();
        feed(m, nan, &poison, 1, 1);
        m.reset(nan);
        infer::Sequence s = m.make_sequence();
        run(s);
        require(drafted(m, s, last, depth) == first, "a first pass read its slot's NaN carried row");
        m.reset(s);
    }
    // The second draft row's id marked invalid on the device: one draft is kept, the rows after it end too.
    {
        auto invalid = std::make_shared<InvalidDraft>();
        auto m = make(invalid, options);
        infer::Sequence s = m->make_sequence();
        history(*m, s, 2);
        const Drafted d = drafted(*m, s, last, depth);
        require(d.ids.size() == 1 && d.ids[0] == want.ids[0], "drafting went on past an invalid draft");
    }
    // Over four CPU stages, a layer each, the drafter on the last.
    {
        infer::Placement place;
        place.mixer_device = place.ffn_device = {0, 1, 2, 3};
        place.embed_device = 0;
        place.output_device = 3;
        infer::Model four(w, with, {backend::make_cpu_backend(), backend::make_cpu_backend(), backend::make_cpu_backend(), backend::make_cpu_backend()}, place, options);
        infer::Sequence s = four.make_sequence();
        history(four, s, 2);
        require(drafted(four, s, last, depth) == want, "a split over four stages drafts otherwise");
        place.output_device = 2;
        refuses("a drafter away from the last stage", "an embedded drafter runs on the last stage's device",
                [&] { infer::Model off(w, with, {backend::make_cpu_backend(), backend::make_cpu_backend(), backend::make_cpu_backend(), backend::make_cpu_backend()}, place, options); });
    }
    // Two blocks of KV, one held by another sequence: a draft needing a new block takes none and changes nothing, and once the block is free it drafts as before; a draft inside its last block needs none.
    {
        infer::ModelOptions small = options;
        small.kv_tokens = 256;
        auto m = make(backend::make_cpu_backend(), small);
        infer::Sequence s = m->make_sequence(), o = m->make_sequence();
        feed(*m, s, prompt.data(), 127, 127);
        feed(*m, o, prompt.data(), 3, 3);
        require(drafted(*m, s, last, 1).ids.size() == 1, "a draft inside its last block was refused");
        feed(*m, s, &steps[0], 1, 1);
        refuses("a draft past the blocks", "KV cache: block budget exhausted", [&] { drafted(*m, s, last, 2); });
        require(s.length() == 128, "a failed draft changed the history");
        m->reset(o);
        auto other = make(backend::make_cpu_backend(), small);
        infer::Sequence f = other->make_sequence();
        feed(*other, f, prompt.data(), 127, 127);
        feed(*other, f, &steps[0], 1, 1);
        require(drafted(*m, s, last, 2) == drafted(*other, f, last, 2), "a draft after a failed one drafts otherwise");
    }
}

// A prompt in slices of 1, 3 and 16, and its decode, give the bytes of the prompt in one pass; two sequences in one pass give each one's bytes alone.
void slices(const gguf::GGUFModel& m, backend::Dtype dtype = backend::Dtype::f32) {
    const infer::ModelWeights w = infer::gguf_weights(m);
    const std::vector<uint32_t> ids = {7, 3, 11, 30, 2, 19, 5, 8, 13, 21, 1, 17, 4};
    std::vector<std::vector<float>> runs;
    for (int ubatch : {16, 1, 3}) {
        infer::ModelOptions options;
        options.dtype = dtype;
        infer::Model model(w, backend::make_cpu_backend(), options);
        model.set_ubatch(ubatch);
        std::vector<float> out = model.prefill(ids);
        for (uint32_t t : {9u, 14u, 27u}) {
            const std::vector<float> next = model.step((int)t);
            out.insert(out.end(), next.begin(), next.end());
        }
        runs.push_back(out);
    }
    require(same(runs[1], runs[0]) && same(runs[2], runs[0]), "a prompt's slices or its decode differ from one pass");

    infer::ModelOptions options;
    options.dtype = dtype;
    options.state_slots = 2;
    options.kv_tokens = 2 * 128;
    infer::Model model(w, backend::make_cpu_backend(), options);
    const std::vector<float> alone_a = model.prefill(ids);
    model.reset();
    const std::vector<float> alone_b = model.prefill({ids.begin(), ids.begin() + 5});
    model.reset();
    infer::Sequence a = model.make_sequence(), b = model.make_sequence();
    infer::ExecContext ctx;
    const infer::BatchEntry both[] = {{&a, ids.data(), ids.size(), true}, {&b, ids.data(), 5, true}};
    model.forward(ctx, both, 2);
    require(!std::memcmp(ctx.logits(0), alone_a.data(), VOCAB * sizeof(float)) && !std::memcmp(ctx.logits(1), alone_b.data(), VOCAB * sizeof(float)),
            "two sequences in one pass differ from each alone");
    model.reset(a);
    model.reset(b);
}

// Two CPU backends, the first stage holding only a linear-attention layer, which keeps a state and no KV, give one CPU's bytes: a prompt, decode steps and two sequences in one pass.
// With `ffn_apart` every layer's feed-forward part runs on the second backend and its mixer on the first, as experts on the CPU beside a device run.
void split_with_a_stage_of_states(const gguf::GGUFModel& m, bool ffn_apart = false, backend::Dtype dtype = backend::Dtype::f32) {
    const infer::ModelWeights w = infer::gguf_weights(m);
    infer::ModelOptions options;
    options.dtype = dtype;
    options.state_slots = 2;
    options.kv_tokens = 2 * 128;
    infer::Model one(w, backend::make_cpu_backend(), options);
    infer::Placement place;
    place.mixer_device = place.ffn_device = {0, 1, 1, 1};
    if (ffn_apart) {
        place.mixer_device = {0, 0, 0, 0};
        place.ffn_device = {1, 1, 1, 1};
    }
    place.embed_device = 0;
    place.output_device = 1;
    infer::Model two(w, {backend::make_cpu_backend(), backend::make_cpu_backend()}, place, options);
    require(two.kv_pools() == 1 && one.kv_pools() == 1 && (ffn_apart || two.stage_count() == 2), "a stage of states given a KV storage");
    for (int ubatch : {1, 3, 16}) {
        one.set_ubatch(ubatch);
        two.set_ubatch(ubatch);
        one.reset();
        two.reset();
        const std::vector<uint32_t> ids = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        require(same(one.prefill(ids), two.prefill(ids)), "a split's prompt differs at ubatch " + std::to_string(ubatch));
        for (uint32_t t : {1u, 2u, 3u}) require(same(one.step((int)t), two.step((int)t)), "a split's decode differs");
        require(one.kv_used_bytes() == two.kv_used_bytes() && one.n_tokens() == two.n_tokens(), "a split's counters differ");
    }
    one.reset();
    two.reset();
    const std::vector<uint32_t> ids = {4, 8, 15, 16, 23, 31};
    infer::Sequence a = one.make_sequence(), b = one.make_sequence(), c = two.make_sequence(), d = two.make_sequence();
    infer::ExecContext x, y;
    const infer::BatchEntry warm_one{&a, ids.data(), 3, false}, warm_two{&c, ids.data(), 3, false};
    one.forward(x, &warm_one, 1);
    two.forward(y, &warm_two, 1);
    const infer::BatchEntry pass_one[] = {{&a, ids.data() + 3, 1, true}, {&b, ids.data(), 6, true, true}};
    const infer::BatchEntry pass_two[] = {{&c, ids.data() + 3, 1, true}, {&d, ids.data(), 6, true, true}};
    one.forward(x, pass_one, 2);
    two.forward(y, pass_two, 2);
    for (size_t r = 0; r < 7; ++r) require(!std::memcmp(x.logits(r), y.logits(r), VOCAB * sizeof(float)), "a split's two-sequence pass differs");
    one.reset(a);
    one.reset(b);
    two.reset(c);
    two.reset(d);
}

// In this fixture only the router projects E to X and only the shared gate E to 1.
// Hold their policy separately: a whole-model witness cannot show one projection silently using the default.
struct GatePolicy : backend::CpuBackend {
    backend::Dtype selected;
    size_t routers = 0, shared = 0;
    explicit GatePolicy(backend::Dtype dtype) : selected(dtype) { set_threads(1); }
    void matmul(uint32_t type, backend::CSlice weights, backend::CSlice x, backend::Slice y,
                size_t nin, size_t nout, size_t rows, backend::RowRuns runs = {}, backend::Dtype dtype = backend::Dtype::f16) override {
        if (nin == E && nout == X) {
            require(dtype == backend::Dtype::f32, "router did not keep F32 inputs");
            ++routers;
        } else if (nin == E && nout == 1) {
            require(dtype == selected, "shared expert gate lost the selected dtype");
            ++shared;
        }
        backend::CpuBackend::matmul(type, weights, x, y, nin, nout, rows, runs, dtype);
    }
};

// qwen35moe: its refused keys, a routed layer's plan, and a prompt, its slices, two sequences and splits giving one pass's and one backend's bytes.
void experts() {
    auto read = [](const gguf::GGUFModel& m) { infer::gguf_weights(m); };
    auto moe_set = [](gguf::GGUFModel& m, const std::string& key, uint64_t value) {
        for (auto& kv : m.kv)
            if (kv.first == "qwen35moe." + key) kv.second.u = value;
    };
    refuses("no shared expert width", "missing metadata qwen35moe.expert_shared_feed_forward_length", [&] {
        gguf::GGUFModel m = routed(tiny());
        for (size_t i = 0; i < m.kv.size(); ++i)
            if (m.kv[i].first == "qwen35moe.expert_shared_feed_forward_length") m.kv.erase(m.kv.begin() + (long)i);
        read(m);
    });
    refuses("more experts a token than a layer has", "more experts per token than the layer has", [&] {
        gguf::GGUFModel m = routed(tiny());
        moe_set(m, "expert_used_count", X + 1);
        read(m);
    });
    refuses("another gating function", "unsupported expert gating function", [&] {
        gguf::GGUFModel m = routed(tiny());
        gguf::MetaValue v;
        v.vtype = gguf::V_UINT32;
        v.u = 2;
        m.kv.push_back({"qwen35moe.expert_gating_func", v});
        read(m);
    });

    const gguf::GGUFModel m = routed(tiny());
    const infer::ModelPlan p = infer::plan_model(infer::gguf_weights(m));
    // Planning reads only the storage tags here; these altered views are not executable quantized fixtures.
    for (const char* name : {"blk.0.ffn_gate_exps.weight", "blk.1.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"}) {
        auto weights = infer::gguf_weights(m);
        for (auto& t : weights.tensors)
            if (t.name == name) t.type = quant::GGML_TYPE_Q8_0;
        const auto mixed = infer::plan_model(weights);
        for (size_t l = 0; l < mixed.layers.size(); ++l) {
            const auto& ops = mixed.layers[l].ops;
            const auto count = std::count_if(ops.begin(), ops.end(), [](const auto& u) {
                return u.part == infer::Part::ffn && u.op == backend::Op::mixed_experts;
            });
            const bool needed = (l == 0 && std::string(name) == "blk.0.ffn_gate_exps.weight") ||
                                (l == 1 && std::string(name) == "blk.1.ffn_up_exps.weight");
            require(count == (needed ? 1 : 0), "a routed layer's mixed gate/up capability differs from its projections");
        }
    }
    for (size_t l = 0; l < 4; ++l) {
        const infer::LayerPlan& layer = p.layers[l];
        const bool full = l % 2;
        require(layer.routed && layer.roles.size() == (full ? 16u : 19u), "routed layer " + std::to_string(l) + "'s roles");
        require(layer.ops.back().part == infer::Part::ffn && layer.ops.back().op == backend::Op::sigmoid_mul, "the shared expert's gate op");
        for (const infer::Role& role : layer.roles) {
            if (role.part != infer::Part::ffn) continue;
            const bool stack = role.kind == infer::RoleKind::experts;
            require(role.stream == (stack ? infer::Stream::window : infer::Stream::copy), "role " + role.name + "'s stream");
        }
    }
    require(p.slots.size() == 14 && p.slots[9] == std::max(K * FE, FS) && p.slots[10] == X && p.slots[11] == K && p.slots[13] == 1, "a routed model's slots");
    require(infer::gguf_weights(m).declared_dtype == backend::Dtype::bf16, "routed architecture dtype default");
    for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16}) {
        auto backend = std::make_shared<GatePolicy>(dtype);
        infer::ModelOptions options;
        options.dtype = dtype;
        infer::Model model(infer::gguf_weights(m), backend, options);
        model.prefill({7, 3, 11});
        model.step(9);
        require(backend->routers == 8 && backend->shared == 8, "gate policy check missed a layer or phase");
        slices(m, dtype);
        split_with_a_stage_of_states(m, false, dtype);
        split_with_a_stage_of_states(m, true, dtype);
    }
}

} // namespace

int main() {
    try {
        configuration();
        plan();
        footprint();
        state_rules();
        checkpoints();
        host_round_trip();
        host_state_fork();
        checkpoint_fit();
        refused_passes_take_no_slot();
        failed_admission_takes_no_slot();
        marks();
        drafts();
        slices(tiny());
        split_with_a_stage_of_states(tiny());
        experts();
    } catch (const std::exception& e) {
        std::cerr << "arch-qwen35: FAIL: " << e.what() << "\n";
        return 1;
    }
    std::cout << "arch-qwen35: " << checked << " checks passed\n";
    return 0;
}
