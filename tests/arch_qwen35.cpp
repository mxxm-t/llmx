// The qwen35 module (model/arch/qwen35.hpp) on tiny models built in memory: every refused key and tensor with its text, the plan of each layer kind, the footprint, and the runtime's rules for a model whose layers keep a recurrent state.
// The math is held to HF by the suite's qwen35 component; here a split and slices are held to one CPU bit for bit, which the per-token recurrence gives.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "model/runtime.hpp"
#include "model/place.hpp"
#include "model/arch/registry.hpp"
#include "tiny_qwen.hpp"

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
                       backend::RowRuns runs = {}) override {
        if (fail) {
            fail = false;
            throw std::runtime_error("injected");
        }
        backend::CpuBackend::matmul_logits(type, data, x, y, nin, nout, nbatch, runs);
    }
};

// The runtime's rules for a model that keeps a state: the op check, no fork, the slots, and a failed pass losing the state.
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
    refuses("a fork", "a fork of a model whose layers keep a recurrent state", [&] { model.fork(a, 0); });
    const infer::BatchEntry rest{&a, ids.data() + 4, 4, true};
    model.forward(ctx, &rest, 1);
    require(!std::memcmp(ctx.logits(0), want.data(), VOCAB * sizeof(float)), "a sequence in two passes differs from the prompt in one");
    model.reset(a);

    // A failed pass on a history loses the state it touched; a failure from length 0 leaves a zero state behind, which is no loss.
    auto failing = std::make_shared<FailingHead>();
    infer::Model broken(w, failing);
    failing->fail = true;
    refuses("the first pass failing", "injected", [&] { broken.prefill(ids); });
    require(broken.n_tokens() == 0, "a failed first prompt left a history");
    require(same(broken.prefill(ids), want), "a sequence after a failed first pass differs");
    failing->fail = true;
    refuses("a step failing", "injected", [&] { broken.step(7); });
    refuses("a step after a lost state", "a sequence whose recurrent state a failed pass lost continues only from a reset", [&] { broken.step(7); });
    broken.reset();
    require(same(broken.prefill(ids), want), "a reset sequence differs from a fresh one");
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

// A prompt in slices of 1, 3 and 16, and its decode, give the bytes of the prompt in one pass; two sequences in one pass give each one's bytes alone.
void slices() {
    const gguf::GGUFModel m = tiny();
    const infer::ModelWeights w = infer::gguf_weights(m);
    const std::vector<uint32_t> ids = {7, 3, 11, 30, 2, 19, 5, 8, 13, 21, 1, 17, 4};
    std::vector<std::vector<float>> runs;
    for (int ubatch : {16, 1, 3}) {
        infer::Model model(w);
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
void split_with_a_stage_of_states() {
    const gguf::GGUFModel m = tiny();
    const infer::ModelWeights w = infer::gguf_weights(m);
    infer::ModelOptions options;
    options.state_slots = 2;
    options.kv_tokens = 2 * 128;
    infer::Model one(w, backend::make_cpu_backend(), options);
    infer::Placement place;
    place.mixer_device = place.ffn_device = {0, 1, 1, 1};
    place.embed_device = 0;
    place.output_device = 1;
    infer::Model two(w, {backend::make_cpu_backend(), backend::make_cpu_backend()}, place, options);
    require(two.kv_pools() == 1 && one.kv_pools() == 1 && two.stage_count() == 2, "a stage of states given a KV storage");
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

} // namespace

int main() {
    try {
        configuration();
        plan();
        footprint();
        state_rules();
        refused_passes_take_no_slot();
        slices();
        split_with_a_stage_of_states();
    } catch (const std::exception& e) {
        std::cerr << "arch-qwen35: FAIL: " << e.what() << "\n";
        return 1;
    }
    std::cout << "arch-qwen35: " << checked << " checks passed\n";
    return 0;
}
