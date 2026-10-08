#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "inference/load.hpp"
#include "loading_backend.hpp"

namespace {
size_t checks = 0;
// Every refusal the checks provoke, as "label: message" in the order provoked, which main holds to the list in tests/data/model_refusals.txt.
std::vector<std::string> refusals;

void require(bool condition, const std::string& label) {
    if (!condition) throw std::runtime_error(label);
}

template<class F> void rejects(const std::string& label, F work) {
    try { work(); }
    catch (const std::runtime_error& e) { refusals.push_back(label + ": " + e.what()); ++checks; return; }
    throw std::runtime_error("accepted invalid model: " + label);
}

gguf::MetaValue integer(uint64_t n, uint32_t type = gguf::V_UINT32) {
    gguf::MetaValue v;
    v.vtype = type; v.u = n; v.i = int64_t(n);
    return v;
}

gguf::MetaValue real(double n, uint32_t type = gguf::V_FLOAT64) {
    gguf::MetaValue v;
    v.vtype = type; v.f64 = n;
    if (type == gguf::V_FLOAT32) {
        const float f = float(n);
        std::memcpy(&v.fb, &f, sizeof(f));
    }
    return v;
}

gguf::MetaValue text(const std::string& s) {
    gguf::MetaValue v;
    v.vtype = gguf::V_STRING; v.s = s;
    return v;
}

void set(gguf::GGUFModel& m, const std::string& key, const gguf::MetaValue& value) {
    for (auto& kv : m.kv) if (kv.first == key) { kv.second = value; return; }
    m.kv.push_back({key, value});
}

void erase_key(gguf::GGUFModel& m, const std::string& key) {
    m.kv.erase(std::remove_if(m.kv.begin(), m.kv.end(),
        [&](const auto& kv) { return kv.first == key; }), m.kv.end());
}

void add(gguf::GGUFModel& m, const std::string& name,
         std::vector<uint64_t> shape, uint32_t type, bool norm = false) {
    size_t n = 1;
    for (uint64_t d : shape) n *= size_t(d);
    size_t bytes = 0;
    switch (type) {
        case 0: bytes = n * 4; break;
        case 2: bytes = n / 32 * 18; break;
        case 3: bytes = n / 32 * 20; break;
        case 8: bytes = n / 32 * 34; break;
        case 12: bytes = n / 256 * 144; break;
        case 13: bytes = n / 256 * 176; break;
        case 14: bytes = n / 256 * 210; break;
        default: throw std::runtime_error("invalid fixture type");
    }
    const size_t offset = (m.blob.size() + 3) / 4 * 4;
    m.blob.resize(offset + bytes, 0);
    if (norm) {
        const float one = 1.0f;
        for (size_t i = 0; i < n; ++i)
            std::memcpy(m.blob.data() + offset + i * 4, &one, 4);
    }
    m.tensors.push_back({name, std::move(shape), type, 0});
    m.offsets.push_back(offset);
}

void erase_tensor(gguf::GGUFModel& m, const std::string& name) {
    const auto it = std::find_if(m.tensors.begin(), m.tensors.end(), [&](const auto& t) { return t.name == name; });
    require(it != m.tensors.end(), "the fixture has no tensor " + name);
    m.offsets.erase(m.offsets.begin() + (it - m.tensors.begin()));
    m.tensors.erase(it);
}

gguf::GGUFModel fixture(bool tied = false, uint32_t type = 0, bool odd = false,
                        bool grouped = true, uint64_t layers = 1) {
    gguf::GGUFModel m;
    const uint64_t emb = type ? 256 : (odd ? 7 : 8);
    const uint64_t ff = type ? 256 : 12;
    const uint64_t hd = type ? 128 : 4;
    for (const auto& kv : std::vector<std::pair<std::string, uint64_t>>{
            {"block_count", layers}, {"embedding_length", emb}, {"feed_forward_length", ff},
            {"attention.head_count", 2}, {"attention.head_count_kv", grouped ? 1u : 2u},
            {"attention.key_length", hd}, {"context_length", 8}})
        m.kv.push_back({"qwen3." + kv.first, integer(kv.second)});
    add(m, "token_embd.weight", {emb, 5}, type);
    add(m, "output_norm.weight", {emb}, 0, true);
    for (uint64_t l = 0; l < layers; ++l) {
        const std::string pre = "blk." + std::to_string(l) + ".";
        for (const char* name : {"attn_norm", "ffn_norm"})
            add(m, pre + name + ".weight", {emb}, 0, true);
        for (const char* name : {"attn_q_norm", "attn_k_norm"})
            add(m, pre + name + ".weight", {hd}, 0, true);
        add(m, pre + "attn_q.weight", {emb, 2 * hd}, type);
        add(m, pre + "attn_k.weight", {emb, hd * (grouped ? 1 : 2)}, type);
        add(m, pre + "attn_v.weight", {emb, hd * (grouped ? 1 : 2)}, type);
        add(m, pre + "attn_output.weight", {2 * hd, emb}, type);
        add(m, pre + "ffn_gate.weight", {emb, ff}, type);
        add(m, pre + "ffn_up.weight", {emb, ff}, type);
        add(m, pre + "ffn_down.weight", {ff, emb}, type);
    }
    if (!tied) add(m, "output.weight", {emb, 5}, type);
    return m;
}

// Hundreds of models are built here, so each one's backend must start no worker threads: its pool would start only for work, and one thread needs none.
void construct(const gguf::GGUFModel& m, bool step = false) {
    auto cpu = std::make_shared<backend::CpuBackend>();
    cpu->set_threads(1);
    infer::Model model(infer::gguf_weights(m), cpu);
    if (step) {
        const auto logits = model.step(0);
        require(logits.size() == 5, "valid fixture vocabulary changed");
        for (float v : logits) require(v == 0.0f, "zero-weight fixture produced nonzero logits");
    }
    require(cpu->workers_started() == 0, "a one-thread model started worker threads");
}



void loading_lifetime_checks() {
    for (int failure = 0; failure < 4; ++failure) {
        auto m = fixture();
        auto b = std::make_shared<LoadingBackend>();
        b->set_threads(1);
        std::string expected;
        if (failure == 0) {
            erase_tensor(m, "blk.0.ffn_down.weight");
            expected = "inference: missing tensor blk.0.ffn_down.weight";
        } else if (failure == 1 || failure == 3) {
            b->fail_adopt = failure == 1 ? 4 : 16;
            expected = "injected adoption failure";
        } else {
            b->fail_cache = true;
            expected = "injected cache allocation failure";
        }
        std::string caught;
        try { infer::Model model(infer::gguf_weights(m), b); }
        catch (const std::runtime_error& e) { caught = e.what(); }
        require(caught == expected, "loading exception was lost or replaced");
        require(b->state->releases > 0 && b->state->premature == 0 && b->state->drains > 0,
                "loading failure freed a buffer before its upload retired");
        require(b->workers_started() == 0, "a one-thread model started worker threads");
        ++checks;
    }
    const auto m = fixture();
    auto a = std::make_shared<LoadingBackend>(), b = std::make_shared<LoadingBackend>();
    auto unused = std::make_shared<LoadingBackend>();
    for (auto& device : {a, b, unused}) device->set_threads(1);
    infer::Placement placement;
    placement.mixer_device = {0}; placement.ffn_device = {1};
    placement.embed_device = 0; placement.output_device = 1;
    b->fail_adopt = 4;
    rejects("split loading failure", [&] { infer::Model model(infer::gguf_weights(m), {a, b, unused}, placement); });
    for (const auto& device : {a, b})
        require(device->state->releases > 0 && device->state->premature == 0 && device->state->drains > 0,
                "split loading failure did not drain each used backend before release");
    require(unused->state->drains == 0, "loading failure drained an unused backend");
    b->fail_adopt = 0;
    const int drained = a->state->drains + b->state->drains;
    {
        infer::Model model(infer::gguf_weights(m), {a, b, unused}, placement);
        require(a->state->pending && b->state->pending, "successful loading unexpectedly became synchronous");
        require(a->state->drains + b->state->drains == drained, "successful loading added a drain");
    }
    for (const auto& device : {a, b})
        require(!device->state->pending && device->state->premature == 0,
                "model teardown freed buffers before pending loading completed");
    require(unused->state->drains == 0, "model teardown drained an unused backend");
    for (const auto& device : {a, b, unused})
        require(device->workers_started() == 0, "a one-thread split model started worker threads");
    ++checks;
}


// The fixture with its one layer routed over two experts of `expert_ff` rows each; the layer keeps the fixture's dense feed-forward tensors, which a routed layer does not read.
gguf::GGUFModel routed_fixture(uint64_t expert_ff = 12) {
    auto m = fixture();
    for (auto& kv : m.kv) kv.first.replace(0, 5, "qwen3moe");
    set(m, "general.architecture", text("qwen3moe"));
    set(m, "qwen3moe.expert_count", integer(2));
    set(m, "qwen3moe.expert_used_count", integer(1));
    set(m, "qwen3moe.expert_feed_forward_length", integer(expert_ff));
    add(m, "blk.0.ffn_gate_inp.weight", {8, 2}, 0);
    add(m, "blk.0.ffn_gate_exps.weight", {8, expert_ff, 2}, 0);
    add(m, "blk.0.ffn_up_exps.weight", {8, expert_ff, 2}, 0);
    add(m, "blk.0.ffn_down_exps.weight", {expert_ff, 8, 2}, 0);
    return m;
}

// The routed fixture without the dense feed-forward tensors, so the model reads every tensor it holds.
gguf::GGUFModel routed_only() {
    auto m = routed_fixture();
    for (const char* name : {"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"}) erase_tensor(m, name);
    return m;
}

// The fit counts what the model reads: a layer with a router is routed, so the dense matrices it also carries neither widen the feed-forward slots nor count among its weights.
void fit_width_checks() {
    const auto m = routed_fixture(6);
    auto only = m;
    for (const char* name : {"blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"}) erase_tensor(only, name);
    const infer::ModelWeights weights = infer::gguf_weights(m), routed = infer::gguf_weights(only);
    const infer::Footprint fp = infer::footprint(weights, infer::plan_model(weights), infer::ModelOptions{});
    const infer::Footprint alone = infer::footprint(routed, infer::plan_model(routed), infer::ModelOptions{});
    require(fp.activations_per_row == alone.activations_per_row, "the fit widens a routed layer's slots for the dense ffn_gate it also carries");
    // The layer's four norms, four attention projections, router and three expert stacks.
    require(fp.layers.size() == 1 && fp.layers[0].size() == 12, "the fit counts the dense matrices a routed layer does not read");
    infer::Model model(weights);
    ++checks;
}

// An architecture that hands the runtime another's plan after `edit`, for the checks every plan passes through.
struct EditedPlan final : infer::Architecture {
    EditedPlan(std::shared_ptr<const infer::Architecture> a, std::function<void(infer::ModelPlan&)> e) : arch(std::move(a)), edit(std::move(e)) {}
    infer::ModelPlan plan(const infer::TensorIndex& tensors) const override {
        infer::ModelPlan p = arch->plan(tensors);
        edit(p);
        return p;
    }
    void fill_tables(std::vector<std::vector<float>>& tables) const override { arch->fill_tables(tables); }
    void embed(const infer::Step& s, const uint32_t* ids) const override { arch->embed(s, ids); }
    void mixer(const infer::Step& s) const override { arch->mixer(s); }
    void ffn(const infer::Step& s) const override { arch->ffn(s); }
    void head(const infer::HeadStep& s) const override { arch->head(s); }
    std::shared_ptr<const infer::Architecture> arch;
    std::function<void(infer::ModelPlan&)> edit;
};

// A wrong plan is the architecture's error, not a refusal of the file: the fit has a field for every pass role and one role for each, and every plan has slot 0 as wide as the residual and its role ids inside its rows of weights.
void plan_checks() {
    const auto file = fixture();
    const infer::ModelWeights weights = infer::gguf_weights(file);
    auto wrong = [&](const std::string& label, const std::function<void()>& work) {
        try { work(); } catch (const std::logic_error&) { ++checks; return; }
        throw std::runtime_error("accepted a wrong plan: " + label);
    };
    auto fit = [&](const std::function<void(infer::ModelPlan&)>& edit) {
        infer::ModelPlan plan = infer::plan_model(weights);
        edit(plan);
        infer::footprint(weights, plan, infer::ModelOptions{});
    };
    wrong("a pass role the fit has no field for", [&] { fit([](infer::ModelPlan& p) { p.pass[0].kind = infer::RoleKind::norm; }); });
    wrong("a second head matrix", [&] { fit([](infer::ModelPlan& p) { p.pass.push_back(p.pass[1]); }); });
    auto planned = [&](std::function<void(infer::ModelPlan&)> edit) {
        infer::ModelWeights edited = weights;
        edited.arch = std::make_shared<const EditedPlan>(weights.arch, std::move(edit));
        infer::plan_model(edited);
    };
    wrong("slot 0 narrower than the residual", [&] { planned([](infer::ModelPlan& p) { --p.slots[0]; }); });
    wrong("a role id past its row of weights", [&] { planned([](infer::ModelPlan& p) { p.layers[0].roles[0].id = uint16_t(p.role_ids); }); });
}

// Layer 0's attention on a device and its experts on a host, streamed to the device from one new token.
infer::Placement streamed_placement() {
    infer::Placement placement;
    placement.mixer_device = {0}; placement.ffn_device = {1};
    placement.stream_from = 1;
    return placement;
}

void loading_window_checks() {
    const auto m = routed_fixture();
    const infer::Placement placement = streamed_placement();
    for (int failure = 1; failure <= 3; ++failure) {
        auto device = std::make_shared<LoadingBackend>();
        auto host = std::make_shared<backend::CpuBackend>();
        device->set_threads(1); host->set_threads(1);
        device->fail_alloc = failure;
        std::string caught;
        try { infer::Model model(infer::gguf_weights(m), {device, host}, placement); }
        catch (const std::runtime_error& e) { caught = e.what(); }
        require(caught == "injected window allocation failure", "streamed window failure was not reached");
        require(device->state->allocations == failure && device->state->releases > 0,
                "streamed window fixture did not create pending storage");
        require(device->state->premature == 0 && device->state->drains > 0,
                "window allocation failure freed pending storage before the constructor catch");
        require(device->workers_started() == 0 && host->workers_started() == 0, "a one-thread model started worker threads");
        ++checks;
    }
}

// The model puts each weight on a backend through the loader's hook, which records the tensors a host reads in place (infer::planning_adopt).
// A copying backend reads none; beside a device, a host that runs a streamed layer's experts reads exactly those, its norm and its router, which the device also takes.
// The hook sees the weights in the order the loader uploads them: the embedding, the head and its norm, then each layer's attention, its feed-forward block, and a streamed layer's copies beside its mixer.
void reader_checks() {
    // What the hook saw: the tensors a host reads in place, sorted, how often each tensor was taken, and each take as "name@backend", in order.
    struct Reads {
        std::vector<std::string> host;
        std::vector<int> takes;
        std::string order;
    };
    auto read_in_place = [](const infer::ModelWeights& weights, std::vector<backend::BackendPtr> backends, infer::Placement placement) {
        Reads r;
        r.takes.assign(weights.tensors.size(), 0);
        std::vector<const backend::Backend*> ids;
        for (const auto& b : backends) ids.push_back(b.get());
        infer::WeightPlan plan;
        const infer::AdoptWeight record = infer::planning_adopt(weights, backends.size(), plan, true);
        const infer::AdoptWeight adopt = [&](size_t i, backend::Backend& b, const std::vector<infer::shard::Run>& runs) {
            ++r.takes[i];
            const size_t id = size_t(std::find(ids.begin(), ids.end(), &b) - ids.begin());
            r.order += (r.order.empty() ? "" : " ") + weights.tensors[i].name + "@" + std::to_string(id);
            return record(i, b, runs);
        };
        { infer::Model model(weights, std::move(backends), std::move(placement), {}, adopt); }
        for (size_t i = 0; i < plan.host_reads.size(); ++i)
            if (plan.host_reads[i]) r.host.push_back(weights.tensors[i].name);
        std::sort(r.host.begin(), r.host.end());
        return r;
    };
    const auto dense = fixture();
    auto device = std::make_shared<LoadingBackend>();
    device->set_threads(1);
    const Reads copied = read_in_place(infer::gguf_weights(dense), {device}, infer::Placement{});
    require(copied.host.empty(), "a copying backend read a weight in place");
    for (int n : copied.takes) require(n == 1, "a dense model did not take every tensor once through the hook");
    require(copied.order == "token_embd.weight@0 output.weight@0 output_norm.weight@0 "
                            "blk.0.attn_norm.weight@0 blk.0.attn_q_norm.weight@0 blk.0.attn_k_norm.weight@0 blk.0.attn_q.weight@0 "
                            "blk.0.attn_k.weight@0 blk.0.attn_v.weight@0 blk.0.attn_output.weight@0 "
                            "blk.0.ffn_norm.weight@0 blk.0.ffn_gate.weight@0 blk.0.ffn_up.weight@0 blk.0.ffn_down.weight@0",
            "a dense model took its weights in another order: " + copied.order);
    require(device->workers_started() == 0, "a one-thread model started worker threads");
    // A tied head on another device than the embedding takes the embedding there too.
    auto second = std::make_shared<LoadingBackend>();
    device = std::make_shared<LoadingBackend>();
    device->set_threads(1); second->set_threads(1);
    infer::Placement across;
    across.mixer_device = {0}; across.ffn_device = {1};
    across.embed_device = 0; across.output_device = 1;
    const Reads tied = read_in_place(infer::gguf_weights(fixture(true)), {device, second}, across);
    require(tied.order == "token_embd.weight@0 token_embd.weight@1 output_norm.weight@1 "
                          "blk.0.attn_norm.weight@0 blk.0.attn_q_norm.weight@0 blk.0.attn_k_norm.weight@0 blk.0.attn_q.weight@0 "
                          "blk.0.attn_k.weight@0 blk.0.attn_v.weight@0 blk.0.attn_output.weight@0 "
                          "blk.0.ffn_norm.weight@1 blk.0.ffn_gate.weight@1 blk.0.ffn_up.weight@1 blk.0.ffn_down.weight@1",
            "a tied head across two devices took its weights in another order: " + tied.order);
    require(device->workers_started() == 0 && second->workers_started() == 0, "a one-thread model started worker threads");
    const auto routed = routed_fixture();
    const infer::ModelWeights weights = infer::gguf_weights(routed);
    auto host = std::make_shared<backend::CpuBackend>();
    device = std::make_shared<LoadingBackend>();
    device->set_threads(1); host->set_threads(1);
    const Reads seen = read_in_place(weights, {device, host}, streamed_placement());
    require(device->workers_started() == 0 && host->workers_started() == 0, "a one-thread model started worker threads");
    require(seen.host == std::vector<std::string>({"blk.0.ffn_down_exps.weight", "blk.0.ffn_gate_exps.weight",
                                                   "blk.0.ffn_gate_inp.weight", "blk.0.ffn_norm.weight",
                                                   "blk.0.ffn_up_exps.weight"}),
            "a streamed layer's host did not read exactly its norm, router and experts in place");
    // The fixture keeps its dense feed-forward matrices, which a routed layer does not take.
    for (size_t i = 0; i < weights.tensors.size(); ++i) {
        const std::string& name = weights.tensors[i].name;
        const bool both = name == "blk.0.ffn_norm.weight" || name == "blk.0.ffn_gate_inp.weight";
        const bool dense_ffn = name == "blk.0.ffn_gate.weight" || name == "blk.0.ffn_up.weight" || name == "blk.0.ffn_down.weight";
        require(seen.takes[i] == (both ? 2 : dense_ffn ? 0 : 1),
                "a streamed layer's norm and router not taken by both backends, or another tensor taken other than once");
    }
    require(seen.order == "token_embd.weight@0 output.weight@0 output_norm.weight@0 "
                          "blk.0.attn_norm.weight@0 blk.0.attn_q_norm.weight@0 blk.0.attn_k_norm.weight@0 blk.0.attn_q.weight@0 "
                          "blk.0.attn_k.weight@0 blk.0.attn_v.weight@0 blk.0.attn_output.weight@0 "
                          "blk.0.ffn_norm.weight@1 blk.0.ffn_gate_inp.weight@1 blk.0.ffn_gate_exps.weight@1 "
                          "blk.0.ffn_up_exps.weight@1 blk.0.ffn_down_exps.weight@1 blk.0.ffn_norm.weight@0 blk.0.ffn_gate_inp.weight@0",
            "a streamed layer took its weights in another order: " + seen.order);
    checks += 5;
}

void metadata_checks() {
    const auto base = fixture();
    const std::vector<std::string> required = {"block_count", "embedding_length",
        "feed_forward_length", "attention.head_count"};
    const std::vector<std::string> optional = {"attention.head_count_kv", "attention.key_length",
        "context_length", "attention.value_length", "rope.dimension_count"};
    for (const auto& key : required) {
        auto m = base; erase_key(m, "qwen3." + key);
        rejects("missing " + key, [&] { infer::gguf_weights(m); });
    }
    auto keys = required;
    keys.insert(keys.end(), optional.begin(), optional.end());
    auto negative = integer(0, gguf::V_INT64); negative.i = -1;
    auto negative32 = integer(0, gguf::V_INT32); negative32.i = -1;
    for (const auto& key : keys) {
        for (const auto& value : std::vector<gguf::MetaValue>{integer(0), negative, negative32,
                integer(uint64_t(std::numeric_limits<int>::max()) + 1, gguf::V_UINT64),
                text("2"), real(2), integer(2, gguf::V_BOOL), integer(2, gguf::V_ARRAY),
                integer(2, gguf::V_UINT8), integer(2, gguf::V_INT8),
                integer(2, gguf::V_UINT16), integer(2, gguf::V_INT16)}) {
            auto m = base; set(m, "qwen3." + key, value);
            rejects("invalid integer " + key, [&] { infer::gguf_weights(m); });
        }
    }
    for (uint32_t type : {gguf::V_UINT32, gguf::V_INT32, gguf::V_UINT64, gguf::V_INT64}) {
        auto m = base;
        for (auto& kv : m.kv) kv.second = integer(kv.second.u, type);
        construct(m); ++checks;
    }
    for (const char* key : {"qwen3.rope.freq_base", "qwen3.attention.layer_norm_rms_epsilon"}) {
        for (const auto& value : std::vector<gguf::MetaValue>{real(0), real(-1),
                real(std::numeric_limits<double>::infinity()), real(std::numeric_limits<double>::quiet_NaN()),
                real(std::numeric_limits<double>::max()), real(std::numeric_limits<double>::denorm_min()),
                real(std::numeric_limits<float>::infinity(), gguf::V_FLOAT32),
                real(std::numeric_limits<float>::quiet_NaN(), gguf::V_FLOAT32), text("1"), integer(1)}) {
            auto m = base; set(m, key, value);
            rejects(std::string("invalid float ") + key, [&] { infer::gguf_weights(m); });
        }
        for (uint32_t type : {gguf::V_FLOAT32, gguf::V_FLOAT64}) {
            auto m = base; set(m, key, real(0.5, type));
            infer::gguf_weights(m); ++checks;
        }
    }
    for (const auto& item : std::vector<std::pair<std::string, std::string>>{
            {"general.architecture", "qwen3"}, {"qwen3.rope.scaling.type", "none"},
            {"qwen3.tensor_data_layout", "reference"}}) {
        auto m = base; set(m, item.first, text(item.second)); construct(m); ++checks;
        for (const auto& value : {text("unsupported"), text(""), integer(1)}) {
            m = base; set(m, item.first, value);
            rejects("invalid " + item.first, [&] { infer::gguf_weights(m); });
        }
    }
    for (const char* key : {"qwen3.rope.scaling.factor", "qwen3.rope.scale_linear"}) {
        for (const auto& value : std::vector<gguf::MetaValue>{real(0), real(0.5), real(2), real(1 + 1e-10),
                real(std::numeric_limits<double>::infinity()), real(std::numeric_limits<double>::quiet_NaN()),
                integer(1), text("1")}) {
            auto m = base; set(m, key, value);
            rejects(std::string("invalid ") + key, [&] { infer::gguf_weights(m); });
        }
        for (uint32_t type : {gguf::V_FLOAT32, gguf::V_FLOAT64}) {
            auto m = base; set(m, key, real(1, type)); construct(m); ++checks;
        }
    }
    auto m = fixture(false, 0, false, false);
    for (const char* key : {"attention.head_count_kv", "attention.key_length", "context_length"})
        erase_key(m, std::string("qwen3.") + key);
    const auto config = infer::qwen3::read_config(m, "qwen3.", false);
    require(config.n_head_kv == 2 && config.head_dim == 4 && config.context_length == 4096 &&
            config.rope_theta == 10000.0f && config.rms_eps == 1e-6f, "incorrect absent defaults");
    construct(m, true);
    ++checks;
    set(m, "qwen3.embedding_length", integer(7));
    rejects("indivisible default head width", [&] { infer::gguf_weights(m); });
    m = base;
    set(m, "qwen3.attention.value_length", integer(4));
    set(m, "qwen3.rope.dimension_count", integer(4));
    construct(m); ++checks;
    for (const auto& item : std::vector<std::pair<std::string, uint64_t>>{
            {"attention.head_count_kv", 3}, {"attention.head_count", 3},
            {"attention.key_length", 3}, {"attention.value_length", 2}, {"rope.dimension_count", 2},
            {"attention.key_length", uint64_t(std::numeric_limits<int>::max()) - 1}}) {
        m = base;
        if (item.first == "attention.head_count") set(m, "qwen3.attention.head_count_kv", integer(2));
        set(m, "qwen3." + item.first, integer(item.second));
        rejects("incompatible geometry " + item.first, [&] { infer::gguf_weights(m); });
    }
    m = base;
    const uint64_t limit = uint64_t(std::numeric_limits<int>::max());
    set(m, "qwen3.attention.head_count", integer(1));
    set(m, "qwen3.attention.key_length", integer(limit - 1));
    set(m, "qwen3.context_length", integer(limit));
    // Only a vector limit below the widths the reader takes reaches this refusal, so its text is held here rather than in the list every platform shares.
    if (limit * (limit - 1) > std::vector<float>().max_size()) {
        std::string caught;
        try { infer::gguf_weights(m); } catch (const std::runtime_error& e) { caught = e.what(); }
        require(caught == "inference: context storage exceeds allocation limit", "KV float capacity not refused with its text");
    } else {
        infer::gguf_weights(m);
    }
    ++checks;
}

// Tensor i of a model that builds, left out (but for the head, whose absence ties it) and in five malformed shapes.
void missing_and_shapes(const gguf::GGUFModel& base, size_t i) {
    const auto name = base.tensors[i].name;
    if (name != "output.weight") {
        auto m = base;
        m.tensors.erase(m.tensors.begin() + i); m.offsets.erase(m.offsets.begin() + i);
        rejects("missing " + name, [&] { construct(m); });
    }
    for (unsigned kind = 0; kind < 5; ++kind) {
        auto m = base;
        auto& shape = m.tensors[i].ne;
        if (kind == 0) --shape[0];
        if (kind == 1) shape.clear();
        if (kind == 2) shape.resize(5, 1);
        if (kind == 3) shape.push_back(2);
        if (kind == 4) shape[0] = 0;
        rejects("shape " + name, [&] { construct(m); });
    }
}

void tensor_checks() {
    const auto base = fixture();
    construct(base, true); ++checks;
    construct(fixture(true), true); ++checks;
    construct(fixture(false, 0, true), true); ++checks;
    rejects("null backend", [&] { infer::Model model(infer::gguf_weights(base), backend::BackendPtr{}); });
    for (size_t i = 0; i < base.tensors.size(); ++i) {
        const auto name = base.tensors[i].name;
        missing_and_shapes(base, i);
        if (base.tensors[i].ne.size() == 2) {
            auto m = base; m.tensors[i].ne.pop_back();
            rejects("matrix rank one " + name, [&] { construct(m); });
            m = base; --m.tensors[i].ne[1];
            rejects("matrix output width " + name, [&] { construct(m); });
        }
        for (size_t rank = base.tensors[i].ne.size() + 1; rank <= 4; ++rank) {
            auto m = base; m.tensors[i].ne.resize(rank, 1);
            construct(m); ++checks;
        }
    }
    for (uint32_t type : {2u, 3u, 8u, 12u, 13u, 14u}) {
        auto m = fixture(false, type); construct(m); ++checks;
        m.tensors[1].type = type;
        rejects("quantized norm", [&] { construct(m); });
        m = fixture(false, type); --m.tensors[0].ne[0];
        rejects("partial quantized row", [&] { construct(m); });
    }
    auto m = base; m.tensors.push_back(m.tensors[0]); m.offsets.push_back(m.offsets[0]);
    rejects("duplicate tensor", [&] { construct(m); });
    // A reader's views reach the model without gguf_weights' checks, so the model refuses a duplicate name itself.
    auto views = infer::gguf_weights(base); views.tensors.push_back(views.tensors[0]);
    rejects("duplicate view", [&] {
        auto cpu = std::make_shared<backend::CpuBackend>();
        cpu->set_threads(1);
        infer::Model model(views, {cpu}, infer::Placement{});
    });
    m = base; m.release_payload();
    rejects("released payload", [&] { construct(m); });
    m = base; m.offsets.pop_back();
    rejects("missing offset", [&] { construct(m); });
    m = base; m.offsets.push_back(0);
    rejects("extra offset", [&] { construct(m); });
    for (size_t offset : {size_t(1), base.blob.size(), std::numeric_limits<size_t>::max()}) {
        m = base; m.offsets[0] = offset;
        rejects("invalid storage offset", [&] { construct(m); });
    }
    m = base; m.blob.pop_back();
    rejects("truncated blob", [&] { construct(m); });
    m = base; set(m, "qwen3.block_count", integer(2));
    rejects("missing second layer", [&] { construct(m); });
    m = base; m.tensors[0].type = 1;
    rejects("unsupported tensor type", [&] { construct(m); });
    m = base; m.tensors[0].ne[1] = uint64_t(std::numeric_limits<int>::max()) + 1;
    rejects("oversized vocabulary", [&] { construct(m); });
    m = base; m.tensors[0].ne[1] = 0;
    rejects("zero vocabulary", [&] { construct(m); });
    m = base; m.tensors[0].ne = {std::numeric_limits<uint64_t>::max(), 2};
    rejects("overflowing tensor extent", [&] { construct(m); });
    m = base; m.tensors.push_back({"unused", {8}, 1, 0}); m.offsets.push_back(0);
    rejects("unsupported unused tensor", [&] { construct(m); });
    m = base; m.tensors.push_back({"unused", {8}, 0, 0}); m.offsets.push_back(m.offsets[1]);
    construct(m); ++checks;
    m = base; m.tensors.push_back({"unused.scalar", {}, 0, 0}); m.offsets.push_back(0);
    construct(m); ++checks;
    m = base; m.tensors.push_back({"unused.empty", {0}, 0, 0}); m.offsets.push_back(m.blob.size());
    construct(m); ++checks;
    m = base; std::swap(m.offsets[1], m.offsets[2]);
    construct(m); ++checks;
}

// The qwen3moe keys, read under the qwen3moe prefix: the expert counts and width, and the routing forms another architecture uses, refused.
// A layer with a router is routed, so a dense architecture refuses one, and a file without the dense width refuses a layer without one.
void moe_metadata_checks() {
    const auto base = routed_fixture();
    construct(base); ++checks;
    for (const char* key : {"block_count", "embedding_length", "attention.head_count",
                            "expert_count", "expert_used_count", "expert_feed_forward_length"}) {
        auto m = base; erase_key(m, std::string("qwen3moe.") + key);
        rejects(std::string("missing qwen3moe.") + key, [&] { infer::gguf_weights(m); });
    }
    for (const char* key : {"expert_count", "expert_used_count", "expert_feed_forward_length"})
        for (const auto& value : {integer(0), text("2")}) {
            auto m = base; set(m, std::string("qwen3moe.") + key, value);
            rejects(std::string("invalid integer qwen3moe.") + key, [&] { infer::gguf_weights(m); });
        }
    auto m = base; set(m, "qwen3moe.expert_used_count", integer(3));
    rejects("more experts a token than the layer has", [&] { infer::gguf_weights(m); });
    m = base; set(m, "qwen3moe.expert_count", integer(300)); set(m, "qwen3moe.expert_used_count", integer(257));
    rejects("more than 256 experts a token", [&] { infer::gguf_weights(m); });
    for (const bool norm : {false, true}) {
        gguf::MetaValue v; v.vtype = gguf::V_BOOL; v.b = norm;
        m = base; set(m, "qwen3moe.expert_weights_norm", v);
        require(infer::qwen3::read_config(m, "qwen3moe.", true).expert_norm == norm, "expert_weights_norm not read");
        ++checks;
    }
    m = base; set(m, "qwen3moe.expert_weights_norm", integer(1));
    rejects("invalid qwen3moe.expert_weights_norm", [&] { infer::gguf_weights(m); });
    m = base; set(m, "qwen3moe.expert_gating_func", integer(1)); construct(m); ++checks;
    for (const auto& value : {integer(2), integer(1, gguf::V_INT32), text("softmax")}) {
        m = base; set(m, "qwen3moe.expert_gating_func", value);
        rejects("invalid qwen3moe.expert_gating_func", [&] { infer::gguf_weights(m); });
    }
    for (const char* key : {"qwen3moe.expert_shared_count", "qwen3moe.expert_shared_feed_forward_length"}) {
        m = base; set(m, key, integer(1));
        rejects(std::string("present ") + key, [&] { infer::gguf_weights(m); });
    }
    m = base; set(m, "qwen3moe.expert_weights_scale", real(1)); construct(m); ++checks;
    for (const auto& value : {real(2), real(0.5), text("1")}) {
        m = base; set(m, "qwen3moe.expert_weights_scale", value);
        rejects("invalid qwen3moe.expert_weights_scale", [&] { infer::gguf_weights(m); });
    }
    m = routed_only(); erase_key(m, "qwen3moe.feed_forward_length"); construct(m, true); ++checks;
    m = base; erase_key(m, "qwen3moe.feed_forward_length"); erase_tensor(m, "blk.0.ffn_gate_inp.weight");
    rejects("dense layer without a feed-forward width", [&] { construct(m); });
    m = fixture(); add(m, "blk.0.ffn_gate_inp.weight", {8, 2}, 0);
    rejects("router in a dense architecture", [&] { construct(m); });
}

// The routed roles one by one, on a model that reads every tensor it holds: the router is a matrix with trailing axes of one, and each expert stack is exactly [in, out, experts].
void routed_tensor_checks() {
    const auto base = routed_only();
    construct(base, true); ++checks;
    for (size_t i = 0; i < base.tensors.size(); ++i) {
        const auto name = base.tensors[i].name;
        const bool stack = name.find("_exps.") != std::string::npos;
        if (!stack && name != "blk.0.ffn_gate_inp.weight") continue;
        missing_and_shapes(base, i);
        auto m = base; --m.tensors[i].ne[1];
        rejects("output width " + name, [&] { construct(m); });
        m = base; m.tensors[i].ne.pop_back();
        rejects("an axis short " + name, [&] { construct(m); });
        if (stack) {
            m = base; --m.tensors[i].ne[2];
            rejects("expert count " + name, [&] { construct(m); });
            m = base; m.tensors[i].ne.push_back(1);
            rejects("a fourth axis of one " + name, [&] { construct(m); });
            continue;
        }
        for (size_t rank = 3; rank <= 4; ++rank) {
            m = base; m.tensors[i].ne.resize(rank, 1);
            construct(m); ++checks;
        }
    }
}

// A placement names a device the model has for every role and every layer, each device's attention layers are one run, and the devices' cache blocks nest.
// place_model refuses what it cannot honour: no device, a stream point without experts on the CPU, experts on the CPU beside several devices or on a model without routed layers, and shares that do not fit the devices.
void placement_checks() {
    const auto dense_file = fixture(), routed_file = routed_fixture();
    const auto three = fixture(false, 0, false, true, 3), two = fixture(false, 0, false, true, 2);
    const auto weights = infer::gguf_weights(dense_file), routed = infer::gguf_weights(routed_file);
    const auto cpu = [] {
        auto b = std::make_shared<backend::CpuBackend>();
        b->set_threads(1);
        return b;
    };
    rejects("no backend", [&] { infer::Model model(weights, {}, infer::Placement{}); });
    rejects("a null second backend", [&] { infer::Model model(weights, {cpu(), nullptr}, infer::Placement{}); });
    struct Case { const char* label; std::vector<int> attn, ffn; int embed, output; };
    for (const Case& c : std::vector<Case>{
            {"attention placed and feed-forward not", {0}, {}, 0, 0},
            {"a placement past the layers", {0, 0}, {0, 0}, 0, 0},
            {"attention on a device the model does not have", {2}, {0}, 0, 0},
            {"attention on a negative device", {-1}, {0}, 0, 0},
            {"feed-forward on a device the model does not have", {0}, {2}, 0, 0},
            {"the embedding on a device the model does not have", {0}, {0}, 2, 0},
            {"the head on a negative device", {0}, {0}, 0, -1}}) {
        infer::Placement p;
        p.mixer_device = c.attn; p.ffn_device = c.ffn; p.embed_device = c.embed; p.output_device = c.output;
        rejects(c.label, [&] { infer::Model model(weights, {cpu(), cpu()}, p); });
    }
    infer::Placement split;
    split.mixer_device = {0, 1, 0}; split.ffn_device = {0, 1, 0};
    rejects("a device's attention layers in two runs", [&] {
        infer::Model model(infer::gguf_weights(three), {cpu(), cpu()}, split);
    });
    // A backend whose cache blocks hold one token more than the CPU's, so neither size divides the other.
    struct OddBlocks : backend::CpuBackend {
        backend::KVLayout kv_layout() const override { return {backend::CpuBackend::kv_layout().block_tokens + 1}; }
    };
    auto odd = std::make_shared<OddBlocks>();
    odd->set_threads(1);
    split.mixer_device = {0, 1}; split.ffn_device = {0, 1};
    rejects("cache blocks that do not nest", [&] {
        infer::Model model(infer::gguf_weights(two), {cpu(), odd}, split);
    });
    const infer::ModelOptions options;
    rejects("place_model without a device", [&] { infer::place_model(weights, {}, infer::PlacementRequest{}, options); });
    infer::PlacementRequest request;
    request.stream_from = 4;
    rejects("a stream point without experts on the CPU", [&] { infer::place_model(routed, {cpu()}, request, options); });
    request = {}; request.histories = 2; request.history_tokens = 4;
    rejects("histories over a null backend", [&] { infer::place_model(weights, {nullptr}, request, options); });
    request = {}; request.cpu_moe = 1; request.names = {"a", "b"};
    rejects("experts on the CPU beside two devices", [&] { infer::place_model(routed, {cpu(), cpu()}, request, options); });
    request = {}; request.cpu_moe = -1; request.shares = {1}; request.names = {"a"};
    rejects("experts on the CPU with layer shares", [&] { infer::place_model(routed, {cpu()}, request, options); });
    request = {}; request.cpu_moe = 1;
    auto device = std::make_shared<LoadingBackend>();
    device->set_threads(1);
    rejects("experts on the CPU of a model without routed layers", [&] { infer::place_model(weights, {device}, request, options); });
    // A width a model's shards cannot take is refused by its name before the fit reads a member's shards: a Q8_0 feed-forward width of 96, 48 columns a member of two.
    request = {}; request.width = 2; request.names = {"a", "b"};
    const gguf::GGUFModel off_block = infer::synthetic_model({1, 64, 96, 2, 2, 32, 64, 7u});
    rejects("a tensor width refused before the fit", [&] { infer::place_model(infer::gguf_weights(off_block), {cpu(), cpu()}, request, options); });
    request = {};
    rejects("a split without device names", [&] { infer::place_model(weights, {cpu(), cpu()}, request, options); });
    request.names = {"a", "b"}; request.shares = {1, 1, 1};
    rejects("three layer shares for two devices", [&] { infer::place_model(weights, {cpu(), cpu()}, request, options); });
    request.shares = {0, 0};
    rejects("every layer share zero", [&] { infer::place_model(weights, {cpu(), cpu()}, request, options); });
    request.shares = {1, -1};
    rejects("a negative layer share", [&] { infer::place_model(weights, {cpu(), cpu()}, request, options); });
}

// A device with a missing weight type or mixed routed products, whose counters distinguish an early refusal from a failed upload.
struct TypeBackend : backend::CpuBackend {
    uint32_t refused = quant::GGML_TYPE_Q8_0;
    size_t adoptions = 0, allocations = 0, routed_calls = 0;
    bool device = false;
    bool mixed_experts = true;
    bool reads_in_place() const override { return !device; }
    bool implements(backend::Op op) const override { return op != backend::Op::mixed_experts || mixed_experts; }
    void matmul_experts(std::initializer_list<backend::Projection> projections, backend::CSlice x, size_t nin,
                        size_t rows, const Routing& routing, backend::RowRuns runs = {}, backend::Dtype dtype = backend::Dtype::f16) override {
        for (const auto& p : projections) require(supports_type(p.type), "a routed product reached an unsupported backend");
        for (const auto& p : projections)
            require(mixed_experts || p.type == projections.begin()->type, "mixed routed products reached an unsupported backend");
        ++routed_calls;
        backend::CpuBackend::matmul_experts(projections, x, nin, rows, routing, runs, dtype);
    }
    bool supports_type(uint32_t type) const override { return type != refused && quant::Registry::instance().get(type); }
    backend::BufferPtr adopt(const void* data, size_t bytes) override {
        ++adoptions;
        return backend::CpuBackend::adopt(data, bytes);
    }
    backend::BufferPtr alloc(size_t bytes, backend::Memory where) override {
        ++allocations;
        return backend::CpuBackend::alloc(bytes, where);
    }
};

void type_capability_checks() {
    const auto file = fixture(true, quant::GGML_TYPE_Q8_0, false, true, 2);
    const auto weights = infer::gguf_weights(file);
    for (const int part : {0, 1, 2}) {
        auto home = std::make_shared<TypeBackend>(), limited = std::make_shared<TypeBackend>();
        home->refused = UINT32_MAX;
        home->set_threads(1); limited->set_threads(1);
        infer::Placement place;
        place.mixer_device = {0, part == 2 ? 1 : 0};
        place.ffn_device = {0, 0};
        place.embed_device = part == 0 ? 1 : 0;
        place.output_device = part == 1 ? 1 : 0;
        std::string error;
        try { infer::Model model(weights, {home, limited}, place); }
        catch (const std::runtime_error& e) { error = e.what(); }
        const std::string role = part == 0 ? "embedding" : part == 1 ? "head" : "layer 1's mixer";
        require(error.find(role) != std::string::npos && error.find("type Q8_0 (8)") != std::string::npos &&
                    error.find("device 1") != std::string::npos,
                "missing early type refusal for " + role + ": " + error);
        require(!home->adoptions && !limited->adoptions && !home->allocations && !limited->allocations,
                "unsupported " + role + " reached adoption or allocation");
        refusals.push_back("unsupported " + role + ": " + error);
        ++checks;
    }
}

// The first layer has a missing type or mixed routed products; a separately typed down projection must remain eligible for streaming.
void type_stream_checks() {
    for (const int defect : {0, 1, 2, 3}) {
        const bool copied_role = defect == 1, mixed = defect >= 2, down_only = defect == 3;
        auto file = fixture(false, quant::GGML_TYPE_Q8_0, false, true, 2);
        for (auto& kv : file.kv) kv.first.replace(0, 5, "qwen3moe");
        set(file, "general.architecture", text("qwen3moe"));
        set(file, "qwen3moe.expert_count", integer(2));
        set(file, "qwen3moe.expert_used_count", integer(1));
        set(file, "qwen3moe.expert_feed_forward_length", integer(256));
        for (int l = 0; l < 2; ++l) {
            const std::string pre = "blk." + std::to_string(l) + ".";
            add(file, pre + "ffn_gate_inp.weight", {256, 2}, l == 0 && copied_role ? 2 : 0);
            for (const char* name : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
                const bool different = !mixed || std::string(name) == (down_only ? "ffn_down_exps.weight" : "ffn_gate_exps.weight");
                add(file, pre + name, {256, 256, 2}, l == 0 && !copied_role && different ? 2 : 8);
            }
        }
        for (size_t t = 0; t < file.tensors.size(); ++t) {
            const auto& info = file.tensors[t];
            if (info.name.find("norm") != std::string::npos) continue;
            size_t n = 1;
            for (uint64_t d : info.ne) n *= size_t(d);
            std::vector<float> values(n);
            for (size_t i = 0; i < n; ++i) values[i] = float(int((i * 17 + t * 3) % 29) - 14) / 256.0f;
            uint8_t* dst = file.blob.data() + file.offsets[t];
            if (!info.type) std::memcpy(dst, values.data(), n * sizeof(float));
            else {
                const auto* qt = quant::Registry::instance().get(info.type);
                qt->quantize(values.data(), dst, n / qt->block_size);
            }
        }
        const auto weights = infer::gguf_weights(file);
        if (mixed) {
            auto limited = std::make_shared<TypeBackend>();
            limited->refused = UINT32_MAX; limited->mixed_experts = false; limited->set_threads(1);
            std::string error;
            try { infer::Model model(weights, limited); }
            catch (const std::runtime_error& e) { error = e.what(); }
            if (down_only) require(error.empty(), "a separately typed down projection was refused: " + error);
            else {
                require(error == "inference: layer 0's feed-forward part needs mixed_experts, which the backend of its device does not implement",
                        "missing early mixed expert refusal: " + error);
                require(!limited->adoptions && !limited->allocations, "mixed expert refusal reached adoption or allocation");
                refusals.push_back("unsupported mixed expert pair: " + error);
                for (const char* name : {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"}) {
                    auto missing = weights;
                    missing.tensors.erase(std::remove_if(missing.tensors.begin(), missing.tensors.end(),
                        [&](const auto& t) { return t.name == name; }), missing.tensors.end());
                    error.clear();
                    try { infer::Model model(missing, limited); }
                    catch (const std::runtime_error& e) { error = e.what(); }
                    require(error == std::string("inference: missing tensor ") + name, "mixed expert planning hid a missing tensor: " + error);
                    refusals.push_back(std::string("missing mixed pair ") + name + ": " + error);
                }
            }
            const auto plan = infer::plan_model(weights);
            for (size_t l = 0; l < plan.layers.size(); ++l) {
                const auto& ops = plan.layers[l].ops;
                const auto count = std::count_if(ops.begin(), ops.end(), [](const auto& u) {
                    return u.part == infer::Part::ffn && u.op == backend::Op::mixed_experts;
                });
                require(count == (l == 0 && !down_only ? 1 : 0), "mixed expert operation declared for the wrong projections");
            }
            ++checks;
        }
        std::vector<float> expected;
        for (const bool streaming : {false, true}) {
            auto device = std::make_shared<TypeBackend>(), host = std::make_shared<TypeBackend>();
            device->refused = mixed ? UINT32_MAX : quant::GGML_TYPE_Q4_0; device->device = true;
            device->mixed_experts = !mixed;
            host->refused = UINT32_MAX;
            device->set_threads(1); host->set_threads(1);
            infer::Placement place;
            place.mixer_device = {0, 0}; place.ffn_device = {1, 1}; place.stream_from = streaming ? 2 : 0;
            infer::Model model(weights, {device, host}, place);
            std::vector<float> logits = model.prefill({0, 1, 2});
            const size_t on_device = streaming ? (down_only ? 2u : 1u) : 0u;
            require(device->routed_calls == on_device && host->routed_calls == 2u - on_device,
                    "stream eligibility did not keep only unsupported layers on the host");
            for (uint32_t id : {3u, 4u}) {
                const auto row = model.step(id);
                logits.insert(logits.end(), row.begin(), row.end());
            }
            for (float v : logits) require(std::isfinite(v), "nonfinite stream fixture logits");
            require(std::any_of(logits.begin(), logits.end(), [](float v) { return v != 0; }), "zero stream fixture logits");
            if (!streaming) expected = logits;
            else require(logits.size() == expected.size() && std::memcmp(logits.data(), expected.data(), logits.size() * sizeof(float)) == 0,
                         "type fallback changed the prompt or follow-up logits");
            ++checks;
        }
        if (down_only) continue;
        auto device = std::make_shared<TypeBackend>(), host = std::make_shared<TypeBackend>();
        device->refused = UINT32_MAX; device->device = true;
        host->refused = mixed ? UINT32_MAX : quant::GGML_TYPE_Q4_0; host->mixed_experts = !mixed;
        device->set_threads(1); host->set_threads(1);
        infer::Placement place;
        place.mixer_device = {0, 0}; place.ffn_device = {1, 1}; place.stream_from = 2;
        std::string error;
        try { infer::Model model(weights, {device, host}, place); }
        catch (const std::runtime_error& e) { error = e.what(); }
        require(error.find("layer 0's feed-forward part") != std::string::npos &&
                    (mixed ? error.find("needs mixed_experts") != std::string::npos :
                             error.find("type Q4_0 (2)") != std::string::npos && error.find("device 1") != std::string::npos) &&
                    !device->adoptions && !host->adoptions &&
                    !device->allocations && !host->allocations, "streaming hid unsupported home weights: " + error);
        if (!mixed) refusals.push_back(std::string("unsupported host ") + (copied_role ? "router" : "expert") + " type: " + error);
        ++checks;
    }
}

// A pass whose positions reach past the context is refused, whether a prompt or a step takes it there.
void context_checks() {
    auto cpu = std::make_shared<backend::CpuBackend>();
    cpu->set_threads(1);
    const auto file = fixture();
    infer::Model model(infer::gguf_weights(file), cpu);
    rejects("a prompt past the context", [&] { model.prefill(std::vector<uint32_t>(9, 0)); });
    model.prefill(std::vector<uint32_t>(8, 0));
    rejects("a step past the context", [&] { model.step(0); });
}

// Each refusal's label and text against the list, one "label: message" a line in the order the checks provoke them.
void compare_refusals(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<std::string> expected;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        expected.push_back(line);
    }
    for (size_t i = 0; i < std::min(expected.size(), refusals.size()); ++i)
        require(expected[i] == refusals[i], "refusal " + std::to_string(i + 1) + " is\n  " + refusals[i] + "\nwhere the list has\n  " + expected[i]);
    require(expected.size() == refusals.size(), std::to_string(refusals.size()) + " refusals where the list has " + std::to_string(expected.size()));
}
}

// The loader's hook, deferring the copies (the streamed load): a copying backend gets storage for every weight while the model is built and nothing is written or adopted, so a model missing a tensor, or whose cache does not fit, fails after storage but before any weight is uploaded.
// Without deferring (the mapped load) the backend adopts each weight as the model resolves it.
void hook_checks() {
    const auto m = fixture();
    const infer::ModelWeights weights = infer::gguf_weights(m);
    // The model adopts the position tables it computes itself, whatever the hook; `tables` counts those adoptions.
    int tables = -1;
    for (const bool defer : {true, false}) {
        auto device = std::make_shared<LoadingBackend>();
        device->set_threads(1);
        infer::WeightPlan plan;
        { infer::Model model(weights, {device}, infer::Placement{}, {}, infer::planning_adopt(weights, 1, plan, defer)); }
        const int n = (int)weights.tensors.size();
        if (defer) tables = device->state->adoptions;
        require(tables > 0 && (defer ? plan.uploads.size() == weights.tensors.size() && device->state->weights == n
                                     : plan.uploads.empty() && device->state->weights == 0 && device->state->adoptions == n + tables),
                defer ? "the deferring hook did not give every copied weight storage, or adopted one" : "the inline hook did not adopt every weight");
        require(device->state->writes == 0, "the hook wrote a weight while the model was built");
        ++checks;
    }
    auto missing = fixture();
    erase_tensor(missing, "blk.0.ffn_down.weight");
    const infer::ModelWeights partial = infer::gguf_weights(missing);
    for (const bool cache : {false, true}) {
        auto device = std::make_shared<LoadingBackend>();
        device->set_threads(1);
        device->fail_cache = cache;
        const infer::ModelWeights& w = cache ? weights : partial;
        std::string caught;
        infer::WeightPlan plan;
        try { infer::Model model(w, {device}, infer::Placement{}, {}, infer::planning_adopt(w, 1, plan, true)); }
        catch (const std::runtime_error& e) { caught = e.what(); }
        require(caught == (cache ? "injected cache allocation failure" : "inference: missing tensor blk.0.ffn_down.weight"),
                "a model that cannot be built lost its error");
        require(device->state->weights > 0 && (!cache || device->state->weights == (int)weights.tensors.size()) &&
                    device->state->adoptions <= tables && device->state->writes == 0,
                "a model that cannot be built uploaded a weight, or its storage was not allocated first");
        require(device->state->premature == 0, "a failed build freed storage before its backend drained");
        ++checks;
    }
}

// A tensor width a model's shards cannot take is refused naming the projection (model/shard.hpp, docs/TENSOR-SPLIT.md, section 4.2): heads, KV heads neither divided nor a multiple, K heads, vocabulary rows, columns off whole quant blocks, a state layer's saved row, and routed layers, which a group does not split yet.
void shard_checks() {
    // A two-layer dense qwen3 plan over views without bytes, F32 but for ffn_down's `down` type, with or without routed experts in its first layer.
    auto dense = [](int heads, int kv, int dim, int ff, uint64_t vocab, uint32_t down, bool routed) {
        infer::qwen3::Config c;
        c.n_layer = 2, c.n_embd = 256, c.n_ff = ff, c.n_head = heads, c.n_head_kv = kv, c.head_dim = dim, c.context_length = 64;
        if (routed) c.n_expert = 4, c.n_expert_used = 2, c.n_ff_exp = 64;
        std::vector<infer::TensorView> v;
        auto add = [&](const std::string& name, std::vector<uint64_t> shape, uint32_t type = quant::GGML_TYPE_F32) { v.push_back({name, std::move(shape), type, nullptr, 0}); };
        const uint64_t E = 256, D = uint64_t(dim), Q = uint64_t(heads) * D, KV = uint64_t(kv) * D, F = uint64_t(ff);
        add("token_embd.weight", {E, vocab});
        add("output.weight", {E, vocab});
        add("output_norm.weight", {E});
        for (int l = 0; l < 2; ++l) {
            const std::string pre = "blk." + std::to_string(l) + ".";
            for (const char* n : {"attn_norm", "ffn_norm"}) add(pre + n + ".weight", {E});
            for (const char* n : {"attn_q_norm", "attn_k_norm"}) add(pre + n + ".weight", {D});
            add(pre + "attn_q.weight", {E, Q});
            add(pre + "attn_k.weight", {E, KV});
            add(pre + "attn_v.weight", {E, KV});
            add(pre + "attn_output.weight", {Q, E});
            if (routed && l == 0) {
                add(pre + "ffn_gate_inp.weight", {E, 4});
                add(pre + "ffn_gate_exps.weight", {E, 64, 4});
                add(pre + "ffn_up_exps.weight", {E, 64, 4});
                add(pre + "ffn_down_exps.weight", {64, E, 4});
                continue;
            }
            add(pre + "ffn_gate.weight", {E, F});
            add(pre + "ffn_up.weight", {E, F});
            add(pre + "ffn_down.weight", {F, E}, down);
        }
        infer::ModelWeights w{std::make_shared<const infer::qwen3::Qwen3>(c), v};
        return std::make_pair(infer::plan_model(w), w.tensors);
    };
    const uint32_t f32 = quant::GGML_TYPE_F32;
    {
        const auto pv = dense(8, 2, 128, 1024, 48, quant::GGML_TYPE_Q4_K, false);
        const infer::ModelPlan& plan = pv.first;
        const std::vector<infer::TensorView>& views = pv.second;
        for (size_t width : {size_t(1), size_t(2), size_t(4)}) infer::shard::check_plan(plan, views, width);
        ++checks;
        rejects("tensor width heads", [&] { infer::shard::check_plan(plan, views, 3); });
    }
    {
        const auto pv = dense(12, 4, 64, 1536, 48, f32, false);
        const infer::ModelPlan& plan = pv.first;
        const std::vector<infer::TensorView>& views = pv.second;
        rejects("tensor width KV heads", [&] { infer::shard::check_plan(plan, views, 6); });
    }
    {
        const auto pv = dense(8, 8, 128, 1024, 48, quant::GGML_TYPE_Q4_K, false);
        const infer::ModelPlan& plan = pv.first;
        const std::vector<infer::TensorView>& views = pv.second;
        rejects("tensor width quant blocks", [&] { infer::shard::check_plan(plan, views, 8); });
    }
    {
        const auto pv = dense(8, 2, 128, 1024, 15, f32, false);
        const infer::ModelPlan& plan = pv.first;
        const std::vector<infer::TensorView>& views = pv.second;
        rejects("tensor width vocabulary", [&] { infer::shard::check_plan(plan, views, 2); });
    }
    {
        const auto pv = dense(8, 2, 128, 1024, 48, f32, true);
        const infer::ModelPlan& plan = pv.first;
        const std::vector<infer::TensorView>& views = pv.second;
        rejects("tensor width routed layer", [&] { infer::shard::check_plan(plan, views, 2); });
    }
    {
        // A qwen35 linear-attention layer of 2 K heads, then a full-attention layer, which 4 members cannot split by K head.
        infer::qwen35::Config c;
        c.n_layer = 2, c.n_embd = 256, c.n_ff = 1024, c.n_head = 4, c.n_head_kv = 4, c.head_dim = 128, c.rope_dim = 64, c.context_length = 64;
        c.k_heads = 2, c.v_heads = 4, c.k_dim = 64, c.v_dim = 64;
        c.full = {0, 1};
        std::vector<infer::TensorView> v;
        auto add = [&](const std::string& name, std::vector<uint64_t> shape) { v.push_back({name, std::move(shape), f32, nullptr, 0}); };
        const uint64_t E = 256, V = 4 * 64, C = 2 * 2 * 64 + V;
        add("token_embd.weight", {E, 48});
        add("output.weight", {E, 48});
        add("output_norm.weight", {E});
        for (const char* n : {"attn_norm", "post_attention_norm"}) add(std::string("blk.0.") + n + ".weight", {E});
        add("blk.0.attn_qkv.weight", {E, C});
        add("blk.0.attn_gate.weight", {E, V});
        add("blk.0.ssm_alpha.weight", {E, 4});
        add("blk.0.ssm_beta.weight", {E, 4});
        add("blk.0.ssm_conv1d.weight", {4, C});
        add("blk.0.ssm_a", {4});
        add("blk.0.ssm_dt.bias", {4});
        add("blk.0.ssm_norm.weight", {64});
        add("blk.0.ssm_out.weight", {V, E});
        for (const char* n : {"ffn_gate", "ffn_up"}) add(std::string("blk.0.") + n + ".weight", {E, 1024});
        add("blk.0.ffn_down.weight", {1024, E});
        infer::ModelWeights w{std::make_shared<const infer::qwen35::Qwen35>(c), v};
        const infer::ModelPlan plan = infer::plan_model(w);
        infer::shard::check_plan(plan, w.tensors, 2);
        ++checks;
        rejects("tensor width K heads", [&] { infer::shard::check_plan(plan, w.tensors, 4); });
        // A saved row of a state layer the width does not divide is refused naming the layer, where the spans of its roles divide.
        infer::ModelPlan odd = plan;
        odd.layers[0].saved[0].width += 1;
        rejects("tensor width saved row", [&] { infer::shard::check_plan(odd, w.tensors, 2); });
    }
}

// With a list, every refusal must have its label and text; with --write, the refusals are written as that list.
int main(int argc, char** argv) {
    try {
        const bool write = argc == 3 && std::string(argv[1]) == "--write";
        if (argc != 2 && !write) throw std::runtime_error("usage: llmx-model-validation-test REFUSALS.txt | --write REFUSALS.txt");
        type_capability_checks();
        type_stream_checks();
        metadata_checks();
        tensor_checks();
        moe_metadata_checks();
        routed_tensor_checks();
        placement_checks();
        context_checks();
        loading_lifetime_checks();
        loading_window_checks();
        reader_checks();
        fit_width_checks();
        plan_checks();
        hook_checks();
        shard_checks();
        if (write) {
            std::ofstream out(argv[2], std::ios::binary);
            for (const auto& r : refusals) out << r << '\n';
            if (!out) throw std::runtime_error(std::string("cannot write ") + argv[2]);
            std::cout << "model-validation: wrote " << refusals.size() << " refusals\n";
            return 0;
        }
        compare_refusals(argv[1]);
        std::cout << "model-validation: " << checks << " checks passed, " << refusals.size() << " refusals with their texts\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "model-validation: " << e.what() << '\n';
        return 1;
    }
}
