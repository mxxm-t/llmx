#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "inference/load.hpp"

namespace {
size_t checks = 0;

void require(bool condition, const std::string& label) {
    if (!condition) throw std::runtime_error(label);
}

template<class F> void rejects(const std::string& label, F work) {
    try { work(); }
    catch (const std::runtime_error&) { ++checks; return; }
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

gguf::GGUFModel fixture(bool tied = false, uint32_t type = 0, bool odd = false,
                        bool grouped = true) {
    gguf::GGUFModel m;
    const uint64_t emb = type ? 256 : (odd ? 7 : 8);
    const uint64_t ff = type ? 256 : 12;
    const uint64_t hd = type ? 128 : 4;
    for (const auto& kv : std::vector<std::pair<std::string, uint64_t>>{
            {"block_count", 1}, {"embedding_length", emb}, {"feed_forward_length", ff},
            {"attention.head_count", 2}, {"attention.head_count_kv", grouped ? 1u : 2u},
            {"attention.key_length", hd}, {"context_length", 8}})
        m.kv.push_back({"qwen3." + kv.first, integer(kv.second)});
    add(m, "token_embd.weight", {emb, 5}, type);
    add(m, "output_norm.weight", {emb}, 0, true);
    for (const char* name : {"attn_norm", "ffn_norm"})
        add(m, std::string("blk.0.") + name + ".weight", {emb}, 0, true);
    for (const char* name : {"attn_q_norm", "attn_k_norm"})
        add(m, std::string("blk.0.") + name + ".weight", {hd}, 0, true);
    add(m, "blk.0.attn_q.weight", {emb, 2 * hd}, type);
    add(m, "blk.0.attn_k.weight", {emb, hd * (grouped ? 1 : 2)}, type);
    add(m, "blk.0.attn_v.weight", {emb, hd * (grouped ? 1 : 2)}, type);
    add(m, "blk.0.attn_output.weight", {2 * hd, emb}, type);
    add(m, "blk.0.ffn_gate.weight", {emb, ff}, type);
    add(m, "blk.0.ffn_up.weight", {emb, ff}, type);
    add(m, "blk.0.ffn_down.weight", {ff, emb}, type);
    if (!tied) add(m, "output.weight", {emb, 5}, type);
    return m;
}

// Hundreds of models are built here, so each one's backend must start no worker threads: its pool would start only for work, and one thread needs none.
void construct(const gguf::GGUFModel& m, bool step = false) {
    auto cpu = std::make_shared<backend::CpuBackend>();
    cpu->set_threads(1);
    infer::Model model(m, cpu);
    if (step) {
        const auto logits = model.step(0);
        require(logits.size() == 5, "valid fixture vocabulary changed");
        for (float v : logits) require(v == 0.0f, "zero-weight fixture produced nonzero logits");
    }
    require(cpu->workers_started() == 0, "a one-thread model started worker threads");
}


struct LoadingState {
    int adoptions = 0, allocations = 0, drains = 0, releases = 0, premature = 0;
    bool pending = false;
};

// Model loading sees a device-like buffer whose outstanding upload ends only at wait or sync.
struct LoadingBuffer : backend::Buffer {
    backend::BufferPtr storage;
    std::shared_ptr<LoadingState> state;
    LoadingBuffer(backend::BufferPtr b, std::shared_ptr<LoadingState> s)
        : storage(std::move(b)), state(std::move(s)) {}
    ~LoadingBuffer() override {
        ++state->releases;
        if (state->pending) ++state->premature;
    }
    size_t size() const override { return storage->size(); }
    const void* host_ptr() const override { return nullptr; }
};

struct LoadingBackend : backend::CpuBackend {
    std::shared_ptr<LoadingState> state = std::make_shared<LoadingState>();
    int fail_adopt = 0, fail_alloc = 0;
    bool fail_cache = false;
    // It plays a device, whose buffers hide their host bytes, so it copies what it adopts.
    bool reads_in_place() const override { return false; }
    backend::BufferPtr adopt(const void* src, size_t bytes) override {
        if (++state->adoptions == fail_adopt) throw std::runtime_error("injected adoption failure");
        auto storage = backend::CpuBackend::alloc(bytes, backend::Memory::device);
        backend::CpuBackend::write(*storage, 0, src, bytes);
        auto buffer = std::make_shared<LoadingBuffer>(std::move(storage), state);
        state->pending = true;
        return buffer;
    }
    backend::BufferPtr alloc(size_t bytes, backend::Memory where) override {
        if (++state->allocations == fail_alloc) throw std::runtime_error("injected window allocation failure");
        auto buffer = std::make_shared<LoadingBuffer>(backend::CpuBackend::alloc(bytes, where), state);
        state->pending = true;
        return buffer;
    }
    void sync() noexcept override { ++state->drains; state->pending = false; }
    void wait(backend::Ticket) noexcept override { sync(); }
    std::unique_ptr<backend::KVStorage> kv_alloc(size_t layers, size_t heads, size_t dim, size_t tokens,
                                               backend::KVType k, backend::KVType v) override {
        if (fail_cache) throw std::runtime_error("injected cache allocation failure");
        return backend::CpuBackend::kv_alloc(layers, heads, dim, tokens, k, v);
    }
};

void loading_lifetime_checks() {
    for (int failure = 0; failure < 4; ++failure) {
        auto m = fixture();
        auto b = std::make_shared<LoadingBackend>();
        b->set_threads(1);
        std::string expected;
        if (failure == 0) {
            auto it = std::find_if(m.tensors.begin(), m.tensors.end(), [](const auto& t) {
                return t.name == "blk.0.ffn_down.weight";
            });
            const auto i = it - m.tensors.begin();
            m.tensors.erase(it); m.offsets.erase(m.offsets.begin() + i);
            expected = "inference: missing tensor blk.0.ffn_down.weight";
        } else if (failure == 1 || failure == 3) {
            b->fail_adopt = failure == 1 ? 4 : 16;
            expected = "injected adoption failure";
        } else {
            b->fail_cache = true;
            expected = "injected cache allocation failure";
        }
        std::string caught;
        try { infer::Model model(m, b); }
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
    placement.attn_device = {0}; placement.ffn_device = {1};
    placement.embed_device = 0; placement.output_device = 1;
    b->fail_adopt = 4;
    rejects("split loading failure", [&] { infer::Model model(m, {a, b, unused}, placement); });
    for (const auto& device : {a, b})
        require(device->state->releases > 0 && device->state->premature == 0 && device->state->drains > 0,
                "split loading failure did not drain each used backend before release");
    require(unused->state->drains == 0, "loading failure drained an unused backend");
    b->fail_adopt = 0;
    const int drained = a->state->drains + b->state->drains;
    {
        infer::Model model(m, {a, b, unused}, placement);
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

// The fit counts a pass's activation rows as the model allocates them: a layer with a router is routed, so the dense ffn_gate it also carries does not widen the feed-forward slots.
void fit_width_checks() {
    const auto m = routed_fixture(6);
    size_t routed = 0;
    for (size_t w : infer::slot_widths(infer::load_config(m), false)) routed += w * sizeof(float);
    require(infer::footprint(infer::gguf_weights(m), infer::ModelOptions{}).activations_per_row == routed,
            "the fit widens a routed layer's slots for the dense ffn_gate it also carries");
    infer::Model model(m);
    ++checks;
}

// Layer 0's attention on a device and its experts on a host, streamed to the device from one new token.
infer::Placement streamed_placement() {
    infer::Placement placement;
    placement.attn_device = {0}; placement.ffn_device = {1};
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
        try { infer::Model model(m, {device, host}, placement); }
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

// The model puts each weight on a backend through the loader's hook, which records the tensors a host reads in place (infer::recording_adopt).
// A copying backend reads none; beside a device, a host that runs a streamed layer's experts reads exactly those, its norm and its router, which the device also takes.
void reader_checks() {
    auto read_in_place = [](const infer::QwenWeights& weights, std::vector<backend::BackendPtr> backends, infer::Placement placement) {
        std::vector<int> takes(weights.tensors.size(), 0);
        std::vector<char> host_reads;
        const infer::AdoptWeight record = infer::recording_adopt(weights, host_reads);
        const infer::AdoptWeight adopt = [&](size_t i, backend::Backend& b) {
            ++takes[i];
            return record(i, b);
        };
        { infer::Model model(weights, std::move(backends), std::move(placement), {}, adopt); }
        std::vector<std::string> host;
        for (size_t i = 0; i < host_reads.size(); ++i)
            if (host_reads[i]) host.push_back(weights.tensors[i].name);
        std::sort(host.begin(), host.end());
        return std::make_pair(host, takes);
    };
    const auto dense = fixture();
    auto device = std::make_shared<LoadingBackend>();
    device->set_threads(1);
    const auto copied = read_in_place(infer::gguf_weights(dense), {device}, infer::Placement{});
    require(copied.first.empty(), "a copying backend read a weight in place");
    for (int n : copied.second) require(n == 1, "a dense model did not take every tensor once through the hook");
    require(device->workers_started() == 0, "a one-thread model started worker threads");
    const auto routed = routed_fixture();
    const infer::QwenWeights weights = infer::gguf_weights(routed);
    auto host = std::make_shared<backend::CpuBackend>();
    device = std::make_shared<LoadingBackend>();
    device->set_threads(1); host->set_threads(1);
    const auto seen = read_in_place(weights, {device, host}, streamed_placement());
    require(device->workers_started() == 0 && host->workers_started() == 0, "a one-thread model started worker threads");
    require(seen.first == std::vector<std::string>({"blk.0.ffn_down_exps.weight", "blk.0.ffn_gate_exps.weight",
                                                    "blk.0.ffn_gate_inp.weight", "blk.0.ffn_norm.weight",
                                                    "blk.0.ffn_up_exps.weight"}),
            "a streamed layer's host did not read exactly its norm, router and experts in place");
    // The fixture keeps its dense feed-forward matrices, which a routed layer does not take.
    for (size_t i = 0; i < weights.tensors.size(); ++i) {
        const std::string& name = weights.tensors[i].name;
        const bool both = name == "blk.0.ffn_norm.weight" || name == "blk.0.ffn_gate_inp.weight";
        const bool dense_ffn = name == "blk.0.ffn_gate.weight" || name == "blk.0.ffn_up.weight" || name == "blk.0.ffn_down.weight";
        require(seen.second[i] == (both ? 2 : dense_ffn ? 0 : 1),
                "a streamed layer's norm and router not taken by both backends, or another tensor taken other than once");
    }
    checks += 2;
}

void metadata_checks() {
    const auto base = fixture();
    const std::vector<std::string> required = {"block_count", "embedding_length",
        "feed_forward_length", "attention.head_count"};
    const std::vector<std::string> optional = {"attention.head_count_kv", "attention.key_length",
        "context_length", "attention.value_length", "rope.dimension_count"};
    for (const auto& key : required) {
        auto m = base; erase_key(m, "qwen3." + key);
        rejects("missing " + key, [&] { infer::load_config(m); });
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
            rejects("invalid integer " + key, [&] { infer::load_config(m); });
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
            rejects(std::string("invalid float ") + key, [&] { infer::load_config(m); });
        }
        for (uint32_t type : {gguf::V_FLOAT32, gguf::V_FLOAT64}) {
            auto m = base; set(m, key, real(0.5, type));
            infer::load_config(m); ++checks;
        }
    }
    for (const auto& item : std::vector<std::pair<std::string, std::string>>{
            {"general.architecture", "qwen3"}, {"qwen3.rope.scaling.type", "none"},
            {"qwen3.tensor_data_layout", "reference"}}) {
        auto m = base; set(m, item.first, text(item.second)); construct(m); ++checks;
        for (const auto& value : {text("unsupported"), text(""), integer(1)}) {
            m = base; set(m, item.first, value);
            rejects("invalid " + item.first, [&] { infer::load_config(m); });
        }
    }
    for (const char* key : {"qwen3.rope.scaling.factor", "qwen3.rope.scale_linear"}) {
        for (const auto& value : std::vector<gguf::MetaValue>{real(0), real(0.5), real(2), real(1 + 1e-10),
                real(std::numeric_limits<double>::infinity()), real(std::numeric_limits<double>::quiet_NaN()),
                integer(1), text("1")}) {
            auto m = base; set(m, key, value);
            rejects(std::string("invalid ") + key, [&] { infer::load_config(m); });
        }
        for (uint32_t type : {gguf::V_FLOAT32, gguf::V_FLOAT64}) {
            auto m = base; set(m, key, real(1, type)); construct(m); ++checks;
        }
    }
    auto m = fixture(false, 0, false, false);
    for (const char* key : {"attention.head_count_kv", "attention.key_length", "context_length"})
        erase_key(m, std::string("qwen3.") + key);
    auto config = infer::load_config(m);
    require(config.n_head_kv == 2 && config.head_dim == 4 && config.context_length == 4096 &&
            config.rope_theta == 10000.0f && config.rms_eps == 1e-6f, "incorrect absent defaults");
    construct(m, true);
    ++checks;
    set(m, "qwen3.embedding_length", integer(7));
    rejects("indivisible default head width", [&] { infer::load_config(m); });
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
        rejects("incompatible geometry " + item.first, [&] { infer::load_config(m); });
    }
    m = base;
    const uint64_t limit = uint64_t(std::numeric_limits<int>::max());
    set(m, "qwen3.attention.head_count", integer(1));
    set(m, "qwen3.attention.key_length", integer(limit - 1));
    set(m, "qwen3.context_length", integer(limit));
    if (limit * (limit - 1) > std::vector<float>().max_size())
        rejects("KV float capacity", [&] { infer::load_config(m); });
    else { infer::load_config(m); ++checks; }
}

void tensor_checks() {
    const auto base = fixture();
    construct(base, true); ++checks;
    construct(fixture(true), true); ++checks;
    construct(fixture(false, 0, true), true); ++checks;
    rejects("null backend", [&] { infer::Model model(base, backend::BackendPtr{}); });
    for (size_t i = 0; i < base.tensors.size(); ++i) {
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
}

int main() {
    try {
        metadata_checks();
        tensor_checks();
        loading_lifetime_checks();
        loading_window_checks();
        reader_checks();
        fit_width_checks();
        std::cout << "model-validation: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "model-validation: " << e.what() << '\n';
        return 1;
    }
}
