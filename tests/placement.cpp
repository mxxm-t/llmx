// Placement across backends (docs/EXECUTION.md step 6): a model split over two CPU backends must produce the bytes of the same model on one, because per-role arithmetic is unchanged and only the residual stream crosses.
// Crossings are counted so they happen exactly where the placement changes and nowhere on a single device; bad placements are refused at load.
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include "model/arch_qwen.hpp"

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// A two-layer Qwen3-shaped F32 model with deterministic weights, so a layer boundary and a within-layer boundary can both be placed.
// Context of two CPU blocks so a prompt can cross one.
gguf::GGUFModel fixture() {
    gguf::GGUFModel m;
    for (const auto& kv : std::vector<std::pair<std::string, uint64_t>>{
            {"block_count", 2}, {"embedding_length", 8}, {"feed_forward_length", 12},
            {"attention.head_count", 2}, {"attention.head_count_kv", 1},
            {"attention.key_length", 4}, {"context_length", 2 * 128}}) {
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
    for (int l = 0; l < 2; ++l) {
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
    return m;
}

struct CountingCpu : backend::CpuBackend {
    int reads = 0, writes = 0, submits = 0;
    void read(const backend::Buffer& src, size_t off, void* dst, size_t bytes) override {
        ++reads;
        backend::CpuBackend::read(src, off, dst, bytes);
    }
    void write(backend::Buffer& dst, size_t off, const void* src, size_t bytes) override {
        ++writes;
        backend::CpuBackend::write(dst, off, src, bytes);
    }
    backend::Ticket submit() override { ++submits; return backend::CpuBackend::submit(); }
};

void exact(const std::vector<float>& a, const std::vector<float>& b, const char* what) {
    require(a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), what);
    for (float v : a) require(std::isfinite(v), "nonfinite logits");
}

size_t checked = 0;

// Embedding and layer 0's attention on A, layer 0's feed-forward and layer 1's attention on B, layer 1's feed-forward and the head on A: two crossings per pass, both inside a layer, and none at the layer boundary because both halves of it sit on B.
void split_matches_single() {
    const auto weights = fixture();
    auto one = std::make_shared<CountingCpu>();
    auto a = std::make_shared<CountingCpu>(), b = std::make_shared<CountingCpu>();
    for (auto& c : {one, a, b}) c->set_threads(1);
    infer::Model single(weights, one);
    infer::Placement p;
    p.attn_device = {0, 1};
    p.ffn_device = {1, 0};
    p.embed_device = 0;
    p.output_device = 0;
    infer::Model split(weights, {a, b}, p);
    single.set_ubatch(2);
    split.set_ubatch(2);

    const std::vector<uint32_t> prompt{3, 1, 4, 1, 5};
    exact(single.prefill(prompt), split.prefill(prompt), "split prefill differs from one device");
    // Three passes for five tokens at ubatch 2, two crossings each, one in each direction.
    // A and B each submit once per pass, as does the single device.
    require(a->reads == 3 && b->writes == 3 && b->reads == 3 && a->writes == 3,
            "crossings are not where the placement changes");
    require(a->submits == 3 && b->submits == 3 && one->submits == 3, "one submission per device per pass");
    require(one->reads == 0 && one->writes == 0, "a single device crossed");
    for (int t : {9, 2, 6}) exact(single.step(t), split.step(t), "split step differs from one device");
    require(split.n_tokens() == 8 && split.kv_used_bytes() == single.kv_used_bytes(),
            "split history differs");
    // Storage is per device and only for the layers it runs: two storages of one layer each back the same bytes as one storage of two.
    require(split.kv_allocated_bytes() == single.kv_allocated_bytes(),
            "split storage differs from one device");
    checked += 4;

    // A history crossing a block boundary and a reset on both storages.
    std::vector<uint32_t> fill(130);
    for (size_t i = 0; i < fill.size(); ++i) fill[i] = (uint32_t)(1 + i % 15);
    exact(single.prefill(fill), split.prefill(fill), "long prefill differs across a block edge");
    single.reset();
    split.reset();
    require(split.n_tokens() == 0, "reset left tokens");
    exact(single.step(7), split.step(7), "step after reset differs");
    checked += 2;

    // Two sequences in one pass across the split, against the same two on one device.
    // The budget is two blocks, so the default sequences give theirs back first.
    single.reset();
    split.reset();
    infer::Sequence s1 = split.make_sequence(), s2 = split.make_sequence();
    infer::Sequence t1 = single.make_sequence(), t2 = single.make_sequence();
    infer::ExecContext cs, ct;
    const uint32_t x1[2] = {5, 6}, x2[1] = {8};
    const infer::BatchEntry es[2] = {{&s1, x1, 2, true}, {&s2, x2, 1, true}};
    const infer::BatchEntry et[2] = {{&t1, x1, 2, true}, {&t2, x2, 1, true}};
    split.forward(cs, es, 2);
    single.forward(ct, et, 2);
    for (size_t r = 0; r < 2; ++r)
        for (size_t i = 0; i < 16; ++i)
            require(cs.logits(r)[i] == ct.logits(r)[i], "batched pass differs across the split");
    require(s1.length() == 2 && s2.length() == 1, "batched pass did not commit");
    checked += 1;
}

void bad_placements_refused() {
    const auto weights = fixture();
    auto a = std::make_shared<backend::CpuBackend>(), b = std::make_shared<backend::CpuBackend>();
    auto rejects = [&](infer::Placement p, const char* what) {
        bool caught = false;
        try { infer::Model m(weights, {a, b}, p); } catch (const std::runtime_error&) { caught = true; }
        require(caught, what);
        ++checked;
    };
    infer::Placement p;
    p.attn_device = {0};
    p.ffn_device = {0, 0};
    rejects(p, "placement short of a layer accepted");
    p.attn_device = {0, 2};
    rejects(p, "placement naming a missing device accepted");
    p.attn_device = {0, 0};
    p.output_device = -1;
    rejects(p, "negative device accepted");
    bool caught = false;
    try { infer::Model m(weights, {a, nullptr}, infer::Placement{}); }
    catch (const std::runtime_error&) { caught = true; }
    require(caught, "null backend accepted");
    ++checked;
    // A sequence of another model, even one of the same shape, is refused by forward and by reset: its block ids belong to the other pool.
    infer::Model m1(weights, a), m2(weights, b);
    infer::Sequence s = m1.make_sequence();
    infer::ExecContext ctx;
    const uint32_t id = 1;
    const infer::BatchEntry e{&s, &id, 1, true};
    caught = false;
    try { m2.forward(ctx, &e, 1); } catch (const std::runtime_error&) { caught = true; }
    require(caught && s.length() == 0, "a sequence of another model was accepted");
    caught = false;
    try { m2.reset(s); } catch (const std::runtime_error&) { caught = true; }
    require(caught, "reset accepted a sequence of another model");
    m1.forward(ctx, &e, 1);
    require(s.length() == 1, "the owning model refused its sequence");
    ++checked;
}
}

int main() {
    try {
        quant::register_builtins();
        split_matches_single();
        bad_placements_refused();
        std::cout << "placement: " << checked << " checks across two CPU backends\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
