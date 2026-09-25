// Placement across backends (docs/EXECUTION.md step 6): a model split over two CPU backends must produce the bytes of the same model on one, because per-role arithmetic is unchanged and only the residual stream crosses.
// Crossings are counted so they happen exactly where the placement changes and nowhere on a single device; bad placements are refused at load.
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include "model/arch_qwen.hpp"
#include "model/layer_split.hpp"
#include "tiny_qwen.hpp"

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// Two layers, so a layer boundary and a within-layer boundary can both be placed, and a context of two CPU blocks so a prompt can cross one.
gguf::GGUFModel fixture() { return tiny_qwen(2, 2 * 128, true); }

struct CountingCpu : backend::CpuBackend {
    int copies = 0, writes = 0, submits = 0;
    void copy(backend::Buffer& dst, size_t dst_off, const backend::Buffer& src, size_t src_off, size_t bytes) override {
        ++copies;
        backend::CpuBackend::copy(dst, dst_off, src, src_off, bytes);
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
    // Three passes for five tokens at ubatch 2, two crossings each, one in each direction, each a copy out of the source and a write into the destination.
    // The residual ends layer 0 on B, where layer 1's attention runs, so the stage boundary crosses nothing.
    // Each stage submits the devices it records on, both here, and a crossing its source: three submissions a pass on A and on B, one on the single device.
    require(a->copies == 3 && b->writes == 3 && b->copies == 3 && a->writes == 3,
            "crossings are not where the placement changes");
    require(a->submits == 9 && b->submits == 9 && one->submits == 3, "submissions are not one per stage and crossing");
    require(one->copies == 0 && one->writes == 0, "a single device crossed");
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

// The layer split fitted to device budgets (model/layer_split.hpp): even shares where room allows, a device without room left out, a host device given only what the others cannot hold and taken back when endpoint weights leave no room, tied weights counted once and the output norm always, resident copies counted, unknown and zero budgets kept apart, shares honored and refused when wrong, and the fitted placement exact against one device.
void layer_split_fits() {
    const auto weights = fixture();
    const infer::ModelOptions options;
    const size_t GiB = size_t(1) << 30, MiB = size_t(1) << 20;
    const infer::Footprint fp = infer::footprint(weights, options);
    // The fixture's two layers of equal shape, and no output.weight, so the head reads the embedding.
    size_t layer0 = 0, layer1 = 0;
    for (const auto& m : fp.layers.at(0)) layer0 += m.bytes;
    for (const auto& m : fp.layers.at(1)) layer1 += m.bytes;
    require(fp.layers.size() == 2 && layer0 == layer1 && layer0 > 0 && fp.tied && fp.output.bytes == fp.embedding.bytes &&
                fp.output_norm.bytes == 8 * sizeof(float) && fp.cache_per_layer > 0,
            "footprint does not describe the model");
    ++checked;
    // A device that copies weights keeps back the scratch a Vulkan device reports (Backend::scratch_reserve); a host keeps none.
    auto budget = [](const char* name, std::optional<size_t> bytes, bool host = false) {
        const size_t scratch = host ? 0 : ((size_t)256 << 20) + bytes.value_or(0) / 20;
        return infer::DeviceBudget{name, bytes, host, {}, 0, scratch};
    };
    auto split = [&](std::vector<infer::DeviceBudget> d, std::vector<int> shares = {}) {
        return infer::split_layers(fp, d, 2, shares);
    };
    auto fails = [&](std::vector<infer::DeviceBudget> d, std::vector<int> shares, const char* what) {
        bool caught = false;
        try { split(d, shares); } catch (const std::runtime_error&) { caught = true; }
        require(caught, what);
        ++checked;
    };

    auto even = split({budget("a", GiB), budget("b", GiB)});
    require(even.stages[0].count == 1 && even.stages[1].count == 1, "two devices with room do not share the layers");
    require(even.embed_device == 0 && even.output_device == 1, "embedding and head not on the first and last stages");
    const infer::Placement placed = infer::placement_for(even);
    require(placed.attn_device == std::vector<int>({0, 1}) && placed.ffn_device == std::vector<int>({0, 1}) &&
                placed.embed_device == 0 && placed.output_device == 1,
            "a layer's attention and feed-forward block on different devices");
    ++checked;

    // 100 MiB is below a device's reserve, so it takes nothing and the other carries the embedding and head too.
    auto one = split({budget("small", GiB / 10), budget("b", GiB)});
    require(one.stages[0].count == 0 && one.stages[1].count == 2 && one.embed_device == 1 && one.output_device == 1,
            "a device without room was given layers");
    ++checked;

    auto host = split({budget("cpu", GiB, true), budget("gpu", GiB)});
    require(host.stages[0].count == 0 && host.stages[1].count == 2, "a host device took layers another device could hold");
    // No reading is a device that cannot tell, which is not checked; a reading of zero is a full device.
    auto unknown = split({budget("a", std::nullopt), budget("b", std::nullopt)});
    require(unknown.stages[0].count == 1 && unknown.stages[1].count == 1, "devices of unknown room not shared evenly");
    checked += 2;
    fails({budget("a", 0), budget("b", 0)}, {}, "devices reporting no free memory were given layers");

    // Two 100 MiB layers, an untied 300 MiB embedding and head, 1 MiB of cache a layer, a CPU and a device of 1 GiB each.
    // With every layer on the device, the embedding joins the head there and one layer no longer fits; the CPU takes it back as its first stage.
    infer::Footprint heavy;
    heavy.layers.assign(2, {infer::Matrix{8, 4096, 1, 100 * MiB}});
    heavy.embedding = infer::Matrix{8, 4096, 1, 300 * MiB};
    heavy.output = infer::Matrix{8, 4096, 1, 300 * MiB};
    heavy.output_norm = infer::Matrix{0, 4096, 1, 16384};
    heavy.cache_per_layer = MiB;
    auto back = infer::split_layers(heavy, {budget("cpu", GiB, true), budget("gpu", GiB)}, 1);
    require(back.stages[0].count == 1 && back.stages[1].count == 1 && back.embed_device == 0 && back.output_device == 1,
            "a host device dropped from the fit did not take the layer the device could not hold");
    // A tied head on the embedding's device is kept once, and its norm is still counted.
    heavy.tied = true;
    heavy.output = heavy.embedding;
    auto tied = infer::split_layers(heavy, {budget("gpu", 4 * GiB)}, 1);
    require(tied.stages[0].weights == 2 * 100 * MiB + 300 * MiB + 16384, "tied embedding and head not counted as one buffer and a norm");
    // What a backend keeps beside a matrix counts: with a copy of every weight, 600 MiB holds one layer where it held both.
    infer::DeviceBudget copying = budget("gpu", 600 * MiB);
    copying.resident = [](const infer::Matrix& m) { return 2 * m.bytes; };
    heavy.tied = false;
    heavy.embedding.bytes = heavy.output.bytes = MiB;
    auto plain = infer::split_layers(heavy, {budget("gpu", 600 * MiB), budget("cpu", GiB, true)}, 1);
    auto doubled = infer::split_layers(heavy, {copying, budget("cpu", GiB, true)}, 1);
    require(plain.stages[0].count == 2 && doubled.stages[0].count < 2 && doubled.stages[1].count > 0,
            "a backend's resident copies were not counted");
    checked += 3;

    // Layers of unequal size are placed by their own sizes: a 400 MiB layer and three of 10 MiB fit devices of 800 and 500 MiB, the large one alone on the first.
    infer::Footprint uneven;
    for (size_t mib : {400, 10, 10, 10}) uneven.layers.push_back({infer::Matrix{8, 4096, 1, mib * MiB, true}});
    uneven.embedding = uneven.output = infer::Matrix{8, 4096, 1, MiB, true};
    uneven.cache_per_layer = MiB;
    auto sized = infer::split_layers(uneven, {budget("a", 800 * MiB), budget("b", 500 * MiB)}, 1);
    require(sized.stages[0].count >= 1 && sized.stages[0].count + sized.stages[1].count == 4, "layers of unequal size did not fit by their sizes");
    // Three equal devices share three equal layers one each.
    infer::Footprint three;
    three.layers.assign(3, {infer::Matrix{8, 4096, 1, 10 * MiB, true}});
    three.embedding = three.output = infer::Matrix{8, 4096, 1, MiB, true};
    auto thirds = infer::split_layers(three, {budget("a", GiB), budget("b", GiB), budget("c", GiB)}, 1);
    require(thirds.stages[0].count == 1 && thirds.stages[1].count == 1 && thirds.stages[2].count == 1,
            "three equal devices did not share three equal layers");
    // Every layer costs the same, so the busiest device runs as few as fit: six 10 MiB layers beside a 40 MiB embedding and a 40 MiB head go two a device, where balancing bytes would give the middle device four.
    infer::Footprint ends;
    ends.layers.assign(6, {infer::Matrix{8, 4096, 1, 10 * MiB, true}});
    ends.embedding = ends.output = infer::Matrix{8, 4096, 1, 40 * MiB, true};
    auto counted = infer::split_layers(ends, {budget("a", GiB), budget("b", GiB), budget("c", GiB)}, 1);
    require(counted.stages[0].count == 2 && counted.stages[1].count == 2 && counted.stages[2].count == 2,
            "the fit balanced bytes rather than layers");
    checked += 3;

    // A tied head on the embedding's device is charged as the head keeps it: a backend's copy of a product matrix counts once, beside the shared buffer.
    infer::Footprint tied_f32;
    tied_f32.layers.assign(2, {infer::Matrix{0, 4096, 1, 10 * MiB, true}});
    tied_f32.embedding = infer::Matrix{0, 4096, 1000, 16 * MiB, false};
    tied_f32.output = tied_f32.embedding;
    tied_f32.output.product = true;
    tied_f32.tied = true;
    infer::DeviceBudget padding = budget("gpu", 4 * GiB);
    padding.resident = [](const infer::Matrix& m) { return m.product ? 2 * m.bytes : m.bytes; };
    auto tied_padded = infer::split_layers(tied_f32, {padding}, 1);
    require(tied_padded.stages[0].weights == 2 * 2 * 10 * MiB + 2 * 16 * MiB, "a tied head's copy on the embedding's device not counted");
    // The host's logits and staging fit the host: refused beside GPUs alone when the host has too little, carried by a CPU that runs layers.
    infer::Footprint logits = three;
    logits.logits_per_row = 64 * MiB;
    bool refused = false;
    try { infer::split_layers(logits, {budget("a", GiB), budget("b", GiB)}, 4, {}, 128 * MiB); } catch (const std::runtime_error&) { refused = true; }
    auto carried = infer::split_layers(logits, {budget("cpu", 2 * GiB, true), budget("a", GiB)}, 4, {1, 2}, 128 * MiB);
    require(refused && carried.host == 4 * 64 * MiB && carried.stages[0].other >= carried.host, "the host's logits not fitted to the host");
    checked += 2;
    // The host keeps the position tables, two handoff buffers per used device, and each used backend's staging, never an unused one's.
    // Two devices staging 68 MiB each beside 64 MiB of tables need 204 MiB of the host: with 140 MiB free one device runs every layer, and with 100 MiB none can.
    infer::Footprint staged = three;
    staged.tables = 64 * MiB;
    staged.handoff_per_row = MiB;
    auto device = [&](const char* name) {
        infer::DeviceBudget d = budget(name, GiB);
        d.host_side = 68 * MiB;
        return d;
    };
    auto alone = infer::split_layers(staged, {device("a"), device("b")}, 1, {}, 140 * MiB);
    require(alone.stages[0].count + alone.stages[1].count == 3 && (alone.stages[0].count == 0 || alone.stages[1].count == 0) &&
                alone.host == 64 * MiB + 68 * MiB,
            "a device was used whose staging the host cannot hold");
    bool short_host = false;
    try { infer::split_layers(staged, {device("a"), device("b")}, 1, {}, 100 * MiB); } catch (const std::runtime_error&) { short_host = true; }
    auto both = infer::split_layers(staged, {device("a"), device("b")}, 1, {}, GiB);
    require(short_host && both.stages[0].count && both.stages[1].count && both.host == 64 * MiB + 2 * 68 * MiB + 4 * MiB,
            "the host's tables, handoff and staging not counted");
    // Shares that leave a device out do not charge its staging.
    auto first_only = infer::split_layers(staged, {device("a"), device("b")}, 1, {1, 0}, 140 * MiB);
    require(first_only.stages[0].count == 3 && first_only.host == 64 * MiB + 68 * MiB, "an unused device's staging charged to the host");
    // A CPU that runs layers reads the host's tables in place: counted once, in the host's needs it carries, not again as its own.
    auto on_cpu = infer::split_layers(staged, {budget("cpu", 2 * GiB, true), device("a")}, 1, {1, 2});
    require(on_cpu.host == 64 * MiB + 68 * MiB + 4 * MiB && on_cpu.stages[0].other == staged.activations_per_row + on_cpu.host,
            "the CPU's alias of the host's tables counted twice");
    checked += 4;

    // A projection with a trailing singleton axis is the same product weight: the footprint follows the tensor's role, not its rank.
    auto singleton = weights;
    for (auto& t : singleton.tensors)
        if (t.name.compare(0, 4, "blk.") == 0 && t.ne.size() == 2) t.ne.push_back(1);
    const infer::Footprint fs = infer::footprint(singleton, options);
    size_t products = 0, singleton_products = 0;
    for (size_t l = 0; l < fp.layers.size(); ++l)
        for (size_t i = 0; i < fp.layers[l].size(); ++i) {
            products += fp.layers[l][i].product;
            singleton_products += fs.layers[l][i].product && fs.layers[l][i].rows == fp.layers[l][i].rows;
        }
    require(products == 2 * 7 && singleton_products == products, "projection roles not recognized with a singleton axis");
    ++checked;

    auto shared = split({budget("a", GiB), budget("b", GiB)}, {2, 0});
    require(shared.stages[0].count == 2 && shared.output_device == 0, "layer shares not honored");
    ++checked;
    fails({budget("a", GiB), budget("b", GiB)}, {1}, "a share per device not required");
    fails({budget("a", GiB), budget("b", GiB)}, {0, 0}, "shares that are all zero accepted");
    // Shares are proportions: 5:3 of two layers rounds to one each, 1:0 gives both to the first.
    auto ratio = split({budget("a", GiB), budget("b", GiB)}, {5, 3});
    auto whole = split({budget("a", GiB), budget("b", GiB)}, {1, 0});
    require(ratio.stages[0].count == 1 && ratio.stages[1].count == 1 && whole.stages[0].count == 2 && whole.stages[1].count == 0,
            "layer shares not taken as proportions");
    ++checked;
    fails({budget("a", GiB / 10), budget("b", GiB / 10)}, {}, "a model with no room accepted");
    fails({budget("a", GiB / 10), budget("b", GiB)}, {1, 1}, "shares past a device's room accepted");

    auto one_cpu = std::make_shared<backend::CpuBackend>();
    auto a = std::make_shared<backend::CpuBackend>(), b = std::make_shared<backend::CpuBackend>();
    for (auto& c : {one_cpu, a, b}) c->set_threads(1);
    infer::Model single(weights, one_cpu);
    infer::Model fitted(weights, {a, b}, placed);
    const std::vector<uint32_t> prompt{2, 7, 1, 8, 2, 8};
    exact(single.prefill(prompt), fitted.prefill(prompt), "fitted split prefill differs from one device");
    for (int t : {3, 14, 1}) exact(single.step(t), fitted.step(t), "fitted split step differs from one device");
    ++checked;

    // The one placement entry (infer::place_model): budgets asked of the backends, a split by shares exact against one device, the request's ubatch applied, and experts on the CPU refused beside several devices.
    auto cpus = [] {
        std::vector<backend::BackendPtr> v{std::make_shared<backend::CpuBackend>(), std::make_shared<backend::CpuBackend>()};
        for (auto& c : v) c->set_threads(1);
        return v;
    };
    const auto asked = infer::budgets_for(cpus(), {"cpu", "cpu"});
    require(asked.size() == 2 && asked[0].host && asked[0].scratch == 0 && asked[0].resident, "a CPU's budget not asked of the backend");
    infer::PlacementRequest request;
    request.names = {"cpu", "cpu"};
    request.shares = {1, 1};
    request.ubatch = 3;
    infer::PlacedModel placed_split = infer::place_model(weights, cpus(), request, options);
    single.reset();
    exact(single.prefill(prompt), placed_split.model->prefill(prompt), "place_model split prefill differs from one device");
    require(placed_split.model->prefill_batch() == 3 && !placed_split.plan.empty(), "place_model did not apply the ubatch or describe the split");
    request.cpu_moe = 1;
    bool experts_refused = false;
    try { infer::place_model(weights, cpus(), request, options); } catch (const std::runtime_error&) { experts_refused = true; }
    require(experts_refused, "experts on the CPU accepted beside several devices");
    // A CPU reads its weights in place, so experts on the CPU beside it are the one device alone.
    infer::PlacementRequest experts_on_cpu;
    experts_on_cpu.names = {"cpu"};
    experts_on_cpu.cpu_moe = -1;
    require(infer::place_model(weights, {std::make_shared<backend::CpuBackend>()}, experts_on_cpu, options).model->prefill_batch() ==
                (size_t)infer::kDefaultUbatch,
            "experts on the CPU beside a CPU not taken as one device");
    checked += 4;
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
    // A device whose attention layers come back after another's would own a storage two stages write.
    const auto three = tiny_qwen(3, 2 * 128, true);
    infer::Placement split;
    split.attn_device = split.ffn_device = {0, 1, 0};
    bool split_refused = false;
    try { infer::Model m(three, {a, b}, split); } catch (const std::runtime_error&) { split_refused = true; }
    require(split_refused, "a device's attention layers split in two accepted");
    ++checked;
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
        layer_split_fits();
        bad_placements_refused();
        std::cout << "placement: " << checked << " checks across two CPU backends\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
