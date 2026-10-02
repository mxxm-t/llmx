// Placement across backends (docs/EXECUTION.md step 6): a model split over two, three or four CPU backends must produce the bytes of the same model on one, because per-role arithmetic is unchanged and only the residual stream crosses.
// Crossings are counted so they happen exactly where the placement changes and nowhere on a single device; bad placements are refused at load.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>
#include "model/runtime.hpp"
#include "model/place.hpp"
#include "model/arch/registry.hpp"
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

// A device that reads host memory in place, as one sharing the host's memory would, and is not the CPU.
struct HostMemoryDevice : CountingCpu {
    bool is_cpu() const override { return false; }
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
    infer::Model single(infer::gguf_weights(weights), one);
    infer::Placement p;
    p.mixer_device = {0, 1};
    p.ffn_device = {1, 0};
    p.embed_device = 0;
    p.output_device = 0;
    infer::Model split(infer::gguf_weights(weights), {a, b}, p);
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

    // The embedding alone on A and the rest on B: the residual leaves A only, so A keeps the one handoff buffer of a placement that is not pipelined and B keeps none.
    infer::Placement embed_apart;
    embed_apart.mixer_device = embed_apart.ffn_device = {1, 1};
    embed_apart.embed_device = 0;
    embed_apart.output_device = 1;
    infer::Model apart(infer::gguf_weights(weights), {a, b}, embed_apart);
    infer::Sequence sa = apart.make_sequence();
    infer::ExecContext ca;
    const infer::BatchEntry ea{&sa, x1, 2, true};
    apart.forward(ca, &ea, 1);
    require(!std::memcmp(ca.logits(0), ct.logits(0), single.n_vocab() * sizeof(float)), "the embedding apart differs from one device");
    require(ca.handoff.size() == 2 && ca.handoff[0].size() == 1 && ca.handoff[1].empty(),
            "handoff buffers not on exactly the devices a crossing leaves");
    checked += 2;
}

// The layer split fitted to device budgets (model/layer_split.hpp): even shares where room allows, a device without room left out, a host device given only what the others cannot hold and taken back when endpoint weights leave no room, tied weights counted once and the output norm always, resident copies counted, unknown and zero budgets kept apart, shares honored and refused when wrong, and the fitted placement exact against one device.
void host_scratch_fits() {
    const size_t MiB = size_t(1) << 20;
    infer::Footprint fp;
    fp.layers.resize(1);
    fp.cache = {7 * MiB};
    fp.tables = 4096;
    fp.activations_per_row = 4096;
    fp.logits_per_row = 4096;
    infer::DeviceBudget host;
    host.name = "cpu";
    host.host = true;
    host.bytes = 8 * MiB;
    host.scratch = 2 * MiB;
    bool refused = false;
    try { infer::split_layers(fp, {host}, 1); } catch (const std::runtime_error&) { refused = true; }
    require(refused, "a host fit consumed the backend's scratch reserve");
    host.bytes = 10 * MiB;
    const auto one = infer::split_layers(fp, {host}, 1);
    require(one.stages[0].other == host.scratch + 3 * 4096,
            "a host's scratch, tables, arena and logits were not each counted once");
    fp.layers.resize(2);
    fp.cache = {3 * MiB, 3 * MiB};
    host.bytes = 6 * MiB;
    const auto two = infer::split_layers(fp, {host, host}, 1);
    require(two.stages[0].count == 1 && two.stages[1].count == 1,
            "host scratch did not constrain the automatic layer placement");
    checked += 3;
}

void layer_split_fits() {
    const auto weights = fixture();
    const infer::ModelOptions options;
    const size_t GiB = size_t(1) << 30, MiB = size_t(1) << 20;
    const infer::ModelWeights views = infer::gguf_weights(weights);
    const infer::Footprint fp = infer::footprint(views, infer::plan_model(views), options);
    // The fixture's two layers of equal shape, and no output.weight, so the head reads the embedding.
    size_t layer0 = 0, layer1 = 0;
    for (const auto& m : fp.layers.at(0)) layer0 += m.bytes;
    for (const auto& m : fp.layers.at(1)) layer1 += m.bytes;
    require(fp.layers.size() == 2 && layer0 == layer1 && layer0 > 0 && fp.tied && fp.output.bytes == fp.embedding.bytes &&
                fp.output_norm.bytes == 8 * sizeof(float) && fp.cache.size() == 2 && fp.cache[0] > 0 && fp.cache[1] == fp.cache[0],
            "footprint does not describe the model");
    ++checked;
    // A device that copies weights keeps back the scratch a Vulkan device reports (Backend::scratch_reserve); a host keeps none.
    auto budget = [](const char* name, std::optional<size_t> bytes, bool host = false) {
        infer::DeviceBudget d;
        d.name = name;
        d.bytes = bytes;
        d.host = host;
        d.scratch = host ? 0 : ((size_t)256 << 20) + bytes.value_or(0) / 20;
        return d;
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
    require(placed.mixer_device == std::vector<int>({0, 1}) && placed.ffn_device == std::vector<int>({0, 1}) &&
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
    heavy.cache.assign(2, MiB);
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
    uneven.cache.assign(4, MiB);
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

    // Each layer's cache is its own, as a model whose layers keep KV and states has them: four 10 MiB layers, the last keeping 500 MiB of cache and the others none.
    // A device with room for 400 MiB cannot run that last layer, which its suffix of layers always holds, so it takes none, where caches spread evenly would give it two.
    const std::vector<infer::DeviceBudget> roomy{infer::DeviceBudget{"a", 2 * GiB, false, {}, 0, 0}, infer::DeviceBudget{"b", 400 * MiB, false, {}, 0, 0}};
    infer::Footprint caches;
    caches.layers.assign(4, {infer::Matrix{8, 4096, 1, 10 * MiB, true}});
    caches.embedding = caches.output = infer::Matrix{8, 4096, 1, MiB, true};
    caches.cache = {0, 0, 0, 500 * MiB};
    auto last_heavy = infer::split_layers(caches, roomy, 1);
    require(last_heavy.stages[0].count == 4 && last_heavy.stages[1].count == 0 && last_heavy.stages[0].cache == 500 * MiB,
            "a layer's own cache was not counted where it sits");
    // With the cache on the first layer the same devices share the layers two each, and each stage counts the cache of its own layers.
    caches.cache = {500 * MiB, 0, 0, 0};
    auto first_heavy = infer::split_layers(caches, roomy, 1);
    require(first_heavy.stages[0].count == 2 && first_heavy.stages[1].count == 2 && first_heavy.stages[0].cache == 500 * MiB &&
                first_heavy.stages[1].cache == 0,
            "a stage's cache is not the sum of its own layers' caches");
    // A cache for some layers and not others is the caller's error.
    caches.cache = {MiB, MiB, MiB};
    bool partial = false;
    try { infer::split_layers(caches, roomy, 1); } catch (const std::logic_error&) { partial = true; }
    require(partial, "a cache for three of four layers was accepted");
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
    // A caller that keeps its own count of logits rows, as a server does for its passes in flight, is fitted for that count, fewer or more than a pass's rows.
    auto fewer = infer::split_layers(logits, {budget("cpu", 2 * GiB, true), budget("a", GiB)}, 4, {1, 2}, 128 * MiB, 2, 2);
    auto more = infer::split_layers(logits, {budget("cpu", 2 * GiB, true), budget("a", GiB)}, 4, {1, 2}, 128 * MiB, 2, 8);
    require(fewer.host == 2 * 64 * MiB && more.host == 8 * 64 * MiB, "the logits rows the caller keeps not the ones fitted");
    checked += 3;
    // The host keeps the position tables, two handoff buffers on each used device but the last, which sends nothing, and each used backend's staging, never an unused one's.
    // Two devices staging 68 MiB each beside 64 MiB of tables need 202 MiB of the host: with 140 MiB free one device runs every layer, and with 100 MiB none can.
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
    require(short_host && both.stages[0].count && both.stages[1].count && both.host == 64 * MiB + 2 * 68 * MiB + 2 * MiB,
            "the host's tables, handoff and staging not counted, or the last stage's handoff counted");
    // A handoff buffer per pass slot on each device that sends, two at least: four slots over two devices hold four rows, one slot two, and over three devices eight.
    auto four_slots = infer::split_layers(staged, {device("a"), device("b")}, 1, {}, GiB, 4);
    auto one_slot = infer::split_layers(staged, {device("a"), device("b")}, 1, {}, GiB, 1);
    auto three_four = infer::split_layers(staged, {device("a"), device("b"), device("c")}, 1, {}, GiB, 4);
    require(four_slots.host == 64 * MiB + 2 * 68 * MiB + 4 * MiB && one_slot.host == both.host &&
                three_four.host == 64 * MiB + 3 * 68 * MiB + 8 * MiB,
            "the handoff buffers of the pass slots not counted on the sending devices alone");
    // Shares that leave a device out do not charge its staging.
    auto first_only = infer::split_layers(staged, {device("a"), device("b")}, 1, {1, 0}, 140 * MiB);
    require(first_only.stages[0].count == 3 && first_only.host == 64 * MiB + 68 * MiB, "an unused device's staging charged to the host");
    // A CPU that runs layers reads the host's tables in place: counted once, in the host's needs it carries, not again as its own.
    auto on_cpu = infer::split_layers(staged, {budget("cpu", 2 * GiB, true), device("a")}, 1, {1, 2});
    require(on_cpu.host == 64 * MiB + 68 * MiB + 2 * MiB && on_cpu.stages[0].other == staged.activations_per_row + on_cpu.host,
            "the CPU's alias of the host's tables counted twice");
    checked += 5;

    // A projection with a trailing singleton axis is the same product weight: the footprint follows the tensor's role, not its rank.
    auto singleton = weights;
    for (auto& t : singleton.tensors)
        if (t.name.compare(0, 4, "blk.") == 0 && t.ne.size() == 2) t.ne.push_back(1);
    const infer::ModelWeights singleton_views = infer::gguf_weights(singleton);
    const infer::Footprint fs = infer::footprint(singleton_views, infer::plan_model(singleton_views), options);
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
    infer::Model single(infer::gguf_weights(weights), one_cpu);
    infer::Model fitted(infer::gguf_weights(weights), {a, b}, placed);
    const std::vector<uint32_t> prompt{2, 7, 1, 8, 2, 8};
    exact(single.prefill(prompt), fitted.prefill(prompt), "fitted split prefill differs from one device");
    for (int t : {3, 14, 1}) exact(single.step(t), fitted.step(t), "fitted split step differs from one device");
    ++checked;

    // The one placement entry (infer::place_model): budgets asked of the backends, a split by shares exact against one device, the request's ubatch applied, experts on the CPU refused beside several devices and on a model without routed layers, and a stream point refused without experts on the CPU.
    auto cpus = [] {
        std::vector<backend::BackendPtr> v{std::make_shared<backend::CpuBackend>(), std::make_shared<backend::CpuBackend>()};
        for (auto& c : v) c->set_threads(1);
        return v;
    };
    // What place_model refuses the request with, or nothing when it places the model.
    auto refusal = [&](const gguf::GGUFModel& m, std::vector<backend::BackendPtr> backends, const infer::PlacementRequest& r) {
        try { infer::place_model(infer::gguf_weights(m), std::move(backends), r, options); }
        catch (const std::runtime_error& e) { return std::string(e.what()); }
        return std::string();
    };
    const auto asked = infer::budgets_for(cpus(), {"cpu", "cpu"});
    require(asked.size() == 2 && asked[0].host && asked[0].scratch == 0 && asked[0].resident, "a CPU's budget not asked of the backend");
    infer::PlacementRequest request;
    request.names = {"cpu", "cpu"};
    request.shares = {1, 1};
    request.ubatch = 3;
    infer::PlacedModel placed_split = infer::place_model(infer::gguf_weights(weights), cpus(), request, options);
    single.reset();
    exact(single.prefill(prompt), placed_split.model->prefill(prompt), "place_model split prefill differs from one device");
    require(placed_split.model->prefill_batch() == 3 && !placed_split.plan.empty(), "place_model did not apply the ubatch or describe the split");
    // Experts on the CPU are a placement of one device, so a routed model refuses them beside several, by the name of the flag given.
    const auto moe = tiny_qwen_moe(2, 2 * 128, true);
    for (const int cpu_moe : {1, -1}) {
        request.cpu_moe = cpu_moe;
        const std::string flag = cpu_moe < 0 ? "--cpu-moe" : "--n-cpu-moe";
        require(refusal(moe, cpus(), request).rfind(flag + ": not with several devices", 0) == 0,
                "experts on the CPU accepted beside several devices, or refused by another flag's name");
    }
    // The request's pass slots reach the fit: two CPUs reporting 1 GiB free hold two slots' handoff buffers and not 2^30 slots'.
    struct SmallCpu : backend::CpuBackend {
        std::optional<size_t> memory_available() const override { return size_t(1) << 30; }
    };
    auto small = [] {
        std::vector<backend::BackendPtr> v{std::make_shared<SmallCpu>(), std::make_shared<SmallCpu>()};
        for (auto& c : v) c->set_threads(1);
        return v;
    };
    request.cpu_moe = 0;
    request.slots = 2;
    require(infer::place_model(infer::gguf_weights(weights), small(), request, options).model->pipelined(), "two pass slots do not fit two CPUs");
    request.slots = size_t(1) << 30;
    bool slots_refused = false;
    try { infer::place_model(infer::gguf_weights(weights), small(), request, options); } catch (const std::runtime_error&) { slots_refused = true; }
    require(slots_refused, "the handoff buffers of the request's pass slots not counted by place_model");
    // A stream point has nothing to stream without experts on the CPU, so place_model refuses it for every caller, whatever the caller checked first.
    infer::PlacementRequest stream_alone;
    stream_alone.names = {"cpu"};
    stream_alone.stream_from = 1;
    require(!refusal(weights, {std::make_shared<backend::CpuBackend>()}, stream_alone).empty(), "a stream point accepted without experts on the CPU");
    // A model without routed layers has no experts to put on the CPU, so the CPU and a device that reads host memory in place refuse the flags alike, by the name of the flag given.
    for (const int cpu_moe : {1, -1}) {
        infer::PlacementRequest dense;
        dense.names = {"cpu"};
        dense.cpu_moe = cpu_moe;
        const std::string expected = std::string(cpu_moe < 0 ? "--cpu-moe" : "--n-cpu-moe") + ": the model has no expert layers";
        require(refusal(weights, {std::make_shared<backend::CpuBackend>()}, dense) == expected,
                "experts on the CPU accepted on a model without routed layers on the CPU, or refused by another flag's name");
        require(refusal(weights, {std::make_shared<HostMemoryDevice>()}, dense) == expected,
                "experts on the CPU accepted on a model without routed layers beside a device, or refused by another flag's name");
    }
    // A CPU runs the experts where they are, so experts on the CPU beside it are that one CPU: nothing crosses, and the logits are those of the model placed without the flag.
    auto cpu_counted = std::make_shared<CountingCpu>(), cpu_plain = std::make_shared<CountingCpu>();
    cpu_counted->set_threads(1);
    cpu_plain->set_threads(1);
    infer::PlacementRequest experts_on_cpu;
    experts_on_cpu.names = {"cpu"};
    experts_on_cpu.cpu_moe = -1;
    const infer::PlacedModel experts_here = infer::place_model(infer::gguf_weights(moe), {cpu_counted}, experts_on_cpu, options);
    const std::vector<float> routed = infer::Model(infer::gguf_weights(moe), cpu_plain).prefill(prompt);
    exact(routed, experts_here.model->prefill(prompt), "experts on the CPU beside a CPU differ from the model without them");
    require(cpu_counted->copies == 0 && cpu_counted->writes == 0, "experts on the CPU beside a CPU crossed to another backend");
    // Reading weights in place does not make a backend the CPU: experts on the CPU beside such a device run on a CPU placed beside it, the residual crossing each way, with the same logits.
    auto host_device = std::make_shared<HostMemoryDevice>();
    host_device->set_threads(1);
    const infer::PlacedModel experts_beside = infer::place_model(infer::gguf_weights(moe), {host_device}, experts_on_cpu, options);
    exact(routed, experts_beside.model->prefill(prompt), "experts on a CPU beside a device differ from the model on one CPU");
    require(host_device->copies > 0 && host_device->writes > 0, "experts on the CPU stayed on a device that reads in place");
    checked += 9;
}

// Three layers placed by place_model over two CPU backends at shares 1:2 and over three at 1:1:1, prompts chunked at ubatch 3.
// A 13-token prompt is five chunks, more than the stages, so the pipelined prefill reuses its pass slots and both handoff buffers of every stage but the last.
struct PipelinedSplit {
    std::vector<int> shares;
    int last_layers;   // the layers the last stage runs
};
const PipelinedSplit kPipelined[] = {{{1, 2}, 2}, {{1, 1, 1}, 1}};
const std::vector<uint32_t> kPrompt{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9};

infer::PlacedModel pipelined(const gguf::GGUFModel& weights, std::vector<backend::BackendPtr> backends, const std::vector<int>& shares) {
    infer::PlacementRequest request;
    request.names.assign(backends.size(), "cpu");
    request.shares = shares;
    request.ubatch = 3;
    for (auto& b : backends) b->set_threads(1);
    return infer::place_model(infer::gguf_weights(weights), std::move(backends), request, infer::ModelOptions{});
}

// The prompt, decode steps after it, a second prompt continuing that history, every row score() hands out, and a pass of a decoding sequence beside a fresh prompt, each exact against one backend.
void pipelined_matches_single() {
    for (bool tied : {true, false}) {
        const auto weights = tiny_qwen(3, 2 * 128, tied);
        for (const PipelinedSplit& ps : kPipelined) {
            auto one = std::make_shared<backend::CpuBackend>();
            one->set_threads(1);
            infer::Model single(infer::gguf_weights(weights), one);
            single.set_ubatch(3);
            std::vector<backend::BackendPtr> cpus;
            for (size_t i = 0; i < ps.shares.size(); ++i) cpus.push_back(std::make_shared<backend::CpuBackend>());
            infer::PlacedModel placed = pipelined(weights, cpus, ps.shares);
            infer::Model& split = *placed.model;
            const size_t V = single.n_vocab();

            exact(single.prefill(kPrompt), split.prefill(kPrompt), "pipelined prefill differs from one device");
            for (int t : {7, 9, 3}) exact(single.step(t), split.step(t), "step after a pipelined prefill differs from one device");
            const std::vector<uint32_t> more{2, 7, 1, 8, 2, 8, 1};
            exact(single.prefill(more), split.prefill(more), "a pipelined prompt continuing a history differs from one device");
            require(split.n_tokens() == 23 && single.n_tokens() == 23 && split.kv_used_bytes() == single.kv_used_bytes(),
                    "pipelined history differs from one device");
            checked += 3;

            std::vector<float> scored;
            single.score(kPrompt, [&](size_t, const float* logits) { scored.insert(scored.end(), logits, logits + V); });
            size_t rows = 0;
            bool same = scored.size() == kPrompt.size() * V;
            split.score(kPrompt, [&](size_t pos, const float* logits) {
                same = same && pos == rows && rows < kPrompt.size() && !std::memcmp(logits, &scored[pos * V], V * sizeof(float));
                ++rows;
            });
            require(same && rows == kPrompt.size(), "a scored row differs from one device");
            ++checked;

            // The budget is two blocks, so the default sequences give theirs back first.
            single.reset();
            split.reset();
            infer::Sequence a = split.make_sequence(), b = split.make_sequence();
            infer::Sequence c = single.make_sequence(), d = single.make_sequence();
            infer::ExecContext xs, xc;
            const infer::BatchEntry history_split{&a, kPrompt.data(), 4, false}, history_single{&c, kPrompt.data(), 4, false};
            split.forward(xs, &history_split, 1);
            single.forward(xc, &history_single, 1);
            const uint32_t next = 6;
            const infer::BatchEntry es[2] = {{&a, &next, 1, true}, {&b, more.data(), more.size(), true}};
            const infer::BatchEntry ec[2] = {{&c, &next, 1, true}, {&d, more.data(), more.size(), true}};
            split.forward(xs, es, 2);
            single.forward(xc, ec, 2);
            for (size_t r = 0; r < 2; ++r)
                require(!std::memcmp(xs.logits(r), xc.logits(r), V * sizeof(float)), "a two-sequence pass differs from one device");
            // Every stage but the last sends, through two handoff buffers; the last sends nothing and keeps none.
            bool handoffs = xs.handoff.size() == ps.shares.size();
            for (size_t dev = 0; handoffs && dev < ps.shares.size(); ++dev) handoffs = xs.handoff[dev].size() == (dev + 1 < ps.shares.size() ? 2u : 0u);
            require(handoffs, "handoff buffers of a pipelined split not two on each stage that sends and none on the last");
            require(a.length() == 5 && b.length() == more.size(), "a two-sequence pass did not commit both");
            ++checked;
        }
    }
}

// A backend on the last stage fails while the first stage is chunks ahead, on top of a history: at its first attention on chunk 2 of 5, and at the head on the last chunk.
// Every storage must be back at the history, which the first attention each device runs next reads, and the same prompt again must be exact.
void pipelined_failure_rolls_back() {
    const auto weights = tiny_qwen(3, 2 * 128, true);
    for (const PipelinedSplit& ps : kPipelined) {
        auto plain = std::make_shared<backend::CpuBackend>();
        plain->set_threads(1);
        infer::Model control(infer::gguf_weights(weights), plain);
        control.set_ubatch(3);
        std::vector<std::shared_ptr<FailingCpu>> stages;
        for (size_t i = 0; i < ps.shares.size(); ++i) stages.push_back(std::make_shared<FailingCpu>());
        infer::PlacedModel placed = pipelined(weights, std::vector<backend::BackendPtr>(stages.begin(), stages.end()), ps.shares);
        infer::Model& split = *placed.model;
        FailingCpu& first = *stages.front();
        FailingCpu& last = *stages.back();
        const std::vector<uint32_t> history{2, 7, 1, 8};
        exact(control.prefill(history), split.prefill(history), "history before a failure differs from one device");

        for (bool at_head : {false, true}) {
            const size_t before = (size_t)split.n_tokens(), used = split.kv_used_bytes();
            for (auto& s : stages) s->histories.clear();
            if (at_head)
                last.fail_output = true;
            else
                last.fail_attention = 2 * ps.last_layers + 1;
            bool failed = false;
            try { split.prefill(kPrompt); } catch (const std::runtime_error&) { failed = true; }
            require(failed && !last.fail_output && !last.fail_attention, "the injected failure did not fire");
            // Chunk 2 starts 6 tokens past the history and the last chunk 12; the first stage had begun chunk 3 at least.
            require(last.histories.back() == before + (at_head ? 12 : 6) && first.histories.back() >= before + 9,
                    "the failure was not on the last stage with the first chunks ahead");
            require((size_t)split.n_tokens() == before && split.kv_used_bytes() == used, "a failed pipelined prompt changed the history");

            for (auto& s : stages) s->histories.clear();
            exact(control.prefill(kPrompt), split.prefill(kPrompt), "a pipelined prompt after a failure differs from one device");
            for (auto& s : stages)
                require(!s->histories.empty() && s->histories.front() == before, "a storage kept tokens of a failed pipelined prompt");
            checked += 2;
        }
        exact(control.step(5), split.step(5), "a step after the retried prompts differs from one device");
        ++checked;
    }
}

// A request's histories (PlacementRequest::histories) grow the cache budget only where it cannot hold them, each in whole blocks up to the context: the fixture's context is two CPU blocks.
void histories_fit_the_pool() {
    const auto weights = fixture();
    const infer::ModelOptions options;
    auto place = [&](size_t histories, size_t tokens, size_t devices) {
        infer::PlacementRequest request;
        std::vector<backend::BackendPtr> backends;
        for (size_t d = 0; d < devices; ++d) {
            auto cpu = std::make_shared<backend::CpuBackend>();
            cpu->set_threads(1);
            backends.push_back(cpu);
            request.names.push_back("cpu");
        }
        if (devices > 1) request.shares.assign(devices, 1);
        request.histories = histories;
        request.history_tokens = tokens;
        return infer::place_model(infer::gguf_weights(weights), std::move(backends), request, options).model;
    };
    // One history of one block leaves the two-block budget as it is, where a budget set to what the histories take would shrink to that block.
    require(place(0, 0, 1)->kv_tokens_total() == 256 && place(1, 100, 1)->kv_tokens_total() == 256 &&
                place(2, 128, 1)->kv_tokens_total() == 256,
            "a budget that holds the histories changed");
    require(place(3, 100, 1)->kv_tokens_total() == 384 && place(2, 129, 1)->kv_tokens_total() == 512,
            "a budget short of whole blocks for each history did not grow");
    ++checked;
    // A history is counted up to the context: one of 1000 tokens takes the context's two blocks and leaves the budget as it is, and three take six.
    require(place(1, 1000, 1)->kv_tokens_total() == 256 && place(3, 1000, 1)->kv_tokens_total() == 768,
            "a history past the context was counted past it");
    ++checked;
    // Three histories of 100 tokens in one pass, which the context's two blocks cannot hold, on one device and over a split.
    for (size_t devices : {1, 2}) {
        auto model = place(3, 100, devices);
        std::vector<infer::Sequence> s;
        for (int i = 0; i < 3; ++i) s.push_back(model->make_sequence());
        const std::vector<uint32_t> ids(100, 5);
        std::vector<infer::BatchEntry> batch;
        for (auto& q : s) batch.push_back(infer::BatchEntry{&q, ids.data(), ids.size(), true});
        infer::ExecContext ctx;
        model->forward(ctx, batch.data(), batch.size());
        for (const auto& q : s) require(q.length() == 100, "a history the grown pool holds was not committed");
        ++checked;
    }
}

// A device reporting `room` bytes free, which is not the CPU; with `copying` it keeps copies of the weights it adopts, so the fit counts them.
// With `rise_after`, it reports one byte free for that many reads before `room`, as a device does while it reclaims the memory of a process that has just ended.
struct SizedDevice : HostMemoryDevice {
    size_t room = 0;
    bool copying = false;
    mutable int rise_after = 0;
    std::optional<size_t> memory_available() const override { return rise_after-- > 0 ? 1 : room; }
    bool reads_in_place() const override { return !copying; }
};

// A server's KV budget (PlacementRequest::fit_kv): the request's budget where it fits, backed whole as the model is made; the most whole blocks that fit where it does not, one more block not fitting; a load refused where not one block fits; and beside experts on the CPU, the device not charged for those layers' feed-forward weights.
void kv_fitted() {
    const auto weights = fixture();
    auto place = [&](size_t room, size_t budget, bool fit, const gguf::GGUFModel& m, int cpu_moe = 0, bool copying = false, int rise_after = 0) {
        auto device = std::make_shared<SizedDevice>();
        device->room = room;
        device->copying = copying;
        device->rise_after = rise_after;
        device->set_threads(1);
        infer::PlacementRequest request;
        request.names = {"device"};
        request.fit_kv = fit;
        request.cpu_moe = cpu_moe;
        infer::ModelOptions options;
        options.kv_tokens = budget;
        return infer::place_model(infer::gguf_weights(m), {device}, request, options).model;
    };
    const size_t roomy = size_t(1) << 30;
    auto grown = place(roomy, 0, false, weights);
    require(grown->kv_tokens_total() == 256 && grown->kv_allocated_bytes() == 0, "a budget not fitted was backed before a pass");
    grown->prefill(std::vector<uint32_t>(256, 3));
    auto kept = place(roomy, 0, true, weights);
    require(kept->kv_tokens_total() == 256 && kept->kv_allocated_bytes() == grown->kv_allocated_bytes(),
            "a fitted budget that fits was cut, or not backed whole as the model was made");
    // A budget of 2^20 tokens on a device with 8 MiB free is cut to whole blocks, which fit, and one more block does not.
    const size_t cut = place(8 << 20, size_t(1) << 20, true, weights)->kv_tokens_total();
    require(cut > 0 && cut < (size_t(1) << 20) && cut % 128 == 0, "a budget past the device's memory was not cut to whole blocks");
    require(place(8 << 20, cut, true, weights)->kv_tokens_total() == cut && place(8 << 20, cut + 128, true, weights)->kv_tokens_total() == cut,
            "the cut budget is not the most whole blocks that fit");
    // Free memory that rises while the fit reads it, as a device reclaims an ended process's memory, is read again until it settles, so the load takes what the device holds once it has.
    require(place(8 << 20, cut, true, weights, 0, false, 2)->kv_tokens_total() == cut, "a fit did not wait for a device's free memory to settle");
    // A device may hold an ended process's memory for a while between frees: free memory that stays level for four seconds and then rises is waited for too.
    require(place(8 << 20, cut, true, weights, 0, false, 16)->kv_tokens_total() == cut, "a fit gave up before a device began to give memory back");
    // A split's placement, which every command fits, waits for it the same way: two devices reporting a byte free for two reads each are placed once their memory has come back.
    {
        std::vector<backend::BackendPtr> two;
        for (int d = 0; d < 2; ++d) {
            auto device = std::make_shared<SizedDevice>();
            device->room = 8 << 20;
            device->rise_after = 2;
            device->set_threads(1);
            two.push_back(device);
        }
        infer::PlacementRequest request;
        request.names = {"device 0", "device 1"};
        request.shares = {1, 1};
        require(infer::place_model(infer::gguf_weights(weights), two, request, infer::ModelOptions{}).model->stage_count() == 2,
                "a split did not wait for its devices' free memory to settle");
    }
    // A budget the device holds is taken at once, while one it cuts waits for its free memory to settle, so a load meant to succeed asks only for what it holds.
    const auto refusal_of = [&](size_t room, size_t budget = 0) {
        try {
            place(room, budget, true, weights);
        } catch (const std::runtime_error& e) {
            return std::string(e.what());
        }
        return std::string();
    };
    require(refusal_of(1).find("does not fit the devices' free memory even without its KV") != std::string::npos,
            "a device that cannot hold the model was not refused as such");
    // The least room that holds one block beside the model, found through the fit's own terms without placing a model, since every refused load waits for the memory to settle; a byte less holds the model without a whole block of KV.
    const auto holds_block = [&](size_t room) {
        auto device = std::make_shared<SizedDevice>();
        device->room = room;
        infer::ModelOptions options;
        options.kv_tokens = 128;
        try {
            infer::split_layers(infer::footprint(infer::gguf_weights(weights), infer::plan_model(infer::gguf_weights(weights)), options),
                                infer::budgets_for({device}, {"device"}), infer::kDefaultUbatch, {}, core::host_memory_available());
            return true;
        } catch (const std::runtime_error&) {
            return false;
        }
    };
    size_t lo = 1, hi = 8 << 20;
    while (hi - lo > 1) {
        const size_t mid = lo + (hi - lo) / 2;
        (holds_block(mid) ? hi : lo) = mid;
    }
    require(refusal_of(hi, 128).empty(), "the least room that holds a block of KV beside the model did not load");
    require(refusal_of(lo).find("no room for one KV block") != std::string::npos, "a device with no room for one KV block was not refused as such");
    // Beside experts on the CPU a copying device holds every layer's attention and not the routed feed-forward blocks, which take more than a block, so it fits more KV than it does holding them.
    const auto moe = tiny_qwen_moe(2, 2 * 128, true);
    const size_t with_experts = place(8 << 20, size_t(1) << 20, true, moe, 0, true)->kv_tokens_total();
    const size_t without = place(8 << 20, size_t(1) << 20, true, moe, -1, true)->kv_tokens_total();
    require(without > with_experts, "a device beside experts on the CPU was charged for their weights");
    checked += 6;
}

// A history recomputed in the classes that first computed it, over two CPU stages, as a paused request's resume recomputes it (docs/SERVER.md, pausing): a 40-token prompt at its extent in slices of 16, then 199 greedy tokens as entries of extent 1 of up to 64 rows, logits only on the last.
// The synthetic Q8_0 model's decode rows take the 8-bit dots, so the replay must give the logits one backend gives after the prompt and 199 single decode steps, bit for bit; and so must a fork at the first block replaying the rest.
void replay_over_stages() {
    const gguf::GGUFModel weights = infer::synthetic_model({2, 64, 128, 4, 2, 16, 64, 11u});
    auto one = std::make_shared<backend::CpuBackend>();
    one->set_threads(1);
    infer::Model single(infer::gguf_weights(weights), one);
    std::vector<backend::BackendPtr> two{std::make_shared<backend::CpuBackend>(), std::make_shared<backend::CpuBackend>()};
    infer::PlacementRequest request;
    request.names = {"cpu", "cpu"};
    request.shares = {1, 1};
    for (auto& b : two) b->set_threads(1);
    infer::PlacedModel placed = infer::place_model(infer::gguf_weights(weights), two, request, infer::ModelOptions{});
    infer::Model& split = *placed.model;
    const size_t prompt = 40, steps = 199, V = single.n_vocab(), bt = one->kv_layout().block_tokens;
    std::vector<uint32_t> ids(prompt);
    for (size_t i = 0; i < prompt; ++i) ids[i] = (uint32_t)((i * 11 + 3) % V);
    std::vector<float> want = single.prefill(ids);
    for (size_t s = 0; s < steps; ++s) {
        uint32_t next = 0;
        for (uint32_t v = 1; v < V; ++v)
            if (want[v] > want[next]) next = v;
        ids.push_back(next);
        want = single.step((int)next);
    }
    single.reset();
    auto replay = [&](infer::Model& m, infer::Sequence& seq) {
        infer::ExecContext ctx;
        std::vector<float> got;
        while (seq.length() < ids.size()) {
            const size_t at = seq.length();
            const bool generated = at >= prompt;
            const size_t n = generated ? std::min<size_t>(64, ids.size() - at) : std::min<size_t>(16, prompt - at);
            infer::BatchEntry e{&seq, ids.data() + at, n, at + n == ids.size()};
            e.extent = generated ? 1 : prompt;
            m.forward(ctx, &e, 1);
            if (e.want_logits) got.assign(ctx.logits(0), ctx.logits(0) + V);
        }
        return got;
    };
    infer::Sequence whole = split.make_sequence();
    exact(replay(split, whole), want, "a history replayed by class over two stages differs from its decode on one backend");
    infer::Sequence forked = split.fork(whole, bt);
    split.reset(whole);
    exact(replay(split, forked), want, "a fork at a block replaying the rest by class over two stages differs from the decode");
    split.reset(forked);
    checked += 2;
}

void bad_placements_refused() {
    const auto weights = fixture();
    auto a = std::make_shared<backend::CpuBackend>(), b = std::make_shared<backend::CpuBackend>();
    auto rejects = [&](infer::Placement p, const char* what) {
        bool caught = false;
        try { infer::Model m(infer::gguf_weights(weights), {a, b}, p); } catch (const std::runtime_error&) { caught = true; }
        require(caught, what);
        ++checked;
    };
    infer::Placement p;
    p.mixer_device = {0};
    p.ffn_device = {0, 0};
    rejects(p, "placement short of a layer accepted");
    p.mixer_device = {0, 2};
    rejects(p, "placement naming a missing device accepted");
    p.mixer_device = {0, 0};
    p.output_device = -1;
    rejects(p, "negative device accepted");
    // A device whose attention layers come back after another's would own a storage two stages write.
    const auto three = tiny_qwen(3, 2 * 128, true);
    infer::Placement split;
    split.mixer_device = split.ffn_device = {0, 1, 0};
    bool split_refused = false;
    try { infer::Model m(infer::gguf_weights(three), {a, b}, split); } catch (const std::runtime_error&) { split_refused = true; }
    require(split_refused, "a device's attention layers split in two accepted");
    ++checked;
    bool caught = false;
    try { infer::Model m(infer::gguf_weights(weights), {a, nullptr}, infer::Placement{}); }
    catch (const std::runtime_error&) { caught = true; }
    require(caught, "null backend accepted");
    ++checked;
    // A sequence of another model, even one of the same shape, is refused by forward and by reset: its block ids belong to the other pool.
    infer::Model m1(infer::gguf_weights(weights), a), m2(infer::gguf_weights(weights), b);
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

// Passes in flight through the pass API (Model::reserve_passes and the calls after it): five requests, the second a fork of the first's first block, each prompt sliced at random and each generated token given.
// Passes of random groups of ready requests are formed, advanced a stage and retired in random order, so they queue between stages, hand off through their own buffers and take logits rows anywhere in the reserved range.
struct PassRequest {
    std::vector<uint32_t> prompt;            // the tokens it prefills after the prefix it forks
    std::vector<uint32_t> gen;               // the tokens it is given after its prompt, one a pass
    size_t prefix = 0;                       // the first request's tokens it forks
    infer::Sequence seq;
    bool made = false;
    size_t done = 0;                         // its own tokens committed
    std::vector<std::vector<float>> alone;   // its rows run alone: after its prompt, then after each generated token
    size_t seen = 0;                         // rows checked
    bool finished() const { return done == prompt.size() + gen.size(); }
};

struct FormedPass {
    std::vector<size_t> reqs, tokens;   // per entry
    std::vector<char> wants;
    size_t base = 0, want = 0, ran = 0;
    bool live = false;
};

constexpr size_t kPassUbatch = 16, kPassSeqs = 5, kPassLogitRows = 2 * kPassSeqs;

// One run over S CPU stages with P pass slots, in which every logits row must be the bytes of its request run alone through prefill and step on one CPU backend.
// Once in the run, a backend fails a pass of several sequences at its next stage: that pass's histories go back to where it found them in every storage, the other passes go on, and its requests, formed again, still give their rows alone.
void passes_run(const gguf::GGUFModel& weights, size_t S, size_t P, uint32_t seed) {
    std::mt19937 rng(seed);
    auto below = [&](size_t n) { return (size_t)(rng() % (uint32_t)n); };
    auto tokens = [&](size_t n) {
        std::vector<uint32_t> v(n);
        for (auto& t : v) t = (uint32_t)below(16);
        return v;
    };
    infer::ModelOptions options;
    options.kv_tokens = 16 * 128;
    std::vector<std::shared_ptr<FailingCpu>> stages;
    for (size_t s = 0; s < S; ++s) {
        stages.push_back(std::make_shared<FailingCpu>());
        stages.back()->set_threads(1);
    }
    infer::PlacementRequest request;
    request.names.assign(S, "cpu");
    request.shares.assign(S, 1);
    request.ubatch = (int)kPassUbatch;
    request.slots = P;
    infer::PlacedModel placed = infer::place_model(infer::gguf_weights(weights), std::vector<backend::BackendPtr>(stages.begin(), stages.end()), request, options);
    infer::Model& split = *placed.model;
    require(split.pipelined() && split.stage_count() == S, "the passes' split is not pipelined over every device");
    infer::ExecContext ctx;
    split.reserve_passes(ctx, P, kPassUbatch + kPassSeqs, kPassLogitRows);
    bool handoffs = ctx.handoff.size() == S;
    for (size_t s = 0; handoffs && s < S; ++s) handoffs = ctx.handoff[s].size() == (s + 1 < S ? std::max<size_t>(2, P) : 0);
    require(handoffs, "reserved handoff buffers not one a slot, two at least, on each stage that sends and none on the last");

    // Declared after the model, so their sequences return their blocks before it goes.
    std::vector<PassRequest> reqs(kPassSeqs);
    reqs[0].prompt = tokens(129 + below(40));
    reqs[1].prefix = 128;
    reqs[1].prompt = tokens(2 + below(30));
    for (size_t r = 2; r < kPassSeqs; ++r) reqs[r].prompt = tokens(2 + below(60));
    for (auto& q : reqs) q.gen = tokens(1 + below(6));
    auto one = std::make_shared<backend::CpuBackend>();
    one->set_threads(1);
    infer::Model single(infer::gguf_weights(weights), one, options);
    single.set_ubatch((int)kPassUbatch);
    for (auto& q : reqs) {
        std::vector<uint32_t> all(reqs[0].prompt.begin(), reqs[0].prompt.begin() + (std::ptrdiff_t)q.prefix);
        all.insert(all.end(), q.prompt.begin(), q.prompt.end());
        single.reset();
        q.alone.push_back(single.prefill(all));
        for (uint32_t t : q.gen) q.alone.push_back(single.step((int)t));
    }
    const size_t V = single.n_vocab();
    for (auto& q : reqs)
        if (!q.prefix) {
            q.seq = split.make_sequence();
            q.made = true;
        }

    std::vector<FormedPass> slots(P);
    std::vector<char> rows_used(kPassLogitRows, 0), flying(kPassSeqs, 0);
    std::vector<infer::BatchEntry> entries;
    const size_t fail_after = below(4);
    size_t formed = 0, most_between = 0;
    bool failed = false;
    // A request not in flight with tokens left; the fork once the first request has committed its prefix.
    auto ready = [&](size_t r) {
        const PassRequest& q = reqs[r];
        return !flying[r] && !q.finished() && (q.made || (!flying[0] && reqs[0].done >= q.prefix));
    };
    auto free_rows = [&](const FormedPass& f) {
        for (size_t i = f.base; i < f.base + f.want; ++i) rows_used[i] = 0;
    };

    // Every ready request or a random group of them, each a prompt slice within the ubatch or its next generated token, in random order, at a free run of logits rows.
    auto form = [&](size_t k) {
        std::vector<size_t> pick;
        const bool all = below(2);
        for (size_t r = 0; r < kPassSeqs; ++r)
            if (ready(r) && (all || below(2))) pick.push_back(r);
        if (pick.empty()) {
            std::vector<size_t> any;
            for (size_t r = 0; r < kPassSeqs; ++r)
                if (ready(r)) any.push_back(r);
            pick.push_back(any[below(any.size())]);
        }
        for (size_t i = pick.size(); i > 1; --i) std::swap(pick[i - 1], pick[below(i)]);
        FormedPass f;
        entries.clear();
        size_t budget = kPassUbatch;
        for (size_t r : pick) {
            PassRequest& q = reqs[r];
            if (!ready(r)) continue;
            const bool prompt = q.done < q.prompt.size();
            if (prompt && !budget) continue;
            if (!q.made) {
                q.seq = split.fork(reqs[0].seq, q.prefix);
                q.made = true;
            }
            infer::BatchEntry e{&q.seq, q.gen.data() + (prompt ? 0 : q.done - q.prompt.size()), 1, true};
            if (prompt) {
                e.n = 1 + below(std::min(budget, q.prompt.size() - q.done));
                e.ids = q.prompt.data() + q.done;
                e.want_logits = q.done + e.n == q.prompt.size();
                e.extent = q.prefix + q.prompt.size();
                budget -= e.n;
            }
            entries.push_back(e);
            f.reqs.push_back(r);
            f.tokens.push_back(e.n);
            f.wants.push_back(e.want_logits);
            f.want += e.want_logits;
        }
        std::vector<size_t> starts;
        for (size_t b = 0; b + f.want <= kPassLogitRows; ++b) {
            bool open = true;
            for (size_t i = b; i < b + f.want; ++i) open = open && !rows_used[i];
            if (open) starts.push_back(b);
        }
        if (entries.empty() || starts.empty()) return;
        f.base = starts[below(starts.size())];
        split.begin_pass(ctx, k, entries.data(), entries.size(), f.base);
        for (size_t i = f.base; i < f.base + f.want; ++i) rows_used[i] = 1;
        for (size_t r : f.reqs) {
            require(reqs[r].seq.in_flight(), "begin_pass did not put a sequence in flight");
            flying[r] = 1;
        }
        f.live = true;
        slots[k] = f;
        ++formed;
    };

    // The pass's next stage; once in the run, on a pass of several sequences, its backend fails first.
    auto advance = [&](size_t k) {
        FormedPass& f = slots[k];
        const size_t s = f.ran;
        const bool arm = !failed && formed > fail_after && f.reqs.size() > 1;
        if (arm) {
            if (s + 1 == S && f.want && below(2)) stages[s]->fail_output = true;
            else stages[s]->fail_attention = 1;
        }
        try {
            split.run_pass_stage(ctx, k, s);
        } catch (const std::runtime_error& e) {
            require(arm && !std::strcmp(e.what(), "injected") && !stages[s]->fail_output && !stages[s]->fail_attention,
                    "a pass failed where no failure was injected");
            for (size_t r : f.reqs) {
                require(!reqs[r].seq.in_flight() && reqs[r].seq.length() == reqs[r].prefix + reqs[r].done,
                        "a failed pass left a sequence in flight or changed its history");
                flying[r] = 0;
            }
            free_rows(f);
            f.live = false;
            failed = true;
            ++checked;
            return;
        }
        require(!arm, "the injected failure did not fire");
        ++f.ran;
    };

    // Every wanting row against the request's row alone, then the pass ends and its tokens count.
    auto retire = [&](size_t k) {
        FormedPass& f = slots[k];
        for (size_t e = 0, row = 0; e < f.reqs.size(); ++e) {
            if (!f.wants[e]) continue;
            PassRequest& q = reqs[f.reqs[e]];
            const size_t expect = q.done < q.prompt.size() ? 0 : q.done - q.prompt.size() + 1;
            require(q.seen == expect && expect < q.alone.size(), "a request's rows out of order");
            require(!std::memcmp(split.pass_logits(ctx, k, row++), q.alone[expect].data(), V * sizeof(float)),
                    "a row of a pass in flight differs from its sequence run alone");
            ++q.seen;
            ++checked;
        }
        split.end_pass(ctx, k);
        for (size_t e = 0; e < f.reqs.size(); ++e) {
            PassRequest& q = reqs[f.reqs[e]];
            q.done += f.tokens[e];
            flying[f.reqs[e]] = 0;
            require(!q.seq.in_flight() && q.seq.length() == q.prefix + q.done, "end_pass left a sequence in flight or short of its tokens");
        }
        free_rows(f);
        f.live = false;
    };

    for (size_t step = 0;; ++step) {
        require(step < 100000, "the passes did not finish");
        bool any_ready = false;
        for (size_t r = 0; r < kPassSeqs; ++r) any_ready = any_ready || ready(r);
        // A kind of action first, forming, advancing or retiring, then a slot it can take, so free slots do not crowd out the passes in flight.
        std::vector<size_t> can[3];
        size_t between = 0;
        for (size_t k = 0; k < P; ++k) {
            if (!slots[k].live) {
                if (any_ready) can[0].push_back(k);
                continue;
            }
            can[slots[k].ran < S ? 1 : 2].push_back(k);
            between += slots[k].ran > 0 && slots[k].ran < S;
        }
        most_between = std::max(most_between, between);
        std::vector<int> kinds;
        for (int i = 0; i < 3; ++i)
            if (!can[i].empty()) kinds.push_back(i);
        if (kinds.empty()) break;
        const int kind = kinds[below(kinds.size())];
        const size_t k = can[kind][below(can[kind].size())];
        if (kind == 0) form(k);
        else if (kind == 1) advance(k);
        else retire(k);
    }
    require(failed, "the failure was never injected");
    require(most_between >= 2, "no two passes waited between stages at once");
    for (const auto& q : reqs)
        require(q.finished() && q.seen == q.alone.size() && !q.seq.in_flight() && q.seq.length() == q.prefix + q.done,
                "a request did not run to its end");
    checked += 2;
}

// Over 2, 3 and 4 stages of a four-layer model, tied and untied, at P = S, S + 1 and 2S.
void passes_match_alone() {
    for (bool tied : {true, false}) {
        const auto weights = tiny_qwen(4, 2 * 128, tied);
        for (size_t S : {2, 3, 4})
            for (size_t P : {S, S + 1, 2 * S}) passes_run(weights, S, P, (uint32_t)(1000 * tied + 100 * S + P));
    }
}

// A CPU backend whose allocations above `limit` bytes fail, as a device's do when its memory runs out.
struct TightCpu : backend::CpuBackend {
    size_t limit = SIZE_MAX;
    backend::BufferPtr alloc(size_t bytes, backend::Memory where) override {
        if (bytes > limit) throw std::runtime_error("out of memory");
        return backend::CpuBackend::alloc(bytes, where);
    }
};

// What the pass API refuses, each before any work and with nothing changed: a context not reserved, reserved twice, used before or on a placement that is not pipelined for more than one slot; forward through a reserved context; a slot beyond the reservation or in use; more rows or logits rows than reserved; a sequence listed twice or already in flight, and reset, fork and forward of one; stages out of order or twice, and logits or an end before the last stage.
// Beside them, a reservation the devices cannot hold fails with their error and leaves the context fresh, so a smaller one on it succeeds, and a pass ended, one aborted after its first stage and the aborted one run again each leave the histories and rows of the sequences run alone.
void passes_refused() {
    const auto weights = tiny_qwen(4, 2 * 128, true);
    std::vector<std::shared_ptr<TightCpu>> tight{std::make_shared<TightCpu>(), std::make_shared<TightCpu>()};
    auto cpus = [&] {
        std::vector<backend::BackendPtr> v(tight.begin(), tight.end());
        for (auto& c : v) c->set_threads(1);
        return v;
    };
    infer::PlacementRequest request;
    request.names = {"cpu", "cpu"};
    request.shares = {1, 1};
    request.ubatch = 4;
    infer::ModelOptions options;
    options.kv_tokens = 8 * 128;
    infer::PlacedModel placed = infer::place_model(infer::gguf_weights(weights), cpus(), request, options);
    infer::Model& m = *placed.model;
    auto cpu = std::make_shared<backend::CpuBackend>();
    cpu->set_threads(1);
    infer::Model single(infer::gguf_weights(weights), cpu);
    single.set_ubatch(4);
    const size_t V = single.n_vocab();
    auto refused = [&](const std::function<void()>& call, const char* what) {
        bool caught = false;
        try { call(); } catch (const std::logic_error&) { caught = true; }
        require(caught, what);
        ++checked;
    };

    infer::Sequence a = m.make_sequence(), b = m.make_sequence();
    const std::vector<uint32_t> ids{3, 1, 4, 1, 5};
    const infer::BatchEntry ea{&a, ids.data(), 3, true}, eb{&b, ids.data(), 1, true}, five{&a, ids.data(), 5, true};
    infer::ExecContext ctx;
    refused([&] { m.begin_pass(ctx, 0, &ea, 1, 0); }, "a pass began in a context not reserved for passes");
    auto own = std::make_shared<backend::CpuBackend>();
    own->set_threads(1);
    infer::Model alone(infer::gguf_weights(weights), own);
    infer::ExecContext one;
    refused([&] { alone.reserve_passes(one, 2, 4, 2); }, "two passes in flight reserved on one device");
    refused([&] { m.reserve_passes(ctx, 0, 4, 2); }, "a reservation without a slot");
    infer::ExecContext used;
    infer::Sequence c = m.make_sequence();
    const infer::BatchEntry ec{&c, ids.data(), 1, true};
    m.forward(used, &ec, 1);
    refused([&] { m.reserve_passes(used, 2, 4, 2); }, "a context a forward used reserved for passes");
    for (auto& t : tight) t->limit = size_t(1) << 20;
    bool short_memory = false;
    try { m.reserve_passes(ctx, 2, size_t(1) << 16, 2); } catch (const std::runtime_error&) { short_memory = true; }
    for (auto& t : tight) t->limit = SIZE_MAX;
    require(short_memory && !ctx.slots && ctx.scratch.empty() && ctx.handoff.empty() && !ctx.logits_buf && ctx.passes.empty(),
            "a reservation the devices cannot hold did not fail, or left the context changed");
    ++checked;
    m.reserve_passes(ctx, 2, 4, 2);
    refused([&] { m.reserve_passes(ctx, 2, 4, 2); }, "a context reserved twice");
    refused([&] { m.forward(ctx, &ea, 1); }, "forward through a context reserved for passes");
    refused([&] { m.begin_pass(ctx, 2, &ea, 1, 0); }, "a pass beyond the reserved slots");
    refused([&] { m.begin_pass(ctx, 0, &five, 1, 0); }, "a pass of more rows than reserved");
    refused([&] { m.begin_pass(ctx, 0, &ea, 1, 2); }, "a pass's logits rows beyond the reservation");
    const infer::BatchEntry twice[2] = {ea, ea};
    refused([&] { m.begin_pass(ctx, 0, twice, 2, 0); }, "a sequence listed twice in a pass");
    require(!a.in_flight(), "a refused pass left its sequence in flight");
    refused([&] { m.run_pass_stage(ctx, 0, 0); }, "a stage ran in a slot a refused pass left");

    m.begin_pass(ctx, 0, &ea, 1, 1);
    require(a.in_flight() && !b.in_flight(), "begin_pass did not put exactly its sequence in flight");
    refused([&] { m.begin_pass(ctx, 0, &eb, 1, 0); }, "a slot in flight took a second pass");
    const infer::BatchEntry both[2] = {eb, ea};
    refused([&] { m.begin_pass(ctx, 1, both, 2, 0); }, "a sequence in flight joined a second pass");
    require(!b.in_flight(), "a refused pass left a sequence in flight");
    refused([&] { m.reset(a); }, "a sequence in flight reset");
    refused([&] { m.fork(a, 0); }, "a sequence in flight forked");
    infer::ExecContext other;
    refused([&] { m.forward(other, &ea, 1); }, "forward took a sequence in flight");
    refused([&] { m.run_pass_stage(ctx, 0, 1); }, "a pass's second stage ran before its first");
    refused([&] { m.run_pass_stage(ctx, 1, 0); }, "a stage ran in a slot with no pass");
    refused([&] { m.pass_logits(ctx, 0, 0); }, "logits read before a pass's last stage");
    refused([&] { m.end_pass(ctx, 0); }, "a pass ended before its last stage");
    require(a.in_flight(), "a refused call took its sequence out of flight");

    // A second pass's first stage between the first pass's two, through the other slot's handoff buffers.
    m.run_pass_stage(ctx, 0, 0);
    m.begin_pass(ctx, 1, &eb, 1, 0);
    m.run_pass_stage(ctx, 1, 0);
    m.run_pass_stage(ctx, 0, 1);
    refused([&] { m.run_pass_stage(ctx, 0, 1); }, "a stage ran twice");
    bool beyond = false;
    try { m.pass_logits(ctx, 0, 1); } catch (const std::out_of_range&) { beyond = true; }
    require(beyond, "a logits row beyond the pass was served");
    const std::vector<float> first = single.prefill({3, 1, 4});
    require(!std::memcmp(m.pass_logits(ctx, 0, 0), first.data(), V * sizeof(float)), "a pass's row differs from its sequence alone");
    m.end_pass(ctx, 0);
    require(!a.in_flight() && a.length() == 3, "end_pass left its sequence in flight or short of its tokens");
    refused([&] { m.end_pass(ctx, 0); }, "a pass ended twice");

    // The second pass aborted after its first stage: its sequence out of flight and back at an empty history in both storages, then run again whole.
    m.abort_pass(ctx, 1);
    require(!b.in_flight() && b.length() == 0, "abort_pass left its sequence in flight or its history changed");
    refused([&] { m.abort_pass(ctx, 1); }, "a pass aborted twice");
    m.begin_pass(ctx, 1, &eb, 1, 0);
    m.run_pass_stage(ctx, 1, 0);
    m.run_pass_stage(ctx, 1, 1);
    single.reset();
    const std::vector<float> retried = single.step(3);
    require(!std::memcmp(m.pass_logits(ctx, 1, 0), retried.data(), V * sizeof(float)), "an aborted pass run again differs from its sequence alone");
    m.end_pass(ctx, 1);
    require(b.length() == 1, "the pass run again did not commit");
    m.reset(a);
    m.reset(b);
    checked += 6;
}
}

int main() {
    try {
        host_scratch_fits();
        split_matches_single();
        layer_split_fits();
        histories_fit_the_pool();
        kv_fitted();
        bad_placements_refused();
        pipelined_matches_single();
        pipelined_failure_rolls_back();
        replay_over_stages();
        passes_refused();
        passes_match_alone();
        std::cout << "placement: " << checked << " checks across two, three and four CPU backends\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
