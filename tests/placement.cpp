// Placement across backends (docs/EXECUTION.md step 6): a model split over two, three or four CPU backends must produce the bytes of the same model on one, because per-role arithmetic is unchanged and only the residual stream crosses.
// Crossings are counted so they happen exactly where the placement changes and nowhere on a single device; bad placements are refused at load.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "model/runtime.hpp"
#include "model/place.hpp"
#include "model/arch/registry.hpp"
#include "model/layer_split.hpp"
#include "core/host_memory.hpp"
#include "tiny_qwen.hpp"
#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// Two layers, so a layer boundary and a within-layer boundary can both be placed, and a context of two CPU blocks so a prompt can cross one.
gguf::GGUFModel fixture() { return tiny_qwen(2, 2 * 128, true); }

// A lookup table's source over `bytes` written to a file of its own, which main removes once every source and its reader are gone.
std::vector<std::filesystem::path> table_files;
std::shared_ptr<const infer::WeightRows> table_source(uint32_t type, size_t columns, size_t rows, const uint8_t* bytes, size_t size) {
    const auto path = std::filesystem::temp_directory_path() /
        ("llmx-placement-table-" + std::to_string(std::random_device{}()) + "-" + std::to_string(table_files.size()) + ".bin");
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(bytes), std::streamsize(size));
        if (!out) throw std::runtime_error("cannot write " + path.string());
    }
    table_files.push_back(path);
    return std::make_shared<const infer::WeightRows>(type, columns, rows, std::make_shared<const format::FileReader>(path.u8string()), 0, size);
}

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
    int holds = 0, hold_calls = 0;
    void hold_between_submissions(bool on) override {
        holds += on ? 1 : -1;
        ++hold_calls;
    }
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

// Embedding and layer 0's attention on A, layer 0's feed-forward and layer 1's attention on B, layer 1's feed-forward and the head on A: two crossings per pass, both inside a layer, none at the layer boundary.
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
    // Three passes for five tokens at ubatch 2, two crossings each, one in each direction, each a copy out of the source and a write into the destination; the residual ends layer 0 on B, so the stage boundary crosses nothing.
    // Each stage submits the devices it records on, both here, and a crossing its source: three submissions a pass on A and on B, one on the single device.
    require(a->copies == 3 && b->writes == 3 && b->copies == 3 && a->writes == 3,
            "crossings are not where the placement changes");
    require(a->submits == 9 && b->submits == 9 && one->submits == 3, "submissions are not one per stage and crossing");
    require(one->copies == 0 && one->writes == 0, "a single device crossed");
    // Each device of the split waits while the other runs its part, so the model asks both to hold between submissions, and the single device not.
    require(a->holds == 1 && b->holds == 1 && one->holds == 0, "holds not asked of exactly a split's devices");
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

    // Two models over the same backends each ask, so the backends hold while either is alive.
    require(a->holds == 2 && b->holds == 2, "a second model over the split did not ask for holds");

    // A model gives its request back when it goes; one using one of its two devices, and one whose construction fails, ask nothing.
    auto c = std::make_shared<CountingCpu>(), d = std::make_shared<CountingCpu>();
    for (auto& x : {c, d}) x->set_threads(1);
    {
        infer::Model held(infer::gguf_weights(weights), {c, d}, p);
        require(c->holds == 1 && d->holds == 1, "a split did not ask for holds");
    }
    require(c->holds == 0 && d->holds == 0 && c->hold_calls == 2 && d->hold_calls == 2, "a model gone kept its holds");
    infer::Placement all_on_one;
    all_on_one.mixer_device = all_on_one.ffn_device = {1, 1};
    all_on_one.embed_device = all_on_one.output_device = 1;
    infer::Model alone(infer::gguf_weights(weights), {c, d}, all_on_one);
    require(c->hold_calls == 2 && d->hold_calls == 2, "a placement on one of its devices asked for holds");
    const auto three = tiny_qwen(3, 2 * 128, true);
    infer::Placement apart_runs;
    apart_runs.mixer_device = apart_runs.ffn_device = {0, 1, 0};
    bool refused = false;
    try { infer::Model broken(infer::gguf_weights(three), {c, d}, apart_runs); }
    catch (const std::runtime_error&) { refused = true; }
    require(refused && c->hold_calls == 2 && d->hold_calls == 2, "a model refused during construction asked for holds");
    checked += 3;
}

// Host scratch constrains the fit even though the host borrows its weights and position tables.
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

// A head may own auxiliary projections as well as its canonical logits matrix and norm.
void head_aux_footprint_fits() {
    const gguf::GGUFModel file = fixture();
    infer::ModelWeights weights = infer::gguf_weights(file);
    infer::ModelPlan plan = infer::plan_model(weights);
    const infer::ModelOptions options;
    const infer::Footprint base = infer::footprint(weights, plan, options);

    auto add_head_weight = [&](const char* name, std::vector<uint64_t> shape, size_t bytes, uint64_t in, uint64_t out,
                               bool split_rows) {
        const size_t index = weights.tensors.size();
        weights.tensors.push_back(infer::TensorView{name, std::move(shape), quant::GGML_TYPE_F32, nullptr, bytes});
        infer::Role role{static_cast<uint16_t>(plan.role_ids++), infer::Part::head, infer::RoleKind::matrix,
                         name, "", in, out};
        role.tensor = index;
        if (split_rows) {
            role.shard.axis = infer::Axis::rows;
            role.shard.sections.push_back(infer::ShardSection{out, 1, 1, "head auxiliary rows"});
        }
        plan.pass.push_back(std::move(role));
    };
    add_head_weight("output_hc_down.weight", {4, 3}, 48, 4, 3, false);
    add_head_weight("output_hc_up.weight", {4, 4}, 64, 4, 4, true);

    const infer::Footprint with_aux = infer::footprint(weights, plan, options);
    require(with_aux.tied && with_aux.head_aux.size() == 2 && with_aux.head_aux[0].bytes == 48 &&
                with_aux.head_aux[0].product && with_aux.head_aux[1].bytes == 64 && with_aux.head_aux[1].product,
            "additional head projections were not captured by the footprint");

    auto budget = [](const char* name) {
        infer::DeviceBudget d;
        d.name = name;
        d.bytes = size_t(1) << 30;
        return d;
    };
    const auto base_one = infer::split_layers(base, {budget("one")}, 1);
    const auto aux_one = infer::split_layers(with_aux, {budget("one")}, 1);
    require(aux_one.stages[0].weights == base_one.stages[0].weights + 48 + 64,
            "a tied first-and-last stage did not count its auxiliary head weights");

    const auto base_split = infer::split_layers(base, {budget("first"), budget("last")}, 1, {1, 1});
    const auto aux_split = infer::split_layers(with_aux, {budget("first"), budget("last")}, 1, {1, 1});
    require(base_split.stages[0].count == 1 && aux_split.stages[0].count == 1 &&
                aux_split.stages[0].weights == base_split.stages[0].weights &&
                aux_split.stages[1].weights == base_split.stages[1].weights + 48 + 64,
            "auxiliary head weights were not charged only to the separate last stage");

    const infer::Footprint member0 = infer::footprint(weights, plan, options, 2, 0);
    const infer::Footprint member1 = infer::footprint(weights, plan, options, 2, 1);
    const infer::Footprint group = infer::footprint(weights, plan, options, 2);
    require(member0.head_aux.size() == 2 && member1.head_aux.size() == 2 && group.head_aux.size() == 2 &&
                member0.head_aux[0].bytes == 48 && member1.head_aux[0].bytes == 48 &&
                member0.head_aux[1].bytes == 32 && member1.head_aux[1].bytes == 32 &&
                group.head_aux[0].bytes == 48 && group.head_aux[1].bytes == 32,
            "a tensor group did not combine each member's auxiliary head weights");
    checked += 4;
}

void lookup_scratch_fits() {
    const gguf::GGUFModel file = fixture();
    const infer::ModelWeights weights = infer::gguf_weights(file);
    infer::ModelPlan plan = infer::plan_model(weights);
    const infer::ModelOptions options;
    const auto original = infer::footprint(weights, plan, options);
    plan.index_width = 7;
    const auto indexed = infer::footprint(weights, plan, options);
    infer::DeviceBudget device;
    device.bytes = size_t(1) << 30;
    const size_t rows = 11;
    for (size_t count : {size_t(1), size_t(2)}) {
        const std::vector<infer::DeviceBudget> devices(count, device);
        const std::vector<int> shares(count, 1);
        const auto base = infer::split_layers(original, devices, rows, shares);
        const auto with_indices = infer::split_layers(indexed, devices, rows, shares);
        require(with_indices.host == base.host + count * rows * 7 * sizeof(uint32_t),
                "lookup scratch is not charged to the host once per used stage");
    }
    const auto grouped = infer::footprint(weights, plan, options, 2);
    require(grouped.host_scratch_per_row == 2 * indexed.host_scratch_per_row,
            "lookup scratch is not counted for every tensor-group member");
    checked += 3;
}

void heterogeneous_saved_fits() {
    const gguf::GGUFModel file = fixture();
    const infer::ModelWeights weights = infer::gguf_weights(file);
    infer::ModelPlan plan = infer::plan_model(weights);
    plan.state = {2, 2, 4, 4};
    plan.slots = {8, 6};
    for (auto& layer : plan.layers) {
        layer.cache = infer::Cache::state;
        layer.state_tail = 0;
    }
    plan.layers[0].saved = {{0, 8, 0, true}, {1, 6}};
    plan.layers[0].recur_writes = {0, 1};
    plan.layers[1].saved = {{1, 6}};
    plan.layers[1].recur_writes = {1};
    infer::ModelOptions options;
    options.mark_slots = 2;
    options.mark_rows = 3;
    for (size_t width : {size_t(1), size_t(2)}) {
        const infer::Footprint fp = infer::footprint(weights, plan, options, width);
        const size_t saved = 8 + 6 / width;
        const size_t rerun = 8 + 6;
        const size_t slots = options.state_slots + options.checkpoint_slots + options.mark_slots;
        const size_t state = infer::shard::state(plan, width).slot_floats();
        const size_t expected = (slots * state + options.mark_slots * options.mark_rows * saved + options.mark_rows * rerun) * sizeof(float);
        require(fp.cache.size() == 2 && fp.cache[0] == expected && fp.cache[1] == expected,
                "heterogeneous saved and rerun rows were not conservatively fitted per state layer");
        ++checked;
    }
}

void external_lookup_fits() {
    const auto file = fixture();
    infer::ModelWeights weights = infer::gguf_weights(file);
    infer::ModelPlan plan = infer::plan_model(weights);
    const uint32_t type = quant::GGML_TYPE_Q8_0;
    const size_t columns = 32, rows_in_table = 8;
    const size_t row_bytes = quant::row_bytes(type, columns);
    std::vector<uint8_t> table(row_bytes * rows_in_table);
    const size_t tensor = weights.tensors.size();
    weights.tensors.push_back(infer::TensorView{"external.lookup", {columns, rows_in_table}, type, table.data(), table.size()});
    infer::Role role{static_cast<uint16_t>(plan.role_ids++), infer::Part::mixer, infer::RoleKind::gather,
                     "external.lookup", "", columns, rows_in_table};
    role.tensor = tensor;
    plan.layers[0].roles.push_back(role);
    plan.layers[0].lookup = infer::Lookup{role.id, 2};

    const infer::ModelOptions options;
    const infer::Footprint mapped = infer::footprint(weights, plan, options);
    weights.tensors[tensor].row_source = table_source(type, columns, rows_in_table, table.data(), table.size());
    const infer::Footprint external = infer::footprint(weights, plan, options);
    size_t mapped_layer_bytes = 0, external_layer_bytes = 0;
    for (const auto& matrix : mapped.layers[0]) mapped_layer_bytes += matrix.bytes;
    for (const auto& matrix : external.layers[0]) external_layer_bytes += matrix.bytes;
    require(mapped.lookups[0].rows_per_token == 0 && external.lookups[0].rows_per_token == 2 &&
                external.lookups[0].row_bytes == row_bytes && external.lookups[0].io_bytes == 0 &&
                mapped_layer_bytes == external_layer_bytes + table.size(),
            "an external lookup was not separated from its full-table weight cost");
    const infer::Footprint group = infer::footprint(weights, plan, options, 2);
    require(group.lookups[0].rows_per_token == 2 && group.lookups[0].row_bytes == row_bytes,
            "a tensor group did not retain its external lookup footprint");

    infer::Footprint fit;
    fit.layers.resize(2);
    fit.cache = {0, 0};
    fit.lookups.resize(2);
    const size_t page = core::page_size();
    fit.lookups[0] = {2, row_bytes, page + 1};
    const size_t normal_rows = 3, apart_rows = 2;
    const size_t device_capacity = (normal_rows + apart_rows) * 2 * row_bytes;
    const auto rounded = [page](size_t bytes) { return (bytes + page - 1) / page * page; };
    const auto host_capacity = [&](size_t token_rows) {
        if (!token_rows) return size_t(0);
        const size_t count = token_rows * 2;
        return rounded(count * row_bytes) + count * (2 * sizeof(size_t) + 2 * sizeof(uint32_t));
    };
    // The table's read windows are counted once, whatever scratches read it (WeightRows::staging_bytes).
    const size_t windows = rounded(page + 1);
    const size_t host_capacity_total = host_capacity(normal_rows) + host_capacity(apart_rows) + windows;
    auto device = [](size_t bytes) {
        infer::DeviceBudget d;
        d.name = "device";
        d.bytes = bytes;
        return d;
    };
    const auto normal_only = infer::split_layers(fit, {device(normal_rows * 2 * row_bytes), device(size_t(1) << 20)}, normal_rows,
                                                 {1, 1}, host_capacity(normal_rows) + windows, 1, size_t(0));
    require(normal_only.stages[0].other == normal_rows * 2 * row_bytes && normal_only.host == host_capacity(normal_rows) + windows,
            "a normal-only lookup did not fit its rows and the table's rounded read windows");
    const auto split = infer::split_layers(fit, {device(device_capacity), device(size_t(1) << 20)}, 5, {1, 1},
                                           host_capacity_total, 1, size_t(0), apart_rows);
    require(split.stages[0].other == device_capacity && split.stages[1].other == 0 && split.host == host_capacity_total,
            "external lookup memory was not charged to its layer device, separate host rows and the table's windows once");
    bool device_short = false;
    try {
        infer::split_layers(fit, {device(device_capacity - 1), device(size_t(1) << 20)}, 5, {1, 1},
                            host_capacity_total, 1, size_t(0), apart_rows);
    } catch (const std::runtime_error&) { device_short = true; }
    require(device_short, "a device one byte short of packed lookup capacity was accepted");
    bool host_short = false;
    try {
        infer::split_layers(fit, {device(device_capacity), device(size_t(1) << 20)}, 5, {1, 1},
                            host_capacity_total - 1, 1, size_t(0), apart_rows);
    } catch (const std::runtime_error&) { host_short = true; }
    require(host_short, "host lookup pages and index vectors were not fitted exactly");
    checked += 6;
}

// The layer split fitted to device budgets (model/layer_split.hpp): even shares where room allows, a device without room left out, a host given only what the others cannot hold, tied weights counted once, and the fit exact against one device.
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

    // The one placement entry (infer::place_model): budgets asked of the backends, a split by shares exact against one device, the request's ubatch applied, and the refusals of experts on the CPU and of a stream point without them.
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
    require(asked.size() == 2 && asked[0].host && asked[0].scratch == asked[0].bytes.value_or(0) / 20 && asked[0].resident,
            "a CPU's budget and scratch reserve not asked of the backend");
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
// With backends playing devices every stage past the first waits on its devices, so the prompt records it on a thread of its own (Model::run_chunks), and those stages must have submitted from no thread but theirs.
void pipelined_matches_single(bool devices) {
    for (bool tied : {true, false}) {
        const auto weights = tiny_qwen(3, 2 * 128, tied);
        for (const PipelinedSplit& ps : kPipelined) {
            auto one = std::make_shared<backend::CpuBackend>();
            one->set_threads(1);
            infer::Model single(infer::gguf_weights(weights), one);
            single.set_ubatch(3);
            std::vector<std::shared_ptr<FailingCpu>> stages;
            std::vector<backend::BackendPtr> cpus;
            for (size_t i = 0; i < ps.shares.size(); ++i) {
                stages.push_back(std::make_shared<FailingCpu>());
                stages.back()->plays_device = devices;
                cpus.push_back(stages.back());
            }
            infer::PlacedModel placed = pipelined(weights, cpus, ps.shares);
            infer::Model& split = *placed.model;
            const size_t V = single.n_vocab();

            for (auto& s : stages) s->submitters.clear();
            exact(single.prefill(kPrompt), split.prefill(kPrompt), "pipelined prefill differs from one device");
            for (size_t i = 1; i < stages.size(); ++i)
                require(!stages[i]->submitters.empty() && (stages[i]->submitters.count(std::this_thread::get_id()) == 0) == devices,
                        devices ? "a stage that waits on its devices was recorded on the prompt's thread" : "a stage on the host was recorded on another thread");
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
// Every storage must be back at the history and the same prompt again exact; with backends playing devices the last stage fails on its own thread, the first stage then at the failing chunk at least.
void pipelined_failure_rolls_back(bool devices) {
    const auto weights = tiny_qwen(3, 2 * 128, true);
    for (const PipelinedSplit& ps : kPipelined) {
        auto plain = std::make_shared<backend::CpuBackend>();
        plain->set_threads(1);
        infer::Model control(infer::gguf_weights(weights), plain);
        control.set_ubatch(3);
        std::vector<std::shared_ptr<FailingCpu>> stages;
        for (size_t i = 0; i < ps.shares.size(); ++i) {
            stages.push_back(std::make_shared<FailingCpu>());
            stages.back()->plays_device = devices;
        }
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
            require(last.histories.back() == before + (at_head ? 12 : 6) && first.histories.back() >= before + (devices ? (at_head ? 12 : 6) : 9),
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
    bool copying = false, unknown = false;
    mutable int rise_after = 0;
    bool memory_is_host() const override { return false; }
    std::optional<size_t> memory_available() const override {
        if (unknown) return std::nullopt;
        return rise_after-- > 0 ? 1 : room;
    }
    bool reads_in_place() const override { return !copying; }
};

// A normal tiny architecture with an additional declared table, used to exercise storage planning without recording a graph.
struct LookupArchitecture : infer::Architecture {
    std::shared_ptr<const infer::Architecture> base;
    explicit LookupArchitecture(std::shared_ptr<const infer::Architecture> b) : base(std::move(b)) {}
    infer::ModelPlan plan(const infer::TensorIndex& tensors) const override {
        infer::ModelPlan p = base->plan(tensors);
        const size_t tensor = tensors.at("placement.lookup");
        const auto& t = tensors.view(tensor);
        infer::Role role{static_cast<uint16_t>(p.role_ids++), infer::Part::mixer, infer::RoleKind::gather,
                         t.name, "", t.shape[0], t.shape[1]};
        role.tensor = tensor;
        p.layers[0].roles.push_back(role);
        p.layers[0].lookup = infer::Lookup{role.id, 2};
        return p;
    }
    void fill_tables(std::vector<std::vector<float>>& tables) const override { base->fill_tables(tables); }
    void embed(const infer::Step& s, const uint32_t* ids) const override { base->embed(s, ids); }
    void mixer(const infer::Step& s) const override { base->mixer(s); }
    void ffn(const infer::Step& s) const override { base->ffn(s); }
    void head(const infer::HeadStep& s) const override { base->head(s); }
};

void lookup_storage_planned() {
    const auto file = fixture();
    infer::ModelWeights weights = infer::gguf_weights(file);
    weights.arch = std::make_shared<LookupArchitecture>(weights.arch);
    const size_t columns = 16, rows = size_t(1) << 18;
    std::vector<float> table(columns * rows);
    const size_t tensor = weights.tensors.size();
    weights.tensors.push_back({"placement.lookup", {columns, rows}, quant::GGML_TYPE_F32,
                              reinterpret_cast<const uint8_t*>(table.data()), table.size() * sizeof(float)});
    const auto source = table_source(quant::GGML_TYPE_F32, columns, rows, weights.tensors[tensor].data, weights.tensors[tensor].bytes);
    const std::vector<infer::LookupCandidate> candidates{{tensor, source}};
    struct PlanningDevice : SizedDevice {
        size_t allocations = 0, adoptions = 0;
        size_t scratch_reserve(size_t) const override { return 0; }
        backend::BufferPtr alloc(size_t bytes, backend::Memory where = backend::Memory::device) override {
            ++allocations;
            return SizedDevice::alloc(bytes, where);
        }
        backend::BufferPtr adopt(const void* data, size_t bytes) override {
            ++adoptions;
            return SizedDevice::adopt(data, bytes);
        }
    };
    auto device = std::make_shared<PlanningDevice>();
    device->room = size_t(64) << 20;
    device->copying = true;
    device->set_threads(1);
    infer::PlacementRequest request;
    request.names = {"device"};
    request.ubatch = 2;
    infer::ModelOptions options;
    const auto resident = infer::plan_placement(weights, {device}, request, options, candidates);
    require(resident.lookups == std::vector<infer::LookupStorage>{infer::LookupStorage::device} &&
                !weights.tensors[tensor].row_source && device->allocations == 0 && device->adoptions == 0 && device->writes == 0,
            "lookup planning allocated model buffers, mutated its input or failed to preserve device residency");
    const int reads = device->rise_after;
    require(reads == -1, "lookup candidates did not share one device budget reading");

    device->room = size_t(8) << 20;
    request.fit_kv = true;
    const auto hosted = infer::plan_placement(weights, {device}, request, options, candidates);
    require(hosted.lookups == std::vector<infer::LookupStorage>{infer::LookupStorage::file} && hosted.options.kv_tokens == resident.architecture.context_length,
            "automatic lookup storage cut KV rather than reading a table that does not fit from its file");
    request.lookup_storage = infer::LookupStorage::file;
    auto from_file = infer::plan_placement(weights, {device}, request, options, candidates);
    require(from_file.lookups == std::vector<infer::LookupStorage>{infer::LookupStorage::file} &&
                from_file.options.kv_tokens == hosted.options.kv_tokens,
            "explicit file lookup storage changed the fitted KV budget");
    request.lookup_storage = infer::LookupStorage::device;
    request.fit_kv = false;
    bool device_short = false;
    try { infer::plan_placement(weights, {device}, request, options, candidates); }
    catch (const std::runtime_error& e) { device_short = std::string(e.what()).find("lookup storage device does not fit beside the KV budget") != std::string::npos; }
    require(device_short && device->allocations == 0 && device->adoptions == 0,
            "explicit device lookup storage was not refused before allocation at the bounded fit's KV budget");

    // Beside an unused host device, automatic storage still reads a table that fits no device from its file.
    auto host = std::make_shared<SizedDevice>();
    host->room = size_t(1) << 20;
    host->set_threads(1);
    request.names = {"host", "device"};
    request.shares = {0, 1};
    request.fit_kv = false;
    request.lookup_storage = infer::LookupStorage::automatic;
    const auto bounded = infer::plan_placement(weights, {host, device}, request, options, candidates);
    require(bounded.lookups == std::vector<infer::LookupStorage>{infer::LookupStorage::file},
            "automatic lookup storage kept a table on a device it exceeds");

    request.names = {"device"};
    request.shares.clear();
    const auto refused = [&](const std::vector<infer::LookupCandidate>& bad) {
        try { infer::plan_placement(weights, {device}, request, options, bad); }
        catch (const std::runtime_error&) { return true; }
        return false;
    };
    require(refused({{tensor, source}, {tensor, source}}) && refused({{tensor + 1, source}}) && refused({{tensor, {}}}),
            "invalid lookup candidates reached a placement probe");
    const auto wrong = table_source(quant::GGML_TYPE_F32, columns / 2, rows * 2, weights.tensors[tensor].data, weights.tensors[tensor].bytes);
    require(refused({{tensor, wrong}}), "a lookup source with another row geometry reached a placement probe");
    for (infer::LookupStorage storage : {infer::LookupStorage::automatic, infer::LookupStorage::device, infer::LookupStorage::file})
        require(infer::lookup_storage_of(infer::lookup_storage_name(storage)) == storage, "lookup storage policy names do not round trip");

    // A one-row table fits where a large pass's packed row capacity does not, so residency must be considered before requiring that capacity.
    infer::ModelWeights tiny = weights;
    tiny.tensors[tensor].shape[1] = 1;
    tiny.tensors[tensor].bytes = columns * sizeof(float);
    const auto tiny_source = table_source(quant::GGML_TYPE_F32, columns, 1, tiny.tensors[tensor].data, tiny.tensors[tensor].bytes);
    const std::vector<infer::LookupCandidate> tiny_candidate{{tensor, tiny_source}};
    const auto tiny_plan = infer::plan_model(tiny);
    request.ubatch = 1024;
    infer::DeviceBudget generous;
    generous.name = "device";
    generous.bytes = size_t(64) << 20;
    const auto exact_fit = infer::split_layers(infer::footprint(tiny, tiny_plan, options), {generous}, request.ubatch);
    device->room = exact_fit.stages[0].weights + exact_fit.stages[0].cache + exact_fit.stages[0].other;
    generous.bytes = device->room;
    tiny.tensors[tensor].row_source = tiny_source;
    bool packed_short = false;
    try { infer::split_layers(infer::footprint(tiny, tiny_plan, options), {generous}, request.ubatch); }
    catch (const std::runtime_error&) { packed_short = true; }
    tiny.tensors[tensor].row_source.reset();
    require(packed_short, "the tiny-table fixture did not exceed its resident budget with bounded row staging");
    request.lookup_storage = infer::LookupStorage::automatic;
    const auto tiny_auto = infer::plan_placement(tiny, {device}, request, options, tiny_candidate);
    require(tiny_auto.lookups == std::vector<infer::LookupStorage>{infer::LookupStorage::device},
            "automatic lookup planning required packed staging for a smaller resident table");
    request.lookup_storage = infer::LookupStorage::device;
    request.fit_kv = true;
    const auto tiny_device = infer::plan_placement(tiny, {device}, request, options, tiny_candidate);
    require(tiny_device.lookups == std::vector<infer::LookupStorage>{infer::LookupStorage::device} &&
                tiny_device.options.kv_tokens == tiny_plan.context_length && device->allocations == 0 && device->adoptions == 0,
            "explicit device lookup fitting used nonexistent row staging or allocated before construction");

    // A larger resident table can also be the only fit when the host has no room for packed row staging.
    request.names = {"host", "device"};
    request.shares = {0, 1};
    request.ubatch = 2;
    request.fit_kv = false;
    request.lookup_storage = infer::LookupStorage::automatic;
    device->room = size_t(64) << 20;
    host->room = size_t(1) << 20;
    const auto full_plan = infer::plan_model(weights);
    const auto host_fit = infer::split_layers(infer::footprint(weights, full_plan, options), infer::budgets_for({host, device}, request.names),
                                              request.ubatch, request.shares, core::host_memory_available());
    host->room = host_fit.host;
    const auto no_staging = infer::plan_placement(weights, {host, device}, request, options, candidates);
    require(no_staging.lookups == std::vector<infer::LookupStorage>{infer::LookupStorage::device},
            "automatic lookup fitting rejected residency when only bounded host staging exceeded its budget");
    request.fit_kv = true;
    const auto no_staging_kv = infer::plan_placement(weights, {host, device}, request, options, candidates);
    require(no_staging_kv.lookups == std::vector<infer::LookupStorage>{infer::LookupStorage::device} &&
                no_staging_kv.options.kv_tokens == full_plan.context_length,
            "automatic KV fitting rejected a resident lookup that avoids bounded host staging");

    // The loader attaches the selected source after planning, then construction consumes that same plan once.
    weights.tensors[tensor].row_source = source;
    auto constructed = infer::construct_model(weights, std::move(from_file), 2);
    require(constructed.model && constructed.model->kv_tokens_total() == hosted.options.kv_tokens && device->allocations > 0,
            "a planned external lookup model was not constructed with its fitted budget");
    checked += 20;
}

// A server's KV budget (PlacementRequest::fit_kv): the request's budget where it fits, backed whole as the model is made, else the most whole blocks that fit; a load where not one block fits is refused.
// Beside experts on the CPU the device is not charged for those layers' feed-forward weights.
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

    // The automatic budget (PlacementRequest::kv_auto): every whole block the device's free memory holds, past the context of 256, the blocks an explicit budget past it is cut to.
    // The model has no embedded drafter, so nothing is reserved for one; a device of unknown free memory keeps the context.
    const auto automatic = [&](size_t room, bool unknown) {
        auto device = std::make_shared<SizedDevice>();
        device->room = room;
        device->unknown = unknown;
        device->set_threads(1);
        infer::PlacementRequest request;
        request.names = {"device"};
        request.fit_kv = request.kv_auto = true;
        return infer::place_model(infer::gguf_weights(weights), {device}, request, infer::ModelOptions{}).model->kv_tokens_total();
    };
    const size_t filled = automatic(8 << 20, false);
    require(filled == cut && filled > 256, ("the automatic budget on 8 MiB took " + std::to_string(filled) + " tokens, against the " + std::to_string(cut) + " an explicit budget past the device is cut to").c_str());
    require(automatic(8 << 20, true) == 256, "the automatic budget on a device of unknown free memory was not the model's context");
    {
        // The CPU's free memory is the host's (Backend::memory_is_host), shared with the host tier and the rest of the host, so on the CPU, and on a split with a CPU stage, the automatic budget keeps the context and says why.
        auto cpu = std::make_shared<backend::CpuBackend>();
        cpu->set_threads(1);
        infer::PlacementRequest request;
        request.names = {"cpu"};
        request.fit_kv = request.kv_auto = true;
        const infer::PlacedModel on_cpu = infer::place_model(infer::gguf_weights(weights), {cpu}, request, infer::ModelOptions{});
        require(on_cpu.model->kv_tokens_total() == 256 && on_cpu.kv_kept_context.find("the host's") != std::string::npos,
                "the automatic budget on the CPU took the host's memory past the model's context, or did not say why it kept the context");
        // Over a split the stage whose free memory is the host's is a device reporting a steady figure, as the CPU's host memory, which moves between reads, would keep the split's level reading waiting.
        struct HostLike : SizedDevice {
            bool memory_is_host() const override { return true; }
        };
        auto device = std::make_shared<SizedDevice>();
        device->room = 8 << 20;
        device->set_threads(1);
        auto other = std::make_shared<HostLike>();
        other->room = 8 << 20;
        other->set_threads(1);
        infer::PlacementRequest split = request;
        split.names = {"device", "host"};
        split.shares = {1, 1};
        const infer::PlacedModel beside = infer::place_model(infer::gguf_weights(weights), {device, other}, split, infer::ModelOptions{});
        require(beside.model->kv_tokens_total() == 256 && !beside.kv_kept_context.empty(), "the automatic budget over a device and a CPU stage did not keep the model's context");
        const infer::PlacedModel unknown = [&] {
            auto d = std::make_shared<SizedDevice>();
            d->unknown = true;
            d->set_threads(1);
            return infer::place_model(infer::gguf_weights(weights), {d}, request, infer::ModelOptions{});
        }();
        require(unknown.kv_kept_context.find("no free memory") != std::string::npos, "the automatic budget on a device of unknown memory did not say why it kept the context");
    }
    // Its search bound: the context where the memory is unknown, a token costs no KV bytes (states alone) or the memory holds less, else the whole blocks the memory holds.
    // At the edge of a size_t it is never a number whose product with a token's bytes passes the free memory, nor more blocks than an int32 block id names.
    require(infer::kv_search_bound(256, std::nullopt, 100, 128) == 256 && infer::kv_search_bound(256, size_t(1) << 20, 0, 128) == 256 &&
                infer::kv_search_bound(256, 1000, 100, 128) == 256 && infer::kv_search_bound(256, size_t(1) << 20, 64, 128) == 16384,
            "the automatic budget's search bound was not the context or the whole blocks the memory holds");
    const size_t per = SIZE_MAX / (size_t(1) << 30), edge = infer::kv_search_bound(256, SIZE_MAX, per, 128);
    require(edge % 128 == 0 && edge <= SIZE_MAX / per && edge + 128 > SIZE_MAX / per, "the search bound at the edge of a size_t was not the most whole blocks whose bytes it states");
    require(infer::kv_search_bound(256, SIZE_MAX, 1, 128) == size_t(std::numeric_limits<int32_t>::max()) * 128, "the search bound passed the blocks an int32 block id names");
    checked += 3;
}

// A history recomputed in the classes that first computed it, over two CPU stages, as a paused request's resume does (docs/SERVER.md, pausing): a 40-token prompt in slices of 16, then 199 greedy tokens as entries of extent 1 of up to 64 rows.
// The synthetic Q8_0 model's decode rows take the 8-bit dots, so the replay, whole and from a fork at the first block, must give the logits one backend gives after the prompt and 199 single decode steps.
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

// What the pass API refuses, each before any work and with nothing changed: contexts not reserved or reserved twice, slots beyond or in use, too many rows, a sequence listed twice or in flight, and stages or logits out of order.
// A reservation the devices cannot hold fails with their error and leaves the context fresh, so a smaller one succeeds, and ended, aborted and rerun passes leave the histories and rows of the sequences run alone.
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

// A prompt in consecutive passes at consecutive stages against prefill on one device, bit for bit, over 2, 3 and 4 stages (docs/MULTI-DEVICE.md, phase 4).
// Each refusal changes nothing, and abandoning a slice, or a failure at its last stage, leaves the history where the first slice found it.
void passes_continue() {
    const auto weights = tiny_qwen(4, 2 * 128, true);
    const std::vector<std::vector<int>> splits{{1, 1}, {1, 1, 2}, {1, 1, 1, 1}};
    for (const std::vector<int>& shares : splits) {
        const size_t S = shares.size();
        int total = 0;
        for (int sh : shares) total += sh;
        const int last_layers = 4 * shares.back() / total;   // the layers the last stage runs, an attention each a slice
        auto one = std::make_shared<backend::CpuBackend>();
        one->set_threads(1);
        infer::Model single(infer::gguf_weights(weights), one);
        single.set_ubatch(3);
        std::vector<std::shared_ptr<FailingCpu>> stages;
        for (size_t i = 0; i < S; ++i) stages.push_back(std::make_shared<FailingCpu>());
        infer::PlacedModel placed = pipelined(weights, std::vector<backend::BackendPtr>(stages.begin(), stages.end()), shares);
        infer::Model& m = *placed.model;
        const size_t V = single.n_vocab();
        infer::ExecContext ctx;
        m.reserve_passes(ctx, S, 3, 1);
        infer::Sequence q = m.make_sequence(), other = m.make_sequence();
        const std::vector<uint32_t>& ids = kPrompt;
        std::vector<infer::BatchEntry> chunk;
        for (size_t i = 0; i < ids.size(); i += 3) {
            const size_t n = std::min<size_t>(3, ids.size() - i);
            infer::BatchEntry e{&q, ids.data() + i, n, i + n == ids.size()};
            e.extent = ids.size();
            chunk.push_back(e);
        }
        const size_t C = chunk.size();
        auto refused = [&](const std::function<void()>& call, const char* what) {
            bool caught = false;
            try { call(); } catch (const std::logic_error&) { caught = true; }
            require(caught, what);
            ++checked;
        };
        // Stage s of chunk t - s at step t, the first stage first, as prefill runs them; a chunk is begun as its first stage runs and ended after its last.
        const auto run = [&](size_t until) {
            std::vector<float> logits;
            for (size_t t = 0; t + 1 < C + S && t < until; ++t)
                for (size_t st = 0; st < S; ++st) {
                    if (t < st || t - st >= C) continue;
                    const size_t c = t - st, slot = c % S;
                    if (st == 0) m.begin_pass(ctx, slot, &chunk[c], 1, 0);
                    m.run_pass_stage(ctx, slot, st);
                    if (st + 1 < S) continue;
                    if (chunk[c].want_logits) logits.assign(m.pass_logits(ctx, slot, 0), m.pass_logits(ctx, slot, 0) + V);
                    m.end_pass(ctx, slot);
                }
            return logits;
        };
        exact(single.prefill(ids), run(SIZE_MAX), "a prompt in consecutive passes differs from prefill on one device");
        require(!q.in_flight() && q.length() == ids.size(), "a prompt in consecutive passes left its sequence in flight or short");
        const uint32_t next = 7;
        const infer::BatchEntry step{&q, &next, 1, true};
        m.begin_pass(ctx, 0, &step, 1, 0);
        for (size_t st = 0; st < S; ++st) m.run_pass_stage(ctx, 0, st);
        require(!std::memcmp(m.pass_logits(ctx, 0, 0), single.step(next).data(), V * sizeof(float)), "a step after a prompt in consecutive passes differs from one device");
        m.end_pass(ctx, 0);
        checked += 2;

        // Refusals, each leaving the history and the passes in flight as they were.
        m.reset(q);
        single.reset();
        m.begin_pass(ctx, 0, &chunk[0], 1, 0);
        refused([&] { m.begin_pass(ctx, 1, &chunk[1], 1, 0); }, "a slice begun before the one before it ran its first stage");
        m.run_pass_stage(ctx, 0, 0);
        const uint32_t lone = 5;
        const infer::BatchEntry beside[2] = {chunk[1], {&other, &lone, 1, true}};
        refused([&] { m.begin_pass(ctx, 1, beside, 2, 0); }, "a continued sequence beside another in its pass");
        require(!other.in_flight() && q.length() == 3, "a refused continuation changed a sequence");
        m.begin_pass(ctx, 1, &chunk[1], 1, 0);
        m.run_pass_stage(ctx, 1, 0);
        refused([&] { m.run_pass_stage(ctx, 1, 1); }, "a slice's stage run before the slice before it ran that stage");
        require(q.in_flight() && q.length() == 6, "a refused stage changed the history or took its pass out of flight");
        // Abandoning the first slice takes the second with it: both slots free, the history empty.
        const std::vector<size_t> later = m.abort_pass(ctx, 0);
        require(later == std::vector<size_t>{1} && !q.in_flight() && q.length() == 0, "abandoning a slice did not take the later one with it or left a history");
        refused([&] { m.abort_pass(ctx, 1); }, "a slice abandoned with an earlier one was still in flight");
        exact(single.prefill(ids), run(SIZE_MAX), "a prompt after its slices were abandoned differs from one device");
        checked += 3;

        // A slice that wants its logits takes no continuation.
        m.reset(q);
        single.reset();
        infer::BatchEntry wanting = chunk[0];
        wanting.want_logits = true;
        m.begin_pass(ctx, 0, &wanting, 1, 0);
        m.run_pass_stage(ctx, 0, 0);
        refused([&] { m.begin_pass(ctx, 1, &chunk[1], 1, 0); }, "a continuation beside a pass that wants the sequence's logits");
        m.abort_pass(ctx, 0);

        // Nor does a slice whose pass holds another sequence, which abandoning a later slice would otherwise take with it.
        infer::BatchEntry head{&q, ids.data(), 2, false};
        head.extent = ids.size();
        const infer::BatchEntry shared[2] = {head, {&other, &lone, 1, false}};
        m.begin_pass(ctx, 0, shared, 2, 0);
        m.run_pass_stage(ctx, 0, 0);
        refused([&] { m.begin_pass(ctx, 1, &chunk[1], 1, 0); }, "a continuation of a slice whose pass holds another sequence");
        require(!ctx.passes[1].in_flight && q.length() == 2 && other.length() == 1, "a refused continuation changed a sequence or took a slot");
        m.abort_pass(ctx, 0);
        m.reset(other);

        // A failure at the last stage on the second slice: that slice gone, the later ones refused at their next stage and abandoned, the history back where the first found it.
        m.reset(q);
        single.reset();
        stages.back()->fail_attention = last_layers + 1;
        bool failed = false;
        try { run(SIZE_MAX); } catch (const std::runtime_error&) { failed = true; }
        require(failed && !stages.back()->fail_attention, "the injected failure did not fire");
        std::vector<size_t> live;
        for (size_t slot = 0; slot < S; ++slot)
            if (ctx.passes[slot].in_flight) live.push_back(slot);
        require(!live.empty(), "no later slice was in flight at the failure");
        std::sort(live.begin(), live.end(), [&](size_t a, size_t b) { return ctx.passes[a].start[0] < ctx.passes[b].start[0]; });
        refused([&] { m.run_pass_stage(ctx, live.front(), ctx.passes[live.front()].ran); }, "a slice after a failed one took its next stage");
        const std::vector<size_t> rest = m.abort_pass(ctx, live.front());
        require(rest.size() + 1 == live.size() && !q.in_flight(), "abandoning the slice after a failed one did not take the rest");
        require(q.length() == 3, "a failed slice did not leave the history where it found it");
        m.reset(q);
        exact(single.prefill(ids), run(SIZE_MAX), "a prompt after a failed slice differs from one device");
        checked += 4;
    }
}

// A tensor split (docs/TENSOR-SPLIT.md, step 2): one stage of width 2 against two stages of width 2 on four CPU backends, bit for bit, through prompts, decode steps, scored rows and a two-sequence pass; against one device within an F32 bound.
void tensor_groups() {
    auto grouped = [](const gguf::GGUFModel& weights, size_t devices, const std::vector<int>& shares, int ubatch) {
        std::vector<backend::BackendPtr> cpus;
        for (size_t i = 0; i < devices; ++i) {
            cpus.push_back(std::make_shared<backend::CpuBackend>());
            cpus.back()->set_threads(1);
        }
        infer::PlacementRequest request;
        request.names.assign(devices, "cpu");
        request.shares = shares;
        request.ubatch = ubatch;
        request.width = 2;
        return infer::place_model(infer::gguf_weights(weights), std::move(cpus), request, infer::ModelOptions{});
    };
    for (bool tied : {true, false}) {
        const auto weights = tiny_qwen(3, 2 * 128, tied);
        infer::PlacedModel one_stage = grouped(weights, 2, {}, 3), two_stages = grouped(weights, 4, {1, 2}, 3), whole = grouped(weights, 2, {}, 13);
        infer::Model& a = *one_stage.model;
        infer::Model& b = *two_stages.model;
        auto cpu = std::make_shared<backend::CpuBackend>();
        cpu->set_threads(1);
        infer::Model single(infer::gguf_weights(weights), cpu);
        single.set_ubatch(3);
        const size_t V = a.n_vocab();
        require(b.stage_count() == 2 && b.pipelined() && a.stage_count() == 1, "the groups did not form the stages asked for");
        auto close = [](const std::vector<float>& x, const std::vector<float>& y) {
            bool ok = x.size() == y.size();
            for (size_t i = 0; ok && i < x.size(); ++i) ok = std::fabs(x[i] - y[i]) <= 1e-4f * (1.0f + std::fabs(y[i]));
            return ok;
        };
        const std::vector<float> first = a.prefill(kPrompt);
        exact(first, b.prefill(kPrompt), "two stages of width 2 differ from one stage of width 2 on a prompt");
        exact(first, whole.model->prefill(kPrompt), "a prompt in slices of the ubatch differs from the prompt in one pass on a group");
        require(close(first, single.prefill(kPrompt)), "a group's prompt is not one device's within the F32 bound");
        for (int t : {7, 9, 3}) {
            const std::vector<float> x = a.step(t);
            exact(x, b.step(t), "a decode step differs between one and two stages of width 2");
            require(close(x, single.step(t)), "a group's decode step is not one device's within the F32 bound");
        }
        const std::vector<uint32_t> more{2, 7, 1, 8, 2, 8, 1};
        exact(a.prefill(more), b.prefill(more), "a prompt continuing a history differs between one and two stages of width 2");
        std::vector<float> scored;
        a.score(kPrompt, [&](size_t, const float* logits) { scored.insert(scored.end(), logits, logits + V); });
        bool same = true;
        b.score(kPrompt, [&](size_t pos, const float* logits) { same = same && !std::memcmp(logits, &scored[pos * V], V * sizeof(float)); });
        require(same, "a scored row differs between one and two stages of width 2");
        a.reset();
        b.reset();
        infer::Sequence sa = a.make_sequence(), ta = a.make_sequence(), sb = b.make_sequence(), tb = b.make_sequence();
        infer::ExecContext xa, xb;
        const infer::BatchEntry ha{&sa, kPrompt.data(), 4, false}, hb{&sb, kPrompt.data(), 4, false};
        a.forward(xa, &ha, 1);
        b.forward(xb, &hb, 1);
        const uint32_t next = 6;
        const infer::BatchEntry ea[2] = {{&sa, &next, 1, true}, {&ta, more.data(), more.size(), true}};
        const infer::BatchEntry eb[2] = {{&sb, &next, 1, true}, {&tb, more.data(), more.size(), true}};
        a.forward(xa, ea, 2);
        b.forward(xb, eb, 2);
        for (size_t r = 0; r < 2; ++r) require(!std::memcmp(xa.logits(r), xb.logits(r), V * sizeof(float)), "a two-sequence pass differs between one and two stages of width 2");
        checked += 6;
    }
}

// Whether the page at `p` is still the process's to read.
bool mapped(const void* p) {
#if defined(_WIN32)
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(p, &info, sizeof info) && info.State == MEM_COMMIT;
#else
    return msync(const_cast<void*>(p), 1, MS_ASYNC) == 0;
#endif
}

// A tensor group's logits rows are host pages each member imports, and a device keeps imported memory while its buffer lives, so a context must drop every view before the pages.
// A buffer that looks, as it goes, whether its pages are still mapped stands for a member's view.
void context_keeps_logits_pages() {
    struct View : backend::Buffer {
        const void* pages;
        bool* intact;
        View(const void* p, bool* ok) : pages(p), intact(ok) {}
        ~View() override { *intact = *intact && mapped(pages); }
        size_t size() const override { return 4096; }
        const void* host_ptr() const override { return pages; }
    };
    bool intact = true;
    {
        infer::ExecContext ctx;
        ctx.logits_host = core::HostPages(4096);
        ctx.logits_buf = std::make_shared<View>(ctx.logits_host.data(), &intact);
        ctx.member_rows.push_back(ctx.logits_buf);
        ctx.member_rows.push_back(std::make_shared<View>(ctx.logits_host.data(), &intact));
        require(mapped(ctx.logits_host.data()), "the context's logits pages are not mapped while it lives");
    }
    require(intact, "a context freed the host pages of its logits rows before the last buffer that imports them");
    ++checked;
}

// A CPU backend that says it is a device, as a stage fed in pieces needs; it is a CPU member for as long as its collective is made, and tells no free memory.
// It keeps the tickets waited for, and fails its next submission when told to.
struct PacedDevice : backend::CpuBackend {
    static inline thread_local bool joining = false;
    std::vector<backend::Ticket> waited;
    bool fail = false;
    bool is_cpu() const override { return joining; }
    std::optional<size_t> memory_available() const override { return std::nullopt; }
    std::unique_ptr<backend::Collective> join(const std::vector<backend::Backend*>& members, size_t rows, size_t width) override {
        joining = true;
        std::unique_ptr<backend::Collective> c = CpuBackend::join(members, rows, width);
        joining = false;
        return c;
    }
    backend::Ticket submit() override {
        if (fail) {
            fail = false;
            throw std::runtime_error("injected submission failure");
        }
        return CpuBackend::submit();
    }
    void wait(backend::Ticket t) noexcept override { waited.push_back(t); }
};

// The fit counts the second arena's rows and a logits ring for the passes apart only where a stage is fed in pieces: on one device and one group that are no CPU's, not on a CPU or over a layer split.
void apart_rows_fitted() {
    const auto devices = [](size_t n, bool cpu) {
        std::vector<backend::BackendPtr> v;
        for (size_t i = 0; i < n; ++i) v.push_back(cpu ? std::static_pointer_cast<backend::Backend>(std::make_shared<backend::CpuBackend>()) : std::make_shared<PacedDevice>());
        return v;
    };
    infer::PlacementRequest request;
    request.apart_rows = 20;
    request.logit_rows = 16;
    require(infer::unpaced(devices(1, false), request).empty() && infer::apart_rows(devices(1, false), request) == 20 && infer::fitted_logit_rows(devices(1, false), request) == std::optional<size_t>(36),
            "one device that is fed in pieces is not fitted with the second arena's rows and their logits ring");
    require(!infer::unpaced(devices(1, true), request).empty() && infer::apart_rows(devices(1, true), request) == 0 && infer::fitted_logit_rows(devices(1, true), request) == std::optional<size_t>(16),
            "a CPU is fitted with rows for passes apart");
    require(!infer::unpaced(devices(2, false), request).empty() && infer::fitted_logit_rows(devices(2, false), request) == std::optional<size_t>(16), "a layer split is fitted with rows for passes apart");
    request.width = 2;
    require(infer::unpaced(devices(2, false), request).empty() && infer::fitted_logit_rows(devices(2, false), request) == std::optional<size_t>(36), "one tensor group is not fitted with the passes apart's logits ring");
    request.logit_rows = 0;
    require(!infer::fitted_logit_rows(devices(2, false), request), "a request that names no logits rows was given some");
    // The rows of a pass the fit counts hold a verify's draft rows beside a ubatch and the decode rows, and the second arena's where a stage is fed in pieces.
    infer::PlacementRequest drafting;
    drafting.ubatch = 512;
    drafting.decode_rows = 16;
    drafting.draft_rows = 64;
    drafting.apart_rows = 80;
    require(infer::fitted_rows(devices(1, true), drafting) == 512 + 16 + 64 && infer::fitted_rows(devices(1, false), drafting) == 512 + 16 + 64 + 80,
            "the fit's rows of a pass do not hold a verify's draft rows");
    drafting.draft_rows = 0;
    drafting.ubatch = 0;
    require(infer::fitted_rows(devices(1, true), drafting) == (size_t)infer::kDefaultUbatch + 16, "the fit's rows of a pass without drafts are not a ubatch and the decode rows");
    ++checked;
}

// A pass recorded within another's stage at a piece boundary is observed once, after its ticket was waited for, whichever boundary took it, the stage's last among them, after which no piece follows.
// Where the stage then fails it is still observed once and the error escapes, and a stage that fails with no pass taken observes none; on one device and a group of two, with and without logits.
void final_piece_observed() {
    const auto weights = tiny_qwen(2, 2 * 128, true);
    const std::vector<uint32_t> prompt{1, 2, 3, 4, 5}, token{6};
    struct Case {
        int take;     // the boundary a pass is taken at: 0 none, -1 the last
        bool fail;    // the first member's next submission after that boundary fails
        bool taken;   // a pass is taken there
    };
    for (const size_t width : {size_t(1), size_t(2)})
        for (const bool logits : {true, false}) {
            size_t pieces = 0;
            for (const Case c : {Case{0, false, false}, Case{1, false, true}, Case{-1, false, true}, Case{1, true, true}, Case{-1, true, true}, Case{1, true, false}}) {
                std::vector<backend::BackendPtr> cpus;
                for (size_t d = 0; d < width; ++d) {
                    cpus.push_back(std::make_shared<PacedDevice>());
                    cpus.back()->set_threads(1);
                }
                PacedDevice& first = static_cast<PacedDevice&>(*cpus[0]);
                infer::PlacementRequest request;
                request.names.assign(width, "device");
                request.width = width;
                infer::PlacedModel placed = infer::place_model(infer::gguf_weights(weights), std::move(cpus), request, infer::ModelOptions{});
                infer::Model& m = *placed.model;
                infer::ExecContext ctx;
                m.reserve_passes(ctx, 2, 8, 4, 4);
                infer::Sequence a = m.make_sequence(), b = m.make_sequence();
                const infer::BatchEntry ea{&a, prompt.data(), prompt.size(), logits}, eb{&b, token.data(), token.size(), true};
                m.begin_pass(ctx, 0, &ea, 1, 0);
                m.begin_pass(ctx, 1, &eb, 1, 2, true);
                m.prepare_pass_stage(ctx, 0, 0);
                m.prepare_pass_stage(ctx, 1, 0);
                const size_t at = c.take < 0 ? pieces : (size_t)c.take;
                size_t boundary = 0, observed = 0;
                backend::Ticket ticket = 0;
                bool waited_first = true;
                infer::Pace pace;
                pace.ahead = false;
                pace.between = [&] {
                    if (++boundary != at) return;
                    if (c.taken) {
                        m.record_pass_within(ctx, 1, 0);
                        pace.within = ticket = m.stage_ticket(ctx, 1);
                        first.waited.clear();
                    }
                    first.fail = c.fail;
                };
                pace.passed = [&] {
                    ++observed;
                    waited_first = waited_first && std::find(first.waited.begin(), first.waited.end(), ticket) != first.waited.end();
                };
                std::string caught;
                try {
                    m.record_pass_stage(ctx, 0, 0, &pace);
                } catch (const std::runtime_error& e) {
                    caught = e.what();
                }
                const std::string what = std::string(width > 1 ? "on a group" : "on one device") + (c.take < 0 ? ", at the last boundary" : c.take ? ", at an interior boundary" : "");
                if (!c.take) {
                    pieces = boundary;
                    require(pieces > 1 && !observed && caught.empty(), "a paced stage ended no pieces, or observed a pass none recorded");
                } else if (!c.taken) {
                    require(caught == "injected submission failure" && !observed && pace.within == 0, ("a stage that failed with no pass taken observed one, or lost its error, " + what).c_str());
                } else {
                    require(caught == (c.fail ? "injected submission failure" : ""), ("a stage's failure after a pass was taken did not escape, or a whole stage failed, " + what).c_str());
                    require(observed == 1 && pace.within == 0, ("a pass recorded at a piece boundary was not observed exactly once before the stage's recording returned, " + what).c_str());
                    require(waited_first, ("a pass recorded at a piece boundary was observed before its ticket was waited for, " + what).c_str());
                }
                if (!caught.empty()) {
                    m.abort_pass(ctx, 0);
                    m.abort_pass(ctx, 1);
                }
                ++checked;
            }
        }
}

// The fit of a tensor group counts what a member keeps beside its layers: its collective's rows (Backend::collective_bytes_per_row) and, on the head's group, its slice of every logits row, while the host's rows hold the whole vocabulary.
void group_fits() {
    const auto weights = tiny_qwen(3, 2 * 128, true);
    const infer::ModelWeights views = infer::gguf_weights(weights);
    const infer::ModelPlan plan = infer::plan_model(views);
    const infer::ModelOptions options;
    const infer::Footprint whole = infer::footprint(views, plan, options), member = infer::footprint(views, plan, options, 2, 0);
    require(whole.logits_per_row == plan.vocab * sizeof(float) && whole.head_slice_per_row == 0 && member.logits_per_row == whole.logits_per_row &&
                member.head_slice_per_row == plan.vocab / 2 * sizeof(float),
            "a member's footprint does not count its slice of the logits rows beside the host's whole rows");
    auto cpu = std::make_shared<backend::CpuBackend>();
    require(cpu->collective_bytes_per_row(2, plan.residual) == plan.residual * sizeof(float), "the CPU's collective keeps other than its partial rows");
    ++checked;
    const size_t rows = 5, logits = 3;
    auto device = [&](size_t per_member) {
        infer::DeviceBudget d;
        d.name = "d";
        d.collective = [per_member](size_t members, size_t width) { return members * width * per_member; };
        return d;
    };
    const std::vector<infer::DeviceBudget> devices{device(4), device(7)};
    const std::vector<infer::DeviceBudget> group = infer::group_budgets(devices, 2, plan.residual);
    require(group.size() == 1 && group[0].collective_per_row == 2 * plan.residual * 7, "a group's budget does not take its members' largest collective rows");
    const infer::LayerSplit roomy = infer::split_layers(member, group, rows, {}, std::nullopt, 1, logits);
    const infer::LayerSplit::Stage& st = roomy.stages.at(0);
    const size_t without = rows * member.activations_per_row + member.tables + group[0].scratch;
    require(st.other == without + rows * group[0].collective_per_row, "a member's collective rows are not counted beside its activations");
    const size_t need = st.weights + st.cache + st.other + logits * member.head_slice_per_row;
    auto fits = [&](size_t bytes) {
        std::vector<infer::DeviceBudget> b = group;
        b[0].bytes = bytes;
        try {
            infer::split_layers(member, b, rows, {}, std::nullopt, 1, logits);
            return true;
        } catch (const std::runtime_error&) {
            return false;
        }
    };
    require(fits(need) && !fits(need - 1), "a group's member is not fitted to its layers, its collective rows and its slice of the logits rows exactly");
    ++checked;
    // The order that keeps every group under one PCI root: given only where a group spans roots and each root holds whole groups.
    using Order = std::vector<size_t>;
    require(infer::one_root_order({"a", "b", "a", "b"}, 2) == Order({0, 2, 1, 3}) && infer::one_root_order({"a", "a", "b", "b"}, 2).empty() &&
                infer::one_root_order({"a", "b", "b", "b"}, 2).empty() && infer::one_root_order({"a", "", "a", "b"}, 2).empty() &&
                infer::one_root_order({"b", "a", "c", "a", "b", "c", "a", "b"}, 4).empty() && infer::one_root_order({"b", "a", "a", "b", "b", "a", "a", "b"}, 4) == Order({0, 3, 4, 7, 1, 2, 5, 6}) &&
                infer::one_root_order({"a", "b"}, 1).empty(),
            "the order that keeps every group under one root is not the roots' devices together, or is given where none helps");
    ++checked;
}

int main() {
    try {
        tensor_groups();
        group_fits();
        context_keeps_logits_pages();
        apart_rows_fitted();
        final_piece_observed();
        host_scratch_fits();
        head_aux_footprint_fits();
        lookup_scratch_fits();
        heterogeneous_saved_fits();
        external_lookup_fits();
        lookup_storage_planned();
        split_matches_single();
        layer_split_fits();
        histories_fit_the_pool();
        kv_fitted();
        bad_placements_refused();
        for (bool devices : {false, true}) {
            pipelined_matches_single(devices);
            pipelined_failure_rolls_back(devices);
        }
        replay_over_stages();
        passes_refused();
        passes_continue();
        passes_match_alone();
        std::cout << "placement: " << checked << " checks across two, three and four CPU backends\n";
        for (const auto& path : table_files) std::filesystem::remove(path);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
