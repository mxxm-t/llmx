// Auxiliary F32 rows share paged KV block IDs, growth and host-history ownership.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backends/devices.hpp"
#include "backends/kv_storage.hpp"
#include "model/arch/qwen3.hpp"
#include "model/arch/registry.hpp"
#include "model/place.hpp"
#include "model/runtime.hpp"
#include "tiny_qwen.hpp"

namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <class F> void rejects(F fn, const char* what) {
    bool failed = false;
    try { fn(); }
    catch (const std::exception&) { failed = true; }
    require(failed, what);
}

float key_value(size_t layer, size_t position, size_t head, size_t lane) {
    return float(100000 + layer * 10000 + position * 100 + head * 10 + lane);
}

float value_value(size_t layer, size_t position, size_t head, size_t lane) {
    return -key_value(layer, position, head, lane) - 0.5f;
}

float auxiliary_value(size_t layer, size_t position, size_t lane) {
    return float(2000 + layer * 1000 + position * 10 + lane) + 0.25f;
}

std::vector<uint8_t> bytes_of(backend::Backend& device, const backend::Buffer& buffer) {
    std::vector<uint8_t> bytes(buffer.size());
    if (!bytes.empty()) device.read(buffer, 0, bytes.data(), bytes.size());
    return bytes;
}

float read_f32(const std::vector<uint8_t>& bytes, size_t offset) {
    float value;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

struct Source {
    std::vector<float> k, v, auxiliary;
    backend::BufferPtr kb, vb, ab;
};

Source source_rows(backend::Backend& device, size_t rows, size_t heads, size_t dim, size_t width,
                   const std::vector<size_t>& positions, size_t layer) {
    require(positions.size() == rows, "source position count differs from its rows");
    Source src;
    constexpr size_t guard = 3;
    src.k.assign(guard + rows * heads * dim + guard, -777.0f);
    src.v.assign(guard + rows * heads * dim + guard, -888.0f);
    src.auxiliary.assign(guard + rows * width + guard, -999.0f);
    for (size_t r = 0; r < rows; ++r) {
        for (size_t h = 0; h < heads; ++h)
            for (size_t d = 0; d < dim; ++d) {
                const size_t at = guard + (r * heads + h) * dim + d;
                src.k[at] = key_value(layer, positions[r], h, d);
                src.v[at] = value_value(layer, positions[r], h, d);
            }
        for (size_t d = 0; d < width; ++d)
            src.auxiliary[guard + r * width + d] = auxiliary_value(layer, positions[r], d);
    }
    src.kb = device.adopt(src.k.data(), src.k.size() * sizeof(float));
    src.vb = device.adopt(src.v.data(), src.v.size() * sizeof(float));
    if (width) src.ab = device.adopt(src.auxiliary.data(), src.auxiliary.size() * sizeof(float));
    return src;
}

void write_rows(backend::Backend& device, size_t layer, const backend::KVView* views, size_t n_views,
                size_t heads, size_t dim, size_t width, const std::vector<size_t>& positions) {
    size_t rows = 0;
    for (size_t i = 0; i < n_views; ++i) rows = backend::size_add(rows, views[i].nq);
    Source src = source_rows(device, rows, heads, dim, width, positions, layer);
    device.kv_write(layer, views, n_views, {src.kb.get(), 3}, {src.vb.get(), 3},
                    width ? backend::CSlice{src.ab.get(), 3} : backend::CSlice{});
    device.sync();
}

void storage_contract(backend::Backend& device) {
    const size_t bt = device.kv_layout().block_tokens;
    const size_t layers = 3, heads = 2, dim = 3, max_blocks = 8;
    const std::vector<size_t> widths{0, 3, 128};
    rejects([&] { device.kv_alloc(layers, heads, dim, max_blocks * bt, backend::KVType::f32,
                                  backend::KVType::f32, {0, 3}); },
            "auxiliary layer-count mismatch accepted");
    rejects([&] { device.kv_alloc(1, 1, 1, bt, backend::KVType::f32,
                                  backend::KVType::f32, {std::numeric_limits<size_t>::max()}); },
            "overflowing auxiliary block width accepted");

    std::unique_ptr<backend::KVStorage> owner = device.kv_alloc(
        layers, heads, dim, max_blocks * bt, backend::KVType::f32, backend::KVType::f32, widths);
    auto& storage = dynamic_cast<backend::BlockKVStorage&>(*owner);
    const size_t expected_block = layers * heads * bt * dim * sizeof(float) * 2 +
                                  (3 + 128) * bt * sizeof(float);
    require(storage.max_blocks() == max_blocks, "KV budget did not round to its whole block count");
    require(storage.auxiliary_width(0) == 0 && storage.auxiliary_width(1) == 3 &&
            storage.auxiliary_width(2) == 128, "per-layer auxiliary widths changed");
    require(storage.auxiliary_block_bytes(0) == 0 &&
            storage.auxiliary_block_bytes(1) == 3 * bt * sizeof(float) &&
            storage.auxiliary_block_bytes(2) == 128 * bt * sizeof(float),
            "auxiliary bytes are not F32 token-major rows per block");
    require(storage.total_block_bytes() == expected_block, "total block bytes omit or miscount auxiliary rows");
    require(storage.allocated_bytes() == 0 && storage.peak_bytes() == 0,
            "KV storage was backed before its first write");

    // Seed block zero, then grow through block seven while writing nonmonotonic physical IDs across two views.
    const int32_t seed_id = 0;
    const backend::KVView seed{owner.get(), &seed_id, 1, 0, 1};
    for (size_t layer = 0; layer < layers; ++layer)
        write_rows(device, layer, &seed, 1, heads, dim, widths[layer], {0});
    require(storage.allocated_bytes() == expected_block, "first KV write did not back one complete block");

    const int32_t a_ids[] = {7, 1, 6};
    const int32_t b_ids[] = {2, 0, 5};
    const backend::KVView views[] = {
        {owner.get(), a_ids, 3, bt - 2, 4},
        {owner.get(), b_ids, 3, 2 * bt - 1, 3},
    };
    const std::vector<size_t> positions{bt - 2, bt - 1, bt, bt + 1,
                                        2 * bt - 1, 2 * bt, 2 * bt + 1};
    for (size_t layer = 0; layer < layers; ++layer)
        write_rows(device, layer, views, 2, heads, dim, widths[layer], positions);
    const size_t backed = 8;
    require(storage.allocated_bytes() == backed * expected_block,
            "growth accounting omitted an auxiliary plane");
    require(storage.peak_bytes() == 9 * expected_block,
            "growth peak did not count old and new KV plus auxiliary blocks");

    std::vector<std::vector<float>> expected_k(layers, std::vector<float>(backed * heads * bt * dim, 0.0f));
    std::vector<std::vector<float>> expected_v = expected_k;
    std::vector<std::vector<float>> expected_aux(layers);
    for (size_t layer = 0; layer < layers; ++layer)
        if (widths[layer]) expected_aux[layer].assign(backed * bt * widths[layer], 0.0f);
    const auto mark = [&](size_t layer, int32_t block, size_t pos) {
        for (size_t h = 0; h < heads; ++h)
            for (size_t d = 0; d < dim; ++d) {
                const size_t at = ((size_t(block) * heads + h) * bt + pos % bt) * dim + d;
                expected_k[layer][at] = key_value(layer, pos, h, d);
                expected_v[layer][at] = value_value(layer, pos, h, d);
            }
        if (widths[layer])
            for (size_t d = 0; d < widths[layer]; ++d)
                expected_aux[layer][(size_t(block) * bt + pos % bt) * widths[layer] + d] =
                    auxiliary_value(layer, pos, d);
    };
    for (size_t layer = 0; layer < layers; ++layer) {
        mark(layer, seed_id, 0);
        for (size_t i = 0; i < 4; ++i) mark(layer, a_ids[(bt - 2 + i) / bt], bt - 2 + i);
        for (size_t i = 0; i < 3; ++i) mark(layer, b_ids[(2 * bt - 1 + i) / bt], 2 * bt - 1 + i);
        const auto& kb = storage.k_buffer(layer);
        const auto& vb = storage.v_buffer(layer);
        const std::vector<uint8_t> kbytes = bytes_of(device, *kb);
        const std::vector<uint8_t> vbytes = bytes_of(device, *vb);
        for (size_t i = 0; i < expected_k[layer].size(); ++i) {
            require(read_f32(kbytes, i * sizeof(float)) == expected_k[layer][i], "K values changed across auxiliary growth");
            require(read_f32(vbytes, i * sizeof(float)) == expected_v[layer][i], "V values changed across auxiliary growth");
        }
        if (widths[layer]) {
            const auto& ab = storage.auxiliary_buffer(layer);
            require(ab && ab->size() == expected_aux[layer].size() * sizeof(float),
                    "auxiliary storage is not backed beside the shared block pool");
            const std::vector<uint8_t> abytes = bytes_of(device, *ab);
            for (size_t i = 0; i < expected_aux[layer].size(); ++i)
                require(read_f32(abytes, i * sizeof(float)) == expected_aux[layer][i],
                        "auxiliary row changed across block-boundary writes or growth");
        } else {
            require(!storage.auxiliary_buffer(layer), "zero-width auxiliary layer allocated a buffer");
        }
    }

    // Refusals target an already-backed row and must leave K, V and auxiliary bytes unchanged.
    const backend::KVView overwrite{owner.get(), &seed_id, 1, 0, 1};
    const auto snapshot = [&] {
        std::vector<std::vector<uint8_t>> result;
        for (size_t layer = 0; layer < layers; ++layer) {
            result.push_back(bytes_of(device, *storage.k_buffer(layer)));
            result.push_back(bytes_of(device, *storage.v_buffer(layer)));
            if (widths[layer]) result.push_back(bytes_of(device, *storage.auxiliary_buffer(layer)));
        }
        return result;
    };
    const auto before = snapshot();
    Source valid = source_rows(device, 1, heads, dim, 3, {0}, 9);
    rejects([&] { device.kv_write(1, &overwrite, 1, {valid.kb.get(), 3}, {valid.vb.get(), 3}); },
            "required auxiliary source was omitted");
    std::vector<float> too_small(3 + 2, -9.0f);
    auto small = device.adopt(too_small.data(), too_small.size() * sizeof(float));
    rejects([&] { device.kv_write(1, &overwrite, 1, {valid.kb.get(), 3}, {valid.vb.get(), 3}, {small.get(), 3}); },
            "undersized auxiliary source accepted");
    rejects([&] { device.kv_write(0, &overwrite, 1, {valid.kb.get(), 3}, {valid.vb.get(), 3}, {valid.ab.get(), 3}); },
            "unexpected auxiliary source on a zero-width layer accepted");
    const int32_t invalid_block = int32_t(max_blocks);
    const backend::KVView invalid_views[] = {
        overwrite,
        {owner.get(), &invalid_block, 1, 0, 1},
    };
    Source two_rows = source_rows(device, 2, heads, dim, 3, {0, 0}, 9);
    rejects([&] {
        device.kv_write(1, invalid_views, 2, {two_rows.kb.get(), 3}, {two_rows.vb.get(), 3},
                        {two_rows.ab.get(), 3});
    }, "invalid block in a later view was accepted");
    device.sync();
    require(snapshot() == before, "a refused auxiliary write changed K, V or auxiliary bytes");

    auto mixed = device.kv_alloc(1, 1, 1, bt, backend::KVType::f16, backend::KVType::f16, {3});
    const int32_t mixed_id = 0;
    const backend::KVView mixed_view{mixed.get(), &mixed_id, 1, 0, 1};
    const float zero = 0.0f;
    const float raw_values[] = {1e-10f, 1.0001f, 1e10f};
    auto mixed_k = device.adopt(&zero, sizeof(zero));
    auto mixed_v = device.adopt(&zero, sizeof(zero));
    auto mixed_aux = device.adopt(raw_values, sizeof(raw_values));
    device.kv_write(0, &mixed_view, 1, {mixed_k.get(), 0}, {mixed_v.get(), 0}, {mixed_aux.get(), 0});
    device.sync();
    const auto& mixed_storage = dynamic_cast<const backend::BlockKVStorage&>(*mixed);
    const std::vector<uint8_t> raw_bytes = bytes_of(device, *mixed_storage.auxiliary_buffer(0));
    for (size_t i = 0; i < 3; ++i)
        require(read_f32(raw_bytes, i * sizeof(float)) == raw_values[i],
                "F16 K/V cache types narrowed an auxiliary F32 row");
}

class FailingAuxCpu final : public backend::CpuBackend {
public:
    backend::BufferPtr alloc(size_t bytes, backend::Memory where) override {
        ++allocations;
        if (fail_at && allocations == fail_at) throw std::runtime_error("injected KV allocation failure");
        return backend::CpuBackend::alloc(bytes, where);
    }
    size_t allocations = 0, fail_at = 0;
};

void growth_failure_is_atomic() {
    FailingAuxCpu device;
    device.set_threads(1);
    const size_t bt = device.kv_layout().block_tokens;
    const std::vector<size_t> widths{0, 3, 128};
    auto owner = device.kv_alloc(3, 1, 2, 8 * bt, backend::KVType::f32, backend::KVType::f32, widths);
    auto& storage = dynamic_cast<backend::BlockKVStorage&>(*owner);
    const int32_t zero = 0;
    const backend::KVView first{owner.get(), &zero, 1, 0, 1};
    for (size_t layer = 0; layer < 3; ++layer)
        write_rows(device, layer, &first, 1, 1, 2, widths[layer], {0});
    const auto snapshot = [&] {
        std::vector<std::vector<uint8_t>> result;
        for (size_t layer = 0; layer < 3; ++layer) {
            result.push_back(bytes_of(device, *storage.k_buffer(layer)));
            result.push_back(bytes_of(device, *storage.v_buffer(layer)));
            if (widths[layer]) result.push_back(bytes_of(device, *storage.auxiliary_buffer(layer)));
        }
        return result;
    };
    const auto before = snapshot();
    const size_t allocated = storage.allocated_bytes(), peak = storage.peak_bytes();
    const int32_t high = 7;
    const backend::KVView target{owner.get(), &high, 1, 0, 1};
    Source src = source_rows(device, 1, 1, 2, widths[1], {0}, 1);
    device.fail_at = device.allocations + 8;
    rejects([&] {
        device.kv_write(1, &target, 1, {src.kb.get(), 3}, {src.vb.get(), 3}, {src.ab.get(), 3});
    }, "injected auxiliary growth allocation did not fail");
    require(!storage.backed(7) && storage.allocated_bytes() == allocated && storage.peak_bytes() == peak,
            "partial auxiliary growth changed storage accounting");
    require(snapshot() == before, "partial auxiliary growth published or changed a buffer");
    device.kv_write(1, &target, 1, {src.kb.get(), 3}, {src.vb.get(), 3}, {src.ab.get(), 3});
    device.sync();
    require(storage.backed(7) && storage.allocated_bytes() == 8 * storage.total_block_bytes(),
            "retry after failed auxiliary growth did not back the full shared block set");
}

class AuxiliaryArchitecture final : public infer::Architecture {
public:
    AuxiliaryArchitecture(std::shared_ptr<const infer::Architecture> base, size_t layer, size_t width,
                          bool non_kv = false)
        : base_(std::move(base)), layer_(layer), width_(width), non_kv_(non_kv) {}
    infer::ModelPlan plan(const infer::TensorIndex& tensors) const override {
        infer::ModelPlan p = base_->plan(tensors);
        p.layers.at(layer_).kv_aux = width_;
        if (non_kv_) p.layers.at(layer_).cache = infer::Cache::none;
        return p;
    }
    void fill_tables(std::vector<std::vector<float>>& tables) const override { base_->fill_tables(tables); }
    void embed(const infer::Step& s, const uint32_t* ids) const override { base_->embed(s, ids); }
    void mixer(const infer::Step& s) const override { base_->mixer(s); }
    void recur(const infer::Step& s) const override { base_->recur(s); }
    void ffn(const infer::Step& s) const override { base_->ffn(s); }
    void head(const infer::HeadStep& s) const override { base_->head(s); }

private:
    std::shared_ptr<const infer::Architecture> base_;
    size_t layer_, width_;
    bool non_kv_;
};

class AuxiliaryCpu final : public backend::CpuBackend {
public:
    void kv_write(size_t layer, const backend::KVView* views, size_t n_views,
                  backend::CSlice k, backend::CSlice v, backend::CSlice auxiliary = {}) override {
        size_t rows = 0;
        for (size_t i = 0; i < n_views; ++i) rows = backend::size_add(rows, views[i].nq);
        auto* blocks = n_views ? dynamic_cast<backend::BlockKVStorage*>(views[0].storage) : nullptr;
        if (!blocks) throw std::runtime_error("test: model did not use shared block KV storage");
        storage = blocks;
        if (n_views) {
            last_blocks.assign(views[0].blocks, views[0].blocks + views[0].n_blocks);
            last_length = views[0].length;
        }
        const size_t width = blocks->auxiliary_width(layer);
        if (!width || auxiliary.buffer) {
            backend::CpuBackend::kv_write(layer, views, n_views, k, v, auxiliary);
            return;
        }
        std::vector<float> values(backend::size_mul(rows, width));
        size_t row = 0;
        for (size_t i = 0; i < n_views; ++i)
            for (size_t r = 0; r < views[i].nq; ++r, ++row)
                for (size_t d = 0; d < width; ++d)
                    values[row * width + d] = auxiliary_value(layer, views[i].length + r, d);
        auto data = adopt(values.data(), values.size() * sizeof(float));
        backend::CpuBackend::kv_write(layer, views, n_views, k, v, {data.get(), 0});
    }

    backend::BlockKVStorage* storage = nullptr;
    std::vector<int32_t> last_blocks;
    size_t last_length = 0;
};

class RecordingAuxCpu final : public backend::CpuBackend {
public:
    std::unique_ptr<backend::KVStorage> kv_alloc(size_t layers, size_t n_head_kv, size_t head_dim,
                                                  size_t max_tokens, backend::KVType k_type,
                                                  backend::KVType v_type,
                                                  const std::vector<size_t>& auxiliary_widths = {},
                                                  const std::vector<size_t>& auxiliary_pools = {}) override {
        auxiliary_allocations.push_back(auxiliary_widths);
        return backend::CpuBackend::kv_alloc(layers, n_head_kv, head_dim, max_tokens,
                                             k_type, v_type, auxiliary_widths, auxiliary_pools);
    }
    std::vector<std::vector<size_t>> auxiliary_allocations;
};

void check_model_aux(const backend::BlockKVStorage& storage, backend::Backend& device,
                     const std::vector<int32_t>& blocks, size_t first, size_t end) {
    const size_t width = storage.auxiliary_width(0), bt = storage.block_tokens();
    const auto& buffer = storage.auxiliary_buffer(0);
    const std::vector<uint8_t> bytes = bytes_of(device, *buffer);
    for (size_t pos = first; pos < end; ++pos) {
        const int32_t id = blocks.at(pos / bt);
        for (size_t d = 0; d < width; ++d) {
            const size_t index = (size_t(id) * bt + pos % bt) * width + d;
            require(read_f32(bytes, index * sizeof(float)) == auxiliary_value(0, pos, d),
                    "model history lost or changed an auxiliary row");
        }
    }
}

std::vector<uint8_t> host_bytes_of(const infer::HostHistory& host,
                                   const std::vector<infer::HostRange>& ranges) {
    constexpr size_t slab_bytes = infer::Model::host_slab_bytes();
    std::vector<uint8_t> bytes;
    for (const infer::HostRange& range : ranges) {
        require(range.device == 0, "single-device host range named another device");
        for (size_t at = range.offset; at < range.offset + range.bytes; ++at)
            bytes.push_back(((const uint8_t*)host.slabs[range.device][at / slab_bytes]->host_ptr())[at % slab_bytes]);
    }
    return bytes;
}

std::vector<uint8_t> auxiliary_host_bytes(size_t first, size_t count) {
    std::vector<uint8_t> bytes(count * 3 * sizeof(float));
    for (size_t pos = first; pos < first + count; ++pos)
        for (size_t lane = 0; lane < 3; ++lane) {
            const float value = auxiliary_value(0, pos, lane);
            std::memcpy(bytes.data() + ((pos - first) * 3 + lane) * sizeof(float), &value, sizeof(value));
        }
    return bytes;
}

void forward(infer::Model& model, infer::Sequence& sequence, const std::vector<uint32_t>& ids) {
    infer::ExecContext context;
    const infer::BatchEntry entry{&sequence, ids.data(), ids.size(), false};
    model.forward(context, &entry, 1);
}

void model_history() {
    auto cpu = std::make_shared<AuxiliaryCpu>();
    cpu->set_threads(1);
    const size_t bt = cpu->kv_layout().block_tokens;
    const gguf::GGUFModel gguf = tiny_qwen(2, 4 * bt, true);
    infer::ModelWeights weights = infer::gguf_weights(gguf);
    const std::shared_ptr<const infer::Architecture> base_arch = weights.arch;
    infer::ModelOptions options;
    options.kv_k = backend::KVType::f32;
    options.kv_v = backend::KVType::f32;
    options.kv_tokens = 4 * bt;

    const auto configured_weights = [&](size_t width) {
        infer::ModelWeights configured = weights;
        configured.arch = std::make_shared<AuxiliaryArchitecture>(base_arch, 0, width);
        return configured;
    };
    {
        infer::ModelWeights invalid = weights;
        invalid.arch = std::make_shared<AuxiliaryArchitecture>(base_arch, 0, 3, true);
        rejects([&] { infer::Model bad(invalid, cpu, options); },
                "model accepted auxiliary columns on a non-KV layer");
    }
    const infer::ModelWeights no_aux_weights = configured_weights(0);
    const infer::ModelWeights aux_weights = configured_weights(3);
    const infer::ModelPlan no_aux_plan = infer::plan_model(no_aux_weights);
    const infer::ModelPlan aux_plan = infer::plan_model(aux_weights);
    const infer::Footprint no_aux_footprint = infer::footprint(no_aux_weights, no_aux_plan, options);
    const infer::Footprint aux_footprint = infer::footprint(aux_weights, aux_plan, options);
    require(aux_footprint.cache.at(0) - no_aux_footprint.cache.at(0) == 4 * bt * 3 * sizeof(float),
            "model footprint omitted the per-position auxiliary F32 columns");

    infer::ModelWeights mapped_weights = weights;
    mapped_weights.arch = std::make_shared<AuxiliaryArchitecture>(base_arch, 1, 3);
    const infer::ModelPlan mapped_plan = infer::plan_model(mapped_weights);
    for (size_t member = 0; member < 2; ++member) {
        const infer::Footprint no_aux_group = infer::footprint(no_aux_weights, no_aux_plan, options, 2, member);
        const infer::Footprint aux_group = infer::footprint(mapped_weights, mapped_plan, options, 2, member);
        require(aux_group.cache.at(1) - no_aux_group.cache.at(1) == 4 * bt * 3 * sizeof(float),
                "tensor-group footprint divided auxiliary columns between members");
    }

    {
        auto first = std::make_shared<RecordingAuxCpu>();
        auto second = std::make_shared<RecordingAuxCpu>();
        first->set_threads(1);
        second->set_threads(1);
        infer::Placement placement;
        placement.mixer_device = {0, 1};
        placement.ffn_device = {0, 1};
        infer::Model split(mapped_weights, std::vector<backend::BackendPtr>{first, second}, placement, options);
        require(first->auxiliary_allocations == std::vector<std::vector<size_t>>{{0}} &&
                second->auxiliary_allocations == std::vector<std::vector<size_t>>{{3}},
                "layer placement did not allocate auxiliary widths only on the owning device");
    }
    {
        auto first = std::make_shared<RecordingAuxCpu>();
        auto second = std::make_shared<RecordingAuxCpu>();
        first->set_threads(1);
        second->set_threads(1);
        infer::Placement placement;
        placement.width = 2;
        infer::Model split(mapped_weights, std::vector<backend::BackendPtr>{first, second}, placement, options);
        const std::vector<std::vector<size_t>> expected{{0, 3}};
        require(first->auxiliary_allocations == expected && second->auxiliary_allocations == expected,
                "tensor-group members did not each allocate the full per-layer auxiliary widths");
    }

    auto identity_cpu = [] {
        auto device = std::make_shared<backend::CpuBackend>();
        device->set_threads(1);
        return device;
    };
    const std::string aux_identity = infer::Model(aux_weights, identity_cpu(), options).host_identity();
    const std::string same_aux_identity = infer::Model(configured_weights(3), identity_cpu(), options).host_identity();
    const std::string wider_aux_identity = infer::Model(configured_weights(4), identity_cpu(), options).host_identity();
    require(aux_identity == same_aux_identity && aux_identity != wider_aux_identity,
            "host identity did not distinguish the auxiliary layout width");

    weights = aux_weights;
    infer::Model model(std::move(weights), cpu, options);
    model.set_ubatch(47);
    infer::Sequence original = model.make_sequence();
    std::vector<uint32_t> prompt(2 * bt);
    for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = uint32_t(i % 16);
    forward(model, original, prompt);
    require(original.length() == 2 * bt && cpu->storage, "synthetic Qwen3 did not write the auxiliary-enabled history");
    auto& storage = *cpu->storage;
    require(storage.layers() == 2 && storage.auxiliary_width(0) == 3 && storage.auxiliary_width(1) == 0,
            "the model did not map its one auxiliary layer to local KV storage");
    check_model_aux(storage, *cpu, cpu->last_blocks, 0, 2 * bt);

    infer::HostHistory host;
    model.save_host(original, 2 * bt, host, std::numeric_limits<size_t>::max());
    model.wait_host(host);
    const size_t per_token = 2 * (2 * 1 * 4 * sizeof(float)) + 3 * sizeof(float);
    require(host.bytes == 2 * bt * per_token && host.held == model.host_bytes(2 * bt),
            "host history accounting omitted auxiliary F32 columns");
    const auto ranges = model.host_ranges(host, 0, 2 * bt);
    size_t range_bytes = 0;
    for (const infer::HostRange& range : ranges) range_bytes += range.bytes;
    require(range_bytes == host.bytes && host.device_bytes.size() == 1 && host.device_bytes[0] == host.bytes,
            "host ranges do not include all auxiliary block bytes");

    const size_t kv_block_bytes = storage.k_block_bytes();
    const size_t aux_offset = 4 * kv_block_bytes;
    const size_t aux_bytes = 2 * bt * 3 * sizeof(float);
    const auto has_range = [](const std::vector<infer::HostRange>& rs, size_t offset, size_t bytes) {
        return std::any_of(rs.begin(), rs.end(), [&](const infer::HostRange& r) {
            return r.offset == offset && r.bytes == bytes;
        });
    };
    require(has_range(ranges, aux_offset, aux_bytes),
            "whole-history host ranges omitted the auxiliary plane extent");
    const auto whole_aux = host_bytes_of(host, {{0, aux_offset, aux_bytes}});
    require(whole_aux == auxiliary_host_bytes(0, 2 * bt),
            "whole-history auxiliary host bytes do not match the token-major rows");
    const auto second_block_ranges = model.host_ranges(host, bt, 2 * bt);
    const size_t one_aux_bytes = bt * 3 * sizeof(float);
    const size_t second_aux_offset = 4 * kv_block_bytes + one_aux_bytes;
    require(has_range(second_block_ranges, second_aux_offset, one_aux_bytes),
            "partial host range omitted the second block's auxiliary extent");
    const auto second_aux = host_bytes_of(host, {{0, second_aux_offset, one_aux_bytes}});
    require(second_aux == auxiliary_host_bytes(bt, bt),
            "partial host range auxiliary bytes do not match the second token block");

    model.reset(original);
    {
        const auto& aux_buffer = storage.auxiliary_buffer(0);
        std::vector<float> poison(aux_buffer->size() / sizeof(float), -12345.75f);
        cpu->write(*aux_buffer, 0, poison.data(), poison.size() * sizeof(float));
        cpu->sync();
    }
    infer::Sequence restored = model.restore_host(host);
    require(restored.length() == 2 * bt, "host restore changed the auxiliary history length");
    forward(model, restored, {7});
    require(cpu->last_length == 2 * bt, "restored continuation started from the wrong position");
    check_model_aux(storage, *cpu, cpu->last_blocks, 0, 2 * bt + 1);

    infer::Sequence forked = model.fork(restored, bt);
    require(forked.length() == bt, "auxiliary history fork length changed");
    model.reset(restored);
    forward(model, forked, {9});
    require(cpu->last_length == bt, "fork continuation started from the wrong position");
    check_model_aux(storage, *cpu, cpu->last_blocks, 0, bt + 1);
    model.reset(forked);
    model.release_host(host);
    require(host.bytes == 0 && host.slabs.empty(), "released auxiliary host history kept its slabs");
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string device = "cpu";
        if (argc == 3 && std::string(argv[1]) == "--device") device = argv[2];
        else if (argc != 1) throw std::runtime_error("usage: kv-auxiliary-test [--device cpu|vulkan:N]");
        backend::BackendPtr storage_device = backend::make_backend(device);
        storage_contract(*storage_device);
        growth_failure_is_atomic();
        model_history();
        std::cout << "KV auxiliary: mixed per-layer F32 rows, shared paging and growth, refusal atomicity, host history and fork passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "KV auxiliary: FAIL: " << error.what() << '\n';
        return 1;
    }
}
