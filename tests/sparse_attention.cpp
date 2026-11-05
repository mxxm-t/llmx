// Sparse causal GQA against an independent double-precision oracle.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backends/devices.hpp"
#if LLMX_HAS_BACKEND_ROCM
#include "backends/rocm/rocm_backend.hpp"
#endif
#include "backends/kv_storage.hpp"
#include "core/fp16.hpp"
#if LLMX_HAS_BACKEND_VULKAN
#include "backends/vulkan/vulkan_backend.hpp"
#endif

namespace {
using backend::Backend;
using backend::BufferPtr;
using backend::CSlice;
using backend::KVView;
using backend::Slice;

constexpr size_t kQHeads = 4;
constexpr size_t kKVHeads = 2;
constexpr size_t kNq = 5;
constexpr size_t kAuxDim = 8;
// The groups whose pooled rows the auxiliary blocks keep, the selection's compression.
constexpr size_t kCompression = 2;
constexpr float kBound = 5e-4f;
size_t checks = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

size_t next_power_of_two(size_t value) {
    size_t out = 1;
    while (out < value) out *= 2;
    return out;
}

std::vector<int32_t> block_map(size_t count, size_t base, size_t capacity) {
    std::vector<int32_t> out(capacity);
    size_t step = 3;
    while (std::gcd(step, capacity) != 1) ++step;
    for (size_t i = 0; i < capacity; ++i) out[i] = int32_t(base + ((i * step) % capacity));
    require(count <= capacity, "block map capacity is too small");
    return out;
}

template<class T>
BufferPtr upload(Backend& device, const std::vector<T>& values) {
    BufferPtr out = device.alloc(std::max<size_t>(values.size(), 1) * sizeof(T), backend::Memory::device);
    if (!values.empty()) device.write(*out, 0, values.data(), values.size() * sizeof(T));
    return out;
}

template<class T>
std::vector<T> read(Backend& device, const backend::Buffer& buffer, size_t byte_offset, size_t count) {
    std::vector<T> out(count);
    if (count) device.read(buffer, byte_offset, out.data(), count * sizeof(T));
    return out;
}

struct Data {
    size_t dim = 0, block_tokens = 0, length = 0, total_tokens = 0, id_stride = 8;
    backend::KVType k_type = backend::KVType::f32, v_type = backend::KVType::f32;
    std::unique_ptr<backend::KVStorage> storage;
    std::vector<int32_t> maps[2];
    std::vector<KVView> views;
    std::vector<float> keys, values, auxiliary, queries;
    std::vector<uint32_t> ids;
};

float key_value(size_t seq, size_t pos, size_t head, size_t d) {
    const double x = double((seq + 1) * 17 + pos * 11 + head * 7 + d * 3);
    return float(0.19 * std::sin(x * 0.071) + 0.08 * std::cos(x * 0.037));
}

float value_value(size_t seq, size_t pos, size_t head, size_t d) {
    const double x = double((seq + 2) * 13 + pos * 5 + head * 19 + d * 2);
    return float(0.31 * std::cos(x * 0.053) - 0.11 * std::sin(x * 0.029));
}

float aux_value(size_t seq, size_t pos, size_t d) {
    return float(0.23 * std::sin(double(1 + seq * 101 + pos * 7 + d * 13) * 0.09));
}

float query_value(size_t seq, size_t row, size_t head, size_t d) {
    const double x = double((seq + 3) * 29 + row * 17 + head * 11 + d * 5);
    return float(0.17 * std::sin(x * 0.043) + 0.09 * std::cos(x * 0.031));
}

Data make_data(Backend& device, size_t dim, backend::KVType k_type, backend::KVType v_type) {
    Data data;
    data.dim = dim;
    data.block_tokens = device.kv_layout().block_tokens;
    data.length = 2 * data.block_tokens + 3;
    data.total_tokens = data.length + kNq;
    data.k_type = k_type;
    data.v_type = v_type;
    const size_t logical_blocks = backend::blocks_for(data.total_tokens, data.block_tokens);
    const size_t per_view = next_power_of_two(logical_blocks);
    data.storage = device.kv_alloc(1, kKVHeads, dim, 2 * per_view * data.block_tokens,
                                   k_type, v_type, {kAuxDim}, {kCompression});
    data.maps[0] = block_map(logical_blocks, 0, per_view);
    data.maps[1] = block_map(logical_blocks, per_view, per_view);
    data.views = {
        KVView{data.storage.get(), data.maps[0].data(), data.maps[0].size(), data.length, kNq},
        KVView{data.storage.get(), data.maps[1].data(), data.maps[1].size(), data.length, kNq}
    };

    const size_t all_tokens = 2 * data.total_tokens;
    data.keys.resize(all_tokens * kKVHeads * dim);
    data.values.resize(data.keys.size());
    data.auxiliary.resize(all_tokens * kAuxDim);
    for (size_t seq = 0; seq < 2; ++seq) for (size_t pos = 0; pos < data.total_tokens; ++pos) {
        const size_t row = seq * data.total_tokens + pos;
        for (size_t h = 0; h < kKVHeads; ++h) for (size_t d = 0; d < dim; ++d) {
            data.keys[row * kKVHeads * dim + h * dim + d] = key_value(seq, pos, h, d);
            data.values[row * kKVHeads * dim + h * dim + d] = value_value(seq, pos, h, d);
        }
        for (size_t d = 0; d < kAuxDim; ++d)
            data.auxiliary[row * kAuxDim + d] = aux_value(seq, pos, d);
    }

    std::vector<KVView> writes{
        KVView{data.storage.get(), data.maps[0].data(), data.maps[0].size(), 0, data.total_tokens},
        KVView{data.storage.get(), data.maps[1].data(), data.maps[1].size(), 0, data.total_tokens}
    };
    BufferPtr k = upload(device, data.keys), v = upload(device, data.values), aux = upload(device, data.auxiliary);
    device.kv_write(0, writes.data(), writes.size(), {k.get(), 0}, {v.get(), 0}, {aux.get(), 0});
    device.sync();

    data.queries.resize(2 * kNq * kQHeads * dim);
    for (size_t seq = 0; seq < 2; ++seq) for (size_t row = 0; row < kNq; ++row)
        for (size_t h = 0; h < kQHeads; ++h) for (size_t d = 0; d < dim; ++d)
            data.queries[((seq * kNq + row) * kQHeads + h) * dim + d] = query_value(seq, row, h, d);

    data.ids.assign(2 * kNq * data.id_stride, UINT32_MAX);
    for (size_t seq = 0; seq < 2; ++seq) for (size_t row = 0; row < kNq; ++row) {
        if (seq == 1 && row == 2) continue;
        const size_t causal_end = data.length + row + 1;
        std::vector<uint32_t> picks{0, 1, uint32_t(data.block_tokens - 1), uint32_t(data.block_tokens),
                                    uint32_t(causal_end - 1)};
        if (causal_end < data.total_tokens) picks.push_back(uint32_t(causal_end));
        if (causal_end + 1 < data.total_tokens) picks.push_back(uint32_t(causal_end + 1));
        std::sort(picks.begin(), picks.end());
        picks.erase(std::unique(picks.begin(), picks.end()), picks.end());
        require(picks.size() < data.id_stride, "synthetic sparse ID stride is too small");
        const size_t row_id = seq * kNq + row;
        std::copy(picks.begin(), picks.end(), data.ids.begin() + row_id * data.id_stride);
    }
    return data;
}

double cache_value(float x, backend::KVType type) {
    if (type == backend::KVType::f16) return double(f16_to_f32(f32_to_f16(x)));
    return double(x);
}

std::vector<double> oracle_row(const Data& data, size_t seq, size_t row,
                              const std::vector<uint32_t>& ids) {
    const size_t dim = data.dim;
    const size_t qrow = (seq * kNq + row) * kQHeads * dim;
    const size_t causal_end = data.length + row + 1;
    const size_t q_per_kv = kQHeads / kKVHeads;
    const double scale = 1.0 / std::sqrt(double(dim));
    std::vector<uint32_t> visible;
    for (uint32_t id : ids)
        if (id != UINT32_MAX && size_t(id) < causal_end && size_t(id) < data.total_tokens)
            visible.push_back(id);
    std::vector<double> out(kQHeads * dim, 0.0);
    for (size_t h = 0; h < kQHeads; ++h) {
        const size_t kvh = h / q_per_kv;
        std::vector<double> scores;
        scores.reserve(visible.size());
        double max_score = -std::numeric_limits<double>::infinity();
        for (uint32_t pos : visible) {
            const size_t key_row = (seq * data.total_tokens + pos) * kKVHeads * dim + kvh * dim;
            double dot = 0.0;
            for (size_t d = 0; d < dim; ++d)
                dot += double(data.queries[qrow + h * dim + d]) * cache_value(data.keys[key_row + d], data.k_type);
            const double score = dot * scale;
            scores.push_back(score);
            max_score = std::max(max_score, score);
        }
        if (visible.empty()) continue;
        double total = 0.0;
        for (double& score : scores) {
            score = std::exp(score - max_score);
            total += score;
        }
        for (size_t d = 0; d < dim; ++d) {
            double acc = 0.0;
            for (size_t j = 0; j < visible.size(); ++j) {
                const size_t value_row = (seq * data.total_tokens + visible[j]) * kKVHeads * dim + kvh * dim;
                acc += (scores[j] / total) * cache_value(data.values[value_row + d], data.v_type);
            }
            out[h * dim + d] = acc;
        }
    }
    return out;
}

std::vector<float> run_batch(Backend& device, const Data& data,
                             const std::vector<uint32_t>& ids, size_t id_stride,
                             const std::string& label) {
    const size_t rows = 2 * kNq, width = kQHeads * data.dim;
    BufferPtr q = upload(device, data.queries), idsbuf = upload(device, ids);
    BufferPtr out = device.alloc((rows * width + 2) * sizeof(float), backend::Memory::device);
    const float guard = -719.125f;
    std::vector<float> init(rows * width + 2, guard);
    device.write(*out, 0, init.data(), init.size() * sizeof(float));
    device.sparse_attention({q.get(), 0}, 0, data.views.data(), data.views.size(),
                            {idsbuf.get(), 0}, id_stride, {out.get(), 1},
                            int(kQHeads), int(kKVHeads), int(data.dim));
    device.sync();
    const auto guarded = read<float>(device, *out, 0, init.size());
    require(guarded.front() == guard && guarded.back() == guard, label + " changed output guards");
    return read<float>(device, *out, sizeof(float), rows * width);
}

double check_against_oracle(const Data& data, const std::vector<uint32_t>& ids,
                            const std::vector<float>& actual, const std::string& label,
                            size_t id_stride = 0) {
    const size_t width = kQHeads * data.dim;
    if (!id_stride) id_stride = data.id_stride;
    require(ids.size() == 2 * kNq * id_stride, label + " ID shape differs");
    require(actual.size() == 2 * kNq * width, label + " output shape differs");
    double overall_worst = 0.0;
    for (size_t seq = 0; seq < 2; ++seq) for (size_t row = 0; row < kNq; ++row) {
        const size_t r = seq * kNq + row;
        std::vector<uint32_t> row_ids(ids.begin() + r * id_stride,
                                      ids.begin() + (r + 1) * id_stride);
        const auto expected = oracle_row(data, seq, row, row_ids);
        double worst = 0.0;
        for (size_t i = 0; i < width; ++i) {
            require(std::isfinite(actual[r * width + i]), label + " produced a nonfinite output");
            const double error = std::fabs(double(actual[r * width + i]) - expected[i]);
            require(std::isfinite(error), label + " produced a nonfinite oracle error");
            worst = std::max(worst, error);
        }
        require(worst <= kBound, label + " oracle error " + std::to_string(worst) +
                " at row " + std::to_string(r));
        overall_worst = std::max(overall_worst, worst);
        if (row_ids.front() == UINT32_MAX)
            for (size_t i = 0; i < width; ++i)
                require(actual[r * width + i] == 0.0f, label + " empty selection was not zero");
    }
    return overall_worst;
}

void check_per_row(Backend& device, const Data& data, const std::vector<uint32_t>& ids, size_t id_stride,
                   const std::vector<float>& batch, const std::string& label) {
    const size_t q_width = kQHeads * data.dim, out_width = q_width;
    for (size_t seq = 0; seq < 2; ++seq) for (size_t row = 0; row < kNq; ++row) {
        const size_t global = seq * kNq + row;
        KVView view{data.storage.get(), data.maps[seq].data(), data.maps[seq].size(),
                    data.length + row, 1};
        BufferPtr q = upload(device, std::vector<float>(data.queries.begin() + global * q_width,
                                                         data.queries.begin() + (global + 1) * q_width));
        std::vector<uint32_t> row_ids(ids.begin() + global * id_stride,
                                      ids.begin() + (global + 1) * id_stride);
        BufferPtr idbuf = upload(device, row_ids);
        BufferPtr out = device.alloc((out_width + 2) * sizeof(float), backend::Memory::device);
        std::vector<float> init(out_width + 2, -818.25f);
        device.write(*out, 0, init.data(), init.size() * sizeof(float));
        device.sparse_attention({q.get(), 0}, 0, &view, 1, {idbuf.get(), 0}, id_stride,
                                {out.get(), 1}, int(kQHeads), int(kKVHeads), int(data.dim));
        device.sync();
        const auto guarded = read<float>(device, *out, 0, init.size());
        require(guarded.front() == init.front() && guarded.back() == init.back(),
                label + " per-row call changed guards");
        const auto alone = read<float>(device, *out, sizeof(float), out_width);
        require(std::memcmp(alone.data(), batch.data() + global * out_width,
                            out_width * sizeof(float)) == 0,
                label + " per-row result differs bitwise from batched result");
    }
}

std::vector<uint32_t> read_ids(Backend& device, const backend::Buffer& buffer,
                               size_t offset_floats, size_t count) {
    return read<uint32_t>(device, buffer, offset_floats * sizeof(float), count);
}

void selector_chain(Backend& device, const Data& data) {
    require(data.dim == 8, "selector chain must use dim 8");
    const size_t rows = 2 * kNq, q_width = kQHeads * data.dim;
    std::vector<float> index_q(rows * q_width), gamma(kAuxDim, 1.0f);
    std::vector<uint32_t> pos(rows);
    for (size_t seq = 0; seq < 2; ++seq) for (size_t row = 0; row < kNq; ++row) {
        const size_t r = seq * kNq + row;
        pos[r] = uint32_t(data.length + row);
        for (size_t h = 0; h < kQHeads; ++h) for (size_t d = 0; d < data.dim; ++d)
            index_q[r * q_width + h * data.dim + d] = float(0.2 * std::sin(double(3 + r * 31 + h * 7 + d) * 0.08));
    }
    const size_t half = 2;
    std::vector<float> cos(data.total_tokens * half, 1.0f), sin(data.total_tokens * half, 0.0f);
    BufferPtr raw_q = upload(device, index_q), norm_q = device.alloc(index_q.size() * sizeof(float), backend::Memory::device);
    BufferPtr q_gamma = upload(device, gamma), k_gamma = upload(device, gamma);
    BufferPtr cosbuf = upload(device, cos), sinbuf = upload(device, sin);
    device.norm_rope_partial({norm_q.get(), 0}, {raw_q.get(), 0}, rows, q_width, data.dim,
                             kQHeads, data.dim, 4, {q_gamma.get(), 0}, 1e-6f,
                             {cosbuf.get(), 0}, {sinbuf.get(), 0}, pos.data());

    constexpr size_t compression = kCompression, budget = 4;
    const size_t stride = budget + compression - 1;
    BufferPtr selected = device.alloc((rows * stride + 2) * sizeof(uint32_t), backend::Memory::device);
    const uint32_t canary = 0xabcddcbau;
    std::vector<uint32_t> before(rows * stride + 2, canary);
    device.write(*selected, 0, before.data(), before.size() * sizeof(uint32_t));
    const Backend::IndexArgs args{kQHeads, data.dim, 4, compression, budget, 1e-6f};
    // The history's groups are pooled as the passes that wrote it would have pooled them (Backend::pool_kv).
    const KVView history[2] = {{data.storage.get(), data.maps[0].data(), data.maps[0].size(), 0, data.length},
                               {data.storage.get(), data.maps[1].data(), data.maps[1].size(), 0, data.length}};
    device.pool_kv(0, history, 2, args, {k_gamma.get(), 0}, {cosbuf.get(), 0}, {sinbuf.get(), 0});
    device.select_kv({norm_q.get(), 0}, 0, data.views.data(), data.views.size(), {selected.get(), 1}, args,
                     {k_gamma.get(), 0}, {cosbuf.get(), 0}, {sinbuf.get(), 0});

    BufferPtr q = upload(device, data.queries);
    BufferPtr out = device.alloc((rows * q_width + 2) * sizeof(float), backend::Memory::device);
    const float out_guard = -917.5f;
    std::vector<float> out_init(rows * q_width + 2, out_guard);
    device.write(*out, 0, out_init.data(), out_init.size() * sizeof(float));
    device.sparse_attention({q.get(), 0}, 0, data.views.data(), data.views.size(),
                            {selected.get(), 1}, stride, {out.get(), 1},
                            int(kQHeads), int(kKVHeads), int(data.dim));
    device.sync();
    const auto selected_guarded = read_ids(device, *selected, 0, before.size());
    require(selected_guarded.front() == canary && selected_guarded.back() == canary,
            "selector-to-attention chain changed ID guards");
    const std::vector<uint32_t> selected_ids = read_ids(device, *selected, 1, rows * stride);
    const auto output = read<float>(device, *out, 0, out_init.size());
    require(output.front() == out_guard && output.back() == out_guard,
            "selector-to-attention chain changed output guards");
    const std::vector<float> actual(output.begin() + 1, output.end() - 1);
    const double error = check_against_oracle(data, selected_ids, actual,
                                              "selector-to-sparse-attention chain", stride);
    std::cout << "selector-to-sparse-attention d8/K" << backend::kv_type_name(data.k_type)
              << "/V" << backend::kv_type_name(data.v_type) << " max_abs_error=" << error << '\n';
}

void refusal_checks(Backend& device, const Data& data) {
    const size_t rows = 2 * kNq, q_width = kQHeads * data.dim;
    BufferPtr q = upload(device, data.queries), ids = upload(device, data.ids);
    BufferPtr out = device.alloc((rows * q_width + 2) * sizeof(float), backend::Memory::device);
    const std::vector<float> before(rows * q_width + 2, 431.25f);
    device.write(*out, 0, before.data(), before.size() * sizeof(float));
    const auto reject_unchanged = [&](const KVView* views, size_t n_views, CSlice qslice,
                                      CSlice id_slice, size_t stride, Slice output,
                                      const std::vector<float>& output_before,
                                      int q_heads, int kv_heads, int head_dim, const std::string& label) {
        bool rejected = false;
        try { device.sparse_attention(qslice, 0, views, n_views, id_slice, stride, output,
                                      q_heads, kv_heads, head_dim); }
        catch (const std::exception&) { rejected = true; }
        require(rejected, label + " was accepted");
        device.sync();
        require(read<float>(device, *output.buffer, 0, output_before.size()) == output_before,
                label + " modified output before refusal");
    };
    const auto snapshot_bytes = [&](const backend::Buffer& buffer) {
        std::vector<uint8_t> bytes(buffer.size());
        if (!bytes.empty()) device.read(buffer, 0, bytes.data(), bytes.size());
        return bytes;
    };
    const auto reject_alias_unchanged = [&](CSlice qslice, CSlice id_slice, Slice output,
                                            backend::Buffer& watched, const std::string& label) {
        const auto before_bytes = snapshot_bytes(watched);
        bool rejected = false;
        try { device.sparse_attention(qslice, 0, data.views.data(), data.views.size(), id_slice,
                                      data.id_stride, output, int(kQHeads), int(kKVHeads), int(data.dim)); }
        catch (const std::exception&) { rejected = true; }
        require(rejected, label + " was accepted");
        device.sync();
        require(snapshot_bytes(watched) == before_bytes, label + " modified an aliased buffer before refusal");
    };

    const size_t output_bytes = rows * q_width * sizeof(float);
    reject_alias_unchanged({q.get(), 0}, {ids.get(), 0}, {q.get(), 0}, *q,
                           "sparse output overlapping query");

    BufferPtr shared_ids_out = device.alloc(std::max(output_bytes, data.ids.size() * sizeof(uint32_t)),
                                            backend::Memory::device);
    device.write(*shared_ids_out, 0, data.ids.data(), data.ids.size() * sizeof(uint32_t));
    reject_alias_unchanged({q.get(), 0}, {shared_ids_out.get(), 0}, {shared_ids_out.get(), 0},
                           *shared_ids_out, "sparse output overlapping IDs");

    auto* storage = dynamic_cast<backend::BlockKVStorage*>(data.storage.get());
    require(storage != nullptr, "sparse test storage is not block KV storage");
    reject_alias_unchanged({q.get(), 0}, {ids.get(), 0}, {storage->k_buffer(0).get(), 0},
                           *storage->k_buffer(0), "sparse output overlapping K storage");
    reject_alias_unchanged({q.get(), 0}, {ids.get(), 0}, {storage->v_buffer(0).get(), 0},
                           *storage->v_buffer(0), "sparse output overlapping V storage");
    if (data.dim == 8) {
        auto& auxiliary = *storage->auxiliary_buffer(0);
        require(auxiliary.size() >= output_bytes, "sparse auxiliary buffer is too small for alias check");
        reject_alias_unchanged({q.get(), 0}, {ids.get(), 0}, {&auxiliary, 0}, auxiliary,
                               "sparse output overlapping auxiliary storage");
    }

    reject_unchanged(data.views.data(), data.views.size(), {q.get(), 0}, {ids.get(), 0}, data.id_stride,
                     {out.get(), 1}, before, 3, int(kKVHeads), int(data.dim), "invalid sparse GQA ratio");
    reject_unchanged(data.views.data(), data.views.size(), {q.get(), 0}, {ids.get(), 0}, 0,
                     {out.get(), 1}, before, int(kQHeads), int(kKVHeads), int(data.dim), "zero sparse ID stride");
    BufferPtr short_ids = device.alloc((rows * data.id_stride - 1) * sizeof(uint32_t), backend::Memory::device);
    const std::vector<uint32_t> short_init(rows * data.id_stride - 1, 0x98765432u);
    device.write(*short_ids, 0, short_init.data(), short_init.size() * sizeof(uint32_t));
    reject_unchanged(data.views.data(), data.views.size(), {q.get(), 0}, {short_ids.get(), 1}, data.id_stride,
                     {out.get(), 1}, before, int(kQHeads), int(kKVHeads), int(data.dim), "short sparse ID span");
    BufferPtr short_out = device.alloc(rows * q_width * sizeof(float), backend::Memory::device);
    const std::vector<float> short_before(rows * q_width, 97.5f);
    device.write(*short_out, 0, short_before.data(), short_before.size() * sizeof(float));
    reject_unchanged(data.views.data(), data.views.size(), {q.get(), 0}, {ids.get(), 0}, data.id_stride,
                     {short_out.get(), 1}, short_before, int(kQHeads), int(kKVHeads), int(data.dim), "short sparse output span");

    const size_t per_view = data.maps[0].size();
    auto wrong = device.kv_alloc(1, kKVHeads + 1, data.dim,
                                 2 * per_view * data.block_tokens,
                                 data.k_type, data.v_type, {kAuxDim});
    wrong->back_all();
    std::vector<int32_t> wrong_map = block_map(backend::blocks_for(data.total_tokens, data.block_tokens),
                                               0, per_view);
    KVView mixed[2] = {
        data.views[0],
        KVView{wrong.get(), wrong_map.data(), wrong_map.size(), data.length, kNq}
    };
    reject_unchanged(mixed, 2, {q.get(), 0}, {ids.get(), 0}, data.id_stride,
                     {out.get(), 1}, before, int(kQHeads), int(kKVHeads), int(data.dim),
                     "invalid second sparse KV view");
}

// The same rows' ids at Flash-Next's stride of 2051 slots, each id in a slot of its own spread over the stride and the rest padding, so a device that splits a row's ids by the stride reads ids in several splits and splits of none.
std::vector<uint32_t> spread_ids(const Data& data, size_t stride) {
    std::vector<uint32_t> out(2 * kNq * stride, UINT32_MAX);
    for (size_t r = 0; r < 2 * kNq; ++r) {
        size_t n = 0;
        while (n < data.id_stride && data.ids[r * data.id_stride + n] != UINT32_MAX) ++n;
        for (size_t i = 0; i < n; ++i) out[r * stride + (n > 1 ? i * (stride - 1) / (n - 1) : 0)] = data.ids[r * data.id_stride + i];
    }
    return out;
}

// The kernels launched since the last reading, on a ROCm device, and nothing elsewhere.
std::map<std::string, size_t> launches(Backend& device, const std::string& name) {
    if (name.rfind("rocm", 0) != 0) return {};
#if LLMX_HAS_BACKEND_ROCM
    return backend::rocm_take_launches(device);
#else
    (void)device;
    return {};
#endif
}

void run_case(Backend& device, const std::string& name, size_t dim, backend::KVType k_type, backend::KVType v_type) {
    Data data = make_data(device, dim, k_type, v_type);
    const std::string label = "d" + std::to_string(dim) + "/K" + backend::kv_type_name(k_type) +
                              "/V" + backend::kv_type_name(v_type);
    const auto actual = run_batch(device, data, data.ids, data.id_stride, label);
    const double error = check_against_oracle(data, data.ids, actual, label);
    // Flash-Next's stride, split in several; and a stride past 256 splits of the profile's ids, whose split doubles so the merge's splits cover it.
    for (const size_t stride : {size_t(2051), size_t(65537)}) {
        const std::vector<uint32_t> spread = spread_ids(data, stride);
        const std::string at = label + " at stride " + std::to_string(stride);
        launches(device, name);
        const auto wide = run_batch(device, data, spread, stride, at);
        check_against_oracle(data, spread, wide, at, stride);
        // On ROCm heads 128 and 256 wide take the per-row body's sparse build and its split merge, other widths the chunk body's.
        const auto ran = launches(device, name);
        if (name.rfind("rocm", 0) == 0) {
            const std::string body = dim == 256 ? "llmx_sparse_attention_vec256" : dim == 128 ? "llmx_sparse_attention_vec" : "llmx_sparse_attention_chunk";
            const std::string merge = dim == 256 ? "llmx_sparse_attention_vec_merge256" : dim == 128 ? "llmx_sparse_attention_vec_merge" : "llmx_sparse_attention_merge";
            require(ran.size() == 2 && ran.count(body) && ran.count(merge), at + " ran other kernels than " + body + " and " + merge + " on " + name);
        }
        check_per_row(device, data, spread, stride, wide, at);
    }
    std::cout << label << " max_abs_error=" << error << '\n';
    check_per_row(device, data, data.ids, data.id_stride, actual, label);
    refusal_checks(device, data);
    if (dim == 8) selector_chain(device, data);
}


// The kernels a device timed since it was last asked, by name; empty where it times none.
std::vector<std::string> timed_kernels(Backend& device) {
    std::vector<std::string> names;
#if LLMX_HAS_BACKEND_VULKAN
    for (const auto& k : backend::vulkan_kernel_times(device)) names.push_back(k.first);
#else
    (void)device;
#endif
    return names;
}

// Grouped heads at width 256 (VulkanBackend::sparse_attention): every row of a batch must give the bits it gives alone, where one head takes a workgroup.
// Strides 78, 85 and 93 leave every remainder of a stream's four-token loop, and ids past a row's visible tokens land in every loop.
void grouped_heads(Backend& device, const std::string& name, backend::KVType k_type, backend::KVType v_type) {
    const size_t dim = 256, kv_heads = 2, bt = device.kv_layout().block_tokens, length = 3 * bt + 5;
    uint32_t seed = 4242;
    const auto next = [&seed] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    bool grouped_seen = false, timed = false, alone_grouped = false;
    for (const size_t per_kv : {size_t(2), size_t(3), size_t(4)})
        for (const size_t stride : {size_t(78), size_t(85), size_t(93)}) {
            const size_t heads = per_kv * kv_heads;
            const std::string label = name + " d256/K" + backend::kv_type_name(k_type) + "/V" + backend::kv_type_name(v_type) + ", " + std::to_string(heads) +
                                      " query heads, stride " + std::to_string(stride);
            // Rows double until the device takes the grouped build, where it says which kernels ran.
            for (size_t rows = 256;; rows *= 2) {
                const size_t total = length + rows;
                auto storage = device.kv_alloc(1, kv_heads, dim, backend::blocks_for(total, bt) * bt, k_type, v_type);
                std::vector<int32_t> map(backend::blocks_for(total, bt));
                for (size_t i = 0; i < map.size(); ++i) map[i] = int32_t(map.size() - 1 - i);
                std::vector<float> keys(total * kv_heads * dim), values(keys.size()), queries(rows * heads * dim);
                for (size_t i = 0; i < keys.size(); ++i) {
                    keys[i] = float(int(next() % 2001u) - 1000) / 1600.0f;
                    values[i] = float(int(next() % 2001u) - 1000) / 900.0f;
                }
                for (float& x : queries) x = float(int(next() % 2001u) - 1000) / 700.0f;
                std::vector<uint32_t> ids(rows * stride);
                for (size_t r = 0; r < rows; ++r) {
                    const size_t visible = length + r + 1;
                    for (size_t t = 0; t < stride; ++t) {
                        const uint32_t pick = next();
                        ids[r * stride + t] = (t * 7 + r) % 6 == 0 ? (pick % 2 ? UINT32_MAX : uint32_t(visible + pick % (total - visible + 1))) : uint32_t(pick % visible);
                    }
                }
                KVView write{storage.get(), map.data(), map.size(), 0, total};
                BufferPtr kb = upload(device, keys), vb = upload(device, values);
                device.kv_write(0, &write, 1, {kb.get(), 0}, {vb.get(), 0});
                device.sync();
                BufferPtr q = upload(device, queries), idbuf = upload(device, ids);
                const size_t width = heads * dim;
                BufferPtr out = device.alloc(rows * width * sizeof(float), backend::Memory::device);
                KVView view{storage.get(), map.data(), map.size(), length, rows};
                timed_kernels(device);
                device.sparse_attention({q.get(), 0}, 0, &view, 1, {idbuf.get(), 0}, stride, {out.get(), 0}, int(heads), int(kv_heads), int(dim));
                device.sync();
                const auto batch_kernels = timed_kernels(device);
                const bool grouped = std::any_of(batch_kernels.begin(), batch_kernels.end(), [](const std::string& k) { return k.find("_g4") != std::string::npos; });
                timed = timed || !batch_kernels.empty();
                if (!batch_kernels.empty() && !grouped && rows < 8192) continue;
                require(batch_kernels.empty() || grouped, label + ": the grouped build was not reached at " + std::to_string(rows) + " rows");
                grouped_seen = grouped_seen || grouped;
                const auto batch = read<float>(device, *out, 0, rows * width);
                BufferPtr alone = device.alloc(width * sizeof(float), backend::Memory::device);
                for (size_t r = 0; r < rows; ++r) {
                    KVView one{storage.get(), map.data(), map.size(), length + r, 1};
                    device.sparse_attention({q.get(), r * width}, 0, &one, 1, {idbuf.get(), r * stride}, stride, {alone.get(), 0}, int(heads), int(kv_heads), int(dim));
                    device.sync();
                    if (r == 0) {
                        const auto one_kernels = timed_kernels(device);
                        alone_grouped = alone_grouped || std::any_of(one_kernels.begin(), one_kernels.end(), [](const std::string& k) { return k.find("_g4") != std::string::npos; });
                    }
                    const auto got = read<float>(device, *alone, 0, width);
                    require(std::memcmp(got.data(), batch.data() + r * width, width * sizeof(float)) == 0,
                            label + ": row " + std::to_string(r) + " of " + std::to_string(rows) + " differs bitwise from the row alone");
                }
                break;
            }
        }
    std::cout << name << " d256/K" << backend::kv_type_name(k_type) << "/V" << backend::kv_type_name(v_type) << " grouped heads: every row the row alone bit for bit"
              << (timed ? (grouped_seen ? ", the grouped build reached" : "") : ", kernels not timed on this device")
              << (alone_grouped ? ", a row alone grouped too on this device's few compute units" : "") << '\n';
}

// Each call's time at Flash-Next's main attention, 24 heads 256 wide over 2 KV heads in f16 caches, 2051 ids of a 16384-token history, rows of one and of 16: printed, not checked.
void rates(Backend& device, const std::string& name) {
    const size_t heads = 24, kv_heads = 2, dim = 256, stride = 2051, history = 16384, bt = device.kv_layout().block_tokens;
    const size_t tokens = history + 16, blocks = backend::blocks_for(tokens, bt);
    auto storage = device.kv_alloc(1, kv_heads, dim, blocks * bt, backend::KVType::f16, backend::KVType::f16, {kAuxDim}, {kCompression});
    std::vector<int32_t> map(blocks);
    std::iota(map.begin(), map.end(), 0);
    for (size_t at = 0; at < tokens; at += 1024) {
        const size_t n = std::min<size_t>(1024, tokens - at);
        std::vector<float> kv(n * kv_heads * dim), aux(n * kAuxDim, 0.0f);
        for (size_t i = 0; i < kv.size(); ++i) kv[i] = key_value(0, at + i / (kv_heads * dim), (i / dim) % kv_heads, i % dim);
        BufferPtr k = upload(device, kv), a = upload(device, aux);
        const KVView w{storage.get(), map.data(), map.size(), at, n};
        device.kv_write(0, &w, 1, {k.get(), 0}, {k.get(), 0}, {a.get(), 0});
    }
    std::vector<float> query(16 * heads * dim);
    for (size_t i = 0; i < query.size(); ++i) query[i] = query_value(0, i / (heads * dim), (i / dim) % heads, i % dim);
    // 512 groups of four tokens spread over the history, then the row's causal tail, as a selection writes them.
    std::vector<uint32_t> ids(16 * stride, UINT32_MAX);
    for (size_t r = 0; r < 16; ++r) {
        for (size_t g = 0; g < 512; ++g)
            for (size_t t = 0; t < 4; ++t) ids[r * stride + g * 4 + t] = uint32_t(g * (history / 512) + t);
        for (size_t t = 0; t < 3; ++t) ids[r * stride + 2048 + t] = uint32_t(history - 3 + t);
    }
    BufferPtr q = upload(device, query), idb = upload(device, ids);
    BufferPtr out = device.alloc(16 * heads * dim * sizeof(float), backend::Memory::device);
    std::cout << "sparse-attention: rates on " << name << "; rows: us a call\n";
    for (const size_t rows : {size_t(1), size_t(16)}) {
        const KVView view{storage.get(), map.data(), map.size(), history - rows + 1, rows};
        const auto call = [&] { device.sparse_attention({q.get(), 0}, 0, &view, 1, {idb.get(), 0}, stride, {out.get(), 0}, int(heads), int(kv_heads), int(dim)); };
        for (int i = 0; i < 3; ++i) call();
        device.sync();
        const int iters = 50;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) call();
        device.sync();
        std::printf("sparse-attention: rate %zu: %.1f us\n", rows, 1e6 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters);
    }
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string device_name = "cpu";
        bool timing = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--device" && i + 1 < argc) device_name = argv[++i];
            else if (arg == "--rates") timing = true;
            else throw std::runtime_error("usage: sparse-attention-test [--device cpu|vulkan:N|rocm:N] [--rates]");
        }
        backend::BackendPtr device = backend::make_backend(device_name, 1);
        // `--rates` prints each call's time and checks nothing.
        if (timing) {
            rates(*device, device_name);
            return 0;
        }
        for (size_t dim : {size_t(8), size_t(128), size_t(256)})
            for (backend::KVType k_type : {backend::KVType::f32, backend::KVType::f16})
                for (backend::KVType v_type : {backend::KVType::f32, backend::KVType::f16})
                    run_case(*device, device_name, dim, k_type, v_type);
        for (backend::KVType k_type : {backend::KVType::f32, backend::KVType::f16})
            for (backend::KVType v_type : {backend::KVType::f32, backend::KVType::f16})
                grouped_heads(*device, device_name, k_type, v_type);
        std::cout << "sparse-attention: PASS (" << checks << " checks) on " << device_name << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sparse-attention: FAIL: " << error.what() << '\n';
        return 1;
    }
}
