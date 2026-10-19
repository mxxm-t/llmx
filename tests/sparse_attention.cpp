// Sparse causal GQA against an independent double-precision oracle.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backends/devices.hpp"
#include "backends/kv_storage.hpp"
#include "core/fp16.hpp"

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
                                   k_type, v_type, {kAuxDim});
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

void check_per_row(Backend& device, const Data& data, const std::vector<uint32_t>& ids,
                   const std::vector<float>& batch, const std::string& label) {
    const size_t q_width = kQHeads * data.dim, out_width = q_width;
    for (size_t seq = 0; seq < 2; ++seq) for (size_t row = 0; row < kNq; ++row) {
        const size_t global = seq * kNq + row;
        KVView view{data.storage.get(), data.maps[seq].data(), data.maps[seq].size(),
                    data.length + row, 1};
        BufferPtr q = upload(device, std::vector<float>(data.queries.begin() + global * q_width,
                                                         data.queries.begin() + (global + 1) * q_width));
        std::vector<uint32_t> row_ids(ids.begin() + global * data.id_stride,
                                      ids.begin() + (global + 1) * data.id_stride);
        BufferPtr idbuf = upload(device, row_ids);
        BufferPtr out = device.alloc((out_width + 2) * sizeof(float), backend::Memory::device);
        std::vector<float> init(out_width + 2, -818.25f);
        device.write(*out, 0, init.data(), init.size() * sizeof(float));
        device.sparse_attention({q.get(), 0}, 0, &view, 1, {idbuf.get(), 0}, data.id_stride,
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

    constexpr size_t compression = 2, budget = 4;
    const size_t stride = budget + compression - 1;
    BufferPtr selected = device.alloc((rows * stride + 2) * sizeof(uint32_t), backend::Memory::device);
    const uint32_t canary = 0xabcddcbau;
    std::vector<uint32_t> before(rows * stride + 2, canary);
    device.write(*selected, 0, before.data(), before.size() * sizeof(uint32_t));
    const Backend::IndexArgs args{kQHeads, data.dim, 4, compression, budget, 1e-6f};
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

void run_case(Backend& device, size_t dim, backend::KVType k_type, backend::KVType v_type) {
    Data data = make_data(device, dim, k_type, v_type);
    const std::string label = "d" + std::to_string(dim) + "/K" + backend::kv_type_name(k_type) +
                              "/V" + backend::kv_type_name(v_type);
    const auto actual = run_batch(device, data, data.ids, data.id_stride, label);
    const double error = check_against_oracle(data, data.ids, actual, label);
    std::cout << label << " max_abs_error=" << error << '\n';
    check_per_row(device, data, data.ids, actual, label);
    refusal_checks(device, data);
    if (dim == 8) selector_chain(device, data);
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string device_name = "cpu";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--device" && i + 1 < argc) device_name = argv[++i];
            else throw std::runtime_error("usage: sparse-attention-test [--device cpu|vulkan:N]");
        }
        backend::BackendPtr device = backend::make_backend(device_name, 1);
        for (size_t dim : {size_t(8), size_t(128), size_t(256)})
            for (backend::KVType k_type : {backend::KVType::f32, backend::KVType::f16})
                for (backend::KVType v_type : {backend::KVType::f32, backend::KVType::f16})
                    run_case(*device, dim, k_type, v_type);
        std::cout << "sparse-attention: PASS (" << checks << " checks) on " << device_name << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sparse-attention: FAIL: " << error.what() << '\n';
        return 1;
    }
}
