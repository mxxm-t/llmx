// QSA index selection against pinned HF intermediates and an independent oracle.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backends/devices.hpp"
#include "core/json.hpp"

namespace {
using backend::Backend;
using backend::BufferPtr;
using backend::CSlice;
using backend::Slice;
using Json = jmini::Value;

size_t checks = 0;
void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

const Json& field(const Json& value, const std::string& key) {
    const Json* found = value.get(key);
    if (!found) throw std::runtime_error("reference: missing field " + key);
    return *found;
}

std::vector<float> floats(const Json& value) {
    std::vector<float> out;
    const auto append = [&](auto&& self, const Json& v) -> void {
        if (v.isNumber()) out.push_back(float(v.asNumber()));
        else if (v.isArray()) for (const Json& item : v.asArray()) self(self, item);
        else throw std::runtime_error("reference: expected numeric array");
    };
    append(append, value);
    return out;
}

size_t integer(const Json& value, const std::string& key) {
    const double n = field(value, key).asNumber();
    if (n < 0 || n > double(std::numeric_limits<size_t>::max()) || std::floor(n) != n)
        throw std::runtime_error("reference: invalid integer " + key);
    return size_t(n);
}

std::vector<std::vector<uint32_t>> id_rows(const Json& value) {
    std::vector<std::vector<uint32_t>> out;
    for (const Json& row : value.asArray()) {
        std::vector<uint32_t> ids;
        for (const Json& item : row.asArray()) {
            const double n = item.asNumber();
            if (n < 0 || n > double(std::numeric_limits<uint32_t>::max()) || std::floor(n) != n)
                throw std::runtime_error("reference: invalid selected token id");
            ids.push_back(uint32_t(n));
        }
        out.push_back(std::move(ids));
    }
    return out;
}

BufferPtr upload(Backend& device, const std::vector<float>& values) {
    BufferPtr out = device.alloc(std::max<size_t>(values.size(), 1) * sizeof(float), backend::Memory::device);
    if (!values.empty()) device.write(*out, 0, values.data(), values.size() * sizeof(float));
    return out;
}

template<class T>
std::vector<T> read_values(Backend& device, const backend::Buffer& buffer, size_t offset, size_t count) {
    std::vector<T> out(count);
    if (count) device.read(buffer, offset, out.data(), count * sizeof(T));
    return out;
}

struct QsaPhase {
    std::vector<float> query;
    std::vector<float> raw_keys;
    std::vector<std::vector<uint32_t>> selected;
};

QsaPhase phase(const Json& item) {
    QsaPhase out;
    out.query = floats(field(item, "query_projection_pre_norm_rope"));
    out.raw_keys = floats(field(item, "raw_index_keys_pre_pool_norm_rope"));
    out.selected = id_rows(field(item, "selected_token_ids"));
    return out;
}

struct FixtureCase {
    std::string name;
    std::vector<uint32_t> input_ids;
    std::optional<uint32_t> decode_token;
    QsaPhase full, sliced, cached;
    std::optional<QsaPhase> decode;
};

std::vector<FixtureCase> fixture_cases(const Json& root) {
    std::vector<FixtureCase> out;
    for (const Json& item : field(root, "cases").asArray()) {
        FixtureCase c;
        c.name = field(item, "name").asString();
        for (const Json& id : field(item, "input_ids").asArray()) c.input_ids.push_back(uint32_t(id.asNumber()));
        const Json& decode = field(item, "decode_token");
        if (decode.t != Json::T::Null) c.decode_token = uint32_t(decode.asNumber());
        const Json& trace = field(item, "qsa_indexer");
        c.full = phase(field(trace, "full"));
        c.sliced = phase(field(trace, "sliced"));
        c.cached = phase(field(trace, "cached"));
        if (field(trace, "decode").t != Json::T::Null) c.decode = phase(field(trace, "decode"));
        out.push_back(std::move(c));
    }
    return out;
}

struct Config {
    size_t heads = 0, dim = 0, rope_dim = 0, compression = 0, budget = 0;
    float eps = 0.0f, theta = 0.0f;
};

Config fixture_config(const Json& root) {
    const Json& c = field(root, "config");
    Config out;
    out.heads = integer(c, "indexer_n_heads");
    out.dim = integer(c, "indexer_head_dim");
    out.rope_dim = size_t(double(field(field(c, "rope_parameters"), "partial_rotary_factor").asNumber()) * out.dim);
    out.compression = integer(c, "indexer_compress_ratio");
    out.budget = integer(c, "indexer_budget");
    out.eps = float(field(c, "rms_norm_eps").asNumber());
    out.theta = float(field(field(c, "rope_parameters"), "rope_theta").asNumber());
    require(out.heads == 4 && out.dim == 8 && out.rope_dim == 4 && out.compression == 2 && out.budget == 4,
            "tiny QSA fixture geometry changed");
    return out;
}

std::vector<float> state_weight(const Json& state, const std::string& name, size_t width) {
    const Json& tensor = field(state, name);
    const auto& shape = field(tensor, "shape").asArray();
    require(shape.size() == 1 && size_t(shape[0].asNumber()) == width,
            name + " shape differs from the QSA graph");
    std::vector<float> out = floats(field(tensor, "values"));
    require(out.size() == width, name + " element count differs from the QSA graph");
    // HF Qwen4Exp stores zero-centered RMSNorm gamma; the backend API takes the GGUF multiplier.
    for (float& x : out) x += 1.0f;
    return out;
}

struct RopeTables {
    std::vector<float> cos, sin;
};

RopeTables rope_tables(const Config& c, size_t positions) {
    RopeTables out;
    const size_t half = c.rope_dim / 2;
    out.cos.resize(positions * half);
    out.sin.resize(positions * half);
    for (size_t p = 0; p < positions; ++p) for (size_t i = 0; i < half; ++i) {
        const double frequency = std::pow(double(c.theta), -2.0 * double(i) / double(c.rope_dim));
        const float angle = float(double(p) * frequency);
        out.cos[p * half + i] = std::cos(angle);
        out.sin[p * half + i] = std::sin(angle);
    }
    return out;
}

struct Cache {
    Backend& device;
    std::unique_ptr<backend::KVStorage> storage;
    std::vector<int32_t> blocks;
    size_t kv_row = 0;
    size_t block_tokens = 0;

    // The auxiliary blocks keep the rows and the pooled row of each complete group of `pool` tokens, as a QSA layer's do.
    Cache(Backend& b, size_t max_tokens, size_t aux_width, size_t pool, size_t kv_width = 1)
        : device(b), block_tokens(b.kv_layout().block_tokens), kv_row(kv_width) {
        const size_t block_count = (max_tokens + block_tokens - 1) / block_tokens;
        const size_t padded_tokens = block_count * block_tokens;
        storage = device.kv_alloc(1, 1, kv_width, padded_tokens, backend::KVType::f32,
                                  backend::KVType::f32, {aux_width}, {pool});
        blocks.resize(block_count);
        for (size_t i = 0; i < block_count; ++i) blocks[i] = int32_t(i);
    }

    backend::KVView view(size_t length, size_t rows, const std::vector<int32_t>& ids) {
        return {storage.get(), ids.data(), ids.size(), length, rows};
    }

    void write(const backend::KVView* views, size_t n_views, const std::vector<float>& raw_keys,
               size_t rows) {
        std::vector<float> k(rows * kv_row, 0.0f), v(rows * kv_row, 0.0f);
        BufferPtr kb = upload(device, k), vb = upload(device, v), ab = upload(device, raw_keys);
        device.kv_write(0, views, n_views, {kb.get(), 0}, {vb.get(), 0}, {ab.get(), 0});
        device.sync();
    }
};

std::vector<uint32_t> run_selection(Backend& device, const Config& c,
                                    const backend::KVView* views, size_t n_views,
                                    const std::vector<float>& query_raw, size_t rows,
                                    const std::vector<float>& qnorm, const std::vector<float>& knorm,
                                    const RopeTables& rope,
                                    const std::vector<uint32_t>& positions,
                                    const std::vector<std::vector<uint32_t>>* expected,
                                    const std::string& label, bool unordered = false) {
    const size_t row_width = c.heads * c.dim;
    require(query_raw.size() == rows * row_width && positions.size() == rows,
            label + " query dimensions differ");
    BufferPtr qsrc = upload(device, query_raw);
    BufferPtr qbuf = device.alloc(std::max<size_t>(rows * row_width, 1) * sizeof(float), backend::Memory::device);
    BufferPtr q_gamma = upload(device, qnorm), k_gamma = upload(device, knorm);
    BufferPtr cos = upload(device, rope.cos), sin = upload(device, rope.sin);
    const size_t width = c.budget + c.compression - 1;
    BufferPtr ids = device.alloc((rows * width + 2) * sizeof(float), backend::Memory::device);
    const float guard = -913.25f;
    std::vector<float> initialized(rows * width + 2, guard);
    device.write(*ids, 0, initialized.data(), initialized.size() * sizeof(float));
    const size_t max_position = rope.cos.size() / (c.rope_dim / 2);
    require(max_position >= 1, label + " empty RoPE table");

    device.norm_rope_partial({qbuf.get(), 0}, {qsrc.get(), 0}, rows, row_width, c.dim,
                             c.heads, c.dim, c.rope_dim, {q_gamma.get(), 0}, c.eps,
                             {cos.get(), 0}, {sin.get(), 0}, positions.data());
    const Backend::IndexArgs args{c.heads, c.dim, c.rope_dim, c.compression, c.budget, c.eps};
    device.unordered(unordered);
    // The rows written before the selected ones pool their groups first, as the passes that wrote them would have (Backend::pool_kv); groups pooled already are pooled again to the same values.
    std::vector<backend::KVView> history;
    for (size_t v = 0; v < n_views; ++v)
        if (views[v].length) history.push_back({views[v].storage, views[v].blocks, views[v].n_blocks, 0, views[v].length});
    if (!history.empty()) device.pool_kv(0, history.data(), history.size(), args, {k_gamma.get(), 0}, {cos.get(), 0}, {sin.get(), 0});
    device.select_kv({qbuf.get(), 0}, 0, views, n_views, {ids.get(), 1}, args,
                     {k_gamma.get(), 0}, {cos.get(), 0}, {sin.get(), 0});
    device.unordered(false);
    device.sync();

    const std::vector<float> guards = read_values<float>(device, *ids, 0, initialized.size());
    require(guards.front() == guard && guards.back() == guard, label + " changed output guards");
    const size_t output_rows = rows;
    std::vector<uint32_t> flat = read_values<uint32_t>(device, *ids, sizeof(float), output_rows * width);
    std::vector<uint32_t> result;
    for (size_t r = 0; r < output_rows; ++r) {
        size_t valid = 0;
        while (valid < width && flat[r * width + valid] != UINT32_MAX) ++valid;
        for (size_t i = valid; i < width; ++i)
            require(flat[r * width + i] == UINT32_MAX, label + " has a non-sentinel after padding");
        if (expected) {
            require(r < expected->size(), label + " expected row count differs");
            const auto& row = expected->at(r);
            require(valid == row.size(), label + " selected count differs on row " + std::to_string(r));
            for (size_t i = 0; i < valid; ++i)
                require(flat[r * width + i] == row[i], label + " selected ID differs on row " + std::to_string(r));
        }
        result.insert(result.end(), flat.begin() + r * width, flat.begin() + r * width + valid);
    }
    return result;
}

std::vector<uint32_t> sequence_ids(size_t block_count, size_t salt) {
    if (!block_count) throw std::runtime_error("empty physical block map");
    std::vector<uint32_t> out(block_count);
    size_t step = 3;
    while (std::gcd(step, block_count) != 1) ++step;
    for (size_t i = 0; i < block_count; ++i) out[i] = uint32_t((i * step + salt) % block_count);
    return out;
}

std::vector<float> slice_rows(const std::vector<float>& values, size_t first, size_t count, size_t width) {
    return {values.begin() + first * width, values.begin() + (first + count) * width};
}

std::vector<uint32_t> positions(size_t first, size_t rows) {
    std::vector<uint32_t> out(rows);
    for (size_t i = 0; i < rows; ++i) out[i] = uint32_t(first + i);
    return out;
}

std::vector<uint32_t> fixture_full(Backend& device, const Config& c, const FixtureCase& test,
                                   const QsaPhase& data, const std::vector<float>& qnorm,
                                   const std::vector<float>& knorm,
                                   const RopeTables& rope, const std::string& label) {
    const size_t rows = test.input_ids.size(), key_width = c.dim;
    require(data.raw_keys.size() == rows * key_width && data.query.size() == rows * c.heads * c.dim && data.selected.size() == rows,
            label + " HF intermediate dimensions differ");
    Cache cache(device, 4 * device.kv_layout().block_tokens, key_width, c.compression);
    const std::vector<uint32_t> logical = sequence_ids(cache.blocks.size(), 0);
    std::vector<int32_t> table(logical.begin(), logical.end());
    backend::KVView write_view = cache.view(0, rows, table);
    cache.write(&write_view, 1, data.raw_keys, rows);
    backend::KVView select_view = cache.view(0, rows, table);
    const auto pos = positions(0, rows);
    return run_selection(device, c, &select_view, 1, data.query, rows, qnorm, knorm, rope, pos,
                         &data.selected, label);
}

std::vector<uint32_t> fixture_sliced(Backend& device, const Config& c, const FixtureCase& test,
                                     const std::vector<float>& qnorm, const std::vector<float>& knorm,
                                     const RopeTables& rope) {
    Cache cache(device, 4 * device.kv_layout().block_tokens, c.dim, c.compression);
    std::vector<int32_t> table(cache.blocks.begin(), cache.blocks.end());
    std::vector<uint32_t> result;
    size_t offset = 0;
    for (const size_t count : {size_t(3), size_t(2), size_t(4), size_t(4)}) {
        const QsaPhase& data = test.sliced;
        backend::KVView view = cache.view(offset, count, table);
        cache.write(&view, 1, slice_rows(data.raw_keys, offset, count, c.dim), count);
        view = cache.view(offset, count, table);
        const auto expected = std::vector<std::vector<uint32_t>>(data.selected.begin() + offset,
                                                                  data.selected.begin() + offset + count);
        const auto got = run_selection(device, c, &view, 1,
            slice_rows(data.query, offset, count, c.heads * c.dim), count, qnorm, knorm, rope,
            positions(offset, count), &expected,
            test.name + " sliced rows " + std::to_string(offset));
        result.insert(result.end(), got.begin(), got.end());
        offset += count;
    }
    require(offset == test.input_ids.size(), test.name + " slice plan did not cover prompt");
    return result;
}

std::vector<uint32_t> fixture_cached(Backend& device, const Config& c, const FixtureCase& test,
                                     const std::vector<float>& qnorm, const std::vector<float>& knorm,
                                     const RopeTables& rope) {
    Cache cache(device, 4 * device.kv_layout().block_tokens, c.dim, c.compression);
    std::vector<int32_t> table(cache.blocks.begin(), cache.blocks.end());
    std::vector<uint32_t> result;
    for (size_t offset = 0; offset < test.input_ids.size(); ++offset) {
        backend::KVView view = cache.view(offset, 1, table);
        cache.write(&view, 1, slice_rows(test.cached.raw_keys, offset, 1, c.dim), 1);
        view = cache.view(offset, 1, table);
        const std::vector<std::vector<uint32_t>> expected{test.cached.selected.at(offset)};
        const auto got = run_selection(device, c, &view, 1,
            slice_rows(test.cached.query, offset, 1, c.heads * c.dim), 1, qnorm, knorm, rope,
            positions(offset, 1), &expected,
            test.name + " cached token " + std::to_string(offset));
        result.insert(result.end(), got.begin(), got.end());
    }
    return result;
}

std::vector<uint32_t> fixture_decode(Backend& device, const Config& c, const FixtureCase& test,
                                     const std::vector<float>& qnorm, const std::vector<float>& knorm,
                                     const RopeTables& rope) {
    require(test.decode.has_value(), test.name + " is missing its HF decode trace");
    Cache cache(device, 4 * device.kv_layout().block_tokens, c.dim, c.compression);
    std::vector<int32_t> table(cache.blocks.begin(), cache.blocks.end());
    backend::KVView prefix = cache.view(0, test.input_ids.size(), table);
    cache.write(&prefix, 1, test.cached.raw_keys, test.input_ids.size());
    backend::KVView current = cache.view(test.input_ids.size(), 1, table);
    cache.write(&current, 1, test.decode->raw_keys, 1);
    current = cache.view(test.input_ids.size(), 1, table);
    const std::vector<uint32_t> pos = positions(test.input_ids.size(), 1);
    return run_selection(device, c, &current, 1, test.decode->query, 1, qnorm, knorm, rope, pos,
                         &test.decode->selected, test.name + " decode");
}

std::vector<uint32_t> flat_ids(const QsaPhase& data) {
    std::vector<uint32_t> out;
    for (const auto& row : data.selected) out.insert(out.end(), row.begin(), row.end());
    return out;
}

void append_group(std::vector<float>& keys, size_t count, size_t dim, double angle) {
    for (size_t t = 0; t < count; ++t) {
        std::vector<float> row(dim, 0.0f);
        row[0] = float(std::cos(angle));
        row[1] = float(std::sin(angle));
        keys.insert(keys.end(), row.begin(), row.end());
    }
}

std::vector<uint32_t> oracle_row(const std::vector<float>& keys, size_t row_tokens,
                                size_t heads, size_t dim, size_t compression,
                                size_t budget, float eps) {
    const size_t complete = row_tokens / compression;
    const size_t take = std::min(complete, budget / compression);
    struct Scored { double score; size_t group; };
    std::vector<Scored> scores;
    for (size_t group = 0; group < complete; ++group) {
        std::vector<double> pooled(dim, 0.0);
        for (size_t t = 0; t < compression; ++t)
            for (size_t d = 0; d < dim; ++d)
                pooled[d] += double(keys[(group * compression + t) * dim + d]) / double(compression);
        double mean_square = 0.0;
        for (double value : pooled) mean_square += value * value / double(dim);
        const double inv = 1.0 / std::sqrt(mean_square + double(eps));
        double score = 0.0;
        for (size_t h = 0; h < heads; ++h) {
            // The synthetic query is the unit vector along dimension zero in every head.
            score += std::max(0.0, pooled[0] * inv);
        }
        score /= std::sqrt(double(dim));
        scores.push_back({score, group});
    }
    std::stable_sort(scores.begin(), scores.end(), [](const Scored& a, const Scored& b) {
        return a.score > b.score;
    });
    std::vector<size_t> selected;
    for (size_t i = 0; i < take; ++i) selected.push_back(scores[i].group);
    std::sort(selected.begin(), selected.end());
    std::vector<uint32_t> out;
    for (size_t group : selected)
        for (size_t t = 0; t < compression; ++t) out.push_back(uint32_t(group * compression + t));
    const size_t tail_start = complete * compression;
    for (size_t t = 0; t < row_tokens % compression; ++t) out.push_back(uint32_t(tail_start + t));
    return out;
}

std::vector<int32_t> physical_map(size_t needed, size_t base, size_t capacity) {
    std::vector<int32_t> out;
    out.reserve(needed);
    for (size_t i = 0; i < needed; ++i) out.push_back(int32_t(base + ((i * 3) % capacity)));
    return out;
}

void synthetic_long(Backend& device) {
    Config c{4, 128, 4, 4, 2048, 1e-6f, 10000.0f};
    const size_t block_tokens = device.kv_layout().block_tokens;
    const size_t first_groups = 523, second_groups = 273;
    const size_t first_partial = 2, second_partial = 3, rows_per_view = 19;
    const size_t first_tokens = first_groups * c.compression + first_partial;
    const size_t second_tokens = second_groups * c.compression + second_partial;
    const size_t first_blocks = (first_tokens + block_tokens - 1) / block_tokens;
    const size_t second_blocks = (second_tokens + block_tokens - 1) / block_tokens;
    size_t per_view_capacity = 1;
    while (per_view_capacity < std::max(first_blocks, second_blocks)) per_view_capacity *= 2;
    const size_t physical_capacity = 2 * per_view_capacity;
    Cache cache(device, physical_capacity * block_tokens, c.dim, c.compression);
    std::vector<int32_t> first_map = physical_map(first_blocks, 0, per_view_capacity);
    std::vector<int32_t> second_map = physical_map(second_blocks, per_view_capacity, per_view_capacity);

    std::vector<float> all_keys;
    all_keys.reserve((first_tokens + second_tokens) * c.dim);
    for (size_t g = 0; g < first_groups; ++g)
        append_group(all_keys, c.compression, c.dim, double(g) * 0.0017);
    for (size_t t = 0; t < first_partial; ++t) append_group(all_keys, 1, c.dim, 0.2 + t * 0.1);
    const size_t second_offset = all_keys.size() / c.dim;
    for (size_t g = 0; g < second_groups; ++g)
        append_group(all_keys, c.compression, c.dim, 0.35 + double(g) * 0.0013);
    for (size_t t = 0; t < second_partial; ++t) append_group(all_keys, 1, c.dim, 0.6 + t * 0.1);

    backend::KVView writes[2] = {
        cache.view(0, first_tokens, first_map), cache.view(0, second_tokens, second_map)
    };
    cache.write(writes, 2, all_keys, first_tokens + second_tokens);
    backend::KVView views[2] = {
        cache.view(first_tokens - rows_per_view, rows_per_view, first_map),
        cache.view(second_tokens - rows_per_view, rows_per_view, second_map)
    };
    std::vector<float> queries(2 * rows_per_view * c.heads * c.dim, 0.0f);
    for (size_t r = 0; r < 2 * rows_per_view; ++r) for (size_t h = 0; h < c.heads; ++h)
        queries[(r * c.heads + h) * c.dim] = 1.0f;
    const size_t positions_count = std::max(first_tokens, second_tokens);
    RopeTables identity;
    identity.cos.assign(positions_count * (c.rope_dim / 2), 1.0f);
    identity.sin.assign(positions_count * (c.rope_dim / 2), 0.0f);
    std::vector<float> multiplier(c.dim, 1.0f);
    std::vector<uint32_t> pos;
    for (size_t i = 0; i < rows_per_view; ++i) pos.push_back(uint32_t(first_tokens - rows_per_view + i));
    for (size_t i = 0; i < rows_per_view; ++i) pos.push_back(uint32_t(second_tokens - rows_per_view + i));
    std::vector<std::vector<uint32_t>> expected{
    };
    const std::vector<float> second_keys(all_keys.begin() + second_offset * c.dim, all_keys.end());
    for (size_t i = 0; i < rows_per_view; ++i)
        expected.push_back(oracle_row(all_keys, first_tokens - rows_per_view + i + 1,
                                      c.heads, c.dim, c.compression, c.budget, c.eps));
    for (size_t i = 0; i < rows_per_view; ++i)
        expected.push_back(oracle_row(second_keys, second_tokens - rows_per_view + i + 1,
                                      c.heads, c.dim, c.compression, c.budget, c.eps));
    const auto got = run_selection(device, c, views, 2, queries, 2 * rows_per_view,
                                   multiplier, multiplier, identity, pos,
                                   &expected, "long synthetic selector");
    require(run_selection(device, c, views, 2, queries, rows_per_view * 2, multiplier, multiplier, identity, pos,
                          &expected, "unordered long synthetic selector", true) == got,
            "unordered selector differs");
    size_t expected_count = 0;
    for (const auto& row : expected) expected_count += row.size();
    require(got.size() == expected_count,
            "long synthetic selector output sizes differ");
    require(first_groups > c.budget / c.compression && second_groups > c.budget / c.compression / 2,
            "synthetic selector did not exceed candidate budgets");

    Config ties{4, 128, 4, 4, 8, 1e-6f, 10000.0f};
    const size_t tie_tokens = 273 * ties.compression + 2;
    Cache tie_cache(device, ((tie_tokens + block_tokens - 1) / block_tokens) * block_tokens, ties.dim, ties.compression);
    std::vector<int32_t> tie_map(tie_cache.blocks.begin(), tie_cache.blocks.end());
    std::vector<float> same(tie_tokens * ties.dim, 0.0f);
    for (size_t t = 0; t < 273 * ties.compression; ++t) same[t * ties.dim] = 1.0f;
    backend::KVView tie_write = tie_cache.view(0, tie_tokens, tie_map);
    tie_cache.write(&tie_write, 1, same, tie_tokens);
    backend::KVView tie_view = tie_cache.view(tie_tokens - 2, 2, tie_map);
    std::vector<float> tie_query(2 * ties.heads * ties.dim, 0.0f);
    for (size_t r = 0; r < 2; ++r) for (size_t h = 0; h < ties.heads; ++h)
        tie_query[(r * ties.heads + h) * ties.dim] = 1.0f;
    RopeTables tie_rope;
    tie_rope.cos.assign(tie_tokens * (ties.rope_dim / 2), 1.0f);
    tie_rope.sin.assign(tie_tokens * (ties.rope_dim / 2), 0.0f);
    std::vector<uint32_t> tie_pos{uint32_t(tie_tokens - 2), uint32_t(tie_tokens - 1)};
    const uint32_t tie_tail = uint32_t(273 * ties.compression);
    std::vector<std::vector<uint32_t>> tie_expected{{0,1,2,3,4,5,6,7,tie_tail},
                                                     {0,1,2,3,4,5,6,7,tie_tail,tie_tail + 1}};
    std::vector<float> tie_multiplier(ties.dim, 1.0f);
    run_selection(device, ties, &tie_view, 1, tie_query, 2, tie_multiplier, tie_multiplier, tie_rope, tie_pos,
                  &tie_expected, "equal-score earlier-group tie");
    ties.budget = 258 * ties.compression;
    std::vector<std::vector<uint32_t>> across(2);
    for (size_t r = 0; r < across.size(); ++r) {
        for (size_t t = 0; t < ties.budget; ++t) across[r].push_back(uint32_t(t));
        for (size_t t = 0; t <= r; ++t) across[r].push_back(tie_tail + uint32_t(t));
    }
    run_selection(device, ties, &tie_view, 1, tie_query, 2, tie_multiplier, tie_multiplier, tie_rope, tie_pos,
                  &across, "selected ties across scan boundary");
}

// The pooled rows are kept: pool_kv writes the rows of exactly the groups its rows complete, and a later selection reads a group's kept row rather than pooling it again.
// Group 0's kept row is replaced by one the query favours; pooled again from its raw keys it would lose to group 5, whose keys lean furthest toward the query.
void pooled_rows_kept(Backend& device) {
    const Config k{1, 8, 2, 4, 4, 1e-6f, 10000.0f};
    const size_t groups = 6, tokens = groups * k.compression + 1;
    Cache cache(device, tokens, k.dim, k.compression);
    std::vector<int32_t> map(cache.blocks.begin(), cache.blocks.end());
    std::vector<float> raw(tokens * k.dim, 1.0f);
    for (size_t t = 0; t < tokens; ++t) raw[t * k.dim] = 0.1f * float(t / k.compression + 1);
    backend::KVView all = cache.view(0, tokens, map);
    cache.write(&all, 1, raw, tokens);
    const std::vector<float> ones(k.dim, 1.0f), cosine(tokens + 1, 1.0f), sine(tokens + 1, 0.0f);
    BufferPtr gamma = upload(device, ones), cos = upload(device, cosine), sin = upload(device, sine);
    const Backend::IndexArgs args{k.heads, k.dim, k.rope_dim, k.compression, k.budget, k.eps};
    auto& storage = dynamic_cast<backend::BlockKVStorage&>(*cache.storage);
    backend::Buffer& plane = *storage.auxiliary_buffer(0);
    const auto row_at = [&](size_t g) {
        const size_t t = g * k.compression;
        return size_t(map[t / cache.block_tokens]) * storage.auxiliary_block_bytes(0) + storage.auxiliary_rows_bytes(0) + (t % cache.block_tokens) / k.compression * k.dim * sizeof(float);
    };
    require(storage.auxiliary_pool(0) == k.compression && storage.auxiliary_rows_bytes(0) == cache.block_tokens * k.dim * sizeof(float) &&
            storage.auxiliary_block_bytes(0) == (cache.block_tokens + cache.block_tokens / k.compression) * k.dim * sizeof(float),
            "the auxiliary blocks do not hold the token rows and then a row a group");
    // Rows 11 and 12 complete group 2 alone.
    backend::KVView two = cache.view(2 * k.compression + 3, 2, map);
    device.pool_kv(0, &two, 1, args, {gamma.get(), 0}, {cos.get(), 0}, {sin.get(), 0});
    for (size_t g = 0; g < groups; ++g) {
        const auto row = read_values<float>(device, plane, row_at(g), k.dim);
        const bool written = std::any_of(row.begin(), row.end(), [](float v) { return v != 0.0f; });
        require(written == (g == 2), "pool_kv wrote the pooled row of group " + std::to_string(g) + (written ? "" : " not") + " completed by its rows");
    }
    device.pool_kv(0, &all, 1, args, {gamma.get(), 0}, {cos.get(), 0}, {sin.get(), 0});
    std::vector<float> favoured(k.dim, 0.0f);
    favoured[0] = 100.0f;
    device.write(plane, row_at(0), favoured.data(), favoured.size() * sizeof(float));
    std::vector<float> query(k.heads * k.dim, 0.0f);
    query[0] = 1.0f;
    BufferPtr q = upload(device, query);
    const size_t stride = k.budget + k.compression - 1;
    BufferPtr out = device.alloc(stride * sizeof(uint32_t), backend::Memory::device);
    // One more row at the end of the history reads every complete group's kept row.
    std::vector<float> one(k.dim, 1.0f);
    std::vector<int32_t> grown = map;
    backend::KVView next = cache.view(tokens, 1, grown);
    cache.write(&next, 1, one, 1);
    next = cache.view(tokens, 1, grown);
    device.select_kv({q.get(), 0}, 0, &next, 1, {out.get(), 0}, args, {gamma.get(), 0}, {cos.get(), 0}, {sin.get(), 0});
    const auto ids = read_values<uint32_t>(device, *out, 0, stride);
    require(ids[0] == 0 && ids[k.compression - 1] == k.compression - 1, "a selection pooled group 0 again rather than reading its kept row, taking group " +
                                                                        std::to_string(ids[0] / k.compression));
}


// Pooled groups turning at the positions a history with images gives them (backend::GroupPositions): each pair by the position its stream names, against a host reference of the mean, the norm and the rotation within 4 float units.
// Positions each group's first token holds give the bits of a call without positions, and two views take their groups' positions in view order.
void pooled_at_positions(Backend& device) {
    const Config k{1, 8, 6, 4, 4, 1e-6f, 10000.0f};
    const size_t groups = 9, tokens = groups * k.compression, half = k.rope_dim / 2;
    std::vector<float> cosine(tokens * half), sine(tokens * half);
    for (size_t p = 0; p < tokens; ++p)
        for (size_t i = 0; i < half; ++i) {
            const float f = float(1.0 / std::pow(double(k.theta), double(2 * i) / double(k.rope_dim)));
            cosine[p * half + i] = std::cos(float(p) * f);
            sine[p * half + i] = std::sin(float(p) * f);
        }
    std::vector<float> raw(tokens * k.dim), gamma_values(k.dim);
    for (size_t i = 0; i < raw.size(); ++i) raw[i] = float(int((i * 37 + 11) % 29) - 14) / 16.0f;
    for (size_t d = 0; d < k.dim; ++d) gamma_values[d] = 0.75f + 0.0625f * float(d);
    BufferPtr gamma = upload(device, gamma_values), cos = upload(device, cosine), sin = upload(device, sine);
    const Backend::IndexArgs args{k.heads, k.dim, k.rope_dim, k.compression, k.budget, k.eps};
    const uint8_t of[] = {0, 1, 2};
    // Group g of the first view turns at (s, s + g % 3, s + g % 2), s below its first token as images before it would leave it.
    std::vector<uint32_t> pos;
    for (size_t g = 0; g < groups; ++g) {
        const uint32_t s = uint32_t(g * k.compression) / 2;
        pos.insert(pos.end(), {s, s + uint32_t(g % 3), s + uint32_t(g % 2)});
    }
    const auto pooled = [&](const backend::GroupPositions& at, size_t split) {
        Cache cache(device, tokens, k.dim, k.compression);
        std::vector<int32_t> map(cache.blocks.begin(), cache.blocks.end());
        backend::KVView all = cache.view(0, tokens, map);
        cache.write(&all, 1, raw, tokens);
        // The rows in two views of one history, the first completing `split` groups, as two entries of a pass would not, so the offsets are those of view order.
        backend::KVView views[] = {cache.view(0, split * k.compression, map), cache.view(split * k.compression, tokens - split * k.compression, map)};
        device.pool_kv(0, &views[0], 1, args, {gamma.get(), 0}, {cos.get(), 0}, {sin.get(), 0}, at);
        backend::GroupPositions rest = at;
        if (at.pos) rest.pos = at.pos + split * at.streams.count;
        device.pool_kv(0, &views[1], 1, args, {gamma.get(), 0}, {cos.get(), 0}, {sin.get(), 0}, rest);
        auto& storage = dynamic_cast<backend::BlockKVStorage&>(*cache.storage);
        std::vector<float> out;
        for (size_t g = 0; g < groups; ++g) {
            const size_t t = g * k.compression;
            const size_t at_byte = size_t(map[t / cache.block_tokens]) * storage.auxiliary_block_bytes(0) + storage.auxiliary_rows_bytes(0) +
                                   (t % cache.block_tokens) / k.compression * k.dim * sizeof(float);
            const auto row = read_values<float>(device, *storage.auxiliary_buffer(0), at_byte, k.dim);
            out.insert(out.end(), row.begin(), row.end());
        }
        return out;
    };
    const std::vector<float> got = pooled({pos.data(), {3, of}}, 4);
    double worst = 0.0;
    for (size_t g = 0; g < groups; ++g) {
        std::vector<double> mean(k.dim, 0.0);
        for (size_t t = g * k.compression; t < (g + 1) * k.compression; ++t)
            for (size_t d = 0; d < k.dim; ++d) mean[d] += raw[t * k.dim + d] / double(k.compression);
        double squares = 0.0;
        for (double v : mean) squares += v * v;
        const double inv = 1.0 / std::sqrt(squares / double(k.dim) + k.eps);
        std::vector<double> ref(k.dim);
        for (size_t d = 0; d < k.dim; ++d) ref[d] = mean[d] * inv * gamma_values[d];
        for (size_t i = 0; i < half; ++i) {
            const size_t p = pos[g * 3 + of[i]];
            const double c = cosine[p * half + i], s = sine[p * half + i], a = ref[i], b = ref[i + half];
            ref[i] = a * c - b * s;
            ref[i + half] = b * c + a * s;
        }
        for (size_t d = 0; d < k.dim; ++d)
            worst = std::max(worst, std::fabs(got[g * k.dim + d] - ref[d]) / (4.0 * std::ldexp(std::max(std::fabs(ref[d]), 1.0), -23)));
    }
    require(worst <= 1.0, "pooled groups at their positions are " + std::to_string(worst) + " of their bound from the reference");
    std::vector<uint32_t> own;
    for (size_t g = 0; g < groups; ++g) own.insert(own.end(), 3, uint32_t(g * k.compression));
    const std::vector<float> plain = pooled({}, 4), at_own = pooled({own.data(), {3, of}}, 4);
    require(plain.size() == at_own.size() && !std::memcmp(plain.data(), at_own.data(), plain.size() * sizeof(float)),
            "pooled groups at their first tokens' positions differ from a call without positions");
}

void invalid_inputs(Backend& device, const Config& c, const FixtureCase& test,
                    const std::vector<float>& qnorm, const std::vector<float>& knorm,
                    const RopeTables& rope) {
    Cache cache(device, 4 * device.kv_layout().block_tokens, c.dim, c.compression);
    std::vector<int32_t> map(cache.blocks.begin(), cache.blocks.end());
    backend::KVView write = cache.view(0, test.input_ids.size(), map);
    cache.write(&write, 1, test.full.raw_keys, test.input_ids.size());
    backend::KVView view = cache.view(0, test.input_ids.size(), map);
    const size_t row_width = c.heads * c.dim, out_width = c.budget + c.compression - 1;
    BufferPtr q = upload(device, test.full.query);
    std::vector<float> short_query(test.full.query.begin(), test.full.query.end() - 1);
    BufferPtr q_short = upload(device, short_query);
    BufferPtr q_gamma = upload(device, qnorm), k_gamma = upload(device, knorm);
    BufferPtr cos = upload(device, rope.cos), sin = upload(device, rope.sin);
    const size_t needed_rope = test.input_ids.size() * (c.rope_dim / 2);
    std::vector<float> short_cos(rope.cos.begin(), rope.cos.begin() + needed_rope - 1);
    BufferPtr cos_short = upload(device, short_cos);
    BufferPtr out = device.alloc((test.input_ids.size() * out_width + 2) * sizeof(uint32_t), backend::Memory::device);
    std::vector<uint32_t> before(test.input_ids.size() * out_width + 2, 0x1234abcdu);
    device.write(*out, 0, before.data(), before.size() * sizeof(uint32_t));
    const auto must_reject_unchanged = [&](CSlice query, Backend::IndexArgs args, CSlice norm,
                                           CSlice rc, CSlice rs, const std::string& what) {
        bool rejected = false;
        try { device.select_kv(query, 0, &view, 1, {out.get(), 1}, args, norm, rc, rs); }
        catch (const std::exception&) { rejected = true; }
        require(rejected, what + " was accepted");
        const auto after = read_values<uint32_t>(device, *out, 0, before.size());
        require(after == before, what + " wrote output before refusal");
    };
    BufferPtr qnorm_buf = device.alloc(test.full.query.size() * sizeof(float), backend::Memory::device);
    const std::vector<uint32_t> q_positions = positions(0, test.input_ids.size());
    device.norm_rope_partial({qnorm_buf.get(), 0}, {q.get(), 0}, test.input_ids.size(), row_width, c.dim,
                             c.heads, c.dim, c.rope_dim, {q_gamma.get(), 0}, c.eps,
                             {cos.get(), 0}, {sin.get(), 0}, q_positions.data());
    const Backend::IndexArgs args{c.heads, c.dim, c.rope_dim, c.compression, c.budget, c.eps};
    must_reject_unchanged({qnorm_buf.get(), 0}, Backend::IndexArgs{0, c.dim, c.rope_dim, c.compression, c.budget, c.eps},
                          {k_gamma.get(), 0}, {cos.get(), 0}, {sin.get(), 0}, "invalid index dimensions");
    must_reject_unchanged({q_short.get(), 0}, args,
                          {k_gamma.get(), 0}, {cos.get(), 0}, {sin.get(), 0}, "short query span");
    must_reject_unchanged({qnorm_buf.get(), 0}, args, {k_gamma.get(), 1},
                          {cos.get(), 0}, {sin.get(), 0}, "short key norm span");
    must_reject_unchanged({qnorm_buf.get(), 0}, args, {k_gamma.get(), 0},
                          {cos_short.get(), 0}, {sin.get(), 0}, "short RoPE span");
    BufferPtr short_out = device.alloc(test.input_ids.size() * out_width * sizeof(uint32_t), backend::Memory::device);
    std::vector<uint32_t> short_before(test.input_ids.size() * out_width, 0x76543210u);
    device.write(*short_out, 0, short_before.data(), short_before.size() * sizeof(uint32_t));
    bool rejected = false;
    try { device.select_kv({qnorm_buf.get(), 0}, 0, &view, 1, {short_out.get(), 1}, args,
                           {k_gamma.get(), 0}, {cos.get(), 0}, {sin.get(), 0}); }
    catch (const std::exception&) { rejected = true; }
    require(rejected, "short output span was accepted");
    require(read_values<uint32_t>(device, *short_out, 0, short_before.size()) == short_before,
            "short output span was modified before refusal");

    // The selector validates every view before writing even the first output row.
    Cache bad_cache(device, 4 * device.kv_layout().block_tokens, c.dim - 1, c.compression);
    std::vector<int32_t> bad_map(bad_cache.blocks.begin(), bad_cache.blocks.end());
    backend::KVView bad_view = bad_cache.view(0, 1, bad_map);
    std::vector<float> q14 = test.full.query;
    const std::vector<float> extra_query(q14.end() - row_width, q14.end());
    q14.insert(q14.end(), extra_query.begin(), extra_query.end());
    BufferPtr q14src = upload(device, q14);
    BufferPtr q14norm = device.alloc(q14.size() * sizeof(float), backend::Memory::device);
    const std::vector<uint32_t> pos14 = positions(0, test.input_ids.size() + 1);
    device.norm_rope_partial({q14norm.get(), 0}, {q14src.get(), 0}, test.input_ids.size() + 1,
                             row_width, c.dim, c.heads, c.dim, c.rope_dim,
                             {q_gamma.get(), 0}, c.eps, {cos.get(), 0}, {sin.get(), 0}, pos14.data());
    const std::vector<uint32_t> multi_before(test.input_ids.size() * out_width + out_width + 2,
                                             0x87654321u);
    BufferPtr multi_out = device.alloc(multi_before.size() * sizeof(uint32_t), backend::Memory::device);
    device.write(*multi_out, 0, multi_before.data(), multi_before.size() * sizeof(uint32_t));
    const backend::KVView two_views[2] = {view, bad_view};
    rejected = false;
    try { device.select_kv({q14norm.get(), 0}, 0, two_views, 2, {multi_out.get(), 1}, args,
                           {k_gamma.get(), 0}, {cos.get(), 0}, {sin.get(), 0}); }
    catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid second selector view was accepted");
    require(read_values<uint32_t>(device, *multi_out, 0, multi_before.size()) == multi_before,
            "valid first selector view wrote output before second-view refusal");

    // An output alias of Q is refused with the original query unchanged.
    device.sync();
    const std::vector<float> alias_before = read_values<float>(device, *qnorm_buf, 0, test.full.query.size());
    rejected = false;
    try { device.select_kv({qnorm_buf.get(), 0}, 0, &view, 1, {qnorm_buf.get(), 0}, args,
                           {k_gamma.get(), 0}, {cos.get(), 0}, {sin.get(), 0}); }
    catch (const std::exception&) { rejected = true; }
    require(rejected, "selector output aliasing Q was accepted");
    require(read_values<float>(device, *qnorm_buf, 0, alias_before.size()) == alias_before,
            "selector Q/output alias changed query values before refusal");
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string reference = "tests/data/qwen4exp_reference.json";
        std::string device_name = "cpu";
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--reference" && i + 1 < argc) reference = argv[++i];
            else if (arg == "--device" && i + 1 < argc) device_name = argv[++i];
            else throw std::runtime_error("usage: kv-selection-test [--reference FILE] [--device cpu|vulkan:N]");
        }
        std::ifstream input(reference, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open HF reference: " + reference);
        const std::string json{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        const Json root = jmini::parse(json);
        const Config c = fixture_config(root);
        const Json& state = field(root, "state_dict");
        const std::vector<float> qnorm = state_weight(state, "model.layers.1.self_attn.indexer.q_layernorm.weight", c.dim);
        const std::vector<float> knorm = state_weight(state, "model.layers.1.self_attn.indexer.k_layernorm.weight", c.dim);
        const RopeTables rope = rope_tables(c, 4096);
        backend::BackendPtr device = backend::make_backend(device_name);
        const auto cases = fixture_cases(root);
        require(!cases.empty(), "HF reference has no QSA cases");
        for (const FixtureCase& test : cases) {
            const auto full = fixture_full(*device, c, test, test.full, qnorm, knorm, rope, test.name + " full");
            const auto sliced = fixture_sliced(*device, c, test, qnorm, knorm, rope);
            const auto cached = fixture_cached(*device, c, test, qnorm, knorm, rope);
            require(full == flat_ids(test.full), test.name + " full HF membership differs");
            require(sliced == flat_ids(test.sliced), test.name + " sliced HF membership differs");
            require(cached == flat_ids(test.cached), test.name + " cached HF membership differs");
            if (test.decode) {
                const auto decoded = fixture_decode(*device, c, test, qnorm, knorm, rope);
                require(decoded == flat_ids(*test.decode), test.name + " decode HF membership differs");
            }
            invalid_inputs(*device, c, test, qnorm, knorm, rope);
        }
        synthetic_long(*device);
        pooled_rows_kept(*device);
        pooled_at_positions(*device);
        std::cout << "kv-selection: PASS (" << checks << " checks) on " << device_name << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "kv-selection: FAIL: " << error.what() << '\n';
        return 1;
    }
}
