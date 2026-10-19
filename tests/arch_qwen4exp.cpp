// Qwen4Exp text metadata parsing and refusal checks, independent of architecture registration.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "format/gguf.hpp"
#include "core/json.hpp"
#include "model/arch/qwen4exp.hpp"

namespace {

int checked = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checked;
}

void refuses(const std::string& what, const std::string& text, const std::function<void()>& run) {
    try {
        run();
    } catch (const std::exception& e) {
        require(std::string(e.what()).find(text) != std::string::npos,
                what + ": refused with \"" + e.what() + "\", not \"" + text + "\"");
        return;
    }
    throw std::runtime_error(what + ": not refused");
}

constexpr const char* prefix = "qwen4exp.";

gguf::MetaValue integer_value(uint64_t n) {
    gguf::MetaValue v;
    v.vtype = gguf::V_UINT32;
    v.u = n;
    return v;
}

void set(gguf::GGUFModel& m, const std::string& key, uint64_t n) {
    for (auto& kv : m.kv) if (kv.first == prefix + key) { kv.second = integer_value(n); return; }
    m.kv.push_back({prefix + key, integer_value(n)});
}

gguf::MetaValue& value(gguf::GGUFModel& m, const std::string& key) {
    for (auto& kv : m.kv) if (kv.first == prefix + key) return kv.second;
    gguf::MetaValue v;
    m.kv.push_back({prefix + key, v});
    return m.kv.back().second;
}

void counts(gguf::GGUFModel& m, const std::string& key, const std::vector<int>& entries) {
    gguf::MetaValue& v = value(m, key);
    v.vtype = gguf::V_ARRAY;
    v.u = gguf::V_INT32;
    v.arr.clear();
    for (int n : entries) {
        gguf::MetaValue e;
        e.vtype = gguf::V_INT32;
        e.i = n;
        v.arr.push_back(e);
    }
}

void booleans(gguf::GGUFModel& m, const std::string& key, const std::vector<bool>& entries) {
    gguf::MetaValue& v = value(m, key);
    v.vtype = gguf::V_ARRAY;
    v.u = gguf::V_BOOL;
    v.arr.clear();
    for (bool b : entries) {
        gguf::MetaValue e;
        e.vtype = gguf::V_BOOL;
        e.b = b;
        v.arr.push_back(e);
    }
}

void uint64s(gguf::GGUFModel& m, const std::string& key, const std::vector<uint64_t>& entries) {
    gguf::MetaValue& v = value(m, key);
    v.vtype = gguf::V_ARRAY;
    v.u = gguf::V_UINT64;
    v.arr.clear();
    for (uint64_t n : entries) {
        gguf::MetaValue e;
        e.vtype = gguf::V_UINT64;
        e.u = n;
        v.arr.push_back(e);
    }
}

void drop(gguf::GGUFModel& m, const std::string& key) {
    for (size_t i = 0; i < m.kv.size(); ++i)
        if (m.kv[i].first == prefix + key) { m.kv.erase(m.kv.begin() + static_cast<std::ptrdiff_t>(i)); return; }
}

gguf::GGUFModel tiny() {
    gguf::GGUFModel m;
    gguf::MetaValue arch;
    arch.vtype = gguf::V_STRING;
    arch.s = "qwen4exp";
    m.kv.push_back({"general.architecture", arch});
    for (const auto& kv : std::vector<std::pair<std::string, uint64_t>>{
             {"block_count", 4}, {"embedding_length", 16}, {"expert_count", 4},
             {"expert_used_count", 2}, {"expert_feed_forward_length", 8},
             {"expert_shared_feed_forward_length", 8}, {"attention.head_count", 4},
             {"attention.head_count_kv", 2}, {"attention.key_length", 8},
             {"attention.value_length", 8}, {"rope.dimension_count", 4}, {"context_length", 128},
             {"ssm.conv_kernel", 4}, {"ssm.state_size", 4}, {"ssm.group_count", 2},
             {"ssm.time_step_rank", 4}, {"ssm.inner_size", 16},
             {"hyper_connection.count", 4}, {"hyper_connection.low_rank", 4},
             {"attention.indexer.head_count", 2}, {"attention.indexer.key_length", 4},
             {"attention.indexer.top_k", 8}, {"ple.ngram_size", 3},
             {"ple.heads_per_ngram", 2}, {"ple.conv_kernel", 4},
             {"embedding_length_per_layer_input", 4}, {"ple.eos_token_id", 3}})
        set(m, kv.first, kv.second);
    booleans(m, "attention.recurrent_layers", {true, false, true, false});
    counts(m, "attention.compress_ratios", {0, 4, 0, 4});
    counts(m, "rope.dimension_sections", {1, 1, 0, 0});
    counts(m, "ple.layers", {2});
    uint64s(m, "ple.layer_multipliers", {1, 3, 5});
    uint64s(m, "ple.head_offsets", {0, 10, 20, 30});
    uint64s(m, "ple.head_vocab_sizes", {10, 10, 10, 10});
    return m;
}

infer::qwen4exp::Config read(const gguf::GGUFModel& m) {
    return infer::qwen4exp::read_config(m, "qwen4exp.");
}

void rotary_tables() {
    auto c = read(tiny());
    c.head_dim = 128;
    c.indexer_head_dim = 128;
    c.rope_dim = 64;
    c.rope_sections = {16, 16, 0, 0};
    c.context_length = 16384;
    c.rope_theta = 10000000.0f;
    infer::qwen4exp::Qwen4Exp arch(c, 32);
    std::vector<std::vector<float>> tables(3);
    tables[0].resize(size_t(c.context_length) * 32);
    tables[1].resize(tables[0].size());
    tables[2].resize(size_t(c.hc_count) * size_t(c.n_embd));
    arch.fill_tables(tables);
    struct Cell { size_t position, column; float cosine, sine; };
    // Original HF rotary outputs, including all 32 frequencies at the 16K boundary; provenance is in docs/QWEN38-FLASH.md.
    const Cell cells[] = {
        {0, 0, 1.0f, 0.0f},
        {1, 2, 0.934061587f, 0.357111961f},
        {128, 5, -0.629410982f, -0.777072549f},
        {2047, 2, 0.982458949f, -0.186478898f},
        {8191, 2, 0.941024303f, 0.338338971f},
        {8344, 2, 0.945493221f, -0.325641751f},
        {13197, 2, 1.0f, -5.60674289e-06f},
        {16350, 2, -0.000329821341f, 0.99999994f},
        {16383, 0, -0.918830931f, 0.394651443f},
        {16383, 1, -0.515634298f, -0.856808782f},
        {16383, 2, 0.492760718f, 0.870164871f},
        {16383, 3, -0.7770015f, 0.62949878f},
        {16383, 4, -0.26619035f, -0.963920474f},
        {16383, 5, 0.73693794f, 0.675960362f},
        {16383, 6, 0.986312211f, -0.164888397f},
        {16383, 7, -0.12710245f, -0.991889596f},
        {16383, 8, -0.672950029f, 0.73968792f},
        {16383, 9, 0.992344022f, 0.123504281f},
        {16383, 10, 0.910646081f, -0.413187325f},
        {16383, 11, 0.112410389f, 0.99366188f},
        {16383, 12, 0.407479435f, 0.913214386f},
        {16383, 13, -0.0847959891f, -0.99639833f},
        {16383, 14, -0.0499060526f, 0.998753905f},
        {16383, 15, -0.658803463f, 0.752315104f},
        {16383, 16, 0.451433271f, -0.892304897f},
        {16383, 17, -0.999940813f, 0.0108782612f},
        {16383, 18, -0.315594286f, 0.948894203f},
        {16383, 19, 0.41463393f, 0.909988284f},
        {16383, 20, 0.770694852f, 0.637204409f},
        {16383, 21, 0.914110601f, 0.405464977f},
        {16383, 22, 0.968344271f, 0.249618381f},
        {16383, 23, 0.988401115f, 0.151865736f},
        {16383, 24, 0.995759189f, 0.0919981077f},
        {16383, 25, 0.998450637f, 0.0556440912f},
        {16383, 26, 0.999434114f, 0.0336365514f},
        {16383, 27, 0.999793351f, 0.0203288831f},
        {16383, 28, 0.999924541f, 0.0122852083f},
        {16383, 29, 0.999972463f, 0.0074240258f},
        {16383, 30, 0.999989927f, 0.00448633824f},
        {16383, 31, 0.999996305f, 0.00271108351f},
    };
    const float tolerance = 2.0f * std::numeric_limits<float>::epsilon();
    for (const Cell& cell : cells) {
        const size_t at = cell.position * 32 + cell.column;
        require(std::abs(tables[0][at] - cell.cosine) <= tolerance &&
                std::abs(tables[1][at] - cell.sine) <= tolerance,
                "rotary table differs from HF at position " + std::to_string(cell.position) +
                " column " + std::to_string(cell.column));
    }
}

void configuration() {
    const auto c = read(tiny());
    require(c.n_layer == 4 && c.n_embd == 16 && c.n_expert == 4 && c.n_expert_used == 2,
            "the text and expert dimensions");
    require(c.layers.size() == 4 && c.layers[0] == infer::qwen4exp::linear_attention &&
                c.layers[1] == infer::qwen4exp::qwen_sparse_attention &&
                c.compress_ratios[1] == 4 && c.compress_ratios[3] == 4,
            "normalized attention layer kinds and ratios");
    require(c.hc_count == 4 && c.hc_low_rank == 4 && c.indexer_n_heads == 2 &&
                c.indexer_kv_heads == 1 && c.indexer_head_dim == 4 && c.indexer_budget == 8,
            "hyper-connection and QSA geometry");
    require(c.has_ple && c.ple_layer == 2 && c.ple_ngram_size == 3 &&
                c.ple_layer_multipliers == std::vector<uint64_t>({1, 3, 5}) &&
                c.ple_head_offsets.size() == 4 && c.ple_head_vocab_sizes[3] == 10,
            "exact-width PLE constants and layer");

    refuses("direct query/KV head mismatch", "head count must be divisible by KV head count", [&] {
        auto bad = c; bad.n_head_kv = 3; infer::qwen4exp::check_config(bad);
    });
    refuses("direct PLE head range", "PLE head range does not fit the int32 row index", [&] {
        auto bad = c; bad.ple_head_offsets[0] = uint64_t(std::numeric_limits<int32_t>::max());
        infer::qwen4exp::check_config(bad);
    });
    {
        auto edge = c;
        edge.ple_head_offsets[0] = uint64_t(std::numeric_limits<int32_t>::max());
        edge.ple_head_vocab_sizes[0] = 1;
        infer::qwen4exp::check_config(edge);
        require(true, "PLE row at INT32_MAX is representable");
    }

    // A single MTP block is outside the trunk; its kind must be non-recurrent and its QSA ratio is retained in metadata.
    auto mtp = tiny();
    set(mtp, "block_count", 5);
    set(mtp, "nextn_predict_layers", 1);
    booleans(mtp, "attention.recurrent_layers", {true, false, true, false, false});
    counts(mtp, "attention.compress_ratios", {0, 4, 0, 4, 4});
    const auto with_mtp = read(mtp);
    require(with_mtp.n_nextn == 1 && with_mtp.n_layer == 4 && with_mtp.layers.size() == 4,
            "MTP is excluded from decoder layers");
    require(with_mtp.nextn_compress_ratio == 4, "MTP retains its own compression ratio");
    counts(mtp, "attention.compress_ratios", {0, 4, 0, 4, 0});
    require(read(mtp).nextn_compress_ratio == 0, "released dense MTP compression tail");
    for (int ratio : {-1, 1}) {
        refuses("direct MTP ratio", "MTP compression ratio must be zero or greater than one", [&] {
            auto bad = with_mtp; bad.nextn_compress_ratio = ratio; infer::qwen4exp::check_config(bad);
        });
    }
    refuses("MTP ratio without MTP", "MTP compression metadata without an MTP block", [&] {
        auto bad = c; bad.nextn_compress_ratio = 4; infer::qwen4exp::check_config(bad);
    });
    refuses("MTP budget mismatch", "MTP compression ratio must divide the indexer budget", [&] {
        auto bad = with_mtp; bad.nextn_compress_ratio = 3; infer::qwen4exp::check_config(bad);
    });

    refuses("recurrent array length", "unsupported metadata qwen4exp.attention.recurrent_layers", [&] {
        auto m = tiny(); booleans(m, "attention.recurrent_layers", {true, false}); (void)read(m);
    });
    refuses("recurrent MTP block", "unsupported metadata qwen4exp.attention.recurrent_layers", [&] {
        auto m = mtp; booleans(m, "attention.recurrent_layers", {true, false, true, false, true}); (void)read(m);
    });
    refuses("linear layer with QSA ratio", "compression ratio on a linear-attention layer", [&] {
        auto m = tiny(); counts(m, "attention.compress_ratios", {4, 4, 0, 4}); (void)read(m);
    });
    refuses("different QSA ratios", "QSA layers must share one compression ratio", [&] {
        auto m = tiny(); counts(m, "attention.compress_ratios", {0, 4, 0, 8}); (void)read(m);
    });
    refuses("partial indexer metadata", "QSA indexer metadata is incomplete", [&] {
        auto m = tiny(); drop(m, "attention.indexer.top_k"); (void)read(m);
    });
    refuses("indexer key smaller than rotary width", "indexer head width is smaller than the rotary width", [&] {
        auto m = tiny(); set(m, "attention.indexer.key_length", 2); (void)read(m);
    });
    refuses("invalid section sum", "dimension_sections does not cover the rotary width", [&] {
        auto m = tiny(); counts(m, "rope.dimension_sections", {1, 0, 0, 0}); (void)read(m);
    });
    refuses("missing MRoPE sections", "missing metadata qwen4exp.rope.dimension_sections", [&] {
        auto m = tiny(); drop(m, "rope.dimension_sections"); (void)read(m);
    });
    refuses("HC count one", "hyper_connection.count must be greater than one", [&] {
        auto m = tiny(); set(m, "hyper_connection.count", 1); (void)read(m);
    });
    refuses("PLE on QSA", "PLE layer must be an in-range linear-attention layer", [&] {
        auto m = tiny(); counts(m, "ple.layers", {1}); (void)read(m);
    });
    refuses("PLE multiplier length", "invalid uint64 array length", [&] {
        auto m = tiny(); uint64s(m, "ple.layer_multipliers", {1, 3}); (void)read(m);
    });
    refuses("PLE constants must remain uint64", "invalid uint64 array type", [&] {
        auto m = tiny(); counts(m, "ple.layer_multipliers", {1, 3, 5}); (void)read(m);
    });
    refuses("PLE row extent overflow", "PLE head range does not fit the int32 row index", [&] {
        auto m = tiny();
        uint64s(m, "ple.head_offsets", {0, 10, 20, uint64_t(std::numeric_limits<int32_t>::max())});
        uint64s(m, "ple.head_vocab_sizes", {10, 10, 10, 2});
        (void)read(m);
    });
    refuses("scaled rope", "scaled RoPE is unsupported", [&] {
        auto m = tiny(); gguf::MetaValue v; v.vtype = gguf::V_FLOAT32;
        float two = 2.0f; std::memcpy(&v.fb, &two, sizeof(two));
        value(m, "rope.scaling.factor") = v; (void)read(m);
    });
}

void ple_selection() {
    using infer::qwen4exp::ple_rows;
    const auto c = read(tiny());
    const uint32_t ids[] = {1, 2, 3, 4};
    const int32_t expected[] = {8, 18, 27, 37, 1, 11, 24, 34,
                                5, 15, 20, 30, 3, 13, 22, 32};
    int32_t rows[16] = {};
    require(ple_rows(c, 10, nullptr, 0, ids, 4, rows, 16) == 16,
            "PLE row count for a token chunk");
    require(std::equal(std::begin(expected), std::end(expected), rows),
            "PLE n-gram hashes, EOS padding, and head order");

    // Splitting after two tokens with their oldest-first tail preserves lookup rows.
    const uint32_t tail[] = {1, 2};
    int32_t sliced[8] = {};
    require(ple_rows(c, 10, tail, 2, ids + 2, 2, sliced, 8) == 8 &&
                std::equal(std::begin(expected) + 8, std::end(expected), sliced),
            "PLE rows are invariant across a cached token slice");

    // EOS is hashed with its own preceding context; it resets history for the next token.
    const uint32_t before_eos[] = {2};
    int32_t eos_rows[8] = {};
    require(ple_rows(c, 10, before_eos, 1, ids + 2, 2, eos_rows, 8) == 8 &&
                std::equal(std::begin(expected) + 8, std::end(expected), eos_rows),
            "PLE EOS token keeps its own context and resets the following context");

    int32_t untouched[16];
    std::fill(std::begin(untouched), std::end(untouched), 71);
    refuses("PLE output capacity", "output buffer is too small", [&] {
        (void)ple_rows(c, 10, nullptr, 0, ids, 4, untouched, 15);
    });
    require(std::all_of(std::begin(untouched), std::end(untouched), [](int32_t x) { return x == 71; }),
            "small PLE output buffer is rejected before writing");
    const uint32_t invalid[] = {1, 10};
    refuses("PLE invalid current id", "token id is outside the vocabulary", [&] {
        (void)ple_rows(c, 10, nullptr, 0, invalid, 2, untouched, 16);
    });
    require(std::all_of(std::begin(untouched), std::end(untouched), [](int32_t x) { return x == 71; }),
            "invalid current token is rejected before writing");
    const uint32_t invalid_tail[] = {10};
    refuses("PLE invalid history id", "token id is outside the vocabulary", [&] {
        (void)ple_rows(c, 10, invalid_tail, 1, ids, 4, untouched, 16);
    });
    require(std::all_of(std::begin(untouched), std::end(untouched), [](int32_t x) { return x == 71; }),
            "invalid history token is rejected before writing");
    refuses("PLE hash product overflow", "hash product exceeds int64 range", [&] {
        auto bad = c;
        bad.ple_layer_multipliers[0] = uint64_t(std::numeric_limits<int64_t>::max());
        (void)ple_rows(bad, 10, nullptr, 0, ids, 4, untouched, 16);
    });
    require(std::all_of(std::begin(untouched), std::end(untouched), [](int32_t x) { return x == 71; }),
            "overflowing hash product is rejected before writing");
    refuses("PLE EOS outside vocabulary", "token id is outside the vocabulary", [&] {
        auto bad = c; bad.ple_eos_token_id = 10;
        (void)ple_rows(bad, 10, nullptr, 0, ids, 4, untouched, 16);
    });

    auto large = c;
    large.ple_eos_token_id = 16777218;
    const uint32_t wide_ids[] = {16777216, 16777217};
    const int32_t wide_expected[] = {8, 18, 24, 34, 3, 13, 23, 33};
    int32_t wide_rows[8] = {};
    require(ple_rows(large, 16777219, nullptr, 0, wide_ids, 2, wide_rows, 8) == 8 &&
                std::equal(std::begin(wide_expected), std::end(wide_expected), wide_rows),
            "PLE hashes preserve uint32 token ids above the exact float-integer range");
}

uint32_t json_token(const jmini::Value& value) {
    if (!value.isNumber() || value.asNumber() < 0 || value.asNumber() > std::numeric_limits<uint32_t>::max() ||
        std::floor(value.asNumber()) != value.asNumber())
        throw std::runtime_error("reference contains an invalid token id");
    return uint32_t(value.asNumber());
}

void ple_reference(const infer::qwen4exp::Config& c, uint32_t vocab_size, const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open Qwen4Exp reference JSON");
    const std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    const jmini::Value root = jmini::parse(source);
    const jmini::Value* cases = root.get("cases");
    if (!cases || !cases->isArray()) throw std::runtime_error("reference has no cases array");

    size_t verified = 0;
    for (const jmini::Value& item : cases->asArray()) {
        const jmini::Value* input = item.get("input_ids");
        const jmini::Value* expected = item.get("ngram_ids");
        const jmini::Value* token = item.get("decode_token");
        const jmini::Value* decode_expected = item.get("decode_ngram_ids");
        if (!input || !input->isArray() || !expected || !expected->isArray() ||
            !token || !decode_expected || !decode_expected->isArray())
            throw std::runtime_error("reference case is missing PLE inputs or rows");
        std::vector<uint32_t> ids;
        for (const jmini::Value& value : input->asArray()) ids.push_back(json_token(value));
        std::vector<int32_t> rows(ids.size() * c.ple_head_offsets.size());
        ple_rows(c, vocab_size, nullptr, 0, ids.data(), ids.size(), rows.data(), rows.size());
        if (expected->asArray().size() != ids.size())
            throw std::runtime_error("reference PLE prompt row count mismatch");
        for (size_t i = 0; i < ids.size(); ++i) {
            const jmini::Value& expected_row = expected->asArray()[i];
            if (!expected_row.isArray() || expected_row.asArray().size() != c.ple_head_offsets.size())
                throw std::runtime_error("reference PLE prompt head count mismatch");
            for (size_t h = 0; h < c.ple_head_offsets.size(); ++h)
                if (rows[i * c.ple_head_offsets.size() + h] != int32_t(json_token(expected_row.asArray()[h])))
                    throw std::runtime_error("PLE prompt rows differ from the HF reference");
        }
        ++verified;

        for (size_t slice_size = 1; slice_size <= 4; ++slice_size) {
            std::vector<int32_t> sliced(rows.size());
            std::vector<uint32_t> history;
            const size_t max_history = size_t(c.ple_ngram_size - 1);
            for (size_t start = 0; start < ids.size();) {
                const size_t count = std::min(slice_size, ids.size() - start);
                const size_t offset = start * c.ple_head_offsets.size();
                ple_rows(c, vocab_size, history.data(), history.size(), ids.data() + start, count,
                         sliced.data() + offset, sliced.size() - offset);
                history.insert(history.end(), ids.begin() + start, ids.begin() + start + count);
                if (history.size() > max_history)
                    history.erase(history.begin(), history.end() - static_cast<std::ptrdiff_t>(max_history));
                start += count;
            }
            if (sliced != rows)
                throw std::runtime_error("sliced PLE prompt rows differ from full prompt rows");
            ++verified;

            if (token->t != jmini::Value::T::Null) {
                const uint32_t decode_id = json_token(*token);
                std::vector<int32_t> decoded(c.ple_head_offsets.size());
                ple_rows(c, vocab_size, history.data(), history.size(), &decode_id, 1,
                         decoded.data(), decoded.size());
                if (decode_expected->asArray().size() != 1 || !decode_expected->asArray()[0].isArray() ||
                    decode_expected->asArray()[0].asArray().size() != c.ple_head_offsets.size())
                    throw std::runtime_error("reference PLE decode row shape mismatch");
                for (size_t h = 0; h < c.ple_head_offsets.size(); ++h)
                    if (decoded[h] != int32_t(json_token(decode_expected->asArray()[0].asArray()[h])))
                        throw std::runtime_error("PLE decode rows differ after a sliced prompt");
                ++verified;
            }
        }

        if (token->t != jmini::Value::T::Null) {
            if (ids.empty()) throw std::runtime_error("reference decode has no preceding prompt");
            const uint32_t decode_id = json_token(*token);
            const size_t tail_count = std::min(ids.size(), size_t(c.ple_ngram_size - 1));
            const uint32_t* tail = ids.data() + (ids.size() - tail_count);
            std::vector<int32_t> decoded(c.ple_head_offsets.size());
            ple_rows(c, vocab_size, tail, tail_count, &decode_id, 1, decoded.data(), decoded.size());
            if (decode_expected->asArray().size() != 1 || !decode_expected->asArray()[0].isArray() ||
                decode_expected->asArray()[0].asArray().size() != c.ple_head_offsets.size())
                throw std::runtime_error("reference PLE decode row shape mismatch");
            for (size_t h = 0; h < c.ple_head_offsets.size(); ++h)
                if (decoded[h] != int32_t(json_token(decode_expected->asArray()[0].asArray()[h])))
                    throw std::runtime_error("PLE decode rows differ from the HF reference");
            ++verified;
        } else if (!decode_expected->asArray().empty()) {
            throw std::runtime_error("terminal EOS reference unexpectedly has decode rows");
        }
    }
    if (!verified) throw std::runtime_error("reference contains no PLE checks");
    require(true, "HF reference PLE prompt and decode rows");
}

uint32_t gguf_vocab_size(const gguf::GGUFModel& model) {
    for (const gguf::TensorInfo& tensor : model.tensors) {
        if (tensor.name == "token_embd.weight" && tensor.ne.size() == 2 && tensor.ne[1] &&
            tensor.ne[1] <= std::numeric_limits<uint32_t>::max())
            return uint32_t(tensor.ne[1]);
    }
    const gguf::MetaValue* tokens = model.find("tokenizer.ggml.tokens");
    if (tokens && tokens->vtype == gguf::V_ARRAY && tokens->u == gguf::V_STRING &&
        !tokens->arr.empty() && tokens->arr.size() <= std::numeric_limits<uint32_t>::max())
        return uint32_t(tokens->arr.size());
    throw std::runtime_error("GGUF has no usable vocabulary size");
}

infer::qwen4exp::Config reference_config() {
    // These are the exact uint64 hash constants and table ranges in run10's HF state dict.
    auto c = read(tiny());
    c.ple_layer = 0;
    c.ple_ngram_size = 3;
    c.ple_heads_per_ngram = 2;
    c.ple_eos_token_id = 31;
    c.ple_embedding_length = 2;
    c.ple_embedding_dim = 8;
    c.ple_layer_multipliers = {57442287457681827ull, 29154299761009867ull, 208413376719075541ull};
    c.ple_head_offsets = {0, 17, 36, 59};
    c.ple_head_vocab_sizes = {17, 19, 23, 29};
    return c;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--gguf") {
            const gguf::GGUFModel model = gguf::read_gguf(argv[2]);
            const infer::qwen4exp::Config c = read(model);
            require(c.n_layer > 0 && !c.layers.empty(), "the emitted GGUF config");
        } else if (argc == 5 && std::string(argv[1]) == "--gguf" && std::string(argv[3]) == "--reference") {
            const gguf::GGUFModel model = gguf::read_gguf(argv[2]);
            const infer::qwen4exp::Config c = read(model);
            require(c.n_layer > 0 && !c.layers.empty(), "the emitted GGUF config");
            ple_reference(c, gguf_vocab_size(model), argv[4]);
        } else if (argc == 3 && std::string(argv[1]) == "--reference") {
            configuration();
            rotary_tables();
            ple_selection();
            const auto c = reference_config();
            ple_reference(c, 32, argv[2]);
        } else if (argc == 1) {
            configuration();
            rotary_tables();
            ple_selection();
        } else {
            std::cerr << "usage: llmx-arch-qwen4exp-test [--gguf FILE [--reference FILE] | --reference FILE]" << std::endl;
            return 2;
        }
        std::cout << "arch-qwen4exp: " << checked << " checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "arch-qwen4exp: FAIL: " << e.what() << "\n";
        return 1;
    }
}
