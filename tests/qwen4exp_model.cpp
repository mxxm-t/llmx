// Full Qwen4Exp GGUF through the native Model runtime, checked against the frozen HF logits.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backends/devices.hpp"
#include "core/json.hpp"
#include "core/list.hpp"
#include "format/file_reader.hpp"
#include "format/gguf.hpp"
#include "inference/load.hpp"
#include "model/arch/qwen4exp.hpp"
#include "model/runtime.hpp"
#include "model/weight_rows.hpp"
#include "tiny_qwen.hpp"

namespace {
using Json = jmini::Value;

const Json& field(const Json& value, const std::string& name) {
    const Json* result = value.get(name);
    if (!result) throw std::runtime_error("reference: missing field " + name);
    return *result;
}

std::vector<uint32_t> ids(const Json& value) {
    std::vector<uint32_t> result;
    for (const Json& item : value.asArray()) {
        const double x = item.asNumber();
        if (!std::isfinite(x) || x < 0 || x > UINT32_MAX || std::floor(x) != x)
            throw std::runtime_error("reference: invalid token id");
        result.push_back(uint32_t(x));
    }
    return result;
}

std::vector<float> floats(const Json& value) {
    std::vector<float> result;
    auto append = [&](auto&& self, const Json& item) -> void {
        if (item.isNumber()) result.push_back(float(item.asNumber()));
        else if (item.isArray()) for (const Json& child : item.asArray()) self(self, child);
        else throw std::runtime_error("reference: expected numeric array");
    };
    append(append, value);
    return result;
}

void compare(const std::vector<float>& actual, const std::vector<float>& expected,
             const std::string& label, float bound = 2e-5f) {
    if (actual.size() != expected.size()) throw std::runtime_error(label + ": logit count mismatch");
    float worst = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]))
            throw std::runtime_error(label + ": non-finite logit");
        worst = std::max(worst, std::fabs(actual[i] - expected[i]));
    }
    if (worst > bound) throw std::runtime_error(label + ": max abs error " + std::to_string(worst));
    std::cout << label << " max_abs=" << worst << '\n';
}

std::vector<float> slice_rows(const std::vector<float>& rows, size_t width, size_t first, size_t count) {
    return {rows.begin() + first * width, rows.begin() + (first + count) * width};
}

std::vector<float> logits(infer::ExecContext& ctx, size_t first, size_t count, size_t width) {
    std::vector<float> out;
    out.reserve(count * width);
    for (size_t i = 0; i < count; ++i) {
        const float* row = ctx.logits(first + i);
        out.insert(out.end(), row, row + width);
    }
    return out;
}

infer::ModelOptions model_options(size_t state_slots = 1, size_t checkpoints = 0, size_t marks = 0,
                                 size_t mark_rows = 0) {
    infer::ModelOptions result;
    result.dtype = backend::Dtype::f32;
    result.kv_k = result.kv_v = backend::KVType::f32;
    result.kv_tokens = 1024;
    result.state_slots = state_slots;
    result.checkpoint_slots = checkpoints;
    result.mark_slots = marks;
    result.mark_rows = mark_rows;
    return result;
}

backend::BackendPtr test_backend(const std::string& device) {
    backend::BackendPtr result = backend::make_backend(device);
    if (result->is_cpu()) result->set_threads(1);
    return result;
}

std::vector<std::string> tensor_group_device_names(const std::string& devices) {
    if (devices == "cpu") return {"cpu", "cpu"};
    const auto names = core::comma_list(devices);
    if (names.size() != 2 || names[0].empty() || names[1].empty())
        throw std::runtime_error("--tensor-width 2 requires exactly two --device names, such as vulkan:0,vulkan:1");
    return names;
}

std::vector<backend::BackendPtr> tensor_group_backends(const std::string& devices) {
    const auto names = tensor_group_device_names(devices);
    return {test_backend(names[0]), test_backend(names[1])};
}

infer::Placement tensor_placement() {
    infer::Placement result;
    result.width = 2;
    return result;
}

struct FailAfterDelta final : backend::CpuBackend {
    FailAfterDelta() { set_threads(1); }
    bool fail_after_delta = false;
    bool injected_after_write = false;
    size_t delta_writes = 0;

    void gated_delta_rule(backend::Slice out, backend::CSlice qkv, backend::CSlice alpha, backend::CSlice b,
                          backend::CSlice a, backend::CSlice dt_bias, size_t layer,
                          const backend::StateView* views, size_t n_views) override {
        backend::CpuBackend::gated_delta_rule(out, qkv, alpha, b, a, dt_bias, layer, views, n_views);
        ++delta_writes;
        if (fail_after_delta) {
            fail_after_delta = false;
            injected_after_write = true;
            throw std::runtime_error("injected after gated_delta_rule");
        }
    }
};

std::vector<float> score(infer::Model& model, const std::vector<uint32_t>& prompt);

void history_checks(const infer::ModelWeights& weights, backend::BackendPtr backend,
                    const std::vector<uint32_t>& prompt, const std::vector<float>& expected, size_t vocab,
                    const std::string& name) {
    const size_t split = prompt.size() / 2;
    {
        infer::Model model(weights, backend, model_options(1, 0, 1, prompt.size()));
        infer::Sequence seq = model.make_sequence();
        infer::ExecContext ctx;
        infer::BatchEntry head{&seq, prompt.data(), split, false};
        head.extent = prompt.size();
        model.forward(ctx, &head, 1);
        if (!model.mark(seq)) throw std::runtime_error(name + ": mark refused");
        infer::BatchEntry tail{&seq, prompt.data() + split, prompt.size() - split, true};
        tail.every_logits = true;
        tail.extent = prompt.size();
        model.forward(ctx, &tail, 1);
        const size_t target = std::min(prompt.size(), split + 2);
        if (model.retract(seq, target) != target)
            throw std::runtime_error(name + ": mark/retract did not replay to requested prefix");
        if (target < prompt.size()) {
            infer::BatchEntry remainder{&seq, prompt.data() + target, prompt.size() - target, true};
            remainder.every_logits = true;
            remainder.extent = prompt.size();
            model.forward(ctx, &remainder, 1);
            compare(logits(ctx, 0, prompt.size() - target, vocab),
                    slice_rows(expected, vocab, target, prompt.size() - target), name + " marked continuation");
        }
    }
}

void edited_suffix_retract_checks(const infer::ModelWeights& weights, backend::BackendPtr backend,
                                  const std::vector<uint32_t>& prompt, size_t vocab, const std::string& name) {
    constexpr size_t prefix = 6, retract_to = 8, total = 13;
    if (prompt.size() != total || vocab < 2)
        throw std::runtime_error(name + ": edited-suffix check requires the 13-token fixture");
    std::vector<uint32_t> edited(prompt.begin(), prompt.begin() + total);
    for (size_t i = retract_to; i < total; ++i)
        edited[i] = (edited[i] + 1) % uint32_t(vocab);
    const uint32_t decode_id = prompt.back();
    auto fresh = [&](const std::vector<uint32_t>& ids) {
        infer::Model model(weights, backend, model_options(1, 0, 1, total));
        infer::Sequence seq = model.make_sequence();
        infer::ExecContext ctx;
        infer::BatchEntry head{&seq, ids.data(), prefix, false};
        head.extent = total;
        model.forward(ctx, &head, 1);
        infer::BatchEntry accepted{&seq, ids.data() + prefix, retract_to - prefix, true};
        accepted.extent = total;
        model.forward(ctx, &accepted, 1);
        infer::BatchEntry suffix{&seq, ids.data() + retract_to, total - retract_to, true};
        suffix.every_logits = true;
        suffix.extent = total;
        model.forward(ctx, &suffix, 1);
        auto suffix_logits = logits(ctx, 0, total - retract_to, vocab);
        infer::BatchEntry decode{&seq, &decode_id, 1, true};
        decode.extent = 1;
        model.forward(ctx, &decode, 1);
        return std::pair<std::vector<float>, std::vector<float>>{
            std::move(suffix_logits), logits(ctx, 0, 1, vocab)};
    };

    const auto unchanged = fresh(std::vector<uint32_t>(prompt.begin(), prompt.begin() + total));
    const auto control = fresh(edited);
    infer::Model model(weights, backend, model_options(1, 0, 1, total));
    infer::Sequence seq = model.make_sequence();
    infer::ExecContext ctx;
    infer::BatchEntry head{&seq, prompt.data(), prefix, false};
    head.extent = total;
    model.forward(ctx, &head, 1);
    if (!model.mark(seq)) throw std::runtime_error(name + ": edited-suffix mark refused");
    infer::BatchEntry original_tail{&seq, prompt.data() + prefix, total - prefix, true};
    original_tail.extent = total;
    model.forward(ctx, &original_tail, 1);
    if (model.retract(seq, retract_to) != retract_to || seq.length() != retract_to)
        throw std::runtime_error(name + ": edited-suffix retract did not restore length 8");
    infer::BatchEntry replacement{&seq, edited.data() + retract_to, total - retract_to, true};
    replacement.every_logits = true;
    replacement.extent = total;
    model.forward(ctx, &replacement, 1);
    const auto actual_logits = logits(ctx, 0, total - retract_to, vocab);
    if (actual_logits.size() != control.first.size() ||
        std::memcmp(actual_logits.data(), control.first.data(), actual_logits.size() * sizeof(float)) != 0)
        throw std::runtime_error(name + ": edited suffix after retract differs from fresh control");
    if (actual_logits.size() == unchanged.first.size() &&
        std::memcmp(actual_logits.data(), unchanged.first.data(), actual_logits.size() * sizeof(float)) == 0)
        throw std::runtime_error(name + ": edited token did not change suffix logits");
    infer::BatchEntry decode{&seq, &decode_id, 1, true};
    decode.extent = 1;
    model.forward(ctx, &decode, 1);
    const auto actual_decode = logits(ctx, 0, 1, vocab);
    if (actual_decode.size() != control.second.size() ||
        std::memcmp(actual_decode.data(), control.second.data(), actual_decode.size() * sizeof(float)) != 0)
        throw std::runtime_error(name + ": edited suffix after retract changed next decode bytes");
    std::cout << name << " edited suffix after retract: fresh-control suffix and decode byte-identical\n";
}

void failed_marked_delta_rollback_checks(const infer::ModelWeights& weights,
                                         const std::vector<uint32_t>& prompt,
                                         const std::vector<float>& expected, size_t vocab,
                                         const std::string& name) {
    const size_t split = prompt.size() / 2;
    auto backend = std::make_shared<FailAfterDelta>();
    infer::Model model(weights, backend, model_options(1, 0, 1, prompt.size()));
    infer::Sequence seq = model.make_sequence();
    infer::ExecContext ctx;
    infer::BatchEntry head{&seq, prompt.data(), split, false};
    head.extent = prompt.size();
    model.forward(ctx, &head, 1);
    if (!model.mark(seq)) throw std::runtime_error(name + ": mark refused before failure check");

    infer::BatchEntry tail{&seq, prompt.data() + split, prompt.size() - split, true};
    tail.every_logits = true;
    tail.extent = prompt.size();
    const size_t writes_before = backend->delta_writes;
    backend->fail_after_delta = true;
    bool failed = false;
    try {
        model.forward(ctx, &tail, 1);
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()) != "injected after gated_delta_rule") throw;
        failed = true;
    }
    if (!failed || !backend->injected_after_write || backend->delta_writes != writes_before + 1)
        throw std::runtime_error(name + ": injected failure did not follow a GDN state write");
    if (seq.length() != split)
        throw std::runtime_error(name + ": failed marked pass did not restore its prefix history");

    model.forward(ctx, &tail, 1);
    compare(logits(ctx, 0, prompt.size() - split, vocab),
            slice_rows(expected, vocab, split, prompt.size() - split), name + " failed-pass retry HF logits");
}

void aborted_stage_rollback_checks(const infer::ModelWeights& weights,
                                   std::vector<backend::BackendPtr> backends, infer::Placement placement,
                                   const std::vector<uint32_t>& prompt, const std::vector<float>& expected,
                                   size_t vocab, const std::string& name) {
    const size_t split = prompt.size() / 2, tail_rows = prompt.size() - split;
    if (!split || !tail_rows || vocab < 2) throw std::runtime_error(name + ": abort fixture is too small");
    const auto options = model_options(1, 0, 1, prompt.size());
    infer::Model clean(weights, backends, placement, options);
    infer::Sequence reference = clean.make_sequence();
    infer::ExecContext reference_ctx;
    infer::BatchEntry head{&reference, prompt.data(), split, false};
    head.extent = prompt.size();
    clean.forward(reference_ctx, &head, 1);
    infer::BatchEntry tail{&reference, prompt.data() + split, tail_rows, true};
    tail.every_logits = true;
    tail.extent = prompt.size();
    clean.forward(reference_ctx, &tail, 1);
    const auto clean_tail = logits(reference_ctx, 0, tail_rows, vocab);
    compare(clean_tail, slice_rows(expected, vocab, split, tail_rows), name + " abort control HF logits");
    const uint32_t next = prompt.back();
    infer::BatchEntry decode{&reference, &next, 1, true};
    decode.extent = 1;
    clean.forward(reference_ctx, &decode, 1);
    const auto clean_decode = logits(reference_ctx, 0, 1, vocab);

    std::vector<uint32_t> poisoned(tail_rows);
    for (size_t i = 0; i < tail_rows; ++i) poisoned[i] = (prompt[split + i] + 1) % uint32_t(vocab);
    for (size_t stop = 1; stop <= clean.stage_count(); ++stop) {
        infer::Model model(weights, backends, placement, options);
        infer::Sequence seq = model.make_sequence();
        infer::ExecContext ctx, pending;
        head.seq = &seq;
        model.forward(ctx, &head, 1);
        if (!model.mark(seq)) throw std::runtime_error(name + ": abort prefix mark refused");
        model.reserve_passes(pending, 1, tail_rows, tail_rows);
        infer::BatchEntry wrong{&seq, poisoned.data(), tail_rows, true};
        wrong.every_logits = true;
        wrong.extent = prompt.size();
        model.begin_pass(pending, 0, &wrong, 1, 0);
        for (size_t stage = 0; stage < stop; ++stage) model.run_pass_stage(pending, 0, stage);
        if (stop == model.stage_count()) {
            bool changed = false;
            for (size_t row = 0; row < tail_rows; ++row)
                changed |= std::memcmp(model.pass_logits(pending, 0, row), clean_tail.data() + row * vocab,
                                       vocab * sizeof(float)) != 0;
            if (!changed) throw std::runtime_error(name + ": poisoned suffix did not change any logits");
        }
        model.abort_pass(pending, 0);
        if (seq.length() != split) throw std::runtime_error(name + ": aborted stage lost its marked prefix");
        tail.seq = &seq;
        model.begin_pass(pending, 0, &tail, 1, 0);
        for (size_t stage = 0; stage < model.stage_count(); ++stage) model.run_pass_stage(pending, 0, stage);
        std::vector<float> actual;
        actual.reserve(tail_rows * vocab);
        for (size_t row = 0; row < tail_rows; ++row) {
            const float* values = model.pass_logits(pending, 0, row);
            actual.insert(actual.end(), values, values + vocab);
        }
        model.end_pass(pending, 0);
        if (actual.size() != clean_tail.size() ||
            std::memcmp(actual.data(), clean_tail.data(), actual.size() * sizeof(float)) != 0)
            throw std::runtime_error(name + ": aborted stage changed retry bytes");
        compare(actual, slice_rows(expected, vocab, split, tail_rows), name + " aborted stage retry HF logits");
        decode.seq = &seq;
        model.forward(ctx, &decode, 1);
        const auto actual_decode = logits(ctx, 0, 1, vocab);
        if (actual_decode.size() != clean_decode.size() ||
            std::memcmp(actual_decode.data(), clean_decode.data(), actual_decode.size() * sizeof(float)) != 0)
            throw std::runtime_error(name + ": aborted stage changed next decode bytes");
        std::cout << name << " abort after stage " << stop << ": retry and decode byte-identical\n";
    }
}

void batch_and_scratch_checks(const infer::ModelWeights& weights, backend::BackendPtr backend,
                              const std::vector<uint32_t>& prompt, const std::vector<float>& expected,
                              size_t vocab, const std::string& name) {
    infer::Model model(weights, backend, model_options(2));
    infer::Sequence a = model.make_sequence(), b = model.make_sequence();
    infer::ExecContext first, second;
    const size_t rows = prompt.size() * 2;
    model.reserve_passes(first, 1, rows, rows);
    model.reserve_passes(second, 1, rows, rows);
    if (first.scratch.empty() || first.scratch.size() != second.scratch.size())
        throw std::runtime_error(name + ": per-context index scratch is missing or shared");
    for (size_t d = 0; d < first.scratch.size(); ++d)
        if (first.scratch[d].indices.empty() || second.scratch[d].indices.empty() ||
            first.scratch[d].indices.data() == second.scratch[d].indices.data())
            throw std::runtime_error(name + ": per-context index scratch is missing or shared");
    const auto first_ptrs = [&] {
        std::vector<const uint32_t*> out;
        for (const auto& s : first.scratch) out.push_back(s.indices.data());
        return out;
    }();
    const auto first_sizes = [&] {
        std::vector<size_t> out;
        for (const auto& s : first.scratch) out.push_back(s.indices.size());
        return out;
    }();
    const auto first_capacities = [&] {
        std::vector<size_t> out;
        for (const auto& s : first.scratch) out.push_back(s.indices.capacity());
        return out;
    }();
    auto run_reserved = [&](infer::ExecContext& ctx, infer::BatchEntry* entries, size_t n) {
        model.begin_pass(ctx, 0, entries, n, 0);
        for (size_t stage = 0; stage < model.stage_count(); ++stage) model.run_pass_stage(ctx, 0, stage);
        size_t rows_wanted = 0;
        for (size_t entry = 0; entry < n; ++entry)
            if (entries[entry].want_logits) rows_wanted += entries[entry].every_logits ? entries[entry].n : 1;
        std::vector<float> result;
        result.reserve(rows_wanted * vocab);
        for (size_t row = 0; row < rows_wanted; ++row) {
            const float* values = model.pass_logits(ctx, 0, row);
            result.insert(result.end(), values, values + vocab);
        }
        model.end_pass(ctx, 0);
        return result;
    };
    infer::BatchEntry entries[] = {{&a, prompt.data(), prompt.size(), true},
                                   {&b, prompt.data(), prompt.size(), true}};
    entries[0].every_logits = entries[1].every_logits = true;
    entries[0].extent = entries[1].extent = prompt.size();
    const auto batched = run_reserved(first, entries, 2);
    compare(slice_rows(batched, vocab, 0, prompt.size()), expected, name + " batched sequence A");
    compare(slice_rows(batched, vocab, prompt.size(), prompt.size()), expected, name + " batched sequence B");

    const uint32_t short_ids[] = {prompt[0], prompt[1]};
    infer::BatchEntry shorter[] = {{&a, short_ids, 2, true}, {&b, short_ids, 2, true}};
    shorter[0].every_logits = shorter[1].every_logits = true;
    shorter[0].extent = shorter[1].extent = prompt.size() + 2;
    run_reserved(first, shorter, 2);
    for (size_t d = 0; d < first.scratch.size(); ++d)
        if (first.scratch[d].indices.data() != first_ptrs[d] || first.scratch[d].indices.size() != first_sizes[d] ||
            first.scratch[d].indices.capacity() != first_capacities[d])
            throw std::runtime_error(name + ": scratch changed on an equal-or-smaller pass");
}

void distinct_history_batch_checks(const infer::ModelWeights& weights,
                                   std::vector<backend::BackendPtr> backends, infer::Placement placement,
                                   const Json& first, const Json& second, size_t vocab,
                                   const std::string& name) {
    const std::vector<uint32_t> first_ids = ids(field(first, "input_ids"));
    const std::vector<uint32_t> second_ids = ids(field(second, "input_ids"));
    if (first_ids == second_ids)
        throw std::runtime_error(name + ": fixture histories are not distinct");

    infer::Model model(weights, std::move(backends), placement, model_options(2));
    infer::Sequence a = model.make_sequence(), b = model.make_sequence();
    infer::ExecContext ctx;
    infer::BatchEntry entries[] = {{&a, first_ids.data(), first_ids.size(), true},
                                   {&b, second_ids.data(), second_ids.size(), true}};
    entries[0].every_logits = entries[1].every_logits = true;
    entries[0].extent = first_ids.size();
    entries[1].extent = second_ids.size();
    model.forward(ctx, entries, 2);

    compare(logits(ctx, 0, first_ids.size(), vocab), floats(field(first, "full_logits")),
            name + " sequence A HF logits");
    compare(logits(ctx, first_ids.size(), second_ids.size(), vocab), floats(field(second, "full_logits")),
            name + " sequence B HF logits");

    std::vector<uint32_t> decode_tokens;
    std::vector<infer::BatchEntry> followups;
    std::vector<const Json*> expected_decode;
    std::vector<std::string> sequence_names;
    decode_tokens.reserve(2);
    followups.reserve(2);
    expected_decode.reserve(2);
    sequence_names.reserve(2);
    auto append_decode = [&](const Json& item, infer::Sequence& sequence, const std::string& sequence_name) {
        const Json& token_json = field(item, "decode_token");
        if (!token_json.isNumber()) return;
        const double raw = token_json.asNumber();
        if (!std::isfinite(raw) || raw < 0 || raw > UINT32_MAX || std::floor(raw) != raw)
            throw std::runtime_error(name + ": invalid " + sequence_name + " decode token");
        decode_tokens.push_back(uint32_t(raw));
        infer::BatchEntry entry{&sequence, &decode_tokens.back(), 1, true};
        entry.extent = 1;
        followups.push_back(entry);
        expected_decode.push_back(&item);
        sequence_names.push_back(sequence_name);
    };
    // Reverse the two entries to exercise follow-up routing independently of their initial batch order.
    append_decode(second, b, "sequence B");
    append_decode(first, a, "sequence A");
    if (!followups.empty()) {
        model.forward(ctx, followups.data(), followups.size());
        for (size_t i = 0; i < followups.size(); ++i)
            compare(logits(ctx, i, 1, vocab), floats(field(*expected_decode[i], "decode_logits")),
                    name + " " + sequence_names[i] + " reversed-order decode HF logits");
    }
}

void cpu_layer_split_checks(const infer::ModelWeights& weights, const std::vector<uint32_t>& prompt,
                           const std::vector<float>& expected, size_t vocab, const std::string& name) {
    std::vector<backend::BackendPtr> backends{backend::make_cpu_backend(), backend::make_cpu_backend()};
    backends[0]->set_threads(1);
    backends[1]->set_threads(1);
    infer::Placement placement;
    placement.mixer_device = {0, 1};
    placement.ffn_device = {0, 1};
    infer::Model model(weights, std::move(backends), placement, model_options(2));
    compare(score(model, prompt), expected, name + " two-CPU layer split");
    infer::Sequence a = model.make_sequence(), b = model.make_sequence();
    infer::ExecContext ctx;
    infer::BatchEntry entries[] = {{&a, prompt.data(), prompt.size(), true},
                                   {&b, prompt.data(), prompt.size(), true}};
    entries[0].every_logits = entries[1].every_logits = true;
    entries[0].extent = entries[1].extent = prompt.size();
    model.forward(ctx, entries, 2);
    compare(logits(ctx, 0, prompt.size(), vocab), expected, name + " split sequence A");
    compare(logits(ctx, prompt.size(), prompt.size(), vocab), expected, name + " split sequence B");
}

// The prompt in consecutive passes over the two-CPU split, each slice entering the first stage while the one before it is at the second (docs/MULTI-DEVICE.md, phase 4).
// The QSA layer pools a group two slices share in the later pass: the last slice's rows and a decode step must be the bits of the same slices run one after another, and HF's.
void continued_split_checks(const infer::ModelWeights& weights, const std::vector<uint32_t>& prompt, const std::vector<float>& hf_sliced,
                            const Json& slice_lengths, const Json& item, size_t vocab, const std::string& name) {
    const auto split = [&] {
        std::vector<backend::BackendPtr> backends{backend::make_cpu_backend(), backend::make_cpu_backend()};
        backends[0]->set_threads(1);
        backends[1]->set_threads(1);
        infer::Placement placement;
        placement.mixer_device = {0, 1};
        placement.ffn_device = {0, 1};
        placement.output_device = 1;
        return std::make_unique<infer::Model>(weights, std::move(backends), placement, model_options(2));
    };
    const auto flight = split(), serial = split();
    infer::Sequence q = flight->make_sequence(), s = serial->make_sequence();
    std::vector<infer::BatchEntry> chunks;
    size_t at = 0, widest = 0;
    for (const Json& length : slice_lengths.asArray()) {
        const size_t n = size_t(length.asNumber());
        infer::BatchEntry e{&q, prompt.data() + at, n, at + n == prompt.size()};
        e.every_logits = e.want_logits;
        e.extent = prompt.size();
        chunks.push_back(e);
        widest = std::max(widest, n);
        at += n;
    }
    const size_t last = chunks.back().n;
    infer::ExecContext ctx, serial_ctx;
    flight->reserve_passes(ctx, 2, widest, widest);
    std::vector<float> got, want;
    for (size_t t = 0; t <= chunks.size(); ++t)
        for (size_t st = 0; st < 2; ++st) {
            if (t < st || t - st >= chunks.size()) continue;
            const size_t c = t - st, slot = c % 2;
            if (st == 0) flight->begin_pass(ctx, slot, &chunks[c], 1, 0);
            flight->run_pass_stage(ctx, slot, st);
            if (st == 0) continue;
            for (size_t row = 0; chunks[c].want_logits && row < last; ++row)
                got.insert(got.end(), flight->pass_logits(ctx, slot, row), flight->pass_logits(ctx, slot, row) + vocab);
            flight->end_pass(ctx, slot);
        }
    for (infer::BatchEntry e : chunks) {
        e.seq = &s;
        serial->forward(serial_ctx, &e, 1);
    }
    want = logits(serial_ctx, 0, last, vocab);
    if (q.length() != prompt.size() || got != want)
        throw std::runtime_error(name + ": a prompt in consecutive passes in flight differs from its slices one after another");
    compare(got, slice_rows(hf_sliced, vocab, prompt.size() - last, last), name + " consecutive passes in flight");
    const Json& decode = field(item, "decode_token");
    if (!decode.isNumber()) return;
    const uint32_t token = uint32_t(decode.asNumber());
    infer::BatchEntry step{&q, &token, 1, true};
    flight->begin_pass(ctx, 0, &step, 1, 0);
    flight->run_pass_stage(ctx, 0, 0);
    flight->run_pass_stage(ctx, 0, 1);
    got.assign(flight->pass_logits(ctx, 0, 0), flight->pass_logits(ctx, 0, 0) + vocab);
    flight->end_pass(ctx, 0);
    step.seq = &s;
    serial->forward(serial_ctx, &step, 1);
    if (got != logits(serial_ctx, 0, 1, vocab)) throw std::runtime_error(name + ": a step after consecutive passes in flight differs from one after the slices");
    compare(got, floats(field(item, "decode_logits")), name + " step after consecutive passes in flight");
}

void keep_fork_parity(const infer::ModelWeights& base, const gguf::GGUFModel& file,
                      const infer::qwen4exp::Config& configuration,
                      std::vector<backend::BackendPtr> backends, infer::Placement placement,
                      const std::vector<uint32_t>& seed,
                      const std::string& name) {
    infer::qwen4exp::Config cfg = configuration;
    cfg.context_length = 512;
    uint32_t vocab = 0;
    for (const auto& tensor : file.tensors)
        if (tensor.name == "token_embd.weight" && tensor.ne.size() >= 2) vocab = uint32_t(tensor.ne[1]);
    if (!vocab) throw std::runtime_error("fixture: missing token embedding vocabulary");
    infer::ModelWeights weights = base;
    weights.arch = std::make_shared<const infer::qwen4exp::Qwen4Exp>(cfg, vocab);
    infer::Model model(weights, backends, placement, model_options(2, 1));
    const size_t block = model.kv_block_tokens();
    if (!block || block > 256 || 2 * block > size_t(model.context_length()))
        throw std::runtime_error(name + ": unexpected KV block size for tiny keep/fork check");
    std::vector<uint32_t> ids(2 * block);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = seed[i % seed.size()];
    const std::vector<uint32_t> prefix(ids.begin(), ids.begin() + block);
    const std::vector<uint32_t> suffix(ids.begin() + block, ids.end());

    infer::Sequence source = model.make_sequence();
    infer::ExecContext kept_ctx;
    infer::BatchEntry kept_prefix{&source, prefix.data(), prefix.size(), false};
    kept_prefix.extent = ids.size();
    kept_prefix.keep = true;
    model.forward(kept_ctx, &kept_prefix, 1);
    if (model.checkpoint(source) != std::optional<size_t>(block))
        throw std::runtime_error(name + ": whole-block keep did not create the expected checkpoint");
    infer::Sequence fork = model.fork(source, block);
    infer::BatchEntry kept_tails[] = {{&source, suffix.data(), suffix.size(), true},
                                      {&fork, suffix.data(), suffix.size(), true}};
    kept_tails[0].every_logits = kept_tails[1].every_logits = true;
    kept_tails[0].extent = kept_tails[1].extent = ids.size();
    model.forward(kept_ctx, kept_tails, 2);
    const std::vector<float> source_logits = logits(kept_ctx, 0, block, vocab);
    const std::vector<float> fork_logits = logits(kept_ctx, block, block, vocab);

    infer::Model fresh(weights, backends, placement, model_options(1));
    infer::Sequence reference = fresh.make_sequence();
    infer::ExecContext fresh_ctx;
    infer::BatchEntry fresh_prefix{&reference, prefix.data(), prefix.size(), false};
    fresh_prefix.extent = ids.size();
    fresh.forward(fresh_ctx, &fresh_prefix, 1);
    infer::BatchEntry fresh_tail{&reference, suffix.data(), suffix.size(), true};
    fresh_tail.every_logits = true;
    fresh_tail.extent = ids.size();
    fresh.forward(fresh_ctx, &fresh_tail, 1);
    const std::vector<float> reference_logits = logits(fresh_ctx, 0, block, vocab);
    compare(source_logits, reference_logits, name + " kept continuation parity", 0.0f);
    compare(fork_logits, reference_logits, name + " fork continuation parity", 0.0f);
}

void host_snapshot_restore_parity(const infer::ModelWeights& base,
                                  const infer::qwen4exp::Config& configuration,
                                  std::vector<backend::BackendPtr> backends, infer::Placement placement,
                                  const std::vector<uint32_t>& seed,
                                  const std::vector<float>& hf_logits, size_t vocab,
                                  const std::string& name) {
    infer::qwen4exp::Config cfg = configuration;
    cfg.context_length = 512;
    if (!cfg.has_ple || cfg.ple_ngram_size < 2 ||
        std::find(cfg.layers.begin(), cfg.layers.end(), infer::qwen4exp::qwen_sparse_attention) == cfg.layers.end())
        throw std::runtime_error(name + ": host restore fixture needs PLE and QSA");
    infer::ModelWeights weights = base;
    weights.arch = std::make_shared<const infer::qwen4exp::Qwen4Exp>(cfg, uint32_t(vocab));
    const infer::ModelPlan plan = infer::plan_model(weights);
    bool has_token_history = false, has_qsa_aux = false;
    for (const infer::LayerPlan& layer : plan.layers) {
        has_token_history |= layer.token_history != 0;
        has_qsa_aux |= layer.kind == infer::qwen4exp::qwen_sparse_attention && layer.kv_aux != 0;
    }
    if (!has_token_history || !has_qsa_aux)
        throw std::runtime_error(name + ": host restore fixture plan lacks PLE history or QSA auxiliary KV");

    infer::Model model(weights, backends, placement, model_options(1, 1));
    const size_t block = model.kv_block_tokens();
    if (!block || block > 256 || seed.empty() || seed.size() > block || 2 * block > size_t(model.context_length()))
        throw std::runtime_error(name + ": unexpected KV block size for host restore check");
    std::vector<uint32_t> tokens(2 * block);
    for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = seed[i % seed.size()];
    const std::vector<uint32_t> prefix(tokens.begin(), tokens.begin() + block);
    const std::vector<uint32_t> suffix(tokens.begin() + block, tokens.end());

    infer::Sequence source = model.make_sequence();
    infer::ExecContext source_ctx;
    infer::BatchEntry source_prefix{&source, prefix.data(), prefix.size(), true};
    source_prefix.every_logits = true;
    source_prefix.extent = tokens.size();
    source_prefix.keep = true;
    model.forward(source_ctx, &source_prefix, 1);
    if (model.checkpoint(source) != std::optional<size_t>(block))
        throw std::runtime_error(name + ": host snapshot prefix did not keep its state");
    compare(slice_rows(logits(source_ctx, 0, block, vocab), vocab, 0, seed.size()),
            hf_logits, name + " host snapshot prefix HF logits");

    infer::HostHistory snapshot;
    model.save_host(source, block, snapshot, model.host_bytes(block));
    if (snapshot.length != block || !snapshot.blocks || !snapshot.state ||
        snapshot.held != model.host_bytes(block))
        throw std::runtime_error(name + ": host snapshot omitted model history storage");
    model.reset(source);

    std::vector<uint32_t> other(tokens.size());
    for (size_t i = 0; i < other.size(); ++i) other[i] = (tokens[i] + 1) % uint32_t(vocab);
    infer::Sequence clobber = model.make_sequence();
    infer::ExecContext clobber_ctx;
    infer::BatchEntry other_prefix{&clobber, other.data(), block, false};
    other_prefix.extent = tokens.size();
    other_prefix.keep = true;
    model.forward(clobber_ctx, &other_prefix, 1);
    infer::BatchEntry other_tail{&clobber, other.data() + block, block, false};
    other_tail.extent = tokens.size();
    model.forward(clobber_ctx, &other_tail, 1);
    if (clobber.length() != tokens.size())
        throw std::runtime_error(name + ": replacement history did not overwrite the released state slot");
    model.reset(clobber);

    infer::Sequence restored = model.restore_host(snapshot);
    if (restored.length() != block)
        throw std::runtime_error(name + ": host restore returned the wrong prefix length");
    model.release_host(snapshot);
    model.trim_host(0);
    if (model.host_allocated()) throw std::runtime_error(name + ": released host snapshot still holds slabs");
    infer::ExecContext restored_ctx;
    infer::BatchEntry restored_tail{&restored, suffix.data(), suffix.size(), true};
    restored_tail.every_logits = true;
    restored_tail.extent = tokens.size();
    model.forward(restored_ctx, &restored_tail, 1);
    const std::vector<float> restored_logits = logits(restored_ctx, 0, block, vocab);

    infer::Model fresh(weights, backends, placement, model_options(1));
    infer::Sequence reference = fresh.make_sequence();
    infer::ExecContext fresh_ctx;
    infer::BatchEntry fresh_prefix{&reference, prefix.data(), prefix.size(), true};
    fresh_prefix.every_logits = true;
    fresh_prefix.extent = tokens.size();
    fresh.forward(fresh_ctx, &fresh_prefix, 1);
    infer::BatchEntry fresh_tail{&reference, suffix.data(), suffix.size(), true};
    fresh_tail.every_logits = true;
    fresh_tail.extent = tokens.size();
    fresh.forward(fresh_ctx, &fresh_tail, 1);
    const std::vector<float> fresh_logits = logits(fresh_ctx, 0, block, vocab);
    compare(restored_logits, fresh_logits, name + " host snapshot restored continuation", 0.0f);
    if (std::memcmp(restored_logits.data(), fresh_logits.data(), fresh_logits.size() * sizeof(float)))
        throw std::runtime_error(name + ": restored continuation differs in its logit bits");
}

std::vector<float> score(infer::Model& model, const std::vector<uint32_t>& prompt) {
    std::vector<float> out;
    out.reserve(prompt.size() * model.n_vocab());
    model.score(prompt, [&](size_t, const float* row) { out.insert(out.end(), row, row + model.n_vocab()); });
    model.reset(); // score leaves its convenience sequence live; release its recurrent slot for explicit sequences.
    return out;
}

std::vector<float> score_slices(infer::Model& model, const std::vector<uint32_t>& prompt,
                                const Json& slice_lengths) {
    infer::Sequence seq = model.make_sequence();
    infer::ExecContext ctx;
    std::vector<float> out;
    out.reserve(prompt.size() * model.n_vocab());
    size_t offset = 0;
    for (const Json& length_value : slice_lengths.asArray()) {
        const double raw = length_value.asNumber();
        if (raw <= 0 || std::floor(raw) != raw || raw > prompt.size() - offset)
            throw std::runtime_error("reference: invalid slice length");
        const size_t length = size_t(raw);
        infer::BatchEntry entry{&seq, prompt.data() + offset, length, true};
        entry.every_logits = true;
        entry.extent = prompt.size();
        model.forward(ctx, &entry, 1);
        for (size_t row = 0; row < length; ++row) {
            const float* logits = ctx.logits(row);
            out.insert(out.end(), logits, logits + model.n_vocab());
        }
        offset += length;
    }
    if (offset != prompt.size()) throw std::runtime_error("reference: slices do not cover the prompt");
    return out;
}

infer::ModelWeights views(gguf::GGUFModel& gguf) {
    infer::ModelWeights result;
    result.arch = infer::qwen4exp::open(gguf, "qwen4exp.");
    result.declared_dtype = backend::Dtype::f32;
    if (gguf.offsets.size() != gguf.tensors.size()) throw std::runtime_error("fixture tensor storage mismatch");
    result.tensors.reserve(gguf.tensors.size());
    for (size_t i = 0; i < gguf.tensors.size(); ++i) {
        const auto& tensor = gguf.tensors[i];
        std::vector<uint64_t> shape(tensor.ne.begin(), tensor.ne.end());
        const uint64_t bytes = tensor.data_size();
        result.tensors.push_back({tensor.name, std::move(shape), tensor.type,
                                  gguf.tensor_data(i), size_t(bytes)});
    }
    return result;
}

// This uses load-progress's execution-capable copying CPU device pattern and counts the FFN placement under test.
// Ordinary CpuBuffers keep inherited operations usable.
struct ExpertPlacementBackend final : backend::CpuBackend {
    explicit ExpertPlacementBackend(bool copying) : copying(copying) { set_threads(1); }
    bool copying;
    size_t copied_adoptions = 0, routed_rows = 0;
    struct Window { const void* source; size_t bytes, writes = 0; };
    std::vector<Window> windows;

    bool reads_in_place() const override { return !copying; }
    bool is_cpu() const override { return !copying; }
    backend::BufferPtr adopt(const void* source, size_t bytes) override {
        if (!copying) return backend::CpuBackend::adopt(source, bytes);
        auto result = backend::CpuBackend::alloc(bytes, backend::Memory::device);
        backend::CpuBackend::write(*result, 0, source, bytes);
        if (bytes && result->host_ptr() == source)
            throw std::runtime_error("expert placement device did not copy adopted weights");
        ++copied_adoptions;
        return result;
    }
    void write(backend::Buffer& destination, size_t offset, const void* source, size_t bytes) override {
        backend::CpuBackend::write(destination, offset, source, bytes);
        for (Window& window : windows) if (source == window.source) {
            if (offset || bytes != window.bytes || std::memcmp(destination.host_ptr(), source, bytes))
                throw std::runtime_error("expert stream did not copy an entire frozen stack into its window");
            ++window.writes;
        }
    }
    void matmul_experts(std::initializer_list<backend::Projection> projections, backend::CSlice input,
                        size_t nin, size_t rows, const Routing& routing, backend::RowRuns runs = {},
                        backend::Dtype dtype = backend::Dtype::f16) override {
        if (dtype != backend::Dtype::f32) throw std::runtime_error("expert placement changed F32 dtype");
        routed_rows += rows;
        backend::CpuBackend::matmul_experts(projections, input, nin, rows, routing, runs, dtype);
    }
    void reset_trace() {
        routed_rows = 0;
        for (Window& window : windows) window.writes = 0;
    }
};

void expert_placement_checks(const infer::ModelWeights& weights, const Json& reference, const Json& item) {
    const auto plan = infer::plan_model(weights);
    const auto prompt = ids(field(item, "input_ids"));
    const auto full = floats(field(item, "full_logits"));
    const auto sliced = floats(field(item, "sliced_logits"));
    const auto& slice_lengths = field(reference, "slice_lengths");
    const size_t layers = plan.layers.size(), vocab = plan.vocab;
    const std::string name = field(item, "name").asString();
    for (int arm = 0; arm < 3; ++arm) {
        const bool host_experts = arm != 0, streaming = arm == 2;
        const std::string label = name + (streaming ? " whole-stack stream" : host_experts ? " CPU experts" : " resident experts");
        auto device = std::make_shared<ExpertPlacementBackend>(true);
        auto host = std::make_shared<ExpertPlacementBackend>(false);
        for (const auto& layer : plan.layers) {
            if (!layer.routed) throw std::runtime_error("expert placement fixture needs routed layers");
            for (const auto& role : layer.roles) if (role.stream == infer::Stream::window) {
                if (!role.tensor) throw std::runtime_error("expert stream fixture has an unresolved window");
                const auto& tensor = weights.tensors[*role.tensor];
                device->windows.push_back({tensor.data, tensor.bytes});
            }
        }
        if (device->windows.size() != 3 * layers)
            throw std::runtime_error("expert placement fixture needs three whole stacks per layer");
        infer::Placement placement;
        placement.mixer_device.assign(layers, 0);
        placement.ffn_device.assign(layers, host_experts ? 1 : 0);
        placement.stream_from = streaming ? 2 : 0;
        infer::Model model(weights, {device, host}, placement, model_options());
        model.set_ubatch(int(prompt.size()));
        if (!device->copied_adoptions) throw std::runtime_error(label + ": copying device adopted no weights");
        auto clear = [&] { device->reset_trace(); host->reset_trace(); };
        auto witness = [&](size_t rows, size_t passes, bool generated) {
            const bool on_device = !host_experts || (streaming && !generated);
            if (device->routed_rows != (on_device ? layers * rows : 0) ||
                host->routed_rows != (on_device ? 0 : layers * rows))
                throw std::runtime_error(label + ": routed products ran on the wrong backend");
            size_t bytes = 0;
            for (const auto& window : device->windows) {
                const size_t expected = streaming && !generated ? passes : 0;
                if (window.writes != expected)
                    throw std::runtime_error(label + ": wrong whole-stack window copy count");
                bytes += window.writes * window.bytes;
            }
            std::cout << label << (generated ? " generated" : " prompt")
                      << " routed_rows device=" << device->routed_rows << " CPU=" << host->routed_rows
                      << " whole_stack_bytes=" << bytes << '\n';
        };
        clear();
        compare(score(model, prompt), full, label + " full HF logits");
        witness(prompt.size(), 1, false);
        clear();
        compare(score_slices(model, prompt, slice_lengths), sliced, label + " sliced HF logits");
        witness(prompt.size(), slice_lengths.asArray().size(), false);
        const auto& decode = field(item, "decode_token");
        if (decode.isNumber()) {
            infer::Sequence seq = model.make_sequence();
            infer::ExecContext ctx;
            infer::BatchEntry head{&seq, prompt.data(), prompt.size(), true};
            head.extent = prompt.size();
            model.forward(ctx, &head, 1);
            clear();
            const double decode_id = decode.asNumber();
            if (!std::isfinite(decode_id) || decode_id < 0 || decode_id > UINT32_MAX || std::floor(decode_id) != decode_id)
                throw std::runtime_error(label + ": invalid follow-up token");
            const uint32_t token = uint32_t(decode_id);
            infer::BatchEntry next{&seq, &token, 1, true};
            next.extent = 1;
            model.forward(ctx, &next, 1);
            compare(logits(ctx, 0, 1, vocab), floats(field(item, "decode_logits")), label + " follow-up HF logits");
            witness(1, 0, true);
        }
    }
}

struct PaddedSharedWeights {
    infer::ModelWeights weights;
    infer::qwen4exp::Config config;
    std::vector<float> storage;
};

PaddedSharedWeights pad_shared_expert_for_tensor_group(const infer::ModelWeights& base,
                                                       const gguf::GGUFModel& file) {
    infer::qwen4exp::Config config = infer::qwen4exp::read_config(file, "qwen4exp.");
    if (config.n_ff_shexp != 7)
        throw std::runtime_error("tensor-width fixture expected shared-expert width 7");
    const size_t hidden = size_t(config.n_embd), old_width = size_t(config.n_ff_shexp), padded_width = 8;
    const size_t layer_count = size_t(config.n_layer);

    PaddedSharedWeights result;
    result.weights = base;
    result.storage.resize(layer_count * 3 * hidden * padded_width);
    size_t cursor = 0;
    auto pad_matrix = [&](size_t tensor, bool down) {
        if (tensor >= result.weights.tensors.size())
            throw std::runtime_error("tensor-width fixture is missing a shared-expert tensor");
        infer::TensorView& view = result.weights.tensors[tensor];
        const std::vector<uint64_t> wanted = down
            ? std::vector<uint64_t>{uint64_t(old_width), uint64_t(hidden)}
            : std::vector<uint64_t>{uint64_t(hidden), uint64_t(old_width)};
        if (view.shape != wanted || view.type != quant::GGML_TYPE_F32 ||
            view.bytes != hidden * old_width * sizeof(float) || !view.data)
            throw std::runtime_error("tensor-width fixture shared-expert tensor has an unexpected shape or type: " + view.name);

        float* padded = result.storage.data() + cursor;
        const float* source = reinterpret_cast<const float*>(view.data);
        if (down) {
            for (size_t row = 0; row < hidden; ++row) {
                std::memcpy(padded + row * padded_width, source + row * old_width,
                            old_width * sizeof(float));
                padded[row * padded_width + old_width] = 0.0f;
            }
            view.shape[0] = padded_width;
        } else {
            std::memcpy(padded, source, hidden * old_width * sizeof(float));
            std::fill(padded + hidden * old_width, padded + hidden * padded_width, 0.0f);
            view.shape[1] = padded_width;
        }
        view.data = reinterpret_cast<const uint8_t*>(padded);
        view.bytes = hidden * padded_width * sizeof(float);
        cursor += hidden * padded_width;
    };

    for (size_t layer = 0; layer < layer_count; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        for (const auto& [suffix, down] : {
                 std::pair<const char*, bool>{"ffn_gate_shexp.weight", false},
                 {"ffn_up_shexp.weight", false}, {"ffn_down_shexp.weight", true}}) {
            const std::string name = prefix + suffix;
            size_t tensor = result.weights.tensors.size();
            for (size_t i = 0; i < result.weights.tensors.size(); ++i)
                if (result.weights.tensors[i].name == name) { tensor = i; break; }
            pad_matrix(tensor, down);
        }
    }
    if (cursor != result.storage.size()) throw std::logic_error("tensor-width fixture padding size mismatch");
    config.n_ff_shexp = int(padded_width);
    result.config = config;
    uint32_t vocab = 0;
    for (const gguf::TensorInfo& tensor : file.tensors)
        if (tensor.name == "token_embd.weight" && tensor.ne.size() > 1 &&
            tensor.ne[1] <= uint64_t(std::numeric_limits<uint32_t>::max()))
            vocab = uint32_t(tensor.ne[1]);
    if (!vocab) throw std::runtime_error("tensor-width fixture is missing its vocabulary tensor");
    result.weights.arch = std::make_shared<const infer::qwen4exp::Qwen4Exp>(
        config, vocab);
    return result;
}

void loader_checks(const std::string& path, const std::string& device, const std::vector<uint32_t>& prompt,
                   const std::vector<float>& expected) {
    for (infer::LoadMode mode : {infer::LoadMode::automatic, infer::LoadMode::mapped, infer::LoadMode::direct}) {
        if (mode == infer::LoadMode::direct) {
            try { format::FileReader probe(path, true); }
            catch (const format::DirectUnavailable& e) {
                std::cout << "loader direct unavailable: " << e.what() << '\n';
                continue;
            }
        }
        for (infer::LookupStorage storage : {infer::LookupStorage::automatic, infer::LookupStorage::device, infer::LookupStorage::file}) {
            auto loaded = std::make_unique<infer::LoadedModel>();
            loaded->file = gguf::read_gguf(path);
            infer::ModelWeights weights = views(loaded->file);
            size_t payload = 0, table_bytes = 0, table_rows = 0, table_row_bytes = 0;
            uint32_t table_type = 0;
            for (const auto& t : weights.tensors) {
                payload += t.bytes;
                if (t.name == "per_layer_token_embd.weight") {
                    table_bytes = t.bytes;
                    table_rows = size_t(t.shape[1]);
                    table_row_bytes = quant::row_bytes(t.type, size_t(t.shape[0]));
                    table_type = t.type;
                }
            }
            infer::PlacementRequest request;
            request.names = {device};
            request.dtype = backend::Dtype::f32;
            request.lookup_storage = storage;
            request.ubatch = int(prompt.size());
            size_t done = 0, total = 0, reports = 0;
            const auto progress = [&](const infer::LoadStep& step) {
                if (std::string(step.phase) != "weights") return;
                const size_t current = step.completed, bytes = step.total;
                if ((!reports && current) || current < done || current > bytes || (reports && bytes != total))
                    throw std::runtime_error("loader: inconsistent progress");
                done = current;
                total = bytes;
                ++reports;
            };
            loaded.reset();
            loaded = infer::load_model(path, {test_backend(device)}, request, model_options(), progress, mode);
            if (!loaded->tok) throw std::runtime_error("loader: missing tokenizer");
            const bool external = storage == infer::LookupStorage::file;
            if (loaded->model->stage_reads(0) != external) throw std::runtime_error("loader: wrong table storage selected");
            const auto& topology = loaded->model->topology();
            if (topology.lookups.size() != 1) throw std::runtime_error("loader: topology lookup count is wrong");
            const auto& lookup = topology.lookups.front();
            if (lookup.layer >= topology.layers.size() || lookup.rows != table_rows || lookup.row_bytes != table_row_bytes ||
                lookup.bytes != table_bytes || lookup.type != table_type || lookup.file_backed != external ||
                lookup.members != topology.layers[lookup.layer].mixer_members)
                throw std::runtime_error("loader: topology lookup details are wrong");
            if (!reports || done != total || total != payload - (external ? table_bytes : 0))
                throw std::runtime_error("loader: whole-table payload count is wrong");
            if (mode == infer::LoadMode::direct)
                for (const auto& segment : loaded->file.segments)
                    if (segment.file) throw std::runtime_error("loader: direct mode mapped a file");
            // A table read from its file reads through the table's readers in every mode, so it holds no share of the payload's mapping.
            if (external)
                for (const auto& segment : loaded->file.segments)
                    if (segment.file && segment.file.use_count() != 1)
                        throw std::runtime_error(std::string("loader: ") + infer::load_mode_name(mode) + " reads its lookup table through the mapping");
            const std::string label = std::string("loader ") + infer::load_mode_name(mode) + "/" + infer::lookup_storage_name(storage);
            compare(score(*loaded->model, prompt), expected, label);
            compare(score(*loaded->model, prompt), expected, label + " repeat");
        }
    }
}

infer::ModelWeights with_ple_row_source(const infer::ModelWeights& base, gguf::GGUFModel& gguf,
                                       std::shared_ptr<const format::FileReader>& retained_reader,
                                       size_t& table_tensor) {
    infer::ModelWeights result = base;
    table_tensor = result.tensors.size();
    for (size_t i = 0; i < result.tensors.size(); ++i)
        if (result.tensors[i].name == "per_layer_token_embd.weight") table_tensor = i;
    if (table_tensor == result.tensors.size()) throw std::runtime_error("fixture: missing PLE table");
    auto& view = result.tensors[table_tensor];
    if (view.shape.size() != 2) throw std::runtime_error("fixture: PLE table is not a matrix");
    const format::FileSpan span = gguf.span(table_tensor);
    retained_reader = std::make_shared<const format::FileReader>(span.file);
    view.row_source = std::make_shared<const infer::WeightRows>(
        view.type, size_t(view.shape[0]), size_t(view.shape[1]), retained_reader, span.offset, span.bytes);
    view.data = nullptr;
    return result;
}

void row_source_checks(const infer::ModelWeights& base, gguf::GGUFModel& gguf, const std::string& device,
                       const std::vector<uint32_t>& prompt, const std::vector<float>& hf_full,
                       const std::vector<float>& hf_sliced, const std::vector<float>& hf_cached,
                       const Json& slice_lengths, const Json& item, size_t vocab, const std::string& name,
                       bool extended_checks) {
    {
        std::shared_ptr<const format::FileReader> reader_lifetime;
        size_t table_tensor = 0;
        infer::ModelWeights source = with_ple_row_source(base, gguf, reader_lifetime, table_tensor);
        const infer::AdoptWeight adopt = [&](size_t tensor, backend::Backend& b,
                                              const infer::shard::Held* held) {
            if (tensor == table_tensor)
                throw std::runtime_error(name + ": external PLE table was adopted as a whole tensor");
            if (held) throw std::runtime_error(name + ": unexpected grouped tensor in row-source test");
            const auto& view = source.tensors[tensor];
            if (!view.data) throw std::runtime_error(name + ": unrelated test tensor has no mapped bytes");
            return b.adopt(view.data, view.bytes);
        };
        infer::Model model(source, std::vector<backend::BackendPtr>{test_backend(device)},
                           infer::Placement{}, model_options(), adopt);
        // The model and its row source must own the file reader after construction.
        source.tensors[table_tensor].row_source.reset();
        reader_lifetime.reset();
        const std::string mode = "file-backed rows";
        compare(score(model, prompt), hf_full, name + " " + mode + " full prompt");
        compare(score_slices(model, prompt, slice_lengths), hf_sliced, name + " " + mode + " sliced prompt");
        compare(score(model, prompt), hf_cached, name + " " + mode + " cached prompt");
        const Json& decode = field(item, "decode_token");
        if (decode.isNumber()) {
            const double raw = decode.asNumber();
            if (!std::isfinite(raw) || raw < 0 || raw > UINT32_MAX || std::floor(raw) != raw)
                throw std::runtime_error(name + ": invalid row-source decode token");
            const uint32_t token = uint32_t(raw);
            const float* got = model.step(prompt.data(), prompt.size());
            compare(std::vector<float>(got, got + prompt.size() * vocab), hf_cached,
                    name + " " + mode + " decode-path prompt");
            got = model.step(&token, 1);
            compare(std::vector<float>(got, got + vocab), floats(field(item, "decode_logits")),
                    name + " " + mode + " follow-up");
        }
        if (vocab != model.n_vocab()) throw std::runtime_error(name + ": row-source model vocabulary changed");

        if (extended_checks) {
            // Reattach an owned source for models constructed by the history and batch helpers.
            source = with_ple_row_source(base, gguf, reader_lifetime, table_tensor);
            history_checks(source, test_backend(device), prompt, hf_full, vocab, name + " " + mode);
            batch_and_scratch_checks(source, test_backend(device), prompt, hf_full, vocab, name + " " + mode);
            if (device == "cpu") cpu_layer_split_checks(source, prompt, hf_full, vocab, name + " " + mode);
        }
    }
}

void tensor_group_row_source_checks(const infer::ModelWeights& base, gguf::GGUFModel& gguf,
                                    const std::string& group_devices,
                                    const infer::qwen4exp::Config& config,
                                    const std::vector<uint32_t>& prompt, const std::vector<float>& hf_full,
                                    const std::vector<float>& hf_sliced, const std::vector<float>& hf_cached,
                                    const Json& slice_lengths, const Json& item, size_t vocab,
                                    const std::string& name) {
    {
        std::shared_ptr<const format::FileReader> reader_lifetime;
        size_t table_tensor = 0;
        infer::ModelWeights source = with_ple_row_source(base, gguf, reader_lifetime, table_tensor);
        infer::Model model(source, tensor_group_backends(group_devices), tensor_placement(), model_options(2));
        source.tensors[table_tensor].row_source.reset();
        reader_lifetime.reset();
        const std::string mode = "file-backed rows";
        compare(score(model, prompt), hf_full,
                name + " padded shared width 8 tensor width 2 " + mode + " full prompt");
        compare(score_slices(model, prompt, slice_lengths), hf_sliced,
                name + " padded shared width 8 tensor width 2 " + mode + " sliced prompt");
        compare(score(model, prompt), hf_cached,
                name + " padded shared width 8 tensor width 2 " + mode + " cached prompt");

        source = with_ple_row_source(base, gguf, reader_lifetime, table_tensor);
        host_snapshot_restore_parity(source, config, tensor_group_backends(group_devices), tensor_placement(),
                                     prompt, hf_full, vocab,
                                     name + " padded shared width 8 tensor width 2 " + mode);
        compare(score(model, prompt), hf_full,
                name + " padded shared width 8 tensor width 2 " + mode + " after the snapshot");

        const Json& decode = field(item, "decode_token");
        if (!decode.isNumber()) return;
        const double raw = decode.asNumber();
        if (!std::isfinite(raw) || raw < 0 || raw > UINT32_MAX || std::floor(raw) != raw)
            throw std::runtime_error(name + ": invalid tensor-group row-source decode token");
        const uint32_t token = uint32_t(raw);
        const float* got = model.step(prompt.data(), prompt.size());
        compare(std::vector<float>(got, got + prompt.size() * vocab), hf_cached,
                name + " padded shared width 8 tensor width 2 " + mode + " decode-path prompt");
        got = model.step(&token, 1);
        compare(std::vector<float>(got, got + vocab), floats(field(item, "decode_logits")),
                name + " padded shared width 8 tensor width 2 " + mode + " follow-up");
    }
}

void tensor_group_case_checks(const infer::ModelWeights& weights, gguf::GGUFModel& gguf,
                              const infer::qwen4exp::Config& config,
                              const std::vector<uint32_t>& prompt, const std::vector<float>& hf_full,
                              const std::vector<float>& hf_sliced, const std::vector<float>& hf_cached,
                              const Json& slice_lengths, const Json& item, size_t vocab,
                              const std::string& name, bool row_source_case, const std::string& group_devices) {
    infer::Model control(weights, test_backend("cpu"), model_options());
    compare(score(control, prompt), hf_full, name + " padded shared width 8 tensor width 1 full prompt");
    compare(score_slices(control, prompt, slice_lengths), hf_sliced,
            name + " padded shared width 8 tensor width 1 sliced prompt");
    compare(score(control, prompt), hf_cached, name + " padded shared width 8 tensor width 1 cached prompt");
    const Json& decode = field(item, "decode_token");
    if (decode.isNumber()) {
        const double raw = decode.asNumber();
        if (!std::isfinite(raw) || raw < 0 || raw > UINT32_MAX || std::floor(raw) != raw)
            throw std::runtime_error(name + ": invalid padded tensor width 1 decode token");
        const uint32_t token = uint32_t(raw);
        const float* got = control.step(prompt.data(), prompt.size());
        compare(std::vector<float>(got, got + prompt.size() * vocab), hf_cached,
                name + " padded shared width 8 tensor width 1 decode-path prompt");
        got = control.step(&token, 1);
        compare(std::vector<float>(got, got + vocab), floats(field(item, "decode_logits")),
                name + " padded shared width 8 tensor width 1 follow-up");
    }

    infer::Model model(weights, tensor_group_backends(group_devices), tensor_placement(), model_options());
    compare(score(model, prompt), hf_full, name + " padded shared width 8 tensor width 2 full prompt");
    compare(score_slices(model, prompt, slice_lengths), hf_sliced,
            name + " padded shared width 8 tensor width 2 sliced prompt");
    compare(score(model, prompt), hf_cached, name + " padded shared width 8 tensor width 2 cached prompt");

    {
        infer::Model marked(weights, tensor_group_backends(group_devices), tensor_placement(),
                            model_options(1, 0, 1, prompt.size()));
        infer::Sequence seq = marked.make_sequence();
        infer::ExecContext ctx;
        const size_t split = prompt.size() / 2;
        infer::BatchEntry head{&seq, prompt.data(), split, false};
        head.extent = prompt.size();
        marked.forward(ctx, &head, 1);
        if (!marked.mark(seq)) throw std::runtime_error(name + ": padded tensor width 2 mark refused");
        infer::BatchEntry tail{&seq, prompt.data() + split, prompt.size() - split, true};
        tail.every_logits = true;
        tail.extent = prompt.size();
        marked.forward(ctx, &tail, 1);
        const size_t target = std::min(prompt.size(), split + 2);
        if (marked.retract(seq, target) != target)
            throw std::runtime_error(name + ": padded tensor width 2 mark/retract did not reach requested prefix");
        if (target < prompt.size()) {
            infer::BatchEntry remainder{&seq, prompt.data() + target, prompt.size() - target, true};
            remainder.every_logits = true;
            remainder.extent = prompt.size();
            marked.forward(ctx, &remainder, 1);
            compare(logits(ctx, 0, prompt.size() - target, vocab),
                    slice_rows(hf_full, vocab, target, prompt.size() - target),
                    name + " padded shared width 8 tensor width 2 marked continuation");
        }
    }

    keep_fork_parity(weights, gguf, config, tensor_group_backends(group_devices), tensor_placement(),
                     prompt, name + " padded shared width 8 tensor width 2");

    if (decode.isNumber()) {
        const double raw = decode.asNumber();
        if (!std::isfinite(raw) || raw < 0 || raw > UINT32_MAX || std::floor(raw) != raw)
            throw std::runtime_error(name + ": invalid tensor width 2 decode token");
        const uint32_t token = uint32_t(raw);
        const float* got = model.step(prompt.data(), prompt.size());
        compare(std::vector<float>(got, got + prompt.size() * vocab), hf_cached,
                name + " padded shared width 8 tensor width 2 decode-path prompt");
        got = model.step(&token, 1);
        compare(std::vector<float>(got, got + vocab), floats(field(item, "decode_logits")),
                name + " padded shared width 8 tensor width 2 follow-up");
    }

    if (row_source_case)
        tensor_group_row_source_checks(weights, gguf, group_devices, config, prompt, hf_full, hf_sliced, hf_cached,
                                       slice_lengths, item, vocab, name);
}

struct AdmissionProbe final : backend::CpuBackend {
    explicit AdmissionProbe(backend::Op denied) : denied_(denied) { set_threads(1); }
    bool implements(backend::Op op) const override {
        return op != denied_ && backend::CpuBackend::implements(op);
    }
    backend::BufferPtr alloc(size_t bytes, backend::Memory where = backend::Memory::device) override {
        ++allocations;
        return backend::CpuBackend::alloc(bytes, where);
    }
    backend::BufferPtr adopt(const void* data, size_t bytes) override {
        ++adoptions;
        return backend::CpuBackend::adopt(data, bytes);
    }
    size_t allocations = 0;
    size_t adoptions = 0;
private:
    backend::Op denied_;
};

void external_row_source_refusal(const infer::ModelWeights& base, gguf::GGUFModel& gguf) {
    std::shared_ptr<const format::FileReader> reader;
    size_t table = 0;
    infer::ModelWeights malformed = with_ple_row_source(base, gguf, reader, table);
    size_t output = malformed.tensors.size();
    for (size_t i = 0; i < malformed.tensors.size(); ++i)
        if (malformed.tensors[i].name == "output.weight") output = i;
    if (output == malformed.tensors.size()) throw std::runtime_error("fixture: missing output weight");
    malformed.tensors[output].row_source = malformed.tensors[table].row_source;
    auto probe = std::make_shared<AdmissionProbe>(backend::Op::group_rms_norm);
    bool refused = false;
    try {
        infer::Model rejected(malformed, probe, model_options());
    } catch (const std::runtime_error& error) {
        refused = std::string(error.what()) ==
            "inference: external row source needs a declared lookup output.weight";
        if (!refused) throw;
    }
    if (!refused) throw std::runtime_error("external row source on an output matrix was accepted");
    if (probe->allocations || probe->adoptions)
        throw std::runtime_error("invalid external row source allocated or adopted model storage");
}

void head_op_admission(const infer::ModelWeights& weights, const std::string& name) {
    for (backend::Op op : {backend::Op::group_rms_norm, backend::Op::silu_scaled, backend::Op::mix_streams}) {
        auto probe = std::make_shared<AdmissionProbe>(op);
        const std::string expected = std::string("inference: the head needs ") + backend::op_name(op) +
            ", which the backend of its device does not implement";
        bool refused = false;
        try {
            infer::Model rejected(weights, probe, model_options());
        } catch (const std::runtime_error& error) {
            refused = error.what() == expected;
            if (!refused) throw;
        }
        if (!refused) throw std::runtime_error(name + ": missing head op was not refused by name: " + expected);
        if (probe->allocations || probe->adoptions)
            throw std::runtime_error(name + ": missing head op allocated or adopted model storage");
        std::cout << name << " admission " << backend::op_name(op) << " refused before storage\n";
    }
}


// Every rope call and every pooled group's positions, a row of one stream written out thrice.
struct ImageProbe : backend::CpuBackend {
    std::vector<std::vector<uint32_t>> ropes;
    std::vector<std::pair<size_t, std::vector<uint32_t>>> groups;   // per pooling call, its group size and its groups' positions, three a group
    static std::vector<uint32_t> three(const uint32_t* pos, size_t n, backend::RopeStreams streams) {
        std::vector<uint32_t> out;
        for (size_t r = 0; r < n; ++r)
            for (size_t k = 0; k < 3; ++k) out.push_back(pos[r * streams.count + (streams.count == 3 ? k : 0)]);
        return out;
    }
    void norm_rope_partial(backend::Slice dst, backend::CSlice src, size_t rows, size_t src_stride, size_t src_head_stride, size_t heads, size_t head_dim,
                           size_t rope_dim, backend::CSlice w, float eps, backend::CSlice cos, backend::CSlice sin, const uint32_t* pos,
                           backend::RopeStreams streams = {}) override {
        ropes.push_back(three(pos, rows, streams));
        CpuBackend::norm_rope_partial(dst, src, rows, src_stride, src_head_stride, heads, head_dim, rope_dim, w, eps, cos, sin, pos, streams);
    }
    void record(const backend::KVView* views, size_t n_views, size_t c, const backend::GroupPositions& at) {
        size_t count = 0;
        for (size_t v = 0; v < n_views; ++v) count += (views[v].length + views[v].nq) / c - views[v].length / c;
        std::vector<uint32_t> out;
        if (at.pos) out = three(at.pos, count, at.streams);
        else
            for (size_t v = 0; v < n_views; ++v)
                for (size_t g = views[v].length / c; g < (views[v].length + views[v].nq) / c; ++g) out.insert(out.end(), 3, uint32_t(g * c));
        groups.push_back({c, out});
    }
    void select_kv(backend::CSlice q, size_t layer, const backend::KVView* views, size_t n_views, backend::Slice ids, const IndexArgs& args, backend::CSlice norm,
                   backend::CSlice cos, backend::CSlice sin, const backend::GroupPositions& at = {}) override {
        record(views, n_views, args.compression, at);
        CpuBackend::select_kv(q, layer, views, n_views, ids, args, norm, cos, sin, at);
    }
};

// A prompt of text and images and HF's get_rope_index written out: a text run counting on, an image's row at (start, start + row, start + column), and the start then past the larger side of its grid.
struct ImageSegment {
    size_t n = 0;
    uint32_t h = 0, w = 0;
};

std::vector<uint32_t> hf_image_positions(const std::vector<ImageSegment>& segments, size_t extra) {
    std::vector<uint32_t> out;
    uint32_t st = 0;
    for (const ImageSegment& s : segments) {
        if (s.n) {
            for (size_t k = 0; k < s.n; ++k) out.insert(out.end(), 3, uint32_t(st + k));
            st += uint32_t(s.n);
        } else {
            for (uint32_t r = 0; r < s.h; ++r)
                for (uint32_t c = 0; c < s.w; ++c) out.insert(out.end(), {st, st + r, st + c});
            st += std::max(s.h, s.w);
        }
    }
    for (size_t k = 0; k < extra; ++k) out.insert(out.end(), 3, uint32_t(st + k));
    return out;
}

// Images on Flash-Next (docs/VISION.md, section 8): every rope and pooled group's position get_rope_index's, a group turning at its first row's position wherever that row lies.
// The logits are the same bits however the prompt is sliced.
void image_checks(const infer::ModelWeights& weights) {
    const std::vector<ImageSegment> segments = {{3, 0, 0}, {0, 2, 3}, {2, 0, 0}, {0, 2, 4}, {3, 0, 0}};
    const std::vector<uint32_t> steps = {9, 14, 27};
    std::vector<float> first;
    for (int ubatch : {64, 1, 3, 7}) {
        auto probe = std::make_shared<ImageProbe>();
        infer::ModelOptions options;
        options.dtype = backend::Dtype::f32;
        options.kv_k = options.kv_v = backend::KVType::f32;
        infer::Model model(weights, probe, options);
        model.set_ubatch(ubatch);
        std::vector<uint32_t> prompt;
        std::vector<infer::MediaRun> runs;
        for (const ImageSegment& s : segments) {
            if (s.n) {
                for (size_t k = 0; k < s.n; ++k) prompt.push_back(uint32_t((prompt.size() * 7 + 3) % model.n_vocab()));
                continue;
            }
            infer::MediaRun run;
            run.first = prompt.size();
            run.rows = size_t(s.h) * s.w;
            run.grid_h = s.h;
            run.grid_w = s.w;
            std::vector<float> rows(run.rows * model.embedding_width());
            for (size_t i = 0; i < rows.size(); ++i) rows[i] = float(int((i * 13 + runs.size() * 5) % 23) - 11) / 32.0f;
            run.encoded = std::make_shared<const std::vector<float>>(std::move(rows));
            prompt.insert(prompt.end(), run.rows, 0u);
            runs.push_back(std::move(run));
        }
        std::vector<float> out = model.prefill(prompt, 0, runs);
        for (uint32_t t : steps) {
            const std::vector<float> next = step_row(model, int(t));
            out.insert(out.end(), next.begin(), next.end());
        }
        const std::vector<uint32_t> expected = hf_image_positions(segments, steps.size());
        // Every rope call reads its rows' positions, so the calls of one pass agree and the passes' rows in order are the prompt's and the steps'.
        for (const auto& call : probe->ropes) {
            bool found = false;
            for (size_t at = 0; !found && at + call.size() <= expected.size(); at += 3)
                found = std::equal(call.begin(), call.end(), expected.begin() + std::ptrdiff_t(at));
            if (!found) throw std::runtime_error("qwen4exp images: a rope call's positions are not HF's get_rope_index at ubatch " + std::to_string(ubatch));
        }
        // Every pooled group turns at its first row's position.
        if (probe->groups.empty()) throw std::runtime_error("qwen4exp images: no pooled layer selected");
        for (const auto& [c, got] : probe->groups)
            for (size_t i = 0; i < got.size(); i += 3) {
                bool found = false;
                for (size_t g = 0; !found && (g + 1) * c * 3 <= expected.size(); ++g)
                    found = std::equal(got.begin() + std::ptrdiff_t(i), got.begin() + std::ptrdiff_t(i + 3), expected.begin() + std::ptrdiff_t(g * c * 3));
                if (!found) throw std::runtime_error("qwen4exp images: a pooled group's position is no group's first row at ubatch " + std::to_string(ubatch));
            }
        if (first.empty()) first = out;
        else if (out.size() != first.size() || std::memcmp(out.data(), first.data(), out.size() * sizeof(float)))
            throw std::runtime_error("qwen4exp images: the logits at ubatch " + std::to_string(ubatch) + " differ from one pass");
    }
    std::cout << "qwen4exp images: rope and pooled-group positions HF's get_rope_index, the same bits over slices of 64, 1, 3 and 7\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const char* usage = "usage: qwen4exp-model-test --gguf FILE --reference FILE [--device cpu|vulkan:N[,vulkan:M]] [--tensor-width 2]";
        std::string gguf_path, reference_path, device = "cpu";
        size_t tensor_width = 1;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if ((arg == "--gguf" || arg == "--reference" || arg == "--device" ||
                 arg == "--tensor-width") && i + 1 < argc) {
                const std::string value = argv[++i];
                if (arg == "--gguf") gguf_path = value;
                else if (arg == "--reference") reference_path = value;
                else if (arg == "--device") device = value;
                else {
                    if (value != "1" && value != "2")
                        throw std::runtime_error("qwen4exp model test supports tensor widths 1 and 2");
                    tensor_width = size_t(value[0] - '0');
                }
            } else throw std::runtime_error(usage);
        }
        if (gguf_path.empty() || reference_path.empty())
            throw std::runtime_error(usage);
        if (tensor_width == 2) (void)tensor_group_device_names(device);
        else if (device.find(',') != std::string::npos)
            throw std::runtime_error("a device pair requires --tensor-width 2");

        std::ifstream ref_file(reference_path, std::ios::binary);
        if (!ref_file) throw std::runtime_error("cannot open reference file");
        const std::string ref_text((std::istreambuf_iterator<char>(ref_file)), {});
        const Json root = jmini::Parser(ref_text).parse();

        gguf::GGUFModel file = gguf::read_gguf(gguf_path);
        gguf::map_payload(file);
        infer::ModelWeights weights = views(file);
        if (tensor_width == 2) {
            PaddedSharedWeights padded = pad_shared_expert_for_tensor_group(weights, file);
            const auto& cases = field(root, "cases").asArray();
            for (size_t a = 0; a < cases.size(); ++a)
                for (size_t b = a + 1; b < cases.size(); ++b)
                    distinct_history_batch_checks(
                        padded.weights, tensor_group_backends(device), tensor_placement(), cases[a], cases[b], 32,
                        field(cases[a], "name").asString() + " + " + field(cases[b], "name").asString() +
                            " padded shared width 8 tensor width 2 distinct batch");
            size_t decode_count = 0;
            for (size_t i = 0; i < cases.size(); ++i) {
                const Json& item = cases[i];
                const std::string name = field(item, "name").asString();
                const std::vector<uint32_t> prompt = ids(field(item, "input_ids"));
                aborted_stage_rollback_checks(padded.weights, tensor_group_backends(device), tensor_placement(),
                                               prompt, floats(field(item, "full_logits")), 32,
                                               name + " tensor width 2");
                tensor_group_case_checks(padded.weights, file, padded.config, prompt, floats(field(item, "full_logits")),
                                         floats(field(item, "sliced_logits")), floats(field(item, "cached_logits")),
                                         field(root, "slice_lengths"), item, 32, name, i == 0, device);
                if (field(item, "decode_token").isNumber()) ++decode_count;
            }
            if (decode_count != 2) throw std::runtime_error("expected two tensor-width follow-up decode cases");
            std::cout << "qwen4exp tensor width 2: " << cases.size()
                      << " HF cases passed with shared-expert width padded from 7 to 8\n";
            return 0;
        }
        external_row_source_refusal(weights, file);
        head_op_admission(weights, "qwen4exp");
        if (device == "cpu") image_checks(weights);
        infer::ModelOptions options;
        options.dtype = backend::Dtype::f32;
        options.kv_k = options.kv_v = backend::KVType::f32;
        options.state_slots = 1;
        backend::BackendPtr primary_backend = test_backend(device);
        infer::Model model(weights, primary_backend, options);
        if (model.n_vocab() != 32) throw std::runtime_error("unexpected fixture vocabulary width");

        const auto& cases = field(root, "cases").asArray();
        for (size_t a = 0; a < cases.size(); ++a)
            for (size_t b = a + 1; b < cases.size(); ++b)
                distinct_history_batch_checks(weights, {test_backend(device)}, infer::Placement{},
                                              cases[a], cases[b], model.n_vocab(),
                                              field(cases[a], "name").asString() + " + " +
                                                  field(cases[b], "name").asString() + " distinct batch");

        size_t count = 0, case_index = 0;
        for (const Json& item : cases) {
            const bool first_case = case_index++ == 0;
            const std::string name = field(item, "name").asString();
            std::cout << "case " << name << ": full/sliced/cached HF logits" << std::endl;
            const std::vector<uint32_t> prompt = ids(field(item, "input_ids"));
            const auto hf_full = floats(field(item, "full_logits"));
            const auto hf_cached = floats(field(item, "cached_logits"));
            const auto hf_sliced = floats(field(item, "sliced_logits"));

            if (first_case) loader_checks(gguf_path, device, prompt, hf_full);

            row_source_checks(weights, file, device, prompt, hf_full, hf_sliced, hf_cached,
                              field(root, "slice_lengths"), item, model.n_vocab(), name, first_case);

            model.set_ubatch(int(prompt.size()));
            compare(score(model, prompt), hf_full, name + " full prompt");
            compare(score_slices(model, prompt, field(root, "slice_lengths")), hf_sliced,
                    name + " sliced prompt");
            compare(score(model, prompt), hf_cached, name + " cached prompt");
            std::cout << "case " << name << ": mark/retract, batch and scratch" << std::endl;
            history_checks(weights, test_backend(device), prompt, hf_full, model.n_vocab(), name);
            if (first_case)
                edited_suffix_retract_checks(weights, test_backend(device), prompt, model.n_vocab(), name);
            aborted_stage_rollback_checks(weights, {test_backend(device)}, infer::Placement{},
                                           prompt, hf_full, model.n_vocab(), name);
            batch_and_scratch_checks(weights, test_backend(device), prompt, hf_full, model.n_vocab(), name);
            if (device == "cpu") {
                std::cout << "case " << name << ": resident/CPU experts/whole-stack streaming" << std::endl;
                expert_placement_checks(weights, root, item);
                std::cout << "case " << name << ": two-CPU split" << std::endl;
                cpu_layer_split_checks(weights, prompt, hf_full, model.n_vocab(), name);
                continued_split_checks(weights, prompt, hf_sliced, field(root, "slice_lengths"), item, model.n_vocab(), name);
                std::cout << "case " << name << ": failure after GDN write, marked rollback" << std::endl;
                failed_marked_delta_rollback_checks(weights, prompt, hf_full, model.n_vocab(), name);
            }
            std::cout << "case " << name << ": keep/fork parity" << std::endl;
            keep_fork_parity(weights, file, infer::qwen4exp::read_config(file, "qwen4exp."),
                             {test_backend(device)}, infer::Placement{}, prompt, name);
            if (first_case) {
                std::cout << "case " << name << ": host snapshot/restore through PLE and QSA" << std::endl;
                host_snapshot_restore_parity(weights, infer::qwen4exp::read_config(file, "qwen4exp."),
                                             {test_backend(device)}, infer::Placement{}, prompt, hf_full,
                                             model.n_vocab(), name);
            }

            const Json& decode = field(item, "decode_token");
            if (!decode.isNumber()) continue;
            const double decode_id = decode.asNumber();
            if (!std::isfinite(decode_id) || decode_id < 0 || decode_id > UINT32_MAX || std::floor(decode_id) != decode_id)
                throw std::runtime_error(name + ": invalid decode token");
            const uint32_t token = uint32_t(decode_id);
            std::cout << "case " << name << ": generated-token follow-up" << std::endl;
            const float* got = model.step(prompt.data(), prompt.size());
            std::vector<float> cached(got, got + prompt.size() * model.n_vocab());
            compare(cached, hf_cached, name + " decode-path prompt");
            got = model.step(&token, 1);
            compare(std::vector<float>(got, got + model.n_vocab()), floats(field(item, "decode_logits")),
                    name + " follow-up");
            ++count;
        }
        if (count != 2) throw std::runtime_error("expected two follow-up decode cases");
        std::cout << "qwen4exp model: " << field(root, "cases").asArray().size()
                  << " HF cases passed on " << device << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
