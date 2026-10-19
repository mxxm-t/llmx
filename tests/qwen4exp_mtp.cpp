// An assembled scalar MTP reference, not official HF MTP parity: HF omits mtp.*.
// Equations follow the pinned HF HC/attention/MoE components and the pinned converter's MTP wrapper.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <future>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "backends/devices.hpp"
#include "core/list.hpp"
#include "model/arch/qwen4exp.hpp"
#include "model/runtime.hpp"
#include "inference/load.hpp"
#include "server/disk_tier.hpp"
#include "qwen4exp_mtp_fixture.hpp"
#include "tiny_qwen.hpp"

namespace {
using Vec = std::vector<double>;
using flash_mtp::E; using flash_mtp::H; using flash_mtp::R; using flash_mtp::V;
using flash_mtp::Fixture;
using flash_mtp::integer; using flash_mtp::string; using flash_mtp::array;
size_t checks = 0;
void require(bool ok, const std::string& label) {
    if (!ok) throw std::runtime_error(label);
    ++checks;
}
template<class F> void refuses(F&& run, const std::string& text) {
    try { run(); } catch (const std::exception& e) {
        require(std::string(e.what()).find(text) != std::string::npos, "wrong refusal: " + std::string(e.what())); return;
    }
    throw std::runtime_error("missing refusal: " + text);
}

// This oracle uses scalar double arithmetic and never calls a production graph or backend helper.
struct Oracle {
    const Fixture& f;
    std::vector<Vec> keys, values, index;
    bool auxiliary_changes_selection = false;
    explicit Oracle(const Fixture& fixture) : f(fixture) {}
    const std::vector<float>& w(const std::string& name) const { return f.data.at(name + ".weight"); }
    static double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
    Vec product(const std::string& name, const Vec& x, size_t offset = 0, size_t outputs = 0) const {
        const auto& a = w(name);
        const size_t n = outputs ? outputs : a.size() / x.size();
        Vec y(n);
        for (size_t r = 0; r < n; ++r) for (size_t k = 0; k < x.size(); ++k) y[r] += a[offset + r * x.size() + k] * x[k];
        return y;
    }
    Vec norm(const std::string& name, const Vec& x, size_t width) const {
        Vec out(x.size()); const auto& gamma = w(name);
        for (size_t start = 0; start < x.size(); start += width) {
            double sum = 0;
            for (size_t i = 0; i < width; ++i) sum += x[start + i] * x[start + i];
            const double scale = 1.0 / std::sqrt(sum / width + f.c.rms_eps);
            for (size_t i = 0; i < width; ++i) out[start + i] = x[start + i] * scale * gamma[(start + i) % gamma.size()];
        }
        return out;
    }
    Vec mix(const std::string& stem, const Vec& x, Vec* injection = nullptr) const {
        Vec n = norm(stem + "_norm", x, E), low = product(stem + "_down", n);
        for (double& a : low) a = (a / H) * sigmoid(a / H);
        Vec gate = product(stem + "_up", low), out(E);
        for (size_t i = 0; i < R; ++i) out[i % E] += n[i] * sigmoid(gate[i]) / H;
        if (injection) {
            *injection = product(stem + "_inject", n);
            for (double& a : *injection) a = 2 * sigmoid(a / H);
        }
        return out;
    }
    Vec input(uint32_t token, const Vec& hidden) const {
        const auto& table = w("token_embd");
        Vec e(table.begin() + token * E, table.begin() + (token + 1) * E);
        e = norm("blk.2.nextn.enorm", e, E);
        const Vec h = norm("blk.2.nextn.hnorm", hidden, E);
        Vec result;
        const bool split = f.data.count("blk.2.nextn.fc_hidden.weight") != 0;
        for (size_t stream = 0; stream < H; ++stream) {
            Vec part(h.begin() + stream * E, h.begin() + (stream + 1) * E), y;
            if (split) {
                y = product("blk.2.nextn.eh_proj", e);
                const Vec z = product("blk.2.nextn.fc_hidden", part);
                for (size_t i = 0; i < E; ++i) y[i] += z[i];
            } else {
                Vec joined = e; joined.insert(joined.end(), part.begin(), part.end());
                y = product("blk.2.nextn.eh_proj", joined);
            }
            result.insert(result.end(), y.begin(), y.end());
        }
        return result;
    }
    void rotate(Vec& x, size_t position) const {
        const double a = x[0], b = x[1], cs = std::cos(double(position)), sn = std::sin(double(position));
        x[0] = a * cs - b * sn; x[1] = a * sn + b * cs;
    }
    void cache(const Vec& mixed) {
        Vec k = norm("blk.2.attn_k_norm", product("blk.2.attn_k", mixed), E);
        rotate(k, keys.size()); keys.push_back(k);
        values.push_back(product("blk.2.attn_v", mixed));
        if (f.c.nextn_compress_ratio) index.push_back(product("blk.2.indexer.k_proj", mixed));
    }
    std::vector<size_t> selected(const Vec& mixed) const {
        const size_t length = keys.size(), ratio = size_t(f.c.nextn_compress_ratio);
        std::vector<size_t> ids;
        if (!ratio) { for (size_t i = 0; i < length; ++i) ids.push_back(i); return ids; }
        Vec q = norm("blk.2.indexer.q_norm", product("blk.2.indexer.q_proj", mixed), E);
        rotate(q, length - 1);
        std::vector<std::pair<double, size_t>> scores;
        for (size_t pool = 0; pool < length / ratio; ++pool) {
            Vec k(E);
            for (size_t j = 0; j < ratio; ++j) for (size_t i = 0; i < E; ++i) k[i] += index[pool * ratio + j][i] / ratio;
            k = norm("blk.2.indexer.k_norm", k, E); rotate(k, pool * ratio);
            double score = 0; for (size_t i = 0; i < E; ++i) score += q[i] * k[i];
            scores.push_back({std::max(score, 0.0) / std::sqrt(double(E)), pool});
        }
        std::sort(scores.begin(), scores.end(), [](auto a, auto b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
        scores.resize(std::min(scores.size(), size_t(f.c.indexer_budget) / ratio));
        for (auto score : scores) for (size_t j = 0; j < ratio; ++j) ids.push_back(score.second * ratio + j);
        for (size_t i = length / ratio * ratio; i < length; ++i) ids.push_back(i);
        std::sort(ids.begin(), ids.end());
        return ids;
    }
    Vec ffn(const Vec& mixed) const {
        Vec score = product("blk.2.ffn_gate_inp", mixed);
        const size_t expert = size_t(std::max_element(score.begin(), score.end()) - score.begin());
        auto swiglu = [&](const std::string& gate, const std::string& up, const std::string& down, size_t expert_id) {
            Vec g = product(gate, mixed, expert_id * E * E, E), u = product(up, mixed, expert_id * E * E, E);
            for (size_t i = 0; i < E; ++i) g[i] = g[i] * sigmoid(g[i]) * u[i];
            return product(down, g, expert_id * E * E, E);
        };
        Vec out = swiglu("blk.2.ffn_gate_exps", "blk.2.ffn_up_exps", "blk.2.ffn_down_exps", expert);
        const Vec shared = swiglu("blk.2.ffn_gate_shexp", "blk.2.ffn_up_shexp", "blk.2.ffn_down_shexp", 0);
        const double gate = sigmoid(product("blk.2.ffn_gate_inp_shexp", mixed)[0]);
        for (size_t i = 0; i < E; ++i) out[i] += gate * shared[i];
        return out;
    }
    Vec draft(uint32_t token, Vec& hidden) {
        Vec x = input(token, hidden), injection, mixed = mix("blk.2.hc_attn", x, &injection);
        cache(mixed);
        Vec raw = product("blk.2.attn_q", mixed), q(raw.begin(), raw.begin() + E);
        q = norm("blk.2.attn_q_norm", q, E); rotate(q, keys.size() - 1);
        const auto rows = selected(mixed);
        if (f.c.nextn_compress_ratio) {
            const auto saved = index;
            for (Vec& key : index) std::fill(key.begin(), key.end(), 0.0);
            auxiliary_changes_selection = auxiliary_changes_selection || rows != selected(mixed);
            index = saved;
        }
        Vec scores; double peak = -1e100;
        for (size_t row : rows) {
            double dot = 0; for (size_t i = 0; i < E; ++i) dot += q[i] * keys[row][i];
            scores.push_back(dot / std::sqrt(double(E))); peak = std::max(peak, scores.back());
        }
        double sum = 0; for (double& a : scores) { a = std::exp(a - peak); sum += a; }
        Vec attention(E);
        for (size_t j = 0; j < rows.size(); ++j) for (size_t i = 0; i < E; ++i) attention[i] += scores[j] / sum * values[rows[j]][i];
        for (size_t i = 0; i < E; ++i) attention[i] *= sigmoid(raw[E + i]);
        attention = product("blk.2.attn_output", attention);
        for (size_t i = 0; i < R; ++i) x[i] += injection[i / E] * attention[i % E];
        mixed = mix("blk.2.hc_ffn", x, &injection);
        const Vec feed = ffn(mixed);
        for (size_t i = 0; i < R; ++i) x[i] += injection[i / E] * feed[i % E];
        hidden = x;
        return product("output", mix("blk.2.nextn.hc_head", x));
    }
};

backend::BackendPtr cpu() { auto b = backend::make_cpu_backend(); b->set_threads(1); return b; }
infer::ModelOptions options() {
    infer::ModelOptions o; o.dtype = backend::Dtype::f32; o.kv_k = o.kv_v = backend::KVType::f32;
    o.state_slots = 3; o.mark_slots = 1; o.mark_rows = 8; o.kv_tokens = 256; return o;
}
std::unique_ptr<infer::Model> model(Fixture& f, bool drafts, const std::string& device = "cpu") {
    auto b = backend::make_backend(device);
    if (b->is_cpu()) b->set_threads(1);
    return std::make_unique<infer::Model>(f.weights, infer::plan_model(f.weights, drafts), std::vector<backend::BackendPtr>{b}, infer::Placement{}, options());
}
std::unique_ptr<infer::Model> group_model(Fixture& f, bool with_drafts,
                                         const std::vector<std::string>& devices, size_t width) {
    std::vector<backend::BackendPtr> members;
    for (const std::string& name : devices) {
        auto member = backend::make_backend(name);
        if (member->is_cpu()) member->set_threads(1);
        members.push_back(std::move(member));
    }
    infer::Placement placement;
    placement.width = width;
    if (devices.size() == 4 && width == 2) {
        placement.mixer_device = placement.ffn_device = {0, 2};
        placement.output_device = 2;
    }
    return std::make_unique<infer::Model>(f.weights, infer::plan_model(f.weights, with_drafts), std::move(members), placement, options());
}
std::vector<float> feed(infer::Model& m, infer::Sequence& s, const std::vector<uint32_t>& ids, size_t extent) {
    infer::ExecContext ctx; infer::BatchEntry entry{&s, ids.data(), ids.size(), true};
    entry.extent = extent; entry.every_logits = true; m.forward(ctx, &entry, 1);
    return {ctx.logits(0), ctx.logits(0) + ids.size() * V};
}
using Chain = std::pair<std::vector<uint32_t>, std::vector<float>>;
Chain chain(infer::Model& m, infer::Sequence& s, uint32_t last, size_t depth) {
    Chain out;
    m.draft(s, last, depth, out.first); require(out.first.size() == depth, "finite draft chain");
    for (size_t i = 0; i < depth; ++i) out.second.insert(out.second.end(), m.draft_logits(i), m.draft_logits(i) + V);
    return out;
}
std::vector<float> drafts(infer::Model& m, infer::Sequence& s, uint32_t last) {
    return chain(m, s, last, 2).second;
}
void same(const std::vector<float>& a, const std::vector<float>& b, const std::string& what) {
    require(a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), what);
}
void batched_chains(infer::Model& m, infer::Sequence& short_sequence, infer::Sequence& long_sequence) {
    const auto short_chain = chain(m, short_sequence, 7, 1), long_chain = chain(m, long_sequence, 2, 2);
    std::vector<uint32_t> short_ids, long_ids;
    infer::Model::DraftAsk asks[2]{{&short_sequence, 7, 1, &short_ids}, {&long_sequence, 2, 2, &long_ids}};
    m.draft(asks, 2);
    require(short_ids == short_chain.first && long_ids == long_chain.first, "unequal batched draft IDs in reversed length order");
    std::vector<float> short_logits(m.draft_logits(0, 0), m.draft_logits(0, 0) + V), long_logits;
    for (size_t i = 0; i < 2; ++i) long_logits.insert(long_logits.end(), m.draft_logits(i, 1), m.draft_logits(i, 1) + V);
    same(short_logits, short_chain.second, "short batched draft logits");
    same(long_logits, long_chain.second, "long batched draft logits after batch shrinks");
}
double within(const std::vector<float>& actual, const std::vector<float>& expected, const std::string& what) {
    require(actual.size() == expected.size(), what + " dimensions");
    double error = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) throw std::runtime_error(what + " nonfinite output");
        error = std::max(error, std::fabs(double(actual[i]) - expected[i]));
    }
    require(error < 2e-5, what + " max_abs=" + std::to_string(error));
    return error;
}

// Exercise the packer's block-only contract and released private/shared formats through the public loader.
void paired_loader(Fixture& f, const std::vector<uint32_t>& prompt, const std::vector<float>& expected, const std::vector<float>& expected_drafts, int layout) {
    auto [target, side] = flash_mtp::headers(f, layout);
    std::map<std::string, std::vector<float>> private_data;
    infer::ModelWeights private_weights = f.weights;
    std::vector<float> draft_expected = expected_drafts;
    if (layout == 2) {
        private_data = flash_mtp::private_globals(f, side);
        for (const auto& [name, values] : private_data)
            private_weights.tensors.push_back({"blk.2.nextn." + name, {E, V}, quant::GGML_TYPE_F32,
                                               reinterpret_cast<const uint8_t*>(values.data()), values.size() * sizeof(float)});
        infer::Model control(private_weights, infer::plan_model(private_weights, true), {cpu()}, {}, options());
        auto sequence = control.make_sequence();
        same(feed(control, sequence, prompt, prompt.size()), expected, "private globals leave target logits unchanged");
        draft_expected = drafts(control, sequence, 2);
        require(draft_expected != expected_drafts, "different private globals change drafts");
    }
    flash_mtp::Files files(f, std::move(target), std::move(side), private_data);
    const auto& model_path = files.target_path;
    const auto& draft_path = files.side_path;
    // The pairing adapter changes headers only; even private renaming preserves every physical span and unmapped owner.
    auto before = gguf::read_gguf(draft_path);
    const auto offsets = before.offsets;
    const size_t payload = before.payload_size();
    std::vector<format::FileSpan> spans;
    for (size_t i = 0; i < before.tensors.size(); ++i) spans.push_back(before.span(i));
    auto target_header = gguf::read_gguf(model_path);
    if (layout) {
        auto incompatible = gguf::read_gguf(draft_path);
        for (auto& item : incompatible.kv) if (item.first == "qwen4exp.embedding_length") item.second = integer(E + 1);
        refuses([&] { infer::spec::pair(target_header, model_path, incompatible, draft_path); }, "incompatible metadata qwen4exp.embedding_length");
        require(incompatible.tensors.front().name == before.tensors.front().name && incompatible.find("qwen4exp.nextn_predict_layers"), "failed normalization preserves headers");
        auto unrelated = gguf::read_gguf(model_path);
        require(infer::spec::pair(target_header, model_path, unrelated, model_path) == infer::spec::DrafterKind::model, "ordinary same-architecture draft model remains a model");
        auto complete = gguf::read_gguf(model_path), tail = gguf::read_gguf(draft_path);
        (void)infer::spec::pair(complete, model_path, tail, draft_path);
        infer::spec::join_blocks(complete, std::move(tail));
        require(infer::spec::pair(target_header, model_path, complete, model_path) == infer::spec::DrafterKind::model,
                "ordinary same-architecture model with embedded MTP remains a model");
        if (layout == 2) {
            auto wrong_tokenizer = gguf::read_gguf(draft_path);
            wrong_tokenizer.tensors.back().ne[1]++;
            for (auto& item : wrong_tokenizer.kv) if (item.first == "tokenizer.ggml.tokens") item.second.arr[0] = string("changed");
            refuses([&] { infer::spec::pair(target_header, model_path, wrong_tokenizer, draft_path); }, "tokenizer.ggml.tokens is");
            require(wrong_tokenizer.tensors.back().name == "output.weight", "tokenizer refusal precedes normalization");
            auto bad_shape = gguf::read_gguf(draft_path);
            bad_shape.tensors.back().ne[1]++;
            refuses([&] { infer::spec::pair(target_header, model_path, bad_shape, draft_path); }, "invalid private global shape output.weight");
            require(bad_shape.tensors.back().name == "output.weight", "failed shape check does not rename private globals");
            auto bad_role = gguf::read_gguf(draft_path);
            bad_role.tensors.front().name = "unexpected.weight";
            refuses([&] { infer::spec::pair(target_header, model_path, bad_role, draft_path); }, "unsupported tensor unexpected.weight");
            auto bad_ratio = gguf::read_gguf(draft_path);
            for (auto& item : bad_ratio.kv) if (item.first == "qwen4exp.attention.compress_ratios") item.second = array({0, 2, 1});
            refuses([&] { infer::spec::pair(target_header, model_path, bad_ratio, draft_path); }, "MTP compression ratio must be zero or greater than one");
            require(bad_ratio.tensors.back().name == "output.weight", "invalid geometry does not rename private globals");
        } else {
            // A valid target PLE description is absent from the released MTP-only header and restored without reading its table.
            auto ple_target = gguf::read_gguf(model_path), ple_side = gguf::read_gguf(draft_path);
            ple_target.kv.push_back({"qwen4exp.ple.layers", array({0})});
            for (const auto& item : std::vector<std::pair<std::string, uint64_t>>{{"ngram_size", 2}, {"heads_per_ngram", 1}, {"conv_kernel", 4}, {"eos_token_id", 7}})
                ple_target.kv.push_back({"qwen4exp.ple." + item.first, integer(item.second)});
            ple_target.kv.push_back({"qwen4exp.embedding_length_per_layer_input", integer(E)});
            for (const auto& item : std::vector<std::pair<std::string, std::vector<uint64_t>>>{{"layer_multipliers", {1, 3}}, {"head_offsets", {0}}, {"head_vocab_sizes", {17}}}) {
                auto values = array(item.second); values.u = gguf::V_UINT64;
                for (auto& value : values.arr) value.vtype = gguf::V_UINT64;
                ple_target.kv.push_back({"qwen4exp.ple." + item.first, values});
            }
            require(infer::spec::pair(ple_target, model_path, ple_side, draft_path) == infer::spec::DrafterKind::mtp, "omitted PLE metadata restored from target");
            require(ple_side.find("qwen4exp.ple.layers") && ple_side.offsets == offsets, "PLE normalization keeps storage spans");
        }
    }
    require(infer::spec::pair(target_header, model_path, before, draft_path) == infer::spec::DrafterKind::mtp, "pair normalized layout");
    require(before.offsets == offsets && before.payload_size() == payload, "normalization preserves storage offsets and extent");
    for (size_t i = 0; i < spans.size(); ++i) {
        const auto after = before.span(i);
        require(after.file == spans[i].file && after.offset == spans[i].offset && after.bytes == spans[i].bytes,
                "normalization preserves file span");
    }
    infer::PlacementRequest request;
    request.dtype = backend::Dtype::f32;
    auto loaded = infer::load_model(model_path, {cpu()}, request, options(), {}, infer::LoadMode::mapped, draft_path);
    require(loaded->drafter == infer::spec::DrafterKind::mtp, "paired loader selects MTP");
    auto sequence = loaded->model->make_sequence();
    same(feed(*loaded->model, sequence, prompt, prompt.size()), expected, "paired load target logits");
    same(drafts(*loaded->model, sequence, 2), draft_expected, "paired load drafts");
}

void qwen35_pairing() {
    gguf::GGUFModel target, side;
    target.kv = {{"general.architecture", string("qwen35")}, {"qwen35.block_count", integer(2)}, {"qwen35.embedding_length", integer(4)}};
    side.kv = {{"general.architecture", string("qwen35")}, {"qwen35.block_count", integer(3)}, {"qwen35.nextn_predict_layers", integer(1)},
               {"qwen35.embedding_length", integer(4)}};
    gguf::TensorInfo tensor; tensor.name = "blk.2.nextn.eh_proj.weight"; tensor.ne = {8, 4}; tensor.type = quant::GGML_TYPE_F32;
    side.tensors.push_back(tensor);
    require(infer::spec::pair(target, "target", side, "side") == infer::spec::DrafterKind::mtp, "qwen35 pairing unchanged");
    side.kv.back().second = integer(5);
    refuses([&] { infer::spec::pair(target, "target", side, "side"); },
            "inference: the drafter side does not pair with target: qwen35.embedding_length is 5 where the model has 4");
}

// A streamed Flash image must restore the same history as the established whole-host path before follow-up decoding, drafting and a partially accepted verify.
void streamed_host_history(const std::string& device, const std::vector<std::string>& group_devices = {}, size_t width = 1) {
    Fixture fixture(false, 2, !group_devices.empty(), true, 2, width);
    // Keep the tiny random fixture finite across a whole 128-token block, beyond its short-chain oracle inputs.
    for (auto& weight : fixture.data) for (float& value : weight.second) value *= 0.25f;
    fixture.c.context_length = 256;
    fixture.observed->base = std::make_shared<infer::qwen4exp::Qwen4Exp>(fixture.c, uint32_t(V));
    auto settings = options();
    settings.checkpoint_slots = 2;
    settings.kv_tokens = 512;
    std::vector<backend::BackendPtr> backends;
    std::vector<backend::Backend*> destinations;
    for (const auto& name : group_devices.empty() ? std::vector<std::string>{device} : group_devices) {
        auto member = backend::make_backend(name);
        if (member->is_cpu()) member->set_threads(1);
        destinations.push_back(member.get());
        backends.push_back(std::move(member));
    }
    infer::Placement placement;
    placement.width = width;
    if (backends.size() == 4 && width == 2) {
        placement.mixer_device = placement.ffn_device = {0, 2};
        placement.output_device = 2;
    }
    infer::Model loaded(fixture.weights, infer::plan_model(fixture.weights, true), std::move(backends), placement, settings);
    const size_t length = loaded.kv_block_tokens();
    require(length >= 8 && length + 7 <= size_t(fixture.c.context_length), "Flash streamed restore fixture fits one whole backend KV block and its continuation");
    std::vector<uint32_t> prompt(length);
    for (size_t i = 0; i < length; ++i) prompt[i] = uint32_t(1 + i % (V - 1));
    auto source = loaded.make_sequence();
    const auto prefix_logits = feed(loaded, source, prompt, prompt.size());
    require(std::all_of(prefix_logits.begin(), prefix_logits.end(), [](float value) { return std::isfinite(value); }), "Flash restore fixture has finite prefix logits");
    require(loaded.keep(source), "Flash streamed restore keeps its source checkpoint");
    infer::HostHistory saved;
    loaded.save_host(source, length, saved, SIZE_MAX);
    loaded.wait_host(saved);
    const auto continuation = [&](infer::Sequence& sequence) {
        std::vector<float> result = feed(loaded, sequence, {2, 6, 1}, 1);
        const auto append = [&](const std::vector<float>& values) { result.insert(result.end(), values.begin(), values.end()); };
        append(drafts(loaded, sequence, 5));
        require(loaded.mark(sequence), "Flash restored history marks a verify");
        (void)feed(loaded, sequence, {5, 4, 7, 2}, 1);
        require(loaded.retract(sequence, length + 5) == length + 5, "Flash restored history retracts a partially accepted verify");
        append(feed(loaded, sequence, {3}, 1));
        append(drafts(loaded, sequence, 7));
        return result;
    };
    const auto expected = continuation(source);
    loaded.reset(source);
    auto whole = loaded.restore_host(saved);
    same(continuation(whole), expected, "Flash whole-host restore control preserves follow-up, draft and rollback logits");
    loaded.reset(whole);
    auto streamed = loaded.restore_into(length);
    require(streamed.runs.size() == saved.device_bytes.size(), "Flash streamed image has every saved device");
    for (size_t d = 0; d < streamed.runs.size(); ++d) {
        size_t bytes = 0;
        for (const auto& run : streamed.runs[d]) bytes += run.bytes;
        require(bytes == saved.device_bytes[d], "Flash streamed restore image byte count on device " + std::to_string(d) +
                ": destinations=" + std::to_string(bytes) + ", saved=" + std::to_string(saved.device_bytes[d]));
    }
    copy_image(loaded, saved, streamed, destinations, 1024);
    same(continuation(streamed.seq), expected, "Flash chunked restore preserves follow-up, draft and rollback logits");
    loaded.reset(streamed.seq);
    const auto directory = std::filesystem::temp_directory_path() /
                           ("llmx-flash-stream-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    try {
        server::DiskStore::Options disk;
        disk.root = directory.string();
        server::DiskStore store(disk, {});
        std::vector<server::StoreRun> runs;
        server::StreamedRead::File file;
        file.layout = saved.device_bytes;
        for (size_t d = 0; d < saved.device_bytes.size(); ++d) {
            runs.push_back({saved.slabs[d], saved.device_bytes[d], 0});
            file.segments.push_back({d, 0, saved.device_bytes[d], false});
            file.payload += saved.device_bytes[d];
        }
        auto written = std::make_shared<std::promise<std::string>>();
        auto answer = written->get_future();
        file.key = store.put("Flash restore regression", std::move(runs), infer::Model::host_slab_bytes(),
                             [written](bool ok, const std::string& error) { written->set_value(ok ? "" : error); }, file.layout);
        require(answer.wait_for(std::chrono::seconds(30)) == std::future_status::ready, "Flash image write finishes within its bound");
        require(answer.get().empty(), "Flash image written to disk");
        auto disk_into = loaded.restore_into(length);
        server::StreamedRead reader(store, {file}, disk_into, [] {});
        try {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            auto state = reader.poll();
            while (state == server::StreamedRead::State::running && std::chrono::steady_clock::now() < deadline) {
                if (reader.ready()) reader.send();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                state = reader.poll();
            }
            require(state == server::StreamedRead::State::done, "Flash disk stream completes: " + reader.why());
            require(reader.bytes() == file.payload, "Flash disk stream reads the complete saved image");
            loaded.restored(disk_into, reader.last());
            same(continuation(disk_into.seq), expected, "Flash disk ring restore preserves follow-up, draft and rollback logits");
            loaded.reset(disk_into.seq);
        } catch (...) {
            reader.cancel();
            for (auto* destination : destinations) destination->sync();
            throw;
        }
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }
    std::filesystem::remove_all(directory);
    loaded.release_host(saved);
    std::cout << "streamed restore " << device << " width=" << width << " stages=" << destinations.size() / width
              << " block_tokens=" << length << " devices=" << destinations.size() << " passed\n";
}

void run(bool split, int compression, const std::string& device) {
    Fixture fixture(split, compression);
    auto reference = model(fixture, true);
    auto reference_sequence = reference->make_sequence();
    const std::vector<uint32_t> prompt{1, 4, 2, 6, 3, 5, 1, 7, 4};
    fixture.observed->capture = true;
    const auto reference_target = feed(*reference, reference_sequence, prompt, prompt.size());
    const auto hidden = fixture.observed->hidden;
    fixture.observed->capture = false;
    const auto reference_drafts = drafts(*reference, reference_sequence, 2);
    auto loaded = model(fixture, true, device), plain = model(fixture, false, device);
    auto sequence = loaded->make_sequence(), without = plain->make_sequence();
    const auto target = feed(*loaded, sequence, prompt, prompt.size());
    double target_error = 0;
    for (size_t i = 0; i < target.size(); ++i) target_error = std::max(target_error, std::fabs(double(target[i]) - reference_target[i]));
    require(target_error < 2e-5, "selected backend target agrees with CPU fixture");
    same(target, feed(*plain, without, prompt, prompt.size()), "loaded MTP changes no target logit");
    Oracle oracle(fixture); Vec previous(R);
    for (size_t i = 0; i < prompt.size(); ++i) {
        oracle.cache(oracle.mix("blk.2.hc_attn", oracle.input(prompt[i], previous)));
        previous = hidden[i];
    }
    std::vector<uint32_t> ids; loaded->draft(sequence, 2, 2, ids);
    require(ids.size() == 2, "two-step chain");
    uint32_t token = 2;
    double worst = 0;
    for (size_t step = 0; step < 2; ++step) {
        const Vec expected = oracle.draft(token, previous);
        for (size_t i = 0; i < V; ++i) worst = std::max(worst, std::fabs(expected[i] - loaded->draft_logits(step)[i]));
        const uint32_t pick = uint32_t(std::max_element(expected.begin(), expected.end()) - expected.begin());
        require(ids[step] == pick, "independent oracle draft id"); token = pick;
    }
    require(worst < 2e-5, "independent scalar MTP logits max_abs=" + std::to_string(worst));
    if (compression) require(oracle.auxiliary_changes_selection, "nonzero auxiliary keys affect QSA selection");
    const auto before = drafts(*loaded, sequence, 2);
    for (int layout : {0, 1, 2}) paired_loader(fixture, prompt, reference_target, reference_drafts, layout);
    same(before, drafts(*loaded, sequence, 2), "draft call leaves context unchanged");
    auto sliced = loaded->make_sequence();
    (void)feed(*loaded, sliced, {1, 4, 2}, prompt.size());
    (void)feed(*loaded, sliced, {6, 3}, prompt.size());
    (void)feed(*loaded, sliced, {5, 1, 7, 4}, prompt.size());
    same(before, drafts(*loaded, sliced, 2), "prompt slicing preserves raw carried rows and MTP cache");
    (void)feed(*loaded, sliced, {6}, 1);
    batched_chains(*loaded, sliced, sequence);
    loaded->reset(sliced);
    const std::vector<uint32_t> verify{2, 6, 1, 5};
    require(loaded->mark(sequence), "mark before verify");
    (void)feed(*loaded, sequence, verify, 1);
    require(loaded->retract(sequence, prompt.size() + 2) == prompt.size() + 2, "partial acceptance retract");
    auto fresh = loaded->make_sequence();
    (void)feed(*loaded, fresh, prompt, prompt.size());
    (void)feed(*loaded, fresh, {2}, 1); (void)feed(*loaded, fresh, {6}, 1);
    same(drafts(*loaded, sequence, 7), drafts(*loaded, fresh, 7), "rejected suffix restores raw carry and auxiliary keys");
    same(feed(*loaded, sequence, {7}, 1), feed(*loaded, fresh, {7}, 1), "edited continuation target logits");
    same(drafts(*loaded, sequence, 3), drafts(*loaded, fresh, 3), "edited continuation draft logits");
    std::cout << device << ' ' << (split ? "split" : "fused") << " compression=" << compression << " max_abs=" << worst << '\n';
}

// These split-capable fixtures compare to their own one-backend native control, separately from the scalar-oracle cases above.
void groups(bool split, int compression, const std::vector<std::string>& devices, size_t width) {
    Fixture fixture(split, compression, true, false, 2, width);
    auto reference = model(fixture, true, devices.front()), grouped = group_model(fixture, true, devices, width), plain = group_model(fixture, false, devices, width);
    auto control = reference->make_sequence(), sequence = grouped->make_sequence(), without = plain->make_sequence();
    const std::vector<uint32_t> prompt{1, 4, 2, 6, 3, 5, 1, 7, 4};
    const auto expected_target = feed(*reference, control, prompt, prompt.size());
    const auto target = feed(*grouped, sequence, prompt, prompt.size());
    const double target_error = within(target, expected_target, "group target versus one-backend control");
    same(target, feed(*plain, without, prompt, prompt.size()), "group loaded/off target logits");
    const auto expected_chain = chain(*reference, control, 2, 2), actual_chain = chain(*grouped, sequence, 2, 2);
    require(actual_chain.first == expected_chain.first, "group draft IDs versus one-backend control");
    const double draft_error = within(actual_chain.second, expected_chain.second, "group draft logits versus one-backend control");
    same(actual_chain.second, drafts(*grouped, sequence, 2), "group repeated draft logits");
    auto sliced = grouped->make_sequence();
    (void)feed(*grouped, sliced, {1, 4, 2}, prompt.size());
    (void)feed(*grouped, sliced, {6, 3}, prompt.size());
    (void)feed(*grouped, sliced, {5, 1, 7, 4}, prompt.size());
    same(actual_chain.second, drafts(*grouped, sliced, 2), "group prompt slicing");
    (void)feed(*grouped, sliced, {6}, 1);
    batched_chains(*grouped, sliced, sequence);
    grouped->reset(sliced);
    const std::vector<uint32_t> verify{2, 6, 1, 5};
    for (size_t accepted : {size_t(0), size_t(2), size_t(4)}) {
        grouped->reset(sequence);
        (void)feed(*grouped, sequence, prompt, prompt.size());
        require(grouped->mark(sequence), "group mark before verify");
        (void)feed(*grouped, sequence, verify, 1);
        require(grouped->retract(sequence, prompt.size() + accepted) == prompt.size() + accepted, "group acceptance retract");
        auto fresh = grouped->make_sequence();
        (void)feed(*grouped, fresh, prompt, prompt.size());
        for (size_t i = 0; i < accepted; ++i) (void)feed(*grouped, fresh, {verify[i]}, 1);
        const auto fresh_chain = chain(*grouped, fresh, 7, 2), restored_chain = chain(*grouped, sequence, 7, 2);
        require(fresh_chain.first == restored_chain.first, "group acceptance restores draft IDs");
        same(restored_chain.second, fresh_chain.second, "group acceptance restores draft logits");
        same(feed(*grouped, sequence, {7}, 1), feed(*grouped, fresh, {7}, 1), "group edited target continuation");
        same(drafts(*grouped, sequence, 3), drafts(*grouped, fresh, 3), "group edited draft continuation");
        grouped->reset(fresh);
    }
    for (size_t i = 0; i < devices.size(); ++i) std::cout << (i ? "," : "") << devices[i];
    std::cout << " width=" << width << " stages=" << devices.size() / width << ' ' << (split ? "split" : "fused") << " compression=" << compression
              << " target_native_control_max_abs=" << target_error << " draft_native_control_max_abs=" << draft_error << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        const char* usage = "usage: llmx-qwen4exp-mtp-test [--device cpu|vulkan:N] [--group-devices D0,D1[,D2,D3] [--group-width 2|4]]";
        std::string device = "cpu";
        std::vector<std::string> group_devices;
        size_t group_width = 2;
        for (int i = 1; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--device" && i + 1 < argc) device = argv[++i];
            else if (flag == "--group-devices" && i + 1 < argc) group_devices = core::comma_list(argv[++i]);
            else if (flag == "--group-width" && i + 1 < argc) {
                const std::string value = argv[++i];
                if (value != "2" && value != "4") throw std::runtime_error(usage);
                group_width = size_t(value[0] - '0');
            } else throw std::runtime_error(usage);
        }
        if (!group_devices.empty()) {
            if (group_devices.size() != 2 && group_devices.size() != 4) throw std::runtime_error(usage);
            if (group_devices.size() % group_width != 0) throw std::runtime_error(usage);
            for (size_t i = 0; i < group_devices.size(); ++i) {
                if (group_devices[i].empty()) throw std::runtime_error(usage);
                for (size_t j = 0; j < i; ++j)
                    if (group_devices[i] != "cpu" && group_devices[i] == group_devices[j])
                        throw std::runtime_error("--group-devices requires distinct device names except cpu");
            }
        }
        if (group_devices.empty() && group_width != 2) throw std::runtime_error(usage);
        qwen35_pairing();
        streamed_host_history(device);
        if (group_devices.empty()) {
            streamed_host_history("cpu", {"cpu", "cpu"}, 2);
            streamed_host_history("cpu", {"cpu", "cpu", "cpu", "cpu"}, 2);
            streamed_host_history("cpu", {"cpu", "cpu", "cpu", "cpu"}, 4);
        } else streamed_host_history(device, group_devices, group_width);
        for (bool split : {false, true}) for (int compression : {0, 2, 4}) run(split, compression, device);
        for (bool split : {false, true}) for (int compression : {0, 2, 4}) {
            if (group_devices.empty()) {
                groups(split, compression, {"cpu", "cpu"}, 2);
                groups(split, compression, {"cpu", "cpu", "cpu", "cpu"}, 2);
                groups(split, compression, {"cpu", "cpu", "cpu", "cpu"}, 4);
            } else groups(split, compression, group_devices, group_width);
        }
        std::cout << "qwen4exp-mtp: " << checks << " checks passed\n";
    } catch (const std::exception& e) { std::cerr << "qwen4exp-mtp: " << e.what() << '\n'; return 1; }
}
