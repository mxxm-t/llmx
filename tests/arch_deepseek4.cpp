// The deepseek41 module (model/arch/deepseek4.hpp) on the tiny fork-layout file built in memory from the step 2 golden's arguments and Engram constants.
// Every refused key and tensor with its text, the per-layer table, the plan, the footprint, the Engram norm product, and the refusal of the architecture by name.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/bf16.hpp"
#include "core/json.hpp"
#include "format/gguf.hpp"
#include "model/arch/deepseek4.hpp"
#include "model/arch/registry.hpp"
#include "model/layer_split.hpp"
#include "model/place.hpp"
#include "model/runtime.hpp"
#include "loading_backend.hpp"

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
        require(std::string(e.what()) == text, what + ": refused with \"" + e.what() + "\", not \"" + text + "\"");
        return;
    }
    throw std::runtime_error(what + ": not refused");
}

const std::string P = "deepseek41.";

const jmini::Value& field(const jmini::Value& v, const char* name) {
    const jmini::Value* f = v.get(name);
    if (!f) throw std::runtime_error(std::string("reference: no field ") + name);
    return *f;
}

uint64_t whole(const jmini::Value& v) {
    if (!v.isNumber() || !v.whole || v.num < 0 || v.num >= 9007199254740992.0) throw std::runtime_error("reference: not an exact whole number");
    return uint64_t(v.num);
}

std::vector<uint64_t> wholes(const jmini::Value& v) {
    std::vector<uint64_t> out;
    for (const jmini::Value& x : v.asArray()) out.push_back(whole(x));
    return out;
}

// The golden's arguments and Engram constants, which the Python writer (tests/fixture_deepseek4.py) writes the same file from.
struct Golden {
    std::map<std::string, double> args;
    std::vector<uint64_t> compress_ratios, kv_sources, index_sources, engram_layers;
    std::vector<std::vector<uint64_t>> primes, multipliers;
    std::vector<uint64_t> token_map;
    std::vector<std::string> tokens;
    jmini::Value root;
    uint64_t operator[](const std::string& k) const { return uint64_t(args.at(k)); }
};

Golden golden;

void read_golden(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open " + path);
    std::stringstream text;
    text << file.rdbuf();
    const jmini::Value root = jmini::parse(text.str());
    const jmini::Value& args = field(root, "args");
    for (const char* k : {"vocab_size", "dim", "moe_inter_dim", "n_layers", "n_heads", "n_routed_experts", "n_shared_experts",
                          "n_activated_experts", "route_scale", "swiglu_limit", "q_lora_rank", "head_dim", "rope_head_dim", "norm_eps",
                          "o_groups", "o_lora_rank", "window_size", "compress_rope_theta", "original_seq_len", "rope_theta",
                          "rope_factor", "beta_fast", "beta_slow", "index_n_heads", "index_head_dim", "index_topk",
                          "candidate_source_layer", "candidate_topk_blocks", "candidate_block_size", "hc_mult", "hc_sinkhorn_iters",
                          "hc_eps", "engram_max_ngram_size", "engram_n_heads", "engram_head_dim", "engram_pad_id",
                          "engram_compressed_vocab_size", "max_seq_len"})
        golden.args[k] = field(args, k).num;
    golden.compress_ratios = wholes(field(args, "compress_ratios"));
    golden.kv_sources = wholes(field(args, "kv_source_layers"));
    golden.index_sources = wholes(field(args, "index_source_layers"));
    golden.engram_layers = wholes(field(args, "engram_layer_ids"));
    const jmini::Value& e = field(root, "engram");
    for (const jmini::Value& row : field(e, "primes").asArray()) golden.primes.push_back(wholes(row));
    for (const jmini::Value& row : field(e, "multipliers").asArray()) golden.multipliers.push_back(wholes(row));
    golden.token_map = wholes(field(e, "token_map"));
    for (const jmini::Value& t : field(e, "tokens").asArray()) golden.tokens.push_back(t.asString());
    golden.root = root;
}

gguf::MetaValue scalar(uint32_t vtype, uint64_t n) {
    gguf::MetaValue v;
    v.vtype = vtype;
    if (vtype == gguf::V_INT32) v.i = int64_t(n);
    else if (vtype == gguf::V_BOOL) v.b = n != 0;
    else v.u = n;
    return v;
}

gguf::MetaValue f32(double x) {
    gguf::MetaValue v;
    v.vtype = gguf::V_FLOAT32;
    const float f = float(x);
    std::memcpy(&v.fb, &f, sizeof(f));
    return v;
}

gguf::MetaValue text(const std::string& s) {
    gguf::MetaValue v;
    v.vtype = gguf::V_STRING;
    v.s = s;
    return v;
}

gguf::MetaValue array(uint32_t vtype, const std::vector<uint64_t>& entries) {
    gguf::MetaValue v;
    v.vtype = gguf::V_ARRAY;
    v.u = vtype;
    for (uint64_t n : entries) v.arr.push_back(scalar(vtype, n));
    return v;
}

gguf::MetaValue reals(const std::vector<double>& entries) {
    gguf::MetaValue v;
    v.vtype = gguf::V_ARRAY;
    v.u = gguf::V_FLOAT32;
    for (double x : entries) v.arr.push_back(f32(x));
    return v;
}

void put(gguf::GGUFModel& m, const std::string& key, gguf::MetaValue v) {
    for (auto& kv : m.kv)
        if (kv.first == key) { kv.second = std::move(v); return; }
    m.kv.push_back({key, std::move(v)});
}

void drop(gguf::GGUFModel& m, const std::string& key) {
    m.kv.erase(std::remove_if(m.kv.begin(), m.kv.end(), [&](const auto& kv) { return kv.first == key; }), m.kv.end());
}

void add_tensor(gguf::GGUFModel& m, const std::string& name, std::vector<uint64_t> ne, uint32_t type = quant::GGML_TYPE_F32) {
    gguf::TensorInfo t{name, std::move(ne), type, 0};
    m.tensors.push_back(t);
    m.add_tensor_data(std::vector<uint8_t>(size_t(t.data_size())));
}

void drop_tensor(gguf::GGUFModel& m, const std::string& name) {
    for (size_t i = 0; i < m.tensors.size(); ++i)
        if (m.tensors[i].name == name) {
            m.tensors.erase(m.tensors.begin() + std::ptrdiff_t(i));
            m.offsets.erase(m.offsets.begin() + std::ptrdiff_t(i));
            return;
        }
    throw std::runtime_error("no tensor " + name);
}

bool has(const std::vector<uint64_t>& ids, uint64_t l) { return std::find(ids.begin(), ids.end(), l) != ids.end(); }

// The fork layout's metadata as the writer gives it, golden values and GGUF types alike.
void fork_metadata(gguf::GGUFModel& m) {
    const Golden& g = golden;
    const uint64_t n = g["n_layers"];
    put(m, "general.architecture", text("deepseek41"));
    const auto u32 = [&](const std::string& k, uint64_t v) { put(m, P + k, scalar(gguf::V_UINT32, v)); };
    const auto real = [&](const std::string& k, double v) { put(m, P + k, f32(v)); };
    u32("block_count", n);
    u32("context_length", g["max_seq_len"]);
    u32("embedding_length", g["dim"]);
    u32("attention.head_count", g["n_heads"]);
    u32("attention.head_count_kv", 1);
    put(m, P + "rope.scaling.type", text("yarn"));
    real("rope.scaling.factor", g.args.at("rope_factor"));
    u32("rope.scaling.original_context_length", g["original_seq_len"]);
    real("rope.scaling.yarn_beta_fast", g.args.at("beta_fast"));
    real("rope.scaling.yarn_beta_slow", g.args.at("beta_slow"));
    real("rope.freq_base", g.args.at("rope_theta"));
    real("attention.layer_norm_rms_epsilon", g.args.at("norm_eps"));
    u32("expert_count", g["n_routed_experts"]);
    u32("expert_used_count", g["n_activated_experts"]);
    u32("expert_gating_func", 4);
    u32("attention.key_length", g["head_dim"]);
    u32("attention.value_length", g["head_dim"]);
    u32("rope.dimension_count", g["rope_head_dim"]);
    u32("attention.q_lora_rank", g["q_lora_rank"]);
    u32("attention.sliding_window", g["window_size"]);
    u32("expert_feed_forward_length", g["moe_inter_dim"]);
    u32("expert_shared_count", g["n_shared_experts"]);
    real("expert_weights_scale", g.args.at("route_scale"));
    put(m, P + "expert_weights_norm", scalar(gguf::V_BOOL, 1));
    put(m, P + "swiglu_clamp_exp", reals(std::vector<double>(size_t(n), g.args.at("swiglu_limit"))));
    put(m, P + "swiglu_clamp_shexp", reals(std::vector<double>(size_t(n), g.args.at("swiglu_limit"))));
    u32("attention.indexer.head_count", g["index_n_heads"]);
    u32("attention.indexer.key_length", g["index_head_dim"]);
    u32("attention.indexer.top_k", g["index_topk"]);
    u32("attention.output_group_count", g["o_groups"]);
    u32("attention.output_lora_rank", g["o_lora_rank"]);
    put(m, P + "attention.compress_ratios", array(gguf::V_INT32, g.compress_ratios));
    real("attention.compress_rope_freq_base", g.args.at("compress_rope_theta"));
    u32("hyper_connection.count", g["hc_mult"]);
    u32("hyper_connection.sinkhorn_iterations", g["hc_sinkhorn_iters"]);
    real("hyper_connection.epsilon", g.args.at("hc_eps"));
    u32("hash_layer_count", 0);
    gguf::MetaValue tokens;
    tokens.vtype = gguf::V_ARRAY;
    tokens.u = gguf::V_STRING;
    for (const std::string& t : g.tokens) tokens.arr.push_back(text(t));
    put(m, "tokenizer.ggml.tokens", tokens);
    put(m, "tokenizer.ggml.bos_token_id", scalar(gguf::V_UINT32, 0));
    put(m, "tokenizer.ggml.eos_token_id", scalar(gguf::V_UINT32, 1));
    put(m, "tokenizer.ggml.padding_token_id", scalar(gguf::V_UINT32, 1));
    put(m, "tokenizer.ggml.add_bos_token", scalar(gguf::V_BOOL, 0));
    put(m, "tokenizer.ggml.add_eos_token", scalar(gguf::V_BOOL, 0));
    put(m, P + "attention.kv_source_layer_ids", array(gguf::V_UINT32, g.kv_sources));
    put(m, P + "attention.index_source_layer_ids", array(gguf::V_UINT32, g.index_sources));
    u32("attention.candidate_source_layer_id", g["candidate_source_layer"]);
    u32("attention.candidate_block_size", g["candidate_block_size"]);
    u32("attention.candidate_topk_blocks", g["candidate_topk_blocks"]);
    put(m, P + "engram.layer_ids", array(gguf::V_INT32, g.engram_layers));
    u32("engram.head_dim", g["engram_head_dim"]);
    u32("engram.n_heads", g["engram_n_heads"]);
    u32("engram.max_ngram_size", g["engram_max_ngram_size"]);
    u32("engram.pad_token_id", g["engram_pad_id"]);
    u32("engram.compressed_vocab_size", g["engram_compressed_vocab_size"]);
    std::vector<uint64_t> rows, primes, offsets, multipliers;
    for (const auto& row : g.primes) {
        uint64_t sum = 0;
        for (uint64_t p : row) {
            primes.push_back(p);
            offsets.push_back(sum);
            sum += p;
        }
        rows.push_back(sum);
    }
    for (const auto& row : g.multipliers) multipliers.insert(multipliers.end(), row.begin(), row.end());
    put(m, P + "engram.num_embeddings", array(gguf::V_UINT64, rows));
    put(m, P + "engram.primes", array(gguf::V_UINT64, primes));
    put(m, P + "engram.offsets", array(gguf::V_UINT64, offsets));
    put(m, P + "engram.multipliers", array(gguf::V_UINT64, multipliers));
    put(m, P + "engram.token_map", array(gguf::V_INT32, g.token_map));
}

// The fork layout's tensors, as the 1017 file names them and the writer shapes them; every value is zero, since nothing here reads one.
void fork_tensors(gguf::GGUFModel& m) {
    const Golden& g = golden;
    const uint64_t d = g["dim"], hc = g["hc_mult"], h = g["n_heads"], hd = g["head_dim"], q = g["q_lora_rank"], go = g["o_groups"];
    const uint64_t orank = g["o_lora_rank"], e = g["n_routed_experts"], inter = g["moe_inter_dim"], ih = g["index_n_heads"];
    const uint64_t ihd = g["index_head_dim"], V = g["vocab_size"], mix = (2 + hc) * hc, ehd = g["engram_head_dim"];
    const uint64_t cols = (g["engram_max_ngram_size"] - 1) * g["engram_n_heads"] * ehd;
    add_tensor(m, "token_embd.weight", {d, V});
    add_tensor(m, "output_norm.weight", {d});
    add_tensor(m, "output.weight", {d, V});
    for (uint64_t l = 0; l < g["n_layers"]; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const bool kv = has(g.kv_sources, l), index = has(g.index_sources, l);
        for (const char* n : {"attn_norm", "ffn_norm"}) add_tensor(m, p + n + ".weight", {d});
        for (const char* n : {"hc_attn", "hc_ffn"}) {
            add_tensor(m, p + n + "_fn.weight", {hc * d, mix});
            add_tensor(m, p + n + "_base.weight", {mix});
            add_tensor(m, p + n + "_scale.weight", {3});
        }
        add_tensor(m, p + "attn_sinks.weight", {h});
        add_tensor(m, p + "attn_q_a.weight", {d, q});
        add_tensor(m, p + "attn_q_a_norm.weight", {q});
        add_tensor(m, p + "attn_q_b.weight", {q, h * hd});
        add_tensor(m, p + "attn_kv.weight", {d, hd});
        add_tensor(m, p + "attn_kv_a_norm.weight", {hd});
        add_tensor(m, p + "attn_output_a.weight", {h * hd / go, go * orank});
        add_tensor(m, p + "attn_output_b.weight", {go * orank, d});
        add_tensor(m, p + "ffn_gate_inp.weight", {d, e});
        add_tensor(m, p + "exp_probs_b.bias", {e});
        if (kv) {
            add_tensor(m, p + "attn_compressor_norm.weight", {hd});
            add_tensor(m, p + "attn_compressor_kv.weight", {d, hd});
            if (g.compress_ratios[l] > 1) add_tensor(m, p + "attn_compressor_gate.weight", {d, hd});
        }
        if (index) {
            add_tensor(m, p + "indexer.attn_q_b.weight", {q, ih * ihd});
            add_tensor(m, p + "indexer.proj.weight", {d, ih});
            if (kv) {
                add_tensor(m, p + "indexer_compressor_kv.weight", {hd, ihd});
                add_tensor(m, p + "indexer_compressor_norm.weight", {ihd});
            }
        }
        add_tensor(m, p + "ffn_gate_exps.weight", {d, inter, e});
        add_tensor(m, p + "ffn_down_exps.weight", {inter, d, e});
        add_tensor(m, p + "ffn_up_exps.weight", {d, inter, e});
        add_tensor(m, p + "ffn_gate_shexp.weight", {d, inter});
        add_tensor(m, p + "ffn_down_shexp.weight", {inter, d});
        add_tensor(m, p + "ffn_up_shexp.weight", {d, inter});
        if (has(g.engram_layers, l)) {
            const size_t i = size_t(std::find(g.engram_layers.begin(), g.engram_layers.end(), l) - g.engram_layers.begin());
            uint64_t rows = 0;
            for (uint64_t prime : g.primes[i]) rows += prime;
            add_tensor(m, p + "engram_embed.weight", {ehd, rows}, quant::GGML_TYPE_MXFP4);
            add_tensor(m, p + "engram_wkv.weight", {cols, d * (hc + 1)});
            add_tensor(m, p + "engram_q.weight", {d, hc});
            add_tensor(m, p + "engram_k.weight", {d, hc});
        }
    }
}

gguf::GGUFModel fork(const std::function<void(gguf::GGUFModel&)>& edit = {}) {
    gguf::GGUFModel m;
    fork_metadata(m);
    fork_tensors(m);
    if (edit) edit(m);
    return m;
}

infer::deepseek4::Config config_of(const gguf::GGUFModel& m) { return infer::deepseek4::read_config(m, P, uint32_t(golden["vocab_size"])); }

// The module's plan of a file.
infer::ModelPlan built(const infer::ModelWeights& w) {
    const auto* arch = dynamic_cast<const infer::deepseek4::DeepSeek41*>(w.arch.get());
    require(arch != nullptr, "the registry's deepseek41 entry reads a DeepSeek41");
    return arch->plan(infer::TensorIndex(w.tensors));
}

void plans(const gguf::GGUFModel& m) { built(infer::gguf_weights(m)); }
void reads(const gguf::GGUFModel& m) { infer::gguf_weights(m); }
// A model built on the CPU and one pass of a prompt, which the module refuses by name until its math exists.
void runs(const gguf::GGUFModel& m) {
    infer::Model model(infer::gguf_weights(m));
    model.prefill({1, 2, 3});
}

std::string unsupported(const std::string& key) { return "inference: unsupported metadata " + P + key; }
std::string missing(const std::string& key) { return "inference: missing metadata " + key; }

// Every deepseek41 key removed in turn: refused as missing by name, or taken at the default the format documents.
void removed_keys() {
    const gguf::GGUFModel base = fork();
    const std::map<std::string, std::string> defaults = {
        {P + "attention.value_length", ""},  // the key width
        {P + "hash_layer_count", ""},        // no hash-routed layers
        {P + "attention.candidate_source_layer_id", "inference: candidate metadata without " + P + "attention.candidate_source_layer_id"},
    };
    size_t keys = 0;
    for (const auto& kv : base.kv) {
        if (kv.first.compare(0, P.size(), P)) continue;
        ++keys;
        const gguf::GGUFModel m = fork([&](gguf::GGUFModel& f) { drop(f, kv.first); });
        const auto d = defaults.find(kv.first);
        if (d != defaults.end() && d->second.empty()) {
            plans(m);
            require(true, kv.first + " removed takes its default");
        } else {
            refuses(kv.first + " removed", d != defaults.end() ? d->second : missing(kv.first), [&] { config_of(m); });
        }
    }
    require(keys == 53, "the fork file holds the 1017 file's 53 deepseek41 keys, not " + std::to_string(keys));
    const auto c = config_of(fork([](gguf::GGUFModel& f) {
        for (const char* k : {"attention.candidate_source_layer_id", "attention.candidate_block_size", "attention.candidate_topk_blocks"}) drop(f, P + k);
    }));
    require(c.candidate_source == -1, "no candidate keys turn the candidate blocks off");
}

// Each key the math holds to one value, at another value, and the keys' types.
void held_values() {
    const auto at = [](const std::string& key, gguf::MetaValue v) { return fork([&](gguf::GGUFModel& f) { put(f, P + key, v); }); };
    const auto u32 = [](uint64_t n) { return scalar(gguf::V_UINT32, n); };
    for (const auto& [key, v] : std::vector<std::pair<std::string, gguf::MetaValue>>{
             {"attention.head_count_kv", u32(2)}, {"attention.value_length", u32(16)}, {"expert_gating_func", u32(1)},
             {"hash_layer_count", u32(3)}, {"rope.scaling.type", text("linear")}, {"rope.scaling.type", u32(1)},
             {"expert_weights_norm", scalar(gguf::V_BOOL, 0)}, {"expert_shared_count", u32(2)},
             {"attention.compress_ratios", array(gguf::V_INT32, {0, 0, 2, 2, 2, 1, 1})},
             {"attention.compress_ratios", array(gguf::V_INT32, {0, 0, 2, 2, 2, 1, 1, 1, 0, 1})},
             {"swiglu_clamp_exp", reals({1, 1, 1})}, {"swiglu_clamp_shexp", reals({1, 1, 1, 1, 1, 1, 1, 1, 1})}})
        refuses(key + " at another value", unsupported(key), [&, key = key, v = v] { config_of(at(key, v)); });
    const auto tail = config_of(at("attention.compress_ratios", array(gguf::V_INT32, {0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0})));
    require(tail.compress_ratios.size() == 8, "a tail of zeros past the blocks is taken and left out");

    gguf::MetaValue ints = array(gguf::V_INT32, {1, 1, 1, 1, 1, 1, 1, 1});
    gguf::MetaValue inf = reals(std::vector<double>(8, 1.0));
    const float infinity = INFINITY;
    std::memcpy(&inf.arr[3].fb, &infinity, sizeof(infinity));
    for (const auto& [key, v, why] : std::vector<std::tuple<std::string, gguf::MetaValue, std::string>>{
             {"block_count", text("8"), "invalid integer type "}, {"rope.freq_base", u32(10000), "invalid floating-point type "},
             {"expert_weights_norm", u32(1), "invalid boolean type "}, {"attention.compress_ratios", u32(0), "invalid array type "},
             {"engram.primes", array(gguf::V_INT32, {41}), "invalid uint64 array type "},
             {"swiglu_clamp_exp", ints, "invalid floating-point type "}, {"swiglu_clamp_exp", inf, "invalid float "},
             {"attention.kv_source_layer_ids", u32(2), "invalid array type "},
             {"engram.token_map", array(gguf::V_INT32, {uint64_t(-1)}), "integer outside supported range "},
             {"embedding_length", u32(0), "integer outside supported range "},
             {"rope.scaling.factor", f32(0), "invalid positive float "}})
        refuses(key + " malformed", "inference: " + why + P + key, [&, key = key, v = v] { config_of(at(key, v)); });
}

// Each rule of check_config broken once, through the file where a file can break it and through the configuration otherwise.
void rules() {
    using infer::deepseek4::Config;
    const auto ids = [](const std::string& key, std::vector<uint64_t> v, uint32_t type = gguf::V_UINT32) {
        return [=](gguf::GGUFModel& f) { put(f, P + key, array(type, v)); };
    };
    const auto both = [](std::function<void(gguf::GGUFModel&)> a, std::function<void(gguf::GGUFModel&)> b) {
        return [=](gguf::GGUFModel& f) { a(f); b(f); };
    };
    const auto ratios = [&](std::vector<uint64_t> v) { return ids("attention.compress_ratios", v, gguf::V_INT32); };
    const auto u32 = [](const std::string& key, uint64_t n) { return [=](gguf::GGUFModel& f) { put(f, P + key, scalar(gguf::V_UINT32, n)); }; };
    const std::vector<std::tuple<std::string, std::function<void(gguf::GGUFModel&)>, std::string>> cases = {
        {"KV sources out of order", ids("attention.kv_source_layer_ids", {4, 2, 5}), "the KV source layers must be ascending distinct layers of the model"},
        {"an index source past the blocks", ids("attention.index_source_layer_ids", {2, 4, 5, 8}), "the index source layers must be ascending distinct layers of the model"},
        {"an Engram layer twice", ids("engram.layer_ids", {1, 1}, gguf::V_INT32), "the Engram layers must be ascending distinct layers of the model"},
        {"a KV source without a ratio", both(ids("attention.kv_source_layer_ids", {1, 2, 4, 5}), ids("attention.index_source_layer_ids", {1, 2, 4, 5, 7})),
         "source layer 1 has no compression ratio"},
        {"an index source without a ratio", ids("attention.index_source_layer_ids", {0, 2, 4, 5, 7}), "source layer 0 has no compression ratio"},
        {"a reader before its source", ratios({0, 2, 2, 2, 2, 1, 1, 1}), "compressed layer 1 has no KV source of its ratio at or before it"},
        {"a reader of another ratio", ratios({0, 0, 2, 1, 2, 1, 1, 1}), "compressed layer 3 has no KV source of its ratio at or before it"},
        {"a KV source that is no index source", ids("attention.index_source_layer_ids", {4, 5, 7}), "KV source layer 2 is not an index source"},
        {"a candidate source that is no index source", u32("attention.candidate_source_layer_id", 6), "the candidate source layer is not an index source"},
        {"a candidate source past the blocks", u32("attention.candidate_source_layer_id", 9), "the candidate source layer is not an index source"},
        {"candidates read at another ratio", u32("attention.candidate_source_layer_id", 4), "index source layer 5 reads candidates of another ratio"},
        {"more experts a token than a layer has", u32("expert_used_count", 5), "more experts per token than the layer has"},
        {"an odd rotary width", u32("rope.dimension_count", 7), "invalid rotary width, which must be even and at most the head and indexer head widths"},
        {"a rotary width past the indexer head", u32("attention.indexer.key_length", 4), "invalid rotary width, which must be even and at most the head and indexer head widths"},
        {"output groups that do not divide the heads", u32("attention.output_group_count", 3), "the output groups do not divide the attention heads' width"},
        {"an Engram n-gram of one token", u32("engram.max_ngram_size", 1), "invalid Engram dimensions"},
        {"a pad id past the vocabulary", u32("engram.pad_token_id", 128), "the Engram pad id is outside the vocabulary"},
        {"one table size for two layers", ids("engram.num_embeddings", {304}, gguf::V_UINT64), "Engram array lengths do not match its layers and heads"},
        {"a prime short", ids("engram.primes", std::vector<uint64_t>(11, 41), gguf::V_UINT64), "Engram array lengths do not match its layers and heads"},
        {"an offset short", ids("engram.offsets", std::vector<uint64_t>(11, 0), gguf::V_UINT64), "Engram array lengths do not match its layers and heads"},
        {"a multiplier short", ids("engram.multipliers", std::vector<uint64_t>(7, 1), gguf::V_UINT64), "Engram array lengths do not match its layers and heads"},
        {"offsets not from 0", ids("engram.offsets", {1, 41, 84, 131, 184, 243, 0, 67, 138, 211, 290, 373}, gguf::V_UINT64), "an Engram layer's offsets do not start at 0"},
        {"a head past its table", ids("engram.num_embeddings", {303, 462}, gguf::V_UINT64), "an Engram head's rows lie outside its table"},
        {"an empty table", ids("engram.num_embeddings", {0, 462}, gguf::V_UINT64), "invalid Engram table rows"},
        {"a token map short", [](gguf::GGUFModel& f) { put(f, P + "engram.token_map", array(gguf::V_INT32, std::vector<uint64_t>(golden.token_map.begin(), golden.token_map.end() - 1))); },
         "the Engram token map does not cover the vocabulary"},
        {"a token map id past the compressed ids", [](gguf::GGUFModel& f) {
             std::vector<uint64_t> map = golden.token_map;
             map[7] = golden["engram_compressed_vocab_size"];
             put(f, P + "engram.token_map", array(gguf::V_INT32, map));
         }, "an Engram token map id is outside the compressed vocabulary"},
        {"a negative SwiGLU clamp", [](gguf::GGUFModel& f) { put(f, P + "swiglu_clamp_shexp", reals({1, 1, 1, 1, -1, 1, 1, 1})); },
         "invalid compression ratio or SwiGLU clamp of layer 4"},
    };
    for (const auto& [what, edit, why] : cases) refuses(what, "inference: " + why, [&, edit = edit] { config_of(fork(edit)); });
    refuses("a multiplier whose products leave int64", "inference: n-gram token hash product exceeds int64 range", [] {
        std::vector<uint64_t> m;
        for (const auto& row : golden.multipliers) m.insert(m.end(), row.begin(), row.end());
        m[5] = uint64_t(1) << 62;
        reads(fork([&](gguf::GGUFModel& f) { put(f, P + "engram.multipliers", array(gguf::V_UINT64, m)); }));
    });
    // Rules a GGUF reader cannot reach, since its integer keys are positive, but another format's reader may.
    const Config good = config_of(fork());
    for (const auto& [what, edit, why] : std::vector<std::tuple<std::string, std::function<void(Config&)>, std::string>>{
             {"a candidate block size of 0", [](Config& c) { c.candidate_block = 0; }, "candidate blocks need a positive block size and count"},
             {"candidate blocks without a source", [](Config& c) { c.candidate_source = -1; }, "candidate blocks without a candidate source layer"},
             {"no blocks", [](Config& c) { c.n_layer = 0; }, "a model without decoder layers"},
             {"no indexer heads", [](Config& c) { c.index_heads = 0; }, "invalid DeepSeek-V4.1 dimensions"},
             {"a NaN epsilon", [](Config& c) { c.rms_eps = NAN; }, "invalid DeepSeek-V4.1 floating-point configuration"},
             {"ratios of another length", [](Config& c) { c.compress_ratios.pop_back(); }, "invalid DeepSeek-V4.1 layer metadata length"},
             {"a context past an allocation", [](Config& c) { c.context_length = 2147483647; c.rope_dim = 8; c.hc_count = 1 << 20; c.n_embd = 1 << 12; },
              "DeepSeek-V4.1 dimensions or context exceed allocation limit"}}) {
        Config c = good;
        edit(c);
        refuses(what, "inference: " + why, [&] { infer::deepseek4::check_config(c); });
    }
}

// A tensor of a kind its layer does not read, and a role without its tensor.
void tensors() {
    const auto added = [](const std::string& name, std::vector<uint64_t> ne) { return [=](gguf::GGUFModel& f) { add_tensor(f, name, ne); }; };
    const std::string kind = "inference: a tensor its layer's kind does not read ";
    for (const auto& [what, edit, why] : std::vector<std::tuple<std::string, std::function<void(gguf::GGUFModel&)>, std::string>>{
             {"a compressor on a ratio-0 layer", added("blk.0.attn_compressor_kv.weight", {64, 32}), kind + "blk.0.attn_compressor_kv.weight"},
             {"a compressor's gate on a ratio-1 source", added("blk.5.attn_compressor_gate.weight", {64, 32}), kind + "blk.5.attn_compressor_gate.weight"},
             {"a compressor on a reader", added("blk.3.attn_compressor_norm.weight", {32}), kind + "blk.3.attn_compressor_norm.weight"},
             {"an indexer on a layer that is no index source", added("blk.3.indexer.proj.weight", {64, 8}), kind + "blk.3.indexer.proj.weight"},
             {"index keys on an index source that owns none", added("blk.7.indexer_compressor_kv.weight", {32, 32}), kind + "blk.7.indexer_compressor_kv.weight"},
             {"Engram outside its layers", added("blk.0.engram_wkv.weight", {192, 320}), kind + "blk.0.engram_wkv.weight"},
             {"a missing router", [](gguf::GGUFModel& f) { drop_tensor(f, "blk.2.ffn_gate_inp.weight"); }, "inference: missing tensor blk.2.ffn_gate_inp.weight"},
             {"a missing head", [](gguf::GGUFModel& f) { drop_tensor(f, "output.weight"); }, "inference: missing tensor output.weight"},
             {"a missing Engram key norm", [](gguf::GGUFModel& f) { drop_tensor(f, "blk.4.engram_k.weight"); }, "inference: missing tensor blk.4.engram_k.weight"},
             {"a missing index key norm", [](gguf::GGUFModel& f) { drop_tensor(f, "blk.2.indexer_compressor_norm.weight"); }, "inference: missing tensor blk.2.indexer_compressor_norm.weight"},
             {"an Engram table of other rows", [](gguf::GGUFModel& f) {
                  for (auto& t : f.tensors) if (t.name == "blk.1.engram_embed.weight") t.ne[1] = 303;
              }, "inference: blk.1.engram_embed.weight does not hold deepseek41.engram.num_embeddings rows"}})
        refuses(what, why, [&, edit = edit] { plans(fork(edit)); });
    refuses("no embedding", "inference: missing tensor token_embd.weight", [] {
        reads(fork([](gguf::GGUFModel& f) { drop_tensor(f, "token_embd.weight"); }));
    });
}

// The upstream layout, refused by its key, and a file holding both layouts' keys.
void layouts() {
    refuses("the upstream layout", unsupported("ple.layers"), [] {
        reads(fork([](gguf::GGUFModel& f) {
            for (const char* k : {"engram.layer_ids", "engram.head_dim", "engram.n_heads", "engram.max_ngram_size", "engram.num_embeddings",
                                  "engram.primes", "engram.offsets", "engram.multipliers", "engram.compressed_vocab_size"}) drop(f, P + k);
            put(f, P + "ple.layers", array(gguf::V_UINT32, golden.engram_layers));
        }));
    });
    refuses("both layouts in one file", "inference: unsupported metadata deepseek41.ple.layers beside deepseek41.engram.layer_ids", [] {
        reads(fork([](gguf::GGUFModel& f) { put(f, P + "ple.layers", array(gguf::V_UINT32, golden.engram_layers)); }));
    });
}

// The tiny file's eight layers against the table of F2DEV's review of 2026-11-05 16:01, and the plan of each.
void plan() {
    const gguf::GGUFModel m = fork();
    const infer::ModelWeights w = infer::gguf_weights(m);
    const auto table = infer::deepseek4::layer_table(config_of(m));
    struct Row { int ratio; bool kv, index, keys, candidates, uses, engram; int source; };
    const Row want[] = {
        {0, false, false, false, false, false, false, -1},  // the window alone
        {0, false, false, false, false, false, true, -1},   // the window with Engram
        {2, true, true, true, false, false, false, 2},      // KV and index source owning its keys
        {2, false, false, false, false, false, false, 2},   // a ratio-2 reader
        {2, true, true, true, false, false, true, 4},       // KV and index source with Engram
        {1, true, true, true, true, false, false, 5},       // KV and index source, the candidate source, at ratio 1
        {1, false, false, false, false, false, false, 5},   // a ratio-1 reader
        {1, false, true, false, false, true, false, 5},     // an index source without keys reading the candidates
    };
    require(table.size() == 8, "eight layers");
    for (size_t l = 0; l < 8; ++l) {
        const auto& t = table[l];
        const Row& r = want[l];
        require(t.ratio == r.ratio && t.kv_source == r.kv && t.index_source == r.index && t.owns_keys == r.keys &&
                    t.candidate_source == r.candidates && t.uses_candidates == r.uses && t.engram == r.engram && t.source == r.source,
                "layer " + std::to_string(l) + "'s place among the sources");
    }
    using namespace infer::deepseek4;
    const infer::ModelPlan p = built(w);
    const uint8_t kinds[] = {0, runs_engram, reads_compressed | compresses_kv | runs_indexer | owns_index_keys, reads_compressed,
                             reads_compressed | compresses_kv | runs_indexer | owns_index_keys | runs_engram,
                             reads_compressed | compresses_kv | runs_indexer | owns_index_keys | picks_candidates, reads_compressed,
                             reads_compressed | runs_indexer | reads_candidates};
    const uint64_t hc = golden["hc_mult"], d = golden["dim"], heads = (golden["engram_max_ngram_size"] - 1) * golden["engram_n_heads"];
    const infer::TensorIndex index(w.tensors);
    size_t roles = p.pass.size();
    for (const infer::Role& role : p.pass) require(infer::role_fits(role, index.view(index.at(role.name)).shape, index.view(index.at(role.name)).type), role.name + " fits its role");
    for (size_t l = 0; l < 8; ++l) {
        const infer::LayerPlan& layer = p.layers[l];
        require(layer.kind == kinds[l], "layer " + std::to_string(l) + "'s kind");
        // Entries: the window ring of window + H rows with H the window, the open group's kv and score rows on a ratio-2 source, and a source's pooled entry and index key.
        const size_t window = golden["window_size"], dim = golden["head_dim"], key = golden["index_head_dim"];
        const size_t ratio = size_t(table[l].ratio), tail = 2 * window * dim + (table[l].kv_source && ratio > 1 ? (ratio + window) * 2 * dim : 0);
        require(layer.cache == infer::Cache::entries && layer.routed && layer.headroom == window && layer.state_tail == tail && !layer.kv_aux,
                "layer " + std::to_string(l) + " keeps entries with its rings");
        require(layer.kv_pool == (table[l].kv_source ? ratio : 0) && layer.kv_pool_width == (table[l].kv_source ? dim + key : 0),
                "layer " + std::to_string(l) + "'s pooled rows");
        // The tiny file's readers 3, 6 and 7 take their entries and keys from sources 2, 5 and 5.
        const int sources[8] = {-1, -1, -1, 2, -1, -1, 5, 5};
        require(layer.entry_source == sources[l] && (sources[l] < 0 || table[size_t(sources[l])].owns_keys) &&
                    layer.entry_source == (table[l].ratio && !table[l].kv_source ? table[l].source : -1),
                "layer " + std::to_string(l) + "'s entry source");
        require(layer.token_history == (table[l].engram ? size_t(golden["engram_max_ngram_size"]) - 1 : 0) &&
                    layer.token_padding == (table[l].engram ? uint32_t(golden["engram_pad_id"]) : 0),
                "layer " + std::to_string(l) + "'s token history");
        require(layer.lookup.has_value() == table[l].engram && (!layer.lookup || (layer.lookup->role == engram_embed && layer.lookup->rows == heads)),
                "layer " + std::to_string(l) + "'s Engram lookup");
        for (const infer::Role& role : layer.roles) {
            const infer::TensorView& t = index.view(index.at(role.name));
            require(infer::role_fits(role, t.shape, t.type), role.name + " fits its role");
            require(role.name.rfind("blk." + std::to_string(l) + ".", 0) == 0, role.name + " is its layer's");
        }
        roles += layer.roles.size();
    }
    require(roles == w.tensors.size() + golden.engram_layers.size(), "every tensor of the file is a role of the plan, engram_wkv two");
    for (const char* name : {"blk.0.attn_sinks.weight", "blk.0.exp_probs_b.bias", "blk.0.hc_attn_base.weight", "blk.0.hc_ffn_scale.weight"})
        for (const infer::Role& role : p.layers[0].roles)
            if (role.name == name) require(role.kind == infer::RoleKind::norm, std::string(name) + " is a norm role");
    for (const infer::Role& role : p.layers[1].roles) {
        if (role.name == "blk.1.hc_attn_fn.weight") require(role.kind == infer::RoleKind::table && role.in == hc * d && role.out == (2 + hc) * hc, "hc_attn_fn is a table");
        if (role.name == "blk.1.engram_embed.weight") require(role.kind == infer::RoleKind::gather && role.out == 304, "engram_embed is gathered");
        if (role.name == "blk.1.engram_wkv.weight")
            require(role.kind == infer::RoleKind::matrix && role.tensor_rows == d * (hc + 1) &&
                        (role.id == infer::deepseek4::engram_key ? role.first_row == 0 && role.out == hc * d : role.first_row == hc * d && role.out == d),
                    "engram_wkv's key rows and value row are two runs of the one tensor");
        if (role.name == "blk.1.engram_q.weight" || role.name == "blk.1.engram_k.weight")
            require(role.kind == infer::RoleKind::source && role.in == d && role.out == hc, role.name + " is a source of the norm table");
    }
    const size_t ctx = golden["max_seq_len"], rd = golden["rope_head_dim"];
    const size_t hd = size_t(golden["n_heads"]) * size_t(golden["head_dim"]), groups = golden["o_groups"], window = golden["window_size"];
    const std::vector<size_t> slots{hc * d, hc, hc, (2 + hc) * hc, hc, hc * hc, d, hc * d, hc, golden["q_lora_rank"], hd, golden["head_dim"], hd, window, groups,
                                    groups * size_t(golden["o_lora_rank"]), d, golden["n_routed_experts"], golden["n_activated_experts"],
                                    golden["n_activated_experts"], (size_t(golden["n_activated_experts"]) + 1) * size_t(golden["moe_inter_dim"]),
                                    (size_t(golden["n_activated_experts"]) + 1) * size_t(golden["moe_inter_dim"]),
                                    (size_t(golden["n_activated_experts"]) + 1) * size_t(golden["moe_inter_dim"]),
                                    heads * size_t(golden["engram_head_dim"]), hc * d, d, hc * d, hc * d, golden["head_dim"], golden["head_dim"],
                                    golden["head_dim"], golden["index_head_dim"], size_t(golden["head_dim"]) + size_t(golden["index_head_dim"]),
                                    size_t(golden["index_n_heads"]) * size_t(golden["index_head_dim"]), golden["index_n_heads"], ctx, 1, golden["index_topk"], golden["candidate_topk_blocks"]};
    require(p.residual == hc * d && p.embed_copies == hc && p.context_length == ctx && p.slots == slots &&
                p.vocab == golden["vocab_size"] && p.index_width == std::max(size_t(heads), window) && !p.kv_heads && !p.head_dim,
            "the residual of the streams, the context, the hyper-connection, window attention, FFN, Engram, compressor, indexer and candidate slots and the index scratch of the lookup and the window");
    std::vector<size_t> tables(4, 0);
    tables.push_back(hc * d);
    tables.push_back(hc);
    tables.push_back(groups);
    tables.push_back(d);
    tables.insert(tables.end(), golden.engram_layers.size(), d * hc);
    require(p.tables == tables && p.table_steps == std::vector<size_t>(4, rd / 2),
            "the cosines and sines of two ropes a position, the ones table, the first pre, the group ids, the FFN's zero row and an Engram layer's norm table each");
    require(p.position_slots == std::vector<std::pair<size_t, size_t>>{{infer::deepseek4::index_scores, 1}}, "the index scores a float a position at the least ratio, 1");
}

// The runtime's footprint over the plan: each layer's tensors once, the Engram tables resident since no row source is given, the tables, slot 0, the index scratch, and each layer's entries against a sum written here.
void footprint() {
    const gguf::GGUFModel m = fork();
    const infer::ModelWeights w = infer::gguf_weights(m);
    const infer::ModelPlan p = infer::plan_model(w);
    infer::ModelOptions options;
    options.state_slots = 3;
    options.checkpoint_slots = 2;
    const infer::Footprint fp = infer::footprint(w, p, options);
    std::vector<size_t> bytes(8, 0);
    size_t embedding = 0, output = 0, norm = 0;
    for (const auto& t : w.tensors) {
        if (t.name == "token_embd.weight") embedding = t.bytes;
        else if (t.name == "output.weight") output = t.bytes;
        else if (t.name == "output_norm.weight") norm = t.bytes;
        else if (t.name.find(".engram_q.") == std::string::npos && t.name.find(".engram_k.") == std::string::npos) bytes.at(size_t(std::stoul(t.name.substr(4)))) += t.bytes;
    }
    for (size_t l = 0; l < 8; ++l) {
        size_t sum = 0;
        for (const infer::Matrix& x : fp.layers[l]) sum += x.bytes;
        require(sum == bytes[l], "layer " + std::to_string(l) + "'s weights counted once each");
        require(!fp.lookups[l].rows_per_token, "layer " + std::to_string(l) + " counts no row source");
    }
    // The tiny file's ratios 0 0 2 2 2 1 1 1 and KV sources 2, 4 and 5: a ring of 256 rows of 32 in each of 5 slots, 130 kv and 130 score rows more at ratio 2.
    // A source's 64-float pooled rows over the 256-token context, one each 2 positions or each one.
    const size_t ring = 256 * 32, group = 130 * 2 * 32;
    const size_t want[8] = {5 * ring, 5 * ring, 5 * (ring + group) + 128 * 64, 5 * ring, 5 * (ring + group) + 128 * 64, 5 * ring + 256 * 64, 5 * ring, 5 * ring};
    for (size_t l = 0; l < 8; ++l) require(fp.cache[l] == want[l] * sizeof(float), "layer " + std::to_string(l) + "'s entries counted as their slots and pooled rows");
    const size_t pool2 = 128 * 64 * sizeof(float), pool5 = 256 * 64 * sizeof(float);
    require(fp.reads == std::vector<int>{-1, -1, -1, 2, -1, -1, 5, 5} && fp.pooled == std::vector<size_t>{0, 0, pool2, 0, pool2, pool5, 0, 0},
            "the fit holds each reader's source and each source's pooled rows");
    // Two stages cut at layer 3 charge the second a replica of source 2's rows, which reader 3 reads there; source 5's readers share its stage and take none, and a cut at 6 charges source 5's instead.
    infer::DeviceBudget d;
    d.name = "cpu";
    d.bytes = size_t(1) << 32;
    const auto sum = [&](size_t a, size_t b) {
        size_t s = 0;
        for (size_t l = a; l < b; ++l) s += fp.cache[l];
        return s;
    };
    const infer::LayerSplit at3 = infer::split_layers(fp, {d, d}, 4, {3, 5}), at6 = infer::split_layers(fp, {d, d}, 4, {6, 2});
    require(at3.stages.at(0).cache == sum(0, 3) && at3.stages.at(1).cache == sum(3, 8) + pool2, "a cut at layer 3 charges the second stage source 2's replica once");
    require(at6.stages.at(0).cache == sum(0, 6) && at6.stages.at(1).cache == sum(6, 8) + pool5, "a cut at layer 6 charges the second stage source 5's replica once");
    require(fp.host_state == 2 * (5 * 3) * sizeof(uint32_t), "each Engram layer's three preceding ids in every slot");
    // Settled to native, the default (docs/DEEPSEEK4.md, section 4.3): a ring row of 32 values is one 33-byte FP8 block in 9 words, the open group's 8320 floats padded to 925 rows of 9.
    // A pooled row is two 9-byte FP4 entry blocks in 5 words, then a 17-byte MXFP4 key block in 5; settling back to f32 restores the F32 widths.
    infer::ModelPlan settled = p;
    infer::ModelOptions given = options;
    infer::settle_cache(settled, given);
    require(settled.native && given.kv_k == backend::KVType::native && given.kv_v == backend::KVType::native, "a plan that declares its format settles to native by default");
    const infer::Footprint nf = infer::footprint(w, settled, given);
    const size_t nring = 256 * 9, ngroup = 925 * 9, npool = 10;
    const size_t packed[8] = {5 * nring, 5 * nring, 5 * (nring + ngroup) + 128 * npool, 5 * nring, 5 * (nring + ngroup) + 128 * npool, 5 * nring + 256 * npool, 5 * nring,
                              5 * nring};
    for (size_t l = 0; l < 8; ++l) require(nf.cache[l] == packed[l] * sizeof(float), "layer " + std::to_string(l) + "'s entries counted at the packed sizes");
    infer::ModelOptions f32 = options;
    f32.kv_k = f32.kv_v = backend::KVType::f32;
    f32.kv_k_given = f32.kv_v_given = true;
    infer::settle_cache(settled, f32);
    require(!settled.native && infer::footprint(w, settled, f32).cache == fp.cache, "a plan settled back to f32 counts the F32 widths");
    const size_t ctx = golden["max_seq_len"], rd = golden["rope_head_dim"], R = golden["hc_mult"] * golden["dim"];
    const size_t norms = golden.engram_layers.size() * R;
    require(fp.embedding.bytes == embedding && fp.output.bytes == output && fp.output_norm.bytes == norm && !fp.tied && fp.head_aux.empty(), "the pass's weights");
    const size_t hd = size_t(golden["n_heads"]) * size_t(golden["head_dim"]), groups = golden["o_groups"], window = golden["window_size"];
    const size_t hc = golden["hc_mult"], slots = 2 * R + 4 * hc + (2 + hc) * hc + hc * hc + 2 * size_t(golden["dim"]) + size_t(golden["q_lora_rank"]) + 2 * hd +
                                                 size_t(golden["head_dim"]) + window + groups + groups * size_t(golden["o_lora_rank"]) +
                       size_t(golden["n_routed_experts"]) + 2 * size_t(golden["n_activated_experts"]) +
                       3 * (size_t(golden["n_activated_experts"]) + 1) * size_t(golden["moe_inter_dim"]) +
                       size_t(golden["engram_n_heads"]) * (size_t(golden["engram_max_ngram_size"]) - 1) * size_t(golden["engram_head_dim"]) + 3 * R + size_t(golden["dim"]) +
                       4 * size_t(golden["head_dim"]) + 2 * size_t(golden["index_head_dim"]) + size_t(golden["index_n_heads"]) * (size_t(golden["index_head_dim"]) + 1) +
                       ctx + 1 + size_t(golden["index_topk"]) + size_t(golden["candidate_topk_blocks"]);
    require(fp.tables == (2 * ctx * rd + R + norms + hc + groups + size_t(golden["dim"])) * sizeof(float) && fp.activations_per_row == slots * sizeof(float) &&
                fp.handoff_per_row == (R + 2 * hc + size_t(golden["head_dim"]) + size_t(golden["index_head_dim"]) + size_t(golden["index_topk"]) + size_t(golden["candidate_topk_blocks"])) * sizeof(float) && fp.host_scratch_per_row == window * sizeof(uint32_t),
            "the tables, the slots, the handoff row with the carried slots and the index scratch");
}

// A device reporting `room` bytes free.
struct Room : backend::CpuBackend {
    size_t room = 0;
    std::optional<size_t> memory_available() const override { return room; }
    bool reads_in_place() const override { return false; }
    bool is_cpu() const override { return false; }
    bool memory_is_host() const override { return false; }
};

// The tables and the index scores cover the positions the KV budget reaches, inside the fit's search: a device too small for them at the whole context holds a budget between half and the whole, and a pass past it is refused.
void positions() {
    const gguf::GGUFModel m = fork();
    const infer::ModelWeights w = infer::gguf_weights(m);
    const infer::ModelPlan p = infer::plan_model(w);
    const size_t ctx = golden["max_seq_len"], rd = golden["rope_head_dim"], half = ctx / 2, rows = 512;
    infer::ModelOptions whole, halved;
    halved.kv_tokens = half;
    const infer::Footprint a = infer::footprint(w, p, whole), b = infer::footprint(w, p, halved);
    require(a.tables - b.tables == 2 * half * rd * sizeof(float) && a.activations_per_row - b.activations_per_row == half * sizeof(float),
            "the four rope tables and the index scores row not sized by the budget");
    const auto device = [](size_t room) {
        auto d = std::make_shared<Room>();
        d->room = room;
        d->set_threads(1);
        return std::vector<backend::BackendPtr>{d};
    };
    infer::PlacementRequest request;
    request.names = {"device"};
    request.ubatch = int(rows);
    // The least rooms the footprint gives half and the whole budget; the placement charges a little more (the lookup rows, the host's share), so rooms past the half's by half the gap and then whole gaps are tried.
    const auto least = [&](const infer::Footprint& fp) {
        const auto holds = [&](size_t room) {
            infer::DeviceBudget d;
            d.name = "device";
            d.bytes = room;
            try {
                infer::split_layers(fp, {d}, rows);
                return true;
            } catch (const std::runtime_error&) {
                return false;
            }
        };
        size_t lo = 1, hi = size_t(1) << 34;
        while (hi - lo > 1) {
            const size_t mid = lo + (hi - lo) / 2;
            (holds(mid) ? hi : lo) = mid;
        }
        return hi;
    };
    const size_t half_room = least(b), gap = least(a) - half_room;
    require(gap > 0, "the whole budget's footprint is not larger than half's");
    std::optional<infer::PlacedModel> placed;
    for (size_t i = 0; i < 8 && !placed; ++i) {
        try {
            placed = infer::place_model(w, device(half_room + gap / 2 + i * gap), request, whole);
        } catch (const std::runtime_error&) {
        }
    }
    require(placed.has_value(), "no room past half the budget's footprint placed the model");
    const size_t total = placed->model->kv_tokens_total();
    require(total >= half && total < ctx && placed->model->positions() == total, "the fit did not settle between half and the whole budget with tables to match");
    std::vector<uint32_t> ids(total + 1, 1);
    refuses("a prompt past the covered positions", "inference: context length exceeded (" + std::to_string(total) + " tokens)", [&] { placed->model->prefill(ids); });
    bool refused = false;
    try { infer::place_model(w, device(half_room / 4), request, whole); } catch (const std::runtime_error& e) { refused = std::string(e.what()).rfind("placement: ", 0) == 0; }
    require(refused, "a device too small for one block not refused by the placement");
}

// engram_q times engram_k against doubles: the fixture's 1 + j/64 values, every pair, and random BF16 pairs whose products are normal F32.
void norm_product() {
    std::vector<uint16_t> q, k;
    for (int a = -16; a <= 16; ++a)
        for (int b = -16; b <= 16; ++b) {
            q.push_back(f32_to_bf16(1.0f + float(a) / 64.0f));
            k.push_back(f32_to_bf16(1.0f + float(b) / 64.0f));
        }
    uint64_t state = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < 100000; ++i) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        const uint32_t bits = uint32_t(state >> 33);
        // A sign, an exponent within 2^-60 to 2^60 and seven mantissa bits each, so every product stays a normal F32.
        const auto draw = [](uint32_t r) { return uint16_t((r & 0x8000u) | (((r >> 7) % 121u + 67u) << 7) | (r & 0x7fu)); };
        q.push_back(draw(bits));
        k.push_back(draw(bits >> 15 | bits << 17));
    }
    const auto view = [](const char* name, const std::vector<uint16_t>& v) {
        return infer::TensorView{name, {v.size(), 1}, quant::GGML_TYPE_BF16, reinterpret_cast<const uint8_t*>(v.data()), v.size() * sizeof(uint16_t)};
    };
    std::vector<float> out(q.size());
    infer::deepseek4::engram_norm_weight(view("q", q), view("k", k), out.data(), q.size());
    for (size_t i = 0; i < q.size(); ++i) {
        const double exact = double(bf16_to_f32(q[i])) * double(bf16_to_f32(k[i]));
        if (double(out[i]) != exact) throw std::runtime_error("engram_norm_weight is not exact at " + std::to_string(i));
    }
    require(true, "engram_q times engram_k exact in F32");
    // F32 views of the same values, as the tiny file holds them, give the same products, and another type or size is refused.
    std::vector<float> qf(q.size()), again(q.size());
    for (size_t i = 0; i < q.size(); ++i) qf[i] = bf16_to_f32(q[i]);
    const infer::TensorView fq{"q", {qf.size(), 1}, quant::GGML_TYPE_F32, reinterpret_cast<const uint8_t*>(qf.data()), qf.size() * sizeof(float)};
    infer::deepseek4::engram_norm_weight(fq, view("k", k), again.data(), q.size());
    require(std::memcmp(out.data(), again.data(), out.size() * sizeof(float)) == 0, "F32 and BF16 sources give the same products");
    infer::TensorView f16 = view("blk.1.engram_q.weight", q);
    f16.type = quant::GGML_TYPE_F16;
    refuses("an F16 norm source", "inference: unsupported tensor type blk.1.engram_q.weight, which the Engram norm reads as BF16 or F32",
            [&] { infer::deepseek4::engram_norm_weight(f16, view("k", k), again.data(), q.size()); });
    refuses("a norm source of another size", "inference: unsupported tensor type q, which the Engram norm reads as BF16 or F32",
            [&] { infer::deepseek4::engram_norm_weight(view("q", q), view("k", k), again.data(), q.size() - 1); });
}

// The position tables against a reference written here from model.py's precompute_freqs_cis in double (model.py:369-389 and :681-697 at e5decd76), and each Engram layer's norm table from its sources.
void position_tables() {
    const gguf::GGUFModel m = fork();
    const infer::ModelWeights w = infer::gguf_weights(m);
    const infer::ModelPlan p = built(w);
    std::vector<std::vector<float>> tables(p.tables.size());
    for (size_t i = 0; i < tables.size(); ++i) tables[i].assign(infer::table_floats(p, i, p.context_length), -1.0f);
    // Sources of values 1 + j/64, a pair a layer.
    const size_t n = golden["dim"] * golden["hc_mult"];
    std::vector<std::vector<float>> values;
    std::vector<infer::TensorView> sources;
    values.reserve(2 * golden.engram_layers.size());
    for (size_t s = 0; s < 2 * golden.engram_layers.size(); ++s) {
        values.emplace_back(n);
        for (size_t i = 0; i < n; ++i) values.back()[i] = 1.0f + float(int((i * 7 + s * 5) % 33) - 16) / 64.0f;
        sources.push_back({"source", {n, 1}, quant::GGML_TYPE_F32, reinterpret_cast<const uint8_t*>(values.back().data()), n * sizeof(float)});
    }
    w.arch->fill_tables(tables, sources);
    const Golden& g = golden;
    const int dim = int(g["rope_head_dim"]);
    const size_t half = size_t(dim / 2), ctx = g["max_seq_len"];
    const double pi = 3.14159265358979323846;
    // precompute_freqs_cis: base^-(2i/dim), and with original_seq_len YaRN's linear ramp between the corrected dims of beta_fast and beta_slow.
    const auto freqs = [&](double base, bool yarn) {
        std::vector<double> f(half);
        for (size_t i = 0; i < half; ++i) f[i] = 1.0 / std::pow(base, double(2 * i) / double(dim));
        if (!yarn) return f;
        const double orig = g.args.at("original_seq_len"), factor = g.args.at("rope_factor");
        const auto cd = [&](double r) { return double(dim) * std::log(orig / (r * 2 * pi)) / (2 * std::log(base)); };
        const double low = std::max(std::floor(cd(g.args.at("beta_fast"))), 0.0), high = std::min(std::ceil(cd(g.args.at("beta_slow"))), double(dim - 1));
        for (size_t i = 0; i < half; ++i) {
            const double smooth = 1 - std::min(std::max((double(i) - low) / std::max(high - low, 1e-3), 0.0), 1.0);
            f[i] = f[i] / factor * (1 - smooth) + f[i] * smooth;
        }
        return f;
    };
    const std::vector<double> window = freqs(g.args.at("rope_theta"), false), compressed = freqs(g.args.at("compress_rope_theta"), true);
    // Within one F32 unit of the double value, which each table rounds once.
    const auto within_unit = [](float got, double want) { return std::fabs(double(got) - want) <= std::ldexp(std::max(std::fabs(want), std::ldexp(1.0, -126)), -23); };
    for (size_t pos = 0; pos < ctx; ++pos)
        for (size_t i = 0; i < half; ++i) {
            const size_t at = pos * half + i;
            const double wa = double(pos) * window[i], ca = double(pos) * compressed[i];
            if (!(within_unit(tables[0][at], std::cos(wa)) && within_unit(tables[1][at], std::sin(wa)) && within_unit(tables[2][at], std::cos(ca)) && within_unit(tables[3][at], std::sin(ca))))
                throw std::runtime_error("the rope tables at position " + std::to_string(pos) + " pair " + std::to_string(i));
        }
    require(std::all_of(tables[4].begin(), tables[4].end(), [](float v) { return v == 1.0f; }), "the ones table");
    const std::vector<float>& zero = tables[infer::deepseek4::zeros];
    require(zero.size() == golden["dim"] && std::all_of(zero.begin(), zero.end(), [](float v) { return v == 0.0f && !std::signbit(v); }), "the FFN's zero row");
    const std::vector<float>& first = tables[infer::deepseek4::first_pre];
    require(first.size() == golden["hc_mult"] && first[0] == 1.0f && std::all_of(first.begin() + 1, first.end(), [](float v) { return v == 0.0f; }),
            "the first block's one-hot pre");
    const std::vector<float>& ids = tables[infer::deepseek4::groups];
    require(ids.size() == golden["o_groups"], "a group id a group");
    for (uint32_t gid = 0; gid < ids.size(); ++gid) require(!std::memcmp(&ids[gid], &gid, sizeof(gid)), "group " + std::to_string(gid) + "'s id as U32 bits");
    for (size_t l = 0; l < golden.engram_layers.size(); ++l)
        for (size_t i = 0; i < n; ++i)
            if (double(tables[infer::deepseek4::engram_norms + l][i]) != double(values[2 * l][i]) * double(values[2 * l + 1][i])) throw std::runtime_error("layer " + std::to_string(l) + "'s norm table");
    require(true, "the rope tables within one F32 unit of the double reference and the norm tables exact");
    // What the reference's own float32 pipeline (float32 frequencies, the angle in float32, then polar) differs by from the tables, at 256 and 65536 positions, for STATUS.
    const std::vector<double> f = infer::deepseek4::rope_frequencies(dim, g.args.at("compress_rope_theta"), int(g.args.at("original_seq_len")), g.args.at("rope_factor"),
                                                                     g.args.at("beta_fast"), g.args.at("beta_slow"));
    for (size_t positions : {size_t(256), size_t(65536)}) {
        double most = 0;
        for (size_t pos = 0; pos < positions; ++pos)
            for (size_t i = 0; i < half; ++i) {
                const float angle = float(pos) * float(f[i]);
                const double exact = double(pos) * f[i];
                most = std::max({most, std::fabs(double(std::cos(angle)) - double(float(std::cos(exact)))), std::fabs(double(std::sin(angle)) - double(float(std::sin(exact))))});
            }
        std::cout << "arch-deepseek4: the float32 pipeline's largest difference from the compressed tables over " << positions << " positions: " << most << "\n";
    }
}

// A model on the CPU is built from the plan and runs its first pass, also with the drafter's zero ratios after the blocks; a cache type given as f16 is refused by its flag before.
// A CPU that lacks one op, and a device beside it, to see the refusal at load and the stream check.
struct Lacking : backend::CpuBackend {
    backend::Op lack;
    explicit Lacking(backend::Op op) : lack(op) {}
    bool implements(backend::Op op) const override { return op != lack; }
};
struct LackingDevice : LoadingBackend {
    std::vector<backend::Op> lack;
    bool implements(backend::Op op) const override { return std::find(lack.begin(), lack.end(), op) == lack.end(); }
};

// Every op and mode the layers issue is declared under its part, so a backend without one refuses the model at load by layer and op, and a stream destination without the FFN's keeps its layer at home.
void device_ops() {
    using backend::Op;
    const gguf::GGUFModel m = fork();
    const infer::ModelWeights weights = infer::gguf_weights(m);
    const infer::ModelPlan p = built(weights);
    require(p.pass_ops.size() == 1 && p.pass_ops[0].part == infer::Part::head && p.pass_ops[0].op == Op::hc_collapse_norm, "the head declares hc_collapse_norm");
    refuses("a head without hc_collapse_norm", "inference: the head needs hc_collapse_norm, which the backend of its device does not implement",
            [&] { infer::Model model(weights, std::make_shared<Lacking>(Op::hc_collapse_norm)); });
    const auto declares = [&](size_t l, infer::Part part, Op op) {
        for (const infer::OpUse& u : p.layers[l].ops)
            if (u.part == part && u.op == op) return true;
        return false;
    };
    const Op every[] = {Op::hc_coeffs, Op::hc_expand, Op::norm_rope_partial, Op::rope_forms, Op::model_format_qdq, Op::sparse_attention_sink,
                        Op::routed_experts, Op::sqrt_softplus_routing, Op::silu_mul_clamp, Op::pool_write, Op::compress_pool,
                        Op::indexer_scores, Op::topk_rows, Op::candidate_blocks, Op::group_rms_norm, Op::signed_sqrt_gate};
    for (Op op : every) {
        size_t l = 0;
        infer::Part part = infer::Part::mixer;
        for (; l < p.layers.size(); ++l) {
            if (declares(l, infer::Part::mixer, op)) break;
            if (declares(l, infer::Part::ffn, op)) { part = infer::Part::ffn; break; }
        }
        require(l < p.layers.size(), std::string("some layer declares ") + backend::op_name(op));
        refuses(std::string("a CPU without ") + backend::op_name(op),
                "inference: layer " + std::to_string(l) + "'s " + (part == infer::Part::mixer ? "mixer" : "feed-forward part") + " needs " +
                    backend::op_name(op) + ", which the backend of its device does not implement",
                [&] { infer::Model model(weights, std::make_shared<Lacking>(op)); });
    }
    const infer::deepseek4::Config c = config_of(m);
    for (size_t l = 0; l < p.layers.size(); ++l) {
        require(declares(l, infer::Part::mixer, Op::rope_forms) && declares(l, infer::Part::ffn, Op::sqrt_softplus_routing) &&
                    declares(l, infer::Part::ffn, Op::routed_experts),
                "layer " + std::to_string(l) + " declares its rope forms, its routing and its routed products");
        require(declares(l, infer::Part::ffn, Op::silu_mul_clamp) == (c.swiglu_exp[l] != 0.0f || c.swiglu_shexp[l] != 0.0f),
                "layer " + std::to_string(l) + " declares the clamp where its block has one");
        require(declares(l, infer::Part::mixer, Op::compress_pool) == (p.layers[l].kv_pool != 0), "layer " + std::to_string(l) + " declares compress_pool where it is a KV source");
    }
    // Experts on the CPU beside a device: the layers stream to it while it has every op, and stay home where it lacks the FFN's routing.
    const auto streams = [&](std::vector<Op> lack) {
        auto device = std::make_shared<LackingDevice>();
        device->lack = std::move(lack);
        device->set_threads(1);
        auto host = backend::make_cpu_backend();
        host->set_threads(1);
        infer::Placement place;
        place.mixer_device.assign(p.layers.size(), 0);
        place.ffn_device.assign(p.layers.size(), 1);
        place.stream_from = 1;
        infer::Model model(weights, {device, host}, place);
        size_t n = 0;
        for (const auto& layer : model.topology().layers) n += !layer.stream_members.empty();
        return n;
    };
    require(streams({}) == p.layers.size(), "every layer streams to a device with every op");
    require(streams({Op::sqrt_softplus_routing}) == 0, "no layer streams to a device without sqrt_softplus_routing");
    refuses("a device mixer without rope_forms", "inference: layer 0's mixer needs rope_forms, which the backend of its device does not implement",
            [&] { streams({Op::rope_forms}); });
}

void named() {
    refuses("an f16 key cache", "--cache-type-k: this model keeps its cache in native or f32, not f16", [] {
        infer::ModelOptions options;
        options.kv_k = backend::KVType::f16;
        options.kv_k_given = true;
        const gguf::GGUFModel m = fork();
        infer::Model model(infer::gguf_weights(m), backend::make_cpu_backend(), options);
    });
    refuses("two cache types", "--cache-type-k and --cache-type-v: this model keeps one cache type for both sides, not f32 and native", [] {
        infer::ModelOptions options;
        options.kv_k = backend::KVType::f32;
        options.kv_k_given = true;
        const gguf::GGUFModel m = fork();
        infer::Model model(infer::gguf_weights(m), backend::make_cpu_backend(), options);
    });
    runs(fork());
    require(true, "the first pass runs every layer");
    runs(fork([](gguf::GGUFModel& f) { put(f, P + "attention.compress_ratios", array(gguf::V_INT32, {0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0})); }));
    require(true, "the first pass runs every layer with the drafter's zero ratios after the blocks");
}

// The writer's tiny file split over CPUs gives one CPU's logits bit for bit: every row of a 40-token text in passes of 3, then 4 decode steps.
// A cut at 3 parts reader 3 from layer 2's entries and picks, which its replica and the carried picks give it; a cut at 6 parts readers 6 and 7 from layer 5's entries, picks and candidate blocks; both pres cross at every cut.
void split_cuts(const std::string& path) {
    gguf::GGUFModel m = gguf::read_gguf(path);
    gguf::map_payload(m);
    const infer::ModelWeights w = infer::gguf_weights(m);
    require(infer::plan_model(w).carried == std::vector<size_t>{infer::deepseek4::pre_attn, infer::deepseek4::pre_ffn, infer::deepseek4::pooled, infer::deepseek4::index_picks, infer::deepseek4::candidate_ids}, "the carried slots");
    std::vector<uint32_t> ids;
    for (uint32_t i = 0; i < 40; ++i) ids.push_back((i * 37 + 5) % 128);
    const auto logits = [&](const std::vector<int>& devices) {
        std::vector<backend::BackendPtr> backends;
        for (int d = 0; d <= *std::max_element(devices.begin(), devices.end()); ++d) {
            backends.push_back(backend::make_cpu_backend());
            backends.back()->set_threads(1);
        }
        infer::Placement placement;
        placement.mixer_device = placement.ffn_device = devices;
        placement.output_device = devices.back();
        infer::ModelOptions options;
        options.dtype = backend::Dtype::f32;
        infer::Model model(w, std::move(backends), placement, options);
        model.set_ubatch(3);
        std::vector<float> out;
        model.score(ids, [&](size_t, const float* row) { out.insert(out.end(), row, row + 128); });
        for (size_t i = 0; i < 4; ++i) {
            const float* row = model.step(&ids[i], 1);
            out.insert(out.end(), row, row + 128);
        }
        return out;
    };
    const std::vector<float> one = logits({0, 0, 0, 0, 0, 0, 0, 0});
    require(one.size() == 44 * 128 && std::all_of(one.begin(), one.end(), [](float x) { return std::isfinite(x); }), "one CPU's 44 finite rows");
    require(std::memcmp(one.data(), one.data() + 128, 128 * sizeof(float)) != 0, "the file's weights give rows that differ by position");
    size_t cuts = 0;
    for (const std::vector<int>& cut : {std::vector<int>{0, 0, 0, 1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0, 1, 1}, {0, 0, 0, 1, 1, 1, 2, 2}, {0, 0, 1, 1, 2, 2, 3, 3}}) {
        std::string name;
        for (int d : cut) name += std::to_string(d);
        const std::vector<float> got = logits(cut);
        require(got.size() == one.size() && !std::memcmp(got.data(), one.data(), one.size() * sizeof(float)), "devices " + name + " give one CPU's logits bit for bit");
        ++cuts;
    }
    std::cout << "arch-deepseek4: the writer's file over " << cuts << " cuts of two to four CPUs gives one CPU's 44 rows bit for bit\n";
}

// A CPU backend that records, in call order, each hc_coeffs's pre, post and comb and each hc_expand's sublayer output f (hc_post's input in model.py), as Model::forward runs the module.
struct Recorder : backend::CpuBackend {
    Recorder() { set_threads(1); }
    struct Coefficients {
        std::vector<float> pre, post, comb;
    };
    std::vector<Coefficients> coefficients;
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<uint32_t>> picks, index_picks, candidates;

    std::vector<float> take(backend::CSlice at, size_t n) {
        std::vector<float> v(n);
        read(*at.buffer, at.offset * sizeof(float), v.data(), n * sizeof(float));
        return v;
    }
    void hc_coeffs(backend::Slice pre, backend::Slice post, backend::Slice comb, backend::CSlice mixes, backend::CSlice stream, backend::CSlice scale,
                   backend::CSlice base, size_t rows, size_t hc, size_t dim, float norm_eps, float eps, size_t iters) override {
        CpuBackend::hc_coeffs(pre, post, comb, mixes, stream, scale, base, rows, hc, dim, norm_eps, eps, iters);
        coefficients.push_back({take({pre.buffer, pre.offset}, rows * hc), take({post.buffer, post.offset}, rows * hc), take({comb.buffer, comb.offset}, rows * hc * hc)});
    }
    void route_experts(backend::CSlice scores, size_t rows, size_t n_expert, size_t k, const backend::RouteRule& rule, backend::Slice ids,
                       backend::Slice weights) override {
        CpuBackend::route_experts(scores, rows, n_expert, k, rule, ids, weights);
        std::vector<uint32_t> v(rows * k);
        read(*ids.buffer, ids.offset * sizeof(float), v.data(), v.size() * sizeof(uint32_t));
        picks.push_back(std::move(v));
    }
    void topk_rows(backend::Slice ids, backend::CSlice scores, backend::CSlice lens, size_t rows, size_t n, size_t k, uint32_t offset) override {
        CpuBackend::topk_rows(ids, scores, lens, rows, n, k, offset);
        std::vector<uint32_t> v(rows * k);
        read(*ids.buffer, ids.offset * sizeof(float), v.data(), v.size() * sizeof(uint32_t));
        index_picks.push_back(std::move(v));
        index_k.push_back(k);
    }
    std::vector<size_t> index_k;
    void candidate_blocks(backend::Slice blocks, backend::CSlice scores, backend::CSlice lens, size_t rows, size_t n, size_t block_size, size_t k) override {
        CpuBackend::candidate_blocks(blocks, scores, lens, rows, n, block_size, k);
        std::vector<uint32_t> v(rows * k);
        read(*blocks.buffer, blocks.offset * sizeof(float), v.data(), v.size() * sizeof(uint32_t));
        candidates.push_back(std::move(v));
        candidate_k = k;
    }
    size_t candidate_k = 0;
    void hc_expand(backend::Slice dst, backend::CSlice f, backend::CSlice stream, backend::CSlice post, backend::CSlice comb, size_t rows, size_t hc, size_t dim) override {
        outputs.push_back(take(f, rows * dim));
        CpuBackend::hc_expand(dst, f, stream, post, comb, rows, hc, dim);
    }
};

// The golden's sequence through Model::forward on the writer's file in one F32 pass: every layer's sublayer outputs y against the reference's.
// Second lines: their coefficients and router picks, the index picks of layers 2, 4, 5 and 7, and layer 5's candidate blocks.
void forward_records(const std::string& path) {
    gguf::GGUFModel file = gguf::read_gguf(path);
    gguf::map_payload(file);
    const infer::deepseek4::Config c = config_of(file);
    const size_t H = size_t(c.hc_count), E = size_t(c.n_embd);
    auto recorder = std::make_shared<Recorder>();
    infer::ModelOptions options;
    options.dtype = backend::Dtype::f32;
    infer::Model model(infer::gguf_weights(file), recorder, options);
    const std::vector<uint64_t> sequence = wholes(field(golden.root, "sequence"));
    model.set_ubatch(int(sequence.size()));
    model.prefill(std::vector<uint32_t>(sequence.begin(), sequence.end()));
    const size_t rows = sequence.size(), layers = size_t(c.n_layer);
    require(recorder->outputs.size() == 2 * layers && recorder->coefficients.size() == 2 * layers && recorder->picks.size() == layers &&
                recorder->index_picks.size() == 4 && recorder->candidates.size() == 1,
            "two sublayer outputs, two coefficient sets and a routing a layer, four index selections and one of candidate blocks");
    const auto close = [&](const jmini::Value& want, const float* got, size_t n, const std::string& what, double& most) {
        const auto& list = want.asArray();
        require(list.size() == n, what + " holds " + std::to_string(n) + " values");
        for (size_t i = 0; i < n; ++i) most = std::max(most, std::fabs(double(got[i]) - list[i].num));
    };
    for (size_t l = 0; l < layers; ++l)
        for (const auto& [part, at, what] : {std::tuple{"attn", 2 * l, "attention"}, std::tuple{"ffn", 2 * l + 1, "FFN"}}) {
            const std::string layer = "layer " + std::to_string(l) + "'s ";
            double most = 0;
            for (const auto& [key, want] : field(field(field(golden.root, "outputs"), std::to_string(l).c_str()), part).obj) {
                const size_t r = size_t(std::stoul(key));
                require(r < rows, "the reference's output position " + key);
                close(want, &recorder->outputs[at][r * E], E, layer + what + " output at position " + key, most);
            }
            std::cout << "arch-deepseek4: " << layer << what << " output y through Model::forward differs from the reference's by at most " << most << " (bound 1e-5)\n";
            require(most <= 1e-5, layer + what + " output y within 1e-5 of the reference's");
        }
    const auto coefficients = [&](const Recorder::Coefficients& got, size_t l, const char* part, const std::string& what) {
        double worst = 0;
        for (const auto& [key, at] : field(field(field(golden.root, "sinkhorn"), std::to_string(l).c_str()), part).obj) {
            const size_t r = size_t(std::stoul(key));
            close(field(at, "pre"), &got.pre[r * H], H, what + " pre at " + key, worst);
            close(field(at, "post"), &got.post[r * H], H, what + " post at " + key, worst);
            std::vector<float> flat;
            for (const jmini::Value& row : field(at, "comb").asArray())
                for (const jmini::Value& v : row.asArray()) flat.push_back(float(v.num));
            require(flat.size() == H * H, what + " comb at " + key + " is hc x hc");
            for (size_t i = 0; i < H * H; ++i) worst = std::max(worst, std::fabs(double(got.comb[r * H * H + i]) - double(flat[i])));
        }
        std::cout << "arch-deepseek4: " << what << " through Model::forward differ from the reference's by at most " << worst << " (bound 1e-5)\n";
        require(worst <= 1e-5, what + " within 1e-5 of the reference's");
    };
    const size_t k = size_t(c.n_expert_used);
    for (size_t l = 0; l < layers; ++l) {
        const std::string layer = "layer " + std::to_string(l) + "'s ";
        coefficients(recorder->coefficients[2 * l], l, "attn", layer + "attention coefficients");
        coefficients(recorder->coefficients[2 * l + 1], l, "ffn", layer + "FFN coefficients");
        // The reference lists each row's picks sorted; llmx keeps them in score order.
        const auto& router = field(field(field(golden.root, "selections"), "router"), std::to_string(l).c_str()).asArray();
        require(router.size() == rows, "the reference routes every row of layer " + std::to_string(l));
        for (size_t r = 0; r < rows; ++r) {
            std::vector<uint32_t> got(recorder->picks[l].begin() + std::ptrdiff_t(r * k), recorder->picks[l].begin() + std::ptrdiff_t((r + 1) * k));
            std::sort(got.begin(), got.end());
            const std::vector<uint64_t> want = wholes(router[r]);
            require(std::equal(got.begin(), got.end(), want.begin(), want.end()), layer + "router picks at row " + std::to_string(r));
        }
        require(true, layer + "router picks equal the reference's at all " + std::to_string(rows) + " rows");
    }
    // The reference lists each row's reachable picks in position order, as topk_rows writes them before the unreachable ones.
    const char* const sources[] = {"2", "4", "5", "7"};
    for (size_t i = 0; i < 4; ++i) {
        const char* l = sources[i];
        const auto& index = field(field(field(golden.root, "selections"), "index"), l).asArray();
        const size_t k = recorder->index_k[i];
        require(index.size() == rows, std::string("the reference selects at every row of layer ") + l);
        for (size_t r = 0; r < rows; ++r) {
            std::vector<uint32_t> got;
            for (size_t j = 0; j < k; ++j)
                if (const uint32_t id = recorder->index_picks[i][r * k + j]; id != UINT32_MAX) got.push_back(id);
            const std::vector<uint64_t> want = wholes(index[r]);
            require(std::equal(got.begin(), got.end(), want.begin(), want.end()), std::string("layer ") + l + "'s index picks at row " + std::to_string(r));
        }
        require(true, std::string("layer ") + l + "'s index picks equal the reference's at all " + std::to_string(rows) + " rows");
    }
    // The reference's candidate mask, read back as its blocks in ascending order.
    const auto& blocks = field(field(field(golden.root, "selections"), "candidate"), "5").asArray();
    require(blocks.size() == rows, "the reference keeps candidate blocks at every row of layer 5");
    for (size_t r = 0; r < rows; ++r) {
        std::vector<uint32_t> got;
        for (size_t j = 0; j < recorder->candidate_k; ++j)
            if (const uint32_t id = recorder->candidates[0][r * recorder->candidate_k + j]; id != UINT32_MAX) got.push_back(id);
        std::sort(got.begin(), got.end());
        const std::vector<uint64_t> want = wholes(blocks[r]);
        require(std::equal(got.begin(), got.end(), want.begin(), want.end()), "layer 5's candidate blocks at row " + std::to_string(r));
    }
    require(true, "layer 5's candidate blocks equal the reference's at all " + std::to_string(rows) + " rows");
}

// The writer's file widened to a 1024-token context: a 512-row prompt in one pass and in slices of 1, 3 and 16, then 8 decode steps.
// score gives every slice the prompt's extent, so every prompt row takes the CPU's one prompt row class and every step its decode class, and the rows must be the same bits.
void slices(const std::string& path) {
    gguf::GGUFModel file = gguf::read_gguf(path);
    gguf::map_payload(file);
    put(file, P + "context_length", scalar(gguf::V_UINT32, 1024));
    const infer::ModelWeights w = infer::gguf_weights(file);
    const size_t V = golden["vocab_size"];
    std::vector<uint32_t> ids;
    for (uint32_t i = 0; i < 512; ++i) ids.push_back(uint32_t((i * 2654435761u >> 9) % V));
    const auto run = [&](int ubatch) {
        auto cpu = backend::make_cpu_backend();
        cpu->set_threads(1);
        infer::ModelOptions options;
        options.dtype = backend::Dtype::f32;
        infer::Model model(w, std::move(cpu), options);
        model.set_ubatch(ubatch);
        std::vector<float> out;
        model.score(ids, [&](size_t, const float* row) { out.insert(out.end(), row, row + V); });
        for (size_t i = 0; i < 8; ++i) {
            const float* row = model.step(&ids[i * 61], 1);
            out.insert(out.end(), row, row + V);
        }
        return out;
    };
    const std::vector<float> whole = run(512);
    require(whole.size() == 520 * V && std::all_of(whole.begin(), whole.end(), [](float x) { return std::isfinite(x); }), "the 512-row pass's 520 finite rows");
    for (int ubatch : {1, 3, 16}) {
        const std::vector<float> got = run(ubatch);
        size_t differ = 0;
        for (size_t r = 0; r < 520; ++r) differ += std::memcmp(&got[r * V], &whole[r * V], V * sizeof(float)) != 0;
        std::cout << "arch-deepseek4: a 512-row prompt in slices of " << ubatch << " and 8 steps: " << differ << " of 520 rows differ from one pass\n";
        require(differ == 0, "a 512-row prompt in slices of " + std::to_string(ubatch) + " gives one pass's bits");
    }
}

// The reference's prefill-then-decode runs at 128 and 129 on the writer's file: the prefix read as one prompt pass, then every later id as a decode step, all 200 rows held to the golden's logits within 2e-5.
void prefill_then_decode(const std::string& path) {
    gguf::GGUFModel file = gguf::read_gguf(path);
    gguf::map_payload(file);
    const infer::ModelWeights w = infer::gguf_weights(file);
    const size_t V = golden["vocab_size"];
    const std::vector<uint64_t> wide = wholes(field(golden.root, "sequence"));
    const std::vector<uint32_t> sequence(wide.begin(), wide.end());
    const auto& want = field(golden.root, "logits").asArray();
    require(want.size() == sequence.size(), "the golden's logits hold a row a position");
    for (size_t length : {size_t(128), size_t(129)}) {
        auto cpu = backend::make_cpu_backend();
        cpu->set_threads(1);
        infer::ModelOptions options;
        options.dtype = backend::Dtype::f32;
        infer::Model model(w, std::move(cpu), options);
        model.set_ubatch(int(length));
        std::vector<float> got;
        model.score(std::vector<uint32_t>(sequence.begin(), sequence.begin() + std::ptrdiff_t(length)), [&](size_t, const float* row) { got.insert(got.end(), row, row + V); });
        for (size_t p = length; p < sequence.size(); ++p) {
            const float* row = model.step(&sequence[p], 1);
            got.insert(got.end(), row, row + V);
        }
        require(got.size() == sequence.size() * V, "a row a position");
        double most = 0;
        for (size_t p = 0; p < sequence.size(); ++p) {
            const auto& row = want[p].asArray();
            require(row.size() == V, "the golden's row " + std::to_string(p));
            for (size_t i = 0; i < V; ++i) most = std::max(most, std::fabs(double(got[p * V + i]) - row[i].num));
        }
        std::cout << "arch-deepseek4: a " << length << "-token prefill then " << sequence.size() - length << " decode steps: logits within " << most << " of the reference (bound 2e-5)\n";
        require(most <= 2e-5, "the prefill of " + std::to_string(length) + " then decode within 2e-5 of the reference's logits");
    }
}

// The packed cache against f32 on the writer's file: a 129-token prefill in passes of 7, then decode steps, every logit the same bits, and the storages' bytes those of the packed rows.
void native_cache(const std::string& path) {
    gguf::GGUFModel file = gguf::read_gguf(path);
    gguf::map_payload(file);
    const infer::ModelWeights w = infer::gguf_weights(file);
    const size_t V = golden["vocab_size"];
    const std::vector<uint64_t> wide = wholes(field(golden.root, "sequence"));
    const std::vector<uint32_t> sequence(wide.begin(), wide.end());
    size_t bytes[2] = {0, 0};
    std::vector<float> got[2];
    for (int f32 = 0; f32 < 2; ++f32) {
        infer::ModelOptions options;
        options.dtype = backend::Dtype::f32;
        if (f32) {
            options.kv_k = options.kv_v = backend::KVType::f32;
            options.kv_k_given = options.kv_v_given = true;
        }
        infer::Model model(w, backend::make_cpu_backend(), options);
        model.set_ubatch(7);
        model.score(std::vector<uint32_t>(sequence.begin(), sequence.begin() + 129), [&](size_t, const float* row) { got[f32].insert(got[f32].end(), row, row + V); });
        for (size_t q = 129; q < sequence.size(); ++q) {
            const float* row = model.step(&sequence[q], 1);
            got[f32].insert(got[f32].end(), row, row + V);
        }
        bytes[f32] = model.kv_allocated_bytes();
    }
    require(got[0].size() == sequence.size() * V && !std::memcmp(got[0].data(), got[1].data(), got[0].size() * sizeof(float)),
            "a prefill and decode steps under native give the f32 cache's logits bit for bit");
    std::cout << "arch-deepseek4: native and f32 give the same logits over " << sequence.size() << " positions; the pooled rows take " << bytes[0] << " and " << bytes[1]
              << " bytes\n";
    require(bytes[0] < bytes[1], "the packed pooled rows take fewer bytes than f32's");
}

// Throws in the Nth hc_expand from now, which on a later stage of a split falls after the stages before it have committed.
struct FailingExpand : backend::CpuBackend {
    FailingExpand() { set_threads(1); }
    int fail_at = 0;
    void hc_expand(backend::Slice dst, backend::CSlice f, backend::CSlice stream, backend::CSlice post, backend::CSlice comb, size_t rows, size_t hc, size_t dim) override {
        if (fail_at && --fail_at == 0) throw std::runtime_error("injected after a source stage");
        CpuBackend::hc_expand(dst, f, stream, post, comb, rows, hc, dim);
    }
};

// Rollback by length (docs/DEEPSEEK4.md, section 4.3): verifies of 2 to 17 drafts after a mark, the text's tokens up to the kept count and other ids after it, each retracted to that count, then a decode step.
// Every kept row and each step's row must be a fresh sequence's decode-step bits, on one CPU, after a fork and on two CPUs with passes failing after the first stage committed (docs/TESTS.md).
void rollback(const std::string& path) {
    gguf::GGUFModel file = gguf::read_gguf(path);
    gguf::map_payload(file);
    put(file, P + "context_length", scalar(gguf::V_UINT32, 1024));
    const infer::ModelWeights w = infer::gguf_weights(file);
    const size_t V = golden["vocab_size"], end = 320, most = 17;
    std::vector<uint32_t> ids;
    for (uint32_t i = 0; i < end + most; ++i) ids.push_back(uint32_t((i * 2654435761u >> 11) % V));
    const auto options = [&](size_t states, size_t checkpoints) {
        infer::ModelOptions o;
        o.dtype = backend::Dtype::f32;
        o.kv_tokens = 2048;
        o.state_slots = states;
        o.checkpoint_slots = checkpoints;
        o.mark_slots = 1;
        o.mark_rows = most;
        return o;
    };
    const auto row = [&](infer::ExecContext& ctx, size_t i) { return std::vector<float>(ctx.logits(i), ctx.logits(i) + V); };
    // A fresh sequence's prompt, then one decode step a token: the row of every position from the prompt's end.
    const auto fresh = [&](size_t prompt) {
        auto cpu = backend::make_cpu_backend();
        cpu->set_threads(1);
        infer::Model model(w, std::move(cpu), options(1, 0));
        infer::Sequence seq = model.make_sequence();
        infer::ExecContext ctx;
        infer::BatchEntry head{&seq, ids.data(), prompt, false};
        head.extent = prompt;
        model.forward(ctx, &head, 1);
        std::vector<std::vector<float>> rows(end + most);
        for (size_t p = prompt; p < end + most; ++p) {
            infer::BatchEntry step{&seq, &ids[p], 1, true};
            step.extent = 1;
            model.forward(ctx, &step, 1);
            rows[p] = row(ctx, 0);
        }
        return rows;
    };
    // Rounds from `start` on `seq`; with `fail` every fourth round's verify, and every fourth round's step two rounds later, first fails after the first stage committed.
    const auto rounds = [&](infer::Model& model, infer::Sequence& seq, const std::vector<std::vector<float>>& ref, size_t start, FailingExpand* fail, const std::string& what) {
        size_t verifies = 0, rows = 0, differ = 0, failed = 0;
        bool block = false, group = false;
        infer::ExecContext ctx;
        const auto compare = [&](const std::vector<float>& got, size_t p) {
            ++rows;
            differ += std::memcmp(got.data(), ref[p].data(), V * sizeof(float)) != 0;
        };
        const auto fails = [&](const infer::BatchEntry& entry, size_t length, const std::string& pass) {
            fail->fail_at = 1;
            std::string error = "nothing";
            try {
                model.forward(ctx, &entry, 1);
            } catch (const std::runtime_error& e) {
                error = e.what();
            }
            require(error == "injected after a source stage" && seq.length() == length, what + ": " + pass + " failing after the first stage leaves the history at " +
                                                                                     std::to_string(length) + ", not " + std::to_string(seq.length()) + ", having thrown " + error);
            ++failed;
        };
        for (size_t L = start, r = 0; L + most < end; ++r) {
            const size_t k = 2 + r % 16, a = (r * 7 + 3) % (k + 1);
            std::vector<uint32_t> drafts(ids.begin() + std::ptrdiff_t(L), ids.begin() + std::ptrdiff_t(L + k));
            for (size_t i = a; i < k; ++i) drafts[i] = (drafts[i] + 1 + uint32_t(i)) % uint32_t(V);
            infer::BatchEntry verify{&seq, drafts.data(), k, true};
            verify.every_logits = true;
            verify.extent = 1;
            if (fail && r % 4 == 1) {
                require(model.mark(seq), what + ": a mark before a failing verify");
                fails(verify, L, "a verify");
                model.retract(seq, L);
            }
            require(model.mark(seq), what + ": a mark at " + std::to_string(L));
            model.forward(ctx, &verify, 1);
            ++verifies;
            block = block || (L % 128 && (L + k) / 128 != L / 128);
            group = group || (L % 2 && a);
            for (size_t i = 0; i < a; ++i) compare(row(ctx, i), L + i);
            require(model.retract(seq, L + a) == L + a && seq.length() == L + a, what + ": a retract to " + std::to_string(L + a));
            infer::BatchEntry step{&seq, &ids[L + a], 1, true};
            step.extent = 1;
            if (fail && r % 4 == 3) fails(step, L + a, "a decode step");
            model.forward(ctx, &step, 1);
            compare(row(ctx, 0), L + a);
            L += a + 1;
        }
        std::cout << "arch-deepseek4: " << what << ": " << verifies << " verifies of 2 to 17 drafts, " << failed << " failed passes, " << differ << " of " << rows
                  << " rows differ from a fresh sequence's decode steps\n";
        require(block && group, what + ": verifies across a 128-token block edge and a ratio-2 group edge");
        require(!fail || failed, what + ": failed passes");
        require(differ == 0, what + ": every kept row and the step after each retract are a fresh sequence's bits");
    };
    {
        const auto ref = fresh(120);
        auto cpu = backend::make_cpu_backend();
        cpu->set_threads(1);
        infer::Model model(w, std::move(cpu), options(1, 0));
        infer::Sequence seq = model.make_sequence();
        infer::ExecContext ctx;
        infer::BatchEntry head{&seq, ids.data(), 120, false};
        head.extent = 120;
        model.forward(ctx, &head, 1);
        rounds(model, seq, ref, 120, nullptr, "one CPU from a 120-token prompt");
    }
    {
        const auto ref = fresh(128);
        auto cpu = backend::make_cpu_backend();
        cpu->set_threads(1);
        infer::Model model(w, std::move(cpu), options(2, 1));
        infer::Sequence src = model.make_sequence();
        infer::ExecContext ctx;
        infer::BatchEntry head{&src, ids.data(), 128, false};
        head.extent = 128;
        model.forward(ctx, &head, 1);
        require(model.keep(src), "a checkpoint at 128");
        infer::Sequence seq = model.fork(src, 128);
        rounds(model, seq, ref, 128, nullptr, "a fork at a checkpoint at 128");
    }
    {
        const auto ref = fresh(120);
        std::vector<backend::BackendPtr> backends{backend::make_cpu_backend()};
        backends[0]->set_threads(1);
        auto failing = std::make_shared<FailingExpand>();
        backends.push_back(failing);
        infer::Placement placement;
        placement.mixer_device = placement.ffn_device = {0, 0, 0, 1, 1, 1, 1, 1};
        placement.output_device = 1;
        infer::Model model(w, std::move(backends), placement, options(1, 0));
        infer::Sequence seq = model.make_sequence();
        infer::ExecContext ctx;
        infer::BatchEntry head{&seq, ids.data(), 120, false};
        head.extent = 120;
        model.forward(ctx, &head, 1);
        rounds(model, seq, ref, 120, failing.get(), "two CPUs cut at layer 3 with failures");
    }
    // A prompt continued in two passes in flight on two CPUs cut at layer 3, the first aborted: by lengths where both chunks fit the ring's headroom together, else rewound, though each fits alone.
    const size_t W = golden["window_size"];
    const auto ref = fresh(120);
    const std::pair<size_t, size_t> chunks[] = {{W / 2, W / 2}, {W, W / 2}};
    for (const auto& [a, b] : chunks) {
        std::vector<backend::BackendPtr> backends{backend::make_cpu_backend(), backend::make_cpu_backend()};
        for (auto& b : backends) b->set_threads(1);
        infer::Placement placement;
        placement.mixer_device = placement.ffn_device = {0, 0, 0, 1, 1, 1, 1, 1};
        placement.output_device = 1;
        infer::Model model(w, std::move(backends), placement, options(1, 0));
        infer::Sequence seq = model.make_sequence();
        infer::ExecContext prompt, ctx;
        infer::BatchEntry head{&seq, ids.data(), 120, false};
        head.extent = 120;
        model.forward(prompt, &head, 1);
        model.reserve_passes(ctx, 2, W, 1);
        infer::BatchEntry first{&seq, &ids[120], a, false}, second{&seq, &ids[120 + a], b, false};
        first.extent = a;
        second.extent = b;
        model.begin_pass(ctx, 0, &first, 1, 0);
        model.run_pass_stage(ctx, 0, 0);
        model.run_pass_stage(ctx, 0, 1);
        model.begin_pass(ctx, 1, &second, 1, 0);
        model.run_pass_stage(ctx, 1, 0);
        model.run_pass_stage(ctx, 1, 1);
        const std::string what = "an abort of " + std::to_string(a) + " rows with " + std::to_string(b) + " continuing in flight";
        require(model.abort_pass(ctx, 0) == std::vector<size_t>{1}, what + " abandons the later pass");
        const size_t back = a + b > W ? 0 : 120;
        require(seq.length() == back, what + " leaves the history at " + std::to_string(back) + ", not " + std::to_string(seq.length()));
        if (!back) model.forward(prompt, &head, 1);
        size_t differ = 0;
        for (size_t p = 120; p < 128; ++p) {
            infer::BatchEntry step{&seq, &ids[p], 1, true};
            step.extent = 1;
            model.forward(prompt, &step, 1);
            differ += std::memcmp(row(prompt, 0).data(), ref[p].data(), V * sizeof(float)) != 0;
        }
        std::cout << "arch-deepseek4: " << what << ": back to " << back << ", " << differ << " of 8 steps differ from a fresh sequence's\n";
        require(differ == 0, what + ": the steps after it are a fresh sequence's bits");
    }
}

// A prompt in passes in flight on two CPUs cut at layer 1, its Engram tables read from the file, each pass formed before the one ahead of it runs its second stage, where the token tails are kept.
// Every layer keeps the read begun at its pass's formation, from the tail the pass ahead leaves, and the rows are the slices' one after another; a pass whose early ids are spoiled reads again and gives them too.
void early_reads(const std::string& path) {
    gguf::GGUFModel file = gguf::read_gguf(path);
    gguf::map_payload(file);
    put(file, P + "context_length", scalar(gguf::V_UINT32, 1024));
    infer::ModelWeights w = infer::gguf_weights(file);
    std::shared_ptr<const format::FileReader> reader;
    size_t tables = 0;
    for (size_t i = 0; i < w.tensors.size(); ++i) {
        infer::TensorView& t = w.tensors[i];
        if (t.name.find("engram_embed.weight") == std::string::npos) continue;
        const format::FileSpan span = file.span(i);
        if (!reader) reader = std::make_shared<const format::FileReader>(span.file);
        t.row_source = std::make_shared<const infer::WeightRows>(t.type, size_t(t.shape[0]), size_t(t.shape[1]), reader, span.offset, span.bytes);
        t.data = nullptr;
        ++tables;
    }
    require(tables == golden.engram_layers.size(), "every Engram table reads from the file");
    const size_t V = golden["vocab_size"], chunk = size_t(golden["window_size"]) / 2, chunks = 5;
    std::vector<uint32_t> ids;
    for (uint32_t i = 0; i < chunk * chunks; ++i) ids.push_back(uint32_t((i * 2654435761u >> 11) % V));
    infer::ModelOptions o;
    o.dtype = backend::Dtype::f32;
    o.kv_tokens = 2048;
    o.state_slots = 1;
    o.mark_slots = 1;
    o.mark_rows = chunk;
    const auto split = [&] {
        std::vector<backend::BackendPtr> backends{backend::make_cpu_backend(), backend::make_cpu_backend()};
        for (auto& b : backends) b->set_threads(1);
        infer::Placement placement;
        placement.mixer_device = placement.ffn_device = {0, 1, 1, 1, 1, 1, 1, 1};
        placement.output_device = 1;
        return std::make_unique<infer::Model>(w, std::move(backends), placement, o);
    };
    const auto serial = split();
    infer::Sequence s = serial->make_sequence();
    infer::ExecContext serial_ctx;
    std::vector<infer::BatchEntry> slices;
    for (size_t c = 0; c < chunks; ++c) {
        infer::BatchEntry e{&s, ids.data() + c * chunk, chunk, c + 1 == chunks};
        e.extent = ids.size();
        slices.push_back(e);
        serial->forward(serial_ctx, &slices.back(), 1);
    }
    const std::vector<float> want(serial_ctx.logits(0), serial_ctx.logits(0) + V);
    for (const bool spoil : {false, true}) {
        const auto flight = split();
        infer::Sequence q = flight->make_sequence();
        infer::ExecContext ctx;
        flight->reserve_passes(ctx, 2, chunk, 1);
        for (infer::BatchEntry& e : slices) e.seq = &q;
        std::vector<float> got;
        for (size_t t = 0; t <= chunks; ++t) {
            if (t < chunks) {
                flight->begin_pass(ctx, t % 2, &slices[t], 1, 0);
                // Another id where the first Engram layer's early read asked for its first row, so its stage finds other ids and reads them again.
                if (spoil && t == 2) {
                    uint32_t& id = ctx.passes[t % 2].lookups[golden.engram_layers[0]].ids[0];
                    id = id ? 0 : 1;
                }
                flight->run_pass_stage(ctx, t % 2, 0);
            }
            if (!t) continue;
            const size_t c = t - 1;
            flight->run_pass_stage(ctx, c % 2, 1);
            if (c + 1 == chunks) got.assign(flight->pass_logits(ctx, c % 2, 0), flight->pass_logits(ctx, c % 2, 0) + V);
            flight->end_pass(ctx, c % 2);
        }
        const infer::Model::LookupReads reads = flight->lookup_reads();
        const std::string what = spoil ? "passes in flight with one early read spoiled" : "passes in flight";
        std::cout << "arch-deepseek4: " << what << ": lookup reads kept " << reads.kept << ", restarted " << reads.restarted << ", late " << reads.late << "\n";
        require(reads.kept + reads.restarted == chunks * tables && reads.restarted == (spoil ? 1u : 0u) && reads.late == 0,
                what + ": every early read kept but the spoiled one");
        require(q.length() == ids.size() && got == want, what + ": the last row is the slices' one after another");
    }
}

// A file the Python writer wrote holds the metadata and the tensor table this test builds: the same keys with the same values and types, and the same tensors with the same shapes and types.
void same_file(const std::string& path) {
    const gguf::GGUFModel file = gguf::read_gguf(path), m = fork();
    std::map<std::string, const gguf::MetaValue*> mine;
    for (const auto& kv : m.kv) mine[kv.first] = &kv.second;
    require(file.kv.size() == m.kv.size(), "the writer's file holds " + std::to_string(file.kv.size()) + " keys, not " + std::to_string(m.kv.size()));
    for (const auto& kv : file.kv) {
        const auto it = mine.find(kv.first);
        require(it != mine.end() && gguf::detail::equal_value(*it->second, kv.second), "the writer's key " + kv.first);
    }
    std::map<std::string, const gguf::TensorInfo*> tensors;
    for (const auto& t : m.tensors) tensors[t.name] = &t;
    require(file.tensors.size() == m.tensors.size(), "the writer's file holds " + std::to_string(file.tensors.size()) + " tensors, not " + std::to_string(m.tensors.size()));
    for (const auto& t : file.tensors) {
        const auto it = tensors.find(t.name);
        require(it != tensors.end() && it->second->ne == t.ne && it->second->type == t.type, "the writer's tensor " + t.name);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2 && !(argc == 4 && std::string(argv[2]) == "--gguf"))
            throw std::runtime_error("usage: llmx-arch-deepseek4-test tests/data/deepseek41_reference.json [--gguf FILE]");
        read_golden(argv[1]);
        if (argc == 4) {
            same_file(argv[3]);
            forward_records(argv[3]);
            slices(argv[3]);
            split_cuts(argv[3]);
            prefill_then_decode(argv[3]);
            native_cache(argv[3]);
            rollback(argv[3]);
            early_reads(argv[3]);
        } else {
            removed_keys();
            held_values();
            rules();
            tensors();
            layouts();
            plan();
            footprint();
            positions();
            norm_product();
            position_tables();
            named();
            device_ops();
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "arch-deepseek4: " << checked << " checks passed" << std::endl;
    return 0;
}
