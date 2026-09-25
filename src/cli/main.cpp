#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <random>
#include <limits>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#include <io.h>
#pragma comment(lib, "shell32.lib")
#else
#include <unistd.h>
#endif

#include "config.hpp"
#if LLMX_HAS_BACKEND_VULKAN
#include "backends/vulkan/vulkan_backend.hpp"
#endif
#include "core/list.hpp"
#include "backends/devices.hpp"
#include "core/json.hpp"
#include "hub/pull.hpp"
#include "format/gguf.hpp"
#include "quant/quant.hpp"
#include "tokenizer/tokenizer.hpp"
#include "inference/sampler.hpp"
#include "inference/generate.hpp"
#include "inference/perplexity.hpp"
#include "inference/chat.hpp"
#include "model/arch_qwen.hpp"
#include "server/api.hpp"

// CLI argument parsing and dispatch; format, quantization, inference and model logic stay in their own layers.

namespace {

bool show_progress(const infer::GenParams& gp) {
#if defined(_WIN32)
    return gp.show_prompt_tokens || _isatty(_fileno(stderr));
#else
    return gp.show_prompt_tokens || isatty(fileno(stderr));
#endif
}

// --cache-type-k / --cache-type-v: the same two names on every backend.
backend::KVType kv_type_of(const std::string& name) {
    if (name == "f32") return backend::KVType::f32;
    if (name == "f16") return backend::KVType::f16;
    throw std::runtime_error("unknown cache type '" + name + "' (f32 or f16)");
}
infer::ModelOptions model_options(const infer::GenParams& gp) {
    infer::ModelOptions o;
    o.kv_k = kv_type_of(gp.cache_type_k);
    o.kv_v = kv_type_of(gp.cache_type_v);
    o.kv_tokens = gp.kv_tokens > 0 ? (size_t)gp.kv_tokens : 0;
    return o;
}

gguf::GGUFModel load_model(const std::string& path, bool visible) {
    if (!visible) return gguf::read_gguf(path);
    std::cerr << "Reading model metadata...\n";
    int previous = -1;
    auto model = gguf::read_gguf(path, [&](size_t completed, size_t total) {
        const int percent = total ? int(100.0 * double(completed) / double(total)) : 100;
        if (percent == previous) return;
        previous = percent;
        std::cerr << "\rLoading tensor data: " << percent << "%" << std::flush;
        if (completed == total) std::cerr << "\n";
    });
    std::cerr << "Preparing model...\n";
    return model;
}

void emit_text(const std::string& text) {
    std::cout << text << std::flush;
}

// ---------------------------------------------------------------------------
// model.json / model.bin helper structures (quantize input)
// ---------------------------------------------------------------------------

std::vector<gguf::TensorInfo> parse_model_json(const jmini::Value& root, uint32_t type) {
    std::vector<gguf::TensorInfo> out;
    const jmini::Value* tensors = root.get("tensors");
    if (!tensors || !tensors->isArray())
        throw std::runtime_error("model.json: missing \"tensors\" array");
    for (const auto& t : tensors->asArray()) {
        gguf::TensorInfo jt;
        jt.type = type;
        const jmini::Value* name = t.get("name");
        if (!name || !name->isString())
            throw std::runtime_error("model.json: tensor missing \"name\" string");
        jt.name = name->asString();
        const jmini::Value* shape = t.get("shape");
        if (!shape || !shape->isArray())
            throw std::runtime_error("model.json: tensor missing \"shape\" array: " + jt.name);
        if (shape->asArray().empty() || shape->asArray().size() > 4)
            throw std::runtime_error("model.json: tensor rank must be between 1 and 4: " + jt.name);
        for (const auto& d : shape->asArray()) {
            if (!d.isNumber())
                throw std::runtime_error("model.json: shape dim is not a number: " + jt.name);
            const double v = d.asNumber();
            // JSON numbers are doubles; stay within their consecutive integer range.
            if (!std::isfinite(v) || v < 1 || v > 9007199254740991.0 || std::floor(v) != v)
                throw std::runtime_error("model.json: dimension must be an integer from 1 to 2^53-1: " + jt.name);
            jt.ne.push_back(uint64_t(v));
        }
        out.push_back(std::move(jt));
    }
    return out;
}

// ---------------------------------------------------------------------------
// commands
// ---------------------------------------------------------------------------

int cmd_quantize(const std::string& json_path, const std::string& bin_path,
                 const std::string& out_path, const std::string& type_arg) {
    uint32_t type;
    size_t block;
    void (*quantize)(const float*, uint8_t*, size_t);
    if (type_arg == "q4_0") {
        type = gguf::GGML_TYPE_Q4_0; block = gguf::Q4_0_BLOCK;
        quantize = quant::quantize_row_q4_0;
    } else { // default q8_0
        type = gguf::GGML_TYPE_Q8_0; block = gguf::Q8_0_BLOCK;
        quantize = quant::quantize_row_q8_0;
    }
    std::ifstream jf(json_path);
    if (!jf) throw std::runtime_error("cannot open " + json_path);
    std::stringstream jss;
    jss << jf.rdbuf();
    jmini::Value root = jmini::parse(jss.str());
    gguf::GGUFModel m;
    m.tensors = parse_model_json(root, type);
    uint64_t need = 0;
    uint64_t output_size = 0;
    for (const auto& t : m.tensors) {
        need = gguf::checked_add(need, gguf::checked_multiply(t.n_elements(), sizeof(float)));
        output_size = gguf::checked_add(gguf::aligned_size(output_size, alignof(float)), t.data_size());
    }
    std::vector<float> values;
    if (need / sizeof(float) > values.max_size() ||
        need > uint64_t(std::numeric_limits<std::streamsize>::max()) ||
        output_size > m.blob.max_size())
        throw std::runtime_error("model.json: tensor storage exceeds allocation or stream limit");

    std::ifstream bf(bin_path, std::ios::binary);
    if (!bf) throw std::runtime_error("cannot open " + bin_path);
    bf.exceptions(std::ios::failbit | std::ios::badbit);
    bf.seekg(0, std::ios::end);
    const std::streamoff sz = bf.tellg();
    if (sz < 0) throw std::runtime_error("cannot determine model.bin size");
    if (uint64_t(sz) != need)
        throw std::runtime_error("model.bin size does not match model.json tensor shapes");
    bf.seekg(0, std::ios::beg);
    values.resize(size_t(need / sizeof(float)));
    if (need) bf.read(reinterpret_cast<char*>(values.data()), std::streamsize(need));
    const float* fptr = values.data();
    m.blob.reserve(size_t(output_size));

    const jmini::Value* name = root.get("name");
    std::string model_name = (name && name->isString()) ? name->asString() : "custom";

    gguf::MetaValue mv_name; mv_name.vtype = gguf::V_STRING; mv_name.s = model_name;
    gguf::MetaValue mv_arch; mv_arch.vtype = gguf::V_STRING; mv_arch.s = "custom";
    gguf::MetaValue mv_qver; mv_qver.vtype = gguf::V_UINT32; mv_qver.u = 2;   // quantization_version
    gguf::MetaValue mv_ft;   mv_ft.vtype   = gguf::V_UINT32;
    mv_ft.u = (type == gguf::GGML_TYPE_Q8_0) ? 7 : 2;   // 7 = MOSTLY_Q8_0, 2 = MOSTLY_Q4_0
    m.kv.emplace_back("general.name", mv_name);
    m.kv.emplace_back("general.architecture", mv_arch);
    m.kv.emplace_back("general.quantization_version", mv_qver);
    m.kv.emplace_back("general.file_type", mv_ft);

    for (const auto& t : m.tensors) {
        size_t nblocks = size_t(t.n_elements() / block);
        std::vector<uint8_t> q(size_t(t.data_size()));
        quantize(fptr, q.data(), nblocks);
        fptr += size_t(t.n_elements());
        m.add_tensor_data(q);
    }

    gguf::write_gguf(m, out_path);
    std::cout << "wrote " << out_path << " (" << m.tensors.size() << " tensors, "
              << type_arg << ")\n";
    return 0;
}

int cmd_dequantize(const std::string& in_path, const std::string& out_json,
                   const std::string& out_bin) {
    gguf::GGUFModel m = gguf::read_gguf(in_path);

    std::stringstream js;
    js << "{\n";
    js << "  \"name\": " << jmini::quote(in_path) << ",\n";
    js << "  \"tensors\": [\n";
    for (size_t i = 0; i < m.tensors.size(); i++) {
        const auto& t = m.tensors[i];
        js << "    {\"name\": " << jmini::quote(t.name) << ", \"shape\": [";
        for (size_t d = 0; d < t.ne.size(); d++) {
            if (d) js << ", ";
            js << t.ne[d];
        }
        js << "]}";
        if (i + 1 < m.tensors.size()) js << ",";
        js << "\n";
    }
    js << "  ]\n}\n";

    std::vector<uint8_t> out;
    for (size_t i = 0; i < m.tensors.size(); i++) {
        const auto& t = m.tensors[i];
        const uint8_t* raw = std::as_const(m).tensor_data(i);
        size_t n = (size_t)t.n_elements();
        std::vector<float> f(n);
        if (t.type == gguf::GGML_TYPE_F32) {
            std::memcpy(f.data(), raw, n * 4);
        } else {
            const quant::QuantType* qt = quant::Registry::instance().get(t.type);
            if (!qt || !qt->dequantize)
                throw std::runtime_error("unsupported tensor type in dequantize: " + t.name);
            qt->dequantize(raw, f.data(), n / qt->block_size);
        }
        out.resize(out.size() + n * 4);
        std::memcpy(out.data() + (out.size() - n * 4), f.data(), n * 4);
    }

    std::ofstream oj(out_json);
    if (!oj) throw std::runtime_error("cannot open " + out_json);
    oj << js.str();

    std::ofstream ob(out_bin, std::ios::binary);
    if (!ob) throw std::runtime_error("cannot open " + out_bin);
    ob.write((const char*)out.data(), (std::streamsize)out.size());

    std::cout << "wrote " << out_json << " and " << out_bin << "\n";
    return 0;
}

// Type names come from the quant registry, so a new quant type shows up in `info` without touching the CLI.
const char* type_name(uint32_t t) {
    if (t == gguf::GGML_TYPE_F32) return "F32";
    const quant::QuantType* qt = quant::Registry::instance().get(t);
    return qt ? qt->name : "?";
}

void dump_value(const gguf::MetaValue& v) {
    switch (v.vtype) {
        case gguf::V_UINT8:  std::cout << v.u; break;
        case gguf::V_INT8:   std::cout << v.i; break;
        case gguf::V_UINT16: std::cout << v.u; break;
        case gguf::V_INT16:  std::cout << v.i; break;
        case gguf::V_UINT32: std::cout << v.u; break;
        case gguf::V_INT32:  std::cout << v.i; break;
        case gguf::V_FLOAT32:{ float x; std::memcpy(&x, &v.fb, 4); std::cout << x; break; }
        case gguf::V_BOOL:   std::cout << (v.b ? "true" : "false"); break;
        case gguf::V_STRING: std::cout << '"' << v.s << '"'; break;
        case gguf::V_ARRAY: {
            std::cout << "[";
            for (size_t i = 0; i < v.arr.size(); i++) {
                if (i) std::cout << ", ";
                dump_value(v.arr[i]);
            }
            std::cout << "]";
            break;
        }
        case gguf::V_UINT64:  std::cout << v.u; break;
        case gguf::V_INT64:   std::cout << v.i; break;
        case gguf::V_FLOAT64: std::cout << v.f64; break;
    }
}

int cmd_info(const std::string& in_path) {
    gguf::GGUFModel m = gguf::read_gguf(in_path);
    std::cout << "File: " << in_path << "\n";
    std::cout << "Tensors: " << m.tensors.size() << "\n";
    std::cout << "Metadata: " << m.kv.size() << " entries\n";
    for (const auto& kv : m.kv) {
        std::cout << "  " << kv.first << " = ";
        dump_value(kv.second);
        std::cout << "\n";
    }
    std::cout << "Tensor list:\n";
    for (const auto& t : m.tensors) {
        std::cout << "  " << type_name(t.type) << " " << t.name << " shape=[";
        for (size_t d = 0; d < t.ne.size(); d++) {
            if (d) std::cout << ", ";
            std::cout << t.ne[d];
        }
        std::cout << "] elements=" << t.n_elements()
                  << " bytes=" << t.data_size() << "\n";
    }
    return 0;
}

std::vector<uint32_t> parse_token_ids(const std::string& s) {
    std::vector<uint32_t> ids;
    std::string cur;
    for (char c : s) {
        if (c == ',' || c == ' ') {
            if (!cur.empty()) { ids.push_back((uint32_t)std::stoull(cur)); cur.clear(); }
        } else cur += c;
    }
    if (!cur.empty()) ids.push_back((uint32_t)std::stoull(cur));
    return ids;
}

int cmd_tokenize(const std::string& model_path, const std::string& text) {
    gguf::GGUFModel m = gguf::read_gguf(model_path);
    bpe::Tokenizer t(m);
    std::vector<uint32_t> ids = t.encode(text);
    for (size_t i = 0; i < ids.size(); i++) {
        if (i) std::cout << ", ";
        std::cout << ids[i];
    }
    std::cout << "\n";
    return 0;
}

int cmd_detokenize(const std::string& model_path, const std::string& ids_arg) {
    gguf::GGUFModel m = gguf::read_gguf(model_path);
    bpe::Tokenizer t(m);
    std::vector<uint32_t> ids = parse_token_ids(ids_arg);
    // decode() rejects an out-of-range id, but only the caller knows which id it was and how large the vocabulary is.
    for (uint32_t id : ids)
        if (id >= t.vocab.size())
            throw std::runtime_error("detokenize: token id " + std::to_string(id) +
                                     " is outside the vocabulary of " +
                                     std::to_string(t.vocab.size()) + " tokens");
    std::cout << t.decode(ids) << "\n";
    return 0;
}

// Canonical spelling keeps vulkan and vulkan:00 from naming the same device twice.
std::vector<int> layer_shares(const std::string& value) {
    std::vector<int> shares;
    if (value.empty()) return shares;
    for (const auto& item : core::comma_list(value)) {
        if (item.empty() || item.find_first_not_of("0123456789") != std::string::npos || item.size() > 6)
            throw std::runtime_error("--layer-shares: '" + item + "' is not a whole-number share");
        shares.push_back(std::atoi(item.c_str()));
    }
    return shares;
}

// The execution flags every model command takes, the "Execution options" of its help: where it runs, its workers, its prompt batch and its caches.
// Reads argv[i] (and its value) into `gp` and returns true when it is one of them.
bool exec_flag(int argc, char** argv, int& i, infer::GenParams& gp) {
    const std::string a = argv[i];
    auto value = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
    if (a == "--device") { if (const char* v = value()) gp.device = v; }
    else if (a == "--layer-shares") { if (const char* v = value()) gp.layer_shares = v; }
    else if (a == "--n-cpu-moe") { if (const char* v = value()) gp.cpu_moe = std::atoi(v); }
    else if (a == "--cpu-moe") gp.cpu_moe = -1;
    else if (a == "--moe-stream-from") { if (const char* v = value()) gp.moe_stream_from = std::atoi(v); }
    else if (a == "--threads") { if (const char* v = value()) gp.threads = std::atoi(v); }
    else if (a == "--ubatch") { if (const char* v = value()) gp.ubatch = std::atoi(v); }
    else if (a == "--cache-type-k" || a == "-ctk") { if (const char* v = value()) gp.cache_type_k = v; }
    else if (a == "--cache-type-v" || a == "-ctv") { if (const char* v = value()) gp.cache_type_v = v; }
    else return false;
    return true;
}

// A model file opened for a command: the file, which the model reads while it lives, its tokenizer, and the model placed over the devices --device lists.
// Built in place and never moved, since the model keeps the file's address.
struct Opened {
    gguf::GGUFModel file;
    std::optional<bpe::Tokenizer> tok;
    std::unique_ptr<infer::Model> model;
    backend::Backend* first = nullptr;   // the first device listed, which bench --profile times
};

// Open a model file as the flags ask: read it, showing progress when `progress`, place the model over the listed devices for its ubatch plus `decode_rows` generated tokens a pass (a server's sequences), print a split's plan when `show_plan`, and release the host's copy of the weights when no weight reads it in place.
// `threads` is the worker count to set, 0 to keep the backend's own; `profile` times the one device's kernels.
std::unique_ptr<Opened> open_model(const std::string& path, const infer::GenParams& gp, bool progress, int threads, size_t decode_rows = 0,
                                   bool show_plan = false, bool profile = false) {
    auto opened = std::make_unique<Opened>();
    opened->file = load_model(path, progress);
    opened->tok.emplace(opened->file);
    const auto specs = backend::device_specs(gp.device);
    if (profile && (specs.size() > 1 || !gp.layer_shares.empty())) throw std::runtime_error("bench: --profile times one device; not with several");
    auto backends = backend::make_backends(specs, profile);
    opened->first = backends.front().get();
    infer::PlacementRequest request;
    request.names = specs;
    request.shares = layer_shares(gp.layer_shares);
    request.cpu_moe = gp.cpu_moe;
    request.stream_from = gp.moe_stream_from > 0 ? (size_t)gp.moe_stream_from : 0;
    request.ubatch = gp.ubatch;
    request.decode_rows = decode_rows;
    infer::PlacedModel placed = infer::place_model(opened->file, std::move(backends), request, model_options(gp));
    if (show_plan) std::cerr << placed.plan;
    opened->model = std::move(placed.model);
    if (!opened->model->holds_payload()) opened->file.release_payload();
    if (threads > 0) opened->model->set_threads(threads);
    return opened;
}

int cmd_generate(const std::string& model_path, const std::string& prompt,
                 const infer::GenParams& gp) {
    const bool progress = show_progress(gp);
    const auto opened = open_model(model_path, gp, progress, gp.threads, 0, gp.show_prompt_tokens);
    bpe::Tokenizer& tok = *opened->tok;
    infer::Model& model = *opened->model;
    const int decode_threads = model.threads_available();
    infer::RNG rng;
    if (gp.seed) rng.seed(gp.seed);

    std::vector<uint32_t> ids = tok.encode(prompt);
    if (ids.empty()) throw std::runtime_error("generate: empty prompt");

    // Prefill and decode can use different worker counts (--threads-batch / --threads).
    const int tb = (gp.threads_batch > 0) ? gp.threads_batch : decode_threads;
    model.set_threads(tb);
    if (gp.show_prompt_tokens) std::cerr << "threads: prefill " << model.threads_available() << "\n";
    if (progress) std::cerr << "Processing " << ids.size() << " prompt tokens...\n";
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> logits = model.prefill(ids);
    model.set_threads(decode_threads);
    double pp_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (gp.show_prompt_tokens) std::cout << "prompt tokens: " << ids.size() << "\n";
    // Preserve fractional throughput for slow models.
    printf("pp: %zu tok, %.0f ms, %.2f tok/s\n", ids.size(), pp_ms,
           (double)ids.size() / (pp_ms / 1e3));

    if (gp.show_prompt_tokens) std::cerr << "threads: decode " << model.threads_available() << "\n";
    if (progress) std::cerr << "Generating...\n";
    t0 = std::chrono::steady_clock::now();
    std::vector<uint32_t> gen = infer::generate(model, tok, gp, rng, logits, emit_text);
    std::cout << "\n";
    double tg_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("tg: %zu tok, %.0f ms, %.2f tok/s\n", gen.size(), tg_ms,
           (double)gen.size() / (tg_ms / 1e3));
    if (gp.show_prompt_tokens)
        printf("kv: allocated %zu bytes, peak %zu bytes, used %zu bytes\n",
               model.kv_allocated_bytes(), model.kv_peak_bytes(), model.kv_used_bytes());
    return 0;
}

// `then_ids` appends exact generated IDs without re-tokenizing their text; `last` reports the final positions through batched passes.
// These logits support the external correctness gate in docs/ROADMAP.md #8.
int cmd_logits(const std::string& model_path, const std::string& text,
               int topn, const infer::GenParams& gp, const std::string& then_ids = "", size_t last = 0) {
    const auto opened = open_model(model_path, gp, false, gp.threads, 0, gp.show_prompt_tokens);
    bpe::Tokenizer& tok = *opened->tok;
    infer::Model& model = *opened->model;

    std::vector<uint32_t> ids = tok.encode(text);
    if (!then_ids.empty()) {
        std::ifstream in(std::filesystem::u8path(then_ids));
        if (!in) throw std::runtime_error("logits: cannot open token ids: " + then_ids);
        for (unsigned long long id; in >> id;) {
            if (id >= model.n_vocab()) throw std::runtime_error("logits: token id out of range");
            ids.push_back((uint32_t)id);
        }
        if (!in.eof()) throw std::runtime_error("logits: token ids must be whitespace-separated integers");
    }
    if (ids.empty()) throw std::runtime_error("logits: empty prompt");

    auto top = [&](const float* logits) {
        std::vector<std::pair<float, uint32_t>> ranked;
        ranked.reserve(model.n_vocab());
        for (size_t i = 0; i < model.n_vocab(); i++) ranked.push_back({ logits[i], (uint32_t)i });
        const size_t n = std::min((size_t)topn, ranked.size());
        std::partial_sort(ranked.begin(), ranked.begin() + (std::ptrdiff_t)n, ranked.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        ranked.resize(n);
        return ranked;
    };
    printf("tokens: %zu\n", ids.size());
    if (last) {
        const size_t from = ids.size() - std::min(last, ids.size());
        model.score(ids, [&](size_t pos, const float* logits) {
            if (pos < from) return;
            printf("%zu", pos);
            for (const auto& r : top(logits)) printf(" %u %.6f", r.second, r.first);
            printf("\n");
        });
        return 0;
    }
    std::vector<float> logits = model.prefill(ids);
    for (const auto& r : top(logits.data())) printf("%u %.6f\n", r.second, r.first);
    return 0;
}

std::string read_perplexity_file(const std::string& path) {
    std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
    if (!input) throw std::runtime_error("perplexity: cannot open file: " + path);
    std::string text;
    char buffer[8192];
    while (input.read(buffer, sizeof(buffer)) || input.gcount())
        text.append(buffer, (size_t)input.gcount());
    if (!input.eof()) throw std::runtime_error("perplexity: cannot read file: " + path);
    return text;
}

int cmd_perplexity(const std::string& model_path, const std::string& text,
                   const infer::GenParams& gp, int context_size, int chunks, bool per_token) {
    const int threads = !per_token && gp.threads_batch > 0 ? gp.threads_batch : gp.threads;
    const auto opened = open_model(model_path, gp, false, threads, 0, gp.show_prompt_tokens);
    bpe::Tokenizer& tok = *opened->tok;
    infer::Model& model = *opened->model;
    if (gp.show_prompt_tokens)
        std::cerr << "threads: " << (per_token ? "decode " : "prefill ") << model.threads_available() << "\n";

    std::vector<uint32_t> ids = tok.encode(text);
    const auto result = infer::perplexity(model, ids, context_size, chunks, per_token);
    double mean_nll = result.mean_nll();
    double ppl = std::exp(mean_nll);
    std::cout << "tokens: " << ids.size() << "\n";
    std::cout << "used tokens: " << result.used_tokens << "\n";
    std::cout << "scored tokens: " << result.scored_tokens << "\n";
    std::cout << "chunks: " << result.chunks << "\n";
    std::cout << "context size: " << (context_size ? context_size : model.context_length()) << "\n";
    std::cout << "mean NLL: " << mean_nll << "\n";
    std::cout << "perplexity: " << ppl << "\n";
    return 0;
}

int cmd_chat(const std::string& model_path, const std::string& system,
             const infer::GenParams& gp) {
    const bool progress = show_progress(gp);
    const auto opened = open_model(model_path, gp, progress, gp.threads, 0, gp.show_prompt_tokens);
    const gguf::GGUFModel& m = opened->file;
    bpe::Tokenizer& tok = *opened->tok;
    infer::Model& model = *opened->model;
    const int decode_threads = model.threads_available();
    infer::RNG rng;
    if (gp.seed) rng.seed(gp.seed);

    std::string tpl = chat::get_chat_template(m);
    if (tpl.empty()) {
        tpl = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
              "{{ message['content'] }}<|im_end|>\n{% endfor %}"
              "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
    }
    std::string bos = (tok.bos_id >= 0 && (size_t)tok.bos_id < tok.vocab.size())
                          ? tok.vocab[tok.bos_id] : "";
    std::string eos = (tok.eos_id >= 0 && (size_t)tok.eos_id < tok.vocab.size())
                          ? tok.vocab[tok.eos_id] : "";

    std::vector<chat::Message> messages;
    messages.push_back({ "system", system });
    std::vector<uint32_t> cached_ids;

    std::cout << "Chat ready (type your message; Ctrl+C to quit)\n" << std::flush;
    std::string line;
    while (std::getline(std::cin, line)) {
        messages.push_back({ "user", line });

        std::string gen = chat::render(tpl, messages, true, bos, eos);
        std::vector<uint32_t> gen_ids = tok.encode(gen);
        if (gen_ids.empty()) throw std::runtime_error("chat: template produced an empty prompt");
        // Templates can rewrite previous turns or change token boundaries.
        // Reuse only an exact prefix; an unchanged prompt also needs fresh logits because generate() does not retain its final distribution.
        if (cached_ids.size() >= gen_ids.size() ||
            !std::equal(cached_ids.begin(), cached_ids.end(), gen_ids.begin())) {
            model.reset();
            cached_ids.clear();
        }
        model.set_threads(gp.threads_batch > 0 ? gp.threads_batch : decode_threads);
        if (gp.show_prompt_tokens) std::cerr << "threads: prefill " << model.threads_available() << "\n";
        if (progress) std::cerr << "Processing " << gen_ids.size() - cached_ids.size() << " prompt tokens...\n";
        std::vector<float> logits = model.prefill(
            std::vector<uint32_t>(gen_ids.begin() + cached_ids.size(), gen_ids.end()));
        cached_ids = std::move(gen_ids);

        model.set_threads(decode_threads);
        if (gp.show_prompt_tokens) std::cerr << "threads: decode " << model.threads_available() << "\n";
        if (progress) std::cerr << "Generating...\n";
        std::vector<uint32_t> reply = infer::generate(model, tok, gp, rng, logits, emit_text);
        std::cout << "\n" << std::flush;
        // A stop match may return its final token without feeding it.
        // EOS is excluded; the next rendered turn supplies its own closing tokens.
        const size_t fed = (size_t)model.n_tokens() - cached_ids.size();
        cached_ids.insert(cached_ids.end(), reply.begin(), reply.begin() + fed);

        messages.push_back({ "assistant", tok.decode(reply) });
    }
    return 0;
}

// Build a small random Qwen3 model in memory for end-to-end prefill/decode TPS measurement.
// Matrices are Q8_0, norms F32 (matching what infer::Model expects).
gguf::GGUFModel build_synthetic_model(int n_layer, int n_embd, int n_ff,
                                      int n_head, int n_head_kv, int head_dim,
                                      int n_vocab, uint32_t seed) {
    gguf::GGUFModel m;
    auto u32 = [&](const std::string& k, uint64_t v) {
        gguf::MetaValue mv; mv.vtype = gguf::V_UINT32; mv.u = v;
        m.kv.emplace_back(k, mv);
    };
    u32("qwen3.block_count", (uint64_t)n_layer);
    u32("qwen3.embedding_length", (uint64_t)n_embd);
    u32("qwen3.feed_forward_length", (uint64_t)n_ff);
    u32("qwen3.attention.head_count", (uint64_t)n_head);
    u32("qwen3.attention.head_count_kv", (uint64_t)n_head_kv);
    u32("qwen3.attention.key_length", (uint64_t)head_dim);
    u32("qwen3.context_length", 2048);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // ne = [nin, nout]; f32 tensors are stored raw, others quantized to Q8_0.
    auto add_tensor = [&](const std::string& name, size_t nin, size_t nout, bool f32) {
        gguf::TensorInfo t;
        t.name = name;
        t.ne = { (uint64_t)nin, (uint64_t)nout };
        t.type = f32 ? gguf::GGML_TYPE_F32 : gguf::GGML_TYPE_Q8_0;
        t.offset = 0;
        if (f32) {
            std::vector<uint8_t> buf(nin * nout * 4);
            float* p = (float*)buf.data();
            for (size_t o = 0; o < nout; o++)
                for (size_t i = 0; i < nin; i++) *p++ = dist(rng);
            m.tensors.push_back(std::move(t));
            m.add_tensor_data(buf);
        } else {
            size_t nblocks = nin / gguf::Q8_0_BLOCK;
            std::vector<uint8_t> buf(nout * nblocks * gguf::Q8_0_TYPESIZE);
            std::vector<float> row(nin);
            for (size_t o = 0; o < nout; o++) {
                for (size_t i = 0; i < nin; i++) row[i] = dist(rng);
                quant::quantize_row_q8_0(row.data(), buf.data() + o * nblocks * gguf::Q8_0_TYPESIZE, nblocks);
            }
            m.tensors.push_back(std::move(t));
            m.add_tensor_data(buf);
        }
    };

    size_t kv_dim = (size_t)n_head_kv * head_dim;
    add_tensor("token_embd.weight", n_embd, n_vocab, false);
    add_tensor("output.weight", n_embd, n_vocab, false);
    add_tensor("output_norm.weight", n_embd, 1, true);
    for (int l = 0; l < n_layer; l++) {
        std::string pre = "blk." + std::to_string(l) + ".";
        add_tensor(pre + "attn_norm.weight", n_embd, 1, true);
        add_tensor(pre + "attn_q.weight", n_embd, n_embd, false);
        add_tensor(pre + "attn_k.weight", n_embd, kv_dim, false);
        add_tensor(pre + "attn_v.weight", n_embd, kv_dim, false);
        add_tensor(pre + "attn_output.weight", n_embd, n_embd, false);
        add_tensor(pre + "attn_q_norm.weight", head_dim, 1, true);
        add_tensor(pre + "attn_k_norm.weight", head_dim, 1, true);
        add_tensor(pre + "ffn_norm.weight", n_embd, 1, true);
        add_tensor(pre + "ffn_gate.weight", n_embd, n_ff, false);
        add_tensor(pre + "ffn_up.weight", n_embd, n_ff, false);
        add_tensor(pre + "ffn_down.weight", n_ff, n_embd, false);
    }
    return m;
}

// Micro-benchmark of the backend hot paths (matmul, RMSNorm, norm+RoPE) plus end-to-end prefill/decode TPS on a synthetic Qwen3 model.
// Used by tests/perf.py as the perf-regression gate for hot-path changes.
int cmd_bench(int size, int iters, int threads, int prefill, int decode,
              const std::string& device) {
    // The hot paths are one backend's; with several devices listed, the first one's.
    auto b = backend::make_backend(backend::device_specs(device).front());
    if (threads > 0) b->set_threads(threads);
    std::cout << "bench: threads " << b->threads_available() << "\n";

    // Square matmul: mat is [nin, nout] = [size, size]. x is the input (length nin), out the result (length nout). nout rows, each nin/32 blocks.
    const size_t nblocks = (size_t)size / gguf::Q8_0_BLOCK;
    std::vector<float> x(size, 0.5f);
    std::vector<uint8_t> mat((size_t)size * nblocks * gguf::Q8_0_TYPESIZE);
    std::vector<float> src(size), w(size), dst(size);
    for (int i = 0; i < size; i++) { src[i] = std::sin((float)i * 0.01f); w[i] = 0.1f; }
    std::vector<float> cos(size / 2), sin(size / 2);
    for (int i = 0; i < size / 2; i++) { cos[i] = std::cos(0.1f); sin[i] = std::sin(0.1f); }

    const auto weights = b->adopt(mat.data(), mat.size());
    const auto x_buf = b->adopt(x.data(), x.size() * sizeof(float));
    const auto dst_buf = b->adopt(dst.data(), dst.size() * sizeof(float));

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    for (int it = 0; it < iters; it++)
        b->matmul(gguf::GGML_TYPE_Q8_0, {weights.get(), 0}, {x_buf.get(), 0},
                  {dst_buf.get(), 0}, (size_t)size, (size_t)size, 1);
    double mm_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count() / iters;

    const auto src_buf = b->adopt(src.data(), src.size() * sizeof(float));
    const auto w_buf = b->adopt(w.data(), w.size() * sizeof(float));

    t0 = clock::now();
    for (int it = 0; it < iters; it++)
        b->rms_norm({dst_buf.get(), 0}, {src_buf.get(), 0}, {w_buf.get(), 0},
                    (size_t)size, 1e-6f);
    double rn_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count() / iters;

    // The op the model runs: one row of one head of `size` floats, at position 0 of a one-entry table.
    const auto cos_buf = b->adopt(cos.data(), cos.size() * sizeof(float));
    const auto sin_buf = b->adopt(sin.data(), sin.size() * sizeof(float));
    const uint32_t pos0 = 0;
    t0 = clock::now();
    for (int it = 0; it < iters; it++)
        b->norm_rope_rows({dst_buf.get(), 0}, 1, 0, 1, {w_buf.get(), 0}, 1e-6f,
                          {cos_buf.get(), 0}, {sin_buf.get(), 0}, (size_t)size / 2, &pos0);
    double rp_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count() / iters;

    double mm_gflops = 2.0 * (double)size * (double)size / (mm_ms * 1e6);
    printf("bench: matmul %dx%d  %8.3f ms  %8.2f GFLOPS\n", size, size, mm_ms, mm_gflops);
    printf("bench: rms_norm n=%d  %8.3f ms\n", size, rn_ms);
    printf("bench: norm_rope n=%d  %8.3f ms\n", size, rp_ms);

    // End-to-end TPS on a small synthetic Qwen3 model (2 layers, 256 embd).
    {
        const int nl = 2, ne = 256, nf = 1024, nh = 8, nk = 2, hd = 32, nv = 512;
        gguf::GGUFModel sm = build_synthetic_model(nl, ne, nf, nh, nk, hd, nv, 12345u);
        infer::Model model(sm, b);

        const int P = prefill, G = decode;
        model.reset();
        t0 = clock::now();
        for (int i = 0; i < P; i++) model.step(i % nv);
        double pre_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        double pre_tps = (double)P / (pre_ms / 1e3);

        t0 = clock::now();
        for (int i = 0; i < G; i++) model.step((P + i) % nv);
        double dec_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        double dec_tps = (double)G / (dec_ms / 1e3);

        printf("bench: prefill %3d tok  %8.3f ms  %8.1f tok/s\n", P, pre_ms, pre_tps);
        printf("bench: decode  %3d tok  %8.3f ms  %8.1f tok/s\n", G, dec_ms, dec_tps);
    }
    return 0;
}

// Time model execution over fixed IDs after warm-up; history setup and sampling are outside the timer.
// Multi-sequence decode follows each sequence's prompt, while single-sequence runs may use the requested depth; see docs/USAGE.md.
int cmd_bench_model(const std::string& path, const infer::GenParams& gp, int P, int G, int R, bool profile, int seqs = 1, int D = 0) {
    const auto opened = open_model(path, gp, false, gp.threads, (size_t)seqs, true, profile);
    backend::Backend* b = opened->first;
    infer::Model& model = *opened->model;
    // Ids below 1000, or below a smaller vocabulary's size, such as the test fixtures'.
    const uint32_t vocab = (uint32_t)std::min<size_t>(1000, model.n_vocab());
    auto ids_from = [vocab](uint32_t seed, size_t n) {
        std::vector<uint32_t> ids(n);
        for (auto& t : ids) { seed = seed * 1664525u + 1013904223u; t = (seed >> 8) % vocab; }
        return ids;
    };
    const std::vector<uint32_t> prompt = ids_from(12345u, (size_t)P), gen = ids_from(777u, (size_t)G), history = ids_from(4242u, (size_t)D);
    // A cleared history, then the depth's tokens if any, before a timed test starts.
    auto fresh = [&] {
        model.reset();
        if (D > 0) model.prefill(history);
    };
    using clock = std::chrono::steady_clock;
    auto ms_since = [](clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    };
    auto pp = [&] {
        fresh();
        const auto t0 = clock::now();
        model.prefill(prompt);
        return (double)P / (ms_since(t0) / 1e3);
    };
    infer::ExecContext ctx;
    // `start` runs once the decode is set up and before its first pass, so a profile of it leaves out the sequences' prompts.
    auto tg = [&](const std::function<void()>& start) {
        if (seqs <= 1) {
            fresh();
            if (start) start();
            const auto t0 = clock::now();
            for (uint32_t t : gen) model.step((int)t);
            return (double)G / (ms_since(t0) / 1e3);
        }
        std::vector<infer::Sequence> s;
        for (int i = 0; i < seqs; ++i) s.push_back(model.make_sequence());
        for (auto& q : s) {
            infer::BatchEntry e{&q, prompt.data(), prompt.size(), true};
            model.forward(ctx, &e, 1);
        }
        ctx.logits(0);
        if (start) start();
        std::vector<infer::BatchEntry> batch;
        const auto t0 = clock::now();
        for (int g = 0; g < G; ++g) {
            batch.clear();
            for (auto& q : s) batch.push_back(infer::BatchEntry{&q, &gen[(size_t)g], 1, true});
            model.forward(ctx, batch.data(), batch.size());
            ctx.logits(0);   // a server reads each pass's logits before it samples the next tokens
        }
        const double rate = (double)G * seqs / (ms_since(t0) / 1e3);
        for (auto& q : s) model.reset(q);
        return rate;
    };
    auto report = [&](const char* what, int n, const std::vector<double>& v) {
        double mean = 0, var = 0;
        for (double x : v) mean += x;
        mean /= (double)v.size();
        for (double x : v) var += (x - mean) * (x - mean);
        const double sd = v.size() > 1 ? std::sqrt(var / (double)(v.size() - 1)) : 0.0;
        const std::string at = D > 0 ? " @ d" + std::to_string(D) : "";
        printf("bench: %s%d%s  %8.2f +- %.2f tok/s  (%zu runs)\n", what, n, at.c_str(), mean, sd, v.size());
    };
    pp();
    tg({});
    std::vector<double> ppv, tgv;
    for (int r = 0; r < R; r++) ppv.push_back(pp());
    for (int r = 0; r < R; r++) tgv.push_back(tg({}));
    report("pp", P, ppv);
    report(seqs > 1 ? ("x" + std::to_string(seqs) + " tg").c_str() : "tg", G, tgv);
    if (profile) {
#if LLMX_HAS_BACKEND_VULKAN
        // Device time per kernel over one more prompt and one more decode run, each read on its own, so a pass is attributed to its kernels rather than inferred from kernels timed alone.
        // A run calls its argument where its interval starts: a batched decode's after its sequences' prompts.
        auto section = [&](const char* what, const std::function<double(const std::function<void()>&)>& run) {
            run([&] { backend::vulkan_kernel_times(*b); });
            auto times = backend::vulkan_kernel_times(*b);
            std::sort(times.begin(), times.end(),
                      [](const auto& x, const auto& y) { return x.second > y.second; });
            double total = 0.0;
            for (const auto& t : times) total += t.second;
            std::cout << "profile " << what << ": " << total << " ms of device time over "
                      << backend::vulkan_timed_dispatches(*b) << " dispatches sampled\n";
            for (const auto& t : times)
                std::cout << "profile:   " << t.first << " " << t.second << " ms ("
                          << (total > 0.0 ? 100.0 * t.second / total : 0.0) << "%)\n";
        };
        section("pp", [&](const std::function<void()>& start) { start(); return pp(); });
        section(seqs > 1 ? "batched tg" : "tg", tg);
        // And what the driver made of each kernel that ran: registers, shared memory and waves per SIMD, where it reports them.
        std::istringstream stats(backend::vulkan_kernel_statistics(*b));
        for (std::string line; std::getline(stats, line);) std::cout << "profile: kernel " << line << "\n";
#else
        (void)b;
        std::cerr << "bench --profile: this build has no Vulkan backend\n";
#endif
    }
    return 0;
}

// llmx serve: the multi-user server of docs/SERVER.md over one model.
int cmd_serve(const std::string& model_path, const server::Config& cfg, const infer::GenParams& gp) {
    const auto opened = open_model(model_path, gp, true, gp.threads, cfg.max_seqs, gp.show_prompt_tokens);
    const gguf::GGUFModel& m = opened->file;
    bpe::Tokenizer& tok = *opened->tok;
    infer::Model& model = *opened->model;
    server::Config c = cfg;
    c.model_name = std::filesystem::path(model_path).filename().string();
    c.ubatch = model.prefill_batch();
    http::Listener listener(c.host, c.port);
    std::cerr << "serving " << c.model_name << " on http://" << c.host << ":" << listener.port()
              << " (device " << gp.device << ", up to " << c.max_seqs << " sequences over "
              << model.kv_tokens_total() << " KV tokens, queue of " << c.max_queue << ")\n";
    server::serve(model, tok, m, c, listener);
    return 0;
}

bool print_usage(const std::string& command = {}) {
    if (!command.empty() && command != "chat" && command != "generate" &&
        command != "serve" && command != "pull" && command != "info" &&
        command != "quantize" && command != "dequantize" && command != "tokenize" &&
        command != "detokenize" && command != "logits" && command != "perplexity" &&
        command != "bench") return false;
    std::cout << "llmx " << LLMX_VERSION_STRING << " - ground-up LLM runtime\n\n";
    if (command.empty()) {
        std::cout
            << "Usage: llmx <command> [arguments] [options]\n\n"
            << "Run a model:\n"
            << "  chat        <model>                  Interactive chat with follow-up turns\n"
            << "  generate    <model> \"<prompt>\"       Generate from a raw text prompt\n"
            << "  serve       <model>                  Start the HTTP inference server\n\n"
            << "Manage models:\n"
            << "  pull        <owner/repo>:<quant>     Download a GGUF from Hugging Face\n"
            << "  info        <model>                  Show metadata and tensor layouts\n"
            << "  quantize    <json> <bin> <out.gguf>   Convert F32 tensors to Q8_0/Q4_0\n"
            << "  dequantize  <gguf> <json> <bin>       Export tensors as F32\n\n"
            << "Inspect and measure:\n"
            << "  tokenize    <model> \"<text>\"         Encode text to token IDs\n"
            << "  detokenize  <model> <ids>            Decode comma/space-separated IDs\n"
            << "  logits      <model> \"<text>\"         Inspect next-token scores\n"
            << "  perplexity  <model> \"<text>\"         Score text or a file\n"
            << "  bench                               Measure kernels or a real model\n\n"
            << "A model is a GGUF file, or the first shard of a GGUF set.\n"
            << "Downloads require curl 8.4+; no Python runtime is needed.\n\n"
            << "Help:     llmx <command> --help  (or -h)\n"
            << "Version:  llmx --version\n"
            << "Example:  llmx chat model.gguf --threads 6 --temp 0 -n 256\n";
        return true;
    }
    const infer::GenParams defaults;
    const auto model_options = [&](bool batch_threads) {
        std::cout << "\nExecution options:\n"
            << "  --device D              cpu (default), or vulkan:N when built with Vulkan;\n"
            << "                          several, comma separated, split the model by layers\n"
            << "                          over them in that order, fitted to their free memory\n"
            << "  --layer-shares A,B      With several devices, their proportions of the layers\n"
            << "  --threads N             CPU workers; 0 selects automatically (default)\n";
        if (batch_threads) std::cout
            << "  --threads-batch N, -tb  CPU prefill workers; default follows --threads\n";
        std::cout
            << "  --ubatch N              Prompt tokens per pass (default: 512)\n"
            << "  --cache-type-k T, -ctk  Key cache: f16 (default) or f32\n"
            << "  --cache-type-v T, -ctv  Value cache: f16 (default) or f32\n"
            << "  --n-cpu-moe N           First N routed layers' experts on CPU (default: 0)\n"
            << "  --cpu-moe               All routed layers' experts on CPU\n"
            << "  --moe-stream-from N     Copy those experts to the device for a prompt of\n"
            << "                          at least N new tokens; 0 disables this (default).\n"
            << "                          Generated tokens stay on CPU.\n";
    };
    if (command == "chat" || command == "generate") {
        const bool chat = command == "chat";
        std::cout << (chat ? "Interactive chat with retained conversation history.\n\n"
                          : "Generate from raw text without applying a chat template.\n\n")
            << "Usage: llmx " << command << " <model.gguf>"
            << (chat ? " [options]\n" : " \"<prompt>\" [options]\n")
            << "\nGeneration options:\n"
            << "  -n N, --max-tokens N    Maximum generated tokens per turn (default: " << defaults.max_tokens << ")\n"
            << "  --temp F                Temperature; 0 is greedy (default: " << defaults.temp << ")\n"
            << "  --topk N                Top-k sampling (default: " << defaults.top_k << ")\n"
            << "  --topp F                Nucleus sampling (default: " << defaults.top_p << ")\n"
            << "  --penalty F             Repetition penalty (default: " << defaults.penalty << ")\n"
            << "  --seed N                RNG seed; 0 keeps the fixed default state\n"
            << "  --stop TEXT             Stop when generated text contains TEXT\n"
            << "  --verbose               Show prompt IDs, progress and execution details\n";
        if (chat) std::cout
            << "  --system TEXT           System message (default: You are a helpful assistant.)\n";
        model_options(true);
        if (chat) std::cout << "\nEnter one message per line; Ctrl+C or end of input exits.\n";
        std::cout << "\nExample: llmx " << command << " model.gguf"
            << (chat ? "" : " \"The capital of France is\"") << " --temp 0 -n 256\n";
    } else if (command == "serve") {
        const server::Config cfg;
        std::cout << "Serve concurrent requests with streaming and prefix reuse.\n\n"
            << "Usage: llmx serve <model.gguf> [options]\n\n"
            << "Server options:\n"
            << "  --host H                Listen address (default: " << cfg.host << ")\n"
            << "  --port N                Listen port (default: " << cfg.port << ")\n"
            << "  --max-seqs N            Active request limit (default: " << cfg.max_seqs << ")\n"
            << "  --max-queue N           Waiting request limit (default: " << cfg.max_queue << ")\n"
            << "  --ctx-size N, -c        Total KV token budget (default: model context)\n";
        model_options(false);
        std::cout << "\nRoutes:\n"
            << "  POST /v1/generate             POST /v1/chat\n"
            << "  POST /v1/completions          POST /v1/chat/completions\n"
            << "  GET  /v1/health               GET  /v1/models\n\n"
            << "Sampling settings belong in each request's JSON body.\n"
            << "Example: llmx serve model.gguf --device vulkan:0 --port 8080\n";
    } else if (command == "pull") {
        std::cout << "Download and verify a GGUF model or complete shard set.\n\n"
            << "Usage: llmx pull <owner/repo>:<quant> [options]\n\n"
            << "Options:\n"
            << "  --revision REF          Branch, tag or commit SHA (default: main)\n"
            << "  --file NAME             Choose a file when several match the quant\n"
            << "  --cache-dir PATH        Cache root (default: <home>/.cache/llmx)\n"
            << "  --parallel N            Streams per file, 1..16 (default: 4)\n\n"
            << "Requires curl 8.4+. HF_TOKEN supplies gated-repo credentials.\n"
            << "The verified local path goes to stdout; progress goes to stderr.\n\n"
            << "Example: llmx pull Qwen/Qwen3-0.6B-GGUF:Q8_0 --parallel 4\n";
    } else if (command == "logits" || command == "perplexity") {
        const bool ppl = command == "perplexity";
        std::cout << (ppl ? "Score next-token likelihoods over text or bounded windows.\n\n"
                         : "Print the highest next-token logits after a prompt.\n\n")
            << "Usage: llmx " << command << " <model.gguf> \"<text>\" [options]\n";
        if (!ppl) std::cout
            << "       llmx logits <model.gguf> <path> --file [options]\n";
        if (ppl) std::cout
            << "       llmx perplexity <model.gguf> --file <path> [options]\n"
            << "\nScoring options:\n"
            << "  --file PATH, -f         UTF-8 input file, immediately after the model\n"
            << "  --ctx-size N, -c        Window tokens (default: model context)\n"
            << "  --chunks N              Maximum windows (default: all)\n"
            << "  --per-token             Score through decode; default uses batched passes\n"
            << "  --verbose               Show scoring phase and actual worker count\n";
        else std::cout << "\nOptions:\n  --top N                 Number of logits to print (default: 10)\n"
            << "  --file                  Read the text from the file named in its place\n"
            << "  --then-ids PATH         Append these whitespace-separated token IDs\n"
            << "  --last N                Print each of the last N positions, one per line\n";
        model_options(ppl);
        std::cout << "\nExample: llmx " << command << " model.gguf "
            << (ppl ? "--file corpus.txt --ctx-size 512 --chunks 4\n"
                    : "\"The capital of France is\" --top 10\n");
    } else if (command == "bench") {
        std::cout << "Measure synthetic kernels or a real model after warm-up.\n\n"
            << "Usage: llmx bench [options]\n"
            << "       llmx bench --model <model.gguf> [options]\n\n"
            << "Benchmark options:\n"
            << "  --p N                   Prompt tokens (default: 64)\n"
            << "  --n N                   Decode tokens (default: 64)\n"
            << "  --size N                Synthetic matrix width, multiple of 32 (default: 1024)\n"
            << "  --iters N               Synthetic kernel repetitions (default: 5)\n"
            << "  --model PATH            Benchmark this model instead of synthetic weights\n"
            << "  --r N                   Real-model repetitions (default: 3)\n"
            << "  --seqs N                Sequences decoding together, a pass one token of each (default: 1)\n"
            << "  --depth N               History of N tokens, filled untimed, that each test runs after (default: 0)\n"
            << "  --profile               Real-model device kernel timing and statistics\n";
        model_options(false);
        std::cout << "\nCache options apply only with --model.\n"
            << "Example: llmx bench --model model.gguf --p 512 --n 128 --r 3\n";
    } else if (command == "quantize") {
        std::cout << "Convert raw F32 tensors into a quantized GGUF file.\n\n"
            << "Usage: llmx quantize <model.json> <model.bin> <out.gguf> [q8_0|q4_0]\n\n"
            << "Default quant: q8_0. Input row widths must be divisible by 32.\n"
            << "The JSON describes tensor names/shapes; the binary contains F32 values.\n"
            << "Example: llmx quantize model.json model.bin model.gguf q8_0\n";
    } else if (command == "dequantize") {
        std::cout << "Export supported GGUF tensors as JSON metadata and F32 values.\n\n"
            << "Usage: llmx dequantize <in.gguf> <out.json> <out.bin>\n\n"
            << "Example: llmx dequantize model.gguf model.json model.bin\n";
    } else if (command == "info") {
        std::cout << "Show GGUF metadata, tensor types and dimensions.\n\n"
            << "Usage: llmx info <model.gguf>\n\n"
            << "For a sharded model, pass its first shard.\n"
            << "Example: llmx info model.gguf\n";
    } else if (command == "tokenize") {
        std::cout << "Encode text with the model's tokenizer.\n\n"
            << "Usage: llmx tokenize <model.gguf> \"<text>\"\n\n"
            << "Example: llmx tokenize model.gguf \"hello world\"\n";
    } else {
        std::cout << "Decode comma- or space-separated token IDs.\n\n"
            << "Usage: llmx detokenize <model.gguf> <ids>\n\n"
            << "Example: llmx detokenize model.gguf \"1,2,3\"\n";
    }
    std::cout << "\n-h, --help shows this page. Full reference: docs/USAGE.md\n";
    return true;
}

} // namespace

#if defined(_WIN32)
// On Windows argv arrives in the system ANSI codepage, which cannot represent most non-ASCII text -- a Japanese or Cyrillic prompt is mangled before it reaches us.
// Re-read the command line as UTF-16 and convert to UTF-8 so text arguments survive.
// Storage is owned by the caller and must outlive argv.
static bool utf8_argv(int& argc, char**& argv,
                      std::vector<std::string>& store, std::vector<char*>& ptrs) {
    int wargc = 0;
    LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (!wargv) return false;
    store.reserve((size_t)wargc);
    for (int i = 0; i < wargc; i++) {
        int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string s(n > 0 ? (size_t)(n - 1) : 0, '\0');
        if (n > 1) WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, &s[0], n, nullptr, nullptr);
        store.push_back(std::move(s));
    }
    LocalFree(wargv);
    ptrs.reserve(store.size() + 1);
    for (auto& s : store) ptrs.push_back(&s[0]);
    ptrs.push_back(nullptr);
    argc = wargc;
    argv = ptrs.data();
    return true;
}
#endif

int main(int argc, char** argv) {
#if defined(_WIN32)
    std::vector<std::string> argv_store;
    std::vector<char*> argv_ptrs;
    utf8_argv(argc, argv, argv_store, argv_ptrs);
    SetConsoleOutputCP(CP_UTF8);
#endif
    try {
        // Populate the quant registry here, since info and dequantize never build a Model.
        quant::register_builtins();
        if (argc < 2) { print_usage(); return 1; }
        std::string cmd = argv[1];
        if (argc == 2 && (cmd == "--help" || cmd == "-h")) {
            print_usage();
            return 0;
        }
        if (argc == 3 && (std::string(argv[2]) == "--help" || std::string(argv[2]) == "-h")) {
            if (print_usage(cmd)) return 0;
            std::cerr << "unknown command: " << cmd << '\n';
            return 2;
        }
        if (cmd == "--version") {
            std::cout << "llmx " << LLMX_VERSION_STRING << "\n";
            return 0;
        }

        if (cmd == "pull") {
            if (argc < 3) throw std::runtime_error("usage: llmx pull <owner/repo>:<quant> [flags...]");
            const std::string target = argv[2];
            const auto colon = target.find(':');
            if (colon == std::string::npos) throw std::runtime_error("pull: expected owner/repository:quant");
            hub::PullOptions options;
            options.repo = target.substr(0, colon);
            options.quant = target.substr(colon + 1);
#ifdef _WIN32
            char* token = nullptr;
            size_t token_size = 0;
            if (_dupenv_s(&token, &token_size, "HF_TOKEN")) throw std::runtime_error("pull: cannot read HF_TOKEN");
            std::unique_ptr<char, decltype(&std::free)> token_owner(token, &std::free);
            if (token) options.token = token;
#else
            if (const char* token = std::getenv("HF_TOKEN")) options.token = token;
#endif
            for (int i = 3; i < argc; ++i) {
                const std::string flag = argv[i];
                if (flag != "--revision" && flag != "--file" && flag != "--cache-dir" && flag != "--parallel")
                    throw std::runtime_error("pull: unknown flag: " + flag);
                if (i + 1 == argc) throw std::runtime_error("pull: missing value for " + flag);
                const std::string value = argv[++i];
                if (flag == "--revision") options.revision = value;
                else if (flag == "--file") options.filename = value;
                else if (flag == "--cache-dir") {
                    if (value.empty()) throw std::runtime_error("pull: empty cache path");
                    options.cache = std::filesystem::u8path(value);
                } else {
                    if (value.empty() || value.size() > 2 || value.find_first_not_of("0123456789") != std::string::npos)
                        throw std::runtime_error("pull: --parallel must be between 1 and 16");
                    options.parallel = unsigned(std::stoul(value));
                }
            }
            const auto path = hub::pull(options, [](const std::string& message) { std::cerr << message << '\n'; });
            std::cout << path.u8string() << '\n';
            return 0;
        }


        if (cmd == "generate" || cmd == "chat") {
            if (argc < 3) {
                if (cmd == "chat") std::cerr << "usage: llmx chat <model.gguf> [--system \"<text>\"] [flags...]\n";
                else std::cerr << "usage: llmx generate <model.gguf> \"<prompt>\" [flags...]\n";
                return 2;
            }
            infer::GenParams gp;
            std::string system = "You are a helpful assistant.";
            std::string prompt;
            bool have_prompt = false;
            for (int i = 3; i < argc; i++) {
                std::string a = argv[i];
                if (a == "-n" || a == "--max-tokens") gp.max_tokens = (i + 1 < argc) ? std::atoi(argv[++i]) : gp.max_tokens;
                else if (a == "--temp") gp.temp = (i + 1 < argc) ? (float)std::atof(argv[++i]) : gp.temp;
                else if (a == "--topk") gp.top_k = (i + 1 < argc) ? std::atoi(argv[++i]) : gp.top_k;
                else if (a == "--topp") gp.top_p = (i + 1 < argc) ? (float)std::atof(argv[++i]) : gp.top_p;
                else if (a == "--penalty") gp.penalty = (i + 1 < argc) ? (float)std::atof(argv[++i]) : gp.penalty;
                else if (a == "--seed") gp.seed = (i + 1 < argc) ? std::strtoull(argv[++i], nullptr, 0) : gp.seed;
                else if (a == "--stop") gp.stop = (i + 1 < argc) ? argv[++i] : gp.stop;
                else if (exec_flag(argc, argv, i, gp)) {}
                else if (a == "--threads-batch" || a == "-tb") gp.threads_batch = (i + 1 < argc) ? std::atoi(argv[++i]) : gp.threads_batch;
                else if (a == "--system" && cmd == "chat") system = (i + 1 < argc) ? argv[++i] : system;
                else if (a == "--verbose") gp.show_prompt_tokens = true;
                else if (!a.empty() && a[0] == '-') { std::cerr << "unknown flag: " << a << "\n"; return 2; }
                else { prompt = a; have_prompt = true; }
            }
            if (gp.max_tokens <= 0) { std::cerr << cmd << ": --max-tokens must be positive\n"; return 2; }
            if (cmd == "generate") {
                if (!have_prompt) { std::cerr << "generate requires a prompt\n"; return 2; }
                return cmd_generate(argv[2], prompt, gp);
            }
            return cmd_chat(argv[2], system, gp);
        }

        if (cmd == "perplexity") {
            if (argc < 4) { std::cerr << "usage: llmx perplexity <model.gguf> (\"<text>\" | --file <path>) [flags...]\n"; return 2; }
            infer::GenParams gp;
            int context_size = 0, chunks = 0;
            bool per_token = false;
            const bool from_file = std::string(argv[3]) == "--file" || std::string(argv[3]) == "-f";
            if (from_file && argc < 5) { std::cerr << "perplexity: --file requires a path\n"; return 2; }
            for (int i = from_file ? 5 : 4; i < argc; i++) {
                std::string a = argv[i];
                if (a == "--file" || a == "-f") {
                    std::cerr << "perplexity: use either inline text or one --file <path> immediately after the model\n";
                    return 2;
                }
                if (a == "--ctx-size" || a == "-c" || a == "--chunks") {
                    if (i + 1 >= argc) { std::cerr << a << " requires a positive integer\n"; return 2; }
                    const std::string value = argv[++i];
                    unsigned long long n = 0;
                    for (char c : value) {
                        if (c < '0' || c > '9' || n > (unsigned long long)std::numeric_limits<int>::max() / 10) {
                            std::cerr << a << " requires a positive integer\n"; return 2;
                        }
                        n = n * 10 + (unsigned)(c - '0');
                    }
                    if (n == 0 || n > (unsigned long long)std::numeric_limits<int>::max()) {
                        std::cerr << a << " requires a positive integer\n"; return 2;
                    }
                    if (a == "--chunks") chunks = (int)n;
                    else context_size = (int)n;
                }
                else if (a == "--per-token") per_token = true;
                else if (a == "--verbose") gp.show_prompt_tokens = true;
                else if (exec_flag(argc, argv, i, gp)) {}
                else if (a == "--threads-batch" || a == "-tb") gp.threads_batch = (i + 1 < argc) ? std::atoi(argv[++i]) : gp.threads_batch;
                else { std::cerr << "unknown flag: " << a << "\n"; return 2; }
            }
            const std::string text = from_file ? read_perplexity_file(argv[4]) : argv[3];
            return cmd_perplexity(argv[2], text, gp, context_size, chunks, per_token);
        }

        if (cmd == "logits") {
            if (argc < 4) { std::cerr << "usage: llmx logits <model.gguf> \"<text>\" | <file> --file [--then-ids FILE] [--last N] [--top N] [--threads N] [--device D]\n"; return 2; }
            infer::GenParams gp;
            int topn = 10;
            bool from_file = false;
            std::string then_ids;
            size_t last = 0;
            for (int i = 4; i < argc; i++) {
                std::string a2 = argv[i];
                if (a2 == "--top") topn = (i + 1 < argc) ? std::atoi(argv[++i]) : topn;
                else if (a2 == "--file") from_file = true;
                else if (a2 == "--then-ids") then_ids = (i + 1 < argc) ? argv[++i] : then_ids;
                else if (a2 == "--last") last = (i + 1 < argc) ? (size_t)std::max(0, std::atoi(argv[++i])) : last;
                else if (exec_flag(argc, argv, i, gp)) {}
                else { std::cerr << "unknown flag: " << a2 << "\n"; return 2; }
            }
            if (topn <= 0) topn = 10;
            const std::string text = from_file ? read_perplexity_file(argv[3]) : argv[3];
            return cmd_logits(argv[2], text, topn, gp, then_ids, last);
        }

        if (cmd == "tokenize") {
            if (argc != 4) { std::cerr << "usage: llmx tokenize <model.gguf> \"<text>\"\n"; return 2; }
            return cmd_tokenize(argv[2], argv[3]);
        }
        if (cmd == "detokenize") {
            if (argc != 4) { std::cerr << "usage: llmx detokenize <model.gguf> <id1,id2,...>\n"; return 2; }
            return cmd_detokenize(argv[2], argv[3]);
        }

        if (cmd == "quantize") {
            if (argc < 5 || argc > 6) { std::cerr << "usage: llmx quantize <model.json> <model.bin> <out.gguf> [q8_0|q4_0]\n"; return 2; }
            std::string type = (argc == 6) ? argv[5] : "q8_0";
            if (type != "q8_0" && type != "q4_0") { std::cerr << "unknown quant type: " << type << " (expected q8_0 or q4_0)\n"; return 2; }
            return cmd_quantize(argv[2], argv[3], argv[4], type);
        }
        if (cmd == "dequantize") {
            if (argc != 5) { std::cerr << "usage: llmx dequantize <in.gguf> <out.json> <out.bin>\n"; return 2; }
            return cmd_dequantize(argv[2], argv[3], argv[4]);
        }
        if (cmd == "info") {
            if (argc != 3) { std::cerr << "usage: llmx info <in.gguf>\n"; return 2; }
            return cmd_info(argv[2]);
        }
        if (cmd == "serve") {
            if (argc < 3) { print_usage(); return 1; }
            server::Config cfg;
            infer::GenParams gp;
            for (int i = 3; i < argc; i++) {
                std::string a = argv[i];
                if (a == "--host") cfg.host = (i + 1 < argc) ? argv[++i] : cfg.host;
                else if (a == "--port") cfg.port = (i + 1 < argc) ? (uint16_t)std::atoi(argv[++i]) : cfg.port;
                else if (a == "--max-seqs") cfg.max_seqs = (i + 1 < argc) ? (size_t)std::atoi(argv[++i]) : cfg.max_seqs;
                else if (a == "--max-queue") cfg.max_queue = (i + 1 < argc) ? (size_t)std::atoi(argv[++i]) : cfg.max_queue;
                else if (a == "--ctx-size" || a == "-c") gp.kv_tokens = (i + 1 < argc) ? std::atoi(argv[++i]) : gp.kv_tokens;
                else if (exec_flag(argc, argv, i, gp)) {}
                else { std::cerr << "unknown flag: " << a << "\n"; return 2; }
            }
            if (cfg.max_seqs == 0) { std::cerr << "serve: --max-seqs must be positive\n"; return 2; }
            if (gp.kv_tokens < 0) { std::cerr << "serve: --ctx-size must be positive\n"; return 2; }
            return cmd_serve(argv[2], cfg, gp);
        }
        if (cmd == "bench") {
            int size = 1024, iters = 5, prefill = 64, decode = 64, repeats = 3, seqs = 1, depth = 0;
            bool profile = false;
            std::string model_path, model_only;   // model_only: the first flag given that only --model reads
            infer::GenParams gp;
            for (int i = 2; i < argc; i++) {
                std::string a = argv[i];
                if (a == "--size") size = (i + 1 < argc) ? std::atoi(argv[++i]) : size;
                else if (a == "--iters") iters = (i + 1 < argc) ? std::atoi(argv[++i]) : iters;
                else if (a == "--p") prefill = (i + 1 < argc) ? std::atoi(argv[++i]) : prefill;
                else if (a == "--n") decode = (i + 1 < argc) ? std::atoi(argv[++i]) : decode;
                else if (a == "--model") model_path = (i + 1 < argc) ? argv[++i] : model_path;
                else if (exec_flag(argc, argv, i, gp)) { if (a != "--device" && a != "--threads" && model_only.empty()) model_only = a; }
                else if (a == "--r") { repeats = (i + 1 < argc) ? std::atoi(argv[++i]) : repeats; if (model_only.empty()) model_only = a; }
                else if (a == "--seqs") { seqs = (i + 1 < argc) ? std::atoi(argv[++i]) : seqs; if (model_only.empty()) model_only = a; }
                else if (a == "--depth") { depth = (i + 1 < argc) ? std::atoi(argv[++i]) : depth; if (model_only.empty()) model_only = a; }
                else if (a == "--profile") { profile = true; if (model_only.empty()) model_only = a; }
                else { std::cerr << "unknown flag: " << a << "\n"; return 2; }
            }
            // The synthetic bench times one backend's kernels and reads only --device, --threads, --size, --iters, --p and --n.
            if (model_path.empty() && !model_only.empty()) { std::cerr << "bench: " << model_only << " takes --model\n"; return 2; }
            if (size <= 0 || size % 32 != 0) { std::cerr << "bench: --size must be positive and a multiple of 32\n"; return 2; }
            // Each of these divides a measured duration or token count.
            if (iters <= 0 || prefill <= 0 || decode <= 0 || repeats <= 0 || seqs <= 0) {
                std::cerr << "bench: --iters, --p, --n, --r and --seqs must be positive\n"; return 2;
            }
            if (depth < 0) { std::cerr << "bench: --depth must not be negative\n"; return 2; }
            // Batched decode already starts after each sequence's prompt.
            if (depth > 0 && seqs > 1) { std::cerr << "bench: --depth takes one sequence\n"; return 2; }
            if (!model_path.empty()) return cmd_bench_model(model_path, gp, prefill, decode, repeats, profile, seqs, depth);
            return cmd_bench(size, iters, gp.threads, prefill, decode, gp.device);
        }
        print_usage();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
