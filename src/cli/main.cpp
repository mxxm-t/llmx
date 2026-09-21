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
#include "core/fp16.hpp"
#include "core/json.hpp"
#include "hub/pull.hpp"
#include "format/gguf.hpp"
#include "format/format.hpp"
#include "quant/quant.hpp"
#include "backends/cpu/cpu_backend.hpp"
#include "tokenizer/tokenizer.hpp"
#include "inference/sampler.hpp"
#include "inference/generate.hpp"
#include "inference/perplexity.hpp"
#include "inference/chat.hpp"
#include "model/arch_qwen.hpp"

// llmx CLI. This file is intentionally a thin dispatcher: format logic lives in
// format/, quantization in quant/, inference in inference/, and the model in
// model/. The only code that belongs here is argument parsing and glue.

namespace {

bool show_progress(const infer::GenParams& gp) {
#if defined(_WIN32)
    return gp.show_prompt_tokens || _isatty(_fileno(stderr));
#else
    return gp.show_prompt_tokens || isatty(fileno(stderr));
#endif
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
        const uint8_t* raw = m.tensor_data(i);
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

// Type names come from the quant registry, so a new quant type shows up in
// `info` without touching the CLI.
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
    // decode() rejects an out-of-range id, but only the caller knows which
    // id it was and how large the vocabulary is.
    for (uint32_t id : ids)
        if (id >= t.vocab.size())
            throw std::runtime_error("detokenize: token id " + std::to_string(id) +
                                     " is outside the vocabulary of " +
                                     std::to_string(t.vocab.size()) + " tokens");
    std::cout << t.decode(ids) << "\n";
    return 0;
}

// The backend a --device spec names. "cpu" is the default; "vulkan:N" is
// device N as the loader lists them, in a build with that backend. Any
// other spec, or a device the build lacks, is an error the user can act on
// rather than a silent fallback.
backend::BackendPtr make_backend(const std::string& spec) {
    if (spec == "cpu") return backend::make_cpu_backend();
    const size_t colon = spec.find(':');
    const std::string name = spec.substr(0, colon);
    int index = 0;
    if (colon != std::string::npos) {
        const std::string rest = spec.substr(colon + 1);
        if (rest.empty() || rest.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error("--device: invalid device index in '" + spec + "'");
        index = std::atoi(rest.c_str());
    }
    if (name == "vulkan") {
#if LLMX_HAS_BACKEND_VULKAN
        return backend::make_vulkan_backend(index);
#else
        throw std::runtime_error("--device vulkan: this build has no Vulkan backend (LLMX_HAS_BACKEND_VULKAN)");
#endif
    }
    throw std::runtime_error("--device: unknown backend '" + name + "' (cpu, vulkan:N)");
}

int cmd_generate(const std::string& model_path, const std::string& prompt,
                 const infer::GenParams& gp) {
    const bool progress = show_progress(gp);
    gguf::GGUFModel m = load_model(model_path, progress);
    bpe::Tokenizer tok(m);
    infer::Model model(m, make_backend(gp.device));
    if (gp.threads > 0) model.set_threads(gp.threads);
    const int decode_threads = model.threads_available();
    model.set_ubatch(gp.ubatch);
    infer::RNG rng;
    if (gp.seed) rng.seed(gp.seed);

    std::vector<uint32_t> ids = tok.encode(prompt);
    if (ids.empty()) throw std::runtime_error("generate: empty prompt");

    // Prefill is compute bound and wants every thread; decode is memory
    // bandwidth bound and usually peaks well below the logical core count,
    // so the two phases get their own thread counts (-t / -tb).
    const int tb = (gp.threads_batch > 0) ? gp.threads_batch : decode_threads;
    model.set_threads(tb);
    if (gp.show_prompt_tokens) std::cerr << "threads: prefill " << model.threads_available() << "\n";
    if (progress) std::cerr << "Processing " << ids.size() << " prompt tokens...\n";
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> logits = infer::prefill(model, ids);
    model.set_threads(decode_threads);
    double pp_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (gp.show_prompt_tokens) std::cout << "prompt tokens: " << ids.size() << "\n";
    // Two decimals: at a few tok/s an integer print rounds a 20% change away.
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

// Print the top-N next-token logits for a prompt. This exists for the
// correctness gate: it is the only way to compare llmx against a
// full-precision reference at the level where errors actually appear, rather
// than through sampled text. See docs/ROADMAP.md #8.
int cmd_logits(const std::string& model_path, const std::string& text,
               int topn, const infer::GenParams& gp) {
    gguf::GGUFModel m = gguf::read_gguf(model_path);
    bpe::Tokenizer tok(m);
    infer::Model model(m, make_backend(gp.device));
    if (gp.threads > 0) model.set_threads(gp.threads);
    model.set_ubatch(gp.ubatch);

    std::vector<uint32_t> ids = tok.encode(text);
    if (ids.empty()) throw std::runtime_error("logits: empty prompt");
    std::vector<float> logits = infer::prefill(model, ids);

    std::vector<std::pair<float, uint32_t>> ranked;
    ranked.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); i++)
        ranked.push_back({ logits[i], (uint32_t)i });
    if (topn > (int)ranked.size()) topn = (int)ranked.size();
    std::partial_sort(ranked.begin(), ranked.begin() + topn, ranked.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    printf("tokens: %zu\n", ids.size());
    for (int i = 0; i < topn; i++)
        printf("%u %.6f\n", ranked[(size_t)i].second, ranked[(size_t)i].first);
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
                   const infer::GenParams& gp, int context_size, int chunks) {
    gguf::GGUFModel m = gguf::read_gguf(model_path);
    bpe::Tokenizer tok(m);
    infer::Model model(m, make_backend(gp.device));
    if (gp.threads > 0) model.set_threads(gp.threads);
    model.set_ubatch(gp.ubatch);

    std::vector<uint32_t> ids = tok.encode(text);
    const auto result = infer::perplexity(model, ids, context_size, chunks);
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
    gguf::GGUFModel m = load_model(model_path, progress);
    bpe::Tokenizer tok(m);
    infer::Model model(m, make_backend(gp.device));
    if (gp.threads > 0) model.set_threads(gp.threads);
    const int decode_threads = model.threads_available();
    model.set_ubatch(gp.ubatch);
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
        // Reuse only an exact prefix; an unchanged prompt also needs fresh
        // logits because generate() does not retain its final distribution.
        if (cached_ids.size() >= gen_ids.size() ||
            !std::equal(cached_ids.begin(), cached_ids.end(), gen_ids.begin())) {
            model.reset();
            cached_ids.clear();
        }
        model.set_threads(gp.threads_batch > 0 ? gp.threads_batch : decode_threads);
        if (gp.show_prompt_tokens) std::cerr << "threads: prefill " << model.threads_available() << "\n";
        if (progress) std::cerr << "Processing " << gen_ids.size() - cached_ids.size() << " prompt tokens...\n";
        std::vector<float> logits = infer::prefill(model,
            std::vector<uint32_t>(gen_ids.begin() + cached_ids.size(), gen_ids.end()));
        cached_ids = std::move(gen_ids);

        model.set_threads(decode_threads);
        if (gp.show_prompt_tokens) std::cerr << "threads: decode " << model.threads_available() << "\n";
        if (progress) std::cerr << "Generating...\n";
        std::vector<uint32_t> reply = infer::generate(model, tok, gp, rng, logits, emit_text);
        std::cout << "\n" << std::flush;
        // A stop match may return its final token without feeding it. EOS is
        // excluded; the next rendered turn supplies its own closing tokens.
        const size_t fed = (size_t)model.n_tokens() - cached_ids.size();
        cached_ids.insert(cached_ids.end(), reply.begin(), reply.begin() + fed);

        messages.push_back({ "assistant", tok.decode(reply) });
    }
    return 0;
}

// Build a small random Qwen3 model in memory for end-to-end prefill/decode TPS
// measurement. Matrices are Q8_0, norms F32 (matching what infer::Model expects).
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

// Micro-benchmark of the backend hot paths (matmul, RMSNorm, norm+RoPE) plus
// end-to-end prefill/decode TPS on a synthetic Qwen3 model. Used by
// tests/perf.py as the perf-regression gate for hot-path changes.
int cmd_bench(int size, int iters, int threads, int prefill, int decode,
              const std::string& device) {
    auto b = make_backend(device);
    if (threads > 0) b->set_threads(threads);
    std::cout << "bench: threads " << b->threads_available() << "\n";

    // Square matmul: mat is [nin, nout] = [size, size]. x is the input
    // (length nin), out the result (length nout). nout rows, each nin/32 blocks.
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

    // The op the model runs: one row of one head of `size` floats, at
    // position 0 of a one-entry table.
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

// The matched real-model measurement: a warm-up of each test, then R
// repeats of prompt processing P tokens in one batch into an empty history
// and of generating G tokens one at a time from an empty history, model
// time only, token ids fixed and sampling excluded. Reported as mean and
// standard deviation of tokens per second, so a reference runtime's
// figures for the same P and G compare directly.
int cmd_bench_model(const std::string& path, const std::string& device, int threads,
                    int P, int G, int R) {
    gguf::GGUFModel m = load_model(path, false);
    infer::Model model(m, make_backend(device));
    if (threads > 0) model.set_threads(threads);
    // Ids below 1000 exist in every vocabulary the runtime loads.
    auto ids_from = [](uint32_t seed, size_t n) {
        std::vector<uint32_t> ids(n);
        for (auto& t : ids) { seed = seed * 1664525u + 1013904223u; t = (seed >> 8) % 1000; }
        return ids;
    };
    const std::vector<uint32_t> prompt = ids_from(12345u, (size_t)P), gen = ids_from(777u, (size_t)G);
    using clock = std::chrono::steady_clock;
    auto ms_since = [](clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    };
    auto pp = [&] {
        model.reset();
        const auto t0 = clock::now();
        model.prefill(prompt);
        return (double)P / (ms_since(t0) / 1e3);
    };
    auto tg = [&] {
        model.reset();
        const auto t0 = clock::now();
        for (uint32_t t : gen) model.step((int)t);
        return (double)G / (ms_since(t0) / 1e3);
    };
    auto report = [&](const char* what, int n, const std::vector<double>& v) {
        double mean = 0, var = 0;
        for (double x : v) mean += x;
        mean /= (double)v.size();
        for (double x : v) var += (x - mean) * (x - mean);
        const double sd = v.size() > 1 ? std::sqrt(var / (double)(v.size() - 1)) : 0.0;
        printf("bench: %s%d  %8.2f +- %.2f tok/s  (%zu runs)\n", what, n, mean, sd, v.size());
    };
    pp();
    tg();
    std::vector<double> ppv, tgv;
    for (int r = 0; r < R; r++) ppv.push_back(pp());
    for (int r = 0; r < R; r++) tgv.push_back(tg());
    report("pp", P, ppv);
    report("tg", G, tgv);
    return 0;
}

void print_usage() {
    std::cout
        << "llmx " << LLMX_VERSION_STRING << " - ground-up GGUF runtime (no external libs)\n"
        << "\n"
        << "Usage:\n"
        << "  llmx --version  print release version and build revision\n"
        << "  llmx pull       <owner/repo>:<quant> [--revision <ref>] [--file <name>]\n"
        << "                  [--cache-dir <path>] [--parallel N] (default: 4, range: 1..16)\n"
        << "    pull uses curl 8.4+ for HTTPS; HF_TOKEN supplies gated-repo credentials\n"
        << "  llmx quantize   <model.json> <model.bin> <out.gguf> [q8_0|q4_0]\n"
        << "  llmx dequantize <in.gguf> <out.json> <out.bin>\n"
        << "  llmx info       <in.gguf>\n"
        << "  llmx logits     <in.gguf> \"<text>\" [--top N]\n"
        << "  llmx tokenize   <in.gguf> \"<text>\"\n"
        << "  llmx detokenize <in.gguf> <id1,id2,...>\n"
        << "  llmx perplexity <in.gguf> \"<text>\" [flags...]\n"
        << "  llmx perplexity <in.gguf> -f/--file <path> [flags...]\n"
        << "    perplexity flags: -c/--ctx-size N  window tokens (default: model context)\n"
        << "                      --chunks N  maximum windows (default: all)\n"
        << "  llmx generate   <in.gguf> \"<prompt>\" [flags...]\n"
        << "  llmx chat       <in.gguf> [--system \"<text>\"] [flags...]\n"
        << "  llmx bench      [--size N] [--iters N] [--threads N] [--p N] [--n N] [--device D]\n"
        << "  llmx bench      --model <in.gguf> [--p N] [--n N] [--r N] [--threads N] [--device D]\n"
        << "                  (warm-up, then R repeats of pp N and tg N, model time only)\n"
        << "    flags: -n/--max-tokens N  --temp F  --topk N  --topp F  --penalty F  --threads N\n"
        << "           --device D  backend: cpu (default) or vulkan:N in a build with it\n"
        << "           --ubatch N  prefill physical batch (default 512)\n"
        << "           -tb/--threads-batch N  threads for prefill (default: --threads)\n"
        << "           --seed N  --stop \"<text>\"  --think (show reasoning)  --verbose\n"
        << "           --verbose reports prompt tokens, thread counts, KV bytes and loading/processing status\n";
}

} // namespace

#if defined(_WIN32)
// On Windows argv arrives in the system ANSI codepage, which cannot represent
// most non-ASCII text -- a Japanese or Cyrillic prompt is mangled before it
// reaches us. Re-read the command line as UTF-16 and convert to UTF-8 so text
// arguments survive. Storage is owned by the caller and must outlive argv.
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
        // Populate the quant registry once, here, rather than relying on a
        // Model being constructed. info and dequantize never build one, so
        // they used to run against an empty registry.
        quant::register_builtins();
        if (argc < 2) { print_usage(); return 1; }
        std::string cmd = argv[1];
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
                std::cerr << "usage: llmx " << cmd << " <model.gguf> [--system \"<text>\"] [flags...]\n";
                std::cerr << "       llmx generate <model.gguf> \"<prompt>\" [flags...]\n";
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
                else if (a == "--threads") gp.threads = (i + 1 < argc) ? std::atoi(argv[++i]) : gp.threads;
                else if (a == "--ubatch") gp.ubatch = (i + 1 < argc) ? std::atoi(argv[++i]) : gp.ubatch;
                else if (a == "--threads-batch" || a == "-tb") gp.threads_batch = (i + 1 < argc) ? std::atoi(argv[++i]) : gp.threads_batch;
                else if (a == "--device") gp.device = (i + 1 < argc) ? argv[++i] : gp.device;
                else if (a == "--system") system = (i + 1 < argc) ? argv[++i] : system;
                else if (a == "--verbose") gp.show_prompt_tokens = true;
                else if (a == "--think") gp.show_thinking = true;
                else if (!a.empty() && a[0] == '-') { std::cerr << "unknown flag: " << a << "\n"; return 2; }
                else { prompt = a; have_prompt = true; }
            }
            if (gp.max_tokens <= 0) gp.max_tokens = 32;
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
                else if (a == "--threads") gp.threads = (i + 1 < argc) ? std::atoi(argv[++i]) : gp.threads;
                else if (a == "--ubatch") gp.ubatch = (i + 1 < argc) ? std::atoi(argv[++i]) : gp.ubatch;
                else if (a == "--threads-batch" || a == "-tb") gp.threads_batch = (i + 1 < argc) ? std::atoi(argv[++i]) : gp.threads_batch;
                else if (a == "--device") gp.device = (i + 1 < argc) ? argv[++i] : gp.device;
                else { std::cerr << "unknown flag: " << a << "\n"; return 2; }
            }
            const std::string text = from_file ? read_perplexity_file(argv[4]) : argv[3];
            return cmd_perplexity(argv[2], text, gp, context_size, chunks);
        }

        if (cmd == "logits") {
            if (argc < 4) { std::cerr << "usage: llmx logits <model.gguf> \"<text>\" [--top N] [--threads N] [--device D]\n"; return 2; }
            infer::GenParams gp;
            int topn = 10;
            for (int i = 4; i < argc; i++) {
                std::string a2 = argv[i];
                if (a2 == "--top") topn = (i + 1 < argc) ? std::atoi(argv[++i]) : topn;
                else if (a2 == "--threads") gp.threads = (i + 1 < argc) ? std::atoi(argv[++i]) : gp.threads;
                else if (a2 == "--ubatch") gp.ubatch = (i + 1 < argc) ? std::atoi(argv[++i]) : gp.ubatch;
                else if (a2 == "--device") gp.device = (i + 1 < argc) ? argv[++i] : gp.device;
                else { std::cerr << "unknown flag: " << a2 << "\n"; return 2; }
            }
            if (topn <= 0) topn = 10;
            return cmd_logits(argv[2], argv[3], topn, gp);
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
        if (cmd == "bench") {
            int size = 1024, iters = 5, threads = 0, prefill = 64, decode = 64, repeats = 3;
            std::string device = "cpu", model_path;
            for (int i = 2; i < argc; i++) {
                std::string a = argv[i];
                if (a == "--size") size = (i + 1 < argc) ? std::atoi(argv[++i]) : size;
                else if (a == "--device") device = (i + 1 < argc) ? argv[++i] : device;
                else if (a == "--iters") iters = (i + 1 < argc) ? std::atoi(argv[++i]) : iters;
                else if (a == "--threads") threads = (i + 1 < argc) ? std::atoi(argv[++i]) : threads;
                else if (a == "--p") prefill = (i + 1 < argc) ? std::atoi(argv[++i]) : prefill;
                else if (a == "--n") decode = (i + 1 < argc) ? std::atoi(argv[++i]) : decode;
                else if (a == "--r") repeats = (i + 1 < argc) ? std::atoi(argv[++i]) : repeats;
                else if (a == "--model") model_path = (i + 1 < argc) ? argv[++i] : model_path;
                else { std::cerr << "unknown flag: " << a << "\n"; return 2; }
            }
            if (size <= 0 || size % 32 != 0) { std::cerr << "bench: --size must be positive and a multiple of 32\n"; return 2; }
            // Each of these divides a measured duration or token count.
            if (iters <= 0 || prefill <= 0 || decode <= 0 || repeats <= 0) {
                std::cerr << "bench: --iters, --p, --n and --r must be positive\n"; return 2;
            }
            if (!model_path.empty()) return cmd_bench_model(model_path, device, threads, prefill, decode, repeats);
            return cmd_bench(size, iters, threads, prefill, decode, device);
        }
        print_usage();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
