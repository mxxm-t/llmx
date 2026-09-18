#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <random>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#pragma comment(lib, "shell32.lib")
#endif

#include "config.hpp"
#include "core/fp16.hpp"
#include "core/json.hpp"
#include "format/gguf.hpp"
#include "format/format.hpp"
#include "quant/quant.hpp"
#include "backends/cpu/cpu_backend.hpp"
#include "tokenizer/tokenizer.hpp"
#include "inference/sampler.hpp"
#include "inference/generate.hpp"
#include "inference/chat.hpp"
#include "model/arch_qwen.hpp"

// llmx CLI. This file is intentionally a thin dispatcher: format logic lives in
// format/, quantization in quant/, inference in inference/, and the model in
// model/. The only code that belongs here is argument parsing and glue.

namespace {

// ---------------------------------------------------------------------------
// model.json / model.bin helper structures (quantize input)
// ---------------------------------------------------------------------------

struct JsonTensor {
    std::string name;
    std::vector<uint64_t> shape; // shape[0] -> ne[0] (fastest dim)
};

std::vector<JsonTensor> parse_model_json(const jmini::Value& root) {
    std::vector<JsonTensor> out;
    const jmini::Value* tensors = root.get("tensors");
    if (!tensors || !tensors->isArray())
        throw std::runtime_error("model.json: missing \"tensors\" array");
    for (const auto& t : tensors->asArray()) {
        JsonTensor jt;
        const jmini::Value* name = t.get("name");
        if (!name || !name->isString())
            throw std::runtime_error("model.json: tensor missing \"name\" string");
        jt.name = name->asString();
        const jmini::Value* shape = t.get("shape");
        if (!shape || !shape->isArray())
            throw std::runtime_error("model.json: tensor missing \"shape\" array: " + jt.name);
        for (const auto& d : shape->asArray()) {
            if (!d.isNumber())
                throw std::runtime_error("model.json: shape dim is not a number: " + jt.name);
            uint64_t v = (uint64_t)d.asNumber();
            if (v == 0) throw std::runtime_error("model.json: zero dimension in: " + jt.name);
            jt.shape.push_back(v);
        }
        out.push_back(std::move(jt));
    }
    return out;
}

uint64_t num_elements(const JsonTensor& t) {
    uint64_t n = 1;
    for (auto d : t.shape) n *= d;
    return n;
}

// ---------------------------------------------------------------------------
// commands
// ---------------------------------------------------------------------------

int cmd_quantize(const std::string& json_path, const std::string& bin_path,
                 const std::string& out_path, const std::string& type_arg) {
    uint32_t type;
    size_t block, typesize;
    void (*quantize)(const float*, uint8_t*, size_t);
    if (type_arg == "q4_0") {
        type = gguf::GGML_TYPE_Q4_0; block = gguf::Q4_0_BLOCK;
        typesize = gguf::Q4_0_TYPESIZE; quantize = quant::quantize_row_q4_0;
    } else { // default q8_0
        type = gguf::GGML_TYPE_Q8_0; block = gguf::Q8_0_BLOCK;
        typesize = gguf::Q8_0_TYPESIZE; quantize = quant::quantize_row_q8_0;
    }
    std::ifstream jf(json_path);
    if (!jf) throw std::runtime_error("cannot open " + json_path);
    std::stringstream jss;
    jss << jf.rdbuf();
    jmini::Value root = jmini::parse(jss.str());
    std::vector<JsonTensor> tensors = parse_model_json(root);

    std::ifstream bf(bin_path, std::ios::binary);
    if (!bf) throw std::runtime_error("cannot open " + bin_path);
    bf.seekg(0, std::ios::end);
    std::streampos sz = bf.tellg();
    bf.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(sz);
    if (sz > 0) bf.read((char*)bytes.data(), sz);

    uint64_t need = 0;
    for (auto& t : tensors) {
        if (num_elements(t) % block != 0)
            throw std::runtime_error("tensor has elements not divisible by " +
                std::to_string(block) + " (" + type_arg + " block): " + t.name);
        need += num_elements(t) * 4;
    }
    if ((uint64_t)bytes.size() != need)
        throw std::runtime_error("model.bin size does not match model.json tensor shapes");

    const float* fptr = (const float*)bytes.data();
    gguf::GGUFModel m;

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

    for (auto& t : tensors) {
        gguf::TensorInfo ti;
        ti.name = t.name;
        ti.ne = t.shape;
        ti.type = type;

        size_t nblocks = (size_t)(num_elements(t) / block);
        std::vector<uint8_t> q(nblocks * typesize);
        quantize(fptr, q.data(), nblocks);
        fptr += num_elements(t);

        m.tensors.push_back(std::move(ti));
        m.data.push_back(std::move(q));
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
    js << "  \"name\": \"" << in_path << "\",\n";
    js << "  \"tensors\": [\n";
    for (size_t i = 0; i < m.tensors.size(); i++) {
        const auto& t = m.tensors[i];
        js << "    {\"name\": \"" << t.name << "\", \"shape\": [";
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
        const auto& raw = m.data[i];
        size_t n = (size_t)t.n_elements();
        std::vector<float> f(n);
        if (t.type == gguf::GGML_TYPE_Q8_0) {
            quant::dequantize_row_q8_0(raw.data(), f.data(), n / gguf::Q8_0_BLOCK);
        } else if (t.type == gguf::GGML_TYPE_Q4_0) {
            quant::dequantize_row_q4_0(raw.data(), f.data(), n / gguf::Q4_0_BLOCK);
        } else if (t.type == gguf::GGML_TYPE_F32) {
            std::memcpy(f.data(), raw.data(), n * 4);
        } else {
            throw std::runtime_error("unsupported tensor type in dequantize: " + t.name);
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

const char* type_name(uint32_t t) {
    switch (t) {
        case gguf::GGML_TYPE_F32:  return "F32";
        case gguf::GGML_TYPE_Q4_0: return "Q4_0";
        case gguf::GGML_TYPE_Q8_0: return "Q8_0";
        default: return "?";
    }
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
    std::cout << t.decode(ids) << "\n";
    return 0;
}

int cmd_generate(const std::string& model_path, const std::string& prompt,
                 const infer::GenParams& gp) {
    gguf::GGUFModel m = gguf::read_gguf(model_path);
    bpe::Tokenizer tok(m);
    infer::Model model(m);
    if (gp.threads > 0) model.set_threads(gp.threads);
    infer::RNG rng;
    if (gp.seed) rng.seed(gp.seed);

    std::vector<uint32_t> ids = tok.encode(prompt);
    if (ids.empty()) throw std::runtime_error("generate: empty prompt");

    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> logits = infer::prefill(model, ids);
    double pp_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (gp.show_prompt_tokens) std::cout << "prompt tokens: " << ids.size() << "\n";
    // Two decimals: at a few tok/s an integer print rounds a 20% change away.
    printf("pp: %zu tok, %.0f ms, %.2f tok/s\n", ids.size(), pp_ms,
           (double)ids.size() / (pp_ms / 1e3));

    t0 = std::chrono::steady_clock::now();
    std::vector<uint32_t> gen = infer::generate(model, tok, gp, rng, logits);
    double tg_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("tg: %zu tok, %.0f ms, %.2f tok/s\n", gen.size(), tg_ms,
           (double)gen.size() / (tg_ms / 1e3));
    return 0;
}

int cmd_perplexity(const std::string& model_path, const std::string& text,
                   const infer::GenParams& gp) {
    gguf::GGUFModel m = gguf::read_gguf(model_path);
    bpe::Tokenizer tok(m);
    infer::Model model(m);
    if (gp.threads > 0) model.set_threads(gp.threads);

    std::vector<uint32_t> ids = tok.encode(text);
    if (ids.size() < 2) throw std::runtime_error("perplexity: need at least 2 tokens");

    double nll = 0.0;
    std::vector<float> logits = model.step((int)ids[0]);
    for (size_t i = 1; i < ids.size(); i++) {
        uint32_t target = ids[i];
        float maxv = -1e30f;
        for (float l : logits) maxv = std::max(maxv, l);
        double sum = 0.0;
        for (float l : logits) sum += std::exp((double)l - maxv);
        double logsumexp = maxv + std::log(sum);
        double lp = (double)logits[target] - logsumexp;
        nll -= lp;
        logits = model.step((int)target);
    }
    double mean_nll = nll / (double)(ids.size() - 1);
    double ppl = std::exp(mean_nll);
    std::cout << "tokens: " << ids.size() << "\n";
    std::cout << "mean NLL: " << mean_nll << "\n";
    std::cout << "perplexity: " << ppl << "\n";
    return 0;
}

int cmd_chat(const std::string& model_path, const std::string& system,
             const infer::GenParams& gp) {
    gguf::GGUFModel m = gguf::read_gguf(model_path);
    bpe::Tokenizer tok(m);
    infer::Model model(m);
    if (gp.threads > 0) model.set_threads(gp.threads);
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

    std::cout << "Chat ready (type your message; Ctrl+C to quit)\n";
    std::string line;
    while (std::getline(std::cin, line)) {
        messages.push_back({ "user", line });

        std::string full = chat::render(tpl, messages, false, bos, eos);
        std::vector<uint32_t> full_ids = tok.encode(full);
        int cur = model.n_tokens();
        if ((int)full_ids.size() > cur)
            infer::prefill(model, std::vector<uint32_t>(full_ids.begin() + cur, full_ids.end()));

        std::string gen = chat::render(tpl, messages, true, bos, eos);
        std::vector<uint32_t> gen_ids = tok.encode(gen);
        int cur2 = model.n_tokens();
        std::vector<float> logits;
        if ((int)gen_ids.size() > cur2)
            logits = infer::prefill(model, std::vector<uint32_t>(gen_ids.begin() + cur2, gen_ids.end()));

        std::vector<uint32_t> reply = infer::generate(model, tok, gp, rng, logits);

        if (tok.eos_id >= 0) model.step(tok.eos_id);

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
            m.data.push_back(std::move(buf));
        } else {
            size_t nblocks = nin / gguf::Q8_0_BLOCK;
            std::vector<uint8_t> buf(nout * nblocks * gguf::Q8_0_TYPESIZE);
            std::vector<float> row(nin);
            for (size_t o = 0; o < nout; o++) {
                for (size_t i = 0; i < nin; i++) row[i] = dist(rng);
                quant::quantize_row_q8_0(row.data(), buf.data() + o * nblocks * gguf::Q8_0_TYPESIZE, nblocks);
            }
            m.tensors.push_back(std::move(t));
            m.data.push_back(std::move(buf));
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

// Micro-benchmark of the backend hot paths (matmul, RMSNorm, RoPE) plus
// end-to-end prefill/decode TPS on a synthetic Qwen3 model. Used by
// tests/perf.py as the perf-regression gate for hot-path changes.
int cmd_bench(int size, int iters, int threads, int prefill, int decode) {
    auto b = backend::make_cpu_backend();
    b->set_threads(threads);

    // Square matmul: mat is [nin, nout] = [size, size]. x is the input
    // (length nin), out the result (length nout). nout rows, each nin/32 blocks.
    const size_t nblocks = (size_t)size / gguf::Q8_0_BLOCK;
    std::vector<float> x(size, 0.5f);
    std::vector<uint8_t> mat((size_t)size * nblocks * gguf::Q8_0_TYPESIZE);
    std::vector<float> src(size), w(size), dst(size);
    for (int i = 0; i < size; i++) { src[i] = std::sin((float)i * 0.01f); w[i] = 0.1f; }
    std::vector<float> cos(size / 2), sin(size / 2);
    for (int i = 0; i < size / 2; i++) { cos[i] = std::cos(0.1f); sin[i] = std::sin(0.1f); }

    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    for (int it = 0; it < iters; it++)
        b->matvec_q8_0(mat.data(), x.data(), dst.data(), nblocks, (size_t)size);
    double mm_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count() / iters;

    t0 = clock::now();
    for (int it = 0; it < iters; it++)
        b->rms_norm(dst.data(), src.data(), w.data(), (size_t)size, 1e-6f);
    double rn_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count() / iters;

    t0 = clock::now();
    for (int it = 0; it < iters; it++)
        b->rope(dst.data(), cos.data(), sin.data(), size / 2);
    double rp_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count() / iters;

    double mm_gflops = 2.0 * (double)size * (double)size / (mm_ms * 1e6);
    printf("bench: matmul %dx%d  %8.3f ms  %8.2f GFLOPS\n", size, size, mm_ms, mm_gflops);
    printf("bench: rms_norm n=%d  %8.3f ms\n", size, rn_ms);
    printf("bench: rope     n=%d  %8.3f ms\n", size, rp_ms);

    // End-to-end TPS on a small synthetic Qwen3 model (2 layers, 256 embd).
    {
        const int nl = 2, ne = 256, nf = 1024, nh = 8, nk = 2, hd = 32, nv = 512;
        gguf::GGUFModel sm = build_synthetic_model(nl, ne, nf, nh, nk, hd, nv, 12345u);
        infer::Model model(sm, b);
        model.set_threads(threads);

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

void print_usage() {
    std::cout
        << "llmx " << LLMX_VERSION_STRING << " - ground-up GGUF Q8_0 CLI (no external libs)\n"
        << "\n"
        << "Usage:\n"
        << "  llmx quantize   <model.json> <model.bin> <out.gguf> [q8_0|q4_0]\n"
        << "  llmx dequantize <in.gguf> <out.json> <out.bin>\n"
        << "  llmx info       <in.gguf>\n"
        << "  llmx tokenize   <in.gguf> \"<text>\"\n"
        << "  llmx detokenize <in.gguf> <id1,id2,...>\n"
        << "  llmx perplexity <in.gguf> \"<text>\" [flags...]\n"
        << "  llmx generate   <in.gguf> \"<prompt>\" [flags...]\n"
        << "  llmx chat       <in.gguf> [--system \"<text>\"] [flags...]\n"
        << "  llmx bench      [--size N] [--iters N] [--threads N] [--p N] [--n N]\n"
        << "    flags: -n/--max-tokens N  --temp F  --topk N  --topp F  --penalty F  --threads N\n"
        << "           --seed N  --stop \"<text>\"  --think (show reasoning)  --verbose\n";
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
        if (argc < 2) { print_usage(); return 1; }
        std::string cmd = argv[1];

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
            if (argc < 4) { std::cerr << "usage: llmx perplexity <model.gguf> \"<text>\" [flags...]\n"; return 2; }
            infer::GenParams gp;
            for (int i = 4; i < argc; i++) {
                std::string a = argv[i];
                if (a == "--threads") gp.threads = (i + 1 < argc) ? std::atoi(argv[++i]) : gp.threads;
                else { std::cerr << "unknown flag: " << a << "\n"; return 2; }
            }
            return cmd_perplexity(argv[2], argv[3], gp);
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
            int size = 1024, iters = 5, threads = 0, prefill = 64, decode = 64;
            for (int i = 2; i < argc; i++) {
                std::string a = argv[i];
                if (a == "--size") size = (i + 1 < argc) ? std::atoi(argv[++i]) : size;
                else if (a == "--iters") iters = (i + 1 < argc) ? std::atoi(argv[++i]) : iters;
                else if (a == "--threads") threads = (i + 1 < argc) ? std::atoi(argv[++i]) : threads;
                else if (a == "--p") prefill = (i + 1 < argc) ? std::atoi(argv[++i]) : prefill;
                else if (a == "--n") decode = (i + 1 < argc) ? std::atoi(argv[++i]) : decode;
                else { std::cerr << "unknown flag: " << a << "\n"; return 2; }
            }
            if (size <= 0 || size % 32 != 0) { std::cerr << "bench: --size must be positive and a multiple of 32\n"; return 2; }
            return cmd_bench(size, iters, threads, prefill, decode);
        }
        print_usage();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
