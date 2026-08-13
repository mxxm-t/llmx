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
#include <unordered_set>

#include "fp16.hpp"
#include "json.hpp"
#include "gguf.hpp"
#include "tokenizer.hpp"
#include "inference.hpp"
#include "chat.hpp"

// ---------------------------------------------------------------------------
// Q8_0 quantization kernels (block of 32 floats -> 1 f16 scale + 32 int8)
// ---------------------------------------------------------------------------

static void quantize_row_q8_0(const float* src, uint8_t* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const float* x = src + b * gguf::Q8_0_BLOCK;
        uint8_t*      y = dst + b * gguf::Q8_0_TYPESIZE;

        float amax = 0.0f;
        for (size_t j = 0; j < gguf::Q8_0_BLOCK; j++)
            amax = std::max(amax, std::fabs(x[j]));

        const float d = amax / 127.0f;
        const uint16_t d16 = f32_to_f16(d);
        y[0] = (uint8_t)(d16 & 0xff);
        y[1] = (uint8_t)(d16 >> 8);

        for (size_t j = 0; j < gguf::Q8_0_BLOCK; j++) {
            float q = (d > 0.0f) ? std::round(x[j] / d) : 0.0f;
            int v = (int)q;
            if (v > 127)  v = 127;
            if (v < -127) v = -127;
            y[2 + j] = (uint8_t)(int8_t)v;
        }
    }
}

static void dequantize_row_q8_0(const uint8_t* src, float* dst, size_t nblocks) {
    for (size_t b = 0; b < nblocks; b++) {
        const uint8_t* y = src + b * gguf::Q8_0_TYPESIZE;
        float*         x = dst + b * gguf::Q8_0_BLOCK;

        uint16_t d16 = (uint16_t)(y[0] | ((uint16_t)y[1] << 8));
        const float d = f16_to_f32(d16);
        for (size_t j = 0; j < gguf::Q8_0_BLOCK; j++)
            x[j] = (float)(int8_t)y[2 + j] * d;
    }
}

// ---------------------------------------------------------------------------
// model.json / model.bin helper structures
// ---------------------------------------------------------------------------

struct JsonTensor {
    std::string name;
    std::vector<uint64_t> shape; // shape[0] -> ne[0] (fastest dim)
};

static void print_usage() {
    std::cout
        << "gguf8 - ground-up GGUF Q8_0 CLI (no external libs)\n"
        << "\n"
        << "Usage:\n"
        << "  gguf8 quantize   <model.json> <model.bin> <out.gguf>\n"
        << "  gguf8 dequantize <in.gguf> <out.json> <out.bin>\n"
        << "  gguf8 info       <in.gguf>\n"
        << "  gguf8 tokenize   <in.gguf> \"<text>\"\n"
        << "  gguf8 detokenize <in.gguf> <id1,id2,...>\n"
        << "  gguf8 perplexity <in.gguf> \"<text>\" [flags...]\n"
        << "  gguf8 generate   <in.gguf> \"<prompt>\" [flags...]\n"
        << "  gguf8 chat       <in.gguf> [--system \"<text>\"] [flags...]\n"
        << "    flags: -n/--max-tokens N  --temp F  --topk N  --topp F  --penalty F  --threads N\n"
        << "           --seed N  --stop \"<text>\"  --think (show reasoning)  --verbose\n"
        << "\n"
        << "model.json describes tensors:\n"
        << "  {\n"
        << "    \"name\": \"MyModel\",\n"
        << "    \"tensors\": [\n"
        << "      {\"name\": \"tok_embeddings.weight\", \"shape\": [512, 256]},\n"
        << "      {\"name\": \"norm.weight\",           \"shape\": [256]}\n"
        << "    ]\n"
        << "  }\n"
        << "model.bin holds each tensor's float32 data concatenated in that order.\n"
        << "Every tensor must have a number of elements divisible by 32 (Q8_0 block).\n";
}

static std::vector<JsonTensor> parse_model_json(const jmini::Value& root) {
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

static uint64_t num_elements(const JsonTensor& t) {
    uint64_t n = 1;
    for (auto d : t.shape) n *= d;
    return n;
}

// ---------------------------------------------------------------------------
// quantize command
// ---------------------------------------------------------------------------

static int cmd_quantize(const std::string& json_path, const std::string& bin_path,
                        const std::string& out_path) {
    std::ifstream jf(json_path);
    if (!jf) throw std::runtime_error("cannot open " + json_path);
    std::stringstream jss;
    jss << jf.rdbuf();
    jmini::Value root = jmini::parse(jss.str());
    std::vector<JsonTensor> tensors = parse_model_json(root);

    // read all float32 data
    std::ifstream bf(bin_path, std::ios::binary);
    if (!bf) throw std::runtime_error("cannot open " + bin_path);
    bf.seekg(0, std::ios::end);
    std::streampos sz = bf.tellg();
    bf.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(sz);
    if (sz > 0) bf.read((char*)bytes.data(), sz);

    uint64_t need = 0;
    for (auto& t : tensors) {
        if (num_elements(t) % gguf::Q8_0_BLOCK != 0)
            throw std::runtime_error("tensor has elements not divisible by 32 (Q8_0 block): " + t.name);
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
    gguf::MetaValue mv_ft;   mv_ft.vtype   = gguf::V_UINT32; mv_ft.u   = 2;   // file_type = mostly Q8_0
    m.kv.emplace_back("general.name", mv_name);
    m.kv.emplace_back("general.architecture", mv_arch);
    m.kv.emplace_back("general.quantization_version", mv_qver);
    m.kv.emplace_back("general.file_type", mv_ft);

    for (auto& t : tensors) {
        gguf::TensorInfo ti;
        ti.name = t.name;
        ti.ne = t.shape;
        ti.type = gguf::GGML_TYPE_Q8_0;

        size_t nblocks = (size_t)(num_elements(t) / gguf::Q8_0_BLOCK);
        std::vector<uint8_t> q(nblocks * gguf::Q8_0_TYPESIZE);
        quantize_row_q8_0(fptr, q.data(), nblocks);
        fptr += num_elements(t);

        m.tensors.push_back(std::move(ti));
        m.data.push_back(std::move(q));
    }

    gguf::write_gguf(m, out_path);
    std::cout << "wrote " << out_path << " (" << m.tensors.size() << " tensors, Q8_0)\n";
    return 0;
}

// ---------------------------------------------------------------------------
// dequantize command
// ---------------------------------------------------------------------------

static int cmd_dequantize(const std::string& in_path, const std::string& out_json,
                          const std::string& out_bin) {
    gguf::GGUFModel m = gguf::read_gguf(in_path);

    // build output json
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

    // concatenated float32 data
    std::vector<uint8_t> out;
    for (size_t i = 0; i < m.tensors.size(); i++) {
        const auto& t = m.tensors[i];
        const auto& raw = m.data[i];
        size_t n = (size_t)t.n_elements();
        std::vector<float> f(n);
        if (t.type == gguf::GGML_TYPE_Q8_0) {
            dequantize_row_q8_0(raw.data(), f.data(), n / gguf::Q8_0_BLOCK);
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

// ---------------------------------------------------------------------------
// info command
// ---------------------------------------------------------------------------

static const char* type_name(uint32_t t) {
    switch (t) {
        case gguf::GGML_TYPE_F32:  return "F32";
        case gguf::GGML_TYPE_Q8_0: return "Q8_0";
        default: return "?";
    }
}

static void dump_value(const gguf::MetaValue& v) {
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

static int cmd_info(const std::string& in_path) {
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

// ---------------------------------------------------------------------------
// tokenize / detokenize commands
// ---------------------------------------------------------------------------

static std::vector<uint32_t> parse_token_ids(const std::string& s) {
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

static int cmd_tokenize(const std::string& model_path, const std::string& text) {
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

static int cmd_detokenize(const std::string& model_path, const std::string& ids_arg) {
    gguf::GGUFModel m = gguf::read_gguf(model_path);
    bpe::Tokenizer t(m);
    std::vector<uint32_t> ids = parse_token_ids(ids_arg);
    std::cout << t.decode(ids) << "\n";
    return 0;
}

// ---------------------------------------------------------------------------
// sampling + generation
// ---------------------------------------------------------------------------

// Minimal xorshift64 PRNG (no <random> dependency).
struct RNG {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    void seed(uint64_t x) { if (x) s = x; }
    uint64_t next() {
        uint64_t x = s;
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        s = x;
        return x;
    }
    // uniform float in [0,1)
    float unit() { return (float)((next() >> 40) * (1.0 / 16777216.0)); }
};

// Temperature + top-k + top-p nucleus sampling with repetition penalty.
// `penalty` >= 1: divide the score of each already-generated token by penalty
// to discourage repeats. Returns the chosen token id.
static uint32_t sample(const std::vector<float>& logits, float temp, int top_k,
                       float top_p, float penalty, const std::vector<uint32_t>& gen,
                       RNG& rng) {
    size_t n = logits.size();

    std::vector<std::pair<float, uint32_t>> ranked;
    ranked.reserve(n);
    for (size_t i = 0; i < n; i++) ranked.push_back({ logits[i], (uint32_t)i });

    // repetition penalty
    if (penalty > 0.0f && penalty != 1.0f && !gen.empty()) {
        std::unordered_set<uint32_t> seen;
        for (uint32_t id : gen) seen.insert(id);
        for (auto& pr : ranked) {
            if (seen.count(pr.second)) {
                pr.first = (pr.first > 0.0f) ? (pr.first / penalty) : (pr.first * penalty);
            }
        }
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // top-k truncation
    size_t keep = (top_k > 0 && (size_t)top_k < n) ? (size_t)top_k : n;

    // temperature
    if (temp > 0.0f) {
        float inv = 1.0f / temp;
        for (size_t i = 0; i < keep; i++) ranked[i].first /= inv; // *temp
    } else {
        return ranked[0].second; // argmax (no randomness)
    }

    // softmax over the kept window
    float maxv = ranked[0].first;
    std::vector<float> p(keep);
    double sum = 0.0;
    for (size_t i = 0; i < keep; i++) {
        float v = std::exp((ranked[i].first - maxv) / temp);
        p[i] = v;
        sum += v;
    }
    for (size_t i = 0; i < keep; i++) p[i] = (float)(p[i] / sum);

    // top-p nucleus truncation
    size_t nuc = keep;
    if (top_p < 1.0f) {
        float acc = 0.0f;
        nuc = 0;
        while (nuc < keep && acc < top_p) { acc += p[nuc]; nuc++; }
        if (nuc < 1) nuc = 1;
        // renormalize over the nucleus
        float nsum = 0.0f;
        for (size_t i = 0; i < nuc; i++) nsum += p[i];
        for (size_t i = 0; i < nuc; i++) p[i] /= nsum;
    }

    float r = rng.unit();
    float acc = 0.0f;
    for (size_t i = 0; i < nuc; i++) {
        acc += p[i];
        if (r < acc) return ranked[i].second;
    }
    return ranked[nuc - 1].second;
}

struct GenParams {
    int max_tokens = 64;
    float temp = 0.8f;
    int top_k = 40;
    float top_p = 0.95f;
    int threads = 0; // 0 = auto
    float penalty = 1.0f;   // repetition penalty (>= 1)
    uint64_t seed = 0;      // 0 = non-deterministic
    std::string stop;       // stop generating when decoded output contains this
    bool show_prompt_tokens = false;
    bool show_thinking = false; // show Qwen3 <thinking> block
};

// Feed every id in `ids` through the model (prefill / continue), updating the
// KV cache. Returns the logits predicted by the last token (i.e. the
// distribution over the next token).
static std::vector<float> prefill(infer::Model& model,
                                  const std::vector<uint32_t>& ids) {
    std::vector<float> logits;
    for (uint32_t id : ids) logits = model.step((int)id);
    return logits;
}

// Find a vocab token whose string contains `sub`, or -1.
static int find_token_by_substr(const bpe::Tokenizer& tok, const std::string& sub) {
    for (size_t i = 0; i < tok.vocab.size(); i++)
        if (tok.vocab[i].find(sub) != std::string::npos) return (int)i;
    return -1;
}

// Generate tokens starting from `logits` (the prediction after the last fed
// token), stopping at eos. Returns generated ids (excluding the eos token).
// By default the Qwen3 <thinking_start>...<thinking_end> reasoning block is
// hidden; only the final answer is printed. Pass gp.show_thinking to keep it.
static std::vector<uint32_t> generate(infer::Model& model, bpe::Tokenizer& tok,
                                      const GenParams& gp, RNG& rng,
                                      std::vector<float> logits) {
    uint32_t eos = (uint32_t)((tok.eos_id >= 0) ? tok.eos_id : 0);
    std::vector<uint32_t> gen;
    std::string decoded;
    for (int t = 0; t < gp.max_tokens; t++) {
        uint32_t id = sample(logits, gp.temp, gp.top_k, gp.top_p, gp.penalty, gen, rng);
        if (id == eos) break;
        gen.push_back(id);
        decoded += tok.decode({ id });
        if (!gp.stop.empty() && decoded.find(gp.stop) != std::string::npos) break;
        logits = model.step((int)id); // predict token after `id`
    }

    size_t begin = 0;
    size_t end = gen.size();
    if (!gp.show_thinking) {
        int tstart = find_token_by_substr(tok, "thinking_start");
        int tend   = find_token_by_substr(tok, "thinking_end");
        int astart = find_token_by_substr(tok, "answer_start");
        int aend   = find_token_by_substr(tok, "answer_end");
        if (tstart >= 0) {
            auto ts = std::find(gen.begin(), gen.end(), (uint32_t)tstart);
            if (ts != gen.end()) {
                auto te = std::find(gen.begin(), gen.end(), (uint32_t)tend);
                begin = (te != gen.end()) ? (size_t)(te - gen.begin() + 1) : gen.size();
            }
        }
        if (astart >= 0) {
            auto as = std::find(gen.begin(), gen.end(), (uint32_t)astart);
            if (as != gen.end()) begin = (size_t)(as - gen.begin() + 1);
        }
        if (aend >= 0) {
            auto ae = std::find(gen.begin(), gen.end(), (uint32_t)aend);
            if (ae != gen.end()) end = (size_t)(ae - gen.begin());
        }
    }

    for (size_t i = begin; i < end; i++)
        std::cout << tok.decode({ gen[i] }) << std::flush;
    std::cout << "\n";
    return gen;
}

static int cmd_generate(const std::string& model_path, const std::string& prompt,
                        const GenParams& gp) {
    gguf::GGUFModel m = gguf::read_gguf(model_path);
    bpe::Tokenizer tok(m);
    infer::Model model(m);
    if (gp.threads > 0) model.set_threads(gp.threads);
    RNG rng;
    if (gp.seed) rng.seed(gp.seed);

    std::vector<uint32_t> ids = tok.encode(prompt);
    if (ids.empty()) throw std::runtime_error("generate: empty prompt");

    std::vector<float> logits = prefill(model, ids);
    if (gp.show_prompt_tokens) std::cout << "prompt tokens: " << ids.size() << "\n";
    generate(model, tok, gp, rng, logits);
    return 0;
}

// ---------------------------------------------------------------------------
// perplexity (correctness gate)
// ---------------------------------------------------------------------------
// Walks the text token by token, reusing the KV cache. For each token i>0 it
// measures the log-probability the model assigns to token i given tokens
// [0..i-1], then reports mean per-token negative log-likelihood (natural log)
// and exp of it = perplexity. A low ppl on real text is a sanity check that
// the Q8_0 forward pass is numerically sound.
static int cmd_perplexity(const std::string& model_path, const std::string& text,
                          const GenParams& gp) {
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

// ---------------------------------------------------------------------------
// chat REPL
// ---------------------------------------------------------------------------

static int cmd_chat(const std::string& model_path, const std::string& system,
                    const GenParams& gp) {
    gguf::GGUFModel m = gguf::read_gguf(model_path);
    bpe::Tokenizer tok(m);
    infer::Model model(m);
    if (gp.threads > 0) model.set_threads(gp.threads);
    RNG rng;
    if (gp.seed) rng.seed(gp.seed);

    // Read the chat template from GGUF metadata; fall back to a simple
    // role/content template if the model doesn't ship one.
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

        // Prefill only the delta: everything already in the KV cache up to the
        // last position, then the newly appended user turn.
        std::string full = chat::render(tpl, messages, false, bos, eos);
        std::vector<uint32_t> full_ids = tok.encode(full);
        int cur = model.n_tokens();
        if ((int)full_ids.size() > cur)
            prefill(model, std::vector<uint32_t>(full_ids.begin() + cur, full_ids.end()));

        // Append the assistant generation prompt; its last token predicts the
        // first assistant reply token.
        std::string gen = chat::render(tpl, messages, true, bos, eos);
        std::vector<uint32_t> gen_ids = tok.encode(gen);
        int cur2 = model.n_tokens();
        std::vector<float> logits;
        if ((int)gen_ids.size() > cur2)
            logits = prefill(model, std::vector<uint32_t>(gen_ids.begin() + cur2, gen_ids.end()));

        std::vector<uint32_t> reply = generate(model, tok, gp, rng, logits);

        // Close the assistant turn in the KV cache. generate() stops at eos
        // without feeding it, so without this the model never "sees" the end of
        // its own reply before the next user message.
        if (tok.eos_id >= 0) model.step(tok.eos_id);

        messages.push_back({ "assistant", tok.decode(reply) });
    }
    return 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    try {
        if (argc < 2) { print_usage(); return 1; }
        std::string cmd = argv[1];

        if (cmd == "generate" || cmd == "chat") {
            if (argc < 4) {
                std::cerr << "usage: gguf8 " << cmd << " <model.gguf> [--system \"<text>\"] [flags...]\n";
                std::cerr << "       gguf8 generate <model.gguf> \"<prompt>\" [flags...]\n";
                return 2;
            }
            GenParams gp;
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
            if (argc < 4) { std::cerr << "usage: gguf8 perplexity <model.gguf> \"<text>\" [flags...]\n"; return 2; }
            GenParams gp;
            for (int i = 4; i < argc; i++) {
                std::string a = argv[i];
                if (a == "--threads") gp.threads = (i + 1 < argc) ? std::atoi(argv[++i]) : gp.threads;
                else { std::cerr << "unknown flag: " << a << "\n"; return 2; }
            }
            return cmd_perplexity(argv[2], argv[3], gp);
        }

        if (cmd == "tokenize") {
            if (argc != 4) { std::cerr << "usage: gguf8 tokenize <model.gguf> \"<text>\"\n"; return 2; }
            return cmd_tokenize(argv[2], argv[3]);
        }
        if (cmd == "detokenize") {
            if (argc != 4) { std::cerr << "usage: gguf8 detokenize <model.gguf> <id1,id2,...>\n"; return 2; }
            return cmd_detokenize(argv[2], argv[3]);
        }

        if (cmd == "quantize") {
            if (argc != 5) { std::cerr << "usage: gguf8 quantize <model.json> <model.bin> <out.gguf>\n"; return 2; }
            return cmd_quantize(argv[2], argv[3], argv[4]);
        }
        if (cmd == "dequantize") {
            if (argc != 5) { std::cerr << "usage: gguf8 dequantize <in.gguf> <out.json> <out.bin>\n"; return 2; }
            return cmd_dequantize(argv[2], argv[3], argv[4]);
        }
        if (cmd == "info") {
            if (argc != 3) { std::cerr << "usage: gguf8 info <in.gguf>\n"; return 2; }
            return cmd_info(argv[2]);
        }
        print_usage();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
