#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <optional>
#include <thread>
#include <tuple>
#include "format/gguf.hpp"
#include "inference/load.hpp"
#include "loading_backend.hpp"
#include "tiny_qwen.hpp"

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

// Every tensor of `m` in file order, which is what the loader reads in.
std::vector<size_t> every(const gguf::GGUFModel& m) {
    std::vector<size_t> tensors(m.tensors.size());
    std::iota(tensors.begin(), tensors.end(), size_t(0));
    return tensors;
}

// The file at `path` read, mapped and read in as the loader does, its payload reported to `progress`.
gguf::GGUFModel load_file(const std::string& path, const format::LoadProgress& progress) {
    gguf::GGUFModel m = gguf::read_gguf(path);
    gguf::map_payload(m);
    gguf::warm(m, every(m), progress);
    return m;
}

// A CPU backend playing a device: it copies what it adopts, so no weight reads the file in place.
// A streamed load gives it alloc_weight storage, poisoned so a byte the load does not write shows, filled by a copy out of the ring it reads in place or by a write where it is made not to (`wraps`); the Nth fill can be made to fail.
struct CopyingBackend : backend::CpuBackend {
    int writes = 0, fail_write = 0, foreign = 0, copies = 0;
    int slow = 0;   // fills still to take two milliseconds each, so the stream's readers run ahead and fill the ring
    bool wraps = true;
    std::vector<backend::BufferPtr> weights;
    std::vector<std::pair<const backend::Buffer*, size_t>> written;
    bool reads_in_place() const override { return false; }
    bool is_cpu() const override { return false; }
    backend::BufferPtr adopt(const void* src, size_t bytes) override {
        auto buffer = alloc(bytes, backend::Memory::device);
        backend::CpuBackend::write(*buffer, 0, src, bytes);
        return buffer;
    }
    backend::BufferPtr alloc_weight(size_t bytes) override {
        auto buffer = alloc(bytes, backend::Memory::device);
        const std::vector<uint8_t> poison(bytes, 0xA5);
        backend::CpuBackend::write(*buffer, 0, poison.data(), bytes);
        weights.push_back(buffer);
        return buffer;
    }
    backend::BufferPtr wrap_host(void* memory, size_t bytes) override { return wraps ? backend::CpuBackend::wrap_host(memory, bytes) : nullptr; }
    void write(backend::Buffer& dst, size_t off, const void* src, size_t bytes) override {
        fill(dst, off);
        backend::CpuBackend::write(dst, off, src, bytes);
    }
    void copy(backend::Buffer& dst, size_t dst_off, const backend::Buffer& src, size_t src_off, size_t bytes) override {
        fill(dst, dst_off);
        ++copies;
        backend::CpuBackend::copy(dst, dst_off, src, src_off, bytes);
    }
    void fill(const backend::Buffer& dst, size_t off) {
        if (++writes == fail_write) throw std::runtime_error("injected write failure");
        if (slow > 0 && slow--) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        if (!weights.empty() && std::none_of(weights.begin(), weights.end(), [&](const backend::BufferPtr& w) { return w.get() == &dst; })) ++foreign;
        written.push_back({&dst, off});
    }
};

// A copying backend whose copies out of host memory run only when their submission is waited for, as a device's do, so a slot of the ring refilled before its copies retire leaves other bytes in the weights.
struct DeferredBackend : CopyingBackend {
    struct Queued {
        const backend::Buffer* dst;
        size_t off;
        const uint8_t* src;
        size_t bytes;
    };
    std::vector<Queued> open;
    std::vector<std::pair<backend::Ticket, std::vector<Queued>>> submitted;
    backend::Ticket last = 0;
    void copy(backend::Buffer& dst, size_t dst_off, const backend::Buffer& src, size_t src_off, size_t bytes) override {
        fill(dst, dst_off);
        ++copies;
        open.push_back({&dst, dst_off, (const uint8_t*)src.host_ptr() + src_off, bytes});
    }
    backend::Ticket submit() override {
        submitted.push_back({++last, std::move(open)});
        open.clear();
        return last;
    }
    void wait(backend::Ticket t) noexcept override {
        for (auto& [ticket, copies_of] : submitted) {
            if (ticket > t) break;
            for (const Queued& q : copies_of) std::memcpy((uint8_t*)const_cast<void*>(q.dst->host_ptr()) + q.off, q.src, q.bytes);
            copies_of.clear();
        }
    }
    void sync() noexcept override { wait(submit()); }
};

// The load modes to run on the model at `path`: direct only where its file system takes direct reads, else the refusal a direct load gives is checked, naming the first file with tensors, `named` (the path itself when empty).
std::vector<infer::LoadMode> load_modes(const std::string& path, const std::string& named = {}) {
    try {
        format::FileReader probe(path, true);
    } catch (const format::DirectUnavailable&) {
        infer::PlacementRequest request;
        request.names = {"cpu"};
        std::string error;
        try { infer::load_model(path, {backend::make_cpu_backend()}, request, {}, {}, infer::LoadMode::direct); }
        catch (const std::runtime_error& e) { error = e.what(); }
        require(error.rfind("--load-mode direct: ", 0) == 0 && error.find(named.empty() ? path : named) != std::string::npos,
                "a direct load was not refused, naming its file, where direct reads are not taken");
        return {infer::LoadMode::automatic, infer::LoadMode::mapped};
    }
    return {infer::LoadMode::automatic, infer::LoadMode::mapped, infer::LoadMode::direct};
}

// Whether every buffer holds the bytes of some tensor of `source` of its size, so nothing of the poison is left.
bool holds_tensors(const std::vector<backend::BufferPtr>& buffers, const gguf::GGUFModel& source) {
    for (const auto& b : buffers) {
        bool found = false;
        for (size_t i = 0; i < source.tensors.size() && !found; ++i)
            found = source.tensor_bytes(i) == b->size() && std::memcmp(b->host_ptr(), source.tensor_data(i), b->size()) == 0;
        if (!found) return false;
    }
    return !buffers.empty();
}

// A load's report: metadata first, the fit before the allocation, each phase once, and the weights' bytes starting at 0, only rising and ending at their total.
struct Rising {
    std::vector<std::pair<size_t, size_t>> seen;
    std::vector<std::string> phases;
    infer::LoadReport callback() {
        return [this](const infer::LoadStep& step) {
            if (phases.empty() || phases.back() != step.phase) phases.push_back(step.phase);
            if (std::string(step.phase) == "weights") seen.push_back({step.completed, step.total});
        };
    }
    bool whole(size_t total) const {
        const auto at = [&](const char* p) { return size_t(std::find(phases.begin(), phases.end(), std::string(p)) - phases.begin()); };
        if (phases.size() != 4 || phases[0] != "metadata" || at("fit") >= at("allocate") || at("allocate") == 4 || at("weights") == 4) return false;
        if (seen.empty() || seen.front().first != 0 || seen.back().first != total) return false;
        for (size_t i = 0; i < seen.size(); ++i)
            if (seen[i].second != total || (i && seen[i].first <= seen[i - 1].first)) return false;
        return true;
    }
};

// `m` with the tokenizer metadata a load reads: sixteen one-letter tokens, the last the end of text.
gguf::GGUFModel with_tokens(gguf::GGUFModel m) {
    gguf::MetaValue tokens;
    tokens.vtype = gguf::V_ARRAY;
    tokens.u = gguf::V_STRING;
    for (char c = 'a'; c < 'a' + 16; ++c) {
        gguf::MetaValue token;
        token.vtype = gguf::V_STRING;
        token.s = std::string(1, c);
        tokens.arr.push_back(token);
    }
    gguf::MetaValue eos;
    eos.vtype = gguf::V_UINT32;
    eos.u = 15;
    m.kv.push_back({"tokenizer.ggml.tokens", tokens});
    m.kv.push_back({"tokenizer.ggml.eos_token_id", eos});
    return m;
}

// tiny_qwen with every tensor BF16 but its vectors, which stay F32: the GGUF file the converter writes from directory_checks' directory.
gguf::GGUFModel bf16_twin() {
    const gguf::GGUFModel f32 = with_tokens(tiny_qwen(2, 2 * 128, true));
    gguf::GGUFModel m;
    m.kv = f32.kv;
    for (size_t i = 0; i < f32.tensors.size(); ++i) {
        gguf::TensorInfo t = f32.tensors[i];
        const uint8_t* values = f32.tensor_data(i);
        const size_t n = size_t(t.n_elements());
        std::vector<uint8_t> bytes(values, values + n * 4);
        if (t.ne.size() > 1) {
            // tiny_qwen's values are multiples of 1/64, exact in BF16: each keeps the high half of its F32 bits.
            for (size_t k = 0; k < n; ++k) std::memcpy(bytes.data() + 2 * k, values + 4 * k + 2, 2);
            bytes.resize(n * 2);
            t.type = quant::GGML_TYPE_BF16;
        }
        m.tensors.push_back(t);
        m.add_tensor_data(bytes);
    }
    return m;
}

// The safetensors header entry of a `dtype` tensor of `ne` (fastest first) at [begin, end).
std::string header_entry(const std::string& name, const std::string& dtype, const std::vector<uint64_t>& ne, size_t begin, size_t end) {
    std::string e = "\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":[";
    for (size_t d = ne.size(); d-- > 0;) e += std::to_string(ne[d]) + (d ? "," : "");
    return e + "],\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(end) + "]}";
}

// The Hugging Face name of tiny_qwen's tensor `name` without its ".weight", such as model.layers.0.self_attn.q_proj.
std::string hf_name(const std::string& name) {
    static const std::vector<std::pair<std::string, std::string>> names = {
        {"token_embd", "model.embed_tokens"}, {"output_norm", "model.norm"}, {"attn_norm", "input_layernorm"}, {"ffn_norm", "post_attention_layernorm"},
        {"attn_q_norm", "self_attn.q_norm"}, {"attn_k_norm", "self_attn.k_norm"}, {"attn_q", "self_attn.q_proj"}, {"attn_k", "self_attn.k_proj"},
        {"attn_v", "self_attn.v_proj"}, {"attn_output", "self_attn.o_proj"}, {"ffn_gate", "mlp.gate_proj"}, {"ffn_up", "mlp.up_proj"}, {"ffn_down", "mlp.down_proj"}};
    std::string base = name.substr(0, name.size() - 7), layer;
    if (base.rfind("blk.", 0) == 0) {
        const size_t dot = base.find('.', 4);
        layer = "model.layers." + base.substr(4, dot - 4) + ".";
        base = base.substr(dot + 1);
    }
    for (const auto& [gguf_name, hf] : names)
        if (gguf_name == base) return layer + hf;
    throw std::runtime_error("a tiny_qwen tensor has no HF name");
}

// One tensor of a safetensors file: its name, dtype, shape (fastest first) and bytes.
struct StEntry {
    std::string name, dtype;
    std::vector<uint64_t> ne;
    std::vector<uint8_t> bytes;
};

// `entries` written as the safetensors file `file`, in their order.
void write_safetensors(const std::filesystem::path& file, const std::vector<StEntry>& entries) {
    std::string header = "{";
    std::vector<uint8_t> data;
    for (size_t i = 0; i < entries.size(); ++i) {
        header += (i ? "," : "") + header_entry(entries[i].name, entries[i].dtype, entries[i].ne, data.size(), data.size() + entries[i].bytes.size());
        data.insert(data.end(), entries[i].bytes.begin(), entries[i].bytes.end());
    }
    header += "}";
    while (header.size() % 8) header += ' ';
    std::ofstream out(file, std::ios::binary);
    uint8_t length[8];
    for (int k = 0; k < 8; ++k) length[k] = uint8_t(uint64_t(header.size()) >> (8 * k));
    out.write(reinterpret_cast<const char*>(length), 8);
    out << header;
    out.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
}

// The tokenizer files of a directory of tiny_qwen's sixteen one-letter tokens.
void write_tokenizer(const std::filesystem::path& dir) {
    std::string vocab;
    for (int k = 0; k < 16; ++k) vocab += std::string(k ? "," : "") + "\"" + char('a' + k) + "\":" + std::to_string(k);
    std::ofstream(dir / "tokenizer.json", std::ios::binary) << R"({"added_tokens":[],"normalizer":{"type":"NFC"},)"
        R"("pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":")"
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)"
        R"("},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"use_regex":false}]},)"
        R"("decoder":{"type":"ByteLevel"},"model":{"type":"BPE","vocab":{)" << vocab << R"(},"merges":[]}})";
    std::ofstream(dir / "tokenizer_config.json", std::ios::binary) << "{}";
}

// A model loaded with a chat template given (ModelOptions::chat_template) renders with it and stores it as its own, saying whether it replaced one; a template that is not UTF-8 is refused.
void template_checks(const std::string& path, bool carried) {
    const std::string text = "{% for m in messages %}[{{ m['content'] }}]{% endfor %}";
    infer::PlacementRequest request;
    request.names = {"cpu"};
    infer::ModelOptions options;
    options.chat_template = text;
    const auto loaded = infer::load_model(path, {backend::make_cpu_backend()}, request, options);
    const gguf::MetaValue* stored = loaded->file.find("tokenizer.chat_template");
    require(stored && stored->s == text && loaded->replaced_template == carried && loaded->chat.render({{"user", "x", std::nullopt}}, true, {}) == "[x]",
            "a given chat template did not replace the model's");
    options.chat_template = "{{ '\xff' }}";
    std::string error;
    try { infer::load_model(path, {backend::make_cpu_backend()}, request, options); }
    catch (const std::runtime_error& e) { error = e.what(); }
    require(error == "the chat template is not UTF-8 at byte 4", "a chat template that is not UTF-8 was taken");
}

// A Hugging Face directory of bf16_twin's weights, its vectors BF16 too, loads in each mode on the CPU and on copying backends to the twin's logits, its vectors widened into the loader's buffer.
void directory_checks(const std::string& path) {
    namespace fs = std::filesystem;
    const gguf::GGUFModel twin = bf16_twin();
    const fs::path dir = fs::u8path(path + ".dir");
    fs::create_directories(dir);
    std::vector<StEntry> entries;
    size_t vector_floats = 0;
    for (size_t i = 0; i < twin.tensors.size(); ++i) {
        const gguf::TensorInfo& t = twin.tensors[i];
        std::vector<uint8_t> bytes(twin.tensor_data(i), twin.tensor_data(i) + twin.tensor_bytes(i));
        if (t.ne.size() == 1) {
            const size_t n = bytes.size() / 4;
            vector_floats += n;
            for (size_t k = 0; k < n; ++k) std::memmove(bytes.data() + 2 * k, bytes.data() + 4 * k + 2, 2);
            bytes.resize(n * 2);
        }
        entries.push_back({hf_name(t.name) + ".weight", "BF16", t.ne, std::move(bytes)});
    }
    write_safetensors(dir / "model.safetensors", entries);
    std::ofstream(dir / "config.json", std::ios::binary) << R"({"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","num_hidden_layers":2,)"
        R"("hidden_size":8,"intermediate_size":12,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,)"
        R"("max_position_embeddings":256,"vocab_size":16,"tie_word_embeddings":true,"eos_token_id":15,"torch_dtype":"bfloat16"})";
    write_tokenizer(dir);
    gguf::write_gguf(twin, path);
    const std::vector<uint32_t> ids = {0, 1, 2, 3, 4};
    const std::vector<float> expected = infer::Model(infer::gguf_weights(twin), backend::make_cpu_backend()).prefill(ids);
    const gguf::GGUFModel read = infer::read_model(dir.u8string());
    require(read.tensors.size() == twin.tensors.size() && read.find("tokenizer.ggml.merges") && read.find("qwen3.attention.value_length"),
            "a directory did not read as its GGUF file's tensors and keys");
    for (const infer::LoadMode mode : load_modes(path)) {
        for (const int arm : {0, 1, 2}) {
            const auto copier = std::make_shared<CopyingBackend>();
            copier->wraps = arm == 1;
            const backend::BackendPtr b = arm ? backend::BackendPtr(copier) : backend::make_cpu_backend();
            infer::PlacementRequest request;
            request.names = {"cpu"};
            const auto loaded = infer::load_model(dir.u8string(), {b}, request, {}, {}, mode);
            require(loaded->widened.size() == vector_floats && loaded->tok->encode("abc") == std::vector<uint32_t>({0, 1, 2}),
                    "a directory's vectors were not widened into the loader's buffer, or its tokenizer differs");
            const std::vector<float> logits = loaded->model->prefill(ids);
            require(logits.size() == expected.size() && std::memcmp(logits.data(), expected.data(), logits.size() * sizeof(float)) == 0,
                    "a directory's logits differ from its GGUF file's");
        }
    }
    // A template given beside the directory takes the place of the one it carries, in its metadata and its format alike.
    std::ofstream(dir / "chat_template.jinja", std::ios::binary) << "{{ messages[0]['content'] }}";
    template_checks(dir.u8string(), true);
    fs::remove_all(dir);
}

// One AWQ linear [in, out] of layer `layer` in groups of 32: the checkpoint's qweight, qzeros and scales (F16 on layer 0, BF16 on layer 1), and the Q5_0 rows holding each (q - z) * s, written here from the block layout.
// `last` replaces the bits of the last scale.
struct AwqLinear {
    std::vector<uint32_t> qweight, qzeros;
    std::vector<uint16_t> scales;
    std::vector<uint8_t> q5_0;
};

AwqLinear awq_linear(size_t in, size_t out, size_t layer, std::optional<uint16_t> last = std::nullopt) {
    // Output 8j + order[s] is nibble s of word j.
    static constexpr size_t order[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    const auto nibble = [](size_t n) { return uint32_t(4 * size_t(std::find(order, order + 8, n % 8) - order)); };
    // Powers of two of both signs, the last an F16 subnormal, so each is exact in F16 and BF16 alike.
    static const float powers[5] = {1.0f / 64, -1.0f / 128, 1.0f / 256, -1.0f / 32, 1.0f / 1048576};
    const auto code = [&](size_t k, size_t n) { return uint32_t((n * 7 + k * 3 + layer) % 16); };
    const auto zero = [&](size_t g, size_t n) { return uint32_t((n + g + layer) % 16); };
    const size_t groups = in / 32, words = out / 8;
    AwqLinear a;
    a.qweight.assign(in * words, 0);
    a.qzeros.assign(groups * words, 0);
    a.scales.resize(groups * out);
    std::vector<uint16_t> d(groups * out);
    for (size_t g = 0; g < groups; ++g)
        for (size_t n = 0; n < out; ++n) {
            const float s = powers[(n + g + layer) % 5];
            uint32_t bits;
            std::memcpy(&bits, &s, 4);
            d[g * out + n] = f32_to_f16(s);
            a.scales[g * out + n] = layer ? uint16_t(bits >> 16) : d[g * out + n];
            a.qzeros[g * words + n / 8] |= zero(g, n) << nibble(n);
        }
    if (last) a.scales.back() = *last;
    for (size_t k = 0; k < in; ++k)
        for (size_t n = 0; n < out; ++n) a.qweight[k * words + n / 8] |= code(k, n) << nibble(n);
    // A Q5_0 block of 32 inputs: F16 d, a word of each code's fifth bit, then the low nibbles of codes j and j + 16 in byte j.
    a.q5_0.assign(out * groups * 22, 0);
    for (size_t n = 0; n < out; ++n)
        for (size_t g = 0; g < groups; ++g) {
            uint8_t* block = a.q5_0.data() + (n * groups + g) * 22;
            block[0] = uint8_t(d[g * out + n]);
            block[1] = uint8_t(d[g * out + n] >> 8);
            for (size_t j = 0; j < 32; ++j) {
                const uint32_t q = code(32 * g + j, n) + 16 - zero(g, n);
                if (q & 16) block[2 + j / 8] |= uint8_t(1u << (j % 8));
                block[6 + j % 16] |= uint8_t((q & 15) << (j < 16 ? 0 : 4));
            }
        }
    return a;
}

template <class T> std::vector<uint8_t> bytes_of(const std::vector<T>& v) {
    std::vector<uint8_t> b(v.size() * sizeof(T));
    std::memcpy(b.data(), v.data(), b.size());
    return b;
}

// Quantized directory `dir` loads in each mode on the CPU and on copying backends to the logits of `twin`, written at `path`, whose `linears` repacked tensors it holds in their target type.
// The CPU keeps each repacked tensor in the loader's buffer as the twin's bytes, and a copying backend keeps none.
void twin_checks(const std::filesystem::path& dir, const std::string& path, const gguf::GGUFModel& twin, size_t linears, const std::string& label) {
    const std::vector<uint32_t> ids = {0, 1, 2, 3, 4};
    const std::vector<float> expected = infer::Model(infer::gguf_weights(twin), backend::make_cpu_backend()).prefill(ids);
    const gguf::GGUFModel read = infer::read_model(dir.u8string());
    require(read.tensors.size() == twin.tensors.size() && read.composites.size() == linears && linears == 14,
            (label + " did not read as its twin's tensors, each linear one made of its parts").c_str());
    gguf::write_gguf(twin, path);
    for (const infer::LoadMode mode : load_modes(path)) {
        for (const int arm : {0, 1, 2}) {
            const auto copier = std::make_shared<CopyingBackend>();
            copier->wraps = arm == 1;
            const backend::BackendPtr b = arm ? backend::BackendPtr(copier) : backend::make_cpu_backend();
            infer::PlacementRequest request;
            request.names = {"cpu"};
            const auto loaded = infer::load_model(dir.u8string(), {b}, request, {}, {}, mode);
            if (arm == 0) {
                bool same = loaded->made.size() == linears;
                for (const auto& [i, bytes] : loaded->made) {
                    const std::string& name = loaded->file.tensors[i].name;
                    const auto t = size_t(std::find_if(twin.tensors.begin(), twin.tensors.end(), [&](const gguf::TensorInfo& x) { return x.name == name; }) - twin.tensors.begin());
                    same = same && t < twin.tensors.size() && bytes.size() == twin.tensor_bytes(t) && std::memcmp(bytes.data(), twin.tensor_data(t), bytes.size()) == 0;
                }
                require(same, (label + " on the CPU did not keep its twin's bytes in the loader's buffer").c_str());
            } else {
                // A streamed load makes each linear on the stream's readers into the weights it allocated; a mapped one copies what it made at adoption.
                require(loaded->made.empty() && (mode == infer::LoadMode::mapped ? copier->weights.empty() : holds_tensors(copier->weights, twin)),
                        (label + " on a copying backend kept made bytes, or its weights are not its twin's").c_str());
            }
            const std::vector<float> logits = loaded->model->prefill(ids);
            require(logits.size() == expected.size() && std::memcmp(logits.data(), expected.data(), logits.size() * sizeof(float)) == 0,
                    (label + "'s logits differ from its twin's").c_str());
        }
    }
}

// Writes `base` as directory `dir` with `quantization` as its quantization_config: each linear's parts by `linear`, which returns its twin's type and bytes, every other tensor in BF16, and `entries` beside them.
// `twin` gets the same model as a GGUF file's tensors, and the count of linears is returned.
template <class Linear>
size_t write_quant_dir(const std::filesystem::path& dir, const gguf::GGUFModel& base, const std::string& quantization, gguf::GGUFModel& twin, std::vector<StEntry> entries, Linear linear) {
    twin = gguf::GGUFModel{};
    twin.kv = base.kv;
    size_t linears = 0;
    for (size_t i = 0; i < base.tensors.size(); ++i) {
        gguf::TensorInfo t = base.tensors[i];
        const uint8_t* values = base.tensor_data(i);
        const size_t n = size_t(t.n_elements());
        const std::string hf = hf_name(t.name);
        if (t.ne.size() == 2 && t.name != "token_embd.weight") {
            auto [type, bytes] = linear(t, size_t(t.name[4] - '0'), hf, entries);
            t.type = type;
            twin.tensors.push_back(t);
            twin.add_tensor_data(bytes);
            ++linears;
            continue;
        }
        // tiny_qwen's values are multiples of 1/64, exact in BF16: each keeps the high half of its F32 bits.
        std::vector<uint8_t> high(n * 2);
        for (size_t k = 0; k < n; ++k) std::memcpy(high.data() + 2 * k, values + 4 * k + 2, 2);
        entries.push_back({hf + ".weight", "BF16", t.ne, high});
        if (t.ne.size() == 2) t.type = quant::GGML_TYPE_BF16;
        twin.tensors.push_back(t);
        twin.add_tensor_data(t.ne.size() == 2 ? high : std::vector<uint8_t>(values, values + n * 4));
    }
    write_safetensors(dir / "model.safetensors", entries);
    std::ofstream(dir / "config.json", std::ios::binary) << R"({"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","num_hidden_layers":2,)"
        R"("hidden_size":32,"intermediate_size":32,"num_attention_heads":8,"num_key_value_heads":2,"head_dim":4,)"
        R"("max_position_embeddings":256,"vocab_size":16,"tie_word_embeddings":true,"eos_token_id":15,"torch_dtype":"float16","quantization_config":{)"
        << quantization << "}}";
    return linears;
}

// A CPU load of directory `dir` fails with an error holding `expected`, else the check fails with `message`.
void refused(const std::filesystem::path& dir, const std::string& expected, const char* message) {
    std::string error;
    try {
        infer::PlacementRequest request;
        request.names = {"cpu"};
        infer::load_model(dir.u8string(), {backend::make_cpu_backend()}, request);
    } catch (const std::runtime_error& e) { error = e.what(); }
    if (error.find(expected) == std::string::npos) throw std::runtime_error(std::string(message) + ": " + error);
}

// An AWQ directory of a 32-wide model, whose linears are AwqLinear and whose other tensors are BF16, loads as its Q5_0 twin (twin_checks).
// A BF16 scale F16 cannot hold, a quantized tied head and quantization_config values llmx does not read are refused by name.
void awq_checks(const std::string& path) {
    namespace fs = std::filesystem;
    const gguf::GGUFModel base = with_tokens(tiny_qwen_shaped(2, 2 * 128, true, 8, 2, 32, 16, 32));
    const fs::path dir = fs::u8path(path + ".awq");
    fs::create_directories(dir);
    gguf::GGUFModel twin;
    const auto write_dir = [&](std::optional<uint16_t> last, const std::string& quantization, bool head = false) {
        std::vector<StEntry> extra;
        if (head) {
            // A quantized lm_head beside the tied embedding, [32, 16] as the embedding is.
            const AwqLinear a = awq_linear(32, 16, 0, std::nullopt);
            extra.push_back({"lm_head.qweight", "I32", {2, 32}, bytes_of(a.qweight)});
            extra.push_back({"lm_head.qzeros", "I32", {2, 1}, bytes_of(a.qzeros)});
            extra.push_back({"lm_head.scales", "F16", {16, 1}, bytes_of(a.scales)});
        }
        return write_quant_dir(dir, base, quantization, twin, extra, [&](const gguf::TensorInfo& t, size_t layer, const std::string& hf, std::vector<StEntry>& entries) {
            const AwqLinear a = awq_linear(size_t(t.ne[0]), size_t(t.ne[1]), layer, t.name == "blk.1.ffn_down.weight" ? last : std::nullopt);
            entries.push_back({hf + ".qweight", "I32", {t.ne[1] / 8, t.ne[0]}, bytes_of(a.qweight)});
            entries.push_back({hf + ".qzeros", "I32", {t.ne[1] / 8, t.ne[0] / 32}, bytes_of(a.qzeros)});
            entries.push_back({hf + ".scales", layer ? "BF16" : "F16", {t.ne[1], t.ne[0] / 32}, bytes_of(a.scales)});
            return std::make_pair(uint32_t(quant::GGML_TYPE_Q5_0), a.q5_0);
        });
    };
    const std::string awq = R"("quant_method":"awq","bits":4,"group_size":32,"zero_point":true,"version":"gemm","modules_to_not_convert":null)";
    const size_t linears = write_dir(std::nullopt, awq);
    write_tokenizer(dir);
    twin_checks(dir, path, twin, linears, "an AWQ directory");
    // As the last scale of layer 1's down projection: 2^-26, below F16's smallest subnormal; 1.1000001b * 2^-20, an F16 subnormal with 4 fraction bits there, which rounds; and 65536, which overflows.
    write_dir(uint16_t(0x3280), awq);
    refused(dir, "tensor blk.1.ffn_down.weight: BF16 scale 1.49011612e-08 of output 31, group 0 is not exact in F16", "an inexact BF16 scale was taken");
    write_dir(uint16_t(0x35C1), awq);
    refused(dir, "tensor blk.1.ffn_down.weight: BF16 scale 1.43796206e-06 of output 31, group 0 is not exact in F16", "a BF16 scale F16 rounds was taken");
    write_dir(uint16_t(0x4780), awq);
    refused(dir, "tensor blk.1.ffn_down.weight: BF16 scale 65536 of output 31, group 0 is not exact in F16", "a BF16 scale past F16's range was taken");
    write_dir(std::nullopt, awq, true);
    refused(dir, "tensor lm_head.weight is quantized, and config.json ties the head to the embedding", "a quantized tied head was taken");
    // AutoAWQ's own files write the packing in capitals.
    write_dir(std::nullopt, R"("quant_method":"awq","bits":4,"group_size":32,"zero_point":true,"version":"GEMM")");
    {
        const std::vector<uint32_t> ids = {0, 1, 2, 3, 4};
        const std::vector<float> expected = infer::Model(infer::gguf_weights(twin), backend::make_cpu_backend()).prefill(ids);
        infer::PlacementRequest request;
        request.names = {"cpu"};
        const auto loaded = infer::load_model(dir.u8string(), {backend::make_cpu_backend()}, request);
        const std::vector<float> logits = loaded->model->prefill(ids);
        require(logits.size() == expected.size() && std::memcmp(logits.data(), expected.data(), logits.size() * sizeof(float)) == 0,
                "an AWQ directory whose version is GEMM did not load as its twin");
    }
    write_dir(std::nullopt, awq + R"(,"exllama":1)");
    refused(dir, "quantization_config exllama is not one llmx reads for awq", "an unknown quantization_config key was taken");
    write_dir(std::nullopt, R"("quant_method":"awq","bits":4,"group_size":32,"zero_point":true,"version":"gemv")");
    refused(dir, "quantization_config version: llmx reads AWQ's gemm packing only", "AWQ's gemv packing was taken");
    write_dir(std::nullopt, R"("quant_method":"awq","bits":4,"group_size":32,"zero_point":false,"version":"gemm")");
    refused(dir, "quantization_config zero_point: llmx reads AWQ with zero points only", "AWQ without zero points was taken");
    write_dir(std::nullopt, R"("quant_method":"exl2","bits":4,"group_size":32)");
    refused(dir, "quantization_config quant_method exl2 is not one llmx reads", "an unknown quant_method was taken");
    fs::remove_all(dir);
}

// One GPTQ linear [in, out] of layer `layer` in groups of 32 at `bits`: qweight packed along the inputs, qzeros along the outputs (each less one under v1), F16 scales and g_idx k / 32.
// The twin's rows are written here from the block layouts: Q4_0 for 4-bit symmetric zero points, Q5_0 for other 4-bit ones and Q8_0 for 8-bit symmetric ones.
struct GptqLinear {
    std::vector<uint32_t> qweight, qzeros, g_idx;
    std::vector<uint16_t> scales;
    uint32_t type = 0;
    std::vector<uint8_t> twin;
};

GptqLinear gptq_linear(size_t in, size_t out, size_t layer, unsigned bits, bool sym, bool v1) {
    static const float powers[5] = {1.0f / 64, -1.0f / 128, 1.0f / 256, -1.0f / 32, 1.0f / 1048576};
    const size_t per = 32 / bits, groups = in / 32, levels = size_t(1) << bits;
    const auto code = [&](size_t k, size_t n) { return uint32_t((n * 7 + k * 3 + layer) % levels); };
    // A v1 zero point is 1 to 16 and a v2 one 0 to 15, so each stores as 0 to 15.
    const auto zero = [&](size_t g, size_t n) { return sym ? uint32_t(levels / 2) : uint32_t((n + g + layer) % 16 + (v1 ? 1 : 0)); };
    GptqLinear a;
    a.qweight.assign(in / per * out, 0);
    a.qzeros.assign(groups * out / per, 0);
    a.scales.resize(groups * out);
    for (size_t g = 0; g < groups; ++g)
        for (size_t n = 0; n < out; ++n) {
            a.scales[g * out + n] = f32_to_f16(powers[(n + g + layer) % 5]);
            a.qzeros[g * (out / per) + n / per] |= ((zero(g, n) - (v1 ? 1 : 0)) & uint32_t(levels - 1)) << (bits * (n % per));
        }
    for (size_t k = 0; k < in; ++k) {
        a.g_idx.push_back(uint32_t(k / 32));
        for (size_t n = 0; n < out; ++n) a.qweight[(k / per) * out + n] |= code(k, n) << (bits * (k % per));
    }
    a.type = bits == 8 ? quant::GGML_TYPE_Q8_0 : sym ? quant::GGML_TYPE_Q4_0 : quant::GGML_TYPE_Q5_0;
    const size_t block = bits == 8 ? 34 : sym ? 18 : 22;
    a.twin.assign(out * groups * block, 0);
    for (size_t n = 0; n < out; ++n)
        for (size_t g = 0; g < groups; ++g) {
            uint8_t* b = a.twin.data() + (n * groups + g) * block;
            b[0] = uint8_t(a.scales[g * out + n]);
            b[1] = uint8_t(a.scales[g * out + n] >> 8);
            for (size_t j = 0; j < 32; ++j) {
                const uint32_t q = code(32 * g + j, n);
                if (bits == 8) {
                    // Q8_0: d, then the 32 signed codes q - 128.
                    b[2 + j] = uint8_t(int8_t(int(q) - 128));
                } else if (sym) {
                    // Q4_0: d, then codes j and j + 16 in the low and high nibbles of byte j, each q, whose value is q - 8.
                    b[2 + j % 16] |= uint8_t(q << (j < 16 ? 0 : 4));
                } else {
                    const uint32_t c = q + 16 - zero(g, n);
                    if (c & 16) b[2 + j / 8] |= uint8_t(1u << (j % 8));
                    b[6 + j % 16] |= uint8_t((c & 15) << (j < 16 ? 0 : 4));
                }
            }
        }
    return a;
}

// GPTQ directories of the 32-wide model, 4-bit symmetric under v1, 4-bit asymmetric under v2 and 8-bit symmetric under v1, each load as its Q4_0, Q5_0 or Q8_0 twin (twin_checks).
// A symmetric zero point off the middle code, act-order groups, 8-bit asymmetric zero points and quantization_config values llmx does not read are refused by name.
void gptq_checks(const std::string& path) {
    namespace fs = std::filesystem;
    const gguf::GGUFModel base = with_tokens(tiny_qwen_shaped(2, 2 * 128, true, 8, 2, 32, 16, 32));
    const fs::path dir = fs::u8path(path + ".gptq");
    fs::create_directories(dir);
    write_tokenizer(dir);
    gguf::GGUFModel twin;
    // `edit` changes the parts of layer 1's down projection before they are written.
    const auto write_dir = [&](unsigned bits, bool sym, bool v1, const std::string& quantization, const std::function<void(GptqLinear&)>& edit = {}) {
        return write_quant_dir(dir, base, quantization, twin, {}, [&](const gguf::TensorInfo& t, size_t layer, const std::string& hf, std::vector<StEntry>& entries) {
            GptqLinear a = gptq_linear(size_t(t.ne[0]), size_t(t.ne[1]), layer, bits, sym, v1);
            if (edit && t.name == "blk.1.ffn_down.weight") edit(a);
            const uint64_t per = 32 / bits;
            entries.push_back({hf + ".qweight", "I32", {t.ne[1], t.ne[0] / per}, bytes_of(a.qweight)});
            entries.push_back({hf + ".qzeros", "I32", {t.ne[1] / per, t.ne[0] / 32}, bytes_of(a.qzeros)});
            entries.push_back({hf + ".scales", "F16", {t.ne[1], t.ne[0] / 32}, bytes_of(a.scales)});
            entries.push_back({hf + ".g_idx", "I32", {t.ne[0]}, bytes_of(a.g_idx)});
            return std::make_pair(a.type, a.twin);
        });
    };
    const auto config = [](unsigned bits, bool sym, bool v1) {
        return std::string(R"("quant_method":"gptq","bits":)") + std::to_string(bits) + R"(,"group_size":32,"desc_act":false,"sym":)" + (sym ? "true" : "false") +
               R"(,"checkpoint_format":")" + (v1 ? "gptq" : "gptq_v2") + R"(","lm_head":false,"pack_dtype":"int32","meta":{"quantizer":["gptqmodel:4.0.0"]})";
    };
    for (const auto& [bits, sym, v1, label] : {std::make_tuple(4u, true, true, "a 4-bit symmetric GPTQ directory"), std::make_tuple(4u, false, false, "a 4-bit GPTQ v2 directory"),
                                               std::make_tuple(8u, true, true, "an 8-bit GPTQ directory")}) {
        const size_t linears = write_dir(bits, sym, v1, config(bits, sym, v1));
        twin_checks(dir, path, twin, linears, label);
    }
    write_dir(4, true, true, config(4, true, true), [](GptqLinear& a) { a.qzeros[0] ^= 1; });
    refused(dir, "tensor blk.1.ffn_down.weight: zero point 7 of output 0, group 0 is not 8, as a symmetric GPTQ's is", "a symmetric zero point off the middle was taken");
    write_dir(4, true, true, config(4, true, true), [](GptqLinear& a) { a.g_idx[0] = 1; });
    refused(dir, "tensor blk.1.ffn_down.weight: g_idx of input 0 is 1, not its group 0", "act-order groups were taken");
    write_dir(8, true, true, config(8, false, true));
    refused(dir, "quantization_config sym: llmx reads 8-bit GPTQ with symmetric zero points only", "8-bit asymmetric GPTQ was taken");
    write_dir(4, true, true, R"("quant_method":"gptq","bits":3,"group_size":32)");
    refused(dir, "quantization_config bits: llmx reads 4- and 8-bit GPTQ only", "3-bit GPTQ was taken");
    write_dir(4, true, true, R"("quant_method":"gptq","bits":4,"group_size":-1)");
    refused(dir, "quantization_config group_size: llmx reads GPTQ groups of a positive multiple of 32 inputs", "GPTQ with one group a row was taken");
    write_dir(4, true, true, R"("quant_method":"gptq","bits":4,"group_size":32,"checkpoint_format":"marlin")");
    refused(dir, "quantization_config checkpoint_format: llmx reads GPTQ's gptq and gptq_v2 formats only", "GPTQ's marlin format was taken");
    write_dir(4, true, true, R"("quant_method":"gptq","bits":4,"group_size":32,"lm_head":true)");
    refused(dir, "quantization_config lm_head: llmx reads GPTQ with an unquantized lm_head only", "a quantized GPTQ head was taken");
    write_dir(4, true, true, config(4, true, true) + R"(,"zero_point":true)");
    refused(dir, "quantization_config zero_point is not one llmx reads for gptq", "an unknown GPTQ key was taken");
    fs::remove_all(dir);
}

// A model file's weights have no bytes until its payload is mapped.
// Loaded through infer::load_model it keeps its payload on the CPU and releases it on a backend that copies every weight, and both give the logits of the same model built in memory.
void loader_checks(const std::string& path) {
    const gguf::GGUFModel source = with_tokens(tiny_qwen(2, 2 * 128, false));
    gguf::write_gguf(source, path);
    {
        gguf::GGUFModel file = gguf::read_gguf(path);
        const infer::ModelWeights unmapped = infer::gguf_weights(file);
        require(std::all_of(unmapped.tensors.begin(), unmapped.tensors.end(), [](const infer::TensorView& t) { return !t.data; }),
                "a weight had bytes before its file was mapped");
        gguf::map_payload(file);
        const infer::ModelWeights mapped = infer::gguf_weights(file);
        for (size_t i = 0; i < mapped.tensors.size(); ++i)
            require(mapped.tensors[i].data && std::memcmp(mapped.tensors[i].data, source.tensor_data(i), source.tensor_bytes(i)) == 0,
                    "a mapped weight's bytes differ from the file's");
    }
    const std::vector<uint32_t> ids = {0, 1, 2, 3, 4};
    const std::vector<float> expected = infer::Model(infer::gguf_weights(source), backend::make_cpu_backend()).prefill(ids);
    size_t payload = 0;
    for (size_t i = 0; i < source.tensors.size(); ++i) payload += source.tensor_bytes(i);
    for (const infer::LoadMode mode : load_modes(path)) {
        for (const int arm : {0, 1, 2}) {
            const bool copying = arm != 0;
            const auto copier = std::make_shared<CopyingBackend>();
            copier->wraps = arm == 1;
            const backend::BackendPtr b = copying ? backend::BackendPtr(copier) : backend::make_cpu_backend();
            infer::PlacementRequest request;
            request.names = {"cpu"};
            Rising progress;
            const auto loaded = infer::load_model(path, {b}, request, {}, progress.callback(), mode);
            require(progress.whole(payload), "a load's progress did not rise from 0 to the payload");
            // A streamed load's copies all hold file tensors' bytes; a mapped one adopts, and allocates no weight storage.
            const bool streams = mode != infer::LoadMode::mapped;
            if (copying)
                require(streams ? copier->weights.size() == source.tensors.size() && holds_tensors(copier->weights, source) : copier->weights.empty(),
                        "a copying backend's weights do not hold the file's bytes, or a mapped load allocated weight storage");
            // A backend that reads the ring in place takes every part as a copy out of it, and one that does not as a write.
            if (copying && streams)
                require(arm == 1 ? copier->copies > 0 && copier->copies == copier->writes : copier->copies == 0,
                        "a streamed load wrote where the backend reads the ring in place, or copied where it does not");
            require(copying ? loaded->file.payload_size() == 0 && loaded->host.empty() : loaded->file.payload_size() != 0,
                    copying ? "a model no host reads kept the host copy" : "a model the CPU reads lost its payload");
            // auto streams a copying backend's weights through the cache; direct reads every file around it, into its own copy where the CPU reads in place.
            const bool direct = mode == infer::LoadMode::direct;
            require(loaded->times.mode == mode && loaded->times.files == (copying && mode == infer::LoadMode::automatic ? 1u : 0u) &&
                        loaded->times.direct_files == (direct ? 1u : 0u) && loaded->host.size() == (direct && !copying ? 1u : 0u),
                    "a load read where its mode and backends say it should not, or did not where they say it should");
            // direct maps nothing, and the CPU's copy holds each weight where the file does.
            if (direct && !copying)
                for (size_t i = 0; i < source.tensors.size(); ++i)
                    require(!loaded->file.tensor_data(i) &&
                                std::memcmp(loaded->host[0]->data() + loaded->file.span(i).offset, source.tensor_data(i), source.tensor_bytes(i)) == 0,
                            "a direct load mapped the file, or its copy does not hold a weight where the file does");
            const std::vector<float> logits = loaded->model->prefill(ids);
            require(logits.size() == expected.size() &&
                        std::memcmp(logits.data(), expected.data(), logits.size() * sizeof(float)) == 0,
                    "a loaded model's logits differ from the model built in memory");
        }
    }
    // A write, or the progress, that fails part way through a streamed load stops it with its error, on a backend whose uploads stay outstanding until it drains, and no storage goes before the drain.
    for (const int failure : {1, 3, 0}) {
        auto b = std::make_shared<LoadingBackend>();
        b->fail_write = failure;
        infer::PlacementRequest request;
        request.names = {"cpu"};
        std::string error;
        const infer::LoadReport progress = [&](const infer::LoadStep& step) {
            if (!failure && step.completed) throw std::runtime_error("progress failure");
        };
        try { infer::load_model(path, {b}, request, {}, progress, infer::LoadMode::automatic); }
        catch (const std::runtime_error& e) { error = e.what(); }
        require(error == (failure ? "injected write failure" : "progress failure"), "a streamed load lost a write's or the progress's failure");
        require(b->state->writes > 0 && b->state->premature == 0 && b->state->drains > 0 && b->state->releases > 0,
                "a failed streamed load freed storage before its backend drained");
    }
    {
        infer::PlacementRequest request;
        request.names = {"cpu"};
        const auto loaded = infer::load_model(path, {backend::make_cpu_backend()}, request);
        require(loaded->tok && loaded->tok->encode("abcde") == ids && loaded->chat.eos == "p",
                "the loader lost the tokenizer or the chat format");
        require(loaded->plan.empty(), "one device reported a split");
    }
    template_checks(path, false);
}

// What a SourceArchitecture saw of its source, and whether its fill fails after reading it, as a model that fails as it is built does.
struct SourceSeen {
    std::vector<uint8_t> bytes;
    bool fail = false;
};

// qwen3 with one more role, a source (infer::RoleKind::source) that fills one more table, whose bytes as fill_tables sees them it keeps.
class SourceArchitecture final : public infer::Architecture {
public:
    SourceArchitecture(std::shared_ptr<const infer::Architecture> base, std::shared_ptr<SourceSeen> seen) : base_(std::move(base)), seen_(std::move(seen)) {}
    infer::ModelPlan plan(const infer::TensorIndex& tensors) const override {
        infer::ModelPlan p = base_->plan(tensors);
        const infer::TensorView& t = tensors.view(tensors.at("blk.0.src.weight"));
        p.layers[0].roles.push_back({uint16_t(p.role_ids), infer::Part::mixer, infer::RoleKind::source, t.name, "", t.shape[0], t.shape[1]});
        ++p.role_ids;
        p.tables.push_back(size_t(t.shape[0] * t.shape[1]));
        return p;
    }
    void fill_tables(std::vector<std::vector<float>>& tables, const std::vector<infer::TensorView>& sources) const override {
        require(sources.size() == 1 && sources[0].data && sources[0].name == "blk.0.src.weight", "fill_tables was not handed the source's host view");
        seen_->bytes.assign(sources[0].data, sources[0].data + sources[0].bytes);
        if (seen_->fail) throw std::runtime_error("injected table failure");
        std::vector<std::vector<float>> own(tables.begin(), tables.end() - 1);
        base_->fill_tables(own, {});
        std::copy(own.begin(), own.end(), tables.begin());
    }
    void embed(const infer::Step& s, const uint32_t* ids) const override { base_->embed(s, ids); }
    void pass_values(const infer::Step& s) const override { base_->pass_values(s); }
    void mixer(const infer::Step& s) const override { base_->mixer(s); }
    void ffn(const infer::Step& s) const override { base_->ffn(s); }
    void head(const infer::HeadStep& s) const override { base_->head(s); }

private:
    std::shared_ptr<const infer::Architecture> base_;
    std::shared_ptr<SourceSeen> seen_;
};

// A source's unaligned span reaches fill_tables in every mode on the CPU, a copying backend and a split of two, and nothing places it (docs/TESTS.md).
// A load that fails after reading it, as its model is built, leaves the next load whole.
void source_checks(const std::string& path) {
    gguf::GGUFModel source = with_tokens(tiny_qwen(2, 2 * 128, false));
    const std::vector<uint32_t> ids = {0, 1, 2, 3, 4};
    const std::vector<float> expected = infer::Model(infer::gguf_weights(source), backend::make_cpu_backend()).prefill(ids);
    {
        const uint64_t width = source.tensors[0].ne[0];
        gguf::TensorInfo t{"blk.0.src.weight", {width, 2}, quant::GGML_TYPE_BF16, 0};
        std::vector<uint8_t> bytes(size_t(t.data_size()));
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(i * 37 + 11);
        source.tensors.push_back(t);
        source.add_tensor_data(bytes);
    }
    gguf::write_gguf(source, path);
    const size_t last = source.tensors.size() - 1;
    const std::vector<uint8_t> want(source.tensor_data(last), source.tensor_data(last) + source.tensor_bytes(last));
    {
        // The fit counts the table and not the source: layer 0's weights are the model's without it, and the tables hold one more of width x 2 floats.
        infer::ModelWeights plain = infer::gguf_weights(source), with = plain;
        with.arch = std::make_shared<SourceArchitecture>(plain.arch, std::make_shared<SourceSeen>());
        const infer::Footprint a = infer::footprint(plain, infer::plan_model(plain), {}), b = infer::footprint(with, infer::plan_model(with), {});
        size_t wa = 0, wb = 0;
        for (const infer::Matrix& m : a.layers[0]) wa += m.bytes;
        for (const infer::Matrix& m : b.layers[0]) wb += m.bytes;
        require(wa == wb && b.tables == a.tables + want.size() * 2, "the fit counted a source's bytes, or not its table");
        infer::Model model(with, backend::make_cpu_backend());
        require(model.prefill(ids) == expected, "a model built in memory with a source gives other logits");
        // The loader's hook records no upload of it and no host read of it, deferred or not.
        for (const bool defer : {true, false}) {
            infer::WeightPlan plan;
            { infer::Model hooked(with, {std::make_shared<LoadingBackend>()}, infer::Placement{}, {}, infer::planning_adopt(with, 1, plan, defer, {}, {})); }
            require(plan.host_reads[last] == 0 && std::none_of(plan.uploads.begin(), plan.uploads.end(), [&](const infer::Upload& u) { return u.tensor == last; }),
                    "the loader's hook recorded the source");
        }
        // A source without bytes, as a view of a file not mapped has, is refused by name as the model is built.
        infer::ModelWeights empty = with;
        empty.tensors[last].data = nullptr;
        std::string error;
        try { infer::Model refused(empty, backend::make_cpu_backend()); } catch (const std::runtime_error& e) { error = e.what(); }
        require(error == "inference: no bytes for the source tensor blk.0.src.weight", "a source without bytes was not refused by name");
    }
    const auto load = [&](infer::LoadMode mode, std::vector<backend::BackendPtr> backends, const std::shared_ptr<SourceSeen>& seen) {
        auto loaded = std::make_unique<infer::LoadedModel>();
        loaded->file = gguf::read_gguf(path);
        loaded->tok.emplace(loaded->file);
        infer::ModelWeights weights = infer::gguf_weights(loaded->file);
        weights.arch = std::make_shared<SourceArchitecture>(weights.arch, seen);
        infer::PlacementRequest request;
        request.names.assign(backends.size(), "cpu");
        if (backends.size() > 1) request.shares.assign(backends.size(), 1);
        return infer::detail::load_weights(std::move(loaded), std::move(weights), std::move(backends), request, {}, {}, mode, false);
    };
    require(gguf::read_gguf(path).span(last).offset % core::page_size() != 0, "the source's span lies on a page, so no unaligned read is made");
    for (const infer::LoadMode mode : load_modes(path))
        for (const int arm : {0, 1, 2}) {
            auto seen = std::make_shared<SourceSeen>();
            std::vector<std::shared_ptr<CopyingBackend>> copiers;
            std::vector<backend::BackendPtr> backends;
            if (arm == 0) backends.push_back(backend::make_cpu_backend());
            for (int d = 0; d < arm; ++d) {
                copiers.push_back(std::make_shared<CopyingBackend>());
                backends.push_back(copiers.back());
            }
            const auto loaded = load(mode, backends, seen);
            require(seen->bytes == want, "fill_tables did not see the source's bytes from the file");
            for (const auto& c : copiers)
                require(std::none_of(c->weights.begin(), c->weights.end(), [&](const backend::BufferPtr& w) {
                            return w->size() == want.size() && std::memcmp(w->host_ptr(), want.data(), w->size()) == 0;
                        }) && (mode == infer::LoadMode::mapped || !c->weights.empty()),
                        "a copying backend took the source");
            if (arm == 1 && mode != infer::LoadMode::mapped)
                require(copiers[0]->weights.size() == source.tensors.size() - 1, "a streamed load placed the source or left another weight out");
            const std::vector<float> logits = loaded->model->prefill(ids);
            require(logits.size() == expected.size() && std::memcmp(logits.data(), expected.data(), logits.size() * sizeof(float)) == 0,
                    "a model with a source gives other logits than the model without it");
        }
    // A load whose model fails as it is built, after its source was read, then a whole one.
    for (const infer::LoadMode mode : load_modes(path)) {
        auto seen = std::make_shared<SourceSeen>();
        seen->fail = true;
        std::string error;
        try { load(mode, {std::make_shared<CopyingBackend>()}, seen); } catch (const std::runtime_error& e) { error = e.what(); }
        require(error == "injected table failure" && seen->bytes == want, "a load failing after its source was read lost the failure");
        seen->fail = false;
        seen->bytes.clear();
        const auto loaded = load(mode, {std::make_shared<CopyingBackend>()}, seen);
        require(seen->bytes == want && loaded->model->prefill(ids) == expected, "the load after a failed one was not whole");
    }
}

// A small qwen3moe model with tokenizer metadata (tiny_qwen_moe).
gguf::GGUFModel tiny_moe() { return with_tokens(tiny_qwen_moe(2, 2 * 128, false)); }

// Experts on the CPU beside a device: the placement adds a CPU backend that reads the experts in place, so each load mode maps the file for it though the device copies, and the logits are those of the same model built in memory.
void experts_checks(const std::string& path) {
    const gguf::GGUFModel source = tiny_moe();
    gguf::write_gguf(source, path);
    const std::vector<uint32_t> ids = {0, 1, 2, 3, 4};
    const std::vector<float> expected = infer::Model(infer::gguf_weights(source), backend::make_cpu_backend()).prefill(ids);
    for (const infer::LoadMode mode : load_modes(path)) {
        infer::PlacementRequest request;
        request.names = {"cpu"};
        request.cpu_moe = -1;
        const auto loaded = infer::load_model(path, {std::make_shared<CopyingBackend>()}, request, {}, {}, mode);
        require(loaded->file.payload_size() != 0, "experts on the CPU lost the payload they read in place");
        const std::vector<float> logits = loaded->model->prefill(ids);
        require(logits.size() == expected.size() && std::memcmp(logits.data(), expected.data(), logits.size() * sizeof(float)) == 0,
                "a model with its experts on the CPU gives other logits than the model built in memory");
    }
}

// A split over two copying devices with a tied head: the embedding reaches the first device and, as the head, the second, each write goes to storage its own device gave, and the logits are those of the model built in memory.
// Then the CPU in place of the first device, so one weight is both read in place and copied.
void split_checks(const std::string& path) {
    const gguf::GGUFModel source = with_tokens(tiny_qwen(2, 2 * 128, true));
    gguf::write_gguf(source, path);
    const std::vector<uint32_t> ids = {0, 1, 2, 3, 4};
    const std::vector<float> expected = infer::Model(infer::gguf_weights(source), backend::make_cpu_backend()).prefill(ids);
    size_t embedding = 0;
    while (source.tensors[embedding].name != "token_embd.weight") ++embedding;
    for (const infer::LoadMode mode : load_modes(path)) {
        auto a = std::make_shared<CopyingBackend>(), b = std::make_shared<CopyingBackend>();
        infer::PlacementRequest request;
        request.names = {"cpu", "cpu"};
        request.shares = {1, 1};
        const auto loaded = infer::load_model(path, {a, b}, request, {}, {}, mode);
        if (mode != infer::LoadMode::mapped) {
            auto takes = [&](const CopyingBackend& d) {
                return std::any_of(d.weights.begin(), d.weights.end(), [&](const backend::BufferPtr& w) {
                    return w->size() == source.tensor_bytes(embedding) && std::memcmp(w->host_ptr(), source.tensor_data(embedding), w->size()) == 0;
                });
            };
            require(holds_tensors(a->weights, source) && holds_tensors(b->weights, source) && takes(*a) && takes(*b),
                    "a split's devices do not both hold the tied embedding, or hold other bytes than the file's");
            require(a->foreign == 0 && b->foreign == 0, "a stream wrote into storage another device gave");
        }
        const std::vector<float> logits = loaded->model->prefill(ids);
        require(logits.size() == expected.size() && std::memcmp(logits.data(), expected.data(), logits.size() * sizeof(float)) == 0,
                "a split load gives other logits than the model built in memory");
    }
    // The CPU reading its half in place beside a copying device: the tied embedding the CPU reads and the device's head copies is read and reported once, in direct from the CPU's copy, and reaches the device whole.
    size_t payload = 0;
    for (size_t i = 0; i < source.tensors.size(); ++i) payload += source.tensor_bytes(i);
    for (const infer::LoadMode mode : load_modes(path)) {
        auto device = std::make_shared<CopyingBackend>();
        infer::PlacementRequest request;
        request.names = {"cpu", "cpu"};
        request.shares = {1, 1};
        Rising progress;
        const auto loaded = infer::load_model(path, {backend::make_cpu_backend(), device}, request, {}, progress.callback(), mode);
        require(progress.whole(payload), "a split with the CPU reading in place reported a weight twice");
        if (mode != infer::LoadMode::mapped)
            require(holds_tensors(device->weights, source) &&
                        std::any_of(device->weights.begin(), device->weights.end(), [&](const backend::BufferPtr& w) {
                            return w->size() == source.tensor_bytes(embedding) && std::memcmp(w->host_ptr(), source.tensor_data(embedding), w->size()) == 0;
                        }),
                    "a device beside the CPU does not hold the tied embedding, or holds other bytes than the file's");
        const std::vector<float> logits = loaded->model->prefill(ids);
        require(logits.size() == expected.size() && std::memcmp(logits.data(), expected.data(), logits.size() * sizeof(float)) == 0,
                "a split with the CPU reading in place gives other logits than the model built in memory");
    }
}

// A set of three shards, the first holding metadata alone: each load mode reads the two with tensors, and the model gives the logits of the one built in memory.
void shard_checks(const std::filesystem::path& dir) {
    const gguf::GGUFModel source = with_tokens(tiny_qwen(2, 2 * 128, false));
    const size_t half = source.tensors.size() / 2;
    auto split_keys = [](gguf::GGUFModel& m, uint64_t no, uint64_t tensors) {
        auto key = [&](const std::string& name, uint32_t type, uint64_t value) {
            gguf::MetaValue v;
            v.vtype = type;
            if (type == gguf::V_INT32) v.i = int64_t(value);   // signed values are kept in `i`
            else v.u = value;
            m.kv.push_back({name, v});
        };
        key("split.no", gguf::V_UINT16, no);
        key("split.count", gguf::V_UINT16, 3);
        key("split.tensors.count", gguf::V_INT32, tensors);
    };
    std::vector<std::string> paths;
    for (uint64_t no = 0; no < 3; ++no) {
        gguf::GGUFModel shard;
        if (no == 0) shard.kv = source.kv;
        split_keys(shard, no, source.tensors.size());
        const size_t from = no == 0 ? 0 : no == 1 ? 0 : half, to = no == 0 ? 0 : no == 1 ? half : source.tensors.size();
        for (size_t i = from; i < to; ++i) {
            shard.tensors.push_back(source.tensors[i]);
            const uint8_t* data = source.tensor_data(i);
            shard.add_tensor_data(std::vector<uint8_t>(data, data + source.tensor_bytes(i)));
        }
        const std::string name = "load-shard-0000" + std::to_string(no + 1) + "-of-00003.gguf";
        paths.push_back((dir / name).u8string());
        gguf::write_gguf(shard, paths.back());
    }
    const std::vector<uint32_t> ids = {0, 1, 2, 3, 4};
    const std::vector<float> expected = infer::Model(infer::gguf_weights(source), backend::make_cpu_backend()).prefill(ids);
    for (const infer::LoadMode mode : load_modes(paths[0], paths[1])) {
        for (const int arm : {0, 1, 2}) {
            const bool copying = arm != 0;
            const auto copier = std::make_shared<CopyingBackend>();
            copier->wraps = arm == 1;
            const backend::BackendPtr b = copying ? backend::BackendPtr(copier) : backend::make_cpu_backend();
            infer::PlacementRequest request;
            request.names = {"cpu"};
            const auto loaded = infer::load_model(paths[0], {b}, request, {}, {}, mode);
            const size_t files = loaded->times.files + loaded->times.direct_files;
            require(mode == infer::LoadMode::mapped || (!copying && mode == infer::LoadMode::automatic) ? files == 0 : files == 2,
                    "a sharded load streamed from other than its two shards with tensors");
            if (copying && mode != infer::LoadMode::mapped) require(holds_tensors(copier->weights, source), "a sharded load's copies differ from the file's");
            const std::vector<float> logits = loaded->model->prefill(ids);
            require(logits.size() == expected.size() && std::memcmp(logits.data(), expected.data(), logits.size() * sizeof(float)) == 0,
                    "a sharded load gives other logits than the model built in memory");
        }
    }
    for (const auto& p : paths) std::filesystem::remove(std::filesystem::u8path(p));
}

// The reads the stream plans, on spans that need no file: pieces start and end on the granule and hold at most the limit, a gap of one granule is read through and a longer one starts a piece, and every tensor's bytes lie in exactly one part.
void plan_checks() {
    const size_t g = 7168, limit = size_t(16) << 20, cap = limit / g * g;
    const std::vector<format::FileSpan> spans = {
        {"a", 0, 1000},                  // 0
        {"a", 1000 + 7168, 500},         // 1: a gap of exactly one granule, read through
        {"a", 1000 + 7168 + 500 + 7169, 300},   // 2: one granule and a byte, a new piece
        {"a", 40000, 3 * cap + 5},       // 3: crosses four pieces
        {"b", 0, 0},                     // 4: nothing to read, at the next file's start
        {"b", 0, 2000},                  // 5: a new file
    };
    const std::vector<size_t> tensors = {0, 1, 2, 3, 4, 5}, file_of = {0, 0, 0, 0, 1, 1};
    const auto pieces = infer::detail::plan_pieces(tensors, spans, file_of, {g, g}, limit);
    std::vector<size_t> covered(spans.size(), 0);
    for (size_t k = 0; k < pieces.size(); ++k) {
        const auto& p = pieces[k];
        require(p.offset % g == 0 && p.bytes % g == 0 && p.bytes <= cap && !p.parts.empty(), "a planned read is not whole granules within the limit");
        if (k && pieces[k - 1].file == p.file) require(pieces[k - 1].offset + pieces[k - 1].bytes <= p.offset, "two planned reads overlap");
        for (const auto& part : p.parts) {
            require(part.tensor_offset == covered[part.tensor] && p.offset + part.piece_offset == spans[part.tensor].offset + part.tensor_offset &&
                        part.piece_offset + part.bytes <= p.bytes && file_of[part.tensor] == p.file,
                    "a planned part is not where its tensor's bytes are");
            covered[part.tensor] += part.bytes;
        }
    }
    for (size_t t = 0; t < spans.size(); ++t) require(covered[t] == spans[t].bytes, "a planned read left out or repeated a tensor's bytes");
    require(pieces.size() == 7 && pieces[0].parts.size() == 2 && pieces[1].parts.size() == 1 && pieces[6].file == 1,
            "the reads are not the ones the gaps, the limit and the files call for");
}

// auto reads around the file cache only when the bytes it streams are more than the host has, and a file system that refuses direct reads gives a reader through the cache instead.
void choice_checks(const std::string& path) {
    require(!infer::detail::reads_around(5, std::nullopt) && !infer::detail::reads_around(5, 5) && infer::detail::reads_around(6, 5) &&
                !infer::detail::reads_around(0, 0),
            "auto reads around the cache at other bytes than more than the host has");
    gguf::write_gguf(with_tokens(tiny_qwen(1, 64, false)), path);
    bool direct = true;
    try { format::FileReader probe(path, true); } catch (const format::DirectUnavailable&) { direct = false; }
    require(infer::detail::open_reader(path, true)->direct() == direct && !infer::detail::open_reader(path, false)->direct(),
            "a reader asked to read around the cache did not, where the file system takes it, or did where it does not");
}

// The stream itself, in reads of one granule: some tensors cross reads and some reads hold several tensors, a tensor two backends take reaches both, and every copy holds the file's bytes.
// A read that comes up short, because the file was cut after its header was read, stops the stream with where the file ended.
void stream_checks(const std::string& path) {
    const gguf::GGUFModel source = tiny_qwen(2, 2 * 128, false);
    gguf::write_gguf(source, path);
    gguf::GGUFModel file = gguf::read_gguf(path);
    auto cpu = std::make_shared<CopyingBackend>();
    std::vector<infer::Upload> uploads;
    for (size_t i = 0; i < file.tensors.size(); ++i) {
        uploads.push_back({i, cpu.get(), cpu->alloc_weight(file.tensor_bytes(i)), {}});
        if (i == 0) uploads.push_back({i, cpu.get(), cpu->alloc_weight(file.tensor_bytes(i)), {}});
    }
    auto streamed = [&](const std::string& p, gguf::GGUFModel& f) {
        std::vector<std::shared_ptr<const format::FileReader>> readers;
        readers.push_back(std::make_unique<format::FileReader>(p));
        std::vector<size_t> order(f.tensors.size()), file_of(f.tensors.size(), 0);
        std::vector<format::FileSpan> spans(f.tensors.size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
            spans[i] = f.span(i);
        }
        // Reads planned on a page, whatever block the file system reports, so this small fixture crosses them; buffered reads need no alignment.
        const size_t granule = core::page_size();
        const auto pieces = infer::detail::plan_pieces(order, spans, file_of, {granule}, granule);
        bool crossing = false, shared = false;
        for (const auto& piece : pieces) {
            shared = shared || piece.parts.size() > 1;
            for (const auto& part : piece.parts) crossing = crossing || part.bytes < spans[part.tensor].bytes;
            require(piece.offset % granule == 0 && piece.bytes % granule == 0 && piece.bytes <= granule, "a read is not on the granule");
        }
        require(crossing && shared, "the fixture does not make tensors cross reads and reads hold several tensors");
        std::vector<std::vector<const infer::Upload*>> destinations(f.tensors.size());
        for (const auto& u : uploads) destinations[u.tensor].push_back(&u);
        infer::LoadTimes times;
        size_t bytes = 0;
        infer::detail::stream(pieces, readers, destinations, [&](size_t, size_t n) { bytes += n; }, times);
        return bytes;
    };
    size_t payload = 0;
    for (size_t i = 0; i < source.tensors.size(); ++i) payload += source.tensor_bytes(i);
    require(streamed(path, file) == payload, "the stream reported other bytes than its tensors'");
    for (const auto& u : uploads)
        require(std::memcmp(u.buffer->host_ptr(), source.tensor_data(u.tensor), source.tensor_bytes(u.tensor)) == 0,
                "a streamed copy differs from the file");
    // The writes follow the file: each lands at or after the one before, where the tensor it fills lies in the file, and the two copies of tensor 0 take each part in turn.
    uint64_t last = 0;
    for (size_t w = 0; w < cpu->written.size(); ++w) {
        const auto [dst, off] = cpu->written[w];
        size_t u = 0;
        while (uploads[u].buffer.get() != dst) ++u;
        const uint64_t at = file.span(uploads[u].tensor).offset + off;
        require(at >= last, "a streamed write went back in the file");
        last = at;
        if (u == 0) require(w + 1 < cpu->written.size() && cpu->written[w + 1] == std::make_pair((const backend::Buffer*)uploads[1].buffer.get(), off),
                            "a part of a tensor with two copies did not go to both before the next");
    }
    // Dozens of 64-byte reads through the four slots, the writes slow enough that both readers fill the ring and wait: every copy still holds the file's bytes, and a write failing late stops the stream while the readers wait on the ring.
    {
        std::vector<std::shared_ptr<const format::FileReader>> readers;
        readers.push_back(std::make_unique<format::FileReader>(path));
        std::vector<size_t> order(file.tensors.size()), file_of(file.tensors.size(), 0);
        std::vector<format::FileSpan> spans(file.tensors.size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
            spans[i] = file.span(i);
        }
        const auto pieces = infer::detail::plan_pieces(order, spans, file_of, {64}, 64);
        require(pieces.size() > 40, "the fixture does not make dozens of 64-byte reads");
        for (const int failure : {0, 30, -1, -2}) {
            // -1 fills by writes, and -2 by copies that run only when waited for.
            auto late = failure == -2 ? std::make_shared<DeferredBackend>() : std::make_shared<CopyingBackend>();
            late->slow = 40;
            late->fail_write = std::max(failure, 0);
            late->wraps = failure != -1;
            std::vector<infer::Upload> copies;
            for (size_t i = 0; i < file.tensors.size(); ++i) copies.push_back({i, late.get(), late->alloc_weight(file.tensor_bytes(i)), {}});
            std::vector<std::vector<const infer::Upload*>> destinations(file.tensors.size());
            for (const auto& u : copies) destinations[u.tensor].push_back(&u);
            infer::LoadTimes times;
            std::string error;
            try { infer::detail::stream(pieces, readers, destinations, {}, times); } catch (const std::runtime_error& e) { error = e.what(); }
            require(failure > 0 ? error == "injected write failure" : error.empty() && holds_tensors(late->weights, source) && (late->copies == 0) == (failure == -1),
                    "a stream of many reads through the ring lost bytes, a late fill's failure, or refilled a slot before the copies out of it ran");
        }
        // Two members of a tensor group take their shards of every role (model/shard.hpp), runs crossing the 64-byte reads, by copies, by writes and by copies that run only when waited for: each member's storage holds its packed shard and no poison.
        {
            const infer::ModelWeights weights = infer::gguf_weights(file);
            const infer::ModelPlan plan = infer::plan_model(weights);
            std::vector<const infer::Role*> roles;
            for (const infer::Role& r : plan.pass) roles.push_back(&r);
            for (const infer::LayerPlan& layer : plan.layers)
                for (const infer::Role& r : layer.roles) roles.push_back(&r);
            for (const int fill : {0, -1, -2}) {
                auto member = fill == -2 ? std::make_shared<DeferredBackend>() : std::make_shared<CopyingBackend>();
                member->wraps = fill != -1;
                std::vector<infer::Upload> shards;
                std::vector<std::vector<uint8_t>> wanted;
                size_t split = 0;
                for (const infer::Role* r : roles) {
                    if (!r->tensor) continue;
                    infer::TensorView t = weights.tensors[*r->tensor];
                    t.data = source.tensor_data(*r->tensor);
                    for (size_t m = 0; m < 2; ++m) {
                        const infer::shard::Held held = infer::shard::held(*r, t, 2, m);
                        size_t runs = 0;
                        infer::shard::file_runs(held, 0, t.bytes, [&](const infer::shard::Run&) { ++runs; });
                        split += runs > 1;
                        wanted.emplace_back(held.bytes);
                        infer::shard::pack(held, t.data, wanted.back().data());
                        shards.push_back({*r->tensor, member.get(), member->alloc_weight(wanted.back().size()), held});
                    }
                }
                require(split >= 4, "the fixture's members do not take shards of several runs");
                std::vector<std::vector<const infer::Upload*>> destinations(file.tensors.size());
                for (const auto& u : shards) destinations[u.tensor].push_back(&u);
                infer::LoadTimes times;
                infer::detail::stream(pieces, readers, destinations, {}, times);
                for (size_t k = 0; k < shards.size(); ++k)
                    require(shards[k].buffer->size() == wanted[k].size() && std::memcmp(shards[k].buffer->host_ptr(), wanted[k].data(), wanted[k].size()) == 0,
                            "a member's streamed shard differs from its packed bytes");
            }
        }
        // A read that fails on a reader thread, a direct read off its granule where the file system takes direct reads, comes out of the stream with the readers joined.
        try {
            std::vector<std::shared_ptr<const format::FileReader>> direct;
            direct.push_back(std::make_unique<format::FileReader>(path, true));
            auto skewed = infer::detail::plan_pieces(order, spans, file_of, {direct[0]->granule()}, direct[0]->granule());
            skewed.back().bytes += 1;
            std::vector<std::vector<const infer::Upload*>> none(file.tensors.size());
            infer::LoadTimes times;
            bool refused = false;
            try { infer::detail::stream(skewed, direct, none, {}, times); } catch (const std::logic_error&) { refused = true; }
            require(refused, "a reader thread's failing read did not come out of the stream");
        } catch (const format::DirectUnavailable&) {
        }
    }
    const auto length = std::filesystem::file_size(path);
    std::filesystem::resize_file(path, length - 4096);
    std::string error;
    try { streamed(path, file); } catch (const std::runtime_error& e) { error = e.what(); }
    require(error.rfind(path + " ended at ", 0) == 0 && error.find(" bytes, before its tensors") != std::string::npos,
            "a stream over a file cut after its header was read did not say where it ended");
}

// The read ring stepped without waiting, as the server drives it (ReadRing), its cases listed in docs/TESTS.md.
void ring_checks() {
    // A ring not started has no reader, so its pages can be set up, and fail to be, before any fill runs, and it goes with nothing to join.
    {
        std::atomic<size_t> unstarted{0};
        {
            infer::ReadRing ring(3, 16, [&](size_t, uint8_t*) -> size_t { return ++unstarted, 16; }, 2, 2);
            require(ring.idle() && !ring.next(false) && ring.pages(0).data(), "a ring not started was not idle with its pages");
        }
        require(unstarted.load() == 0, "a ring not started ran its fill");
    }
    std::mutex m;
    std::condition_variable cv;
    bool open = false;   // the first fill waits until the test opens it
    size_t fills = 0;
    {
        infer::ReadRing ring(6, 64, [&](size_t k, uint8_t* into) {
            std::unique_lock<std::mutex> lk(m);
            cv.wait(lk, [&] { return open || k > 0; });
            ++fills;
            std::fill(into, into + 64, uint8_t(k + 1));
            return size_t(k + 1);
        }, 2, 1);
        ring.start();
        require(!ring.next(false), "a ring gave a piece not yet read");
        {
            std::lock_guard<std::mutex> lk(m);
            open = true;
        }
        cv.notify_all();
        std::optional<infer::ReadRing::Full> first;
        while (!(first = ring.next(false))) std::this_thread::yield();
        require(first->piece == 0 && first->bytes == 1 && ring.pages(first->slot).data()[0] == 1, "a ring gave the wrong piece first");
        const auto second = ring.next(true);
        require(second && second->piece == 1 && second->bytes == 2, "a ring gave the wrong piece second");
        // Two slots are full and neither given back, so the third piece is not read.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        {
            std::lock_guard<std::mutex> lk(m);
            require(fills == 2, "a ring refilled a slot not given back");
        }
        ring.free(0);
        const auto third = ring.next(true);
        require(third && third->piece == 2 && third->slot == first->slot && ring.pages(third->slot).data()[0] == 3, "a ring did not refill the slot given back");
    }
    {
        infer::ReadRing failing(3, 16, [](size_t k, uint8_t*) -> size_t {
            if (k == 1) throw std::runtime_error("a chunk that fails its check");
            return 16;
        }, 2, 1);
        failing.start();
        require(failing.next(true)->piece == 0, "a failing ring lost its first piece");
        failing.free(0);
        bool thrown = false;
        try { failing.next(true); } catch (const std::runtime_error& e) { thrown = std::string(e.what()) == "a chunk that fails its check"; }
        require(thrown, "a fill's throw did not come out of the step");
    }
    {
        std::atomic<bool> release{false};
        infer::ReadRing held(4, 16, [&](size_t, uint8_t*) -> size_t {
            while (!release.load()) std::this_thread::yield();
            return 16;
        }, 2, 1);
        held.start();
        held.stop();
        require(!held.idle(), "a ring was idle while a reader was in a fill");
        release.store(true);
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!held.idle()) {
            require(std::chrono::steady_clock::now() < until, "a stopped ring never went idle");
            std::this_thread::yield();
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string path = argv[1];
    try {
        gguf::GGUFModel source;
        source.tensors.push_back({"quant", {32}, quant::GGML_TYPE_Q8_0, 0});
        source.add_tensor_data(std::vector<uint8_t>(34, 17));
        const size_t bytes = 8 * 1024 * 1024 + 64;
        source.tensors.push_back({"large", {bytes / 4}, quant::GGML_TYPE_F32, 0});
        std::vector<uint8_t> values(bytes);
        for (size_t i = 0; i < values.size(); ++i) values[i] = uint8_t(i * 31);
        source.add_tensor_data(values);
        gguf::write_gguf(source, path);
        std::vector<size_t> seen;
        const format::LoadProgress progress = [&](size_t done, size_t total) {
            require(total == bytes + 34, "progress counts padding as payload");
            require(done <= total, "progress exceeds total");
            require(seen.empty() ? done == 0 : done > seen.back(), "non-monotonic progress");
            seen.push_back(done);
        };
        // A mapped model holds its file, which a rewrite below must not see held.
        {
            const auto in_file = std::filesystem::file_size(path);
            auto loaded = gguf::read_gguf(path);
            require(loaded.tensors.size() == 2 && loaded.tensors[0].name == "quant" && loaded.tensors[1].name == "large" &&
                    loaded.tensors[1].ne == source.tensors[1].ne && loaded.tensors[1].type == quant::GGML_TYPE_F32,
                    "reading changed the tensor table");
            std::ifstream in(path, std::ios::binary);
            for (size_t i = 0; i < source.tensors.size(); ++i) {
                require(!loaded.tensor_data(i), "reading the file mapped it");
                const format::FileSpan span = loaded.span(i);
                std::vector<char> bytes_read(span.bytes);
                in.seekg(std::streamoff(span.offset));
                in.read(bytes_read.data(), std::streamsize(span.bytes));
                require(span.file == path && span.bytes == source.tensor_bytes(i) && in &&
                        std::memcmp(bytes_read.data(), source.tensor_data(i), span.bytes) == 0,
                        "a tensor's span does not hold its bytes");
            }
            require(seen.empty(), "reading the file reported progress");
            // A model whose file is not mapped is refused before anything is written or reported, even over the file it was read from.
            std::string refused;
            try { gguf::write_gguf(loaded, path); } catch (const std::runtime_error& e) { refused = e.what(); }
            require(refused == "GGUF tensor is not mapped: quant" && std::filesystem::file_size(path) == in_file,
                    "writing a model that is not mapped touched the output");
            refused.clear();
            try { gguf::warm(loaded, every(loaded), progress); } catch (const std::logic_error& e) { refused = e.what(); }
            require(refused == "GGUF tensor is not mapped: quant" && seen.empty(), "warming a model that is not mapped reported progress");
            gguf::map_payload(loaded);
            gguf::warm(loaded, every(loaded), progress);
            require(seen.size() >= 4 && seen.back() == bytes + 34, "missing intermediate/final progress");
            for (size_t i = 0; i < source.tensors.size(); ++i)
                require(std::memcmp(source.tensor_data(i), loaded.tensor_data(i), source.tensor_bytes(i)) == 0,
                        "loading changed tensor bytes");
            require(loaded.offsets[1] % alignof(float) == 0, "lost F32 alignment");
            std::vector<std::pair<size_t, size_t>> calls;
            gguf::warm(loaded, {0}, [&](size_t done, size_t total) { calls.push_back({done, total}); });
            require(calls == std::vector<std::pair<size_t, size_t>>{{0, 34}, {34, 34}}, "warming one tensor reported other bytes");
        }
        bool threw = false;
        try {
            load_file(path, [](size_t done, size_t) {
                if (done) throw std::runtime_error("consumer failure");
            });
        } catch (const std::runtime_error& e) {
            threw = std::string(e.what()) == "consumer failure";
        }
        require(threw, "consumer exception lost");
        // The writer pads the last tensor; truncate its actual payload, not just the padding.
        const auto length = std::filesystem::file_size(path);
        std::filesystem::resize_file(path, length - 33);
        seen.clear();
        threw = false;
        try { load_file(path, progress); }
        catch (const std::runtime_error&) { threw = true; }
        require(threw && seen.empty(), "truncated payload emitted progress before structural rejection");
        // Reading a file holds nothing, so its size can change before it is mapped, which refuses it before any progress; a file truncated under a live mapping is not visible.
        // A streamed load's readers make the same check on the size they open, which is the one check.
        for (const auto size : {length - 33, length + 64}) {
            gguf::write_gguf(source, path);
            gguf::GGUFModel read = gguf::read_gguf(path);
            gguf::check_size(read, path, length);
            std::filesystem::resize_file(path, size);
            seen.clear();
            std::string error, checked;
            try {
                gguf::map_payload(read);
                gguf::warm(read, every(read), progress);
            } catch (const std::runtime_error& e) { error = e.what(); }
            require(error == "GGUF file changed size since its header was read: " + path && seen.empty(),
                    "a file whose size changed after it was read was mapped");
            try { gguf::check_size(read, path, format::FileReader(path).size()); } catch (const std::runtime_error& e) { checked = e.what(); }
            require(checked == error, "a reader's size check differs from the mapping's");
        }
        gguf::write_gguf(source, path);
        std::filesystem::resize_file(path, 8);
        seen.clear();
        threw = false;
        try { load_file(path, progress); }
        catch (const std::ios_base::failure&) { threw = true; }
        require(threw && seen.empty(), "truncated header reported progress");
        {
            std::ofstream os(path, std::ios::binary | std::ios::trunc);
            const uint32_t header[] = {gguf::MAGIC, gguf::VERSION};
            const uint64_t counts[] = {2, 0};
            os.write((const char*)header, sizeof(header));
            os.write((const char*)counts, sizeof(counts));
            for (int i = 0; i < 2; ++i) {
                gguf::write_string(os, i ? "empty" : "value");
                const uint32_t nd = 1, type = quant::GGML_TYPE_F32;
                const uint64_t ne = i ? 0 : 1, offset = i ? (uint64_t(1) << 63) : 0;
                os.write((const char*)&nd, 4);
                os.write((const char*)&ne, 8);
                os.write((const char*)&type, 4);
                os.write((const char*)&offset, 8);
            }
            gguf::pad_to(os, gguf::ALIGNMENT);
            const float value = 1;
            os.write((const char*)&value, 4);
        }
        size_t invalid_calls = 0;
        threw = false;
        try {
            load_file(path, [&](size_t, size_t) { ++invalid_calls; });
        } catch (const std::runtime_error&) { threw = true; }
        require(threw && invalid_calls == 0, "invalid trailing offset emitted progress");
        gguf::write_gguf(gguf::GGUFModel{}, path);
        size_t empty_calls = 0;
        load_file(path, [&](size_t done, size_t total) {
            require(done == 0 && total == 0, "empty payload progress");
            ++empty_calls;
        });
        require(empty_calls == 1, "empty model completion missing");
        loader_checks(path);
        directory_checks(path);
        awq_checks(path);
        gptq_checks(path);
        stream_checks(path);
        ring_checks();
        experts_checks(path);
        source_checks(path);
        split_checks(path);
        shard_checks(std::filesystem::u8path(path).parent_path());
        plan_checks();
        choice_checks(path);
        std::filesystem::remove(path);
        std::cout << "load progress: payload bytes, chunks, spans, truncation before and after reading, consumer failures, the loader in each mode, a Hugging Face directory, an AWQ directory, GPTQ directories, experts on the CPU, a table's source, splits, shards, the planned reads, the read choice and the stream\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        std::filesystem::remove(path);
        return 1;
    }
}
