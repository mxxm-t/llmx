#include <cstring>
#include <iostream>
#include "model/arch/registry.hpp"

static size_t cases = 0;

static void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

// A two-layer Qwen3 directory's config, `edit` replacing one "key": value pair or adding one before the closing brace.
static std::string config(const std::string& from = "", const std::string& to = "") {
    std::string c = R"({"architectures":["Qwen3ForCausalLM"],"attention_bias":false,"attention_dropout":0.0,)"
        R"("bos_token_id":1,"eos_token_id":2,"head_dim":4,"hidden_act":"silu","hidden_size":8,)"
        R"("initializer_range":0.02,"intermediate_size":12,"max_position_embeddings":64,"max_window_layers":2,)"
        R"("model_type":"qwen3","num_attention_heads":2,"num_hidden_layers":2,"num_key_value_heads":1,)"
        R"("rms_norm_eps":1e-06,"rope_scaling":null,"rope_theta":1000000,"sliding_window":null,)"
        R"("tie_word_embeddings":true,"torch_dtype":"bfloat16","transformers_version":"4.51.0",)"
        R"("use_cache":true,"use_sliding_window":false,"vocab_size":32})";
    if (from.empty() && !to.empty()) c.insert(c.size() - 1, "," + to);
    if (!from.empty()) {
        const size_t at = c.find(from);
        require(at != std::string::npos, "test config has no " + from);
        c.replace(at, from.size(), to);
    }
    return c;
}

// Tensors under HF names in two files: the head alone in the first, the rest in the second.
static gguf::GGUFModel tensors(bool head = true, const std::string& extra = "") {
    gguf::GGUFModel m;
    auto add = [&](const std::string& name, std::vector<uint64_t> ne) {
        m.offsets.push_back(m.tensors.size() * 4096);
        m.tensors.push_back({name, ne, quant::GGML_TYPE_BF16, 0});
    };
    if (head) {
        add("lm_head.weight", {8, 32});
        m.segments.push_back({"a.safetensors", nullptr, 8, 0, 4096, 0});
    }
    m.segments.push_back({"b.safetensors", nullptr, 8, m.tensors.size() * 4096, 1 << 20, m.tensors.size()});
    add("model.embed_tokens.weight", {8, 32});
    add("model.norm.weight", {8});
    for (int l = 0; l < 2; ++l) {
        const std::string p = "model.layers." + std::to_string(l) + ".";
        for (const char* n : {"input_layernorm", "post_attention_layernorm"}) add(p + n + ".weight", {8});
        for (const char* n : {"self_attn.q_norm", "self_attn.k_norm"}) add(p + n + ".weight", {4});
        add(p + "self_attn.q_proj.weight", {8, 8});
        add(p + "self_attn.k_proj.weight", {8, 4});
        add(p + "self_attn.v_proj.weight", {8, 4});
        add(p + "self_attn.o_proj.weight", {8, 8});
        add(p + "mlp.gate_proj.weight", {8, 12});
        add(p + "mlp.up_proj.weight", {8, 12});
        add(p + "mlp.down_proj.weight", {12, 8});
    }
    if (!extra.empty()) add(extra, {8});
    return m;
}

static void read_config(const std::string& c, gguf::GGUFModel& m) {
    const std::string type = "qwen3";
    const infer::ArchEntry& entry = infer::hf_architecture(type);
    infer::hf::apply(entry.hf_table(), entry.name, jmini::parse(c), m);
}

static void refused(const std::string& c, gguf::GGUFModel m, const std::string& expected) {
    ++cases;
    try {
        read_config(c, m);
    } catch (const std::exception& e) {
        if (std::string(e.what()).find(expected) == std::string::npos)
            throw std::runtime_error("refused with \"" + std::string(e.what()) + "\", expected \"" + expected + "\"");
        return;
    }
    throw std::runtime_error("accepted what should be refused for \"" + expected + "\"");
}

static uint64_t integer(const gguf::GGUFModel& m, const std::string& key) {
    const gguf::MetaValue* v = m.find(key);
    require(v && v->vtype == gguf::V_UINT32, key + " is not a UINT32");
    return v->u;
}

static float real(const gguf::GGUFModel& m, const std::string& key) {
    const gguf::MetaValue* v = m.find(key);
    require(v && v->vtype == gguf::V_FLOAT32, key + " is not a FLOAT32");
    float f;
    std::memcpy(&f, &v->fb, sizeof(f));
    return f;
}

int main() {
    try {
        {
            ++cases;
            gguf::GGUFModel m = tensors();
            read_config(config(), m);
            const gguf::MetaValue* arch = m.find("general.architecture");
            require(arch && arch->vtype == gguf::V_STRING && arch->s == "qwen3", "general.architecture");
            require(integer(m, "qwen3.block_count") == 2 && integer(m, "qwen3.context_length") == 64 &&
                    integer(m, "qwen3.embedding_length") == 8 && integer(m, "qwen3.feed_forward_length") == 12 &&
                    integer(m, "qwen3.attention.head_count") == 2 && integer(m, "qwen3.attention.head_count_kv") == 1 &&
                    integer(m, "qwen3.attention.key_length") == 4 && integer(m, "qwen3.attention.value_length") == 4, "integer keys");
            require(real(m, "qwen3.rope.freq_base") == 1e6f && real(m, "qwen3.attention.layer_norm_rms_epsilon") == 1e-6f, "real keys");
            require(m.kv.size() == 11, "no other keys");
            // The tied head is dropped with the file that held it alone.
            require(m.tensors.size() == 24 && m.tensors[0].name == "token_embd.weight" && m.offsets[0] == 4096, "head dropped");
            require(m.segments.size() == 1 && m.segments[0].first == 0 && m.segments[0].path == "b.safetensors", "head's file dropped");
            require(m.tensors[2].name == "blk.0.attn_norm.weight" && m.tensors[23].name == "blk.1.ffn_down.weight", "names");
            const infer::qwen3::Config c = infer::qwen3::read_config(m, "qwen3.", false);
            require(c.n_layer == 2 && c.n_embd == 8 && c.n_head_kv == 1 && c.head_dim == 4 && c.context_length == 64, "read_config");
        }
        {
            ++cases;
            gguf::GGUFModel m = tensors();
            read_config(config("\"tie_word_embeddings\":true", "\"tie_word_embeddings\":false"), m);
            require(m.tensors.size() == 25 && m.tensors[0].name == "output.weight" && m.segments.size() == 2, "untied head kept");
        }
        {
            ++cases;
            gguf::GGUFModel m = tensors(false);
            read_config(config("\"torch_dtype\":\"bfloat16\"", "\"dtype\":\"float16\""), m);
            require(m.tensors.size() == 24, "tied, no head stored");
        }
        ++cases;
        try {
            const std::string made_up = "made_up_model";
            infer::hf_architecture(made_up);
            throw std::logic_error("an unknown model_type accepted");
        } catch (const std::runtime_error& e) {
            require(std::string(e.what()).find("model_type made_up_model") != std::string::npos, "refusal names the type");
        }

        refused(config("", "\"pad_token_id\":0"), tensors(), "key pad_token_id is not one");
        refused(config("[\"Qwen3ForCausalLM\"]", "[\"Qwen3ForCausalLM\",\"X\"]"), tensors(), "architectures must be exactly");
        refused(config("[\"Qwen3ForCausalLM\"]", "[\"Qwen3MoeForCausalLM\"]"), tensors(), "architectures must be exactly");
        refused(config("\"model_type\":\"qwen3\"", "\"model_type\":\"qwen2\""), tensors(), "model_type must be qwen3");
        refused(config("\"attention_bias\":false", "\"attention_bias\":true"), tensors(), "attention_bias = true");
        refused(config("\"hidden_act\":\"silu\"", "\"hidden_act\":\"gelu\""), tensors(), "hidden_act");
        refused(config("\"rope_scaling\":null", "\"rope_scaling\":{\"type\":\"yarn\"}"), tensors(), "rope_scaling");
        refused(config("\"use_sliding_window\":false", "\"use_sliding_window\":true"), tensors(), "use_sliding_window");
        refused(config("\"torch_dtype\":\"bfloat16\"", "\"torch_dtype\":\"float8_e4m3fn\""), tensors(), "torch_dtype");
        refused(config("\"tie_word_embeddings\":true", "\"tie_word_embeddings\":1"), tensors(), "tie_word_embeddings");
        refused(config("\"num_hidden_layers\":2", "\"num_hidden_layers\":2.5"), tensors(), "positive 32-bit integer");
        refused(config("\"num_hidden_layers\":2", "\"num_hidden_layers\":0"), tensors(), "positive 32-bit integer");
        refused(config("\"num_hidden_layers\":2", "\"num_hidden_layers\":1"), tensors(), "past the layer count");
        refused(config("\"rms_norm_eps\":1e-06", "\"rms_norm_eps\":-1"), tensors(), "finite positive");
        refused(config("\"rms_norm_eps\":1e-06", "\"rms_norm_eps\":\"x\""), tensors(), "finite positive");
        refused(config("\"num_hidden_layers\":2,", ""), tensors(), "no layer count");
        refused(config("", "\"use_cache\":false"), tensors(), "repeats key use_cache");
        refused(config("\"vocab_size\":32", "\"vocab_size\":33"), tensors(), "vocab_size");
        refused(config("\"tie_word_embeddings\":true", "\"tie_word_embeddings\":false"), tensors(false), "holds no lm_head");
        refused(config(), tensors(true, "model.layers.0.mlp.gate.weight"), "tensor model.layers.0.mlp.gate.weight is not one");
        refused(config(), tensors(true, "model.layers.01.input_layernorm.weight"), "is not one");
        refused("[]", tensors(), "not an object");
    } catch (const std::exception& e) {
        std::cerr << "hf-config: FAIL after " << cases << " cases: " << e.what() << "\n";
        return 1;
    }
    std::cout << "hf-config: " << cases << " cases passed\n";
    return 0;
}
