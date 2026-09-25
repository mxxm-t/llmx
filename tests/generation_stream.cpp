#include <iostream>
#include "inference/generate.hpp"

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

gguf::GGUFModel fixture(const std::vector<uint32_t>& sequence) {
    gguf::GGUFModel m;
    for (const auto& setting : std::vector<std::pair<std::string, int>>{
            {"block_count", 1}, {"embedding_length", 16}, {"feed_forward_length", 16},
            {"attention.head_count", 1}, {"attention.head_count_kv", 1},
            {"attention.key_length", 16}, {"context_length", 32}}) {
        gguf::MetaValue value;
        value.vtype = gguf::V_UINT32;
        value.u = setting.second;
        m.kv.push_back({"qwen3." + setting.first, value});
    }
    const auto encoder = bpe::build_byte_encoder();
    const std::vector<std::string> vocab = {"EOS", "A", encoder.at(0xc3), encoder.at(0xa9), "Z"};
    gguf::MetaValue tokens;
    tokens.vtype = gguf::V_ARRAY;
    tokens.u = gguf::V_STRING;
    for (const auto& text : vocab) {
        gguf::MetaValue token;
        token.vtype = gguf::V_STRING;
        token.s = text;
        tokens.arr.push_back(token);
    }
    m.kv.push_back({"tokenizer.ggml.tokens", tokens});
    gguf::MetaValue eos;
    eos.vtype = gguf::V_UINT32;
    eos.u = 0;                  // vocab[0] is "EOS" above
    m.kv.push_back({"tokenizer.ggml.eos_token_id", eos});
    auto add = [&](const std::string& name, std::vector<uint64_t> shape, const std::vector<float>& data) {
        m.tensors.push_back({name, shape, gguf::GGML_TYPE_F32, 0});
        std::vector<uint8_t> bytes(data.size() * sizeof(float));
        std::memcpy(bytes.data(), data.data(), bytes.size());
        m.add_tensor_data(bytes);
    };
    std::vector<float> embedding(16 * vocab.size(), 0), output(embedding.size(), 0);
    for (size_t i = 0; i < vocab.size(); ++i) embedding[i * 16 + i] = 1;
    // One-hot embeddings and zero residual branches make the output projection a token transition table.
    for (size_t i = 0; i < sequence.size(); ++i) {
        const uint32_t next = i + 1 < sequence.size() ? sequence[i + 1] : 0;
        output[next * 16 + sequence[i]] = 1;
    }
    add("token_embd.weight", {16, vocab.size()}, embedding);
    add("output.weight", {16, vocab.size()}, output);
    add("output_norm.weight", {16}, std::vector<float>(16, 1));
    for (const char* norm : {"attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"})
        add(std::string("blk.0.") + norm + ".weight", {16}, std::vector<float>(16, 1));
    for (const char* matrix : {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"})
        add(std::string("blk.0.") + matrix + ".weight", {16, 16}, std::vector<float>(256, 0));
    return m;
}

void check(const std::vector<uint32_t>& sequence, const std::string& stop, int limit, const std::string& expected, size_t generated,
           size_t fed) {
    auto weights = fixture(sequence);
    bpe::Tokenizer tok(weights);
    infer::Model model(weights);
    model.set_threads(1);
    infer::GenParams gp;
    gp.temp = 0;
    gp.max_tokens = limit;
    gp.stop = stop;
    infer::RNG rng;
    std::vector<float> logits(tok.vocab.size(), 0);
    logits[sequence.empty() ? 0 : sequence.front()] = 1;
    std::string text;
    size_t calls = 0;
    auto ids = infer::generate(model, tok, gp, rng, logits, [&](const std::string& piece) {
        require(size_t(model.n_tokens()) == calls, "text was not delivered before the next model step");
        ++calls;
        text += piece;
    });
    require(text == expected, "displayed bytes changed");
    require(ids.size() == generated && size_t(model.n_tokens()) == fed, "stop/EOS/token-limit accounting changed");
    require(std::equal(ids.begin(), ids.end(), sequence.begin()), "generated IDs changed");
}

int main() {
    try {
        check({1, 2, 3, 4}, "", 8, "A\xc3\xa9Z", 4, 4);
        check({1, 2, 3, 4}, "\xc3\xa9", 8, "A\xc3\xa9", 3, 2);
        check({1, 2, 3, 4}, "", 2, "A\xc3", 2, 2);
        check({}, "", 8, "", 0, 0);
        check({1}, "", 0, "", 0, 0);
        auto weights = fixture({1});
        bpe::Tokenizer tok(weights);
        infer::Model model(weights);
        model.set_threads(1);
        infer::GenParams gp;
        gp.temp = 0;
        infer::RNG rng;
        std::vector<float> logits(tok.vocab.size(), 0);
        logits[1] = 1;
        bool threw = false;
        try {
            infer::generate(model, tok, gp, rng, logits, [](const std::string&) {
                throw std::runtime_error("consumer failure");
            });
        } catch (const std::runtime_error& e) { threw = std::string(e.what()) == "consumer failure"; }
        require(threw && model.n_tokens() == 0, "consumer failure continued generation");
        auto ids = infer::generate(model, tok, gp, rng, logits);
        require(ids == std::vector<uint32_t>{1}, "generation without output callback failed");
        std::cout << "generation stream: early delivery, UTF-8, stop/EOS and consumer failures pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
