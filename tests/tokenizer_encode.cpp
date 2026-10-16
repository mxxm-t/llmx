// bpe::Tokenizer::encode against the encode it replaced, written out here: every byte checked against every special token, and BPE run on every word, on a small vocabulary with merges and special tokens (src/tokenizer/tokenizer.hpp).
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "tokenizer/tokenizer.hpp"

namespace {

size_t checks = 0;
void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

gguf::MetaValue text(const std::string& s) {
    gguf::MetaValue v;
    v.vtype = gguf::V_STRING;
    v.s = s;
    return v;
}

gguf::MetaValue array(uint32_t type) {
    gguf::MetaValue v;
    v.vtype = gguf::V_ARRAY;
    v.u = type;
    return v;
}

// Every byte as a token, merged words, and special tokens, two of which share a prefix and one of which is a prefix of another.
bpe::Tokenizer tokenizer() {
    const auto bytes = bpe::build_byte_encoder();
    gguf::MetaValue tokens = array(gguf::V_STRING), types = array(gguf::V_UINT32), merges = array(gguf::V_STRING);
    const auto add = [&](const std::string& s, uint32_t type) {
        tokens.arr.push_back(text(s));
        gguf::MetaValue t;
        t.vtype = gguf::V_UINT32;
        t.u = type;
        types.arr.push_back(t);
    };
    for (int b = 0; b < 256; ++b) add(bytes.at((uint8_t)b), 1);
    const std::string sp = bytes.at((uint8_t)' ');
    for (const std::pair<std::string, std::string>& m : std::vector<std::pair<std::string, std::string>>{
             {"h", "e"}, {"l", "l"}, {"he", "ll"}, {"hell", "o"}, {sp, "w"}, {"o", "r"}, {sp + "w", "or"}, {sp + "wor", "l"}, {sp + "worl", "d"}, {sp, "h"}, {sp + "h", "ello"}}) {
        merges.arr.push_back(text(m.first + " " + m.second));
        if (m.second == "ello") continue;
        add(m.first + m.second, 1);
    }
    add("ello", 1);
    merges.arr.push_back(text("e llo"));
    add("llo", 1);
    merges.arr.push_back(text("l lo"));
    add("lo", 1);
    merges.arr.push_back(text("l o"));
    add(sp + "hello", 1);
    for (const char* special : {"<|im_start|>", "<|im_end|>", "<|im", "<think>", "</think>", "<tool_call>"}) add(special, 3);
    gguf::GGUFModel m;
    m.kv.push_back({"tokenizer.ggml.tokens", tokens});
    m.kv.push_back({"tokenizer.ggml.token_type", types});
    m.kv.push_back({"tokenizer.ggml.merges", merges});
    return bpe::Tokenizer(m);
}

// The encode the tokenizer had: at every byte the longest special token, and BPE on every pretokenized word between them.
std::vector<uint32_t> walked(const bpe::Tokenizer& tok, const std::string& text) {
    std::vector<uint32_t> ids;
    std::string plain;
    const auto flush = [&]() {
        for (const auto& word : bpe::Tokenizer::pretokenize(plain))
            for (const auto& sym : tok.bpe(tok.byte_encode(word))) ids.push_back(tok.token_to_id.at(sym));
        plain.clear();
    };
    for (size_t i = 0; i < text.size();) {
        const bpe::SpecialToken* match = nullptr;
        size_t best = 0;
        for (const auto& sp : tok.specials)
            if (sp.text.size() > best && text.compare(i, sp.text.size(), sp.text) == 0) { match = &sp; best = sp.text.size(); }
        if (match) { flush(); ids.push_back(match->id); i += best; }
        else plain += text[i++];
    }
    flush();
    return ids;
}

}   // namespace

int main() {
    try {
        const bpe::Tokenizer tok = tokenizer();
        const std::vector<std::string> pieces = {
            "<|im_start|>", "<|im_end|>", "<|im", "<think>", "</think>", "<tool_call>",   // specials, and one a prefix of two others
            "<|im_end", "<|i", "<", "a < b", "<th", "</", "<<|im_end|>",                    // a '<' that begins no special, and specials cut short
            "hello", " hello", "hello,", "Hello", "hellohello", " world", "world", "\nhello\n", "  hello", "hello's",
            "caf\xc3\xa9", "\xe6\x9d\xb1\xe4\xba\xac", "123", " 4.5", "\n\n", "\t", "x", " ",
        };
        for (const auto& p : pieces) require(tok.encode(p) == walked(tok, p), "the piece \"" + p + "\" alone encodes otherwise");
        // Specials at the start, at the end and side by side, and a special cut short at the text's end.
        for (const std::string& t : {std::string("<|im_start|><|im_end|>hello world<think></think>"), std::string("<|im_end|>"), std::string("hello<|im_en"),
                                    std::string("<|im<|im_start|>x<|im_end|><|im"), std::string("")})
            require(tok.encode(t) == walked(tok, t), "\"" + t + "\" encodes otherwise");
        require(tok.encode("<|im_start|>").size() == 1 && tok.encode("<|im").size() == 1 && tok.encode("<|im_en").size() > 1,
                "a whole special is not one id, or a special cut short is one");
        // Long texts of the pieces in random order, so a word stands beside every kind of neighbour and repeats.
        std::mt19937 rng(11);
        for (int round = 0; round < 50; ++round) {
            std::string t;
            const size_t n = 1 + rng() % 4000;
            for (size_t k = 0; k < n; ++k) t += pieces[rng() % pieces.size()];
            require(tok.encode(t) == walked(tok, t), "a text of " + std::to_string(n) + " random pieces, round " + std::to_string(round) + ", encodes otherwise");
        }
        std::cout << "tokenizer-encode: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "tokenizer-encode: FAIL: " << e.what() << "\n";
        return 1;
    }
}
