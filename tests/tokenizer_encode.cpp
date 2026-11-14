// bpe::Tokenizer::encode against the encode it replaced, written out here: every byte checked against every special token, and BPE run on every word, on a small vocabulary with merges and special tokens (src/tokenizer/tokenizer.hpp).
// On a vocabulary of thousands of generated merges it also holds the merge loop to a plain lowest-rank-leftmost loop on random words, and bounds the time two one-word texts of 2 MiB take.
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
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

// A vocabulary with thousands of merges the way training makes them, each joining two tokens that exist, over the ASCII letters and the continuation bytes 0x80 to 0xBF, so a run of either is one word of `pretokenize`.
// No valid UTF-8 sequence is continuation bytes alone, so `encode`'s NFC pass leaves such a word as it is; the tokens' raw byte texts are returned beside the vocabulary, so a word can be built from tokens whose merges then apply.
bpe::Tokenizer rich_tokenizer(std::vector<std::string>& raw_tokens) {
    const auto bytes = bpe::build_byte_encoder();
    const auto encode = [&](const std::string& raw) {
        std::string s;
        for (unsigned char b : raw) s += bytes.at(b);
        return s;
    };
    gguf::MetaValue tokens = array(gguf::V_STRING), types = array(gguf::V_UINT32), merges = array(gguf::V_STRING);
    const auto add = [&](const std::string& s) {
        tokens.arr.push_back(text(s));
        gguf::MetaValue t;
        t.vtype = gguf::V_UINT32;
        t.u = 1;
        types.arr.push_back(t);
    };
    for (int b = 0; b < 256; ++b) add(bytes.at((uint8_t)b));
    std::unordered_set<std::string> have, joined;
    for (int c = 0; c < 256; ++c)
        if (bpe::Tokenizer::is_letter((unsigned char)c) && (unsigned char)c < 0xC0) {
            raw_tokens.push_back(std::string(1, (char)c));
            have.insert(raw_tokens.back());
        }
    std::mt19937 rng(7);
    while (merges.arr.size() < 4000) {
        const std::string& a = raw_tokens[rng() % raw_tokens.size()];
        const std::string& b = raw_tokens[rng() % raw_tokens.size()];
        if (a.size() + b.size() > 12) continue;
        const std::string ab = a + b;
        // The tokenizer ranks a merge by its joined text, so a second split of the same text would overwrite the first.
        if (!joined.insert(ab).second) continue;
        merges.arr.push_back(text(encode(a) + " " + encode(b)));
        if (have.insert(ab).second) {
            add(encode(ab));
            raw_tokens.push_back(ab);
        }
    }
    gguf::GGUFModel m;
    m.kv.push_back({"tokenizer.ggml.tokens", tokens});
    m.kv.push_back({"tokenizer.ggml.token_type", types});
    m.kv.push_back({"tokenizer.ggml.merges", merges});
    return bpe::Tokenizer(m);
}

// The merge loop as a plain scan: the adjacent pair of lowest rank, the leftmost on a tie, merged until no pair has a rank.
std::vector<uint32_t> reference_bpe(const bpe::Tokenizer& tok, const std::string& raw) {
    const std::string word = tok.byte_encode(raw);
    std::vector<std::string> sym;
    for (size_t i = 0; i < word.size();) {
        const size_t l = utf8::lead_length((unsigned char)word[i]);
        sym.push_back(word.substr(i, l));
        i += l;
    }
    while (sym.size() > 1) {
        uint32_t best = UINT32_MAX;
        size_t at = sym.size();
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            const auto it = tok.merge_rank.find(sym[i] + sym[i + 1]);
            if (it != tok.merge_rank.end() && it->second < best) { best = it->second; at = i; }
        }
        if (at == sym.size()) break;
        sym[at] += sym[at + 1];
        sym.erase(sym.begin() + at + 1);
    }
    std::vector<uint32_t> ids;
    for (const auto& s : sym) ids.push_back(tok.token_to_id.at(s));
    return ids;
}

// A one-word text of `size` bytes made of the given tokens, so merges apply throughout it.
std::string long_word(const std::vector<std::string>& parts, size_t size, std::mt19937& rng) {
    std::string w;
    while (w.size() < size) w += parts[rng() % parts.size()];
    w.resize(size);
    return w;
}

}   // namespace

int main() {
    try {
        const bpe::Tokenizer tok = tokenizer();
        // A tokens array of another element type, and a token_type array shorter than the tokens, are refused at load.
        {
            const auto refused = [&](const gguf::MetaValue& tokens, const gguf::MetaValue& types, const std::string& what) {
                gguf::GGUFModel m;
                m.kv.push_back({"tokenizer.ggml.tokens", tokens});
                m.kv.push_back({"tokenizer.ggml.token_type", types});
                try { bpe::Tokenizer t(m); } catch (const std::runtime_error& e) {
                    require(std::string(e.what()).find(what) != std::string::npos, "refused as \"" + std::string(e.what()) + "\", not for \"" + what + "\"");
                    return;
                }
                throw std::runtime_error("a tokenizer loaded that should be refused for \"" + what + "\"");
            };
            gguf::MetaValue numbers = array(gguf::V_UINT32), strings = array(gguf::V_STRING), one_type = array(gguf::V_UINT32);
            gguf::MetaValue n;
            n.vtype = gguf::V_UINT32;
            n.u = 1;
            numbers.arr = {n, n};
            strings.arr = {text("a"), text("b")};
            one_type.arr = {n};
            refused(numbers, one_type, "tokenizer.ggml.tokens is an array of type 4, not of strings");
            refused(strings, one_type, "tokenizer.ggml.token_type has 1 entries for 2 tokens");
        }
        {
            std::vector<std::string> raw;
            const bpe::Tokenizer rich = rich_tokenizer(raw);
            std::mt19937 rng(23);
            // Words of a few tokens, of random letters and of random continuation bytes, each one word of the pretokenizer.
            for (int round = 0; round < 600; ++round) {
                std::string w;
                if (round < 300) { const size_t n = 1 + rng() % 6; for (size_t k = 0; k < n; ++k) w += raw[rng() % raw.size()]; }
                else if (round < 450) { const size_t n = 1 + rng() % 30; for (size_t k = 0; k < n; ++k) w += (char)("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[rng() % 52]); }
                else { const size_t n = 1 + rng() % 30; for (size_t k = 0; k < n; ++k) w += (char)(0x80 + rng() % 64); }
                require(rich.encode(w) == reference_bpe(rich, w), "random word " + std::to_string(round) + " merges otherwise than the plain loop");
            }
            // One word of 2 MiB, of letters and then of continuation bytes, must encode in bounded time, since a client can send one.
            std::vector<std::string> letters, high;
            for (const auto& t : raw) ((unsigned char)t[0] < 0x80 ? letters : high).push_back(t);
            for (const auto* parts : {&letters, &high}) {
                const std::string w = long_word(*parts, size_t(2) << 20, rng);
                const auto t0 = std::chrono::steady_clock::now();
                const std::vector<uint32_t> ids = rich.encode(w);
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                require(rich.decode(ids) == w, "the 2 MiB word does not decode back");
                require(seconds < 15.0, "the 2 MiB word took " + std::to_string(seconds) + " s to encode");
            }
        }
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
