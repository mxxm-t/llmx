#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

#include "format/gguf.hpp"

// GPT-2 style byte-level BPE tokenizer, implemented from scratch.
// Reads tokenizer metadata from a GGUF model:
//   tokenizer.ggml.model        = "gpt2"
//   tokenizer.ggml.tokens       = array<string>   (token id -> byte-mapped token)
//   tokenizer.ggml.token_type   = array<u32>      (per-token type)
//   tokenizer.ggml.merges       = array<string>   ("s1 s2", rank = index)
//   tokenizer.ggml.bos/eos/pad_token_id, add_bos_token

namespace bpe {

// Encode a unicode code point to UTF-8.
inline std::string utf8_encode(uint32_t cp) {
    if (cp < 0x80) return std::string(1, (char)cp);
    if (cp < 0x800) return std::string({ (char)(0xC0 | (cp >> 6)), (char)(0x80 | (cp & 0x3F)) });
    if (cp < 0x10000) return std::string({ (char)(0xE0 | (cp >> 12)), (char)(0x80 | ((cp >> 6) & 0x3F)), (char)(0x80 | (cp & 0x3F)) });
    return std::string({ (char)(0xF0 | (cp >> 18)), (char)(0x80 | ((cp >> 12) & 0x3F)), (char)(0x80 | ((cp >> 6) & 0x3F)), (char)(0x80 | (cp & 0x3F)) });
}

inline size_t utf8_char_len(unsigned char c) {
    if (c >= 0xF0) return 4;
    if (c >= 0xE0) return 3;
    if (c >= 0xC0) return 2;
    return 1;
}

// GPT-2 bytes_to_unicode(): map each byte to a printable unicode code point.
inline std::unordered_map<uint8_t, std::string> build_byte_encoder() {
    std::unordered_map<uint8_t, std::string> m;
    for (int b = 33; b <= 126; b++) m[(uint8_t)b] = utf8_encode((uint32_t)b);
    for (int b = 161; b <= 255; b++) m[(uint8_t)b] = utf8_encode((uint32_t)b);
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool in = (b >= 33 && b <= 126) || (b >= 161 && b <= 255);
        if (!in) { m[(uint8_t)b] = utf8_encode((uint32_t)(256 + n)); n++; }
    }
    return m;
}

struct SpecialToken {
    std::string text;
    uint32_t id;
};

class Tokenizer {
public:
    std::vector<std::string> vocab;
    std::unordered_map<std::string, uint32_t> token_to_id;
    std::unordered_map<std::string, uint32_t> merge_rank;
    std::unordered_map<uint8_t, std::string> byte_to_char;
    std::unordered_map<std::string, uint8_t> char_to_byte;
    std::vector<SpecialToken> specials; // control / user-defined tokens
    int32_t bos_id = -1;
    int32_t eos_id = -1;
    int32_t pad_id = -1;
    bool add_bos = false;

    Tokenizer(const gguf::GGUFModel& m) {
        byte_to_char = build_byte_encoder();
        for (const auto& kv : byte_to_char) char_to_byte[kv.second] = kv.first;

        // tokens
        const gguf::MetaValue* toks = find_kv(m, "tokenizer.ggml.tokens");
        if (!toks || toks->vtype != gguf::V_ARRAY)
            throw std::runtime_error("tokenizer: missing tokenizer.ggml.tokens");
        vocab.reserve(toks->arr.size());
        for (const auto& e : toks->arr) {
            vocab.push_back(e.s);
            uint32_t id = (uint32_t)vocab.size() - 1;
            token_to_id[e.s] = id;
        }

        // token types (for specials)
        std::vector<uint32_t> types(vocab.size(), 1);
        const gguf::MetaValue* tt = find_kv(m, "tokenizer.ggml.token_type");
        if (tt && tt->vtype == gguf::V_ARRAY) {
            for (size_t i = 0; i < tt->arr.size() && i < types.size(); i++) {
                const auto& e = tt->arr[i];
                if (e.vtype == gguf::V_INT8 || e.vtype == gguf::V_INT16 ||
                    e.vtype == gguf::V_INT32 || e.vtype == gguf::V_INT64)
                    types[i] = (uint32_t)e.i;
                else
                    types[i] = (uint32_t)e.u;
            }
        }

        // merges
        const gguf::MetaValue* mg = find_kv(m, "tokenizer.ggml.merges");
        if (mg && mg->vtype == gguf::V_ARRAY) {
            for (size_t i = 0; i < mg->arr.size(); i++) {
                const std::string& s = mg->arr[i].s;
                size_t sp = s.find(' ');
                std::string merged = (sp == std::string::npos) ? s : (s.substr(0, sp) + s.substr(sp + 1));
                merge_rank[merged] = (uint32_t)i;
            }
        }

        // special tokens: control(3) and user-defined(4)
        for (size_t i = 0; i < vocab.size(); i++) {
            if (types[i] == 3 || types[i] == 4)
                specials.push_back({ vocab[i], (uint32_t)i });
        }

        // ids
        if (const gguf::MetaValue* v = find_kv(m, "tokenizer.ggml.bos_token_id"))
            if (v->vtype == gguf::V_UINT32) bos_id = (int32_t)v->u;
        if (const gguf::MetaValue* v = find_kv(m, "tokenizer.ggml.eos_token_id"))
            if (v->vtype == gguf::V_UINT32) eos_id = (int32_t)v->u;
        if (const gguf::MetaValue* v = find_kv(m, "tokenizer.ggml.padding_token_id"))
            if (v->vtype == gguf::V_UINT32) pad_id = (int32_t)v->u;
        if (const gguf::MetaValue* v = find_kv(m, "tokenizer.ggml.add_bos_token"))
            if (v->vtype == gguf::V_BOOL) add_bos = v->b;
    }

    int32_t token_id(const std::string& s) const {
        auto it = token_to_id.find(s);
        return (it == token_to_id.end()) ? -1 : (int32_t)it->second;
    }

    // --- pre-tokenization (GPT-2 regex on the raw UTF-8 input) ---
    static bool is_space(unsigned char c) { return c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\f'||c=='\v'; }
    static bool is_ascii_letter(unsigned char c) { return (c>='a'&&c<='z')||(c>='A'&&c<='Z'); }
    static bool is_digit(unsigned char c) { return c>='0'&&c<='9'; }
    static bool is_letter(unsigned char c) { return is_ascii_letter(c) || c >= 0x80; }

    static std::vector<std::string> pretokenize(const std::string& text) {
        std::vector<std::string> out;
        static const char* contractions[] = {"'s","'t","'re","'ve","'m","'ll","'d"};
        size_t i = 0;
        const size_t n = text.size();
        auto emit = [&](size_t b, size_t e) { out.push_back(text.substr(b, e - b)); };
        while (i < n) {
            unsigned char c = (unsigned char)text[i];
            // 1. contractions
            if (c == '\'') {
                bool matched = false;
                for (const char* p : contractions) {
                    size_t len = std::strlen(p);
                    if (text.compare(i, len, p) == 0) { emit(i, i + len); i += len; matched = true; break; }
                }
                if (matched) continue;
            }
            // 2. optional space + letters
            { size_t j = i; if (j < n && is_space((unsigned char)text[j])) j++; size_t k = j;
              while (k < n && is_letter((unsigned char)text[k])) k++; if (k > j) { emit(i, k); i = k; continue; } }
            // 3. optional space + digits
            { size_t j = i; if (j < n && is_space((unsigned char)text[j])) j++; size_t k = j;
              while (k < n && is_digit((unsigned char)text[k])) k++; if (k > j) { emit(i, k); i = k; continue; } }
            // 4. optional space + non-space non-alnum
            { size_t j = i; if (j < n && is_space((unsigned char)text[j])) j++; size_t k = j;
              while (k < n && !is_space((unsigned char)text[k]) && !is_letter((unsigned char)text[k]) && !is_digit((unsigned char)text[k])) k++;
              if (k > j) { emit(i, k); i = k; continue; } }
            // 5. GPT-2 `\s+(?!\S)`: a whitespace run, minus its last space when
            //    the run is followed by a non-space char -- that space belongs to
            //    the next token, which rules 2-4 pick up as their optional leading
            //    space. (The old guard could never fire: after the loop text[k] is
            //    never a space, so runs of 2+ spaces were emitted whole and
            //    diverged from the reference tokenizer.)
            { size_t k = i; while (k < n && is_space((unsigned char)text[k])) k++;
              if (k > i) { size_t e = (k < n) ? k - 1 : k;
                           if (e > i) { emit(i, e); i = e; continue; } } }
            // 6. whitespace run (safety net)
            { size_t k = i; while (k < n && is_space((unsigned char)text[k])) k++;
              if (k > i) { emit(i, k); i = k; continue; } }
            // safety
            emit(i, i + utf8_char_len((unsigned char)text[i])); i += utf8_char_len((unsigned char)text[i]);
        }
        return out;
    }

    std::string byte_encode(const std::string& raw) const {
        std::string s;
        for (unsigned char b : raw) s += byte_to_char.at(b);
        return s;
    }

    // BPE merge loop over a byte-mapped word.
    std::vector<std::string> bpe(const std::string& word) const {
        std::vector<std::string> sym;
        for (size_t i = 0; i < word.size(); ) {
            size_t l = utf8_char_len((unsigned char)word[i]);
            sym.push_back(word.substr(i, l));
            i += l;
        }
        while (sym.size() > 1) {
            uint32_t best = UINT32_MAX; int best_pos = -1;
            for (size_t i = 0; i + 1 < sym.size(); i++) {
                std::string pair = sym[i] + sym[i + 1];
                auto it = merge_rank.find(pair);
                if (it != merge_rank.end() && it->second < best) { best = it->second; best_pos = (int)i; }
            }
            if (best_pos < 0) break;
            sym[best_pos] += sym[best_pos + 1];
            sym.erase(sym.begin() + best_pos + 1);
        }
        return sym;
    }

    // Encode text, handling special tokens as single ids.
    std::vector<uint32_t> encode(const std::string& text) const {
        std::vector<uint32_t> ids;
        // greedy longest-match split on special tokens
        size_t i = 0; std::string plain;
        auto flush = [&]() {
            if (plain.empty()) return;
            for (const auto& tok : pretokenize(plain)) {
                std::string w = byte_encode(tok);
                for (const auto& sym : bpe(w)) {
                    auto it = token_to_id.find(sym);
                    if (it == token_to_id.end())
                        throw std::runtime_error("tokenizer: symbol not in vocab: " + sym);
                    ids.push_back(it->second);
                }
            }
            plain.clear();
        };
        while (i < text.size()) {
            const SpecialToken* match = nullptr; size_t best_len = 0;
            for (const auto& sp : specials) {
                if (sp.text.size() > best_len && text.compare(i, sp.text.size(), sp.text) == 0) {
                    match = &sp; best_len = sp.text.size();
                }
            }
            if (match) { flush(); ids.push_back(match->id); i += best_len; }
            else { plain += text[i++]; }
        }
        flush();
        return ids;
    }

    std::string decode(const std::vector<uint32_t>& ids) const {
        std::string out;
        for (uint32_t id : ids) {
            const std::string& tok = vocab[id];
            for (size_t i = 0; i < tok.size(); ) {
                size_t l = utf8_char_len((unsigned char)tok[i]);
                std::string ch = tok.substr(i, l);
                auto it = char_to_byte.find(ch);
                if (it != char_to_byte.end()) out += (char)it->second;
                else out += ch;
                i += l;
            }
        }
        return out;
    }

private:
    static const gguf::MetaValue* find_kv(const gguf::GGUFModel& m, const std::string& key) {
        for (const auto& kv : m.kv) if (kv.first == key) return &kv.second;
        return nullptr;
    }
};

} // namespace bpe
