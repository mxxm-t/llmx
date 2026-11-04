#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <initializer_list>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <utility>

#include "core/unicode_class.hpp"
#include "core/utf8.hpp"
#include "format/gguf.hpp"

// GPT-2 style byte-level BPE tokenizer, implemented from scratch.
// Reads tokenizer metadata from a GGUF model:
//   tokenizer.ggml.model        = "gpt2"           (refused when present with another value)
//   tokenizer.ggml.pre          = "qwen2"/"qwen35"/"joyai-llm" (refused when present with another value)
//   tokenizer.ggml.tokens       = array<string>    (token id -> byte-mapped token)
//   tokenizer.ggml.token_type   = array<u32>       (per-token type)
//   tokenizer.ggml.merges       = array<string>    ("s1 s2", rank = index)
//   tokenizer.ggml.bos/eos_token_id

namespace bpe {

// GPT-2 bytes_to_unicode(): map each byte to a printable unicode code point.
inline std::unordered_map<uint8_t, std::string> build_byte_encoder() {
    std::unordered_map<uint8_t, std::string> m;
    for (int b = 33; b <= 126; b++) m[(uint8_t)b] = utf8::encode((uint32_t)b);
    for (int b = 161; b <= 255; b++)
        if (b != 173) m[(uint8_t)b] = utf8::encode((uint32_t)b);
    int n = 0;
    for (int b = 0; b < 256; b++) {
        // GPT-2 excludes the soft-hyphen byte, mapping it to U+0143 instead.
        bool in = (b >= 33 && b <= 126) || (b >= 161 && b <= 255 && b != 173);
        if (!in) { m[(uint8_t)b] = utf8::encode((uint32_t)(256 + n)); n++; }
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
    std::array<bool, 256> special_lead{};   // the bytes a special token begins with, so the scan for one skips every other byte
    bool joyai = false;                     // whether the file's pretokenizer is joyai-llm rather than qwen2/qwen35
    int32_t bos_id = -1;
    int32_t eos_id = -1;

    // Whether `id` ends a generation: a model without an EOS id has no stop token at all, since folding it to 0 made token zero, an ordinary token, end every generation.
    // A request's ignore_eos masks the same id before sampling (infer::sample).
    bool is_eos(uint32_t id) const { return eos_id >= 0 && id == (uint32_t)eos_id; }

    Tokenizer(const gguf::GGUFModel& m) {
        // A file made for another tokenizer or pretokenizer would encode to valid-looking but wrong ids, so naming one is refused.
        // A key the file omits is not checked, which the synthetic test models rely on.
        auto require = [&](const std::string& key, std::initializer_list<const char*> implemented) {
            const gguf::MetaValue* v = m.find(key);
            if (!v) return;
            std::string names;
            for (const char* name : implemented) {
                if (v->vtype == gguf::V_STRING && v->s == name) return;
                names += (names.empty() ? "'" : ", '") + std::string(name) + "'";
            }
            throw std::runtime_error("tokenizer: unsupported " + key +
                                     (v->vtype == gguf::V_STRING ? " '" + v->s + "'" : std::string()) +
                                     "; implemented: " + names);
        };
        require("tokenizer.ggml.model", {"gpt2"});
        // qwen35 differs from qwen2 only in combining marks (see pretokenize); joyai-llm has its own splits (pretokenize_joyai).
        require("tokenizer.ggml.pre", {"qwen2", "qwen35", "joyai-llm"});
        if (const gguf::MetaValue* v = m.find("tokenizer.ggml.pre")) joyai = v->s == "joyai-llm";

        byte_to_char = build_byte_encoder();
        for (const auto& kv : byte_to_char) char_to_byte[kv.second] = kv.first;

        // tokens
        const gguf::MetaValue* toks = m.find("tokenizer.ggml.tokens");
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
        const gguf::MetaValue* tt = m.find("tokenizer.ggml.token_type");
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
        const gguf::MetaValue* mg = m.find("tokenizer.ggml.merges");
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
        for (const auto& sp : specials)
            if (!sp.text.empty()) special_lead[(unsigned char)sp.text[0]] = true;

        // ids
        if (const gguf::MetaValue* v = m.find("tokenizer.ggml.bos_token_id"))
            if (v->vtype == gguf::V_UINT32) bos_id = (int32_t)v->u;
        if (const gguf::MetaValue* v = m.find("tokenizer.ggml.eos_token_id"))
            if (v->vtype == gguf::V_UINT32) eos_id = (int32_t)v->u;
    }

    // --- pre-tokenization (Qwen2/Qwen3 Split regex from tokenizer.json) ---
    //   (?i:'s|'t|'re|'ve|'m|'ll|'d)
    //   | [^\r\n\p{L}\p{N}]?\p{L}+
    //   | \p{N}
    //   |  ?[^\s\p{L}\p{N}]+[\r\n]*
    //   | \s*[\r\n]+
    //   | \s+(?!\S)
    //   | \s+
    // This is not the GPT-2 regex: leading punctuation or underscore binds to the following word ("_snake", "(x"), digits are emitted one at a time, and a whitespace run ending in newlines stays one piece.
    // Alternatives are ordered and the first match wins; qwen35's regex adds \p{M} to the second alternative's letter run and the fourth's excluded class, which every byte at or above 0x80 being a letter here already does.
    static bool is_space(unsigned char c) { return c==' '||c=='\t'||c=='\n'||c=='\r'||c=='\f'||c=='\v'; }
    static bool is_ascii_letter(unsigned char c) { return (c>='a'&&c<='z')||(c>='A'&&c<='Z'); }
    static bool is_digit(unsigned char c) { return c>='0'&&c<='9'; }
    static bool is_letter(unsigned char c) { return is_ascii_letter(c) || c >= 0x80; }
    static bool is_nl(unsigned char c) { return c=='\r' || c=='\n'; }

    static std::vector<std::string> pretokenize(const std::string& text) {
        std::vector<std::string> out;
        static const char* contractions[] = {"'s","'t","'re","'ve","'m","'ll","'d"};
        size_t i = 0;
        const size_t n = text.size();
        auto emit = [&](size_t b, size_t e) { out.push_back(text.substr(b, e - b)); };
        auto ieq = [](char a, char b) {
            return std::tolower((unsigned char)a) == std::tolower((unsigned char)b);
        };
        while (i < n) {
            // 1. contractions, case-insensitive per the (?i:) group
            if (text[i] == '\'') {
                bool matched = false;
                for (const char* q : contractions) {
                    size_t len = std::strlen(q);
                    if (i + len > n) continue;
                    bool eq = true;
                    for (size_t k = 0; k < len; k++)
                        if (!ieq(text[i + k], q[k])) { eq = false; break; }
                    if (eq) { emit(i, i + len); i += len; matched = true; break; }
                }
                if (matched) continue;
            }
            // 2. optional non-newline non-alnum char, then one or more letters
            { size_t j = i;
              unsigned char c0 = (unsigned char)text[j];
              if (!is_nl(c0) && !is_letter(c0) && !is_digit(c0)) j++;
              size_t k = j;
              while (k < n && is_letter((unsigned char)text[k])) k++;
              if (k > j) { emit(i, k); i = k; continue; } }
            // 3. exactly one digit
            if (is_digit((unsigned char)text[i])) { emit(i, i + 1); i += 1; continue; }
            // 4. optional space, a run of non-space non-alnum, trailing newlines
            { size_t j = i;
              if (text[j] == ' ') j++;
              size_t k = j;
              while (k < n) { unsigned char c = (unsigned char)text[k];
                              if (is_space(c) || is_letter(c) || is_digit(c)) break;
                              k++; }
              if (k > j) { while (k < n && is_nl((unsigned char)text[k])) k++;
                           emit(i, k); i = k; continue; } }
            // 5. whitespace run truncated at its LAST newline, which is what keeps a blank line a single piece rather than two.
            { size_t k = i;
              while (k < n && is_space((unsigned char)text[k])) k++;
              size_t last_nl = std::string::npos;
              for (size_t q = i; q < k; q++) if (is_nl((unsigned char)text[q])) last_nl = q;
              if (last_nl != std::string::npos) { emit(i, last_nl + 1); i = last_nl + 1; continue; } }
            // 6. whitespace run minus its last char when a non-space follows; that last space belongs to the next token via rules 2 and 4.
            { size_t k = i;
              while (k < n && is_space((unsigned char)text[k])) k++;
              if (k > i) { size_t e = (k < n) ? k - 1 : k;
                           if (e > i) { emit(i, e); i = e; continue; } } }
            // 7. any remaining whitespace run
            { size_t k = i;
              while (k < n && is_space((unsigned char)text[k])) k++;
              if (k > i) { emit(i, k); i = k; continue; } }
            // safety net: consume one character by the length its lead byte announces, so invalid input still moves on.
            emit(i, i + utf8::lead_length((unsigned char)text[i]));
            i += utf8::lead_length((unsigned char)text[i]);
        }
        return out;
    }

    // joyai-llm: the file's three Isolated splits in turn, each on the previous one's pieces, over Unicode 16.0 classes (docs/src/tokenizer-tokenizer.md).
    // A byte that does not begin a valid UTF-8 sequence is read as one character of no class.
    struct CodePoint { uint32_t cp; uint32_t at; uint8_t bits; };

    static size_t joyai_numbers(const std::vector<CodePoint>& t, size_t i, size_t end) {
        size_t k = i;
        while (k < end && k - i < 3 && (t[k].bits & unicode_class::N)) k++;
        return k - i;
    }

    static size_t joyai_kana_han(const std::vector<CodePoint>& t, size_t i, size_t end) {
        size_t k = i;
        while (k < end && ((t[k].cp >= 0x4E00 && t[k].cp <= 0x9FA5) || (t[k].cp >= 0x3040 && t[k].cp <= 0x30FF))) k++;
        return k - i;
    }

    static size_t joyai_words(const std::vector<CodePoint>& t, size_t i, size_t end) {
        using namespace unicode_class;
        const auto nl = [&](size_t k) { return t[k].cp == '\r' || t[k].cp == '\n'; };
        const auto run = [&](size_t k, uint8_t bits) { while (k < end && (t[k].bits & bits)) k++; return k; };
        const auto ascii_letter = [&](size_t k) { return k < end && t[k].cp < 0x80 && (t[k].bits & L); };
        // ASCII punctuation is exactly the ASCII characters of classes P and S.
        if (t[i].cp < 0x80 && (t[i].bits & (P | S)) && ascii_letter(i + 1)) {
            size_t k = i + 1;
            while (ascii_letter(k)) k++;
            return k - i;
        }
        // The optional character falls back to none when no letter or mark follows it, which matters only when it is itself a mark.
        if (!nl(i) && !(t[i].bits & (L | P | S))) {
            size_t k = run(i + 1, L | M);
            if (k > i + 1) return k - i;
        }
        if (size_t k = run(i, L | M); k > i) return k - i;
        {
            size_t j = i + (t[i].cp == ' ' ? 1 : 0);
            size_t k = run(j, P | S);
            if (k > j) {
                while (k < end && nl(k)) k++;
                return k - i;
            }
        }
        size_t k = run(i, W);
        for (size_t q = k; q > i; q--)
            if (nl(q - 1)) return q - i;
        if (k == end && k > i) return k - i;
        if (k > i + 1) return k - 1 - i;
        return k - i;
    }

    static std::vector<std::string> pretokenize_joyai(const std::string& text) {
        std::vector<CodePoint> t;
        for (size_t i = 0; i < text.size();) {
            size_t len = utf8::valid_length(text, i);
            uint32_t cp = (unsigned char)text[i];
            if (len > 1) {
                cp &= 0x7F >> len;
                for (size_t k = 1; k < len; k++) cp = (cp << 6) | ((unsigned char)text[i + k] & 0x3F);
            }
            t.push_back({ cp, (uint32_t)i, len ? unicode_class::of(cp) : (uint8_t)0 });
            i += len ? len : 1;
        }
        using Match = size_t (*)(const std::vector<CodePoint>&, size_t, size_t);
        std::vector<std::pair<size_t, size_t>> pieces{ { 0, t.size() } }, next;
        for (Match match : { Match(joyai_numbers), Match(joyai_kana_han), Match(joyai_words) }) {
            next.clear();
            for (auto [b, e] : pieces) {
                size_t gap = b;
                for (size_t i = b; i < e;) {
                    size_t m = match(t, i, e);
                    if (!m) { i++; continue; }
                    if (gap < i) next.push_back({ gap, i });
                    next.push_back({ i, i + m });
                    i += m;
                    gap = i;
                }
                if (gap < e) next.push_back({ gap, e });
            }
            pieces.swap(next);
        }
        std::vector<std::string> out;
        out.reserve(pieces.size());
        for (auto [b, e] : pieces) {
            size_t from = t[b].at, to = e < t.size() ? t[e].at : text.size();
            out.push_back(text.substr(from, to - from));
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
            size_t l = utf8::lead_length((unsigned char)word[i]);
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

    // Encode text, handling special tokens as single ids: the longest special token at a byte one can begin with, and BPE over the pretokenized text between them.
    // A word merged once is not merged again within the call, since BPE gives a word the same ids wherever it stands; the map lives for one call, so calls share nothing.
    std::vector<uint32_t> encode(const std::string& text) const {
        std::vector<uint32_t> ids;
        std::unordered_map<std::string, std::vector<uint32_t>> merged;
        auto flush = [&](size_t from, size_t to) {
            if (from == to) return;
            for (const auto& tok : (joyai ? pretokenize_joyai : pretokenize)(text.substr(from, to - from))) {
                auto known = merged.find(tok);
                if (known == merged.end()) {
                    std::vector<uint32_t> word;
                    for (const auto& sym : bpe(byte_encode(tok))) {
                        auto it = token_to_id.find(sym);
                        if (it == token_to_id.end())
                            throw std::runtime_error("tokenizer: symbol not in vocab: " + sym);
                        word.push_back(it->second);
                    }
                    known = merged.emplace(tok, std::move(word)).first;
                }
                ids.insert(ids.end(), known->second.begin(), known->second.end());
            }
        };
        size_t start = 0;
        for (size_t i = 0; i < text.size();) {
            const SpecialToken* match = nullptr; size_t best_len = 0;
            if (special_lead[(unsigned char)text[i]])
                for (const auto& sp : specials)
                    if (sp.text.size() > best_len && text.compare(i, sp.text.size(), sp.text) == 0) {
                        match = &sp; best_len = sp.text.size();
                    }
            if (match) { flush(start, i); ids.push_back(match->id); i += best_len; start = i; }
            else ++i;
        }
        flush(start, text.size());
        return ids;
    }

    std::string decode(const std::vector<uint32_t>& ids) const {
        std::string out;
        for (uint32_t id : ids) {
            const std::string& tok = vocab.at(id);
            for (size_t i = 0; i < tok.size(); ) {
                size_t l = utf8::lead_length((unsigned char)tok[i]);
                std::string ch = tok.substr(i, l);
                auto it = char_to_byte.find(ch);
                if (it != char_to_byte.end()) out += (char)it->second;
                else out += ch;
                i += l;
            }
        }
        return out;
    }
};

} // namespace bpe
