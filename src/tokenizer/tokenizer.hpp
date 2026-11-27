#pragma once
#include <algorithm>
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
#include "core/unicode_nfc.hpp"
#include "core/utf8.hpp"
#include "format/gguf.hpp"

// GPT-2 style byte-level BPE tokenizer, implemented from scratch.
// Reads tokenizer metadata from a GGUF model:
//   tokenizer.ggml.model        = "gpt2"           (refused when present with another value)
//   tokenizer.ggml.pre          = a name of kPres          (refused when present with another value)
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

// The split regex a pretokenizer runs, named by the pattern rather than by a model.
enum class Split : uint8_t { qwen2, joyai, o200k };

// What a tokenizer.ggml.pre name selects: its split, whether HF's NFC normalizer runs first, and whether a piece the vocabulary holds whole is that token before BPE (HF's ignore_merges).
struct Pre { const char* name; Split split; bool nfc; bool whole_pieces; };

// qwen35 differs from qwen2 only in combining marks (see pretokenize); minimax-m2 and inkling2 files carry the o200k_base pattern, inkling2 with ignore_merges and without NFC.
inline constexpr Pre kPres[] = {
    {"qwen2", Split::qwen2, true, false},
    {"qwen35", Split::qwen2, true, false},
    {"joyai-llm", Split::joyai, false, false},
    {"minimax-m2", Split::o200k, true, false},
    {"inkling2", Split::o200k, false, true},
};

class Tokenizer {
public:
    std::vector<std::string> vocab;
    std::unordered_map<std::string, uint32_t> token_to_id;
    std::unordered_map<std::string, uint32_t> merge_rank;
    std::unordered_map<uint8_t, std::string> byte_to_char;
    std::unordered_map<std::string, uint8_t> char_to_byte;
    std::vector<SpecialToken> specials;     // control / user-defined tokens, ordered by their first byte
    std::array<uint32_t, 257> special_at{}; // where the specials beginning with each byte start in `specials`, so a position visits only its candidates
    Pre pre = kPres[0];                     // the file's pretokenizer; a file naming none is read as qwen2
    int32_t bos_id = -1;
    int32_t eos_id = -1;

    // Whether `id` ends a generation: a model without an EOS id has no stop token at all, since folding it to 0 made token zero, an ordinary token, end every generation.
    // A request's ignore_eos masks the same id before sampling (infer::sample).
    bool is_eos(uint32_t id) const { return eos_id >= 0 && id == (uint32_t)eos_id; }

    Tokenizer(const gguf::GGUFModel& m) {
        // A file made for another tokenizer or pretokenizer would encode to valid-looking but wrong ids, so naming one is refused.
        // A key the file omits is not checked, which the synthetic test models rely on.
        auto require = [&](const std::string& key, const auto& implemented, auto name_of) -> int {
            const gguf::MetaValue* v = m.find(key);
            if (!v) return -1;
            std::string names;
            int at = 0;
            for (const auto& entry : implemented) {
                if (v->vtype == gguf::V_STRING && v->s == name_of(entry)) return at;
                names += (names.empty() ? "'" : ", '") + std::string(name_of(entry)) + "'";
                at++;
            }
            throw std::runtime_error("tokenizer: unsupported " + key +
                                     (v->vtype == gguf::V_STRING ? " '" + v->s + "'" : std::string()) +
                                     "; implemented: " + names);
        };
        static constexpr const char* kModels[] = {"gpt2"};
        require("tokenizer.ggml.model", kModels, [](const char* name) { return name; });
        if (int at = require("tokenizer.ggml.pre", kPres, [](const Pre& p) { return p.name; }); at >= 0) pre = kPres[at];

        byte_to_char = build_byte_encoder();
        for (const auto& kv : byte_to_char) char_to_byte[kv.second] = kv.first;

        // tokens
        const gguf::MetaValue* toks = m.find("tokenizer.ggml.tokens");
        if (!toks || toks->vtype != gguf::V_ARRAY)
            throw std::runtime_error("tokenizer: missing tokenizer.ggml.tokens");
        if (toks->u != gguf::V_STRING)
            throw std::runtime_error("tokenizer: tokenizer.ggml.tokens is an array of type " + std::to_string(toks->u) + ", not of strings");
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
            if (tt->arr.size() < vocab.size())
                throw std::runtime_error("tokenizer: tokenizer.ggml.token_type has " + std::to_string(tt->arr.size()) + " entries for " +
                                         std::to_string(vocab.size()) + " tokens");
            for (size_t i = 0; i < types.size(); i++) {
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
            if ((types[i] == 3 || types[i] == 4) && !vocab[i].empty())
                specials.push_back({ vocab[i], (uint32_t)i });
        }
        std::stable_sort(specials.begin(), specials.end(), [](const SpecialToken& a, const SpecialToken& b) {
            return (unsigned char)a.text[0] < (unsigned char)b.text[0];
        });
        for (size_t i = 0, b = 0; b < 256; b++) {
            special_at[b] = (uint32_t)i;
            while (i < specials.size() && (unsigned char)specials[i].text[0] == b) i++;
        }
        special_at[256] = (uint32_t)specials.size();

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

    // The joyai-llm and o200k splits run over code points and their Unicode 16.0 classes (docs/src/tokenizer-tokenizer.md).
    // A byte that does not begin a valid UTF-8 sequence is read as one character of no class.
    struct CodePoint { uint32_t cp; uint32_t at; uint8_t bits; };

    // \p{N}{1,3}
    static size_t numbers(const std::vector<CodePoint>& t, size_t i, size_t end) {
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
        return spaces(t, i, end);
    }

    // \s*[\r\n]+|\s+(?!\S)|\s+
    static size_t spaces(const std::vector<CodePoint>& t, size_t i, size_t end) {
        size_t k = i;
        while (k < end && (t[k].bits & unicode_class::W)) k++;
        for (size_t q = k; q > i; q--)
            if (t[q - 1].cp == '\r' || t[q - 1].cp == '\n') return q - i;
        if (k == end && k > i) return k - i;
        if (k > i + 1) return k - 1 - i;
        return k - i;
    }

    // (?i:'s|'t|'re|'ve|'m|'ll|'d), where the regex engine's case folding also takes U+017F, long s, for s.
    static size_t contraction(const std::vector<CodePoint>& t, size_t i, size_t end) {
        if (i >= end || t[i].cp != '\'') return 0;
        const auto folded = [&](size_t k) -> uint32_t {
            if (k >= end) return 0;
            const uint32_t c = t[k].cp;
            return c == 0x17F ? 's' : c >= 'A' && c <= 'Z' ? c + 32 : c;
        };
        const uint32_t a = folded(i + 1), b = folded(i + 2);
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
        if (((a == 'r' || a == 'v') && b == 'e') || (a == 'l' && b == 'l')) return 3;
        return 0;
    }

    // tiktoken's o200k_base split as one Isolated split, its pattern given in docs/src/tokenizer-tokenizer.md.
    // The two letter classes overlap in Lm, Lo and M, so the first alternative backtracks its upper run to the last place a lower run can start.
    static size_t o200k_words(const std::vector<CodePoint>& t, size_t i, size_t end) {
        using namespace unicode_class;
        const auto upper = [](uint8_t b) { return (b & M) || ((b & L) && !(b & LL)); };
        const auto lower = [](uint8_t b) { return (b & M) || ((b & L) && !(b & LU)); };
        const auto run = [&](size_t k, auto in) { while (k < end && in(t[k].bits)) k++; return k; };
        // The optional first character is tried before it is left out, in each letter alternative in turn.
        const bool lead = t[i].cp != '\r' && t[i].cp != '\n' && !(t[i].bits & (L | N));
        for (size_t j = i + lead;; j = i) {
            const size_t a = run(j, upper);
            for (size_t k = a + 1; k-- > j;)
                if (k < end && lower(t[k].bits)) {
                    const size_t b = run(k, lower);
                    return b + contraction(t, b, end) - i;
                }
            if (j == i) break;
        }
        for (size_t j = i + lead;; j = i) {
            if (const size_t a = run(j, upper); a > j) {
                const size_t b = run(a, lower);
                return b + contraction(t, b, end) - i;
            }
            if (j == i) break;
        }
        if (size_t k = numbers(t, i, end)) return k;
        {
            const size_t j = i + (t[i].cp == ' ' ? 1 : 0);
            size_t k = run(j, [](uint8_t b) { return !(b & (W | L | N)); });
            if (k > j) {
                while (k < end && (t[k].cp == '\r' || t[k].cp == '\n' || t[k].cp == '/')) k++;
                return k - i;
            }
        }
        return spaces(t, i, end);
    }

    using Match = size_t (*)(const std::vector<CodePoint>&, size_t, size_t);

    // The Isolated splits `matches` in turn, each on the previous one's pieces: a match is a piece, and so is each gap between matches.
    static std::vector<std::string> split(const std::string& text, std::initializer_list<Match> matches) {
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
        std::vector<std::pair<size_t, size_t>> pieces{ { 0, t.size() } }, next;
        for (Match match : matches) {
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

    // joyai-llm: the file's three Isolated splits.
    static std::vector<std::string> pretokenize_joyai(const std::string& text) { return split(text, { numbers, joyai_kana_han, joyai_words }); }

    static std::vector<std::string> pretokenize_o200k(const std::string& text) { return split(text, { o200k_words }); }

    std::string byte_encode(const std::string& raw) const {
        std::string s;
        for (unsigned char b : raw) s += byte_to_char.at(b);
        return s;
    }

    // BPE merge loop over a byte-mapped word: the adjacent pair of lowest rank, the leftmost on a tie, merged until no pair has a rank.
    // The symbols are spans of the word in a linked list and the candidate pairs a heap keyed by rank and the left span's index, so a word of n symbols takes O(n log n) rather than a scan per merge.
    std::vector<std::string> bpe(const std::string& word) const {
        struct Span { uint32_t begin, len; int32_t prev, next; };
        struct Pair { uint32_t rank, left; int32_t right; uint32_t left_len, right_len; };
        // A pair is live while both spans are as they were pushed and still adjacent; a span merged away has length 0 and one that grew a longer length, so a stale pair fails the length check.
        std::vector<Span> spans;
        for (size_t i = 0; i < word.size();) {
            const uint32_t l = (uint32_t)utf8::lead_length((unsigned char)word[i]);
            spans.push_back({ (uint32_t)i, l, (int32_t)spans.size() - 1, (int32_t)spans.size() + 1 });
            i += l;
        }
        if (spans.empty()) return {};
        spans.back().next = -1;
        std::vector<Pair> heap;
        // A merged span keeps its left half's index and the survivors keep their order, so the lowest index among equal ranks is the leftmost pair, as a scan chooses it.
        const auto later = [](const Pair& a, const Pair& b) { return a.rank != b.rank ? a.rank > b.rank : a.left > b.left; };
        std::string text;
        const auto offer = [&](uint32_t left) {
            const Span& l = spans[left];
            if (l.next < 0) return;
            const Span& r = spans[(size_t)l.next];
            text.assign(word, l.begin, l.len + r.len);
            const auto it = merge_rank.find(text);
            if (it == merge_rank.end()) return;
            heap.push_back({ it->second, left, l.next, l.len, r.len });
            std::push_heap(heap.begin(), heap.end(), later);
        };
        for (uint32_t i = 0; i + 1 < spans.size(); ++i) offer(i);
        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(), later);
            const Pair p = heap.back();
            heap.pop_back();
            Span& l = spans[p.left];
            if (l.len != p.left_len || l.next != p.right || spans[(size_t)p.right].len != p.right_len) continue;
            Span& r = spans[(size_t)p.right];
            l.len += r.len;
            l.next = r.next;
            r.len = 0;
            if (l.next >= 0) spans[(size_t)l.next].prev = (int32_t)p.left;
            if (l.prev >= 0) offer((uint32_t)l.prev);
            offer(p.left);
        }
        std::vector<std::string> sym;
        for (int32_t i = 0; i >= 0; i = spans[(size_t)i].next) sym.push_back(word.substr(spans[(size_t)i].begin, spans[(size_t)i].len));
        return sym;
    }

    // Encode text, handling special tokens as single ids: the longest special token at a byte one can begin with, and BPE over the pretokenized text between them.
    // A word merged once is not merged again within the call, since BPE gives a word the same ids wherever it stands; the map lives for one call, so calls share nothing.
    std::vector<uint32_t> encode(const std::string& text) const {
        std::vector<uint32_t> ids;
        std::unordered_map<std::string, std::vector<uint32_t>> merged;
        auto flush = [&](size_t from, size_t to) {
            if (from == to) return;
            // HF's NFC normalizer, where the file names it, runs on the text between special tokens before the split.
            const std::string piece = pre.nfc ? unicode_nfc::normalize(text.substr(from, to - from)) : text.substr(from, to - from);
            const auto pretokenized = pre.split == Split::joyai ? pretokenize_joyai(piece) : pre.split == Split::o200k ? pretokenize_o200k(piece) : pretokenize(piece);
            for (const auto& tok : pretokenized) {
                auto known = merged.find(tok);
                if (known == merged.end()) {
                    std::vector<uint32_t> word;
                    const std::string mapped = byte_encode(tok);
                    // A piece is never a special token's text, which the scan below takes first, so a whole piece is an ordinary token.
                    if (const auto whole = pre.whole_pieces ? token_to_id.find(mapped) : token_to_id.end(); whole != token_to_id.end())
                        word.push_back(whole->second);
                    else for (const auto& sym : bpe(mapped)) {
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
            const unsigned char lead = (unsigned char)text[i];
            for (uint32_t k = special_at[lead]; k < special_at[lead + 1]; k++) {
                const SpecialToken& sp = specials[k];
                if (sp.text.size() > best_len && text.compare(i, sp.text.size(), sp.text) == 0) {
                    match = &sp; best_len = sp.text.size();
                }
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
