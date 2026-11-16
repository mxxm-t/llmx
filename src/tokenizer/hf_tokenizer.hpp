#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/json.hpp"
#include "format/gguf.hpp"

// A Hugging Face directory's tokenizer.json, tokenizer_config.json and chat_template.jinja read into the tokenizer keys the GGUF converter writes, so bpe::Tokenizer and the chat template read a directory as they read its GGUF file.
// Strict: a tokenizer other than byte-level BPE with the qwen2 split and the NFC normalizer, or an option the tokenizer does not implement, is refused.

namespace bpe::hf {

using KeyValues = std::vector<std::pair<std::string, gguf::MetaValue>>;

// The Split pattern of Qwen2 and Qwen3 tokenizer.json files, which Tokenizer::pretokenize implements as pre "qwen2".
inline const char* const qwen2_pattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

// GGUF token types.
enum : int32_t { token_normal = 1, token_control = 3, token_user = 4, token_unused = 5 };

namespace detail {

inline std::string read_text(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open file: " + path.u8string());
    return std::string((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
}

inline jmini::Value read_json(const std::filesystem::path& path) {
    jmini::Value v = jmini::parse(read_text(path));
    if (!v.isObject()) throw std::runtime_error(path.filename().u8string() + " is not an object");
    std::set<std::string> keys;
    for (const auto& item : v.obj)
        if (!keys.insert(item.first).second) throw std::runtime_error(path.filename().u8string() + " repeats key " + item.first);
    return v;
}

// A JSON scalar as the rules below write it: null, true, false, a quoted string or a whole number.
inline std::string text(const jmini::Value& v) {
    switch (v.t) {
        case jmini::Value::T::Null: return "null";
        case jmini::Value::T::Bool: return v.b ? "true" : "false";
        case jmini::Value::T::String: return "\"" + v.str + "\"";
        case jmini::Value::T::Number: return v.whole ? std::to_string(int64_t(v.num)) : std::string("<fraction>");
        default: return "<structure>";
    }
}

struct Rule {
    const char* key;
    const char* allowed;   // the values accepted, as JSON texts separated by '|', or null for any value
};

// `v`, named `what`, is an object whose keys are all listed in `rules`, each holding a value its rule allows; `required` keys must be present.
inline void check(const jmini::Value* v, const std::string& what, std::initializer_list<Rule> rules, std::initializer_list<const char*> required = {}) {
    if (!v || !v->isObject()) throw std::runtime_error(what + " is not an object");
    for (const char* key : required)
        if (!v->get(key)) throw std::runtime_error(what + " has no " + key);
    std::set<std::string> seen;
    for (const auto& [key, value] : v->obj) {
        if (!seen.insert(key).second) throw std::runtime_error(what + " repeats key " + key);
        const Rule* rule = nullptr;
        for (const Rule& r : rules)
            if (key == r.key) rule = &r;
        if (!rule) throw std::runtime_error(what + " key " + key + " is not one llmx reads");
        if (!rule->allowed) continue;
        const std::string value_text = text(value), list = rule->allowed;
        bool ok = false;
        for (size_t start = 0;;) {
            const size_t bar = list.find('|', start);
            ok |= list.substr(start, bar == std::string::npos ? std::string::npos : bar - start) == value_text;
            if (bar == std::string::npos) break;
            start = bar + 1;
        }
        if (!ok) throw std::runtime_error(what + " " + key + " = " + value_text + " is not supported (" + list + ")");
    }
}

inline gguf::MetaValue string_value(const std::string& s) {
    gguf::MetaValue v;
    v.vtype = gguf::V_STRING;
    v.s = s;
    return v;
}

inline gguf::MetaValue array_value(uint32_t element_type, size_t n) {
    gguf::MetaValue v;
    v.vtype = gguf::V_ARRAY;
    v.u = element_type;
    v.arr.reserve(n);
    return v;
}

inline uint32_t token_id(const jmini::Value& v, uint64_t n_vocab, const std::string& what) {
    if (!v.isNumber() || !v.whole || v.num < 0 || v.num >= double(n_vocab))
        throw std::runtime_error(what + " is not a token id below the embedding's " + std::to_string(n_vocab) + " rows");
    return uint32_t(v.num);
}

} // namespace detail

// The tokenizer keys of the directory `dir` whose embedding has `n_vocab` rows, and whose config.json is `config`.
// Token ids the files do not use are padded as "[PAD<id>]" of type unused up to `n_vocab`, as the converter pads them.
inline KeyValues read_keys(const std::string& dir, const jmini::Value& config, uint64_t n_vocab) {
    using detail::check;
    const std::filesystem::path root = std::filesystem::u8path(dir);
    const jmini::Value tj = detail::read_json(root / "tokenizer.json");
    const jmini::Value tc = detail::read_json(root / "tokenizer_config.json");

    check(&tj, "tokenizer.json", {{"version", nullptr}, {"truncation", "null"}, {"padding", "null"}, {"added_tokens", nullptr},
        {"normalizer", nullptr}, {"pre_tokenizer", nullptr}, {"post_processor", nullptr}, {"decoder", nullptr}, {"model", nullptr}},
        {"added_tokens", "normalizer", "pre_tokenizer", "model"});
    check(tj.get("normalizer"), "tokenizer.json normalizer", {{"type", "\"NFC\""}}, {"type"});
    const jmini::Value* pre = tj.get("pre_tokenizer");
    check(pre, "tokenizer.json pre_tokenizer", {{"type", "\"Sequence\""}, {"pretokenizers", nullptr}}, {"type", "pretokenizers"});
    const jmini::Value* steps = pre->get("pretokenizers");
    if (!steps->isArray() || steps->arr.size() != 2) throw std::runtime_error("tokenizer.json pre_tokenizer is not a Split and a ByteLevel step");
    check(&steps->arr[0], "tokenizer.json Split", {{"type", "\"Split\""}, {"pattern", nullptr}, {"behavior", "\"Isolated\""}, {"invert", "false"}},
        {"type", "pattern", "behavior", "invert"});
    const jmini::Value* pattern = steps->arr[0].get("pattern");
    check(pattern, "tokenizer.json Split pattern", {{"Regex", nullptr}}, {"Regex"});
    if (!pattern->get("Regex")->isString() || pattern->get("Regex")->str != qwen2_pattern)
        throw std::runtime_error("tokenizer.json Split pattern is not the qwen2 pattern, the one llmx implements for this format");
    const std::initializer_list<detail::Rule> byte_level = {{"type", "\"ByteLevel\""}, {"add_prefix_space", "false"}, {"trim_offsets", "true|false"},
        {"use_regex", "false"}};
    check(&steps->arr[1], "tokenizer.json ByteLevel pre_tokenizer", byte_level, {"type"});
    // A null post-processor adds nothing; offsets, which ByteLevel trims, are not used.
    if (const jmini::Value* post = tj.get("post_processor"); post && post->t != jmini::Value::T::Null)
        check(post, "tokenizer.json post_processor", byte_level, {"type"});
    check(tj.get("decoder"), "tokenizer.json decoder", byte_level, {"type"});
    const jmini::Value* model = tj.get("model");
    check(model, "tokenizer.json model", {{"type", "\"BPE\""}, {"dropout", "null"}, {"unk_token", "null"}, {"continuing_subword_prefix", "null|\"\""},
        {"end_of_word_suffix", "null|\"\""}, {"fuse_unk", "false"}, {"byte_fallback", "false"}, {"ignore_merges", "false"}, {"vocab", nullptr},
        {"merges", nullptr}}, {"type", "vocab", "merges"});

    // Tokens: the model's vocabulary and the added tokens, each id once.
    std::vector<std::string> tokens(n_vocab);
    std::vector<uint32_t> types(n_vocab, 0);
    const jmini::Value* vocab = model->get("vocab");
    if (!vocab->isObject()) throw std::runtime_error("tokenizer.json model vocab is not an object");
    for (const auto& [token, id_value] : vocab->obj) {
        const uint32_t id = detail::token_id(id_value, n_vocab, "tokenizer.json vocab id of " + token);
        if (types[id]) throw std::runtime_error("tokenizer.json vocab gives id " + std::to_string(id) + " twice");
        tokens[id] = token;
        types[id] = token_normal;
    }
    const jmini::Value* added = tj.get("added_tokens");
    if (!added->isArray()) throw std::runtime_error("tokenizer.json added_tokens is not an array");
    for (const jmini::Value& a : added->arr) {
        check(&a, "tokenizer.json added token", {{"id", nullptr}, {"content", nullptr}, {"single_word", "false"}, {"lstrip", "false"},
            {"rstrip", "false"}, {"normalized", "false"}, {"special", "true|false"}}, {"id", "content", "special"});
        if (!a.get("content")->isString() || a.get("content")->str.empty()) throw std::runtime_error("tokenizer.json added token has no content");
        const std::string& content = a.get("content")->str;
        const uint32_t id = detail::token_id(*a.get("id"), n_vocab, "tokenizer.json added token id of " + content);
        if (types[id] && tokens[id] != content) throw std::runtime_error("tokenizer.json added token " + content + " takes an id the vocab gives another token");
        tokens[id] = content;
        // The converter also makes a token that looks like <|name|> a control token when tokenizer.json does not call it special.
        const bool looks_special = content.size() >= 4 && content.compare(0, 2, "<|") == 0 && content.compare(content.size() - 2, 2, "|>") == 0;
        types[id] = a.get("special")->b || looks_special ? token_control : token_user;
    }
    gguf::MetaValue token_array = detail::array_value(gguf::V_STRING, n_vocab), type_array = detail::array_value(gguf::V_INT32, n_vocab);
    for (uint64_t i = 0; i < n_vocab; ++i) {
        if (!types[i]) tokens[i] = "[PAD" + std::to_string(i) + "]", types[i] = token_unused;
        token_array.arr.push_back(detail::string_value(tokens[i]));
        gguf::MetaValue t;
        t.vtype = gguf::V_INT32;
        t.i = types[i];
        type_array.arr.push_back(t);
    }

    // Merges, "a b" or ["a", "b"], stored as "a b"; a part holding a space could not be split again.
    const jmini::Value* merges = model->get("merges");
    if (!merges->isArray()) throw std::runtime_error("tokenizer.json model merges is not an array");
    gguf::MetaValue merge_array = detail::array_value(gguf::V_STRING, merges->arr.size());
    for (const jmini::Value& mv : merges->arr) {
        std::string merge;
        if (mv.isString()) {
            merge = mv.str;
            const size_t space = merge.find(' ');
            if (space == std::string::npos || space == 0 || space + 1 == merge.size() || merge.find(' ', space + 1) != std::string::npos)
                throw std::runtime_error("tokenizer.json merge is not two parts: " + merge);
        } else if (mv.isArray() && mv.arr.size() == 2 && mv.arr[0].isString() && mv.arr[1].isString()) {
            for (const auto& part : mv.arr)
                if (part.str.empty() || part.str.find(' ') != std::string::npos) throw std::runtime_error("tokenizer.json merge part is empty or holds a space");
            merge = mv.arr[0].str + " " + mv.arr[1].str;
        } else {
            throw std::runtime_error("tokenizer.json merge is neither a string nor a pair of strings");
        }
        merge_array.arr.push_back(detail::string_value(merge));
    }

    KeyValues kv;
    kv.emplace_back("tokenizer.ggml.model", detail::string_value("gpt2"));
    kv.emplace_back("tokenizer.ggml.pre", detail::string_value("qwen2"));
    kv.emplace_back("tokenizer.ggml.tokens", std::move(token_array));
    kv.emplace_back("tokenizer.ggml.token_type", std::move(type_array));
    kv.emplace_back("tokenizer.ggml.merges", std::move(merge_array));

    // tokenizer_config.json: the options the tokenizer implements, special tokens named by their text, and the chat template.
    check(&tc, "tokenizer_config.json", {{"add_bos_token", "false"}, {"add_eos_token", "false"}, {"add_prefix_space", "false|null"},
        {"added_tokens_decoder", nullptr}, {"additional_special_tokens", nullptr}, {"bos_token", nullptr}, {"chat_template", nullptr},
        {"clean_up_tokenization_spaces", "false"}, {"eos_token", nullptr}, {"errors", "\"replace\""}, {"model_max_length", nullptr},
        {"pad_token", nullptr}, {"split_special_tokens", "false"}, {"tokenizer_class", nullptr}, {"unk_token", nullptr},
        {"extra_special_tokens", nullptr}, {"_commit_hash", nullptr}});
    // Special ids: tokenizer_config.json's text first, then config.json's id, each kept only below n_vocab, as the converter takes them.
    // A pad text naming no token gives no id from the text, as HF's tokenizer gives none; the other texts must name an added token.
    struct Special { const char* hf; const char* gguf; bool must_name; };
    static const Special specials[] = {{"bos", "bos_token_id", true}, {"eos", "eos_token_id", true}, {"unk", "unknown_token_id", true},
        {"pad", "padding_token_id", false}};
    for (const Special& s : specials) {
        const std::string name = std::string(s.hf) + "_token";
        int64_t id = -1;
        if (const jmini::Value* entry = tc.get(name); entry && entry->t != jmini::Value::T::Null) {
            const jmini::Value* content = entry->isObject() ? entry->get("content") : entry;
            if (!content || !content->isString()) throw std::runtime_error("tokenizer_config.json " + name + " is not a token's text");
            for (const jmini::Value& a : added->arr)
                if (a.get("content")->str == content->str) id = int64_t(a.get("id")->num);
            if (id < 0 && s.must_name) throw std::runtime_error("tokenizer_config.json " + name + " " + content->str + " is not an added token");
        }
        if (const jmini::Value* v = config.get(std::string(s.hf) + "_token_id"); id < 0 && v && v->isNumber() && v->whole && v->num >= 0 && v->num < double(n_vocab))
            id = int64_t(v->num);
        if (id < 0) continue;
        gguf::MetaValue value;
        value.vtype = gguf::V_UINT32;
        value.u = uint64_t(id);
        kv.emplace_back(std::string("tokenizer.ggml.") + s.gguf, value);
    }
    for (const char* which : {"bos", "eos"})
        if (const jmini::Value* add = tc.get(std::string("add_") + which + "_token")) {
            gguf::MetaValue value;
            value.vtype = gguf::V_BOOL;
            value.b = add->b;
            kv.emplace_back(std::string("tokenizer.ggml.add_") + which + "_token", value);
        }
    // extra_special_tokens, a map of names to texts or a list of texts, may name only added tokens; HF keeps a listed text whole, which a text outside tokenizer.json's added tokens would not be here.
    if (const jmini::Value* extra = tc.get("extra_special_tokens"); extra && extra->t != jmini::Value::T::Null) {
        if (!extra->isObject() && !extra->isArray()) throw std::runtime_error("tokenizer_config.json extra_special_tokens is not a map or a list");
        std::vector<const jmini::Value*> texts;
        if (extra->isObject()) for (const auto& entry : extra->obj) texts.push_back(&entry.second);
        else for (const jmini::Value& v : extra->arr) texts.push_back(&v);
        for (const jmini::Value* v : texts) {
            bool named = v->isString();
            if (named) {
                named = false;
                for (const jmini::Value& a : added->arr) named = named || a.get("content")->str == v->str;
            }
            if (!named) throw std::runtime_error("tokenizer_config.json extra_special_tokens " + (v->isString() ? v->str : std::string("entry")) + " is not an added token");
        }
    }
    // added_tokens_decoder, which HF reads over tokenizer.json's list, must agree with it.
    if (const jmini::Value* decoder = tc.get("added_tokens_decoder")) {
        if (!decoder->isObject()) throw std::runtime_error("tokenizer_config.json added_tokens_decoder is not an object");
        for (const auto& [id, entry] : decoder->obj) {
            const jmini::Value* match = nullptr;
            for (const jmini::Value& a : added->arr)
                if (std::to_string(int64_t(a.get("id")->num)) == id) match = &a;
            if (!match || !entry.isObject() || !entry.get("content") || !entry.get("special") ||
                detail::text(*entry.get("content")) != detail::text(*match->get("content")) ||
                detail::text(*entry.get("special")) != detail::text(*match->get("special")))
                throw std::runtime_error("tokenizer_config.json added_tokens_decoder " + id + " differs from tokenizer.json's added token");
        }
    }
    // chat_template.jinja, when present, is the template, as for the converter; a list of named templates is refused.
    const std::filesystem::path jinja = root / "chat_template.jinja";
    if (std::filesystem::exists(jinja)) {
        kv.emplace_back("tokenizer.chat_template", detail::string_value(detail::read_text(jinja)));
    } else if (const jmini::Value* t = tc.get("chat_template"); t && t->t != jmini::Value::T::Null) {
        if (!t->isString()) throw std::runtime_error("tokenizer_config.json chat_template is not one template");
        kv.emplace_back("tokenizer.chat_template", detail::string_value(t->str));
    }
    return kv;
}

} // namespace bpe::hf
