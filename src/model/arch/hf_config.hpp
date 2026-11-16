#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "format/gguf.hpp"

// A Hugging Face directory's config.json and tensor names turned into the metadata and names an architecture's GGUF reader takes, by one table per architecture module.
// Strict: a config key or a tensor the table does not name is refused, so a checkpoint the module does not run is never read as one it does.

namespace infer::hf {

enum class Rule {
    integer,   // a whole number from 1 up to 2^32 - 1, stored as UINT32
    real,      // a finite positive number, stored as FLOAT32
    check,     // checked against `allowed` and not stored
    each,      // an array whose every entry is checked as `check` checks a value, not stored
};

struct Key {
    const char* hf;
    const char* gguf;      // the key under the architecture's prefix, for integer and real
    Rule rule;
    const char* allowed;   // for check: the values accepted, as JSON texts separated by '|', or null for any value
};

// "{N}" in both names stands for a layer number below the block count.
struct Name {
    const char* hf;
    const char* gguf;
};

struct Table {
    const char* model_type;     // config.json model_type
    const char* class_name;     // config.json architectures' one entry
    std::vector<Key> keys;
    std::vector<Name> names;
    const char* embedding;      // the GGUF names of the token embedding and the head, which the tie and vocab_size checks read
    const char* head;
};

namespace detail {

// A JSON scalar as the table writes it: null, true, false, a quoted string or a whole number.
inline std::string text(const jmini::Value& v) {
    switch (v.t) {
        case jmini::Value::T::Null: return "null";
        case jmini::Value::T::Bool: return v.b ? "true" : "false";
        case jmini::Value::T::String: return "\"" + v.str + "\"";
        case jmini::Value::T::Number: return v.whole ? std::to_string(int64_t(v.num)) : std::string("<fraction>");
        default: return "<structure>";
    }
}

inline bool allowed(const std::string& list, const std::string& value) {
    for (size_t start = 0;;) {
        const size_t bar = list.find('|', start);
        if (list.substr(start, bar == std::string::npos ? std::string::npos : bar - start) == value) return true;
        if (bar == std::string::npos) return false;
        start = bar + 1;
    }
}

inline void check(const Key& k, const std::string& key, const jmini::Value& v) {
    if (k.allowed && !allowed(k.allowed, text(v)))
        throw std::runtime_error("config.json " + key + " = " + text(v) + " is not supported (" + k.allowed + ")");
}

inline gguf::MetaValue meta(const jmini::Value& v, Rule rule, const std::string& key) {
    gguf::MetaValue out;
    if (rule == Rule::integer) {
        if (!v.isNumber() || !v.whole || v.num < 1 || v.num > 4294967295.0)
            throw std::runtime_error("config.json " + key + " is not a positive 32-bit integer");
        out.vtype = gguf::V_UINT32;
        out.u = uint64_t(v.num);
    } else {
        const float f = float(v.num);
        if (!v.isNumber() || !(v.num > 0) || !std::isfinite(f) || f == 0)
            throw std::runtime_error("config.json " + key + " is not a finite positive number");
        out.vtype = gguf::V_FLOAT32;
        std::memcpy(&out.fb, &f, sizeof(f));
    }
    return out;
}

// The layer number `name` gives where `pattern` has "{N}", if it matches.
inline bool match(const std::string& pattern, const std::string& name, uint64_t& layer) {
    const size_t at = pattern.find("{N}");
    if (at == std::string::npos) return pattern == name;
    const std::string head = pattern.substr(0, at), tail = pattern.substr(at + 3);
    if (name.size() <= head.size() + tail.size() || name.compare(0, head.size(), head) ||
        name.compare(name.size() - tail.size(), tail.size(), tail)) return false;
    const std::string digits = name.substr(head.size(), name.size() - head.size() - tail.size());
    if (digits.size() > 9 || (digits.size() > 1 && digits[0] == '0')) return false;
    for (char c : digits)
        if (c < '0' || c > '9') return false;
    layer = std::stoull(digits);
    return true;
}

} // namespace detail

// `m`, the tensors of a directory under their HF names, given the metadata and names `table` makes of `config` for the architecture `arch`, with the table's keys under its prefix ahead of those `m` already holds, such as hf_quant's.
// A tied head stored beside the embedding is dropped, as the HF model ignores it, so the head reads the embedding as the GGUF file of the same checkpoint has it.
inline void apply(const Table& table, const std::string& arch, const jmini::Value& config, gguf::GGUFModel& m) {
    if (!config.isObject()) throw std::runtime_error("config.json is not an object");
    const jmini::Value* classes = config.get("architectures");
    if (!classes || !classes->isArray() || classes->arr.size() != 1 || !classes->arr[0].isString() ||
        classes->arr[0].str != table.class_name)
        throw std::runtime_error(std::string("config.json architectures must be exactly [\"") + table.class_name + "\"]");
    const jmini::Value* type = config.get("model_type");
    if (!type || !type->isString() || type->str != table.model_type)
        throw std::runtime_error(std::string("config.json model_type must be ") + table.model_type);

    std::vector<std::pair<std::string, gguf::MetaValue>> kv;
    gguf::MetaValue name;
    name.vtype = gguf::V_STRING;
    name.s = arch;
    kv.emplace_back("general.architecture", name);
    std::set<std::string> seen;
    for (const auto& [key, value] : config.obj) {
        if (!seen.insert(key).second) throw std::runtime_error("config.json repeats key " + key);
        if (key == "architectures" || key == "model_type") continue;
        bool known = false;
        for (const Key& k : table.keys) {
            if (key != k.hf) continue;
            known = true;
            if (k.rule == Rule::check) {
                detail::check(k, key, value);
            } else if (k.rule == Rule::each) {
                if (!value.isArray()) throw std::runtime_error("config.json " + key + " is not an array");
                for (const jmini::Value& entry : value.arr) detail::check(k, key, entry);
            } else {
                kv.emplace_back(arch + "." + k.gguf, detail::meta(value, k.rule, key));
            }
        }
        if (!known) throw std::runtime_error("config.json key " + key + " is not one llmx reads for " + table.model_type);
    }
    const auto find = [&](const std::string& key) -> const gguf::MetaValue* {
        for (const auto& item : kv)
            if (item.first == key) return &item.second;
        return nullptr;
    };
    const gguf::MetaValue* layers = find(arch + ".block_count");
    if (!layers) throw std::runtime_error("config.json gives no layer count");

    // Tensors: renamed, a layer past the count refused, a tied head dropped.
    const jmini::Value* tie = config.get("tie_word_embeddings");
    const bool tied = tie && tie->t == jmini::Value::T::Bool && tie->b;
    std::vector<size_t> drop;
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        auto& t = m.tensors[i];
        const std::string hf_name = t.name;
        bool renamed = false;
        for (const Name& n : table.names) {
            uint64_t layer = 0;
            if (!detail::match(n.hf, t.name, layer)) continue;
            if (layer >= layers->u) throw std::runtime_error("tensor " + t.name + " is past the layer count");
            std::string out = n.gguf;
            const size_t at = out.find("{N}");
            if (at != std::string::npos) out.replace(at, 3, std::to_string(layer));
            t.name = out;
            renamed = true;
            break;
        }
        if (!renamed) throw std::runtime_error("tensor " + t.name + " is not one llmx reads for " + table.model_type);
        if (tied && t.name == table.head) {
            if (m.composites.count(i)) throw std::runtime_error("tensor " + hf_name + " is quantized, and config.json ties the head to the embedding");
            drop.push_back(i);
        }
    }
    m.erase_tensors(drop);
    const gguf::TensorInfo* embedding = nullptr;
    bool head = false;
    for (const auto& t : m.tensors) {
        if (t.name == table.embedding) embedding = &t;
        head |= t.name == table.head;
    }
    if (!tied && !head) throw std::runtime_error("config.json unties the head, and the directory holds no lm_head");
    const jmini::Value* vocab = config.get("vocab_size");
    if (!embedding || embedding->ne.size() != 2 || !vocab || !vocab->isNumber() || vocab->num != double(embedding->ne[1]))
        throw std::runtime_error("config.json vocab_size does not give the embedding's rows");
    kv.insert(kv.end(), std::make_move_iterator(m.kv.begin()), std::make_move_iterator(m.kv.end()));
    m.kv = std::move(kv);
}

} // namespace infer::hf
