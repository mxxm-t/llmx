#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/bf16.hpp"
#include "core/fp16.hpp"
#include "core/json.hpp"
#include "format/gguf.hpp"
#include "format/hf_quant.hpp"

// A Hugging Face directory's config.json and tensor names turned into the metadata and names an architecture's GGUF reader takes, by one table per architecture module.
// Strict: a config key or a tensor the table does not name is refused, so a checkpoint the module does not run is never read as one it does.

namespace infer::hf {

enum class Rule {
    integer,   // a whole number from 1 up to 2^32 - 1, stored as UINT32
    real,      // a finite positive number, stored as FLOAT32
    check,     // checked against `allowed` and not stored
    each,      // an array whose every entry is checked as `check` checks a value, not stored
    rope_scaling,  // null, or YaRN's parameters stored as the rope.scaling keys the GGUF converter writes
};

struct Key {
    const char* hf;
    const char* gguf;      // the key under the architecture's prefix, for integer and real
    Rule rule;
    const char* allowed;   // for check: the values accepted, as JSON texts separated by '|', or null for any value
};

// A tensor's form made at load from its HF values, as the converter writes it; a composite of this kind (gguf::GGUFModel::Composite) is made by `transform`.
// The kinds follow hf_quant's (format/hf_quant.hpp), whose repack makes the others.
enum Transform : uint32_t {
    none = 0,
    plus_one = 16,   // F32 of 1 + w, as HF's norms compute it
    neg_exp = 17,    // F32 of -exp(w), rounded once from double
    widen = 18,      // F32 of w, a 3-D tensor's two slower dimensions made one
    reorder = 19,    // the values moved, their type kept
};
inline bool transforms(uint32_t kind) { return kind >= plus_one && kind <= reorder; }
static_assert(hf_quant::fp8_grid < plus_one, "hf_quant's composite kinds must stay below the transforms'");

// Which of a tensor's rows or columns are indexed by V head, which the converter stores in tiled order: every row, the rows after the 2 Hk Dk of q and k, or every row's columns.
enum class VHeads { none, rows, rows_after_qk, columns };

// "{N}" in both names stands for a layer number below the decoder layer count, and "{L}" in the GGUF name for the MTP layer after them.
// An HF name ending in '*' matches every name it begins, and an empty GGUF name drops the tensor.
struct Name {
    std::string hf;
    std::string gguf;
    Transform transform = none;
    VHeads heads = VHeads::none;
};

using Meta = std::vector<std::pair<std::string, gguf::MetaValue>>;

struct Table {
    const char* model_type;     // config.json model_type
    const char* class_name;     // config.json architectures' one entry
    std::vector<Key> keys;
    std::vector<Name> names;
    const char* embedding;      // the GGUF names of the token embedding and the head, which the tie and vocab_size checks read
    const char* head;
    const char* config = nullptr;     // the object of config.json `keys` read, or null for config.json itself
    std::vector<Key> outer{};         // with `config`: config.json's own keys beside it
    void (*derive)(const jmini::Value& config, Meta& kv, const std::string& prefix) = nullptr;   // the keys no single config value gives
    const char* k_heads = nullptr;    // for VHeads: the GGUF keys of Hk, Hv and Dk under the prefix
    const char* v_heads = nullptr;
    const char* k_width = nullptr;
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

// rope_scaling under the prefix `p`: null for the unscaled rope, or YaRN's object, its original length max_position_embeddings where it gives none, as transformers reads it.
// HF's keys for an attention factor other than 0.1 ln(factor) + 1 are refused by name, as is any other key or type.
inline void rope_scaling(const jmini::Value& v, const jmini::Value& config, const std::string& p, std::vector<std::pair<std::string, gguf::MetaValue>>& kv) {
    if (v.t == jmini::Value::T::Null) return;
    if (!v.isObject()) throw std::runtime_error("config.json rope_scaling is not an object or null");
    gguf::MetaValue type;
    type.vtype = gguf::V_STRING;
    type.s = "yarn";
    kv.emplace_back(p + "rope.scaling.type", type);
    std::set<std::string> seen;
    for (const auto& [key, value] : v.obj) {
        const std::string what = "rope_scaling." + key;
        if (!seen.insert(key).second) throw std::runtime_error("config.json repeats key " + what);
        if (key == "rope_type" || key == "type") {
            if (!value.isString() || value.str != "yarn") throw std::runtime_error("config.json " + what + " = " + text(value) + " is not supported (\"yarn\")");
        } else if (key == "factor") {
            kv.emplace_back(p + "rope.scaling.factor", meta(value, Rule::real, what));
        } else if (key == "original_max_position_embeddings") {
            kv.emplace_back(p + "rope.scaling.original_context_length", meta(value, Rule::integer, what));
        } else if (key == "beta_fast" || key == "beta_slow") {
            kv.emplace_back(p + "rope.scaling.yarn_" + key, meta(value, Rule::real, what));
        } else if (key == "attention_factor" || key == "mscale" || key == "mscale_all_dim") {
            throw std::runtime_error("config.json " + what + " is not supported");
        } else {
            throw std::runtime_error("config.json key " + what + " is not one llmx reads");
        }
    }
    if (!seen.count("rope_type") && !seen.count("type")) throw std::runtime_error("config.json rope_scaling gives no rope_type");
    if (!seen.count("factor")) throw std::runtime_error("config.json rope_scaling gives no factor");
    const jmini::Value* longest = config.get("max_position_embeddings");
    if (!seen.count("original_max_position_embeddings") && longest)
        kv.emplace_back(p + "rope.scaling.original_context_length", meta(*longest, Rule::integer, "max_position_embeddings"));
}

// The layer number `name` gives where `pattern` has "{N}", if it matches.
inline bool match(const std::string& pattern, const std::string& name, uint64_t& layer) {
    if (!pattern.empty() && pattern.back() == '*') return name.compare(0, pattern.size() - 1, pattern, 0, pattern.size() - 1) == 0;
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

// The keys of `obj` stored as `keys` give them; a key `keys` does not list, but for those in `skip`, is refused by name.
inline void read_keys(const std::vector<Key>& keys, const jmini::Value& obj, const std::vector<std::string>& skip, const std::string& arch,
                      const char* model_type, Meta& kv) {
    std::set<std::string> seen;
    for (const auto& [key, value] : obj.obj) {
        if (!seen.insert(key).second) throw std::runtime_error("config.json repeats key " + key);
        if (std::find(skip.begin(), skip.end(), key) != skip.end()) continue;
        bool known = false;
        for (const Key& k : keys) {
            if (key != k.hf) continue;
            known = true;
            if (k.rule == Rule::check) {
                check(k, key, value);
            } else if (k.rule == Rule::rope_scaling) {
                rope_scaling(value, obj, arch + ".", kv);
            } else if (k.rule == Rule::each) {
                if (!value.isArray()) throw std::runtime_error("config.json " + key + " is not an array");
                for (const jmini::Value& entry : value.arr) check(k, key, entry);
            } else {
                kv.emplace_back(arch + "." + k.gguf, meta(value, k.rule, key));
            }
        }
        if (!known) throw std::runtime_error("config.json key " + key + " is not one llmx reads for " + model_type);
    }
}

// The tiled order of `units` rows or columns, those from `offset` on Hv heads of equal width: unit n holds HF's unit order[n], so GGUF V head j = s Hk + h holds HF's V head h r + s.
inline std::vector<uint32_t> tiled(uint64_t units, uint64_t offset, uint64_t hk, uint64_t hv, const std::string& name) {
    if (!hk || !hv || hv % hk || offset >= units || (units - offset) % hv || units > UINT32_MAX)
        throw std::runtime_error("tensor " + name + " does not hold whole V heads");
    const uint64_t width = (units - offset) / hv, r = hv / hk;
    std::vector<uint32_t> order(static_cast<size_t>(units));
    for (uint64_t n = 0; n < units; ++n) {
        const uint64_t j = n < offset ? 0 : (n - offset) / width;
        order[size_t(n)] = uint32_t(n < offset ? n : offset + ((j % hk) * r + j / hk) * width + (n - offset) % width);
    }
    return order;
}

// The order of a weight's scale units of `unit` rows or columns, from the weight's own order of single ones, where every unit moves whole.
inline std::vector<uint32_t> unit_order(const std::vector<uint32_t>& order, uint64_t unit, const std::string& name) {
    const size_t n = order.size(), units = size_t((n + unit - 1) / unit);
    std::vector<uint32_t> out(units);
    std::vector<bool> seen(units);
    for (size_t u = 0; u < units; ++u) {
        const uint64_t first = order[u * unit], to = first / unit;
        // A partial last unit stays last, so no whole unit moves onto its short tail.
        bool whole = first % unit == 0 && to < units && !seen[size_t(to)] && (u + 1 == units) == (to + 1 == units);
        for (uint64_t k = 1; whole && k < unit && u * unit + k < n; ++k) whole = order[size_t(u * unit + k)] == first + k;
        if (!whole) throw std::runtime_error("FP8 tensor " + name + " moves its V heads across its scale blocks of " + std::to_string(unit) + ", which llmx does not regrid");
        seen[size_t(to)] = true;
        out[u] = uint32_t(to);
    }
    return out;
}

// Each FP8 grid composite whose weight moves by V head takes the order of its scales: a scale a row moves with its row, a block of 128 rows or inputs moves whole, and one scale for the tensor stays.
inline void follow_weights(gguf::GGUFModel& m) {
    std::map<std::string, size_t> names;
    for (size_t i = 0; i < m.tensors.size(); ++i) names.emplace(m.tensors[i].name, i);
    for (auto& [i, c] : m.composites) {
        if (c.kind != hf_quant::fp8_grid) continue;
        const std::string& g = m.tensors[i].name;
        const auto w = names.find(hf_quant::fp8_weight_of(g));
        if (w == names.end()) throw std::logic_error("hf: FP8 grid " + g + " has no weight");
        const auto wc = m.composites.find(w->second);
        if (wc == m.composites.end() || wc->second.kind != reorder || c.parts[0].first.n_elements() == 1) continue;
        const gguf::TensorInfo& weight = wc->second.parts[0].first;
        const auto shift = gguf::fp8_scale_shift(weight.ne, m.tensors[i].ne);
        if (!shift) throw std::logic_error("hf: FP8 grid " + g + " does not pair with its weight");
        if (wc->second.columns && *shift == 0) continue;   // a scale a row covers every input of its row
        c.columns = wc->second.columns;
        c.order = unit_order(wc->second.order, c.columns ? gguf::kFp8ScaleInputs : uint64_t(1) << *shift, weight.name);
    }
}

} // namespace detail

// Composite `c`'s bytes, tensor `t`, at `out` from its one part's bytes: each value of the part taken at its place in `c.order`, then `c.kind` applied (Transform).
inline void transform(const gguf::GGUFModel::Composite& c, const gguf::TensorInfo& t, const std::vector<const uint8_t*>& parts, uint8_t* out) {
    if (!transforms(c.kind) || parts.size() != 1 || c.parts.size() != 1) throw std::logic_error("hf: unknown transform of " + t.name);
    const gguf::TensorInfo& in = c.parts[0].first;
    const size_t n = size_t(in.n_elements()), width = in.ne.size() > 1 ? size_t(in.ne[0]) : 1, rows = n / width;
    const size_t size = in.type == quant::GGML_TYPE_F32 ? 4 : in.type == quant::GGML_TYPE_F8_E4M3 ? 1 : 2;
    const uint8_t* src = parts[0];
    if (c.kind == reorder) return gguf::permute(src, out, width, rows, size, c.order, c.columns);
    for (size_t r = 0; r < rows; ++r)
        for (size_t w = 0; w < width; ++w) {
            const size_t from = c.order.empty() ? r * width + w : c.columns ? r * width + c.order[w] : size_t(c.order[r]) * width + w;
            const size_t to = r * width + w;
            float v;
            if (size == 4) {
                std::memcpy(&v, src + from * 4, 4);
            } else {
                uint16_t h;
                std::memcpy(&h, src + from * 2, 2);
                v = in.type == quant::GGML_TYPE_BF16 ? bf16_to_f32(h) : f16_to_f32(h);
            }
            const float f = c.kind == plus_one ? 1.0f + v : c.kind == neg_exp ? -float(std::exp(double(v))) : v;
            std::memcpy(out + to * 4, &f, 4);
        }
}

// The object of `config` that `table`'s keys read: config.json itself, or the one it nests them in.
inline const jmini::Value& keys_of(const Table& table, const jmini::Value& config) {
    if (!table.config) return config;
    const jmini::Value* inner = config.isObject() ? config.get(table.config) : nullptr;
    if (!inner || !inner->isObject()) throw std::runtime_error(std::string("config.json gives no ") + table.config + " object");
    return *inner;
}

// `m`, the tensors of a directory under their HF names, given the metadata and names `table` makes of `config` for the architecture `arch`, with the table's keys under its prefix ahead of those `m` already holds, such as hf_quant's.
// A tied head and a tensor the table drops are not read, and one the converter stores in another form is made at load (Transform).
inline void apply(const Table& table, const std::string& arch, const jmini::Value& config, gguf::GGUFModel& m) {
    if (!config.isObject()) throw std::runtime_error("config.json is not an object");
    const jmini::Value* classes = config.get("architectures");
    if (!classes || !classes->isArray() || classes->arr.size() != 1 || !classes->arr[0].isString() ||
        classes->arr[0].str != table.class_name)
        throw std::runtime_error(std::string("config.json architectures must be exactly [\"") + table.class_name + "\"]");
    const jmini::Value* type = config.get("model_type");
    if (!type || !type->isString() || type->str != table.model_type)
        throw std::runtime_error(std::string("config.json model_type must be ") + table.model_type);

    Meta kv;
    gguf::MetaValue name;
    name.vtype = gguf::V_STRING;
    name.s = arch;
    kv.emplace_back("general.architecture", name);
    const jmini::Value& keys = keys_of(table, config);
    if (table.config) detail::read_keys(table.outer, config, {"architectures", "model_type", table.config}, arch, table.model_type, kv);
    detail::read_keys(table.keys, keys, table.config ? std::vector<std::string>{} : std::vector<std::string>{"architectures", "model_type"}, arch,
                      table.model_type, kv);
    if (table.derive) table.derive(keys, kv, arch + ".");
    const auto find = [&](const std::string& key) -> const gguf::MetaValue* {
        for (const auto& item : kv)
            if (item.first == key) return &item.second;
        return nullptr;
    };
    const gguf::MetaValue* blocks = find(arch + ".block_count");
    if (!blocks) throw std::runtime_error("config.json gives no layer count");
    const gguf::MetaValue* nextn = find(arch + ".nextn_predict_layers");
    const uint64_t layers = blocks->u - (nextn ? nextn->u : 0);
    const auto count = [&](const char* key) -> uint64_t {
        const gguf::MetaValue* v = key ? find(arch + "." + key) : nullptr;
        if (!v) throw std::runtime_error(std::string("config.json gives no value for ") + arch + "." + (key ? key : "the V heads"));
        return v->u;
    };

    // Tensors: renamed, a layer past the count refused, a dropped one or a tied head dropped, one stored in another form made at load.
    // With nested keys the top level's tie_word_embeddings, which HF's outer model reads, holds where the nested object gives none.
    const jmini::Value* inner_tie = keys.get("tie_word_embeddings");
    const jmini::Value* outer_tie = table.config ? config.get("tie_word_embeddings") : nullptr;
    if (inner_tie && outer_tie && detail::text(*inner_tie) != detail::text(*outer_tie))
        throw std::runtime_error(std::string("config.json tie_word_embeddings differs from ") + table.config + "'s");
    const jmini::Value* tie = inner_tie ? inner_tie : outer_tie;
    const bool tied = tie && tie->t == jmini::Value::T::Bool && tie->b;
    std::vector<size_t> drop;
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        auto& t = m.tensors[i];
        const std::string hf_name = t.name;
        // An FP8 linear's scale grid, which hf_quant made `X.scale` beside `X.weight`, takes its weight's name with `.scale` for `.weight`.
        const bool grid = m.composites.count(i) && hf_name.size() > 6 && hf_name.compare(hf_name.size() - 6, 6, ".scale") == 0;
        const std::string looked = grid ? hf_quant::fp8_weight_of(hf_name) : hf_name;
        const Name* row = nullptr;
        for (const Name& n : table.names) {
            uint64_t layer = 0;
            if (!detail::match(n.hf, looked, layer)) continue;
            if (layer >= layers) throw std::runtime_error("tensor " + t.name + " is past the layer count");
            std::string out = n.gguf;
            if (const size_t at = out.find("{N}"); at != std::string::npos) out.replace(at, 3, std::to_string(layer));
            if (const size_t at = out.find("{L}"); at != std::string::npos) {
                if (!nextn) throw std::runtime_error("tensor " + t.name + " is an MTP tensor, and config.json gives no MTP layer");
                out.replace(at, 3, std::to_string(layers));
            }
            if (grid) out = hf_quant::fp8_grid_of(out);
            t.name = out;
            row = &n;
            break;
        }
        if (!row) throw std::runtime_error("tensor " + t.name + " is not one llmx reads for " + table.model_type);
        if (row->gguf.empty() || (tied && t.name == table.head)) {
            if (m.composites.count(i))
                throw std::runtime_error("tensor " + hf_name + " is quantized, and " + (row->gguf.empty() ? std::string("llmx does not read it") : "config.json ties the head to the embedding"));
            drop.push_back(i);
            continue;
        }
        if (t.ne.size() > 4) throw std::runtime_error("tensor " + hf_name + " has " + std::to_string(t.ne.size()) + " dimensions, more than a GGUF tensor's 4");
        if (grid) continue;   // a grid takes its weight's order once every weight has one (detail::follow_weights)
        std::vector<uint32_t> order;
        if (row->heads != VHeads::none) {
            const uint64_t hk = count(table.k_heads), hv = count(table.v_heads);
            const uint64_t width = t.ne.size() > 1 ? t.ne[0] : 1;
            if (!width || !t.n_elements()) throw std::runtime_error("tensor " + hf_name + " does not hold whole V heads");
            const uint64_t units = row->heads == VHeads::columns ? width : uint64_t(t.n_elements()) / width;
            if (hv != hk) order = detail::tiled(units, row->heads == VHeads::rows_after_qk ? 2 * hk * count(table.k_width) : 0, hk, hv, hf_name);
        }
        // A 1-D tensor that only moves is widened as it moves, as the loader widens every other 16-bit vector.
        const Transform kind = row->transform != none ? row->transform : order.empty() ? none : t.ne.size() == 1 ? widen : reorder;
        if (kind == none) continue;
        if (m.composites.count(i)) throw std::runtime_error("tensor " + hf_name + " is quantized, which llmx reads only in the form the directory stores");
        const bool fp8 = t.type == quant::GGML_TYPE_F8_E4M3;
        if (fp8 && kind != reorder) throw std::runtime_error("tensor " + hf_name + " holds F8_E4M3 codes, which llmx moves but does not transform");
        if (!fp8 && t.type != quant::GGML_TYPE_F32 && t.type != quant::GGML_TYPE_BF16 && t.type != quant::GGML_TYPE_F16)
            throw std::runtime_error("tensor " + hf_name + " is not F32, BF16, F16 or F8_E4M3, the types llmx transforms");
        gguf::GGUFModel::Composite c;
        c.kind = kind;
        c.parts.emplace_back(t, m.offsets[i]);
        c.order = std::move(order);
        c.columns = row->heads == VHeads::columns;
        if (kind != reorder) t.type = quant::GGML_TYPE_F32;
        if (kind == widen && t.ne.size() == 3) t.ne = {t.ne[0], t.ne[1] * t.ne[2]};
        m.composites[i] = std::move(c);
    }
    m.erase_tensors(drop);
    detail::follow_weights(m);
    const gguf::TensorInfo* embedding = nullptr;
    bool head = false;
    for (const auto& t : m.tensors) {
        if (t.name == table.embedding) embedding = &t;
        head |= t.name == table.head;
    }
    if (!tied && !head) throw std::runtime_error("config.json unties the head, and the directory holds no lm_head");
    const jmini::Value* vocab = keys.get("vocab_size");
    if (!embedding || embedding->ne.size() != 2 || !vocab || !vocab->isNumber() || vocab->num != double(embedding->ne[1]))
        throw std::runtime_error("config.json vocab_size does not give the embedding's rows");
    kv.insert(kv.end(), std::make_move_iterator(m.kv.begin()), std::make_move_iterator(m.kv.end()));
    m.kv = std::move(kv);
}

} // namespace infer::hf
