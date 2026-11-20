#pragma once
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/bf16.hpp"
#include "core/fp16.hpp"
#include "core/json.hpp"
#include "format/gguf.hpp"
#include "format/weight_scales.hpp"
#include "quant/quant.hpp"

// Quantized Hugging Face checkpoints: config.json's quantization_config read strictly, and each quantized linear's companion tensors made one tensor of an existing type whose values are the checkpoint's exactly (docs/src/format-hf_quant.md).
// The tensor is a gguf::GGUFModel::Composite; the loader makes its bytes with repack.

namespace hf_quant {

// What a repack writes; the disk tier's identity carries it, so a change to any kind's bytes moves it.
constexpr uint32_t kRepackVersion = 3;

enum Kind : uint32_t { awq_q5_0 = 1, gptq_q4_0 = 2, gptq_q5_0 = 3, gptq_q8_0 = 4, ct_q4_0 = 5, ct_q5_0 = 6, ct_q8_0 = 7, fp8_grid = gguf::kFp8GridComposite };

namespace detail {

inline bool ends_with(const std::string& s, const std::string& tail) {
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

// A group_size a repack into blocks of 32 can share a scale over.
inline uint32_t group_of(const jmini::Value& v, const std::string& what, const char* method) {
    if (!v.isNumber() || !v.whole || v.num < 32 || v.num > 65536 || uint32_t(v.num) % 32)
        throw std::runtime_error(what + ": llmx reads " + method + " groups of a positive multiple of 32 inputs");
    return uint32_t(v.num);
}

// AWQ GEMM's quantization_config: each key checked against what the repack reads, refused when it would change the values, and taken as it is when it names only how a runtime fuses or schedules kernels.
inline uint32_t awq_group(const jmini::Value& qc) {
    uint32_t group = 0;
    for (const auto& [key, v] : qc.obj) {
        const std::string what = "quantization_config " + key;
        if (key == "quant_method") continue;
        if (key == "bits") {
            if (!v.isNumber() || v.num != 4) throw std::runtime_error(what + ": llmx reads 4-bit AWQ only");
        } else if (key == "group_size") {
            group = group_of(v, what, "AWQ");
        } else if (key == "zero_point") {
            if (v.t != jmini::Value::T::Bool || !v.b) throw std::runtime_error(what + ": llmx reads AWQ with zero points only");
        } else if (key == "version") {
            std::string version = v.isString() ? v.str : std::string();
            std::transform(version.begin(), version.end(), version.begin(), [](unsigned char ch) { return char(std::tolower(ch)); });
            if (version != "gemm") throw std::runtime_error(what + ": llmx reads AWQ's gemm packing only");
        } else if (key == "backend") {
            if (!v.isString() || v.str != "autoawq") throw std::runtime_error(what + " is not one llmx reads");
        } else if (key == "modules_to_not_convert") {
            if (v.isArray()) {
                for (const auto& item : v.arr)
                    if (!item.isString()) throw std::runtime_error(what + " is not a list of module names");
            } else if (v.t != jmini::Value::T::Null) {
                throw std::runtime_error(what + " is not a list of module names");
            }
        } else if (key != "do_fuse" && key != "fuse_max_seq_len" && key != "modules_to_fuse" && key != "exllama_config") {
            throw std::runtime_error(what + " is not one llmx reads for awq");
        }
    }
    if (!group) throw std::runtime_error("quantization_config gives no group_size");
    return group;
}

struct Gptq {
    uint32_t bits = 0, group = 0;
    bool sym = true;  // transformers' GPTQConfig default
    bool v1 = true;   // checkpoint_format gptq, which stores each zero point less one
};

// GPTQ's quantization_config, read as AWQ's is; desc_act and static_groups are taken as they are, since g_idx itself decides whether a linear's groups follow its inputs.
inline Gptq gptq_config(const jmini::Value& qc) {
    // Keys that name how the checkpoint was quantized or how a runtime schedules kernels, and change no stored value.
    static const char* const taken[] = {"damp_percent", "damp_auto_increment", "true_sequential", "static_groups", "desc_act", "hyb_act", "mse", "meta",
                                        "use_exllama", "exllama_config", "use_cuda_fp16", "backend", "cache_block_outputs", "model_seqlen", "batch_size",
                                        "max_input_length", "dataset", "tokenizer", "pad_token_id", "block_name_to_quantize", "module_name_preceding_first_block",
                                        "modules_in_block_to_quantize", "model_name_or_path", "model_file_base_name", "quantizer"};
    Gptq q;
    for (const auto& [key, v] : qc.obj) {
        const std::string what = "quantization_config " + key;
        if (key == "quant_method") continue;
        if (key == "bits") {
            if (!v.isNumber() || (v.num != 4 && v.num != 8)) throw std::runtime_error(what + ": llmx reads 4- and 8-bit GPTQ only");
            q.bits = uint32_t(v.num);
        } else if (key == "group_size") {
            q.group = group_of(v, what, "GPTQ");
        } else if (key == "sym") {
            if (v.t != jmini::Value::T::Bool) throw std::runtime_error(what + " is not true or false");
            q.sym = v.b;
        } else if (key == "checkpoint_format") {
            if (!v.isString() || (v.str != "gptq" && v.str != "gptq_v2")) throw std::runtime_error(what + ": llmx reads GPTQ's gptq and gptq_v2 formats only");
            q.v1 = v.str == "gptq";
        } else if (key == "lm_head") {
            if (v.t != jmini::Value::T::Bool || v.b) throw std::runtime_error(what + ": llmx reads GPTQ with an unquantized lm_head only");
        } else if (key == "pack_dtype") {
            if (!v.isString() || v.str != "int32") throw std::runtime_error(what + ": llmx reads GPTQ packed in int32 only");
        } else if (std::find_if(std::begin(taken), std::end(taken), [&](const char* k) { return key == k; }) == std::end(taken)) {
            throw std::runtime_error(what + " is not one llmx reads for gptq");
        }
    }
    if (!q.bits) throw std::runtime_error("quantization_config gives no bits");
    if (!q.group) throw std::runtime_error("quantization_config gives no group_size");
    if (q.bits == 8 && !q.sym) throw std::runtime_error("quantization_config sym: llmx reads 8-bit GPTQ with symmetric zero points only");
    return q;
}

// Each `X.<main>` of `m` that holds integers, or F8_E4M3 codes under `fp8`, and its companions `X.<suffix>`, the first `required` of them required: made one composite `X.weight` at the main's place, the companions leaving the table.
// Under `fp8` the main stays as it is and the composite is its scale grid `X.scale` at the first companion's place, made from that companion alone; `make` checks the parts and sets the composite's kind and the tensor's type and shape.
template <class Make> void linears(gguf::GGUFModel& m, const std::string& main, const std::vector<std::string>& companions, size_t required, Make make, bool fp8 = false) {
    std::map<std::string, size_t> index;
    for (size_t i = 0; i < m.tensors.size(); ++i) index[m.tensors[i].name] = i;
    std::vector<size_t> drop;
    std::vector<bool> coded(m.tensors.size());
    for (size_t i = 0; i < m.tensors.size(); ++i)
        coded[i] = fp8 ? m.tensors[i].type == quant::GGML_TYPE_F8_E4M3 : m.tensors[i].type == quant::GGML_TYPE_I32 || m.tensors[i].type == quant::GGML_TYPE_I8;
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        gguf::TensorInfo& t = m.tensors[i];
        const auto companion = std::find_if(companions.begin(), companions.end(), [&](const std::string& s) { return ends_with(t.name, "." + s); });
        if (companion != companions.end()) {
            const auto beside = index.find(t.name.substr(0, t.name.size() - companion->size()) + main);
            if (beside == index.end()) throw std::runtime_error("tensor " + t.name + " has no " + main + " beside it");
            if (!coded[beside->second])
                throw std::runtime_error("tensor " + t.name + " has no " + (fp8 ? "F8_E4M3 " : "integer ") + main + " beside it");
            continue;
        }
        if (!ends_with(t.name, "." + main) || !coded[i]) continue;
        const std::string base = t.name.substr(0, t.name.size() - main.size() - 1);
        gguf::GGUFModel::Composite c;
        if (!fp8) c.parts = {{t, m.offsets[i]}};
        size_t at = i;
        for (size_t j = 0; j < companions.size(); ++j) {
            const auto found = index.find(base + "." + companions[j]);
            if (found == index.end()) {
                if (j < required) throw std::runtime_error("tensor " + t.name + " lacks its " + companions[j]);
                continue;
            }
            if (fp8 && j == 0) {
                at = found->second;
                c.parts.push_back({m.tensors[at], m.offsets[at]});
                continue;
            }
            if (!fp8) c.parts.push_back({m.tensors[found->second], m.offsets[found->second]});
            drop.push_back(found->second);
        }
        gguf::TensorInfo& made = m.tensors[at];
        if (fp8) made = t;
        make(base, c, made);
        made.name = base + (fp8 ? ".scale" : ".weight");
        m.composites[at] = std::move(c);
    }
    std::sort(drop.begin(), drop.end());
    m.erase_tensors(drop);
}

inline bool scales_type(const gguf::TensorInfo& s) { return s.type == quant::GGML_TYPE_F16 || s.type == quant::GGML_TYPE_BF16; }

// AWQ linears: qweight [in, out / 8], qzeros [in / group, out / 8] and scales [in / group, out] (HF order) make one Q5_0 tensor [in, out].
inline void awq(uint32_t group, gguf::GGUFModel& m) {
    linears(m, "qweight", {"qzeros", "scales"}, 2, [&](const std::string& base, gguf::GGUFModel::Composite& c, gguf::TensorInfo& t) {
        const gguf::TensorInfo &z = c.parts[1].first, &s = c.parts[2].first;
        if (t.type != quant::GGML_TYPE_I32 || z.type != quant::GGML_TYPE_I32 || !scales_type(s))
            throw std::runtime_error("AWQ linear " + base + " is not I32 qweight and qzeros with 16-bit scales");
        if (t.ne.size() != 2 || z.ne.size() != 2 || s.ne.size() != 2) throw std::runtime_error("AWQ linear " + base + " is not a matrix");
        const uint64_t in = t.ne[1], out = s.ne[0];
        if (!in || !out || in % group || t.ne[0] * 8 != out || z.ne[0] * 8 != out || z.ne[1] * group != in || s.ne[1] * group != in)
            throw std::runtime_error("AWQ linear " + base + " has shapes that do not match a group of " + std::to_string(group));
        c.kind = awq_q5_0;
        c.group = group;
        t.type = quant::GGML_TYPE_Q5_0;
        t.ne = {in, out};
    });
}

// GPTQ linears: qweight [in * bits / 32, out] packed along the inputs, qzeros [in / group, out * bits / 32] packed along the outputs, scales [in / group, out] and, when present, g_idx [in] make one tensor [in, out].
// It is Q4_0 for 4-bit symmetric zero points, Q5_0 for other 4-bit ones and Q8_0 for 8-bit symmetric ones.
inline void gptq(const Gptq& q, gguf::GGUFModel& m) {
    const uint64_t per = 32 / q.bits;
    linears(m, "qweight", {"qzeros", "scales", "g_idx"}, 2, [&](const std::string& base, gguf::GGUFModel::Composite& c, gguf::TensorInfo& t) {
        const gguf::TensorInfo &z = c.parts[1].first, &s = c.parts[2].first;
        const bool g_idx = c.parts.size() == 4;
        if (t.type != quant::GGML_TYPE_I32 || z.type != quant::GGML_TYPE_I32 || !scales_type(s) || (g_idx && c.parts[3].first.type != quant::GGML_TYPE_I32))
            throw std::runtime_error("GPTQ linear " + base + " is not I32 qweight, qzeros and g_idx with 16-bit scales");
        if (t.ne.size() != 2 || z.ne.size() != 2 || s.ne.size() != 2 || (g_idx && c.parts[3].first.ne.size() != 1))
            throw std::runtime_error("GPTQ linear " + base + " is not a matrix");
        const uint64_t in = t.ne[1] * per, out = t.ne[0];
        if (!in || !out || in % q.group || out % per || s.ne[0] != out || s.ne[1] * q.group != in || z.ne[0] * per != out || z.ne[1] * q.group != in ||
            (g_idx && c.parts[3].first.ne[0] != in))
            throw std::runtime_error("GPTQ linear " + base + " has shapes that do not match " + std::to_string(q.bits) + "-bit codes in groups of " + std::to_string(q.group));
        c.kind = q.bits == 8 ? gptq_q8_0 : q.sym ? gptq_q4_0 : gptq_q5_0;
        c.group = q.group;
        c.zero_offset = q.v1 ? 1 : 0;
        t.type = q.bits == 8 ? quant::GGML_TYPE_Q8_0 : q.sym ? quant::GGML_TYPE_Q4_0 : quant::GGML_TYPE_Q5_0;
        t.ne = {in, out};
    });
}

// How an FP8 checkpoint's weight scales cover a linear: one scale, one a row, or one a 128 x 128 block.
enum class Fp8Scales { tensor, channel, block };

struct Compressed {
    uint32_t bits = 0, group = 0;  // group 0: one scale a row (the channel strategy)
    bool sym = true, packed = true;
    bool fp8 = false;              // float-quantized: E4M3 codes under scales of `scales`' form
    Fp8Scales scales = Fp8Scales::channel;
    std::string activations;  // the input activations' scheme in words, empty when none; read and not applied
    std::string kv;           // a KV cache scheme in words, empty when none; its k_scale and v_scale are read and not applied
};

// A config group's input_activations in words, for `llmx info`; llmx runs every file's activations by --dtype, not by this scheme.
inline std::string activation_scheme(const jmini::Value& a) {
    const auto text = [&](const char* k) { const jmini::Value* v = a.get(k); return v && v->isString() ? v->str : std::string(); };
    const auto flag = [&](const char* k) { const jmini::Value* v = a.get(k); return v && v->t == jmini::Value::T::Bool && v->b; };
    const jmini::Value* bits = a.get("num_bits");
    std::string s = (bits && bits->isNumber() ? std::to_string(int64_t(bits->num)) + "-bit " : std::string()) + (text("type").empty() ? "given" : text("type"));
    if (!text("strategy").empty()) s += " by " + text("strategy");
    s += flag("dynamic") ? ", dynamic" : ", static";
    s += flag("symmetric") ? ", symmetric" : ", asymmetric";
    return s + "; not applied, activations follow --dtype";
}

// Whether `v` is the block shape [128, 128], the one FP8 block llmx reads.
inline bool square_blocks(const jmini::Value& v) {
    return v.isArray() && v.arr.size() == 2 &&
           std::all_of(v.arr.begin(), v.arr.end(), [](const jmini::Value& x) { return x.isNumber() && x.num == double(gguf::kFp8ScaleInputs); });
}

// One config group's weights scheme of compressed-tensors; actorder is taken as it is, since weight_g_idx itself decides whether a linear's groups follow its inputs.
inline Compressed compressed_scheme(const jmini::Value& w, const std::string& where) {
    if (!w.isObject()) throw std::runtime_error(where + " is not an object");
    static const char* const taken[] = {"actorder", "observer", "observer_kwargs"};
    Compressed q;
    std::string strategy;
    const jmini::Value *group = nullptr, *block = nullptr;
    for (const auto& [key, v] : w.obj) {
        const std::string what = where + " " + key;
        if (key == "num_bits") {
            if (!v.isNumber() || (v.num != 4 && v.num != 8)) throw std::runtime_error(what + ": llmx reads 4- and 8-bit compressed-tensors only");
            q.bits = uint32_t(v.num);
        } else if (key == "type") {
            if (!v.isString() || (v.str != "int" && v.str != "float")) throw std::runtime_error(what + ": llmx reads compressed-tensors int and float weights only");
            q.fp8 = v.str == "float";
        } else if (key == "symmetric") {
            if (v.t != jmini::Value::T::Bool) throw std::runtime_error(what + " is not true or false");
            q.sym = v.b;
        } else if (key == "strategy") {
            if (!v.isString()) throw std::runtime_error(what + " is not a string");
            strategy = v.str;
        } else if (key == "group_size") {
            group = &v;
        } else if (key == "dynamic") {
            if (v.t != jmini::Value::T::Bool || v.b) throw std::runtime_error(what + ": llmx reads compressed-tensors with stored weight scales only");
        } else if (key == "block_structure") {
            block = &v;
        } else if (std::find_if(std::begin(taken), std::end(taken), [&](const char* k) { return key == k; }) == std::end(taken)) {
            throw std::runtime_error(what + " is not one llmx reads for compressed-tensors");
        }
    }
    if (!q.bits) throw std::runtime_error(where + " gives no num_bits");
    if (strategy.empty()) throw std::runtime_error(where + " gives no strategy");
    const bool blocks = block && block->t != jmini::Value::T::Null;
    const bool groups = group && group->t != jmini::Value::T::Null && !(group->isNumber() && group->num == -1);
    if (q.fp8) {
        if (q.bits != 8 || !q.sym) throw std::runtime_error(where + ": llmx reads 8-bit symmetric float weights only");
        if (strategy != "tensor" && strategy != "channel" && strategy != "block")
            throw std::runtime_error(where + " strategy: llmx reads compressed-tensors float weights by tensor, channel or 128 x 128 block only");
        q.scales = strategy == "tensor" ? Fp8Scales::tensor : strategy == "channel" ? Fp8Scales::channel : Fp8Scales::block;
        if (blocks != (q.scales == Fp8Scales::block) || (blocks && !square_blocks(*block)))
            throw std::runtime_error(where + " block_structure: llmx reads float weight blocks of 128 x 128 under the block strategy only");
        if (groups) throw std::runtime_error(where + " group_size: llmx reads float weights without groups");
        return q;
    }
    if (blocks) throw std::runtime_error(where + " block_structure: llmx reads compressed-tensors int weights without weight blocks only");
    if (strategy != "group" && strategy != "channel") throw std::runtime_error(where + " strategy: llmx reads compressed-tensors group and channel scales only");
    if (strategy == "group") {
        if (!group) throw std::runtime_error(where + " gives no group_size");
        q.group = group_of(*group, where + " group_size", "compressed-tensors");
    } else if (groups) {
        throw std::runtime_error(where + " group_size: a channel scheme has no groups");
    }
    if (q.bits == 8 && !q.sym) throw std::runtime_error(where + " symmetric: llmx reads 8-bit compressed-tensors with symmetric zero points only");
    return q;
}

// compressed-tensors' quantization_config, read as AWQ's is: one weights scheme over config groups that each target every linear, activation schemes taken and not applied.
// The keys taken are those llm-compressor's files carry; `ignore` may name only the head, or under float weights any modules, names and `re:` patterns alike, since a linear is FP8 exactly where its weight is (docs/src/format-hf_quant.md).
inline Compressed compressed_config(const jmini::Value& qc) {
    static const char* const taken[] = {"global_compression_ratio", "version"};
    Compressed q;
    bool schemes = false;
    std::string format, activations;
    const jmini::Value *kv = nullptr, *ignore = nullptr;
    for (const auto& [key, v] : qc.obj) {
        const std::string what = "quantization_config " + key;
        if (key == "quant_method") continue;
        if (key == "format") {
            if (!v.isString() || (v.str != "pack-quantized" && v.str != "int-quantized" && v.str != "float-quantized"))
                throw std::runtime_error(what + ": llmx reads compressed-tensors pack-quantized, int-quantized and float-quantized only");
            format = v.str;
        } else if (key == "quantization_status") {
            if (!v.isString() || v.str != "compressed") throw std::runtime_error(what + ": llmx reads compressed checkpoints only");
        } else if (key == "kv_cache_scheme") {
            kv = &v;
        } else if (key == "sparsity_config") {
            if (v.t != jmini::Value::T::Null && !(v.isObject() && v.obj.empty())) throw std::runtime_error(what + ": llmx reads dense compressed-tensors only");
        } else if (key == "ignore") {
            ignore = &v;
        } else if (key == "config_groups") {
            if (!v.isObject()) throw std::runtime_error(what + " is not an object");
            for (const auto& [name, g] : v.obj) {
                const std::string where = what + " " + name;
                if (!g.isObject()) throw std::runtime_error(where + " is not an object");
                if (!g.get("targets")) throw std::runtime_error(where + " gives no targets");
                for (const auto& [part, x] : g.obj) {
                    if (part == "weights") {
                        const Compressed one = compressed_scheme(x, where + " weights");
                        if (schemes && (one.bits != q.bits || one.group != q.group || one.sym != q.sym || one.fp8 != q.fp8 || one.scales != q.scales))
                            throw std::runtime_error(where + " weights differ from another group's; llmx reads one weights scheme");
                        q = one;
                        schemes = true;
                    } else if (part == "output_activations") {
                        if (x.t != jmini::Value::T::Null) throw std::runtime_error(where + " output_activations: llmx reads compressed-tensors with unquantized outputs only");
                    } else if (part == "targets") {
                        if (!x.isArray() || std::none_of(x.arr.begin(), x.arr.end(), [](const jmini::Value& y) { return y.isString() && y.str == "Linear"; }))
                            throw std::runtime_error(where + " targets: llmx reads config groups that target every Linear");
                    } else if (part == "input_activations") {
                        if (x.t != jmini::Value::T::Null) activations = activation_scheme(x);
                    } else {
                        throw std::runtime_error(where + " " + part + " is not one llmx reads for compressed-tensors");
                    }
                }
            }
        } else if (std::find_if(std::begin(taken), std::end(taken), [&](const char* k) { return key == k; }) == std::end(taken)) {
            throw std::runtime_error(what + " is not one llmx reads for compressed-tensors");
        }
    }
    if (format.empty()) throw std::runtime_error("quantization_config gives no format");
    if (!schemes) throw std::runtime_error("quantization_config config_groups give no weights scheme");
    if (q.fp8 != (format == "float-quantized")) throw std::runtime_error("quantization_config format " + format + " does not hold " + (q.fp8 ? "float" : "int") + " weights");
    const bool null_kv = !kv || kv->t == jmini::Value::T::Null;
    if (!null_kv && (!q.fp8 || !kv->isObject()))
        throw std::runtime_error("quantization_config kv_cache_scheme: llmx reads compressed-tensors int weights with an unquantized KV cache only");
    if (!null_kv) q.kv = activation_scheme(*kv).substr(0, activation_scheme(*kv).find(';')) + "; not applied, the KV cache follows --cache-type-k and --cache-type-v";
    if (ignore && ignore->t != jmini::Value::T::Null &&
        !(ignore->isArray() && std::all_of(ignore->arr.begin(), ignore->arr.end(), [&](const jmini::Value& x) { return x.isString() && (q.fp8 || x.str == "lm_head"); })))
        throw std::runtime_error(std::string("quantization_config ignore: llmx reads compressed-tensors ") + (q.fp8 ? "that leave out a list of modules" : "that leave out the head alone"));
    q.packed = format == "pack-quantized";
    q.activations = activations;
    if (!q.packed && !q.sym) throw std::runtime_error("quantization_config format: llmx reads int-quantized compressed-tensors with symmetric codes only");
    return q;
}

// The activation and KV cache schemes a file gives, in words, kept in its metadata as `quantization.input_activations` and `quantization.kv_cache_scheme` for `llmx info`; neither is applied.
inline void schemes(const std::string& activations, const std::string& kv, gguf::GGUFModel& m) {
    for (const auto& [key, text] : {std::pair<const char*, const std::string&>{"quantization.input_activations", activations}, {"quantization.kv_cache_scheme", kv}}) {
        if (text.empty()) continue;
        gguf::MetaValue v;
        v.vtype = gguf::V_STRING;
        v.s = text;
        m.kv.emplace_back(key, v);
    }
}

// An FP8 checkpoint's linears: the companion holding the weight scales, their form, and the activation and KV cache schemes in words (schemes).
struct Fp8 {
    std::string scale;
    Fp8Scales scales = Fp8Scales::block;
    std::string activations, kv;
};

// quant_method fp8's quantization_config: E4M3 codes with `weight_scale_inv` scales (a multiplier despite its name) per 128 x 128 block, or one a tensor without weight_block_size.
inline Fp8 fp8_config(const jmini::Value& qc) {
    Fp8 q{"weight_scale_inv", Fp8Scales::block, "", ""};
    for (const auto& [key, v] : qc.obj) {
        const std::string what = "quantization_config " + key;
        if (key == "quant_method" || key == "dequantize") continue;
        if (key == "fmt") {
            if (!v.isString() || v.str != "e4m3") throw std::runtime_error(what + ": llmx reads E4M3 FP8 only");
        } else if (key == "weight_block_size") {
            if (v.t == jmini::Value::T::Null) q.scales = Fp8Scales::tensor;
            else if (!square_blocks(v)) throw std::runtime_error(what + ": llmx reads FP8 blocks of 128 x 128 only");
        } else if (key == "activation_scheme") {
            if (!v.isString() || (v.str != "dynamic" && v.str != "static")) throw std::runtime_error(what + " is neither dynamic nor static");
            q.activations = "8-bit float by tensor, " + v.str + "; not applied, activations follow --dtype";
        } else if (key == "modules_to_not_convert") {
            if (v.t != jmini::Value::T::Null && !(v.isArray() && std::all_of(v.arr.begin(), v.arr.end(), [](const jmini::Value& x) { return x.isString(); })))
                throw std::runtime_error(what + " is not a list of module names");
        } else {
            throw std::runtime_error(what + " is not one llmx reads for fp8");
        }
    }
    return q;
}

// FP8 linears: an F8_E4M3 weight [in, out] and its scales of BF16, F16 or F32, one a tensor, [1, out] by channel or [in / 128, ceil(out / 128)] by block, made the F32 grid `X.scale` (gguf::fp8_grid).
// A static `input_scale` and a KV cache's `k_scale` and `v_scale` leave the table unapplied; the modules listed to leave out need no match, since a linear is FP8 exactly where its weight is.
inline void fp8(const Fp8& q, gguf::GGUFModel& m) {
    const char* form = q.scales == Fp8Scales::tensor ? "tensor" : q.scales == Fp8Scales::channel ? "channel" : "block";
    linears(m, "weight", {q.scale, "input_scale"}, 1, [&](const std::string& base, gguf::GGUFModel::Composite& c, gguf::TensorInfo& t) {
        const gguf::TensorInfo& s = c.parts[0].first;
        if (s.type != quant::GGML_TYPE_F32 && s.type != quant::GGML_TYPE_F16 && s.type != quant::GGML_TYPE_BF16)
            throw std::runtime_error("tensor " + s.name + " holds " + quant::storage_type(s.type)->name + " scales; llmx reads BF16, F16 and F32 FP8 scales");
        if (t.ne.size() != 2 || !t.ne[0] || !t.ne[1] || t.ne[0] % gguf::kFp8ScaleInputs) throw std::runtime_error("FP8 linear " + base + " is not a matrix of whole 128-input blocks");
        const uint64_t in = t.ne[0], out = t.ne[1];
        const bool fits = q.scales == Fp8Scales::tensor    ? s.n_elements() == 1 && s.ne.size() <= 1
                          : q.scales == Fp8Scales::channel ? s.ne == std::vector<uint64_t>{1, out}
                                                           : s.ne == gguf::fp8_grid_shape(t.ne, 7);
        if (!fits) throw std::runtime_error("FP8 linear " + base + " has scales that do not match one " + (q.scales == Fp8Scales::tensor ? "scale" : std::string("scale a ") + form) +
                                            " over " + std::to_string(in) + " inputs and " + std::to_string(out) + " outputs");
        c.kind = fp8_grid;
        t.type = quant::GGML_TYPE_F32;
        t.ne = gguf::fp8_grid_shape(t.ne, q.scales == Fp8Scales::channel ? 0 : 7);
    }, true);
    if (!q.kv.empty()) {
        std::vector<size_t> drop;
        for (size_t i = 0; i < m.tensors.size(); ++i)
            if (ends_with(m.tensors[i].name, ".k_scale") || ends_with(m.tensors[i].name, ".v_scale")) drop.push_back(i);
        m.erase_tensors(drop);
    }
    schemes(q.activations, q.kv, m);
}

// The part of composite `c` whose name ends in `.suffix`, or -1.
inline int part_of(const gguf::GGUFModel::Composite& c, const std::string& suffix) {
    for (size_t j = 0; j < c.parts.size(); ++j)
        if (ends_with(c.parts[j].first.name, "." + suffix)) return int(j);
    return -1;
}

// compressed-tensors linears: weight_packed or int-quantized's I8 weight, weight_scale, weight_zero_point when asymmetric, weight_shape and weight_g_idx make one tensor [in, out] (shapes in docs/src/format-hf_quant.md).
// It is Q4_0 for 4-bit symmetric codes, Q5_0 for 4-bit asymmetric ones and Q8_0 for 8-bit ones.
inline void compressed(const Compressed& q, gguf::GGUFModel& m) {
    if (q.fp8) {
        fp8(Fp8{"weight_scale", q.scales, q.activations, q.kv}, m);
        return;
    }
    const uint64_t per = 32 / q.bits;
    linears(m, q.packed ? "weight_packed" : "weight", {"weight_scale", "weight_zero_point", "weight_shape", "weight_g_idx"}, 1,
            [&](const std::string& base, gguf::GGUFModel::Composite& c, gguf::TensorInfo& t) {
        const int zero = part_of(c, "weight_zero_point"), shape = part_of(c, "weight_shape"), g_idx = part_of(c, "weight_g_idx");
        const gguf::TensorInfo& s = c.parts[1].first;
        const auto typed = [&](int j, uint32_t type) { return j < 0 || c.parts[size_t(j)].first.type == type; };
        if (t.type != (q.packed ? quant::GGML_TYPE_I32 : quant::GGML_TYPE_I8) || !scales_type(s) || !typed(zero, quant::GGML_TYPE_I32) || !typed(shape, quant::GGML_TYPE_I64) ||
            !typed(g_idx, quant::GGML_TYPE_I32))
            throw std::runtime_error("compressed-tensors linear " + base + (q.packed ? " is not I32 weight_packed" : " is not I8 weight") +
                                     " with 16-bit scales, I32 zero points and g_idx and an I64 weight_shape");
        if ((zero >= 0) == q.sym) throw std::runtime_error("compressed-tensors linear " + base + (q.sym ? " has zero points, and its scheme is symmetric" : " lacks its weight_zero_point"));
        if (q.packed && shape < 0) throw std::runtime_error("compressed-tensors linear " + base + " lacks its weight_shape");
        const auto ne = [&](int j) { return j < 0 ? std::vector<uint64_t>{} : c.parts[size_t(j)].first.ne; };
        const uint64_t in = t.ne.empty() ? 0 : q.packed ? t.ne[0] * per : t.ne[0], out = t.ne.size() == 2 ? t.ne[1] : 0, width = q.group ? q.group : in;
        const uint64_t groups = width ? in / width : 0;
        if (t.ne.size() != 2 || !in || !out || in % 32 || in % width || s.ne != std::vector<uint64_t>{groups, out} || (zero >= 0 && (out % per || ne(zero) != std::vector<uint64_t>{groups, out / per})) ||
            (shape >= 0 && ne(shape) != std::vector<uint64_t>{2}) || (g_idx >= 0 && ne(g_idx) != std::vector<uint64_t>{in}))
            throw std::runtime_error("compressed-tensors linear " + base + " has shapes that do not match " + std::to_string(q.bits) + "-bit codes in " +
                                     (q.group ? "groups of " + std::to_string(q.group) : std::string("rows")) + " of a multiple of 32 inputs");
        c.kind = q.bits == 8 ? ct_q8_0 : q.sym ? ct_q4_0 : ct_q5_0;
        c.group = uint32_t(width);
        t.type = q.bits == 8 ? quant::GGML_TYPE_Q8_0 : q.sym ? quant::GGML_TYPE_Q4_0 : quant::GGML_TYPE_Q5_0;
        t.ne = {in, out};
    });
    schemes(q.activations, {}, m);
}

inline uint32_t word(const uint8_t* p, size_t at) {
    uint32_t w;
    std::memcpy(&w, p + 4 * at, 4);
    return w;
}

// The F16 bits of 16-bit scale `at` of `p`: F16 as stored, and BF16 where F16 holds it exactly, else refused naming tensor `name`, output `n` and group `g`.
inline uint16_t scale_f16(const uint8_t* p, size_t at, bool bf16, const std::string& name, size_t n, size_t g) {
    uint16_t h;
    std::memcpy(&h, p + 2 * at, 2);
    if (!bf16) return h;
    const float x = bf16_to_f32(h);
    h = f32_to_f16(x);
    if (!(f16_to_f32(h) == x)) {
        char text[32];
        std::snprintf(text, sizeof text, "%.9g", double(x));
        throw std::runtime_error("tensor " + name + ": BF16 scale " + text + " of output " + std::to_string(n) + ", group " + std::to_string(g) + " is not exact in F16");
    }
    return h;
}

// Refuses tensor `name` unless g_idx `p` gives input `k` its group of `group` inputs.
inline void follows(const uint8_t* p, size_t k, size_t group, const std::string& name) {
    if (word(p, k) != k / group)
        throw std::runtime_error("tensor " + name + ": g_idx of input " + std::to_string(k) + " is " + std::to_string(int32_t(word(p, k))) + ", not its group " +
                                 std::to_string(k / group) + "; llmx reads groups that follow the inputs only");
}

} // namespace detail

// The weight beside an FP8 scale grid `X.scale` that linears made: `X.weight`.
inline std::string fp8_weight_of(const std::string& grid) { return grid.substr(0, grid.size() - 6) + ".weight"; }

// The scale grid beside an FP8 weight `X.weight`: `X.scale`.
inline std::string fp8_grid_of(const std::string& weight) { return weight.substr(0, weight.size() - 7) + ".scale"; }

// Reads `qc`, config.json's quantization_config or null, and makes `m`'s quantized linears composites; integer tensors no rule consumed are refused by name.
inline void apply(const jmini::Value* qc, gguf::GGUFModel& m) {
    if (qc) {
        if (!qc->isObject()) throw std::runtime_error("quantization_config is not an object");
        const jmini::Value* method = qc->get("quant_method");
        if (!method || !method->isString()) throw std::runtime_error("quantization_config gives no quant_method");
        if (method->str == "awq") detail::awq(detail::awq_group(*qc), m);
        else if (method->str == "gptq") detail::gptq(detail::gptq_config(*qc), m);
        else if (method->str == "compressed-tensors") detail::compressed(detail::compressed_config(*qc), m);
        else if (method->str == "fp8") detail::fp8(detail::fp8_config(*qc), m);
        else throw std::runtime_error("quantization_config quant_method " + method->str + " is not one llmx reads");
    }
    std::map<std::string, size_t> names;
    for (size_t i = 0; i < m.tensors.size(); ++i) names[m.tensors[i].name] = i;
    for (const auto& t : m.tensors) {
        if (t.type == quant::GGML_TYPE_I8 || t.type == quant::GGML_TYPE_I32 || t.type == quant::GGML_TYPE_I64)
            throw std::runtime_error("tensor " + t.name + " holds integers that no quantization rule of the directory reads");
        if (t.type != quant::GGML_TYPE_F8_E4M3) continue;
        const auto grid = detail::ends_with(t.name, ".weight") ? names.find(fp8_grid_of(t.name)) : names.end();
        if (grid == names.end() || !m.composites.count(grid->second))
            throw std::runtime_error("tensor " + t.name + " holds F8_E4M3 codes that no quantization rule of the directory reads");
    }
}

// Writes composite `c`'s bytes, tensor `t`, at `out` from its parts' bytes read whole, in the order of c.parts, a block of 32 inputs of every output at a time.
// Each weight is (q - z) * s, held as d = s in F16 and an integer code; what no code holds exactly is refused by tensor (docs/src/format-hf_quant.md).
inline void repack(const gguf::GGUFModel::Composite& c, const gguf::TensorInfo& t, const std::vector<const uint8_t*>& parts, uint8_t* out) {
    if (c.kind == fp8_grid) {
        if (parts.size() != 1 || c.parts.size() != 1) throw std::logic_error("hf_quant: unknown repack of " + t.name);
        const size_t count = size_t(t.n_elements()), n = size_t(c.parts[0].first.n_elements());
        if (c.order.empty()) return gguf::fp8_grid(parts[0], c.parts[0].first.type, n, out, count);
        // A grid that follows its weight's V-head order (infer::hf::apply) is widened, then moved by whole scales.
        std::vector<uint8_t> grid(4 * count);
        gguf::fp8_grid(parts[0], c.parts[0].first.type, n, grid.data(), count);
        return gguf::permute(grid.data(), out, size_t(t.ne[0]), count / size_t(t.ne[0]), 4, c.order, c.columns);
    }
    const bool awq = c.kind == awq_q5_0, gptq = c.kind == gptq_q4_0 || c.kind == gptq_q5_0 || c.kind == gptq_q8_0, ct = c.kind == ct_q4_0 || c.kind == ct_q5_0 || c.kind == ct_q8_0;
    if (!(awq && parts.size() == 3) && !(gptq && (parts.size() == 3 || parts.size() == 4)) && !(ct && parts.size() == c.parts.size() && parts.size() >= 2))
        throw std::logic_error("hf_quant: unknown repack of " + t.name);
    static constexpr unsigned kOrder[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    static constexpr unsigned kNibble[8] = {0, 4, 1, 5, 2, 6, 3, 7};  // AWQ's nibble of output 8j + o, the inverse of kOrder
    constexpr size_t kBlock = 32;
    static_assert(quant::Q4_0_BLOCK == kBlock && quant::Q5_0_BLOCK == kBlock && quant::Q8_0_BLOCK == kBlock, "repacks write blocks of 32 inputs");
    const bool q4 = c.kind == gptq_q4_0 || c.kind == ct_q4_0, q8 = c.kind == gptq_q8_0 || c.kind == ct_q8_0;
    const unsigned bits = q8 ? 8 : 4, per = 32 / bits, mask = (1u << bits) - 1;
    const size_t in = size_t(t.ne[0]), rows = size_t(t.ne[1]), blocks = in / kBlock;
    const size_t block_bytes = q4 ? quant::Q4_0_TYPESIZE : q8 ? quant::Q8_0_TYPESIZE : quant::Q5_0_TYPESIZE;
    const size_t row_bytes = blocks * block_bytes;
    const auto put = [&](size_t n, size_t b, uint16_t d, const uint8_t* codes) {
        uint8_t* block = out + n * row_bytes + b * block_bytes;
        if (q4) quant::write_block_q4_0(d, codes, block);
        else if (q8) quant::write_block_q8_0(d, codes, block);
        else quant::write_block_q5_0(d, codes, block);
    };
    if (ct) {
        // compressed-tensors rows are outputs: codes, scales and zero points of output n are read whole a row at a time.
        const int zero = detail::part_of(c, "weight_zero_point"), shape = detail::part_of(c, "weight_shape"), g_idx = detail::part_of(c, "weight_g_idx");
        const bool packed = c.parts[0].first.type == quant::GGML_TYPE_I32, bf16 = c.parts[1].first.type == quant::GGML_TYPE_BF16;
        const size_t groups = in / c.group;
        const int half = 1 << (bits - 1);
        if (shape >= 0) {
            int64_t dims[2];
            std::memcpy(dims, parts[size_t(shape)], sizeof dims);
            if (dims[0] != int64_t(rows) || dims[1] != int64_t(in))
                throw std::runtime_error("tensor " + t.name + ": weight_shape is [" + std::to_string(dims[0]) + ", " + std::to_string(dims[1]) + "], not [" + std::to_string(rows) + ", " +
                                         std::to_string(in) + "]");
        }
        if (g_idx >= 0)
            for (size_t k = 0; k < in; ++k) detail::follows(parts[size_t(g_idx)], k, c.group, t.name);
        uint8_t codes[kBlock];
        for (size_t n = 0; n < rows; ++n)
            for (size_t b = 0; b < blocks; ++b) {
                const size_t g = b * kBlock / c.group;
                const int z = zero < 0 ? 0 : int((detail::word(parts[size_t(zero)], (n / per) * groups + g) >> (bits * (n % per))) & mask) - half;
                for (size_t i = 0; i < kBlock; ++i) {
                    const size_t k = b * kBlock + i;
                    const int q = packed ? int((detail::word(parts[0], n * (in / per) + k / per) >> (bits * (k % per))) & mask) - half : int(int8_t(parts[0][n * in + k]));
                    if (q < -half || q >= half)
                        throw std::runtime_error("tensor " + t.name + ": code " + std::to_string(q) + " of output " + std::to_string(n) + ", input " + std::to_string(k) + " is not " +
                                                 std::to_string(bits) + "-bit");
                    codes[i] = q8 ? uint8_t(int8_t(q)) : q4 ? uint8_t(q + 8) : uint8_t(q - z + 16);
                }
                put(n, b, detail::scale_f16(parts[1], n * groups + g, bf16, t.name, n, g), codes);
            }
        return;
    }
    const bool bf16 = c.parts[2].first.type == quant::GGML_TYPE_BF16;
    std::vector<uint16_t> d(rows);
    std::vector<uint32_t> zero(rows);
    std::vector<uint8_t> codes(rows * kBlock);
    size_t group = SIZE_MAX;
    for (size_t b = 0; b < blocks; ++b) {
        const size_t g = b * kBlock / c.group;
        if (g != group) {
            group = g;
            for (size_t n = 0; n < rows; ++n) {
                if (awq) {
                    zero[n] = (detail::word(parts[1], g * (rows / 8) + n / 8) >> (4 * kNibble[n % 8])) & 15;
                } else {
                    zero[n] = ((detail::word(parts[1], g * (rows / per) + n / per) >> (bits * (n % per))) & mask) + c.zero_offset;
                    if (c.kind != gptq_q5_0 && zero[n] != 1u << (bits - 1))
                        throw std::runtime_error("tensor " + t.name + ": zero point " + std::to_string(zero[n]) + " of output " + std::to_string(n) + ", group " +
                                                 std::to_string(g) + " is not " + std::to_string(1u << (bits - 1)) + ", as a symmetric GPTQ's is");
                }
                d[n] = detail::scale_f16(parts[2], g * rows + n, bf16, t.name, n, g);
            }
        }
        for (size_t i = 0; i < kBlock; ++i) {
            const size_t k = b * kBlock + i;
            if (awq) {
                for (size_t j = 0; j < rows / 8; ++j) {
                    const uint32_t w = detail::word(parts[0], k * (rows / 8) + j);
                    for (unsigned s = 0; s < 8; ++s) {
                        const size_t n = 8 * j + kOrder[s];
                        codes[n * kBlock + i] = uint8_t(int((w >> (4 * s)) & 15) - int(zero[n]) + 16);
                    }
                }
                continue;
            }
            if (parts.size() == 4) detail::follows(parts[3], k, c.group, t.name);
            const size_t at = (k / per) * rows, shift = bits * (k % per);
            for (size_t n = 0; n < rows; ++n) {
                const uint32_t q = (detail::word(parts[0], at + n) >> shift) & mask;
                codes[n * kBlock + i] = c.kind == gptq_q4_0 ? uint8_t(q) : c.kind == gptq_q8_0 ? uint8_t(int8_t(int(q) - 128)) : uint8_t(int(q) - int(zero[n]) + 16);
            }
        }
        for (size_t n = 0; n < rows; ++n) put(n, b, d[n], &codes[n * kBlock]);
    }
}

} // namespace hf_quant
