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
#include "quant/quant.hpp"

// Quantized Hugging Face checkpoints: config.json's quantization_config read strictly, and each quantized linear's companion tensors made one tensor of an existing type whose values are the checkpoint's exactly (docs/src/format-hf_quant.md).
// The tensor is a gguf::GGUFModel::Composite; the loader makes its bytes with repack.

namespace hf_quant {

// What a repack writes; the disk tier's identity carries it, so a change to any kind's bytes moves it.
constexpr uint32_t kRepackVersion = 2;

enum Kind : uint32_t { awq_q5_0 = 1, gptq_q4_0 = 2, gptq_q5_0 = 3, gptq_q8_0 = 4 };

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

// Each `X.qweight` of `m` and its companions `X.<suffix>`, the first `required` of them required, made one composite `X.weight` at qweight's place, the companions leaving the table.
// `make` checks the parts and sets the composite's kind and the tensor's type and shape.
template <class Make> void linears(gguf::GGUFModel& m, const std::vector<std::string>& companions, size_t required, Make make) {
    std::map<std::string, size_t> index;
    for (size_t i = 0; i < m.tensors.size(); ++i) index[m.tensors[i].name] = i;
    std::vector<size_t> drop;
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        gguf::TensorInfo& t = m.tensors[i];
        const auto companion = std::find_if(companions.begin(), companions.end(), [&](const std::string& s) { return ends_with(t.name, "." + s); });
        if (companion != companions.end()) {
            if (!index.count(t.name.substr(0, t.name.size() - companion->size()) + "qweight")) throw std::runtime_error("tensor " + t.name + " has no qweight beside it");
            continue;
        }
        if (!ends_with(t.name, ".qweight")) continue;
        const std::string base = t.name.substr(0, t.name.size() - 8);
        gguf::GGUFModel::Composite c;
        c.parts = {{t, m.offsets[i]}};
        for (size_t j = 0; j < companions.size(); ++j) {
            const auto found = index.find(base + "." + companions[j]);
            if (found == index.end()) {
                if (j < required) throw std::runtime_error("tensor " + t.name + " lacks its " + companions[j]);
                continue;
            }
            c.parts.push_back({m.tensors[found->second], m.offsets[found->second]});
            drop.push_back(found->second);
        }
        make(base, c, t);
        t.name = base + ".weight";
        m.composites[i] = std::move(c);
    }
    std::sort(drop.begin(), drop.end());
    m.erase_tensors(drop);
}

inline bool scales_type(const gguf::TensorInfo& s) { return s.type == quant::GGML_TYPE_F16 || s.type == quant::GGML_TYPE_BF16; }

// AWQ linears: qweight [in, out / 8], qzeros [in / group, out / 8] and scales [in / group, out] (HF order) make one Q5_0 tensor [in, out].
inline void awq(uint32_t group, gguf::GGUFModel& m) {
    linears(m, {"qzeros", "scales"}, 2, [&](const std::string& base, gguf::GGUFModel::Composite& c, gguf::TensorInfo& t) {
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
    linears(m, {"qzeros", "scales", "g_idx"}, 2, [&](const std::string& base, gguf::GGUFModel::Composite& c, gguf::TensorInfo& t) {
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

inline uint32_t word(const uint8_t* p, size_t at) {
    uint32_t w;
    std::memcpy(&w, p + 4 * at, 4);
    return w;
}

} // namespace detail

// Reads `qc`, config.json's quantization_config or null, and makes `m`'s quantized linears composites; integer tensors no rule consumed are refused by name.
inline void apply(const jmini::Value* qc, gguf::GGUFModel& m) {
    if (qc) {
        if (!qc->isObject()) throw std::runtime_error("quantization_config is not an object");
        const jmini::Value* method = qc->get("quant_method");
        if (!method || !method->isString()) throw std::runtime_error("quantization_config gives no quant_method");
        if (method->str == "awq") detail::awq(detail::awq_group(*qc), m);
        else if (method->str == "gptq") detail::gptq(detail::gptq_config(*qc), m);
        else throw std::runtime_error("quantization_config quant_method " + method->str + " is not one llmx reads");
    }
    for (const auto& t : m.tensors)
        if (t.type == quant::GGML_TYPE_I32 || t.type == quant::GGML_TYPE_I64)
            throw std::runtime_error("tensor " + t.name + " holds integers that no quantization rule of the directory reads");
}

// Writes composite `c`'s bytes, tensor `t`, at `out` from its parts' bytes read whole, in the order of c.parts, a block of 32 inputs of every output at a time.
// Each weight is (q - z) * s, held as d = s in F16 and an integer code; a scale F16 cannot hold, a symmetric GPTQ zero point other than the middle code and a g_idx whose groups do not follow the inputs are refused by tensor.
inline void repack(const gguf::GGUFModel::Composite& c, const gguf::TensorInfo& t, const std::vector<const uint8_t*>& parts, uint8_t* out) {
    const bool awq = c.kind == awq_q5_0, gptq = c.kind == gptq_q4_0 || c.kind == gptq_q5_0 || c.kind == gptq_q8_0;
    if (!(awq && parts.size() == 3) && !(gptq && (parts.size() == 3 || parts.size() == 4))) throw std::logic_error("hf_quant: unknown repack of " + t.name);
    static constexpr unsigned kOrder[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    static constexpr unsigned kNibble[8] = {0, 4, 1, 5, 2, 6, 3, 7};  // AWQ's nibble of output 8j + o, the inverse of kOrder
    constexpr size_t kBlock = 32;
    static_assert(quant::Q4_0_BLOCK == kBlock && quant::Q5_0_BLOCK == kBlock && quant::Q8_0_BLOCK == kBlock, "repacks write blocks of 32 inputs");
    const unsigned bits = c.kind == gptq_q8_0 ? 8 : 4, per = 32 / bits, mask = (1u << bits) - 1;
    const size_t in = size_t(t.ne[0]), rows = size_t(t.ne[1]), blocks = in / kBlock;
    const size_t block_bytes = c.kind == gptq_q4_0 ? quant::Q4_0_TYPESIZE : c.kind == gptq_q8_0 ? quant::Q8_0_TYPESIZE : quant::Q5_0_TYPESIZE;
    const size_t row_bytes = blocks * block_bytes;
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
                uint16_t h;
                std::memcpy(&h, parts[2] + 2 * (g * rows + n), 2);
                if (bf16) {
                    const float x = bf16_to_f32(h);
                    h = f32_to_f16(x);
                    if (!(f16_to_f32(h) == x)) {
                        char text[32];
                        std::snprintf(text, sizeof text, "%.9g", double(x));
                        throw std::runtime_error("tensor " + t.name + ": BF16 scale " + text + " of output " + std::to_string(n) + ", group " + std::to_string(g) + " is not exact in F16");
                    }
                }
                d[n] = h;
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
            if (parts.size() == 4 && detail::word(parts[3], k) != k / c.group)
                throw std::runtime_error("tensor " + t.name + ": g_idx of input " + std::to_string(k) + " is " + std::to_string(int32_t(detail::word(parts[3], k))) +
                                         ", not its group " + std::to_string(k / c.group) + "; llmx reads GPTQ whose groups follow the inputs only");
            const size_t at = (k / per) * rows, shift = bits * (k % per);
            for (size_t n = 0; n < rows; ++n) {
                const uint32_t q = (detail::word(parts[0], at + n) >> shift) & mask;
                codes[n * kBlock + i] = c.kind == gptq_q4_0 ? uint8_t(q) : c.kind == gptq_q8_0 ? uint8_t(int8_t(int(q) - 128)) : uint8_t(int(q) - int(zero[n]) + 16);
            }
        }
        for (size_t n = 0; n < rows; ++n) {
            uint8_t* block = out + n * row_bytes + b * block_bytes;
            if (c.kind == gptq_q4_0) quant::write_block_q4_0(d[n], &codes[n * kBlock], block);
            else if (c.kind == gptq_q8_0) quant::write_block_q8_0(d[n], &codes[n * kBlock], block);
            else quant::write_block_q5_0(d[n], &codes[n * kBlock], block);
        }
    }
}

} // namespace hf_quant
