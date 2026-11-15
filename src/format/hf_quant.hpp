#pragma once
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
constexpr uint32_t kRepackVersion = 1;

enum Kind : uint32_t { awq_q5_0 = 1 };

namespace detail {

inline bool ends_with(const std::string& s, const std::string& tail) {
    return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
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
            if (!v.isNumber() || !v.whole || v.num < 32 || v.num > 65536 || uint32_t(v.num) % 32)
                throw std::runtime_error(what + ": llmx reads AWQ groups of a positive multiple of 32 inputs");
            group = uint32_t(v.num);
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

// Each AWQ linear's qweight [in, out / 8], qzeros [in / group, out / 8] and scales [in / group, out] (HF order) made one Q5_0 tensor [in, out] at qweight's place, named as the unquantized weight.
inline void awq(uint32_t group, gguf::GGUFModel& m) {
    std::map<std::string, size_t> index;
    for (size_t i = 0; i < m.tensors.size(); ++i) index[m.tensors[i].name] = i;
    std::vector<size_t> drop;
    for (size_t i = 0; i < m.tensors.size(); ++i) {
        gguf::TensorInfo& t = m.tensors[i];
        if (ends_with(t.name, ".qzeros") || ends_with(t.name, ".scales")) {
            const std::string base = t.name.substr(0, t.name.rfind('.'));
            if (!index.count(base + ".qweight")) throw std::runtime_error("tensor " + t.name + " has no qweight beside it");
            continue;
        }
        if (!ends_with(t.name, ".qweight")) continue;
        const std::string base = t.name.substr(0, t.name.size() - 8);
        const auto zeros = index.find(base + ".qzeros"), scales = index.find(base + ".scales");
        if (zeros == index.end() || scales == index.end()) throw std::runtime_error("tensor " + t.name + " lacks its qzeros or scales");
        const gguf::TensorInfo &z = m.tensors[zeros->second], &s = m.tensors[scales->second];
        if (t.type != quant::GGML_TYPE_I32 || z.type != quant::GGML_TYPE_I32 || (s.type != quant::GGML_TYPE_F16 && s.type != quant::GGML_TYPE_BF16))
            throw std::runtime_error("AWQ linear " + base + " is not I32 qweight and qzeros with 16-bit scales");
        if (t.ne.size() != 2 || z.ne.size() != 2 || s.ne.size() != 2) throw std::runtime_error("AWQ linear " + base + " is not a matrix");
        const uint64_t in = t.ne[1], out = s.ne[0];
        if (!in || !out || in % group || t.ne[0] * 8 != out || z.ne[0] * 8 != out || z.ne[1] * group != in || s.ne[1] * group != in)
            throw std::runtime_error("AWQ linear " + base + " has shapes that do not match a group of " + std::to_string(group));
        gguf::GGUFModel::Composite c;
        c.kind = awq_q5_0;
        c.group = group;
        c.parts = {{t, m.offsets[i]}, {z, m.offsets[zeros->second]}, {s, m.offsets[scales->second]}};
        t.name = base + ".weight";
        t.type = quant::GGML_TYPE_Q5_0;
        t.ne = {in, out};
        m.composites[i] = std::move(c);
        drop.push_back(zeros->second);
        drop.push_back(scales->second);
    }
    std::sort(drop.begin(), drop.end());
    m.erase_tensors(drop);
}

} // namespace detail

// Reads `qc`, config.json's quantization_config or null, and makes `m`'s quantized linears composites; integer tensors no rule consumed are refused by name.
inline void apply(const jmini::Value* qc, gguf::GGUFModel& m) {
    if (qc) {
        if (!qc->isObject()) throw std::runtime_error("quantization_config is not an object");
        const jmini::Value* method = qc->get("quant_method");
        if (!method || !method->isString()) throw std::runtime_error("quantization_config gives no quant_method");
        if (method->str != "awq") throw std::runtime_error("quantization_config quant_method " + method->str + " is not one llmx reads");
        detail::awq(detail::awq_group(*qc), m);
    }
    for (const auto& t : m.tensors)
        if (t.type == quant::GGML_TYPE_I32 || t.type == quant::GGML_TYPE_I64)
            throw std::runtime_error("tensor " + t.name + " holds integers that no quantization rule of the directory reads");
}

// Writes composite `c`'s bytes, tensor `t`, at `out` from its parts' bytes read whole, in the order of c.parts.
// AWQ: the weight at output n and input k is (q - z) * s; Q5_0 holds it as d = s in F16 and code q - z + 16, so a 16-bit scale F16 cannot hold exactly is refused by tensor and scale.
inline void repack(const gguf::GGUFModel::Composite& c, const gguf::TensorInfo& t, const std::vector<const uint8_t*>& parts, uint8_t* out) {
    if (c.kind != awq_q5_0 || parts.size() != 3) throw std::logic_error("hf_quant: unknown repack of " + t.name);
    static constexpr unsigned kOrder[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    const size_t in = size_t(t.ne[0]), rows = size_t(t.ne[1]), words = rows / 8, blocks = in / quant::Q5_0_BLOCK;
    const size_t row_bytes = blocks * quant::Q5_0_TYPESIZE;
    const bool bf16 = c.parts[2].first.type == quant::GGML_TYPE_BF16;
    const auto word = [](const uint8_t* p, size_t at) {
        uint32_t w;
        std::memcpy(&w, p + 4 * at, 4);
        return w;
    };
    std::vector<uint16_t> d(rows);
    std::vector<uint8_t> zero(rows), codes(rows * quant::Q5_0_BLOCK);
    size_t group = SIZE_MAX;
    for (size_t b = 0; b < blocks; ++b) {
        const size_t g = b * quant::Q5_0_BLOCK / c.group;
        if (g != group) {
            group = g;
            for (size_t j = 0; j < words; ++j) {
                const uint32_t w = word(parts[1], g * words + j);
                for (unsigned s = 0; s < 8; ++s) zero[8 * j + kOrder[s]] = uint8_t((w >> (4 * s)) & 15);
            }
            for (size_t n = 0; n < rows; ++n) {
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
        for (size_t i = 0; i < quant::Q5_0_BLOCK; ++i) {
            const size_t k = b * quant::Q5_0_BLOCK + i;
            for (size_t j = 0; j < words; ++j) {
                const uint32_t w = word(parts[0], k * words + j);
                for (unsigned s = 0; s < 8; ++s) {
                    const size_t n = 8 * j + kOrder[s];
                    codes[n * quant::Q5_0_BLOCK + i] = uint8_t(int((w >> (4 * s)) & 15) - int(zero[n]) + 16);
                }
            }
        }
        for (size_t n = 0; n < rows; ++n) quant::write_block_q5_0(d[n], &codes[n * quant::Q5_0_BLOCK], out + n * row_bytes + b * quant::Q5_0_TYPESIZE);
    }
}

} // namespace hf_quant
