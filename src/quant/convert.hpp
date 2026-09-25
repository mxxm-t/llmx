#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "core/json.hpp"
#include "format/gguf.hpp"
#include "quant/quant.hpp"

// Conversion between a model as raw F32 tensors and a GGUF file: model.json names the model and each tensor with its shape, model.bin holds their floats in that order.
namespace quant {

// The tensors model.json describes, each to be written as `type`.
inline std::vector<gguf::TensorInfo> raw_tensors(const jmini::Value& root, uint32_t type) {
    std::vector<gguf::TensorInfo> out;
    const jmini::Value* tensors = root.get("tensors");
    if (!tensors || !tensors->isArray())
        throw std::runtime_error("model.json: missing \"tensors\" array");
    for (const auto& t : tensors->asArray()) {
        gguf::TensorInfo jt;
        jt.type = type;
        const jmini::Value* name = t.get("name");
        if (!name || !name->isString())
            throw std::runtime_error("model.json: tensor missing \"name\" string");
        jt.name = name->asString();
        const jmini::Value* shape = t.get("shape");
        if (!shape || !shape->isArray())
            throw std::runtime_error("model.json: tensor missing \"shape\" array: " + jt.name);
        if (shape->asArray().empty() || shape->asArray().size() > 4)
            throw std::runtime_error("model.json: tensor rank must be between 1 and 4: " + jt.name);
        for (const auto& d : shape->asArray()) {
            if (!d.isNumber())
                throw std::runtime_error("model.json: shape dim is not a number: " + jt.name);
            const double v = d.asNumber();
            // JSON numbers are doubles; stay within their consecutive integer range.
            if (!std::isfinite(v) || v < 1 || v > 9007199254740991.0 || std::floor(v) != v)
                throw std::runtime_error("model.json: dimension must be an integer from 1 to 2^53-1: " + jt.name);
            jt.ne.push_back(uint64_t(v));
        }
        out.push_back(std::move(jt));
    }
    return out;
}

// The GGML type a quantize type name selects, or none for any other name: q8_0 and q4_0 are the only types quantize writes.
inline std::optional<uint32_t> quant_type_of(const std::string& name) {
    if (name == "q8_0") return gguf::GGML_TYPE_Q8_0;
    if (name == "q4_0") return gguf::GGML_TYPE_Q4_0;
    return std::nullopt;
}

// GGUF's general.file_type for a model whose matrices are all `type`.
inline uint32_t file_type_of(uint32_t type) {
    if (type == gguf::GGML_TYPE_Q8_0) return 7;   // MOSTLY_Q8_0
    if (type == gguf::GGML_TYPE_Q4_0) return 2;   // MOSTLY_Q4_0
    throw std::runtime_error("quantize: no file type for this quant type");
}

// Write the model model.json and model.bin describe as a GGUF file whose tensors are all `type`; returns the tensor count.
inline size_t quantize_raw(const std::string& json_path, const std::string& bin_path, const std::string& out_path, uint32_t type) {
    const QuantType* qt = Registry::instance().get(type);
    if (!qt || !qt->quantize) throw std::runtime_error("quantize: unsupported quant type");
    std::ifstream jf(std::filesystem::u8path(json_path));
    if (!jf) throw std::runtime_error("cannot open " + json_path);
    std::stringstream jss;
    jss << jf.rdbuf();
    jmini::Value root = jmini::parse(jss.str());
    gguf::GGUFModel m;
    m.tensors = raw_tensors(root, type);
    uint64_t need = 0;
    uint64_t output_size = 0;
    for (const auto& t : m.tensors) {
        need = gguf::checked_add(need, gguf::checked_multiply(t.n_elements(), sizeof(float)));
        output_size = gguf::checked_add(gguf::aligned_size(output_size, alignof(float)), t.data_size());
    }
    std::vector<float> values;
    if (need / sizeof(float) > values.max_size() ||
        need > uint64_t(std::numeric_limits<std::streamsize>::max()) ||
        output_size > m.blob.max_size())
        throw std::runtime_error("model.json: tensor storage exceeds allocation or stream limit");

    std::ifstream bf(std::filesystem::u8path(bin_path), std::ios::binary);
    if (!bf) throw std::runtime_error("cannot open " + bin_path);
    bf.exceptions(std::ios::failbit | std::ios::badbit);
    bf.seekg(0, std::ios::end);
    const std::streamoff sz = bf.tellg();
    if (sz < 0) throw std::runtime_error("cannot determine model.bin size");
    if (uint64_t(sz) != need)
        throw std::runtime_error("model.bin size does not match model.json tensor shapes");
    bf.seekg(0, std::ios::beg);
    values.resize(size_t(need / sizeof(float)));
    if (need) bf.read(reinterpret_cast<char*>(values.data()), std::streamsize(need));
    const float* fptr = values.data();
    m.blob.reserve(size_t(output_size));

    const jmini::Value* name = root.get("name");
    gguf::MetaValue mv_name; mv_name.vtype = gguf::V_STRING; mv_name.s = (name && name->isString()) ? name->asString() : "custom";
    gguf::MetaValue mv_arch; mv_arch.vtype = gguf::V_STRING; mv_arch.s = "custom";
    gguf::MetaValue mv_qver; mv_qver.vtype = gguf::V_UINT32; mv_qver.u = 2;   // quantization_version
    gguf::MetaValue mv_ft;   mv_ft.vtype = gguf::V_UINT32; mv_ft.u = file_type_of(type);
    m.kv.emplace_back("general.name", mv_name);
    m.kv.emplace_back("general.architecture", mv_arch);
    m.kv.emplace_back("general.quantization_version", mv_qver);
    m.kv.emplace_back("general.file_type", mv_ft);

    for (const auto& t : m.tensors) {
        const size_t nblocks = size_t(t.n_elements() / qt->block_size);
        std::vector<uint8_t> q(size_t(t.data_size()));
        qt->quantize(fptr, q.data(), nblocks);
        fptr += size_t(t.n_elements());
        m.add_tensor_data(q);
    }
    gguf::write_gguf(m, out_path);
    return m.tensors.size();
}

// Write a GGUF file's tensors as raw F32, the model.json and model.bin that quantize_raw reads.
inline void dequantize_to_raw(const std::string& in_path, const std::string& out_json, const std::string& out_bin) {
    const gguf::GGUFModel m = gguf::read_gguf(in_path);
    std::stringstream js;
    js << "{\n";
    js << "  \"name\": " << jmini::quote(in_path) << ",\n";
    js << "  \"tensors\": [\n";
    for (size_t i = 0; i < m.tensors.size(); i++) {
        const auto& t = m.tensors[i];
        js << "    {\"name\": " << jmini::quote(t.name) << ", \"shape\": [";
        for (size_t d = 0; d < t.ne.size(); d++) {
            if (d) js << ", ";
            js << t.ne[d];
        }
        js << "]}";
        if (i + 1 < m.tensors.size()) js << ",";
        js << "\n";
    }
    js << "  ]\n}\n";

    std::vector<uint8_t> out;
    for (size_t i = 0; i < m.tensors.size(); i++) {
        const auto& t = m.tensors[i];
        const uint8_t* raw = m.tensor_data(i);
        const size_t n = (size_t)t.n_elements();
        std::vector<float> f(n);
        if (t.type == gguf::GGML_TYPE_F32) {
            std::memcpy(f.data(), raw, n * 4);
        } else {
            const QuantType* qt = Registry::instance().get(t.type);
            if (!qt || !qt->dequantize)
                throw std::runtime_error("unsupported tensor type in dequantize: " + t.name);
            qt->dequantize(raw, f.data(), n / qt->block_size);
        }
        out.resize(out.size() + n * 4);
        std::memcpy(out.data() + (out.size() - n * 4), f.data(), n * 4);
    }

    std::ofstream oj(std::filesystem::u8path(out_json));
    if (!oj) throw std::runtime_error("cannot open " + out_json);
    oj << js.str();
    std::ofstream ob(std::filesystem::u8path(out_bin), std::ios::binary);
    if (!ob) throw std::runtime_error("cannot open " + out_bin);
    ob.write((const char*)out.data(), (std::streamsize)out.size());
}

} // namespace quant
