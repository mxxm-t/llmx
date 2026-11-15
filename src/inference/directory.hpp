#pragma once
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "format/gguf.hpp"
#include "format/safetensors.hpp"
#include "model/arch/registry.hpp"
#include "tokenizer/hf_tokenizer.hpp"

// A Hugging Face model directory read as the GGUF file its converter writes would read: the tensors under the architecture's names, its metadata and its tokenizer keys.
// The tensors keep the directory's types; the loader widens 16-bit vectors to F32, as the converter stores them (inference/load.hpp).

namespace infer {

// Whether `path` names a directory, which the loader reads as a Hugging Face model.
inline bool is_model_directory(const std::string& path) {
    std::error_code ec;
    return std::filesystem::is_directory(std::filesystem::u8path(path), ec);
}

inline gguf::GGUFModel read_directory(const std::string& dir) {
    const std::filesystem::path config_path = std::filesystem::u8path(dir) / "config.json";
    std::ifstream stream(config_path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open file: " + config_path.u8string());
    const jmini::Value config = jmini::parse(std::string((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>()));
    const jmini::Value* type = config.isObject() ? config.get("model_type") : nullptr;
    if (!type || !type->isString()) throw std::runtime_error("config.json gives no model_type");
    gguf::GGUFModel m = safetensors::read_directory(dir);
    const ArchEntry& entry = hf_architecture(type->str);
    hf::apply(entry.hf_table(), entry.name, config, m);
    // hf::apply has held vocab_size to the embedding's rows.
    const uint64_t rows = uint64_t(config.get("vocab_size")->num);
    for (auto& item : bpe::hf::read_keys(dir, config, rows)) m.kv.push_back(std::move(item));
    return m;
}

// The model at `path`: a GGUF file, the first of a shard set, or a Hugging Face directory.
inline gguf::GGUFModel read_model(const std::string& path) {
    return is_model_directory(path) ? read_directory(path) : gguf::read_gguf(path);
}

// The files of a directory its metadata is read from, in a fixed order, those present.
inline std::vector<std::string> directory_metadata_files(const std::string& dir) {
    std::vector<std::string> files;
    for (const char* name : {"config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json"}) {
        const std::filesystem::path p = std::filesystem::u8path(dir) / name;
        if (std::filesystem::exists(p)) files.push_back(p.u8string());
    }
    return files;
}

// Whether a directory's tensor `t` is widened to F32 on load: a vector of 16-bit floats.
inline bool widened_on_load(const gguf::TensorInfo& t) {
    return t.ne.size() == 1 && (t.type == quant::GGML_TYPE_BF16 || t.type == quant::GGML_TYPE_F16);
}

} // namespace infer
