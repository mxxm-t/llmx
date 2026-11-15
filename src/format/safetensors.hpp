#pragma once
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "format/gguf.hpp"

// safetensors files and directories, read into the tensor table and file segments of a gguf::GGUFModel.
// File layout: header length (u64, little endian), a JSON object of that many bytes naming each tensor's dtype, shape and data_offsets, then the tensors' bytes.

namespace safetensors {

// A larger header is refused before it is read: the JSON values it parses into take many times its size.
constexpr uint64_t max_header_bytes = 100ull << 20;
// Shapes and offsets are JSON numbers, exact as integers up to 2^53.
constexpr double max_integer = 9007199254740992.0;
constexpr size_t max_rank = 4;

namespace detail {

struct Entry {
    gguf::TensorInfo tensor;
    uint64_t begin = 0, end = 0;
};

struct File {
    std::string path;
    uint64_t data_start = 0, size = 0;
    std::vector<Entry> entries;   // in data_offsets order
};

inline void unique_keys(const jmini::Object& object, const std::string& what) {
    std::set<std::string> keys;
    for (const auto& item : object)
        if (!keys.insert(item.first).second) throw std::runtime_error(what + " repeats key " + item.first);
}

inline uint64_t integer(const jmini::Value& v, const std::string& what) {
    if (!v.isNumber() || !v.whole || v.num < 0 || v.num > max_integer)
        throw std::runtime_error(what + " is not a non-negative integer up to 2^53");
    return uint64_t(v.num);
}

inline uint32_t storage_type(const std::string& dtype, const std::string& name) {
    if (dtype == "F32") return quant::GGML_TYPE_F32;
    if (dtype == "F16") return quant::GGML_TYPE_F16;
    if (dtype == "BF16") return quant::GGML_TYPE_BF16;
    if (dtype == "I32") return quant::GGML_TYPE_I32;
    if (dtype == "I64") return quant::GGML_TYPE_I64;
    throw std::runtime_error("safetensors tensor " + name + " has dtype " + dtype + ", which llmx does not read");
}

inline Entry entry(const std::string& name, const jmini::Value& v, const std::string& path) {
    const std::string what = "safetensors tensor " + name + " in " + path;
    if (!v.isObject()) throw std::runtime_error(what + " is not an object");
    unique_keys(v.obj, what);
    const jmini::Value *dtype = v.get("dtype"), *shape = v.get("shape"), *offsets = v.get("data_offsets");
    if (v.obj.size() != 3 || !dtype || !shape || !offsets)
        throw std::runtime_error(what + " must hold exactly dtype, shape and data_offsets");
    if (!dtype->isString()) throw std::runtime_error(what + ": dtype is not a string");
    if (!shape->isArray() || shape->arr.size() > max_rank) throw std::runtime_error(what + ": shape is not an array of rank 4 or less");
    if (!offsets->isArray() || offsets->arr.size() != 2) throw std::runtime_error(what + ": data_offsets is not a pair");
    Entry e;
    e.tensor.name = name;
    e.tensor.type = storage_type(dtype->str, name);
    // HF shapes list the slowest dimension first; GGUF's ne lists the fastest first.
    for (auto it = shape->arr.rbegin(); it != shape->arr.rend(); ++it) e.tensor.ne.push_back(integer(*it, what + " shape"));
    e.begin = integer(offsets->arr[0], what + " data_offsets");
    e.end = integer(offsets->arr[1], what + " data_offsets");
    if (e.end < e.begin) throw std::runtime_error(what + ": data_offsets end before they begin");
    if (e.tensor.data_size() != e.end - e.begin) throw std::runtime_error(what + ": data_offsets do not span its shape");
    return e;
}

// One file's header, checked: tensors cover the data section from its start to the file's end in order, with no gap or overlap, each at a file offset a float may be read at.
inline File read_file(const std::string& path) {
    File f;
    f.path = path;
    std::ifstream stream(std::filesystem::u8path(path), std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open file: " + path);
    stream.seekg(0, std::ios::end);
    f.size = uint64_t(stream.tellg());
    stream.seekg(0);
    unsigned char length_bytes[8];
    if (f.size < 8 || !stream.read(reinterpret_cast<char*>(length_bytes), 8))
        throw std::runtime_error("safetensors file too short: " + path);
    uint64_t length = 0;
    for (int i = 7; i >= 0; --i) length = (length << 8) | length_bytes[i];
    if (length > max_header_bytes) throw std::runtime_error("safetensors header over 100 MiB: " + path);
    if (length > f.size - 8) throw std::runtime_error("safetensors header runs past the end of " + path);
    std::string text(size_t(length), '\0');
    if (!stream.read(text.data(), std::streamsize(length))) throw std::runtime_error("cannot read safetensors header: " + path);
    f.data_start = 8 + length;
    const jmini::Value header = jmini::parse(text);
    if (!header.isObject()) throw std::runtime_error("safetensors header is not an object: " + path);
    unique_keys(header.obj, "safetensors header of " + path);
    for (const auto& item : header.obj) {
        if (item.first == "__metadata__") {
            if (!item.second.isObject()) throw std::runtime_error("safetensors __metadata__ is not an object: " + path);
            for (const auto& meta : item.second.obj)
                if (!meta.second.isString()) throw std::runtime_error("safetensors __metadata__ value is not a string: " + meta.first);
            continue;
        }
        f.entries.push_back(entry(item.first, item.second, path));
    }
    std::stable_sort(f.entries.begin(), f.entries.end(), [](const Entry& a, const Entry& b) {
        return a.begin != b.begin ? a.begin < b.begin : a.end < b.end;
    });
    uint64_t next = 0;
    for (const auto& e : f.entries) {
        if (e.begin != next) throw std::runtime_error("safetensors tensor " + e.tensor.name + (e.begin < next ? " overlaps another" : " leaves a gap before it") + " in " + path);
        if ((f.data_start + e.begin) % alignof(float)) throw std::runtime_error("safetensors tensor " + e.tensor.name + " is not at a float-aligned file offset in " + path);
        next = e.end;
    }
    if (next != f.size - f.data_start) throw std::runtime_error("safetensors tensors do not end where the file does: " + path);
    return f;
}

// A file named by an index must sit beside it.
inline void check_basename(const std::string& name) {
    if (name.empty() || name == "." || name == ".." || name.find_first_of("/\\:") != std::string::npos)
        throw std::runtime_error("safetensors index names a file outside its directory: " + name);
}

// The files of a directory: those its index names, each holding exactly the tensors the index maps to it, or its single model.safetensors.
inline std::vector<File> read_files(const std::filesystem::path& dir) {
    const std::filesystem::path index = dir / "model.safetensors.index.json";
    std::vector<File> files;
    if (!std::filesystem::exists(index)) {
        files.push_back(read_file((dir / "model.safetensors").u8string()));
        return files;
    }
    std::ifstream stream(index, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open file: " + index.u8string());
    const std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    const jmini::Value root = jmini::parse(text);
    if (!root.isObject()) throw std::runtime_error("safetensors index is not an object");
    unique_keys(root.obj, "safetensors index");
    const jmini::Value* map = root.get("weight_map");
    if (!map || !map->isObject()) throw std::runtime_error("safetensors index has no weight_map object");
    unique_keys(map->obj, "safetensors weight_map");
    std::map<std::string, std::set<std::string>> named;
    for (const auto& item : map->obj) {
        if (!item.second.isString()) throw std::runtime_error("safetensors weight_map value is not a file name: " + item.first);
        check_basename(item.second.str);
        named[item.second.str].insert(item.first);
    }
    for (const auto& [name, tensors] : named) {
        files.push_back(read_file((dir / std::filesystem::u8path(name)).u8string()));
        std::set<std::string> held;
        for (const auto& e : files.back().entries) held.insert(e.tensor.name);
        if (held != tensors) throw std::runtime_error("safetensors index does not name exactly the tensors of " + name);
    }
    return files;
}

} // namespace detail

// The tensors of a safetensors directory with their HF names, laid out as read_gguf lays out a set of shards; the metadata is left to the caller.
// Nothing past the headers is read; gguf::map_payload maps the files.
inline gguf::GGUFModel read_directory(const std::string& dir) {
    gguf::GGUFModel m;
    size_t base = 0;
    for (auto& file : detail::read_files(std::filesystem::u8path(dir))) {
        if (file.entries.empty()) continue;
        const size_t first = m.tensors.size();
        for (auto& e : file.entries) {
            m.offsets.push_back(size_t(gguf::checked_add(base, e.begin)));
            m.tensors.push_back(std::move(e.tensor));
        }
        const size_t size = size_t(file.size - file.data_start);
        m.segments.push_back({file.path, nullptr, size_t(file.data_start), base, size, first});
        base = size_t(gguf::aligned_size(gguf::checked_add(base, size), alignof(float)));
    }
    return m;
}

} // namespace safetensors
