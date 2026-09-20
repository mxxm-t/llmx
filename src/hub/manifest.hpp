#pragma once
#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "core/json.hpp"

namespace hub {

struct File {
    std::string name;
    uint64_t size = 0;
    std::string digest;
    bool lfs = false;
};

struct Manifest {
    std::string revision;
    std::vector<File> files;
};

inline std::string lower(std::string text) {
    for (char& c : text) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    return text;
}

inline bool hex_digest(const std::string& s, size_t length) {
    if (s.size() != length) return false;
    for (char c : s) if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    return true;
}

inline void validate_path(const std::string& path) {
    if (path.empty() || path.size() > 4096) throw std::runtime_error("pull: invalid repository file path");
    size_t start = 0;
    while (start < path.size()) {
        const size_t end = path.find('/', start);
        const std::string part = path.substr(start, end == std::string::npos ? end : end - start);
        if (part.empty() || part == "." || part == ".." || part.back() == '.' || part.back() == ' ')
            throw std::runtime_error("pull: unsafe repository file path");
        for (unsigned char c : part)
            if (c < 32 || c == 127 || std::string("\\:*?\"<>|").find(char(c)) != std::string::npos)
                throw std::runtime_error("pull: repository path is not portable");
        const std::string stem = lower(part.substr(0, part.find('.')));
        if (stem == "con" || stem == "prn" || stem == "aux" || stem == "nul" ||
            (stem.size() == 4 && (stem.substr(0, 3) == "com" || stem.substr(0, 3) == "lpt") &&
             stem[3] >= '0' && stem[3] <= '9'))
            throw std::runtime_error("pull: reserved repository file name");
        if (end == std::string::npos) return;
        start = end + 1;
    }
    throw std::runtime_error("pull: repository path ends in slash");
}

inline void validate_repo(const std::string& repo) {
    const size_t slash = repo.find('/');
    if (slash == std::string::npos || slash == 0 || slash + 1 == repo.size() ||
        repo.find('/', slash + 1) != std::string::npos || repo.size() > 193)
        throw std::runtime_error("pull: expected owner/repository:quant");
    validate_path(repo);
    for (char c : repo)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '/' || c == '-' || c == '_' || c == '.'))
            throw std::runtime_error("pull: invalid repository ID");
    if (repo.find("..") != std::string::npos || repo.find("--") != std::string::npos)
        throw std::runtime_error("pull: invalid repository ID");
}

inline std::string url_encode(const std::string& text, bool keep_slashes = false) {
    static constexpr char digits[] = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : text) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~' ||
            (keep_slashes && c == '/')) out += char(c);
        else { out += '%'; out += digits[c >> 4]; out += digits[c & 15]; }
    }
    return out;
}

inline const jmini::Value* optional_member(const jmini::Value& value, const std::string& key) {
    if (!value.isObject()) throw std::runtime_error("pull: metadata object required");
    const jmini::Value* result = nullptr;
    for (const auto& item : value.obj) if (item.first == key) {
        if (result) throw std::runtime_error("pull: duplicate metadata field: " + key);
        result = &item.second;
    }
    return result;
}

inline const jmini::Value& member(const jmini::Value& value, const std::string& key) {
    const auto* result = optional_member(value, key);
    if (!result) throw std::runtime_error("pull: missing metadata field: " + key);
    return *result;
}

inline std::string string_member(const jmini::Value& value, const std::string& key) {
    const auto& result = member(value, key);
    if (!result.isString()) throw std::runtime_error("pull: metadata string required: " + key);
    return result.str;
}

inline uint64_t size_member(const jmini::Value& value) {
    const auto& size = member(value, "size");
    if (!size.isNumber() || !std::isfinite(size.num) || size.num < 0 ||
        size.num > 9007199254740991.0 || std::floor(size.num) != size.num)
        throw std::runtime_error("pull: invalid file size");
    return uint64_t(size.num);
}

struct ShardName {
    std::string base;
    unsigned index = 0, count = 0;
};

inline ShardName shard_name(const std::string& name, bool validate = true) {
    ShardName out{name, 0, 0};
    if (name.size() < 21 || lower(name.substr(name.size() - 5)) != ".gguf") return out;
    const size_t p = name.size() - 20;
    if (name[p] != '-' || name.compare(p + 6, 4, "-of-") != 0) return out;
    for (size_t i : {p + 1, p + 10})
        for (size_t j = i; j < i + 5; ++j)
            if (name[j] < '0' || name[j] > '9') return out;
    out.index = unsigned(std::stoul(name.substr(p + 1, 5)));
    out.count = unsigned(std::stoul(name.substr(p + 10, 5)));
    out.base = name.substr(0, p) + name.substr(name.size() - 5);
    if (!validate) return out;
    if (!out.index || !out.count || out.index > out.count || out.count > 65535)
        throw std::runtime_error("pull: invalid shard filename");
    if (name.substr(name.size() - 5) != ".gguf")
        throw std::runtime_error("pull: sharded GGUF requires canonical .gguf filenames");
    return out;
}

inline void validate_quant(const std::string& quant) {
    if (quant.empty() || quant.size() > 32) throw std::runtime_error("pull: quant is required");
    for (char c : quant)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_')) throw std::runtime_error("pull: invalid quant");
}

inline Manifest select(const std::string& metadata, const std::string& quant,
                       const std::string& filename = {}) {
    validate_quant(quant);
    if (!filename.empty()) validate_path(filename);
    const auto root = jmini::parse(metadata);
    Manifest result;
    result.revision = string_member(root, "sha");
    if (!hex_digest(result.revision, 40)) throw std::runtime_error("pull: invalid resolved commit SHA");
    const auto& siblings = member(root, "siblings");
    if (!siblings.isArray()) throw std::runtime_error("pull: siblings array required");
    std::map<std::string, std::vector<File>> groups;
    std::set<std::string> paths;
    const std::string suffix = lower(quant) + ".gguf";
    for (const auto& sibling : siblings.arr) {
        const std::string name = string_member(sibling, "rfilename");
        if (name.size() < 5 || lower(name.substr(name.size() - 5)) != ".gguf") continue;
        const ShardName shard = shard_name(name, false);
        const std::string base = lower(shard.base);
        if (base.size() < suffix.size() || base.compare(base.size() - suffix.size(), suffix.size(), suffix) != 0)
            continue;
        const size_t prefix = base.size() - suffix.size();
        if (prefix && base[prefix - 1] != '-' && base[prefix - 1] != '.' && base[prefix - 1] != '/' && base[prefix - 1] != '_') continue;
        shard_name(name);
        validate_path(name);
        if (!paths.insert(lower(name)).second) throw std::runtime_error("pull: duplicate or case-colliding GGUF paths");
        File file{name, size_member(sibling), {}, false};
        if (!file.size) throw std::runtime_error("pull: empty GGUF file");
        const auto* lfs_value = optional_member(sibling, "lfs");
        if (lfs_value && lfs_value->t != jmini::Value::T::Null) {
            const auto& lfs = *lfs_value;
            file.digest = string_member(lfs, "sha256");
            file.lfs = true;
            if (size_member(lfs) != file.size || !hex_digest(file.digest, 64))
                throw std::runtime_error("pull: invalid LFS size or SHA256");
        } else {
            file.digest = string_member(sibling, "blobId");
            if (!hex_digest(file.digest, 40)) throw std::runtime_error("pull: invalid Git blob SHA1");
        }
        groups[shard.base + '\0' + std::to_string(shard.count)].push_back(std::move(file));
    }
    if (groups.empty()) throw std::runtime_error("pull: no GGUF file matches quant " + quant);
    auto chosen = groups.begin();
    if (!filename.empty()) {
        chosen = groups.end();
        for (auto it = groups.begin(); it != groups.end(); ++it)
            for (const auto& file : it->second) if (file.name == filename) chosen = it;
        if (chosen == groups.end()) throw std::runtime_error("pull: --file does not match the requested quant");
    } else if (groups.size() != 1) {
        std::string message = "pull: ambiguous quant; select a file with --file:";
        for (const auto& group : groups) message += "\n  " + group.second.front().name;
        throw std::runtime_error(message);
    }
    result.files = std::move(chosen->second);
    std::sort(result.files.begin(), result.files.end(), [](const File& a, const File& b) { return a.name < b.name; });
    const auto first = shard_name(result.files.front().name);
    if (first.count) {
        if (result.files.size() != first.count) throw std::runtime_error("pull: incomplete GGUF shard set");
        for (size_t i = 0; i < result.files.size(); ++i) {
            const auto shard = shard_name(result.files[i].name);
            if (shard.index != i + 1 || shard.count != first.count)
                throw std::runtime_error("pull: inconsistent GGUF shard set");
        }
    } else if (result.files.size() != 1) throw std::runtime_error("pull: mixed sharded and single GGUF files");
    return result;
}

} // namespace hub
