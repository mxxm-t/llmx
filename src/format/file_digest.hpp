#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>
#include "core/sha.hpp"
#include "format/file_reader.hpp"
#include "format/output_file.hpp"
#if !defined(_WIN32)
#include <sys/stat.h>
#endif

// A file's SHA-256, and a cache of it keyed by the file's path, size, modification time and inode, so a large model file is read for its digest once (docs/DISK-TIER.md, The entry file, identity).

namespace format {

// What a cached digest is held to: the file's size, its modification time in nanoseconds and its inode (on Windows, its file index), any change of which means the file may have changed.
struct FileStamp {
    uint64_t size = 0, mtime = 0, inode = 0;
    bool operator==(const FileStamp& o) const { return size == o.size && mtime == o.mtime && inode == o.inode; }
};

// The stamp of the file at `path`; a file that cannot be read throws.
inline FileStamp file_stamp(const std::string& path) {
    FileStamp s;
#if defined(_WIN32)
    HANDLE file = CreateFileW(std::filesystem::u8path(path).c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot read file: " + path);
    BY_HANDLE_FILE_INFORMATION info;
    const BOOL ok = GetFileInformationByHandle(file, &info);
    CloseHandle(file);
    if (!ok) throw std::runtime_error("cannot read file: " + path);
    s.size = (uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
    s.mtime = ((uint64_t(info.ftLastWriteTime.dwHighDateTime) << 32) | info.ftLastWriteTime.dwLowDateTime) * 100;
    s.inode = (uint64_t(info.nFileIndexHigh) << 32) | info.nFileIndexLow;
#else
    struct stat st;
    if (stat(path.c_str(), &st) != 0) throw std::runtime_error("cannot read file: " + path);
    s.size = (uint64_t)st.st_size;
#if defined(__APPLE__)
    s.mtime = (uint64_t)st.st_mtimespec.tv_sec * 1000000000ull + (uint64_t)st.st_mtimespec.tv_nsec;
#else
    s.mtime = (uint64_t)st.st_mtim.tv_sec * 1000000000ull + (uint64_t)st.st_mtim.tv_nsec;
#endif
    s.inode = (uint64_t)st.st_ino;
#endif
    return s;
}

// The SHA-256 of the whole file at `path`, as lowercase hex, read in order in 64 MiB pieces through the file cache.
inline std::string file_sha256(const std::string& path) {
    FileReader reader(path);
    core::Sha hash(true);
    std::vector<uint8_t> piece(size_t(64) << 20);
    for (uint64_t at = 0; at < reader.size();) {
        const size_t n = reader.read(at, piece.data(), piece.size());
        if (!n) throw std::runtime_error("cannot read file: " + path + " ends before its size");
        hash.update(piece.data(), n);
        at += n;
    }
    return hash.hex();
}

// The SHA-256 of the file at `path`, from the cache in directory `dir` where it holds one written with the file's present stamp, else read whole and cached there.
// The cache holds a small file a path, named by the SHA-256 of the path, written whole or not at all; a digest whose file changed while it was read is returned but not cached.
inline std::string cached_file_sha256(const std::string& dir, const std::string& path) {
    const std::string absolute = std::filesystem::absolute(std::filesystem::u8path(path)).u8string();
    core::Sha name(true);
    name.update(absolute.data(), absolute.size());
    const std::filesystem::path entry = std::filesystem::u8path(dir) / ("sha256-" + name.hex() + ".txt");
    const FileStamp before = file_stamp(path);
    {
        std::ifstream in(entry, std::ios::binary);
        std::string magic, cached_path, digest;
        FileStamp stamp;
        if (std::getline(in, magic) && magic == "llmx-file-sha256 1" && std::getline(in, cached_path) && cached_path == absolute &&
            (in >> stamp.size >> stamp.mtime >> stamp.inode >> digest) && stamp == before && digest.size() == 64)
            return digest;
    }
    const std::string digest = file_sha256(path);
    if (!(file_stamp(path) == before)) return digest;
    std::filesystem::create_directories(std::filesystem::u8path(dir));
    OutputFile out(entry.u8string());
    out.write([&](std::ostream& o) {
        o << "llmx-file-sha256 1\n" << absolute << "\n" << before.size << " " << before.mtime << " " << before.inode << " " << digest << "\n";
    });
    out.publish();
    return digest;
}

} // namespace format
