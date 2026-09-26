#pragma once
// A file read at given offsets into the caller's memory, by several threads at once (docs/src/format-file_reader.md).
// The loader streams weights through it in large reads in file order, where a mapping would fault them in a page at a time.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "core/host_memory.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace format {

class FileReader {
public:
    explicit FileReader(const std::string& path) : path_(path) {
#if defined(_WIN32)
        const std::wstring wide = std::filesystem::u8path(path).wstring();
        file_ = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_FLAG_SEQUENTIAL_SCAN | FILE_FLAG_OVERLAPPED, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open file: " + path);
        LARGE_INTEGER size;
        if (!GetFileSizeEx(file_, &size)) { close(); throw std::runtime_error("cannot size file: " + path); }
        size_ = (uint64_t)size.QuadPart;
        granule_ = core::page_size();
#else
        fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) throw std::runtime_error("cannot open file: " + path);
        struct stat st {};
        if (fstat(fd_, &st) != 0) { close(); throw std::runtime_error("cannot size file: " + path); }
        size_ = (uint64_t)st.st_size;
        granule_ = std::max(core::page_size(), st.st_blksize > 0 ? (size_t)st.st_blksize : size_t(0));
#if defined(POSIX_FADV_SEQUENTIAL)
        posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
#endif
    }
    ~FileReader() { close(); }
    FileReader(const FileReader&) = delete;
    FileReader& operator=(const FileReader&) = delete;

    const std::string& path() const { return path_; }
    uint64_t size() const { return size_; }
    // The unit a read starts and ends on for the file system to serve it whole: the larger of the page and the file's block size, which is the recordsize on ZFS.
    size_t granule() const { return granule_; }

    // Read up to `bytes` from `offset` into `dst` and return how many were read, fewer only at the end of the file.
    size_t read(uint64_t offset, void* dst, size_t bytes) const {
        size_t done = 0;
        while (done < bytes) {
            const size_t n = read_some(offset + done, (uint8_t*)dst + done, bytes - done);
            if (!n) break;
            done += n;
        }
        return done;
    }

private:
    // One call's worth, 0 at the end of the file.
    size_t read_some(uint64_t offset, uint8_t* dst, size_t bytes) const {
#if defined(_WIN32)
        // Each call waits on its own event, since the handle's own signal is shared by every read in flight on it.
        OVERLAPPED ov{};
        ov.Offset = (DWORD)offset;
        ov.OffsetHigh = (DWORD)(offset >> 32);
        ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!ov.hEvent) throw std::runtime_error("cannot read file: " + path_);
        const DWORD want = (DWORD)std::min(bytes, size_t(1) << 30);
        DWORD got = 0;
        BOOL ok = ReadFile(file_, dst, want, nullptr, &ov);
        DWORD err = ok ? ERROR_SUCCESS : GetLastError();
        if (ok || err == ERROR_IO_PENDING) {
            ok = GetOverlappedResult(file_, &ov, &got, TRUE);
            err = ok ? ERROR_SUCCESS : GetLastError();
        }
        CloseHandle(ov.hEvent);
        if (err == ERROR_HANDLE_EOF) return 0;
        if (err != ERROR_SUCCESS) throw std::runtime_error("cannot read file: " + path_);
        return got;
#else
        for (;;) {
            const ssize_t n = pread(fd_, dst, bytes, (off_t)offset);
            if (n >= 0) return (size_t)n;
            if (errno != EINTR) throw std::runtime_error("cannot read file: " + path_);
        }
#endif
    }

    void close() {
#if defined(_WIN32)
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
        file_ = INVALID_HANDLE_VALUE;
#else
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
#endif
    }

    std::string path_;
    uint64_t size_ = 0;
    size_t granule_ = 0;
#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};

} // namespace format
