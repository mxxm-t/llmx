#pragma once
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include "core/host_memory.hpp"
#include "format/file_reader.hpp"

// A new file written at offsets, through the file cache or around it, flushed before it is published; the disk tier's entries are written through it (docs/DISK-TIER.md).

namespace format {

class FileWriter {
public:
    // Creates `path`, which must not exist, readable and writable by its owner alone.
    // With `direct`, writes go around the file cache, start and end on granule() and come from a page; a file system that does not take them throws DirectUnavailable and leaves no file.
    FileWriter(const std::string& path, bool direct) : path_(path), direct_(direct) {
#if defined(_WIN32)
        const std::wstring wide = std::filesystem::u8path(path).wstring();
        const DWORD flags = FILE_FLAG_OVERLAPPED | (direct ? FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH : FILE_ATTRIBUTE_NORMAL);
        file_ = CreateFileW(wide.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW, flags, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot create file: " + path + " (" + std::to_string(GetLastError()) + ")");
        granule_ = core::page_size();
        if (direct) {
            FILE_STORAGE_INFO info{};
            if (!GetFileInformationByHandleEx(file_, FileStorageInfo, &info, sizeof(info))) {
                discard();
                throw DirectUnavailable(path + " does not report its sector sizes");
            }
            granule_ = std::max({granule_, (size_t)info.LogicalBytesPerSector, (size_t)info.PhysicalBytesPerSectorForPerformance});
        }
#else
        int mode = O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC;
#if defined(__linux__)
        if (direct) mode |= O_DIRECT;
#else
        if (direct) throw DirectUnavailable(path + ": direct writes are not implemented on this system");
#endif
        fd_ = ::open(path.c_str(), mode, 0600);
        if (fd_ < 0) {
            if (direct && errno == EINVAL) throw DirectUnavailable(path + " is on a file system that does not take direct writes");
            throw std::runtime_error("cannot create file: " + path + " (" + std::strerror(errno) + ")");
        }
        struct stat st {};
        granule_ = std::max(core::page_size(), fstat(fd_, &st) == 0 && st.st_blksize > 0 ? (size_t)st.st_blksize : size_t(0));
#if defined(__linux__)
        if (direct) {
            KernelStatx sx{};
            const long rc = syscall(SYS_statx, fd_, "", AT_EMPTY_PATH, 0x00002000u /* STATX_DIOALIGN */, &sx);
            if (rc != 0 || !(sx.mask & 0x00002000u) || !sx.dio_mem_align || !sx.dio_offset_align || sx.dio_mem_align > core::page_size()) {
                discard();
                throw DirectUnavailable(path + " is on a file system that does not take direct writes");
            }
            granule_ = std::max(core::page_size(), (size_t)sx.dio_offset_align);
        }
#endif
#endif
    }
    ~FileWriter() { close(); }
    FileWriter(const FileWriter&) = delete;
    FileWriter& operator=(const FileWriter&) = delete;

    // The unit a direct write starts and ends on, at least a page.
    size_t granule() const { return granule_; }

    // Writes `bytes` from `src` at `offset`; a direct write must start and end on granule() and come from a page.
    // A write that fails, the disk full among them, throws naming the file and the system's reason.
    void write(uint64_t offset, const void* src, size_t bytes) {
        if (direct_ && (offset % granule_ || bytes % granule_ || (uintptr_t)src % core::page_size()))
            throw std::logic_error("direct write of " + path_ + " not on its alignment");
        const auto* p = static_cast<const uint8_t*>(src);
        while (bytes) {
            const size_t n = write_some(offset, p, std::min(bytes, size_t(1) << 30));
            offset += n;
            p += n;
            bytes -= n;
        }
    }

    // Everything written reaches the disk before this returns.
    void sync() {
#if defined(_WIN32)
        if (!FlushFileBuffers(file_)) throw std::runtime_error("cannot flush file: " + path_);
#elif defined(__APPLE__)
        // fsync leaves the bytes in the drive's cache on macOS; F_FULLFSYNC flushes it, and fsync stands in where the file system refuses it.
        if (fcntl(fd_, F_FULLFSYNC) != 0 && fsync(fd_) != 0) throw std::runtime_error("cannot flush file: " + path_ + " (" + std::strerror(errno) + ")");
#else
        if (fdatasync(fd_) != 0) throw std::runtime_error("cannot flush file: " + path_ + " (" + std::strerror(errno) + ")");
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

private:
    // Closes and removes the file, for a writer refused as it is made.
    void discard() {
        close();
        std::error_code ec;
        std::filesystem::remove(std::filesystem::u8path(path_), ec);
    }

    size_t write_some(uint64_t offset, const uint8_t* src, size_t bytes) {
#if defined(_WIN32)
        OVERLAPPED ov{};
        ov.Offset = (DWORD)offset;
        ov.OffsetHigh = (DWORD)(offset >> 32);
        ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (!ov.hEvent) throw std::runtime_error("cannot write file: " + path_);
        DWORD done = 0;
        BOOL ok = WriteFile(file_, src, (DWORD)bytes, nullptr, &ov);
        DWORD err = ok ? ERROR_SUCCESS : GetLastError();
        if (ok || err == ERROR_IO_PENDING) {
            ok = GetOverlappedResult(file_, &ov, &done, TRUE);
            err = ok ? ERROR_SUCCESS : GetLastError();
        }
        CloseHandle(ov.hEvent);
        if (err != ERROR_SUCCESS || !done) throw std::runtime_error("cannot write file: " + path_ + " (" + std::to_string(err) + ")");
        return done;
#else
        for (;;) {
            const ssize_t n = pwrite(fd_, src, bytes, (off_t)offset);
            if (n > 0) return (size_t)n;
            if (n < 0 && errno == EINTR) continue;
            throw std::runtime_error("cannot write file: " + path_ + " (" + std::strerror(n < 0 ? errno : ENOSPC) + ")");
        }
#endif
    }

    std::string path_;
    bool direct_ = false;
    size_t granule_ = 4096;
#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};

} // namespace format
