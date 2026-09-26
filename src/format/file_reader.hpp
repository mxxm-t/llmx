#pragma once
// A file read at given offsets into the caller's memory, by several threads at once, through the file cache or around it (docs/src/format-file_reader.md).
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
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#endif

namespace format {

// A file whose file system does not take direct reads.
struct DirectUnavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

#if defined(__linux__)
// struct statx as the kernel lays it out (include/uapi/linux/stat.h), so the direct-I/O alignment is read whatever the C library's headers declare.
struct KernelStatx {
    uint32_t mask, blksize;
    uint64_t attributes;
    uint32_t nlink, uid, gid;
    uint16_t mode, spare0;
    uint64_t ino, size, blocks, attributes_mask;
    uint8_t times[4 * 16];
    uint32_t rdev_major, rdev_minor, dev_major, dev_minor;
    uint64_t mnt_id;
    uint32_t dio_mem_align, dio_offset_align;
    uint64_t spare[12];
};
static_assert(offsetof(KernelStatx, dio_mem_align) == 152 && sizeof(KernelStatx) == 256, "struct statx layout");
#endif

class FileReader {
public:
    // With `direct`, reads go around the file cache and must start, end and land on granule(); a file system that does not take them throws DirectUnavailable.
    explicit FileReader(const std::string& path, bool direct = false) : path_(path), direct_(direct) {
#if defined(_WIN32)
        const std::wstring wide = std::filesystem::u8path(path).wstring();
        const DWORD flags = FILE_FLAG_OVERLAPPED | (direct ? FILE_FLAG_NO_BUFFERING : FILE_FLAG_SEQUENTIAL_SCAN);
        file_ = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, flags, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open file: " + path);
        LARGE_INTEGER size;
        if (!GetFileSizeEx(file_, &size)) { close(); throw std::runtime_error("cannot size file: " + path); }
        size_ = (uint64_t)size.QuadPart;
        granule_ = core::page_size();
        if (direct) {
            FILE_STORAGE_INFO info{};
            if (!GetFileInformationByHandleEx(file_, FileStorageInfo, &info, sizeof(info))) {
                close();
                throw DirectUnavailable(path + " does not report its sector sizes");
            }
            granule_ = std::max({granule_, (size_t)info.LogicalBytesPerSector, (size_t)info.PhysicalBytesPerSectorForPerformance});
            // One aligned read proves the file takes unbuffered reads before anything depends on it.
            core::HostPages probe(granule_);
            try {
                read_some(0, probe.data(), granule_);
            } catch (const std::runtime_error&) {
                close();
                throw DirectUnavailable(path + " refused an unbuffered read");
            }
        }
#else
        int mode = O_RDONLY | O_CLOEXEC;
#if defined(__linux__)
        if (direct) mode |= O_DIRECT;
#else
        if (direct) throw DirectUnavailable(path + ": direct reads are not implemented on this system");
#endif
        fd_ = ::open(path.c_str(), mode);
        if (fd_ < 0) {
            if (direct && errno == EINVAL) throw DirectUnavailable(path + " is on a file system that does not take direct reads");
            throw std::runtime_error("cannot open file: " + path);
        }
        struct stat st {};
        if (fstat(fd_, &st) != 0) { close(); throw std::runtime_error("cannot size file: " + path); }
        size_ = (uint64_t)st.st_size;
        granule_ = std::max(core::page_size(), st.st_blksize > 0 ? (size_t)st.st_blksize : size_t(0));
#if defined(__linux__)
        if (direct) {
            // The alignment direct reads need, from statx (Linux 6.1 on); without it the file system's own rules are unknown, so direct reads are refused.
            KernelStatx sx{};
            const long rc = syscall(SYS_statx, fd_, "", AT_EMPTY_PATH, 0x00002000u /* STATX_DIOALIGN */, &sx);
            if (rc != 0 || !(sx.mask & 0x00002000u) || !sx.dio_mem_align || !sx.dio_offset_align || sx.dio_mem_align > core::page_size()) {
                close();
                throw DirectUnavailable(path + " is on a file system that does not take direct reads");
            }
            granule_ = std::max(core::page_size(), (size_t)sx.dio_offset_align);
        }
#endif
#if defined(POSIX_FADV_SEQUENTIAL)
        if (!direct) posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
#endif
    }
    ~FileReader() { close(); }
    FileReader(const FileReader&) = delete;
    FileReader& operator=(const FileReader&) = delete;

    const std::string& path() const { return path_; }
    uint64_t size() const { return size_; }
    bool direct() const { return direct_; }
    // The unit a read starts and ends on: for buffered reads the larger of the page and the file's block size, which is the recordsize on ZFS; for direct reads the alignment the file system requires, at least a page.
    size_t granule() const { return granule_; }

    // Read up to `bytes` from `offset` into `dst` and return how many were read, fewer only at the end of the file.
    // A direct read must start and end on granule() and land on a page; one that does not is a caller's error, since some file systems would serve it through their cache instead of refusing it.
    size_t read(uint64_t offset, void* dst, size_t bytes) const {
        if (direct_ && (offset % granule_ || bytes % granule_ || (uintptr_t)dst % core::page_size()))
            throw std::logic_error("direct read of " + path_ + " not on its alignment");
        size_t done = 0;
        while (done < bytes) {
            const size_t n = read_some(offset + done, (uint8_t*)dst + done, bytes - done);
            if (!n) break;
            done += n;
            // A direct read comes back short only at the end of the file, and the next would not be aligned.
            if (direct_ && done % granule_) break;
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
    bool direct_ = false;
    uint64_t size_ = 0;
    size_t granule_ = 0;
#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};

} // namespace format
