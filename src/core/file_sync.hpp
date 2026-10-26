#pragma once
#include <atomic>
#include <cstdint>
#include <filesystem>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

// The one place a file or a directory is made durable, with the strongest call each system has (docs/src/core-file_sync.md).

namespace core {

#if defined(_WIN32)
using FileHandle = HANDLE;
#else
using FileHandle = int;
#endif

// How many durable calls this process has made, for tests that hold a caller to this owner.
inline std::atomic<uint64_t>& sync_calls() {
    static std::atomic<uint64_t> calls{0};
    return calls;
}

// An open file's bytes on the disk; macOS asks F_FULLFSYNC, since its fsync leaves them in the drive's cache.
// A directory asks for `whole`, its entries being metadata that fdatasync need not flush.
inline bool sync_handle(FileHandle h, bool whole = false) {
    sync_calls().fetch_add(1, std::memory_order_relaxed);
#if defined(_WIN32)
    (void)whole;
    return FlushFileBuffers(h) != 0;
#elif defined(__APPLE__)
    (void)whole;
    return fcntl(h, F_FULLFSYNC) == 0 || fsync(h) == 0;
#else
    return (whole ? fsync(h) : fdatasync(h)) == 0;
#endif
}

// The same for a file named by its path.
inline bool sync_file(const std::filesystem::path& file) {
#if defined(_WIN32)
    const HANDLE h = CreateFileW(file.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return false;
    const bool ok = sync_handle(h);
    CloseHandle(h);
    return ok;
#else
    const int fd = ::open(file.c_str(), O_WRONLY | O_CLOEXEC);
    if (fd < 0) return false;
    const bool ok = sync_handle(fd);
    ::close(fd);
    return ok;
#endif
}

// A rename reaches the disk with its directory; Windows keeps no directory to flush, so it does nothing there.
inline void sync_directory(const std::filesystem::path& dir) {
#if !defined(_WIN32)
    const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return;
    sync_handle(fd, true);
    ::close(fd);
#else
    (void)dir;
#endif
}

}  // namespace core
