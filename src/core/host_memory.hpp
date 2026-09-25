#pragma once
#include <cstddef>
#include <cstdio>
#include <optional>
#include <stdexcept>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

// What the host can still give a process, as its operating system reports it now, and the page its memory comes in (docs/src/core-host_memory.md).

namespace core {

// The size of a page of memory, which the operating system maps files and moves memory in.
// It throws when the operating system gives no positive size, since every use steps or multiplies by it.
inline size_t page_size() {
    static const size_t page = [] {
#if defined(_WIN32)
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        const long long size = si.dwPageSize;
#else
        const long long size = sysconf(_SC_PAGESIZE);
#endif
        if (size <= 0) throw std::runtime_error("cannot read the size of a memory page");
        return (size_t)size;
    }();
    return page;
}

// Bytes of physical memory available without swapping: what Windows reports as available, what Linux reports as MemAvailable (free memory plus reclaimable page cache); nothing when neither can be read.
inline std::optional<size_t> host_memory_available() {
#if defined(_WIN32)
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof(s);
    if (GlobalMemoryStatusEx(&s)) return (size_t)s.ullAvailPhys;
    return std::nullopt;
#else
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (f) {
        char line[256];
        unsigned long long kb = 0;
        bool found = false;
        while (!found && std::fgets(line, sizeof line, f))
            found = std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1;
        std::fclose(f);
        if (found) return (size_t)kb * 1024;
    }
#if defined(_SC_AVPHYS_PAGES)
    const long pages = sysconf(_SC_AVPHYS_PAGES);
    if (pages > 0) return (size_t)pages * page_size();
#endif
    return std::nullopt;
#endif
}

} // namespace core
