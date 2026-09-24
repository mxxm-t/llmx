#pragma once
#include <cstddef>
#include <cstdio>
#include <cstring>

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

// What the host can still give a process, as its operating system reports it now (docs/src/core-host_memory.md).

namespace core {

// Bytes of physical memory available without swapping: what Windows reports as available, what Linux reports as MemAvailable (free memory plus reclaimable page cache); 0 when neither can be read.
inline size_t host_memory_available() {
#if defined(_WIN32)
    MEMORYSTATUSEX s{};
    s.dwLength = sizeof(s);
    return GlobalMemoryStatusEx(&s) ? (size_t)s.ullAvailPhys : 0;
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
    const long pages = sysconf(_SC_AVPHYS_PAGES), page = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page > 0) return (size_t)pages * (size_t)page;
#endif
    return 0;
#endif
}

} // namespace core
