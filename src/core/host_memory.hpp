#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

// What the host can still give a process, as its operating system reports it now, the page its memory comes in, and pages the process owns (docs/src/core-host_memory.md).

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

// Page-aligned memory the process owns, rounded up to whole pages and given back when it goes.
// A read into it can go around the file cache, and a device copies out of it as out of any host memory (docs/src/core-host_memory.md).
class HostPages {
public:
    HostPages() = default;
    explicit HostPages(size_t bytes) : size_((bytes + page_size() - 1) / page_size() * page_size()) {
        if (!size_) return;
#if defined(_WIN32)
        data_ = (uint8_t*)VirtualAlloc(nullptr, size_, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
        void* p = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        data_ = p == MAP_FAILED ? nullptr : (uint8_t*)p;
#endif
        if (!data_) throw std::runtime_error("cannot allocate " + std::to_string(size_) + " bytes of host pages");
    }
    ~HostPages() { release(); }
    HostPages(HostPages&& o) noexcept : data_(std::exchange(o.data_, nullptr)), size_(std::exchange(o.size_, 0)) {}
    HostPages& operator=(HostPages&& o) noexcept {
        if (this != &o) {
            release();
            data_ = std::exchange(o.data_, nullptr);
            size_ = std::exchange(o.size_, 0);
        }
        return *this;
    }
    HostPages(const HostPages&) = delete;
    HostPages& operator=(const HostPages&) = delete;

    uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

    // `bytes` of address space, rounded up to whole pages, with no memory behind it until commit gives some: a layout whose parts are filled one by one, such as the direct load's copy of the weights a host reads.
    static HostPages reserved(size_t bytes) {
        HostPages p;
        p.size_ = (bytes + page_size() - 1) / page_size() * page_size();
        if (!p.size_) return p;
#if defined(_WIN32)
        p.data_ = (uint8_t*)VirtualAlloc(nullptr, p.size_, MEM_RESERVE, PAGE_NOACCESS);
#else
        void* q = mmap(nullptr, p.size_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        p.data_ = q == MAP_FAILED ? nullptr : (uint8_t*)q;
#endif
        if (!p.data_) throw std::runtime_error("cannot reserve " + std::to_string(p.size_) + " bytes of address space");
        return p;
    }

    // Give memory to the whole pages covering `bytes` from `offset`, which must lie inside; committing a page twice is harmless.
    void commit(size_t offset, size_t bytes) {
        if (!bytes) return;
        if (offset > size_ || bytes > size_ - offset) throw std::logic_error("host pages: commit outside the range");
        const size_t page = page_size(), lo = offset / page * page, hi = (offset + bytes + page - 1) / page * page;
#if defined(_WIN32)
        const bool ok = VirtualAlloc(data_ + lo, hi - lo, MEM_COMMIT, PAGE_READWRITE) != nullptr;
#else
        const bool ok = mprotect(data_ + lo, hi - lo, PROT_READ | PROT_WRITE) == 0;
#endif
        if (!ok) throw std::runtime_error("cannot commit " + std::to_string(hi - lo) + " bytes of host pages");
    }

    // Take back the memory of the whole pages inside `bytes` from `offset`, keeping the address space reserved; a page the range covers only in part keeps its memory.
    void decommit(size_t offset, size_t bytes) {
        if (offset > size_ || bytes > size_ - offset) throw std::logic_error("host pages: decommit outside the range");
        const size_t page = page_size(), lo = (offset + page - 1) / page * page, hi = (offset + bytes) / page * page;
        if (hi <= lo) return;
#if defined(_WIN32)
        VirtualFree(data_ + lo, hi - lo, MEM_DECOMMIT);
#else
        madvise(data_ + lo, hi - lo, MADV_DONTNEED);
        mprotect(data_ + lo, hi - lo, PROT_NONE);
#endif
    }

private:
    void release() noexcept {
        if (!data_) return;
#if defined(_WIN32)
        VirtualFree(data_, 0, MEM_RELEASE);
#else
        munmap(data_, size_);
#endif
        data_ = nullptr;
    }

    uint8_t* data_ = nullptr;
    size_t size_ = 0;
};

} // namespace core
