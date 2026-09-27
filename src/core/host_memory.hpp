#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "core/cgroup.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

// What memory this process can still take, as its operating system and its memory limits give it now, the page its memory comes in, and pages the process owns (docs/src/core-host_memory.md).

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

// Bytes of physical memory the whole host has available without swapping: what Windows reports as available, what Linux reports as MemAvailable (free memory plus reclaimable page cache); nothing when neither can be read.
inline std::optional<size_t> system_memory_available() {
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

namespace detail {

// The one decimal number a small file's text holds; nothing when it holds anything else.
inline std::optional<uint64_t> file_number(std::string_view text) {
    std::string_view f[1];
    if (fields(text, f, 1) != 1) return std::nullopt;
    return number(f[0]);
}

// A limit less what is charged against it, 0 past it.
inline size_t memory_room(uint64_t limit, uint64_t used) { return (size_t)(limit > used ? limit - used : 0); }

// The value on the line of a memory.stat text that `key` starts; nothing when no line does or its value is not one decimal number.
inline std::optional<uint64_t> stat_value(std::string_view stat, std::string_view key) {
    std::optional<uint64_t> value;
    bool found = false;
    each_line(stat, [&](std::string_view line) {
        std::string_view f[2];
        const size_t n = fields(line, f, 2);
        if (found || !n || f[0] != key) return;
        found = true;
        if (n == 2) value = number(f[1]);
    });
    return value;
}

// A cgroup's limit less its working set, 0 past the limit: its usage less the inactive file pages its memory.stat gives under `key`, which the kernel reclaims before it would kill the process, or the usage whole when the stat does not give them.
inline std::optional<size_t> cgroup_room(std::optional<uint64_t> limit, std::optional<uint64_t> usage, std::string_view stat, std::string_view key) {
    if (!limit || !usage) return std::nullopt;
    const auto inactive = stat_value(stat, key);
    return memory_room(*limit, inactive ? (*usage > *inactive ? *usage - *inactive : 0) : *usage);
}

}  // namespace detail

// Bytes a cgroup v2 memory.max text leaves over its memory.current and memory.stat texts, the limit less the usage less the stat's inactive_file, and 0 past it; the usage whole when the stat, empty for none, does not give inactive_file; nothing for "max", no limit, or a limit or usage it cannot read.
inline std::optional<size_t> cgroup_v2_memory_room(std::string_view memory_max, std::string_view memory_current, std::string_view memory_stat) {
    return detail::cgroup_room(detail::file_number(memory_max), detail::file_number(memory_current), memory_stat, "inactive_file");
}

// Bytes a cgroup v1 memory.limit_in_bytes text leaves over its memory.usage_in_bytes and memory.stat texts the same way, with the stat's total_inactive_file, which covers the cgroups under it as the usage does; nothing for a limit of 2^62 bytes or more, which is how v1 writes no limit.
inline std::optional<size_t> cgroup_v1_memory_room(std::string_view limit_in_bytes, std::string_view usage_in_bytes, std::string_view memory_stat) {
    const auto limit = detail::file_number(limit_in_bytes);
    if (limit && *limit >= (uint64_t(1) << 62)) return std::nullopt;
    return detail::cgroup_room(limit, detail::file_number(usage_in_bytes), memory_stat, "total_inactive_file");
}

// Bytes the cgroup memory limits over a process leave it, the fewest, from its /proc/self/cgroup and /proc/self/mountinfo texts and `read`, which gives a file's text or nothing.
// A limit on any cgroup above the process's holds it too, so each is read up to its mount's point with its own memory.stat; a cgroup whose limit or usage cannot be read is passed over.
template <class Read>
std::optional<size_t> cgroup_memory_room(std::string_view proc_self_cgroup, std::string_view mountinfo, Read&& read) {
    std::optional<size_t> fewest;
    const auto take = [&](std::optional<size_t> n) { if (n && (!fewest || *n < *fewest)) fewest = n; };
    const auto stat = [&](const std::string& dir) { return read(dir + "/memory.stat").value_or(std::string()); };
    for (const auto& dir : cgroup_v2_directories(proc_self_cgroup, mountinfo)) {
        const auto max = read(dir + "/memory.max");
        if (!max) continue;
        if (const auto current = read(dir + "/memory.current")) take(cgroup_v2_memory_room(*max, *current, stat(dir)));
    }
    for (const auto& dir : cgroup_v1_directories(proc_self_cgroup, mountinfo, "memory")) {
        const auto limit = read(dir + "/memory.limit_in_bytes");
        if (!limit) continue;
        if (const auto usage = read(dir + "/memory.usage_in_bytes")) take(cgroup_v1_memory_room(*limit, *usage, stat(dir)));
    }
    return fewest;
}

// Bytes this process's cgroup memory limits leave it, on Linux; nothing elsewhere, without a limit or when the files cannot be read.
inline std::optional<size_t> cgroup_memory_room() {
#if defined(__linux__)
    const auto cgroup = detail::read_text("/proc/self/cgroup"), mountinfo = detail::read_text("/proc/self/mountinfo");
    if (!cgroup || !mountinfo) return std::nullopt;
    return cgroup_memory_room(*cgroup, *mountinfo, detail::read_text);
#else
    return std::nullopt;
#endif
}

// A job object's memory limit flags that job_memory_room reads, with the values Windows gives them.
inline constexpr uint32_t job_limit_process_memory = 0x100, job_limit_job_memory = 0x200;
#if defined(_WIN32)
static_assert(job_limit_process_memory == JOB_OBJECT_LIMIT_PROCESS_MEMORY && job_limit_job_memory == JOB_OBJECT_LIMIT_JOB_MEMORY);
#endif

// Bytes a job object's memory limits leave a process: with the process memory limit flag, `process_limit` less the process's committed `process_commit`, with the job memory limit flag, `job_limit` less the job's committed `job_commit`, the fewer when both, 0 past a limit; nothing without either flag.
inline std::optional<size_t> job_memory_room(uint32_t limit_flags, uint64_t process_limit, uint64_t process_commit, uint64_t job_limit, uint64_t job_commit) {
    std::optional<size_t> fewest;
    const auto take = [&](size_t n) { if (!fewest || n < *fewest) fewest = n; };
    if (limit_flags & job_limit_process_memory) take(detail::memory_room(process_limit, process_commit));
    if (limit_flags & job_limit_job_memory) take(detail::memory_room(job_limit, job_commit));
    return fewest;
}

// Bytes the memory limits of this process's own job object leave it, on Windows; nothing elsewhere, outside a job or without a memory limit.
// A limit whose commit cannot be read is passed over.
inline std::optional<size_t> job_memory_room() {
#if defined(_WIN32)
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    if (!QueryInformationJobObject(nullptr, JobObjectExtendedLimitInformation, &limits, sizeof limits, nullptr)) return std::nullopt;
    uint32_t flags = limits.BasicLimitInformation.LimitFlags & (job_limit_process_memory | job_limit_job_memory);
    PROCESS_MEMORY_COUNTERS process{};
    if ((flags & job_limit_process_memory) && !GetProcessMemoryInfo(GetCurrentProcess(), &process, sizeof process)) flags &= ~job_limit_process_memory;
    JOBOBJECT_LIMIT_VIOLATION_INFORMATION job{};
    if ((flags & job_limit_job_memory) && !QueryInformationJobObject(nullptr, JobObjectLimitViolationInformation, &job, sizeof job, nullptr))
        flags &= ~job_limit_job_memory;
    return job_memory_room(flags, limits.ProcessMemoryLimit, process.PagefileUsage, limits.JobMemoryLimit, job.JobMemory);
#else
    return std::nullopt;
#endif
}

// Bytes this process's memory limits leave it: its cgroup memory limits on Linux, its job object's memory limits on Windows.
inline std::optional<size_t> memory_limit_room() {
#if defined(_WIN32)
    return job_memory_room();
#else
    return cgroup_memory_room();
#endif
}

// What a process can still take: the fewer of the host's available memory and the room its memory limits leave, each only when it was read; nothing when neither was.
inline std::optional<size_t> host_memory_available(std::optional<size_t> host, std::optional<size_t> room) {
    if (host && room) return *host < *room ? host : room;
    return host ? host : room;
}

// Bytes of memory this process can still take without swapping or passing a memory limit, read now (docs/src/core-host_memory.md).
inline std::optional<size_t> host_memory_available() { return host_memory_available(system_memory_available(), memory_limit_room()); }

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
