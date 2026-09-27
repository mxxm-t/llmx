#pragma once
#include <algorithm>
#include <climits>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "core/cgroup.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <cerrno>
#include <sched.h>
#endif

// The CPUs a process may use and the worker count a pool takes when given none (docs/src/core-cpus.md).

namespace core {

namespace detail {

// A quota of `quota` microseconds in each `period` as whole CPUs, rounded up.
inline std::optional<unsigned> quota_cpus(std::string_view quota, std::string_view period) {
    const auto q = number(quota), p = number(period);
    if (!q || !p || !*q || !*p) return std::nullopt;
    return (unsigned)std::min<uint64_t>(*q / *p + (*q % *p != 0), UINT_MAX);
}

}  // namespace detail

// CPUs a cgroup v2 cpu.max text allows ("600000 100000" is 6, "150000 100000" rounds up to 2); nothing for "max", no limit, or a text it cannot read.
inline std::optional<unsigned> cgroup_v2_cpus(std::string_view cpu_max) {
    std::string_view f[2];
    if (detail::fields(cpu_max, f, 2) != 2 || f[0] == "max") return std::nullopt;
    return detail::quota_cpus(f[0], f[1]);
}

// CPUs a cgroup v1 cpu.cfs_quota_us text allows over its cpu.cfs_period_us text, rounded up; nothing for a quota of -1, no limit, or a text it cannot read.
inline std::optional<unsigned> cgroup_v1_cpus(std::string_view cfs_quota_us, std::string_view cfs_period_us) {
    std::string_view q[1], p[1];
    if (detail::fields(cfs_quota_us, q, 1) != 1 || detail::fields(cfs_period_us, p, 1) != 1) return std::nullopt;
    return detail::quota_cpus(q[0], p[0]);
}

// CPUs the cgroup CPU quotas over a process allow, the smallest, from its /proc/self/cgroup and /proc/self/mountinfo texts and `read`, which gives a file's text or nothing.
// The process's cgroup is found below its mount by taking the mount's root off its path, and a quota on any cgroup above it holds it too, so each is read up to the mount's point.
template <class Read>
std::optional<unsigned> cgroup_cpus(std::string_view proc_self_cgroup, std::string_view mountinfo, Read&& read) {
    std::optional<unsigned> fewest;
    const auto take = [&](std::optional<unsigned> n) { if (n && (!fewest || *n < *fewest)) fewest = n; };
    for (const auto& dir : cgroup_v2_directories(proc_self_cgroup, mountinfo))
        if (const auto max = read(dir + "/cpu.max")) take(cgroup_v2_cpus(*max));
    for (const auto& dir : cgroup_v1_directories(proc_self_cgroup, mountinfo, "cpu")) {
        const auto quota = read(dir + "/cpu.cfs_quota_us");
        if (!quota) continue;
        if (const auto period = read(dir + "/cpu.cfs_period_us")) take(cgroup_v1_cpus(*quota, *period));
    }
    return fewest;
}

// CPUs this process's cgroup CPU quotas allow, on Linux; nothing elsewhere, without a limit or when the files cannot be read.
inline std::optional<unsigned> cgroup_cpus() {
#if defined(__linux__)
    const auto cgroup = detail::read_text("/proc/self/cgroup"), mountinfo = detail::read_text("/proc/self/mountinfo");
    if (!cgroup || !mountinfo) return std::nullopt;
    return cgroup_cpus(*cgroup, *mountinfo, detail::read_text);
#else
    return std::nullopt;
#endif
}

// A job object's CPU rate control flags that job_cpus reads, with the values Windows gives them.
inline constexpr uint32_t job_rate_enable = 0x1, job_rate_hard_cap = 0x4;
#if defined(_WIN32)
static_assert(job_rate_enable == JOB_OBJECT_CPU_RATE_CONTROL_ENABLE && job_rate_hard_cap == JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP);
#endif

// CPUs a job object's CPU rate control allows over `processors` active processors: an enabled hard cap of `rate` cycles in each 10000 as whole CPUs, rounded up; nothing without one, or for a rate outside 1 to 10000.
inline std::optional<unsigned> job_cpus(uint32_t control_flags, uint32_t rate, unsigned processors) {
    if ((control_flags & (job_rate_enable | job_rate_hard_cap)) != (job_rate_enable | job_rate_hard_cap) || !rate || rate > 10000 || !processors)
        return std::nullopt;
    const uint64_t cycles = (uint64_t)rate * processors;
    return (unsigned)(cycles / 10000 + (cycles % 10000 != 0));
}

// CPUs the CPU rate hard cap of this process's own job object allows, on Windows; nothing elsewhere, outside a job or without a cap.
inline std::optional<unsigned> job_cpus() {
#if defined(_WIN32)
    JOBOBJECT_CPU_RATE_CONTROL_INFORMATION info{};
    if (!QueryInformationJobObject(nullptr, JobObjectCpuRateControlInformation, &info, sizeof info, nullptr)) return std::nullopt;
    return job_cpus(info.ControlFlags, info.CpuRate, GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));
#else
    return std::nullopt;
#endif
}

// CPUs this process's CPU quota allows: its cgroup CPU quotas on Linux, its job object's CPU rate hard cap on Windows.
inline std::optional<unsigned> quota_cpus() {
#if defined(_WIN32)
    return job_cpus();
#else
    return cgroup_cpus();
#endif
}

#if defined(_WIN32)
// The one processor group all of this process's threads lie in, the process affinity mask in it and the group's active processors; nothing when the process spans several groups or the masks cannot be read.
struct GroupAffinity {
    USHORT group = 0;
    DWORD_PTR allowed = 0, system = 0;
};
inline std::optional<GroupAffinity> group_affinity() {
    GroupAffinity a;
    USHORT count = 1;
    if (!GetProcessGroupAffinity(GetCurrentProcess(), &count, &a.group) || count != 1 ||
        !GetProcessAffinityMask(GetCurrentProcess(), &a.allowed, &a.system) || !a.allowed)
        return std::nullopt;
    return a;
}
#endif

// CPUs this process's affinity allows: sched_getaffinity on Linux; on Windows the process affinity mask when its threads lie in one processor group, else the active processors of the groups they lie in; nothing elsewhere or when it cannot be read.
inline std::optional<unsigned> affinity_cpus() {
#if defined(_WIN32)
    unsigned n = 0;
    if (const auto one = group_affinity()) {
        for (DWORD_PTR m = one->allowed; m; m &= m - 1) ++n;
        return n;
    }
    // Threads in several groups get no process mask, so the count is every active processor of those groups.
    std::vector<USHORT> groups(std::max<WORD>(GetActiveProcessorGroupCount(), 1));
    USHORT count = (USHORT)groups.size();
    if (!GetProcessGroupAffinity(GetCurrentProcess(), &count, groups.data()) || count < 2) return std::nullopt;
    for (USHORT i = 0; i < count; ++i) n += GetActiveProcessorCount(groups[i]);
    if (!n) return std::nullopt;
    return n;
#elif defined(__linux__)
    // A kernel with more possible CPUs than the mask holds refuses it with EINVAL, so the mask doubles until it fits.
    for (int size = CPU_SETSIZE; size <= (1 << 20); size *= 2) {
        cpu_set_t* set = CPU_ALLOC(size);
        if (!set) return std::nullopt;
        const size_t bytes = CPU_ALLOC_SIZE(size);
        CPU_ZERO_S(bytes, set);
        const int r = sched_getaffinity(0, bytes, set);
        const int error = errno;
        const int n = r == 0 ? CPU_COUNT_S(bytes, set) : 0;
        CPU_FREE(set);
        if (r == 0) return n > 0 ? std::optional<unsigned>((unsigned)n) : std::nullopt;
        if (error != EINVAL) return std::nullopt;
    }
    return std::nullopt;
#else
    return std::nullopt;
#endif
}

// The worker count a pool takes when given none: the fewest of the hardware threads, the CPUs the affinity allows and the CPUs the CPU quota allows, each only when it was read, 4 when none was, from 1 to 64.
inline int automatic_threads(std::optional<unsigned> hardware, std::optional<unsigned> affinity, std::optional<unsigned> quota) {
    std::optional<unsigned> fewest;
    for (const auto& n : {hardware, affinity, quota})
        if (n && *n && (!fewest || *n < *fewest)) fewest = n;
    return (int)std::clamp(fewest.value_or(4u), 1u, 64u);
}

// This process's automatic worker count, read now.
inline int automatic_threads() {
    const unsigned hardware = std::thread::hardware_concurrency();
    return automatic_threads(hardware ? std::optional<unsigned>(hardware) : std::nullopt, affinity_cpus(), quota_cpus());
}

}  // namespace core
