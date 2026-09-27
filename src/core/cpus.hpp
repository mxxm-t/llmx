#pragma once
#include <algorithm>
#include <charconv>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

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

// The whitespace-separated fields of a small file's text or line.
inline size_t fields(std::string_view text, std::string_view* out, size_t most) {
    size_t n = 0;
    for (size_t i = 0; i < text.size();) {
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t' || text[i] == '\n' || text[i] == '\r')) ++i;
        const size_t start = i;
        while (i < text.size() && text[i] != ' ' && text[i] != '\t' && text[i] != '\n' && text[i] != '\r') ++i;
        if (i == start) break;
        if (n == most) return most + 1;
        out[n++] = text.substr(start, i - start);
    }
    return n;
}

// Calls `f` with each line of `text`, the last one without its newline included.
template <class F>
void each_line(std::string_view text, F&& f) {
    for (size_t at = 0; at < text.size();) {
        size_t end = text.find('\n', at);
        if (end == std::string_view::npos) end = text.size();
        f(text.substr(at, end - at));
        at = end + 1;
    }
}

// Whether a comma-separated list holds `item`.
inline bool lists(std::string_view list, std::string_view item) {
    for (size_t at = 0; at <= list.size();) {
        size_t end = list.find(',', at);
        if (end == std::string_view::npos) end = list.size();
        if (list.substr(at, end - at) == item) return true;
        at = end + 1;
    }
    return false;
}

// A decimal number with nothing else in the field, no sign included.
inline std::optional<uint64_t> number(std::string_view field) {
    uint64_t v = 0;
    const auto r = std::from_chars(field.data(), field.data() + field.size(), v);
    if (r.ec != std::errc{} || r.ptr != field.data() + field.size()) return std::nullopt;
    return v;
}

// A quota of `quota` microseconds in each `period` as whole CPUs, rounded up.
inline std::optional<unsigned> quota_cpus(std::string_view quota, std::string_view period) {
    const auto q = number(quota), p = number(period);
    if (!q || !p || !*q || !*p) return std::nullopt;
    return (unsigned)std::min<uint64_t>(*q / *p + (*q % *p != 0), UINT_MAX);
}

// A cgroup path without its trailing slashes, when it is absolute and does not climb with "..".
inline std::optional<std::string> cgroup_path(std::string path) {
    if (path.empty() || path[0] != '/' || (path + "/").find("/../") != std::string::npos) return std::nullopt;
    while (path.size() > 1 && path.back() == '/') path.pop_back();
    return path;
}

// The path of the first "hierarchy:controllers:path" line `pick` accepts, when cgroup_path takes it.
template <class Pick>
std::optional<std::string> cgroup_line_path(std::string_view text, Pick pick) {
    std::optional<std::string> path;
    bool found = false;
    each_line(text, [&](std::string_view line) {
        const size_t a = line.find(':'), b = a == std::string_view::npos ? a : line.find(':', a + 1);
        if (found || b == std::string_view::npos || !pick(line.substr(0, a), line.substr(a + 1, b - a - 1))) return;
        found = true;
        path = cgroup_path(std::string(line.substr(b + 1)));
    });
    return path;
}

// A mountinfo field with its octal escapes ("\040" for a space) decoded.
inline std::string unescape(std::string_view field) {
    std::string out;
    for (size_t i = 0; i < field.size(); ++i) {
        const auto octal = [&](size_t k) { return field[k] >= '0' && field[k] <= '7'; };
        if (field[i] == '\\' && i + 3 < field.size() && octal(i + 1) && octal(i + 2) && octal(i + 3)) {
            out += (char)((field[i + 1] - '0') * 64 + (field[i + 2] - '0') * 8 + (field[i + 3] - '0'));
            i += 3;
        } else {
            out += field[i];
        }
    }
    return out;
}

#if defined(__linux__)
inline std::optional<std::string> read_text(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return std::nullopt;
    std::string text;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
    const bool failed = std::ferror(f) != 0;
    std::fclose(f);
    if (failed) return std::nullopt;
    return text;
}
#endif

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

// The process's cgroup v2 path in /proc/self/cgroup text, its "0::/path" line.
inline std::optional<std::string> cgroup_v2_path(std::string_view proc_self_cgroup) {
    return detail::cgroup_line_path(proc_self_cgroup, [](std::string_view hierarchy, std::string_view controllers) {
        return hierarchy == "0" && controllers.empty();
    });
}

// The process's cgroup v1 path for the cpu controller in /proc/self/cgroup text, from the first line whose controllers list cpu.
inline std::optional<std::string> cgroup_v1_cpu_path(std::string_view proc_self_cgroup) {
    return detail::cgroup_line_path(proc_self_cgroup, [](std::string_view, std::string_view controllers) {
        return detail::lists(controllers, "cpu");
    });
}

// A cgroup hierarchy's directory `root` mounted at `point`.
struct CgroupMount {
    std::string root, point;
};

namespace detail {

// The mounts in /proc/self/mountinfo text whose file system type and super options `pick` accepts, in the order listed, leaving out a root cgroup_path does not take.
template <class Pick>
std::vector<CgroupMount> cgroup_mounts(std::string_view mountinfo, Pick pick) {
    std::vector<CgroupMount> mounts;
    each_line(mountinfo, [&](std::string_view line) {
        std::string_view f[32];
        const size_t n = fields(line, f, 32);
        if (n > 32) return;
        size_t dash = 6;
        while (dash < n && f[dash] != "-") ++dash;
        if (dash + 3 >= n || !pick(f[dash + 1], f[dash + 3])) return;
        auto root = cgroup_path(unescape(f[3]));
        const std::string point = unescape(f[4]);
        if (root && !point.empty() && point[0] == '/') mounts.push_back({std::move(*root), point});
    });
    return mounts;
}

}  // namespace detail

// The cgroup v2 mounts in /proc/self/mountinfo text.
inline std::vector<CgroupMount> cgroup_v2_mounts(std::string_view mountinfo) {
    return detail::cgroup_mounts(mountinfo, [](std::string_view type, std::string_view) { return type == "cgroup2"; });
}

// The cgroup v1 mounts in /proc/self/mountinfo text whose hierarchy holds the cpu controller.
inline std::vector<CgroupMount> cgroup_v1_cpu_mounts(std::string_view mountinfo) {
    return detail::cgroup_mounts(mountinfo, [](std::string_view type, std::string_view options) {
        return type == "cgroup" && detail::lists(options, "cpu");
    });
}

// Where cgroup `path` lies under the first of `mounts` whose root holds it: that mount's point and the path below its root, empty at the root itself; nothing when no root holds it.
inline std::optional<std::pair<std::string, std::string>> cgroup_directory(const std::string& path, const std::vector<CgroupMount>& mounts) {
    for (const auto& m : mounts) {
        if (m.root == "/") return std::make_pair(m.point, path == "/" ? std::string() : path);
        if (path == m.root) return std::make_pair(m.point, std::string());
        if (path.size() > m.root.size() && path.compare(0, m.root.size(), m.root) == 0 && path[m.root.size()] == '/')
            return std::make_pair(m.point, path.substr(m.root.size()));
    }
    return std::nullopt;
}

// CPUs the cgroup CPU quotas over a process allow, the smallest, from its /proc/self/cgroup and /proc/self/mountinfo texts and `read`, which gives a file's text or nothing.
// The process's cgroup is found below its mount by taking the mount's root off its path, and a quota on any cgroup above it holds it too, so each is read up to the mount's point.
template <class Read>
std::optional<unsigned> cgroup_cpus(std::string_view proc_self_cgroup, std::string_view mountinfo, Read&& read) {
    std::optional<unsigned> fewest;
    const auto take = [&](std::optional<unsigned> n) { if (n && (!fewest || *n < *fewest)) fewest = n; };
    const auto walk = [](const std::optional<std::string>& path, const std::vector<CgroupMount>& mounts, auto&& at) {
        if (!path) return;
        const auto dir = cgroup_directory(*path, mounts);
        if (!dir) return;
        const std::string point = dir->first == "/" ? std::string() : dir->first;
        for (std::string below = dir->second;; below.erase(below.rfind('/'))) {
            at(point + below);
            if (below.empty()) return;
        }
    };
    walk(cgroup_v2_path(proc_self_cgroup), cgroup_v2_mounts(mountinfo), [&](const std::string& dir) {
        if (const auto max = read(dir + "/cpu.max")) take(cgroup_v2_cpus(*max));
    });
    walk(cgroup_v1_cpu_path(proc_self_cgroup), cgroup_v1_cpu_mounts(mountinfo), [&](const std::string& dir) {
        const auto quota = read(dir + "/cpu.cfs_quota_us");
        if (!quota) return;
        if (const auto period = read(dir + "/cpu.cfs_period_us")) take(cgroup_v1_cpus(*quota, *period));
    });
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
