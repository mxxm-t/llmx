// The automatic worker count and the CPU quotas it reads, checked on file texts and field values rather than a real cgroup or job object.
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/cpus.hpp"

namespace {

size_t checks = 0;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
    ++checks;
}

std::string shown(const std::optional<unsigned>& n) { return n ? std::to_string(*n) : "nothing"; }
std::string shown(const std::optional<std::string>& s) { return s ? "\"" + *s + "\"" : "nothing"; }
bool same(const std::vector<core::CgroupMount>& a, const std::vector<core::CgroupMount>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].root != b[i].root || a[i].point != b[i].point) return false;
    return true;
}
std::string shown(const std::vector<core::CgroupMount>& mounts) {
    std::string out = "[";
    for (const auto& m : mounts) out += (out.size() > 1 ? ", " : "") + m.root + " at " + m.point;
    return out + "]";
}

void v2_quota() {
    const std::pair<const char*, std::optional<unsigned>> cases[] = {
        {"max 100000\n", std::nullopt}, {"max 100000", std::nullopt}, {"max", std::nullopt},
        {"600000 100000\n", 6u}, {"150000 100000\n", 2u}, {"100000 100000\n", 1u}, {"100001 100000\n", 2u},
        {"50000 100000\n", 1u}, {"1000 100000\n", 1u}, {"1600000 100000\n", 16u}, {"  600000   100000  \n", 6u},
        {"", std::nullopt}, {"\n", std::nullopt}, {"600000\n", std::nullopt}, {"600000 0\n", std::nullopt},
        {"0 100000\n", std::nullopt}, {"-1 100000\n", std::nullopt}, {"600000 100000 1\n", std::nullopt},
        {"6e5 100000\n", std::nullopt}, {"600000 +100000\n", std::nullopt}, {"600000x 100000\n", std::nullopt},
        {"99999999999999999999999 100000\n", std::nullopt},
    };
    for (const auto& c : cases)
        require(core::cgroup_v2_cpus(c.first) == c.second,
                std::string("cpu.max \"") + c.first + "\" gave " + shown(core::cgroup_v2_cpus(c.first)) + ", expected " + shown(c.second));
}

void v1_quota() {
    struct Case { const char* quota; const char* period; std::optional<unsigned> cpus; };
    const Case cases[] = {
        {"-1\n", "100000\n", std::nullopt}, {"600000\n", "100000\n", 6u}, {"150000\n", "100000\n", 2u},
        {"50000\n", "100000\n", 1u}, {"200000", "50000", 4u},
        {"", "100000\n", std::nullopt}, {"600000\n", "", std::nullopt}, {"600000\n", "0\n", std::nullopt},
        {"0\n", "100000\n", std::nullopt}, {"-2\n", "100000\n", std::nullopt}, {"max\n", "100000\n", std::nullopt},
        {"600000 1\n", "100000\n", std::nullopt},
    };
    for (const auto& c : cases)
        require(core::cgroup_v1_cpus(c.quota, c.period) == c.cpus,
                std::string("cfs quota \"") + c.quota + "\" over period \"" + c.period + "\" gave " +
                shown(core::cgroup_v1_cpus(c.quota, c.period)) + ", expected " + shown(c.cpus));
}

void paths() {
    struct Case { const char* text; std::optional<std::string> v2, v1; };
    const char* container_v1 =
        "12:pids:/containers/abc\n11:cpu,cpuacct:/containers/abc\n10:cpuset:/containers/abc\n"
        "9:cpuacct:/elsewhere\n1:name=init:/containers/abc\n0::/system.slice/runtime.service\n";
    const Case cases[] = {
        {"0::/user.slice/user-0.slice/session-32436.scope\n", "/user.slice/user-0.slice/session-32436.scope", std::nullopt},
        {"0::/\n", "/", std::nullopt},
        {"0::/a:b\n", "/a:b", std::nullopt},
        {"0::/a/b/\n", "/a/b", std::nullopt},
        {container_v1, "/system.slice/runtime.service", "/containers/abc"},
        {"4:cpuacct,cpu:/pods/p1\n", std::nullopt, "/pods/p1"},
        {"4:cpuset:/x\n5:cpuacct:/y\n", std::nullopt, std::nullopt},
        {"1:cpu:/only\n", std::nullopt, "/only"},
        {"1:cpu:/first\n2:cpu,cpuacct:/second\n", std::nullopt, "/first"},
        {"1:cpu:relative\n2:cpu,cpuacct:/second\n", std::nullopt, std::nullopt},
        {"0::/../outside\n", std::nullopt, std::nullopt},
        {"0::relative\n", std::nullopt, std::nullopt},
        {"0:\n", std::nullopt, std::nullopt},
        {"", std::nullopt, std::nullopt},
        {"0::/last-line-without-newline", "/last-line-without-newline", std::nullopt},
    };
    for (const auto& c : cases) {
        require(core::cgroup_v2_path(c.text) == c.v2,
                std::string("v2 path of \"") + c.text + "\" gave " + shown(core::cgroup_v2_path(c.text)) + ", expected " + shown(c.v2));
        require(core::cgroup_v1_path(c.text, "cpu") == c.v1,
                std::string("v1 cpu path of \"") + c.text + "\" gave " + shown(core::cgroup_v1_path(c.text, "cpu")) + ", expected " + shown(c.v1));
    }
}

// Mountinfo lines as the kernel writes them, with the names changed.
const char* const v2_own_namespace = "1340 1321 0:26 / /sys/fs/cgroup ro,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw,nsdelegate\n";
const char* const v1_container =
    "30 25 0:27 /containers/abc /sys/fs/cgroup/cpu,cpuacct ro,nosuid,nodev,noexec,relatime master:11 - cgroup cgroup rw,cpu,cpuacct\n"
    "31 25 0:28 /containers/abc /sys/fs/cgroup/cpuset ro,nosuid,nodev,noexec,relatime master:12 - cgroup cgroup rw,cpuset\n"
    "32 25 0:29 /containers/abc /sys/fs/cgroup/cpuacct ro,nosuid,nodev,noexec,relatime - cgroup cgroup rw,cpuacct\n";

void mounts() {
    using M = std::vector<core::CgroupMount>;
    struct Case { const char* name; const char* mountinfo; M v2, v1; };
    const Case cases[] = {
        {"a container in its own v2 namespace", v2_own_namespace, {{"/", "/sys/fs/cgroup"}}, {}},
        {"a v1 container, cpuset and cpuacct alone left out", v1_container, {}, {{"/containers/abc", "/sys/fs/cgroup/cpu,cpuacct"}}},
        {"a hybrid host",
         "25 24 0:22 / /sys/fs/cgroup ro,nosuid,nodev,noexec shared:9 - tmpfs tmpfs ro,mode=755\n"
         "26 25 0:23 / /sys/fs/cgroup/unified rw,nosuid,nodev,noexec,relatime shared:10 - cgroup2 cgroup2 rw,nsdelegate\n"
         "27 25 0:24 / /sys/fs/cgroup/cpu,cpuacct rw,nosuid shared:11 master:2 propagate_from:3 - cgroup cgroup rw,cpuacct,cpu\n",
         {{"/", "/sys/fs/cgroup/unified"}}, {{"/", "/sys/fs/cgroup/cpu,cpuacct"}}},
        {"escapes in the root and the mount point",
         "40 30 0:26 /a\\040b /mnt/cgroup\\040v2 rw - cgroup2 cgroup2 rw\n41 30 0:27 /c\\134d /mnt/v1\\011cpu rw - cgroup cgroup rw,cpu\n",
         {{"/a b", "/mnt/cgroup v2"}}, {{"/c\\d", "/mnt/v1\tcpu"}}},
        {"a root outside the namespace left out",
         "40 30 0:26 /../.. /sys/fs/cgroup ro - cgroup2 cgroup rw\n41 30 0:26 / /mnt/own rw - cgroup2 cgroup rw\n",
         {{"/", "/mnt/own"}}, {}},
        {"a trailing slash taken off the root", "40 30 0:26 /a/ /mnt/a rw - cgroup2 cgroup rw\n", {{"/a", "/mnt/a"}}, {}},
        {"lines without the separator or its fields",
         "40 30 0:26 / /sys/fs/cgroup rw cgroup2 cgroup rw\n41 30 0:26 / /mnt/a rw - cgroup2\n42 30 0:26 / relative rw - cgroup2 cgroup rw\n",
         {}, {}},
        {"no text", "", {}, {}},
    };
    for (const auto& c : cases) {
        const M v2 = core::cgroup_v2_mounts(c.mountinfo), v1 = core::cgroup_v1_mounts(c.mountinfo, "cpu");
        require(same(v2, c.v2), std::string(c.name) + ": v2 mounts " + shown(v2) + ", expected " + shown(c.v2));
        require(same(v1, c.v1), std::string(c.name) + ": v1 cpu mounts " + shown(v1) + ", expected " + shown(c.v1));
    }
}

// The quota read over a file system held in a map: the process's own cgroup under its mount and each above it up to the mount's point, v2 and v1, the smallest limit.
void walk() {
    struct Case {
        const char* name;
        const char* proc_self_cgroup;
        const char* mountinfo;
        std::map<std::string, std::string> files;
        std::optional<unsigned> cpus;
    };
    const char* v2_host = "35 24 0:30 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:9 - cgroup2 cgroup2 rw,nsdelegate\n";
    const char* v2_subtree = "35 24 0:30 /machines/m1 /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup2 rw\n";
    const char* v1_host = "27 25 0:24 / /sys/fs/cgroup/cpu,cpuacct rw,nosuid shared:11 - cgroup cgroup rw,cpu,cpuacct\n";
    const std::string v1 = "/sys/fs/cgroup/cpu,cpuacct", v1_and_v2 = std::string(v1_container) + v2_own_namespace;
    const Case cases[] = {
        {"a container in its own v2 namespace", "0::/\n", v2_own_namespace, {{"/sys/fs/cgroup/cpu.max", "600000 100000\n"}}, 6u},
        {"no limit on v2", "0::/a\n", v2_host, {{"/sys/fs/cgroup/a/cpu.max", "max 100000\n"}}, std::nullopt},
        {"a v2 limit on a parent", "0::/a/b\n", v2_host,
         {{"/sys/fs/cgroup/a/b/cpu.max", "max 100000\n"}, {"/sys/fs/cgroup/a/cpu.max", "300000 100000\n"}}, 3u},
        {"the smaller of a v2 cgroup and its parent", "0::/a/b\n", v2_host,
         {{"/sys/fs/cgroup/a/b/cpu.max", "150000 100000\n"}, {"/sys/fs/cgroup/a/cpu.max", "300000 100000\n"}}, 2u},
        {"a v1 container, its cgroup at the root of its mount", "11:cpu,cpuacct:/containers/abc\n", v1_container,
         {{v1 + "/cpu.cfs_quota_us", "600000\n"}, {v1 + "/cpu.cfs_period_us", "100000\n"}}, 6u},
        {"a child cgroup inside a v1 container", "11:cpu,cpuacct:/containers/abc/inner\n", v1_container,
         {{v1 + "/cpu.cfs_quota_us", "600000\n"}, {v1 + "/cpu.cfs_period_us", "100000\n"},
          {v1 + "/inner/cpu.cfs_quota_us", "200000\n"}, {v1 + "/inner/cpu.cfs_period_us", "100000\n"}}, 2u},
        {"a cgroup inside a v1 container named like the container's path on the host", "11:cpu,cpuacct:/containers/abc\n", v1_container,
         {{v1 + "/cpu.cfs_quota_us", "600000\n"}, {v1 + "/cpu.cfs_period_us", "100000\n"},
          {v1 + "/containers/cpu.cfs_quota_us", "100000\n"}, {v1 + "/containers/cpu.cfs_period_us", "100000\n"}}, 6u},
        {"a v2 subtree mounted at the mount point", "0::/machines/m1/payload\n", v2_subtree,
         {{"/sys/fs/cgroup/cpu.max", "600000 100000\n"}, {"/sys/fs/cgroup/payload/cpu.max", "300000 100000\n"},
          {"/sys/fs/cgroup/machines/m1/payload/cpu.max", "100000 100000\n"}}, 3u},
        {"a cgroup outside every mount's root", "0::/elsewhere\n", v2_subtree, {{"/sys/fs/cgroup/cpu.max", "600000 100000\n"}}, std::nullopt},
        {"a root that is a prefix of the cgroup's name only", "0::/machines/m10\n", v2_subtree,
         {{"/sys/fs/cgroup/cpu.max", "600000 100000\n"}}, std::nullopt},
        {"the first mount whose root holds the cgroup", "0::/machines/m1/x\n",
         "40 30 0:30 /machines/m2 /mnt/m2 rw - cgroup2 cgroup2 rw\n41 30 0:30 / /mnt/all rw - cgroup2 cgroup2 rw\n"
         "42 30 0:30 /machines/m1 /mnt/m1 rw - cgroup2 cgroup2 rw\n",
         {{"/mnt/all/machines/m1/x/cpu.max", "400000 100000\n"}, {"/mnt/m1/x/cpu.max", "100000 100000\n"}}, 4u},
        {"a mount point with a space", "0::/a\n", "40 30 0:30 / /mnt/cgroup\\040v2 rw - cgroup2 cgroup2 rw\n",
         {{"/mnt/cgroup v2/a/cpu.max", "200000 100000\n"}}, 2u},
        {"a mount at the file system's root", "0::/a\n", "40 30 0:30 / / rw - cgroup2 cgroup2 rw\n",
         {{"/a/cpu.max", "200000 100000\n"}, {"/cpu.max", "500000 100000\n"}}, 2u},
        {"no mountinfo", "0::/\n", "", {{"/sys/fs/cgroup/cpu.max", "600000 100000\n"}}, std::nullopt},
        {"no v1 cpu mount", "11:cpu,cpuacct:/containers/abc\n", v2_host,
         {{v1 + "/cpu.cfs_quota_us", "600000\n"}, {v1 + "/cpu.cfs_period_us", "100000\n"}}, std::nullopt},
        {"no limit on v1", "11:cpu,cpuacct:/containers/abc\n", v1_host,
         {{v1 + "/containers/abc/cpu.cfs_quota_us", "-1\n"}, {v1 + "/containers/abc/cpu.cfs_period_us", "100000\n"},
          {v1 + "/cpu.cfs_quota_us", "-1\n"}, {v1 + "/cpu.cfs_period_us", "100000\n"}}, std::nullopt},
        {"a v1 limit on a parent", "11:cpu,cpuacct:/containers/abc\n", v1_host,
         {{v1 + "/containers/cpu.cfs_quota_us", "300000\n"}, {v1 + "/containers/cpu.cfs_period_us", "100000\n"}}, 3u},
        {"a v1 quota without its period", "11:cpu,cpuacct:/containers/abc\n", v1_host,
         {{v1 + "/containers/abc/cpu.cfs_quota_us", "200000\n"}}, std::nullopt},
        {"hybrid, the limit on v1", "11:cpu,cpuacct:/containers/abc\n0::/system.slice/x.service\n",
         "26 25 0:23 / /sys/fs/cgroup/unified rw - cgroup2 cgroup2 rw\n27 25 0:24 / /sys/fs/cgroup/cpu,cpuacct rw - cgroup cgroup rw,cpu,cpuacct\n",
         {{v1 + "/containers/abc/cpu.cfs_quota_us", "250000\n"}, {v1 + "/containers/abc/cpu.cfs_period_us", "100000\n"}}, 3u},
        {"the smaller of v1 and v2", "11:cpu,cpuacct:/containers/abc\n0::/\n", v1_and_v2.c_str(),
         {{v1 + "/cpu.cfs_quota_us", "800000\n"}, {v1 + "/cpu.cfs_period_us", "100000\n"}, {"/sys/fs/cgroup/cpu.max", "400000 100000\n"}}, 4u},
        {"a process outside its namespace's root", "0::/../x\n", v2_own_namespace, {{"/sys/fs/cgroup/cpu.max", "600000 100000\n"}}, std::nullopt},
        {"an unreadable cpu.max", "0::/\n", v2_own_namespace, {{"/sys/fs/cgroup/cpu.max", "garbage\n"}}, std::nullopt},
        {"no files", "0::/a/b\n", v2_host, {}, std::nullopt},
        {"no cgroup line", "", v2_own_namespace, {{"/sys/fs/cgroup/cpu.max", "600000 100000\n"}}, std::nullopt},
    };
    for (const auto& c : cases) {
        const auto read = [&](const std::string& path) -> std::optional<std::string> {
            const auto it = c.files.find(path);
            if (it == c.files.end()) return std::nullopt;
            return it->second;
        };
        const auto got = core::cgroup_cpus(c.proc_self_cgroup, c.mountinfo, read);
        require(got == c.cpus, std::string(c.name) + ": gave " + shown(got) + ", expected " + shown(c.cpus));
    }
}

// A job object's CPU rate hard cap as CPUs of the active processors, rounded up.
void job() {
    struct Case { uint32_t flags, rate; unsigned processors; std::optional<unsigned> cpus; };
    const uint32_t cap = core::job_rate_enable | core::job_rate_hard_cap;
    const Case cases[] = {
        {cap, 1250, 16, 2u}, {cap, 1251, 16, 3u}, {cap, 10000, 16, 16u}, {cap, 1, 16, 1u}, {cap, 3750, 16, 6u},
        {cap | 0x8, 5000, 12, 6u}, {cap, 5000, 128, 64u},
        {core::job_rate_enable, 1250, 16, std::nullopt}, {core::job_rate_hard_cap, 1250, 16, std::nullopt},
        {core::job_rate_enable | 0x2, 5, 16, std::nullopt}, {0, 0, 16, std::nullopt},
        {cap, 0, 16, std::nullopt}, {cap, 10001, 16, std::nullopt}, {cap, 1250, 0, std::nullopt},
    };
    for (const auto& c : cases) {
        const auto got = core::job_cpus(c.flags, c.rate, c.processors);
        require(got == c.cpus, "job flags " + std::to_string(c.flags) + ", rate " + std::to_string(c.rate) + " over " +
                               std::to_string(c.processors) + " processors gave " + shown(got) + ", expected " + shown(c.cpus));
    }
}

// The automatic count is the fewest of the sources it could read, four when it read none, from 1 to 64.
void fewest() {
    using O = std::optional<unsigned>;
    struct Case { O hardware, affinity, quota; int threads; };
    const Case cases[] = {
        {16u, 16u, 6u, 6}, {16u, 4u, 6u, 4}, {16u, 16u, std::nullopt, 16}, {16u, std::nullopt, 2u, 2},
        {16u, std::nullopt, std::nullopt, 16}, {std::nullopt, 3u, std::nullopt, 3}, {std::nullopt, std::nullopt, 5u, 5},
        {std::nullopt, std::nullopt, std::nullopt, 4}, {0u, 0u, 0u, 4}, {16u, 0u, 6u, 6},
        {128u, std::nullopt, std::nullopt, 64}, {128u, 96u, 80u, 64}, {128u, 128u, 48u, 48}, {16u, 16u, 1u, 1},
    };
    for (const auto& c : cases) {
        const int got = core::automatic_threads(c.hardware, c.affinity, c.quota);
        require(got == c.threads, "hardware " + shown(c.hardware) + ", affinity " + shown(c.affinity) + ", quota " +
                                  shown(c.quota) + " gave " + std::to_string(got) + ", expected " + std::to_string(c.threads));
    }
    const int live = core::automatic_threads();
    require(live >= 1 && live <= 64, "this process's automatic count " + std::to_string(live) + " is outside 1 to 64");
}

}  // namespace

int main() {
    try {
        v2_quota();
        v1_quota();
        paths();
        mounts();
        walk();
        job();
        fewest();
        std::cout << "cpus: " << checks << " checks pass, this process's automatic count " << core::automatic_threads() << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
