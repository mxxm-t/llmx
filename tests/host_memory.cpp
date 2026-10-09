// The memory a process can still take and the memory limits it reads, checked on file texts and field values rather than a real cgroup or job object.
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

#include "core/host_memory.hpp"

namespace {

size_t checks = 0;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
    ++checks;
}

const size_t M = size_t(1) << 20, G = size_t(1) << 30;

// What a container run with --memory 8g leaves once its cgroup has charged it 2318336 bytes.
const size_t container_room = 8 * G - 2318336;

std::string shown(const std::optional<size_t>& n) { return n ? std::to_string(*n) : "nothing"; }

std::string gib(const std::optional<size_t>& n, const char* none) {
    if (!n) return none;
    char text[32];
    std::snprintf(text, sizeof text, "%.2f GiB", double(*n) / double(G));
    return text;
}

// cgroup v2's memory.max less memory.current, without a memory.stat: 0 past the limit, nothing for "max" or a text that is not one decimal number.
void v2_room() {
    struct Case { const char* max; const char* current; std::optional<size_t> room; };
    const Case cases[] = {
        {"8589934592\n", "104857600\n", 8 * G - 100 * M}, {"8589934592", "0", 8 * G}, {"  4294967296 \n", "\t1073741824\n", 3 * G},
        {"4294967296\n", "4294967296\n", 0}, {"4294967296\n", "5368709120\n", 0}, {"0\n", "0\n", 0},
        {"max\n", "104857600\n", std::nullopt}, {"max\n", "", std::nullopt}, {"max 4294967296\n", "0\n", std::nullopt},
        {"", "0\n", std::nullopt}, {"\n", "0\n", std::nullopt}, {"4g\n", "0\n", std::nullopt}, {"-1\n", "0\n", std::nullopt},
        {"+4294967296\n", "0\n", std::nullopt}, {"0x100000000\n", "0\n", std::nullopt}, {"4294967296 0\n", "0\n", std::nullopt},
        {"99999999999999999999999\n", "0\n", std::nullopt}, {"4294967296\n", "", std::nullopt},
        {"4294967296\n", "garbage\n", std::nullopt}, {"4294967296\n", "-5\n", std::nullopt}, {"4294967296\n", "1 2\n", std::nullopt},
    };
    for (const auto& c : cases) {
        const auto got = core::cgroup_v2_memory_room(c.max, c.current, "");
        require(got == c.room, std::string("memory.max \"") + c.max + "\" over memory.current \"" + c.current + "\" gave " + shown(got) +
                               ", expected " + shown(c.room));
    }
}

// cgroup v1's memory.limit_in_bytes less memory.usage_in_bytes, without a memory.stat: 0 past the limit, nothing for a limit of 2^62 bytes or more, which is how v1 writes none, or a text that is not one decimal number.
void v1_room() {
    struct Case { const char* limit; const char* usage; std::optional<size_t> room; };
    const Case cases[] = {
        {"8589934592\n", "1073741824\n", 7 * G}, {"2147483648\n", "3221225472\n", 0}, {"2147483648\n", "2147483648\n", 0},
        {"4611686018427387903\n", "0\n", (size_t(1) << 62) - 1},
        {"9223372036854771712\n", "1073741824\n", std::nullopt}, {"9223372036854710272\n", "0\n", std::nullopt},
        {"4611686018427387904\n", "0\n", std::nullopt}, {"9223372036854771712\n", "garbage\n", std::nullopt},
        {"max\n", "0\n", std::nullopt}, {"-1\n", "0\n", std::nullopt}, {"", "0\n", std::nullopt}, {"8589934592 1\n", "0\n", std::nullopt},
        {"8589934592\n", "", std::nullopt}, {"8589934592\n", "1g\n", std::nullopt}, {"18446744073709551616\n", "0\n", std::nullopt},
    };
    for (const auto& c : cases) {
        const auto got = core::cgroup_v1_memory_room(c.limit, c.usage, "");
        require(got == c.room, std::string("memory.limit_in_bytes \"") + c.limit + "\" over memory.usage_in_bytes \"" + c.usage + "\" gave " +
                               shown(got) + ", expected " + shown(c.room));
    }
}

// The limit less the working set, the usage less the inactive file pages memory.stat gives (inactive_file on v2, total_inactive_file on v1), a working set below 0 taken as 0.
// The usage stands whole when memory.stat is missing, lacks the key or gives it as anything but one decimal number.
void working_set() {
    struct Case { const char* limit; const char* usage; const char* stat; std::optional<size_t> room; };
    const char* const v2_stat = "anon 1073741824\nfile 5368709120\ninactive_anon 0\nactive_anon 1073741824\ninactive_file 4294967296\nactive_file 1073741824\n";
    const Case v2_cases[] = {
        {"8589934592\n", "6442450944\n", v2_stat, 6 * G},
        {"8589934592\n", "1073741824\n", "inactive_file 2147483648\n", 8 * G},
        {"8589934592\n", "8589000000\n", "anon 500000000\ninactive_file 8000000000\n", 8589934592 - 589000000},
        {"1073741824\n", "3221225472\n", "inactive_file 1073741824\n", 0},
        {"8589934592\n", "6442450944\n", "anon 0\ninactive_file 4294967296", 6 * G},
        {"8589934592\n", "6442450944\n", "anon 1073741824\nfile 5368709120\nactive_file 1073741824\n", 2 * G},
        {"8589934592\n", "6442450944\n", "", 2 * G},
        {"8589934592\n", "6442450944\n", "inactive_anon 4294967296\n", 2 * G},
        {"8589934592\n", "6442450944\n", "total_inactive_file 4294967296\n", 2 * G},
        {"8589934592\n", "6442450944\n", "inactive_file 4G\n", 2 * G},
        {"8589934592\n", "6442450944\n", "inactive_file -1\n", 2 * G},
        {"8589934592\n", "6442450944\n", "inactive_file\n", 2 * G},
        {"8589934592\n", "6442450944\n", "inactive_file 1 2\n", 2 * G},
        {"8589934592\n", "6442450944\n", "garbage", 2 * G},
        {"max\n", "6442450944\n", v2_stat, std::nullopt},
        {"8589934592\n", "garbage\n", v2_stat, std::nullopt},
    };
    for (const auto& c : v2_cases) {
        const auto got = core::cgroup_v2_memory_room(c.limit, c.usage, c.stat);
        require(got == c.room, std::string("memory.max \"") + c.limit + "\" over memory.current \"" + c.usage + "\" and memory.stat \"" + c.stat +
                               "\" gave " + shown(got) + ", expected " + shown(c.room));
    }
    const Case v1_cases[] = {
        {"8589934592\n", "6442450944\n", "cache 5368709120\ninactive_file 1073741824\ntotal_cache 5368709120\ntotal_inactive_file 4294967296\n", 6 * G},
        {"8589934592\n", "1073741824\n", "total_inactive_file 2147483648\n", 8 * G},
        {"1073741824\n", "3221225472\n", "total_inactive_file 1073741824\n", 0},
        {"8589934592\n", "6442450944\n", "cache 5368709120\ninactive_file 4294967296\n", 2 * G},
        {"8589934592\n", "6442450944\n", "", 2 * G},
        {"8589934592\n", "6442450944\n", "total_inactive_file x\n", 2 * G},
        {"9223372036854771712\n", "6442450944\n", "total_inactive_file 4294967296\n", std::nullopt},
    };
    for (const auto& c : v1_cases) {
        const auto got = core::cgroup_v1_memory_room(c.limit, c.usage, c.stat);
        require(got == c.room, std::string("memory.limit_in_bytes \"") + c.limit + "\" over memory.usage_in_bytes \"" + c.usage +
                               "\" and memory.stat \"" + c.stat + "\" gave " + shown(got) + ", expected " + shown(c.room));
    }
}

// Mountinfo lines as the kernel writes them, with the names changed; the first is a container's, run with its own cgroup namespace.
const char* const v2_own_namespace =
    "637 636 0:30 / /sys/fs/cgroup ro,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw,nsdelegate,memory_recursiveprot\n";
const char* const v2_host = "35 24 0:30 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:9 - cgroup2 cgroup2 rw,nsdelegate\n";
const char* const v1_container =
    "30 25 0:27 /docker/abc /sys/fs/cgroup/cpu,cpuacct ro,nosuid,nodev,noexec,relatime master:11 - cgroup cgroup rw,cpu,cpuacct\n"
    "33 25 0:31 /docker/abc /sys/fs/cgroup/memory ro,nosuid,nodev,noexec,relatime master:13 - cgroup cgroup rw,memory\n";
const char* const v1_host =
    "27 25 0:24 / /sys/fs/cgroup/cpu,cpuacct rw,nosuid shared:11 - cgroup cgroup rw,cpu,cpuacct\n"
    "34 25 0:31 / /sys/fs/cgroup/memory rw,nosuid shared:15 - cgroup cgroup rw,memory\n";

using Files = std::map<std::string, std::string>;

std::optional<size_t> room_over(const char* proc_self_cgroup, const char* mountinfo, const Files& files) {
    const auto read = [&](const std::string& path) -> std::optional<std::string> {
        const auto it = files.find(path);
        if (it == files.end()) return std::nullopt;
        return it->second;
    };
    return core::cgroup_memory_room(proc_self_cgroup, mountinfo, read);
}

// The room over a file system held in a map: the process's own cgroup and each above it up to the mount's point, v2 and v1, the smallest.
void walk() {
    struct Case {
        const char* name;
        const char* proc_self_cgroup;
        const char* mountinfo;
        Files files;
        std::optional<size_t> room;
    };
    const std::string v2 = "/sys/fs/cgroup", v1 = "/sys/fs/cgroup/memory", huge = "9223372036854771712\n";
    const std::string v1_and_v2 = std::string(v1_host) + "26 25 0:23 / /sys/fs/cgroup/unified rw - cgroup2 cgroup2 rw\n";
    const Case cases[] = {
        {"a container run with --memory 8g in its own v2 namespace", "0::/\n", v2_own_namespace,
         {{v2 + "/memory.max", "8589934592\n"}, {v2 + "/memory.current", "2318336\n"}}, container_room},
        {"no limit on v2", "0::/a\n", v2_host, {{v2 + "/a/memory.max", "max\n"}, {v2 + "/a/memory.current", "1073741824\n"}}, std::nullopt},
        {"a v2 parent's tighter limit", "0::/a/b\n", v2_host,
         {{v2 + "/a/b/memory.max", "8589934592\n"}, {v2 + "/a/b/memory.current", "1073741824\n"},
          {v2 + "/a/memory.max", "4294967296\n"}, {v2 + "/a/memory.current", "3221225472\n"}}, 1 * G},
        {"a v2 child's tighter room", "0::/a/b\n", v2_host,
         {{v2 + "/a/b/memory.max", "2147483648\n"}, {v2 + "/a/b/memory.current", "1610612736\n"},
          {v2 + "/a/memory.max", "17179869184\n"}, {v2 + "/a/memory.current", "3221225472\n"}}, G / 2},
        {"a v2 parent without a limit", "0::/a/b\n", v2_host,
         {{v2 + "/a/b/memory.max", "2147483648\n"}, {v2 + "/a/b/memory.current", "1073741824\n"},
          {v2 + "/a/memory.max", "max\n"}, {v2 + "/a/memory.current", "5368709120\n"}}, 1 * G},
        {"usage past a v2 limit", "0::/\n", v2_own_namespace, {{v2 + "/memory.max", "1073741824\n"}, {v2 + "/memory.current", "1610612736\n"}}, 0},
        {"a v2 limit without its usage", "0::/\n", v2_own_namespace, {{v2 + "/memory.max", "1073741824\n"}}, std::nullopt},
        {"a malformed memory.max, its parent's limit still read", "0::/a/b\n", v2_host,
         {{v2 + "/a/b/memory.max", "8G\n"}, {v2 + "/a/b/memory.current", "1073741824\n"},
          {v2 + "/a/memory.max", "4294967296\n"}, {v2 + "/a/memory.current", "1073741824\n"}}, 3 * G},
        {"a malformed memory.current", "0::/\n", v2_own_namespace, {{v2 + "/memory.max", "1073741824\n"}, {v2 + "/memory.current", "garbage\n"}},
         std::nullopt},
        {"a v1 container", "11:cpu,cpuacct:/docker/abc\n9:memory:/docker/abc\n0::/system.slice/containerd.service\n", v1_container,
         {{v1 + "/memory.limit_in_bytes", "8589934592\n"}, {v1 + "/memory.usage_in_bytes", "1073741824\n"}}, 7 * G},
        {"a child cgroup inside a v1 container", "9:memory:/docker/abc/inner\n", v1_container,
         {{v1 + "/inner/memory.limit_in_bytes", huge}, {v1 + "/inner/memory.usage_in_bytes", "1073741824\n"},
          {v1 + "/memory.limit_in_bytes", "8589934592\n"}, {v1 + "/memory.usage_in_bytes", "6442450944\n"}}, 2 * G},
        {"no limit on v1", "9:memory:/a\n", v1_host,
         {{v1 + "/a/memory.limit_in_bytes", huge}, {v1 + "/a/memory.usage_in_bytes", "1073741824\n"},
          {v1 + "/memory.limit_in_bytes", huge}, {v1 + "/memory.usage_in_bytes", "32212254720\n"}}, std::nullopt},
        {"a v1 parent's tighter limit", "9:memory:/a/b\n", v1_host,
         {{v1 + "/a/b/memory.limit_in_bytes", huge}, {v1 + "/a/b/memory.usage_in_bytes", "1073741824\n"},
          {v1 + "/a/memory.limit_in_bytes", "2147483648\n"}, {v1 + "/a/memory.usage_in_bytes", "1610612736\n"},
          {v1 + "/memory.limit_in_bytes", huge}, {v1 + "/memory.usage_in_bytes", "32212254720\n"}}, G / 2},
        {"the memory controller's v1 cgroup, not the cpu controller's", "11:cpu,cpuacct:/x\n9:memory:/y\n", v1_host,
         {{v1 + "/y/memory.limit_in_bytes", "1073741824\n"}, {v1 + "/y/memory.usage_in_bytes", "0\n"},
          {v1 + "/x/memory.limit_in_bytes", "104857600\n"}, {v1 + "/x/memory.usage_in_bytes", "0\n"},
          {"/sys/fs/cgroup/cpu,cpuacct/y/memory.limit_in_bytes", "104857600\n"}, {"/sys/fs/cgroup/cpu,cpuacct/y/memory.usage_in_bytes", "0\n"}},
         1 * G},
        {"hybrid, the limit on v1", "9:memory:/docker/abc\n0::/system.slice/x.service\n", v1_and_v2.c_str(),
         {{v1 + "/docker/abc/memory.limit_in_bytes", "4294967296\n"}, {v1 + "/docker/abc/memory.usage_in_bytes", "1073741824\n"}}, 3 * G},
        {"the smaller of v1 and v2", "9:memory:/docker/abc\n0::/system.slice/x.service\n", v1_and_v2.c_str(),
         {{v1 + "/docker/abc/memory.limit_in_bytes", "4294967296\n"}, {v1 + "/docker/abc/memory.usage_in_bytes", "1073741824\n"},
          {v2 + "/unified/system.slice/x.service/memory.max", "2147483648\n"},
          {v2 + "/unified/system.slice/x.service/memory.current", "1073741824\n"}}, 1 * G},
        {"a container whose usage is mostly file cache", "0::/\n", v2_own_namespace,
         {{v2 + "/memory.max", "8589934592\n"}, {v2 + "/memory.current", "8547991552\n"},
          {v2 + "/memory.stat", "anon 41943040\nfile 8506048512\ninactive_file 8455716864\nactive_file 50331648\n"}}, 8 * G - 92274688},
        {"each cgroup's memory.stat read at its own level", "0::/a/b\n", v2_host,
         {{v2 + "/a/b/memory.max", "8589934592\n"}, {v2 + "/a/b/memory.current", "1073741824\n"},
          {v2 + "/a/memory.max", "4294967296\n"}, {v2 + "/a/memory.current", "3758096384\n"},
          {v2 + "/a/memory.stat", "inactive_file 3221225472\n"}}, 3 * G + G / 2},
        {"a v1 container's total_inactive_file", "9:memory:/docker/abc\n", v1_container,
         {{v1 + "/memory.limit_in_bytes", "8589934592\n"}, {v1 + "/memory.usage_in_bytes", "8053063680\n"},
          {v1 + "/memory.stat", "inactive_file 0\ntotal_inactive_file 6442450944\n"}}, 6 * G + G / 2},
        {"no files", "0::/a/b\n", v2_host, {}, std::nullopt},
        {"no mountinfo", "0::/\n", "", {{v2 + "/memory.max", "1073741824\n"}, {v2 + "/memory.current", "0\n"}}, std::nullopt},
        {"no cgroup line", "", v2_own_namespace, {{v2 + "/memory.max", "1073741824\n"}, {v2 + "/memory.current", "0\n"}}, std::nullopt},
    };
    for (const auto& c : cases) {
        const auto got = room_over(c.proc_self_cgroup, c.mountinfo, c.files);
        require(got == c.room, std::string(c.name) + ": gave " + shown(got) + ", expected " + shown(c.room));
    }
}

// A job object's memory limits: the process limit less the process's commit and the job limit less the job's, the smaller, 0 past a limit.
void job() {
    struct Case { uint32_t flags; uint64_t process_limit, process_commit, job_limit, job_commit; std::optional<size_t> room; };
    const uint32_t P = core::job_limit_process_memory, J = core::job_limit_job_memory, kill_on_close = 0x2000;
    const Case cases[] = {
        {0, 8 * G, 1 * G, 8 * G, 1 * G, std::nullopt}, {kill_on_close, 8 * G, 1 * G, 8 * G, 1 * G, std::nullopt},
        {P, 8 * G, 1 * G, 0, 0, 7 * G}, {P | kill_on_close, 8 * G, 1 * G, 0, 0, 7 * G}, {J, 0, 0, 4 * G, 3 * G, 1 * G},
        {P | J, 8 * G, 1 * G, 4 * G, 3 * G, 1 * G}, {P | J, 2 * G, 3 * G / 2, 4 * G, 1 * G, G / 2},
        {P, 1 * G, 2 * G, 0, 0, 0}, {J, 0, 0, 1 * G, 1 * G, 0}, {J, 8 * G, 0, 4 * G, 5 * G, 0},
    };
    for (const auto& c : cases) {
        const auto got = core::job_memory_room(c.flags, c.process_limit, c.process_commit, c.job_limit, c.job_commit);
        require(got == c.room, "job flags " + std::to_string(c.flags) + ", process " + std::to_string(c.process_commit) + " of " +
                               std::to_string(c.process_limit) + ", job " + std::to_string(c.job_commit) + " of " + std::to_string(c.job_limit) +
                               " gave " + shown(got) + ", expected " + shown(c.room));
    }
}

// What the process can still take is the fewer of the host's available memory and the room its limits leave, each only when it was read.
void fewest() {
    using O = std::optional<size_t>;
    struct Case { O host, room, available; };
    const Case cases[] = {
        {16 * G, 7 * G, 7 * G}, {4 * G, 7 * G, 4 * G}, {16 * G, std::nullopt, 16 * G}, {std::nullopt, 7 * G, 7 * G},
        {std::nullopt, std::nullopt, std::nullopt}, {16 * G, 0, 0}, {0, 7 * G, 0},
    };
    for (const auto& c : cases) {
        const auto got = core::host_memory_available(c.host, c.room);
        require(got == c.available, "host " + shown(c.host) + ", room " + shown(c.room) + " gave " + shown(got) + ", expected " + shown(c.available));
    }
    // A container limited to 8 GiB on a host with 26 GiB available can take what its limit leaves, and without the limit's files the host's figure stands.
    const Files limited = {{"/sys/fs/cgroup/memory.max", "8589934592\n"}, {"/sys/fs/cgroup/memory.current", "2318336\n"}};
    const O host = 26 * G;
    const O in_container = core::host_memory_available(host, room_over("0::/\n", v2_own_namespace, limited));
    require(in_container == container_room, "a container limited to 8 GiB was given " + shown(in_container));
    const O without_files = core::host_memory_available(host, room_over("0::/\n", v2_own_namespace, {}));
    require(without_files == host, "without the limit's files the host's figure became " + shown(without_files));
}

}  // namespace

int main() {
    try {
        v2_room();
        v1_room();
        working_set();
        walk();
        job();
        fewest();
        std::cout << "host-memory: " << checks << " checks pass; this process can take " << gib(core::host_memory_available(), "an unknown amount")
                  << ", the host has " << gib(core::system_memory_available(), "an unknown amount") << " available and its memory limits leave "
                  << gib(core::memory_limit_room(), "no limit") << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
