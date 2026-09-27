#pragma once
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

// The cgroups over this process and the directories their files are read in, from the texts of /proc/self/cgroup and /proc/self/mountinfo (docs/src/core-cgroup.md).

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

// The process's cgroup v2 path in /proc/self/cgroup text, its "0::/path" line.
inline std::optional<std::string> cgroup_v2_path(std::string_view proc_self_cgroup) {
    return detail::cgroup_line_path(proc_self_cgroup, [](std::string_view hierarchy, std::string_view controllers) {
        return hierarchy == "0" && controllers.empty();
    });
}

// The process's cgroup v1 path for `controller` in /proc/self/cgroup text, from the first line whose controllers list it.
inline std::optional<std::string> cgroup_v1_path(std::string_view proc_self_cgroup, std::string_view controller) {
    return detail::cgroup_line_path(proc_self_cgroup, [&](std::string_view, std::string_view controllers) {
        return detail::lists(controllers, controller);
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

// The cgroup v1 mounts in /proc/self/mountinfo text whose hierarchy holds `controller`.
inline std::vector<CgroupMount> cgroup_v1_mounts(std::string_view mountinfo, std::string_view controller) {
    return detail::cgroup_mounts(mountinfo, [&](std::string_view type, std::string_view options) {
        return type == "cgroup" && detail::lists(options, controller);
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

namespace detail {

// The directory of cgroup `path` under `mounts` and of each cgroup above it up to the mount's point, its own first; none when there is no path or no mount's root holds it.
inline std::vector<std::string> cgroup_directories(const std::optional<std::string>& path, const std::vector<CgroupMount>& mounts) {
    std::vector<std::string> dirs;
    if (!path) return dirs;
    const auto dir = cgroup_directory(*path, mounts);
    if (!dir) return dirs;
    const std::string point = dir->first == "/" ? std::string() : dir->first;
    for (std::string below = dir->second;; below.erase(below.rfind('/'))) {
        dirs.push_back(point + below);
        if (below.empty()) return dirs;
    }
}

}  // namespace detail

// The directories of the process's cgroup v2 and of each cgroup above it up to its mount's point, its own first, from its /proc/self/cgroup and /proc/self/mountinfo texts.
// A limit on any cgroup above the process's holds it too, so a reader takes the smallest over all of them.
inline std::vector<std::string> cgroup_v2_directories(std::string_view proc_self_cgroup, std::string_view mountinfo) {
    return detail::cgroup_directories(cgroup_v2_path(proc_self_cgroup), cgroup_v2_mounts(mountinfo));
}

// The same for the process's cgroup v1 of `controller`, in the hierarchy mounted with it.
inline std::vector<std::string> cgroup_v1_directories(std::string_view proc_self_cgroup, std::string_view mountinfo, std::string_view controller) {
    return detail::cgroup_directories(cgroup_v1_path(proc_self_cgroup, controller), cgroup_v1_mounts(mountinfo, controller));
}

}  // namespace core
