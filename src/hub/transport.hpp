#pragma once
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "core/json.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <io.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace hub {

struct ByteRange { uint64_t offset, length, total; };
struct Response { int status; uint64_t bytes; };
struct TransportError : std::runtime_error {
    int status, curl_code;
    unsigned retry_after_seconds;
    explicit TransportError(const std::string& message, int http = 0, int code = 0, unsigned retry = 0)
        : std::runtime_error("pull: " + message), status(http), curl_code(code), retry_after_seconds(retry) {}
};

namespace transport_detail {

inline std::string config_string(const std::string& value) {
    std::string out = "\"";
    for (char c : value) {
        if (c == '\\' || c == '"') out += '\\';
        if (c == '\n') out += "\\n";
        else out += c;
    }
    return out + '"';
}

inline void clean_text(const std::string& value) {
    for (unsigned char c : value)
        if (c < 32 || c == 127) throw TransportError("control character in request");
}

template<class S> inline bool secret_environment(const S& entry) {
    std::string name;
    for (auto c : entry) {
        if (c == '=') break;
        if (c >= 'a' && c <= 'z') c -= 'a' - 'A';
        if (c > 127) return false;
        name += char(c);
    }
    return name == "HF_TOKEN" || name == "HUGGING_FACE_HUB_TOKEN" ||
           name == "SSLKEYLOGFILE";
}

struct ProcessResult { int code; std::string error, output; };
struct FileClose { void operator()(FILE* file) const { if (file) std::fclose(file); } };
inline std::mutex launch_mutex;
constexpr size_t CONFIG_LIMIT = 4096;
constexpr size_t STDERR_LIMIT = 65536;

#ifdef _WIN32
struct Handle {
    HANDLE value = nullptr;
    Handle() = default;
    explicit Handle(HANDLE h) : value(h) {}
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    ~Handle() { reset(); }
    void reset() {
        if (value && value != INVALID_HANDLE_VALUE) CloseHandle(value);
        value = nullptr;
    }
};
struct Child {
    Handle process;
    bool done = false;
    ~Child() {
        if (process.value && !done) {
            TerminateProcess(process.value, 1);
            WaitForSingleObject(process.value, INFINITE);
        }
    }
};

inline std::filesystem::path curl_path() {
    wchar_t path[MAX_PATH + 1];
    const UINT n = GetSystemDirectoryW(path, MAX_PATH + 1);
    if (!n || n > MAX_PATH) throw TransportError("cannot locate system curl.exe");
    return std::filesystem::path(path) / L"curl.exe";
}
#else
struct Fd {
    int value = -1;
    Fd() = default;
    explicit Fd(int fd) : value(fd) {}
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
    ~Fd() { reset(); }
    void reset() { if (value >= 0) close(value); value = -1; }
};
struct Child {
    pid_t pid = -1;
    ~Child() {
        if (pid > 0) {
            kill(pid, SIGKILL);
            while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {}
        }
    }
};
inline void cloexec(int fd) {
    if (fcntl(fd, F_SETFD, FD_CLOEXEC) < 0)
        throw TransportError("cannot protect curl file descriptors");
}
inline void above_stdio(Fd& fd) {
    if (fd.value >= 3) { cloexec(fd.value); return; }
    const int copy = fcntl(fd.value, F_DUPFD_CLOEXEC, 3);
    if (copy < 0) throw TransportError("cannot protect curl file descriptors");
    fd.reset(); fd.value = copy;
}
inline std::filesystem::path curl_path() { return "curl"; }
#endif

inline ProcessResult process(const std::filesystem::path& executable,
                             const std::string& config,
                             const std::filesystem::path* destination, bool version = false) {
    if (config.size() > CONFIG_LIMIT) throw TransportError("curl request exceeds 4096 bytes");
    ProcessResult result{};
    result.error.reserve(STDERR_LIMIT + 4096);
    // Serialize descriptor creation through spawn; transfers themselves run in parallel.
    std::unique_lock<std::mutex> lock(launch_mutex);
#ifdef _WIN32
    FILE* opened = nullptr;
    if (destination) _wfopen_s(&opened, destination->c_str(), L"w+b");
    std::unique_ptr<FILE, FileClose> output(opened);
#else
    std::unique_ptr<FILE, FileClose> output(destination ?
        std::fopen(destination->c_str(), "w+b") : nullptr);
#endif
    if (destination && !output) throw TransportError("cannot open partial download file");
#ifdef _WIN32
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    Handle input_read, input_write, error_read, error_write, output_child;
    if (!CreatePipe(&input_read.value, &input_write.value, &security, 8192) ||
        !CreatePipe(&error_read.value, &error_write.value, &security, 8192) ||
        !SetHandleInformation(error_read.value, HANDLE_FLAG_INHERIT, 0) ||
        !SetHandleInformation(input_write.value, HANDLE_FLAG_INHERIT, 0) ||
        !DuplicateHandle(GetCurrentProcess(), version ? error_write.value : reinterpret_cast<HANDLE>(
            _get_osfhandle(_fileno(output.get()))), GetCurrentProcess(),
            &output_child.value, 0, TRUE, DUPLICATE_SAME_ACCESS))
        throw TransportError("cannot prepare curl pipes");
    DWORD written = 0;
    if (!WriteFile(input_write.value, config.data(), DWORD(config.size()), &written, nullptr) ||
        written != config.size()) throw TransportError("cannot prepare curl configuration");
    input_write.reset();
    std::wstring environment;
    wchar_t* inherited = GetEnvironmentStringsW();
    if (!inherited) throw TransportError("cannot read process environment");
    struct FreeEnvironment { wchar_t* p; ~FreeEnvironment() { FreeEnvironmentStringsW(p); } } env_guard{inherited};
    for (const wchar_t* p = inherited; *p; p += wcslen(p) + 1) {
        const std::wstring entry(p);
        if (!secret_environment(entry)) { environment += entry; environment += L'\0'; }
    }
    environment += L'\0';
    if (environment.size() == 1) environment += L'\0';
    SIZE_T bytes = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
    std::vector<unsigned char> storage(bytes);
    auto* attributes = reinterpret_cast<PPROC_THREAD_ATTRIBUTE_LIST>(storage.data());
    if (!InitializeProcThreadAttributeList(attributes, 1, 0, &bytes))
        throw TransportError("cannot initialize curl handle inheritance");
    struct Attributes { PPROC_THREAD_ATTRIBUTE_LIST p; ~Attributes() { DeleteProcThreadAttributeList(p); } } attr_guard{attributes};
    HANDLE handles[]{input_read.value, output_child.value, error_write.value};
    if (!UpdateProcThreadAttribute(attributes, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                   handles, sizeof(handles), nullptr, nullptr))
        throw TransportError("cannot restrict curl handle inheritance");
    STARTUPINFOEXW start{};
    start.StartupInfo.cb = sizeof(start);
    start.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    start.StartupInfo.hStdInput = input_read.value;
    start.StartupInfo.hStdOutput = output_child.value;
    start.StartupInfo.hStdError = error_write.value;
    start.lpAttributeList = attributes;
    std::wstring command = L"\"" + executable.wstring() +
        (version ? L"\" -q --version" : L"\" -q --config -");
    Child child;
    PROCESS_INFORMATION info{};
    if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE,
            CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
            environment.data(), nullptr, &start.StartupInfo, &info))
        throw TransportError("cannot launch curl; install an up-to-date system curl");
    child.process.value = info.hProcess;
    Handle thread(info.hThread);
    input_read.reset(); error_write.reset(); output_child.reset();
    lock.unlock();
    char buffer[4096];
    for (;;) {
        DWORD count = 0;
        if (!ReadFile(error_read.value, buffer, sizeof(buffer), &count, nullptr)) {
            if (GetLastError() != ERROR_BROKEN_PIPE) throw TransportError("cannot read curl status");
            break;
        }
        if (!count) break;
        result.error.append(buffer, count);
        if (result.error.size() > STDERR_LIMIT)
            result.error.erase(0, result.error.size() - STDERR_LIMIT);
    }
    DWORD code = 0;
    if (WaitForSingleObject(child.process.value, INFINITE) != WAIT_OBJECT_0 ||
        !GetExitCodeProcess(child.process.value, &code))
        throw TransportError("cannot wait for curl");
    child.done = true;
    result.code = code <= INT_MAX ? int(code) : -1;
#else
    Fd output_child;
    if (output) {
        cloexec(fileno(output.get()));
        output_child.value = fcntl(fileno(output.get()), F_DUPFD_CLOEXEC, 3);
        if (output_child.value < 0) throw TransportError("cannot prepare curl output");
    }
    Fd input_read, input_write, error_read, error_write;
    int ends[2];
    if (pipe(ends) != 0) throw TransportError("cannot prepare curl input pipe");
    input_read.value = ends[0]; input_write.value = ends[1];
    above_stdio(input_read); above_stdio(input_write);
    if (pipe(ends) != 0) throw TransportError("cannot prepare curl status pipe");
    error_read.value = ends[0]; error_write.value = ends[1];
    above_stdio(error_read); above_stdio(error_write);
    const int flags = fcntl(input_write.value, F_GETFL);
    if (flags < 0 || fcntl(input_write.value, F_SETFL, flags | O_NONBLOCK) < 0)
        throw TransportError("cannot prepare nonblocking curl configuration pipe");
    // Fill before spawn with our read end open. Nonblocking avoids a PIPE_BUF/capacity assumption.
    ssize_t written;
    do { written = write(input_write.value, config.data(), config.size()); }
    while (written < 0 && errno == EINTR);
    if (written < 0 || size_t(written) != config.size())
        throw TransportError("curl request exceeds available configuration pipe capacity");
    input_write.reset();
    std::vector<std::string> environment;
    for (char** p = environ; *p; ++p)
        if (!secret_environment(std::string(*p))) environment.emplace_back(*p);
    std::vector<char*> env;
    for (auto& entry : environment) env.push_back(entry.data());
    env.push_back(nullptr);
    std::string command = executable.string();
    char disable[] = "-q", option[] = "--config", input[] = "-", version_option[] = "--version";
    char* argv[]{command.data(), disable, version ? version_option : option, version ? nullptr : input, nullptr};
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions)) throw TransportError("cannot prepare curl spawn");
    struct Actions { posix_spawn_file_actions_t* p; ~Actions() { posix_spawn_file_actions_destroy(p); } } action_guard{&actions};
    if (posix_spawn_file_actions_adddup2(&actions, input_read.value, STDIN_FILENO) ||
        posix_spawn_file_actions_adddup2(&actions, version ? error_write.value : output_child.value, STDOUT_FILENO) ||
        posix_spawn_file_actions_adddup2(&actions, error_write.value, STDERR_FILENO))
        throw TransportError("cannot prepare curl standard streams");
    Child child;
    const int spawned = posix_spawnp(&child.pid, command.c_str(), &actions, nullptr, argv, env.data());
    if (spawned) { child.pid = -1; throw TransportError("cannot launch curl; install curl 8.4 or newer"); }
    input_read.reset(); error_write.reset(); output_child.reset();
    lock.unlock();
    char buffer[4096];
    for (;;) {
        const ssize_t count = read(error_read.value, buffer, sizeof(buffer));
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) throw TransportError("cannot read curl status");
        if (!count) break;
        result.error.append(buffer, size_t(count));
        if (result.error.size() > STDERR_LIMIT)
            result.error.erase(0, result.error.size() - STDERR_LIMIT);
    }
    int status = 0;
    pid_t waited;
    do { waited = waitpid(child.pid, &status, 0); } while (waited < 0 && errno == EINTR);
    if (waited < 0) throw TransportError("cannot wait for curl");
    child.pid = -1;
    result.code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
    if (version) result.output = result.error;
    if (output && std::fclose(output.release()) != 0) throw TransportError("cannot close partial download file");
    return result;
}

inline void check_version(const std::filesystem::path& executable) {
    const auto r = process(executable, "", nullptr, true);
    std::istringstream in(r.output);
    std::string name;
    unsigned major = 0, minor = 0, patch = 0;
    char dot1 = 0, dot2 = 0;
    if (r.code || !(in >> name >> major >> dot1 >> minor >> dot2 >> patch) ||
        name != "curl" || dot1 != '.' || dot2 != '.' || major < 8 || (major == 8 && minor < 4))
        throw TransportError("curl 8.4 or newer is required for safe redirects and bounded downloads");
}

inline Response download_with(const std::filesystem::path& executable,
                             const std::string& url, const std::string& token,
                             const std::filesystem::path& partial, uint64_t max_bytes,
                             std::optional<ByteRange> range) {
    clean_text(url); clean_text(token);
    if (url.compare(0, 8, "https://") != 0) throw TransportError("download URL must use HTTPS");
    const size_t host_end = url.find_first_of("/?#", 8);
    const auto host = url.substr(8, host_end == std::string::npos ? host_end : host_end - 8);
    if (host.empty() || host.find('@') != std::string::npos || host.find('\\') != std::string::npos)
        throw TransportError("invalid download URL authority");
    if (range && (!range->length || range->offset > range->total ||
                  range->length > range->total - range->offset))
        throw TransportError("invalid byte range");
    if (range && (!max_bytes || range->length < max_bytes)) max_bytes = range->length;
    std::string config = "silent\nshow-error\nfail\nlocation\ngloboff\n"
        "proto = \"=https\"\nproto-redir = \"=https\"\nmax-redirs = 5\n"
        "connect-timeout = 30\nspeed-limit = 1\nspeed-time = 60\n";
    config += "url = " + config_string(url) + '\n';
    if (!token.empty()) config += "header = " + config_string("Authorization: Bearer " + token) + '\n';
    if (range) config += "range = " + config_string(std::to_string(range->offset) + "-" +
        std::to_string(range->offset + range->length - 1)) + '\n';
    if (max_bytes) config += "max-filesize = " + std::to_string(max_bytes) + '\n';
    config += "write-out = " + config_string("%{stderr}\nLLMX_RESPONSE\n%{http_code}\n%{header_json}\nLLMX_END\n") + '\n';
    if (config.size() > CONFIG_LIMIT) throw TransportError("curl request exceeds 4096 bytes");
    check_version(executable);
    auto result = process(executable, config, &partial);
    // Windows curl writes its status stream in text mode, independently of the binary body.
    size_t normalized = 0;
    for (size_t i = 0; i < result.error.size(); ++i) {
        if (result.error[i] == '\r' && i + 1 < result.error.size() && result.error[i + 1] == '\n') continue;
        result.error[normalized++] = result.error[i];
    }
    result.error.resize(normalized);
    const std::string marker = "\nLLMX_RESPONSE\n", end = "\nLLMX_END\n";
    const auto start = result.error.rfind(marker);
    int status = 0;
    std::string headers;
    if (start != std::string::npos && result.error.size() >= end.size() &&
        result.error.compare(result.error.size() - end.size(), end.size(), end) == 0) {
        const size_t at = start + marker.size();
        if (result.error.size() > at + 4 && result.error[at + 3] == '\n' &&
            result.error[at] >= '0' && result.error[at] <= '9' &&
            result.error[at+1] >= '0' && result.error[at+1] <= '9' &&
            result.error[at+2] >= '0' && result.error[at+2] <= '9') {
            status = (result.error[at]-'0')*100 + (result.error[at+1]-'0')*10 + result.error[at+2]-'0';
            headers = result.error.substr(at + 4, result.error.size() - end.size() - at - 4);
        }
    }
    jmini::Value header_object;
    try { header_object = jmini::parse(headers); } catch (const std::runtime_error&) {}
    unsigned retry = 0;
    if (status == 429 || status == 503) {
        const auto* values = header_object.get("retry-after");
        if (values && values->isArray() && values->arr.size() == 1 && values->arr[0].isString()) {
            const auto& text = values->arr[0].str;
            if (!text.empty() && text.find_first_not_of("0123456789") == std::string::npos)
                // 61 is a refusal sentinel: never retry earlier than a longer server delay.
                for (char c : text) retry = std::min(61u, retry * 10 + unsigned(c - '0'));
        }
    }
    if (result.code) throw TransportError("curl transfer failed (exit " + std::to_string(result.code) +
        ", HTTP " + std::to_string(status) + ")", status, result.code, retry);
    if (status != (range ? 206 : 200)) throw TransportError(range && status == 200 ?
        "server ignored byte range (HTTP 200)" : "unexpected HTTP status " + std::to_string(status), status, 0, retry);
    if (!header_object.isObject()) throw TransportError("invalid curl response headers", status);
    if (range) {
        bool valid = false;
        {
            const auto* values = header_object.get("content-range");
            const std::string expected = "bytes " + std::to_string(range->offset) + "-" +
                std::to_string(range->offset + range->length - 1) + "/" + std::to_string(range->total);
            valid = values && values->isArray() && values->arr.size() == 1 &&
                values->arr[0].isString() && values->arr[0].str == expected;
        }
        if (!valid) throw TransportError("invalid Content-Range response", status);
    }
    std::error_code ec;
    const auto size = std::filesystem::file_size(partial, ec);
    if (ec) throw TransportError("cannot inspect partial download size", status);
    if ((max_bytes && size > max_bytes) || (range && size != range->length))
        throw TransportError("download byte count does not match request", status);
    return {status, uint64_t(size)};
}
} // namespace transport_detail

inline Response download(const std::string& url, const std::string& token,
                         const std::filesystem::path& partial, uint64_t max_bytes = 0,
                         std::optional<ByteRange> range = std::nullopt) {
    return transport_detail::download_with(transport_detail::curl_path(), url, token,
                                            partial, max_bytes, range);
}
} // namespace hub
