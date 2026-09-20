#include "hub/transport.hpp"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#ifdef _WIN32
#include <fcntl.h>
#endif

namespace fs = std::filesystem;
static int checks = 0;
static void require(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
static std::string read(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
static void write(const fs::path& p, const std::string& value) {
    std::ofstream f(p, std::ios::binary);
    f << value;
    if (!f) throw std::runtime_error("fixture write failed");
}
static void env_set(const char* key, const char* value) {
#ifdef _WIN32
    if (_putenv_s(key, value ? value : "")) throw std::runtime_error("environment failed");
#else
    if (value ? setenv(key, value, 1) : unsetenv(key)) throw std::runtime_error("environment failed");
#endif
}
struct Env {
    std::string key;
    std::optional<std::string> old;
    Env(const char* name, const char* value) : key(name) {
        if (const auto* v = std::getenv(name)) old = v;
        env_set(name, value);
    }
    ~Env() { env_set(key.c_str(), old ? old->c_str() : nullptr); }
};

static int fake(int argc, char** argv) {
    if (std::getenv("HF_TOKEN") || std::getenv("HUGGING_FACE_HUB_TOKEN") ||
        std::getenv("SSLKEYLOGFILE")) return 92;
    if (argc == 3 && std::string(argv[1]) == "-q" && std::string(argv[2]) == "--version") {
        const auto* version = std::getenv("LLMX_TEST_CURL_VERSION");
        std::cout << "curl " << (version ? version : "8.4.0") << " fake\n";
        return 0;
    }
    if (argc != 4 || std::string(argv[1]) != "-q" ||
        std::string(argv[2]) != "--config" || std::string(argv[3]) != "-") return 91;
    const std::string config{std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>()};
    for (const auto* option : {"silent\n", "show-error\n", "fail\n", "location\n", "globoff\n",
            "proto = \"=https\"\n", "proto-redir = \"=https\"\n", "max-redirs = 5\n",
            "connect-timeout = 30\n", "speed-limit = 1\n", "speed-time = 60\n"})
        if (config.find(option) == std::string::npos) return 93;
    if (config.find("location-trusted") != std::string::npos || config.find("\nretry") != std::string::npos)
        return 94;
    std::string url, authorization, requested_range;
    std::istringstream lines(config);
    std::string line;
    while (std::getline(lines, line)) {
        const auto equal = line.find(" = ");
        if (equal == std::string::npos) continue;
        const auto key = line.substr(0, equal), value = line.substr(equal + 3);
        if (key == "url") url = jmini::parse(value).str;
        if (key == "header") authorization = jmini::parse(value).str;
        if (key == "range") requested_range = jmini::parse(value).str;
    }
    const auto slash = url.rfind('/');
    const auto scenario = slash == std::string::npos ? url : url.substr(slash + 1);
    if (scenario == "early") return 7;
    if (scenario == "auth" && authorization != "Authorization: Bearer secret_\"\\$&") return 95;
    if (scenario == "quote\"\\$&{}[]" && url != "https://example.test/quote\"\\$&{}[]") return 96;
    if (!requested_range.empty() && requested_range != "3-6") return 97;
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
#endif
    std::string body("A\0\r\nZ", 5), headers = "{}";
    int status = 200;
    if (!requested_range.empty()) {
        status = 206; body = "DATA";
        headers = "{\"content-range\":[\"bytes 3-6/10\"]}";
    }
    if (scenario == "ignored") status = 200;
    if (scenario == "missing") headers = "{}";
    if (scenario == "wrong") headers = "{\"content-range\":[\"bytes 4-7/10\"]}";
    if (scenario == "total") headers = "{\"content-range\":[\"bytes 3-6/11\"]}";
    if (scenario == "duplicate") headers = "{\"content-range\":[\"bytes 3-6/10\",\"bytes 3-6/10\"]}";
    if (scenario == "short") body.pop_back();
    if (scenario == "extra") body += 'X';
    if (scenario == "204") status = 204;
    if (scenario == "206") status = 206;
    if (scenario == "500") status = 500;
    if (scenario.find("retry") == 0) {
        status = scenario == "retry-503" ? 503 : 429;
        headers = "{\"retry-after\":[\"7\"]}";
        if (scenario == "retry-large") headers = "{\"retry-after\":[\"999999999999999999999\"]}";
        if (scenario == "retry-date") headers = "{\"retry-after\":[\"Sun, 20 Sep 2026 12:00:00 GMT\"]}";
        if (scenario == "retry-negative") headers = "{\"retry-after\":[\"-1\"]}";
        if (scenario == "retry-duplicate") headers = "{\"retry-after\":[\"7\",\"9\"]}";
    }
    if (scenario == "noise") std::cerr << std::string(300000, 'x');
    std::cout.write(body.data(), std::streamsize(body.size()));
    if (scenario == "nomarker") return 0;
    if (scenario == "500") std::cerr << "secret_\"\\$& https://secret.invalid/signed?token=x\n";
    const char* nl = scenario == "crlf" ? "\r\n" : "\n";
    std::cerr << nl << "LLMX_RESPONSE" << nl << status << nl << headers << nl << "LLMX_END" << nl;
    return status >= 400 ? 22 : 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "-q") {
        try { return fake(argc, argv); } catch (...) { return 98; }
    }
    fs::path root;
    try {
        root = fs::temp_directory_path() / ("llmx-transport-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
        if (argc == 2 && std::string(argv[1]) == "--https-smoke") {
            fs::create_directories(root);
            const auto response = hub::download("https://huggingface.co/api/models/Qwen/Qwen3-0.6B-GGUF",
                "", root / "metadata.json", 1024 * 1024);
            require(response.status == 200 && response.bytes > 0, "HTTPS GET failed");
            require(jmini::parse(read(root / "metadata.json")).isObject(), "HTTPS JSON invalid");
            std::cout << "HTTPS smoke: status=" << response.status << " bytes=" << response.bytes
                      << " evidence=" << root << '\n';
            return 0;
        }
        fs::create_directories(root / fs::u8path(u8"space \u00e4"));
        const fs::path self = fs::absolute(argv[0]);
        const fs::path child = root / fs::u8path(u8"space \u00e4") /
#ifdef _WIN32
            "fake curl.exe";
#else
            "fake curl";
#endif
        fs::copy_file(self, child);
        const fs::path out = root / fs::u8path(u8"space \u00e4") / fs::u8path(u8"body \u4e2d.bin");
        Env token("HF_TOKEN", "must-not-inherit"), legacy("HUGGING_FACE_HUB_TOKEN", "must-not-inherit"),
            tls("SSLKEYLOGFILE", "must-not-inherit");
#ifdef _WIN32
        DWORD handles_before = 0;
#else
        auto fd_count = [] { int count = 0; for (int i = 0; i < 256; ++i) if (fcntl(i, F_GETFD) >= 0) ++count; return count; };
        const auto fds_before = fd_count();
        struct sigaction signal_before{};
        require(sigaction(SIGPIPE, nullptr, &signal_before) == 0, "signal query failed");
#endif
        auto get = [&](const std::string& name, uint64_t cap = 100,
                       std::optional<hub::ByteRange> range = std::nullopt) {
            return hub::transport_detail::download_with(child, "https://example.test/" + name,
                "secret_\"\\$&", out, cap, range);
        };
        auto reject = [&](const std::string& name, uint64_t cap = 100,
                          std::optional<hub::ByteRange> range = std::nullopt) {
            try { get(name, cap, range); }
            catch (const hub::TransportError& e) {
                require(std::string(e.what()).find("secret") == std::string::npos, "secret leaked in error");
                return e.status;
            }
            throw std::runtime_error("invalid response accepted: " + name);
        };
        const auto ok = get("auth");
#ifdef _WIN32
        // Establish the baseline after Windows' first process-launch initialization.
        require(GetProcessHandleCount(GetCurrentProcess(), &handles_before) != FALSE, "handle query failed");
#endif
        require(ok.status == 200 && ok.bytes == 5, "full response failed");
        require(read(out) == std::string("A\0\r\nZ", 5), "binary output changed");
        require(get("quote\"\\$&{}[]").bytes == 5, "request quoting failed");
        require(get(std::string(2000, 'x')).bytes == 5, "long configuration failed");
        require(get("noise").bytes == 5, "bounded stderr draining failed");
        require(get("crlf").bytes == 5, "Windows text status failed");
        require(reject("204") == 204, "204 accepted");
        require(reject("206") == 206, "unexpected206 accepted");
        require(reject("500") == 500, "HTTP error status missing");
        for (const auto& sample : std::vector<std::pair<std::string,unsigned>>{
                 {"retry",7}, {"retry-large",61}, {"retry-date",0}, {"retry-negative",0}, {"retry-duplicate",0}, {"retry-503",7}}) {
            bool caught = false;
            try { get(sample.first); }
            catch (const hub::TransportError& e) {
                caught = e.status == (sample.first == "retry-503" ? 503 : 429) &&
                    e.curl_code == 22 && e.retry_after_seconds == sample.second;
            }
            require(caught, "Retry-After handling failed");
        }
        reject("early"); reject("nomarker"); reject("auth", 4);
        const hub::ByteRange range{3,4,10};
        require(get("range", 100, range).bytes == 4 && read(out) == "DATA", "range failed");
        require(reject("ignored", 100, range) == 200, "ignored range status missing");
        for (const auto* name : {"missing", "wrong", "total", "duplicate", "short", "extra"})
            reject(name, 100, range);
        reject("range", 100, hub::ByteRange{9,2,10});
        reject("range", 100, hub::ByteRange{0,0,10});
        reject("range", 100, hub::ByteRange{UINT64_MAX,2,UINT64_MAX});
        for (const auto* version : {"7.83.0", "8.3.9", "invalid"}) {
            Env old("LLMX_TEST_CURL_VERSION", version);
            write(out, "preserve"); reject("auth");
            require(read(out) == "preserve", "old curl modified output");
        }
        for (const auto& url : {std::string("http://example.test/a"), std::string("https://user:pass@example.test/a"),
             std::string("https://example.test/a\nheader=x"), std::string("https://example.test/a\0b",24)}) {
            bool failed = false;
            try { hub::transport_detail::download_with(child, url, "", out, 100, {}); }
            catch (const hub::TransportError&) { failed = true; }
            require(failed, "invalid URL accepted");
        }
        bool failed = false;
        try { hub::transport_detail::download_with(child, "https://example.test/a", "a\rb", out, 100, {}); }
        catch (const hub::TransportError&) { failed = true; }
        require(failed, "header injection accepted");
        failed = false;
        try { hub::transport_detail::download_with(child, "https://example.test/" + std::string(5000,'x'), "", out, 100, {}); }
        catch (const hub::TransportError&) { failed = true; }
        require(failed, "unbounded configuration accepted");
        failed = false;
        try { hub::transport_detail::download_with(root / "absent-curl", "https://example.test/a", "", out, 100, {}); }
        catch (const hub::TransportError&) { failed = true; }
        require(failed, "missing executable accepted");
#ifdef _WIN32
        DWORD handles_after = 0;
        require(GetProcessHandleCount(GetCurrentProcess(), &handles_after) != FALSE, "handle query failed");
        if (handles_after != handles_before) std::cerr << "handle counts: " << handles_before << " -> " << handles_after << '\n';
        require(handles_after == handles_before, "child handle leaked");
#endif
        std::vector<std::future<hub::Response>> concurrent;
        for (int i = 0; i < 8; ++i) concurrent.push_back(std::async(std::launch::async, [&,i] {
            return hub::transport_detail::download_with(child, "https://example.test/noise", "",
                root / ("parallel-" + std::to_string(i)), 100, {});
        }));
        for (auto& f : concurrent) require(f.get().bytes == 5, "parallel process failure");
        require(get("auth").bytes == 5, "reuse after failure failed");
#ifndef _WIN32
        require(fd_count() == fds_before, "child descriptor leaked");
        struct sigaction signal_after{};
        require(sigaction(SIGPIPE, nullptr, &signal_after) == 0 &&
                signal_after.sa_handler == signal_before.sa_handler &&
                signal_after.sa_flags == signal_before.sa_flags, "SIGPIPE policy changed");
#endif
        fs::remove_all(root);
        std::cout << "hub-transport: " << checks << " checks passed (fake child; no network)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "hub-transport: " << e.what() << '\n';
        if (!root.empty()) std::cerr << "fixtures retained: " << root << '\n';
        return 1;
    }
}
