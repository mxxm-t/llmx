#pragma once
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <random>
#include <thread>
#include "core/sha.hpp"
#include "hub/manifest.hpp"
#include "hub/transport.hpp"

namespace hub {

struct PullOptions {
    std::string repo, quant, revision = "main", filename, token;
    std::filesystem::path cache;
    unsigned parallel = 4;
};
using PullProgress = std::function<void(const std::string&)>;

namespace pull_detail {

constexpr unsigned max_retry_after_seconds = 60;
constexpr uint64_t max_metadata_bytes = 16 * 1024 * 1024;

inline std::filesystem::path default_cache() {
#ifdef _WIN32
    const DWORD count = GetEnvironmentVariableW(L"USERPROFILE", nullptr, 0);
    if (count) {
        std::wstring home(count, L'\0');
        const DWORD copied = GetEnvironmentVariableW(L"USERPROFILE", home.data(), count);
        if (copied && copied < count) {
            home.resize(copied);
            return std::filesystem::path(home) / L".cache" / L"llmx";
        }
    }
#else
    const char* home = std::getenv("HOME");
    if (home && *home) return std::filesystem::path(home) / ".cache" / "llmx";
#endif
    throw std::runtime_error("pull: home directory unavailable; use --cache-dir");
}

class Temporary {
public:
    std::filesystem::path path;
    explicit Temporary(const std::filesystem::path& root) {
        std::random_device random;
        for (unsigned attempt = 0; attempt < 20; ++attempt) {
            auto candidate = root / (".pull-" + std::to_string(random()) + "-" + std::to_string(random()));
            if (std::filesystem::create_directory(candidate)) { path = std::move(candidate); return; }
        }
        throw std::runtime_error("pull: cannot create temporary cache directory");
    }
    Temporary(const Temporary&) = delete;
    Temporary& operator=(const Temporary&) = delete;
    ~Temporary() {
        std::error_code ec;
        if (!path.empty()) std::filesystem::remove_all(path, ec);
    }
};

inline std::string read_metadata(const std::filesystem::path& path) {
    const auto size = std::filesystem::file_size(path);
    if (size > max_metadata_bytes)
        throw std::runtime_error("pull: Hub metadata exceeds " + std::to_string(max_metadata_bytes / (1024 * 1024)) + " MiB");
    std::ifstream input(path, std::ios::binary);
    input.exceptions(std::ios::badbit | std::ios::failbit);
    std::string text(size_t(size), '\0');
    if (size) input.read(&text[0], std::streamsize(size));
    return text;
}

inline core::Sha file_hasher(const File& file) {
    core::Sha hash(file.lfs);
    if (!file.lfs) {
        const std::string header = "blob " + std::to_string(file.size) + '\0';
        hash.update(header.data(), header.size());
    }
    return hash;
}

inline void consume(std::istream& input, uint64_t bytes, core::Sha& hash, std::ostream* output = nullptr) {
    std::vector<char> buffer(1024 * 1024);
    while (bytes) {
        const size_t count = size_t(std::min<uint64_t>(bytes, buffer.size()));
        input.read(buffer.data(), std::streamsize(count));
        if (!input) throw std::runtime_error("pull: truncated file during verification");
        hash.update(buffer.data(), count);
        if (output) output->write(buffer.data(), std::streamsize(count));
        bytes -= count;
    }
    if (input.peek() != std::char_traits<char>::eof() || input.bad())
        throw std::runtime_error("pull: file extent changed during verification");
}

inline bool verified(const std::filesystem::path& path, const File& file) {
    if (std::filesystem::is_symlink(std::filesystem::symlink_status(path)))
        throw std::runtime_error("pull: cache file must not be a symlink");
    if (!std::filesystem::exists(path)) return false;
    if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("pull: cache entry is not a file");
    if (std::filesystem::file_size(path) != file.size) return false;
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("pull: cannot read cached file");
    auto hash = file_hasher(file);
    consume(input, file.size, hash);
    return hash.hex() == file.digest;
}

inline std::filesystem::path cache_path(const std::filesystem::path& root,
                                      const std::string& relative) {
    validate_path(relative);
    std::filesystem::path result = root;
    const auto path = std::filesystem::u8path(relative);
    for (const auto& component : path.parent_path()) {
        result /= component;
        if (std::filesystem::is_symlink(std::filesystem::symlink_status(result)))
            throw std::runtime_error("pull: cache directory must not be a symlink");
        std::filesystem::create_directory(result);
        if (!std::filesystem::is_directory(result)) throw std::runtime_error("pull: invalid cache directory");
    }
    return result / path.filename();
}

inline void publish(const std::filesystem::path& partial, const std::filesystem::path& destination) {
#ifdef _WIN32
    if (!MoveFileExW(partial.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING))
        throw std::runtime_error("pull: cannot publish verified cache file");
#else
    std::filesystem::rename(partial, destination);
#endif
}

inline bool transient(const TransportError& error) {
    return error.status == 408 || error.status == 429 || error.status == 500 ||
           error.status == 502 || error.status == 503 || error.status == 504 ||
           (!error.status && (error.curl_code == 5 || error.curl_code == 6 || error.curl_code == 7)) ||
           error.curl_code == 18 || error.curl_code == 28 || error.curl_code == 52 ||
           error.curl_code == 55 || error.curl_code == 56 || error.curl_code == 92;
}

template<class Fetch>
inline void request(Fetch& fetch, const std::string& url, const std::string& token,
                    const std::filesystem::path& destination, uint64_t limit,
                    std::optional<ByteRange> range = std::nullopt) {
    for (unsigned attempt = 0; ; ++attempt) {
        try { fetch(url, token, destination, limit, range); return; }
        catch (const TransportError& error) {
            if (attempt == 4 || !transient(error)) throw;
            if (error.retry_after_seconds > max_retry_after_seconds)
                throw TransportError("server requests a wait longer than " + std::to_string(max_retry_after_seconds) +
                                     " seconds; retry the pull later", error.status, error.curl_code);
            const unsigned seconds = std::max(1u << attempt, error.retry_after_seconds);
            std::this_thread::sleep_for(std::chrono::seconds(seconds));
        }
    }
}

template<class Fetch>
inline void acquire(Fetch& fetch, const PullOptions& options, const Manifest& manifest,
                    const File& file, const std::filesystem::path& temporary,
                    const std::filesystem::path& destination, const PullProgress& progress) {
    if (progress) progress("Checking " + file.name);
    if (verified(destination, file)) {
        if (progress) progress("Cached " + file.name);
        return;
    }
    const std::string url = "https://huggingface.co/" + options.repo + "/resolve/" +
                            manifest.revision + "/" + url_encode(file.name, true);
    constexpr uint64_t minimum_part = 8 * 1024 * 1024;
    const unsigned streams = unsigned(std::min<uint64_t>(options.parallel,
        std::max<uint64_t>(1, file.size / minimum_part)));
    if (progress) progress("Downloading " + file.name + " (" + std::to_string(streams) + " streams)");
    const auto assembled = temporary / "verified.part";
    if (streams == 1) {
        request(fetch, url, options.token, assembled, file.size);
        if (!verified(assembled, file)) throw std::runtime_error("pull: downloaded file size or digest mismatch");
    } else {
        std::vector<ByteRange> ranges;
        std::vector<std::future<void>> tasks;
        ranges.reserve(streams);
        tasks.reserve(streams);
        const uint64_t width = file.size / streams, remainder = file.size % streams;
        uint64_t offset = 0;
        for (unsigned i = 0; i < streams; ++i) {
            const uint64_t length = width + (i < remainder ? 1 : 0);
            ranges.push_back({offset, length, file.size}); offset += length;
        }
        for (unsigned i = 0; i < streams; ++i) {
            tasks.push_back(std::async(std::launch::async, [&, i] {
                request(fetch, url, options.token, temporary / (std::to_string(i) + ".part"),
                        ranges[i].length, ranges[i]);
            }));
        }
        std::exception_ptr failure;
        for (auto& task : tasks) {
            try { task.get(); } catch (...) { if (!failure) failure = std::current_exception(); }
        }
        if (failure) std::rethrow_exception(failure);
        if (progress) progress("Verifying " + file.name);
        std::ofstream output(assembled, std::ios::binary | std::ios::trunc);
        output.exceptions(std::ios::badbit | std::ios::failbit);
        auto hash = file_hasher(file);
        for (unsigned i = 0; i < streams; ++i) {
            const auto part = temporary / (std::to_string(i) + ".part");
            if (std::filesystem::file_size(part) != ranges[i].length)
                throw std::runtime_error("pull: incomplete byte range");
            std::ifstream input(part, std::ios::binary);
            consume(input, ranges[i].length, hash, &output);
            input.close();
            std::filesystem::remove(part);
        }
        output.close();
        if (hash.hex() != file.digest) throw std::runtime_error("pull: assembled download digest mismatch");
    }
    publish(assembled, destination);
    if (progress) progress("Ready " + file.name);
}

template<class Fetch>
inline std::filesystem::path pull(PullOptions options, const PullProgress& progress, Fetch fetch) {
    validate_repo(options.repo);
    validate_quant(options.quant);
    if (!options.filename.empty()) validate_path(options.filename);
    if (options.revision.empty() || options.revision.size() > 256)
        throw std::runtime_error("pull: invalid revision");
    transport_detail::clean_text(options.revision);
    if (options.parallel < 1 || options.parallel > 16) throw std::runtime_error("pull: --parallel must be between 1 and 16");
    if (options.cache.empty()) options.cache = default_cache();
    std::filesystem::create_directories(options.cache);
    const auto root = std::filesystem::canonical(options.cache);
    Temporary temporary(root);
    if (progress) progress("Resolving " + options.repo + " at " + options.revision);
    const std::string url = "https://huggingface.co/api/models/" + options.repo + "/revision/" +
                            url_encode(options.revision) + "?blobs=true";
    const auto metadata = temporary.path / "metadata.json";
    request(fetch, url, options.token, metadata, max_metadata_bytes);
    const Manifest manifest = select(read_metadata(metadata), options.quant, options.filename);
    if (hex_digest(options.revision, 40) && manifest.revision != options.revision)
        throw std::runtime_error("pull: Hub returned a different revision");
    if (progress) progress("Resolved commit " + manifest.revision);
    std::string repo_directory = "models--" + options.repo;
    repo_directory.replace(repo_directory.find('/'), 1, "--");
    std::filesystem::path first;
    for (const auto& file : manifest.files) {
        const auto destination = cache_path(root, repo_directory + "/snapshots/" + manifest.revision + "/" + file.name);
        acquire(fetch, options, manifest, file, temporary.path, destination, progress);
        if (first.empty()) first = destination;
    }
    return first;
}

} // namespace pull_detail

inline std::filesystem::path pull(const PullOptions& options, const PullProgress& progress = {}) {
    return pull_detail::pull(options, progress, download);
}

} // namespace hub
