#include "hub/pull.hpp"
#include <atomic>
#include <iostream>

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

static void rejects(const std::function<void()>& body) {
    bool caught = false;
    try { body(); } catch (const std::exception&) { caught = true; }
    require(caught, "failure was accepted");
}

static void write(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.write(text.data(), std::streamsize(text.size()));
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("fixture root argument required");
        const auto parent = std::filesystem::u8path(argv[1]);
        std::filesystem::create_directories(parent);
        hub::pull_detail::Temporary fixture(parent);
        hub::PullOptions options;
        options.repo = "test/model"; options.quant = "Q8_0";
        options.cache = fixture.path / std::filesystem::u8path("cache-\xc3\xa9");
        options.token = "private-fixture-token";
        const uint64_t size = 33554449;
        const std::string digest = "dff5b2f0725ee66b6c1fec1e6118b91622d727cda303e120c46c6e53d2deae96";
        const std::string revision(40, 'a');
        const std::string metadata = "{\"sha\":\"" + revision + "\",\"siblings\":[{\"rfilename\":\"model-Q8_0.gguf\",\"size\":" +
            std::to_string(size) + ",\"lfs\":{\"size\":" + std::to_string(size) + ",\"sha256\":\"" + digest + "\"}}]}";
        std::atomic<unsigned> active{0}, peak{0}, requests{0}, metadata_requests{0};
        bool fail = false, truncate = false, corrupt = false, retry = false;
        auto fetch = [&](const std::string& url, const std::string& token, const std::filesystem::path& partial,
                         uint64_t limit, std::optional<hub::ByteRange> range) -> hub::Response {
            require(token == options.token, "credential not passed to transport");
            if (url.find("/api/models/") != std::string::npos) {
                ++metadata_requests;
                require(url.find("/revision/main?blobs=true") != std::string::npos, "revision metadata URL");
                write(partial, metadata); return {200, uint64_t(metadata.size())};
            }
            require(url == "https://huggingface.co/test/model/resolve/" + revision + "/model-Q8_0.gguf", "mutable download URL");
            const unsigned request = requests++;
            if (retry && request == 0) {
                write(partial, "partial response");
                throw hub::TransportError("retry fixture", 429, 22);
            }
            const unsigned concurrent = ++active;
            unsigned previous = peak.load();
            while (concurrent > previous && !peak.compare_exchange_weak(previous, concurrent)) {}
            struct Done { std::atomic<unsigned>& n; ~Done() { --n; } } done{active};
            if (range) {
                require(range->total == size && range->length == limit && range->offset + range->length <= size, "invalid range plan");
                std::this_thread::sleep_for(std::chrono::milliseconds(30));
                if (fail && !range->offset) throw hub::TransportError("range failure fixture", 403, 22);
            }
            std::ofstream output(partial, std::ios::binary | std::ios::trunc);
            output.exceptions(std::ios::badbit | std::ios::failbit);
            const uint64_t offset = range ? range->offset : 0;
            const uint64_t length = (range ? range->length : size) - (truncate ? 1 : 0);
            std::vector<char> buffer(65536);
            for (uint64_t pos = 0; pos < length;) {
                const size_t count = size_t(std::min<uint64_t>(buffer.size(), length - pos));
                for (size_t i = 0; i < count; ++i) buffer[i] = char((offset + pos + i + (corrupt ? 1 : 0)) % 251);
                output.write(buffer.data(), std::streamsize(count)); pos += count;
            }
            return {range ? 206 : 200, length};
        };
        std::vector<std::string> events;
        auto progress = [&](const std::string& event) { events.push_back(event); };
        const auto downloaded = hub::pull_detail::pull(options, progress, fetch);
        require(peak == 4 && requests == 4 && active == 0, "four streams did not overlap or join");
        require(downloaded == std::filesystem::canonical(options.cache) / "models--test--model" / "snapshots" / revision / "model-Q8_0.gguf", "cache layout");
        require(events.back() == "Ready model-Q8_0.gguf", "missing completion progress");
        const hub::File file{"model-Q8_0.gguf", size, digest, true};
        require(hub::pull_detail::verified(downloaded, file), "assembled digest differs");
        requests = 0;
        require(hub::pull_detail::pull(options, {}, fetch) == downloaded && requests == 0, "cache hit downloaded again");
        require(metadata_requests == 2, "revision was not refreshed");
        {
            std::fstream changed(downloaded, std::ios::in | std::ios::out | std::ios::binary);
            changed.put('X');
        }
        hub::pull_detail::pull(options, {}, fetch);
        require(requests == 4 && hub::pull_detail::verified(downloaded, file), "corrupt cache was not repaired");
        std::filesystem::remove(downloaded);
        fail = true; requests = 0;
        rejects([&] { hub::pull_detail::pull(options, {}, fetch); });
        require(active == 0 && !std::filesystem::exists(downloaded), "failure published output or left active streams");
        fail = false; truncate = true;
        rejects([&] { hub::pull_detail::pull(options, {}, fetch); });
        require(!std::filesystem::exists(downloaded), "truncated file published");
        truncate = false; corrupt = true;
        rejects([&] { hub::pull_detail::pull(options, {}, fetch); });
        require(!std::filesystem::exists(downloaded), "incorrect digest published");
        corrupt = false; retry = true; options.parallel = 1; requests = 0; peak = 0;
        hub::pull_detail::pull(options, {}, fetch);
        require(requests == 2 && peak == 1 && hub::pull_detail::verified(downloaded, file), "serial retry appended partial bytes");
        for (const auto& entry : std::filesystem::directory_iterator(options.cache))
            require(entry.path().filename().string().find(".pull-") != 0, "temporary state leaked");
        const auto git_file = fixture.path / "git.bin";
        write(git_file, "abc");
        require(hub::pull_detail::verified(git_file, {"git.bin",3,"f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f",false}), "Git blob identity");
        options.parallel = 17;
        rejects([&] { hub::pull_detail::pull(options, {}, fetch); });
        unsigned refused_requests = 0;
        auto long_wait = [&](const std::string&, const std::string&, const std::filesystem::path&,
                             uint64_t, std::optional<hub::ByteRange>) -> hub::Response {
            ++refused_requests;
            throw hub::TransportError("long wait fixture", 429, 22, 61);
        };
        rejects([&] { hub::pull_detail::request(long_wait, "https://example.test/model", "", git_file, 3); });
        require(refused_requests == 1, "retried before long server delay");
        std::cout << "hub: four overlapping ranges, exact assembly, cache repair, failure cleanup, serial retry and Git digest pass\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
