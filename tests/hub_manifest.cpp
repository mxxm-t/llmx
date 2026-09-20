#include "hub/manifest.hpp"
#include "core/sha.hpp"
#include <iostream>
#include <functional>

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

static void rejects(const std::function<void()>& body) {
    bool caught = false;
    try { body(); } catch (const std::runtime_error&) { caught = true; }
    require(caught, "invalid metadata accepted");
}

static std::string file(const std::string& name, bool lfs = true) {
    return "{\"rfilename\":\"" + name + "\",\"size\":42," + (lfs ?
        "\"lfs\":{\"size\":42,\"sha256\":\"" + std::string(64, 'a') + "\"}" :
        "\"blobId\":\"" + std::string(40, 'b') + "\"") + "}";
}

static std::string metadata(const std::string& files) {
    return "{\"sha\":\"" + std::string(40, 'c') + "\",\"siblings\":[" + files + "]}";
}

int main() {
    try {
        struct Vector { std::string text, sha1, sha256; };
        const Vector vectors[] = {
            {"", "da39a3ee5e6b4b0d3255bfef95601890afd80709", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
            {"abc", "a9993e364706816aba3e25717850c26c9cd0d89d", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
            {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", "84983e441c3bd26ebaae4aa1f95129e5e54670f1", "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
            {std::string(1000000, 'a'), "34aa973cd4c4daa4f61eeb2bdbad27316534016f", "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"}
        };
        for (const auto& v : vectors) for (bool sha256 : {false, true}) {
            for (size_t chunk : {size_t(1), size_t(63), size_t(64), size_t(129), size_t(65536)}) {
                core::Sha hash(sha256);
                for (size_t i = 0; i < v.text.size(); i += chunk)
                    hash.update(v.text.data() + i, std::min(chunk, v.text.size() - i));
                const auto expected = sha256 ? v.sha256 : v.sha1;
                require(hash.hex() == expected && hash.hex() == expected, "SHA standard vector differs");
            }
        }
        const auto single = hub::select(metadata(file("Qwen-Q8_0.gguf")), "q8_0");
        require(single.files.size() == 1 && single.files[0].lfs && single.files[0].size == 42, "single selection");
        require(!hub::select(metadata(file("Qwen.F32.gguf", false)), "F32").files[0].lfs, "Git selection");
        require(hub::select(metadata(file("model_Q8_0.gguf")), "q8_0").files.size() == 1, "underscore quant separator");
        rejects([&] { hub::select(metadata("{\"rfilename\":\"x-Q8_0.gguf\",\"size\":42,\"lfs\":null,\"lfs\":{},\"blobId\":\"" + std::string(40, 'a') + "\"}"), "q8_0"); });
        const std::string first = "Q8_0/model-Q8_0-00001-of-00002.gguf";
        const std::string second = "Q8_0/model-Q8_0-00002-of-00002.gguf";
        const auto shards = hub::select(metadata(file(second) + "," + file(first)), "Q8_0", second);
        require(shards.files.size() == 2 && shards.files[0].name == first, "shard ordering");
        const auto mixed = metadata(file(second) + "," + file(first) + "," + file("Q8_0/model-Q8_0.gguf"));
        require(hub::select(mixed, "q8_0", first).files.size() == 2, "explicit shards alongside single file");
        require(hub::select(mixed, "q8_0", "Q8_0/model-Q8_0.gguf").files.size() == 1, "explicit single alongside shards");
        rejects([&] { hub::select(metadata(file("x-Q8_0-00001-of-65536.gguf")), "q8_0"); });
        rejects([&] { hub::select(metadata(file("x-Q8_0-00001-of-00001.GGUF")), "q8_0"); });
        require(hub::select(metadata(file("x-Q8_0.gguf") + "," + file("x-Q4_K_M-00001-of-00002.GGUF")), "q8_0").files.size() == 1, "unrelated quant validation interferes");
        rejects([&] { hub::select(metadata(file(first)), "q8_0"); });
        rejects([&] { hub::select(metadata(file(first) + "," + file(first)), "q8_0"); });
        rejects([&] { hub::select(metadata(file("x-Q8_0.gguf") + "," + file("X-q8_0.gguf")), "q8_0"); });
        rejects([&] { hub::select(metadata(file("a-Q8_0.gguf") + "," + file("b-Q8_0.gguf")), "q8_0"); });
        const auto selected = hub::select(metadata(file("a-Q8_0.gguf") + "," + file("b-Q8_0.gguf")), "q8_0", "b-Q8_0.gguf");
        require(selected.files[0].name == "b-Q8_0.gguf", "explicit file selection");
        rejects([&] { hub::select(metadata(file("x-IQ8_0.gguf")), "q8_0"); });
        rejects([&] { hub::select(metadata(file("../x-Q8_0.gguf")), "q8_0"); });
        rejects([&] { hub::select(metadata(file("x-Q8_0.gguf")), "Q4_0"); });
        rejects([&] { hub::select("{\"sha\":\"main\",\"siblings\":[]}", "q8_0"); });
        rejects([&] { hub::select("{\"sha\":\"" + std::string(40,'a') + "\",\"sha\":\"" + std::string(40,'b') + "\",\"siblings\":[]}", "q8_0"); });
        for (const std::string path : {"/root.gguf", "x/../a", "x//a", "C:a", "x\\a", "CON.gguf", "x/lpt1.GGUF", "x./a", "x/a ", "a/"})
            rejects([&] { hub::validate_path(path); });
        for (const std::string repo : {"owner", "/repo", "owner/", "a/b/c", "a/../b", "a/b--c", "a/b?c"})
            rejects([&] { hub::validate_repo(repo); });
        hub::validate_repo("Qwen/Qwen3-0.6B-GGUF");
        require(hub::url_encode("refs/pr/1") == "refs%2Fpr%2F1", "revision URL quoting");
        require(hub::url_encode("folder/a b.gguf", true) == "folder/a%20b.gguf", "file URL quoting");
        std::cout << "hub: SHA vectors, quant selection, shards, metadata and paths pass\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
