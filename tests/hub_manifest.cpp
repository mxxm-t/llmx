#include "hub/manifest.hpp"
#include "core/sha.hpp"
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

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

// The refusal of `body`, which must name `text`.
static void refuses(const std::function<void()>& body, const std::string& text) {
    std::string error;
    try { body(); } catch (const std::runtime_error& e) { error = e.what(); }
    require(error.find(text) != std::string::npos, ("refusal without '" + text + "': " + error).c_str());
}

// The names a directory selection takes, in its order.
static std::string names(const hub::Manifest& m) {
    std::string out;
    for (const auto& f : m.files) out += (out.empty() ? "" : ",") + f.name;
    return out;
}

// select_directory on the sibling list the Hub gave for Qwen/Qwen3-0.6B at c1899de, recorded, and on lists made from it.
static void directory_checks() {
    const std::string qwen = R"({"sha":"c1899de289a04d12100db370d81485cdf75e47ca","siblings":[)"
        R"({"rfilename":".gitattributes","size":1570,"blobId":"52373fe24473b1aa44333d318f578ae6bf04b49b"},)"
        R"({"rfilename":"LICENSE","size":11343,"blobId":"6634c8cc3133b3848ec74b9f275acaaa1ea618ab"},)"
        R"({"rfilename":"README.md","size":13965,"blobId":"a50b19e76f5274f9ec99f5a5d99873dca5bff25e"},)"
        R"({"rfilename":"config.json","size":726,"blobId":"f5c3703b78ae2a478ae15b247e9f855e0ce2107b"},)"
        R"({"rfilename":"generation_config.json","size":239,"blobId":"20a8a9156fc8c3f25295ca067f61fdf120d517c5"},)"
        R"({"rfilename":"merges.txt","size":1671853,"blobId":"31349551d90c7606f325fe0f11bbb8bd5fa0d7c7"},)"
        R"({"rfilename":"model.safetensors","size":1503300328,"blobId":"a0458bbfda764d86cf930900d5f0f933933da9a3",)"
        R"("lfs":{"sha256":"f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b","size":1503300328,"pointerSize":135}},)"
        R"({"rfilename":"tokenizer.json","size":11422654,"blobId":"cd71f61a15a522601badb3dc960d800d9cb3766c",)"
        R"("lfs":{"sha256":"aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4","size":11422654,"pointerSize":133}},)"
        R"({"rfilename":"tokenizer_config.json","size":9732,"blobId":"417d038a63fa3de29cfde265caedae14d1a58d92"},)"
        R"({"rfilename":"vocab.json","size":2776833,"blobId":"4783fe10ac3adce15ac8f358ef5462739852c569"}]})";
    const auto m = hub::select_directory(qwen);
    require(m.revision == "c1899de289a04d12100db370d81485cdf75e47ca" && names(m) == "config.json,tokenizer.json,tokenizer_config.json,model.safetensors",
            "the Qwen3-0.6B directory's files");
    require(m.files[3].lfs && m.files[3].size == 1503300328 && m.files[3].digest == "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b" &&
            !m.files[0].lfs && m.files[0].digest == "f5c3703b78ae2a478ae15b247e9f855e0ce2107b", "the Qwen3-0.6B directory's digests");
    const std::string common = file("config.json", false) + "," + file("tokenizer.json") + "," + file("tokenizer_config.json", false);
    const std::string index = file("model.safetensors.index.json", false);
    const std::string one = file("model-00001-of-00002.safetensors"), two = file("model-00002-of-00002.safetensors");
    require(names(hub::select_directory(metadata(common + "," + index + "," + two + "," + one + "," + file("chat_template.jinja", false) + "," +
                                                 file("onnx/model.safetensors") + "," + file("original/consolidated.safetensors")))) ==
            "config.json,tokenizer.json,tokenizer_config.json,chat_template.jinja,model.safetensors.index.json,model-00001-of-00002.safetensors,model-00002-of-00002.safetensors",
            "a sharded directory's files, its subfolders left out");
    refuses([&] { hub::select_directory(metadata(common + "," + index + "," + one)); }, "incomplete safetensors shard set");
    refuses([&] { hub::select_directory(metadata(common + "," + index + "," + two)); }, "incomplete safetensors shard set");
    refuses([&] { hub::select_directory(metadata(common + "," + index + "," + one + "," + two + "," + file("consolidated.safetensors"))); }, "consolidated.safetensors is not a shard");
    refuses([&] { hub::select_directory(metadata(common + "," + index + "," + one + "," + two + "," + file("model.safetensors"))); }, "model.safetensors is not a shard");
    refuses([&] { hub::select_directory(metadata(common + "," + index + "," + file("model-00001-of-00001.safetensors") + "," + two)); }, "model-00002-of-00002.safetensors is not a shard");
    refuses([&] { hub::select_directory(metadata(common + "," + one + "," + two)); }, "model-00001-of-00002.safetensors has no model.safetensors.index.json");
    refuses([&] { hub::select_directory(metadata(common + "," + file("model.safetensors") + "," + file("adapter.safetensors"))); }, "adapter.safetensors has no model.safetensors.index.json");
    refuses([&] { hub::select_directory(metadata(file("tokenizer.json") + "," + file("model.safetensors"))); }, "no config.json");
    refuses([&] { hub::select_directory(metadata(common + "," + file("pytorch_model.bin"))); }, ".bin files only");
    refuses([&] { hub::select_directory(metadata(common)); }, "no .safetensors weights");
    refuses([&] { hub::select_directory(metadata(file("config.json", false) + "," + file("tokenizer_config.json", false) + "," + file("model.safetensors"))); }, "no tokenizer.json");
    refuses([&] { hub::select_directory(metadata(common + "," + file("model.safetensors") + "," + file("model.safetensors"))); }, "duplicate or case-colliding repository file model.safetensors");
    refuses([&] { hub::select_directory(metadata(common + "," + file("Config.json", false) + "," + file("model.safetensors"))); }, "case-colliding repository file Config.json");
    refuses([&] { hub::select_directory(metadata(common + ",{\"rfilename\":\"model.safetensors\",\"size\":0,\"blobId\":\"" + std::string(40, 'b') + "\"}")); }, "empty file model.safetensors");
    refuses([&] { hub::select_directory("{\"sha\":\"main\",\"siblings\":[]}"); }, "invalid resolved commit SHA");
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
        // The SHA extensions' compression gives the portable code's state, block counts from 1 to 33 over random data, where the CPU has them.
#ifdef LLMX_SHA_EXTENSIONS
        if (core::sha_detail::sha_extensions()) {
            uint64_t seed = 0x243f6a8885a308d3ULL;
            std::vector<uint8_t> data(64 * 33);
            for (auto& byte : data) {
                seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
                byte = uint8_t(seed >> 56);
            }
            for (size_t n = 1; n <= 33; ++n) {
                uint32_t portable[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
                uint32_t extensions[8];
                std::copy(portable, portable + 8, extensions);
                core::sha_detail::sha256_portable(portable, data.data(), n);
                core::sha_detail::sha256_extensions(extensions, data.data(), n);
                require(std::equal(portable, portable + 8, extensions), ("the SHA extensions' state differs from the portable code's after " + std::to_string(n) + " blocks").c_str());
            }
            std::printf("hub-manifest: SHA-256 checked with the SHA extensions\n");
        }
#endif
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
        directory_checks();
        hub::validate_repo("Qwen/Qwen3-0.6B-GGUF");
        require(hub::url_encode("refs/pr/1") == "refs%2Fpr%2F1", "revision URL quoting");
        require(hub::url_encode("folder/a b.gguf", true) == "folder/a%20b.gguf", "file URL quoting");
        std::cout << "hub: SHA vectors, quant selection, shards, safetensors directories, metadata and paths pass\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
