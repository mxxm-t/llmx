#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "format/gguf.hpp"
#include "inference/load.hpp"
#include "tiny_qwen.hpp"

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

// The file at `path` read, mapped and read in as the loader does, its payload reported to `progress`.
gguf::GGUFModel load_file(const std::string& path, const format::LoadProgress& progress) {
    gguf::GGUFModel m = gguf::read_gguf(path);
    gguf::map_payload(m);
    gguf::warm(m, progress);
    return m;
}

// A CPU backend playing a device: it copies what it adopts, so no weight reads the file in place.
struct CopyingBackend : backend::CpuBackend {
    bool reads_in_place() const override { return false; }
    backend::BufferPtr adopt(const void* src, size_t bytes) override {
        auto buffer = alloc(bytes, backend::Memory::device);
        write(*buffer, 0, src, bytes);
        return buffer;
    }
};

// A model file's weights have no bytes until its payload is mapped.
// Loaded through infer::load_model it keeps its payload on the CPU and releases it on a backend that copies every weight, and both give the logits of the same model built in memory.
void loader_checks(const std::string& path) {
    gguf::GGUFModel source = tiny_qwen(2, 2 * 128, false);
    gguf::MetaValue tokens;
    tokens.vtype = gguf::V_ARRAY;
    tokens.u = gguf::V_STRING;
    for (char c = 'a'; c < 'a' + 16; ++c) {
        gguf::MetaValue token;
        token.vtype = gguf::V_STRING;
        token.s = std::string(1, c);
        tokens.arr.push_back(token);
    }
    gguf::MetaValue eos;
    eos.vtype = gguf::V_UINT32;
    eos.u = 15;
    source.kv.push_back({"tokenizer.ggml.tokens", tokens});
    source.kv.push_back({"tokenizer.ggml.eos_token_id", eos});
    gguf::write_gguf(source, path);
    {
        gguf::GGUFModel file = gguf::read_gguf(path);
        const infer::QwenWeights unmapped = infer::gguf_weights(file);
        require(std::all_of(unmapped.tensors.begin(), unmapped.tensors.end(), [](const infer::TensorView& t) { return !t.data; }),
                "a weight had bytes before its file was mapped");
        gguf::map_payload(file);
        const infer::QwenWeights mapped = infer::gguf_weights(file);
        for (size_t i = 0; i < mapped.tensors.size(); ++i)
            require(mapped.tensors[i].data && std::memcmp(mapped.tensors[i].data, source.tensor_data(i), source.tensor_bytes(i)) == 0,
                    "a mapped weight's bytes differ from the file's");
    }
    const std::vector<uint32_t> ids = {0, 1, 2, 3, 4};
    const std::vector<float> expected = infer::Model(source, backend::make_cpu_backend()).prefill(ids);
    for (const bool copying : {false, true}) {
        backend::BackendPtr b = copying ? std::make_shared<CopyingBackend>() : backend::make_cpu_backend();
        infer::PlacementRequest request;
        request.names = {"cpu"};
        size_t total = 0, completed = 0;
        const auto loaded = infer::load_model(path, {b}, request, {}, [&](size_t done, size_t all) { completed = done; total = all; });
        require(total && completed == total, "the loader lost the file's progress");
        require(loaded->tok && loaded->tok->encode("abcde") == ids && loaded->chat.eos == "p",
                "the loader lost the tokenizer or the chat format");
        require(loaded->plan.empty(), "one device reported a split");
        require(copying ? loaded->file.payload_size() == 0 : loaded->file.payload_size() != 0,
                copying ? "a model no host reads kept the host copy" : "a model the CPU reads lost its payload");
        const std::vector<float> logits = loaded->model->prefill(ids);
        require(logits.size() == expected.size() &&
                std::memcmp(logits.data(), expected.data(), logits.size() * sizeof(float)) == 0,
                "a loaded model's logits differ from the model built in memory");
    }
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string path = argv[1];
    try {
        gguf::GGUFModel source;
        source.tensors.push_back({"quant", {32}, gguf::GGML_TYPE_Q8_0, 0});
        source.add_tensor_data(std::vector<uint8_t>(34, 17));
        const size_t bytes = 8 * 1024 * 1024 + 64;
        source.tensors.push_back({"large", {bytes / 4}, gguf::GGML_TYPE_F32, 0});
        std::vector<uint8_t> values(bytes);
        for (size_t i = 0; i < values.size(); ++i) values[i] = uint8_t(i * 31);
        source.add_tensor_data(values);
        gguf::write_gguf(source, path);
        std::vector<size_t> seen;
        const format::LoadProgress progress = [&](size_t done, size_t total) {
            require(total == bytes + 34, "progress counts padding as payload");
            require(done <= total, "progress exceeds total");
            require(seen.empty() ? done == 0 : done > seen.back(), "non-monotonic progress");
            seen.push_back(done);
        };
        // A mapped model holds its file, which a rewrite below must not see held.
        {
            const auto in_file = std::filesystem::file_size(path);
            auto loaded = gguf::read_gguf(path);
            require(loaded.tensors.size() == 2 && loaded.tensors[0].name == "quant" && loaded.tensors[1].name == "large" &&
                    loaded.tensors[1].ne == source.tensors[1].ne && loaded.tensors[1].type == gguf::GGML_TYPE_F32,
                    "reading changed the tensor table");
            for (size_t i = 0; i < source.tensors.size(); ++i) require(!loaded.tensor_data(i), "reading the file mapped it");
            require(seen.empty(), "reading the file reported progress");
            // A model whose file is not mapped is refused before anything is written or reported, even over the file it was read from.
            std::string refused;
            try { gguf::write_gguf(loaded, path); } catch (const std::runtime_error& e) { refused = e.what(); }
            require(refused == "GGUF tensor is not mapped: quant" && std::filesystem::file_size(path) == in_file,
                    "writing a model that is not mapped touched the output");
            refused.clear();
            try { gguf::warm(loaded, progress); } catch (const std::logic_error& e) { refused = e.what(); }
            require(refused == "GGUF tensor is not mapped: quant" && seen.empty(), "warming a model that is not mapped reported progress");
            gguf::map_payload(loaded);
            gguf::warm(loaded, progress);
            require(seen.size() >= 4 && seen.back() == bytes + 34, "missing intermediate/final progress");
            for (size_t i = 0; i < source.tensors.size(); ++i)
                require(std::memcmp(source.tensor_data(i), loaded.tensor_data(i), source.tensor_bytes(i)) == 0,
                        "loading changed tensor bytes");
            require(loaded.offsets[1] % alignof(float) == 0, "lost F32 alignment");
        }
        bool threw = false;
        try {
            load_file(path, [](size_t done, size_t) {
                if (done) throw std::runtime_error("consumer failure");
            });
        } catch (const std::runtime_error& e) {
            threw = std::string(e.what()) == "consumer failure";
        }
        require(threw, "consumer exception lost");
        // The writer pads the last tensor; truncate its actual payload, not just the padding.
        const auto length = std::filesystem::file_size(path);
        std::filesystem::resize_file(path, length - 33);
        seen.clear();
        threw = false;
        try { load_file(path, progress); }
        catch (const std::runtime_error&) { threw = true; }
        require(threw && seen.empty(), "truncated payload emitted progress before structural rejection");
        // Reading a file holds nothing, so its size can change before it is mapped, which refuses it before any progress; a file truncated under a live mapping is not visible.
        for (const auto size : {length - 33, length + 64}) {
            gguf::write_gguf(source, path);
            gguf::GGUFModel read = gguf::read_gguf(path);
            std::filesystem::resize_file(path, size);
            seen.clear();
            std::string error;
            try {
                gguf::map_payload(read);
                gguf::warm(read, progress);
            } catch (const std::runtime_error& e) { error = e.what(); }
            require(error == "GGUF file changed size since its header was read: " + path && seen.empty(),
                    "a file whose size changed after it was read was mapped");
        }
        gguf::write_gguf(source, path);
        std::filesystem::resize_file(path, 8);
        seen.clear();
        threw = false;
        try { load_file(path, progress); }
        catch (const std::ios_base::failure&) { threw = true; }
        require(threw && seen.empty(), "truncated header reported progress");
        {
            std::ofstream os(path, std::ios::binary | std::ios::trunc);
            const uint32_t header[] = {gguf::MAGIC, gguf::VERSION};
            const uint64_t counts[] = {2, 0};
            os.write((const char*)header, sizeof(header));
            os.write((const char*)counts, sizeof(counts));
            for (int i = 0; i < 2; ++i) {
                gguf::write_string(os, i ? "empty" : "value");
                const uint32_t nd = 1, type = gguf::GGML_TYPE_F32;
                const uint64_t ne = i ? 0 : 1, offset = i ? (uint64_t(1) << 63) : 0;
                os.write((const char*)&nd, 4);
                os.write((const char*)&ne, 8);
                os.write((const char*)&type, 4);
                os.write((const char*)&offset, 8);
            }
            gguf::pad_to(os, gguf::ALIGNMENT);
            const float value = 1;
            os.write((const char*)&value, 4);
        }
        size_t invalid_calls = 0;
        threw = false;
        try {
            load_file(path, [&](size_t, size_t) { ++invalid_calls; });
        } catch (const std::runtime_error&) { threw = true; }
        require(threw && invalid_calls == 0, "invalid trailing offset emitted progress");
        gguf::write_gguf(gguf::GGUFModel{}, path);
        size_t empty_calls = 0;
        load_file(path, [&](size_t done, size_t total) {
            require(done == 0 && total == 0, "empty payload progress");
            ++empty_calls;
        });
        require(empty_calls == 1, "empty model completion missing");
        loader_checks(path);
        std::filesystem::remove(path);
        std::cout << "load progress: payload bytes, chunks, truncation before and after reading, consumer failures and the loader pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        std::filesystem::remove(path);
        return 1;
    }
}
