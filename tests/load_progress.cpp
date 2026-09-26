#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include "format/gguf.hpp"
#include "inference/load.hpp"
#include "tiny_qwen.hpp"

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

// Every tensor of `m` in file order, which is what the loader reads in.
std::vector<size_t> every(const gguf::GGUFModel& m) {
    std::vector<size_t> tensors(m.tensors.size());
    std::iota(tensors.begin(), tensors.end(), size_t(0));
    return tensors;
}

// The file at `path` read, mapped and read in as the loader does, its payload reported to `progress`.
gguf::GGUFModel load_file(const std::string& path, const format::LoadProgress& progress) {
    gguf::GGUFModel m = gguf::read_gguf(path);
    gguf::map_payload(m);
    gguf::warm(m, every(m), progress);
    return m;
}

// A CPU backend playing a device: it copies what it adopts, so no weight reads the file in place.
// A streamed load gives it alloc_weight storage, poisoned here so a byte the load does not write shows, and writes into it; it keeps that storage and each write's destination and offset, and the Nth write can be made to fail.
struct CopyingBackend : backend::CpuBackend {
    int writes = 0, fail_write = 0;
    std::vector<backend::BufferPtr> weights;
    std::vector<std::pair<const backend::Buffer*, size_t>> written;
    bool reads_in_place() const override { return false; }
    backend::BufferPtr adopt(const void* src, size_t bytes) override {
        auto buffer = alloc(bytes, backend::Memory::device);
        backend::CpuBackend::write(*buffer, 0, src, bytes);
        return buffer;
    }
    backend::BufferPtr alloc_weight(size_t bytes) override {
        auto buffer = alloc(bytes, backend::Memory::device);
        const std::vector<uint8_t> poison(bytes, 0xA5);
        backend::CpuBackend::write(*buffer, 0, poison.data(), bytes);
        weights.push_back(buffer);
        return buffer;
    }
    void write(backend::Buffer& dst, size_t off, const void* src, size_t bytes) override {
        if (++writes == fail_write) throw std::runtime_error("injected write failure");
        written.push_back({&dst, off});
        backend::CpuBackend::write(dst, off, src, bytes);
    }
};

// The load modes to run on the file at `path`: direct only where its file system takes direct reads, and where it does not, the refusal a direct load gives is checked instead.
std::vector<infer::LoadMode> load_modes(const std::string& path) {
    try {
        format::FileReader probe(path, true);
    } catch (const format::DirectUnavailable&) {
        infer::PlacementRequest request;
        request.names = {"cpu"};
        std::string error;
        try { infer::load_model(path, {backend::make_cpu_backend()}, request, {}, {}, infer::LoadMode::direct); }
        catch (const std::runtime_error& e) { error = e.what(); }
        require(error == "--load-mode direct: " + path + " is on a file system that does not take direct reads", "a direct load was not refused where direct reads are not taken");
        return {infer::LoadMode::automatic, infer::LoadMode::mapped};
    }
    return {infer::LoadMode::automatic, infer::LoadMode::mapped, infer::LoadMode::direct};
}

// Whether every buffer holds the bytes of some tensor of `source` of its size, so nothing of the poison is left.
bool holds_tensors(const std::vector<backend::BufferPtr>& buffers, const gguf::GGUFModel& source) {
    for (const auto& b : buffers) {
        bool found = false;
        for (size_t i = 0; i < source.tensors.size() && !found; ++i)
            found = source.tensor_bytes(i) == b->size() && std::memcmp(b->host_ptr(), source.tensor_data(i), b->size()) == 0;
        if (!found) return false;
    }
    return !buffers.empty();
}

// Progress that starts at 0, only rises, and ends at its total.
struct Rising {
    std::vector<std::pair<size_t, size_t>> seen;
    format::LoadProgress callback() {
        return [this](size_t done, size_t total) { seen.push_back({done, total}); };
    }
    bool whole(size_t total) const {
        if (seen.empty() || seen.front().first != 0 || seen.back().first != total) return false;
        for (size_t i = 0; i < seen.size(); ++i)
            if (seen[i].second != total || (i && seen[i].first <= seen[i - 1].first)) return false;
        return true;
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
    size_t payload = 0;
    for (size_t i = 0; i < source.tensors.size(); ++i) payload += source.tensor_bytes(i);
    for (const infer::LoadMode mode : load_modes(path)) {
        for (const bool copying : {false, true}) {
            const auto copier = std::make_shared<CopyingBackend>();
            const backend::BackendPtr b = copying ? backend::BackendPtr(copier) : backend::make_cpu_backend();
            infer::PlacementRequest request;
            request.names = {"cpu"};
            Rising progress;
            const auto loaded = infer::load_model(path, {b}, request, {}, progress.callback(), mode);
            require(progress.whole(payload), "a load's progress did not rise from 0 to the payload");
            // A streamed load's copies all hold file tensors' bytes; a mapped one adopts, and allocates no weight storage.
            const bool streams = mode != infer::LoadMode::mapped;
            if (copying)
                require(streams ? copier->weights.size() == source.tensors.size() && holds_tensors(copier->weights, source) : copier->weights.empty(),
                        "a copying backend's weights do not hold the file's bytes, or a mapped load allocated weight storage");
            require(copying ? loaded->file.payload_size() == 0 && loaded->host.empty() : loaded->file.payload_size() != 0,
                    copying ? "a model no host reads kept the host copy" : "a model the CPU reads lost its payload");
            // auto streams a copying backend's weights through the cache; direct reads every file around it, into its own copy where the CPU reads in place.
            const bool direct = mode == infer::LoadMode::direct;
            require(loaded->times.mode == mode && loaded->times.files == (copying && mode == infer::LoadMode::automatic ? 1u : 0u) &&
                        loaded->times.direct_files == (direct ? 1u : 0u) && loaded->host.size() == (direct && !copying ? 1u : 0u),
                    "a load read where its mode and backends say it should not, or did not where they say it should");
            const std::vector<float> logits = loaded->model->prefill(ids);
            require(logits.size() == expected.size() &&
                        std::memcmp(logits.data(), expected.data(), logits.size() * sizeof(float)) == 0,
                    "a loaded model's logits differ from the model built in memory");
        }
    }
    // A write, or the progress, that fails part way through a streamed load stops it with its error.
    for (int failure : {1, 3}) {
        auto b = std::make_shared<CopyingBackend>();
        b->fail_write = failure;
        infer::PlacementRequest request;
        request.names = {"cpu"};
        std::string error;
        try { infer::load_model(path, {b}, request, {}, {}, infer::LoadMode::automatic); }
        catch (const std::runtime_error& e) { error = e.what(); }
        require(error == "injected write failure", "a streamed load lost a write's failure");
    }
    {
        infer::PlacementRequest request;
        request.names = {"cpu"};
        std::string error;
        try {
            infer::load_model(path, {std::make_shared<CopyingBackend>()}, request, {}, [](size_t done, size_t) {
                if (done) throw std::runtime_error("progress failure");
            }, infer::LoadMode::automatic);
        } catch (const std::runtime_error& e) { error = e.what(); }
        require(error == "progress failure", "a streamed load lost its progress callback's failure");
    }
    {
        infer::PlacementRequest request;
        request.names = {"cpu"};
        const auto loaded = infer::load_model(path, {backend::make_cpu_backend()}, request);
        require(loaded->tok && loaded->tok->encode("abcde") == ids && loaded->chat.eos == "p",
                "the loader lost the tokenizer or the chat format");
        require(loaded->plan.empty(), "one device reported a split");
    }
}

// A small qwen3moe model with tokenizer metadata: tiny_qwen with each layer routed over two experts of 12 rows, one used.
gguf::GGUFModel tiny_moe() {
    gguf::GGUFModel m = tiny_qwen(2, 2 * 128, false);
    for (auto& kv : m.kv) kv.first.replace(0, 5, "qwen3moe");
    auto meta = [&](const std::string& key, uint32_t type, uint64_t u, const std::string& s = {}) {
        gguf::MetaValue v;
        v.vtype = type;
        v.u = u;
        v.s = s;
        m.kv.push_back({key, v});
    };
    meta("general.architecture", gguf::V_STRING, 0, "qwen3moe");
    meta("qwen3moe.expert_count", gguf::V_UINT32, 2);
    meta("qwen3moe.expert_used_count", gguf::V_UINT32, 1);
    meta("qwen3moe.expert_feed_forward_length", gguf::V_UINT32, 12);
    auto add = [&](const std::string& name, std::vector<uint64_t> shape) {
        size_t count = 1;
        for (uint64_t d : shape) count *= size_t(d);
        const size_t offset = m.blob.size();
        m.blob.resize(offset + count * sizeof(float));
        for (size_t i = 0; i < count; ++i) {
            const float v = float(int((i * 13 + m.tensors.size() * 5) % 31) - 15) / 64.0f;
            std::memcpy(m.blob.data() + offset + i * sizeof(float), &v, sizeof(v));
        }
        m.tensors.push_back({name, std::move(shape), gguf::GGML_TYPE_F32, 0});
        m.offsets.push_back(offset);
    };
    for (int l = 0; l < 2; ++l) {
        const std::string pre = "blk." + std::to_string(l) + ".";
        add(pre + "ffn_gate_inp.weight", {8, 2});
        add(pre + "ffn_gate_exps.weight", {8, 12, 2});
        add(pre + "ffn_up_exps.weight", {8, 12, 2});
        add(pre + "ffn_down_exps.weight", {12, 8, 2});
    }
    gguf::MetaValue tokens;
    tokens.vtype = gguf::V_ARRAY;
    tokens.u = gguf::V_STRING;
    for (char c = 'a'; c < 'a' + 16; ++c) {
        gguf::MetaValue token;
        token.vtype = gguf::V_STRING;
        token.s = std::string(1, c);
        tokens.arr.push_back(token);
    }
    m.kv.push_back({"tokenizer.ggml.tokens", tokens});
    meta("tokenizer.ggml.eos_token_id", gguf::V_UINT32, 15);
    return m;
}

// Experts on the CPU beside a device: the placement adds a CPU backend that reads the experts in place, so each load mode maps the file for it even though the device copies, and the model gives the logits of the same model built in memory on the CPU.
void experts_checks(const std::string& path) {
    const gguf::GGUFModel source = tiny_moe();
    gguf::write_gguf(source, path);
    const std::vector<uint32_t> ids = {0, 1, 2, 3, 4};
    const std::vector<float> expected = infer::Model(source, backend::make_cpu_backend()).prefill(ids);
    for (const infer::LoadMode mode : load_modes(path)) {
        infer::PlacementRequest request;
        request.names = {"cpu"};
        request.cpu_moe = -1;
        const auto loaded = infer::load_model(path, {std::make_shared<CopyingBackend>()}, request, {}, {}, mode);
        require(loaded->file.payload_size() != 0, "experts on the CPU lost the payload they read in place");
        const std::vector<float> logits = loaded->model->prefill(ids);
        require(logits.size() == expected.size() && std::memcmp(logits.data(), expected.data(), logits.size() * sizeof(float)) == 0,
                "a model with its experts on the CPU gives other logits than the model built in memory");
    }
}

// The stream itself, in reads of one granule: some tensors cross reads and some reads hold several tensors, a tensor two backends take reaches both, and every copy holds the file's bytes.
// A read that comes up short, because the file was cut after its header was read, stops the stream with where the file ended.
void stream_checks(const std::string& path) {
    const gguf::GGUFModel source = tiny_qwen(2, 2 * 128, false);
    gguf::write_gguf(source, path);
    gguf::GGUFModel file = gguf::read_gguf(path);
    auto cpu = std::make_shared<CopyingBackend>();
    std::vector<infer::Upload> uploads;
    for (size_t i = 0; i < file.tensors.size(); ++i) {
        uploads.push_back({i, cpu.get(), cpu->alloc_weight(file.tensor_bytes(i))});
        if (i == 0) uploads.push_back({i, cpu.get(), cpu->alloc_weight(file.tensor_bytes(i))});
    }
    auto streamed = [&](const std::string& p, gguf::GGUFModel& f) {
        std::vector<std::unique_ptr<format::FileReader>> readers;
        readers.push_back(std::make_unique<format::FileReader>(p));
        std::vector<size_t> order(f.tensors.size()), file_of(f.tensors.size(), 0);
        std::vector<format::FileSpan> spans(f.tensors.size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
            spans[i] = f.span(i);
        }
        // Reads planned on a page, whatever block the file system reports, so this small fixture crosses them; buffered reads need no alignment.
        const size_t granule = core::page_size();
        const auto pieces = infer::detail::plan_pieces(order, spans, file_of, {granule}, granule);
        bool crossing = false, shared = false;
        for (const auto& piece : pieces) {
            shared = shared || piece.parts.size() > 1;
            for (const auto& part : piece.parts) crossing = crossing || part.bytes < spans[part.tensor].bytes;
            require(piece.offset % granule == 0 && piece.bytes % granule == 0 && piece.bytes <= granule, "a read is not on the granule");
        }
        require(crossing && shared, "the fixture does not make tensors cross reads and reads hold several tensors");
        std::vector<std::vector<const infer::Upload*>> destinations(f.tensors.size());
        for (const auto& u : uploads) destinations[u.tensor].push_back(&u);
        infer::LoadTimes times;
        size_t bytes = 0;
        infer::detail::stream(pieces, readers, destinations, [&](size_t n) { bytes += n; }, times, 2);
        return bytes;
    };
    size_t payload = 0;
    for (size_t i = 0; i < source.tensors.size(); ++i) payload += source.tensor_bytes(i);
    require(streamed(path, file) == payload, "the stream reported other bytes than its tensors'");
    for (const auto& u : uploads)
        require(std::memcmp(u.buffer->host_ptr(), source.tensor_data(u.tensor), source.tensor_bytes(u.tensor)) == 0,
                "a streamed copy differs from the file");
    // The writes follow the file: each lands at or after the one before, where the tensor it fills lies in the file, and the two copies of tensor 0 take each part in turn.
    uint64_t last = 0;
    for (size_t w = 0; w < cpu->written.size(); ++w) {
        const auto [dst, off] = cpu->written[w];
        size_t u = 0;
        while (uploads[u].buffer.get() != dst) ++u;
        const uint64_t at = file.span(uploads[u].tensor).offset + off;
        require(at >= last, "a streamed write went back in the file");
        last = at;
        if (u == 0) require(w + 1 < cpu->written.size() && cpu->written[w + 1] == std::make_pair((const backend::Buffer*)uploads[1].buffer.get(), off),
                            "a part of a tensor with two copies did not go to both before the next");
    }
    const auto length = std::filesystem::file_size(path);
    std::filesystem::resize_file(path, length - 4096);
    std::string error;
    try { streamed(path, file); } catch (const std::runtime_error& e) { error = e.what(); }
    require(error.rfind(path + " ended at ", 0) == 0 && error.find(" bytes, before its tensors") != std::string::npos,
            "a stream over a file cut after its header was read did not say where it ended");
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
            std::ifstream in(path, std::ios::binary);
            for (size_t i = 0; i < source.tensors.size(); ++i) {
                require(!loaded.tensor_data(i), "reading the file mapped it");
                const format::FileSpan span = loaded.span(i);
                std::vector<char> bytes_read(span.bytes);
                in.seekg(std::streamoff(span.offset));
                in.read(bytes_read.data(), std::streamsize(span.bytes));
                require(span.file == path && span.bytes == source.tensor_bytes(i) && in &&
                        std::memcmp(bytes_read.data(), source.tensor_data(i), span.bytes) == 0,
                        "a tensor's span does not hold its bytes");
            }
            require(seen.empty(), "reading the file reported progress");
            // A model whose file is not mapped is refused before anything is written or reported, even over the file it was read from.
            std::string refused;
            try { gguf::write_gguf(loaded, path); } catch (const std::runtime_error& e) { refused = e.what(); }
            require(refused == "GGUF tensor is not mapped: quant" && std::filesystem::file_size(path) == in_file,
                    "writing a model that is not mapped touched the output");
            refused.clear();
            try { gguf::warm(loaded, every(loaded), progress); } catch (const std::logic_error& e) { refused = e.what(); }
            require(refused == "GGUF tensor is not mapped: quant" && seen.empty(), "warming a model that is not mapped reported progress");
            gguf::map_payload(loaded);
            gguf::warm(loaded, every(loaded), progress);
            require(seen.size() >= 4 && seen.back() == bytes + 34, "missing intermediate/final progress");
            for (size_t i = 0; i < source.tensors.size(); ++i)
                require(std::memcmp(source.tensor_data(i), loaded.tensor_data(i), source.tensor_bytes(i)) == 0,
                        "loading changed tensor bytes");
            require(loaded.offsets[1] % alignof(float) == 0, "lost F32 alignment");
            std::vector<std::pair<size_t, size_t>> calls;
            gguf::warm(loaded, {0}, [&](size_t done, size_t total) { calls.push_back({done, total}); });
            require(calls == std::vector<std::pair<size_t, size_t>>{{0, 34}, {34, 34}}, "warming one tensor reported other bytes");
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
                gguf::warm(read, every(read), progress);
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
        stream_checks(path);
        experts_checks(path);
        std::filesystem::remove(path);
        std::cout << "load progress: payload bytes, chunks, spans, truncation before and after reading, consumer failures, the loader in each mode, experts on the CPU and the stream\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        std::filesystem::remove(path);
        return 1;
    }
}
