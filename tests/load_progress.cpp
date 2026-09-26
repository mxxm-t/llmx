#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <thread>
#include "format/gguf.hpp"
#include "inference/load.hpp"
#include "loading_backend.hpp"
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
    int writes = 0, fail_write = 0, foreign = 0;
    int slow = 0;   // writes still to take two milliseconds each, so the stream's readers run ahead and fill the ring
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
        if (slow > 0 && slow--) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        if (!weights.empty() && std::none_of(weights.begin(), weights.end(), [&](const backend::BufferPtr& w) { return w.get() == &dst; })) ++foreign;
        written.push_back({&dst, off});
        backend::CpuBackend::write(dst, off, src, bytes);
    }
};

// The load modes to run on the model at `path`: direct only where its file system takes direct reads, and where it does not, the refusal a direct load gives is checked instead, naming the first file with tensors, `named` (the path itself when empty).
std::vector<infer::LoadMode> load_modes(const std::string& path, const std::string& named = {}) {
    try {
        format::FileReader probe(path, true);
    } catch (const format::DirectUnavailable&) {
        infer::PlacementRequest request;
        request.names = {"cpu"};
        std::string error;
        try { infer::load_model(path, {backend::make_cpu_backend()}, request, {}, {}, infer::LoadMode::direct); }
        catch (const std::runtime_error& e) { error = e.what(); }
        require(error.rfind("--load-mode direct: ", 0) == 0 && error.find(named.empty() ? path : named) != std::string::npos,
                "a direct load was not refused, naming its file, where direct reads are not taken");
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

// `m` with the tokenizer metadata a load reads: sixteen one-letter tokens, the last the end of text.
gguf::GGUFModel with_tokens(gguf::GGUFModel m) {
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
    m.kv.push_back({"tokenizer.ggml.tokens", tokens});
    m.kv.push_back({"tokenizer.ggml.eos_token_id", eos});
    return m;
}

// A model file's weights have no bytes until its payload is mapped.
// Loaded through infer::load_model it keeps its payload on the CPU and releases it on a backend that copies every weight, and both give the logits of the same model built in memory.
void loader_checks(const std::string& path) {
    const gguf::GGUFModel source = with_tokens(tiny_qwen(2, 2 * 128, false));
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
            // direct maps nothing, and the CPU's copy holds each weight where the file does.
            if (direct && !copying)
                for (size_t i = 0; i < source.tensors.size(); ++i)
                    require(!loaded->file.tensor_data(i) &&
                                std::memcmp(loaded->host[0].data() + loaded->file.span(i).offset, source.tensor_data(i), source.tensor_bytes(i)) == 0,
                            "a direct load mapped the file, or its copy does not hold a weight where the file does");
            const std::vector<float> logits = loaded->model->prefill(ids);
            require(logits.size() == expected.size() &&
                        std::memcmp(logits.data(), expected.data(), logits.size() * sizeof(float)) == 0,
                    "a loaded model's logits differ from the model built in memory");
        }
    }
    // A write, or the progress, that fails part way through a streamed load stops it with its error, on a backend whose uploads stay outstanding until it drains, and no storage goes before the drain.
    for (const int failure : {1, 3, 0}) {
        auto b = std::make_shared<LoadingBackend>();
        b->fail_write = failure;
        infer::PlacementRequest request;
        request.names = {"cpu"};
        std::string error;
        const format::LoadProgress progress = [&](size_t done, size_t) {
            if (!failure && done) throw std::runtime_error("progress failure");
        };
        try { infer::load_model(path, {b}, request, {}, progress, infer::LoadMode::automatic); }
        catch (const std::runtime_error& e) { error = e.what(); }
        require(error == (failure ? "injected write failure" : "progress failure"), "a streamed load lost a write's or the progress's failure");
        require(b->state->writes > 0 && b->state->premature == 0 && b->state->drains > 0 && b->state->releases > 0,
                "a failed streamed load freed storage before its backend drained");
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
    return with_tokens(std::move(m));
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

// A split over two copying devices with a tied head: the embedding reaches the first device and, as the head, the second, each write goes to storage its own device gave, and the logits are those of the model built in memory.
// Then the CPU in place of the first device, so one weight is both read in place and copied.
void split_checks(const std::string& path) {
    const gguf::GGUFModel source = with_tokens(tiny_qwen(2, 2 * 128, true));
    gguf::write_gguf(source, path);
    const std::vector<uint32_t> ids = {0, 1, 2, 3, 4};
    const std::vector<float> expected = infer::Model(source, backend::make_cpu_backend()).prefill(ids);
    size_t embedding = 0;
    while (source.tensors[embedding].name != "token_embd.weight") ++embedding;
    for (const infer::LoadMode mode : load_modes(path)) {
        auto a = std::make_shared<CopyingBackend>(), b = std::make_shared<CopyingBackend>();
        infer::PlacementRequest request;
        request.names = {"cpu", "cpu"};
        request.shares = {1, 1};
        const auto loaded = infer::load_model(path, {a, b}, request, {}, {}, mode);
        if (mode != infer::LoadMode::mapped) {
            auto takes = [&](const CopyingBackend& d) {
                return std::any_of(d.weights.begin(), d.weights.end(), [&](const backend::BufferPtr& w) {
                    return w->size() == source.tensor_bytes(embedding) && std::memcmp(w->host_ptr(), source.tensor_data(embedding), w->size()) == 0;
                });
            };
            require(holds_tensors(a->weights, source) && holds_tensors(b->weights, source) && takes(*a) && takes(*b),
                    "a split's devices do not both hold the tied embedding, or hold other bytes than the file's");
            require(a->foreign == 0 && b->foreign == 0, "a stream wrote into storage another device gave");
        }
        const std::vector<float> logits = loaded->model->prefill(ids);
        require(logits.size() == expected.size() && std::memcmp(logits.data(), expected.data(), logits.size() * sizeof(float)) == 0,
                "a split load gives other logits than the model built in memory");
    }
    // The CPU reading its half in place beside a copying device: the tied embedding the CPU reads and the device's head copies is read and reported once, in direct from the CPU's copy, and reaches the device whole.
    size_t payload = 0;
    for (size_t i = 0; i < source.tensors.size(); ++i) payload += source.tensor_bytes(i);
    for (const infer::LoadMode mode : load_modes(path)) {
        auto device = std::make_shared<CopyingBackend>();
        infer::PlacementRequest request;
        request.names = {"cpu", "cpu"};
        request.shares = {1, 1};
        Rising progress;
        const auto loaded = infer::load_model(path, {backend::make_cpu_backend(), device}, request, {}, progress.callback(), mode);
        require(progress.whole(payload), "a split with the CPU reading in place reported a weight twice");
        if (mode != infer::LoadMode::mapped)
            require(holds_tensors(device->weights, source) &&
                        std::any_of(device->weights.begin(), device->weights.end(), [&](const backend::BufferPtr& w) {
                            return w->size() == source.tensor_bytes(embedding) && std::memcmp(w->host_ptr(), source.tensor_data(embedding), w->size()) == 0;
                        }),
                    "a device beside the CPU does not hold the tied embedding, or holds other bytes than the file's");
        const std::vector<float> logits = loaded->model->prefill(ids);
        require(logits.size() == expected.size() && std::memcmp(logits.data(), expected.data(), logits.size() * sizeof(float)) == 0,
                "a split with the CPU reading in place gives other logits than the model built in memory");
    }
}

// A set of three shards, the first holding metadata alone: each load mode reads the two with tensors, and the model gives the logits of the one built in memory.
void shard_checks(const std::filesystem::path& dir) {
    const gguf::GGUFModel source = with_tokens(tiny_qwen(2, 2 * 128, false));
    const size_t half = source.tensors.size() / 2;
    auto split_keys = [](gguf::GGUFModel& m, uint64_t no, uint64_t tensors) {
        auto key = [&](const std::string& name, uint32_t type, uint64_t value) {
            gguf::MetaValue v;
            v.vtype = type;
            if (type == gguf::V_INT32) v.i = int64_t(value);   // signed values are kept in `i`
            else v.u = value;
            m.kv.push_back({name, v});
        };
        key("split.no", gguf::V_UINT16, no);
        key("split.count", gguf::V_UINT16, 3);
        key("split.tensors.count", gguf::V_INT32, tensors);
    };
    std::vector<std::string> paths;
    for (uint64_t no = 0; no < 3; ++no) {
        gguf::GGUFModel shard;
        if (no == 0) shard.kv = source.kv;
        split_keys(shard, no, source.tensors.size());
        const size_t from = no == 0 ? 0 : no == 1 ? 0 : half, to = no == 0 ? 0 : no == 1 ? half : source.tensors.size();
        for (size_t i = from; i < to; ++i) {
            shard.tensors.push_back(source.tensors[i]);
            const uint8_t* data = source.tensor_data(i);
            shard.add_tensor_data(std::vector<uint8_t>(data, data + source.tensor_bytes(i)));
        }
        const std::string name = "load-shard-0000" + std::to_string(no + 1) + "-of-00003.gguf";
        paths.push_back((dir / name).u8string());
        gguf::write_gguf(shard, paths.back());
    }
    const std::vector<uint32_t> ids = {0, 1, 2, 3, 4};
    const std::vector<float> expected = infer::Model(source, backend::make_cpu_backend()).prefill(ids);
    for (const infer::LoadMode mode : load_modes(paths[0], paths[1])) {
        for (const bool copying : {false, true}) {
            const auto copier = std::make_shared<CopyingBackend>();
            const backend::BackendPtr b = copying ? backend::BackendPtr(copier) : backend::make_cpu_backend();
            infer::PlacementRequest request;
            request.names = {"cpu"};
            const auto loaded = infer::load_model(paths[0], {b}, request, {}, {}, mode);
            const size_t files = loaded->times.files + loaded->times.direct_files;
            require(mode == infer::LoadMode::mapped || (!copying && mode == infer::LoadMode::automatic) ? files == 0 : files == 2,
                    "a sharded load streamed from other than its two shards with tensors");
            if (copying && mode != infer::LoadMode::mapped) require(holds_tensors(copier->weights, source), "a sharded load's copies differ from the file's");
            const std::vector<float> logits = loaded->model->prefill(ids);
            require(logits.size() == expected.size() && std::memcmp(logits.data(), expected.data(), logits.size() * sizeof(float)) == 0,
                    "a sharded load gives other logits than the model built in memory");
        }
    }
    for (const auto& p : paths) std::filesystem::remove(std::filesystem::u8path(p));
}

// The reads the stream plans, on spans that need no file: pieces start and end on the granule and hold at most the limit rounded down to it, a gap of one granule is read through and a longer one starts a new piece, a tensor longer than a piece crosses several, a new file starts a new piece, and every tensor's bytes lie in exactly one part each.
void plan_checks() {
    const size_t g = 7168, limit = size_t(16) << 20, cap = limit / g * g;
    const std::vector<format::FileSpan> spans = {
        {"a", 0, 1000},                  // 0
        {"a", 1000 + 7168, 500},         // 1: a gap of exactly one granule, read through
        {"a", 1000 + 7168 + 500 + 7169, 300},   // 2: one granule and a byte, a new piece
        {"a", 40000, 3 * cap + 5},       // 3: crosses four pieces
        {"b", 0, 0},                     // 4: nothing to read, at the next file's start
        {"b", 0, 2000},                  // 5: a new file
    };
    const std::vector<size_t> tensors = {0, 1, 2, 3, 4, 5}, file_of = {0, 0, 0, 0, 1, 1};
    const auto pieces = infer::detail::plan_pieces(tensors, spans, file_of, {g, g}, limit);
    std::vector<size_t> covered(spans.size(), 0);
    for (size_t k = 0; k < pieces.size(); ++k) {
        const auto& p = pieces[k];
        require(p.offset % g == 0 && p.bytes % g == 0 && p.bytes <= cap && !p.parts.empty(), "a planned read is not whole granules within the limit");
        if (k && pieces[k - 1].file == p.file) require(pieces[k - 1].offset + pieces[k - 1].bytes <= p.offset, "two planned reads overlap");
        for (const auto& part : p.parts) {
            require(part.tensor_offset == covered[part.tensor] && p.offset + part.piece_offset == spans[part.tensor].offset + part.tensor_offset &&
                        part.piece_offset + part.bytes <= p.bytes && file_of[part.tensor] == p.file,
                    "a planned part is not where its tensor's bytes are");
            covered[part.tensor] += part.bytes;
        }
    }
    for (size_t t = 0; t < spans.size(); ++t) require(covered[t] == spans[t].bytes, "a planned read left out or repeated a tensor's bytes");
    require(pieces.size() == 7 && pieces[0].parts.size() == 2 && pieces[1].parts.size() == 1 && pieces[6].file == 1,
            "the reads are not the ones the gaps, the limit and the files call for");
}

// auto reads around the file cache only when the bytes it streams are more than the host has, and a file system that refuses direct reads gives a reader through the cache instead.
void choice_checks(const std::string& path) {
    require(!infer::detail::reads_around(5, std::nullopt) && !infer::detail::reads_around(5, 5) && infer::detail::reads_around(6, 5) &&
                !infer::detail::reads_around(0, 0),
            "auto reads around the cache at other bytes than more than the host has");
    gguf::write_gguf(with_tokens(tiny_qwen(1, 64, false)), path);
    bool direct = true;
    try { format::FileReader probe(path, true); } catch (const format::DirectUnavailable&) { direct = false; }
    require(infer::detail::open_reader(path, true)->direct() == direct && !infer::detail::open_reader(path, false)->direct(),
            "a reader asked to read around the cache did not, where the file system takes it, or did where it does not");
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
        infer::detail::stream(pieces, readers, destinations, [&](size_t n) { bytes += n; }, times);
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
    // Dozens of 64-byte reads through the four slots, the writes slow enough that both readers fill the ring and wait: every copy still holds the file's bytes, and a write failing late stops the stream while the readers wait on the ring.
    {
        std::vector<std::unique_ptr<format::FileReader>> readers;
        readers.push_back(std::make_unique<format::FileReader>(path));
        std::vector<size_t> order(file.tensors.size()), file_of(file.tensors.size(), 0);
        std::vector<format::FileSpan> spans(file.tensors.size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
            spans[i] = file.span(i);
        }
        const auto pieces = infer::detail::plan_pieces(order, spans, file_of, {64}, 64);
        require(pieces.size() > 40, "the fixture does not make dozens of 64-byte reads");
        for (const int failure : {0, 30}) {
            auto late = std::make_shared<CopyingBackend>();
            late->slow = 40;
            late->fail_write = failure;
            std::vector<infer::Upload> copies;
            for (size_t i = 0; i < file.tensors.size(); ++i) copies.push_back({i, late.get(), late->alloc_weight(file.tensor_bytes(i))});
            std::vector<std::vector<const infer::Upload*>> destinations(file.tensors.size());
            for (const auto& u : copies) destinations[u.tensor].push_back(&u);
            infer::LoadTimes times;
            std::string error;
            try { infer::detail::stream(pieces, readers, destinations, {}, times); } catch (const std::runtime_error& e) { error = e.what(); }
            require(failure ? error == "injected write failure" : error.empty() && holds_tensors(late->weights, source),
                    "a stream of many reads through the ring lost bytes, or a late write's failure");
        }
        // A read that fails on a reader thread, a direct read off its granule where the file system takes direct reads, comes out of the stream with the readers joined.
        try {
            std::vector<std::unique_ptr<format::FileReader>> direct;
            direct.push_back(std::make_unique<format::FileReader>(path, true));
            auto skewed = infer::detail::plan_pieces(order, spans, file_of, {direct[0]->granule()}, direct[0]->granule());
            skewed.back().bytes += 1;
            std::vector<std::vector<const infer::Upload*>> none(file.tensors.size());
            infer::LoadTimes times;
            bool refused = false;
            try { infer::detail::stream(skewed, direct, none, {}, times); } catch (const std::logic_error&) { refused = true; }
            require(refused, "a reader thread's failing read did not come out of the stream");
        } catch (const format::DirectUnavailable&) {
        }
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
        // A streamed load's readers make the same check on the size they open, which is the one check.
        for (const auto size : {length - 33, length + 64}) {
            gguf::write_gguf(source, path);
            gguf::GGUFModel read = gguf::read_gguf(path);
            gguf::check_size(read, path, length);
            std::filesystem::resize_file(path, size);
            seen.clear();
            std::string error, checked;
            try {
                gguf::map_payload(read);
                gguf::warm(read, every(read), progress);
            } catch (const std::runtime_error& e) { error = e.what(); }
            require(error == "GGUF file changed size since its header was read: " + path && seen.empty(),
                    "a file whose size changed after it was read was mapped");
            try { gguf::check_size(read, path, format::FileReader(path).size()); } catch (const std::runtime_error& e) { checked = e.what(); }
            require(checked == error, "a reader's size check differs from the mapping's");
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
        split_checks(path);
        shard_checks(std::filesystem::u8path(path).parent_path());
        plan_checks();
        choice_checks(path);
        std::filesystem::remove(path);
        std::cout << "load progress: payload bytes, chunks, spans, truncation before and after reading, consumer failures, the loader in each mode, experts on the CPU, splits, shards, the planned reads, the read choice and the stream\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        std::filesystem::remove(path);
        return 1;
    }
}
