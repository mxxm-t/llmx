#include <filesystem>
#include <fstream>
#include <iterator>
#include <iostream>
#include <limits>
#include "format/gguf.hpp"

using Bytes = std::vector<uint8_t>;
static size_t cases = 0;

static void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

static void integer(Bytes& bytes, uint64_t value, size_t width) {
    for (size_t i = 0; i < width; ++i) bytes.push_back(uint8_t(value >> (8 * i)));
}

static void text(Bytes& bytes, const std::string& value) {
    integer(bytes, value.size(), 8);
    bytes.insert(bytes.end(), value.begin(), value.end());
}

struct Metadata {
    std::string key;
    uint32_t type;
    Bytes value;
};

static Metadata number(const std::string& key, uint64_t value, uint32_t type, size_t width) {
    Bytes bytes;
    integer(bytes, value, width);
    return {key, type, std::move(bytes)};
}

static Metadata string_value(const std::string& key, const std::string& value) {
    Bytes bytes;
    text(bytes, value);
    return {key, 8, std::move(bytes)};
}

static Metadata array_value() {
    Bytes bytes;
    integer(bytes, 8, 4);
    integer(bytes, 2, 8);
    text(bytes, "token zero");
    text(bytes, "token one");
    return {"tokenizer.ggml.tokens", 9, std::move(bytes)};
}

static std::vector<Metadata> split(int no, int count, int tensors) {
    return {number("split.no", uint64_t(no), 2, 2),
            number("split.count", uint64_t(count), 2, 2),
            number("split.tensors.count", uint64_t(tensors), 5, 4)};
}

struct Tensor {
    std::string name;
    std::vector<uint64_t> dimensions;
    uint32_t type;
    Bytes data;
    uint64_t offset = std::numeric_limits<uint64_t>::max();
};

static Bytes encode(const std::vector<Metadata>& metadata, const std::vector<Tensor>& tensors,
                    size_t alignment = 32) {
    Bytes bytes;
    integer(bytes, 0x46554747, 4);
    integer(bytes, 3, 4);
    integer(bytes, tensors.size(), 8);
    integer(bytes, metadata.size(), 8);
    for (const auto& kv : metadata) {
        text(bytes, kv.key);
        integer(bytes, kv.type, 4);
        bytes.insert(bytes.end(), kv.value.begin(), kv.value.end());
    }
    size_t position = 0;
    for (const auto& t : tensors) {
        text(bytes, t.name);
        integer(bytes, t.dimensions.size(), 4);
        for (auto dimension : t.dimensions) integer(bytes, dimension, 8);
        integer(bytes, t.type, 4);
        integer(bytes, t.offset == std::numeric_limits<uint64_t>::max() ? position : t.offset, 8);
        position += t.data.size();
        position += (alignment - position % alignment) % alignment;
    }
    bytes.resize(bytes.size() + (alignment - bytes.size() % alignment) % alignment);
    for (const auto& t : tensors) {
        bytes.insert(bytes.end(), t.data.begin(), t.data.end());
        bytes.resize(bytes.size() + (alignment - bytes.size() % alignment) % alignment);
    }
    return bytes;
}

static void save(const std::filesystem::path& path, const Bytes& bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.exceptions(std::ios::badbit | std::ios::failbit);
    file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}

// The set read, mapped and read in as the loader does, its payload reported to `progress`.
static gguf::GGUFModel load(const std::filesystem::path& path, const format::LoadProgress& progress) {
    auto model = gguf::read_gguf(path.u8string());
    gguf::map_payload(model);
    gguf::warm(model, progress);
    return model;
}

static gguf::GGUFModel accepted(const std::filesystem::path& path, const std::vector<Tensor>& expected,
                                const std::vector<std::filesystem::path>& files = {}) {
    size_t total = 0;
    for (const auto& t : expected) total += t.data.size();
    // Reading the set maps nothing, and lays each tensor in the segment of the file given for it, where one is given: a segment holds the tensors from its first on.
    {
        const auto read = gguf::read_gguf(path.u8string());
        require(read.tensors.size() == expected.size(), "wrong aggregate tensor count");
        for (size_t i = 0; i < expected.size(); ++i) {
            require(!read.tensor_data(i), "reading the set mapped it");
            if (files.empty()) continue;
            size_t s = 0;
            while (s + 1 < read.segments.size() && read.segments[s + 1].first <= i) ++s;
            require(read.segments[s].path == files[i].u8string(), "a tensor lies in another shard's segment");
        }
    }
    std::vector<size_t> seen;
    auto model = load(path, [&](size_t done, size_t size) {
        require(size == total && done <= size, "wrong aggregate progress bounds");
        require(seen.empty() ? done == 0 : done > seen.back(), "nonmonotonic aggregate progress");
        seen.push_back(done);
    });
    require(!seen.empty() && seen.back() == total, "missing aggregate completion");
    require(model.tensors.size() == expected.size(), "wrong aggregate tensor count");
    for (size_t i = 0; i < expected.size(); ++i) {
        const auto& t = expected[i];
        require(model.tensors[i].name == t.name && model.tensors[i].ne == t.dimensions &&
                model.tensors[i].type == t.type, "tensor descriptor changed");
        require(model.tensor_bytes(i) == t.data.size(), "wrong tensor bytes");
        require(model.offsets[i] % alignof(float) == 0, "unaligned destination tensor");
        if (!t.data.empty())
            require(std::memcmp(model.tensor_data(i), t.data.data(), t.data.size()) == 0, "tensor payload changed");
    }
    ++cases;
    return model;
}

static void rejected(const std::filesystem::path& path, const std::string& diagnostic) {
    size_t progress = 0;
    bool failed = false;
    try { load(path, [&](size_t, size_t) { ++progress; }); }
    catch (const std::bad_alloc&) { throw std::runtime_error("allocation attempted before rejection"); }
    catch (const std::exception& error) { failed = std::string(error.what()).find(diagnostic) != std::string::npos; }
    require(failed, "missing expected rejection: " + diagnostic);
    require(progress == 0, "invalid shard set emitted payload progress");
    ++cases;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::filesystem::path directory(argv[1]);
    try {
        require(std::filesystem::create_directory(directory), "fixture directory must not already exist");
        const auto first = directory / "model-00001-of-00002.gguf";
        const auto second = directory / "model-00002-of-00002.gguf";
        const auto plain = directory / "single.gguf";
        const Tensor quant{"quant", {32}, 8, Bytes(34, 23)};
        const Tensor scalar{"scalar", {}, 0, {0, 0, 128, 63}};
        const Tensor empty{"empty", {0}, 0, {}};
        auto first_meta = split(0, 2, 3);
        first_meta.push_back(string_value("general.architecture", "qwen3"));
        first_meta.push_back(array_value());
        first_meta.push_back(number("general.alignment", 64, 4, 4));
        auto reset = [&] {
            save(first, encode(first_meta, {quant}, 64));
            save(second, encode(split(1, 2, 3), {scalar, empty}));
        };
        reset();
        auto model = accepted(first, {quant, scalar, empty}, {first, second, second});
        require(model.kv.size() == 3, "split bookkeeping survived assembly");
        gguf::write_gguf(model, plain.string());
        // A loaded model maps its shards, and Windows keeps a mapped file from being rewritten or removed.
        model.release_payload();
        require(accepted(plain, {quant, scalar, empty}).kv.size() == 3, "single-file roundtrip changed metadata");
        const auto read = gguf::read_gguf(first.string());
        const gguf::MetaValue* architecture = read.find("general.architecture");
        require(read.tensors.size() == 3 && architecture && architecture->s == "qwen3",
                "reading the set lost its tensors or metadata");
        ++cases;

        const auto unicode_directory = directory / std::filesystem::u8path(u8"cache-\u00e9-\u4e2d-\U0001f680");
        require(std::filesystem::create_directory(unicode_directory), "Unicode fixture directory already exists");
        const auto unicode_first = unicode_directory / "model-00001-of-00002.gguf";
        const auto unicode_second = unicode_directory / "model-00002-of-00002.gguf";
        const auto unicode_single = unicode_directory / std::filesystem::u8path(u8"model-\u4e2d.gguf");
        save(unicode_first, encode(first_meta, {quant}, 64));
        save(unicode_second, encode(split(1, 2, 3), {scalar, empty}));
        auto unicode_model = accepted(unicode_first, {quant, scalar, empty});
        gguf::write_gguf(unicode_model, unicode_single.u8string());
        accepted(unicode_single, {quant, scalar, empty});
        for (const auto& path : {unicode_first, unicode_single}) {
            const auto unicode_read = gguf::read_gguf(path.u8string());
            const gguf::MetaValue* unicode_architecture = unicode_read.find("general.architecture");
            require(unicode_read.tensors.size() == 3 && unicode_architecture && unicode_architecture->s == "qwen3",
                    "reading a Unicode path lost the model");
            ++cases;
        }
        unicode_model.release_payload();
        for (const auto& path : {unicode_first, unicode_second, unicode_single}) std::filesystem::remove(path);
        std::filesystem::remove(unicode_directory);

        auto repeated = first_meta;
        repeated[0] = number("split.no", 1, 2, 2);
        repeated.back() = number("general.alignment", 16, 4, 4);
        std::reverse(repeated.begin(), repeated.end());
        save(second, encode(repeated, {scalar, empty}, 16));
        accepted(first, {quant, scalar, empty});
        const std::vector<Metadata> typed_values = {
            number("u8", 255, 0, 1), number("i8", 255, 1, 1),
            number("u16", 65535, 2, 2), number("i16", 65535, 3, 2),
            number("u32", 0xffffffff, 4, 4), number("i32", 0xffffffff, 5, 4),
            number("f32", 0x7fc00001, 6, 4), number("bool", 1, 7, 1),
            number("u64", 0xffffffffffffffffull, 10, 8),
            number("i64", 0xffffffffffffffffull, 11, 8),
            number("f64", 0x7ff8000000000001ull, 12, 8)
        };
        auto typed_first = first_meta;
        auto typed_second = split(1, 2, 3);
        for (const auto& value : typed_values) {
            typed_first.push_back(value);
            typed_second.push_back(value);
        }
        Bytes nested;
        integer(nested, 9, 4);
        integer(nested, 1, 8);
        const auto strings = array_value();
        nested.insert(nested.end(), strings.value.begin(), strings.value.end());
        typed_first.push_back({"nested", 9, nested});
        typed_second.push_back({"nested", 9, nested});
        save(first, encode(typed_first, {quant}, 64));
        save(second, encode(typed_second, {scalar, empty}));
        accepted(first, {quant, scalar, empty});
        for (size_t i = 0; i < typed_values.size(); ++i) {
            auto changed = typed_second;
            changed[3 + i].value[0] ^= 1;
            save(second, encode(changed, {scalar, empty}));
            rejected(first, "inconsistent GGUF shard metadata");
        }
        typed_second.back().value.back() = 'x';
        save(second, encode(typed_second, {scalar, empty}));
        rejected(first, "inconsistent GGUF shard metadata");
        reset();
        save(first, encode(first_meta, {}, 64));
        save(second, encode(split(1, 2, 3), {quant, scalar, empty}));
        accepted(first, {quant, scalar, empty}, {second, second, second});
        // A zero-sized tensor that ends a shard lies at the offset the next shard starts at, and still belongs to its own shard.
        save(first, encode(first_meta, {quant, empty}, 64));
        save(second, encode(split(1, 2, 3), {scalar}));
        accepted(first, {quant, empty, scalar}, {first, first, second});
        save(first, encode(split(0, 2, 0), {}));
        save(second, encode(split(1, 2, 0), {}));
        accepted(first, {});
        save(plain, encode({}, {quant, scalar}));
        accepted(plain, {quant, scalar});
        save(plain, encode(split(0, 1, 1), {scalar}));
        require(accepted(plain, {scalar}).kv.empty(), "one-part split bookkeeping retained");

        for (int missing = 0; missing < 3; ++missing) {
            auto metadata = split(0, 2, 3);
            metadata.erase(metadata.begin() + missing);
            save(first, encode(metadata, {quant}));
            rejected(first, "invalid GGUF split metadata");
        }
        for (int field = 0; field < 3; ++field) {
            auto metadata = split(0, 2, 3);
            metadata[size_t(field)] = number(metadata[size_t(field)].key, uint64_t(field ? 3 : 0), 4, 4);
            save(first, encode(metadata, {quant}));
            rejected(first, "invalid GGUF split metadata");
        }
        for (auto values : {split(0, 0, 3), split(2, 2, 3), split(0, 2, -1)}) {
            save(first, encode(values, {quant}));
            rejected(first, "invalid GGUF split metadata");
        }
        reset();
        rejected(second, "open the first GGUF shard");
        save(plain, encode(split(0, 2, 3), {quant}));
        rejected(plain, "GGUF shard filename");
        std::filesystem::remove(second);
        rejected(first, "cannot open file");

        for (auto metadata : {split(0, 2, 3), split(1, 3, 3), split(1, 2, 4), std::vector<Metadata>{}}) {
            reset();
            save(second, encode(metadata, {scalar, empty}));
            rejected(first, "inconsistent GGUF split metadata");
        }
        reset();
        save(second, encode(split(1, 2, 3), {scalar}));
        rejected(first, "GGUF split tensor count mismatch");
        save(second, encode(split(1, 2, 3), {scalar, empty, empty}));
        rejected(first, "GGUF split tensor count mismatch");
        save(second, encode(split(1, 2, 3), {quant, empty}));
        rejected(first, "duplicate GGUF tensor");
        save(second, encode(split(1, 2, 3), {scalar, scalar}));
        rejected(first, "duplicate GGUF tensor");
        reset();
        save(first, encode(first_meta, {quant, quant}, 64));
        rejected(first, "duplicate GGUF tensor");

        for (bool in_first : {false, true}) {
            reset();
            auto metadata = in_first ? first_meta : split(1, 2, 3);
            metadata.push_back(metadata.front());
            save(in_first ? first : second, encode(metadata, in_first ? std::vector<Tensor>{quant} : std::vector<Tensor>{scalar, empty}, in_first ? 64 : 32));
            rejected(first, "duplicate GGUF metadata: split.");
            reset();
            metadata = in_first ? first_meta : split(1, 2, 3);
            metadata.push_back(string_value("general.architecture", "qwen3"));
            if (!in_first) metadata.push_back(metadata.back());
            save(in_first ? first : second, encode(metadata, in_first ? std::vector<Tensor>{quant} : std::vector<Tensor>{scalar, empty}, in_first ? 64 : 32));
            rejected(first, "duplicate GGUF metadata: general.architecture");
        }
        for (auto extra : {string_value("general.architecture", "llama"),
                           number("general.architecture", 1, 4, 4),
                           string_value("unknown.key", "later-only"),
                           string_value("tokenizer.ggml.tokens", "wrong type")}) {
            reset();
            auto metadata = split(1, 2, 3);
            metadata.push_back(extra);
            save(second, encode(metadata, {scalar, empty}));
            rejected(first, "inconsistent GGUF shard metadata");
        }
        reset();
        auto metadata = split(1, 2, 3);
        auto changed_array = array_value();
        changed_array.value.back() = 'x';
        metadata.push_back(changed_array);
        save(second, encode(metadata, {scalar, empty}));
        rejected(first, "inconsistent GGUF shard metadata");

        for (int failure = 0; failure < 5; ++failure) {
            reset();
            auto broken = scalar;
            if (failure == 0) broken.type = 999;
            if (failure == 1) broken.offset = 1;
            if (failure == 2) broken.offset = uint64_t(1) << 63;
            if (failure == 3) broken.dimensions = {std::numeric_limits<uint64_t>::max(), 2};
            if (failure == 4) broken.dimensions = {100};
            save(second, encode(split(1, 2, 3), {broken, empty}));
            rejected(first, failure == 0 ? "unsupported tensor type" : failure == 1 ? "unaligned GGUF" :
                            failure == 3 ? "overflow" : "file extent");
        }
        reset();
        const auto third_first = directory / "three-00001-of-00003.gguf";
        const auto third_last = directory / "three-00003-of-00003.gguf";
        save(third_first, encode(split(0, 3, 3), {quant}));
        save(directory / "three-00002-of-00003.gguf", encode(split(1, 3, 3), {}));
        save(third_last, encode(split(2, 3, 3), {scalar, empty}));
        accepted(third_first, {quant, scalar, empty}, {third_first, third_last, third_last});

        const size_t big_size = 8 * 1024 * 1024 + 4;
        Tensor big{"large", {big_size / 4}, 0, Bytes(big_size, 67)};
        save(second, encode(split(1, 2, 3), {big, empty}));
        accepted(first, {quant, big, empty});
        std::vector<size_t> seen;
        const auto progress = [&](size_t done, size_t total) {
            require(total == 34 + big_size, "chunk progress changed total");
            seen.push_back(done);
        };
        load(first, progress);
        require(seen == std::vector<size_t>{0, 34, 34 + 8 * 1024 * 1024, 34 + big_size}, "shard/chunk completion sequence differs");
        ++cases;
        for (bool at_start : {false, true}) {
            bool failed = false;
            try {
                load(first, [&](size_t done, size_t) {
                    if (at_start || done) throw std::runtime_error("callback failure");
                });
            } catch (const std::runtime_error& error) { failed = std::string(error.what()) == "callback failure"; }
            require(failed, "callback exception lost");
            ++cases;
        }
        // A shard truncated before it is read is refused before any progress, as a single file is.
        save(second, encode(split(1, 2, 3), {big, empty}));
        Bytes whole;
        {
            std::ifstream in(second, std::ios::binary);
            whole.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        }
        save(second, Bytes(whole.begin(), whole.end() - 64));
        seen.clear();
        bool failed = false;
        try {
            load(first, [&](size_t done, size_t) { seen.push_back(done); });
        } catch (const std::ios_base::failure&) { failed = true; }
        require(failed && seen.empty(), "a truncated shard was loaded or reported progress");
        ++cases;
        // So is a shard truncated after the set was read and before it is mapped; one truncated under a live mapping is not a case the reader can see.
        save(second, whole);
        auto read_set = gguf::read_gguf(first.string());
        save(second, Bytes(whole.begin(), whole.end() - 64));
        std::string error;
        try { gguf::map_payload(read_set); }
        catch (const std::runtime_error& e) { error = e.what(); }
        require(error == "GGUF file changed size since its header was read: " + second.string(),
                "a shard truncated after it was read was mapped");
        read_set = {};   // unmaps the first shard, mapped before the second was refused
        ++cases;
        std::filesystem::remove_all(directory);
        std::cout << "GGUF shards: " << cases << " cases pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        // Leaving the fixture behind makes every later run fail on the create_directory guard rather than on what actually broke.
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
        return 1;
    }
}
