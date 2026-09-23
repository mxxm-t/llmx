#include <filesystem>
#include <iostream>
#include <limits>
#include <new>
#include "format/gguf.hpp"

using Bytes = std::vector<uint8_t>;
static size_t cases = 0;

static void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

static void integer(Bytes& out, uint64_t value, size_t size) {
    for (size_t i = 0; i < size; ++i) out.push_back(uint8_t(value >> (i * 8)));
}

static void text(Bytes& out, const std::string& value) {
    integer(out, value.size(), 8);
    out.insert(out.end(), value.begin(), value.end());
}

static Bytes header(uint64_t tensors, uint64_t metadata = 0, uint32_t version = 3) {
    Bytes out;
    integer(out, 0x46554747, 4);
    integer(out, version, 4);
    integer(out, tensors, 8);
    integer(out, metadata, 8);
    return out;
}

static void tensor(Bytes& out, const std::string& name, const std::vector<uint64_t>& dimensions,
                   uint32_t type, uint64_t offset) {
    text(out, name);
    integer(out, dimensions.size(), 4);
    for (auto dimension : dimensions) integer(out, dimension, 8);
    integer(out, type, 4);
    integer(out, offset, 8);
}

static void alignment(Bytes& out, uint32_t value, uint32_t type = 4) {
    text(out, "general.alignment");
    integer(out, type, 4);
    integer(out, value, type == 10 ? 8 : 4);
}

static size_t payload(Bytes& out, const Bytes& data, size_t align = 32) {
    out.resize(out.size() + (align - out.size() % align) % align, 0);
    const size_t start = out.size();
    out.insert(out.end(), data.begin(), data.end());
    return start;
}

static Bytes one(const std::vector<uint64_t>& dimensions, uint32_t type, uint64_t offset, size_t bytes) {
    auto out = header(1);
    tensor(out, "value", dimensions, type, offset);
    payload(out, Bytes(bytes, 23));
    return out;
}

static void save(const std::string& path, const Bytes& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    output.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}

static void rejected(const std::string& path, const std::string& name, const Bytes& bytes) {
    save(path, bytes);
    size_t callbacks = 0;
    bool threw = false;
    try { gguf::read_gguf(path, [&](size_t, size_t) { ++callbacks; }); }
    catch (const std::bad_alloc&) { throw std::runtime_error(name + ": attempted allocation instead of structural rejection"); }
    catch (const std::exception&) { threw = true; }
    require(threw, name + ": malformed file accepted");
    require(callbacks == 0, name + ": structural rejection emitted progress");
    ++cases;
}

static gguf::GGUFModel accepted(const std::string& path, const std::string& name, const Bytes& bytes,
                                const std::vector<Bytes>& expected) {
    save(path, bytes);
    size_t total = 0;
    for (const auto& data : expected) total += data.size();
    std::vector<size_t> progress;
    auto result = gguf::read_gguf(path, [&](size_t done, size_t size) {
        require(size == total && done <= size, name + ": wrong progress bounds");
        require(progress.empty() ? done == 0 : done > progress.back(), name + ": nonmonotonic progress");
        progress.push_back(done);
    });
    require(result.tensors.size() == expected.size(), name + ": wrong tensor count");
    require(!progress.empty() && progress.back() == total, name + ": completion missing");
    for (size_t i = 0; i < expected.size(); ++i) {
        require(result.tensor_bytes(i) == expected[i].size(), name + ": wrong tensor size");
        if (!expected[i].empty())
            require(std::memcmp(result.tensor_data(i), expected[i].data(), expected[i].size()) == 0, name + ": payload differs");
    }
    ++cases;
    return result;
}

static Bytes nested(size_t depth, bool empty_leaf = false) {
    auto out = header(0, 1);
    text(out, "nested");
    integer(out, 9, 4);
    for (size_t i = 0; i < depth; ++i) {
        integer(out, i + 1 == depth ? 0 : 9, 4);
        integer(out, empty_leaf && i + 1 == depth ? 0 : 1, 8);
    }
    if (!empty_leaf) integer(out, 77, 1);
    payload(out, {});
    return out;
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::string path = argv[1];
    try {
        const uint64_t maximum = std::numeric_limits<uint64_t>::max();
        for (uint32_t version : {0u, 1u, 2u, 4u, std::numeric_limits<uint32_t>::max()}) {
            auto bytes = header(0, 0, version);
            payload(bytes, {});
            rejected(path, "version " + std::to_string(version), bytes);
        }
        auto bytes = header(0);
        bytes[0] = 0;
        rejected(path, "bad magic", bytes);
        rejected(path, "short header", Bytes(20));
        rejected(path, "tensor count exceeds file", header(maximum));
        rejected(path, "metadata count exceeds file", header(0, maximum));
        bytes = header(0, 1);
        integer(bytes, maximum, 8);
        rejected(path, "metadata key length exceeds file", bytes);
        bytes = header(0, 1);
        text(bytes, "string");
        integer(bytes, 8, 4);
        integer(bytes, maximum, 8);
        rejected(path, "metadata string length exceeds file", bytes);
        bytes = header(1);
        integer(bytes, maximum, 8);
        rejected(path, "tensor name length exceeds file", bytes);
        for (uint32_t element : {0u, 8u, 9u, 10u}) {
            bytes = header(0, 1);
            text(bytes, "array");
            integer(bytes, 9, 4);
            integer(bytes, element, 4);
            integer(bytes, maximum, 8);
            rejected(path, "array count exceeds file " + std::to_string(element), bytes);
        }
        bytes = header(0, 1);
        text(bytes, "array");
        integer(bytes, 9, 4);
        integer(bytes, 99, 4);
        integer(bytes, 0, 8);
        payload(bytes, {});
        rejected(path, "unknown empty array element", bytes);
        for (uint32_t type = 0; type <= 12; ++type) {
            bytes = header(0, 1);
            text(bytes, "array");
            integer(bytes, 9, 4);
            integer(bytes, type, 4);
            integer(bytes, 0, 8);
            payload(bytes, {});
            auto model = accepted(path, "valid empty array " + std::to_string(type), bytes, {});
            require(model.kv[0].second.arr.empty(), "empty array changed");
        }
        auto model = accepted(path, "nested array depth256", nested(256), {});
        const auto* value = &model.kv[0].second;
        for (size_t i = 0; i < 256; ++i) {
            require(value->vtype == 9 && value->arr.size() == 1, "nested array structure changed");
            value = &value->arr[0];
        }
        require(value->vtype == 0 && value->u == 77, "nested array leaf changed");
        rejected(path, "nested array depth257", nested(257));
        accepted(path, "nested empty array depth256", nested(256, true), {});
        rejected(path, "nested empty array depth257", nested(257, true));
        for (uint32_t rank : {4u, 5u, std::numeric_limits<uint32_t>::max()}) {
            bytes = header(1);
            text(bytes, "rank");
            integer(bytes, rank, 4);
            integer(bytes, 1, 8);
            rejected(path, "rank exceeds available dimensions " + std::to_string(rank), bytes);
        }
        rejected(path, "rank5 complete header", one({1, 1, 1, 1, 1}, 0, 0, 4));
        accepted(path, "rank4", one({1, 1, 1, 1}, 0, 0, 4), {Bytes(4, 23)});
        accepted(path, "scalarF32", one({}, 0, 0, 4), {Bytes(4, 23)});
        accepted(path, "zero first extent", one({0, 4}, 0, 0, 0), {Bytes{}});
        accepted(path, "zero later extent", one({4, 0}, 0, 0, 0), {Bytes{}});
        accepted(path, "zero F32 product after huge dimensions", one({maximum, 2, 0}, 0, 0, 0), {Bytes{}});
        accepted(path, "zero Q8 product after huge dimensions", one({maximum - 31, 2, 0}, 8, 0, 0), {Bytes{}});
        rejected(path, "zero Q8 product still requires complete rows", one({maximum, 2, 0}, 8, 0, 0));
        rejected(path, "zero extent outside offset", one({0}, 0, 32, 0));
        rejected(path, "zero extent wrapping offset", one({0}, 0, maximum - 31, 0));
        rejected(path, "dimension product overflow", one({maximum, 2}, 0, 0, 0));
        rejected(path, "F32 byte count overflow", one({uint64_t(1) << 62}, 0, 0, 0));
        rejected(path, "Q8 byte count overflow", one({maximum - 31}, 8, 0, 0));
        rejected(path, "large valid size absent payload", one({uint64_t(1) << 40}, 0, 0, 0));
        rejected(path, "unknown tensor type", one({32}, 99, 0, 128));
        struct Type { uint32_t id; uint64_t block; size_t bytes; };
        const Type types[] = {{0, 1, 4}, {2, 32, 18}, {3, 32, 20}, {8, 32, 34},
                              {12, 256, 144}, {13, 256, 176}, {14, 256, 210}};
        for (const auto& type : types) {
            const auto name = "type " + std::to_string(type.id);
            accepted(path, name, one({type.block, 2}, type.id, 0, 2 * type.bytes), {Bytes(2 * type.bytes, 23)});
            rejected(path, name + " truncated payload", one({type.block, 2}, type.id, 0, 2 * type.bytes - 1));
            rejected(path, name + " outside payload", one({type.block}, type.id, 32, type.bytes));
            rejected(path, name + " offset wraps", one({type.block}, type.id, maximum - 31, type.bytes));
            if (type.block > 1) {
                rejected(path, name + " total divisible but row is not", one({type.block / 2, 2}, type.id, 0, type.bytes));
                rejected(path, name + " partial block", one({type.block + 1}, type.id, 0, 2 * type.bytes));
                accepted(path, name + " zero rows", one({type.block, 0}, type.id, 0, 0), {Bytes{}});
                accepted(path, name + " zero columns", one({0, 2}, type.id, 0, 0), {Bytes{}});
            }
        }
        bytes = header(2);
        tensor(bytes, "last-first", {1}, 0, 64);
        tensor(bytes, "first-last", {1}, 0, 0);
        Bytes data(68, 0);
        std::fill(data.begin(), data.begin() + 4, uint8_t(11));
        std::fill(data.begin() + 64, data.end(), uint8_t(91));
        payload(bytes, data);
        accepted(path, "unordered offsets", bytes, {Bytes(4, 91), Bytes(4, 11)});
        bytes = header(2);
        tensor(bytes, "overlap-a", {16}, 0, 0);
        tensor(bytes, "overlap-b", {16}, 0, 32);
        payload(bytes, Bytes(96, 37));
        accepted(path, "valid overlapping ranges", bytes, {Bytes(64, 37), Bytes(64, 37)});
        bytes = header(2);
        tensor(bytes, "alias-a", {1}, 0, 0);
        tensor(bytes, "alias-b", {1}, 0, 0);
        payload(bytes, Bytes(4, 42));
        accepted(path, "valid identical ranges", bytes, {Bytes(4, 42), Bytes(4, 42)});
        bytes = header(2);
        tensor(bytes, "value", {1}, 0, 0);
        tensor(bytes, "empty-at-eof", {0}, 0, 32);
        payload(bytes, Bytes(32, 19));
        accepted(path, "zero extent at payload EOF", bytes, {Bytes(4, 19), Bytes{}});
        for (uint32_t align : {8u, 16u, 24u, 32u, 64u}) {
            bytes = header(2, 1);
            alignment(bytes, align);
            tensor(bytes, "a", {1}, 0, 0);
            tensor(bytes, "b", {1}, 0, align);
            data.assign(align + 4, 9);
            const size_t start = payload(bytes, data, align);
            require(start % align == 0, "test fixture alignment bug");
            accepted(path, "alignment " + std::to_string(align), bytes, {Bytes(4, 9), Bytes(4, 9)});
        }
        for (uint32_t align : {0u, 1u, 7u, 12u, 25u}) {
            bytes = header(0, 1);
            alignment(bytes, align);
            payload(bytes, {});
            rejected(path, "invalid alignment " + std::to_string(align), bytes);
        }
        bytes = header(0, 1);
        alignment(bytes, 32, 10);
        payload(bytes, {});
        rejected(path, "wrong alignment metadata type", bytes);
        bytes = header(0, 2);
        alignment(bytes, 32);
        alignment(bytes, 32);
        payload(bytes, {});
        rejected(path, "duplicate alignment metadata", bytes);
        bytes = header(1);
        tensor(bytes, "unaligned", {1}, 0, 1);
        payload(bytes, Bytes(5));
        rejected(path, "unaligned tensor offset", bytes);
        gguf::GGUFModel written;
        gguf::MetaValue align;
        align.vtype = 4;
        align.u = 24;
        written.kv.emplace_back("general.alignment", align);
        written.tensors.push_back({"a", {1}, 0, 0});
        written.add_tensor_data(Bytes{0, 0, 128, 63});
        written.tensors.push_back({"b", {32}, 8, 0});
        written.add_tensor_data(Bytes(34, 5));
        gguf::write_gguf(written, path);
        model = gguf::read_gguf(path);
        require(model.tensors[0].offset == 0 && model.tensors[1].offset == 24,
                "writer ignored custom alignment");
        require(model.tensor_bytes(0) == 4 && model.tensor_bytes(1) == 34 &&
                std::memcmp(model.tensor_data(0), written.tensor_data(0), 4) == 0 &&
                std::memcmp(model.tensor_data(1), written.tensor_data(1), 34) == 0, "custom alignment writer roundtrip changed bytes");
        ++cases;
        model = {};   // unmaps the file before it is removed
        std::filesystem::remove(path);
        std::cout << "GGUF validation: " << cases << " independent file cases pass\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        std::filesystem::remove(path);
        return 1;
    }
}
