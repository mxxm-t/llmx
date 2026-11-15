#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "format/safetensors.hpp"

namespace fs = std::filesystem;
using Bytes = std::vector<uint8_t>;
static size_t cases = 0;

static void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

// A file of `header` padded with spaces until its data starts `misalign` bytes past a multiple of 8, then `data` bytes counting up from 1.
static void write_file(const fs::path& path, std::string header, size_t data, size_t misalign = 0) {
    while ((8 + header.size()) % 8 != misalign) header += ' ';
    Bytes bytes;
    for (int i = 0; i < 8; ++i) bytes.push_back(uint8_t(uint64_t(header.size()) >> (8 * i)));
    bytes.insert(bytes.end(), header.begin(), header.end());
    for (size_t i = 0; i < data; ++i) bytes.push_back(uint8_t(i + 1));
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
}

static void write_text(const fs::path& path, const std::string& text) {
    std::ofstream(path, std::ios::binary) << text;
}

static void refused(const fs::path& dir, const std::string& expected) {
    ++cases;
    try {
        safetensors::read_directory(dir.u8string());
    } catch (const std::exception& e) {
        if (std::string(e.what()).find(expected) == std::string::npos)
            throw std::runtime_error("refused with \"" + std::string(e.what()) + "\", expected \"" + expected + "\"");
        return;
    }
    throw std::runtime_error("accepted a directory that should be refused for \"" + expected + "\"");
}

// One model.safetensors of `header` and `data` bytes, refused with `expected`.
static void refused_file(const fs::path& dir, const std::string& header, size_t data, const std::string& expected, size_t misalign = 0) {
    fs::remove_all(dir);
    fs::create_directories(dir);
    write_file(dir / "model.safetensors", header, data, misalign);
    refused(dir, expected);
}

int main() {
    const fs::path root = fs::temp_directory_path() /
        ("llmx-safetensors-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        // One file: a BF16 matrix of 2 rows of 4 and an F32 vector of 3, listed out of offset order.
        const fs::path one = root / "one";
        fs::create_directories(one);
        write_file(one / "model.safetensors",
            R"({"b":{"dtype":"F32","shape":[3],"data_offsets":[16,28]},"__metadata__":{"format":"pt"},)"
            R"("a.weight":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]}})", 28);
        {
            ++cases;
            gguf::GGUFModel m = safetensors::read_directory(one.u8string());
            require(m.tensors.size() == 2 && m.tensors[0].name == "a.weight" && m.tensors[1].name == "b", "tensors in offset order");
            require(m.tensors[0].ne == std::vector<uint64_t>{4, 2} && m.tensors[0].type == quant::GGML_TYPE_BF16, "shape reversed, BF16");
            require(m.tensors[1].ne == std::vector<uint64_t>{3} && m.tensors[1].type == quant::GGML_TYPE_F32, "F32 vector");
            require(m.kv.empty() && m.segments.size() == 1, "one segment, no metadata");
            gguf::map_payload(m);
            require(m.tensor_data(0)[0] == 1 && m.tensor_data(1)[0] == 17, "payload addressed in the file");
            require(m.span(1).offset == m.segments[0].start + 16 && m.span(1).bytes == 12, "span of the second tensor");
        }

        // Two files and an index; a file of no tensors is named by no index, and the index's own metadata is ignored.
        const fs::path two = root / "two";
        fs::create_directories(two);
        write_file(two / "model-00001-of-00002.safetensors", R"({"x":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]}})", 8);
        write_file(two / "model-00002-of-00002.safetensors", R"({"y":{"dtype":"F32","shape":[],"data_offsets":[0,4]}})", 4);
        write_text(two / "model.safetensors.index.json",
            R"({"metadata":{"total_size":12},"weight_map":{"y":"model-00002-of-00002.safetensors","x":"model-00001-of-00002.safetensors"}})");
        {
            ++cases;
            gguf::GGUFModel m = safetensors::read_directory(two.u8string());
            require(m.tensors.size() == 2 && m.tensors[0].name == "x" && m.tensors[1].name == "y", "files in name order");
            require(m.segments.size() == 2 && m.segments[1].first == 1 && m.offsets[1] == 8, "second file after the first");
            require(m.tensors[1].ne.empty(), "rank zero");
            gguf::map_payload(m);
            require(m.tensor_data(1)[0] == 1, "second file's payload");
        }

        const fs::path bad = root / "bad";
        const std::string a = R"("a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]})";
        refused_file(bad, R"({"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}})", 8, "do not span");
        refused_file(bad, "{" + a + "}", 12, "do not end where the file does");
        refused_file(bad, "{" + a + R"(,"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})", 8, "overlaps");
        refused_file(bad, "{" + a + R"(,"b":{"dtype":"F32","shape":[1],"data_offsets":[12,16]}})", 16, "leaves a gap");
        refused_file(bad, R"({"a":{"dtype":"I8","shape":[2],"data_offsets":[0,2]}})", 2, "dtype I8");
        refused_file(bad, "{" + a + "," + a + "}", 8, "repeats key a");
        refused_file(bad, R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8],"x":1}})", 8, "exactly dtype");
        refused_file(bad, R"({"a":{"dtype":"F32","shape":[2.0],"data_offsets":[0,8]}})", 8, "non-negative integer");
        refused_file(bad, R"({"a":{"dtype":"F32","shape":[-2],"data_offsets":[0,8]}})", 8, "non-negative integer");
        refused_file(bad, R"({"a":{"dtype":"F32","shape":[1,1,1,1,2],"data_offsets":[0,8]}})", 8, "rank 4");
        refused_file(bad, R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}})", 8, "end before");
        refused_file(bad, R"({"__metadata__":{"n":1},)" + a + "}", 8, "not a string");
        refused_file(bad, R"({"a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},"b":{"dtype":"F32","shape":[1],"data_offsets":[2,6]}})", 6, "float-aligned");
        refused_file(bad, "{" + a + "}", 8, "float-aligned", 2);
        refused_file(bad, "[]", 0, "not an object");
        {
            fs::remove_all(bad);
            fs::create_directories(bad);
            Bytes bytes(8, 0);
            bytes[0] = 0xff; bytes[1] = 0xff; bytes[2] = 0xff; bytes[3] = 0x7f;
            std::ofstream(bad / "model.safetensors", std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), 8);
            refused(bad, "over 100 MiB");
            bytes[2] = 0; bytes[3] = 0;
            std::ofstream(bad / "model.safetensors", std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), 8);
            refused(bad, "runs past the end");
            std::ofstream(bad / "model.safetensors", std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()), 4);
            refused(bad, "too short");
            fs::remove(bad / "model.safetensors");
            refused(bad, "cannot open");
        }

        // Index refusals: a tensor the index does not name, a name the file lacks, a path outside the directory, a repeated name.
        auto index = [&](const std::string& map) {
            fs::remove_all(bad);
            fs::create_directories(bad);
            write_file(bad / "m.safetensors", R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"y":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})", 8);
            write_text(bad / "model.safetensors.index.json", R"({"weight_map":{)" + map + "}}");
        };
        index(R"("x":"m.safetensors")");
        refused(bad, "exactly the tensors of m.safetensors");
        index(R"("x":"m.safetensors","y":"m.safetensors","z":"m.safetensors")");
        refused(bad, "exactly the tensors of m.safetensors");
        index(R"("x":"../m.safetensors","y":"m.safetensors")");
        refused(bad, "outside its directory");
        index(R"("x":"sub/m.safetensors","y":"m.safetensors")");
        refused(bad, "outside its directory");
        index(R"("x":"m.safetensors","x":"m.safetensors","y":"m.safetensors")");
        refused(bad, "repeats key x");
        index(R"("x":1)");
        refused(bad, "not a file name");
    } catch (const std::exception& e) {
        std::error_code ignored;
        fs::remove_all(root, ignored);
        std::cerr << "safetensors: FAIL after " << cases << " cases: " << e.what() << "\n";
        return 1;
    }
    std::error_code ignored;
    fs::remove_all(root, ignored);
    std::cout << "safetensors: " << cases << " cases passed\n";
    return 0;
}
