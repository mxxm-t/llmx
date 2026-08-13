#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

// Pluggable model-file format abstraction. A format knows how to open a file
// and enumerate its tensors plus metadata. GGUF (format/gguf.hpp) is the first
// implementation; future formats (safetensors, raw) implement the same
// interface so the rest of the stack never cares which container was used.
namespace format {

// A single tensor as described by the file's metadata (not yet loaded).
struct Tensor {
    std::string name;
    std::vector<uint64_t> shape; // shape[0] is the fastest dim (ne[0])
    uint32_t type = 0;           // GGML type id (see format/gguf.hpp)
};

class ModelFormat {
public:
    virtual ~ModelFormat() = default;

    // List tensors in the order their data appears in the file.
    virtual std::vector<Tensor> tensors() const = 0;

    // Read a string metadata key. Returns empty if absent.
    virtual std::string metadata_string(const std::string& key) const = 0;

    // Read an integer metadata key. Returns 0 if absent.
    virtual uint64_t metadata_u64(const std::string& key) const = 0;
};

using ModelFormatPtr = std::shared_ptr<ModelFormat>;

// Open a model file, auto-detecting the format from its header. Returns nullptr
// if the format is not recognized.
ModelFormatPtr open(const std::string& path);

} // namespace format
