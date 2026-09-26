#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

// What a model file's reader hands on whatever the container: where a tensor's bytes lie in its file, and how much of the payload has been read (docs/src/format-format.md).
namespace format {

// A tensor's bytes in a model file: `bytes` bytes from `offset` in the file at `file`, a UTF-8 path.
struct FileSpan {
    std::string file;
    uint64_t offset = 0;
    size_t bytes = 0;
};

// Completed tensor payload bytes, excluding metadata and padding; called synchronously.
using LoadProgress = std::function<void(size_t completed, size_t total)>;

} // namespace format
