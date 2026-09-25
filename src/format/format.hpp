#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>

// What a model file's reader reports whatever the container: how much of the payload has been read (docs/src/format-format.md).
namespace format {

// Completed tensor payload bytes, excluding metadata and padding; called synchronously.
using LoadProgress = std::function<void(size_t completed, size_t total)>;

} // namespace format
