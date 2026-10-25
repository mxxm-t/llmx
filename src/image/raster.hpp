#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace image {

// The most pixels a decoder accepts: Pillow's decompression-bomb warning threshold, so an image Pillow opens without complaint is never refused for its size.
inline constexpr uint64_t kMaxDecodedPixels = 89478485;

// An 8-bit RGB image, rows top to bottom, three bytes a pixel.
struct Raster {
    uint32_t width = 0, height = 0;
    std::vector<uint8_t> rgb;

    Raster() = default;
    Raster(uint32_t w, uint32_t h) : width(w), height(h), rgb(size_t(w) * h * 3) {}
    uint8_t* at(size_t x, size_t y) { return rgb.data() + (y * width + x) * 3; }
    const uint8_t* at(size_t x, size_t y) const { return rgb.data() + (y * width + x) * 3; }
};

}  // namespace image
