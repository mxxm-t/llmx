#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <nmmintrin.h>

namespace core {

// CRC32C (Castagnoli) of `n` bytes continuing from `crc` (0 to start), with the SSE4.2 instruction the AVX2 baseline includes; the disk tier checks its files with it (docs/DISK-TIER.md, The entry file).
inline uint32_t crc32c(uint32_t crc, const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t c = (uint32_t)~crc;
    for (; n >= 8; n -= 8, p += 8) {
        uint64_t v;
        std::memcpy(&v, p, 8);
        c = _mm_crc32_u64(c, v);
    }
    uint32_t c32 = (uint32_t)c;
    for (; n; --n, ++p) c32 = _mm_crc32_u8(c32, *p);
    return ~c32;
}

} // namespace core
