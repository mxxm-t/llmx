#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

// UTF-8 encoding, and the two ways its readers measure a character: from the lead byte alone, or as a validated sequence.

namespace utf8 {

// The UTF-8 bytes of `cp`, which the caller keeps at or below U+10FFFF.
inline std::string encode(uint32_t cp) {
    if (cp < 0x80) return std::string(1, char(cp));
    if (cp < 0x800) return { char(0xC0 | (cp >> 6)), char(0x80 | (cp & 0x3F)) };
    if (cp < 0x10000) return { char(0xE0 | (cp >> 12)), char(0x80 | ((cp >> 6) & 0x3F)), char(0x80 | (cp & 0x3F)) };
    return { char(0xF0 | (cp >> 18)), char(0x80 | ((cp >> 12) & 0x3F)), char(0x80 | ((cp >> 6) & 0x3F)), char(0x80 | (cp & 0x3F)) };
}

// The length a lead byte announces, without looking at what follows: 1 for ASCII and for a continuation byte, so a reader always moves on.
inline size_t lead_length(unsigned char c) {
    if (c >= 0xF0) return 4;
    if (c >= 0xE0) return 3;
    if (c >= 0xC0) return 2;
    return 1;
}

// The length of the valid UTF-8 sequence at `s[i]`, or 0 when it is truncated, overlong, a surrogate or above U+10FFFF.
inline size_t valid_length(const std::string& s, size_t i) {
    const auto first = static_cast<unsigned char>(s[i]);
    if (first < 0x80) return 1;
    size_t count;
    uint32_t code, minimum;
    if (first >= 0xC2 && first <= 0xDF) {
        count = 1; code = first & 0x1F; minimum = 0x80;
    } else if (first >= 0xE0 && first <= 0xEF) {
        count = 2; code = first & 0x0F; minimum = 0x800;
    } else if (first >= 0xF0 && first <= 0xF4) {
        count = 3; code = first & 0x07; minimum = 0x10000;
    } else return 0;
    if (s.size() - i - 1 < count) return 0;
    for (size_t n = 1; n <= count; ++n) {
        const auto c = static_cast<unsigned char>(s[i + n]);
        if ((c & 0xC0) != 0x80) return 0;
        code = (code << 6) | (c & 0x3F);
    }
    if (code < minimum || code > 0x10FFFF || (code >= 0xD800 && code <= 0xDFFF)) return 0;
    return count + 1;
}

} // namespace utf8
