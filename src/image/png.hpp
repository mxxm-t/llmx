#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "image/inflate.hpp"
#include "image/raster.hpp"

// PNG decoded to 8-bit RGB as Pillow opens it and converts it to RGB: gray replicated, palettes expanded, alpha and tRNS dropped, low bit depths scaled to 0..255 as Pillow's unpackers scale them.
// Sixteen-bit samples are refused, since Pillow's conversion of them to RGB clips rather than scales; so are a palette index past the palette and every structural fault.

namespace image {

namespace png_detail {

inline uint32_t be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }

inline uint32_t crc32(const uint8_t* p, size_t n) {
    static const auto table = [] {
        std::vector<uint32_t> t(256);
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xedb88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    uint32_t c = 0xffffffffu;
    for (size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xff] ^ (c >> 8);
    return c ^ 0xffffffffu;
}

inline uint8_t paeth(int a, int b, int c) {
    const int p = a + b - c, pa = p > a ? p - a : a - p, pb = p > b ? p - b : b - p, pc = p > c ? p - c : c - p;
    return uint8_t(pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
}

// Reverses the filters of `rows` scanlines of `stride` bytes, each led by its filter byte, in place in `data`, writing the filtered-out lines to `out`.
inline void unfilter(const uint8_t* data, size_t rows, size_t stride, size_t bpp, uint8_t* out) {
    for (size_t y = 0; y < rows; ++y) {
        const uint8_t f = data[y * (stride + 1)];
        const uint8_t* in = data + y * (stride + 1) + 1;
        uint8_t* cur = out + y * stride;
        const uint8_t* up = y ? cur - stride : nullptr;
        for (size_t x = 0; x < stride; ++x) {
            const int a = x >= bpp ? cur[x - bpp] : 0, b = up ? up[x] : 0, c = (up && x >= bpp) ? up[x - bpp] : 0;
            int v = in[x];
            switch (f) {
                case 0: break;
                case 1: v += a; break;
                case 2: v += b; break;
                case 3: v += (a + b) >> 1; break;
                case 4: v += paeth(a, b, c); break;
                default: throw std::runtime_error("png: an invalid filter type");
            }
            cur[x] = uint8_t(v);
        }
    }
}

}  // namespace png_detail

// What a PNG holds apart from its pixels that the conversion reads, as Pillow's chunk readers keep it, each empty when absent:
// the EXIF of an eXIf chunk or a tEXt chunk named "exif", whichever comes last; the text of a "Raw profile type exif" chunk; and the XMP of a chunk named "XML:com.adobe.xmp".
struct PngExtras {
    std::vector<uint8_t> exif;
    std::string raw_profile_exif, xmp;
};

// A text chunk's keyword and text as Pillow reads them, kept in `extras` where the keyword is one the conversion reads.
// A compressed text that fails to inflate reads as empty, as Pillow drops it; one past Pillow's 1 MiB text limit is refused, as Pillow refuses it.
inline void text_chunk(const std::string& type, const uint8_t* data, size_t len, PngExtras& extras);

inline bool is_png(const uint8_t* p, size_t n) {
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    return n >= 8 && std::memcmp(p, sig, 8) == 0;
}

inline Raster decode_png(const uint8_t* p, size_t n, PngExtras* extras = nullptr, uint64_t max_pixels = kMaxDecodedPixels) {
    using namespace png_detail;
    if (!is_png(p, n)) throw std::runtime_error("png: not a PNG file");
    size_t at = 8;
    uint32_t w = 0, h = 0;
    int depth = 0, color = -1, interlace = 0;
    std::vector<uint8_t> idat, palette;
    bool header = false, ended = false;
    while (!ended) {
        if (at + 12 > n) throw std::runtime_error("png: the file ends inside a chunk");
        const uint32_t len = be32(p + at);
        if (len > n - at - 12) throw std::runtime_error("png: a chunk runs past the end of the file");
        const uint8_t* type = p + at + 4;
        const uint8_t* data = p + at + 8;
        if (crc32(type, size_t(len) + 4) != be32(data + len)) throw std::runtime_error("png: a chunk's CRC fails");
        const std::string t(reinterpret_cast<const char*>(type), 4);
        if (!header && t != "IHDR") throw std::runtime_error("png: the first chunk is not IHDR");
        if (t == "IHDR") {
            if (header || len != 13) throw std::runtime_error("png: a malformed IHDR");
            w = be32(data);
            h = be32(data + 4);
            depth = data[8];
            color = data[9];
            interlace = data[12];
            if (w == 0 || h == 0 || w > 0x7fffffffu || h > 0x7fffffffu) throw std::runtime_error("png: an invalid image size");
            if (data[10] != 0 || data[11] != 0 || interlace > 1) throw std::runtime_error("png: an unknown compression, filter or interlace method");
            if (depth == 16) throw std::runtime_error("png: 16-bit samples are not supported");
            const bool ok = (color == 0 && (depth == 1 || depth == 2 || depth == 4 || depth == 8)) ||
                            (color == 3 && (depth == 1 || depth == 2 || depth == 4 || depth == 8)) ||
                            ((color == 2 || color == 4 || color == 6) && depth == 8);
            if (!ok) throw std::runtime_error("png: an invalid color type and bit depth");
            if (uint64_t(w) * h > max_pixels) throw std::runtime_error("png: the image has more pixels than the limit");
            header = true;
        } else if (t == "PLTE") {
            if (len % 3 || len == 0 || len > 768) throw std::runtime_error("png: a malformed PLTE");
            palette.assign(data, data + len);
        } else if (t == "IDAT") {
            idat.insert(idat.end(), data, data + len);
        } else if (t == "IEND") {
            ended = true;
        } else if (t == "eXIf" && extras) {
            extras->exif.assign(data, data + len);
        } else if ((t == "tEXt" || t == "zTXt" || t == "iTXt") && extras) {
            text_chunk(t, data, len, *extras);
        }
        at += size_t(len) + 12;
    }
    if (color == 3 && palette.empty()) throw std::runtime_error("png: a palette image without a PLTE");
    if (idat.empty()) throw std::runtime_error("png: no image data");

    const int channels = color == 2 ? 3 : color == 4 ? 2 : color == 6 ? 4 : 1;
    const size_t bits_per_pixel = size_t(channels) * size_t(depth);
    const size_t bpp = (bits_per_pixel + 7) / 8;
    // The passes of Adam7, or the whole image as one pass.
    struct Pass { uint32_t x0, y0, dx, dy; };
    static const Pass adam7[7] = {{0, 0, 8, 8}, {4, 0, 8, 8}, {0, 4, 4, 8}, {2, 0, 4, 4}, {0, 2, 2, 4}, {1, 0, 2, 2}, {0, 1, 1, 2}};
    static const Pass whole[1] = {{0, 0, 1, 1}};
    const Pass* passes = interlace ? adam7 : whole;
    const int npass = interlace ? 7 : 1;
    size_t expected = 0;
    for (int i = 0; i < npass; ++i) {
        const Pass& ps = passes[i];
        const size_t pw = w > ps.x0 ? (size_t(w) - ps.x0 + ps.dx - 1) / ps.dx : 0, ph = h > ps.y0 ? (size_t(h) - ps.y0 + ps.dy - 1) / ps.dy : 0;
        if (pw && ph) expected += ph * ((pw * bits_per_pixel + 7) / 8 + 1);
    }
    const std::vector<uint8_t> raw = inflate_zlib(idat.data(), idat.size(), expected);
    if (raw.size() != expected) throw std::runtime_error("png: the image data is shorter than the image");

    Raster img(w, h);
    size_t off = 0;
    std::vector<uint8_t> lines;
    for (int i = 0; i < npass; ++i) {
        const Pass& ps = passes[i];
        const size_t pw = w > ps.x0 ? (size_t(w) - ps.x0 + ps.dx - 1) / ps.dx : 0, ph = h > ps.y0 ? (size_t(h) - ps.y0 + ps.dy - 1) / ps.dy : 0;
        if (!pw || !ph) continue;
        const size_t stride = (pw * bits_per_pixel + 7) / 8;
        lines.resize(ph * stride);
        unfilter(raw.data() + off, ph, stride, bpp, lines.data());
        off += ph * (stride + 1);
        for (size_t y = 0; y < ph; ++y) {
            const uint8_t* line = lines.data() + y * stride;
            for (size_t x = 0; x < pw; ++x) {
                uint8_t* px = img.at(ps.x0 + x * ps.dx, ps.y0 + y * ps.dy);
                if (depth < 8) {
                    const size_t bit = x * size_t(depth);
                    const unsigned v = (line[bit / 8] >> (8 - depth - bit % 8)) & ((1u << depth) - 1u);
                    if (color == 3) {
                        if (size_t(v) * 3 >= palette.size()) throw std::runtime_error("png: a palette index past the palette");
                        std::memcpy(px, &palette[size_t(v) * 3], 3);
                    } else {
                        const uint8_t g = uint8_t(v * (255u / ((1u << depth) - 1u)));
                        px[0] = px[1] = px[2] = g;
                    }
                    continue;
                }
                const uint8_t* s = line + x * size_t(channels);
                if (color == 3) {
                    if (size_t(s[0]) * 3 >= palette.size()) throw std::runtime_error("png: a palette index past the palette");
                    std::memcpy(px, &palette[size_t(s[0]) * 3], 3);
                } else if (channels <= 2) {
                    px[0] = px[1] = px[2] = s[0];
                } else {
                    px[0] = s[0];
                    px[1] = s[1];
                    px[2] = s[2];
                }
            }
        }
    }
    return img;
}

inline void text_chunk(const std::string& type, const uint8_t* data, size_t len, PngExtras& extras) {
    const char* k = reinterpret_cast<const char*>(data);
    const size_t klen = strnlen(k, len);
    const std::string key(k, klen);
    if (key != "exif" && key != "Raw profile type exif" && key != "XML:com.adobe.xmp") return;
    constexpr size_t kTextLimit = size_t(1) << 20;
    const auto inflated = [&](const uint8_t* z, size_t zn) -> std::string {
        std::vector<uint8_t> v;
        try {
            v = inflate_zlib(z, zn, kTextLimit + 1);
        } catch (const std::runtime_error&) {
            return {};
        }
        if (v.size() > kTextLimit) throw std::runtime_error("png: a compressed text chunk inflates past 1 MiB");
        return std::string(v.begin(), v.end());
    };
    std::string text;
    const size_t rest = klen < len ? klen + 1 : len;
    if (type == "tEXt") {
        text.assign(reinterpret_cast<const char*>(data + rest), len - rest);
        if (key == "exif") {
            extras.exif.assign(text.begin(), text.end());
            return;
        }
    } else if (type == "zTXt") {
        if (rest < len && data[rest] != 0) throw std::runtime_error("png: an unknown zTXt compression method");
        text = rest + 1 < len ? inflated(data + rest + 1, len - rest - 1) : std::string();
        if (key == "exif") return;  // Pillow keeps a zTXt named exif as text, which getexif does not read
    } else {
        if (klen >= len || len - rest < 2) return;
        const uint8_t flag = data[rest], method = data[rest + 1];
        size_t q = rest + 2;
        for (int field = 0; field < 2; ++field) {
            while (q < len && data[q]) ++q;
            if (q >= len) return;
            ++q;
        }
        if (flag) {
            if (method) return;
            text = inflated(data + q, len - q);
        } else {
            text.assign(reinterpret_cast<const char*>(data + q), len - q);
        }
        if (key == "exif") return;
    }
    if (key == "Raw profile type exif") extras.raw_profile_exif = text;
    else extras.xmp = text;
}

}  // namespace image
