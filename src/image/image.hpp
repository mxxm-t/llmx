#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "image/jpeg.hpp"
#include "image/png.hpp"
#include "image/raster.hpp"

// An image file as the vision models' processor receives it, decoded, oriented and resized to the reference's bytes (docs/VISION.md, section 2).

namespace image {

namespace image_detail {

// Pillow's reading of the orientation tag in IFD0 of a TIFF-structured EXIF block: 0 when the tag is absent, else its first value, or -1 for a value no orientation has.
// As Pillow does, a directory cut short keeps the entries read before the cut, an entry of a type Pillow does not know is skipped, and an orientation of several values is its first.
inline int exif_orientation(const std::vector<uint8_t>& exif_in) {
    const uint8_t* d = exif_in.data();
    size_t n = exif_in.size();
    if (n >= 6 && std::memcmp(d, "Exif\0\0", 6) == 0) { d += 6; n -= 6; }
    if (n == 0) return 0;
    if (n < 8) throw std::runtime_error("image: malformed EXIF (no TIFF header)");
    bool le;
    if (std::memcmp(d, "II\x2a\x00", 4) == 0) le = true;
    else if (std::memcmp(d, "MM\x00\x2a", 4) == 0) le = false;
    else throw std::runtime_error("image: malformed EXIF (not a TIFF header)");
    const auto u16 = [&](size_t i) -> uint32_t { return le ? uint32_t(d[i]) | uint32_t(d[i + 1]) << 8 : uint32_t(d[i]) << 8 | d[i + 1]; };
    const auto u32 = [&](size_t i) -> uint32_t { return le ? u16(i) | u16(i + 2) << 16 : u16(i) << 16 | u16(i + 2); };
    const size_t ifd = u32(4);
    if (ifd > n || n - ifd < 2) return 0;
    const size_t count = u16(ifd);
    int orientation = 0;
    for (size_t i = 0; i < count; ++i) {
        const size_t e = ifd + 2 + 12 * i;
        if (e + 12 > n) break;
        const uint32_t type = u16(e + 2);
        const size_t unit = type == 1 || type == 6 ? 1 : type == 3 || type == 8 ? 2 : type == 4 || type == 9 ? 4 : 0;
        const bool known = (type >= 1 && type <= 13) || type == 16;
        if (!known) continue;
        const size_t cnt = u32(e + 4);
        // Values past four bytes lie at an offset, and one past the block ends the directory as Pillow's reader does.
        size_t at = e + 8;
        if (unit && cnt * unit > 4) {
            at = u32(e + 8);
            if (cnt > n / unit || at > n - cnt * unit) break;
        }
        if (u16(e) != 0x0112) continue;
        if (!unit) throw std::runtime_error("image: an EXIF orientation of an unexpected type");
        if (cnt == 0) throw std::runtime_error("image: an EXIF orientation with no value");
        int v;
        switch (type) {
            case 1: v = d[at]; break;
            case 6: v = int8_t(d[at]); break;
            case 3: v = int(u16(at)); break;
            case 8: v = int16_t(u16(at)); break;
            case 4: v = u32(at) > 8 ? 9 : int(u32(at)); break;
            default: v = int32_t(u32(at)); break;
        }
        orientation = v > 0 ? v : -1;
    }
    return orientation;
}

// Pillow's fallback when EXIF has no orientation: the first `tiff:Orientation` in the XMP followed by `="` or `>` and a digit.
inline int xmp_orientation(const std::string& xmp) {
    static const std::string key = "tiff:Orientation";
    for (size_t at = xmp.find(key); at != std::string::npos; at = xmp.find(key, at + 1)) {
        size_t q = at + key.size();
        if (q + 1 < xmp.size() && xmp[q] == '=' && xmp[q + 1] == '"') q += 2;
        else if (q < xmp.size() && xmp[q] == '>') q += 1;
        else continue;
        if (q < xmp.size() && xmp[q] >= '0' && xmp[q] <= '9') return xmp[q] - '0';
    }
    return 0;
}

// The image turned as Pillow's exif_transpose turns it for orientations 2 to 8; others leave it as it is.
inline Raster orient(Raster in, int orientation) {
    if (orientation < 2 || orientation > 8) return in;
    const size_t w = in.width, h = in.height;
    const bool swap = orientation >= 5;
    Raster out(uint32_t(swap ? h : w), uint32_t(swap ? w : h));
    for (size_t y = 0; y < out.height; ++y)
        for (size_t x = 0; x < out.width; ++x) {
            size_t sx = x, sy = y;
            switch (orientation) {
                case 2: sx = w - 1 - x; break;                   // mirror left to right
                case 3: sx = w - 1 - x; sy = h - 1 - y; break;   // rotate 180
                case 4: sy = h - 1 - y; break;                   // mirror top to bottom
                case 5: sx = y; sy = x; break;                   // transpose
                case 6: sx = y; sy = h - 1 - x; break;           // rotate 90 clockwise
                case 7: sx = w - 1 - y; sy = h - 1 - x; break;   // transverse
                case 8: sx = w - 1 - y; sy = x; break;           // rotate 90 counter-clockwise
            }
            std::memcpy(out.at(x, y), in.at(sx, sy), 3);
        }
    return out;
}

inline int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// The EXIF of a PNG text chunk "Raw profile type exif", as Pillow reads it: the hex digits after its first three lines, whitespace between byte pairs skipped.
inline std::vector<uint8_t> raw_profile_exif(const std::string& text) {
    size_t at = 0;
    for (int line = 0; line < 3; ++line) {
        at = text.find('\n', at);
        if (at == std::string::npos) return {};
        ++at;
    }
    std::vector<uint8_t> out;
    for (size_t i = at; i < text.size();) {
        const char c = text[i];
        if (c == '\n') { ++i; continue; }  // the lines are joined before the hex is read
        if (c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f') { ++i; continue; }
        const int hi = hex_digit(c), lo = i + 1 < text.size() ? hex_digit(text[i + 1]) : -1;
        if (hi < 0 || lo < 0) throw std::runtime_error("image: a malformed raw EXIF profile in a PNG text chunk");
        out.push_back(uint8_t(hi << 4 | lo));
        i += 2;
    }
    return out;
}

}  // namespace image_detail

// The image in file `p` as 8-bit RGB, turned by its EXIF orientation as transformers' load_image turns it; anything other than a PNG or JPEG is refused by name.
inline Raster decode_image(const uint8_t* p, size_t n, uint64_t max_pixels = kMaxDecodedPixels) {
    using namespace image_detail;
    std::vector<uint8_t> exif;
    std::string xmp;
    Raster img;
    if (is_png(p, n)) {
        PngExtras e;
        img = decode_png(p, n, &e, max_pixels);
        exif = e.exif;
        if (exif.empty() && !e.raw_profile_exif.empty()) exif = raw_profile_exif(e.raw_profile_exif);
        xmp = e.xmp;
    } else if (is_jpeg(p, n)) {
        JpegExtras e;
        img = decode_jpeg(p, n, &e, max_pixels);
        exif = std::move(e.exif);
        xmp = std::move(e.xmp);
    } else if (n >= 6 && (std::memcmp(p, "GIF87a", 6) == 0 || std::memcmp(p, "GIF89a", 6) == 0)) {
        throw std::runtime_error("image: GIF is not supported");
    } else if (n >= 12 && std::memcmp(p, "RIFF", 4) == 0 && std::memcmp(p + 8, "WEBP", 4) == 0) {
        throw std::runtime_error("image: WebP is not supported");
    } else {
        throw std::runtime_error("image: not a PNG or JPEG file");
    }
    int orientation = exif_orientation(exif);
    if (orientation == 0) orientation = xmp_orientation(xmp);
    return orient(std::move(img), orientation);
}

// HF's smart_resize: the height and width nearest the image's, each a multiple of `factor`, with between `min_pixels` and `max_pixels` pixels; an aspect ratio past 200 is refused.
// Its float steps are Python's, so each size is the reference's to the bit: a quotient rounded half to even, then scaled by the square root of a pixel ratio and floored or ceiled.
inline std::pair<uint32_t, uint32_t> smart_resize(uint32_t height, uint32_t width, uint32_t factor, uint64_t min_pixels, uint64_t max_pixels) {
    if (height == 0 || width == 0 || factor == 0) throw std::runtime_error("image: an empty image");
    const double hd = height, wd = width;
    if ((height > width ? hd / wd : wd / hd) > 200.0) throw std::runtime_error("image: the aspect ratio is past 200");
    const double f = factor;
    uint64_t h_bar = uint64_t(std::nearbyint(hd / f)) * factor, w_bar = uint64_t(std::nearbyint(wd / f)) * factor;
    if (h_bar * w_bar > max_pixels) {
        const double beta = std::sqrt(double(uint64_t(height) * width) / double(max_pixels));
        h_bar = std::max<uint64_t>(factor, uint64_t(std::floor(hd / beta / f)) * factor);
        w_bar = std::max<uint64_t>(factor, uint64_t(std::floor(wd / beta / f)) * factor);
    } else if (h_bar * w_bar < min_pixels) {
        const double beta = std::sqrt(double(min_pixels) / double(uint64_t(height) * width));
        h_bar = uint64_t(std::ceil(hd * beta / f)) * factor;
        w_bar = uint64_t(std::ceil(wd * beta / f)) * factor;
    }
    if (h_bar > 0xffffffffu || w_bar > 0xffffffffu) throw std::runtime_error("image: the resized image is too large");
    return {uint32_t(h_bar), uint32_t(w_bar)};
}

namespace image_detail {

// One axis of torchvision's antialiased bicubic resize of 8-bit images: each output sample a fixed-point weighted sum of the inputs under the Keys kernel (a = -0.5) widened by the downscale factor.
struct Taps {
    std::vector<size_t> first, count;
    std::vector<int16_t> weights;  // `count[i]` weights for output i, `stride` apart
    size_t stride = 0;
    int precision = 0;
};

inline double keys_cubic(double x) {
    const double a = -0.5;
    x = std::fabs(x);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}

inline Taps taps(size_t in, size_t out) {
    const double scale = double(in) / double(out);
    const double support = scale >= 1.0 ? 2.0 * scale : 2.0, inv = scale >= 1.0 ? 1.0 / scale : 1.0;
    Taps t;
    t.stride = size_t(std::ceil(support)) * 2 + 1;
    t.first.resize(out);
    t.count.resize(out);
    std::vector<double> w(out * t.stride, 0.0);
    double max_w = 0.0;
    for (size_t i = 0; i < out; ++i) {
        const double center = scale * (double(i) + 0.5);
        const int64_t lo = std::max<int64_t>(int64_t(center - support + 0.5), 0);
        const int64_t hi = std::min<int64_t>(int64_t(center + support + 0.5), int64_t(in));
        const size_t cnt = hi > lo ? size_t(hi - lo) : 0;
        if (cnt > t.stride) throw std::logic_error("image: resize taps past their stride");
        double total = 0.0;
        for (size_t j = 0; j < cnt; ++j) {
            const double v = keys_cubic((double(j) + double(lo) - center + 0.5) * inv);
            w[i * t.stride + j] = v;
            total += v;
        }
        for (size_t j = 0; j < cnt; ++j) {
            if (total != 0.0) w[i * t.stride + j] /= total;
            max_w = std::max(max_w, w[i * t.stride + j]);
        }
        t.first[i] = size_t(lo);
        t.count[i] = cnt;
    }
    // The most fraction bits that keep the largest weight inside 16 bits.
    int precision = 0;
    while (precision < 22 && int64_t(0.5 + max_w * double(int64_t(1) << (precision + 1))) < (int64_t(1) << 15)) ++precision;
    t.precision = precision;
    t.weights.resize(w.size());
    for (size_t k = 0; k < w.size(); ++k) t.weights[k] = int16_t(std::round(w[k] * double(int64_t(1) << precision)));
    return t;
}

inline Raster resize_axis(const Raster& in, size_t out_w, size_t out_h, bool horizontal) {
    const Taps t = horizontal ? taps(in.width, out_w) : taps(in.height, out_h);
    Raster out{uint32_t(out_w), uint32_t(out_h)};
    const int32_t round = t.precision ? int32_t(1) << (t.precision - 1) : 0;
    for (size_t y = 0; y < out_h; ++y)
        for (size_t x = 0; x < out_w; ++x) {
            const size_t i = horizontal ? x : y;
            const int16_t* w = t.weights.data() + i * t.stride;
            for (int c = 0; c < 3; ++c) {
                int32_t acc = round;
                for (size_t j = 0; j < t.count[i]; ++j) {
                    const uint8_t* s = horizontal ? in.at(t.first[i] + j, y) : in.at(x, t.first[i] + j);
                    acc += int32_t(s[c]) * w[j];
                }
                acc >>= t.precision;
                out.at(x, y)[c] = uint8_t(acc < 0 ? 0 : acc > 255 ? 255 : acc);
            }
        }
    return out;
}

}  // namespace image_detail

// The image resized as torchvision's v2 bicubic antialiased resize of 8-bit images: width, then height, each pass rounded to 8 bits and skipped at an unchanged size.
inline Raster resize_bicubic(const Raster& in, uint32_t out_w, uint32_t out_h) {
    if (out_w == 0 || out_h == 0) throw std::runtime_error("image: an empty resize");
    Raster cur = in;
    if (out_w != cur.width) cur = image_detail::resize_axis(cur, out_w, cur.height, true);
    if (out_h != cur.height) cur = image_detail::resize_axis(cur, cur.width, out_h, false);
    return cur;
}

// How a vision encoder wants its images, which its module hands to prepare (docs/VISION.md, section 9): its patch and merge sizes, the normalization, and smart_resize's pixel bounds.
struct Params {
    uint32_t patch = 16, merge = 2;
    float mean[3] = {0.5f, 0.5f, 0.5f}, std[3] = {0.5f, 0.5f, 0.5f};
    uint64_t min_pixels = 65536, max_pixels = 16777216;
};

// An image ready for its encoder: its patch grid and each patch's values in (channel, row, column) order, normalized, the patches in merge-block order.
struct Prepared {
    uint32_t grid_h = 0, grid_w = 0;
    std::vector<float> patches;
};

// The image sized by smart_resize, resized, normalized in F32 as the processor's fused rescale and normalize, (x - 255 mean) / (255 std), and cut into patches block by block of merge by merge.
inline Prepared prepare(const Raster& img, const Params& p) {
    if (!p.patch || !p.merge) throw std::runtime_error("image: an empty patch or merge");
    const auto [h, w] = smart_resize(img.height, img.width, p.patch * p.merge, p.min_pixels, p.max_pixels);
    const Raster r = resize_bicubic(img, w, h);
    Prepared out;
    out.grid_h = h / p.patch;
    out.grid_w = w / p.patch;
    const size_t P = p.patch, M = p.merge, row = 3 * P * P;
    out.patches.resize(size_t(out.grid_h) * out.grid_w * row);
    // The processor fuses its 1/255 rescale into the mean and std in float, so 0.5 becomes 127.5.
    float m[3], s[3];
    for (int c = 0; c < 3; ++c) {
        m[c] = p.mean[c] * float(1.0 / (1.0 / 255.0));
        s[c] = p.std[c] * float(1.0 / (1.0 / 255.0));
    }
    size_t k = 0;
    for (size_t by = 0; by < out.grid_h / M; ++by)
        for (size_t bx = 0; bx < out.grid_w / M; ++bx)
            for (size_t iy = 0; iy < M; ++iy)
                for (size_t ix = 0; ix < M; ++ix) {
                    const size_t y0 = (by * M + iy) * P, x0 = (bx * M + ix) * P;
                    float* dst = out.patches.data() + k * row;
                    for (int c = 0; c < 3; ++c)
                        for (size_t y = 0; y < P; ++y)
                            for (size_t x = 0; x < P; ++x) dst[(size_t(c) * P + y) * P + x] = (float(r.at(x0 + x, y0 + y)[c]) - m[c]) / s[c];
                    ++k;
                }
    return out;
}

}  // namespace image
