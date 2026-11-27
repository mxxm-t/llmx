// image::decode_image, smart_resize, resize_bicubic and the padded and pad_end fits held to the bytes of the vision models' processors (tests/data/image_decode.json, deepseek41_image_fit.json and inkling2_image_fit.json).
// Unread formats are refused with their texts, and prefixes and flipped bits of every image are refused or decoded, never read past their end.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/json.hpp"
#include "core/sha.hpp"
#include "image/image.hpp"

namespace {

int checks = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

std::vector<uint8_t> load(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path.string());
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

std::string sha256(const uint8_t* p, size_t n) {
    core::Sha sha(true);
    sha.update(p, n);
    return sha.hex();
}

size_t number(const jmini::Value& v, const char* key) {
    const jmini::Value* x = v.get(key);
    if (!x || !x->isNumber()) throw std::runtime_error(std::string("fixture: missing ") + key);
    return size_t(x->asNumber());
}

const std::string& text(const jmini::Value& v, const char* key) {
    const jmini::Value* x = v.get(key);
    if (!x || !x->isString()) throw std::runtime_error(std::string("fixture: missing ") + key);
    return x->asString();
}

// Where two decodes first part, for a failure's message.
std::string first_difference(const image::Raster& got, const std::vector<uint8_t>& want) {
    for (size_t i = 0; i < std::min(got.rgb.size(), want.size()); ++i)
        if (got.rgb[i] != want[i]) {
            const size_t px = i / 3;
            return " (first at x " + std::to_string(px % got.width) + ", y " + std::to_string(px / got.width) + ", channel " + std::to_string(i % 3) +
                   ": " + std::to_string(got.rgb[i]) + " against " + std::to_string(want[i]) + ")";
        }
    return got.rgb.size() == want.size() ? "" : " (sizes differ)";
}

void decodes(const std::filesystem::path& dir, const jmini::Value& doc) {
    for (const auto& c : doc.get("decode")->asArray()) {
        const std::string& name = text(c, "file");
        const std::vector<uint8_t> bytes = load(dir / name);
        require(sha256(bytes.data(), bytes.size()) == text(c, "sha256"), name + ": the file is not the fixture's");
        const image::Raster img = image::decode_image(bytes.data(), bytes.size());
        require(img.width == number(c, "width") && img.height == number(c, "height"),
                name + ": decoded as " + std::to_string(img.width) + "x" + std::to_string(img.height) + ", the reference " +
                    std::to_string(number(c, "width")) + "x" + std::to_string(number(c, "height")));
        if (sha256(img.rgb.data(), img.rgb.size()) == text(c, "rgb_sha256")) {
            ++checks;
            continue;
        }
        std::string where;
        if (const jmini::Value* rgb = c.get("rgb")) where = first_difference(img, load(dir / rgb->asString()));
        throw std::runtime_error(name + ": the decoded bytes are not the reference's" + where);
    }
}

void resizes(const std::filesystem::path& dir, const jmini::Value& doc) {
    const size_t factor = number(*doc.get("processor"), "patch_size") * number(*doc.get("processor"), "merge_size");
    for (const auto& c : doc.get("resize")->asArray()) {
        const std::string& name = text(c, "file");
        const std::vector<uint8_t> bytes = load(dir / name);
        const image::Raster img = image::decode_image(bytes.data(), bytes.size());
        const auto [h, w] = image::smart_resize(img.height, img.width, uint32_t(factor), number(c, "min_pixels"), number(c, "max_pixels"));
        const std::string at = name + " at " + std::to_string(number(c, "min_pixels")) + ".." + std::to_string(number(c, "max_pixels"));
        require(h == number(c, "height") && w == number(c, "width"), at + ": resized to " + std::to_string(w) + "x" + std::to_string(h) + ", the reference " +
                                                                          std::to_string(number(c, "width")) + "x" + std::to_string(number(c, "height")));
        const image::Raster out = image::resize_bicubic(img, w, h);
        require(sha256(out.rgb.data(), out.rgb.size()) == text(c, "rgb_sha256"), at + ": the resized bytes are not torchvision's");
        // The processor's pixel_values: the same image normalized and cut into patches in merge-block order.
        image::Params params;
        params.patch = uint32_t(number(*doc.get("processor"), "patch_size"));
        params.merge = uint32_t(number(*doc.get("processor"), "merge_size"));
        params.min_pixels = number(c, "min_pixels");
        params.max_pixels = number(c, "max_pixels");
        const image::Prepared prep = image::prepare(img, params);
        const auto& grid = c.get("grid")->asArray();
        require(prep.grid_h == size_t(grid[0].asNumber()) && prep.grid_w == size_t(grid[1].asNumber()), at + ": the patch grid is not the processor's");
        require(sha256(reinterpret_cast<const uint8_t*>(prep.patches.data()), prep.patches.size() * sizeof(float)) == text(c, "patches_sha256"),
                at + ": the prepared patches are not the processor's pixel_values");
    }
}

void smart_resizes(const jmini::Value& doc) {
    for (const auto& c : doc.get("smart_resize")->asArray()) {
        const size_t h = number(c, "height"), w = number(c, "width");
        const std::string at = std::to_string(w) + "x" + std::to_string(h) + " in " + std::to_string(number(c, "min_pixels")) + ".." + std::to_string(number(c, "max_pixels"));
        const jmini::Value* result = c.get("result");
        if (result->isString()) {
            bool refused = false;
            try {
                (void)image::smart_resize(uint32_t(h), uint32_t(w), uint32_t(number(c, "factor")), number(c, "min_pixels"), number(c, "max_pixels"));
            } catch (const std::runtime_error&) {
                refused = true;
            }
            require(refused, at + ": HF refuses it and llmx does not");
            continue;
        }
        const auto [rh, rw] = image::smart_resize(uint32_t(h), uint32_t(w), uint32_t(number(c, "factor")), number(c, "min_pixels"), number(c, "max_pixels"));
        require(rh == size_t(result->asArray()[0].asNumber()) && rw == size_t(result->asArray()[1].asNumber()),
                at + ": " + std::to_string(rw) + "x" + std::to_string(rh) + " against HF's " + std::to_string(size_t(result->asArray()[1].asNumber())) + "x" +
                    std::to_string(size_t(result->asArray()[0].asNumber())));
    }
}

// DeepSeek-V4.1's padded fit held to image_processor.py's plan_image_grid and load_image (tests/data/deepseek41_image_fit.json, from `tools/deepseek4_reference.py --vision`).
// Each case's size, padded bytes and BF16 patches are the processor's, Pillow's 22-bit taps differ from torchvision's 16-bit ones on some case, and budgets below a block of 4 rows are refused.
void padded_fits(const std::filesystem::path& dir, const jmini::Value& doc) {
    const jmini::Value& proc = *doc.get("processor");
    image::Params params;
    params.patch = uint32_t(number(proc, "vision_patch_size"));
    params.merge = uint32_t(number(proc, "vision_downsample_ratio"));
    params.min_pixels = number(proc, "vision_min_pixels");
    params.fit = image::Fit::padded;
    params.fill = uint8_t(number(doc, "fill"));
    size_t forms_differ = 0;
    for (const auto& c : doc.get("cases")->asArray()) {
        const std::string& name = text(c, "file");
        const std::vector<uint8_t> bytes = load(dir / name);
        require(sha256(bytes.data(), bytes.size()) == text(c, "sha256"), name + ": the file is not the fixture's");
        const image::Raster img = image::decode_image(bytes.data(), bytes.size());
        params.max_rows = number(c, "max_rows");
        const std::string at = name + " at " + std::to_string(params.max_rows) + " rows";
        const auto [h, w] = image::fit_size(img.height, img.width, params);
        const auto& best = c.get("best")->asArray();
        require(h == size_t(best[0].asNumber()) && w == size_t(best[1].asNumber()), at + ": fitted to " + std::to_string(w) + "x" + std::to_string(h) + ", the processor " +
                                                                                     std::to_string(size_t(best[1].asNumber())) + "x" + std::to_string(size_t(best[0].asNumber())));
        require(image::padded_rows(h, w, params.patch, params.merge) == number(c, "rows"), at + ": the block's rows are not the processor's");
        const image::Raster padded = image::resize_for(img, params);
        require(sha256(padded.rgb.data(), padded.rgb.size()) == text(c, "rgb_sha256"), at + ": the padded bytes are not ImageOps.pad's");
        const image::Prepared prep = image::patches(padded, params);
        const auto& grid = c.get("grid")->asArray();
        require(prep.grid_h == size_t(grid[0].asNumber()) && prep.grid_w == size_t(grid[1].asNumber()), at + ": the patch grid is not the processor's");
        require(sha256(reinterpret_cast<const uint8_t*>(prep.patches.data()), prep.patches.size() * sizeof(float)) == text(c, "patches_sha256"),
                at + ": the BF16 patches are not load_image's");
        // The image's own resize inside the pad, in torchvision's form: a case where it differs shows the 22-bit taps are what matched.
        const double ratio = double(img.width) / img.height, dest = double(w) / h;
        const uint32_t fw = ratio < dest ? uint32_t(std::nearbyint(ratio * h)) : w, fh = ratio > dest ? uint32_t(std::nearbyint(double(img.height) / img.width * w)) : h;
        const image::Raster pillow = image::resize_bicubic(img, fw, fh, image::Resample::pillow), tv = image::resize_bicubic(img, fw, fh, image::Resample::torchvision);
        forms_differ += pillow.rgb != tv.rgb;
    }
    require(forms_differ > 0, "padded fit: torchvision's 16-bit taps give Pillow's bytes on every case, so no case tells the forms apart");
    for (const auto& c : doc.get("refuse")->asArray()) {
        const std::string& name = text(c, "file");
        const std::vector<uint8_t> bytes = load(dir / name);
        const image::Raster img = image::decode_image(bytes.data(), bytes.size());
        params.max_rows = number(c, "max_rows");
        std::string why;
        try {
            (void)image::fit_size(img.height, img.width, params);
        } catch (const std::runtime_error& e) {
            why = e.what();
        }
        require(why.find(text(c, "error")) != std::string::npos, name + " at " + std::to_string(params.max_rows) + " rows: refused with \"" + why + "\"");
    }
    std::cout << "image-decode: " << doc.get("cases")->asArray().size() << " padded fits, Pillow's taps told from torchvision's on " << forms_differ << "\n";
}

// Inkling's pad_end fit held to transformers' Inkling2ImageProcessor (tests/data/inkling2_image_fit.json, from `tools/inkling2_vision_reference.py --fit`).
// Each image keeps its size, its grid ceils to whole 40 px tokens, and the F32 patches, the pad read as -1 before the normalization, are pixel_values' bytes; one token under the count is refused.
void pad_end_fits(const std::filesystem::path& dir, const jmini::Value& doc) {
    const jmini::Value& proc = *doc.get("processor");
    image::Params params;
    params.patch = uint32_t(number(proc, "patch"));
    params.merge = uint32_t(number(proc, "merge"));
    params.fit = image::Fit::pad_end;
    params.pad_value = float(proc.get("pad_value")->asNumber());
    params.matte = image::Matte::checker;
    size_t transparent = 0;
    for (const auto& c : doc.get("cases")->asArray()) {
        const std::string& name = text(c, "file");
        const std::vector<uint8_t> bytes = load(dir / name);
        require(sha256(bytes.data(), bytes.size()) == text(c, "sha256"), name + ": the file is not the fixture's");
        const image::Raster img = image::decode_image(bytes.data(), bytes.size());
        const auto& size = c.get("size")->asArray();
        require(img.height == size_t(size[0].asNumber()) && img.width == size_t(size[1].asNumber()), name + ": decoded to another size than exif_transpose's");
        const bool alpha = c.get("transparent")->b;
        require(img.alpha.empty() != alpha, name + (alpha ? ": Pillow sees transparency the decoder dropped" : ": the decoder kept an alpha Pillow does not see"));
        transparent += alpha;
        params.max_rows = number(c, "max_rows");
        const image::Raster kept = image::resize_for(img, params);
        require(alpha || kept.rgb == img.rgb, name + ": pad_end changed the image");
        const image::Prepared prep = image::patches(kept, params);
        const auto& grid = c.get("grid")->asArray();
        require(prep.grid_h == size_t(grid[0].asNumber()) && prep.grid_w == size_t(grid[1].asNumber()),
                name + ": grid " + std::to_string(prep.grid_h) + "x" + std::to_string(prep.grid_w) + " is not the processor's");
        require(image::grid(kept, params) == std::make_pair(prep.grid_h, prep.grid_w), name + ": image::grid is not the patches' grid");
        require(sha256(reinterpret_cast<const uint8_t*>(prep.patches.data()), prep.patches.size() * sizeof(float)) == text(c, "patches_sha256"),
                name + ": the patches are not pixel_values");
    }
    for (const auto& c : doc.get("refuse_decode")->asArray()) {
        const std::string& name = text(c, "file");
        const std::vector<uint8_t> bytes = load(dir / name);
        std::string why;
        try {
            (void)image::decode_image(bytes.data(), bytes.size());
        } catch (const std::runtime_error& e) {
            why = e.what();
        }
        require(why.find(text(c, "error")) != std::string::npos, name + ": refused with \"" + why + "\"");
    }
    for (const auto& c : doc.get("refuse")->asArray()) {
        const std::string& name = text(c, "file");
        const std::vector<uint8_t> bytes = load(dir / name);
        const image::Raster img = image::decode_image(bytes.data(), bytes.size());
        params.max_rows = number(c, "max_rows");
        std::string why;
        try {
            (void)image::fit_size(img.height, img.width, params);
        } catch (const std::runtime_error& e) {
            why = e.what();
        }
        require(why.find(text(c, "error")) != std::string::npos, name + " at " + std::to_string(params.max_rows) + " tokens: refused with \"" + why + "\"");
    }
    std::cout << "image-decode: " << doc.get("cases")->asArray().size() << " pad_end fits, " << transparent << " composited onto the checkerboard\n";
}

std::string refusal(const std::vector<uint8_t>& bytes, uint64_t max_pixels = image::kMaxDecodedPixels) {
    try {
        (void)image::decode_image(bytes.data(), bytes.size(), max_pixels);
    } catch (const std::runtime_error& e) {
        return e.what();
    }
    return {};
}

void refusals(const std::filesystem::path& dir, const jmini::Value& doc) {
    for (const auto& c : doc.get("refuse")->asArray()) {
        const std::string& name = text(c, "file");
        const std::string why = refusal(load(dir / name));
        require(why.find(text(c, "error")) != std::string::npos, name + ": refused with \"" + why + "\", not \"" + text(c, "error") + "\"");
    }
    // The pixel limit, one pixel under each image's own count.
    for (const char* name : {"photo-300x200.png", "photo-300x200.jpg"}) {
        const std::string why = refusal(load(dir / name), 300 * 200 - 1);
        require(why.find("more pixels than the limit") != std::string::npos, std::string(name) + ": the pixel limit refused it with \"" + why + "\"");
        require(refusal(load(dir / name), 300 * 200).empty(), std::string(name) + ": refused at exactly its pixel count");
    }
}

// Every prefix of each image is refused, never decoded past its end; corrupted bytes are refused or decoded, never anything else.
void damage(const std::filesystem::path& dir, const jmini::Value& doc) {
    uint32_t x = 0x9e3779b9u;
    size_t prefixes = 0, corrupted = 0;
    for (const auto& c : doc.get("decode")->asArray()) {
        const std::vector<uint8_t> bytes = load(dir / text(c, "file"));
        const size_t step = std::max<size_t>(1, bytes.size() / 97);
        for (size_t n = 0; n < bytes.size(); n += step) {
            const std::vector<uint8_t> cut(bytes.begin(), bytes.begin() + std::ptrdiff_t(n));
            const std::string why = refusal(cut);
            // A JPEG cut after its last scan but before its end-of-image marker is still refused; a PNG cut anywhere loses its IEND.
            require(!why.empty(), text(c, "file") + ": a prefix of " + std::to_string(n) + " bytes decoded");
            ++prefixes;
        }
        for (int k = 0; k < 24; ++k) {
            std::vector<uint8_t> bad = bytes;
            x = x * 1664525u + 1013904223u;
            bad[size_t(x >> 8) % bad.size()] ^= uint8_t(1u << (x & 7));
            try {
                (void)image::decode_image(bad.data(), bad.size());
            } catch (const std::runtime_error&) {
            }
            ++corrupted;
        }
    }
    require(prefixes > 1000 && corrupted > 1000, "damage: too few cases");
}

// Files whose decode would cost far more than their bytes, built here: a progressive JPEG of many scans, a frame header alone at a large size, and a PNG whose compressed texts are long.
namespace synth {

void be16(std::vector<uint8_t>& v, size_t x) {
    v.push_back(uint8_t(x >> 8));
    v.push_back(uint8_t(x));
}

void be32(std::vector<uint8_t>& v, uint32_t x) {
    be16(v, x >> 16);
    be16(v, x & 0xffff);
}

void segment(std::vector<uint8_t>& v, uint8_t marker, const std::vector<uint8_t>& body) {
    v.push_back(0xff);
    v.push_back(marker);
    be16(v, body.size() + 2);
    v.insert(v.end(), body.begin(), body.end());
}

// A quantization table of ones and a frame of one gray component, `sof` the frame marker.
std::vector<uint8_t> frame(uint8_t sof, size_t width, size_t height) {
    std::vector<uint8_t> v{0xff, 0xd8};
    std::vector<uint8_t> dqt{0};
    dqt.insert(dqt.end(), 64, 1);
    segment(v, 0xdb, dqt);
    std::vector<uint8_t> f{8};
    be16(f, height);
    be16(f, width);
    f.insert(f.end(), {1, 1, 0x11, 0});
    segment(v, sof, f);
    return v;
}

// An 8 by 8 progressive JPEG of `scans` scans: the DC first and refinement, then AC bands of one coefficient each; both tables hold one code, so a block is one bit.
std::vector<uint8_t> progressive_jpeg(int scans) {
    std::vector<uint8_t> v = frame(0xc2, 8, 8);
    for (int tc : {0, 1}) {
        std::vector<uint8_t> dht{uint8_t(tc << 4), 1};
        dht.insert(dht.end(), 15, 0);
        dht.push_back(0);
        segment(v, 0xc4, dht);
    }
    const auto scan = [&](int ss, int se, int ah, int al) {
        segment(v, 0xda, {1, 1, 0x00, uint8_t(ss), uint8_t(se), uint8_t(ah << 4 | al)});
        v.push_back(0x00);
    };
    scan(0, 0, 0, 1);
    if (scans > 1) scan(0, 0, 1, 0);
    for (int k = 1; k + 2 <= scans; ++k) scan(k, k, 0, 0);
    v.insert(v.end(), {0xff, 0xd9});
    return v;
}

// A frame header of `side` by `side` gray pixels and the end of image, with no scan between them.
std::vector<uint8_t> frame_only_jpeg(size_t side) {
    std::vector<uint8_t> v = frame(0xc0, side, side);
    v.insert(v.end(), {0xff, 0xd9});
    return v;
}

uint32_t adler32(const std::vector<uint8_t>& d) {
    uint32_t a = 1, b = 0;
    for (uint8_t x : d) {
        a = (a + x) % 65521;
        b = (b + a) % 65521;
    }
    return b << 16 | a;
}

// A zlib stream of `n` zero bytes in stored blocks.
std::vector<uint8_t> zeros_zlib(size_t n) {
    std::vector<uint8_t> z{0x78, 0x01};
    for (size_t at = 0; at < n;) {
        const size_t len = std::min<size_t>(65535, n - at);
        z.push_back(at + len == n ? 1 : 0);
        z.insert(z.end(), {uint8_t(len), uint8_t(len >> 8), uint8_t(~len), uint8_t(~len >> 8)});
        z.insert(z.end(), len, 0);
        at += len;
    }
    be32(z, adler32(std::vector<uint8_t>(n, 0)));
    return z;
}

void chunk(std::vector<uint8_t>& v, const char* type, const std::vector<uint8_t>& data) {
    be32(v, uint32_t(data.size()));
    std::vector<uint8_t> body(type, type + 4);
    body.insert(body.end(), data.begin(), data.end());
    v.insert(v.end(), body.begin(), body.end());
    be32(v, image::png_detail::crc32(body.data(), body.size()));
}

// A 1 by 1 gray PNG with a zTXt chunk named XML:com.adobe.xmp inflating to each of `texts` bytes.
std::vector<uint8_t> text_png(const std::vector<size_t>& texts) {
    std::vector<uint8_t> v{0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
    chunk(v, "IHDR", {0, 0, 0, 1, 0, 0, 0, 1, 8, 0, 0, 0, 0});
    for (size_t n : texts) {
        std::vector<uint8_t> t(reinterpret_cast<const uint8_t*>("XML:com.adobe.xmp"), reinterpret_cast<const uint8_t*>("XML:com.adobe.xmp") + 18);
        t.push_back(0);
        const std::vector<uint8_t> z = zeros_zlib(n);
        t.insert(t.end(), z.begin(), z.end());
        chunk(v, "zTXt", t);
    }
    std::vector<uint8_t> raw{0x78, 0x01, 0x01, 0x02, 0x00, 0xfd, 0xff, 0x00, 0x80};
    be32(raw, adler32({0x00, 0x80}));
    chunk(v, "IDAT", raw);
    chunk(v, "IEND", {});
    return v;
}

}  // namespace synth

// What a hostile file may cost: scans past the cap are refused, a frame without a scan commits no plane, and compressed texts are read within one budget a file.
void limits() {
    using Clock = std::chrono::steady_clock;
    require(refusal(synth::progressive_jpeg(64)).empty(), "a progressive JPEG of 64 scans was refused");
    const std::string scans = refusal(synth::progressive_jpeg(65));
    require(scans.find("scans") != std::string::npos, "a progressive JPEG of 65 scans was not refused for its scans: \"" + scans + "\"");
    // 9459 squared is under the pixel limit; a plane of it is 179 MB, which a refusal without one never takes the time to fill.
    const std::vector<uint8_t> header = synth::frame_only_jpeg(9459);
    const Clock::time_point t0 = Clock::now();
    const std::string frame = refusal(header);
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    require(frame.find("no scan") != std::string::npos, "a frame header alone at 9459 by 9459 was not refused for its missing scan: \"" + frame + "\"");
    require(ms < 200.0, "a frame header alone at 9459 by 9459 took " + std::to_string(ms) + " ms to refuse");
    constexpr size_t mib = size_t(1) << 20;
    require(refusal(synth::text_png({mib / 2, mib / 2})).empty(), "a PNG whose texts inflate to 1 MiB together was refused");
    const std::string one = refusal(synth::text_png({2 * mib}));
    require(one.find("inflate past 1 MiB") != std::string::npos, "a PNG whose text inflates to 2 MiB was not refused: \"" + one + "\"");
    const std::string together = refusal(synth::text_png({mib / 2, mib / 2, mib / 2}));
    require(together.find("inflate past 1 MiB") != std::string::npos, "a PNG whose texts inflate to 1.5 MiB together was not refused: \"" + together + "\"");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: llmx-image-decode-test tests/data/image_decode.json\n";
        return 2;
    }
    try {
        const std::filesystem::path json = argv[1], dir = json.parent_path() / "images";
        const std::vector<uint8_t> raw = load(json);
        const jmini::Value doc = jmini::parse(std::string(raw.begin(), raw.end()));
        smart_resizes(doc);
        decodes(dir, doc);
        resizes(dir, doc);
        const std::vector<uint8_t> fit = load(json.parent_path() / "deepseek41_image_fit.json");
        padded_fits(dir, jmini::parse(std::string(fit.begin(), fit.end())));
        const std::vector<uint8_t> ink = load(json.parent_path() / "inkling2_image_fit.json");
        pad_end_fits(dir, jmini::parse(std::string(ink.begin(), ink.end())));
        refusals(dir, doc);
        damage(dir, doc);
        limits();
    } catch (const std::exception& e) {
        std::cerr << "image-decode: FAIL: " << e.what() << "\n";
        return 1;
    }
    std::cout << "image-decode: " << checks << " checks passed\n";
    return 0;
}
