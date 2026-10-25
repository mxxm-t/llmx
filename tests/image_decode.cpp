// image::decode_image, smart_resize and resize_bicubic held to the bytes of the vision models' processor (tests/data/image_decode.json, from `tools/gen_baseline.py image-decode`).
// Unread formats are refused with their texts, and prefixes and flipped bits of every image are refused or decoded, never read past their end.
#include <algorithm>
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
        refusals(dir, doc);
        damage(dir, doc);
    } catch (const std::exception& e) {
        std::cerr << "image-decode: FAIL: " << e.what() << "\n";
        return 1;
    }
    std::cout << "image-decode: " << checks << " checks passed\n";
    return 0;
}
