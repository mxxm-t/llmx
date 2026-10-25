// llmx-encode-image MMPROJ IMAGE OUT: an image's rows from a vision encoder's mmproj file, written as F32, with its grid, tokens and encode time printed as one JSON line (docs/VISION.md, section 8, step 3).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/devices.hpp"
#include "core/json.hpp"
#include "format/gguf.hpp"
#include "image/image.hpp"
#include "model/arch/registry.hpp"
#include "model/encoder.hpp"
#include "model/place.hpp"

namespace {

int usage(const std::string& why) {
    std::cerr << "usage: llmx-encode-image MMPROJ IMAGE OUT [--device D] [--threads N] [--dtype auto|f32|f16|bf16] [--image-max-pixels N]\n"
              << "error: " << why << "\n";
    return 2;
}

uint64_t positive(const std::string& flag, const std::string& v) {
    size_t used = 0;
    unsigned long long n = 0;
    try {
        n = std::stoull(v, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != v.size() || !n || v[0] == '-' || v[0] == '+') throw std::invalid_argument(flag + " needs a positive whole number");
    return n;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> positional;
    std::string device = "cpu", dtype = "auto";
    int threads = 0;
    std::optional<uint64_t> max_pixels;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            const auto value = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
                return argv[++i];
            };
            if (a == "--device") device = value();
            else if (a == "--threads") threads = int(positive(a, value()));
            else if (a == "--dtype") dtype = value();
            else if (a == "--image-max-pixels") max_pixels = positive(a, value());
            else if (a.size() > 1 && a[0] == '-') throw std::invalid_argument("unknown flag " + a);
            else positional.push_back(a);
        }
        if (positional.size() != 3) throw std::invalid_argument("three files are needed");
    } catch (const std::invalid_argument& e) {
        return usage(e.what());
    }
    try {
        gguf::GGUFModel file = gguf::read_gguf(positional[0]);
        gguf::map_payload(file);
        const infer::VisionWeights weights = infer::vision_weights(file);
        std::vector<backend::BackendPtr> backends = backend::make_backends(backend::device_specs(device));
        if (backends.size() != 1) throw std::runtime_error("--device: the encoder runs on one device");
        if (threads) backends[0]->set_threads(threads);
        std::optional<backend::Dtype> requested;
        if (dtype == "f32") requested = backend::Dtype::f32;
        else if (dtype == "f16") requested = backend::Dtype::f16;
        else if (dtype == "bf16") requested = backend::Dtype::bf16;
        else if (dtype != "auto") throw std::invalid_argument("--dtype takes auto, f32, f16 or bf16");
        const infer::DtypePlan plan = infer::resolve_dtype(weights.declared_dtype, backends, {device}, requested);
        infer::VisionEncoder encoder(weights.arch, weights.tensors, *backends[0], plan.effective);
        std::ifstream in(positional[1], std::ios::binary);
        if (!in) throw std::runtime_error("cannot open " + positional[1]);
        const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const image::Raster raster = image::decode_image(bytes.data(), bytes.size());
        image::Params params = weights.arch->preprocess();
        if (max_pixels) params.max_pixels = *max_pixels;
        const auto t0 = std::chrono::steady_clock::now();
        const image::Prepared prepared = image::prepare(raster, params);
        const auto t1 = std::chrono::steady_clock::now();
        const std::vector<float> rows = encoder.encode(prepared);
        const auto t2 = std::chrono::steady_clock::now();
        std::ofstream out(positional[2], std::ios::binary);
        out.write(reinterpret_cast<const char*>(rows.data()), std::streamsize(rows.size() * sizeof(float)));
        if (!out) throw std::runtime_error("cannot write " + positional[2]);
        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::cout << "{\"grid\":[" << prepared.grid_h << "," << prepared.grid_w << "],\"tokens\":" << encoder.architecture().tokens(prepared.grid_h, prepared.grid_w)
                  << ",\"width\":" << encoder.plan().out_width << ",\"dtype\":" << jmini::quote(backend::dtype_name(plan.effective))
                  << ",\"prepare_ms\":" << jmini::number(float(ms(t0, t1))) << ",\"encode_ms\":" << jmini::number(float(ms(t1, t2))) << "}\n";
    } catch (const std::exception& e) {
        std::cerr << "llmx-encode-image: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
