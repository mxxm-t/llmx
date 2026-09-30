#include <charconv>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include "backends/cpu/cpu_backend.hpp"
#include "inference/load.hpp"

// A test-only control: packed weights on the CPU with original F32 activations, so independent HF checks can separate weight decoding from activation rounding.
int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cerr << "usage: llmx-cpu-f32-check MODEL TEXT THREADS UBATCH\n";
            return 2;
        }
        const auto positive = [](const char* text, int maximum) {
            const std::string_view s(text);
            int value = 0;
            const auto result = std::from_chars(s.data(), s.data() + s.size(), value);
            if (result.ec != std::errc{} || result.ptr != s.data() + s.size() || value < 1 || value > maximum)
                throw std::runtime_error("integer outside the tool's range");
            return value;
        };
        const int threads = positive(argv[3], 64), ubatch = positive(argv[4], 4096);
        auto cpu = std::make_shared<backend::CpuBackend>();
        cpu->set_decode_activations8(false);
        cpu->set_threads(threads);
        infer::PlacementRequest request;
        request.names = {"cpu"};
        request.ubatch = ubatch;
        infer::ModelOptions options;
        options.kv_k = options.kv_v = backend::KVType::f32;
        const auto loaded = infer::load_model(argv[1], {cpu}, request, options);
        loaded->model->set_threads(threads);
        const auto logits = loaded->model->prefill(loaded->tok->encode(argv[2]));
        std::cout << std::setprecision(9);
        for (size_t i = 0; i < logits.size(); ++i) std::cout << i << " " << logits[i] << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
