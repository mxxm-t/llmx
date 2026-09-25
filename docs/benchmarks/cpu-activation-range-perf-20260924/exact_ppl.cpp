#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include "config.hpp"
#include "inference/perplexity.hpp"
#include "tokenizer/tokenizer.hpp"

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: exact-ppl MODEL CORPUS");
        std::ifstream input(argv[2], std::ios::binary);
        if (!input) throw std::runtime_error("cannot open corpus");
        const std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        auto weights = gguf::read_gguf(argv[1]);
        bpe::Tokenizer tokenizer(weights);
        const auto ids = tokenizer.encode(text);
        if (ids.size() < 1024) throw std::runtime_error("corpus is too short");
        infer::ModelOptions options;
        options.kv_k = options.kv_v = backend::KVType::f16;
        options.kv_tokens = 512;
        infer::Model model(weights, backend::make_cpu_backend(), options);
        model.set_threads(6);
        model.set_ubatch(128);
        std::cout << "version: " << LLMX_VERSION_STRING << '\n';
        for (bool per_token : {false, true}) {
            const auto result = infer::perplexity(model, ids, 512, 2, per_token);
            if (result.used_tokens != 1024 || result.scored_tokens != 1022 || result.chunks != 2 || !std::isfinite(result.nll))
                throw std::runtime_error("unexpected scoring result");
            std::cout << (per_token ? "decode" : "batched") << " " << result.used_tokens << " "
                      << result.scored_tokens << " " << result.chunks << " " << std::hexfloat
                      << result.nll << " " << result.mean_nll() << std::defaultfloat << std::endl;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
