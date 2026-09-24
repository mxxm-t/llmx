// A model on one Vulkan device against the same model split by layers over two, compared as raw float logits: every position of a scored text through the prompt path, then greedy decode steps, bit for bit (docs/MULTI-DEVICE.md, phase 1).
// Usage: llmx-split-check <model.gguf> <text file> [single device] [first split device] [second split device] [decode steps]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include "backends/vulkan/vulkan_backend.hpp"
#include "format/gguf.hpp"
#include "model/arch_qwen.hpp"
#include "model/layer_split.hpp"
#include "tokenizer/tokenizer.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: llmx-split-check <model.gguf> <text file> [single] [first] [second] [steps]\n");
        return 2;
    }
    try {
        const int single = argc > 3 ? std::atoi(argv[3]) : 0, first = argc > 4 ? std::atoi(argv[4]) : 0, second = argc > 5 ? std::atoi(argv[5]) : 1;
        const int steps = argc > 6 ? std::atoi(argv[6]) : 32;
        gguf::GGUFModel m = gguf::read_gguf(argv[1]);
        bpe::Tokenizer tok(m);
        std::ifstream in(argv[2], std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::vector<uint32_t> ids = tok.encode(text);
        if (ids.size() < 2) throw std::runtime_error("the text holds fewer than two tokens");

        infer::ModelOptions options;
        options.kv_tokens = 4096;
        infer::Model one(m, backend::make_vulkan_backend(single), options);
        std::vector<backend::BackendPtr> pair{backend::make_vulkan_backend(first), backend::make_vulkan_backend(second)};
        std::vector<infer::DeviceBudget> budgets;
        for (size_t d = 0; d < pair.size(); ++d) budgets.push_back(infer::DeviceBudget{"vulkan:" + std::to_string(d ? second : first), std::nullopt, false, {}, 0});
        const infer::LayerSplit split = infer::split_layers(infer::footprint(m, options), budgets, 512, {1, 1});
        infer::Model two(m, std::move(pair), infer::placement_for(split), options);
        std::printf("%s: %zu tokens; single vulkan:%d, split layers 0-%d on vulkan:%d and %d-%d on vulkan:%d\n", argv[1], ids.size(), single,
                    split.stages[0].count - 1, first, split.stages[0].count, split.stages[0].count + split.stages[1].count - 1, second);

        const size_t vocab = one.n_vocab();
        std::vector<float> scored(ids.size() * vocab);
        one.score(ids, [&](size_t pos, const float* logits) { std::memcpy(&scored[pos * vocab], logits, vocab * sizeof(float)); });
        size_t differ = 0;
        two.score(ids, [&](size_t pos, const float* logits) { differ += std::memcmp(&scored[pos * vocab], logits, vocab * sizeof(float)) != 0; });
        std::printf("prompt path: %zu positions x %zu logits, %zu positions differ\n", ids.size(), vocab, differ);

        // Decode from the prompt, both models fed the same greedy token each step.
        one.reset();
        two.reset();
        std::vector<float> a = one.prefill(ids), b = two.prefill(ids);
        size_t steps_differ = std::memcmp(a.data(), b.data(), vocab * sizeof(float)) != 0;
        for (int s = 0; s < steps; ++s) {
            uint32_t next = 0;
            for (size_t v = 1; v < vocab; ++v)
                if (a[v] > a[next]) next = (uint32_t)v;
            a = one.step((int)next);
            b = two.step((int)next);
            steps_differ += std::memcmp(a.data(), b.data(), vocab * sizeof(float)) != 0;
        }
        std::printf("decode path: prefill and %d greedy steps, %zu differ\n", steps, steps_differ);
        const bool same = !differ && !steps_differ;
        std::printf("%s\n", same ? "bit-identical" : "DIFFERENT");
        return same ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "llmx-split-check: %s\n", e.what());
        return 2;
    }
}
