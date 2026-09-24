// A model on one device against the same model split by layers over two, compared as raw float logits: every position of a scored text through the prompt path, then greedy decode steps, bit for bit (docs/MULTI-DEVICE.md, phase 1).
// Usage: llmx-split-check <model.gguf> <text file> [single device] [first split device] [second split device] [decode steps]; a device is `cpu` or a Vulkan index.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include "backends/cpu/cpu_backend.hpp"
#include "backends/vulkan/vulkan_backend.hpp"
#include "format/gguf.hpp"
#include "model/arch_qwen.hpp"
#include "model/layer_split.hpp"
#include "tokenizer/tokenizer.hpp"

static backend::BackendPtr device(const std::string& spec) {
    return spec == "cpu" ? backend::make_cpu_backend() : backend::make_vulkan_backend(std::atoi(spec.c_str()));
}

static std::string name(const std::string& spec) { return spec == "cpu" ? spec : "vulkan:" + spec; }

// A decoding sequence with an established history beside a fresh prompt, then both decoding, in the same row order on one device and the split.
static size_t mixed(infer::Model& one, infer::Model& two, const std::vector<uint32_t>& ids) {
    one.reset();
    two.reset();
    infer::Sequence a = one.make_sequence(), b = one.make_sequence();
    infer::Sequence c = two.make_sequence(), d = two.make_sequence();
    infer::ExecContext x, y;
    const size_t warm = std::min(size_t(4), ids.size()), prompt = std::min(size_t(64), ids.size());
    const infer::BatchEntry warm_one{&a, ids.data(), warm, false};
    const infer::BatchEntry warm_two{&c, ids.data(), warm, false};
    one.forward(x, &warm_one, 1);
    two.forward(y, &warm_two, 1);
    size_t differ = 0, checked = 0;
    for (size_t pass = 0; pass < 3; ++pass) {
        const uint32_t next_a = ids[pass % ids.size()], next_b = ids[(pass + 1) % ids.size()];
        const size_t fresh = pass == 0 ? prompt : 1;
        const uint32_t* input = pass == 0 ? ids.data() : &next_b;
        const infer::BatchEntry first[] = {{&a, &next_a, 1, true}, {&b, input, fresh, true, true}};
        const infer::BatchEntry second[] = {{&c, &next_a, 1, true}, {&d, input, fresh, true, true}};
        one.forward(x, first, 2);
        two.forward(y, second, 2);
        for (size_t row = 0; row < 1 + fresh; ++row) {
            differ += std::memcmp(x.logits(row), y.logits(row), one.n_vocab() * sizeof(float)) != 0;
            ++checked;
        }
        if (a.length() != warm + pass + 1 || c.length() != a.length() ||
            b.length() != prompt + pass || d.length() != b.length())
            throw std::runtime_error("mixed pass did not preserve both histories");
    }
    std::printf("mixed path: 3 passes, %zu logits rows, %zu differ\n", checked, differ);
    one.reset(a);
    one.reset(b);
    two.reset(c);
    two.reset(d);
    return differ;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: llmx-split-check <model.gguf> <text file> [single] [first] [second] [steps]\n");
        return 2;
    }
    try {
        const std::string single = argc > 3 ? argv[3] : "0", first = argc > 4 ? argv[4] : "0", second = argc > 5 ? argv[5] : "1";
        const int steps = argc > 6 ? std::atoi(argv[6]) : 32;
        gguf::GGUFModel m = gguf::read_gguf(argv[1]);
        bpe::Tokenizer tok(m);
        std::ifstream in(argv[2], std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::vector<uint32_t> ids = tok.encode(text);
        if (ids.size() < 2) throw std::runtime_error("the text holds fewer than two tokens");

        infer::ModelOptions options;
        options.kv_tokens = 4096;
        infer::Model one(m, device(single), options);
        std::vector<backend::BackendPtr> pair{device(first), device(second)};
        std::vector<infer::DeviceBudget> budgets;
        for (const std::string& spec : {first, second}) budgets.push_back(infer::DeviceBudget{name(spec), std::nullopt, spec == "cpu", {}, 0});
        const infer::LayerSplit split = infer::split_layers(infer::footprint(m, options), budgets, 512, {1, 1});
        infer::Model two(m, std::move(pair), infer::placement_for(split), options);
        std::printf("%s: %zu tokens; single %s, split layers 0-%d on %s and %d-%d on %s\n", argv[1], ids.size(), name(single).c_str(),
                    split.stages[0].count - 1, name(first).c_str(), split.stages[0].count, split.stages[0].count + split.stages[1].count - 1,
                    name(second).c_str());

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
        const size_t mixed_differ = mixed(one, two, ids);
        const bool same = !differ && !steps_differ && !mixed_differ;
        std::printf("%s\n", same ? "bit-identical" : "DIFFERENT");
        return same ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "llmx-split-check: %s\n", e.what());
        return 2;
    }
}
