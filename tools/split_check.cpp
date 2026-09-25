// A model on one device against the same model split by layers over several, compared as raw float logits: every position of a scored text through the prompt path, then a prefill in chunks of the ubatch, which a split pipelines over its stages, and greedy decode steps, bit for bit (docs/MULTI-DEVICE.md, phases 1 and 2).
// Usage: llmx-split-check <model.gguf> <text file> [single device] [split devices, comma separated] [decode steps] [ubatch] [cache type]; a device is `cpu` or a Vulkan index, and the cache type, f16 or f32, stores both sides of both models' caches, the model's default when left out.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include "backends/devices.hpp"
#include "format/gguf.hpp"
#include "model/arch_qwen.hpp"
#include "tokenizer/tokenizer.hpp"

// A device given as `cpu` or a Vulkan index, in the spelling a device list takes.
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
        std::fprintf(stderr, "usage: llmx-split-check <model.gguf> <text file> [single] [split, e.g. 0,1,2] [steps] [ubatch] [f16|f32]\n");
        return 2;
    }
    try {
        const std::string single = argc > 3 ? argv[3] : "0", split = argc > 4 ? argv[4] : "0,1";
        const int steps = argc > 5 ? std::atoi(argv[5]) : 32, ubatch = argc > 6 ? std::atoi(argv[6]) : 0;
        // The cache type is checked before the model file is read, so a wrong name costs no load.
        infer::ModelOptions options;
        if (argc > 7) options.kv_k = options.kv_v = backend::kv_type_of(argv[7]);
        options.kv_tokens = 4096;
        gguf::GGUFModel m = gguf::read_gguf(argv[1]);
        bpe::Tokenizer tok(m);
        std::ifstream in(argv[2], std::ios::binary);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::vector<uint32_t> ids = tok.encode(text);
        if (ids.size() < 2) throw std::runtime_error("the text holds fewer than two tokens");

        infer::Model one(m, backend::make_backend(name(single)), options);
        one.set_ubatch(ubatch);
        // The split takes equal shares of the layers, placed as a device list with --layer-shares 1,1,... places them.
        // Each entry is a backend of its own, without the CLI's listed-once rule (backend::device_specs), so `cpu,cpu` splits over two CPU backends.
        infer::PlacementRequest request;
        for (const std::string& d : core::comma_list(split)) {
            request.names.push_back(name(d));
            request.shares.push_back(1);
        }
        request.ubatch = ubatch;
        infer::PlacedModel placed = infer::place_model(m, backend::make_backends(request.names), request, options);
        infer::Model& two = *placed.model;
        std::printf("%s: %zu tokens, %s caches; single %s, split:\n%s", argv[1], ids.size(), backend::kv_type_name(options.kv_k), name(single).c_str(), placed.plan.c_str());

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
