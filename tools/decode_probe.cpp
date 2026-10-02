// A reply's decode path on a real model: the prompt read as one prefill, then each forced id of a fixture fed as a decode step, as a request alone runs through the server.
// Each step prints its greedy token and the forced one with their logits; after the last forced id it prints the step's five best, or every id of a smaller vocabulary, and the logits of the fixture's two tokens, where two builds' greedy replies parted.
// A fixture's `draft`, a count, loads the file's embedded drafter and prints its drafts after the last step's greedy token, each draft row's id and every logit of its row (docs/SPECULATIVE.md, section 7); its `cache`, f16 or f32, stores both cache sides so, f16 when left out.
// Usage: llmx-decode-probe <model.gguf> <fixture.json> [device]; the device is `cpu` or a Vulkan index, 0 when left out. It exits 1 where a forced id is not its step's greedy token, and 2 on a fixture entry that is not a whole number inside the vocabulary.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include "backends/devices.hpp"
#include "core/json.hpp"
#include "inference/load.hpp"
#include "inference/logprobs.hpp"

// Entry i of a fixture's array `name` as a token id: a finite whole number from 0 to 2^32 - 1 below `vocab`, or an error naming the entry.
static uint32_t fixture_id(const jmini::Value& v, const char* name, size_t i, size_t vocab) {
    const std::string at = std::string(name) + "[" + std::to_string(i) + "]";
    if (!v.isNumber()) throw std::runtime_error(at + " is not a number");
    const double x = v.asNumber();
    if (!std::isfinite(x) || x != std::floor(x) || x < 0 || x > 4294967295.0) throw std::runtime_error(at + " is not a whole number from 0 to 4294967295");
    if (x >= (double)vocab) throw std::runtime_error(at + " lies outside the vocabulary");
    return (uint32_t)x;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: llmx-decode-probe <model.gguf> <fixture.json> [device]\n");
        return 2;
    }
    try {
        std::ifstream in(argv[2], std::ios::binary);
        if (!in) throw std::runtime_error(std::string("cannot read ") + argv[2]);
        const jmini::Value fixture = jmini::parse(std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
        const jmini::Value *prompt = fixture.get("prompt"), *ids = fixture.get("ids"), *tokens = fixture.get("tokens");
        if (!prompt || !prompt->isString() || !ids || !ids->isArray() || !tokens || !tokens->isArray() || tokens->asArray().size() != 2)
            throw std::runtime_error("the fixture needs a prompt, its forced ids and the two tokens where the replies part");
        const std::string device = argc > 3 ? argv[3] : "0";
        const jmini::Value* draft = fixture.get("draft");
        if (draft && (!draft->isNumber() || draft->asNumber() != std::floor(draft->asNumber()) || draft->asNumber() < 1 || draft->asNumber() > 63))
            throw std::runtime_error("draft is not a whole number from 1 to 63");
        infer::PlacementRequest request;
        request.names = {device == "cpu" ? device : "vulkan:" + device};
        request.drafter = draft != nullptr;
        infer::ModelOptions options;
        options.kv_tokens = 4096;
        if (const jmini::Value* cache = fixture.get("cache")) {
            if (!cache->isString()) throw std::runtime_error("cache is not a cache type");
            options.kv_k = options.kv_v = backend::kv_type_of(cache->asString());
        }
        const auto loaded = infer::load_model(argv[1], backend::make_backends(request.names), request, options);
        infer::Model& model = *loaded->model;
        const size_t vocab = model.n_vocab();
        std::vector<uint32_t> forced;
        for (size_t i = 0; i < ids->asArray().size(); ++i) forced.push_back(fixture_id(ids->asArray()[i], "ids", i, vocab));
        const uint32_t a = fixture_id(tokens->asArray()[0], "tokens", 0, vocab), b = fixture_id(tokens->asArray()[1], "tokens", 1, vocab);

        const std::vector<uint32_t> prompt_ids = loaded->tok->encode(prompt->asString());
        std::printf("%s on %s: %zu prompt tokens, %zu forced ids\n", argv[1], request.names[0].c_str(), prompt_ids.size(), forced.size());
        std::vector<float> logits = model.prefill(prompt_ids);
        // The n best ids, or every id of a smaller vocabulary, best first and a tie to the lower id, as greedy sampling takes the first.
        auto best = [&](size_t n) { return infer::top_logprobs(logits.data(), vocab, 0.0, n); };
        size_t off = 0;
        for (size_t s = 0; s < forced.size(); ++s) {
            const auto top = best(2);
            off += top[0].id != forced[s];
            std::printf("step %zu greedy %u %.6f", s, top[0].id, logits[top[0].id]);
            if (top.size() > 1) std::printf(" second %u %.6f", top[1].id, logits[top[1].id]);
            std::printf(" forced %u %.6f\n", forced[s], logits[forced[s]]);
            logits = model.step((int)forced[s]);
        }
        std::printf("step %zu best", forced.size());
        for (const auto& t : best(5)) std::printf(" %u %.6f", t.id, logits[t.id]);
        std::printf("\nstep %zu tokens %u %.6f %u %.6f, the first ahead by %.6f\n", forced.size(), a, logits[a], b, logits[b], logits[a] - logits[b]);
        if (draft) {
            const uint32_t pick = best(1)[0].id;
            std::vector<uint32_t> drafts;
            model.draft(pick, (size_t)draft->asNumber(), drafts);
            std::printf("draft pick %u, %zu drafts\n", pick, drafts.size());
            for (size_t m = 0; m < drafts.size(); ++m) {
                const float* row = model.draft_logits(m);
                std::printf("draft %zu %u:", m, drafts[m]);
                for (size_t i = 0; i < vocab; ++i) std::printf(" %.9g", row[i]);
                std::printf("\n");
            }
        }
        std::printf("%s\n", off ? "the forced ids left the greedy path" : "every forced id is its step's greedy token");
        return off ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "llmx-decode-probe: %s\n", e.what());
        return 2;
    }
}
