// A reply's decode path on a real model: the prompt read as one prefill, then each forced id of a fixture fed as a decode step, as a request alone runs through the server.
// Each step prints its greedy token and the forced one with their logits; after the last forced id it prints the step's five best and the logits of the fixture's two tokens, where two builds' greedy replies parted.
// Usage: llmx-decode-probe <model.gguf> <fixture.json> [device]; the device is `cpu` or a Vulkan index, 0 when left out. It exits 1 where a forced id is not its step's greedy token.
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>
#include "backends/devices.hpp"
#include "core/json.hpp"
#include "inference/load.hpp"

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
        std::vector<uint32_t> forced;
        for (const jmini::Value& v : ids->asArray()) forced.push_back((uint32_t)v.asNumber());
        const uint32_t a = (uint32_t)tokens->asArray()[0].asNumber(), b = (uint32_t)tokens->asArray()[1].asNumber();

        const std::string device = argc > 3 ? argv[3] : "0";
        infer::PlacementRequest request;
        request.names = {device == "cpu" ? device : "vulkan:" + device};
        infer::ModelOptions options;
        options.kv_tokens = 4096;
        const auto loaded = infer::load_model(argv[1], backend::make_backends(request.names), request, options);
        infer::Model& model = *loaded->model;
        const size_t vocab = model.n_vocab();
        for (uint32_t id : forced)
            if (id >= vocab) throw std::runtime_error("a forced id lies outside the vocabulary");
        if (a >= vocab || b >= vocab) throw std::runtime_error("a token lies outside the vocabulary");

        const std::vector<uint32_t> prompt_ids = loaded->tok->encode(prompt->asString());
        std::printf("%s on %s: %zu prompt tokens, %zu forced ids\n", argv[1], request.names[0].c_str(), prompt_ids.size(), forced.size());
        std::vector<float> logits = model.prefill(prompt_ids);
        // The n best ids, best first and a tie to the lower id, as greedy sampling takes the first.
        auto best = [&](size_t n) {
            std::vector<uint32_t> top;
            while (top.size() < n) {
                uint32_t g = (uint32_t)vocab;
                for (uint32_t v = 0; v < (uint32_t)vocab; ++v)
                    if ((g == vocab || logits[v] > logits[g]) && std::find(top.begin(), top.end(), v) == top.end()) g = v;
                top.push_back(g);
            }
            return top;
        };
        size_t off = 0;
        for (size_t s = 0; s < forced.size(); ++s) {
            const std::vector<uint32_t> top = best(2);
            off += top[0] != forced[s];
            std::printf("step %zu greedy %u %.6f second %u %.6f forced %u %.6f\n", s, top[0], logits[top[0]], top[1], logits[top[1]], forced[s], logits[forced[s]]);
            logits = model.step((int)forced[s]);
        }
        std::printf("step %zu best", forced.size());
        for (uint32_t id : best(5)) std::printf(" %u %.6f", id, logits[id]);
        std::printf("\nstep %zu tokens %u %.6f %u %.6f, the first ahead by %.6f\n", forced.size(), a, logits[a], b, logits[b], logits[a] - logits[b]);
        std::printf("%s\n", off ? "the forced ids left the greedy path" : "every forced id is its step's greedy token");
        return off ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "llmx-decode-probe: %s\n", e.what());
        return 2;
    }
}
