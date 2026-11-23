// A prompt's logits rows and the expert ids each routed step picked, for a reference that follows llmx's picks where it cannot order the experts itself (docs/TESTS.md, Suite components).
// The rows are what `llmx logits --last LAST --top VOCAB --dtype f32 --cache-type-k f32 --cache-type-v f32 --load-mode mapped --ubatch UBATCH` prints after an empty text with the ids through --then-ids.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "backends/devices.hpp"
#include "inference/load.hpp"
#include "inference/logprobs.hpp"

static size_t count(const char* value, const char* name) {
    const std::string s(value);
    size_t used = 0;
    const unsigned long long n = s.empty() || s[0] == '-' ? 0 : std::stoull(s, &used);
    if (used != s.size() || n == 0) throw std::runtime_error(std::string(name) + " is not a positive whole number");
    return (size_t)n;
}

static int probe(int argc, const char* const* argv) {
    if (argc < 6 || argc > 8) {
        std::fprintf(stderr, "usage: llmx-route-probe MODEL IDS OUTPUT_PREFIX LAST UBATCH [DEVICES [STREAM_LAYERS]]\n"
                             "  writes OUTPUT_PREFIX.txt, the logits rows of the last LAST positions, and OUTPUT_PREFIX.routes, every routed step's picks and router scores\n"
                             "  with STREAM_LAYERS, layers separated by commas, OUTPUT_PREFIX.streams, the residual entering each of those layers\n");
        return 2;
    }
    try {
        const std::string prefix(argv[3]);
        const size_t last = count(argv[4], "LAST");
        infer::PlacementRequest request;
        request.names = backend::device_specs(argc > 6 ? argv[6] : "cpu");
        request.ubatch = (int)count(argv[5], "UBATCH");
        request.logit_rows = (size_t)request.ubatch;
        request.dtype = backend::Dtype::f32;
        std::ifstream input(std::filesystem::u8path(argv[2]));
        if (!input) throw std::runtime_error("cannot read token IDs");
        std::vector<uint32_t> ids;
        uint64_t value;
        while (input >> value) {
            if (value > UINT32_MAX) throw std::runtime_error("token ID exceeds uint32");
            ids.push_back((uint32_t)value);
        }
        if (!input.eof() || ids.empty()) throw std::runtime_error("expected whole token IDs");

        // Each record: the layer, rows, k and n_expert as int32 and uint32s, then rows * k ids and rows * n_expert scores, little-endian; a layer's records come in position order.
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "routes need IEEE binary32");
        std::ofstream routes(std::filesystem::u8path(prefix + ".routes"), std::ios::binary);
        routes.exceptions(std::ios::badbit | std::ios::failbit);
        infer::ModelOptions options;
        options.kv_k = options.kv_v = backend::KVType::f32;
        options.kv_k_given = options.kv_v_given = true;
        // Each record: the layer, rows and floats a row as int32 and uint32s, then rows of residual floats, little-endian; a layer's records come in position order.
        std::set<int> stream_layers;
        std::ofstream streams;
        if (argc > 7) {
            std::stringstream list(argv[7]);
            for (std::string item; std::getline(list, item, ',');) stream_layers.insert(item == "0" ? 0 : (int)count(item.c_str(), "STREAM_LAYERS"));
            streams.open(std::filesystem::u8path(prefix + ".streams"), std::ios::binary);
            streams.exceptions(std::ios::badbit | std::ios::failbit);
            options.streams = [&](int layer, size_t rows, size_t width, const float* x) {
                if (!stream_layers.count(layer)) return;
                const int32_t head[3] = {layer, (int32_t)rows, (int32_t)width};
                streams.write(reinterpret_cast<const char*>(head), sizeof(head));
                streams.write(reinterpret_cast<const char*>(x), (std::streamsize)(rows * width * sizeof(float)));
            };
        }
        options.routes = [&](int layer, size_t rows, size_t k, const uint32_t* picked, size_t n_expert, const float* scores) {
            const int32_t head[4] = {layer, (int32_t)rows, (int32_t)k, (int32_t)n_expert};
            routes.write(reinterpret_cast<const char*>(head), sizeof(head));
            routes.write(reinterpret_cast<const char*>(picked), (std::streamsize)(rows * k * sizeof(uint32_t)));
            routes.write(reinterpret_cast<const char*>(scores), (std::streamsize)(rows * n_expert * sizeof(float)));
        };
        const auto loaded = infer::load_model(argv[1], backend::make_backends(request.names), request, options, {}, infer::LoadMode::mapped);
        infer::Model& model = *loaded->model;
        const size_t vocab = model.n_vocab();
        for (uint32_t id : ids)
            if (id >= vocab) throw std::runtime_error("token ID outside vocabulary");
        std::FILE* text = std::fopen((prefix + ".txt").c_str(), "wb");
        if (!text) throw std::runtime_error("cannot write " + prefix + ".txt");
        std::fprintf(text, "tokens: %zu\n", ids.size());
        const size_t from = ids.size() - std::min(last, ids.size());
        // Ranked as `llmx logits` ranks them, a tie to the lower id, with the raw logits.
        model.score(ids, [&](size_t pos, const float* logits) {
            if (pos < from) return;
            std::fprintf(text, "%zu", pos);
            for (const auto& t : infer::top_logprobs(logits, vocab, 0.0, vocab)) std::fprintf(text, " %u %.6f", t.id, logits[t.id]);
            std::fprintf(text, "\n");
        });
        const bool written = std::fclose(text) == 0;
        routes.close();
        if (streams.is_open()) streams.close();
        if (!written) throw std::runtime_error("cannot write " + prefix + ".txt");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "llmx-route-probe: %s\n", e.what());
        return 1;
    }
}

#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) args.push_back(std::filesystem::path(argv[i]).u8string());
    std::vector<const char*> values;
    for (const auto& arg : args) values.push_back(arg.c_str());
    return probe(argc, values.data());
}
#else
int main(int argc, char** argv) { return probe(argc, argv); }
#endif
