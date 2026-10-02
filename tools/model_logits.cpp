// Full-vocabulary captures for the quantization plan's device-versus-CPU gate: every prompt position in batched and per-token execution, then 64 argmax steps after one prefill.
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>
#include "backends/devices.hpp"
#include "core/json.hpp"
#include "core/list.hpp"
#include "inference/load.hpp"

static int integer(const char* value, int minimum, const char* name) {
    const std::string s(value);
    int n = 0;
    const auto parsed = std::from_chars(s.data(), s.data() + s.size(), n);
    if (parsed.ec != std::errc{} || parsed.ptr != s.data() + s.size() || n < minimum)
        throw std::runtime_error(std::string(name) + " is not an integer in range");
    return n;
}

static int capture(int argc, const char* const* argv) {
    if (argc != 8 && argc != 9) {
        std::cerr << "usage: llmx-model-logits MODEL IDS OUTPUT_PREFIX DEVICE CACHE UBATCH CPU_EXPERTS [SHARES]\n"
                  << "  CACHE: f16 or f32; UBATCH: positive; CPU_EXPERTS: -1 for all, otherwise a count; six CPU threads\n"
                  << "  SHARES: comma-separated whole-number layer proportions, one per device; omitted uses automatic placement\n";
        return 2;
    }
    try {
        const std::string cache(argv[5]), prefix(argv[3]);
        if (cache != "f16" && cache != "f32") throw std::runtime_error("CACHE must be f16 or f32");
        infer::PlacementRequest request;
        request.names = backend::device_specs(argv[4]);
        request.ubatch = integer(argv[6], 1, "UBATCH");
        request.cpu_moe = integer(argv[7], -1, "CPU_EXPERTS");
        if (argc == 9)
            for (const auto& share : core::comma_list(argv[8])) request.shares.push_back(integer(share.c_str(), 0, "SHARES"));
        std::ifstream input(std::filesystem::u8path(argv[2]));
        if (!input) throw std::runtime_error("cannot read token IDs");
        std::vector<uint32_t> ids;
        uint64_t value;
        while (input >> value) {
            if (value > UINT32_MAX) throw std::runtime_error("token ID exceeds uint32");
            ids.push_back((uint32_t)value);
        }
        if (!input.eof() || ids.size() < 2) throw std::runtime_error("expected at least two whole token IDs");
        infer::ModelOptions options;
        options.kv_k = options.kv_v = cache == "f16" ? backend::KVType::f16 : backend::KVType::f32;
        options.kv_tokens = ids.size() + 64;
        const auto loaded = infer::load_model(argv[1], backend::make_backends(request.names), request, options);
        auto& model = *loaded->model;
        model.set_threads(6);
        const size_t vocab = model.n_vocab();
        for (uint32_t id : ids) if (id >= vocab) throw std::runtime_error("token ID outside vocabulary");
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "captures need IEEE binary32");
        auto output = [&](const char* phase) {
            std::ofstream file(std::filesystem::u8path(prefix + "." + phase + ".bin"), std::ios::binary);
            file.exceptions(std::ios::badbit | std::ios::failbit);
            return file;
        };
        auto save = [&](std::ofstream& file, const float* row) {
            for (size_t i = 0; i < vocab; ++i) if (!std::isfinite(row[i])) throw std::runtime_error("nonfinite captured logit");
            file.write(reinterpret_cast<const char*>(row), (std::streamsize)(vocab * sizeof(float)));
        };
        auto batched = output("batched");
        size_t rows = 0;
        model.score(ids, [&](size_t pos, const float* row) {
            if (pos != rows++) throw std::runtime_error("score positions out of order");
            save(batched, row);
        });
        if (rows != ids.size()) throw std::runtime_error("score positions missing");
        batched.close();
        const auto batched_paths = model.take_matrix_paths();
        model.reset();
        auto decode = output("decode");
        for (uint32_t id : ids) save(decode, model.step((int)id).data());
        decode.close();
        const auto decode_paths = model.take_matrix_paths();
        model.reset();
        auto greedy = output("greedy");
        auto logits = model.prefill(ids);
        std::vector<uint32_t> reply;
        for (size_t i = 0; i < 64; ++i) {
            save(greedy, logits.data());
            const uint32_t next = (uint32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin());
            reply.push_back(next);
            if (i + 1 < 64) logits = model.step((int)next);
        }
        greedy.close();
        const auto greedy_paths = model.take_matrix_paths();
        const auto* arch = loaded->file.find("general.architecture");
        std::cout << "{\"version\":" << jmini::quote(LLMX_VERSION_STRING) << ",\"architecture\":" << jmini::quote(arch ? arch->s : "") << ",\"vocab\":" << vocab << ",\"tokens\":[";
        for (size_t i = 0; i < ids.size(); ++i) std::cout << (i ? "," : "") << ids[i];
        std::set<uint32_t> types;
        for (const auto& tensor : loaded->file.tensors) types.insert(tensor.type);
        std::cout << "],\"storage_types\":[";
        bool comma = false;
        for (uint32_t type : types) { std::cout << (comma ? "," : "") << type; comma = true; }
        std::cout << "],\"greedy\":[";
        for (size_t i = 0; i < reply.size(); ++i) std::cout << (i ? "," : "") << reply[i];
        std::cout << "],\"dtype\":" << jmini::quote(backend::dtype_name(loaded->dtype.effective)) << ",\"matrix_paths\":{";
        const auto paths = [](const char* phase, const std::vector<std::vector<std::string>>& devices) {
            std::cout << jmini::quote(phase) << ":[";
            for (size_t i = 0; i < devices.size(); ++i) {
                std::cout << (i ? "," : "") << "[";
                for (size_t j = 0; j < devices[i].size(); ++j) std::cout << (j ? "," : "") << jmini::quote(devices[i][j]);
                std::cout << "]";
            }
            std::cout << "]";
        };
        paths("batched", batched_paths); std::cout << ",";
        paths("decode", decode_paths); std::cout << ",";
        paths("greedy", greedy_paths);
        std::cout << "}}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model-logits: " << error.what() << '\n';
        return 1;
    }
}

#if defined(_WIN32)
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) args.push_back(std::filesystem::path(argv[i]).u8string());
    std::vector<const char*> values;
    for (const auto& arg : args) values.push_back(arg.c_str());
    return capture(argc, values.data());
}
#else
int main(int argc, char** argv) { return capture(argc, argv); }
#endif
