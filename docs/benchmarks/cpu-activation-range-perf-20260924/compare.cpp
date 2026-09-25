#include <algorithm>
#include <chrono>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>
#ifdef LLMX_COMPARE_REFERENCE
#include "llama.h"
#else
#include "config.hpp"
#include "model/arch_qwen.hpp"
#endif
using Clock = std::chrono::steady_clock;
int main(int argc, char ** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--version") {
#ifdef LLMX_COMPARE_REFERENCE
        std::cout << "mx-llama.cpp 5542318e748c harness-f16" << std::endl;
#else
        std::cout << "llmx " << LLMX_VERSION_STRING << " harness-f16" << std::endl;
#endif
        return 0;
    }
    int threads = 6;
    if (argc == 5 && std::string_view(argv[3]) == "--threads") {
        const std::string_view value(argv[4]);
        const auto result = std::from_chars(value.data(), value.data() + value.size(), threads);
        if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
            threads < 1 || threads > 64) {
            std::cerr << "--threads must be an integer from 1 to 64\n";
            return 2;
        }
    } else if (argc != 3) {
        std::cerr << "usage: compare-cpu MODEL.gguf TOKEN_IDS.txt [--threads N]\n"
                  << "  --threads: 1-64, default 6 (both prefill and decode)\n";
        return 2;
    }
    std::ifstream input(argv[2]);
    std::vector<uint32_t> ids;
    uint32_t id;
    while (input >> id) ids.push_back(id);
    if (ids.size() != 247) throw std::runtime_error("expected 247 pinned HF tokens");
    const size_t np = ids.size() - 32;
    std::vector<uint32_t> prompt(ids.begin(), ids.begin() + np);
#ifdef LLMX_COMPARE_REFERENCE
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = (uint32_t)np;
    cp.n_ubatch = 128;
    cp.n_threads = cp.n_threads_batch = threads;
    cp.type_k = cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv = cp.op_offload = false;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 4;
    const size_t nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> ref_ids(ids.begin(), ids.end());
#else
    auto weights = gguf::read_gguf(argv[1]);
    infer::ModelOptions options;
    options.kv_k = options.kv_v = backend::KVType::f16;
    options.kv_tokens = 512;
    infer::Model model(weights, backend::make_cpu_backend(), options);
    model.set_threads(threads);
    model.set_ubatch(128);
#endif
    for (int run = 0; run < 2; ++run) {
#ifdef LLMX_COMPARE_REFERENCE
        llama_memory_clear(llama_get_memory(ctx), false);
#else
        model.reset();
#endif
        const auto start = Clock::now();
#ifdef LLMX_COMPARE_REFERENCE
        if (llama_decode(ctx, llama_batch_get_one(ref_ids.data(), (int32_t)np))) return 5;
        llama_synchronize(ctx);
        const float* first = llama_get_logits_ith(ctx, -1);
        auto logits = std::vector<float>(first, first + nv);
#else
        auto logits = model.prefill(prompt);
#endif
        const auto pp_end = Clock::now();
        for (size_t i = np; i < ids.size(); ++i) {
#ifdef LLMX_COMPARE_REFERENCE
            if (llama_decode(ctx, llama_batch_get_one(&ref_ids[i], 1))) return 6;
            llama_synchronize(ctx);
            const float* next = llama_get_logits_ith(ctx, -1);
            logits = std::vector<float>(next, next + nv);
#else
            logits = model.step(ids[i]);
#endif
        }
        const auto tg_end = Clock::now();
#ifdef LLMX_COMPARE_REFERENCE
        const float * values = logits.data();
#else
        const float * values = logits.data();
        const size_t nv = logits.size();
#endif
        double sum = 0;
        for (size_t i = 0; i < nv; ++i) {
            if (!std::isfinite(values[i])) return 7;
            sum += values[i];
        }
        std::cout << std::setprecision(10) << "{\"run\":" << run
                  << ",\"threads\":" << threads
                  << ",\"pp_tokens\":" << np << ",\"tg_tokens\":32,\"pp_ms\":"
                  << std::chrono::duration<double,std::milli>(pp_end-start).count()
                  << ",\"tg_ms\":" << std::chrono::duration<double,std::milli>(tg_end-pp_end).count()
                  << ",\"top1\":" << (std::max_element(values,values+nv)-values)
                  << ",\"logit_sum\":" << sum << "}" << std::endl;
    }
#ifdef LLMX_COMPARE_REFERENCE
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
#endif
}
