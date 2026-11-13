// cuda-driver: the driver call table (cuda_runtime.hpp) against the toolkit's stub driver and a library without the calls, then the installed driver if any.
#include "backends/cuda/cuda_runtime.hpp"

#include <cstdio>
#include <cstring>
#include <string>

using namespace backend;

// The symbol each entry of the table loads, in table order, as cuda.h maps its name.
const char* const kCuSymbols[] = {
#define LLMX_CU_NAME(name) LLMX_CU_SYMBOL(name),
    LLMX_CU_FUNCTIONS(LLMX_CU_NAME)
#undef LLMX_CU_NAME
};

int main(int argc, char** argv) {
    int failures = 0;
    auto require = [&](bool ok, const std::string& what) {
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
        if (!ok) ++failures;
    };

    // The table loads the versioned symbols the header maps its names to, never a call's first ABI.
    bool v2 = false;
    for (const char* s : kCuSymbols) v2 = v2 || std::strcmp(s, "cuMemAlloc_v2") == 0;
    require(v2, "cuMemAlloc loads as cuMemAlloc_v2");

    // The stub is the toolkit's, so a machine with only the driver runs the rest without it.
    void* stub = argc > 1 ? dlopen(argv[1], RTLD_NOW | RTLD_LOCAL) : nullptr;
    if (argc > 1) require(stub != nullptr, std::string("the stub driver opens: ") + argv[1]);
    if (stub) {
        std::string missing;
        for (const char* s : kCuSymbols)
            if (!dlsym(stub, s)) missing += std::string(" ") + s;
        require(missing.empty(), "the stub exports every symbol of the table" + missing);
        std::string why;
        try {
            (void)cu_load(stub);
        } catch (const CudaUnavailable& e) {
            why = e.what();
        }
        require(why == "cuda: the driver did not initialise", "a driver that does not initialise is unavailable once every call resolved: " + why);
    }

    // A library without the calls is unavailable, naming the first it lacks.
    std::string why;
    try {
        (void)cu_load(dlopen(nullptr, RTLD_NOW));
    } catch (const CudaUnavailable& e) {
        why = e.what();
    }
    require(why == "cuda: the driver has no cuInit", "a library without the calls names the first: " + why);

    // The installed driver: opened where there is one, else unavailable, never another failure.
    try {
        int version = 0;
        cu_check(cu().cuDriverGetVersion(&version), "cuDriverGetVersion");
        require(version >= 12080, "the installed driver serves CUDA 12.8 or newer: " + std::to_string(version));
    } catch (const CudaUnavailable& e) {
        std::printf("no driver here: %s\n", e.what());
    }
    return failures ? 1 : 0;
}
