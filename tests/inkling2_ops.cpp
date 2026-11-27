// inkling2-ops: the CPU forms of Inkling-2-Mini's layer ops against the math of HF's modeling_inkling2_loop.py in double (docs/INKLING2.md, section 5).
// The device forms run in backend-vulkan and backend-rocm through the same header.

#include <exception>
#include <iostream>

#include "inkling2_checks.hpp"

int main(int argc, char**) {
    if (argc != 1) {
        std::cerr << "usage: llmx-inkling2-ops-test\n";
        return 2;
    }
    try {
        ink2::l2_clip();
        ink2::query_mix();
        ink2::tanh_gate_mix();
        ink2::ring_conv();
        ink2::attention_rel_sink();
        ink2::attention_terms();
        ink2::softcaps();
        ink2::sigmoid_eps();
        ink2::expert_window();
        ink2::copy_rows();
        ink2::conv_strides();
        {
            backend::CpuBackend cpu;
            ink2::row_products(cpu);
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "inkling2-ops: " << ink2::checked << " checks passed" << std::endl;
    return 0;
}
