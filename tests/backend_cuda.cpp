// backend-cuda: CUDA device 0 as the driver reports it, buffers, a refused allocation named, the F32 matmul against the CPU backend, and the F16 class's Q8_0, Q4_K, Q5_K and Q6_K products by their twin's rule and row class.
// `backend-cuda types` runs only the type checks; it exits 77 where there is no driver or device.
#include "backends/cuda/cuda_backend.hpp"
#include "backends/cuda/cuda_device.hpp"
#include "backends/devices.hpp"
#include "quant/types.hpp"
#include "row_classes.hpp"
#include "twin_checks.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <random>
#include <string>
#include <vector>

using namespace backend;

namespace {

int failures = 0;
void require(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// A check that throws is a failure, and the checks after it still run.
void guarded(const std::string& name, const std::function<void()>& check) {
    try {
        check();
    } catch (const std::exception& e) {
        require(false, name + " threw: " + e.what());
    }
}

// Identity, a round trip, a refused allocation and the F32 matmul.
void basics(const CudaDevice& d, Backend& g, Backend& cpu) {
    require(g.identity().rfind("cuda:0 ", 0) == 0 && g.identity().find(d.pci_bus) != std::string::npos, "identity names cuda:0 and its PCI address: " + g.identity());

    // 16 MiB to the device and back through its stream.
    {
        const size_t n = size_t(4) << 20;
        std::vector<float> in(n), out(n);
        for (size_t i = 0; i < n; ++i) in[i] = (float)(i % 9973) * 0.25f - 11.0f;
        BufferPtr b = g.alloc(n * sizeof(float), Memory::device);
        g.write(*b, 0, in.data(), n * sizeof(float));
        g.read(*b, 0, out.data(), n * sizeof(float));
        require(std::memcmp(in.data(), out.data(), n * sizeof(float)) == 0, "16 MiB round trip host to device to host");
    }
    {
        bool named = false;
        try {
            BufferPtr huge = g.alloc(d.total_memory * 4, Memory::device);
        } catch (const std::exception& e) {
            named = std::string(e.what()).rfind("cuda: ", 0) == 0;
            std::printf("  refused: %s\n", e.what());
        }
        require(named, "an allocation beyond the device is refused with the backend's name");
    }

    // F32 matmul at Qwen3-0.6B's widths and an odd inner width, at decode and prompt row counts, against the CPU's F32 sums: each element within 4 * nin * eps of the sum of |w * x|.
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    struct Shape { size_t nin, nout, rows; };
    for (Shape s : {Shape{1024, 3072, 1}, Shape{1024, 3072, 4}, Shape{1024, 3072, 64}, Shape{3072, 1024, 512}, Shape{1000, 77, 3}}) {
        std::vector<float> w(s.nin * s.nout), x(s.nin * s.rows), want(s.nout * s.rows), got(s.nout * s.rows);
        for (float& v : w) v = u(rng);
        for (float& v : x) v = u(rng);
        auto run = [&](Backend& b, std::vector<float>& y) {
            BufferPtr wb = b.alloc(w.size() * sizeof(float), Memory::device), xb = b.alloc(x.size() * sizeof(float), Memory::device), yb = b.alloc(y.size() * sizeof(float), Memory::device);
            b.write(*wb, 0, w.data(), w.size() * sizeof(float));
            b.write(*xb, 0, x.data(), x.size() * sizeof(float));
            b.matmul({quant::GGML_TYPE_F32, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, s.nin, s.nout, s.rows, {}, Dtype::f32);
            b.read(*yb, 0, y.data(), y.size() * sizeof(float));
        };
        (void)cuda_take_launches(g);
        run(cpu, want);
        run(g, got);
        std::string kernels;
        for (const auto& [name, count] : cuda_take_launches(g)) kernels += " " + name + "x" + std::to_string(count);
        double worst = 0.0;
        for (size_t r = 0; r < s.rows; ++r)
            for (size_t o = 0; o < s.nout; ++o) {
                double mag = 0.0;
                for (size_t i = 0; i < s.nin; ++i) mag += std::fabs((double)w[o * s.nin + i] * x[r * s.nin + i]);
                const double bound = 4.0 * (double)s.nin * 1.1920929e-7 * mag + 1e-30;
                worst = std::max(worst, std::fabs((double)got[r * s.nout + o] - want[r * s.nout + o]) / bound);
            }
        char what[256];
        std::snprintf(what, sizeof(what), "F32 matmul nin %zu nout %zu rows %zu within the F32 bound, worst %.3f of it, ran%s", s.nin, s.nout, s.rows, worst, kernels.c_str());
        require(worst <= 1.0 && !kernels.empty(), what);
    }
}

// The F16 class's types (docs/CUDA.md, step 3) through the checks backend-rocm shares (twin_checks.hpp, row_classes.hpp): the exact rows, the Q8_0 twin products, the K-quants' twin products against the CPU, and row-class identity.
void types(const CudaDevice& d, Backend& g, Backend& cpu) {
    const std::vector<uint32_t> quants = {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K};
    guarded("exact_rows", [&] { twin_checks::exact_rows(g, {quant::GGML_TYPE_F32, quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K}, cuda_take_launches, require); });
    guarded("twin_products", [&] { twin_checks::twin_products(g, d.caps, Dtype::f16, require); });
    guarded("twin_types", [&] {
        twin_checks::twin_types(g, cpu, {quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K}, {Dtype::f16}, d.caps.subgroup_size, cuda_take_launches, require);
    });
    guarded("row classes", [&] { require(row_classes::check(g, quants, twin_checks::typed_matrix, {Dtype::f32, Dtype::f16}, false) > 0, "every pair of extents of one class gives the same bits"); });
}

} // namespace

int main(int argc, char** argv) {
    const bool only_types = argc > 1 && std::string(argv[1]) == "types";
    CudaDevice d;
    try {
        d = cuda_device(0);
    } catch (const CudaUnavailable& e) {
        std::printf("backend-cuda: skipped, %s\n", e.what());
        return 77;
    }
    std::printf("device 0: %s, warp %u, %u multiprocessors, %zu MiB, bus %s, %s\n", d.caps.device.c_str(), d.caps.subgroup_size, d.caps.compute_units,
                d.total_memory >> 20, d.pci_bus.c_str(), d.caps.driver.c_str());
    require(d.caps.subgroup_size == 32, "the warp is 32 lanes");
    require(std::stoi(d.arch.substr(3)) >= 80, "the device is sm_80 or newer: " + d.arch);
    require(d.caps.shared_bytes >= 48u << 10 && d.caps.compute_units > 0 && d.total_memory > 0, "the device reports its shared memory, units and memory");

    BackendPtr g = make_backend("cuda:0");
    BackendPtr cpu = make_backend("cpu");
    if (!only_types) basics(d, *g, *cpu);
    types(d, *g, *cpu);
    return failures ? 1 : 0;
}
