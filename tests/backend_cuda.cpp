// backend-cuda: CUDA device 0 as the driver reports it, buffers, a refused allocation, the F32 matmul against the CPU, the F16 and int8 classes' Q8_0 and K-quant products by their twin's rule and row class, and the qwen35 layers' ops.
// `backend-cuda types` runs only the type checks, `backend-cuda qwen35` only the qwen35 ops and `backend-cuda bench` only the product timings; it exits 77 where there is no driver or device.
#include "backends/cuda/cuda_backend.hpp"
#include "backends/cuda/cuda_device.hpp"
#include "backends/devices.hpp"
#include "quant/types.hpp"
#include "qwen35_checks.hpp"
#include "row_classes.hpp"
#include "twin_checks.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
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

// A K-quant block's value i (0 to 255) as the int8 tile sums it, before Q6_K's offset of 32, and the scale and min of the 16 or 32 values holding it, as the kernel's decode forms them.
int k_value(uint32_t type, const uint8_t* b, size_t i, float& ds, float& dmin) {
    const float d = f16_to_f32(uint16_t(b[0] | (b[1] << 8))), dm = f16_to_f32(uint16_t(b[2] | (b[3] << 8)));
    if (type == quant::GGML_TYPE_Q6_K) {
        const size_t n = i / 128, part = i % 128 / 32, l = i % 32;
        const uint8_t lo = b[64 * n + l + 32 * (part & 1)], hi = b[128 + 32 * n + l];
        ds = f16_to_f32(uint16_t(b[208] | (b[209] << 8))) * float(int8_t(b[192 + 8 * n + l / 16 + 2 * part]));
        dmin = 0.0f;
        return ((part < 2 ? lo : lo >> 4) & 15) | (((hi >> (2 * part)) & 3) << 4);
    }
    const size_t sub = i / 32, j = i % 32;
    const uint8_t* sc = b + 4;
    const int s = sub < 4 ? sc[sub] & 63 : (sc[sub + 4] & 15) | ((sc[sub - 4] >> 6) << 4), m = sub < 4 ? sc[sub + 4] & 63 : (sc[sub + 4] >> 4) | ((sc[sub] >> 6) << 4);
    ds = d * float(s), dmin = dm * float(m);
    const bool q5 = type == quant::GGML_TYPE_Q5_K;
    const uint8_t q = b[(q5 ? 48 : 16) + 32 * (sub / 2) + j];
    return ((sub & 1) ? q >> 4 : q & 15) | (q5 ? ((b[16 + j] >> sub) & 1) << 4 : 0);
}

// A 32-value block's value i of Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL or MXFP4 as the int8 tile sums it, before Q4_0's and Q5_0's offsets, with its block's scale and min as the kernel's decode forms them.
int block_value(uint32_t type, const uint8_t* b, size_t i, float& ds, float& dmin) {
    const auto half = [&](size_t at) { return f16_to_f32(uint16_t(b[at] | (b[at + 1] << 8))); };
    dmin = 0.0f;
    if (type == quant::GGML_TYPE_MXFP4) {
        const unsigned e = b[0];
        uint32_t bits = e <= 1 ? 0x00200000u << e : (e - 1) << 23;
        std::memcpy(&ds, &bits, 4);
        const int q = (i < 16 ? b[1 + i] : b[1 + i - 16] >> 4) & 15;
        static const int doubled[8] = {0, 1, 2, 3, 4, 6, 8, 12};
        return q & 8 ? -doubled[q & 7] : doubled[q & 7];
    }
    const bool minned = type == quant::GGML_TYPE_Q4_1 || type == quant::GGML_TYPE_Q5_1, five = type == quant::GGML_TYPE_Q5_0 || type == quant::GGML_TYPE_Q5_1;
    ds = half(0);
    if (minned) dmin = -half(2);
    const size_t qs = (minned ? 4 : 2) + (five ? 4 : 0);
    int q = (i < 16 ? b[qs + i] : b[qs + i - 16] >> 4) & 15;
    if (five) q |= int((uint32_t(b[qs - 4]) | uint32_t(b[qs - 3]) << 8 | uint32_t(b[qs - 2]) << 16 | uint32_t(b[qs - 1]) << 24) >> i & 1) << 4;
    return type == quant::GGML_TYPE_IQ4_NL ? quant::IQ4_NL_VALUES[q] : q;
}

// One output of the int8 prompt tile by the host in its order: each 32-value block, a K-quant's in halves where its scale changes every 16, as acc = fmaf(dx, term, acc), the term from exact integer sums.
float tile_fold(uint32_t type, const uint8_t* row, const int16_t* xq, const float* xd, size_t nin) {
    const quant::StorageType& st = *quant::storage_type(type);
    const bool kq = type == quant::GGML_TYPE_Q4_K || type == quant::GGML_TYPE_Q5_K || type == quant::GGML_TYPE_Q6_K;
    const size_t H = type == quant::GGML_TYPE_Q6_K ? 2 : 1, n = 32 / H;
    const int offset = type == quant::GGML_TYPE_Q6_K ? 32 : type == quant::GGML_TYPE_Q4_0 ? 8 : type == quant::GGML_TYPE_Q5_0 ? 16 : 0;
    float acc = 0.0f;
    for (size_t k = 0; k < nin / 32; ++k) {
        const uint8_t* blk = row + k * 32 / st.block_size * st.type_size;
        if (type == quant::GGML_TYPE_Q8_0) {
            int isum = 0;
            for (size_t j = 0; j < 32; ++j) isum += int(int8_t(blk[2 + j])) * xq[k * 32 + j];
            acc = std::fmaf(f16_to_f32(uint16_t(blk[0] | (blk[1] << 8))) * xd[k], float(isum), acc);
            continue;
        }
        for (size_t h = 0; h < H; ++h) {
            int isum = 0, xsum = 0;
            float ds = 0.0f, dmin = 0.0f;
            for (size_t j = h * n; j < (h + 1) * n; ++j) {
                const int v = kq ? k_value(type, blk, k * 32 % st.block_size + j, ds, dmin) : block_value(type, blk, j, ds, dmin);
                isum += v * xq[k * 32 + j];
                xsum += xq[k * 32 + j];
            }
            const float term = offset ? ds * float(isum - offset * xsum) : std::fmaf(ds, float(isum), -(dmin * float(xsum)));
            acc = std::fmaf(xd[k], term, acc);
        }
    }
    return acc;
}

// The int8 prompt tiles against the host's fold of their order (docs/STATUS.md, CUDA int8 tiles of every type on the tensor cores): the sums are exact, so an unsplit tile matches the fold bit for bit.
void imma_tile(Backend& g, const backend::DeviceCaps& caps) {
    struct Shape { size_t nin, nout, rows; };
    std::vector<Shape> shapes = {{1024, 3072, 64}, {1024, 3072, 200}, {2048, 1000, 512}};
    // Those split on every card run so far, so one shape takes the fewest outputs at which 64 rows of 1024 do not split on this device's units (split_blocks).
    // A call without row runs is one sequence of the prompt class, which splits for one column tile (split_tiles_of of extent 2), whatever rows it holds.
    const backend::DeviceProfile prof = backend::profile_for(caps);
    size_t whole = 64;
    while (whole < 65536 && backend::split_blocks(prof, caps, (whole + 63) / 64, 1024 / 32) != 1024 / 32) whole += 64;
    shapes.push_back({1024, whole, 64});
    for (uint32_t type : {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_IQ4_NL,
                          quant::GGML_TYPE_MXFP4, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K}) {
        std::string tname = quant::storage_type(type)->name;
        for (char& ch : tname) ch = char(std::tolower(ch));
        const quant::StorageType& st = *quant::storage_type(type);
        bool unsplit = false;
        for (Shape s : shapes) {
            const std::vector<uint8_t> w = type == quant::GGML_TYPE_Q8_0 ? twin_checks::q8_matrix(s.nin, s.nout, uint32_t(s.nout + s.rows)) : twin_checks::typed_matrix(type, s.nin, s.nout, uint32_t(s.nout + s.rows));
            const std::vector<float> x = row_classes::values(s.nin * s.rows, uint32_t(s.rows));
            BufferPtr wb = g.alloc(w.size(), Memory::device), xb = g.alloc(x.size() * sizeof(float), Memory::device), yb = g.alloc(s.nout * s.rows * sizeof(float), Memory::device);
            g.write(*wb, 0, w.data(), w.size());
            g.write(*xb, 0, x.data(), x.size() * sizeof(float));
            (void)cuda_take_launches(g);
            g.matmul({type, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, s.nin, s.nout, s.rows, {}, Dtype::int8);
            const std::vector<float> y = twin_checks::read(g, *yb, s.nout * s.rows);
            const auto ran = cuda_take_launches(g);
            std::string kernels;
            for (const auto& [name, count] : ran) kernels += " " + name;
            const bool tile = ran.count("llmx_" + tname + "_tile8") || ran.count("llmx_" + tname + "_tile8_tall"), split = ran.count("llmx_tile_reduce") > 0;
            const size_t nb = s.nin / 32, row_bytes = s.nin / st.block_size * st.type_size;
            std::vector<float> xd(s.rows * nb);
            std::vector<int16_t> xq(s.rows * s.nin);
            for (size_t r = 0; r < s.rows; ++r)
                for (size_t k = 0; k < nb; ++k) twin_checks::twin_rule(&x[r * s.nin + k * 32], xd[r * nb + k], &xq[r * s.nin + k * 32], 127.0f);
            size_t off = 0;
            for (size_t r = 0; r < s.rows; ++r)
                for (size_t o = 0; o < s.nout; ++o) {
                    const float acc = tile_fold(type, &w[o * row_bytes], &xq[r * s.nin], &xd[r * nb], s.nin);
                    if (std::memcmp(&acc, &y[r * s.nout + o], sizeof(float)) != 0) ++off;
                }
            char what[320];
            std::snprintf(what, sizeof(what), "int8 %s tile nin %zu nout %zu rows %zu: %zu of %zu outputs off the host fold%s, ran%s", st.name, s.nin, s.nout, s.rows, off, s.nout * s.rows,
                          split ? " (split: the parts' order is not folded, so only the tile's run is held)" : "", kernels.c_str());
            require(tile && (split || off == 0), what);
            unsplit = unsplit || (tile && !split && off == 0);
        }
        require(unsplit, std::string("an int8 ") + st.name + " tile call that did not split matched the host fold bit for bit");
    }
}

// Product timings at Qwen3-0.6B's and Qwen3.6-27B's widths, f16 against int8, decode and prompt row counts: `backend-cuda bench` prints one line a cell with the median of its runs and the kernels it ran.
void bench(Backend& g) {
    struct Shape { const char* model; size_t nin, nout; };
    const Shape shapes[] = {{"0.6B qkv", 1024, 4096}, {"0.6B gate_up", 1024, 6144}, {"0.6B down", 3072, 1024}, {"0.6B lm_head", 1024, 151936},
                            {"27B qkv", 5120, 10240}, {"27B gate_up", 5120, 34816}, {"27B down", 17408, 5120}};
    for (uint32_t type : {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q6_K})
        for (const Shape& sh : shapes)
            for (size_t rows : {size_t(1), size_t(4), size_t(512)}) {
                const std::vector<uint8_t> w = type == quant::GGML_TYPE_Q8_0 ? twin_checks::q8_matrix(sh.nin, sh.nout, 5) : twin_checks::typed_matrix(type, sh.nin, sh.nout, 5);
                const std::vector<float> x = row_classes::values(sh.nin * rows, 9);
                BufferPtr wb = g.alloc(w.size(), Memory::device), xb = g.alloc(x.size() * sizeof(float), Memory::device), yb = g.alloc(sh.nout * rows * sizeof(float), Memory::device);
                g.write(*wb, 0, w.data(), w.size());
                g.write(*xb, 0, x.data(), x.size() * sizeof(float));
                for (Dtype dt : {Dtype::f16, Dtype::int8}) {
                    const auto run = [&] { g.matmul({type, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, sh.nin, sh.nout, rows, {}, dt); };
                    run();
                    g.sync();
                    (void)cuda_take_launches(g);
                    std::vector<double> us;
                    for (int rep = 0; rep < 7; ++rep) {
                        const int n = rows == 512 ? 5 : 50;
                        const auto t0 = std::chrono::steady_clock::now();
                        for (int i = 0; i < n; ++i) run();
                        g.sync();
                        us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n);
                    }
                    std::sort(us.begin(), us.end());
                    std::string kernels;
                    for (const auto& [name, count] : cuda_take_launches(g)) kernels += " " + name;
                    const double ops = 2.0 * double(sh.nin) * double(sh.nout) * double(rows);
                    std::printf("BENCH %s %s %s rows %zu: %.1f us, %.2f GB/s weights, %.2f TOPS,%s\n", quant::storage_type(type)->name, sh.model, dt == Dtype::int8 ? "int8" : "f16", rows, us[3],
                                double(w.size()) / us[3] * 1e-3, ops / us[3] * 1e-6, kernels.c_str());
                }
            }
}

// The F16 and int8 classes' types (docs/CUDA.md, steps 3 and 4) through the checks backend-rocm shares (twin_checks.hpp, row_classes.hpp): exact rows, twin products against the CPU, row-class identity.
void types(const CudaDevice& d, Backend& g, Backend& cpu) {
    const std::vector<uint32_t> quants = {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K};
    guarded("exact_rows", [&] { twin_checks::exact_rows(g, {quant::GGML_TYPE_F32, quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K}, cuda_take_launches, require); });
    guarded("twin_products", [&] { twin_checks::twin_products(g, d.caps, Dtype::f16, require); });
    guarded("twin_products int8", [&] { twin_checks::twin_products(g, d.caps, Dtype::int8, require); });
    guarded("imma_tile", [&] { imma_tile(g, d.caps); });
    guarded("twin_types", [&] {
        twin_checks::twin_types(g, cpu, {quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K}, {Dtype::f16, Dtype::int8}, d.caps.subgroup_size, cuda_take_launches, require);
    });
    guarded("row classes", [&] { require(row_classes::check(g, quants, twin_checks::typed_matrix, {Dtype::f32, Dtype::f16, Dtype::int8}, false) > 0, "every pair of extents of one class gives the same bits"); });
}

// The qwen35 layers' ops (docs/CUDA.md, step 5) by the checks backend-rocm shares: against the CPU and bit for bit across views and passes, the drafter ops, and the fused step against the base's.
// The delta rule and the fused step sum a token's eight lanes in a branch other lanes of the warp skip, so these are the checks a full-warp exchange there fails or hangs.
void qwen35(const CudaDevice& d, Backend& g) {
    guarded("qwen35 ops", [&] {
        q35::Pair p(g);
        size_t runs = 0;
        (void)cuda_take_launches(g);
        size_t values = q35::check_ops(p, runs);
        const auto ran = cuda_take_launches(g);
        require(ran.count("llmx_gated_delta_rule") && ran.count("llmx_gated_delta_rule_short"), "the qwen35 checks run both delta rule builds");
        values += q35::check_drafter_ops(p, d.profile.argmax_slice, twin_checks::typed_matrix);
        require(values > 0, "qwen35 ops: " + std::to_string(values) + " values against the CPU, " + std::to_string(runs) + " device runs bit for bit across views, orders and passes");
    });
    guarded("delta_step", [&] { twin_checks::delta_step_bits(g, cuda_take_launches, require); });
}

} // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    const bool only_types = mode == "types";
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
    if (mode == "bench") {
        bench(*g);
        return 0;
    }
    if (mode == "qwen35") {
        qwen35(d, *g);
        return failures ? 1 : 0;
    }
    if (!only_types) basics(d, *g, *cpu);
    types(d, *g, *cpu);
    if (!only_types) qwen35(d, *g);
    return failures ? 1 : 0;
}
