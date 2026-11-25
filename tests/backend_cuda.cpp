// backend-cuda: CUDA device 0, buffers, a refused allocation, the F32 matmul against the CPU, every weight type's products and row classes, the qwen35, routed and DeepSeek-V4.1 ops, attention tiles, and two-device groups and handoffs.
// Its modes `types`, `qwen35`, `moe`, `deepseek4`, `groups`, `bench` and `sums` run one part each (docs/TESTS.md); it exits 77 where there is no driver or device.
#include "attention_tiles.hpp"
#include "dspine_checks.hpp"
#include "backends/cuda/cuda_backend.hpp"
#include "backends/cuda/cuda_device.hpp"
#include "backends/devices.hpp"
#include "deepseek4_checks.hpp"
#include "group_sums.hpp"
#include "moe_checks.hpp"
#include "quant/types.hpp"
#include "qwen35_checks.hpp"
#include "row_classes.hpp"
#include "twin_checks.hpp"
#include "vision_checks.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <memory>
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
    require(g.identity() == "cuda " + d.caps.device + "; " + d.caps.driver, "identity names the backend, the device and its driver, not the ordinal or PCI address: " + g.identity());

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
float tile_fold(uint32_t type, const uint8_t* row, const int16_t* xq, const float* xd, size_t k0, size_t k1) {
    const quant::StorageType& st = *quant::storage_type(type);
    const bool kq = type == quant::GGML_TYPE_Q4_K || type == quant::GGML_TYPE_Q5_K || type == quant::GGML_TYPE_Q6_K;
    const size_t H = type == quant::GGML_TYPE_Q6_K ? 2 : 1, n = 32 / H;
    const int offset = type == quant::GGML_TYPE_Q6_K ? 32 : type == quant::GGML_TYPE_Q4_0 ? 8 : type == quant::GGML_TYPE_Q5_0 ? 16 : 0;
    float acc = 0.0f;
    for (size_t k = k0; k < k1; ++k) {
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

// The int8 prompt tiles against the host's fold of their order (docs/STATUS.md, CUDA int8 tiles of every type on the tensor cores): the sums are exact, so a tile matches the fold bit for bit, a split one part by part.
void imma_tile(Backend& g, const backend::DeviceCaps& caps) {
    struct Shape { size_t nin, nout, rows; };
    std::vector<Shape> shapes = {{1024, 3072, 64}, {1024, 3072, 200}, {2048, 1000, 512}};
    // Those split on every card run so far, so one shape takes the fewest outputs at which 64 rows of 1024 do not split on this device's units (split_blocks).
    // A call without row runs is one sequence of the prompt class, which splits for one column tile (split_tiles_of of extent 2), whatever rows it holds.
    const backend::DeviceProfile prof = backend::profile_for(caps);
    size_t whole = 64;
    while (whole < 65536 && backend::split_blocks(prof, caps, (whole + 63) / 64, 1024 / 32) != 1024 / 32) whole += 64;
    shapes.push_back({1024, whole, 64});
    // Unsplit edges of both widths (docs/TESTS.md): 256 values are 8 blocks, too few to split, at 200 outputs for the 64-wide build and at the fewest outputs 128k - 56 that take the tall one, rows not a multiple of 64.
    const size_t edge_nin = std::max<size_t>(256, (prof.qtile_tall_from_8bit + 255) / 256 * 256);
    shapes.push_back({256, 200, 100});
    size_t tall = 0;
    for (size_t n = 72; n < 65536 && !tall; n += 128)
        if (backend::tile_rows_for(caps, prof, 64, 64, 128, n, 1, edge_nin, prof.qtile_tall_from_8bit, true) == 128 && backend::split_blocks(prof, caps, (n + 63) / 64, edge_nin / 32) == edge_nin / 32) tall = n;
    require(tall != 0, "an output count at which a 72-row call takes the tall tile unsplit");
    // At 72 rows a K-type takes its square build, at 40 its 64-row tall one.
    shapes.push_back({edge_nin, tall, 72});
    shapes.push_back({edge_nin, tall, 40});
    for (uint32_t type : {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_IQ4_NL,
                          quant::GGML_TYPE_MXFP4, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K}) {
        std::string tname = quant::storage_type(type)->name;
        for (char& ch : tname) ch = char(std::tolower(ch));
        const quant::StorageType& st = *quant::storage_type(type);
        bool unsplit = false, unsplit64 = false, unsplit_tall = false, unsplit_square = false, parted = false;
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
            const bool tile = ran.count("llmx_" + tname + "_tile8") || ran.count("llmx_" + tname + "_tile8_tall") || ran.count("llmx_" + tname + "_tile8_square"), split = ran.count("llmx_tile_reduce") > 0;
            const size_t nb = s.nin / 32, row_bytes = s.nin / st.block_size * st.type_size;
            std::vector<float> xd(s.rows * nb);
            std::vector<int16_t> xq(s.rows * s.nin);
            for (size_t r = 0; r < s.rows; ++r)
                for (size_t k = 0; k < nb; ++k) twin_checks::twin_rule(&x[r * s.nin + k * 32], xd[r * nb + k], &xq[r * s.nin + k * 32], 127.0f);
            // A split call's parts are the fold over kper blocks each, which llmx_tile_reduce adds from zero in part order; kper is the host's, over the 64-output grid of one column tile.
            const size_t kper = backend::split_blocks(prof, caps, (s.nout + 63) / 64, nb), parts = (nb + kper - 1) / kper;
            size_t off = 0;
            for (size_t r = 0; r < s.rows; ++r)
                for (size_t o = 0; o < s.nout; ++o) {
                    float acc = parts == 1 ? tile_fold(type, &w[o * row_bytes], &xq[r * s.nin], &xd[r * nb], 0, nb) : 0.0f;
                    for (size_t z = 0; parts > 1 && z < parts; ++z) acc += tile_fold(type, &w[o * row_bytes], &xq[r * s.nin], &xd[r * nb], z * kper, std::min(nb, (z + 1) * kper));
                    if (std::memcmp(&acc, &y[r * s.nout + o], sizeof(float)) != 0) ++off;
                }
            char what[320];
            std::snprintf(what, sizeof(what), "int8 %s tile nin %zu nout %zu rows %zu in %zu parts: %zu of %zu outputs off the host fold, ran%s", st.name, s.nin, s.nout, s.rows, parts, off,
                          s.nout * s.rows, kernels.c_str());
            require(tile && split == (parts > 1) && off == 0, what);
            unsplit = unsplit || (tile && !split && off == 0);
            unsplit64 = unsplit64 || (ran.count("llmx_" + tname + "_tile8") && !split && off == 0);
            unsplit_tall = unsplit_tall || (ran.count("llmx_" + tname + "_tile8_tall") && !split && off == 0);
            unsplit_square = unsplit_square || (ran.count("llmx_" + tname + "_tile8_square") && !split && off == 0);
            parted = parted || (tile && split);
        }
        require(unsplit, std::string("an int8 ") + st.name + " tile call that did not split matched the host fold bit for bit");
        require(parted, std::string("an int8 ") + st.name + " tile call split its inner dimension into parts");
        // The types with a tall build match unsplit at both widths, so the one tensor-core body is held at 64 and 128 outputs, and the K-types' at 64 and 128 rows.
        const bool square = type == quant::GGML_TYPE_Q4_K || type == quant::GGML_TYPE_Q5_K || type == quant::GGML_TYPE_Q6_K, has_tall = square || type == quant::GGML_TYPE_Q8_0;
        require(!has_tall || (unsplit64 && unsplit_tall), std::string("an int8 ") + st.name + " tile matched the host fold unsplit at 64 and at 128 outputs");
        require(!square || unsplit_square, std::string("an int8 ") + st.name + " square tile matched the host fold unsplit");
    }
}

// Product timings at Qwen3-0.6B's and Qwen3.6-27B's widths, f16 against int8, decode and prompt row counts: `backend-cuda bench` prints one line a cell with the median of its runs and the kernels it ran.
void bench(Backend& g) {
    struct Shape { const char* model; size_t nin, nout; };
    const Shape shapes[] = {{"0.6B qkv", 1024, 4096}, {"0.6B gate_up", 1024, 6144}, {"0.6B down", 3072, 1024}, {"0.6B lm_head", 1024, 151936},
                            {"27B qkv", 5120, 10240}, {"27B gate_up", 5120, 34816}, {"27B down", 17408, 5120}};
    for (uint32_t type : {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q6_K})
        for (const Shape& sh : shapes)
            for (size_t rows : {size_t(1), size_t(4), size_t(64), size_t(65), size_t(96), size_t(128), size_t(192), size_t(512)}) {
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
                        const int n = rows >= 64 ? 5 : 50;
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

// The weight types ROCm runs (docs/CUDA.md, steps 3, 4 and 8) through the checks backend-rocm shares (twin_checks.hpp, row_classes.hpp): exact rows, twin products against the CPU, row-class identity.
void types(const CudaDevice& d, Backend& g, Backend& cpu) {
    const std::vector<uint32_t> quants = {quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_MXFP4};
    guarded("exact_rows", [&] { twin_checks::exact_rows(g, cuda_profile(g), d.caps.subgroup_size, {quant::GGML_TYPE_F32, quant::GGML_TYPE_Q8_0, quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_BF16, quant::GGML_TYPE_F16, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_MXFP4, quant::GGML_TYPE_Q2_K, quant::GGML_TYPE_Q3_K, quant::GGML_TYPE_IQ4_XS, quant::GGML_TYPE_IQ2_XXS, quant::GGML_TYPE_IQ2_XS, quant::GGML_TYPE_IQ2_S, quant::GGML_TYPE_IQ3_XXS, quant::GGML_TYPE_IQ3_S, quant::GGML_TYPE_IQ1_S, quant::GGML_TYPE_IQ1_M, quant::GGML_TYPE_TQ1_0, quant::GGML_TYPE_TQ2_0, quant::GGML_TYPE_Q1_0, quant::GGML_TYPE_Q2_0}, cuda_take_launches, require); });
    guarded("nvfp4_scales", [&] { twin_checks::nvfp4_scales(g, cpu, cuda_take_launches, require); });
    guarded("fp8_grids", [&] { twin_checks::fp8_grids(g, cpu, cuda_take_launches, require); });
    guarded("twin_products", [&] { twin_checks::twin_products(g, d.caps, Dtype::f16, cuda_take_launches, require); });
    guarded("twin_products int8", [&] { twin_checks::twin_products(g, d.caps, Dtype::int8, cuda_take_launches, require); });
    for (const Dtype dt : {Dtype::f16, Dtype::int8}) {
        const std::string form = dt == Dtype::int8 ? " int8" : "";
        guarded("producer_twins" + form, [&] { twin_checks::producer_twins(g, dt, require); });
        guarded("stream_twins" + form, [&] { twin_checks::stream_twins(g, dt, cuda_take_launches, require); });
    }
    guarded("imma_tile", [&] { imma_tile(g, d.caps); });
    // An explicit bf16 is carried on the BF16 tile and F32 rows, so it is emulated and never in the auto list (docs/PRECISION.md, section 2).
    require(g.native_dtypes() == std::vector<Dtype>{Dtype::f16, Dtype::f32, Dtype::int8} && g.emulates_dtype(Dtype::bf16) && !g.emulates_dtype(Dtype::f16), "CUDA lists f16, f32 and int8 and emulates bf16");
    require(g.dtype_path(Dtype::bf16) == "bf16 (BF16 tile inputs), f32 (other products, decode columns, routers)", "CUDA's bf16 catalog names the BF16 tile and its F32 products: " + g.dtype_path(Dtype::bf16));
    twin_checks::Bits bits{d.caps, cuda_profile, cuda_take_launches, [](bool ok, const std::string& what) { if (!ok) require(false, what); }};
    const int before = failures;
    guarded("half_tile", [&] { bits.half_tile(g, true); });
    guarded("q8_decode_bits", [&] { bits.q8_decode_bits(g); });
    guarded("q8_columns_bits", [&] { bits.q8_columns_bits(g); });
    guarded("mxfp4_decode_bits", [&] { bits.mxfp4_decode_bits(g); });
    require(failures == before, "the F16 half tile, the Q8_0 and MXFP4 decode rows in the host's order and a decode call's columns each its column alone");
    guarded("twin_types", [&] {
        // A K-type's 8-bit prompt takes its tall build where tile_rows_for picks 128 outputs over the extent's column tiles (backend_core.hpp tile, split_tiles_of), the square one above 64 rows.
        const backend::DeviceProfile prof = backend::profile_for(d.caps);
        const auto tall = [&](uint32_t type, Dtype dt, size_t nin, size_t nout, size_t extent) {
            const bool k = type == quant::GGML_TYPE_Q4_K || type == quant::GGML_TYPE_Q5_K || type == quant::GGML_TYPE_Q6_K;
            if (!k || dt != Dtype::int8 || backend::tile_rows_for(d.caps, prof, 64, 64, 128, nout, (std::min<size_t>(extent, 512) + 63) / 64, nin, prof.qtile_tall_from_8bit, true) != 128) return std::string();
            return std::string(extent > 64 ? "_square" : "_tall");
        };
        twin_checks::twin_types(g, cpu, {quant::GGML_TYPE_Q4_K, quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_IQ4_NL, quant::GGML_TYPE_MXFP4,
                                         quant::GGML_TYPE_Q3_K, quant::GGML_TYPE_Q2_K, quant::GGML_TYPE_IQ4_XS, quant::GGML_TYPE_IQ2_XXS, quant::GGML_TYPE_IQ2_XS, quant::GGML_TYPE_IQ3_S, quant::GGML_TYPE_IQ2_S,
                                         quant::GGML_TYPE_IQ3_XXS, quant::GGML_TYPE_IQ1_S, quant::GGML_TYPE_IQ1_M,
                                         quant::GGML_TYPE_Q1_0, quant::GGML_TYPE_Q2_0},
                                {Dtype::f16, Dtype::int8}, d.caps.subgroup_size, cuda_take_launches, tall, require);
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
    guarded("delta_step", [&] { twin_checks::delta_step_bits(g, d.caps, cuda_take_launches, require); });
}

// The vision encoder's ops by the checks backend-rocm and backend-vulkan share, against the CPU within their bounds, each kernel the core adds for them reached.
void vision_ops(Backend& g) {
    guarded("vision ops", [&] {
        q35::Pair p(g);
        (void)cuda_take_launches(g);
        const size_t values = vision::check_ops(p);
        const auto ran = cuda_take_launches(g);
        for (const char* k : {"llmx_layer_norm_rows", "llmx_gelu", "llmx_add_bias", "llmx_mix_rows", "llmx_norm_rope_streams", "llmx_attention_chunk"})
            require(ran.count(k), std::string("the vision checks take ") + k);
        // The per-row body writes a block-major twin once a product has read one (twin_build), so either build counts.
        require(ran.count("llmx_attention_vec") || ran.count("llmx_attention_vec_major"), "the vision checks take llmx_attention_vec");
        require(values > 0, "vision ops: " + std::to_string(values) + " values against the CPU");
    });
}

// DeepSeek-V4.1's ops (docs/CUDA.md, step 8) against the CPU by the check backend-rocm shares (deepseek4_checks.hpp), with the ops ROCm holds to the CPU's bits held to them here too.
void deepseek4(Backend& g) {
    guarded("deepseek4 ops", [&] {
        q35::Pair p(g);
        (void)cuda_take_launches(g);
        const size_t values = ds4::check_ops(p, {"hc_collapse_norm", "hc_expand", "indexer_scores", "rope_forms", "mix_rows"});
        require(values > 0, "deepseek4 ops: " + std::to_string(values) + " values against the CPU");
        // Every CUDA target builds the 8-bit dot, so the FP4 keys 128 wide reach the integer index tile.
        require(cuda_take_launches(g).count("llmx_indexer_scores_int") == 1, "deepseek4's index checks take llmx_indexer_scores_int");
    });
    guarded("sparse_tile_pair", [&] { ds4::sparse_tile_pair(g, cuda_caps, cuda_take_launches, require); });
    guarded("sparse_grouped", [&] { ds4::sparse_grouped(g, cuda_caps, cuda_take_launches, require); });
    guarded("router", [&] { ds4::router(g, cuda_caps, cuda_take_launches, require); });
}

// The routed experts' ops (docs/CUDA.md, step 8) by the checks backend-rocm shares (moe_checks.hpp): routing against the CPU and every routed build's entries their expert's one-column product bit for bit.
// A routed build's lanes sum in branches that other lanes of the warp skip, so these are also the checks a full-warp exchange there fails or hangs.
void moe(const CudaDevice& d, Backend& g, Backend& cpu) {
    const int before = failures;
    moe_checks::Moe m{d.caps.subgroup_size, cuda_take_launches, cuda_route_only, [](bool ok, const std::string& what) { if (!ok) require(false, what); }};
    guarded("routing", [&] { m.routing(g, cpu); });
    guarded("experts", [&] { m.experts(g, cpu); });
    guarded("experts_grouped", [&] { m.experts_grouped(g); });
    guarded("experts_tile", [&] { m.experts_tile(g, cpu); });
    guarded("experts_quads", [&] { m.experts_quads(g); });
    guarded("routed_builds_witness", [&] { m.routed_builds_witness(g); });
    guarded("experts_add", [&] { m.experts_add(g); });
    require(failures == before, "moe: routing and every routed build against the CPU and their one-column products");
}

// Counts of the launches and copies a group sum makes, read through the driver's call table.
struct SumCalls {
    int launches = 0, copies = 0;
};
SumCalls sum_calls_seen;
decltype(CuFn::cuLaunchKernel) real_launch = nullptr;
decltype(CuFn::cuMemcpyAsync) real_copy = nullptr;
CUresult counted_launch(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared, CUstream s, void** args, void** extra) {
    ++sum_calls_seen.launches;
    return real_launch(f, gx, gy, gz, bx, by, bz, shared, s, args, extra);
}
CUresult counted_copy(CUdeviceptr dst, CUdeviceptr src, size_t bytes, CUstream s) {
    ++sum_calls_seen.copies;
    return real_copy(dst, src, bytes, s);
}
// While set, a stream wait the backend issues waits for the handoff check's gate word to reach 1 instead, so the sender's work cannot end before the check opens the gate.
CUdeviceptr handoff_gate = 0;
decltype(CuFn::cuStreamWaitValue64) real_wait = nullptr;
CUresult gated_wait(CUstream s, CUdeviceptr, cuuint64_t, unsigned flags) { return real_wait(s, handoff_gate, 1, flags); }
// Whether devices 0 and 1 reach each other's memory, as the driver says.
bool peer_pair(CuFn& f) {
    CUdevice a = 0, b = 0;
    int can = 0;
    return f.cuDeviceGet(&a, 0) == CUDA_SUCCESS && f.cuDeviceGet(&b, 1) == CUDA_SUCCESS && f.cuDeviceCanAccessPeer(&can, a, b) == CUDA_SUCCESS && can;
}
// Disables the peer access of every directed pair of the first `most` devices, as a fresh process has none.
void drop_peer_access(CuFn& f, size_t most) {
    CUcontext kept = nullptr;
    (void)f.cuCtxGetCurrent(&kept);
    std::vector<CUcontext> ctx(most, nullptr);
    for (size_t i = 0; i < most; ++i) {
        CUdevice dev = 0;
        if (f.cuDeviceGet(&dev, (int)i) != CUDA_SUCCESS || f.cuDevicePrimaryCtxRetain(&ctx[i], dev) != CUDA_SUCCESS) return;
    }
    for (size_t a = 0; a < most; ++a)
        for (size_t b = 0; b < most; ++b)
            if (a != b && f.cuCtxSetCurrent(ctx[a]) == CUDA_SUCCESS) (void)f.cuCtxDisablePeerAccess(ctx[b]);
    (void)f.cuCtxSetCurrent(kept);
}
CUresult no_peer(int* can, CUdevice, CUdevice) {
    *can = 0;
    return CUDA_SUCCESS;
}

std::vector<BackendPtr> cuda_devices(size_t most) {
    std::vector<BackendPtr> d;
    for (size_t i = 0; i < most; ++i) {
        try {
            d.push_back(make_cuda_backend((int)i));
        } catch (const CudaUnavailable&) {
            break;
        }
    }
    return d;
}
std::vector<Backend*> raw(const std::vector<BackendPtr>& d, size_t W) {
    std::vector<Backend*> r;
    for (size_t m = 0; m < W; ++m) r.push_back(d[m].get());
    return r;
}

// A fast sum is one launch a member and no copy, a conservative one a launch a member and a copy a peer.
void sum_calls(Collective& c, size_t W, bool fast, const std::string& what) {
    CuFn& f = cu();
    sum_calls_seen = {};
    f.cuLaunchKernel = counted_launch;
    f.cuMemcpyAsync = counted_copy;
    c.sum_in_place(1, 97);
    f.cuLaunchKernel = real_launch;
    f.cuMemcpyAsync = real_copy;
    require(sum_calls_seen.launches == (int)W && sum_calls_seen.copies == (fast ? 0 : (int)(W * (W - 1))),
            what + ": " + std::to_string(sum_calls_seen.launches) + " launches and " + std::to_string(sum_calls_seen.copies) + " copies, " + (fast ? "fast" : "conservative"));
}

// group_sums::check with one line for all its comparisons.
void oracle_sums(const std::vector<Backend*>& group, Collective& c, size_t width, const std::vector<size_t>& counts, uint32_t seed, const std::string& what) {
    int bad = 0;
    const size_t n = group_sums::check(group, c, width, counts, seed, [&](bool ok, const std::string&) { bad += ok ? 0 : 1; });
    require(bad == 0, what + ": " + std::to_string(n) + " sums of " + std::to_string(group.size()) + " members in member order, " + std::to_string(bad) + " members off the oracle");
}

// Picks that switch transport, form, block cap or all of them at every size, as a join's timing may keep them where neither transport wins at every size: the crossover the collective follows within one group.
const std::map<size_t, GpuSumPick> kCrossover = {
    {64, {GroupTransport::host_writes, GpuSumForm::shares}}, {256, {GroupTransport::peer_writes, GpuSumForm::chain}}, {512, {GroupTransport::host_writes, GpuSumForm::chain}},
    {1024, {GroupTransport::peer_writes, GpuSumForm::one_shot}}, {1027, {GroupTransport::host_writes, GpuSumForm::pulls}}, {4096, {GroupTransport::host_writes, GpuSumForm::one_shot, 2}},
    {4 * 4096, {GroupTransport::peer_writes, GpuSumForm::shares, 2}}, {8 * 4096, {GroupTransport::host_writes, GpuSumForm::chain, 1}}};

// The arms over the armed bits, NaN payloads, infinities and zeros, many sums so each parity half is armed again many times: peer writes where the devices allow them, host writes and pulls, each against the pulls' bits.
// Where the devices allow peer writes, one arm switches transport by size as kCrossover picks, two sums in flight across a switch.
void sentinel_sums(size_t W, bool peer) {
    static const uint32_t bits[] = {0x7f800001u, 0x7fc00001u, 0xff800001u, 0xffc00001u, 0x7fc00000u, 0x7f812345u, 0x7fd12345u, 0x7f800000u, 0xff800000u, 0x80000000u, 0x0u, 0x00000001u};
    const size_t cap = 8 * 1031, sizes[] = {1, 3, 5, 63, 64, 97, 255, 256, 1023, 1024, 1027, cap};
    const GpuSumForm forms[] = {GpuSumForm::one_shot, GpuSumForm::chain, GpuSumForm::wrap, GpuSumForm::shares, GpuSumForm::copies, GpuSumForm::pulls};
    std::vector<std::vector<BackendPtr>> sides;
    std::vector<std::unique_ptr<Collective>> arms;
    std::vector<std::string> names;
    if (peer) {
        sides.push_back(cuda_devices(W));
        arms.push_back(cuda_collective(raw(sides.back(), W), cap, 1, kCrossover));
        names.push_back("crossover");
    }
    for (const GroupTransport t : {GroupTransport::peer_writes, GroupTransport::host_writes, GroupTransport::pulls}) {
        if (t == GroupTransport::peer_writes && !peer) continue;
        for (const GpuSumForm form : forms) {
            if (t == GroupTransport::pulls ? form != GpuSumForm::one_shot : (form == GpuSumForm::pulls || form == GpuSumForm::copies) && t != GroupTransport::host_writes) continue;
            sides.push_back(cuda_devices(W));
            arms.push_back(cuda_collective(raw(sides.back(), W), cap, 1, form, t));
            names.push_back(std::string(t == GroupTransport::peer_writes ? "peer writes " : t == GroupTransport::host_writes ? "host writes " : "pulls ") + std::to_string((int)form));
        }
    }
    const size_t A = arms.size(), ref = A - 1;
    uint32_t seed = 9001 + (uint32_t)W;
    auto value = [&] {
        seed = seed * 1664525u + 1013904223u;
        const uint32_t r = seed >> 8;
        float v;
        if (r % 3) std::memcpy(&v, &bits[r % (sizeof(bits) / sizeof(bits[0]))], sizeof(v));
        else v = float(int(r % 4001) - 2000) / 16.0f;
        return v;
    };
    std::vector<int> bad(A, 0);
    for (size_t it = 0; it < 600; ++it) {
        const size_t n = sizes[(it * 7 + it / 12) % (sizeof(sizes) / sizeof(sizes[0]))];
        std::vector<std::vector<float>> partial(W, std::vector<float>(n));
        for (size_t m = 0; m < W; ++m)
            for (size_t i = 0; i < n; ++i) partial[m][i] = value();
        for (size_t a = 0; a < A; ++a) {
            for (size_t m = 0; m < W; ++m) {
                const Slice p = arms[a]->partial(m);
                sides[a][m]->write(*p.buffer, p.offset * sizeof(float), partial[m].data(), n * sizeof(float));
            }
            arms[a]->sum_in_place(n, 1);
        }
        for (size_t m = 0; m < W; ++m) {
            std::vector<std::vector<float>> got(A, std::vector<float>(n));
            for (size_t a = 0; a < A; ++a) {
                sides[a][m]->submit();
                const Slice p = arms[a]->partial(m);
                sides[a][m]->read(*p.buffer, p.offset * sizeof(float), got[a].data(), n * sizeof(float));
            }
            for (size_t a = 0; a < ref; ++a)
                for (size_t i = 0; i < n; ++i) {
                    uint32_t u, v;
                    std::memcpy(&u, &got[a][i], 4);
                    std::memcpy(&v, &got[ref][i], 4);
                    if (u != v && !(std::isnan(got[a][i]) && std::isnan(got[ref][i]))) {
                        ++bad[a];
                        break;
                    }
                }
        }
    }
    for (size_t a = 0; a < ref; ++a) require(bad[a] == 0, std::to_string(W) + " devices, 600 sums of armed bits, NaN payloads, infinities and zeros by " + names[a] + " give the pulls' bits");
}

// Device 0 sends rows through a host-visible buffer to every other device in three rounds, the second behind a sender's wait the check releases only after every take returned, each reader's copy waiting on the device.
void handoffs(std::vector<BackendPtr>& d) {
    const size_t n = 4099, bytes = n * sizeof(float);
    std::vector<Backend*> readers;
    for (size_t r = 1; r < d.size(); ++r) readers.push_back(d[r].get());
    const std::unique_ptr<Handoff> h = d[0]->handoff(readers);
    require(h != nullptr, "a CUDA handoff to CUDA readers is made");
    if (!h) return;
    const BufferPtr box = d[0]->alloc(bytes, Memory::host_visible), rows = d[0]->alloc(bytes), gate = d[0]->alloc(sizeof(uint64_t), Memory::host_visible);
    std::vector<BufferPtr> dst;
    for (size_t r = 1; r < d.size(); ++r) dst.push_back(d[r]->alloc(bytes));
    for (int round = 1; round <= 3; ++round) {
        std::vector<float> want(n);
        for (size_t i = 0; i < n; ++i) want[i] = float((int)((i * 7 + (size_t)round * 13) % 61) - 30) * 0.25f + (float)round;
        d[0]->write(*rows, 0, want.data(), bytes);
        d[0]->wait(d[0]->submit());
        // Round 2's begin waits for the gate in place of the readers' counts, so the check does not race the sender's work (docs/TESTS.md, backend-cuda).
        volatile uint64_t* open = (volatile uint64_t*)gate->host_ptr();
        *open = 0;
        CuFn& f = cu();
        if (round == 2) {
            handoff_gate = (CUdeviceptr)(uintptr_t)gate->host_ptr();
            real_wait = f.cuStreamWaitValue64;
            f.cuStreamWaitValue64 = gated_wait;
        }
        h->begin();
        d[0]->copy(*box, 0, *rows, 0, bytes);
        h->sent();
        if (round == 2) f.cuStreamWaitValue64 = real_wait;
        const Ticket sent = d[0]->submit();
        for (size_t r = 1; r < d.size(); ++r) {
            h->take(r - 1, *dst[r - 1], 0, *box, 0, bytes);
            h->taken(r - 1);
            d[r]->submit();
        }
        if (round == 2) require(!d[0]->done(sent), "a reader's take returns before the sender's slow work ends");
        *open = 1;
        bool same = true;
        for (size_t r = 1; r < d.size(); ++r) {
            d[r]->sync();
            std::vector<float> got(n);
            d[r]->read(*dst[r - 1], 0, got.data(), bytes);
            same = same && std::memcmp(got.data(), want.data(), bytes) == 0;
        }
        d[0]->sync();
        require(same, "handoff round " + std::to_string(round) + ": every reader reads the rows sent to it");
    }
}

// Tensor groups and handoffs over 2 up to kMaxTensorWidth CUDA devices where they open (docs/TESTS.md, backend-cuda): the probe's transport, host writes and pulls forced, each form, the member order, and the sentinel bits.
void groups() {
    CuFn& f = cu();
    real_launch = f.cuLaunchKernel;
    real_copy = f.cuMemcpyAsync;
    std::vector<BackendPtr> d = cuda_devices(kMaxTensorWidth);
    if (d.size() < 2) {
        std::printf("backend-cuda: one device, so no tensor group or handoff\n");
        return;
    }
    uint32_t seed = 101;
    for (size_t W = 2; W <= d.size(); ++W) {
        const std::vector<Backend*> group = raw(d, W);
        const std::string at = std::to_string(W) + " devices";
        guarded("join " + at, [&] {
            std::unique_ptr<Collective> c = d[0]->join(group, 64, 97);
            sum_calls(*c, W, true, "a joined group of " + at);
            oracle_sums(group, *c, 97, {1, 3, 64}, seed, "joined, " + at);
            std::unique_ptr<Collective> small = d[0]->join(group, 2, 97);
            oracle_sums(group, *small, 97, {1, 2}, seed + 50, "joined for fewer rows, " + at);
        });
        seed += 100;
        const bool peer = peer_pair(f);
        std::printf("backend-cuda: devices 0 and 1 %s each other's memory\n", peer ? "reach" : "do not reach");
        guarded("forced transports " + at, [&] {
            for (const GroupTransport t : {GroupTransport::peer_writes, GroupTransport::host_writes, GroupTransport::pulls}) {
                if (t == GroupTransport::peer_writes && !peer) continue;
                const std::string name = t == GroupTransport::peer_writes ? "peer writes" : t == GroupTransport::host_writes ? "host writes" : "pulls";
                for (const GpuSumForm form : {GpuSumForm::one_shot, GpuSumForm::chain, GpuSumForm::wrap, GpuSumForm::shares, GpuSumForm::copies, GpuSumForm::pulls}) {
                    if (t == GroupTransport::pulls ? form != GpuSumForm::one_shot : (form == GpuSumForm::pulls || form == GpuSumForm::copies) && t != GroupTransport::host_writes) continue;
                    const size_t cap = 17 * 4 * 4096 + 5;
                    std::unique_ptr<Collective> forced = cuda_collective(group, cap, 1, form, t);
                    oracle_sums(group, *forced, 1, {1, 4 * W + 1, 4 * 4096 + 1, cap}, seed, name + " in form " + std::to_string((int)form) + ", " + at);
                    seed += 100;
                }
            }
        });
        guarded("copies in chunks " + at, [&] {
            // Copies in more chunks than their rows fill, so some chunks of a share are empty, then in three, each in place and into.
            const size_t cap = 17 * 4 * 4096 + 5;
            for (const unsigned chunks : {16u, 3u}) {
                std::unique_ptr<Collective> c = cuda_collective(group, cap, 1, {{1, {GroupTransport::host_writes, GpuSumForm::copies, chunks}}});
                oracle_sums(group, *c, 1, {1, 4 * W + 1, 4 * 4096 + 1, cap}, seed, "copies in " + std::to_string(chunks) + " chunks, " + at);
                seed += 100;
            }
        });
        guarded("crossover " + at, [&] {
            if (!peer) return;
            // One collective whose sums switch transport and form at each size kCrossover names, each side of every switch against the oracle.
            const size_t cap = 17 * 4 * 4096 + 5;
            std::unique_ptr<Collective> c = cuda_collective(group, cap, 1, kCrossover);
            oracle_sums(group, *c, 1, {1, 70, 300, 600, 1025, 1027, 2000, 4097, 1, 4 * 4096 + 1, 300, 8 * 4096 + 1, 70, cap}, seed, "picks switching transport, " + at);
        });
        seed += 100;
        guarded("no peer access " + at, [&] {
            std::vector<BackendPtr> e = cuda_devices(W);
            const auto real = f.cuDeviceCanAccessPeer;
            f.cuDeviceCanAccessPeer = no_peer;
            std::unique_ptr<Collective> c = e[0]->join(raw(e, W), 64, 97);
            f.cuDeviceCanAccessPeer = real;
            sum_calls(*c, W, true, "a group without peer access, by host writes, " + at);
            oracle_sums(raw(e, W), *c, 97, {1, 3, 64}, seed, "without peer access, " + at);
            std::unique_ptr<Collective> pulls = cuda_collective(raw(e, W), 64, 97, GpuSumForm::one_shot, GroupTransport::pulls);
            sum_calls(*pulls, W, false, "a group forced to pulls, " + at);
        });
        seed += 100;
        guarded("member order " + at, [&] {
            // Fresh backends over pairs without access, as at a model's first load, so each order of a set probed once is timed through access enabled again.
            if (peer) drop_peer_access(f, W);
            std::vector<BackendPtr> e = cuda_devices(W);
            const std::vector<Backend*> fresh = raw(e, W);
            const std::vector<size_t> order = e[0]->member_order(fresh, W, 64, 97);
            std::vector<size_t> sorted = order;
            std::sort(sorted.begin(), sorted.end());
            bool permutes = sorted.size() == W;
            for (size_t k = 0; permutes && k < W; ++k) permutes = sorted[k] == k;
            require(permutes && e[0]->member_order(fresh, W, 64, 97) == order, "the member order of " + at + " is one order of its members, kept");
            std::vector<Backend*> ordered;
            for (size_t k : order) ordered.push_back(fresh[k]);
            std::unique_ptr<Collective> c = ordered[0]->join(ordered, 64, 97);
            oracle_sums(ordered, *c, 97, {1, 64}, seed, "in the member order, " + at);
        });
        seed += 100;
        guarded("sentinel sums " + at, [&] { sentinel_sums(W, peer); });
    }
    guarded("handoffs", [&] { handoffs(d); });
}

// The copy engine's rates between devices 0 and 1 and the host for the 27B's 512-row sum, one way and both at once, so the link's own rates stand beside the kernels' (`RATE` lines).
void sum_rates(CuFn& f, bool peer) {
    const size_t bytes = size_t(512) * 5120 * sizeof(float);
    CUcontext kept = nullptr, ctx[2] = {nullptr, nullptr};
    CUdeviceptr src[2] = {0, 0}, dst[2] = {0, 0};
    void* host[2] = {nullptr, nullptr};
    CUstream s[2] = {nullptr, nullptr};
    bool ok = f.cuCtxGetCurrent(&kept) == CUDA_SUCCESS;
    for (int i = 0; ok && i < 2; ++i) {
        CUdevice dev = 0;
        ok = f.cuDeviceGet(&dev, i) == CUDA_SUCCESS && f.cuDevicePrimaryCtxRetain(&ctx[i], dev) == CUDA_SUCCESS && f.cuCtxSetCurrent(ctx[i]) == CUDA_SUCCESS &&
             f.cuMemAlloc(&src[i], bytes) == CUDA_SUCCESS && f.cuMemAlloc(&dst[i], bytes) == CUDA_SUCCESS && f.cuMemHostAlloc(&host[i], bytes, CU_MEMHOSTALLOC_PORTABLE) == CUDA_SUCCESS &&
             f.cuStreamCreate(&s[i], CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS;
    }
    // kind 0 copies into device i from its peer, 1 from device i to the host, 2 from the host to device i.
    const auto issue = [&](int i, int kind) {
        bool done = f.cuCtxSetCurrent(ctx[i]) == CUDA_SUCCESS;
        if (kind == 0) done = done && f.cuMemcpyAsync(dst[i], src[1 - i], bytes, s[i]) == CUDA_SUCCESS;
        else if (kind == 1) done = done && f.cuMemcpyAsync((CUdeviceptr)host[i], src[i], bytes, s[i]) == CUDA_SUCCESS;
        else done = done && f.cuMemcpyAsync(dst[i], (CUdeviceptr)host[i], bytes, s[i]) == CUDA_SUCCESS;
        return done;
    };
    const auto rate = [&](const char* what, const std::vector<std::pair<int, int>>& ops) {
        std::vector<double> gb;
        for (int rep = 0; ok && rep < 6; ++rep) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < (rep ? 10 : 1); ++r)
                for (const auto& [i, k] : ops) ok = ok && issue(i, k);
            for (int i = 0; i < 2; ++i) ok = ok && f.cuStreamSynchronize(s[i]) == CUDA_SUCCESS;
            if (rep) gb.push_back(bytes / (std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 10) / 1e3);
        }
        if (!ok) return;
        std::sort(gb.begin(), gb.end());
        std::printf("RATE %s: %.2f GB/s a copy, median of 5 (%.2f to %.2f)\n", what, gb[2], gb[0], gb[4]);
    };
    if (peer) {
        rate("peer copy from device 1 to 0", {{0, 0}});
        rate("peer copy from device 0 to 1", {{1, 0}});
        rate("peer copies both ways at once", {{0, 0}, {1, 0}});
    }
    rate("device 0 to host", {{0, 1}});
    rate("host to device 0", {{0, 2}});
    rate("devices 0 and 1 to host at once", {{0, 1}, {1, 1}});
    rate("host to devices 0 and 1 at once", {{0, 2}, {1, 2}});
    if (!ok) std::printf("RATE copies failed\n");
    for (int i = 0; i < 2; ++i) {
        if (!ctx[i] || f.cuCtxSetCurrent(ctx[i]) != CUDA_SUCCESS) continue;
        if (s[i]) (void)f.cuStreamDestroy(s[i]);
        if (src[i]) (void)f.cuMemFree(src[i]);
        if (dst[i]) (void)f.cuMemFree(dst[i]);
        if (host[i]) (void)f.cuMemFreeHost(host[i]);
    }
    (void)f.cuCtxSetCurrent(kept);
}

// Times a 2-wide group's sums by each transport, form and block cap the pair allows, from one 256-float row to 512 rows of the 27B's 5120, checking nothing (`backend-cuda sums`).
// Each line is the median of 9 runs of back-to-back sums: the time a sum, each member's busy time a sum (`busy_ms`), the host's time to issue one, and that collective's sums of known rows off the oracle.
void sums() {
    CuFn& f = cu();
    std::vector<BackendPtr> d = cuda_devices(2);
    if (d.size() < 2) {
        std::printf("backend-cuda: one device, so no tensor group to time\n");
        return;
    }
    const bool peer = peer_pair(f);
    std::printf("backend-cuda: devices 0 and 1 %s each other's memory\n", peer ? "reach" : "do not reach");
    const std::vector<Backend*> group = raw(d, 2);
    // Joining probes the pair and enables the peer access that the peer-write arms below write through.
    const std::unique_ptr<Collective> joined = d[0]->join(group, 1, 1024);
    sum_rates(f, peer);
    std::fflush(stdout);
    std::vector<GpuSumPick> arms;
    for (const GroupTransport t : {GroupTransport::peer_writes, GroupTransport::host_writes}) {
        if (t == GroupTransport::peer_writes && !peer) continue;
        for (unsigned b : {0u, 8u, 4u, 2u, 1u}) arms.push_back({t, GpuSumForm::one_shot, b});
        for (unsigned b : {0u, 8u, 4u}) arms.push_back({t, GpuSumForm::shares, b});
        for (unsigned b : {0u, 4u, 1u}) arms.push_back({t, GpuSumForm::chain, b});
        if (t == GroupTransport::host_writes) arms.push_back({t, GpuSumForm::pulls});
    }
    const size_t rows_of[] = {1, 1, 1, 1, 2, 4, 8, 32, 128, 512}, widths[] = {256, 1024, 2048, 5120, 5120, 5120, 5120, 5120, 5120, 5120};
    const auto time_sums = [&](Collective& c, size_t rows, size_t width) {
        const size_t n = rows * width;
        const int N = n >= 100000 ? 10 : 32;
        const auto run = [&] {
            c.sum_in_place(rows, width);
            for (Backend* m : group) m->submit();
        };
        run();
        for (Backend* m : group) m->sync();
        std::vector<double> ms, busy0, busy1, issue;
        for (int rep = 0; rep < 9; ++rep) {
            const Ticket a0 = group[0]->last_ticket(), a1 = group[1]->last_ticket();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < N; ++i) run();
            const auto t1 = std::chrono::steady_clock::now();
            for (Backend* m : group) m->sync();
            const auto t2 = std::chrono::steady_clock::now();
            ms.push_back(std::chrono::duration<double, std::milli>(t2 - t0).count() / N);
            issue.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count() / N);
            busy0.push_back(group[0]->busy_ms(a0, group[0]->last_ticket()) / N);
            busy1.push_back(group[1]->busy_ms(a1, group[1]->last_ticket()) / N);
        }
        for (std::vector<double>* v : {&ms, &busy0, &busy1, &issue}) std::sort(v->begin(), v->end());
        char line[160];
        std::snprintf(line, sizeof line, "%.4f ms, %.2f GB/s, busy %.4f and %.4f ms, issue %.1f us", ms[4], n * 4.0 / ms[4] / 1e6, busy0[4], busy1[4], issue[4]);
        return std::string(line);
    };
    // Two rounds, the second in another order, so a drift over the run shows as a difference between them.
    for (int round = 0; round < 2; ++round) {
        for (size_t k = 0; k < arms.size(); ++k) {
            const GpuSumPick& a = arms[round ? arms.size() - 1 - k : k];
            std::unique_ptr<Collective> c = cuda_collective(group, 512, 5120, std::map<size_t, GpuSumPick>{{0, a}});
            for (size_t s = 0; s < sizeof(rows_of) / sizeof(rows_of[0]); ++s) {
                const std::string t = time_sums(*c, rows_of[s], widths[s]);
                size_t off = 0, held = 0;
                if (widths[s] == 5120) (void)group_sums::check(group, *c, 5120, {rows_of[s]}, 7 + (uint32_t)round, [&](bool ok, const std::string&) { off += ok ? 0 : 1; ++held; });
                std::printf("SUMS round %d %s %s cap %u, %zu floats: %s, %zu of %zu results off the oracle\n", round, a.transport == GroupTransport::peer_writes ? "peer writes" : "host writes", sum_form_name(a.form), a.blocks,
                            rows_of[s] * widths[s], t.c_str(), off, held);
            }
            std::fflush(stdout);
        }
    }
    // The pair joined as a model's group joins it, which times every kept transport's forms and caps and reports each size's pick on stderr; then that collective's sums.
    std::unique_ptr<Collective> c = d[0]->join(group, 512, 5120);
    for (size_t s = 0; s < sizeof(rows_of) / sizeof(rows_of[0]); ++s)
        std::printf("SUMS joined, %zu floats: %s\n", rows_of[s] * widths[s], time_sums(*c, rows_of[s], widths[s]).c_str());
}

// Measurement (cuda88): every transport, form and cap at one size of rows x width, both member orders, as vsnccl times them.
void sweepn(size_t rows, size_t width) {
    std::vector<BackendPtr> d = cuda_devices(2);
    if (d.size() < 2) return;
    const std::vector<Backend*> fwd = raw(d, 2);
    const std::unique_ptr<Collective> joined = d[0]->join(fwd, 1, 1024);
    std::vector<GpuSumPick> arms;
    for (const GroupTransport t : {GroupTransport::peer_writes, GroupTransport::host_writes}) {
        for (unsigned b : {0u, 8u, 4u, 2u, 1u}) arms.push_back({t, GpuSumForm::one_shot, b});
        for (unsigned b : {0u, 8u, 4u, 2u}) arms.push_back({t, GpuSumForm::shares, b});
        for (unsigned b : {0u, 4u, 1u}) arms.push_back({t, GpuSumForm::chain, b});
        if (t == GroupTransport::host_writes) arms.push_back({t, GpuSumForm::pulls});
    }
    const size_t n = rows * width;
    for (int round = 0; round < 2; ++round) {
        std::vector<Backend*> group = fwd;
        if (round) std::reverse(group.begin(), group.end());
        for (const GpuSumPick& a : arms) {
            std::unique_ptr<Collective> c = cuda_collective(group, rows, width, std::map<size_t, GpuSumPick>{{0, a}});
            const int N = n >= 100000 ? 10 : 32;
            const auto run = [&] {
                c->sum_in_place(rows, width);
                for (Backend* m : group) m->submit();
            };
            run();
            for (Backend* m : group) m->sync();
            std::vector<double> ms;
            for (int rep = 0; rep < 15; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < N; ++i) run();
                for (Backend* m : group) m->sync();
                ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / N);
            }
            std::sort(ms.begin(), ms.end());
            std::printf("SWEEP order %s %zux%zu %s %s cap %u us %.2f GBs %.3f\n", round ? "10" : "01", rows, width, a.transport == GroupTransport::peer_writes ? "peer" : "host", sum_form_name(a.form), a.blocks, ms[7] * 1e3, n * 4.0 / ms[7] / 1e6);
        }
        std::fflush(stdout);
    }
}

// Measurement for tools/sums_vs_reference.py: the joined group's sums per size in one member order, timed as `sums` times them.
void vsnccl(bool reversed, size_t W, size_t model) {
    std::vector<BackendPtr> d = cuda_devices(W);
    if (d.size() < W) return;
    std::vector<Backend*> group = raw(d, W);
    if (reversed) std::reverse(group.begin(), group.end());
    // A model width of 5120 times the 27B's rows and the odd row its picks cover; another width joins as a model of that width and times its one decode row.
    const std::unique_ptr<Collective> rows = group[0]->join(group, 512, model), odd = model == 5120 ? group[0]->join(group, 1, 65537) : nullptr;
    std::vector<size_t> rows_of = {1, 2, 4, 6, 8, 1, 13, 32, 128, 512}, widths = {5120, 5120, 5120, 5120, 5120, 65537, 5120, 5120, 5120, 5120};
    if (model != 5120) rows_of = {1}, widths = {model};
    for (size_t s = 0; s < rows_of.size(); ++s) {
        Collective& c = odd && widths[s] == 65537 ? *odd : *rows;
        const size_t n = rows_of[s] * widths[s];
        const int N = n >= 100000 ? 10 : 32;
        // A pass submits each device once for many sums, as all_reduce_perf queues its calls: the judged arm submits once per timed window, the llmx-each arm after every sum.
        c.sum_in_place(rows_of[s], widths[s]);
        for (Backend* m : group) m->submit();
        for (Backend* m : group) m->sync();
        for (const bool each : {false, true}) {
            std::vector<double> ms;
            for (int rep = 0; rep < 15; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < N; ++i) {
                    c.sum_in_place(rows_of[s], widths[s]);
                    if (each)
                        for (Backend* m : group) m->submit();
                }
                if (!each)
                    for (Backend* m : group) m->submit();
                for (Backend* m : group) m->sync();
                ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / N);
            }
            std::sort(ms.begin(), ms.end());
            std::printf("ARM %s order %s n %zu us %.2f GBs %.3f min %.2f max %.2f width %zu\n", each ? "llmx-each" : "llmx", reversed ? "10" : "01", n, ms[7] * 1e3,
                        n * 4.0 / ms[7] / 1e6, ms[0] * 1e3, ms[14] * 1e3, W);
        }
    }
    std::fflush(stdout);
}


// Measurement (cuda89): one decode row's sum of `model` floats, the host's time in each call beside the time a sum takes and each member's busy time a submission.
void issue(bool reversed, size_t W, size_t model) {
    std::vector<BackendPtr> d = cuda_devices(W);
    if (d.size() < W) return;
    std::vector<Backend*> group = raw(d, W);
    if (reversed) std::reverse(group.begin(), group.end());
    const std::unique_ptr<Collective> c = group[0]->join(group, 512, model);
    const int N = 32;
    const auto run = [&] {
        c->sum_in_place(1, model);
        for (Backend* m : group) m->submit();
    };
    run();
    for (Backend* m : group) m->sync();
    std::vector<double> total, host, sum_host, busy0, busy1;
    for (int rep = 0; rep < 15; ++rep) {
        std::vector<Ticket> a;
        for (Backend* m : group) a.push_back(m->last_ticket());
        double h = 0, hs = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) {
            const auto s0 = std::chrono::steady_clock::now();
            c->sum_in_place(1, model);
            const auto s1 = std::chrono::steady_clock::now();
            for (Backend* m : group) m->submit();
            const auto s2 = std::chrono::steady_clock::now();
            hs += std::chrono::duration<double, std::micro>(s1 - s0).count();
            h += std::chrono::duration<double, std::micro>(s2 - s0).count();
        }
        for (Backend* m : group) m->sync();
        total.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / N);
        host.push_back(h / N);
        sum_host.push_back(hs / N);
        busy0.push_back(group[0]->busy_ms(a[0], group[0]->last_ticket()) * 1e3 / N);
        busy1.push_back(group[1]->busy_ms(a[1], group[1]->last_ticket()) * 1e3 / N);
    }
    for (std::vector<double>* v : {&total, &host, &sum_host, &busy0, &busy1}) std::sort(v->begin(), v->end());
    std::printf("ISSUE order %s n %zu us %.2f host %.2f sum-host %.2f busy %.2f %.2f\n", reversed ? "10" : "01", model, total[7], host[7], sum_host[7], busy0[7], busy1[7]);
    std::fflush(stdout);
}

// Measurement (cuda89): the driver calls a sum issues, each timed alone over many calls, with its own device's context current and with the peer's.
void apicost() {
    std::vector<BackendPtr> d = cuda_devices(2);
    if (d.size() < 2) return;
    const CuFn& f = cu();
    CUcontext c[2];
    CUstream s[2];
    CUevent timed[2], plain[2];
    for (int i = 0; i < 2; ++i) {
        CUdevice dev = 0;
        cu_check(f.cuDeviceGet(&dev, i), "get");
        cu_check(f.cuDevicePrimaryCtxRetain(&c[i], dev), "retain");
        cu_check(f.cuCtxSetCurrent(c[i]), "set");
        cu_check(f.cuStreamCreate(&s[i], CU_STREAM_NON_BLOCKING), "stream");
        cu_check(f.cuEventCreate(&timed[i], CU_EVENT_DEFAULT), "event");
        cu_check(f.cuEventCreate(&plain[i], CU_EVENT_DISABLE_TIMING), "event");
    }
    const int N = 20000;
    const auto per = [&](const char* name, const std::function<void(int)>& call) {
        for (int i = 0; i < 1000; ++i) call(i);
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) call(i);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / N;
        cu_check(f.cuCtxSetCurrent(c[0]), "set");
        cu_check(f.cuStreamSynchronize(s[0]), "sync");
        cu_check(f.cuCtxSetCurrent(c[1]), "set");
        cu_check(f.cuStreamSynchronize(s[1]), "sync");
        std::printf("APICOST %s us %.3f\n", name, us);
    };
    per("set-context-alternating", [&](int i) { (void)f.cuCtxSetCurrent(c[i & 1]); });
    cu_check(f.cuCtxSetCurrent(c[0]), "set");
    per("record-timed-own", [&](int) { (void)f.cuEventRecord(timed[0], s[0]); });
    per("record-plain-own", [&](int) { (void)f.cuEventRecord(plain[0], s[0]); });
    cu_check(f.cuCtxSetCurrent(c[1]), "set");
    per("record-timed-peer-current", [&](int) { (void)f.cuEventRecord(timed[0], s[0]); });
    per("query-own-done", [&](int) { (void)f.cuEventQuery(timed[1]); });
    per("pair-set-record", [&](int i) { (void)f.cuCtxSetCurrent(c[i & 1]); (void)f.cuEventRecord(timed[i & 1], s[i & 1]); });
    per("pair-record-no-set", [&](int i) { (void)f.cuEventRecord(timed[i & 1], s[i & 1]); });
    std::fflush(stdout);
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
    require(std::stoi(d.arch.substr(3)) >= 70, "the device is sm_70 or newer: " + d.arch);
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
    if (mode == "vision") {
        vision_ops(*g);
        return failures ? 1 : 0;
    }
    if (mode == "deepseek4") {
        deepseek4(*g);
        return failures ? 1 : 0;
    }
    if (mode == "moe") {
        moe(d, *g, *cpu);
        return failures ? 1 : 0;
    }
    if (mode == "sweepn") {
        for (int k = 2; k + 1 < argc; k += 2) sweepn((size_t)std::atoi(argv[k]), (size_t)std::atoi(argv[k + 1]));
        return 0;
    }
    if (mode == "apicost") {
        apicost();
        return 0;
    }
    if (mode == "issue") {
        for (int k = 3; k < argc; ++k) issue(std::string(argv[2]) == "10", 2, (size_t)std::atoi(argv[k]));
        return 0;
    }
    if (mode == "vsnccl") {
        vsnccl(argc > 2 && std::string(argv[2]) == "10", argc > 3 ? (size_t)std::atoi(argv[3]) : 2, argc > 4 ? (size_t)std::atoi(argv[4]) : 5120);
        return 0;
    }
    if (mode == "sums") {
        sums();
        return 0;
    }
    if (mode == "groups") {
        groups();
        return failures ? 1 : 0;
    }
    if (mode == "timing") {
        guarded("kernel_times", [&] { twin_checks::kernel_times(*make_backend("cuda:0", true), *g, cuda_take_launches, require); });
        guarded("host_stall", [&] { twin_checks::host_stall(*make_backend("cuda:0", true), cuda_profile, cuda_delay_launches, require); });
        guarded("gated_drop", [&] { twin_checks::gated_drop(*make_backend("cuda:0", true), *g, cuda_delay_launches, require); });
        std::printf("backend-cuda timing: %s\n", failures ? "FAIL" : "PASS");
        return failures ? 1 : 0;
    }
    if (!only_types) {
        basics(d, *g, *cpu);
        guarded("kernel_times", [&] { twin_checks::kernel_times(*make_backend("cuda:0", true), *g, cuda_take_launches, require); });
        guarded("host_stall", [&] { twin_checks::host_stall(*make_backend("cuda:0", true), cuda_profile, cuda_delay_launches, require); });
        guarded("gated_drop", [&] { twin_checks::gated_drop(*make_backend("cuda:0", true), *g, cuda_delay_launches, require); });
    }
    types(d, *g, *cpu);
    if (!only_types) qwen35(d, *g);
    if (!only_types) moe(d, *g, *cpu);
    if (!only_types) deepseek4(*g);
    if (!only_types) guarded("attention tiles", [&] { attention_tiles::check(*g, cuda_take_launches, require); attention_tiles::check_long(*g, cuda_take_launches, require); dspine_checks::window(*g, *cpu, cuda_take_launches, require); });
    if (!only_types) guarded("dspine ops", [&] { dspine_checks::ops(*g, *cpu, require, {1, 4, 1}); dspine_checks::dflash2_ops(*g, *cpu, require, 0, 1); });
    if (!only_types) vision_ops(*g);
    if (!only_types) groups();
    return failures ? 1 : 0;
}
