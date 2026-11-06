// deepseek4-ops: DeepSeek-V4.1's storage formats and the CPU forms of its layer ops (docs/DEEPSEEK4.md, sections 4.2 and 4.5).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/cpu/cpu_backend.hpp"
#include "quant/model_formats.hpp"

namespace {

namespace mf = quant::model_formats;
using backend::BufferPtr;
using backend::CpuBackend;

size_t checked = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checked;
}

uint32_t bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

float from_bits(uint32_t u) {
    float x;
    std::memcpy(&x, &u, sizeof x);
    return x;
}

bool same(float a, float b) { return bits(a) == bits(b) || (std::isnan(a) && std::isnan(b)); }

std::string hex(float x) { return std::to_string(x) + " (0x" + [&] { char s[9]; std::snprintf(s, sizeof s, "%08x", bits(x)); return std::string(s); }() + ")"; }

uint64_t state = 0x243F6A8885A308D3ull;
uint32_t draw() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return uint32_t(state >> 32);
}

// E4M3 from its text: 1 sign bit, 4 exponent bits of bias 7, 3 mantissa bits, subnormals at exponent 0, no infinities, S.1111.111 NaN.
double e4m3_text(unsigned c) {
    const unsigned e = (c >> 3) & 15, m = c & 7;
    const double v = e ? std::ldexp(1.0 + m / 8.0, int(e) - 7) : std::ldexp(m / 8.0, -6);
    return c & 0x80 ? -v : v;
}

// E2M1 from its text: 1 sign bit, 2 exponent bits of bias 1, 1 mantissa bit, a subnormal 0.5 at exponent 0.
double e2m1_text(unsigned c) {
    const unsigned e = (c >> 1) & 3, m = c & 1;
    const double v = e ? std::ldexp(1.0 + m / 2.0, int(e) - 1) : m / 2.0;
    return c & 8 ? -v : v;
}

// The nearest of the non-negative codes [0, n) to |x| under `value`, a tie to the even code, the sign of x put back in `sign_bit`.
template <class Value>
unsigned nearest(double x, unsigned n, unsigned sign_bit, Value value) {
    const double a = std::fabs(x);
    unsigned best = 0;
    for (unsigned c = 1; c < n; ++c) {
        const double d = std::fabs(value(c) - a), b = std::fabs(value(best) - a);
        if (d < b || (d == b && !(c & 1))) best = c;
    }
    return std::signbit(x) ? best | sign_bit : best;
}

unsigned e4m3_nearest(double x) {
    if (std::fabs(x) > 448.0) return std::signbit(x) ? 0xFE : 0x7E;
    return nearest(x, 0x7F, 0x80, e4m3_text);
}

unsigned e2m1_nearest(double x) { return nearest(x, 8, 8, e2m1_text); }

// The candidates a rounding can turn on: every code's value, each midpoint between neighbours, and the floats beside each, of both signs.
std::vector<float> boundaries(const std::vector<double>& values) {
    std::vector<float> out;
    for (size_t i = 0; i < values.size(); ++i) {
        std::vector<double> at{values[i]};
        if (i + 1 < values.size()) at.push_back((values[i] + values[i + 1]) / 2);
        for (double v : at) {
            const float f = float(v);
            require(double(f) == v, "a boundary is not a float");
            for (float g : {f, std::nextafter(f, 0.0f), std::nextafter(f, 1e30f)})
                for (float s : {g, -g}) out.push_back(s);
        }
    }
    return out;
}

void codes() {
    for (unsigned c = 0; c < 256; ++c) {
        const double t = e4m3_text(c);
        const float v = mf::e4m3_value(uint8_t(c));
        if ((c & 0x7F) == 0x7F) require(std::isnan(v), "E4M3 code " + std::to_string(c) + " is not NaN");
        else require(double(v) == t && std::signbit(v) == bool(c & 0x80), "E4M3 code " + std::to_string(c) + " decodes to " + hex(v));
    }
    for (unsigned c = 0; c < 16; ++c) {
        const float v = mf::e2m1_value(uint8_t(c));
        require(double(v) == e2m1_text(c) && std::signbit(v) == bool(c & 8), "E2M1 code " + std::to_string(c) + " decodes to " + hex(v));
    }
    std::vector<double> e4m3, e2m1;
    for (unsigned c = 0; c < 0x7F; ++c) e4m3.push_back(e4m3_text(c));
    for (unsigned c = 0; c < 8; ++c) e2m1.push_back(e2m1_text(c));
    e4m3.push_back(480.0);
    std::vector<float> xs = boundaries(e4m3);
    for (float x : {464.0f, 465.0f, 1e6f, 3.4e38f, 1e-30f, 0x1p-149f, 0x1p-11f, 0x1.000002p-11f}) {
        xs.push_back(x);
        xs.push_back(-x);
    }
    for (int i = 0; i < 1000000; ++i) {
        const float x = from_bits(draw());
        if (std::isfinite(x) && std::fabs(x) < 1e4f) xs.push_back(x);
    }
    for (float x : xs) {
        const unsigned want = e4m3_nearest(x), got = mf::e4m3_code(x);
        require(got == want, "E4M3 code of " + hex(x) + ": " + std::to_string(got) + " not " + std::to_string(want));
    }
    require(mf::e4m3_code(NAN) == 0x7F, "a NaN's E4M3 code");
    xs = boundaries(e2m1);
    for (int i = 0; i < 1000000; ++i) xs.push_back(float(int32_t(draw())) / float(0x7FFFFFFF) * 6.0f);
    for (float x : xs) {
        if (std::fabs(x) > 6.0f) continue;
        const unsigned want = e2m1_nearest(x), got = mf::e2m1_code(x);
        require(got == want, "E2M1 code of " + hex(x) + ": " + std::to_string(got) + " not " + std::to_string(want));
    }
}

// The bit rule against the least power of two at or above x: exact powers are their own ceiling, a subnormal gives 2^-126.
void scales() {
    for (int e = -149; e <= 126; ++e) {
        const float p = std::ldexp(1.0f, e);
        for (float x : {p, std::nextafter(p, 1e30f), std::nextafter(p, 0.0f)}) {
            if (x <= 0) continue;
            int want = e;
            if (x != p) want = x > p ? e + 1 : e;
            if (x < 0x1p-126f) want = -126;
            require(mf::ceil_log2(x) == want, "ceil_log2 of " + hex(x));
            require(mf::pow2(mf::ceil_log2(x)) == std::ldexp(1.0f, want), "pow2 of ceil_log2 of " + hex(x));
        }
    }
    // The power scale is taken of the F32 product amax * (1 / max), never of the exact quotient.
    for (int i = 0; i < 100000; ++i) {
        const float amax = std::fabs(from_bits(draw() & 0x7EFFFFFF));
        for (float inv : {1.0f / mf::FP8_MAX, 1.0f / mf::FP4_MAX}) {
            const float product = amax * inv;
            if (!std::isfinite(product) || product == 0) continue;
            const int e = product < 0x1p-126f ? -126 : int(std::ceil(std::log2(double(product))));
            require(mf::power_scale(amax, inv) == std::ldexp(1.0f, e), "power scale of " + hex(amax));
        }
    }
}

// A block quantized and dequantized as the format's text gives it, in double but for the F32 operations the rule names.
std::vector<float> qdq_text(std::vector<float> x, int format) {
    float amax = 0;
    for (float v : x) amax = std::fmax(amax, std::fabs(v));
    float s;
    if (format == 0) {
        const float p = std::fmax(amax, 1e-4f) * (1.0f / 448.0f);
        s = std::ldexp(1.0f, p < 0x1p-126f ? -126 : int(std::ceil(std::log2(double(p)))));
    } else if (format == 1) {
        const float p = std::fmax(amax, 6.0f * 0x1p-126f) * (1.0f / 6.0f);
        s = std::ldexp(1.0f, p < 0x1p-126f ? -126 : int(std::ceil(std::log2(double(p)))));
    } else {
        s = float(e4m3_text(e4m3_nearest(double(std::fmax(amax, 6.0f * 0x1p-9f) / 6.0f))));
    }
    for (float& v : x) {
        const float q = v / s, limit = format ? 6.0f : 448.0f;
        const double c = q < -limit ? -limit : (q > limit ? limit : q);
        v = float(format ? e2m1_text(e2m1_nearest(c)) : e4m3_text(e4m3_nearest(c))) * s;
    }
    return x;
}

void blocks() {
    const char* names[] = {"FP8 under a power scale", "FP4 under a power scale", "FP4 under an E4M3 scale"};
    const size_t widths[] = {32, 32, 16};
    for (int format = 0; format < 3; ++format) {
        for (int i = 0; i < 20000; ++i) {
            std::vector<float> x(widths[format]);
            // Magnitudes from 2^-140 to 2^20 so the clamps, the subnormal scales and the saturated scale are reached, a zero block and signed zeros among them.
            const int top = int(draw() % 161) - 140;
            for (float& v : x) {
                const uint32_t r = draw();
                v = (r % 17 == 0) ? (r & 1 ? -0.0f : 0.0f) : std::ldexp(float(r >> 8) / float(1u << 24), top - int(r % 5)) * (r & 2 ? -1.0f : 1.0f);
            }
            if (i == 0) std::fill(x.begin(), x.end(), 0.0f);
            const std::vector<float> want = qdq_text(x, format);
            if (format == 0) mf::qdq_fp8_power(x.data(), x.size());
            else if (format == 1) mf::qdq_fp4_power(x.data(), x.size());
            else mf::qdq_fp4_e4m3(x.data(), x.size());
            for (size_t j = 0; j < x.size(); ++j) require(same(x[j], want[j]), std::string(names[format]) + ": value " + std::to_string(j) + " is " + hex(x[j]) + " not " + hex(want[j]));
        }
    }
    // The clamps by name: a block below each floor takes the floor's scale.
    std::vector<float> tiny(32, 1e-30f);
    mf::qdq_fp8_power(tiny.data(), tiny.size());
    require(tiny[0] == 0.0f, "the window's FP8 floor of 1e-4 rounds 1e-30 to zero");
    std::vector<float> entry(16, 2e-4f);
    mf::qdq_fp4_e4m3(entry.data(), entry.size());
    require(entry[0] == 0.0f, "the entries' floor 6 * 2^-9 gives the scale 2^-9, under which 2e-4 rounds to zero");
    std::vector<float> big(16, 1e6f);
    mf::qdq_fp4_e4m3(big.data(), big.size());
    require(big[0] == 6.0f * 448.0f, "a scale past E4M3's range saturates at 448");
}

// The step 2 port's codecs (tools/deepseek4_kernel.py) on the seeded blocks of tools/deepseek4_reference.py --codecs, held bit for bit through FNV-1a 64 digests of runs of blocks.
// The port's E4M3 cast of a scale past 464 gives NaN where the kernels saturate, so those blocks are left out on both sides and counted.
void port(const std::string& path) {
    std::ifstream in(path);
    require(bool(in), "cannot open " + path);
    std::string line;
    std::getline(in, line);
    std::getline(in, line);
    std::istringstream head(line);
    std::string word;
    uint64_t seed = 0;
    size_t blocks = 0, run = 0;
    head >> word >> seed >> word >> blocks >> word >> run;
    require(blocks > 2 && run > 0 && blocks % run == 0, path + ": a header of seed, blocks and run");
    state = seed;
    const char* names[3] = {"fp8_power", "fp4_power", "fp4_e4m3"};
    for (int format = 0; format < 3; ++format) {
        require(bool(std::getline(in, line)), path + ": no line for " + names[format]);
        std::istringstream row(line);
        std::string name;
        size_t width = 0, left_out = 0;
        row >> name >> width >> left_out;
        require(name == names[format], path + ": " + name + " where " + names[format] + " was expected");
        // The same draws as the generator: block 0 all zeros, block 1 all 1e6, then 24-bit values across exponents, one in 17 a signed zero.
        std::vector<float> x(blocks * width);
        for (size_t i = 0; i < width; ++i) x[i] = 0.0f, x[width + i] = 1e6f;
        for (size_t b = 2; b < blocks; ++b) {
            const int top = -140 + int(draw() % 161);
            for (size_t i = 0; i < width; ++i) {
                const uint32_t r = draw();
                if (r % 17 == 0) x[b * width + i] = r & 1 ? -0.0f : 0.0f;
                else x[b * width + i] = float(std::ldexp(double(r >> 8), top - int(r % 5) - 24) * (r & 2 ? -1.0 : 1.0));
            }
        }
        size_t skipped = 0, compared = 0;
        for (size_t start = 0; start < blocks; start += run) {
            uint64_t h = 0xCBF29CE484222325ull;
            for (size_t b = start; b < start + run; ++b) {
                float* p = x.data() + b * width;
                if (format == 2 && std::fmax(mf::block_amax(p, width), 6.0f * 0x1p-9f) / mf::FP4_MAX >= 464.0f) {
                    ++skipped;
                    continue;
                }
                if (format == 0) mf::qdq_fp8_power(p, width);
                else if (format == 1) mf::qdq_fp4_power(p, width);
                else mf::qdq_fp4_e4m3(p, width);
                const auto* bytes = reinterpret_cast<const unsigned char*>(p);
                for (size_t k = 0; k < width * sizeof(float); ++k) h = (h ^ bytes[k]) * 0x100000001B3ull;
                compared += width;
            }
            std::string want;
            row >> want;
            char got[17];
            std::snprintf(got, sizeof got, "%016llx", (unsigned long long)h);
            require(want == got, std::string(names[format]) + ": blocks " + std::to_string(start) + " to " + std::to_string(start + run - 1) + " differ from the port's");
        }
        require(skipped == left_out, std::string(names[format]) + ": " + std::to_string(skipped) + " blocks left out where the port gave NaN in " + std::to_string(left_out));
        std::cout << "port " << names[format] << ": " << compared << " values bit for bit, " << skipped << " blocks left out" << std::endl;
    }
}

// One unit in the last place of 1 in F32; the bounds below are counts of it times a magnitude.
const double U = std::ldexp(1.0, -24);

std::vector<float> uniform(size_t n, float lo, float hi) {
    std::vector<float> v(n);
    for (float& x : v) x = lo + (hi - lo) * float(draw() >> 8) / float(1 << 24);
    return v;
}

BufferPtr upload(CpuBackend& cpu, const std::vector<float>& v) {
    BufferPtr b = cpu.alloc(std::max<size_t>(v.size(), 1) * sizeof(float), backend::Memory::device);
    if (!v.empty()) cpu.write(*b, 0, v.data(), v.size() * sizeof(float));
    return b;
}

std::vector<float> download(CpuBackend& cpu, const backend::Buffer& b, size_t n) {
    std::vector<float> v(n);
    cpu.read(b, 0, v.data(), n * sizeof(float));
    return v;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) { return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float)); }

template <class F> void refused(F&& f, const std::string& what) {
    bool threw = false;
    try { f(); } catch (const std::runtime_error&) { threw = true; }
    require(threw, what + " is refused");
}

const int kThreads[] = {1, 2, 3, 5, 8};

// model_format_qdq on the CPU: every block of a run the bits of the format's block function alone, at every thread count, and a run that ends inside a block refused.
void format_op() {
    const backend::ModelFormat formats[] = {backend::ModelFormat::fp8_power, backend::ModelFormat::fp4_power, backend::ModelFormat::fp4_e4m3};
    const size_t widths[] = {32, 32, 16};
    for (int f = 0; f < 3; ++f) {
        const size_t width = widths[f], count = 37 * width;
        std::vector<float> x(count);
        for (size_t i = 0; i < count; ++i) {
            const uint32_t r = draw();
            x[i] = std::ldexp(float(r >> 8) / float(1u << 24), int((i / width) % 41) - 20) * (r & 1 ? -1.0f : 1.0f);
        }
        std::vector<float> want = x;
        for (size_t i = 0; i < count; i += width) {
            if (f == 0) mf::qdq_fp8_power(&want[i], width);
            else if (f == 1) mf::qdq_fp4_power(&want[i], width);
            else mf::qdq_fp4_e4m3(&want[i], width);
        }
        CpuBackend cpu;
        for (int t : kThreads) {
            cpu.set_threads(t);
            BufferPtr xb = upload(cpu, x);
            cpu.model_format_qdq({xb.get(), 0}, count, formats[f]);
            require(!std::memcmp(download(cpu, *xb, count).data(), want.data(), count * 4), "model_format_qdq format " + std::to_string(f) + " at " + std::to_string(t) + " threads");
        }
        BufferPtr xb = upload(cpu, x);
        refused([&] { cpu.model_format_qdq({xb.get(), 0}, count - 1, formats[f]); }, "model_format_qdq ending inside a block");
    }
}

// The mHC ops against references in double written from kernel.py:426-460 and model.py:950-967.
void mhc() {
    const size_t hc = 4, dim = 40, rows = 7, mix = (hc + 2) * hc, width = hc * dim, iters = 20;
    const float norm_eps = 1e-6f, eps = 1e-6f, rms_eps = 1e-6f;
    const std::vector<float> mixes = uniform(rows * mix, -40.0f, 40.0f), x = uniform(rows * width, -3.0f, 3.0f);
    const std::vector<float> scale = {0.9f, 1.1f, 1.0f}, base = uniform(mix, -1.0f, 1.0f), w = uniform(dim, 0.5f, 1.5f);
    const std::vector<float> f = uniform(rows * dim, -2.0f, 2.0f);
    CpuBackend cpu;
    BufferPtr mb = upload(cpu, mixes), xb = upload(cpu, x), sb = upload(cpu, scale), bb = upload(cpu, base), wb = upload(cpu, w), fb = upload(cpu, f);
    auto coeffs = [&](size_t first, size_t n) {
        BufferPtr pre = cpu.alloc(n * hc * 4, backend::Memory::device), post = cpu.alloc(n * hc * 4, backend::Memory::device);
        BufferPtr comb = cpu.alloc(n * hc * hc * 4, backend::Memory::device);
        cpu.hc_coeffs({pre.get(), 0}, {post.get(), 0}, {comb.get(), 0}, {mb.get(), first * mix}, {xb.get(), first * width}, {sb.get(), 0}, {bb.get(), 0},
                      n, hc, dim, norm_eps, eps, iters);
        std::vector<float> out = download(cpu, *pre, n * hc), p = download(cpu, *post, n * hc), c = download(cpu, *comb, n * hc * hc);
        out.insert(out.end(), p.begin(), p.end());
        out.insert(out.end(), c.begin(), c.end());
        return out;
    };
    cpu.set_threads(1);
    const std::vector<float> got = coeffs(0, rows);
    std::vector<float> pre(got.begin(), got.begin() + rows * hc), post(got.begin() + rows * hc, got.begin() + 2 * rows * hc);
    std::vector<float> comb(got.begin() + 2 * rows * hc, got.end());
    auto sig = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
    double worst = 0;
    for (size_t r = 0; r < rows; ++r) {
        double squares = 0;
        for (size_t i = 0; i < width; ++i) squares += double(x[r * width + i]) * x[r * width + i];
        const double rs = 1.0 / std::sqrt(squares / width + norm_eps);
        const float* m = mixes.data() + r * mix;
        std::vector<double> c(hc * hc);
        for (size_t k = 0; k < hc; ++k) {
            const double wp = sig(m[k] * rs * scale[0] + base[k]) + eps, wq = 2 * sig(m[hc + k] * rs * scale[1] + base[hc + k]);
            require(std::fabs(pre[r * hc + k] - wp) <= 1e-6 && std::fabs(post[r * hc + k] - wq) <= 1e-6, "mHC pre and post of row " + std::to_string(r));
        }
        for (size_t i = 0; i < hc; ++i) {
            double peak = -INFINITY, sum = 0;
            for (size_t j = 0; j < hc; ++j) peak = std::max(peak, c[i * hc + j] = m[2 * hc + i * hc + j] * rs * scale[2] + base[2 * hc + i * hc + j]);
            for (size_t j = 0; j < hc; ++j) sum += c[i * hc + j] = std::exp(c[i * hc + j] - peak);
            for (size_t j = 0; j < hc; ++j) c[i * hc + j] = c[i * hc + j] / sum + eps;
        }
        for (size_t round = 0; round < iters; ++round) {
            for (size_t i = 0; round && i < hc; ++i) {
                double sum = 0;
                for (size_t j = 0; j < hc; ++j) sum += c[i * hc + j];
                for (size_t j = 0; j < hc; ++j) c[i * hc + j] /= sum + eps;
            }
            for (size_t j = 0; j < hc; ++j) {
                double sum = 0;
                for (size_t i = 0; i < hc; ++i) sum += c[i * hc + j];
                for (size_t i = 0; i < hc; ++i) c[i * hc + j] /= sum + eps;
            }
        }
        for (size_t i = 0; i < hc * hc; ++i) {
            worst = std::max(worst, std::fabs(comb[r * hc * hc + i] - c[i]));
            require(std::fabs(comb[r * hc * hc + i] - c[i]) <= 1e-5, "mHC comb of row " + std::to_string(r) + " entry " + std::to_string(i) + " against double");
        }
    }
    std::cout << "hc_coeffs: comb within " << worst << " of double after " << iters << " Sinkhorn rounds" << std::endl;
    // A row alone gives the bits it gives beside the others, at every thread count.
    for (int t : kThreads) {
        cpu.set_threads(t);
        require(same_bits(coeffs(0, rows), got), "hc_coeffs at " + std::to_string(t) + " threads");
    }
    for (size_t r = 0; r < rows; ++r) {
        const std::vector<float> one = coeffs(r, 1);
        for (size_t k = 0; k < hc; ++k) require(same(one[k], pre[r * hc + k]) && same(one[hc + k], post[r * hc + k]), "hc_coeffs of row " + std::to_string(r) + " alone");
        for (size_t i = 0; i < hc * hc; ++i) require(same(one[2 * hc + i], comb[r * hc * hc + i]), "hc_coeffs comb of row " + std::to_string(r) + " alone");
    }
    BufferPtr preb = upload(cpu, pre), postb = upload(cpu, post), combb = upload(cpu, comb);

    // The collapse u = sum of pre_k X_k, then RMSNorm with w.
    auto collapse = [&](size_t first, size_t n) {
        BufferPtr out = cpu.alloc(n * dim * 4, backend::Memory::device);
        cpu.hc_collapse_norm({out.get(), 0}, {xb.get(), first * width}, {preb.get(), first * hc}, {wb.get(), 0}, n, hc, dim, rms_eps);
        return download(cpu, *out, n * dim);
    };
    cpu.set_threads(1);
    const std::vector<float> u = collapse(0, rows);
    for (size_t r = 0; r < rows; ++r) {
        std::vector<double> y(dim);
        double squares = 0;
        for (size_t d = 0; d < dim; ++d) {
            for (size_t k = 0; k < hc; ++k) y[d] += double(pre[r * hc + k]) * x[r * width + k * dim + d];
            squares += y[d] * y[d];
        }
        const double inv = 1.0 / std::sqrt(squares / dim + rms_eps);
        for (size_t d = 0; d < dim; ++d) {
            const double want = y[d] * inv * w[d];
            require(std::fabs(u[r * dim + d] - want) <= (dim + 8) * U * (std::fabs(want) + 1e-6) + 16 * U, "hc_collapse_norm row " + std::to_string(r) + " value " + std::to_string(d));
        }
        require(!std::memcmp(collapse(r, 1).data(), u.data() + r * dim, dim * 4), "hc_collapse_norm of row " + std::to_string(r) + " alone");
    }
    for (int t : kThreads) {
        cpu.set_threads(t);
        require(same_bits(collapse(0, rows), u), "hc_collapse_norm at " + std::to_string(t) + " threads");
    }

    // The expansion X'_k = post_k f + sum over j of comb[j][k] X_j, out of place and in place.
    auto expand = [&](bool in_place) {
        BufferPtr out = in_place ? upload(cpu, x) : cpu.alloc(rows * width * 4, backend::Memory::device);
        cpu.hc_expand({out.get(), 0}, {fb.get(), 0}, {in_place ? out.get() : xb.get(), 0}, {postb.get(), 0}, {combb.get(), 0}, rows, hc, dim);
        return download(cpu, *out, rows * width);
    };
    cpu.set_threads(1);
    const std::vector<float> y = expand(false);
    for (size_t r = 0; r < rows; ++r)
        for (size_t k = 0; k < hc; ++k)
            for (size_t d = 0; d < dim; ++d) {
                double want = double(post[r * hc + k]) * f[r * dim + d], size = std::fabs(want);
                for (size_t j = 0; j < hc; ++j) {
                    const double term = double(comb[r * hc * hc + j * hc + k]) * x[r * width + j * dim + d];
                    want += term;
                    size += std::fabs(term);
                }
                require(std::fabs(y[r * width + k * dim + d] - want) <= (hc + 2) * U * size, "hc_expand row " + std::to_string(r) + " stream " + std::to_string(k) + " value " + std::to_string(d));
            }
    for (int t : kThreads) {
        cpu.set_threads(t);
        require(same_bits(expand(false), y) && same_bits(expand(true), y), "hc_expand at " + std::to_string(t) + " threads, out of place and in place");
    }

    // Refusals: no streams, an output overlapping its inputs, and an expansion over part of the stream.
    refused([&] { cpu.hc_coeffs({preb.get(), 0}, {postb.get(), 0}, {combb.get(), 0}, {mb.get(), 0}, {xb.get(), 0}, {sb.get(), 0}, {bb.get(), 0}, rows, 0, dim, norm_eps, eps, iters); }, "hc_coeffs of no streams");
    refused([&] { cpu.hc_coeffs({preb.get(), 0}, {preb.get(), 1}, {combb.get(), 0}, {mb.get(), 0}, {xb.get(), 0}, {sb.get(), 0}, {bb.get(), 0}, rows, hc, dim, norm_eps, eps, iters); }, "hc_coeffs with pre and post overlapping");
    refused([&] { cpu.hc_collapse_norm({xb.get(), 0}, {xb.get(), 0}, {preb.get(), 0}, {wb.get(), 0}, rows, hc, dim, rms_eps); }, "hc_collapse_norm into its stream");
    BufferPtr scratch = upload(cpu, x);
    refused([&] { cpu.hc_expand({scratch.get(), 1}, {fb.get(), 0}, {scratch.get(), 0}, {postb.get(), 0}, {combb.get(), 0}, rows - 1, hc, dim); }, "hc_expand over a shifted stream");
}

// U32 ids and lengths travel in float buffers, as the id ops read them.
const uint32_t kNone = UINT32_MAX;
std::vector<float> as_floats(const std::vector<uint32_t>& v) {
    std::vector<float> f(v.size());
    for (size_t i = 0; i < v.size(); ++i) f[i] = from_bits(v[i]);
    return f;
}
std::vector<uint32_t> as_ids(const std::vector<float>& f) {
    std::vector<uint32_t> v(f.size());
    for (size_t i = 0; i < f.size(); ++i) v[i] = bits(f[i]);
    return v;
}

// Scores on a few levels, -inf and both zeros among them, so ties fall across every cut.
std::vector<float> levels(size_t n) {
    const float inf = std::numeric_limits<float>::infinity();
    const float level[] = {-inf, -1.5f, -0.0f, 0.0f, 0.25f, 1.0f, 2.0f};
    std::vector<float> v(n);
    for (float& x : v) x = level[draw() % 7];
    return v;
}

// The selections' order written out: a full stable sort by score, higher first, ties to the lower index.
std::vector<uint32_t> ranked(const float* s, size_t n) {
    std::vector<uint32_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = uint32_t(i);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return s[a] > s[b]; });
    return order;
}

// topk_rows and candidate_blocks against the rules of model.py:575-611: a reachable pick of -inf kept, positions at or past the length dropped, the newest block pinned.
void selections() {
    const float inf = std::numeric_limits<float>::infinity();
    CpuBackend cpu;
    auto topk = [&](const std::vector<float>& s, const std::vector<uint32_t>& lens, size_t first, size_t rows, size_t n, size_t k, uint32_t offset) {
        BufferPtr sb = upload(cpu, s), lb = upload(cpu, as_floats(lens)), ib = cpu.alloc(std::max<size_t>(rows * k, 1) * 4, backend::Memory::device);
        cpu.topk_rows({ib.get(), 0}, {sb.get(), first * n}, {lb.get(), first}, rows, n, k, offset);
        return as_ids(download(cpu, *ib, rows * k));
    };
    auto topk_case = [&](size_t rows, size_t n, size_t k, uint32_t offset, std::vector<float> s, const std::vector<uint32_t>& lens) {
        cpu.set_threads(1);
        const std::vector<uint32_t> got = topk(s, lens, 0, rows, n, k, offset);
        for (size_t r = 0; r < rows; ++r) {
            std::vector<uint32_t> order = ranked(s.data() + r * n, n);
            order.resize(k);
            std::sort(order.begin(), order.end());
            for (size_t i = 0; i < k; ++i)
                require(got[r * k + i] == (order[i] < lens[r] ? order[i] + offset : kNone), "topk_rows row " + std::to_string(r) + " pick " + std::to_string(i));
            require(topk(s, lens, r, 1, n, k, offset) == std::vector<uint32_t>(got.begin() + r * k, got.begin() + (r + 1) * k), "topk_rows of row " + std::to_string(r) + " alone");
        }
        for (int t : kThreads) {
            cpu.set_threads(t);
            require(topk(s, lens, 0, rows, n, k, offset) == got, "topk_rows at " + std::to_string(t) + " threads");
        }
        return got;
    };
    topk_case(6, 50, 12, 128, levels(6 * 50), {0, 1, 7, 30, 49, 50});
    topk_case(3, 3000, 512, 0, levels(3 * 3000), {3000, 2048, 700});
    topk_case(2, 20, 20, 5, levels(2 * 20), {20, 13});
    // A row of -inf whose positions are all reachable still picks, by lower index.
    const std::vector<uint32_t> low = topk_case(1, 16, 4, 0, std::vector<float>(16, -inf), {16});
    require(low == std::vector<uint32_t>({0, 1, 2, 3}), "topk_rows of a row of -inf picks its first positions");

    auto blocks = [&](const std::vector<float>& s, const std::vector<uint32_t>& lens, size_t first, size_t rows, size_t n, size_t bs, size_t k) {
        BufferPtr sb = upload(cpu, s), lb = upload(cpu, as_floats(lens)), ob = cpu.alloc(std::max<size_t>(rows * k, 1) * 4, backend::Memory::device);
        cpu.candidate_blocks({ob.get(), 0}, {sb.get(), first * n}, {lb.get(), first}, rows, n, bs, k);
        return as_ids(download(cpu, *ob, rows * k));
    };
    auto blocks_case = [&](size_t rows, size_t n, size_t bs, size_t k, const std::vector<float>& s, const std::vector<uint32_t>& lens) {
        cpu.set_threads(1);
        const std::vector<uint32_t> got = blocks(s, lens, 0, rows, n, bs, k);
        const size_t nb = (n + bs - 1) / bs;
        for (size_t r = 0; r < rows; ++r) {
            std::vector<float> score(nb, -inf);
            for (size_t i = 0; i < n; ++i) score[i / bs] = std::max(score[i / bs], s[r * n + i]);
            if (lens[r] && (lens[r] - 1) / bs < nb) score[(lens[r] - 1) / bs] = inf;
            std::vector<uint32_t> order = ranked(score.data(), nb), want;
            order.resize(std::min(k, nb));
            for (uint32_t b : order)
                if (score[b] > -inf) want.push_back(b);
            std::sort(want.begin(), want.end());
            want.resize(k, kNone);
            require(std::vector<uint32_t>(got.begin() + r * k, got.begin() + (r + 1) * k) == want, "candidate_blocks row " + std::to_string(r));
            require(blocks(s, lens, r, 1, n, bs, k) == want, "candidate_blocks of row " + std::to_string(r) + " alone");
        }
        for (int t : kThreads) {
            cpu.set_threads(t);
            require(blocks(s, lens, 0, rows, n, bs, k) == got, "candidate_blocks at " + std::to_string(t) + " threads");
        }
        return got;
    };
    // Positions past each row's length at -inf, as the index scores give them.
    std::vector<float> s = levels(6 * 50);
    const std::vector<uint32_t> lens = {0, 1, 9, 20, 49, 50};
    for (size_t r = 0; r < 6; ++r)
        for (size_t i = lens[r]; i < 50; ++i) s[r * 50 + i] = -inf;
    blocks_case(6, 50, 8, 4, s, lens);
    blocks_case(6, 50, 8, 10, s, lens);
    blocks_case(2, 4096, 8, 64, levels(2 * 4096), {4096, 1000});
    // Only the pinned block of a row of -inf survives; a row of length 0 pins nothing and keeps nothing.
    require(blocks_case(1, 40, 8, 3, std::vector<float>(40, -inf), {20}) == std::vector<uint32_t>({2, kNone, kNone}), "candidate_blocks keeps the pinned block of a row of -inf");
    require(blocks_case(1, 40, 8, 3, std::vector<float>(40, -inf), {0}) == std::vector<uint32_t>(3, kNone), "candidate_blocks of length 0 keeps nothing");

    BufferPtr nb = upload(cpu, {0.0f, std::nanf(""), 1.0f, 2.0f}), lb = upload(cpu, as_floats({4})), ob = cpu.alloc(64, backend::Memory::device);
    refused([&] { cpu.topk_rows({ob.get(), 0}, {nb.get(), 0}, {lb.get(), 0}, 1, 4, 2, 0); }, "topk_rows over a NaN");
    refused([&] { cpu.topk_rows({ob.get(), 0}, {lb.get(), 0}, {lb.get(), 0}, 1, 1, 2, 0); }, "topk_rows of more picks than positions");
    refused([&] { cpu.topk_rows({nb.get(), 0}, {nb.get(), 0}, {lb.get(), 0}, 1, 4, 2, 0); }, "topk_rows into its scores");
    refused([&] { cpu.candidate_blocks({ob.get(), 0}, {nb.get(), 0}, {lb.get(), 0}, 1, 4, 2, 2); }, "candidate_blocks over a NaN");
    refused([&] { cpu.candidate_blocks({ob.get(), 0}, {lb.get(), 0}, {lb.get(), 0}, 1, 1, 0, 1); }, "candidate_blocks of blocks of no positions");
}

// sparse_attention_sink against kernel.py:349-351's rule in double, its rows' ids built as model.py:410-426 builds a prefill window of 128 positions including the query, then compressed picks.
void sink_attention() {
    const size_t heads = 3, dim = 24, window = 128, seq = 160, entries = 20, picks = 8, k = window + picks, kv_rows = seq + entries;
    const std::vector<size_t> pos = {0, 5, 127, 128, 159};
    const size_t rows = pos.size() + 2;
    std::vector<uint32_t> ids(rows * k, kNone);
    for (size_t r = 0; r < pos.size(); ++r) {
        const size_t start = pos[r] + 1 > window ? pos[r] + 1 - window : 0;
        for (size_t i = 0; i < window && start + i <= pos[r]; ++i) ids[r * k + i] = uint32_t(start + i);
        for (size_t i = 0; i < picks; ++i)
            if (draw() % 3) ids[r * k + window + i] = uint32_t(seq + draw() % entries);
    }
    // A row naming nothing, and one naming a single compressed entry.
    ids[(rows - 1) * k + window] = uint32_t(seq + 3);
    const std::vector<float> q = uniform(rows * heads * dim, -1.0f, 1.0f), kv = uniform(kv_rows * dim, -1.0f, 1.0f), sink = {-0.5f, 0.0f, 6.0f};
    const float scale = 1.0f / std::sqrt(float(dim));
    CpuBackend cpu;
    BufferPtr qb = upload(cpu, q), kb = upload(cpu, kv), ib = upload(cpu, as_floats(ids)), sb = upload(cpu, sink);
    auto run = [&](size_t first, size_t n) {
        BufferPtr ob = cpu.alloc(n * heads * dim * 4, backend::Memory::device);
        cpu.sparse_attention_sink({ob.get(), 0}, {qb.get(), first * heads * dim}, {kb.get(), 0}, {ib.get(), first * k}, {sb.get(), 0}, n, heads, dim, kv_rows, k, scale, 0, nullptr, 0, {}, 0);
        return download(cpu, *ob, n * heads * dim);
    };
    cpu.set_threads(1);
    const std::vector<float> got = run(0, rows);
    for (size_t r = 0; r < rows; ++r)
        for (size_t h = 0; h < heads; ++h) {
            std::vector<double> s;
            std::vector<uint32_t> named;
            double peak = -1e30;
            for (size_t t = 0; t < k; ++t) {
                if (ids[r * k + t] == kNone) continue;
                double dot = 0.0;
                for (size_t d = 0; d < dim; ++d) dot += double(q[(r * heads + h) * dim + d]) * kv[ids[r * k + t] * dim + d];
                s.push_back(dot * scale);
                named.push_back(ids[r * k + t]);
                peak = std::max(peak, s.back());
            }
            double denominator = std::exp(double(sink[h]) - peak);
            for (double v : s) denominator += std::exp(v - peak);
            for (size_t d = 0; d < dim; ++d) {
                double want = 0.0, size = 0.0;
                for (size_t t = 0; t < s.size(); ++t) {
                    want += std::exp(s[t] - peak) * kv[named[t] * dim + d] / denominator;
                    size += std::exp(s[t] - peak) * std::fabs(kv[named[t] * dim + d]) / denominator;
                }
                const float o = got[(r * heads + h) * dim + d];
                if (s.empty()) require(bits(o) == 0, "sparse_attention_sink of a row naming nothing is +0");
                else require(std::fabs(o - want) <= 1e-5 * size + 1e-7, "sparse_attention_sink row " + std::to_string(r) + " head " + std::to_string(h) + " value " + std::to_string(d));
            }
        }
    for (size_t r = 0; r < rows; ++r)
        require(!std::memcmp(run(r, 1).data(), got.data() + r * heads * dim, heads * dim * 4), "sparse_attention_sink of row " + std::to_string(r) + " alone");
    for (int t : kThreads) {
        cpu.set_threads(t);
        require(same_bits(run(0, rows), got), "sparse_attention_sink at " + std::to_string(t) + " threads");
    }
    std::vector<uint32_t> bad(k, kNone);
    bad[3] = uint32_t(kv_rows);
    BufferPtr badb = upload(cpu, as_floats(bad)), ob = cpu.alloc(heads * dim * 4, backend::Memory::device);
    refused([&] { cpu.sparse_attention_sink({ob.get(), 0}, {qb.get(), 0}, {kb.get(), 0}, {badb.get(), 0}, {sb.get(), 0}, 1, heads, dim, kv_rows, k, scale, 0, nullptr, 0, {}, 0); }, "sparse_attention_sink of an id past its rows");
    refused([&] { cpu.sparse_attention_sink({qb.get(), 0}, {qb.get(), 0}, {kb.get(), 0}, {ib.get(), 0}, {sb.get(), 0}, 1, heads, dim, kv_rows, k, scale, 0, nullptr, 0, {}, 0); }, "sparse_attention_sink into its queries");
}

// compress_pool against model.py:473-482 in double: a softmax per channel over each group's rows, the weighted sum, then the compressor's RMSNorm.
void pooling() {
    const size_t dim = 40;
    const float eps = 1e-6f;
    for (size_t ratio : {1, 2, 4}) {
        const size_t groups = 5;
        const std::vector<float> kv = uniform(groups * ratio * dim, -2.0f, 2.0f), score = uniform(groups * ratio * dim, -8.0f, 8.0f), w = uniform(dim, 0.5f, 1.5f);
        CpuBackend cpu;
        BufferPtr kb = upload(cpu, kv), sb = upload(cpu, score), wb = upload(cpu, w);
        auto run = [&](size_t first, size_t n) {
            BufferPtr ob = cpu.alloc(n * dim * 4, backend::Memory::device);
            cpu.compress_pool({ob.get(), 0}, {kb.get(), first * ratio * dim}, {sb.get(), first * ratio * dim}, {wb.get(), 0}, n, ratio, dim, eps);
            return download(cpu, *ob, n * dim);
        };
        cpu.set_threads(1);
        const std::vector<float> got = run(0, groups);
        for (size_t g = 0; g < groups; ++g) {
            std::vector<double> pooled(dim), size(dim);
            double squares = 0.0;
            for (size_t d = 0; d < dim; ++d) {
                double peak = -1e300, sum = 0.0;
                for (size_t i = 0; i < ratio; ++i) peak = std::max(peak, double(score[(g * ratio + i) * dim + d]));
                for (size_t i = 0; i < ratio; ++i) sum += std::exp(score[(g * ratio + i) * dim + d] - peak);
                for (size_t i = 0; i < ratio; ++i) {
                    const double p = std::exp(score[(g * ratio + i) * dim + d] - peak) / sum;
                    pooled[d] += p * kv[(g * ratio + i) * dim + d];
                    size[d] += p * std::fabs(kv[(g * ratio + i) * dim + d]);
                }
                squares += pooled[d] * pooled[d];
            }
            const double rs = 1.0 / std::sqrt(squares / double(dim) + eps);
            for (size_t d = 0; d < dim; ++d) {
                const double want = pooled[d] * rs * w[d];
                require(std::fabs(got[g * dim + d] - want) <= (ratio + dim + 16) * U * (size[d] * rs * w[d] + std::fabs(want)), "compress_pool ratio " + std::to_string(ratio) + " group " + std::to_string(g) + " channel " + std::to_string(d));
            }
            require(!std::memcmp(run(g, 1).data(), got.data() + g * dim, dim * 4), "compress_pool of group " + std::to_string(g) + " alone");
        }
        for (int t : kThreads) {
            cpu.set_threads(t);
            require(same_bits(run(0, groups), got), "compress_pool at " + std::to_string(t) + " threads");
        }
        refused([&] { cpu.compress_pool({kb.get(), 0}, {kb.get(), 0}, {sb.get(), 0}, {wb.get(), 0}, groups, ratio, dim, eps); }, "compress_pool into its rows");
        refused([&] { cpu.compress_pool({kb.get(), 0}, {sb.get(), 0}, {sb.get(), 0}, {wb.get(), 0}, groups, 0, dim, eps); }, "compress_pool of groups of no rows");
    }
}

// Pooled rows as a DeepSeek KV source keeps them: groups of 4 positions, each row an entry of 24 floats then an index key of 32, in 128-token blocks whose table is out of order.
// Sequence A holds 300 positions on blocks 2, 0 and 1, so its groups cross two block edges; sequence B holds positions 10 to 16 on block 3, its groups 2 and 3.
struct Pooled {
    static constexpr size_t ratio = 4, entry = 24, key = 32, width = entry + key;
    CpuBackend cpu;
    std::unique_ptr<backend::KVStorage> owned = cpu.kv_alloc(1, 0, 0, 512, backend::KVType::f32, backend::KVType::f32, {0}, {{ratio, width}});
    backend::CpuKVStorage& storage = dynamic_cast<backend::CpuKVStorage&>(*owned);
    const std::vector<int32_t> a_blocks = {2, 0, 1}, b_blocks = {3};
    std::vector<float> a = uniform(75 * width, -1.0f, 1.0f), b = uniform(2 * width, -1.0f, 1.0f);
    backend::KVView view(bool on_a, size_t length, size_t nq) const {
        const std::vector<int32_t>& t = on_a ? a_blocks : b_blocks;
        return {&const_cast<backend::CpuKVStorage&>(storage), t.data(), t.size(), length, nq};
    }
    // Sequence B's rows of group g, which only groups 2 and 3 have.
    const float* row(bool on_a, size_t g) const { return on_a ? &a[g * width] : &b[(g - 2) * width]; }
};

// pool_write stores every view's completed groups in one call, a view completing none taking no row, each at its one address, which the readers below share.
void pool_store(Pooled& p) {
    std::vector<float> rows(p.a);
    rows.insert(rows.end(), p.b.begin(), p.b.end());
    const backend::KVView views[] = {p.view(true, 0, 300), p.view(false, 5, 2), p.view(false, 10, 7)};
    BufferPtr rb = upload(p.cpu, rows);
    p.cpu.pool_write(0, views, 3, {rb.get(), 0});
    for (size_t g = 0; g < 75; ++g)
        require(!std::memcmp(p.storage.pooled_row(0, views[0], g), p.row(true, g), Pooled::width * 4), "pool_write's row of group " + std::to_string(g) + " of the first view");
    for (size_t g = 2; g < 4; ++g)
        require(!std::memcmp(p.storage.pooled_row(0, views[2], g), p.row(false, g), Pooled::width * 4), "pool_write's row of group " + std::to_string(g) + " of the third view");
    require(p.storage.pooled_row(0, views[0], 32) == (const float*)((const uint8_t*)p.storage.pooled_row(0, views[0], 0) - 2 * p.storage.auxiliary_block_bytes(0)),
            "group 32 opens the second block of the table, block 0, two blocks before block 2");
    refused([&] { p.cpu.pool_write(0, views, 3, {rb.get(), Pooled::width}); }, "pool_write of rows past its buffer");
    std::unique_ptr<backend::KVStorage> plain = p.cpu.kv_alloc(1, 1, 8, 512);
    const backend::KVView none{plain.get(), p.a_blocks.data(), 3, 0, 8};
    refused([&] { p.cpu.pool_write(0, &none, 1, {rb.get(), 0}); }, "pool_write to a layer without pooled rows");
}

// sparse_attention_sink with the entries as a second source in double: rows naming window ids alone, entries alone, both and none, over two views in one call, the entries the first 24 floats of the rows the index keys follow.
void sink_entries(Pooled& p) {
    const size_t heads = 2, dim = Pooled::entry, kv_rows = 50, k = 6, ek = 5;
    const std::vector<backend::KVView> views = {p.view(true, 290, 6), p.view(false, 15, 2)};
    const size_t rows = 8;
    std::vector<uint32_t> ids(rows * k, kNone), entries(rows * ek, kNone);
    for (size_t r = 0; r < rows; ++r) {
        const bool on_a = r < 6;
        const size_t visible = (on_a ? 290 + r + 1 : 15 + r - 6 + 1) / Pooled::ratio;
        if (r != 1 && r != 3)
            for (size_t t = 0; t < k; ++t)
                if (draw() % 4) ids[r * k + t] = draw() % kv_rows;
        if (r != 0 && r != 3)
            for (size_t t = 0; t < ek; ++t)
                if (draw() % 4) entries[r * ek + t] = uint32_t(on_a ? draw() % visible : 2 + draw() % 2);
    }
    entries[2 * ek] = uint32_t(32);
    entries[2 * ek + 1] = uint32_t(31);
    const std::vector<float> q = uniform(rows * heads * dim, -1.0f, 1.0f), kv = uniform(kv_rows * dim, -1.0f, 1.0f), sink = {0.25f, -1.0f};
    const float scale = 1.0f / std::sqrt(float(dim));
    CpuBackend& cpu = p.cpu;
    BufferPtr qb = upload(cpu, q), kb = upload(cpu, kv), ib = upload(cpu, as_floats(ids)), eb = upload(cpu, as_floats(entries)), sb = upload(cpu, sink);
    auto run = [&](const std::vector<backend::KVView>& v, size_t first, size_t n) {
        BufferPtr ob = cpu.alloc(n * heads * dim * 4, backend::Memory::device);
        cpu.sparse_attention_sink({ob.get(), 0}, {qb.get(), first * heads * dim}, {kb.get(), 0}, {ib.get(), first * k}, {sb.get(), 0}, n, heads, dim, kv_rows, k, scale, 0,
                                  v.data(), v.size(), {eb.get(), first * ek}, ek);
        return download(cpu, *ob, n * heads * dim);
    };
    cpu.set_threads(1);
    const std::vector<float> got = run(views, 0, rows);
    for (size_t r = 0; r < rows; ++r)
        for (size_t h = 0; h < heads; ++h) {
            std::vector<const float*> named;
            for (size_t t = 0; t < k; ++t)
                if (ids[r * k + t] != kNone) named.push_back(&kv[ids[r * k + t] * dim]);
            for (size_t t = 0; t < ek; ++t)
                if (entries[r * ek + t] != kNone) named.push_back(p.row(r < 6, entries[r * ek + t]));
            std::vector<double> sc;
            double peak = -1e30;
            for (const float* row : named) {
                double dot = 0.0;
                for (size_t d = 0; d < dim; ++d) dot += double(q[(r * heads + h) * dim + d]) * row[d];
                sc.push_back(dot * scale);
                peak = std::max(peak, sc.back());
            }
            double denominator = std::exp(double(sink[h]) - peak);
            for (double v : sc) denominator += std::exp(v - peak);
            for (size_t d = 0; d < dim; ++d) {
                double want = 0.0, size = 0.0;
                for (size_t t = 0; t < sc.size(); ++t) {
                    want += std::exp(sc[t] - peak) * named[t][d] / denominator;
                    size += std::exp(sc[t] - peak) * std::fabs(named[t][d]) / denominator;
                }
                const float o = got[(r * heads + h) * dim + d];
                if (sc.empty()) require(bits(o) == 0, "sparse_attention_sink of a row naming no window row and no entry is +0");
                else require(std::fabs(o - want) <= 1e-5 * size + 1e-7, "sparse_attention_sink with entries row " + std::to_string(r) + " head " + std::to_string(h) + " value " + std::to_string(d));
            }
        }
    for (size_t r = 0; r < rows; ++r) {
        const std::vector<backend::KVView> alone = {r < 6 ? p.view(true, 290 + r, 1) : p.view(false, 15 + r - 6, 1)};
        require(!std::memcmp(run(alone, r, 1).data(), got.data() + r * heads * dim, heads * dim * 4), "sparse_attention_sink with entries of row " + std::to_string(r) + " alone");
    }
    for (int t : kThreads) {
        cpu.set_threads(t);
        require(same_bits(run(views, 0, rows), got), "sparse_attention_sink with entries at " + std::to_string(t) + " threads");
    }
    std::vector<uint32_t> past(ek, kNone);
    past[1] = uint32_t((290 + 1) / Pooled::ratio);
    BufferPtr pb = upload(cpu, as_floats(past)), nb = upload(cpu, as_floats(std::vector<uint32_t>(rows * k, kNone)));
    BufferPtr ob = cpu.alloc(rows * Pooled::width * heads * 4, backend::Memory::device);
    const backend::KVView first = p.view(true, 290, 1);
    refused([&] { cpu.sparse_attention_sink({ob.get(), 0}, {qb.get(), 0}, {kb.get(), 0}, {ib.get(), 0}, {sb.get(), 0}, 1, heads, dim, kv_rows, k, scale, 0, &first, 1, {pb.get(), 0}, ek); },
            "sparse_attention_sink of an entry its row cannot see yet");
    refused([&] { cpu.sparse_attention_sink({ob.get(), 0}, {qb.get(), 0}, {kb.get(), 0}, {nb.get(), 0}, {sb.get(), 0}, 2, heads, dim, kv_rows, k, scale, 0, &first, 1, {eb.get(), 0}, ek); },
            "sparse_attention_sink of more rows than its views hold");
    refused([&] { cpu.sparse_attention_sink({ob.get(), 0}, {qb.get(), 0}, {kb.get(), 0}, {nb.get(), 0}, {sb.get(), 0}, 1, heads, Pooled::width + 1, 1, k, scale, 0, &first, 1, {nb.get(), 0}, ek); },
            "sparse_attention_sink of entries wider than the pooled rows");
    refused([&] { cpu.sparse_attention_sink({ob.get(), 0}, {qb.get(), 0}, {kb.get(), 0}, {ib.get(), 0}, {sb.get(), 0}, 1, heads, dim, kv_rows, k, scale, 0, nullptr, 0, {eb.get(), 0}, ek); },
            "sparse_attention_sink of entries without views");
}

// indexer_scores against model.py:557-571 in double: the sum over heads of w w_scale ReLU(q . k), the keys the last 32 floats of the views' pooled rows, -inf at or past each row's groups and outside its candidate blocks.
void index_scores(Pooled& p) {
    const float inf = std::numeric_limits<float>::infinity(), w_scale = 0.375f;
    const size_t heads = 4, dim = Pooled::key, n = 40, bs = 8, bk = 2;
    const std::vector<backend::KVView> views = {p.view(true, 120, 10), p.view(false, 13, 3)};
    const size_t rows = 13;
    const std::vector<float> q = uniform(rows * heads * dim, -1.0f, 1.0f), w = uniform(rows * heads, -0.5f, 0.5f);
    std::vector<uint32_t> cand(rows * bk, kNone);
    for (size_t i = 0; i < cand.size(); ++i)
        if (draw() % 3) cand[i] = draw() % 5;
    CpuBackend& cpu = p.cpu;
    BufferPtr qb = upload(cpu, q), wb = upload(cpu, w), cb = upload(cpu, as_floats(cand));
    for (bool with_blocks : {false, true}) {
        auto run = [&](const std::vector<backend::KVView>& v, size_t first) {
            size_t m = 0;
            for (const backend::KVView& x : v) m += x.nq;
            BufferPtr ob = cpu.alloc(m * n * 4, backend::Memory::device);
            const backend::CSlice blocks = with_blocks ? backend::CSlice{cb.get(), first * bk} : backend::CSlice{};
            cpu.indexer_scores({ob.get(), 0}, {qb.get(), first * heads * dim}, 0, v.data(), v.size(), {wb.get(), first * heads}, w_scale, heads, dim, n, blocks, bk, bs);
            return download(cpu, *ob, m * n);
        };
        cpu.set_threads(1);
        const std::vector<float> got = run(views, 0);
        for (size_t r = 0; r < rows; ++r) {
            const bool on_a = r < 10;
            const size_t position = on_a ? 120 + r : 13 + r - 10, len = (position + 1) / Pooled::ratio;
            for (size_t t = 0; t < n; ++t) {
                bool reached = t < len;
                if (with_blocks) reached = reached && (cand[r * bk] == t / bs || cand[r * bk + 1] == t / bs);
                const float o = got[r * n + t];
                if (!reached) {
                    require(o == -inf, "indexer_scores of an unreached position is -inf");
                    continue;
                }
                const float zeros[Pooled::width] = {};
                const float* key = (on_a ? p.row(true, t) : t >= 2 ? p.row(false, t) : zeros) + Pooled::entry;
                double want = 0.0, size = 0.0;
                for (size_t h = 0; h < heads; ++h) {
                    double dot = 0.0, mag = 0.0;
                    for (size_t d = 0; d < dim; ++d) {
                        dot += double(q[(r * heads + h) * dim + d]) * key[d];
                        mag += std::fabs(double(q[(r * heads + h) * dim + d]) * key[d]);
                    }
                    want += std::max(dot, 0.0) * w[r * heads + h] * w_scale;
                    size += mag * std::fabs(w[r * heads + h] * w_scale);
                }
                require(std::fabs(o - want) <= (dim + heads + 4) * U * size, "indexer_scores row " + std::to_string(r) + " position " + std::to_string(t));
            }
            const std::vector<backend::KVView> alone = {on_a ? p.view(true, position, 1) : p.view(false, position, 1)};
            require(!std::memcmp(run(alone, r).data(), got.data() + r * n, n * 4), "indexer_scores of row " + std::to_string(r) + " alone");
        }
        for (int t : kThreads) {
            cpu.set_threads(t);
            require(same_bits(run(views, 0), got), "indexer_scores at " + std::to_string(t) + " threads");
        }
    }
    BufferPtr ob = cpu.alloc(rows * n * 4, backend::Memory::device);
    refused([&] { cpu.indexer_scores({qb.get(), 0}, {qb.get(), 0}, 0, views.data(), 1, {wb.get(), 0}, w_scale, heads, dim, n, {}, 0, 0); }, "indexer_scores into its queries");
    refused([&] { cpu.indexer_scores({ob.get(), 0}, {qb.get(), 0}, 0, views.data(), 1, {wb.get(), 0}, w_scale, heads, dim, n, {cb.get(), 0}, bk, 0); }, "indexer_scores of blocks of no positions");
    refused([&] { cpu.indexer_scores({ob.get(), 0}, {qb.get(), 0}, 0, views.data(), 1, {wb.get(), 0}, w_scale, heads, dim, 31, {}, 0, 0); }, "indexer_scores narrower than a row's groups");
    refused([&] { cpu.indexer_scores({ob.get(), 0}, {qb.get(), 0}, 0, views.data(), 1, {wb.get(), 0}, w_scale, heads, Pooled::width + 1, n, {}, 0, 0); },
            "indexer_scores of keys wider than the pooled rows");
}


// route_experts' sqrt_softplus rule against model.py:811-829 in double: scores sqrt(softplus(x)) with torch's threshold of 20, picks by score + bias, ties to the lower id, weights the unbiased scores over their sum + 1e-20 times the scale.
void routing() {
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    const float level[] = {-inf, -30.0f, -2.5f, -0.0f, 0.0f, 0.75f, 3.0f, 19.5f, 20.0f, 20.5f, 31.0f};
    const size_t n_expert = 24, rows = 9;
    std::vector<float> x(rows * n_expert), bias(n_expert);
    for (float& v : x) v = level[draw() % 11];
    for (float& v : bias) v = float(int(draw() % 9) - 4) / 8.0f;
    for (size_t e = 0; e < n_expert; ++e) x[n_expert + e] = 1.0f;
    x[2 * n_expert + 5] = nan;
    x[3 * n_expert + 7] = inf;
    CpuBackend cpu;
    BufferPtr xb = upload(cpu, x), bb = upload(cpu, bias);
    for (size_t k : {1, 6, 24})
        for (bool with_bias : {false, true})
            for (bool normalize : {false, true}) {
                const float scale = 1.5f;
                const backend::RouteRule rule = backend::RouteRule::sqrt_softplus(with_bias ? backend::CSlice{bb.get(), 0} : backend::CSlice{}, scale, normalize);
                auto run = [&](size_t first, size_t n) {
                    BufferPtr ib = cpu.alloc(n * k * 4, backend::Memory::device), wb = cpu.alloc(n * k * 4, backend::Memory::device);
                    cpu.route_experts({xb.get(), first * n_expert}, n, n_expert, k, rule, {ib.get(), 0}, {wb.get(), 0});
                    std::vector<float> out = download(cpu, *ib, n * k), w = download(cpu, *wb, n * k);
                    out.insert(out.end(), w.begin(), w.end());
                    return out;
                };
                cpu.set_threads(1);
                const std::vector<float> got = run(0, rows);
                const std::string name = "route_experts sqrt_softplus k " + std::to_string(k) + (with_bias ? " with a bias" : "") + (normalize ? " normalized" : "");
                for (size_t r = 0; r < rows; ++r) {
                    std::vector<double> s(n_expert);
                    std::vector<float> key(n_expert);
                    bool finite = true;
                    for (size_t e = 0; e < n_expert; ++e) {
                        const double v = x[r * n_expert + e];
                        s[e] = std::sqrt(v > 20.0 ? v : std::log1p(std::exp(v)));
                        const double b = s[e] + (with_bias ? bias[e] : 0.0);
                        finite = finite && std::isfinite(b);
                        key[e] = float(b);
                    }
                    const std::vector<uint32_t> order = ranked(key.data(), n_expert);
                    double total = 0.0;
                    for (size_t j = 0; j < k; ++j) total += s[order[j]];
                    for (size_t j = 0; j < k; ++j) {
                        const uint32_t id = bits(got[r * k + j]);
                        const float w = got[rows * k + r * k + j];
                        if (!finite) {
                            require(id == kNone && bits(w) == 0, name + " row " + std::to_string(r) + " without a candidate");
                            continue;
                        }
                        require(id == order[j], name + " row " + std::to_string(r) + " pick " + std::to_string(j));
                        const double want = (normalize && k > 1 ? s[order[j]] / (total + 1e-20) : s[order[j]]) * scale;
                        require(std::fabs(w - want) <= (k + 8) * U * std::fabs(want), name + " row " + std::to_string(r) + " weight " + std::to_string(j));
                    }
                    const std::vector<float> one = run(r, 1);
                    require(!std::memcmp(one.data(), got.data() + r * k, k * 4) && !std::memcmp(one.data() + k, got.data() + rows * k + r * k, k * 4), name + " of row " + std::to_string(r) + " alone");
                }
                for (int t : kThreads) {
                    cpu.set_threads(t);
                    require(same_bits(run(0, rows), got), name + " at " + std::to_string(t) + " threads");
                }
            }
    BufferPtr ib = cpu.alloc(n_expert * 4, backend::Memory::device), wb = cpu.alloc(n_expert * 4, backend::Memory::device);
    const backend::RouteRule plain = backend::RouteRule::sqrt_softplus({}, 1.5f, true);
    refused([&] { cpu.route_experts({xb.get(), 0}, 1, n_expert, 0, plain, {ib.get(), 0}, {wb.get(), 0}); }, "route_experts picking no expert");
    refused([&] { cpu.route_experts({xb.get(), 0}, 1, n_expert, n_expert + 1, plain, {ib.get(), 0}, {wb.get(), 0}); }, "route_experts picking more experts than exist");
    refused([&] { cpu.route_experts({xb.get(), 0}, 1, n_expert, 2, backend::RouteRule::sqrt_softplus({}, inf, true), {ib.get(), 0}, {wb.get(), 0}); }, "route_experts at a scale that is not finite");
    backend::RouteRule biased = backend::RouteRule::softmax(true);
    biased.bias = {bb.get(), 0};
    refused([&] { cpu.route_experts({xb.get(), 0}, 1, n_expert, 2, biased, {ib.get(), 0}, {wb.get(), 0}); }, "route_experts under softmax with a bias");
    refused([&] { backend::RouteRule::sqrt_softplus({}, 1.0f, true).softmax_only("device"); }, "a device without sqrt_softplus scores given them");
}

// silu_mul's clamp against model.py:847-850 in double: silu(min(g, L)) * clamp(u, -L, L), a NaN passing through, and a limit of 0 clamping nothing.
void swiglu_clamp() {
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    const size_t rows = 6, width = 37, n = rows * width;
    std::vector<float> g = uniform(n, -30.0f, 30.0f), u = uniform(n, -30.0f, 30.0f);
    g[3] = nan; u[4] = nan; g[5] = inf; g[6] = -inf; u[7] = inf; u[8] = -inf; g[9] = 10.0f; u[9] = -10.0f; g[10] = 1.0f; u[10] = 1.0f;
    CpuBackend cpu;
    BufferPtr gb = upload(cpu, g), ub = upload(cpu, u);
    for (float limit : {0.0f, 1.0f, 10.0f}) {
        auto run = [&](size_t first, size_t m) {
            BufferPtr ob = cpu.alloc(m * width * 4, backend::Memory::device);
            cpu.silu_mul({ob.get(), 0}, {gb.get(), first * width}, {ub.get(), first * width}, m * width, limit);
            return download(cpu, *ob, m * width);
        };
        cpu.set_threads(1);
        const std::vector<float> got = run(0, rows);
        const std::string name = "silu_mul at limit " + std::to_string(limit);
        for (size_t i = 0; i < n; ++i) {
            double gi = g[i], ui = u[i];
            if (limit > 0.0f) {
                if (gi > limit) gi = limit;
                if (ui < -limit) ui = -limit;
                if (ui > limit) ui = limit;
            }
            const double want = gi / (1.0 + std::exp(-gi)) * ui;
            if (std::isnan(want)) {
                require(std::isnan(got[i]), name + " element " + std::to_string(i) + " is NaN");
                continue;
            }
            require(got[i] == want || std::fabs(got[i] - want) <= 8 * U * std::fabs(want), name + " element " + std::to_string(i));
        }
        for (size_t r = 0; r < rows; ++r) require(!std::memcmp(run(r, 1).data(), got.data() + r * width, width * 4), name + " of row " + std::to_string(r) + " alone");
        for (int t : kThreads) {
            cpu.set_threads(t);
            require(same_bits(run(0, rows), got), name + " at " + std::to_string(t) + " threads");
        }
    }
    BufferPtr ob = cpu.alloc(n * 4, backend::Memory::device);
    for (float limit : {-1.0f, nan, inf})
        refused([&] { cpu.silu_mul({ob.get(), 0}, {gb.get(), 0}, {ub.get(), 0}, n, limit); }, "silu_mul at limit " + std::to_string(limit));
}

// norm_rope_partial's forms against model.py:392-406 in double: the head's RMSNorm, then pairs from the offset, adjacent as complex numbers or split in halves, turned by the angle or by minus it.
void rope_forms() {
    const size_t rows = 5, heads = 3, head_dim = 48, rope_dim = 16, half = rope_dim / 2, npos = 300;
    const float eps = 1e-6f;
    std::vector<float> cos(npos * half), sin(npos * half);
    for (size_t p = 0; p < npos; ++p)
        for (size_t j = 0; j < half; ++j) {
            const double angle = double(p) * std::pow(10000.0, -double(2 * j) / double(rope_dim));
            cos[p * half + j] = float(std::cos(angle));
            sin[p * half + j] = float(std::sin(angle));
        }
    const std::vector<uint32_t> pos = {299, 0, 17, 128, 5};
    const size_t stride = heads * head_dim + 8, head_stride = head_dim + 2;
    const std::vector<float> x = uniform(rows * stride, -3.0f, 3.0f), w = uniform(head_dim, 0.5f, 1.5f);
    CpuBackend cpu;
    BufferPtr xb = upload(cpu, x), wb = upload(cpu, w), cb = upload(cpu, cos), sb = upload(cpu, sin);
    for (bool adjacent : {false, true})
        for (size_t offset : {size_t(0), size_t(8), head_dim - rope_dim})
            for (bool inverse : {false, true})
                for (bool norm : {false, true}) {
                    const backend::RopeForm form{adjacent, offset, inverse};
                    if (form.plain()) continue;
                    const backend::CSlice wn = norm ? backend::CSlice{wb.get(), 0} : backend::CSlice{};
                    auto run = [&](size_t first, size_t m) {
                        BufferPtr ob = cpu.alloc(m * heads * head_dim * 4, backend::Memory::device);
                        cpu.norm_rope_partial({ob.get(), 0}, {xb.get(), first * stride}, m, stride, head_stride, heads, head_dim, rope_dim, wn, eps,
                                              {cb.get(), 0}, {sb.get(), 0}, pos.data() + first, form);
                        return download(cpu, *ob, m * heads * head_dim);
                    };
                    cpu.set_threads(1);
                    const std::vector<float> got = run(0, rows);
                    const std::string name = std::string("norm_rope_partial ") + (adjacent ? "adjacent" : "halves") + " from " + std::to_string(offset) +
                                             (inverse ? " inverse" : "") + (norm ? " with a norm" : "");
                    for (size_t r = 0; r < rows; ++r) {
                        for (size_t h = 0; h < heads; ++h) {
                            const float* in = x.data() + r * stride + h * head_stride;
                            std::vector<double> v(in, in + head_dim), mag(head_dim);
                            double rs = 1.0;
                            if (norm) {
                                double squares = 0.0;
                                for (double e : v) squares += e * e;
                                rs = 1.0 / std::sqrt(squares / double(head_dim) + eps);
                                for (size_t d = 0; d < head_dim; ++d) v[d] *= rs * w[d];
                            }
                            for (size_t d = 0; d < head_dim; ++d) mag[d] = std::fabs(v[d]);
                            for (size_t j = 0; j < half; ++j) {
                                const size_t a = offset + (adjacent ? 2 * j : j), b = offset + (adjacent ? 2 * j + 1 : j + half);
                                const double c = cos[pos[r] * half + j], s = (inverse ? -1.0 : 1.0) * sin[pos[r] * half + j];
                                const double va = v[a], vb = v[b];
                                v[a] = va * c - vb * s;
                                v[b] = va * s + vb * c;
                                mag[a] = mag[b] = std::fabs(va) + std::fabs(vb);
                            }
                            for (size_t d = 0; d < head_dim; ++d)
                                require(std::fabs(got[(r * heads + h) * head_dim + d] - v[d]) <= (head_dim + 8) * U * (mag[d] + std::fabs(v[d])),
                                        name + " row " + std::to_string(r) + " head " + std::to_string(h) + " float " + std::to_string(d));
                        }
                        require(!std::memcmp(run(r, 1).data(), got.data() + r * heads * head_dim, heads * head_dim * 4), name + " of row " + std::to_string(r) + " alone");
                    }
                    for (int t : kThreads) {
                        cpu.set_threads(t);
                        require(same_bits(run(0, rows), got), name + " at " + std::to_string(t) + " threads");
                    }
                    // In place over contiguous heads, the same bits as into another buffer.
                    std::vector<float> packed(rows * heads * head_dim);
                    for (size_t r = 0; r < rows; ++r)
                        for (size_t h = 0; h < heads; ++h) std::memcpy(&packed[(r * heads + h) * head_dim], &x[r * stride + h * head_stride], head_dim * 4);
                    BufferPtr pb = upload(cpu, packed);
                    cpu.norm_rope_partial({pb.get(), 0}, {pb.get(), 0}, rows, heads * head_dim, head_dim, heads, head_dim, rope_dim, wn, eps,
                                          {cb.get(), 0}, {sb.get(), 0}, pos.data(), form);
                    require(same_bits(download(cpu, *pb, rows * heads * head_dim), got), name + " in place");
                }
    BufferPtr ob = cpu.alloc(rows * heads * head_dim * 4, backend::Memory::device);
    refused([&] { cpu.norm_rope_partial({ob.get(), 0}, {xb.get(), 0}, rows, stride, head_stride, heads, head_dim, rope_dim, {wb.get(), 0}, eps,
                                        {cb.get(), 0}, {sb.get(), 0}, pos.data(), backend::RopeForm{true, head_dim - rope_dim + 2, false}); }, "norm_rope_partial rotating past the head");
}

// The grouped output projection against model.py:787-790 in double: out[r][g * rank + o] = sum over d of wo_a[g][o][d] x[r][g][d], one matrix read as its groups' stacks.
void grouped_projection() {
    const size_t rows = 5, groups = 4, rank = 8, nin = 64, nout = groups * rank;
    const std::vector<float> x = uniform(rows * groups * nin, -2.0f, 2.0f), w = uniform(nout * nin, -1.0f, 1.0f);
    std::vector<uint32_t> ids(rows * groups);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = uint32_t(i % groups);
    std::vector<uint8_t> q8(nout * (nin / 32) * 34);
    for (size_t o = 0; o < nout; ++o) quant::quantize_row_q8_0(&w[o * nin], &q8[o * (nin / 32) * 34], nin / 32);
    std::vector<float> wq(nout * nin);
    for (size_t o = 0; o < nout; ++o) quant::dequantize_row_q8_0(&q8[o * (nin / 32) * 34], &wq[o * nin], nin / 32);
    CpuBackend cpu;
    BufferPtr xb = upload(cpu, x), wb = upload(cpu, w), ib = upload(cpu, as_floats(ids));
    BufferPtr qb = cpu.alloc(q8.size(), backend::Memory::device);
    cpu.write(*qb, 0, q8.data(), q8.size());
    std::vector<backend::RowRun> scratch;
    for (bool quantized : {false, true})
        for (bool decode : {false, true}) {
            const backend::WeightView view{quantized ? uint32_t(quant::GGML_TYPE_Q8_0) : uint32_t(quant::GGML_TYPE_F32), {quantized ? qb.get() : wb.get(), 0}};
            const std::vector<float>& ref = quantized ? wq : w;
            auto run = [&](size_t first, size_t m) {
                std::vector<backend::RowRun> runs;
                for (size_t r = 0; r < m; ++r)
                    if (decode) runs.push_back({r + 1, 1});
                if (!decode) runs.push_back({m, m});
                BufferPtr ob = cpu.alloc(m * nout * 4, backend::Memory::device);
                cpu.matmul_grouped(backend::Projection(view, {ob.get(), 0}, rank), {xb.get(), first * groups * nin}, nin, m, groups, {ib.get(), 0},
                                   {runs.data(), runs.size()}, backend::Dtype::f32, scratch);
                return download(cpu, *ob, m * nout);
            };
            cpu.set_threads(1);
            const std::vector<float> got = run(0, rows);
            const std::string name = std::string("matmul_grouped ") + (quantized ? "Q8_0" : "F32") + (decode ? " decode rows" : " a prompt");
            for (size_t r = 0; r < rows; ++r) {
                for (size_t g = 0; g < groups; ++g)
                    for (size_t o = 0; o < rank; ++o) {
                        double want = 0.0, size = 0.0;
                        for (size_t d = 0; d < nin; ++d) {
                            const double p = double(ref[(g * rank + o) * nin + d]) * x[(r * groups + g) * nin + d];
                            want += p;
                            size += std::fabs(p);
                        }
                        require(std::fabs(got[r * nout + g * rank + o] - want) <= (nin + 8) * U * size, name + " row " + std::to_string(r) + " group " + std::to_string(g) + " output " + std::to_string(o));
                    }
                if (decode) require(!std::memcmp(run(r, 1).data(), got.data() + r * nout, nout * 4), name + " of row " + std::to_string(r) + " alone");
            }
            for (int t : kThreads) {
                cpu.set_threads(t);
                require(same_bits(run(0, rows), got), name + " at " + std::to_string(t) + " threads");
            }
        }
    std::vector<uint32_t> past = ids;
    past[3] = uint32_t(groups);
    BufferPtr pb = upload(cpu, as_floats(past)), ob = cpu.alloc(rows * nout * 4, backend::Memory::device);
    const backend::RowRun one{rows, rows};
    refused([&] { cpu.matmul_grouped(backend::Projection(backend::WeightView{uint32_t(quant::GGML_TYPE_F32), {wb.get(), 0}}, {ob.get(), 0}, rank), {xb.get(), 0}, nin, rows, groups,
                                     {pb.get(), 0}, {&one, 1}, backend::Dtype::f32, scratch); }, "matmul_grouped naming a group past the matrix");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: llmx-deepseek4-ops-test CODECS");
        codes();
        scales();
        blocks();
        format_op();
        port(argv[1]);
        mhc();
        selections();
        sink_attention();
        pooling();
        {
            Pooled pooled;
            pool_store(pooled);
            sink_entries(pooled);
            index_scores(pooled);
        }
        routing();
        swiglu_clamp();
        rope_forms();
        grouped_projection();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "deepseek4-ops: " << checked << " checks PASS" << std::endl;
    return 0;
}
