// A fused K-quant dot sums q*x and applies the block scale afterwards, so a large activation can drive the sum past FLT_MAX where dequantizing first stays finite (d*Inf is Inf, 0*Inf is NaN).
// Each type's weights are one repeated value with unit group scales, so the exact answer is known in closed form; the value is the type's largest magnitude, 127 for Q8_0, 15 for Q4_K, 31 for Q5_K and Q6_K.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/cpu/cpu_backend.hpp"

namespace {

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

void put16(std::vector<uint8_t>& v, size_t at, uint16_t x) {
    v[at] = (uint8_t)(x & 0xFF);
    v[at + 1] = (uint8_t)(x >> 8);
}

// Q8_0 for contrast.
// Its kernel folds the scale into each weight before the activation, (q*d)*x, rather than accumulating sum(q*x) first, so it should have no overflow window at all.
// Included to verify that by measurement rather than by reading the kernel, and to catch it if that ever changes.
// Eight 32-value blocks, every weight 127.
std::vector<uint8_t> block_q8_0(uint16_t half) {
    std::vector<uint8_t> b(8 * gguf::Q8_0_TYPESIZE, 0);
    for (int blk = 0; blk < 8; blk++) {
        uint8_t* p = b.data() + blk * gguf::Q8_0_TYPESIZE;
        p[0] = (uint8_t)(half & 0xFF);
        p[1] = (uint8_t)(half >> 8);
        for (int i = 0; i < 32; i++) p[2 + i] = 127;
    }
    return b;
}

// Q4_K: every nibble 15, unit group scales, zero mins, so each weight decodes to 15.
// Same 144-byte layout as the dequantizer expects.
std::vector<uint8_t> block_q4_K(uint16_t half) {
    std::vector<uint8_t> b(gguf::Q4_K_TYPESIZE, 0);
    put16(b, 0, half);          // d
    put16(b, 2, 0);             // dmin
    for (int g = 0; g < 8; g++) {   // 6-bit scale 1, min 0, all 8 groups
        if (g < 4) b[4 + g] = 1; else { b[4 + 8 + (g - 4)] = 1; }
    }
    for (int i = 0; i < 128; i++) b[16 + i] = 0xFF;  // every nibble 15
    return b;
}

// One 256-value super-block whose every decoded weight is 31, with the given half-precision super-block scale.
std::vector<uint8_t> block_q5_K(uint16_t half) {
    std::vector<uint8_t> b(gguf::Q5_K_TYPESIZE, 0);
    put16(b, 0, half);          // d
    put16(b, 2, 0);             // dmin: no min contribution
    for (int g = 0; g < 8; g++) {   // 6-bit scale 1, min 0, for all 8 groups
        if (g < 4) b[4 + g] = 1; else { b[4 + 8 + (g - 4)] = 1; }
    }
    for (int i = 0; i < 32; i++) b[16 + i] = 0xFF;   // every high bit set: +16
    for (int i = 0; i < 128; i++) b[48 + i] = 0xFF;  // every nibble 15
    return b;                                        // 15 + 16 = 31
}

std::vector<uint8_t> block_q6_K(uint16_t half) {
    std::vector<uint8_t> b(gguf::Q6_K_TYPESIZE, 0);
    for (int i = 0; i < 128; i++) b[i] = 0xFF;       // low nibbles 15
    for (int i = 0; i < 64; i++) b[128 + i] = 0xFF;  // high 2-bit pairs = 3
    for (int i = 0; i < 16; i++) b[192 + i] = 1;     // group scales 1
    put16(b, 208, half);                             // d
    return b;                                        // (15|3<<4) - 32 = 31
}

struct Case { const char* name; uint16_t half; bool huge; };

// `eight` runs the decode dots over 8-bit activations, which cannot overflow since each block is scaled first; their bound is relative to the sum of magnitudes, since a block's rounding follows its largest value and the ordinary input's products largely cancel.
int run_type(uint32_t type, const char* tname, bool eight) {
    const size_t nin = 256;
    backend::CpuBackend cpu;
    cpu.set_threads(1);
    cpu.set_decode_activations8(eight);

    const Case cases[] = {
        {"tiny scale, huge input", 0x0001, true},
        {"zero scale, huge input", 0x0000, true},
        {"ordinary scale/input",   0x3555, false},
        {"tiny scale, ordinary",   0x0001, false},
    };
    int checked = 0;
    for (const auto& c : cases) {
        const float q = type == gguf::GGML_TYPE_Q8_0 ? 127.0f
                      : type == gguf::GGML_TYPE_Q4_K ? 15.0f : 31.0f;
        std::vector<uint8_t> w = type == gguf::GGML_TYPE_Q8_0 ? block_q8_0(c.half)
                               : type == gguf::GGML_TYPE_Q4_K ? block_q4_K(c.half)
                               : type == gguf::GGML_TYPE_Q5_K ? block_q5_K(c.half)
                                                              : block_q6_K(c.half);
        std::vector<float> x(nin), y(1, 0.0f);
        for (size_t i = 0; i < nin; i++)
            x[i] = c.huge ? std::ldexp(1.0f, 123)
                          : (float)((int)(i * 17 % 37) - 18) / 37.0f;

        // Every weight decodes to the same constant, so the dot is q*d*sum(x).
        const float d = f16_to_f32(c.half);
        long double exact = 0.0L, magnitude = 0.0L;
        for (size_t i = 0; i < nin; i++) {
            exact += (long double)q * d * x[i];
            magnitude += std::fabs((long double)q * d * x[i]);
        }

        const auto w_buf = cpu.adopt(w.data(), w.size());
        const auto x_buf = cpu.adopt(x.data(), x.size() * sizeof(float));
        const auto y_buf = cpu.adopt(y.data(), y.size() * sizeof(float));
        cpu.matmul(type, {w_buf.get(), 0}, {x_buf.get(), 0}, {y_buf.get(), 0}, nin, 1, 1);

        require(std::isfinite(y[0]),
                std::string(tname) + " / " + c.name + ": produced a nonfinite result");
        const long double err = std::fabs((long double)y[0] - exact);
        const long double tol = eight ? magnitude * 1e-2L + 1e-30L : std::fabs(exact) * 1e-5L + 1e-30L;
        require(err <= tol, std::string(tname) + (eight ? " 8-bit" : "") + " / " + c.name +
                            ": " + std::to_string((double)y[0]) +
                            " differs from " + std::to_string((double)exact));
        checked++;
    }
    return checked;
}

}  // namespace

int main() {
    try {
        quant::register_builtins();
        int n = 0;
        for (bool eight : {false, true}) {
            n += run_type(gguf::GGML_TYPE_Q8_0, "Q8_0", eight);
            n += run_type(gguf::GGML_TYPE_Q4_K, "Q4_K", eight);
            n += run_type(gguf::GGML_TYPE_Q5_K, "Q5_K", eight);
            n += run_type(gguf::GGML_TYPE_Q6_K, "Q6_K", eight);
        }
        printf("fused dot overflow: %d cases finite and exact\n", n);
        return 0;
    } catch (const std::exception& e) {
        printf("FAIL: %s\n", e.what());
        return 1;
    }
}
