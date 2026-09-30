#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "backends/cpu/q8_dots.hpp"
#include "backends/cpu/cpu_backend.hpp"
#include "mxfp4_prompt.hpp"
#include "quant/mxfp4.hpp"

namespace {

uint32_t bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

// Decode the element's sign, exponent and mantissa mathematically in double, then apply the block exponent and round once to f32.
float expected(unsigned e, unsigned code) {
    const unsigned magnitude = code & 7, exponent = magnitude >> 1, mantissa = magnitude & 1;
    if (!magnitude) return 0.0f;
    const double element = exponent ? std::ldexp(1.0 + mantissa * 0.5, int(exponent) - 1) : mantissa * 0.5;
    const double value = std::ldexp(element, int(e) - 127);
    const float rounded = value > std::numeric_limits<float>::max() ? std::numeric_limits<float>::infinity() : float(value);
    return code & 8 ? -rounded : rounded;
}

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

size_t check_scales() {
    size_t checked = 0;
    for (unsigned e = 0; e < 256; ++e) {
        const float weight = std::ldexp(1.0f, int(e) - 128);
        require(bits(quant::mxfp4_scale(uint8_t(e))) == bits(weight), "MXFP4 scale differs from power of two");
        for (uint32_t exponent = 0; exponent < 256; ++exponent)
            for (uint32_t mantissa : {0u, 1u, 0x3fffffu, 0x7fffffu})
                for (uint32_t sign : {0u, 0x80000000u}) {
                    const uint32_t raw = sign | (exponent << 23) | mantissa;
                    float activation;
                    std::memcpy(&activation, &raw, sizeof activation);
                    const float expected = weight * activation;
                    float got;
                    const bool accepted = backend::q8::mxfp4_dot_scale(uint8_t(e), activation, got);
                    require(accepted == (e < 253 && std::isnormal(expected)), "MXFP4 scale classification differs from finite normal range");
                    require(!accepted || bits(got) == bits(expected), "MXFP4 accepted scale product changed");
                    ++checked;
                }
    }
    return checked;
}

size_t check_integer_lanes() {
    size_t checked = 0;
    std::array<uint8_t, 17> packed{};
    std::array<int16_t, 33> input{};
    uint8_t* p = packed.data() + 1;
    int16_t* x = input.data() + 1;
    for (unsigned code = 0; code < 256; ++code) {
        for (size_t j = 0; j < 16; ++j) p[j] = uint8_t(code + 17 * j);
        for (size_t pattern = 0; pattern < 66; ++pattern) {
            std::fill(input.begin(), input.end(), int16_t(0));
            if (pattern < 64) x[pattern % 32] = pattern < 32 ? int16_t(32767) : int16_t(-32768);
            else for (size_t j = 0; j < 32; ++j)
                x[j] = pattern == 64 ? (j % 2 ? int16_t(-32768) : int16_t(32767)) : int16_t(int((j * 997 + code * 257) % 65536) - 32768);
            int32_t got[8];
            _mm256_storeu_si256((__m256i*)got, backend::q8::dot_mxfp4_block(p, x));
            for (size_t lane = 0; lane < 8; ++lane) {
                int32_t want = 0;
                for (size_t half = 0; half < 2; ++half) {
                    for (size_t pair = 0; pair < 2; ++pair) {
                        const size_t at = 2 * lane + pair;
                        const unsigned nibble = (unsigned(p[at]) >> (4 * half)) & 15;
                        want += int32_t(expected(128, nibble)) * int32_t(x[at + 16 * half]);
                    }
                }
                require(got[lane] == want, "MXFP4 integer lane differs from independent E2M1 products");
                ++checked;
            }
        }
    }
    return checked;
}

size_t check_dots() {
    backend::q8::Rows16 x;
    backend::q8::size_rows(1, 32, x);
    x.d[0] = 1.0f;
    std::array<uint8_t, 18> storage{};
    uint8_t* p = storage.data() + 1;
    size_t checked = 0;
    // One-hot inputs isolate each nibble at every exponent whose weights are finite, and do not share the production lookup table.
    for (unsigned e = 0; e < 253; ++e) {
        p[0] = uint8_t(e);
        for (unsigned shift = 0; shift < 16; ++shift) {
            for (size_t j = 0; j < 16; ++j)
                p[j + 1] = uint8_t(((j + shift) % 16) | (((15 - j + shift) % 16) << 4));
            for (size_t j = 0; j < 32; ++j) {
                std::fill(x.q.begin(), x.q.end(), int16_t(0));
                x.q[j] = 1;
                const unsigned code = j < 16 ? unsigned(j + shift) % 16 : unsigned(31 - j + shift) % 16;
                require(backend::q8::dot_mxfp4(p, x, 0) == expected(e, code), "MXFP4 one-hot dot differs from independent decode");
                ++checked;
            }
        }
    }
    // Odd block counts and unaligned rows, with cancellation and scales whose premature product loses a representable result.
    for (size_t blocks : {size_t(1), size_t(3), size_t(9), size_t(90)}) {
        backend::q8::size_rows(1, blocks * 32, x);
        std::vector<uint8_t> bytes(1 + blocks * 17);
        for (int mode = 0; mode < 4; ++mode) {
            double oracle = 0.0, magnitude = 0.0;
            for (size_t b = 0; b < blocks; ++b) {
                uint8_t* block = bytes.data() + 1 + b * 17;
                const unsigned e = mode == 1 ? 0 : mode == 2 ? 240 : mode == 3 ? 2 : 122 + unsigned(b % 7);
                block[0] = uint8_t(e);
                x.d[b] = mode == 1 ? std::ldexp(1.0f, -15) : mode == 2 ? std::ldexp(1.0f, 30) : mode == 3 ? std::ldexp(1.0f, -24) : 0.0031f;
                for (size_t j = 0; j < 32; ++j) {
                    const unsigned code = unsigned(j + b) % 16;
                    if (j < 16) block[j + 1] = uint8_t(code);
                    else block[j - 15] |= uint8_t(code << 4);
                    // In the overflowing-scale case each block's nonzero weights cancel exactly.
                    const int value = mode == 2 ? 1 : int((j * 971 + b * 17) % 65535) - 32767;
                    x.q[b * 32 + j] = int16_t(value);
                    const double product = double(expected(e, code)) * (double(value) * double(x.d[b]));
                    oracle += product;
                    magnitude += std::fabs(product);
                }
            }
            const float got = backend::q8::dot_mxfp4(bytes.data() + 1, x, 0);
            const double bound = std::max(double(std::numeric_limits<float>::denorm_min()), magnitude * 2e-6);
            require(std::isfinite(got) && std::fabs(double(got) - oracle) <= bound, "MXFP4 multi-block dot exceeds its independent error bound");
            ++checked;
        }
    }
    // High exponents must multiply decoded weights: a small activation cannot make an infinite stored weight finite.
    backend::q8::size_rows(1, 32, x);
    std::fill(x.q.begin(), x.q.end(), int16_t(1));
    x.d[0] = std::ldexp(1.0f, -120);
    for (unsigned e : {253u, 254u, 255u}) {
        p[0] = uint8_t(e);
        for (unsigned code : {1u, 9u, 7u, 15u}) {
            std::fill(p + 1, p + 17, uint8_t(code | (code << 4)));
            const float want = float(double(expected(e, code)) * double(x.d[0]) * 32.0);
            const float got = backend::q8::dot_mxfp4(p, x, 0);
            require(bits(got) == bits(want), "MXFP4 high-exponent dot changed a decoded weight's range");
            ++checked;
        }
    }
    // Each decode row retains a representable dot under normal and subnormal activation scales.
    backend::q8::size_rows(2, 32, x);
    std::fill(x.q.begin(), x.q.end(), int16_t(32767));
    x.d[0] = 1.0f / 32767.0f;
    x.d[1] = std::numeric_limits<float>::denorm_min();
    p[0] = 127;
    std::fill(p + 1, p + 17, uint8_t(0x11));
    for (size_t row = 0; row < 2; ++row) {
        const float want = float(0.5 * 32767.0 * double(x.d[row]) * 32.0);
        require(want > 0.0f && bits(backend::q8::dot_mxfp4(p, x, row)) == bits(want), "MXFP4 decode row lost its own scale range");
        ++checked;
    }
    return checked;
}

} // namespace

int main() {
    try {
        size_t checked = 0;
        // An unaligned input, guarded output, adjacent blocks with unlike scales, and every code in each position at every exponent.
        for (unsigned e = 0; e < 256; ++e) {
            for (unsigned shift = 0; shift < 16; ++shift) {
                std::array<uint8_t, 1 + 2 * quant::MXFP4_TYPESIZE> storage{};
                uint8_t* p = storage.data() + 1;
                p[0] = uint8_t(e);
                p[quant::MXFP4_TYPESIZE] = uint8_t(255 - e);
                for (size_t j = 0; j < 16; ++j) {
                    const unsigned low = unsigned(j + shift) % 16, high = unsigned(15 - j + shift) % 16;
                    p[j + 1] = uint8_t(low | (high << 4));
                    p[quant::MXFP4_TYPESIZE + j + 1] = uint8_t(high | (low << 4));
                }
                std::array<float, 66> out;
                out.fill(123.0f);
                quant::dequantize_row_mxfp4(p, out.data() + 1, 2);
                require(out.front() == 123.0f && out.back() == 123.0f, "MXFP4 decoder crossed output extent");
                for (size_t b = 0; b < 2; ++b) {
                    for (size_t j = 0; j < 32; ++j) {
                        const unsigned lo = unsigned((j % 16) + shift) % 16, hi = unsigned(15 - (j % 16) + shift) % 16;
                        const unsigned code = (j < 16) != (b != 0) ? lo : hi;
                        const float want = expected(b ? 255 - e : e, code);
                        if (bits(out[1 + b * 32 + j]) != bits(want))
                            throw std::runtime_error("MXFP4 mismatch: exponent " + std::to_string(b ? 255 - e : e) +
                                                     " code " + std::to_string(code) + " position " + std::to_string(j));
                        ++checked;
                    }
                }
            }
        }
        quant::dequantize_row_mxfp4(nullptr, nullptr, 0);
        require(bits(expected(0, 1)) == 0x00200000u, "oracle subnormal anchor");
        require(bits(expected(127, 7)) == 0x40c00000u, "oracle E2M1 anchor");
        require(bits(expected(255, 1)) == 0x7f000000u, "oracle exponent 255 anchor");
        size_t prompts = 0;
        for (int threads : {1, 3}) {
            backend::CpuBackend cpu;
            cpu.set_threads(threads);
            for (size_t width : {32u, 544u})
                for (size_t extent : {2u, 7u, 65u}) {
                    prompts += mxfp4_test::prompt_invariance(cpu, width, extent);
                    for (bool add : {false, true}) prompts += mxfp4_test::routed_invariance(cpu, width, extent, add);
                }
        }
        std::cout << "MXFP4 CPU prompt precision: " << prompts << " values passed\n";
        const size_t scales = check_scales();
        std::cout << "MXFP4 scales: " << scales << " acceptance/product checks passed\n";
        const size_t lanes = check_integer_lanes();
        std::cout << "MXFP4 integer lanes: " << lanes << " exact independent sums passed\n";
        const size_t dots = check_dots();
        std::cout << "MXFP4 dots: " << dots << " independent checks passed\n";
        std::cout << "MXFP4: " << checked << " values exact, including subnormals, zero signs and overflow; extents intact\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
