// binary32 <-> binary16 conversion, checked without an oracle library.
//
// The decode direction is exercised constantly by every quantized matmul, so
// it was correct. The encode direction is reached only by `llmx quantize`,
// which nothing compared against a reference, and it was wrong twice: a
// mantissa that rounded up out of ten bits had its carry OR-ed into the
// exponent field instead of added, which silently halved the result whenever
// the exponent was odd, and ties rounded half-up while the subnormal path in
// the same function rounded half-to-even.
//
// The oracle here is binary16 itself. Every finite half is a float exactly, so
// encoding that float must return the same bits; and for any float, the
// encoded half must be at least as close to it as either neighbouring half,
// which is the definition of round-to-nearest with ties resolved to even.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>

#include "core/fp16.hpp"

namespace {

size_t checks = 0;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

bool is_finite_half(uint16_t h) { return (h & 0x7c00u) != 0x7c00u; }

// Every finite half is exactly representable as a float, so the encoder has to
// return the bits it started from. This is the whole range, not a sample.
void round_trip() {
    for (uint32_t h = 0; h < 0x10000u; ++h) {
        if (!is_finite_half((uint16_t)h)) continue;
        const float f = f16_to_f32((uint16_t)h);
        const uint16_t back = f32_to_f16(f);
        if (back != (uint16_t)h) {
            char buf[96];
            std::snprintf(buf, sizeof buf, "half 0x%04X encoded back as 0x%04X",
                          (unsigned)h, (unsigned)back);
            throw std::runtime_error(buf);
        }
        ++checks;
    }
}

// Nearest with ties to even, stated as a property: no other half is strictly
// closer, and where a neighbour is equally close the even significand wins.
void nearest(float f) {
    const uint16_t got = f32_to_f16(f);
    require(is_finite_half(got), "finite input encoded to inf or nan");
    const double target = (double)f;
    const double chosen = (double)f16_to_f32(got);
    for (uint32_t h = 0; h < 0x10000u; ++h) {
        if (!is_finite_half((uint16_t)h)) continue;
        const double other = (double)f16_to_f32((uint16_t)h);
        const double d_other = std::fabs(other - target);
        const double d_chosen = std::fabs(chosen - target);
        require(d_other >= d_chosen, "a nearer half exists");
        if (d_other == d_chosen && other != chosen)
            require((got & 1u) == 0u, "tie did not resolve to the even significand");
    }
    ++checks;
}

float from_bits(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

// The two cases the old encoder got wrong, pinned by name so a regression
// says which defect came back.
void regressions() {
    // Carry out of the mantissa with an odd exponent. 0x3FFFF800 is just below
    // 2.0 and rounds up to it; the OR-ed carry returned 1.0, exactly half.
    const float just_under_two = from_bits(0x3FFFF800u);
    require(f32_to_f16(just_under_two) == 0x4000u, "mantissa carry lost in the exponent");
    require(f16_to_f32(f32_to_f16(just_under_two)) == 2.0f, "carry case did not round to 2");
    ++checks;

    // Same carry one binade down, where the old code produced 0x0400 for a
    // value that is 0x0800.
    require(f32_to_f16(from_bits(0x38FFFFC0u)) == 0x0800u, "mantissa carry lost near 2^-14");
    ++checks;

    // Overflow by rounding: below the largest half but nearer to infinity.
    require(f32_to_f16(70000.0f) == 0x7c00u, "value above the half range is not inf");
    require(f32_to_f16(-70000.0f) == 0xfc00u, "negative overflow is not -inf");
    ++checks;

    // Signed zero and the subnormal edges survive the round trip.
    require(f32_to_f16(0.0f) == 0x0000u, "positive zero");
    require(f32_to_f16(-0.0f) == 0x8000u, "negative zero");
    require(f32_to_f16(from_bits(0x33000000u)) == 0x0000u, "underflow is not zero");
    ++checks;

    require((f32_to_f16(std::numeric_limits<float>::infinity()) & 0x7fffu) == 0x7c00u,
            "infinity");
    const uint16_t nan = f32_to_f16(std::numeric_limits<float>::quiet_NaN());
    require((nan & 0x7c00u) == 0x7c00u && (nan & 0x03ffu) != 0u, "nan lost its payload");
    ++checks;
}

}  // namespace

int main() {
    try {
        round_trip();
        // Exhaustive nearest-half checking is 65536 comparisons per input, so
        // this walks a coprime stride through the float space rather than all
        // of it, plus the boundaries that the stride would miss.
        for (uint32_t bits = 0; bits < 0x40000000u; bits += 7370029u) nearest(from_bits(bits));
        for (uint32_t bits = 0x80000000u; bits < 0xc0000000u; bits += 7370029u)
            nearest(from_bits(bits));
        for (float f : {1.0f, 0.5f, 65504.0f, -65504.0f, 6.103515625e-05f,
                        5.960464477539063e-08f, 1.0f / 3.0f, -1.0f / 3.0f})
            nearest(f);
        regressions();
        std::cout << "fp16: " << checks << " conversion checks pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
