// The Vulkan twin writer's scale and reciprocal (shaders/xquant.glsl, xq_div and xq_rdiv), followed on the host operation for operation for every significand in [1, 2), against the IEEE quotients (AGENTS.md, Tests).
// Built without contraction, so each multiply and add rounds as the shader's precise ones do.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

float bits(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}

uint32_t word(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

// xq_rdiv's reciprocal of a significand: a seed from its bits and four Newton steps.
float recip(float s) {
    float y = bits(0x7EF311C3u - word(s));
    for (int i = 0; i < 4; ++i) y = y * (2.0f - s * y);
    return y;
}

// Units in the last place between two positive normal floats.
long apart(float a, float b) { return std::labs(long(word(a)) - long(word(b))); }

}  // namespace

int main() {
    try {
        long worst_scale = 0, worst_recip = 0;
        size_t checked = 0;
        for (const uint32_t levels : {32767u, 127u}) {
            const float c = bits(levels == 32767u ? 0x38000100u : 0x3C010204u);
            require(c == float(1.0 / double(levels)), "the scale's factor for " + std::to_string(levels) + " is not 1 / levels correctly rounded");
            for (uint32_t s = 0; s < (1u << 23); ++s, ++checked) {
                const float m = bits(0x3F800000u | s);
                worst_scale = std::max(worst_scale, apart(m * c, float(double(m) / double(levels))));
            }
        }
        for (const uint32_t a : {32767u, 127u, 1u})
            for (uint32_t s = 0; s < (1u << 23); ++s, ++checked) {
                const float m = bits(0x3F800000u | s);
                worst_recip = std::max(worst_recip, apart(float(a) * recip(m), float(double(a) / double(m))));
            }
        require(worst_scale <= 1, "the scale is " + std::to_string(worst_scale) + " units from the IEEE quotient");
        require(worst_recip <= 2, "the reciprocal is " + std::to_string(worst_recip) + " units from the IEEE quotient");
        std::cout << "twin-quotients: " << checked << " quotients, the scale within " << worst_scale << " and the reciprocal within " << worst_recip
                  << " units of the IEEE quotient\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "twin-quotients: FAIL: " << e.what() << "\n";
        return 1;
    }
}
