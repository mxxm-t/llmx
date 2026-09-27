#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "quant/quant.hpp"

namespace {

size_t checks = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

// Integer multiples of a power of two have exact expected codes, including where its reciprocal is not finite.
void powers() {
    for (int exponent = -149; exponent <= 124; ++exponent) {
        const float unit = std::ldexp(1.0f, exponent);
        for (int rotation = 0; rotation < 15; ++rotation) {
            std::array<float, 32> input{};
            for (size_t j = 0; j < input.size(); ++j)
                input[j] = float((int(j) + rotation) % 15 - 7) * unit;
            std::array<uint8_t, 20> output;
            output.fill(0xa5);
            quant::quantize_row_q4_0(input.data(), output.data() + 1, 1);
            const std::string label = "Q4_0 exponent " + std::to_string(exponent) + ", rotation " + std::to_string(rotation);
            require(output.front() == 0xa5 && output.back() == 0xa5, label + ": output boundary");
            const unsigned scale = exponent < -24 ? 0u : exponent < -14 ? 1u << (exponent + 24)
                : exponent <= 15 ? unsigned(exponent + 15) << 10 : 0x7c00u;
            require((unsigned(output[1]) | unsigned(output[2]) << 8) == scale, label + ": scale");
            for (size_t j = 0; j < input.size(); ++j) {
                const unsigned code = (output[3 + j % 16] >> (j < 16 ? 0 : 4)) & 15;
                const unsigned expected = unsigned((int(j) + rotation) % 15 + 1);
                require(code == expected, label + ": code at " + std::to_string(j) + " is " +
                        std::to_string(code) + ", expected " + std::to_string(expected));
            }
        }
    }
}

void boundaries() {
    for (int exponent : {-148, -130, -127, 0}) {
        const float unit = std::ldexp(1.0f, exponent);
        const std::array<float, 8> values{{-7, -2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 7}};
        const std::array<unsigned, 8> codes{{1, 5, 6, 7, 9, 10, 11, 15}};
        std::array<float, 32> input{};
        std::array<uint8_t, 18> output{};
        for (size_t j = 0; j < input.size(); ++j) input[j] = values[j % 8] * unit;
        quant::quantize_row_q4_0(input.data(), output.data(), 1);
        for (size_t j = 0; j < input.size(); ++j)
            require(((output[2 + j % 16] >> (j < 16 ? 0 : 4)) & 15u) == codes[j % 8], "Q4_0 half-way code");
    }
    // The smallest input has a scale rounded to zero even in F32; it follows the existing zero-scale encoding.
    std::array<float, 32> input{};
    std::array<uint8_t, 18> output{};
    for (size_t j = 0; j < input.size(); ++j)
        input[j] = j % 2 ? -std::numeric_limits<float>::denorm_min() : std::numeric_limits<float>::denorm_min();
    quant::quantize_row_q4_0(input.data(), output.data(), 1);
    require(output[0] == 0 && output[1] == 0, "Q4_0 underflowed scale");
    for (size_t j = 2; j < output.size(); ++j) require(output[j] == 0x88, "Q4_0 underflowed scale code");
}

} // namespace

int main() {
    try {
        powers();
        boundaries();
        std::cout << "quantize-range: " << checks << " checks pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
