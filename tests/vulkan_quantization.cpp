// Inspect the private packed activation buffers without adding a runtime probe API.
#include "backends/vulkan/vulkan_backend.cpp"
#include <cmath>
#include <iostream>
#include <limits>

namespace backend {
namespace {
struct VulkanQuantizationTest {
    static void run(VulkanBackend& b, CSlice x, CSlice y, uint32_t n, bool tile) {
        if (tile) {
            const uint32_t args[] = {n, 32, n / 32};
            b.dispatch(K_QUANTIZE_X8, {b.bind(x), b.bind(y)}, args, sizeof(args), (n / 4 + 255) / 256);
        } else b.dispatch(K_QUANTIZE_X, {b.bind(x), b.bind(y)}, &n, sizeof(n), (n + 255) / 256, 1, 1);
    }
};
}
}

namespace {
float from_bits(uint32_t bits) {
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}
uint32_t random_bits(uint32_t& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// A representable scale covering the peak gives a bound independent of the device's chosen scale.
double reconstruction_step(float peak, int limit) {
    float scale = float(double(peak) / limit);
    if (double(scale) * limit < double(peak)) scale = std::nextafter(scale, std::numeric_limits<float>::max());
    return std::max(double(scale), double(std::numeric_limits<float>::denorm_min()));
}

// The normalized shader product may round once before restoring a subnormal exponent, so permit two f32 steps.
bool sum_close(float actual, double expected) {
    const float rounded = float(expected);
    if (!std::isfinite(rounded)) return actual == rounded;
    if (!std::isfinite(actual)) return false;
    const double up = double(std::nextafter(rounded, std::numeric_limits<float>::infinity())) - rounded;
    const double down = double(rounded) - std::nextafter(rounded, -std::numeric_limits<float>::infinity());
    const double ulp = std::min(up, down);
    return std::abs(double(actual) - expected) <= 2 * ulp;
}

void check(backend::VulkanBackend& b) {
    std::vector<float> peaks{0};
    for (int e = -149; e <= 127; ++e) peaks.push_back(std::ldexp(1.0f, e));
    peaks.push_back(std::numeric_limits<float>::max());
    for (int limit : {127, 32767}) {
        const float boundary = float(double(limit) / std::numeric_limits<float>::max());
        peaks.push_back(std::nextafter(boundary, 0.0f));
        peaks.push_back(boundary);
        peaks.push_back(std::nextafter(boundary, 1.0f));
    }
    for (uint32_t boundary : {0x08800000u, 0x7e800000u})
        for (int offset = -2; offset <= 2; ++offset) peaks.push_back(from_bits(uint32_t(int64_t(boundary) + offset)));
    uint32_t seed = 173;
    for (int i = 0; i < 4096; ++i) peaks.push_back(from_bits(random_bits(seed) % 0x7f800000u));
    const float factors[] = {1, -1, 0, .5f, -.5f, 1.0f / 3, -1.0f / 7, .015625f};
    std::vector<float> input;
    for (float peak : peaks) for (int j = 0; j < 32; ++j) input.push_back(peak * factors[j % 8]);
    // Uneven exponents exercise values that become subnormal or zero when normalized beside the peak.
    const size_t patterned = peaks.size();
    for (size_t block = 0; block < patterned; ++block) {
        const float peak = peaks[block];
        peaks.push_back(peak);
        input.push_back(peak);
        for (int j = 1; j < 32; ++j) {
            const uint32_t r = random_bits(seed);
            input.push_back(std::ldexp((r & 1) ? peak : -peak, -int((r >> 1) % 277)));
        }
    }
    const uint32_t n = uint32_t(input.size()), base8 = (n / 2 + n / 8 + 63) & ~63u;
    constexpr uint32_t guard = 64, sentinel = 0x12345678u;
    const uint32_t words8 = n / 4 + n / 16;
    std::vector<uint32_t> packed(base8 + words8 + 2 * guard, sentinel), tile(words8 + 2 * guard, sentinel);
    const auto x = b.adopt(input.data(), input.size() * sizeof(float));
    const auto q = b.adopt(packed.data(), packed.size() * sizeof(uint32_t));
    const auto t = b.adopt(tile.data(), tile.size() * sizeof(uint32_t));
    backend::VulkanQuantizationTest::run(b, {x.get(), 0}, {q.get(), guard}, n, false);
    backend::VulkanQuantizationTest::run(b, {x.get(), 0}, {t.get(), guard}, n, true);
    b.read(*q, 0, packed.data(), packed.size() * sizeof(uint32_t));
    b.read(*t, 0, tile.data(), tile.size() * sizeof(uint32_t));
    for (size_t i = 0; i < guard; ++i) {
        require(packed[i] == sentinel && packed[packed.size() - 1 - i] == sentinel, "packed activation guard changed");
        require(tile[i] == sentinel && tile[tile.size() - 1 - i] == sentinel, "tile activation guard changed");
    }
    for (size_t i = n / 2 + n / 8; i < base8; ++i)
        require(packed[guard + i] == sentinel, "packed activation alignment gap changed");
    for (size_t i = 0; i < words8; ++i)
        require(tile[guard + i] == packed[guard + base8 + i], "8-bit word and lane writers differ");
    size_t failures = 0, sums = 0;
    for (int twin = 0; twin < 2; ++twin) for (size_t block = 0; block < peaks.size(); ++block) {
        const int limit = twin ? 127 : 32767;
        const size_t tab = guard + (twin ? base8 + n / 4 : n / 2) + block * 2;
        const float scale = from_bits(packed[tab]);
        const double step = reconstruction_step(peaks[block], limit);
        bool bad = !std::isfinite(scale) || scale < 0 || double(scale) > step || (peaks[block] > 0 && scale == 0);
        int halves[2] = {};
        for (size_t j = 0; j < 32; ++j) {
            const size_t i = block * 32 + j;
            const uint32_t word = packed[guard + (twin ? base8 + i / 4 : block * 16 + 2 * (j / 4) + j % 2)];
            const uint32_t code = twin ? (word >> (8 * (j % 4))) & 255u : (word >> (j % 4 >= 2 ? 16 : 0)) & 65535u;
            const int quantized = int(code) - (code > uint32_t(limit) ? 2 * (limit + 1) : 0);
            const double actual = double(scale) * quantized;
            const double bound = .50001 * step + 3e-7 * std::abs(double(input[i]));
            bad |= std::abs(quantized) > limit || !std::isfinite(actual) || std::abs(actual - input[i]) > bound;
            halves[j / 16] += quantized;
        }
        bad |= !sum_close(from_bits(packed[tab + 1]), double(scale) * (halves[0] + halves[1]));
        ++sums;
        if (!twin) for (size_t half = 0; half < 2; ++half) {
            const size_t at = guard + n / 2 + n / 16 + block * 2 + half;
            bad |= !sum_close(from_bits(packed[at]), double(scale) * halves[half]);
            ++sums;
        }
        if (bad) {
            if (failures < 8) std::cerr << "activation packing: " << (twin ? 8 : 16) << " bits, block " << block
                                      << ", peak " << peaks[block] << ", scale " << scale << '\n';
            ++failures;
        }
    }
    std::cout << "vulkan-quantization: " << input.size() << " values per twin, " << sums << " packed sums, "
              << words8 << " identical lane/word words, " << failures << " failing blocks\n";
    require(failures == 0, "packed activation reconstruction or sums exceed their bounds");
}
}

int main() {
    try {
        backend::VulkanBackend b(0);
        check(b);
        return 0;
    } catch (const backend::VulkanUnavailable& e) {
        std::cerr << e.what() << '\n';
        return 77;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
