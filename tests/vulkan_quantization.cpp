// Inspect the private packed activation buffers without adding a runtime probe API.
#include "backends/vulkan/vulkan_backend.cpp"
#include <cmath>
#include <iostream>
#include <limits>

namespace backend {
namespace {
struct VulkanQuantizationTest {
    static bool preserves_float32(VulkanBackend& b) { return b.dev_->preserve_float32; }
    // Refuse optional float modes at module creation, as a device without those capabilities would.
    struct WithoutPreservation {
        VulkanBackend& backend;
        inline static PFN_vkCreateShaderModule create = nullptr;
        inline static size_t modules = 0;
        static VKAPI_ATTR VkResult VKAPI_CALL checked_create(VkDevice device, const VkShaderModuleCreateInfo* info,
                                                            const VkAllocationCallbacks* alloc, VkShaderModule* module) {
            for (size_t i = 5; i < info->codeSize / 4;) {
                const uint32_t word = info->pCode[i], count = word >> 16;
                if (!count || i + count > info->codeSize / 4) return VK_ERROR_INITIALIZATION_FAILED;
                if ((word & 65535u) == 16u && count >= 4 && info->pCode[i + 3] == 32u &&
                    (info->pCode[i + 2] == 4459u || info->pCode[i + 2] == 4461u)) return VK_ERROR_FEATURE_NOT_PRESENT;
                i += count;
            }
            ++modules;
            return create(device, info, alloc, module);
        }
        explicit WithoutPreservation(VulkanBackend& b) : backend(b) {
            b.dev_->preserve_float32 = false;
            create = b.dev_->fn.vkCreateShaderModule;
            modules = 0;
            b.dev_->fn.vkCreateShaderModule = checked_create;
        }
        ~WithoutPreservation() { backend.dev_->fn.vkCreateShaderModule = create; }
    };
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

// Hold optional consumer arithmetic to the ordinary modules on the same device and normal inputs.
void check_ordinary_identity(backend::VulkanBackend& preserved, backend::VulkanBackend& ordinary) {
    if (!backend::VulkanQuantizationTest::preserves_float32(preserved)) {
        std::cout << "vulkan-quantization: ordinary module comparison skipped without float preservation\n";
        return;
    }
    uint32_t seed = 193;
    size_t checked = 0, failed = 0;
    constexpr size_t rows = 41;
    for (size_t width : {size_t(32), size_t(64), size_t(96), size_t(128), size_t(2080), size_t(2112)}) {
        const size_t blocks = width / 32;
        std::vector<uint8_t> weights(rows * blocks * quant::Q8_0_TYPESIZE);
        for (size_t block = 0; block < rows * blocks; ++block) {
            const uint16_t scale = uint16_t(0x2400u + random_bits(seed) % 0x2800u);
            const size_t start = block * quant::Q8_0_TYPESIZE;
            std::memcpy(weights.data() + start, &scale, sizeof(scale));
            for (size_t j = 0; j < 32; ++j) weights[start + 2 + j] = uint8_t(random_bits(seed));
        }
        const auto wp = preserved.adopt(weights.data(), weights.size());
        const auto wo = ordinary.adopt(weights.data(), weights.size());
        for (size_t columns : {size_t(1), size_t(2), size_t(3), size_t(7), size_t(8), size_t(9), size_t(16), size_t(31), size_t(32), size_t(33), size_t(64)}) {
            std::vector<float> input(width * columns), actual(rows * columns), expected(rows * columns);
            for (float& v : input) v = float(int(random_bits(seed) % 65535u) - 32767) / 1009.0f;
            const auto xp = preserved.adopt(input.data(), input.size() * sizeof(float));
            const auto xo = ordinary.adopt(input.data(), input.size() * sizeof(float));
            const auto yp = preserved.alloc(actual.size() * sizeof(float), backend::Memory::device);
            const auto yo = ordinary.alloc(expected.size() * sizeof(float), backend::Memory::device);
            const backend::RowRun decode{columns, 1};
            preserved.matmul(quant::GGML_TYPE_Q8_0, {wp.get(), 0}, {xp.get(), 0}, {yp.get(), 0}, width, rows, columns, {&decode, 1});
            ordinary.matmul(quant::GGML_TYPE_Q8_0, {wo.get(), 0}, {xo.get(), 0}, {yo.get(), 0}, width, rows, columns, {&decode, 1});
            preserved.read(*yp, 0, actual.data(), actual.size() * sizeof(float));
            ordinary.read(*yo, 0, expected.data(), expected.size() * sizeof(float));
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0) {
                    if (failed < 8) std::fprintf(stderr, "ordinary Q8 identity: width %zu columns %zu output %zu preserved %.9g ordinary %.9g\n", width, columns, i, actual[i], expected[i]);
                    ++failed;
                }
                ++checked;
            }
        }
    }
    std::cout << "vulkan-quantization: " << checked << " ordinary Q8 consumer outputs, " << failed << " differ between modules\n";
    require(failed == 0, "preserved Q8 consumers differ from ordinary modules");
}

// An identity matrix reads each activation back through the public quantized matmul, so the oracle is the original input rather than the device's quantization formula.
size_t check_activation_range(backend::Backend& b, size_t width, bool preserves_float32) {
    const size_t blocks = width / 32, row_bytes = blocks * quant::Q8_0_TYPESIZE;
    std::vector<uint8_t> weights(width * row_bytes, 0);
    for (size_t row = 0; row < width; ++row) {
        const uint16_t scale = 0x3c00; // Half precision 1, with one unit weight in the row.
        const size_t offset = row * row_bytes + (row / 32) * quant::Q8_0_TYPESIZE;
        std::memcpy(weights.data() + offset, &scale, sizeof(scale));
        weights[offset + 2 + row % 32] = 1;
    }
    const auto w = b.adopt(weights.data(), weights.size());
    std::vector<float> peaks{0};
    for (int exponent = -149; exponent <= 127; ++exponent) peaks.push_back(std::ldexp(1.0f, exponent));
    peaks.push_back(std::numeric_limits<float>::max());
    for (int limit : {127, 32767}) {
        const float boundary = float(double(limit) / std::numeric_limits<float>::max());
        peaks.push_back(std::nextafter(boundary, 0.0f));
        peaks.push_back(boundary);
        peaks.push_back(std::nextafter(boundary, 1.0f));
    }
    const float factors[] = {1, -1, 0, .5f, -.5f, 1.0f / 3, -1.0f / 7, .015625f};
    size_t checked = 0, failed = 0, skipped = 0;
    for (int producer = 0; producer < 3; ++producer) for (float peak : peaks) {
        std::vector<float> input(width), output(width);
        for (size_t j = 0; j < width; ++j) input[j] = peak * factors[j % 8];
        const auto x = b.adopt(input.data(), input.size() * sizeof(float));
        const auto y = b.alloc(output.size() * sizeof(float));
        backend::BufferPtr produced;
        if (producer != 0) {
            produced = b.alloc(width * sizeof(float));
            std::vector<float> source(width, producer == 1 ? 32.0f : 1.0f);
            const auto a = b.adopt(source.data(), source.size() * sizeof(float));
            if (producer == 1) {
                for (size_t j = 0; j < width; ++j) source[j] = input[j] / 32;
                const auto up = b.adopt(source.data(), source.size() * sizeof(float));
                b.silu_mul({produced.get(), 0}, {a.get(), 0}, {up.get(), 0}, width);
                b.sync();
            } else {
                b.rms_norm_rows({produced.get(), 0}, {a.get(), 0}, {x.get(), 0}, 1, width, width, 0);
                b.sync();
            }
            // Quantization must reconstruct the producer's actual float output, independently of that producer's own arithmetic error.
            b.read(*produced, 0, input.data(), input.size() * sizeof(float));
            peak = 0;
            for (float v : input) peak = std::max(peak, std::abs(v));
        }
        // Without float preservation the packed twin is still checked in full, but a consumer may flush its subnormal scale.
        if (!preserves_float32 && peak != 0 && peak < std::numeric_limits<float>::min() * 32767.0f) {
            skipped += width;
            continue;
        }
        b.matmul(quant::GGML_TYPE_Q8_0, {w.get(), 0}, {produced ? produced.get() : x.get(), 0}, {y.get(), 0}, width, width, 1);
        b.read(*y, 0, output.data(), output.size() * sizeof(float));
        // A tiny block needs a representable scale rounded up so its peak fits the integer range; both twins fit this 8-bit bound.
        float representable = float(double(peak) / 127);
        if (double(representable) * 127 < double(peak)) representable = std::nextafter(representable, std::numeric_limits<float>::max());
        const double step = std::max(double(representable), double(std::numeric_limits<float>::denorm_min()));
        for (size_t j = 0; j < width; ++j) {
            const double bound = .50001 * step + 3e-7 * std::abs(double(input[j]));
            if (!std::isfinite(output[j]) || std::abs(double(output[j]) - input[j]) > bound) {
                if (failed < 8) std::fprintf(stderr, "activation range: producer %d peak %.9g position %zu expected %.9g device %.9g bound %.9g\n",
                                              producer, peak, j, input[j], output[j], bound);
                ++failed;
            }
            ++checked;
        }
    }
    std::cout << "vulkan-quantization: width " << width << ", " << checked << " activation reconstruction values, "
              << failed << " failed, " << skipped << " skipped without float preservation\n";
    require(failed == 0, "finite activation reconstruction exceeds its quantization bound");
    return checked;
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
        for (size_t width : {size_t(32), size_t(64)})
            check_activation_range(b, width, backend::VulkanQuantizationTest::preserves_float32(b));
        backend::VulkanBackend fallback(0);
        backend::VulkanQuantizationTest::WithoutPreservation guard(fallback);
        for (size_t width : {size_t(32), size_t(64)}) check_activation_range(fallback, width, false);
        require(guard.modules != 0, "unsupported-mode check created no modules");
        std::cout << "vulkan-quantization: " << guard.modules << " modules accepted without float preservation\n";
        check_ordinary_identity(b, fallback);
        return 0;
    } catch (const backend::VulkanUnavailable& e) {
        std::cerr << e.what() << '\n';
        return 77;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
