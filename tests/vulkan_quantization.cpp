// Inspect the private packed activation buffers without adding a runtime probe API.
#include "backends/vulkan/vulkan_backend.cpp"
#include <cmath>
#include <iostream>
#include <limits>
#include "matrix_precision.hpp"

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
    static void run(VulkanBackend& b, CSlice x, CSlice y, uint32_t n, bool words) {
        const uint32_t pc[] = {n, 0};
        if (words) b.dispatch(K_QUANTIZE_XW, {b.bind(x), b.bind(y)}, pc, sizeof(pc), (n / 4 + 255) / 256);
        else b.dispatch(K_QUANTIZE_X, {b.bind(x), b.bind(y)}, &n, sizeof(n), (n + 255) / 256);
    }
    // The quantizers' builds that also write the 8-bit twin of --dtype int8 after the 16-bit one (xquant.glsl, TWIN8).
    static void run8(VulkanBackend& b, CSlice x, CSlice y, uint32_t n, bool words) {
        const uint32_t pc[] = {n, 0};
        if (words) b.dispatch(K_QUANTIZE_XW, {b.bind(x), b.bind(y)}, pc, sizeof(pc), (n / 4 + 255) / 256, 1, 1);
        else b.dispatch(K_QUANTIZE_X, {b.bind(x), b.bind(y)}, &n, sizeof(n), (n + 255) / 256, 1, 1);
    }
    static size_t x8_base_words(size_t n) { return VulkanBackend::x8_base_bytes(n) / 4; }
    static uint32_t x8_columns(uint32_t cols) { return VulkanBackend::x8_columns(cols); }
    static void set_twin8(VulkanBackend& b, bool on) { b.want_x8_ = on; }
    static bool prefers_integer_dot(VulkanBackend& b) { return b.dev_->profile.prefer_integer_dot; }
    // Where the device does not prefer the integer dot, preferring it reaches the tile's twin layouts, which change only where a twin's values lie.
    static void prefer_integer_dot(VulkanBackend& b, bool on) { b.dev_->profile.prefer_integer_dot = on; }
    static bool tile_reads(VulkanBackend& b, size_t nin, size_t nbatch, RowRuns runs) { return b.tile_reads(nin, nbatch, runs); }
    // The words of the twin the table keeps for the n floats at x, and its layout; empty when no producer left one.
    static std::vector<uint32_t> kept_twin(VulkanBackend& b, CSlice x, size_t n, uint32_t& major, bool& has8) {
        const VkDescriptorBufferInfo xf = b.bind(x);
        for (const auto& t : b.twins_)
            if (t.n == n && t.x.buffer == xf.buffer && t.x.offset == xf.offset) {
                std::vector<uint32_t> words(t.buffer->size() / 4);
                b.read(*t.buffer, 0, words.data(), words.size() * 4);
                major = t.major;
                has8 = t.has8;
                return words;
            }
        return {};
    }
    // The twin the quantizers write for the n floats at x into y: the lane writer, or the word writer in `major`'s layout, the 8-bit twin after it when twin8.
    static void quantize(VulkanBackend& b, CSlice x, CSlice y, uint32_t n, bool words, uint32_t major, bool twin8) {
        const uint32_t pc[] = {n, major};
        if (words) b.dispatch(K_QUANTIZE_XW, {b.bind(x), b.bind(y)}, pc, sizeof(pc), (n / 4 + 255) / 256, 1, twin8 ? 1 : 0);
        else b.dispatch(K_QUANTIZE_X, {b.bind(x), b.bind(y)}, pc, sizeof(uint32_t), (n + 255) / 256, 1, twin8 ? 1 : 0);
    }
    static void prior_layout(VulkanBackend& b) {
        const uint32_t major = 1;
        const auto& k = b.kernel(K_QUANTIZE_XW, 1);
        b.dev_->fn.vkCmdPushConstants(b.open(), k.layout, VK_SHADER_STAGE_COMPUTE_BIT, sizeof(uint32_t), sizeof(major), &major);
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

// F32 rows share the generic row source with Q8, but must keep their ordinary arithmetic.
void check_float_identity(backend::VulkanBackend& preserved, backend::VulkanBackend& ordinary) {
    uint32_t seed = 913;
    size_t checked = 0, failed = 0;
    constexpr size_t rows = 41;
    for (size_t width : {size_t(33), size_t(128), size_t(2048)}) {
        std::vector<float> weights(rows * width);
        for (float& v : weights) v = float(int(random_bits(seed) % 65535u) - 32767) / 4099.0f;
        const auto wp = preserved.adopt(weights.data(), weights.size() * sizeof(float));
        const auto wo = ordinary.adopt(weights.data(), weights.size() * sizeof(float));
        for (size_t columns : {size_t(1), size_t(3), size_t(9)}) {
            std::vector<float> input(width * columns), actual(rows * columns), expected(rows * columns);
            for (float& v : input) v = float(int(random_bits(seed) % 65535u) - 32767) / 1009.0f;
            const auto xp = preserved.adopt(input.data(), input.size() * sizeof(float));
            const auto xo = ordinary.adopt(input.data(), input.size() * sizeof(float));
            const auto yp = preserved.alloc(actual.size() * sizeof(float), backend::Memory::device);
            const auto yo = ordinary.alloc(expected.size() * sizeof(float), backend::Memory::device);
            const backend::RowRun decode{columns, 1};
            preserved.matmul(quant::GGML_TYPE_F32, {wp.get(), 0}, {xp.get(), 0}, {yp.get(), 0}, width, rows, columns, {&decode, 1});
            ordinary.matmul(quant::GGML_TYPE_F32, {wo.get(), 0}, {xo.get(), 0}, {yo.get(), 0}, width, rows, columns, {&decode, 1});
            preserved.read(*yp, 0, actual.data(), actual.size() * sizeof(float));
            ordinary.read(*yo, 0, expected.data(), expected.size() * sizeof(float));
            for (size_t i = 0; i < actual.size(); ++i) {
                if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || std::memcmp(&actual[i], &expected[i], sizeof(float)) != 0) {
                    if (failed < 8) std::fprintf(stderr, "ordinary F32 identity: width %zu columns %zu output %zu preserved %.9g ordinary %.9g\n", width, columns, i, actual[i], expected[i]);
                    ++failed;
                }
                ++checked;
            }
        }
    }
    std::cout << "vulkan-quantization: " << checked << " ordinary F32 outputs, " << failed << " differ between modules\n";
    require(failed == 0, "F32 rows changed under Q8 preservation");
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
        // A tiny block needs a representable scale rounded up so its peak fits the integer range; the 16-bit twin fits this reconstruction bound.
        float representable = float(double(peak) / 32767);
        if (double(representable) * 32767 < double(peak)) representable = std::nextafter(representable, std::numeric_limits<float>::max());
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

// Exact zero weights and one unit weight distinguish range repair from suppressing nonfinite outputs.
void check_offset_range(backend::Backend& b, bool prompt = false, backend::Dtype dtype = backend::Dtype::f16) {
    size_t checked = 0, failed = 0;
    struct Scale { uint16_t bits; double value; };
    const Scale scales[] = {{1, std::ldexp(1.0, -24)}, {0x3c00, 1}, {0x7bff, 65504}};
    constexpr size_t rows = 3;
    const size_t cols = prompt ? 65 : 3, count = rows * cols;
    // F16 conformance covers its own finite range; F32 keeps the wider boundary cases.
    std::vector<float> peaks;
    if (dtype == backend::Dtype::f32) {
        for (int e : {-130, -112, -76, -75, 0, 94, 95, 100, 120, 124, 127}) peaks.push_back(std::ldexp(1.0f, e));
    } else {
        for (int e : {-24, -23, -15, -14, -13, -1, 0, 14, 15}) peaks.push_back(std::ldexp(1.0f, e));
        peaks.push_back(65504.0f);
    }
    for (uint32_t type : {quant::GGML_TYPE_Q4_0, quant::GGML_TYPE_Q4_1, quant::GGML_TYPE_Q5_0, quant::GGML_TYPE_Q5_1, quant::GGML_TYPE_Q4_K,
                          quant::GGML_TYPE_Q5_K, quant::GGML_TYPE_Q6_K, quant::GGML_TYPE_Q8_0}) {
        const auto* qt = quant::Registry::instance().get(type);
        for (const Scale& scale : scales) for (int encoding : {0, 1}) for (size_t width : {size_t(256), size_t(288), size_t(4096)}) {
            if (width % qt->block_size) continue;
            const size_t stride = quant::row_bytes(type, width);
            std::vector<uint8_t> weights(rows * stride, 0);
            for (size_t offset = 0; offset < weights.size(); offset += qt->type_size) {
                uint8_t* block = weights.data() + offset;
                if (type == quant::GGML_TYPE_Q6_K) {
                    std::fill(block + 128, block + 192, uint8_t(0xaa));
                    std::fill(block + 192, block + 208, uint8_t(1));
                    block[209] = 0x3c;
                } else {
                    block[1] = 0x3c;
                    if (type == quant::GGML_TYPE_Q4_0) std::fill(block + 2, block + 18, uint8_t(0x88));
                    if (type == quant::GGML_TYPE_Q5_0) std::fill(block + 2, block + 6, uint8_t(255));
                }
            }
            uint8_t* unit = weights.data() + 2 * stride - qt->type_size;
            if (type == quant::GGML_TYPE_Q4_0) unit[2] = 0x89;
            else if (type == quant::GGML_TYPE_Q4_1) unit[4] = 1;
            else if (type == quant::GGML_TYPE_Q5_0) unit[6] = 1;
            else if (type == quant::GGML_TYPE_Q5_1) unit[8] = 1;
            else if (type == quant::GGML_TYPE_Q8_0) unit[2] = 1;
            else if (type == quant::GGML_TYPE_Q6_K) unit[0] = 1;
            else { unit[4] = 1; unit[type == quant::GGML_TYPE_Q5_K ? 48 : 16] = 1; }
            if (encoding == 1) {
                for (size_t offset = 0; offset < weights.size(); offset += qt->type_size) {
                    uint8_t* block = weights.data() + offset;
                    if (type == quant::GGML_TYPE_Q4_1 || type == quant::GGML_TYPE_Q5_1) {
                        block[3] = 0xbc;
                        std::fill(block + (type == quant::GGML_TYPE_Q5_1 ? 8 : 4), block + qt->type_size, uint8_t(0x11));
                    } else if (type == quant::GGML_TYPE_Q4_K || type == quant::GGML_TYPE_Q5_K) {
                        block[3] = 0x3c;
                        std::fill(block + 4, block + 12, uint8_t(1));
                        std::fill(block + 12, block + 16, uint8_t(0x11));
                        std::fill(block + (type == quant::GGML_TYPE_Q5_K ? 48 : 16), block + qt->type_size, uint8_t(0x11));
                    } else block[type == quant::GGML_TYPE_Q6_K ? 209 : 1] = 0xbc;
                }
                if (type == quant::GGML_TYPE_Q4_0) unit[2] = 0x87;
                else if (type == quant::GGML_TYPE_Q4_1) unit[4] = 0x12;
                else if (type == quant::GGML_TYPE_Q5_0) { unit[2] &= 0xfe; unit[6] = 0x0f; }
                else if (type == quant::GGML_TYPE_Q5_1) unit[8] = 0x12;
                else if (type == quant::GGML_TYPE_Q8_0) unit[2] = 255;
                else if (type == quant::GGML_TYPE_Q6_K) { unit[0] = 15; unit[128] = 0xa9; }
                else unit[type == quant::GGML_TYPE_Q5_K ? 48 : 16] = 0x12;
            }
            for (size_t offset = 0; offset < weights.size(); offset += qt->type_size) {
                uint8_t* block = weights.data() + offset;
                const size_t at = type == quant::GGML_TYPE_Q6_K ? 208 : 0;
                block[at] = uint8_t(scale.bits);
                block[at + 1] = uint8_t((scale.bits >> 8) | (block[at + 1] & 128));
                if (encoding == 1 && (type == quant::GGML_TYPE_Q4_1 || type == quant::GGML_TYPE_Q5_1 || type == quant::GGML_TYPE_Q4_K || type == quant::GGML_TYPE_Q5_K)) {
                    block[2] = uint8_t(scale.bits);
                    block[3] = uint8_t((scale.bits >> 8) | (block[3] & 128));
                }
            }
            std::vector<float> input(width * cols), output(2 * count + 2), ids(cols, 0), gains(cols, 0.5f);
            const auto w = b.adopt(weights.data(), weights.size()), x = b.alloc(input.size() * sizeof(float)),
                       y = b.alloc(output.size() * sizeof(float)), id = b.adopt(ids.data(), ids.size() * sizeof(float)),
                       gain = b.adopt(gains.data(), gains.size() * sizeof(float));
            const backend::Backend::Routing routing{{id.get(), 0}, {gain.get(), 0}, 1, 1};
            const backend::RowRun runs{cols, prompt ? size_t(512) : size_t(1)};
            for (float peak : peaks) {
                for (size_t c = 0; c < cols; ++c)
                    std::fill(input.begin() + c * width, input.begin() + (c + 1) * width, c == 1 ? -peak : peak);
                b.write(*x, 0, input.data(), input.size() * sizeof(float));
                for (int op = 0; op < 6; ++op) {
                    const float initial = op == 3 || op == 5 ? 0.25f : 0.0f;
                    std::fill(output.begin(), output.end(), initial);
                    output.front() = output.back() = 19.0f;
                    b.write(*y, 0, output.data(), output.size() * sizeof(float));
                    testq::take_matrix_paths(b);
                    if (prompt || dtype == backend::Dtype::f32) backend::vulkan_kernel_times(b);
                    switch (op) {
                    case 0: b.matmul(type, {w.get(), 0}, {x.get(), 0}, {y.get(), 1}, width, rows, cols, {&runs, 1}, dtype); break;
                    case 1: b.matmul_logits(type, {w.get(), 0}, {x.get(), 0}, {y.get(), 1}, width, rows, cols, {&runs, 1}, dtype); break;
                    case 2: b.matmul_group({{type, {w.get(), 0}, {y.get(), 1}, rows}, {type, {w.get(), 0}, {y.get(), count + 1}, rows}},
                                           {x.get(), 0}, width, cols, {&runs, 1}, dtype); break;
                    case 3: b.matmul_add(type, {w.get(), 0}, {x.get(), 0}, {y.get(), 1}, width, rows, cols, {&runs, 1}, dtype); break;
                    case 4: b.matmul_experts({{type, {w.get(), 0}, {y.get(), 1}, rows}}, {x.get(), 0}, width, cols, routing, {&runs, 1}, dtype); break;
                    case 5: b.matmul_experts_add(type, {w.get(), 0}, {x.get(), 0}, {y.get(), 1}, width, rows, cols, routing, {&runs, 1}, dtype); break;
                    }
                    b.read(*y, 0, output.data(), output.size() * sizeof(float));
                    const auto paths = testq::take_matrix_paths(b);
                    require(!paths.empty(), "offset range check has no matrix path witness");
                    if (prompt) {
                        const auto kernels = backend::vulkan_kernel_times(b);
                        require(std::any_of(kernels.begin(), kernels.end(), [](const auto& k) {
                            return k.first.find("matmul_tile") == 0;
                        }), "prompt range check did not witness a tile dispatch");
                    }
                    if (dtype == backend::Dtype::f32) {
                        require(paths == std::vector<std::string>{"f32"}, "explicit F32 call used rounded activations");
                        if (!prompt) {
                            const auto kernels = backend::vulkan_kernel_times(b);
                            require(std::any_of(kernels.begin(), kernels.end(), [](const auto& k) {
                                return k.first.find("matmul_row") == 0;
                            }), "explicit F32 decode did not use a row kernel");
                        }
                    }
                    const size_t values = op == 2 ? 2 * count : count;
                    for (size_t i = 0; i < values; ++i) {
                        const size_t c = (i % count) / rows;
                        const bool unit_row = i % rows == 1;
                        const double product = unit_row ? (c == 1 ? -double(peak) : double(peak)) * scale.value * (op == 5 ? 0.5 : 1.0) : 0;
                        const float expected = float(initial + product);
                        const double bound = unit_row ? std::abs(product) * (dtype == backend::Dtype::f32 ? 2.0 * std::numeric_limits<float>::epsilon() : 1.0 / 32767.0) + 2 * double(std::numeric_limits<float>::denorm_min()) : 0;
                        const bool valid = std::isfinite(expected)
                            ? std::isfinite(output[i + 1]) && std::abs(double(output[i + 1]) - expected) <= bound
                            : output[i + 1] == expected;
                        if (!valid) {
                            if (failed < 12) std::fprintf(stderr, "offset range: type %u scale %u encoding %d width %zu peak %.9g op %d value %zu got %.9g expected %.9g\n",
                                                        type, unsigned(scale.bits), encoding, width, double(peak), op, i, output[i + 1], expected);
                            ++failed;
                        }
                        ++checked;
                    }
                    require(output.front() == 19 && output.back() == 19, "offset range output guard changed");
                    for (size_t i = values + 1; i + 1 < output.size(); ++i) require(output[i] == initial, "offset range wrote past output");
                }
            }
        }
    }
    std::cout << "vulkan-quantization: " << (dtype == backend::Dtype::f32 ? "F32 " : "") << checked << (prompt ? " prompt offset range values, " : " decode offset range values, ") << failed << " failed\n";
    require(failed == 0, "quantized offset arithmetic loses a finite result");
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
    const uint32_t n = uint32_t(input.size());
    constexpr uint32_t guard = 64, sentinel = 0x12345678u;
    const uint32_t words16 = n / 2 + n / 16;
    std::vector<uint32_t> packed(words16 + 2 * guard, sentinel), twin16(words16 + 2 * guard, sentinel);
    const auto x = b.adopt(input.data(), input.size() * sizeof(float));
    const auto q = b.adopt(packed.data(), packed.size() * sizeof(uint32_t));
    const auto t = b.adopt(twin16.data(), twin16.size() * sizeof(uint32_t));
    backend::VulkanQuantizationTest::run(b, {x.get(), 0}, {q.get(), guard}, n, false);
    backend::VulkanQuantizationTest::run(b, {x.get(), 0}, {t.get(), guard}, n, true);
    b.read(*q, 0, packed.data(), packed.size() * sizeof(uint32_t));
    b.read(*t, 0, twin16.data(), twin16.size() * sizeof(uint32_t));
    for (size_t i = 0; i < guard; ++i) {
        require(packed[i] == sentinel && packed[packed.size() - 1 - i] == sentinel, "packed activation guard changed");
        require(twin16[i] == sentinel && twin16[twin16.size() - 1 - i] == sentinel, "word activation guard changed");
    }
    for (size_t i = 0; i < words16; ++i)
        require(twin16[guard + i] == packed[guard + i], "16-bit word and lane writers differ");
    size_t failures = 0, sums = 0;
    for (size_t block = 0; block < peaks.size(); ++block) {
        constexpr int limit = 32767;
        const size_t tab = guard + n / 2 + block * 2;
        const float scale = from_bits(packed[tab]);
        const double step = reconstruction_step(peaks[block], limit);
        bool bad = !std::isfinite(scale) || scale < 0 || double(scale) > step || (peaks[block] > 0 && scale == 0);
        int sum = 0;
        for (size_t j = 0; j < 32; ++j) {
            const size_t i = block * 32 + j;
            const uint32_t word = packed[guard + block * 16 + 2 * (j / 4) + j % 2];
            const uint32_t code = (word >> (j % 4 >= 2 ? 16 : 0)) & 65535u;
            const int quantized = int(code) - (code > uint32_t(limit) ? 2 * (limit + 1) : 0);
            const double actual = double(scale) * quantized;
            const double bound = .50001 * step + 3e-7 * std::abs(double(input[i]));
            bad |= std::abs(quantized) > limit || !std::isfinite(actual) || std::abs(actual - input[i]) > bound;
            sum += quantized;
        }
        bad |= !sum_close(from_bits(packed[tab + 1]), double(scale) * sum);
        ++sums;
        if (bad) {
            if (failures < 8) std::cerr << "activation packing: " << "16 bits, block " << block
                                      << ", peak " << peaks[block] << ", scale " << scale << '\n';
            ++failures;
        }
    }
    std::cout << "vulkan-quantization: " << input.size() << " values per twin, " << sums << " packed sums, "
              << words16 << " identical lane/word words, " << failures << " failing blocks\n";
    require(failures == 0, "packed activation reconstruction or sums exceed their bounds");
    // Both writers must agree on the 8-bit twin and preserve the preceding 16-bit twin.
    const size_t base = backend::VulkanQuantizationTest::x8_base_words(n), words8 = base + n / 4 + n / 16;
    std::vector<uint32_t> twin8(words8 + 2 * guard, sentinel), lanes8(words8 + 2 * guard, sentinel), alone(n / 2 + n / 16 + 2 * guard, sentinel);
    const auto e = b.adopt(twin8.data(), twin8.size() * sizeof(uint32_t));
    const auto f = b.adopt(lanes8.data(), lanes8.size() * sizeof(uint32_t));
    const auto g = b.adopt(alone.data(), alone.size() * sizeof(uint32_t));
    // A prior block-major layout must not leak into the word writer's plain layout.
    // Its at most seven padded blocks fit inside the guard if the helper misses the layout word.
    backend::VulkanQuantizationTest::prior_layout(b);
    backend::VulkanQuantizationTest::run8(b, {x.get(), 0}, {e.get(), guard}, n, true);
    backend::VulkanQuantizationTest::run8(b, {x.get(), 0}, {f.get(), guard}, n, false);
    backend::VulkanQuantizationTest::run(b, {x.get(), 0}, {g.get(), guard}, n, true);
    b.read(*e, 0, twin8.data(), twin8.size() * sizeof(uint32_t));
    b.read(*f, 0, lanes8.data(), lanes8.size() * sizeof(uint32_t));
    b.read(*g, 0, alone.data(), alone.size() * sizeof(uint32_t));
    for (size_t i = 0; i < guard; ++i)
        require(twin8[i] == sentinel && twin8[twin8.size() - 1 - i] == sentinel && lanes8[i] == sentinel && lanes8[lanes8.size() - 1 - i] == sentinel,
                "8-bit activation guard changed");
    for (size_t i = 0; i < n / 2 + n / 16; ++i) require(twin8[guard + i] == alone[guard + i], "16-bit twin beside the 8-bit one changed");
    size_t lane_words8 = 0;
    for (size_t i = base; i < words8; ++i) {
        require(twin8[guard + i] == lanes8[guard + i], "lane and word 8-bit writers differ");
        ++lane_words8;
    }
    size_t failures8 = 0;
    for (size_t block = 0; block < peaks.size(); ++block) {
        constexpr int limit = 127;
        const size_t tab = guard + base + n / 4 + block * 2;
        const float scale = from_bits(twin8[tab]);
        const double step = reconstruction_step(peaks[block], limit);
        bool bad = !std::isfinite(scale) || scale < 0 || double(scale) > step || (peaks[block] > 0 && scale == 0);
        int sum = 0;
        for (size_t j = 0; j < 32; ++j) {
            const size_t i = block * 32 + j;
            const int quantized = int(int8_t(uint8_t(twin8[guard + base + block * 8 + j / 4] >> (8 * (j % 4)))));
            const double actual = double(scale) * quantized;
            const double bound = .50001 * step + 3e-7 * std::abs(double(input[i]));
            bad |= std::abs(quantized) > limit || !std::isfinite(actual) || std::abs(actual - input[i]) > bound;
            sum += quantized;
        }
        bad |= !sum_close(from_bits(twin8[tab + 1]), double(scale) * sum);
        if (bad) {
            if (failures8 < 8) std::cerr << "activation packing: 8 bits, block " << block << ", peak " << peaks[block] << ", scale " << scale << '\n';
            ++failures8;
        }
    }
    std::cout << "vulkan-quantization: " << peaks.size() << " blocks of the 8-bit twin, " << lane_words8 << " identical lane/word words, " << failures8 << " failing\n";
    require(failures8 == 0, "8-bit twin reconstruction or sums exceed their bounds");
}

// The words a twin of n values holds in `major`'s layout: the 16-bit twin, then the 8-bit one when twin8, without a block-major twin's padding columns.
std::vector<size_t> twin_words(size_t n, bool twin8, uint32_t major) {
    std::vector<size_t> at;
    for (size_t i = 0; i < n / 2 + n / 16; ++i) at.push_back(i);
    if (!twin8) return at;
    const size_t base = backend::VulkanQuantizationTest::x8_base_words(n);
    if (!major) {
        for (size_t i = 0; i < n / 4 + n / 16; ++i) at.push_back(base + i);
        return at;
    }
    const size_t cols = n / 32 / major, pad = backend::VulkanQuantizationTest::x8_columns(uint32_t(cols)), tab = base + pad * major * 8;
    for (size_t k = 0; k < major; ++k)
        for (size_t c = 0; c < cols; ++c) {
            const size_t blk = k * pad + c;
            for (size_t w = 0; w < 8; ++w) at.push_back(base + blk * 8 + w);
            at.push_back(tab + 2 * blk);
            at.push_back(tab + 2 * blk + 1);
        }
    return at;
}

// Grouped RMSNorm against a double reference, the twin it leaves against the quantizers' twin of its output, and each row alone against the same row among others, all bit for bit but the reference.
void check_group_rms_norm(backend::VulkanBackend& b) {
    using T = backend::VulkanQuantizationTest;
    constexpr double U = 0x1p-24;
    constexpr float eps = 1e-6f;
    uint32_t seed = 2027;
    size_t cases = 0, values = 0, words = 0, alone = 0, tiled = 0, major_cases = 0, twin8_cases = 0;
    auto run_case = [&](size_t dim, size_t groups, size_t rows, size_t extent, bool twin8) {
        const size_t width = groups * dim, n = rows * width;
        std::vector<float> x(n), w(width), y(n), single(width);
        for (float& v : x) v = float(int(random_bits(seed) % 65535u) - 32767) / 23405.0f;
        for (float& v : w) v = 0.6f + float(random_bits(seed) % 65536u) / 81920.0f;
        // Every allocation first, since a new buffer drops every twin.
        const auto dev = backend::Memory::device;
        const auto xs = b.alloc(n * 4, dev), ws = b.alloc(width * 4, dev), ys = b.alloc(n * 4, dev), one = b.alloc(width * 4, dev), ip = b.alloc(n * 4, dev);
        const auto ref = b.alloc(T::x8_base_words(n) * 4 + (n + n / 32 * 8) / 4 * 5, dev);
        b.write(*xs, 0, x.data(), n * 4);
        b.write(*ws, 0, w.data(), width * 4);
        b.write(*ip, 0, x.data(), n * 4);
        T::set_twin8(b, twin8);
        const backend::RowRun run{rows, extent};
        const backend::RowRuns runs{&run, 1};
        const bool quant = dim % 32 == 0, tile = quant && T::tile_reads(b, width, rows, runs);
        b.group_rms_norm({ys.get(), 0}, {xs.get(), 0}, {ws.get(), 0}, rows, groups, dim, eps, runs);
        uint32_t major = 0;
        bool has8 = false;
        const auto kept = T::kept_twin(b, {ys.get(), 0}, n, major, has8);
        b.read(*ys, 0, y.data(), n * 4);
        require(kept.empty() != quant, "grouped RMSNorm left a twin other than where its groups are whole blocks");
        if (quant) {
            require(has8 == twin8, "grouped RMSNorm's twin lacks or adds the 8-bit twin");
            T::quantize(b, {ys.get(), 0}, {ref.get(), 0}, uint32_t(n), tile, major, twin8);
            std::vector<uint32_t> expected(ref->size() / 4);
            b.read(*ref, 0, expected.data(), expected.size() * 4);
            for (size_t at : twin_words(n, twin8, major)) {
                if (at >= kept.size() || kept[at] != expected[at]) {
                    std::fprintf(stderr, "grouped RMSNorm twin: dim %zu groups %zu rows %zu extent %zu twin8 %d major %u word %zu\n",
                                 dim, groups, rows, extent, int(twin8), major, at);
                    require(false, "grouped RMSNorm's twin differs from the quantizers' twin of its output");
                }
                ++words;
            }
            tiled += tile;
            major_cases += major != 0;
            twin8_cases += twin8;
        }
        for (size_t r = 0; r < rows; ++r)
            for (size_t g = 0; g < groups; ++g) {
                const size_t at = r * width + g * dim;
                double square = 0;
                for (size_t d = 0; d < dim; ++d) square += double(x[at + d]) * x[at + d];
                const double inv = 1.0 / std::sqrt(square / double(dim) + eps);
                for (size_t d = 0; d < dim; ++d) {
                    const double e = double(x[at + d]) * inv * w[g * dim + d];
                    if (!(std::abs(double(y[at + d]) - e) <= double(dim + 12) * U * std::abs(e) + 1e-30)) {
                        std::fprintf(stderr, "grouped RMSNorm: dim %zu groups %zu rows %zu value %zu device %.9g expected %.9g\n",
                                     dim, groups, rows, at + d, y[at + d], e);
                        require(false, "grouped RMSNorm exceeds its bound against the double reference");
                    }
                    ++values;
                }
            }
        // Exactly in place, as the model also calls it, the output and twin must be the same.
        b.group_rms_norm({ip.get(), 0}, {ip.get(), 0}, {ws.get(), 0}, rows, groups, dim, eps, runs);
        uint32_t ip_major = 0;
        const auto ip_kept = T::kept_twin(b, {ip.get(), 0}, n, ip_major, has8);
        std::vector<float> in_place(n);
        b.read(*ip, 0, in_place.data(), n * 4);
        require(std::memcmp(in_place.data(), y.data(), n * 4) == 0, "grouped RMSNorm in place differs from out of place");
        require(ip_kept.empty() == kept.empty() && ip_major == major, "grouped RMSNorm in place left another twin");
        if (quant)
            for (size_t at : twin_words(n, twin8, major))
                require(ip_kept[at] == kept[at], "grouped RMSNorm's twin in place differs from out of place");
        if (rows > 1)
            for (size_t r = 0; r < rows; ++r) {
                const backend::RowRun lone{1, extent};
                b.group_rms_norm({one.get(), 0}, {xs.get(), r * width}, {ws.get(), 0}, 1, groups, dim, eps, {&lone, 1});
                b.read(*one, 0, single.data(), width * 4);
                require(std::memcmp(single.data(), y.data() + r * width, width * 4) == 0,
                        "grouped RMSNorm row alone differs from the same row among others");
                ++alone;
            }
        ++cases;
    };
    const size_t dims[] = {2560, 2048, 4128, 2501, 1000, 260, 64}, group_counts[] = {1, 4}, row_counts[] = {1, 3, 4, 33};
    const bool preferred = T::prefers_integer_dot(b);
    for (bool forced : {false, true}) {
        if (forced && preferred) break;
        T::prefer_integer_dot(b, preferred || forced);
        for (size_t dim : dims)
            for (size_t groups : group_counts)
                for (size_t rows : row_counts)
                    for (size_t extent : {size_t(1), size_t(512)})
                        for (bool twin8 : {false, true})
                            if (!forced || extent != 1) run_case(dim, groups, rows, extent, twin8);
    }
    T::prefer_integer_dot(b, preferred);
    T::set_twin8(b, false);
    std::cout << "vulkan-quantization: grouped RMSNorm, " << cases << " cases, " << values << " values within the reference bound, "
              << words << " twin words equal to the quantizers' (" << twin8_cases << " cases with the 8-bit twin, " << tiled
              << " tile layouts, " << major_cases << " block-major), " << alone << " rows alone equal\n";
    require(tiled != 0 && major_cases != 0, "grouped RMSNorm never wrote the tile's twin or a block-major one");
}
}

int main() {
    try {
        backend::VulkanBackend b(0);
        check(b);
        check_group_rms_norm(b);
        for (size_t width : {size_t(32), size_t(64)})
            check_activation_range(b, width, backend::VulkanQuantizationTest::preserves_float32(b));
        backend::VulkanBackend fallback(0);
        backend::VulkanQuantizationTest::WithoutPreservation guard(fallback);
        for (size_t width : {size_t(32), size_t(64)}) check_activation_range(fallback, width, false);
        require(guard.modules != 0, "unsupported-mode check created no modules");
        std::cout << "vulkan-quantization: " << guard.modules << " modules accepted without float preservation\n";
        check_ordinary_identity(b, fallback);
        check_float_identity(b, fallback);
        check_offset_range(b);
        backend::VulkanBackend tiled(0, true);
        check_offset_range(tiled, true);
        check_offset_range(tiled, false, backend::Dtype::f32);
        check_offset_range(tiled, true, backend::Dtype::f32);
        return 0;
    } catch (const backend::VulkanUnavailable& e) {
        std::cerr << e.what() << '\n';
        return 77;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
