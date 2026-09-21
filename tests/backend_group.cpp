#include <string>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include "backends/cpu/cpu_backend.hpp"

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static size_t check_prefill_reduction() {
    size_t count = 0;
    for (size_t n : std::array<size_t, 19>{0,1,2,7,8,9,15,16,17,31,32,33,127,128,129,1023,1024,1025,2048}) {
        for (size_t offset = 0; offset < 8; ++offset) {
            const size_t stride = n + 13;
            std::vector<float> rows(4 * stride + 8), inputs(3 * stride + 8);
            for (size_t i = 0; i < rows.size(); ++i)
                rows[i] = float(int((i * 1471 + 7) % 65521) - 32760) / 317.0f;
            for (size_t i = 0; i < inputs.size(); ++i)
                inputs[i] = float(int((i * 769 + 23) % 32749) - 16374) / 523.0f;
            const float* r = rows.data() + offset;
            const float* x = inputs.data() + offset;
            float actual[12];
            backend::CpuBackend::dot_f32_x4x3(r, stride, x, x + stride, x + 2 * stride, n, actual, actual + 4, actual + 8);
            for (size_t c = 0; c < 3; ++c) for (size_t k = 0; k < 4; ++k) {
                float lanes[8] = {};
                size_t j = 0;
                for (; j + 8 <= n; j += 8)
                    for (size_t lane = 0; lane < 8; ++lane)
                        lanes[lane] = std::fma(r[k * stride + j + lane], x[c * stride + j + lane], lanes[lane]);
                float v = lanes[0];
                for (size_t lane = 1; lane < 8; ++lane) v += lanes[lane];
                for (; j < n; ++j) v += r[k * stride + j] * x[c * stride + j];
                require(std::memcmp(&v, &actual[c * 4 + k], sizeof(float)) == 0,
                        "prefill reduction differs from ordered scalar FMA oracle");
                ++count;
            }
        }
    }
    return count;
}

static size_t check_q8_scales(backend::CpuBackend& cpu) {
    size_t count = 0;
    std::array<uint8_t, 34> row{};
    std::array<float, 32> x{};
    // The CPU backend adopts the caller's pointer rather than copying, so one
    // handle per array stays valid while the sweep rewrites them in place.
    float actual = 0.0f;
    const auto row_buf = cpu.adopt(row.data(), row.size());
    const auto x_buf = cpu.adopt(x.data(), x.size() * sizeof(float));
    const auto out_buf = cpu.adopt(&actual, sizeof(float));
    for (unsigned h = 0; h < 65536; ++h) {
        if ((h & 0x7c00) == 0x7c00) continue;
        row[0] = uint8_t(h);
        row[1] = uint8_t(h >> 8);
        const size_t lane = h % x.size();
        x[lane] = 1.0f;
        for (int q : {-128, -1, 0, 127}) {
            std::fill(row.begin() + 2, row.end(), uint8_t(q));
            // A one-hot input makes every finite f16 scale times int8 exact
            // in f32, independently of the SIMD reduction order.
            const float expected = f16_to_f32(uint16_t(h)) * float(q);
            actual = 0.0f;
            cpu.matmul(gguf::GGML_TYPE_Q8_0, {row_buf.get(), 0}, {x_buf.get(), 0},
                       {out_buf.get(), 0}, gguf::Q8_0_BLOCK, 1, 1);
            require(std::isfinite(actual) && actual == expected, "Q8 scale or signed weight differs");
            ++count;
        }
        x[lane] = 0.0f;
    }
    return count;
}

struct Matrix {
    uint32_t type;
    size_t rows;
    std::vector<float> weights;
    std::vector<uint8_t> packed;
    std::vector<float> separate, grouped;

    const uint8_t* data() const {
        return type == gguf::GGML_TYPE_F32
            ? reinterpret_cast<const uint8_t*>(weights.data()) : packed.data();
    }
    size_t bytes() const {
        return type == gguf::GGML_TYPE_F32 ? weights.size() * sizeof(float) : packed.size();
    }

    Matrix(uint32_t t, size_t n, size_t width, size_t batch) : type(t), rows(n) {
        weights.resize(rows * width);
        for (size_t i = 0; i < weights.size(); ++i)
            weights[i] = float(int((i * 37 + 13) % 257) - 128) / 128.0f;
        if (type != gguf::GGML_TYPE_F32) {
            const auto* q = quant::Registry::instance().get(type);
            const size_t blocks = weights.size() / q->block_size;
            packed.resize(blocks * q->type_size);
            if (q->quantize) q->quantize(weights.data(), packed.data(), blocks);
            else {
                for (size_t i = 0; i < packed.size(); ++i)
                    packed[i] = uint8_t(i * 73 + 19);
                for (size_t b = 0; b < blocks; ++b) {
                    uint8_t* p = packed.data() + b * q->type_size;
                    // Finite binary-power scales, including f16 subnormals.
                    const uint16_t scale = b % 3 == 0 ? 1 : 0x2000;
                    const size_t offset = type == gguf::GGML_TYPE_Q6_K ? 208 : 0;
                    std::memcpy(p + offset, &scale, sizeof(scale));
                    if (type != gguf::GGML_TYPE_Q6_K)
                        std::memcpy(p + 2, &scale, sizeof(scale));
                }
            }
            q->dequantize(packed.data(), weights.data(), blocks);
        }
        separate.assign(rows * batch + 2, 123456.0f);
        grouped = separate;
    }

    // Adopted once per matrix; the fixture outlives the handle.
    backend::BufferPtr buffer(backend::CpuBackend& cpu) {
        if (!handle) handle = cpu.adopt(data(), bytes());
        return handle;
    }
    backend::Projection projection(backend::CpuBackend& cpu) {
        return {type, {buffer(cpu).get(), 0}, {out_buffer(cpu).get(), 1}, rows};
    }
    backend::BufferPtr out_buffer(backend::CpuBackend& cpu) {
        if (!out_handle) out_handle = cpu.adopt(grouped.data(), grouped.size() * sizeof(float));
        return out_handle;
    }
    backend::BufferPtr sep_buffer(backend::CpuBackend& cpu) {
        if (!sep_handle) sep_handle = cpu.adopt(separate.data(), separate.size() * sizeof(float));
        return sep_handle;
    }
    backend::BufferPtr out_handle, sep_handle;
    backend::BufferPtr handle;
};

static size_t check(backend::CpuBackend& cpu, std::array<uint32_t, 3> types,
                    size_t width, size_t rows, size_t batch) {
    std::array<Matrix, 3> matrices = {
        Matrix(types[0], rows, width, batch),
        Matrix(types[1], rows + 2, width, batch),
        Matrix(types[2], rows + 4, width, batch)};
    std::vector<float> x(width * batch);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = float(int((i * 19 + 7) % 101) - 50) / 32.0f;
    const auto original = x;
    const auto x_buf = cpu.adopt(x.data(), x.size() * sizeof(float));
    for (auto& m : matrices)
        cpu.matmul(m.type, {m.buffer(cpu).get(), 0}, {x_buf.get(), 0},
                   {m.sep_buffer(cpu).get(), 1}, width, m.rows, batch);
    cpu.matmul_group({matrices[0].projection(cpu), matrices[1].projection(cpu),
                      matrices[2].projection(cpu)}, {x_buf.get(), 0}, width, batch);
    require(x == original, "grouped matmul modified activations");
    size_t count = 0;
    for (const auto& m : matrices) {
        require(std::memcmp(m.separate.data(), m.grouped.data(), m.grouped.size() * sizeof(float)) == 0,
                "grouped output differs from separate calls or overwrote a sentinel");
        require(m.grouped.front() == 123456.0f && m.grouped.back() == 123456.0f,
                "output boundary overwritten");
        for (size_t b = 0; b < batch; ++b) {
            for (size_t o = 0; o < m.rows; ++o) {
                double expected = 0, magnitude = 0;
                for (size_t i = 0; i < width; ++i) {
                    const double product = double(m.weights[o * width + i]) * x[b * width + i];
                    expected += product;
                    magnitude += std::abs(product);
                }
                const float actual = m.grouped[1 + b * m.rows + o];
                require(std::isfinite(actual) && std::abs(actual - expected) <= 2e-6 * (1 + magnitude),
                        "output differs from double-precision dot oracle");
                ++count;
            }
        }
    }
    auto& first = matrices[0];
    std::fill(first.grouped.begin() + 1, first.grouped.end() - 1, 123456.0f);
    cpu.matmul_group({first.projection(cpu)}, {x_buf.get(), 0}, width, batch);
    require(first.grouped == first.separate, "single-projection fallback differs");
    return count;
}

// Activation magnitude sweep. The ordinary cases above generate x within
// about +-1.6, which is why this file passed a kernel that produced Inf and
// NaN on large inputs: a fused dot accumulates sum(q*x) and applies the block
// scale afterwards, so the inner sum can overflow before a small scale would
// have bounded it. The oracle bound is relative to magnitude, so extreme
// scales are testable here without loosening anything.
static size_t check_magnitudes(backend::CpuBackend& cpu) {
    size_t values = 0;
    for (float mag : {1e-30f, 1e-8f, 1.0f, 1e8f, 1e30f, 1e36f}) {
        for (uint32_t type : {gguf::GGML_TYPE_Q8_0, gguf::GGML_TYPE_Q4_0,
                              gguf::GGML_TYPE_Q4_1, gguf::GGML_TYPE_Q4_K,
                              gguf::GGML_TYPE_Q5_K, gguf::GGML_TYPE_Q6_K,
                              gguf::GGML_TYPE_F32}) {
            const size_t width = type == gguf::GGML_TYPE_F32 ? 37 : 256;
            Matrix m(type, 17, width, 1);
            std::vector<float> x(width);
            const auto x_buf = cpu.adopt(x.data(), x.size() * sizeof(float));
            // Sign matters as much as magnitude. A mixed-sign pattern lets the
            // inner sum cancel and never reach the overflow window, which is
            // why an earlier version of this sweep passed a kernel that
            // produced Inf. Same-sign inputs maximise sum(q*x) instead.
            for (size_t i = 0; i < width; ++i)
                x[i] = mag >= 1e30f ? mag
                     : mag * float(int((i * 19 + 7) % 101) - 50) / 50.0f;
            cpu.matmul(m.type, {m.buffer(cpu).get(), 0}, {x_buf.get(), 0},
                   {m.sep_buffer(cpu).get(), 1}, width, m.rows, 1);
            for (size_t o = 0; o < m.rows; ++o) {
                double expected = 0, magnitude = 0;
                for (size_t i = 0; i < width; ++i) {
                    const double product = double(m.weights[o * width + i]) * x[i];
                    expected += product;
                    magnitude += std::abs(product);
                }
                // Only meaningful where the true answer is representable. At
                // extreme magnitudes some rows genuinely exceed FLT_MAX, and
                // returning infinity for those is correct rather than a bug.
                // The interesting rows are the ones whose result fits while
                // the kernel's intermediate sum(q*x) does not - which is the
                // overflow this sweep exists to catch.
                if (!(std::abs(expected) <= 3.0e38)) continue;
                const float actual = m.separate[1 + o];
                require(std::isfinite(actual),
                        ("nonfinite output where the exact result fits: type " +
                         std::to_string(type) + " mag " +
                         std::to_string(mag)).c_str());
                require(std::abs(actual - expected) <= 2e-6 * (1 + magnitude),
                        "magnitude sweep differs from double-precision dot oracle");
                ++values;
            }
        }
    }
    return values;
}

int main() {
    try {
        quant::register_builtins();
        backend::CpuBackend cpu;
        cpu.set_threads(1);
        const size_t scales = check_q8_scales(cpu);
        const size_t reductions = check_prefill_reduction();
        const size_t magnitudes = check_magnitudes(cpu);
        size_t values = 0, cases = 0;
        for (int threads : {1, 2, 6}) {
            cpu.set_threads(threads);
            cpu.matmul_group({}, {}, 0, 1);
            for (size_t rows : {size_t(7), size_t(47), size_t(48), size_t(65)}) {
                for (size_t batch : {size_t(0), size_t(1), size_t(2), size_t(3), size_t(4)}) {
                    for (uint32_t type : {gguf::GGML_TYPE_F32, gguf::GGML_TYPE_Q8_0,
                            gguf::GGML_TYPE_Q4_0, gguf::GGML_TYPE_Q4_1, gguf::GGML_TYPE_Q4_K,
                            gguf::GGML_TYPE_Q5_K, gguf::GGML_TYPE_Q6_K}) {
                        values += check(cpu, {type, type, type}, type == gguf::GGML_TYPE_F32 ? 37 : 256,
                                        rows, batch);
                        ++cases;
                    }
                    values += check(cpu, {gguf::GGML_TYPE_Q8_0, gguf::GGML_TYPE_F32, gguf::GGML_TYPE_Q4_K},
                                    256, rows, batch);
                    ++cases;
                    values += check(cpu, {gguf::GGML_TYPE_Q8_0, gguf::GGML_TYPE_Q6_K, gguf::GGML_TYPE_Q4_K},
                                    256, rows, batch);
                    ++cases;
                }
            }
        }
        // An unknown quant type must be rejected, with real storage behind
        // the projection so the type is what fails.
        std::vector<uint8_t> scratch(65 * 256, 0);
        const auto storage = cpu.adopt(scratch.data(), scratch.size());
        std::vector<float> sink(67, 0.0f);
        bool rejected = false;
        try {
            const auto sink_buf = cpu.adopt(sink.data(), sink.size() * sizeof(float));
            cpu.matmul_group({{9999, {storage.get(), 0}, {sink_buf.get(), 0}, 65},
                              {9999, {storage.get(), 0}, {sink_buf.get(), 0}, 67}},
                             {}, 256, 1);
        } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "invalid quant type was not rejected on caller");
        // A projection without storage is rejected before anything reads it.
        rejected = false;
        try {
            const auto sink2 = cpu.adopt(sink.data(), sink.size() * sizeof(float));
            cpu.matmul_group({{gguf::GGML_TYPE_Q8_0, {}, {sink2.get(), 0}, 65},
                              {gguf::GGML_TYPE_Q8_0, {}, {sink2.get(), 0}, 67}},
                             {}, 256, 1);
        } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "projection without storage was accepted");
        std::cout << "grouped projections: " << cases << " cases, " << values
                  << " outputs checked against separate calls and double dots; "
                  << scales << " exact finite Q8 scale/weight cases; "
                  << reductions << " ordered prefill reductions; "
                  << magnitudes << " magnitude-sweep outputs\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
