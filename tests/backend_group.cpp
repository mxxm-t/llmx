#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include "backends/cpu/cpu_backend.hpp"

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
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

    backend::Projection projection() {
        return {type, data(), grouped.data() + 1, rows};
    }
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
    for (auto& m : matrices)
        cpu.matmul(m.type, m.data(), x.data(), m.separate.data() + 1, width, m.rows, batch);
    cpu.matmul_group({matrices[0].projection(), matrices[1].projection(), matrices[2].projection()},
                     x.data(), width, batch);
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
    cpu.matmul_group({first.projection()}, x.data(), width, batch);
    require(first.grouped == first.separate, "single-projection fallback differs");
    return count;
}

int main() {
    try {
        quant::register_builtins();
        backend::CpuBackend cpu;
        size_t values = 0, cases = 0;
        for (int threads : {1, 2, 6}) {
            cpu.set_threads(threads);
            cpu.matmul_group({}, nullptr, 0, 1);
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
        bool rejected = false;
        try { cpu.matmul_group({{9999, nullptr, nullptr, 65}, {9999, nullptr, nullptr, 67}}, nullptr, 256, 1); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "invalid quant type was not rejected on caller");
        std::cout << "grouped projections: " << cases << " cases, " << values
                  << " outputs checked against separate calls and double dots\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
