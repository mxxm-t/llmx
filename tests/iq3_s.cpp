#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
#include "config.hpp"
#include "backends/cpu/cpu_backend.hpp"
#if LLMX_HAS_BACKEND_VULKAN
#include "backends/vulkan/vulkan_backend.hpp"
#endif
#include <stdexcept>
#include "quant/quant.hpp"
#include "matrix_precision.hpp"
#include "row_classes.hpp"

namespace {
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
uint32_t bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }
// Independently pinned decoded bytes, rather than the runtime's packed words: ggml eefc4e7321c869496146697d63362f073941aed6, LE-byte SHA256 bd1af4945a1717c65610b0284e4628b9a1ba3b306fae3a06f6e5f597356e349f.
constexpr char grid[] =
    "0101010103010101050101010b0101010f010101010301010303010105030101090301010d03010101050101030501010b050101070701010109010105090101"
    "0b0901010f090101030b0101070b0101010d0101050d0101030f0101090f01010f0f01010101030103010301050103010901030101030301030303010b030301"
    "01050301070503010f050301030703010b07030109090301030d03010b0d0301050f030101010501030105010b0105010f01050101030501070305010d030501"
    "030505010b0505010107050109070501050905010b0905010f090501030b0501070b0501010f0501070f050107010701030307010b0307010105070105050701"
    "03070701070707010d07070109090701010b0701050b07010f0d0701030f07010b0f070101010901070309010f03090103050901090509010507090101090901"
    "07090901030b0901010f090105010b0109010b0101050b0105050b010d050b0107070b0103090b010b090b010f090b010d0d0b01070f0b010d010d0103030d01"
    "07030d0103070d01050b0d01030f0d0101010f0105010f0109010f0101050f0105050f010d050f0107070f01010b0f01090b0f01010101030301010305010103"
    "090101030103010303030103070301030b0301030f030103010501030505010303070103090701030d070103090b01030d0b0103030d0103050f010301010303"
    "03010303070103030d010303010303030903030303050303010703030707030303090303010b0303050b0303010f03030d0f030301010503050305030b030503"
    "0f03050301050503090505030507050301090503070905030b0b0503010d0503050f050303010703090107030f0107030103070307030703030507030f050703"
    "010707030907070303090703050d0703010f0703070109030b01090305030903090309030307090307070903050909030d090903010b0903090b090303010b03"
    "01030b0307030b0303050b0301070b0305070b03030b0b0301050d0309050d030f050d0309090d030d090d0303010f0307010f0301030f0305030f0303050f03"
    "0b070f0303090f03050d0f03010f0f030101010503010105070101050b0101050f0101050103010505030105090301050d03010503050105070501050f050105"
    "010701050507010503090105070901050b090105010b0105050b01050f0d0105010f0105070f01050b0f0105010103050501030501030305070303050f030305"
    "050503050b050305030703050907030505090305030b030503010505090105050f0105050305050507050505010705050f07050503090505070b05050f0b0505"
    "030f0505090f050501010705050107050b010705030307050505070509050705030707050707070505090705010b07050d0d0705030109050f01090501050905"
    "07050905050709050b07090503090905050f09050b0f090509010b0503030b0505050b050f070b0501090b05070b0b05010f0b0501010d0505010d050f010d05"
    "03050d050b0b0d05030d0d050b010f0503030f050d050f0501070f0507090f05010b0f050501010703030107070301070b0301070f0301070505010703070107"
    "070701070b07010705090107090901070f090107030b0107070d0107030f010703010307070103070b01030709030307030503070705030701090307010d0307"
    "050f03070d0f03070101050705030507010505070507050709070507010b050703010707010307070903070703050707070507070f0507070107070703090707"
    "070907070f0907070b0b0707070f070707010907030309070d0309070505090703070907050b0907010d0907090d090703010b0701030b0705030b070b050b07"
    "05070b0709090b070d0b0b07070f0b070d030d0703090d0703010f0707010f0701050f0705050f070b070f070101010909010109050301090105010909050109"
    "0f0501090507010903090109010b0109010f0109050103090f010309030303090703030905050309010703090b07030907090309030b03090b0b030903010509"
    "07010509010305090b0305090305050907070509010905090f0b0509050d0509010f05090901070903030709070307090105070905050709030707090b070709"
    "0101090905010909090509090f07090901090909030f09090b010b090f010b0903050b09050d0b0907030d0909070d09010d0d0901030f090b030f0901070f09"
    "07090f09030b0f090501010b0103010b0903010b0505010b0109010b0909010b0f09010b050b010b0d0d010b090f010b0301030b0701030b0b01030b0503030b"
    "0305030b0507030b050f030b0101050b0303050b0705050b0107050b0d07050b070b050b0501070b0f01070b0103070b0f05070b0909070b030b070b0b0d070b"
    "070f070b0301090b0901090b0105090b0507090b0d09090b05030b0b0d050b0b030b0b0b070b0b0b05090d0b05010f0b09010f0b05050f0b0303010d0703010d"
    "0b03010d0307010d0707010d010d010d0101030d0105030d0f05030d090d030d0503050d0907050d0509050d0b0b050d050d050d010f050d0101070d0903070d"
    "0305070d0109070d0b05090d0709090d050d090d01010b0d07010b0d09070b0d010d0b0d0b010d0d01090d0d03030f0d07030f0d0101010f0901010f0f01010f"
    "0105010f0505010f0d07010f0109010f090b010f050d010f0501030f0303030f0905030f0709030f0b09030f0301050f0901050f0103050f0d03050f0305050f"
    "0107050f030b050f0501070f0507070f0b07070f070b070f0301090f0b01090f0703090f0105090f010b090f05050b0f05090b0f05010d0f03070d0f01010f0f";
unsigned digit(char c) { return c <= '9' ? unsigned(c - '0') : unsigned(c - 'a' + 10); }
unsigned magnitude(unsigned index, unsigned lane) {
    const size_t at = 2 * (4 * index + lane);
    return 16 * digit(grid[at]) + digit(grid[at + 1]);
}
float expected(uint16_t h, unsigned index, unsigned lane, unsigned scale, bool negative) {
    const unsigned exponent = (h >> 10) & 31, fraction = h & 1023;
    double d = exponent ? std::ldexp(double(1024 + fraction), int(exponent) - 25)
                        : std::ldexp(double(fraction), -24);
    if (h & 0x8000) d = -d;
    const double value = d * (1 + 2 * scale) * magnitude(index, lane);
    return float(negative ? -value : value);
}
using Indices = std::array<unsigned, 64>;
using Scales = std::array<unsigned, 8>;
using Signs = std::array<bool, 256>;
size_t checked = 0;
void check(uint16_t h, const Indices& indices, const Scales& scales, const Signs& signs) {
    // Three blocks start at an odd address and have dissimilar neighbors.
    std::array<uint8_t, 332> source{};
    std::array<float, 770> output{};
    source.fill(0xa5); output.fill(12345.0f);
    for (unsigned b = 0; b < 3; ++b) {
        uint8_t* p = source.data() + 1 + 110 * b;
        std::memset(p, 0, 110);
        const uint16_t hb = h ^ (b == 1 ? 0x8000 : 0);
        p[0] = uint8_t(hb); p[1] = uint8_t(hb >> 8);
        for (unsigned g = 0; g < 64; ++g) {
            const unsigned ix = (indices[g] + b) % 512;
            p[2 + g] = uint8_t(ix);
            if (ix >= 256) p[66 + g / 8] |= uint8_t(1 << (g % 8));
        }
        for (unsigned i = 0; i < 256; ++i)
            if (signs[i] != (b == 2)) p[74 + i / 8] |= uint8_t(1 << (i % 8));
        for (unsigned g = 0; g < 8; ++g)
            p[106 + g / 2] |= uint8_t(scales[g] << (4 * (g % 2)));
    }
    const auto before = source;
    quant::dequantize_row_iq3_s(source.data() + 1, output.data() + 1, 3);
    require(source == before, "decoder changed input");
    require(output.front() == 12345.0f && output.back() == 12345.0f, "output guard changed");
    for (unsigned b = 0; b < 3; ++b)
        for (unsigned i = 0; i < 256; ++i) {
            const float want = expected(h ^ (b == 1 ? 0x8000 : 0), (indices[i / 4] + b) % 512,
                                        i % 4, scales[i / 32], signs[i] != (b == 2));
            require(bits(output[1 + 256 * b + i]) == bits(want), "IQ3_S decode bits differ");
            ++checked;
        }
}
// The packer starts from logical fields and computes expected weights before any backend reads the bytes. No runtime decoder or dictionary is consulted.
void fixture(uint8_t* p, float* decoded, uint16_t h, unsigned shift) {
    std::memset(p, 0, 110);
    p[0] = uint8_t(h); p[1] = uint8_t(h >> 8);
    for (unsigned g = 0; g < 64; ++g) {
        const unsigned index = (shift + 37 * g) % 512;
        p[2 + g] = uint8_t(index);
        p[66 + g / 8] |= uint8_t((index / 256) << (g % 8));
        for (unsigned lane = 0; lane < 4; ++lane) {
            const unsigned i = 4 * g + lane;
            const bool sign = (shift + i) % 3 == 0;
            if (sign) p[74 + i / 8] |= uint8_t(1 << (i % 8));
            decoded[i] = expected(h, index, lane, (shift + i / 32) % 16, sign);
        }
    }
    for (unsigned g = 0; g < 8; ++g)
        p[106 + g / 2] |= uint8_t(((shift + g) % 16) << (4 * (g % 2)));
}
std::vector<float> read(backend::Backend& b, backend::Buffer& buffer, size_t count) {
    std::vector<float> out(count);
    b.read(buffer, 0, out.data(), count * sizeof(float));
    return out;
}
size_t embeddings(backend::Backend& b) {
    constexpr size_t rows = 522, width = 256;
    const uint16_t edges[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff};
    std::vector<uint8_t> packed(rows * 110);
    std::vector<float> decoded(rows * width), initial(rows * width + 2, 12345.0f);
    std::vector<uint32_t> ids(rows);
    for (size_t row = 0; row < rows; ++row) {
        fixture(packed.data() + 110 * row, decoded.data() + width * row,
                row < 512 ? 0x3555 : edges[row - 512], unsigned(row));
        ids[row] = uint32_t(rows - 1 - row);
    }
    const auto w = b.adopt(packed.data(), packed.size());
    const auto out = b.alloc(initial.size() * sizeof(float));
    b.write(*out, 0, initial.data(), initial.size() * sizeof(float));
    b.embed({out.get(), 1}, 21, {w.get(), 0}, width, rows, ids.data(), rows);
    auto got = read(b, *out, initial.size());
    require(got.front() == initial.front() && got.back() == initial.back(), "embedding crossed output guards");
    for (size_t row = 0; row < rows; ++row)
        for (size_t i = 0; i < width; ++i)
            require(bits(got[1 + width * row + i]) == bits(decoded[width * ids[row] + i]),
                    "IQ3_S embedding differs from independent weights or zero signs");
    const uint32_t selected[] = {513, 513, 0, uint32_t(rows), UINT32_MAX};
    const auto idb = b.adopt(selected, sizeof selected);
    b.write(*out, 0, initial.data(), initial.size() * sizeof(float));
    b.embed_ids({out.get(), 1}, 21, {w.get(), 0}, width, rows, {idb.get(), 0}, 5);
    got = read(b, *out, initial.size());
    for (size_t i = 0; i < got.size(); ++i) {
        uint32_t want = bits(initial[i]);
        if (i >= 1 && i <= 5 * width) {
            const size_t row = (i - 1) / width, lane = (i - 1) % width;
            want = row < 3 ? bits(decoded[width * selected[row] + lane]) : 0;
        }
        require(bits(got[i]) == want, "IQ3_S embed_ids differs or crossed output guards");
    }
    return rows * width + 5 * width;
}
// Independent round-to-nearest-even BF16 input rounding, using the discarded half-word.
float bf16_input(float x) {
    uint32_t u = bits(x), top = u >> 16;
    const uint32_t low = u & 65535;
    if (low > 32768 || (low == 32768 && (top & 1))) ++top;
    u = top << 16;
    float rounded; std::memcpy(&rounded, &u, 4); return rounded;
}
std::vector<std::string> paths(backend::Backend& b) {
    backend::MatrixPaths out; b.swap_matrix_paths(out); return out.take();
}
// The profile and logical extent predict dispatch independently of its completed witness.
bool integer_path(backend::Backend& b, backend::Dtype dtype, size_t extent) {
#if LLMX_HAS_BACKEND_VULKAN
    if (!b.is_cpu() && (dtype == backend::Dtype::f16 || dtype == backend::Dtype::int8)) {
        const auto profile = backend::vulkan_device_profile(b);
        return extent == 1 ? profile.lookup_integer_dot : profile.prefer_integer_dot;
    }
#else
    (void)b; (void)dtype; (void)extent;
#endif
    return false;
}
size_t integer_calls = 0;
size_t products(backend::Backend& b, size_t width, size_t rows, size_t cols, bool one_hot) {
    const size_t blocks = width / 256, count = rows * cols;
    std::vector<uint8_t> packed(rows * blocks * 110);
    std::vector<float> decoded(rows * width), input(cols * width + 2, 0), initial(2 * count + 2, 12345.0f);
    for (size_t j = 0; j < rows * blocks; ++j)
        fixture(packed.data() + 110 * j, decoded.data() + 256 * j,
                uint16_t((j % 2 ? 0x8000 : 0) | (0x1800 + j % 1024)), unsigned(j * 13));
    const float ties[] = {1, 1.00390625f, 1.01171875f, 1.005859375f};
    for (size_t col = 0; col < cols; ++col) {
        if (one_hot) input[1 + col * width + (col * 137 + 31) % width] = ties[col % 4];
        else for (size_t i = 0; i < width; ++i)
            input[1 + col * width + i] = i % 7 == 0 ? ties[(col + i) % 4] : float(int((i * 17 + col * 7) % 61) - 30) / 32;
    }
    auto rounded = input;
    for (auto& v : rounded) v = bf16_input(v);
    auto reconstructed = input;
    const auto twin = testq::row_activations(std::vector<float>(input.begin() + 1, input.end() - 1));
    std::copy(twin.begin(), twin.end(), reconstructed.begin() + 1);
    const auto w = b.adopt(packed.data(), packed.size());
    const auto fw = b.adopt(decoded.data(), decoded.size() * sizeof(float));
    const auto x = b.adopt(input.data(), input.size() * sizeof(float));
    const auto bx = b.adopt(rounded.data(), rounded.size() * sizeof(float));
    const auto out = b.alloc(initial.size() * sizeof(float));
    std::vector<uint32_t> ids(cols, 0);
    std::vector<float> gains(cols, 0.5f);
    const auto ib = b.adopt(ids.data(), ids.size() * sizeof(uint32_t));
    const auto gb = b.adopt(gains.data(), gains.size() * sizeof(float));
    const backend::Backend::Routing routing{{ib.get(), 0}, {gb.get(), 0}, 1, 1};
    size_t values = 0;
    for (size_t extent : {size_t(1), size_t(16384)}) {
        const backend::RowRun run{cols, extent};
        for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8}) {
            const bool integer = integer_path(b, dtype, extent);
            const auto& activations = dtype == backend::Dtype::bf16 ? rounded : integer ? reconstructed : input;
            // The small multi-column case exercises shared callers without multiplying the larger shape sweep.
            const int ops = width == 512 && rows == 5 && cols == 3 ? 6 : 1;
            for (int op = 0; op < ops; ++op)
            for (uint32_t type : {uint32_t(21), uint32_t(0)}) {
                const bool additive = op == 3 || op == 5;
                const size_t copies = op == 2 ? 2 : 1;
                const float prior = additive ? 0.25f : 12345.0f;
                std::fill(initial.begin(), initial.end(), prior);
                initial.front() = initial.back() = 12345.0f;
                b.write(*out, 0, initial.data(), initial.size() * sizeof(float));
                paths(b);
                const backend::CSlice weight{type == 21 ? w.get() : fw.get(), 0};
                const backend::CSlice in{type == 0 && dtype == backend::Dtype::bf16 ? bx.get() : x.get(), 1};
                const auto request = type == 0 ? backend::Dtype::f32 : dtype;
                const backend::RowRuns runs{&run, 1};
                switch (op) {
                case 0: b.matmul(type, weight, in, {out.get(), 1}, width, rows, cols, runs, request); break;
                case 1: b.matmul_logits(type, weight, in, {out.get(), 1}, width, rows, cols, runs, request); break;
                case 2: b.matmul_group({{type, weight, {out.get(), 1}, rows}, {type, weight, {out.get(), count + 1}, rows}},
                                       in, width, cols, runs, request); break;
                case 3: b.matmul_add(type, weight, in, {out.get(), 1}, width, rows, cols, runs, request); break;
                case 4: b.matmul_experts({{type, weight, {out.get(), 1}, rows}}, in, width, cols, routing, runs, request); break;
                case 5: b.matmul_experts_add(type, weight, in, {out.get(), 1}, width, rows, cols, routing, runs, request); break;
                }
                const auto& oracle_inputs = type == 21 ? activations : dtype == backend::Dtype::bf16 ? rounded : input;
                const auto got = read(b, *out, initial.size());
                const std::vector<std::string> want_paths{type == 21 && dtype == backend::Dtype::bf16 ? "bf16" : type == 21 && integer ? "block-int16" : "f32"};
                const auto actual_paths = paths(b);
                if (actual_paths != want_paths) {
                    std::cerr << "IQ3_S path type=" << type << " request=" << backend::dtype_name(dtype) << " op=" << op << " extent=" << extent << " actual=";
                    for (const auto& path : actual_paths) std::cerr << path << ' ';
                    std::cerr << '\n';
                }
                require(actual_paths == want_paths, "IQ3_S matrix arithmetic witness differs");
                if (type == 21 && integer) ++integer_calls;
                require(got.front() == initial.front() && got.back() == initial.back(), "matmul crossed output guards");
                for (size_t i = copies * count + 1; i + 1 < got.size(); ++i)
                    require(bits(got[i]) == bits(initial[i]), "matmul wrote beyond output extent");
                for (size_t copy = 0; copy < copies; ++copy)
                for (size_t col = 0; col < cols; ++col)
                    for (size_t row = 0; row < rows; ++row) {
                        const float have = got[1 + copy * count + col * rows + row];
                        const double gain = op == 5 ? 0.5 : 1.0;
                        if (one_hot && !additive && !(type == 21 && integer)) {
                            const size_t selected = (col * 137 + 31) % width;
                            const float want = float(double(decoded[row * width + selected]) * oracle_inputs[1 + col * width + selected]);
                            require(bits(have) == bits(want), "IQ3_S/F32 one-hot matmul is not exact");
                        } else {
                            double want = 0, magnitude = 0;
                            for (size_t i = 0; i < width; ++i) {
                                const double term = double(decoded[row * width + i]) * oracle_inputs[1 + col * width + i];
                                want += term; magnitude += std::fabs(term);
                            }
                            want = gain * want + (additive ? 0.25 : 0);
                            magnitude = gain * magnitude + (additive ? 0.25 : 0);
                            // gamma(2n) accounts for rounded products and summation, including cancellation; n>=256 also covers the final scale and residual addition.
                            const double nu = 2 * double(width) * 0x1p-24;
                            const double bound = nu / (1 - nu) * magnitude;
                            if (!std::isfinite(have) || std::fabs(double(have) - want) > bound)
                                std::cerr << "type=" << type << " dtype=" << backend::dtype_name(dtype) << " op=" << op << " width=" << width << " rows=" << rows
                                          << " cols=" << cols << " extent=" << extent << " row=" << row << " col=" << col << " got=" << have << " expected=" << want << " bound=" << bound << '\n';
                            require(std::isfinite(have) && std::fabs(double(have) - want) <= bound,
                                    "IQ3_S/F32 matmul exceeds double-oracle rounding bound");
                            if (type == 21 && integer) {
                                double original = 0, rounding = 0;
                                for (size_t i = 0; i < width; ++i) {
                                    const double coefficient = decoded[row * width + i];
                                    original += coefficient * input[1 + col * width + i];
                                    rounding += std::fabs(coefficient * (double(input[1 + col * width + i]) - oracle_inputs[1 + col * width + i]));
                                }
                                original = gain * original + (additive ? 0.25 : 0);
                                require(std::fabs(double(have) - original) <= bound + gain * rounding,
                                        "IQ3_S original-input error exceeds reconstruction plus accumulation bounds");
                            }
                        }
                        ++values;
                    }
            }
        }
    }
    return values;
}
size_t routed_products(backend::Backend& b) {
    constexpr size_t width = 512, rows = 7, experts = 3, picks = 2;
    std::vector<uint8_t> packed(experts * rows * 2 * 110);
    std::vector<float> weights(experts * rows * width);
    for (size_t j = 0; j < experts * rows * 2; ++j)
        fixture(packed.data() + 110 * j, weights.data() + 256 * j, uint16_t(0x2400 + j * 7), unsigned(j * 11));
    const auto w = b.adopt(packed.data(), packed.size());
    size_t values = 0;
    for (size_t cols : {size_t(1), size_t(3), size_t(33)}) {
        const size_t entries = cols * picks;
        std::vector<float> input(entries * width), gains(entries);
        std::vector<uint32_t> ids(entries);
        for (size_t e = 0; e < entries; ++e) {
            ids[e] = uint32_t((e + e / picks) % experts);
            gains[e] = e % picks ? 0.75f : 0.25f;
            for (size_t i = 0; i < width; ++i)
                input[e * width + i] = float(int((e * 13 + i * 7) % 71) - 35) / 32.0f;
        }
        const auto twin = testq::row_activations(input);
        auto rounded = input;
        for (float& v : rounded) v = bf16_input(v);
        const auto x = b.adopt(input.data(), input.size() * sizeof(float));
        const auto ib = b.adopt(ids.data(), ids.size() * sizeof(uint32_t));
        const auto gb = b.adopt(gains.data(), gains.size() * sizeof(float));
        const backend::Backend::Routing routing{{ib.get(), 0}, {gb.get(), 0}, picks, experts};
        for (size_t extent : {size_t(1), size_t(16384)})
        for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8})
        for (bool additive : {false, true}) {
            const bool integer = integer_path(b, dtype, extent);
            const auto& activations = dtype == backend::Dtype::bf16 ? rounded : integer ? twin : input;
            const size_t count = (additive ? cols : entries) * rows;
            std::vector<float> initial(count + 2, additive ? 0.25f : 0.0f);
            initial.front() = initial.back() = 12345.0f;
            const auto out = b.adopt(initial.data(), initial.size() * sizeof(float));
            const backend::RowRun run{cols, extent};
            paths(b);
            if (additive) b.matmul_experts_add(21, {w.get(), 0}, {x.get(), 0}, {out.get(), 1}, width, rows, cols, routing, {&run, 1}, dtype);
            else b.matmul_experts({{21, {w.get(), 0}, {out.get(), 1}, rows}}, {x.get(), 0}, width, cols, routing, {&run, 1}, dtype);
            const auto got = read(b, *out, initial.size());
            require(paths(b) == std::vector<std::string>{dtype == backend::Dtype::bf16 ? "bf16" : integer ? "block-int16" : "f32"},
                    "IQ3_S multi-expert matrix witness differs");
            if (integer) ++integer_calls;
            require(got.front() == initial.front() && got.back() == initial.back(), "IQ3_S routed output guards");
            for (size_t c = 0; c < (additive ? cols : entries); ++c)
            for (size_t r = 0; r < rows; ++r) {
                double want = additive ? 0.25 : 0, magnitude = additive ? 0.25 : 0;
                for (size_t pick = 0; pick < (additive ? picks : 1); ++pick) {
                    const size_t entry = additive ? c * picks + pick : c;
                    const size_t col = additive ? entry : entry / picks;
                    const double gain = additive ? gains[entry] : 1.0;
                    for (size_t i = 0; i < width; ++i) {
                        const double term = gain * double(weights[(ids[entry] * rows + r) * width + i]) * activations[col * width + i];
                        want += term; magnitude += std::fabs(term);
                    }
                }
                const double nu = double(2 * picks * width) * 0x1p-24;
                require(std::isfinite(got[1 + c * rows + r]) && std::fabs(double(got[1 + c * rows + r]) - want) <= nu / (1 - nu) * magnitude,
                        "IQ3_S multiple-expert products exceed reconstructed-input double bound");
                ++values;
            }
        }
    }
    return values;
}
size_t precision_products(backend::Backend& b) {
    std::vector<uint8_t> packed(110, 0);
    packed[1] = 0x3c; // Unit weights from dictionary entry zero, no sign or subscale bits.
    const auto w = b.adopt(packed.data(), packed.size());
    size_t values = 0;
    for (float sign : {-1.0f, 1.0f}) {
        std::vector<float> input(256, 0);
        input[0] = 2.0f; input[1] = -2.0f; input[2] = sign / 1024.0f;
        const auto reconstructed = testq::row_activations(input);
        require(reconstructed[2] == sign * (16.0f * (2.0f / 32767.0f)), "IQ3_S independent twin precision anchor");
        const auto x = b.adopt(input.data(), input.size() * sizeof(float));
        const auto out = b.alloc(sizeof(float));
        for (size_t extent : {size_t(1), size_t(16384)})
        for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8}) {
            const bool integer = integer_path(b, dtype, extent);
            const backend::RowRun run{1, extent};
            paths(b);
            b.matmul(21, {w.get(), 0}, {x.get(), 0}, {out.get(), 0}, 256, 1, 1, {&run, 1}, dtype);
            const auto got = read(b, *out, 1);
            require(paths(b) == std::vector<std::string>{dtype == backend::Dtype::bf16 ? "bf16" : integer ? "block-int16" : "f32"},
                    "IQ3_S precision path witness");
            if (integer) ++integer_calls;
            const float want = integer ? reconstructed[2] : input[2];
            require(std::fabs(double(got[0]) - want) <= std::fabs(double(want)) * 0x1p-20,
                    "IQ3_S small value beside cancelling peaks loses int16 precision");
            ++values;
        }
    }
    return values;
}
size_t range_products(backend::Backend& b) {
    constexpr size_t width = 256;
    const uint16_t halves[] = {0, 0x8000, 1, 0x8001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff};
    const float inputs[] = {0.0f, 0x1p-24f, 0x1p-14f, 1.0f, 65504.0f, 0x1p-120f, 0x1p90f};
    size_t values = 0;
    for (uint16_t h : halves) {
        std::vector<uint8_t> packed(110);
        std::vector<float> weights(width);
        fixture(packed.data(), weights.data(), h, 31);
        const auto w = b.adopt(packed.data(), packed.size());
        for (float value : inputs)
        for (bool negative : {false, true}) {
            std::vector<float> input(width, 0.0f);
            const size_t lane = 123;
            input[lane] = negative ? -value : value;
            const auto x = b.adopt(input.data(), input.size() * sizeof(float));
            const float initial[] = {19, 0, 19};
            const auto out = b.adopt(initial, sizeof initial);
            for (size_t extent : {size_t(1), size_t(16384)})
            for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::int8}) {
                const bool integer = integer_path(b, dtype, extent);
                const backend::RowRun run{1, extent};
                paths(b);
                b.matmul(21, {w.get(), 0}, {x.get(), 0}, {out.get(), 1}, width, 1, 1, {&run, 1}, dtype);
                const auto got = read(b, *out, 3);
                require(paths(b) == std::vector<std::string>{integer ? "block-int16" : "f32"}, "IQ3_S range witness");
                if (integer) ++integer_calls;
                const double want = double(weights[lane]) * input[lane];
                // One-hot scale rounding: relative F32 rounding plus at most half a subnormal scale unit per integer level.
                const double reconstruction = integer ? std::fabs(double(value)) * 0x1p-22 + 32767.0 * std::numeric_limits<float>::denorm_min() : 0;
                const double bound = std::fabs(double(weights[lane])) * reconstruction + std::fabs(want) * 0x1p-21 + 2.0 * std::numeric_limits<float>::denorm_min();
                require(got.front() == 19 && got.back() == 19 && std::isfinite(got[1]) && std::fabs(double(got[1]) - want) <= bound,
                        "IQ3_S finite-half and scale-repair range control");
                ++values;
            }
        }
    }
    return values;
}
size_t batch_controls(backend::Backend& b) {
    constexpr size_t width = 4096, rows = 11, nout = 17;
    std::vector<uint8_t> packed(nout * (width / 256) * 110);
    std::vector<float> decoded(nout * width);
    for (size_t j = 0; j < nout * (width / 256); ++j)
        fixture(packed.data() + 110 * j, decoded.data() + 256 * j, uint16_t(0x1800 + j % 1024), unsigned(j * 19));
    const auto input = row_classes::values(rows * width, 133);
    const auto w = b.adopt(packed.data(), packed.size()), x = b.adopt(input.data(), input.size() * sizeof(float));
    std::vector<float> initial(rows * nout + 2, 12345.0f);
    const auto out = b.adopt(initial.data(), initial.size() * sizeof(float));
    const std::vector<uint32_t> ids(rows, 0);
    const std::vector<float> gains(rows, 1.0f);
    const auto ib = b.adopt(ids.data(), ids.size() * sizeof(uint32_t)), gb = b.adopt(gains.data(), gains.size() * sizeof(float));
    const backend::Backend::Routing routing{{ib.get(), 0}, {gb.get(), 0}, 1, 1};
    size_t crossover = 64;
#if LLMX_HAS_BACKEND_VULKAN
    if (!b.is_cpu()) crossover = backend::tile_from_for(backend::vulkan_device_profile(b), false, width);
#endif
    size_t values = 0;
    auto call = [&](size_t first, size_t count, backend::RowRuns runs, backend::Dtype dtype) {
        b.matmul(21, {w.get(), 0}, {x.get(), first * width}, {out.get(), 1 + first * nout}, width, nout, count, runs, dtype);
    };
    auto routed_call = [&](size_t extent, backend::Dtype dtype) {
        const backend::RowRun run{rows, extent};
        b.matmul_experts({{21, {w.get(), 0}, {out.get(), 1}, nout}}, {x.get(), 0}, width, rows, routing, {&run, 1}, dtype);
    };
    auto finish = [&] {
        const auto got = read(b, *out, initial.size());
        require(got.front() == initial.front() && got.back() == initial.back(), "IQ3_S batch-control output guards");
        for (size_t i = 1; i + 1 < got.size(); ++i) require(std::isfinite(got[i]), "IQ3_S batch-control nonfinite result");
        values += rows * nout;
        return got;
    };
    for (auto dtype : {backend::Dtype::f32, backend::Dtype::f16, backend::Dtype::bf16, backend::Dtype::int8}) {
        const backend::RowRun mixed[] = {{2, 1}, {rows, crossover + 1}};
        paths(b); call(0, rows, {mixed, 2}, dtype);
        const auto together = finish();
        const auto together_paths = paths(b);
        const backend::RowRun decode{2, 1}, prompt{rows - 2, crossover + 1};
        paths(b); call(0, 2, {&decode, 1}, dtype); call(2, rows - 2, {&prompt, 1}, dtype);
        const auto apart = finish();
        const auto apart_paths = paths(b);
        require(std::memcmp(together.data(), apart.data(), together.size() * sizeof(float)) == 0 && together_paths == apart_paths, "IQ3_S mixed RowRuns changed bytes or paths when separated");
        const backend::RowRun first_prompt{3, crossover + 1}, tail_prompt{rows - 5, crossover + 1};
        paths(b); call(0, 2, {&decode, 1}, dtype); call(2, 3, {&first_prompt, 1}, dtype); call(5, rows - 5, {&tail_prompt, 1}, dtype);
        const auto sliced = finish();
        const auto sliced_paths = paths(b);
        require(std::memcmp(together.data(), sliced.data(), together.size() * sizeof(float)) == 0 && together_paths == sliced_paths, "IQ3_S prompt slices changed bytes or paths beside decode rows");
    }
    // One shape through the shared extent/class checker, retaining its tile-split boundaries.
    std::vector<std::vector<float>> class_outputs, routed_outputs;
    for (size_t extent : row_classes::kExtents) {
        const backend::RowRun run{rows, extent};
        paths(b); call(0, rows, {&run, 1}, backend::Dtype::f16);
        class_outputs.push_back(finish()); paths(b);
        routed_call(extent, backend::Dtype::f16);
        routed_outputs.push_back(finish()); paths(b);
    }
    values += row_classes::same_within_classes(b, class_outputs, "IQ3_S F16 row-class identity");
    values += row_classes::same_within_classes(b, routed_outputs, "IQ3_S F16 routed row-class identity");
#if LLMX_HAS_BACKEND_VULKAN
    if (!b.is_cpu()) {
        const auto profile = backend::vulkan_device_profile(b);
        for (bool routed : {false, true}) {
            const size_t edge = routed ? profile.moe_tile_from_q4 : crossover;
            for (size_t extent : {edge - 1, edge, edge + 1})
            for (auto dtype : {backend::Dtype::f16, backend::Dtype::int8}) {
                const backend::RowRun run{rows, extent};
                paths(b);
                if (routed) routed_call(extent, dtype);
                else call(0, rows, {&run, 1}, dtype);
                finish();
                const bool integer = extent < edge ? profile.lookup_integer_dot : profile.prefer_integer_dot;
                require(paths(b) == std::vector<std::string>{integer ? "block-int16" : "f32"}, "IQ3_S crossover arithmetic witness");
                if (integer) ++integer_calls;
            }
        }
    }
#endif
    return values;
}
size_t tall_tile_control(backend::Backend& b) {
    if (b.is_cpu()) return 0;
    // Nine column tiles give 65 * 9 workgroups, passing MI50's eight-per-CU narrow-row threshold.
    constexpr size_t width = 256, nout = 8193, cols = 513;
    std::vector<uint8_t> packed(nout * 110);
    std::vector<float> decoded(nout * width), input(cols * width, 0);
    for (size_t r = 0; r < nout; ++r)
        fixture(packed.data() + r * 110, decoded.data() + r * width, uint16_t(0x1800 + r % 1024), unsigned(r * 13));
    for (size_t c = 0; c < cols; ++c) input[c * width + (c * 137 + 31) % width] = 1.0f;
    const auto w = b.adopt(packed.data(), packed.size()), x = b.adopt(input.data(), input.size() * sizeof(float));
    std::vector<float> initial(nout * cols + 2, 12345.0f);
    const auto out = b.adopt(initial.data(), initial.size() * sizeof(float));
    const backend::RowRun run{cols, 16384};
    paths(b); b.matmul(21, {w.get(), 0}, {x.get(), 0}, {out.get(), 1}, width, nout, cols, {&run, 1}, backend::Dtype::f16);
    const auto got = read(b, *out, initial.size());
    const bool integer = integer_path(b, backend::Dtype::f16, 16384);
    require(paths(b) == std::vector<std::string>{integer ? "block-int16" : "f32"}, "IQ3_S tall tile arithmetic witness");
    if (integer) ++integer_calls;
    require(got.front() == initial.front() && got.back() == initial.back(), "IQ3_S tall tile output guards");
    for (size_t c = 0; c < cols; ++c)
    for (size_t r = 0; r < nout; ++r) {
        const double want = decoded[r * width + (c * 137 + 31) % width];
        require(std::isfinite(got[1 + c * nout + r]) && std::fabs(double(got[1 + c * nout + r]) - want) <= std::fabs(want) * 0x1p-20,
                "IQ3_S tall tile one-hot/tail control");
    }
    return nout * cols;
}
void backend_checks(backend::Backend& b) {
    require(b.supports_type(21), "backend does not expose IQ3_S");
    require(bf16_input(1.00390625f) == 1.0f && bf16_input(1.01171875f) == 1.015625f && bf16_input(1.005859375f) == 1.0078125f, "independent BF16 tie anchors");
    const size_t embedded = embeddings(b);
    size_t multiplied = 0;
    for (bool one_hot : {false, true}) {
        multiplied += products(b, 256, 1, 1, one_hot);
        multiplied += products(b, 512, 5, 3, one_hot);
        multiplied += products(b, 4096, 17, 17, one_hot);
        multiplied += products(b, 512, 65, 33, one_hot);
    }
    multiplied += products(b, 256, 512, 1, false);
    multiplied += routed_products(b);
    multiplied += range_products(b);
    multiplied += precision_products(b);
    multiplied += batch_controls(b);
    multiplied += tall_tile_control(b);
    // All 256 one-hot component positions in a prompt-shaped call.
    multiplied += products(b, 256, 5, 256, true);
    std::cout << "IQ3_S backend: " << embedded << " exact embedding values, " << multiplied
              << " product/control values over F32/F16/BF16/int8 requests (F32/BF16 one-hot exact, reconstructed-input gamma(2n), original-input rounding, path witnesses) [ok]\n";
}

}
int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--vulkan") {
#if LLMX_HAS_BACKEND_VULKAN
        try {
            auto device = backend::make_vulkan_backend(0, true);
            backend_checks(*device);
            const auto profile = backend::vulkan_device_profile(*device);
            bool row = false, tile = false, tall = false;
            for (const auto& entry : backend::vulkan_kernel_times(*device)) {
                std::cout << "IQ3_S dispatch: " << entry.first << '\n';
                row |= entry.first.find("matmul_row_iq3_s_dot") != std::string::npos;
                tile |= entry.first.find("matmul_tile_iq3_s_dot") != std::string::npos;
                tall |= entry.first.find("matmul_tile_iq3_s_dot_tall") != std::string::npos;
            }
            if (backend::vulkan_timed_dispatches(*device)) {
                require(row == profile.lookup_integer_dot, "IQ3_S integer-row dispatch differs from profile");
                require(tile == profile.prefer_integer_dot, "IQ3_S integer-tile dispatch differs from profile");
                require(tall == profile.prefer_integer_dot, "IQ3_S tall integer-tile dispatch was not witnessed");
                require((row || tile) == (integer_calls != 0), "IQ3_S dispatched integer modules lack block-int16 witnesses");
            }
            return 0;
        } catch (const backend::VulkanUnavailable& e) {
            std::cout << "IQ3_S Vulkan unavailable: " << e.what() << '\n'; return 77;
        } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
#else
        std::cout << "IQ3_S Vulkan backend not built\n"; return 77;
#endif
    }
    if (argc != 1) { std::cerr << "usage: llmx-iq3-s-test [--vulkan]\n"; return 1; }
    try {
        const auto* qt = quant::Registry::instance().get(21);
        require(qt && qt->block_size == 256 && qt->type_size == 110 &&
                qt->dequantize == quant::dequantize_row_iq3_s && !qt->quantize, "read-only registry");
        require(quant::row_bytes(21, 768, 2) == 660, "row bytes");
        bool refused = false;
        try { (void)quant::row_bytes(21, 257); } catch (const std::runtime_error&) { refused = true; }
        require(refused, "partial block accepted");
        quant::dequantize_row_iq3_s(nullptr, nullptr, 0);
        require(magnitude(0, 0) == 1 && magnitude(1, 0) == 3 && magnitude(1, 3) == 1, "hand dictionary entries");
        Indices indices{}; Scales scales{}; Signs signs{};
        // Each of the 512 vectors appears at every vector position and lane.
        // All 16 scale nibbles appear in every sub-block in both packed nibbles.
        for (unsigned shift = 0; shift < 512; ++shift) {
            for (unsigned g = 0; g < 64; ++g) indices[g] = (shift + g) % 512;
            for (unsigned g = 0; g < 8; ++g) scales[g] = (shift + g) % 16;
            for (unsigned i = 0; i < 256; ++i) signs[i] = ((i + shift) % 3) == 0;
            check(0x3c00, indices, scales, signs);
        }
        indices.fill(0); scales.fill(0); signs.fill(false);
        for (unsigned g = 0; g < 64; ++g) {
            indices[g] = 256; check(0x3c00, indices, scales, signs); indices[g] = 0;
        }
        for (unsigned i = 0; i < 256; ++i) {
            signs[i] = true; check(0x3c00, indices, scales, signs); signs[i] = false;
        }
        for (unsigned g = 0; g < 8; ++g)
            for (unsigned nibble = 0; nibble < 16; ++nibble) {
                scales[g] = nibble; check(0x3c00, indices, scales, signs); scales[g] = 0;
            }
        // Every finite binary16 scale, both zero signs included, against mixed magnitudes, signs and subscales, with an independent double oracle.
        for (unsigned h = 0; h <= 65535; ++h) {
            if ((h & 0x7c00) == 0x7c00) continue;
            for (unsigned g = 0; g < 64; ++g) indices[g] = (h + 37 * g) % 512;
            for (unsigned g = 0; g < 8; ++g) scales[g] = (h + g) % 16;
            for (unsigned i = 0; i < 256; ++i) signs[i] = (i % 2) != 0;
            check(uint16_t(h), indices, scales, signs);
        }
        std::cout << "IQ3_S: " << checked << " exact values, dictionary/fields/finite halves/guards [ok]\n";
        backend::CpuBackend cpu;
        cpu.set_threads(1);
        backend_checks(cpu);
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
