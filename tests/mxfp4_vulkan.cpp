// MXFP4 through the public Vulkan backend: embedding, row and tile products, split partials, accumulation, mixed projections and routed expert products.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include "backends/vulkan/vulkan_backend.hpp"
#include "matrix_precision.hpp"


namespace {
constexpr uint32_t type = 39;
constexpr size_t guard = 8;
size_t checked = 0, cases = 0;
backend::Dtype policy = backend::Dtype::f32;
float rounded(float value) {
    if (policy != backend::Dtype::bf16) return value;
    uint32_t n; std::memcpy(&n, &value, 4);
    const uint32_t lower = n & 65535, upper = n >> 16;
    n = (upper + (lower > 32768 || (lower == 32768 && (upper & 1)))) << 16;
    std::memcpy(&value, &n, 4); return value;
}
std::set<std::string> dispatched, matrix_paths;
bool record_dispatches(backend::Backend& b) {
    for (const auto& item : backend::vulkan_kernel_times(b)) dispatched.insert(item.first);
    const auto paths = testq::take_matrix_paths(b);
    matrix_paths.insert(paths.begin(), paths.end());
    return std::find(paths.begin(), paths.end(), "block-int16") != paths.end();
}
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
uint32_t bits(float x) { uint32_t n; std::memcpy(&n, &x, 4); return n; }
// Values defined by the fixture's raw E8M0 and E2M1 fields, without llmx's decoder.
float weight(const std::vector<uint8_t>& w, size_t row, size_t width, size_t j) {
    const double table[] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    const size_t at = (row * (width / 32) + j / 32) * 17;
    const unsigned byte = w[at + 1 + j % 16], code = j % 32 < 16 ? byte & 15 : byte >> 4;
    if (!(code & 7)) return 0;
    const double value = std::ldexp(table[code & 7], int(w[at]) - 127);
    const float f = value > std::numeric_limits<float>::max() ? std::numeric_limits<float>::infinity() : float(value);
    return code & 8 ? -f : f;
}
backend::CSlice input(const backend::BufferPtr& p) { return {p.get(), 0}; }
void run(backend::Backend& b, size_t width, size_t rows, size_t cols, bool decode, bool add, int mode, bool routed) {
    const size_t experts = routed ? 3 : 1, count = cols * (routed ? 2 : 1);
    std::vector<uint8_t> w(rows * experts * (width / 32) * 17);
    for (size_t k = 0; k < w.size() / 17; ++k) {
        w[17 * k] = 127;
        for (size_t j = 0; j < 16; ++j) w[17 * k + 1 + j] = uint8_t(k * 19 + j * 37);
    }
    std::vector<float> x((routed && add ? count : cols) * width);
    for (size_t i = 0; i < x.size(); ++i) x[i] = i % 3 == 0 ? -1.006111f : .500971f;
    if (mode == 1 || mode == 2) {
        std::fill(w.begin(), w.end(), uint8_t(0));
        std::fill(x.begin(), x.end(), mode == 1 ? 2.0f : std::ldexp(1.0f, width == 544 && policy != backend::Dtype::f16 ? -26 : -24));
        for (size_t row = 0; row < rows * experts; ++row)
            for (size_t k = 0; k < width / 32; ++k) {
                const size_t at = (row * (width / 32) + k) * 17;
                if (mode == 1) {
                    w[at] = 252;
                    const size_t half = width == 544 ? 8 : 1;
                    if (k < 2 * half) std::fill(w.begin() + at + 1, w.begin() + at + 17, uint8_t(k < half ? 0x66 : 0xee));
                } else {
                    w[at] = 0;
                    w[at + 1] = 1; // One tiny product per block. At width 544, three split parts must round together.
                }
            }
    }
    if (mode == 3) {
        std::fill(w.begin(), w.end(), uint8_t(0));
        for (size_t row = 0; row < rows * experts; ++row)
            for (size_t k = 0; k < width / 32; ++k) {
                const size_t at = (row * (width / 32) + k) * 17;
                const bool active = k % 3 == row % 3;
                w[at] = active ? (row % 3 == 0 ? 0 : row % 3 == 1 ? 103 : 127) : 127;
                if (active) std::fill(w.begin() + at + 1, w.begin() + at + 17, uint8_t(row % 3 == 0 ? 0x11 : 0x22));
            }
        // Tiny weight rows and tiny activation columns must each retain their earlier blocks' fallback flag.
        for (size_t i = 0; i < x.size(); ++i) {
            const size_t col = i / width, block = (i % width) / 32;
            x[i] = (block + col) % 3 == 2 ? 0.0f : std::ldexp(1.0f, col % 3 == 0 ? -24 : col % 3 == 1 ? (policy == backend::Dtype::f16 ? -24 : -128) : 0);
        }
    }
    const size_t outputs = routed && add ? cols : count;
    std::vector<float> y(guard + outputs * rows + guard, 71.0f);
    std::fill(y.begin() + guard, y.end() - guard, add ? .25f : 0.0f);
    auto wb = b.adopt(w.data(), w.size()), xb = b.adopt(x.data(), x.size() * sizeof(float));
    auto yb = b.adopt(y.data(), y.size() * sizeof(float));
    const backend::Slice out{yb.get(), guard};
    const backend::RowRun run{cols, decode ? 1 : cols};
    std::vector<uint32_t> ids(count);
    for (size_t i = 0; i < count; ++i) ids[i] = uint32_t((i / 2 + i % 2) % experts);
    if (routed) {
        std::vector<float> probabilities(count, .5f);
        auto ib = b.adopt(ids.data(), ids.size() * sizeof(uint32_t)), pb = b.adopt(probabilities.data(), probabilities.size() * sizeof(float));
        if (add) b.matmul_experts_add(type, input(wb), input(xb), out, width, rows, cols, {input(ib), input(pb), 2, experts}, {&run, 1}, policy);
        else b.matmul_experts({{type, input(wb), out, rows}}, input(xb), width, cols, {input(ib), input(pb), 2, experts}, {&run, 1}, policy);
    } else if (add) b.matmul_add(type, input(wb), input(xb), out, width, rows, cols, {&run, 1}, policy);
    else b.matmul(type, input(wb), input(xb), out, width, rows, cols, {&run, 1}, policy);
    b.read(*yb, 0, y.data(), y.size() * sizeof(float));
    const bool integer = record_dispatches(b);
    for (size_t i = 0; i < guard; ++i) require(y[i] == 71 && y[y.size() - 1 - i] == 71, "output guard overwritten");
    for (size_t c = 0; c < outputs; ++c)
        for (size_t row = 0; row < rows; ++row) {
            double magnitude = 0, input_error = 0;
            float expected = 0;
            for (size_t slot = 0; slot < (routed && add ? 2u : 1u); ++slot) {
                const size_t index = routed && add ? c * 2 + slot : c;
                const size_t xrow = routed && !add ? c / 2 : index;
                double sum = 0;
                for (size_t j = 0; j < width; ++j) {
                    const double product = double(weight(w, (routed ? ids[index] * rows : 0) + row, width, j)) * rounded(x[xrow * width + j]);
                    sum += product; magnitude += std::abs(product);
                    if (integer) {
                        float peak = 0;
                        for (size_t k = j / 32 * 32; k < j / 32 * 32 + 32; ++k) peak = std::max(peak, std::abs(x[xrow * width + k]));
                        input_error += std::abs(double(weight(w, (routed ? ids[index] * rows : 0) + row, width, j))) * peak / (2 * 32767);
                    }
                }
                expected += float(sum) * (routed && add ? .5f : 1.0f);
            }
            const float want = expected + (add ? .25f : 0);
            const float got = y[guard + c * rows + row];
            const bool ok = mode ? bits(want) == bits(got) : std::isfinite(got) && std::abs(double(got) - want) <= input_error + 2e-6 * magnitude + 1e-7;
            if (!ok) {
                std::cerr << "width " << width << " rows " << rows << " cols " << cols << " decode " << decode << " add " << add
                          << " mode " << mode << " routed " << routed << " output " << c << ',' << row << " want " << want << " got " << got << '\n';
                throw std::runtime_error("MXFP4 product differs from fixture");
            }
            ++checked;
        }
    ++cases;
}
// Opposite ends of a long dot lie in different split parts.
// Each part overflows or rounds to zero in F32, while their combined result is representable.
void split_range(backend::Backend& b) {
    constexpr size_t width = 4096, columns = 65;
    const auto& profile = backend::vulkan_device_profile(b);
    for (bool tiny : {false, true}) {
        std::vector<uint8_t> w(width / 32 * 17, 0);
        if (tiny) {
            w[1] = w[w.size() - 16] = 4; // Two products of 2^-150.
        } else {
            w[0] = w[w.size() - 17] = 252;
            w[1] = 6; w[w.size() - 16] = 14; // +2^127 and -2^127.
        }
        std::vector<float> x(width * columns, tiny ? std::ldexp(1.0f, -24) : 2.0f);
        std::vector<float> y(columns);
        auto wb = b.adopt(w.data(), w.size()), xb = b.adopt(x.data(), x.size() * 4), yb = b.alloc(y.size() * 4);
        record_dispatches(b);
        const backend::RowRun run{columns, columns};
        b.matmul(type, input(wb), input(xb), {yb.get(), 0}, width, 1, columns, {&run, 1}, policy);
        b.read(*yb, 0, y.data(), y.size() * 4);
        bool tile = false, reduce = false;
        for (const auto& item : backend::vulkan_kernel_times(b)) {
            dispatched.insert(item.first);
            tile |= item.first == "matmul_tile_q8mx_small" || item.first == "matmul_tile_q8mx";
            reduce |= item.first == "matmul_reduce_mxfp4";
        }
        const bool integer = record_dispatches(b);
        if (policy == backend::Dtype::f16 && profile.prefer_integer_dot)
            require(tile && reduce && integer, "MXFP4 range fixture missed the integer split path");
        const float expected = tiny ? std::numeric_limits<float>::denorm_min() : 0.0f;
        for (float value : y) {
            if (bits(value) != bits(expected)) {
                std::cerr << "split range tiny " << tiny << " policy " << int(policy)
                          << " want " << expected << " got " << value << '\n';
                throw std::runtime_error("MXFP4 split range differs from fixture");
            }
            ++checked;
        }
        ++cases;
    }
}
// Same-type projections exercise one call with uneven split partials; the F32 projection beside them exercises the partition between type families.
void grouped(backend::Backend& b, size_t cols, bool decode) {
    constexpr size_t width = 160, rows[] = {3, 67, 5};
    std::vector<uint8_t> w((rows[0] + rows[1]) * (width / 32) * 17, 0x22);
    for (size_t i = 0; i < w.size(); i += 17) w[i] = 127;
    std::vector<float> fw(rows[2] * width, .5f), x(cols * width, 1.0f);
    const size_t first_bytes = rows[0] * (width / 32) * 17;
    auto wb = b.adopt(w.data(), first_bytes), wb2 = b.adopt(w.data() + first_bytes, w.size() - first_bytes);
    auto fb = b.adopt(fw.data(), fw.size() * sizeof(float));
    auto xb = b.adopt(x.data(), x.size() * sizeof(float));
    backend::BufferPtr yb[3];
    for (size_t i = 0; i < 3; ++i) {
        std::vector<float> out(guard * 2 + rows[i] * cols, 71);
        yb[i] = b.adopt(out.data(), out.size() * sizeof(float));
    }
    const backend::RowRun run{cols, decode ? 1 : cols};
    b.matmul_group({{type, input(wb), {yb[0].get(), guard}, rows[0]},
                    {type, input(wb2), {yb[1].get(), guard}, rows[1]},
                    {0, input(fb), {yb[2].get(), guard}, rows[2]}}, input(xb), width, cols, {&run, 1}, policy);
    for (size_t i = 0; i < 3; ++i) {
        std::vector<float> out(guard * 2 + rows[i] * cols);
        b.read(*yb[i], 0, out.data(), out.size() * sizeof(float));
        for (size_t j = 0; j < guard; ++j) require(out[j] == 71 && out[out.size() - 1 - j] == 71, "group output guard overwritten");
        for (size_t j = guard; j < out.size() - guard; ++j) {
            require(std::isfinite(out[j]) && std::abs(out[j] - (i == 2 ? 80.0f : 160.0f)) < .00032f, "mixed group product differs from fields");
            ++checked;
        }
    }
    record_dispatches(b);
    ++cases;
}
// Mixed prompt and decode rows must be independent of submission boundaries, with nontrivial dot sums.
void batch_invariance(backend::Backend& b, size_t width, size_t extent) {
    constexpr size_t rows = 67, decoding = 3;
    const size_t columns = extent + decoding;
    std::vector<uint8_t> w(rows * (width / 32) * 17);
    for (size_t k = 0; k < w.size() / 17; ++k) {
        w[k * 17] = uint8_t(121 + k % 5);
        for (size_t j = 0; j < 16; ++j) w[k * 17 + j + 1] = uint8_t(k * 31 + j * 17);
    }
    std::vector<float> x(columns * width);
    for (size_t i = 0; i < x.size(); ++i) x[i] = float(int((i * 37) % 1009) - 504) / 511;
    auto wb = b.adopt(w.data(), w.size()), xb = b.adopt(x.data(), x.size() * 4);
    for (int operation = 0; operation < 4; ++operation) {
        std::vector<float> reference;
        for (size_t chunk : {columns, size_t(1), size_t(3)}) {
            std::vector<float> y(columns * rows, .25f);
            auto yb = b.adopt(y.data(), y.size() * 4);
            for (size_t first = 0; first < columns;) {
                const size_t count = std::min(chunk, columns - first);
                backend::RowRun runs[2]; size_t nr = 0;
                if (first < decoding) runs[nr++] = {std::min(count, decoding - first), 1};
                if (first + count > decoding) runs[nr++] = {count, extent};
                const backend::CSlice in{xb.get(), first * width};
                const backend::Slice out{yb.get(), first * rows};
                if (operation == 0) b.matmul(type, input(wb), in, out, width, rows, count, {runs, nr}, policy);
                else if (operation == 1) b.matmul_logits(type, input(wb), in, out, width, rows, count, {runs, nr}, policy);
                else if (operation == 2) b.matmul_add(type, input(wb), in, out, width, rows, count, {runs, nr}, policy);
                else b.matmul_group({{type, input(wb), out, rows}}, in, width, count, {runs, nr}, policy);
                first += count;
            }
            b.read(*yb, 0, y.data(), y.size() * 4);
            if (reference.empty()) reference = y;
            else if (std::memcmp(y.data(), reference.data(), y.size() * 4) != 0) {
                std::cerr << "batch width " << width << " extent " << extent << " chunk " << chunk << " operation " << operation << '\n';
                throw std::runtime_error("MXFP4 mixed rows changed with microbatch boundaries");
            }
            checked += y.size(); ++cases; record_dispatches(b);
        }
    }
}
// One selected finite weight at each scale boundary, on both sides of zero and with inputs that separate F32 and BF16.
void scale_products(backend::Backend& b) {
    constexpr size_t width = 32, rows = 20, columns = 65;
    const uint8_t scales[] = {0, 1, 103, 104, 143, 144, 145, 146, 252, 255};
    std::vector<uint8_t> w(rows * 17, 0);
    for (size_t row = 0; row < rows; ++row) {
        w[row * 17] = scales[row / 2];
        w[row * 17 + 1] = row % 2 ? 9 : 1;
    }
    std::vector<float> x(columns * width, 0), y(rows * columns);
    for (size_t c = 0; c < columns; ++c) x[c * width] = c % 3 == 0 ? -1.006111f : c % 3 == 1 ? 1.0f : std::ldexp(1.0f, -24);
    auto wb = b.adopt(w.data(), w.size()), xb = b.adopt(x.data(), x.size() * 4), yb = b.alloc(y.size() * 4);
    const backend::RowRun run{columns, columns};
    b.matmul_logits(type, input(wb), input(xb), {yb.get(), 0}, width, rows, columns, {&run, 1}, policy);
    b.read(*yb, 0, y.data(), y.size() * 4);
    for (size_t c = 0; c < columns; ++c) for (size_t row = 0; row < rows; ++row) {
        const float expected = float(double(weight(w, row, width, 0)) * double(rounded(x[c * width])));
        require(y[c * rows + row] == expected, "scale boundary product differs from independent rounded-input reference");
        ++checked;
    }
    record_dispatches(b); ++cases;
}
void embedding(backend::Backend& b) {
    std::vector<uint8_t> w(256 * 17);
    for (unsigned e = 0; e < 256; ++e) {
        w[e * 17] = uint8_t(e);
        for (unsigned j = 0; j < 16; ++j) w[e * 17 + 1 + j] = uint8_t(j | (15 - j) << 4);
    }
    std::vector<uint32_t> ids(256);
    for (unsigned i = 0; i < 256; ++i) ids[i] = 255 - i;
    std::vector<float> out(ids.size() * 32);
    auto wb = b.adopt(w.data(), w.size()), yb = b.alloc(out.size() * sizeof(float));
    b.embed({yb.get(), 0}, type, input(wb), 32, 256, ids.data(), ids.size());
    b.read(*yb, 0, out.data(), out.size() * sizeof(float));
    for (size_t i = 0; i < ids.size(); ++i)
        for (size_t j = 0; j < 32; ++j) require(bits(out[i * 32 + j]) == bits(weight(w, ids[i], 32, j)), "embedding differs from fields");
    checked += out.size(); record_dispatches(b); ++cases;
}
}
int main() {
    try {
        auto b = backend::make_vulkan_backend(0, true);
        if (!b->supports_type(type)) { std::cout << "SKIP: MXFP4 float range unavailable\n"; return 77; }
        embedding(*b);
        for (auto dtype : {backend::Dtype::f32, backend::Dtype::bf16, backend::Dtype::f16}) {
            policy = dtype;
            matrix_paths.clear();
            scale_products(*b);
            split_range(*b);
            for (size_t width : {32u, 160u, 544u, 4096u})
                for (size_t rows : {1u, 7u, 67u})
                    for (size_t cols : {1u, 3u, 65u})
                        for (bool add : {false, true}) run(*b, width, rows, cols, cols == 1, add, 0, false);
            for (int mode : {1, 2, 3})
                for (size_t width : {160u, 544u})
                    for (bool add : {false, true}) run(*b, width, 7, 65, false, add, mode, false);
            for (size_t cols : {1u, 3u, 65u}) {
                grouped(*b, cols, cols == 1);
                for (bool add : {false, true}) run(*b, 160, 7, cols, cols == 1, add, 0, true);
            }
            const auto& profile = backend::vulkan_device_profile(*b);
            for (size_t width : {544u, 4096u}) {
                const size_t threshold = backend::tile_from_for(profile, false, width);
                for (size_t extent : {size_t(2), threshold - 1, threshold, threshold + 1}) batch_invariance(*b, width, extent);
            }
            const std::vector<std::string> paths(matrix_paths.begin(), matrix_paths.end());
            const std::vector<std::string> expected = dtype == backend::Dtype::f16 ? std::vector<std::string>{"block-int16", "f32"}
                : std::vector<std::string>{dtype == backend::Dtype::bf16 ? "bf16" : "f32"};
            require(paths == expected, "MXFP4 policy witness differs");
        }
        require(dispatched.count("embed_mxfp4") != 0, "MXFP4 embedding was not dispatched");
        require(dispatched.count("matmul_reduce_mxfp4") != 0, "MXFP4 wide partial reduction was not dispatched");
        const auto& profile = backend::vulkan_device_profile(*b);
        const std::string row = profile.mxfp4_integer_dot ? "matmul_row_mxfp4_dot" : "matmul_row_mxfp4";
        require(dispatched.count(row + "_1col") && dispatched.count(row) && dispatched.count(row + "_grouped"), "MXFP4 row variants were not dispatched");
        require(dispatched.count("matmul_row_mxfp4_float_x_1col") != 0, "MXFP4 F32 row was not dispatched");
        if (profile.prefer_integer_dot) {
            require(dispatched.count("copy_mxfp4") != 0, "MXFP4 weight copy was not dispatched");
            require(dispatched.count("matmul_tile_q8mx_small") || dispatched.count("matmul_tile_q8mx"), "MXFP4 integer tile was not dispatched");
        }
        for (const auto& name : dispatched) std::cout << "dispatched: " << name << '\n';
        std::cout << "MXFP4 policy paths: " << cases << " cases, " << checked << " outputs passed\n";
    } catch (const backend::VulkanUnavailable& e) {
        std::cout << "SKIP: " << e.what() << '\n'; return 77;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
