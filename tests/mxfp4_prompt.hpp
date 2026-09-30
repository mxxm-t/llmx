#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "backends/backend.hpp"

// The same prompt precision and row-run checks apply to CPU and Vulkan MXFP4 products.
namespace mxfp4_test {
constexpr uint32_t type = 39;
inline void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
inline uint32_t bits(float x) { uint32_t n; std::memcpy(&n, &x, 4); return n; }
inline backend::CSlice input(const backend::BufferPtr& p) { return {p.get(), 0}; }
// A prompt must preserve original F32 activations even when each microbatch contains one row, or when decode rows and a different weight type share a call.
inline size_t prompt_invariance(backend::Backend& b, size_t width, size_t extent) {
    size_t checked = 0;
    constexpr size_t rows = 7, other_rows = 5, decode_rows = 2;
    const size_t cols = extent + decode_rows;
    std::vector<uint8_t> w(rows * (width / 32) * 17, 0);
    std::vector<size_t> selected(rows);
    for (size_t row = 0; row < rows; ++row) {
        selected[row] = ((row * 37) % (width / 32)) * 32 + 1 + row % 31;
        for (size_t block = 0; block < width / 32; ++block) w[(row * (width / 32) + block) * 17] = 127;
        const size_t j = selected[row], at = (row * (width / 32) + j / 32) * 17 + 1 + j % 16;
        w[at] = uint8_t(j % 32 < 16 ? 2 : 0x20);
    }
    std::vector<float> x(cols * width), fw(other_rows * width);
    for (size_t i = 0; i < x.size(); ++i) x[i] = i % 32 ? float(1 + (i * 37) % 997) / 1001.0f : 1.0f;
    for (size_t i = 0; i < fw.size(); ++i) fw[i] = float(int((i * 31) % 101) - 50) / 137.0f;
    auto wb = b.adopt(w.data(), w.size()), fb = b.adopt(fw.data(), fw.size() * sizeof(float));
    auto xb = b.adopt(x.data(), x.size() * sizeof(float));
    std::vector<float> reference, other_reference;
    for (bool micro : {false, true})
        for (bool grouped : {false, true}) {
            auto yb = b.alloc(cols * rows * sizeof(float)), ob = b.alloc(cols * other_rows * sizeof(float));
            const auto call = [&](size_t first, size_t count, backend::RowRuns runs) {
                const backend::Slice ys{yb.get(), first * rows}, os{ob.get(), first * other_rows};
                const backend::CSlice xs{xb.get(), first * width};
                if (grouped) b.matmul_group({{type, input(wb), ys, rows}, {0, input(fb), os, other_rows}}, xs, width, count, runs);
                else {
                    b.matmul(type, input(wb), xs, ys, width, rows, count, runs);
                    b.matmul(0, input(fb), xs, os, width, other_rows, count, runs);
                }
            };
            if (micro) {
                for (size_t c = 0; c < cols; ++c) {
                    const backend::RowRun run{1, c < decode_rows ? 1 : extent};
                    call(c, 1, {&run, 1});
                }
            } else {
                const backend::RowRun runs[] = {{decode_rows, 1}, {cols, extent}};
                call(0, cols, {runs, 2});
            }
            std::vector<float> y(cols * rows), other(cols * other_rows);
            b.read(*yb, 0, y.data(), y.size() * sizeof(float));
            b.read(*ob, 0, other.data(), other.size() * sizeof(float));
            for (size_t c = decode_rows; c < cols; ++c)
                for (size_t row = 0; row < rows; ++row)
                    require(bits(y[c * rows + row]) == bits(x[c * width + selected[row]]), "MXFP4 prompt rounded its one-hot activation");
            if (!micro && !grouped) { reference = y; other_reference = other; }
            else {
                require(std::memcmp(y.data(), reference.data(), y.size() * sizeof(float)) == 0, "MXFP4 prompt/decode rows depend on microbatch or mixed type");
                require(std::memcmp(other.data(), other_reference.data(), other.size() * sizeof(float)) == 0, "MXFP4 changed the companion type's choice");
            }
            checked += y.size() + other.size();
        }
    return checked;
}
inline size_t routed_invariance(backend::Backend& b, size_t width, size_t extent, bool add) {
    size_t checked = 0;
    constexpr size_t rows = 7, experts = 3, k = 2, decode_rows = 9;
    const size_t cols = extent + decode_rows, entries = cols * k, xrows = add ? entries : cols;
    std::vector<uint8_t> w(experts * rows * (width / 32) * 17, 0);
    std::vector<size_t> selected(experts * rows);
    for (size_t row = 0; row < selected.size(); ++row) {
        selected[row] = ((row * 37) % (width / 32)) * 32 + 1 + row % 31;
        for (size_t block = 0; block < width / 32; ++block) w[(row * (width / 32) + block) * 17] = 127;
        const size_t j = selected[row], at = (row * (width / 32) + j / 32) * 17 + 1 + j % 16;
        w[at] = uint8_t(j % 32 < 16 ? 2 : 0x20);
    }
    std::vector<float> x(xrows * width), probability(entries, .5f);
    std::vector<uint32_t> ids(entries);
    for (size_t i = 0; i < x.size(); ++i) x[i] = i % 32 ? float(1 + (i * 37) % 997) / 1001.0f : 1.0f;
    for (size_t i = 0; i < entries; ++i) ids[i] = uint32_t((i / k + i % k) % experts);
    auto wb = b.adopt(w.data(), w.size()), xb = b.adopt(x.data(), x.size() * sizeof(float));
    auto ib = b.adopt(ids.data(), ids.size() * sizeof(uint32_t)), pb = b.adopt(probability.data(), probability.size() * sizeof(float));
    std::vector<float> reference;
    for (bool micro : {false, true}) {
        std::vector<float> y((add ? cols : entries) * rows, add ? .25f : 0.0f);
        auto yb = b.adopt(y.data(), y.size() * sizeof(float));
        const auto call = [&](size_t first, size_t count, backend::RowRuns runs) {
            const backend::Backend::Routing routing{{ib.get(), first * k}, {pb.get(), first * k}, k, experts};
            if (add) b.matmul_experts_add(type, input(wb), {xb.get(), first * k * width}, {yb.get(), first * rows}, width, rows, count, routing, runs);
            else b.matmul_experts({{type, input(wb), {yb.get(), first * k * rows}, rows}}, {xb.get(), first * width}, width, count, routing, runs);
        };
        if (micro) {
            for (size_t c = 0; c < cols; ++c) {
                const backend::RowRun run{1, c < decode_rows ? 1 : extent};
                call(c, 1, {&run, 1});
            }
        } else {
            const backend::RowRun runs[] = {{decode_rows, 1}, {cols, extent}};
            call(0, cols, {runs, 2});
        }
        b.read(*yb, 0, y.data(), y.size() * sizeof(float));
        for (size_t c = decode_rows; c < cols; ++c)
            for (size_t row = 0; row < rows; ++row) {
                float sum = 0;
                for (size_t slot = 0; slot < k; ++slot) {
                    const size_t e = c * k + slot, j = selected[ids[e] * rows + row];
                    if (add) sum += .5f * x[e * width + j];
                    else require(bits(y[e * rows + row]) == bits(x[c * width + j]), "MXFP4 routed prompt rounded its one-hot activation");
                }
                if (add) require(bits(y[c * rows + row]) == bits(.25f + sum), "MXFP4 routed residual differs from original activations");
            }
        if (!micro) reference = y;
        else require(std::memcmp(y.data(), reference.data(), y.size() * sizeof(float)) == 0, "MXFP4 routed prompt/decode rows depend on microbatch");
        checked += y.size();
    }
    return checked;
}
} // namespace mxfp4_test
