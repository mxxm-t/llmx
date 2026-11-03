#pragma once
// A tensor group's sums held to one oracle by each backend's collective test (backend-rocm, backend-vulkan), so their collectives give the same bytes on the same inputs.
// The oracle adds as shaders/sum_rows.comp adds: t = p0, t += p1, ..., then residual += t, or residual = t in place; signed zeros, tails and offsets included.
#include "backends/backend.hpp"

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace group_sums {

struct Case {
    std::vector<std::vector<float>> partial;   // per member
    std::vector<float> start;                  // the residual before the sum
};

// Whole sixteenths and both zeros, each member placing +0 and -0 at its own strides, so the oracle meets -0 + -0, -0 + +0 and x + -x in every member order.
inline Case make(size_t W, size_t n, uint32_t seed) {
    Case c;
    c.partial.assign(W, std::vector<float>(n));
    c.start.resize(n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t m = 0; m < W; ++m) {
            seed = seed * 1664525u + 1013904223u;
            const size_t r = (i + 3 * m) % 7;
            c.partial[m][i] = r == 0 ? 0.0f : r == 1 ? -0.0f : r == 2 && m ? -c.partial[0][i] : float(int(seed >> 16) % 2001 - 1000) / 16.0f;
        }
        seed = seed * 1664525u + 1013904223u;
        c.start[i] = i % 5 == 0 ? -0.0f : i % 5 == 1 ? 0.0f : float(int(seed >> 8) % 4001 - 2000) / 16.0f;
    }
    return c;
}

inline std::vector<float> oracle(const Case& c, bool replace) {
    std::vector<float> out(c.start.size());
    for (size_t i = 0; i < out.size(); ++i) {
        float t = c.partial[0][i];
        for (size_t m = 1; m < c.partial.size(); ++m) t += c.partial[m][i];
        out[i] = replace ? t : c.start[i] + t;
    }
    return out;
}

// sum_into on residuals at an aligned and an unaligned offset, then sum_in_place, at each of `counts` rows of `width`; each member's bytes must be the oracle's.
inline size_t check(const std::vector<backend::Backend*>& group, backend::Collective& c, size_t width, std::initializer_list<size_t> counts, uint32_t seed,
                    const std::function<void(bool, const std::string&)>& require) {
    const size_t W = group.size();
    size_t checks = 0;
    for (size_t offset : {size_t(0), size_t(7)})
        for (size_t rows : counts) {
            const size_t n = rows * width;
            const Case k = make(W, n, seed++);
            std::vector<backend::BufferPtr> x;
            std::vector<backend::Slice> xs;
            for (size_t m = 0; m < W; ++m) {
                x.push_back(group[m]->alloc((n + offset) * sizeof(float)));
                group[m]->write(*x.back(), offset * sizeof(float), k.start.data(), n * sizeof(float));
                const backend::Slice p = c.partial(m);
                group[m]->write(*p.buffer, p.offset * sizeof(float), k.partial[m].data(), n * sizeof(float));
                xs.push_back({x.back().get(), offset});
            }
            c.sum_into(xs, rows, width);
            const std::vector<float> into = oracle(k, false), in_place = oracle(k, true);
            for (size_t m = 0; m < W; ++m) {
                group[m]->submit();
                std::vector<float> got(n);
                group[m]->read(*x[m], offset * sizeof(float), got.data(), n * sizeof(float));
                require(std::memcmp(got.data(), into.data(), n * sizeof(float)) == 0, "a member's residual is not the oracle's sum in member order");
            }
            for (size_t m = 0; m < W; ++m) {
                const backend::Slice p = c.partial(m);
                group[m]->write(*p.buffer, p.offset * sizeof(float), k.partial[m].data(), n * sizeof(float));
            }
            c.sum_in_place(rows, width);
            for (size_t m = 0; m < W; ++m) {
                group[m]->submit();
                std::vector<float> got(n);
                const backend::Slice p = c.partial(m);
                group[m]->read(*p.buffer, p.offset * sizeof(float), got.data(), n * sizeof(float));
                require(std::memcmp(got.data(), in_place.data(), n * sizeof(float)) == 0, "a member's partial rows summed in place are not the oracle's sum in member order");
            }
            checks += 2;
        }
    return checks;
}

} // namespace group_sums
