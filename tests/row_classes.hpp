#pragma once
// Every pair of extents a backend puts in one class (Backend::row_class) gives the same bits through each op that chooses its arithmetic by extent: a matmul, the routed products and attention after a history (docs/SPECULATIVE.md, section 1).
// backend-group runs it on the CPU and backend-vulkan on a device, at extents on each side of every crossover and tile split.
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#include "backends/backend.hpp"
#include "model/kv_cache.hpp"

namespace row_classes {

// A quantized matrix of `rows` rows `nin` wide, as each test makes its own.
using Matrix = std::function<std::vector<uint8_t>(uint32_t type, size_t nin, size_t rows, uint32_t seed)>;

inline const std::vector<size_t> kExtents = {1, 2, 3, 4, 5, 15, 16, 17, 23, 24, 25, 31, 32, 33, 39, 40, 41, 47, 48, 49, 63, 64, 65, 95, 96, 97, 127, 128,
                                             129, 130, 191, 192, 193, 194, 255, 256, 257, 258, 319, 320, 321, 322, 383, 384, 385, 386, 447, 448,
                                             449, 450, 511, 512, 513, 1000};

inline std::vector<float> values(size_t n, uint32_t seed, float scale = 1.0f) {
    std::vector<float> v(n);
    uint32_t s = seed * 2654435761u + 1;
    for (float& x : v) {
        s = s * 1664525u + 1013904223u;
        x = scale * (float((s >> 8) % 2001) / 1000.0f - 1.0f);
    }
    return v;
}

// The outputs of each extent, and the pairs of one class among them required to be the same bits; `what` names a mismatch.
inline size_t same_within_classes(const backend::Backend& b, const std::vector<std::vector<float>>& out, const std::string& what) {
    size_t pairs = 0;
    for (size_t i = 0; i < out.size(); ++i)
        for (size_t j = i + 1; j < out.size(); ++j) {
            if (b.row_class(kExtents[i]) != b.row_class(kExtents[j])) continue;
            if (std::memcmp(out[i].data(), out[j].data(), out[i].size() * sizeof(float)))
                throw std::runtime_error(what + ": extents " + std::to_string(kExtents[i]) + " and " + std::to_string(kExtents[j]) + " of one class differ");
            ++pairs;
        }
    return pairs;
}

inline std::vector<float> read(backend::Backend& b, const backend::Buffer& buf, size_t n) {
    std::vector<float> v(n);
    b.read(buf, 0, v.data(), n * sizeof(float));
    return v;
}

// Four rows of one prompt at each extent through a matmul of each type at rows 256 and 4096 wide, the routed gate and down projections of 8 experts taking 2 a row, and attention of 4 query rows over 128-wide heads after a 70-token history; the pairs found the same.
inline size_t check(backend::Backend& b, const std::vector<uint32_t>& types, const Matrix& matrix) {
    if (b.row_class(1) == b.row_class(2)) throw std::runtime_error("a generated token shares a class with a prompt");
    const size_t rows = 4, nout = 64;
    size_t pairs = 0;
    for (uint32_t type : types)
        for (size_t nin : {size_t(256), size_t(4096)}) {
            const std::vector<uint8_t> w = matrix(type, nin, nout, 71);
            const std::vector<float> x = values(rows * nin, 72);
            const backend::BufferPtr wb = b.adopt(w.data(), w.size()), xb = b.adopt(x.data(), x.size() * sizeof(float));
            const backend::BufferPtr y = b.alloc(rows * nout * sizeof(float));
            std::vector<std::vector<float>> out;
            for (size_t e : kExtents) {
                const backend::RowRun run[1] = {{rows, e}};
                b.matmul(type, {wb.get(), 0}, {xb.get(), 0}, {y.get(), 0}, nin, nout, rows, {run, 1});
                out.push_back(read(b, *y, rows * nout));
            }
            pairs += same_within_classes(b, out, "a matmul of type " + std::to_string(type) + ", rows " + std::to_string(nin) + " wide");
        }

    const size_t n_expert = 8, k = 2, nin = 256;
    const std::vector<float> scores = values(rows * n_expert, 73, 3.0f), x = values(rows * nin, 74), x2 = values(rows * k * nin, 75), base = values(rows * nout, 76);
    const backend::BufferPtr sb = b.adopt(scores.data(), scores.size() * sizeof(float));
    const backend::BufferPtr xb = b.adopt(x.data(), x.size() * sizeof(float)), x2b = b.adopt(x2.data(), x2.size() * sizeof(float));
    const backend::BufferPtr ids = b.alloc(rows * k * sizeof(float)), wts = b.alloc(rows * k * sizeof(float));
    b.route_experts({sb.get(), 0}, rows, n_expert, k, true, {ids.get(), 0}, {wts.get(), 0});
    const backend::Backend::Routing routing{{ids.get(), 0}, {wts.get(), 0}, k, n_expert};
    for (uint32_t type : types) {
        const std::vector<uint8_t> g = matrix(type, nin, n_expert * nout, 77), d = matrix(type, nin, n_expert * nout, 78);
        const backend::BufferPtr gb = b.adopt(g.data(), g.size()), db = b.adopt(d.data(), d.size());
        const backend::BufferPtr go = b.alloc(rows * k * nout * sizeof(float)), yo = b.alloc(rows * nout * sizeof(float));
        std::vector<std::vector<float>> gate, down;
        for (size_t e : kExtents) {
            const backend::RowRun run[1] = {{rows, e}};
            b.matmul_experts({{type, {gb.get(), 0}, {go.get(), 0}, nout}}, {xb.get(), 0}, nin, rows, routing, {run, 1});
            gate.push_back(read(b, *go, rows * k * nout));
            b.write(*yo, 0, base.data(), base.size() * sizeof(float));
            b.matmul_experts_add(type, {db.get(), 0}, {x2b.get(), 0}, {yo.get(), 0}, nin, nout, rows, routing, {run, 1});
            down.push_back(read(b, *yo, rows * nout));
        }
        pairs += same_within_classes(b, gate, "a routed projection of type " + std::to_string(type));
        pairs += same_within_classes(b, down, "a routed projection added of type " + std::to_string(type));
    }

    const int n_head = 4, n_head_kv = 2, head_dim = 128;
    const size_t hist = 70, qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim;
    const std::vector<float> keys = values((hist + rows) * kvw, 79), vals = values((hist + rows) * kvw, 80), q = values(rows * qw, 81, 4.0f);
    const backend::BufferPtr kb = b.adopt(keys.data(), keys.size() * sizeof(float)), vb = b.adopt(vals.data(), vals.size() * sizeof(float));
    const backend::BufferPtr qb = b.adopt(q.data(), q.size() * sizeof(float)), ob = b.alloc(rows * qw * sizeof(float));
    auto st = b.kv_alloc(1, n_head_kv, head_dim, 512);
    infer::BlockPool pool(st->max_blocks());
    std::vector<std::vector<float>> att;
    for (size_t e : kExtents) {
        infer::KVSequence seq(&pool, b.kv_layout().block_tokens);
        seq.prepare(hist);
        const backend::KVView h = seq.view(st.get());
        b.kv_write(0, &h, 1, {kb.get(), 0}, {vb.get(), 0});
        seq.commit();
        seq.prepare(rows);
        backend::KVView v = seq.view(st.get());
        v.extent = e;
        b.kv_write(0, &v, 1, {kb.get(), hist * kvw}, {vb.get(), hist * kvw});
        b.attention({qb.get(), 0}, 0, &v, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
        att.push_back(read(b, *ob, rows * qw));
        seq.abort();
    }
    pairs += same_within_classes(b, att, "attention");
    return pairs;
}

} // namespace row_classes
