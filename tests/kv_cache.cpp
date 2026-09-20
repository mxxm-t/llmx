// Paged KV cache oracle (docs/KV-CACHE.md): the logical pool and sequence,
// the CPU storage behind them, and attention over a shuffled block table.
// HF/model history checks remain separate; this does not replace them.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "backends/cpu/cpu_backend.hpp"
#include "model/kv_cache.hpp"

namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

template <class F> void rejects(F fn, const char* what) {
    bool failed = false;
    try { fn(); } catch (const std::exception&) { failed = true; }
    require(failed, what);
}

float expected(size_t layer, size_t head, size_t pos, size_t lane, int seed) {
    return float(seed * 1000000 + layer * 100000 + pos * 1000 + head * 128 + lane);
}

void pool_and_sequence() {
    infer::BlockPool pool(4);
    const int32_t a = pool.alloc(), b = pool.alloc();
    require(a == 0 && b == 1 && pool.in_use() == 2, "ids are dense from zero");
    pool.retain(a);
    pool.release(a);
    require(pool.in_use() == 2, "a retained block stays in use");
    pool.release(a);
    require(pool.in_use() == 1 && pool.alloc() == a, "a freed block is reused first");
    rejects([&] { pool.release(b); pool.release(b); }, "double release accepted");
    require(pool.alloc() == b, "the first release of a double release must still free");
    pool.alloc();
    pool.alloc();
    require(pool.in_use() == 4, "four blocks in use at the budget");
    rejects([&] { pool.alloc(); }, "budget exhaustion accepted");

    infer::BlockPool p2(3);
    infer::KVSequence seq(&p2, 4);
    seq.prepare(5);
    require(seq.n_blocks() == 2 && seq.length() == 0, "prepare must not commit");
    seq.commit();
    require(seq.length() == 5 && p2.in_use() == 2, "commit advances the length");
    seq.prepare(3);
    require(seq.n_blocks() == 2, "positions 5..7 fit the second block");
    seq.abort();
    require(seq.length() == 5 && seq.n_blocks() == 2, "abort keeps committed blocks");
    seq.prepare(4);
    require(seq.n_blocks() == 3, "position 8 needs a third block");
    seq.abort();
    require(seq.n_blocks() == 2 && p2.in_use() == 2, "abort returns attempt blocks");
    rejects([&] { seq.prepare(8); }, "over-budget prepare accepted");
    require(seq.n_blocks() == 2 && p2.in_use() == 2 && seq.length() == 5,
            "a failed prepare must leave the sequence unchanged");
    seq.prepare(1);
    seq.commit();
    seq.reset();
    require(seq.length() == 0 && seq.n_blocks() == 0 && p2.in_use() == 0,
            "reset returns every block");
}

// Append `batch` tokens at `pos` through the backend and commit them.
void append(backend::CpuBackend& cpu, backend::KVStorage& st, infer::KVSequence& seq,
            size_t heads, size_t width, size_t batch, int seed) {
    const size_t pos = seq.length();
    seq.prepare(batch);
    const backend::KVView view = seq.view(&st);
    for (size_t layer = 0; layer < 3; ++layer) {
        std::vector<float> k(batch * heads * width), v(k.size());
        for (size_t b = 0; b < batch; ++b)
            for (size_t h = 0; h < heads; ++h)
                for (size_t d = 0; d < width; ++d) {
                    const size_t i = (b * heads + h) * width + d;
                    k[i] = expected(layer, h, pos + b, d, seed);
                    v[i] = -k[i];
                }
        cpu.kv_write(layer, view, pos, k.data(), v.data(), batch);
    }
    seq.commit();
}

void check(const backend::KVStorage& st, const infer::KVSequence& seq, size_t bt,
           size_t heads, size_t width, int seed) {
    const auto& s = dynamic_cast<const backend::CpuKVStorage&>(st);
    const backend::KVView view = seq.view(const_cast<backend::KVStorage*>(&st));
    for (size_t layer = 0; layer < 3; ++layer)
        for (size_t pos = 0; pos < seq.length(); ++pos) {
            const int32_t id = view.blocks[pos / bt];
            for (size_t h = 0; h < heads; ++h)
                for (size_t d = 0; d < width; ++d) {
                    const size_t at = (h * bt + pos % bt) * width + d;
                    const float value = expected(layer, h, pos, d, seed);
                    require(s.k(layer, id)[at] == value, "key history changed");
                    require(s.v(layer, id)[at] == -value, "value history changed");
                }
        }
}

void storage_growth_and_reset() {
    backend::CpuBackend cpu;
    cpu.set_threads(1);
    const size_t bt = cpu.kv_layout().block_tokens;
    for (size_t width : {size_t(1), size_t(8), size_t(42), size_t(128)}) {
        const size_t heads = 3, limit = 3 * bt;
        const size_t block_bytes = 3 * 2 * heads * bt * width * sizeof(float);
        auto st = cpu.kv_alloc(3, heads, width, 3 * block_bytes + block_bytes / 2);
        require(st->max_blocks() == 3, "budget rounds down to whole blocks");
        require(st->allocated_bytes() == 0, "storage backed before any write");
        infer::BlockPool pool(st->max_blocks());
        infer::KVSequence seq(&pool, bt);
        size_t last = 0;
        for (size_t want : {size_t(1), size_t(2), bt - 1, bt, bt + 1, 2 * bt - 1,
                            2 * bt, 2 * bt + 1, limit - 1, limit}) {
            if (want <= seq.length()) continue;
            append(cpu, *st, seq, heads, width, want - seq.length(), 1);
            check(*st, seq, bt, heads, width, 1);
            require(st->allocated_bytes() >= last, "storage shrank while growing");
            require(st->allocated_bytes() <= 3 * block_bytes, "storage exceeded the budget");
            last = st->allocated_bytes();
        }
        require(seq.length() == limit, "sequence did not reach the budget");
        rejects([&] { append(cpu, *st, seq, heads, width, 1, 1); }, "write past the budget accepted");
        require(seq.length() == limit && pool.in_use() == 3, "failed append changed the sequence");
        check(*st, seq, bt, heads, width, 1);
        {
            const backend::KVView view = seq.view(st.get());
            std::vector<float> row(heads * width, 0.0f);
            rejects([&] { cpu.kv_write(3, view, 0, row.data(), row.data(), 1); },
                    "layer outside storage accepted");
            rejects([&] { cpu.kv_write(0, view, limit, row.data(), row.data(), 1); },
                    "position outside the view accepted");
        }
        // Reset returns the blocks but keeps the storage they occupied.
        seq.reset();
        require(st->allocated_bytes() == last, "reset discarded retained storage");
        append(cpu, *st, seq, heads, width, 3, 2);
        check(*st, seq, bt, heads, width, 2);
    }
}

// Attention over a shuffled table must equal attention over the same history
// in a different table bit for bit, and match a double-precision reference.
void attention_over_blocks() {
    backend::CpuBackend cpu;
    cpu.set_threads(2);
    const size_t bt = cpu.kv_layout().block_tokens;
    const int n_head = 4, n_head_kv = 2, head_dim = 40;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);

    for (size_t n_past : {size_t(0), bt - 1, bt, bt + 1, 2 * bt + 3}) {
        for (int nbatch : {1, 3}) {
            const size_t sequence = n_past + (size_t)nbatch;
            std::vector<float> K(sequence * n_head_kv * head_dim), V(K.size());
            std::vector<float> Q((size_t)nbatch * n_head * head_dim);
            for (auto& x : K) x = uni(rng);
            for (auto& x : V) x = uni(rng);
            for (auto& x : Q) x = uni(rng);

            // Two storages: one whose blocks were taken in order, one whose
            // pool was churned first so the table is out of order.
            std::vector<std::vector<float>> outs;
            for (int churn = 0; churn < 2; ++churn) {
                const size_t block_bytes = 2 * n_head_kv * bt * head_dim * sizeof(float);
                auto st = cpu.kv_alloc(1, n_head_kv, head_dim, 8 * block_bytes);
                infer::BlockPool pool(st->max_blocks());
                if (churn) {
                    std::vector<int32_t> taken;
                    for (int i = 0; i < 6; ++i) taken.push_back(pool.alloc());
                    for (int i : {4, 1, 5, 2}) pool.release(taken[(size_t)i]);
                }
                infer::KVSequence seq(&pool, bt);
                seq.prepare(n_past);
                cpu.kv_write(0, seq.view(st.get()), 0, K.data(), V.data(), n_past);
                seq.commit();
                seq.prepare((size_t)nbatch);
                const backend::KVView view = seq.view(st.get());
                cpu.kv_write(0, view, n_past, K.data() + n_past * n_head_kv * head_dim,
                             V.data() + n_past * n_head_kv * head_dim, (size_t)nbatch);
                std::vector<float> out(Q.size(), 0.0f);
                cpu.attention(Q.data(), 0, view, out.data(), n_head, n_head_kv, head_dim, nbatch);
                if (n_past == 0 && churn == 0) {
                    infer::KVSequence shorter(&pool, bt);
                    rejects([&] {
                        cpu.attention(Q.data(), 0, shorter.view(st.get()), out.data(),
                                      n_head, n_head_kv, head_dim, nbatch);
                    }, "attention over an empty view accepted");
                }
                outs.push_back(out);
            }
            require(outs[0] == outs[1], "block table order changed the result");

            const double scale = 1.0 / std::sqrt((double)head_dim);
            for (int b = 0; b < nbatch; ++b)
                for (int h = 0; h < n_head; ++h) {
                    const int kvh = h / (n_head / n_head_kv);
                    const size_t end = n_past + (size_t)b + 1;
                    std::vector<double> sc(end);
                    double mx = -1e300;
                    for (size_t t = 0; t < end; ++t) {
                        double s = 0;
                        for (int d = 0; d < head_dim; ++d)
                            s += (double)Q[((size_t)b * n_head + h) * head_dim + d] *
                                 K[(t * n_head_kv + kvh) * head_dim + d];
                        sc[t] = s * scale;
                        mx = std::max(mx, sc[t]);
                    }
                    double sum = 0;
                    for (auto& x : sc) { x = std::exp(x - mx); sum += x; }
                    for (int d = 0; d < head_dim; ++d) {
                        double acc = 0;
                        for (size_t t = 0; t < end; ++t)
                            acc += sc[t] / sum * V[(t * n_head_kv + kvh) * head_dim + d];
                        const double got = outs[0][((size_t)b * n_head + h) * head_dim + d];
                        require(std::fabs(got - acc) <= 1e-5 * (1.0 + std::fabs(acc)),
                                "attention differs from the reference");
                    }
                }
        }
    }
}

}  // namespace

int main() {
    try {
        pool_and_sequence();
        storage_growth_and_reset();
        attention_over_blocks();
        std::cout << "KV cache: pool, sequence, on-demand storage, reset and paged attention pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
