// Paged KV cache oracle (docs/KV-CACHE.md): the logical pool and sequence, the CPU storage behind them, and attention over a shuffled block table.
// HF/model history checks remain separate; this does not replace them.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backends/cpu/cpu_backend.hpp"
#include "model/arch_qwen.hpp"
#include "model/kv_cache.hpp"
#include "tiny_qwen.hpp"

// Fails the Nth allocation of at least `min_bytes` after arming, once.
// Used to fail the prefill activation arena, which is one allocation large enough that no other allocation on the path reaches the threshold.
static thread_local size_t fail_large_after = 0, fail_min_bytes = 0;

void* operator new(std::size_t n) {
    if (fail_min_bytes && n >= fail_min_bytes && fail_large_after && --fail_large_after == 0) {
        fail_min_bytes = 0;
        throw std::bad_alloc();
    }
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

// Deliberately narrower than std::exception: this file injects bad_alloc of its own, and catching that here would let an allocation failure stand in for the budget or bounds error each case is checking for.
template <class F> void rejects(F fn, const char* what) {
    bool failed = false;
    try { fn(); }
    catch (const std::bad_alloc&) { throw; }
    catch (const std::exception&) { failed = true; }
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
    rejects([&] { pool.retain(b); }, "retain of a free block accepted");
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

    seq.prepare(1);
    seq.commit();
    rejects([&] { seq.prepare(std::numeric_limits<size_t>::max()); }, "length overflow accepted");
    require(seq.length() == 1 && seq.n_blocks() == 1 && p2.in_use() == 1,
            "failed overflow prepare changed the sequence");
    seq.reset();
    rejects([&] { infer::KVSequence bad(&p2, 0); }, "zero block size accepted");
    rejects([&] { infer::KVSequence unbound; unbound.prepare(1); }, "unbound sequence accepted");
    rejects([&] { infer::BlockPool held(2); held.alloc(); held.configure(4); },
            "pool reconfigured while blocks are held");

    // Ownership: a sequence returns its blocks when destroyed or moved from.
    {
        infer::KVSequence owner(&p2, 4);
        owner.prepare(6);
        owner.commit();
        require(p2.in_use() == 2, "owner holds two blocks");
        infer::KVSequence moved(std::move(owner));
        require(moved.length() == 6 && moved.n_blocks() == 2 && p2.in_use() == 2,
                "move must transfer, not duplicate, the blocks");
        infer::KVSequence other(&p2, 4);
        other.prepare(1);
        other.commit();
        require(p2.in_use() == 3, "second owner holds one block");
        other = std::move(moved);
        require(other.length() == 6 && p2.in_use() == 2, "move assignment must release the old blocks");
    }
    require(p2.in_use() == 0, "destroyed sequences must return their blocks");
    p2.configure(5);
    require(p2.max_blocks() == 5 && p2.alloc() == 0, "an idle pool can be reconfigured in place");
    p2.release(0);

    // A sequence bound before the pool grew must still take ids safely.
    {
        infer::BlockPool grow(1);
        infer::KVSequence bound(&grow, 4);
        grow.configure(6);
        bound.prepare(4 * 6);
        require(bound.n_blocks() == 6 && grow.in_use() == 6, "prepare after reconfiguration");
        bound.abort();
        require(grow.in_use() == 0, "abort after reconfiguration returned every id");
        rejects([&] { bound.prepare(4 * 6 + 1); }, "over-budget prepare after reconfiguration accepted");
        require(grow.in_use() == 0 && bound.n_blocks() == 0, "failed prepare left ids outstanding");
        bound.prepare(3);
        bound.commit();
        require(bound.length() == 3 && grow.in_use() == 1, "retry after failed prepare");
    }
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
        const auto kb = cpu.adopt(k.data(), k.size() * sizeof(float));
        const auto vb = cpu.adopt(v.data(), v.size() * sizeof(float));
        cpu.kv_write(layer, &view, 1, {kb.get(), 0}, {vb.get(), 0});
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
        auto st = cpu.kv_alloc(3, heads, width, limit - 1);
        require(st->max_blocks() == 3, "budget rounds up to whole blocks");
        require(st->allocated_bytes() == 0 && st->peak_bytes() == 0, "storage backed before any write");
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
            require(st->allocated_bytes() % block_bytes == 0,
                    "retained capacity is not a whole number of blocks");
            require(st->peak_bytes() >= st->allocated_bytes() &&
                    st->peak_bytes() <= last + st->allocated_bytes(),
                    "growth peak must be old plus new at most");
            last = st->allocated_bytes();
        }
        require(seq.length() == limit, "sequence did not reach the budget");
        rejects([&] { append(cpu, *st, seq, heads, width, 1, 1); }, "write past the budget accepted");
        require(seq.length() == limit && pool.in_use() == 3, "failed append changed the sequence");
        check(*st, seq, bt, heads, width, 1);
        {
            // A view claiming one more row than its table covers.
            backend::KVView view = seq.view(st.get());
            view.nq = 1;
            std::vector<float> row(heads * width, 0.0f);
            const auto rowb = cpu.adopt(row.data(), row.size() * sizeof(float));
            rejects([&] { cpu.kv_write(3, &view, 1, {rowb.get(), 0}, {rowb.get(), 0}); },
                    "layer outside storage accepted");
            rejects([&] { cpu.kv_write(0, &view, 1, {rowb.get(), 0}, {rowb.get(), 0}); },
                    "position outside the view accepted");
        }
        // Reset returns the blocks but keeps the storage they occupied.
        seq.reset();
        require(st->allocated_bytes() == last, "reset discarded retained storage");
        append(cpu, *st, seq, heads, width, 3, 2);
        check(*st, seq, bt, heads, width, 2);
    }
}

// A fork at a whole-block length shares every block below it, read-only, and allocates and copies nothing, so the two histories agree up to the fork and then diverge without touching each other.
// A length inside a block copies its partial tail into a private block, and a length past the history is refused.
// A history truncated into a shared block cannot be appended to.
// Releases follow the refcounts.
void fork_shares_blocks() {
    backend::CpuBackend cpu;
    cpu.set_threads(1);
    const size_t bt = cpu.kv_layout().block_tokens, heads = 2, width = 8;
    auto st = cpu.kv_alloc(3, heads, width, 6 * bt);
    infer::BlockPool pool(st->max_blocks());
    infer::KVSequence seq(&pool, bt);
    append(cpu, *st, seq, heads, width, 2 * bt + 2, 1);
    const size_t before = pool.in_use();

    infer::KVSequence::Tail tail;
    infer::KVSequence fork = seq.fork(bt, tail);
    require(fork.length() == bt && fork.n_blocks() == 1, "fork length or table");
    require(tail.from == -1 && tail.to == -1 && pool.in_use() == before, "a whole-block fork allocated a block");
    const backend::KVView vs = seq.view(st.get()), vf = fork.view(st.get());
    require(vf.blocks[0] == vs.blocks[0] && pool.refs(vs.blocks[0]) == 2 && pool.refs(vs.blocks[1]) == 1 &&
            pool.refs(vs.blocks[2]) == 1, "shared blocks miscounted");
    check(*st, fork, bt, heads, width, 1);
    rejects([&] { infer::KVSequence::Tail t; seq.fork(2 * bt + 3, t); }, "fork past the history accepted");
    require(pool.in_use() == before && pool.refs(vs.blocks[0]) == 2, "refused fork changed the pool");

    // A fork inside a block takes a private tail for the backend to fill.
    {
        infer::KVSequence::Tail part_tail;
        infer::KVSequence part = seq.fork(bt + 2, part_tail);
        const backend::KVView vp = part.view(st.get());
        require(part.length() == bt + 2 && part.n_blocks() == 2 && vp.blocks[0] == vs.blocks[0] &&
                part_tail.from == vs.blocks[1] && part_tail.to == vp.blocks[1] && pool.refs(vp.blocks[1]) == 1,
                "tail not private");
        cpu.kv_copy(*st, part_tail.from, part_tail.to);
        check(*st, part, bt, heads, width, 1);
    }
    require(pool.in_use() == before && pool.refs(vs.blocks[0]) == 2, "a released fork kept its blocks");

    // Each history grows on its own; the shared block stays as it was.
    append(cpu, *st, seq, heads, width, 3, 2);
    append(cpu, *st, fork, heads, width, 5, 3);
    const auto& s = dynamic_cast<const backend::CpuKVStorage&>(*st);
    for (const infer::KVSequence* q : {&seq, &fork}) {
        const size_t forked = q == &seq ? 2 * bt + 2 : bt;
        const int seed_after = q == &seq ? 2 : 3;
        const backend::KVView view = q->view(st.get());
        for (size_t layer = 0; layer < 3; ++layer)
            for (size_t pos = 0; pos < q->length(); ++pos)
                for (size_t h = 0; h < heads; ++h)
                    for (size_t d = 0; d < width; ++d) {
                        const float want = expected(layer, h, pos, d, pos < forked ? 1 : seed_after);
                        require(s.k(layer, view.blocks[pos / bt])[(h * bt + pos % bt) * width + d] == want,
                                "history diverged in the wrong place");
                    }
    }

    // Truncating into the shared block and appending would write what the other sequence reads.
    seq.truncate(bt / 2);
    rejects([&] { seq.prepare(1); }, "append into a shared block accepted");
    require(seq.length() == bt / 2 && pool.refs(vs.blocks[0]) == 2, "refused append changed state");

    // Releases follow the refcounts: the fork's reset frees its own block and only drops a reference on the shared block, which stays in use until the last holder lets go.
    const size_t held = pool.in_use();
    fork.reset();
    require(pool.refs(vs.blocks[0]) == 1 && pool.in_use() == held - 1, "fork reset released the wrong blocks");
    seq.reset();
    require(pool.in_use() == 0, "blocks leaked across forks");
}

// Attention over a shuffled table must equal attention over the same history in a different table bit for bit, and match a double-precision reference.
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

            // Two storages: one whose blocks were taken in order, one whose pool was churned first so the table is out of order.
            std::vector<std::vector<float>> outs;
            for (int churn = 0; churn < 2; ++churn) {
                auto st = cpu.kv_alloc(1, n_head_kv, head_dim, 8 * bt);
                infer::BlockPool pool(st->max_blocks());
                if (churn) {
                    std::vector<int32_t> taken;
                    for (int i = 0; i < 6; ++i) taken.push_back(pool.alloc());
                    for (int i : {4, 1, 5, 2}) pool.release(taken[(size_t)i]);
                }
                infer::KVSequence seq(&pool, bt);
                seq.prepare(n_past);
                const auto Kb = cpu.adopt(K.data(), K.size() * sizeof(float));
                const auto Vb = cpu.adopt(V.data(), V.size() * sizeof(float));
                const auto Qb = cpu.adopt(Q.data(), Q.size() * sizeof(float));
                const backend::KVView history = seq.view(st.get());
                cpu.kv_write(0, &history, 1, {Kb.get(), 0}, {Vb.get(), 0});
                seq.commit();
                seq.prepare((size_t)nbatch);
                const backend::KVView view = seq.view(st.get());
                const size_t tail = n_past * (size_t)n_head_kv * (size_t)head_dim;
                cpu.kv_write(0, &view, 1, {Kb.get(), tail}, {Vb.get(), tail});
                std::vector<float> out(Q.size(), 0.0f);
                const auto ob = cpu.adopt(out.data(), out.size() * sizeof(float));
                cpu.attention({Qb.get(), 0}, 0, &view, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
                if (n_past == 0 && churn == 0) {
                    infer::KVSequence shorter(&pool, bt);
                    const backend::KVView empty = shorter.view(st.get());
                    rejects([&] {
                        cpu.attention({Qb.get(), 0}, 0, &empty, 1, {ob.get(), 0},
                                      n_head, n_head_kv, head_dim);
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

// Two sequences in one pass: their rows concatenated in view order through one kv_write and one attention call must equal the same two sequences written and attended separately, bit for bit, since the per-view work is the single-sequence work.
// Histories straddle a block edge and differ in length so a row offset or a length taken from the wrong view shows.
void batched_views() {
    backend::CpuBackend cpu;
    cpu.set_threads(2);
    const size_t bt = cpu.kv_layout().block_tokens;
    const int n_head = 4, n_head_kv = 2, head_dim = 24;
    const size_t kvw = (size_t)n_head_kv * head_dim, qw = (size_t)n_head * head_dim;
    const size_t past[2] = {5, bt + 3}, nq[2] = {2, 3};
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::vector<float> K[2], V[2], Q[2];
    for (int i = 0; i < 2; ++i) {
        K[i].resize((past[i] + nq[i]) * kvw); V[i].resize(K[i].size()); Q[i].resize(nq[i] * qw);
        for (auto& x : K[i]) x = uni(rng);
        for (auto& x : V[i]) x = uni(rng);
        for (auto& x : Q[i]) x = uni(rng);
    }
    // Separate: each sequence written and attended on its own.
    std::vector<float> separate;
    std::vector<float> joint;
    for (int mode = 0; mode < 2; ++mode) {
        auto st = cpu.kv_alloc(1, n_head_kv, head_dim, 8 * bt);
        infer::BlockPool pool(st->max_blocks());
        infer::KVSequence seq[2] = {infer::KVSequence(&pool, bt), infer::KVSequence(&pool, bt)};
        backend::BufferPtr keep[8];
        for (int i = 0; i < 2; ++i) {
            seq[i].prepare(past[i]);
            const backend::KVView h = seq[i].view(st.get());
            keep[i * 2] = cpu.adopt(K[i].data(), K[i].size() * sizeof(float));
            keep[i * 2 + 1] = cpu.adopt(V[i].data(), V[i].size() * sizeof(float));
            cpu.kv_write(0, &h, 1, {keep[i * 2].get(), 0}, {keep[i * 2 + 1].get(), 0});
            seq[i].commit();
            seq[i].prepare(nq[i]);
        }
        const backend::KVView views[2] = {seq[0].view(st.get()), seq[1].view(st.get())};
        std::vector<float> out((nq[0] + nq[1]) * qw, 0.0f);
        const auto ob = cpu.adopt(out.data(), out.size() * sizeof(float));
        if (mode == 0) {
            for (int i = 0; i < 2; ++i) {
                const size_t tail = past[i] * kvw;
                const auto qb = cpu.adopt(Q[i].data(), Q[i].size() * sizeof(float));
                cpu.kv_write(0, &views[i], 1, {keep[i * 2].get(), tail}, {keep[i * 2 + 1].get(), tail});
                cpu.attention({qb.get(), 0}, 0, &views[i], 1, {ob.get(), i ? nq[0] * qw : 0},
                              n_head, n_head_kv, head_dim);
            }
            separate = out;
        } else {
            std::vector<float> k, v, q;
            for (int i = 0; i < 2; ++i) {
                k.insert(k.end(), K[i].begin() + past[i] * kvw, K[i].end());
                v.insert(v.end(), V[i].begin() + past[i] * kvw, V[i].end());
                q.insert(q.end(), Q[i].begin(), Q[i].end());
            }
            const auto kb = cpu.adopt(k.data(), k.size() * sizeof(float));
            const auto vb = cpu.adopt(v.data(), v.size() * sizeof(float));
            const auto qb = cpu.adopt(q.data(), q.size() * sizeof(float));
            cpu.kv_write(0, views, 2, {kb.get(), 0}, {vb.get(), 0});
            cpu.attention({qb.get(), 0}, 0, views, 2, {ob.get(), 0}, n_head, n_head_kv, head_dim);
            joint = out;
        }
    }
    require(!separate.empty() && separate == joint,
            "two views in one call differ from the sequences taken separately");
}

// One layer and a context of four CPU blocks, so a step can cross a block boundary, for the transaction check below.
gguf::GGUFModel fixture() { return tiny_qwen(1, 4 * 128, true); }

// Fails the output projection (the only 16-row matmul) once, after every layer's KV has been written for the step.
struct FailingCpu : backend::CpuBackend {
    bool fail_output = false;
    int outputs = 0;
    int syncs = 0, submits = 0, waits = 0, reads = 0;
    backend::Ticket last_wait = 0;
    void matmul(uint32_t type, backend::CSlice data, backend::CSlice x, backend::Slice y,
                size_t nin, size_t nout, size_t nbatch, backend::RowRuns runs = {}) override {
        if (nout == 16) {
            ++outputs;
            if (fail_output) { fail_output = false; throw std::runtime_error("injected"); }
        }
        backend::CpuBackend::matmul(type, data, x, y, nin, nout, nbatch, runs);
    }
    void sync() noexcept override { ++syncs; backend::CpuBackend::sync(); }
    backend::Ticket submit() override { ++submits; return backend::CpuBackend::submit(); }
    void wait(backend::Ticket t) noexcept override {
        ++waits;
        last_wait = t;
        backend::CpuBackend::wait(t);
    }
    void read(const backend::Buffer& src, size_t off, void* dst, size_t bytes) override {
        ++reads;
        backend::CpuBackend::read(src, off, dst, bytes);
    }
};

// Every path handing blocks back to the pool must retire the backend's work first (docs/KV-CACHE.md): failed passes drain with sync(), reset() waits on the pass's ticket. Counting the calls keeps the contract from lapsing on the eager CPU backend.
void release_syncs() {
    const auto weights = fixture();
    auto cpu = std::make_shared<FailingCpu>();
    cpu->set_threads(1);
    infer::Model model(weights, cpu);
    model.set_ubatch(2);

    // A successful pass is one submission, waited on for its logits, which leave through host-visible memory rather than a read op.
    model.step(1);
    require(cpu->submits == 1 && cpu->waits == 1 && cpu->last_wait == 1,
            "a pass is one submission waited on once");
    require(cpu->reads == 0, "logits were copied out with a read op");

    int before = cpu->syncs;
    cpu->fail_output = true;
    rejects([&] { model.step(2); }, "injected decode failure did not propagate");
    require(cpu->syncs > before, "a failed step released blocks without retiring work");

    before = cpu->waits;
    const int syncs = cpu->syncs;
    model.reset();
    require(cpu->waits > before && cpu->last_wait == 1,
            "reset released blocks without waiting on the last pass");
    require(cpu->syncs == syncs, "reset drained instead of waiting on its ticket");

    before = cpu->syncs;
    cpu->fail_output = true;
    rejects([&] { model.prefill({4, 5, 6}); }, "injected prefill failure did not propagate");
    require(cpu->syncs > before, "a failed prefill released blocks without retiring work");

    // A prompt of three tokens at ubatch 2 is two passes: two submissions, one wait, for the last pass only.
    const int submits = cpu->submits, waits = cpu->waits;
    model.prefill({4, 5, 6});
    require(cpu->submits == submits + 2 && cpu->waits == waits + 1 &&
            cpu->last_wait == backend::Ticket(submits + 2),
            "a prompt submits once per pass and waits once, on the last");

    // A step that succeeds commits rather than releases, so it needs no sync of its own: its ticket is the one ordering point per pass.
    before = cpu->syncs;
    model.step(7);
    require(cpu->syncs == before, "a successful step retired work it did not have to");
    require(cpu->reads == 0, "a read op was used where a wait suffices");
}

// Two sequences in one pass through the model: one decoding a token over a three-token history while the other prefills two tokens.
// Each row must see its own position and its own history, so the logits of the joint pass match the two sequences run on their own, to float tolerance (the three-row matmul takes a different reduction path than one- and two-row ones).
// A sequence listed twice is refused with every history unchanged.
void batched_forward() {
    const auto weights = fixture();
    auto cpu = std::make_shared<backend::CpuBackend>();
    cpu->set_threads(1);
    infer::Model joint(weights, cpu), alone(weights, cpu);
    infer::Sequence a = joint.make_sequence(), b = joint.make_sequence();
    infer::Sequence a2 = alone.make_sequence(), b2 = alone.make_sequence();
    infer::ExecContext ctx, ctx2;
    const uint32_t hist[3] = {1, 2, 3}, ta[1] = {4}, tb[2] = {5, 6};
    const infer::BatchEntry h{&a, hist, 3, false};
    joint.forward(ctx, &h, 1);
    require(ctx.n_logits == 0 && a.length() == 3, "history pass produced logits");
    const infer::BatchEntry h2{&a2, hist, 3, false};
    alone.forward(ctx2, &h2, 1);

    const infer::BatchEntry both[2] = {{&a, ta, 1, true}, {&b, tb, 2, true}};
    joint.forward(ctx, both, 2);
    require(ctx.n_logits == 2 && ctx.width == 16 && a.length() == 4 && b.length() == 2,
            "joint pass did not commit both sequences");
    std::vector<float> la, lb;
    const infer::BatchEntry ea{&a2, ta, 1, true};
    alone.forward(ctx2, &ea, 1);
    la.assign(ctx2.logits(0), ctx2.logits(0) + 16);
    const infer::BatchEntry eb{&b2, tb, 2, true};
    alone.forward(ctx2, &eb, 1);
    lb.assign(ctx2.logits(0), ctx2.logits(0) + 16);
    for (size_t i = 0; i < 16; ++i) {
        require(std::fabs(ctx.logits(0)[i] - la[i]) <= 1e-5 * (1.0 + std::fabs(la[i])),
                "decode row of the joint pass differs from the sequence alone");
        require(std::fabs(ctx.logits(1)[i] - lb[i]) <= 1e-5 * (1.0 + std::fabs(lb[i])),
                "prefill row of the joint pass differs from the sequence alone");
    }
    // Rows in the other order must land in the other order.
    require(la != lb, "the two sequences produced the same logits");

    const infer::BatchEntry twice[2] = {{&a, ta, 1, true}, {&a, tb, 2, true}};
    rejects([&] { joint.forward(ctx, twice, 2); }, "a sequence listed twice was accepted");
    require(a.length() == 4 && b.length() == 2, "refused batch changed a history");
    rejects([&] { ctx.logits(2); }, "logits row beyond the pass was served");
    joint.reset(a);
    require(a.length() == 0 && b.length() == 2, "reset touched the other sequence");
}

// Through the model: a fork at a block boundary continues exactly as a fresh sequence fed the same tokens would, and the original continues exactly as if never forked.
// Every history is fed in the same passes, since a pass's width can change a CPU reduction.
void model_fork() {
    const auto weights = fixture();
    auto cpu = std::make_shared<backend::CpuBackend>();
    cpu->set_threads(1);
    infer::Model model(weights, cpu), fresh(weights, cpu);
    const size_t bt = cpu->kv_layout().block_tokens;
    std::vector<uint32_t> history(bt + 3);
    for (size_t i = 0; i < history.size(); ++i) history[i] = (uint32_t)(1 + i % 15);
    const uint32_t six = 6, seven = 7;
    auto run = [&](infer::Model& m, infer::Sequence& s, const uint32_t* ids, size_t n, bool want) {
        infer::ExecContext ctx;
        const infer::BatchEntry e{&s, ids, n, want};
        m.forward(ctx, &e, 1);
        return want ? std::vector<float>(ctx.logits(0), ctx.logits(0) + 16) : std::vector<float>();
    };
    infer::Sequence a = model.make_sequence();
    run(model, a, history.data(), bt, false);
    run(model, a, history.data() + bt, 3, false);
    infer::Sequence b = model.fork(a, bt);
    require(b.length() == bt, "fork length");
    const std::vector<float> la = run(model, a, &six, 1, true), lb = run(model, b, &seven, 1, true);

    infer::Sequence c = fresh.make_sequence(), d = fresh.make_sequence();
    run(fresh, c, history.data(), bt, false);
    run(fresh, c, history.data() + bt, 3, false);
    run(fresh, d, history.data(), bt, false);
    require(run(fresh, c, &six, 1, true) == la, "original after fork differs from an unforked history");
    require(run(fresh, d, &seven, 1, true) == lb, "fork differs from a fresh sequence fed the same history");
    require(la != lb, "the two continuations agree");

    rejects([&] { fresh.fork(a, bt); }, "fork of another model's sequence accepted");
    model.reset(b);
    model.reset(a);
    require(a.length() == 0 && b.length() == 0, "reset after fork");
}

// A failure after the KV writes must leave length, position and bytes as they were, and the retried step must produce the logits of an undisturbed model, exactly.
void model_transaction() {
    const auto weights = fixture();
    auto cpu = std::make_shared<FailingCpu>();
    auto plain = std::make_shared<backend::CpuBackend>();
    cpu->set_threads(1);
    plain->set_threads(1);
    infer::Model model(weights, cpu), control(weights, plain);
    model.set_ubatch(2);
    control.set_ubatch(2);

    // Decode path.
    require(model.step(1) == control.step(1), "first step differs");
    const size_t used = model.kv_used_bytes(), allocated = model.kv_allocated_bytes();
    cpu->fail_output = true;
    rejects([&] { model.step(2); }, "injected failure did not propagate");
    require(model.n_tokens() == 1 && model.kv_used_bytes() == used &&
            model.kv_allocated_bytes() == allocated, "failed step changed the history");
    require(model.step(2) == control.step(2), "retry after failure differs");
    require(model.step(3) == control.step(3), "step after retry differs");
    require(model.n_tokens() == 3, "position after retry");

    // Prefill path, failing in the last microbatch's output stage.
    model.reset();
    control.reset();
    cpu->fail_output = true;
    rejects([&] { model.prefill({4, 5, 6}); }, "injected prefill failure did not propagate");
    require(model.n_tokens() == 0 && model.kv_used_bytes() == 0, "failed prefill changed the history");
    require(model.prefill({4, 5, 6}) == control.prefill({4, 5, 6}), "prefill retry differs");
    require(model.step(7) == control.step(7), "step after prefill retry differs");
    require(model.n_tokens() == 4 && cpu->outputs == 7, "output projection count");

    // A failure on the step that opens a new block.
    // Policy: history and length are restored; capacity the backend grew for the attempt may be retained, bounded by the budget.
    const size_t bt = cpu->kv_layout().block_tokens;
    model.reset();
    control.reset();
    std::vector<uint32_t> fill(bt - 1);
    for (size_t i = 0; i < fill.size(); ++i) fill[i] = (uint32_t)(1 + i % 15);
    require(model.prefill(fill) == control.prefill(fill), "block fill differs");
    require(model.step(9) == control.step(9), "last token of the first block differs");
    const size_t before = model.kv_allocated_bytes();
    cpu->fail_output = true;
    rejects([&] { model.step(10); }, "injected block-crossing failure did not propagate");
    require(model.n_tokens() == (int)bt && model.kv_used_bytes() == bt * 1 * 2 * 1 * 4 * sizeof(float),
            "failed block-crossing step changed the history");
    require(model.kv_allocated_bytes() >= before, "retained capacity shrank");
    require(model.step(10) == control.step(10), "retry across the block boundary differs");
    require(model.step(11) == control.step(11), "step after block-crossing retry differs");
    require(model.n_tokens() == (int)bt + 2, "position after block-crossing retry");

    // The prefill activation arena fails to allocate; the retry must allocate it again and match the control exactly.
    model.reset();
    control.reset();
    model.set_ubatch(3);
    control.set_ubatch(3);
    {
        infer::Model fresh(weights, plain);
        fresh.set_ubatch(3);
        // The arena for ubatch 3 on this fixture is 1216 bytes; nothing else allocated during prefill comes close, so this selects it alone.
        fail_min_bytes = 1024;
        fail_large_after = 1;
        bool failed = false;
        try { fresh.prefill({1, 2, 3}); } catch (const std::bad_alloc&) { failed = true; }
        fail_min_bytes = 0;
        require(failed && fresh.n_tokens() == 0, "partial scratch failure did not propagate cleanly");
        require(fresh.prefill({1, 2, 3}) == control.prefill({1, 2, 3}), "retry after partial scratch differs");
        require(fresh.step(4) == control.step(4), "step after partial scratch retry differs");
    }
}

}  // namespace

int main() {
    try {
        pool_and_sequence();
        storage_growth_and_reset();
        attention_over_blocks();
        model_transaction();
        release_syncs();
        batched_views();
        batched_forward();
        fork_shares_blocks();
        model_fork();
        std::cout << "KV cache: pool, sequence, ownership, on-demand storage, reset, paged attention, "
                     "failed-step transactions and retire-before-release pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
