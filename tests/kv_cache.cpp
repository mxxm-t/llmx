// Paged KV cache oracle (docs/KV-CACHE.md): the logical pool and sequence, the CPU storage behind them and the growth every backend's storage shares, and attention over a shuffled block table.
// HF/model history checks remain separate; this does not replace them.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backends/cpu/cpu_backend.hpp"
#include "model/runtime.hpp"
#include "model/arch/registry.hpp"
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
#if defined(__GNUC__) && !defined(__clang__)
// GCC takes the free below for a mismatch with operator new, though the replaced new above allocates with malloc.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

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

// The growth rule the CPU and Vulkan storages each wrote before they shared one, written out here so a change to the shared rule fails: back through the block written, and at least double what is backed, up to the budget.
size_t rule_before_sharing(size_t backed, size_t id, size_t max_blocks) {
    return std::max(id + 1, std::min(max_blocks, backed * 2));
}

// Writes one row into block `id` of every layer through a one-block view, which is how a pass first reaches a block.
void write_block(backend::CpuBackend& cpu, backend::KVStorage& st, size_t layers, size_t row_floats, int32_t id) {
    std::vector<float> row(row_floats, 1.0f);
    const auto rowb = cpu.adopt(row.data(), row.size() * sizeof(float));
    const backend::KVView view{&st, &id, 1, 0, 1};
    for (size_t layer = 0; layer < layers; ++layer)
        cpu.kv_write(layer, &view, 1, {rowb.get(), 0}, {rowb.get(), 0});
}

// Every growth step and the peak against the rule above: the blocks backed after each write, the bytes retained, and the most held while a growth copies, which is the old blocks plus the new.
// Ids are written out of order and past twice what is backed, so each branch of the rule is taken, with the two sides of different types.
// Attention over a block inside the budget that no write backed is refused and backs nothing.
void growth_steps_and_peak() {
    for (size_t target = 0; target <= 40; ++target)
        for (size_t max_blocks = 1; max_blocks <= 40; ++max_blocks)
            for (size_t backed = 0; backed <= std::min(target, max_blocks); ++backed)
                if (target < max_blocks)
                    require(backend::BlockKVStorage::growth_target(backed, target, max_blocks) ==
                            rule_before_sharing(backed, target, max_blocks),
                            "the growth target differs from the rule before sharing");

    backend::CpuBackend cpu;
    cpu.set_threads(1);
    const size_t bt = cpu.kv_layout().block_tokens, layers = 2, heads = 2, width = 8;
    const size_t max_blocks = 11;
    // One block of every layer, K in f16 and V in f32.
    const size_t block = layers * heads * bt * width * (2 + 4);
    struct Case {
        std::vector<int32_t> ids;
        std::vector<size_t> steps;
    };
    const Case cases[] = {
        {{0, 1, 2, 3, 4, 8, 9, 10}, {1, 2, 4, 4, 8, 11, 11, 11}},
        {{6, 7, 2, 7, 10}, {7, 11, 11, 11, 11}},
        {{1, 0, 5, 4}, {2, 2, 6, 6}},
    };
    for (const Case& c : cases) {
        auto st = cpu.kv_alloc(layers, heads, width, max_blocks * bt, backend::KVType::f16, backend::KVType::f32);
        require(st->max_blocks() == max_blocks, "budget in blocks");
        size_t backed = 0, peak = 0;
        for (size_t i = 0; i < c.ids.size(); ++i) {
            const size_t id = (size_t)c.ids[i];
            if (id >= backed) {
                const size_t grown = rule_before_sharing(backed, id, max_blocks);
                peak = std::max(peak, (backed + grown) * block);
                backed = grown;
            }
            write_block(cpu, *st, layers, heads * width, c.ids[i]);
            require(backed == c.steps[i], "the rule before sharing gives other steps than the test lists");
            require(st->allocated_bytes() == backed * block, "growth step differs from the rule before sharing");
            require(st->peak_bytes() == peak, "growth peak differs from old plus new under the rule before sharing");
        }
        rejects([&] { write_block(cpu, *st, layers, heads * width, (int32_t)max_blocks); },
                "a block past the budget accepted");
        require(st->allocated_bytes() == backed * block && st->peak_bytes() == peak,
                "a refused block changed the accounting");
        if (backed < max_blocks) {
            const int32_t unwritten = (int32_t)max_blocks - 1;
            const backend::KVView view{st.get(), &unwritten, 1, 0, 1};
            const auto q = cpu.alloc(heads * width * sizeof(float), backend::Memory::device);
            const auto out = cpu.alloc(heads * width * sizeof(float), backend::Memory::device);
            std::string error;
            try { cpu.attention({q.get(), 0}, 0, &view, 1, {out.get(), 0}, (int)heads, (int)heads, (int)width); }
            catch (const std::runtime_error& e) { error = e.what(); }
            require(error == "backend: attention over unwritten KV blocks", "attention over an unwritten block accepted");
            require(st->allocated_bytes() == backed * block && st->peak_bytes() == peak,
                    "a refused read changed the accounting");
        }
    }

    // The whole budget is checked in bytes at the types before anything is backed.
    rejects([&] { cpu.kv_alloc(1, 1, 1, std::numeric_limits<size_t>::max()); }, "an overflowing budget accepted");
    rejects([&] {
        cpu.kv_alloc(1, 1, 1, std::numeric_limits<size_t>::max(), backend::KVType::f16, backend::KVType::f16);
    }, "an overflowing f16 budget accepted");
    auto large = cpu.kv_alloc(1, 1, 1, std::numeric_limits<size_t>::max() / 1024);
    require(large->allocated_bytes() == 0, "a large budget in range was backed on allocation");
}

// A backend whose allocation fails on request and whose syncs are counted.
struct FailingAlloc : backend::CpuBackend {
    int allocs = 0, fail_at = 0, syncs = 0;
    backend::BufferPtr alloc(size_t bytes, backend::Memory where) override {
        if (++allocs == fail_at) throw std::runtime_error("injected allocation failure");
        return backend::CpuBackend::alloc(bytes, where);
    }
    void sync() noexcept override { ++syncs; }
};

// A storage that records its hooks, standing in for a backend whose copies run after a growth returns.
struct HookedStorage final : backend::BlockKVStorage {
    std::vector<const backend::Buffer*> retired;
    std::vector<const backend::Buffer*> seen;
    HookedStorage(backend::Backend& owner, size_t max_tokens)
        : BlockKVStorage(owner, "test", 4, 2, 1, 2, max_tokens, backend::KVType::f32, backend::KVType::f16) {}
    void retire(const backend::BufferPtr& old) override { retired.push_back(old.get()); }
    void after_growth(const std::vector<backend::BufferPtr>& keys, const std::vector<backend::BufferPtr>& values) override {
        seen.clear();
        for (size_t l = 0; l < keys.size(); ++l) {
            seen.push_back(keys[l].get());
            seen.push_back(values[l].get());
        }
    }
    std::vector<const backend::Buffer*> held() const {
        std::vector<const backend::Buffer*> out;
        for (size_t l = 0; l < layers(); ++l) {
            out.push_back(k_buffer(l).get());
            out.push_back(v_buffer(l).get());
        }
        return out;
    }
};

// A growth hands every old buffer to `retire` once and shows the new ones to `after_growth` before publishing them.
// A growth that fails drains the backend and leaves buffers, accounting and hooks as they were, and the retry grows.
// Errors carry the storage's prefix.
void growth_hooks() {
    FailingAlloc cpu;
    HookedStorage st(cpu, 4 * 8);
    require(st.max_blocks() == 8 && st.block_tokens() == 4, "hooked storage shape");
    st.ensure(0);
    require(st.retired.empty() && st.seen == st.held(), "first growth retired a buffer or hid the new ones");
    const std::vector<const backend::Buffer*> first = st.held();
    st.ensure(1);
    require(st.retired == first && st.seen == st.held(), "growth did not retire every old buffer once");
    const std::vector<const backend::Buffer*> second = st.held();
    const size_t allocated = st.allocated_bytes(), peak = st.peak_bytes();
    const int syncs = cpu.syncs;
    cpu.fail_at = cpu.allocs + 3;
    rejects([&] { st.ensure(2); }, "injected allocation failure did not propagate");
    require(cpu.syncs == syncs + 1, "a failed growth did not drain the backend");
    require(st.held() == second && st.retired == first && st.seen == second,
            "a failed growth changed the buffers or ran a hook");
    require(st.allocated_bytes() == allocated && st.peak_bytes() == peak && !st.backed(2),
            "a failed growth changed the accounting");
    st.ensure(2);
    require(st.backed(3) && !st.backed(4) && st.retired.size() == 2 * first.size(), "retry after a failed growth");
    require(cpu.syncs == syncs + 1, "a successful growth drained the backend");
    bool prefixed = false;
    try { st.ensure(8); }
    catch (const std::runtime_error& e) { prefixed = std::string(e.what()) == "test: KV block outside the budget"; }
    require(prefixed, "a storage error without its prefix");
}

// A fork at a whole-block length shares every block below it, read-only, and allocates and copies nothing, so the two histories agree up to the fork and then diverge without touching each other.
// A length inside a block or past the history is refused.
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

    infer::KVSequence fork = seq.fork(bt);
    require(fork.length() == bt && fork.n_blocks() == 1, "fork length or table");
    require(pool.in_use() == before, "a fork allocated a block");
    const backend::KVView vs = seq.view(st.get()), vf = fork.view(st.get());
    require(vf.blocks[0] == vs.blocks[0] && pool.refs(vs.blocks[0]) == 2 && pool.refs(vs.blocks[1]) == 1 &&
            pool.refs(vs.blocks[2]) == 1, "shared blocks miscounted");
    check(*st, fork, bt, heads, width, 1);
    rejects([&] { seq.fork(bt + 2); }, "fork inside a block accepted");
    rejects([&] { seq.fork(3 * bt); }, "fork past the history accepted");
    require(pool.in_use() == before && pool.refs(vs.blocks[0]) == 2, "refused fork changed the pool");

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

// Every path handing blocks back to the pool must retire the backend's work first (docs/KV-CACHE.md): failed passes drain with sync(), reset() waits on the pass's ticket. Counting the calls keeps the contract from lapsing on the eager CPU backend.
void release_syncs() {
    const auto weights = fixture();
    auto cpu = std::make_shared<FailingCpu>();
    cpu->set_threads(1);
    infer::Model model(infer::gguf_weights(weights), cpu);
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
    infer::Model joint(infer::gguf_weights(weights), cpu), alone(infer::gguf_weights(weights), cpu);
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
    infer::Model model(infer::gguf_weights(weights), cpu), fresh(infer::gguf_weights(weights), cpu);
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
    rejects([&] { model.fork(a, bt + 1); }, "fork inside a block accepted");
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

// A history copied to host memory and back (Model::save_host, Model::restore_host, docs/SPECULATIVE.md, section 2, Host tier): two blocks of a history saved, the history reset and its blocks taken by another, then restored into other physical blocks, continue as the history never copied does, bit for bit.
// A copy inside a block or past the history and a restore into another model are refused, the refused restore holding nothing.
void host_round_trip() {
    const auto weights = fixture();
    auto cpu = std::make_shared<backend::CpuBackend>();
    cpu->set_threads(1);
    infer::Model model(infer::gguf_weights(weights), cpu), fresh(infer::gguf_weights(weights), cpu);
    const size_t bt = cpu->kv_layout().block_tokens;
    std::vector<uint32_t> history(2 * bt + 5);
    for (size_t i = 0; i < history.size(); ++i) history[i] = (uint32_t)(1 + (i * 7) % 15);
    const uint32_t six = 6;
    auto run = [&](infer::Model& m, infer::Sequence& s, const uint32_t* ids, size_t n, bool want) {
        infer::ExecContext ctx;
        const infer::BatchEntry e{&s, ids, n, want};
        m.forward(ctx, &e, 1);
        return want ? std::vector<float>(ctx.logits(0), ctx.logits(0) + 16) : std::vector<float>();
    };
    infer::Sequence ref = fresh.make_sequence();
    run(fresh, ref, history.data(), 2 * bt, false);
    run(fresh, ref, history.data() + 2 * bt, 5, false);
    const std::vector<float> want = run(fresh, ref, &six, 1, true);

    infer::Sequence a = model.make_sequence();
    run(model, a, history.data(), 2 * bt, false);
    run(model, a, history.data() + 2 * bt, 5, false);
    infer::HostHistory h;
    const size_t any = std::numeric_limits<size_t>::max();
    rejects([&] { model.save_host(a, bt + 1, h, any); }, "a copy to host memory inside a block accepted");
    rejects([&] { model.save_host(a, 3 * bt, h, any); }, "a copy to host memory past the history accepted");
    rejects([&] { model.save_host(a, 2 * bt, h, 0); }, "a copy to host memory past its limit accepted");
    require(model.host_allocated() == 0, "a refused copy to host memory kept a slab");
    model.save_host(a, 2 * bt, h, any);
    require(h.length == 2 * bt && h.held == model.host_bytes(2 * bt) && h.bytes > 0 && h.bytes <= h.held, "a copy to host memory holds other bytes");
    model.reset(a);
    infer::Sequence other = model.make_sequence();
    run(model, other, history.data(), bt, false);
    rejects([&] { fresh.restore_host(h); }, "a history copied by another model restored");
    require(!model.caches_on_devices(), "a model on the CPU alone counted a cache on a device");
    infer::Sequence r = model.restore_host(h);
    require(r.length() == 2 * bt, "a restored history's length");
    run(model, r, history.data() + 2 * bt, 5, false);
    require(run(model, r, &six, 1, true) == want, "a history restored from host memory differs from one never copied");
    const size_t slabs = model.host_allocated();
    model.release_host(h);
    require(h.bytes == 0 && h.slabs.empty(), "a released host history holds memory");
    // A second copy takes the slabs the first left, allocating none.
    model.save_host(r, 2 * bt, h, slabs);
    require(model.host_allocated() == slabs, "a second copy to host memory allocated past the slabs the first left");
    model.release_host(h);
    model.reset(r);
    model.reset(other);
    fresh.reset(ref);
}

// What a history copied to host memory holds and how (Model::host_identity, docs/DISK-TIER.md, The entry file): two models of one file on two CPU backends give one identity, and each of another V cache type, another activation dtype, a backend of another identity, a backend of other row classes and a split over two backends gives another; the identity names each device's backend, its runs and the row classes.
// A copy's runs (HostHistory::device_bytes) add up to its bytes, one a device.
struct OtherCpu : backend::CpuBackend {
    std::string identity() const override { return "another cpu"; }
};
struct OtherClasses : backend::CpuBackend {
    size_t row_class(size_t extent) const override { return extent <= 64 ? 1 : 2; }
};
void host_identity() {
    const auto weights = tiny_qwen(2, 512, true);
    const infer::ModelWeights w = infer::gguf_weights(weights);
    const auto cpu = [] {
        auto c = std::make_shared<backend::CpuBackend>();
        c->set_threads(1);
        return c;
    };
    const auto identity = [&](backend::BackendPtr b, infer::ModelOptions o = {}) { return infer::Model(w, std::move(b), o).host_identity(); };
    const std::string base = identity(cpu());
    require(base == identity(cpu()), "two models of one file on two CPU backends gave two identities");
    require(base.find("device 0 cpu; dtype f16") != std::string::npos && base.find("kv f16 f16") != std::string::npos && base.find("classes 1:1,0 2:2,0") != std::string::npos &&
                base.find("tokens a block") != std::string::npos,
            "the identity does not name the device, its dtype, the cache types, its runs and the row classes");
    infer::ModelOptions f32_v;
    f32_v.kv_v = backend::KVType::f32;
    require(identity(cpu(), f32_v) != base, "another V cache type gave the same identity");
    infer::ModelOptions f32_dtype;
    f32_dtype.dtype = backend::Dtype::f32;
    require(identity(cpu(), f32_dtype) != base, "another activation dtype gave the same identity");
    require(identity(std::make_shared<OtherCpu>()) != base, "a backend of another identity gave the same identity");
    require(identity(std::make_shared<OtherClasses>()) != base, "a backend of other row classes gave the same identity");
    infer::Placement split;
    split.mixer_device = {0, 1};
    split.ffn_device = {0, 1};
    split.output_device = 1;
    const std::string two = infer::Model(w, {cpu(), cpu()}, split).host_identity();
    require(two != base && two.find("device 1 cpu") != std::string::npos, "a split over two backends gave the identity of one");
    infer::Model model(w, cpu());
    infer::Sequence a = model.make_sequence();
    std::vector<uint32_t> ids(300);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (uint32_t)(1 + i % 15);
    infer::ExecContext ctx;
    const infer::BatchEntry e{&a, ids.data(), ids.size(), false};
    model.forward(ctx, &e, 1);
    infer::HostHistory h;
    model.save_host(a, 256, h, std::numeric_limits<size_t>::max());
    require(h.device_bytes.size() == 1 && h.device_bytes[0] == h.bytes, "a copy's runs do not add up to its bytes");
    model.release_host(h);
    model.reset(a);
}

// Which idle slabs a copy to host memory frees before it allocates (infer::detail::slabs_to_free), so the slabs alive stay within the limit across devices: none where the idle ones cover every need; on a split whose two devices each hold two idle slabs, under a limit of four, a copy needing three on the first and one on the second frees the second's spare one before allocating the first's third (the review's case, which kept five alive); and a copy whose shortfall the spare slabs cannot cover is refused.
void host_slab_limit() {
    const auto drop = [](std::vector<size_t> need, std::vector<size_t> idle, size_t alive, size_t limit) {
        return infer::detail::slabs_to_free(need, idle, alive, limit);
    };
    require(drop({1, 1}, {2, 3}, 9, 4) == std::optional<std::vector<size_t>>(std::vector<size_t>{0, 0}), "idle slabs covering every need were freed");
    require(drop({3, 1}, {2, 2}, 4, 4) == std::optional<std::vector<size_t>>(std::vector<size_t>{0, 1}), "the spare slab of the other device was not freed");
    require(drop({3, 1}, {2, 2}, 4, 5) == std::optional<std::vector<size_t>>(std::vector<size_t>{0, 0}), "a slab was freed within the limit");
    require(drop({4, 1}, {0, 3}, 5, 7) == std::optional<std::vector<size_t>>(std::vector<size_t>{0, 2}), "the second device's spare slabs were not freed");
    require(!drop({3, 1}, {0, 0}, 2, 4), "a copy past the limit was not refused");
    // New slabs leave the host the reserve the fit keeps, a twentieth of what it has free: 1900 bytes of 2000 do, one more does not, and an unknown figure refuses nothing.
    require(infer::detail::host_room(2000, 1900) && !infer::detail::host_room(2000, 1901) && !infer::detail::host_room(2000, 2001),
            "the host's reserve was not kept at its edge");
    require(infer::detail::host_room(std::nullopt, (size_t)1 << 40), "an unknown free figure refused a copy");
}

// What a paused request's resume relies on (docs/SERVER.md, pausing): a history recomputed in the classes that first computed it gives the logits it gave, bit for bit.
// The reference is a 40-token prompt at its extent, then 199 greedy tokens each decoded in a pass of its own; the synthetic Q8_0 model's decode rows take the 8-bit dots and its prompt rows the float path, so a class taken wrongly shows.
// The replays: the prompt at its extent in slices of 16, then the 199 tokens as entries of extent 1 of up to 64 rows, logits only on the last; a fork at the first block of the reference history replaying the rest; and the replay beside another sequence's decode row and a third's prompt slice.
void replay_by_class() {
    const gguf::GGUFModel weights = infer::synthetic_model({2, 64, 128, 4, 2, 16, 64, 7u});
    auto cpu = std::make_shared<backend::CpuBackend>();
    cpu->set_threads(2);
    infer::Model model(infer::gguf_weights(weights), cpu);
    const size_t prompt = 40, steps = 199, bt = cpu->kv_layout().block_tokens, V = model.n_vocab();
    std::vector<uint32_t> ids(prompt);
    for (size_t i = 0; i < prompt; ++i) ids[i] = (uint32_t)((i * 11 + 3) % V);
    infer::ExecContext ctx;
    auto greedy = [&](const float* row) {
        uint32_t best = 0;
        for (uint32_t v = 1; v < V; ++v)
            if (row[v] > row[best]) best = v;
        return best;
    };
    infer::Sequence ref = model.make_sequence();
    infer::BatchEntry first{&ref, ids.data(), prompt, true};
    first.extent = prompt;
    model.forward(ctx, &first, 1);
    std::vector<float> want;
    for (size_t s = 0; s < steps; ++s) {
        ids.push_back(greedy(ctx.logits(0)));
        const infer::BatchEntry step{&ref, &ids.back(), 1, true};
        model.forward(ctx, &step, 1);
    }
    want.assign(ctx.logits(0), ctx.logits(0) + V);
    require(ref.length() == prompt + steps, "the reference history");

    // The rows `seq` lacks, by class, `others` sharing each pass; the logits after the last.
    auto replay = [&](infer::Sequence& seq, const std::function<void(std::vector<infer::BatchEntry>&)>& others) {
        std::vector<float> got;
        while (seq.length() < ids.size()) {
            const size_t at = seq.length();
            std::vector<infer::BatchEntry> pass;
            others(pass);
            const bool generated = at >= prompt;
            const size_t n = generated ? std::min<size_t>(64, ids.size() - at) : std::min<size_t>(16, prompt - at);
            infer::BatchEntry e{&seq, ids.data() + at, n, at + n == ids.size()};
            e.extent = generated ? 1 : prompt;
            pass.push_back(e);
            model.forward(ctx, pass.data(), pass.size());
            if (e.want_logits) got.assign(ctx.logits(ctx.n_logits - 1), ctx.logits(ctx.n_logits - 1) + V);
        }
        return got;
    };
    auto exact_logits = [&](const std::vector<float>& got, const char* what) {
        require(got.size() == want.size() && !std::memcmp(got.data(), want.data(), V * sizeof(float)), what);
    };
    infer::Sequence alone = model.make_sequence();
    exact_logits(replay(alone, [](std::vector<infer::BatchEntry>&) {}), "a history replayed by class differs from its decode");
    model.reset(alone);

    infer::Sequence forked = model.fork(ref, bt);
    exact_logits(replay(forked, [](std::vector<infer::BatchEntry>&) {}), "a fork at a block replaying the rest by class differs from the decode");
    model.reset(forked);

    // Beside it, a sequence decoding its own tokens and a third reading a 50-token prompt in slices of 7.
    infer::Sequence mixed = model.make_sequence(), decoding = model.make_sequence(), reading = model.make_sequence();
    const uint32_t start[3] = {5, 6, 7};
    const infer::BatchEntry warm{&decoding, start, 3, false};
    model.forward(ctx, &warm, 1);
    std::vector<uint32_t> other(50);
    for (size_t i = 0; i < other.size(); ++i) other[i] = (uint32_t)((i * 5 + 1) % V);
    uint32_t next = 9;
    exact_logits(replay(mixed, [&](std::vector<infer::BatchEntry>& pass) {
        next = (next * 7 + 3) % (uint32_t)V;
        pass.push_back(infer::BatchEntry{&decoding, &next, 1, true});
        if (reading.length() < other.size()) {
            const size_t at = reading.length(), n = std::min<size_t>(7, other.size() - at);
            infer::BatchEntry slice{&reading, other.data() + at, n, at + n == other.size()};
            slice.extent = other.size();
            pass.push_back(slice);
        }
    }), "a history replayed by class beside other sequences differs from its decode");
    model.reset(mixed);
    model.reset(decoding);
    model.reset(reading);
    model.reset(ref);
}

// A failure after the KV writes must leave length, position and bytes as they were, and the retried step must produce the logits of an undisturbed model, exactly.
void model_transaction() {
    const auto weights = fixture();
    auto cpu = std::make_shared<FailingCpu>();
    auto plain = std::make_shared<backend::CpuBackend>();
    cpu->set_threads(1);
    plain->set_threads(1);
    // The history's bytes are counted below as f32 sides.
    infer::ModelOptions f32;
    f32.kv_k = f32.kv_v = backend::KVType::f32;
    infer::Model model(infer::gguf_weights(weights), cpu, f32), control(infer::gguf_weights(weights), plain, f32);
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
        infer::Model fresh(infer::gguf_weights(weights), plain, f32);
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
        growth_steps_and_peak();
        growth_hooks();
        attention_over_blocks();
        model_transaction();
        release_syncs();
        batched_views();
        batched_forward();
        fork_shares_blocks();
        model_fork();
        host_round_trip();
        host_identity();
        host_slab_limit();
        replay_by_class();
        std::cout << "KV cache: pool, sequence, ownership, on-demand storage, growth steps and peak, growth hooks, "
                     "reset, paged attention, failed-step transactions, retire-before-release and replay by class pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
