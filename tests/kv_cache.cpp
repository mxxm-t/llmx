// Paged KV cache oracle (docs/KV-CACHE.md): the logical pool and sequence,
// the CPU storage behind them, and attention over a shuffled block table.
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

// Fails the Nth allocation of at least `min_bytes` after arming, once. Used
// to break the batch-scratch allocation part way through.
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
        cpu.kv_write(layer, view, pos, {kb.get(), 0}, {vb.get(), 0}, batch);
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
            const backend::KVView view = seq.view(st.get());
            std::vector<float> row(heads * width, 0.0f);
            const auto rowb = cpu.adopt(row.data(), row.size() * sizeof(float));
            rejects([&] { cpu.kv_write(3, view, 0, {rowb.get(), 0}, {rowb.get(), 0}, 1); },
                    "layer outside storage accepted");
            rejects([&] { cpu.kv_write(0, view, limit, {rowb.get(), 0}, {rowb.get(), 0}, 1); },
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
                cpu.kv_write(0, seq.view(st.get()), 0, {Kb.get(), 0}, {Vb.get(), 0}, n_past);
                seq.commit();
                seq.prepare((size_t)nbatch);
                const backend::KVView view = seq.view(st.get());
                const size_t tail = n_past * (size_t)n_head_kv * (size_t)head_dim;
                cpu.kv_write(0, view, n_past, {Kb.get(), tail}, {Vb.get(), tail}, (size_t)nbatch);
                std::vector<float> out(Q.size(), 0.0f);
                const auto ob = cpu.adopt(out.data(), out.size() * sizeof(float));
                cpu.attention({Qb.get(), 0}, 0, view, {ob.get(), 0}, n_head, n_head_kv, head_dim, nbatch);
                if (n_past == 0 && churn == 0) {
                    infer::KVSequence shorter(&pool, bt);
                    rejects([&] {
                        cpu.attention({Qb.get(), 0}, 0, shorter.view(st.get()), {ob.get(), 0},
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

// A one-layer Qwen3-shaped F32 model, deterministic weights, for the
// transaction check below. Mirrors the prefill-scope fixture, with a context
// of four CPU blocks so a step can cross a block boundary.
gguf::GGUFModel fixture() {
    gguf::GGUFModel m;
    for (const auto& kv : std::vector<std::pair<std::string, uint64_t>>{
            {"block_count", 1}, {"embedding_length", 8}, {"feed_forward_length", 12},
            {"attention.head_count", 2}, {"attention.head_count_kv", 1},
            {"attention.key_length", 4}, {"context_length", 4 * 128}}) {
        gguf::MetaValue v; v.vtype = gguf::V_UINT32; v.u = kv.second;
        m.kv.push_back({"qwen3." + kv.first, v});
    }
    auto add = [&](const std::string& name, std::vector<uint64_t> shape, bool norm = false) {
        size_t count = 1;
        for (uint64_t d : shape) count *= size_t(d);
        const size_t offset = m.blob.size();
        m.blob.resize(offset + count * sizeof(float));
        for (size_t i = 0; i < count; ++i) {
            const float v = norm ? 1.0f : float(int((i * 17 + m.tensors.size() * 3) % 29) - 14) / 64.0f;
            std::memcpy(m.blob.data() + offset + i * sizeof(float), &v, sizeof(v));
        }
        m.tensors.push_back({name, std::move(shape), gguf::GGML_TYPE_F32, 0});
        m.offsets.push_back(offset);
    };
    add("token_embd.weight", {8, 16});
    add("output_norm.weight", {8}, true);
    for (const char* name : {"attn_norm", "ffn_norm"})
        add(std::string("blk.0.") + name + ".weight", {8}, true);
    for (const char* name : {"attn_q_norm", "attn_k_norm"})
        add(std::string("blk.0.") + name + ".weight", {4}, true);
    add("blk.0.attn_q.weight", {8, 8});
    add("blk.0.attn_k.weight", {8, 4});
    add("blk.0.attn_v.weight", {8, 4});
    add("blk.0.attn_output.weight", {8, 8});
    add("blk.0.ffn_gate.weight", {8, 12});
    add("blk.0.ffn_up.weight", {8, 12});
    add("blk.0.ffn_down.weight", {12, 8});
    return m;
}

// Fails the output projection (the only 16-row matmul) once, after every
// layer's KV has been written for the step.
struct FailingCpu : backend::CpuBackend {
    bool fail_output = false;
    int outputs = 0;
    void matmul(uint32_t type, backend::CSlice data, backend::CSlice x, backend::Slice y,
                size_t nin, size_t nout, size_t nbatch) override {
        if (nout == 16) {
            ++outputs;
            if (fail_output) { fail_output = false; throw std::runtime_error("injected"); }
        }
        backend::CpuBackend::matmul(type, data, x, y, nin, nout, nbatch);
    }
};

// A failure after the KV writes must leave length, position and bytes as
// they were, and the retried step must produce the logits of an undisturbed
// model, exactly.
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

    // A failure on the step that opens a new block. Policy: history and
    // length are restored; capacity the backend grew for the attempt may be
    // retained, bounded by the budget.
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

    // Partial scratch allocation: the second large batch buffer fails, the
    // retry must allocate the whole set and match the control exactly.
    model.reset();
    control.reset();
    model.set_ubatch(3);
    control.set_ubatch(3);
    {
        infer::Model fresh(weights, plain);
        fresh.set_ubatch(3);
        fail_min_bytes = 3 * 8 * sizeof(float);
        fail_large_after = 2;
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
        std::cout << "KV cache: pool, sequence, ownership, on-demand storage, reset, paged attention "
                     "and failed-step transactions pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
