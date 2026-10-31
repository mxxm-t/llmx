// backend-rocm: the ROCm backend's buffers, tickets, refusals and dense F32 ops against the CPU backend, and each op's batch invariance; exits 77 without a runtime or a device.
#include "backends/cpu/cpu_backend.hpp"
#include "backends/kv_storage.hpp"
#include "backends/rocm/rocm_backend.hpp"
#include "backends/rocm/rocm_runtime.hpp"
#include "core/fp16.hpp"
#include "model/kv_cache.hpp"
#include "quant/types.hpp"
#include "row_classes.hpp"

#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace backend;

namespace {

int failures = 0;
void require(bool ok, const std::string& what) {
    if (!ok) {
        std::cout << "FAIL " << what << "\n";
        ++failures;
    }
}
template <class F> bool throws(F f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}
std::vector<float> read(Backend& b, const Buffer& buf, size_t n, size_t off = 0) {
    std::vector<float> v(n);
    b.read(buf, off * sizeof(float), v.data(), n * sizeof(float));
    return v;
}
// Every value within 1e-4 * (1 + |reference|) of the CPU's, the bound the device tests hold F32 ops to.
void close(const std::vector<float>& got, const std::vector<float>& want, const std::string& what) {
    bool ok = got.size() == want.size();
    for (size_t i = 0; ok && i < got.size(); ++i) ok = std::fabs(got[i] - want[i]) <= 1e-4f * (1.0f + std::fabs(want[i]));
    require(ok, what);
}
bool same(const std::vector<float>& a, const std::vector<float>& b) { return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0; }

void buffers(Backend& g) {
    const std::vector<float> x = row_classes::values(1000, 1);
    auto d = g.alloc(4000), h = g.alloc(4000, Memory::host_visible);
    require(read(g, *d, 1000) == std::vector<float>(1000, 0.0f), "device memory reads as zeros");
    require(read(g, *h, 1000) == std::vector<float>(1000, 0.0f), "host-visible memory reads as zeros");
    g.write(*d, 40, x.data() + 10, 400);
    g.copy(*h, 0, *d, 0, 4000);
    g.sync();
    std::vector<float> want(1000, 0.0f);
    std::copy(x.begin() + 10, x.begin() + 110, want.begin() + 10);
    require(read(g, *d, 1000) == want, "a write at an offset reads back");
    require(std::memcmp(h->host_ptr(), want.data(), 4000) == 0, "host-visible memory holds a copy in place after sync");
    auto a = g.adopt(x.data(), 4000);
    require(read(g, *a, 1000) == x, "an adopted buffer holds its bytes");
    // A buffer dropped right after the work that reads it: the work is ordered on the stream before the free.
    {
        auto tmp = g.adopt(x.data(), 4000);
        g.copy(*d, 0, *tmp, 0, 4000);
    }
    require(read(g, *d, 1000) == x, "a buffer dropped after its copy was enqueued gave its bytes");
    // A large buffer dropped while many copies out of it are still queued: hipFree unmaps it at once, so its memory must wait for them.
    {
        const size_t n = size_t(256) << 20;
        auto dst = g.alloc(n);
        {
            auto src = g.alloc(n);
            g.write(*src, 0, x.data(), 4000);
            for (int i = 0; i < 64; ++i) g.copy(*dst, 0, *src, 0, n);
        }
        require(read(g, *dst, 1000) == x, "a large buffer dropped with copies out of it queued gave its bytes");
    }
    require(throws([&] { g.read(*d, 3999, nullptr, 2); }), "a read past the buffer is refused");
    require(throws([&] { g.write(*d, 0, x.data(), 4001); }), "a write past the buffer is refused");
    auto big = [&] { g.alloc(size_t(1) << 50); };
    require(throws(big), "an allocation past the device's memory is refused");
    const Ticket t1 = g.submit(), t2 = g.submit();
    require(t2 == t1 + 1, "tickets increase by one");
    g.wait(t2);
    require(g.done(t1) && g.done(t2), "waited tickets read as done");
    // More tickets than events, so slots are reused.
    for (int i = 0; i < 2100; ++i) g.submit();
    g.sync();
    require(g.done(t2 + 2100), "tickets past the event ring retire");
    // Another thread waits on each ticket while this one keeps launching and submitting, as a scheduler waits on a pass while the next is recorded.
    {
        auto buf = g.alloc(1 << 20);
        std::atomic<Ticket> handed{0};
        std::atomic<bool> stop{false}, ok{true};
        std::thread waiter([&] {
            Ticket seen = 0;
            while (!stop || seen < handed) {
                const Ticket t = handed;
                if (t == seen) continue;
                g.wait(t);
                if (!g.done(t)) ok = false;
                seen = t;
            }
        });
        for (int i = 0; i < 3000; ++i) {
            g.copy(*buf, 0, *buf, 1 << 19, 1 << 19);
            handed = g.submit();
        }
        stop = true;
        waiter.join();
        g.sync();
        require(ok && g.done(handed), "tickets waited on from another thread while this one submits read as done");
    }
}

void refusals(Backend& g) {
    std::vector<uint8_t> q4k(144 * 4, 0);
    auto w = g.adopt(q4k.data(), q4k.size());
    auto x = g.alloc(256 * 4), y = g.alloc(16);
    require(g.supports_type(quant::GGML_TYPE_F32) && g.supports_type(quant::GGML_TYPE_Q8_0) && !g.supports_type(quant::GGML_TYPE_Q4_K),
            "the backend supports F32 and Q8_0 weights only");
    require(throws([&] { g.matmul({quant::GGML_TYPE_Q4_K, {w.get(), 0}}, {x.get(), 0}, {y.get(), 0}, 256, 4, 1); }), "a Q4_K matmul is refused");
    require(throws([&] { g.matmul({quant::GGML_TYPE_Q8_0, {w.get(), 0}}, {x.get(), 0}, {y.get(), 0}, 48, 4, 1); }), "a Q8_0 row ending inside a block is refused");
    require(!g.implements(Op::gated_delta_rule) && g.implements(Op::norm_rope_partial), "the backend implements norm_rope_partial and no other named op");
    require(throws([&] { g.gated_rms_norm({y.get(), 0}, {x.get(), 0}, {x.get(), 0}, {x.get(), 0}, 1, 1, 4, 1e-6f); }),
            "an op the backend lacks throws");
}

// A Q8_0 matrix of `rows` rows `nin` wide: each block's scale and values from the seed, every value in range.
std::vector<uint8_t> q8_matrix(size_t nin, size_t rows, uint32_t seed) {
    std::vector<uint8_t> m;
    const std::vector<float> scale = row_classes::values(rows * nin / 32, seed, 0.05f), v = row_classes::values(rows * nin, seed + 1, 127.0f);
    for (size_t b = 0; b < rows * nin / 32; ++b) {
        const uint16_t d = f32_to_f16(scale[b]);
        m.push_back(uint8_t(d));
        m.push_back(uint8_t(d >> 8));
        for (size_t j = 0; j < 32; ++j) m.push_back(uint8_t(int8_t(std::max(-127.0f, std::min(127.0f, std::round(v[b * 32 + j]))))));
    }
    return m;
}

// A product into pinned host memory, as the logits are, gives the bits of the same product into device memory, both written and added, for a generated row and a prompt's rows.
void host_output(Backend& g) {
    const size_t nin = 256, nout = 300;
    const std::vector<uint8_t> w = q8_matrix(nin, nout, 51);
    const std::vector<float> x = row_classes::values(nin * 3, 52), y0 = row_classes::values(nout * 3, 53);
    auto wb = g.adopt(w.data(), w.size()), xb = g.adopt(x.data(), x.size() * 4);
    const WeightView wv{quant::GGML_TYPE_Q8_0, {wb.get(), 0}};
    for (size_t rows : {size_t(1), size_t(3)})
        for (bool add : {false, true}) {
            auto dev = g.adopt(y0.data(), nout * rows * 4), host = g.alloc(nout * rows * 4, Memory::host_visible);
            g.write(*host, 0, y0.data(), nout * rows * 4);
            if (add) {
                g.matmul_add(wv, {xb.get(), 0}, {dev.get(), 0}, nin, nout, rows);
                g.matmul_add(wv, {xb.get(), 0}, {host.get(), 0}, nin, nout, rows);
            } else {
                g.matmul(wv, {xb.get(), 0}, {dev.get(), 0}, nin, nout, rows);
                g.matmul(wv, {xb.get(), 0}, {host.get(), 0}, nin, nout, rows);
            }
            g.sync();
            require(same(read(g, *dev, nout * rows), read(g, *host, nout * rows)),
                    std::string("a product into host memory is the product into device memory (") + (add ? "added" : "written") + ", " + std::to_string(rows) + " rows)");
        }
}

void ops(Backend& g, Backend& c) {
    // Q8_0 rows decoded exactly: a matmul and an embedding against the CPU.
    {
        const size_t nin = 4096, nout = 37, rows = 3;
        const std::vector<uint8_t> w = q8_matrix(nin, nout, 21);
        const std::vector<float> x = row_classes::values(nin * rows, 22);
        const uint32_t ids[3] = {36, 0, 5};
        std::vector<std::vector<float>> prod(2), emb(2);
        Backend* bs[2] = {&g, &c};
        for (int k = 0; k < 2; ++k) {
            Backend& b = *bs[k];
            auto wb = b.adopt(w.data(), w.size()), xb = b.adopt(x.data(), x.size() * 4), yb = b.alloc(nout * rows * 4), eb = b.alloc(3 * nin * 4);
            b.matmul({quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, rows, {}, Dtype::f32);
            prod[k] = read(b, *yb, nout * rows);
            b.embed({eb.get(), 0}, {quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, nin, nout, ids, 3);
            emb[k] = read(b, *eb, 3 * nin);
        }
        close(prod[0], prod[1], "a Q8_0 matmul against the CPU");
        require(same(emb[0], emb[1]), "a Q8_0 embedding gives the CPU's rows, bit for bit");
    }
    // Products: odd widths, several rows, and each row of a call equal to the row alone, bit for bit.
    for (size_t nin : {size_t(1000), size_t(4096)}) {
        const size_t nout = 37, rows = 5;
        const std::vector<float> w = row_classes::values(nin * nout, 2), x = row_classes::values(nin * rows, 3), base = row_classes::values(nout * rows, 4);
        std::vector<std::vector<float>> out(2);
        Backend* bs[2] = {&g, &c};
        for (int k = 0; k < 2; ++k) {
            Backend& b = *bs[k];
            auto wb = b.adopt(w.data(), w.size() * 4), xb = b.adopt(x.data(), x.size() * 4), yb = b.adopt(base.data(), base.size() * 4);
            b.matmul_add({quant::GGML_TYPE_F32, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, rows);
            out[k] = read(b, *yb, nout * rows);
            if (k == 0) {
                auto ya = b.alloc(nout * rows * 4);
                b.matmul({quant::GGML_TYPE_F32, {wb.get(), 0}}, {xb.get(), 0}, {ya.get(), 0}, nin, nout, rows);
                const std::vector<float> all = read(b, *ya, nout * rows);
                bool alone = true;
                for (size_t r = 0; r < rows; ++r) {
                    auto y1 = b.alloc(nout * 4);
                    b.matmul({quant::GGML_TYPE_F32, {wb.get(), 0}}, {xb.get(), r * nin}, {y1.get(), 0}, nin, nout, 1);
                    alone = alone && std::memcmp(read(b, *y1, nout).data(), all.data() + r * nout, nout * 4) == 0;
                }
                require(alone, "each row of a matmul " + std::to_string(nin) + " wide is the row alone, bit for bit");
            }
        }
        close(out[0], out[1], "matmul_add " + std::to_string(nin) + " wide against the CPU");
    }
    // Norms and elementwise ops.
    {
        const size_t rows = 3, n = 1000, stride = 1100;
        const std::vector<float> x = row_classes::values(rows * stride, 5), w = row_classes::values(n, 6), y = row_classes::values(rows * stride, 7);
        std::vector<std::vector<float>> norm(2), sm(2), ad(2);
        Backend* bs[2] = {&g, &c};
        for (int k = 0; k < 2; ++k) {
            Backend& b = *bs[k];
            auto xb = b.adopt(x.data(), x.size() * 4), wb = b.adopt(w.data(), w.size() * 4), yb = b.adopt(y.data(), y.size() * 4), ob = b.alloc(x.size() * 4);
            b.rms_norm_rows({ob.get(), 0}, {xb.get(), 0}, {wb.get(), 0}, rows, n, stride, 1e-6f);
            norm[k] = read(b, *ob, rows * stride);
            b.silu_mul({ob.get(), 0}, {xb.get(), 0}, {yb.get(), 0}, rows * stride);
            sm[k] = read(b, *ob, rows * stride);
            b.add({ob.get(), 0}, {yb.get(), 0}, rows * stride);
            ad[k] = read(b, *ob, rows * stride);
        }
        close(norm[0], norm[1], "rms_norm_rows against the CPU");
        close(sm[0], sm[1], "silu_mul against the CPU");
        close(ad[0], ad[1], "add against the CPU");
    }
    // The rope with and without a norm, positions out of order.
    for (bool weighted : {true, false}) {
        const size_t rows = 4, heads = 3, head_dim = 128, half = 64, positions = 50;
        const std::vector<float> x = row_classes::values(rows * heads * head_dim, 8), w = row_classes::values(head_dim, 9);
        std::vector<float> cos(positions * half), sin(positions * half);
        for (size_t p = 0; p < positions; ++p)
            for (size_t j = 0; j < half; ++j) {
                const float a = (float)p * std::pow(10000.0f, -2.0f * (float)j / (float)head_dim);
                cos[p * half + j] = std::cos(a);
                sin[p * half + j] = std::sin(a);
            }
        const uint32_t pos[rows] = {7, 0, 49, 3};
        std::vector<std::vector<float>> out(2);
        Backend* bs[2] = {&g, &c};
        for (int k = 0; k < 2; ++k) {
            Backend& b = *bs[k];
            auto xb = b.adopt(x.data(), x.size() * 4), wb = b.adopt(w.data(), w.size() * 4);
            auto cb = b.adopt(cos.data(), cos.size() * 4), sb = b.adopt(sin.data(), sin.size() * 4);
            b.norm_rope_partial({xb.get(), 0}, {xb.get(), 0}, rows, heads * head_dim, head_dim, heads, head_dim, 2 * half,
                                weighted ? CSlice{wb.get(), 0} : CSlice{}, 1e-6f, {cb.get(), 0}, {sb.get(), 0}, pos);
            out[k] = read(b, *xb, x.size());
        }
        close(out[0], out[1], std::string("norm_rope_partial ") + (weighted ? "with" : "without") + " a norm against the CPU");
    }
    // Rows of a table: embed and gather_rows.
    {
        const size_t width = 300, nrows = 20;
        const std::vector<float> t = row_classes::values(width * nrows, 10);
        const uint32_t ids[4] = {19, 0, 7, 7};
        auto tb = g.adopt(t.data(), t.size() * 4), ob = g.alloc(4 * width * 4);
        g.embed({ob.get(), 0}, {quant::GGML_TYPE_F32, {tb.get(), 0}}, width, nrows, ids, 4);
        std::vector<float> want;
        for (uint32_t id : ids) want.insert(want.end(), t.begin() + id * width, t.begin() + (id + 1) * width);
        require(same(read(g, *ob, 4 * width), want), "embed gives the table's rows");
        g.gather_rows({ob.get(), 0}, {tb.get(), 0}, width, ids, 4);
        require(same(read(g, *ob, 4 * width), want), "gather_rows gives the source's rows");
        const uint32_t bad[1] = {20};
        require(throws([&] { g.embed({ob.get(), 0}, {quant::GGML_TYPE_F32, {tb.get(), 0}}, width, nrows, bad, 1); }), "an embedding row past the table is refused");
    }
}

// KV writes and attention after histories across block edges, both cache types and two sequences in one call, against the CPU; then attention's rows, alone and beside others, bit for bit.
void attention(Backend& g, Backend& c) {
    const int n_head = 4, n_head_kv = 2, head_dim = 128;
    const size_t qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim;
    for (KVType kt : {KVType::f32, KVType::f16})
        for (size_t hist : {size_t(0), size_t(70), size_t(130)}) {
            const size_t rows = 3;
            const std::vector<float> keys = row_classes::values((hist + rows) * kvw * 2, 11), vals = row_classes::values((hist + rows) * kvw * 2, 12);
            const std::vector<float> q = row_classes::values(rows * 2 * qw, 13, 4.0f);
            std::vector<std::vector<float>> out(2);
            Backend* bs[2] = {&g, &c};
            for (int k = 0; k < 2; ++k) {
                Backend& b = *bs[k];
                auto kb = b.adopt(keys.data(), keys.size() * 4), vb = b.adopt(vals.data(), vals.size() * 4);
                auto qb = b.adopt(q.data(), q.size() * 4), ob = b.alloc(rows * 2 * qw * 4);
                auto st = b.kv_alloc(1, n_head_kv, head_dim, 512, kt, kt);
                infer::BlockPool pool;
                pool.configure(st->max_blocks());
                infer::KVSequence s1(&pool, b.kv_layout().block_tokens), s2(&pool, b.kv_layout().block_tokens);
                for (infer::KVSequence* s : {&s1, &s2}) {
                    if (!hist) continue;
                    s->prepare(hist);
                    const KVView h = s->view(st.get());
                    const size_t at = s == &s1 ? 0 : (hist + rows) * kvw;
                    b.kv_write(0, &h, 1, {kb.get(), at}, {vb.get(), at});
                    s->commit();
                }
                s1.prepare(rows);
                s2.prepare(rows);
                const KVView v[2] = {s1.view(st.get()), s2.view(st.get())};
                std::vector<float> kr, vr;
                for (size_t sq = 0; sq < 2; ++sq) {
                    const size_t at = sq * (hist + rows) * kvw + hist * kvw;
                    kr.insert(kr.end(), keys.begin() + at, keys.begin() + at + rows * kvw);
                    vr.insert(vr.end(), vals.begin() + at, vals.begin() + at + rows * kvw);
                }
                auto krb = b.adopt(kr.data(), kr.size() * 4), vrb = b.adopt(vr.data(), vr.size() * 4);
                b.kv_write(0, v, 2, {krb.get(), 0}, {vrb.get(), 0});
                b.attention({qb.get(), 0}, 0, v, 2, {ob.get(), 0}, n_head, n_head_kv, head_dim);
                out[k] = read(b, *ob, rows * 2 * qw);
                if (k == 0) {
                    // The first sequence alone gives its rows' bits.
                    auto o1 = b.alloc(rows * qw * 4);
                    b.attention({qb.get(), 0}, 0, v, 1, {o1.get(), 0}, n_head, n_head_kv, head_dim);
                    require(std::memcmp(read(b, *o1, rows * qw).data(), out[0].data(), rows * qw * 4) == 0,
                            "attention of a sequence alone gives its rows beside another, bit for bit");
                }
                s1.abort();
                s2.abort();
            }
            close(out[0], out[1], std::string("attention, ") + kv_type_name(kt) + " cache, history " + std::to_string(hist) + ", two sequences, against the CPU");
        }
}

// The twin's rule as docs/STATUS.md gives it, on the host, at L levels: a block's scale and its values, 32767 for block-int16 and 127 for block-int8.
void twin_rule(const float* x, float& d, int16_t* q, float L) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
    if (amax == 0.0f) {
        d = 0.0f;
        for (int j = 0; j < 32; ++j) q[j] = 0;
        return;
    }
    int e = 0;
    std::frexp(amax, &e);
    const int s = (amax >= 0x1p-100f && amax <= 0x1p100f) ? 0 : 1 - e;
    const float an = std::ldexp(amax, s);
    const float k = L / an;
    for (int j = 0; j < 32; ++j) {
        const float r = std::ldexp(x[j], s) * k;
        const float t = std::min(std::floor(std::fabs(r) + 0.5f), L);
        q[j] = (int16_t)(r < 0.0f ? -t : t);
    }
    d = std::ldexp(an / L, -s);
}

// A Q8_0 row of `nin` values whose block b holds value 1 at position j and scale 1, zero elsewhere: its product reads the twin's value d q at (b, j) alone.
std::vector<uint8_t> one_hot_q8(size_t nin, size_t b, size_t j) {
    std::vector<uint8_t> row(nin / 32 * 34, 0);
    const uint16_t one = 0x3c00;
    row[b * 34] = uint8_t(one);
    row[b * 34 + 1] = uint8_t(one >> 8);
    row[b * 34 + 2 + j] = 1;
    return row;
}

// A class's Q8_0 products, f16 on the block-int16 twin and int8 on the block-int8 one, against the twin's rule and a double oracle on the twin.
// Every decode column must be the column alone, and a prompt row the same bits in any call.
void twin_products(Backend& g, Dtype dt) {
    const float L = dt == Dtype::int8 ? 127.0f : 32767.0f;
    const std::string form = dt == Dtype::int8 ? "8-bit" : "16-bit";
    // Twin: blocks of ordinary, tiny, huge, tied and zero values, read through one-hot rows on both kernels.
    const size_t nin = 32 * 6;
    std::vector<float> x = row_classes::values(nin * 2, 31);
    for (size_t j = 0; j < 32; ++j) {
        x[32 + j] *= 1e-35f;              // a block whose scale needs the power of two
        x[64 + j] *= 1e35f;               // and one whose largest value would overflow its reciprocal's product
        x[96 + j] = 0.0f;                 // a block of zeros
        x[128 + j] = j == 0 ? L : (float)j + 0.5f;   // exact halves: ties away from zero
    }
    x[160] = -L;
    x[161] = -0.5f;
    std::vector<uint8_t> w;
    for (size_t o = 0; o < nin; ++o) {
        const std::vector<uint8_t> row = one_hot_q8(nin, o / 32, o % 32);
        w.insert(w.end(), row.begin(), row.end());
    }
    auto wb = g.adopt(w.data(), w.size()), xb = g.adopt(x.data(), x.size() * 4), yb = g.alloc(nin * 2 * 4);
    bool rule = true;
    for (size_t extent : {size_t(1), size_t(2)}) {
        const RowRun run[1] = {{2, extent}};
        g.matmul({quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nin, 2, {run, 1}, dt);
        const std::vector<float> y = read(g, *yb, nin * 2);
        for (size_t r = 0; r < 2; ++r)
            for (size_t b = 0; b < nin / 32; ++b) {
                float d;
                int16_t q[32];
                twin_rule(x.data() + r * nin + b * 32, d, q, L);
                for (size_t j = 0; j < 32; ++j) {
                    const float want = d * (float)q[j];
                    rule = rule && std::memcmp(&y[r * nin + b * 32 + j], &want, 4) == 0;
                }
            }
    }
    require(rule, "the " + form + " twin is its rule, through the row kernel and the tile");
    // Products against a double oracle on the twin, rows 4096 wide, generated and prompt rows.
    const size_t width = 4096, nout = 40, rows = 21;
    const std::vector<uint8_t> m = q8_matrix(width, nout, 33);
    const std::vector<float> xs = row_classes::values(width * rows, 34);
    auto mb = g.adopt(m.data(), m.size()), xsb = g.adopt(xs.data(), xs.size() * 4), ys = g.alloc(nout * rows * 4);
    std::vector<std::vector<float>> twin_d(rows, std::vector<float>(width / 32));
    std::vector<std::vector<int16_t>> twin_q(rows, std::vector<int16_t>(width));
    for (size_t r = 0; r < rows; ++r)
        for (size_t b = 0; b < width / 32; ++b) twin_rule(xs.data() + r * width + b * 32, twin_d[r][b], twin_q[r].data() + b * 32, L);
    for (size_t extent : {size_t(1), size_t(2)}) {
        const RowRun run[1] = {{rows, extent}};
        g.matmul({quant::GGML_TYPE_Q8_0, {mb.get(), 0}}, {xsb.get(), 0}, {ys.get(), 0}, width, nout, rows, {run, 1}, dt);
        const std::vector<float> y = read(g, *ys, nout * rows);
        bool close_ok = true;
        for (size_t r = 0; r < rows; ++r)
            for (size_t o = 0; o < nout; ++o) {
                double sum = 0.0, mag = 0.0;
                for (size_t b = 0; b < width / 32; ++b) {
                    const uint8_t* block = m.data() + (o * (width / 32) + b) * 34;
                    const double dw = f16_to_f32((uint16_t)(block[0] | block[1] << 8));
                    long isum = 0;
                    for (size_t j = 0; j < 32; ++j) isum += (long)(int8_t)block[2 + j] * twin_q[r][b * 32 + j];
                    sum += dw * twin_d[r][b] * (double)isum;
                    mag += std::fabs(dw * twin_d[r][b] * (double)isum);
                }
                close_ok = close_ok && std::fabs(y[r * nout + o] - sum) <= 1e-6 * mag + 1e-30;
            }
        require(close_ok, std::string("the ") + (extent == 1 ? "row kernel" : "tile") + " within 1e-6 of a double sum over the " + form + " twin");
    }
    // A group of three projections of one X in one tile call, its inner dimension split for a prompt of 21 rows, each projection within 1e-6 of the double sum over the twin.
    {
        const size_t outs[3] = {40, 104, 64};
        std::vector<std::vector<uint8_t>> gm;
        std::vector<BufferPtr> gw, gy;
        for (size_t p = 0; p < 3; ++p) {
            gm.push_back(q8_matrix(width, outs[p], 81 + (uint32_t)p));
            gw.push_back(g.adopt(gm[p].data(), gm[p].size()));
            gy.push_back(g.alloc(outs[p] * rows * 4));
        }
        const RowRun run[1] = {{rows, size_t(21)}};
        g.matmul_group({Projection({quant::GGML_TYPE_Q8_0, {gw[0].get(), 0}}, {gy[0].get(), 0}, outs[0]),
                        Projection({quant::GGML_TYPE_Q8_0, {gw[1].get(), 0}}, {gy[1].get(), 0}, outs[1]),
                        Projection({quant::GGML_TYPE_Q8_0, {gw[2].get(), 0}}, {gy[2].get(), 0}, outs[2])},
                       {xsb.get(), 0}, width, rows, {run, 1}, dt);
        bool group_ok = true;
        for (size_t p = 0; p < 3; ++p) {
            const std::vector<float> y = read(g, *gy[p], outs[p] * rows);
            for (size_t r = 0; r < rows; ++r)
                for (size_t o = 0; o < outs[p]; ++o) {
                    double sum = 0.0, mag = 0.0;
                    for (size_t b = 0; b < width / 32; ++b) {
                        const uint8_t* block = gm[p].data() + (o * (width / 32) + b) * 34;
                        const double dw = f16_to_f32((uint16_t)(block[0] | block[1] << 8));
                        long isum = 0;
                        for (size_t j = 0; j < 32; ++j) isum += (long)(int8_t)block[2 + j] * twin_q[r][b * 32 + j];
                        sum += dw * twin_d[r][b] * (double)isum;
                        mag += std::fabs(dw * twin_d[r][b] * (double)isum);
                    }
                    group_ok = group_ok && std::fabs(y[r * outs[p] + o] - sum) <= 1e-6 * mag + 1e-30;
                }
        }
        require(group_ok, form + ": a group of three projections in one split tile call within 1e-6 of a double sum over the twin");
    }
    auto y1 = g.alloc(nout * 4);
    // Every decode column of 1 to 64 generated rows, up to four launches of 16, is the column alone, bit for bit.
    // The residual add and a group of three projections are held so at counts either side of each launch's edge.
    {
        const size_t cols = 64;
        bool alone = true, added = true, grouped = true;
        for (size_t w : {size_t(4096), size_t(1280), size_t(1312)}) {
            const std::vector<uint8_t> mw = q8_matrix(w, nout, 61);
            const std::vector<float> xw = row_classes::values(w * cols, 62), base = row_classes::values(nout * cols, 63);
            auto mwb = g.adopt(mw.data(), mw.size()), xwb = g.adopt(xw.data(), xw.size() * 4);
            auto yw = g.alloc(nout * cols * 4), ya = g.alloc(nout * cols * 4), yb = g.alloc(nout * cols * 4), yc = g.alloc(nout * cols * 4);
            const WeightView wv{quant::GGML_TYPE_Q8_0, {mwb.get(), 0}};
            std::vector<std::vector<float>> one(cols), one_added(cols);
            for (size_t r = 0; r < cols; ++r) {
                const RowRun run[1] = {{1, 1}};
                g.matmul(wv, {xwb.get(), r * w}, {y1.get(), 0}, w, nout, 1, {run, 1}, dt);
                one[r] = read(g, *y1, nout);
                g.write(*y1, 0, base.data() + r * nout, nout * 4);
                g.matmul_add(wv, {xwb.get(), r * w}, {y1.get(), 0}, w, nout, 1, {run, 1}, dt);
                one_added[r] = read(g, *y1, nout);
            }
            for (size_t n = 1; n <= cols; ++n) {
                const RowRun run[1] = {{n, 1}};
                g.matmul(wv, {xwb.get(), 0}, {yw.get(), 0}, w, nout, n, {run, 1}, dt);
                const std::vector<float> y = read(g, *yw, nout * n);
                for (size_t r = 0; r < n; ++r) alone = alone && std::memcmp(y.data() + r * nout, one[r].data(), nout * 4) == 0;
            }
            for (size_t n : {size_t(1), size_t(2), size_t(3), size_t(15), size_t(16), size_t(17), size_t(31), size_t(32), size_t(33), size_t(48), size_t(49), size_t(63), size_t(64)}) {
                const RowRun run[1] = {{n, 1}};
                g.write(*ya, 0, base.data(), nout * n * 4);
                g.matmul_add(wv, {xwb.get(), 0}, {ya.get(), 0}, w, nout, n, {run, 1}, dt);
                const std::vector<float> y = read(g, *ya, nout * n);
                for (size_t r = 0; r < n; ++r) added = added && std::memcmp(y.data() + r * nout, one_added[r].data(), nout * 4) == 0;
                g.matmul_group({Projection(wv, {yw.get(), 0}, nout), Projection(wv, {yb.get(), 0}, nout), Projection(wv, {yc.get(), 0}, nout)},
                               {xwb.get(), 0}, w, n, {run, 1}, dt);
                for (BufferPtr* out : {&yw, &yb, &yc}) {
                    const std::vector<float> yg = read(g, **out, nout * n);
                    for (size_t r = 0; r < n; ++r) grouped = grouped && std::memcmp(yg.data() + r * nout, one[r].data(), nout * 4) == 0;
                }
            }
        }
        require(alone, form + ": every decode column of 1 to 64 generated rows is the column alone, bit for bit, rows 4096, 1280 and 1312 wide (41 blocks, the int8 class's quarter layout)");
        require(added, form + ": every decode column of the residual add at 1 to 64 rows is the column alone, bit for bit");
        require(grouped, form + ": every decode column of a group of three projections at 1 to 64 rows is the column alone, bit for bit");
    }
    // A decode group of three different projections in one launch gives each projection's own call, bit for bit; 1248 wide takes the int8 quarter layout.
    {
        const size_t outs[3] = {41, 104, 63};
        bool group_ok = true;
        for (size_t w : {size_t(1280), size_t(1248)}) {
            const std::vector<float> xw = row_classes::values(w * 17, 91);
            auto xwb = g.adopt(xw.data(), xw.size() * 4);
            std::vector<std::vector<uint8_t>> gm;
            std::vector<BufferPtr> gw, gy, ys;
            for (size_t p = 0; p < 3; ++p) {
                gm.push_back(q8_matrix(w, outs[p], 92 + (uint32_t)p));
                gw.push_back(g.adopt(gm[p].data(), gm[p].size()));
                gy.push_back(g.alloc(outs[p] * 17 * 4));
                ys.push_back(g.alloc(outs[p] * 17 * 4));
            }
            for (size_t n : {size_t(1), size_t(2), size_t(17)}) {
                const RowRun run[1] = {{n, 1}};
                for (size_t p = 0; p < 3; ++p) g.matmul({quant::GGML_TYPE_Q8_0, {gw[p].get(), 0}}, {xwb.get(), 0}, {ys[p].get(), 0}, w, outs[p], n, {run, 1}, dt);
                g.matmul_group({Projection({quant::GGML_TYPE_Q8_0, {gw[0].get(), 0}}, {gy[0].get(), 0}, outs[0]),
                                Projection({quant::GGML_TYPE_Q8_0, {gw[1].get(), 0}}, {gy[1].get(), 0}, outs[1]),
                                Projection({quant::GGML_TYPE_Q8_0, {gw[2].get(), 0}}, {gy[2].get(), 0}, outs[2])},
                               {xwb.get(), 0}, w, n, {run, 1}, dt);
                for (size_t p = 0; p < 3; ++p) {
                    const std::vector<float> a = read(g, *gy[p], outs[p] * n), b = read(g, *ys[p], outs[p] * n);
                    group_ok = group_ok && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
                }
            }
        }
        require(group_ok, form + ": a decode group of three different projections in one launch is each projection's own call, bit for bit, at 1, 2 and 17 rows");
    }
    // A prompt row gives the same bits in calls of 1 to 21 rows and of 63 to 65, 128 and 130, which cross the tile's 64 rows, over 100 outputs, which end inside its 64.
    {
        const size_t pn = 130, po = 100;
        const std::vector<uint8_t> mp = q8_matrix(width, po, 71);
        const std::vector<float> xp = row_classes::values(width * pn, 72);
        auto mpb = g.adopt(mp.data(), mp.size()), xpb = g.adopt(xp.data(), xp.size() * 4), yp = g.alloc(po * pn * 4), yp1 = g.alloc(po * 4);
        const WeightView wv{quant::GGML_TYPE_Q8_0, {mpb.get(), 0}};
        std::vector<std::vector<float>> alone(pn);
        for (size_t r = 0; r < pn; ++r) {
            const RowRun run[1] = {{1, 2}};
            g.matmul(wv, {xpb.get(), r * width}, {yp1.get(), 0}, width, po, 1, {run, 1}, dt);
            alone[r] = read(g, *yp1, po);
        }
        bool tiled = true;
        std::vector<size_t> counts;
        for (size_t n = 1; n <= 21; ++n) counts.push_back(n);
        for (size_t n : {size_t(63), size_t(64), size_t(65), size_t(128), size_t(130)}) counts.push_back(n);
        for (size_t n : counts) {
            const RowRun run[1] = {{n, 2}};
            g.matmul(wv, {xpb.get(), 0}, {yp.get(), 0}, width, po, n, {run, 1}, dt);
            const std::vector<float> y = read(g, *yp, po * n);
            for (size_t r = 0; r < n; ++r) tiled = tiled && std::memcmp(y.data() + r * po, alone[r].data(), po * 4) == 0;
        }
        require(tiled, form + ": a prompt row gives the same bits in calls of 1 to 130 rows across the tile's edges");
    }
    // A matrix of an odd number of blocks, 65 outputs of 3 blocks adopted alone, so its last block starts on a word and the tile's last word load of it runs past the matrix: each prompt row the bits of that row alone.
    {
        const size_t sw = 96, so = 65, sn = 70;
        const std::vector<uint8_t> ms = q8_matrix(sw, so, 73);
        const std::vector<float> xs = row_classes::values(sw * sn, 74);
        auto msb = g.adopt(ms.data(), ms.size()), xsb2 = g.adopt(xs.data(), xs.size() * 4), ysn = g.alloc(so * sn * 4), ys1 = g.alloc(so * 4);
        const WeightView wv{quant::GGML_TYPE_Q8_0, {msb.get(), 0}};
        bool edge = true;
        const RowRun all[1] = {{sn, 2}};
        g.matmul(wv, {xsb2.get(), 0}, {ysn.get(), 0}, sw, so, sn, {all, 1}, dt);
        const std::vector<float> y = read(g, *ysn, so * sn);
        for (size_t r = 0; r < sn; ++r) {
            const RowRun run[1] = {{1, 2}};
            g.matmul(wv, {xsb2.get(), r * sw}, {ys1.get(), 0}, sw, so, 1, {run, 1}, dt);
            edge = edge && std::memcmp(y.data() + r * so, read(g, *ys1, so).data(), so * 4) == 0;
        }
        // The same matrix with room after it in its buffer gives the same bits, so the load past its end reads nothing that reaches a sum.
        std::vector<uint8_t> roomy = ms;
        roomy.resize(ms.size() + 64, 0xa5);
        auto rb = g.adopt(roomy.data(), roomy.size());
        g.matmul({quant::GGML_TYPE_Q8_0, {rb.get(), 0}}, {xsb2.get(), 0}, {ysn.get(), 0}, sw, so, sn, {all, 1}, dt);
        edge = edge && std::memcmp(read(g, *ysn, so * sn).data(), y.data(), so * sn * 4) == 0;
        require(edge, form + ": a prompt row of a matrix whose last block ends inside a word gives the bits of that row alone and of the matrix with room after it");
    }
    // Every pair of extents of one class: the same bits through a Q8_0 product in this class.
    std::vector<std::vector<float>> out;
    const size_t four = 4;
    for (size_t e : row_classes::kExtents) {
        const RowRun run[1] = {{four, e}};
        g.matmul({quant::GGML_TYPE_Q8_0, {mb.get(), 0}}, {xsb.get(), 0}, {ys.get(), 0}, width, nout, four, {run, 1}, dt);
        out.push_back(read(g, *ys, nout * four));
    }
    require(row_classes::same_within_classes(g, out, "a Q8_0 product, " + form) > 0, "the row-class check compared pairs, " + form);
}

// The twins producers write beside their outputs, once a product has read one in the class: a product of a norm's or silu_mul's output gives the bits of the same product of a copy of it, whose twin the product makes itself.
// A write into the output between them, or a product into it, drops the producer's twin.
void producer_twins(Backend& g, Dtype dt) {
    const std::string form = dt == Dtype::int8 ? "8-bit" : "16-bit";
    const size_t nout = 40;
    bool norms = true, silus = true, dropped = true;
    // Rows of 1024, 4096 and 5120 take the held builds of 16 and 32 values a thread, 9216 the plain one, and a stride past n writes no twin.
    struct Case { size_t n, rows, stride, extent; };
    for (const Case cs : {Case{4096, 1, 4096, 1}, Case{4096, 3, 4096, 3}, Case{5120, 2, 5120, 1}, Case{9216, 2, 9216, 2}, Case{1024, 2, 1024, 2}, Case{4096, 2, 4200, 1}}) {
        const size_t n = cs.n, rows = cs.rows, stride = cs.stride;
        const std::vector<float> x = row_classes::values(rows * stride, 91), w = row_classes::values(n, 92), up = row_classes::values(rows * stride, 93);
        const std::vector<uint8_t> m = q8_matrix(n, nout, 94);
        auto xb = g.adopt(x.data(), x.size() * 4), wb = g.adopt(w.data(), w.size() * 4), ub = g.adopt(up.data(), up.size() * 4), mb = g.adopt(m.data(), m.size());
        auto ob = g.alloc(rows * stride * 4), cb = g.alloc(rows * n * 4), y = g.alloc(nout * rows * 4), yc = g.alloc(nout * rows * 4);
        const WeightView wv{quant::GGML_TYPE_Q8_0, {mb.get(), 0}};
        const RowRun run[1] = {{rows, cs.extent}};
        // The copy is dense, so the product of a strided output reads row 0 alone.
        const size_t prows = stride == n ? rows : 1;
        const RowRun prun[1] = {{prows, prows == 1 ? size_t(1) : cs.extent}};
        g.rms_norm_rows({ob.get(), 0}, {xb.get(), 0}, {wb.get(), 0}, rows, n, stride, 1e-6f);
        g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, n, nout, prows, {prun, 1}, dt);
        g.copy(*cb, 0, *ob, 0, prows * n * 4);
        g.matmul(wv, {cb.get(), 0}, {yc.get(), 0}, n, nout, prows, {prun, 1}, dt);
        norms = norms && same(read(g, *y, nout * prows), read(g, *yc, nout * prows));
        if (stride != n) continue;
        g.silu_mul({ob.get(), 0}, {xb.get(), 0}, {ub.get(), 0}, rows * n);
        g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, n, nout, rows, {run, 1}, dt);
        g.copy(*cb, 0, *ob, 0, rows * n * 4);
        g.matmul(wv, {cb.get(), 0}, {yc.get(), 0}, n, nout, rows, {run, 1}, dt);
        silus = silus && same(read(g, *y, nout * rows), read(g, *yc, nout * rows));
        // A write into the norm's output: the product reads the written values.
        g.rms_norm_rows({ob.get(), 0}, {xb.get(), 0}, {wb.get(), 0}, rows, n, stride, 1e-6f);
        g.write(*ob, 0, up.data(), rows * n * 4);
        g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, n, nout, rows, {run, 1}, dt);
        g.matmul(wv, {ub.get(), 0}, {yc.get(), 0}, n, nout, rows, {run, 1}, dt);
        dropped = dropped && same(read(g, *y, nout * rows), read(g, *yc, nout * rows));
        // A product into the norm's output, then a product of it.
        if (n >= nout * rows) {
            g.rms_norm_rows({ob.get(), 0}, {xb.get(), 0}, {wb.get(), 0}, rows, n, stride, 1e-6f);
            g.matmul(wv, {ob.get(), 0}, {ob.get(), 0}, n, nout, 1, {}, dt);
            g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, n, nout, rows, {run, 1}, dt);
            g.copy(*cb, 0, *ob, 0, rows * n * 4);
            g.matmul(wv, {cb.get(), 0}, {yc.get(), 0}, n, nout, rows, {run, 1}, dt);
            dropped = dropped && same(read(g, *y, nout * rows), read(g, *yc, nout * rows));
        }
    }
    require(norms, form + ": a product of rms_norm_rows' output gives the bits of a product of its copy");
    require(silus, form + ": a product of silu_mul's output gives the bits of a product of its copy");
    require(dropped, form + ": a write or a product into a producer's output drops its twin");
}

// A prompt's rows whose histories cross a change in the attention's chunks (256 positions, and 4096 where the count stops at 16), and at 16K a prompt of 40 rows on the attention tile: each row the same bits as that row alone in its class.
void attention_chunks(Backend& g) {
    const int n_head = 4, n_head_kv = 2, head_dim = 128;
    const size_t qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim;
    for (size_t hist : {size_t(250), size_t(4090), size_t(16300)}) {
        const size_t rows = hist > 8192 ? 40 : 13;
        const std::vector<float> keys = row_classes::values((hist + rows) * kvw, 41), vals = row_classes::values((hist + rows) * kvw, 42);
        const std::vector<float> q = row_classes::values(rows * qw, 43, 4.0f);
        auto kb = g.adopt(keys.data(), keys.size() * 4), vb = g.adopt(vals.data(), vals.size() * 4);
        auto qb = g.adopt(q.data(), q.size() * 4), ob = g.alloc(rows * qw * 4), o1 = g.alloc(qw * 4);
        auto st = g.kv_alloc(1, n_head_kv, head_dim, 16448, KVType::f16, KVType::f16);
        infer::BlockPool pool;
        pool.configure(st->max_blocks());
        infer::KVSequence s(&pool, g.kv_layout().block_tokens);
        s.prepare(hist + rows);
        KVView all = s.view(st.get());
        g.kv_write(0, &all, 1, {kb.get(), 0}, {vb.get(), 0});
        KVView v = all;
        v.length = hist;
        v.nq = rows;
        v.extent = rows;   // the prompt's extent, which a row alone below keeps, so both are one row class
        g.attention({qb.get(), 0}, 0, &v, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
        const std::vector<float> together = read(g, *ob, rows * qw);
        bool alone = true;
        for (size_t b = 0; b < rows; ++b) {
            KVView one = all;
            one.length = hist + b;
            one.nq = 1;
            one.extent = rows;
            g.attention({qb.get(), b * qw}, 0, &one, 1, {o1.get(), 0}, n_head, n_head_kv, head_dim);
            alone = alone && std::memcmp(read(g, *o1, qw).data(), together.data() + b * qw, qw * 4) == 0;
        }
        require(alone, "attention rows across a chunk change after a history of " + std::to_string(hist) + " are each the row alone, bit for bit");
        s.abort();
    }
}

// The fused norm_rope_kv against the base's ops one at a time on the same backend: q, k and both KV sides bit for bit, two sequences after histories, f32 and f16 caches, with and without the norms' weights.
void fused_norm_rope_kv(Backend& g) {
    const size_t n_head = 8, n_head_kv = 2, dim = 128, half = dim / 2, qw = n_head * dim, kvw = n_head_kv * dim, ctx = 256;
    const std::vector<float> cosv = row_classes::values(ctx * half, 51), sinv = row_classes::values(ctx * half, 52);
    const std::vector<float> qwt = row_classes::values(dim, 53), kwt = row_classes::values(dim, 54);
    bool ok = true;
    for (KVType kt : {KVType::f32, KVType::f16})
        for (bool weighted : {true, false})
            for (size_t hist : {size_t(0), size_t(61)}) {
                const size_t rows = 3;
                const std::vector<float> q = row_classes::values(rows * 2 * qw, 55, 3.0f), k = row_classes::values(rows * 2 * kvw, 56, 3.0f),
                                         v = row_classes::values(rows * 2 * kvw, 57);
                auto cb = g.adopt(cosv.data(), cosv.size() * 4), sb = g.adopt(sinv.data(), sinv.size() * 4);
                auto qwb = g.adopt(qwt.data(), qwt.size() * 4), kwb = g.adopt(kwt.data(), kwt.size() * 4);
                std::vector<float> outs[2][2];
                std::vector<uint8_t> caches[2][2];
                for (int fused = 0; fused < 2; ++fused) {
                    auto qb = g.adopt(q.data(), q.size() * 4), kb = g.adopt(k.data(), k.size() * 4), vb = g.adopt(v.data(), v.size() * 4);
                    auto st = g.kv_alloc(1, n_head_kv, dim, 512, kt, kt);
                    infer::BlockPool pool;
                    pool.configure(st->max_blocks());
                    infer::KVSequence s1(&pool, g.kv_layout().block_tokens), s2(&pool, g.kv_layout().block_tokens);
                    if (hist) {
                        for (infer::KVSequence* s : {&s1, &s2}) {
                            s->prepare(hist);
                            s->commit();
                        }
                    }
                    s1.prepare(rows);
                    s2.prepare(rows);
                    const KVView views[2] = {s1.view(st.get()), s2.view(st.get())};
                    std::vector<uint32_t> pos(rows * 2);
                    for (size_t r = 0; r < pos.size(); ++r) pos[r] = (uint32_t)(hist + r % rows + 7 * (r / rows));
                    const Backend::RopeArgs rope{{cb.get(), 0}, {sb.get(), 0}, half, pos.data(), 1e-6f};
                    const CSlice qn = weighted ? CSlice{qwb.get(), 0} : CSlice{}, kn = weighted ? CSlice{kwb.get(), 0} : CSlice{};
                    if (fused) g.norm_rope_kv({qb.get(), 0}, qw, n_head, qn, {kb.get(), 0}, {vb.get(), 0}, kvw, n_head_kv, kn, rope, rows * 2, 0, views, 2);
                    else g.Backend::norm_rope_kv({qb.get(), 0}, qw, n_head, qn, {kb.get(), 0}, {vb.get(), 0}, kvw, n_head_kv, kn, rope, rows * 2, 0, views, 2);
                    outs[fused][0] = read(g, *qb, q.size());
                    outs[fused][1] = read(g, *kb, k.size());
                    auto* bs = dynamic_cast<BlockKVStorage*>(st.get());
                    for (int side = 0; side < 2; ++side) {
                        const Buffer& buf = side ? *bs->v_buffer(0) : *bs->k_buffer(0);
                        caches[fused][side].resize(buf.size());
                        g.read(buf, 0, caches[fused][side].data(), buf.size());
                    }
                    s1.abort();
                    s2.abort();
                }
                ok = ok && same(outs[0][0], outs[1][0]) && same(outs[0][1], outs[1][1]) && caches[0][0] == caches[1][0] && caches[0][1] == caches[1][1];
            }
    require(ok, "the fused norm_rope_kv gives q, k and both KV sides of the ops one at a time, bit for bit");
}

// Attention's output twin: a product of the output gives the bits of a product of its copy, whose twin llmx_twin8 or llmx_twin16 makes, at one split and at several, and with a row on the tile kernel, which writes none.
void attention_twins(Backend& g, Dtype dt) {
    const int n_head = 8, n_head_kv = 2, head_dim = 128;
    const size_t qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim, nout = 40;
    const std::vector<uint8_t> m = q8_matrix(qw, nout, 61);
    auto mb = g.adopt(m.data(), m.size());
    const WeightView wv{quant::GGML_TYPE_Q8_0, {mb.get(), 0}};
    bool ok = true;
    struct Case { size_t hist, rows, extent; };
    for (const Case cs : {Case{10, 1, 1}, Case{200, 1, 1}, Case{200, 3, 1}, Case{70, 40, 40}}) {
        const size_t rows = cs.rows, all = cs.hist + rows;
        const std::vector<float> keys = row_classes::values(all * kvw, 62), vals = row_classes::values(all * kvw, 63), q = row_classes::values(rows * qw, 64, 4.0f);
        auto kb = g.adopt(keys.data(), keys.size() * 4), vb = g.adopt(vals.data(), vals.size() * 4), qb = g.adopt(q.data(), q.size() * 4);
        auto ob = g.alloc(rows * qw * 4), cb = g.alloc(rows * qw * 4), y = g.alloc(nout * rows * 4), yc = g.alloc(nout * rows * 4);
        auto st = g.kv_alloc(1, n_head_kv, head_dim, 512, KVType::f16, KVType::f16);
        infer::BlockPool pool;
        pool.configure(st->max_blocks());
        infer::KVSequence s(&pool, g.kv_layout().block_tokens);
        s.prepare(all);
        KVView v = s.view(st.get());
        g.kv_write(0, &v, 1, {kb.get(), 0}, {vb.get(), 0});
        v.length = cs.hist;
        v.nq = rows;
        v.extent = cs.extent;
        const RowRun run[1] = {{rows, cs.extent}};
        g.attention({qb.get(), 0}, 0, &v, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
        g.matmul(wv, {ob.get(), 0}, {y.get(), 0}, qw, nout, rows, {run, 1}, dt);
        g.copy(*cb, 0, *ob, 0, rows * qw * 4);
        g.matmul(wv, {cb.get(), 0}, {yc.get(), 0}, qw, nout, rows, {run, 1}, dt);
        ok = ok && same(read(g, *y, nout * rows), read(g, *yc, nout * rows));
        s.abort();
    }
    require(ok, std::string(dt == Dtype::int8 ? "8-bit" : "16-bit") + ": a product of attention's output gives the bits of a product of its copy");
}

// Every pair of extents of one class gives the same bits, through a matmul and through attention after a history (row_classes.hpp's shapes).
void classes(Backend& g) {
    require(g.row_class(1) != g.row_class(2), "a generated token is a class of its own");
    size_t pairs = 0;
    const size_t rows = 4, nout = 64;
    for (size_t nin : {size_t(256), size_t(4096)}) {
        const std::vector<float> w = row_classes::values(nin * nout, 71), x = row_classes::values(rows * nin, 72);
        auto wb = g.adopt(w.data(), w.size() * 4), xb = g.adopt(x.data(), x.size() * 4), y = g.alloc(rows * nout * 4);
        std::vector<std::vector<float>> out;
        for (size_t e : row_classes::kExtents) {
            const RowRun run[1] = {{rows, e}};
            g.matmul({quant::GGML_TYPE_F32, {wb.get(), 0}}, {xb.get(), 0}, {y.get(), 0}, nin, nout, rows, {run, 1}, Dtype::f32);
            out.push_back(read(g, *y, rows * nout));
        }
        pairs += row_classes::same_within_classes(g, out, "an F32 matmul, rows " + std::to_string(nin) + " wide");
    }
    // Attention of four query rows after a 70-token history at each extent, on both sides of the attention tile's threshold: rows of one class give the same bits.
    const int n_head = 4, n_head_kv = 2, head_dim = 128;
    const size_t hist = 70, qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim;
    const std::vector<float> keys = row_classes::values((hist + rows) * kvw, 79), vals = row_classes::values((hist + rows) * kvw, 80);
    const std::vector<float> q = row_classes::values(rows * qw, 81, 4.0f);
    auto kb = g.adopt(keys.data(), keys.size() * 4), vb = g.adopt(vals.data(), vals.size() * 4), qb = g.adopt(q.data(), q.size() * 4);
    auto ob = g.alloc(rows * qw * 4);
    auto st = g.kv_alloc(1, n_head_kv, head_dim, 512, KVType::f16, KVType::f16);
    infer::BlockPool pool;
    pool.configure(st->max_blocks());
    std::vector<std::vector<float>> att;
    for (size_t e : row_classes::kExtents) {
        infer::KVSequence seq(&pool, g.kv_layout().block_tokens);
        seq.prepare(hist);
        const KVView h = seq.view(st.get());
        g.kv_write(0, &h, 1, {kb.get(), 0}, {vb.get(), 0});
        seq.commit();
        seq.prepare(rows);
        KVView v = seq.view(st.get());
        v.extent = e;
        g.kv_write(0, &v, 1, {kb.get(), hist * kvw}, {vb.get(), hist * kvw});
        g.attention({qb.get(), 0}, 0, &v, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
        att.push_back(read(g, *ob, rows * qw));
        seq.abort();
    }
    const size_t att_pairs = row_classes::same_within_classes(g, att, "attention");
    require(att_pairs > 0, "the attention row-class check compared pairs");
    pairs += att_pairs;
    require(pairs > 0, "the row-class check compared pairs");
}

// pass_arg: 9000 decode steps of two layers take the device ring round, and the last step gives a fresh backend's bits.
void pass_args(Backend& g) {
    const int n_head = 4, n_head_kv = 2, head_dim = 128;
    const size_t steps = 9000, qw = (size_t)n_head * head_dim, kvw = (size_t)n_head_kv * head_dim;
    const std::vector<float> keys = row_classes::values(steps * kvw, 91), vals = row_classes::values(steps * kvw, 92), q = row_classes::values(qw, 93, 4.0f);
    std::vector<std::vector<float>> out;
    for (int fresh = 0; fresh < 2; ++fresh) {
        BackendPtr own = fresh ? make_rocm_backend(0) : nullptr;
        Backend& b = fresh ? *own : g;
        auto kb = b.adopt(keys.data(), keys.size() * 4), vb = b.adopt(vals.data(), vals.size() * 4), qb = b.adopt(q.data(), q.size() * 4);
        auto ob = b.alloc(2 * qw * 4);
        auto st = b.kv_alloc(2, n_head_kv, head_dim, steps, KVType::f16, KVType::f16);
        infer::BlockPool pool;
        pool.configure(st->max_blocks());
        infer::KVSequence s(&pool, b.kv_layout().block_tokens);
        if (!fresh) {
            for (size_t t = 0; t < steps; ++t) {
                s.prepare(1);
                const KVView v = s.view(st.get());
                for (size_t l = 0; l < 2; ++l) {
                    b.kv_write(l, &v, 1, {kb.get(), t * kvw}, {vb.get(), t * kvw});
                    b.attention({qb.get(), 0}, l, &v, 1, {ob.get(), l * qw}, n_head, n_head_kv, head_dim);
                }
                s.commit();
                (void)b.submit();
            }
        } else {
            s.prepare(steps);
            KVView v = s.view(st.get());
            for (size_t l = 0; l < 2; ++l) b.kv_write(l, &v, 1, {kb.get(), 0}, {vb.get(), 0});
            v.length = steps - 1;
            v.nq = 1;
            for (size_t l = 0; l < 2; ++l) b.attention({qb.get(), 0}, l, &v, 1, {ob.get(), l * qw}, n_head, n_head_kv, head_dim);
        }
        out.push_back(read(b, *ob, 2 * qw));
    }
    require(same(out[0], out[1]), "attention after 9000 decode steps of two layers, its pass values round the device ring, gives a fresh backend's bits");
}

// A failure the device reports at a ticket's event is kept as the first cause: raise(t) before results are read in place, a read and the next launch throw it.
void failure() {
    BackendPtr g = make_rocm_backend(0);
    HipFn& f = hip();
    const auto synchronize = f.hipEventSynchronize;
    auto buf = g->alloc(4096);
    const Ticket t = g->submit();
    f.hipEventSynchronize = [](hipEvent_t) { return hipErrorLaunchFailure; };
    g->wait(t);
    f.hipEventSynchronize = synchronize;
    std::vector<float> got(1024);
    require(throws([&] { g->raise(t); }), "raise throws a failure the device reported at the ticket's event");
    require(throws([&] { g->read(*buf, 0, got.data(), 4096); }), "a read after the failure throws it");
    require(throws([&] { g->add({buf.get(), 0}, {buf.get(), 0}, 1024); }), "a launch after the failure throws it");
}

} // namespace

int main() {
    BackendPtr g;
    try {
        g = make_rocm_backend(0);
    } catch (const RocmUnavailable& e) {
        std::cout << "backend-rocm: skipped, " << e.what() << "\n";
        return 77;
    }
    BackendPtr c = make_cpu_backend();
    c->set_threads(4);
    try {
        buffers(*g);
        refusals(*g);
        ops(*g, *c);
        attention(*g, *c);
        classes(*g);
        twin_products(*g, Dtype::f16);
        producer_twins(*g, Dtype::f16);
        twin_products(*g, Dtype::int8);
        producer_twins(*g, Dtype::int8);
        fused_norm_rope_kv(*g);
        attention_twins(*g, Dtype::f16);
        attention_twins(*g, Dtype::int8);
        host_output(*g);
        attention_chunks(*g);
        pass_args(*g);
        failure();
    } catch (const std::exception& e) {
        std::cout << "FAIL threw: " << e.what() << "\n";
        ++failures;
    }
    std::cout << (failures ? "backend-rocm: FAIL\n" : "backend-rocm: PASS\n");
    return failures ? 1 : 0;
}
