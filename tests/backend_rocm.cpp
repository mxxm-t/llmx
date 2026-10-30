// backend-rocm: the ROCm backend's buffers, tickets, refusals and dense F32 ops against the CPU backend, and each op's batch invariance; exits 77 without a runtime or a device.
#include "backends/cpu/cpu_backend.hpp"
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

// The twin's rule as docs/STATUS.md gives it, on the host: a block's scale and its 16-bit values.
void twin_rule(const float* x, float& d, int16_t* q) {
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
    const float k = 32767.0f / an;
    for (int j = 0; j < 32; ++j) {
        const float r = std::ldexp(x[j], s) * k;
        const float t = std::min(std::floor(std::fabs(r) + 0.5f), 32767.0f);
        q[j] = (int16_t)(r < 0.0f ? -t : t);
    }
    d = std::ldexp(an / 32767.0f, -s);
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

// The F16 class's Q8_0 products: the twin against its rule reconstructed on the host, the row kernel and the tile against a double oracle on that twin, every decode column the column alone, and a prompt row the same bits in any call.
void int16_products(Backend& g) {
    // Twin: blocks of ordinary, tiny, huge, tied and zero values, read through one-hot rows on both kernels.
    const size_t nin = 32 * 6;
    std::vector<float> x = row_classes::values(nin * 2, 31);
    for (size_t j = 0; j < 32; ++j) {
        x[32 + j] *= 1e-35f;              // a block whose scale needs the power of two
        x[64 + j] *= 1e35f;               // and one whose largest value would overflow its reciprocal's product
        x[96 + j] = 0.0f;                 // a block of zeros
        x[128 + j] = j == 0 ? 32767.0f : (float)j + 0.5f;   // exact halves: ties away from zero
    }
    x[160] = -32767.0f;
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
        g.matmul({quant::GGML_TYPE_Q8_0, {wb.get(), 0}}, {xb.get(), 0}, {yb.get(), 0}, nin, nin, 2, {run, 1}, Dtype::f16);
        const std::vector<float> y = read(g, *yb, nin * 2);
        for (size_t r = 0; r < 2; ++r)
            for (size_t b = 0; b < nin / 32; ++b) {
                float d;
                int16_t q[32];
                twin_rule(x.data() + r * nin + b * 32, d, q);
                for (size_t j = 0; j < 32; ++j) {
                    const float want = d * (float)q[j];
                    rule = rule && std::memcmp(&y[r * nin + b * 32 + j], &want, 4) == 0;
                }
            }
    }
    require(rule, "the 16-bit twin is its rule, through the row kernel and the tile");
    // Products against a double oracle on the twin, rows 4096 wide, generated and prompt rows.
    const size_t width = 4096, nout = 40, rows = 21;
    const std::vector<uint8_t> m = q8_matrix(width, nout, 33);
    const std::vector<float> xs = row_classes::values(width * rows, 34);
    auto mb = g.adopt(m.data(), m.size()), xsb = g.adopt(xs.data(), xs.size() * 4), ys = g.alloc(nout * rows * 4);
    std::vector<std::vector<float>> twin_d(rows, std::vector<float>(width / 32));
    std::vector<std::vector<int16_t>> twin_q(rows, std::vector<int16_t>(width));
    for (size_t r = 0; r < rows; ++r)
        for (size_t b = 0; b < width / 32; ++b) twin_rule(xs.data() + r * width + b * 32, twin_d[r][b], twin_q[r].data() + b * 32);
    for (size_t extent : {size_t(1), size_t(2)}) {
        const RowRun run[1] = {{rows, extent}};
        g.matmul({quant::GGML_TYPE_Q8_0, {mb.get(), 0}}, {xsb.get(), 0}, {ys.get(), 0}, width, nout, rows, {run, 1}, Dtype::f16);
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
        require(close_ok, std::string("the ") + (extent == 1 ? "row kernel" : "tile") + " within 1e-6 of a double sum over the twin");
    }
    std::vector<std::vector<float>> single(rows);
    auto y1 = g.alloc(nout * 4);
    // Every decode column of 1 to 64 generated rows, up to four launches of 16, is the column alone, bit for bit.
    // The residual add and a group of three projections are held so at counts either side of each launch's edge.
    {
        const size_t cols = 64;
        bool alone = true, added = true, grouped = true;
        for (size_t w : {size_t(4096), size_t(1280)}) {
            const std::vector<uint8_t> mw = q8_matrix(w, nout, 61);
            const std::vector<float> xw = row_classes::values(w * cols, 62), base = row_classes::values(nout * cols, 63);
            auto mwb = g.adopt(mw.data(), mw.size()), xwb = g.adopt(xw.data(), xw.size() * 4);
            auto yw = g.alloc(nout * cols * 4), ya = g.alloc(nout * cols * 4), yb = g.alloc(nout * cols * 4), yc = g.alloc(nout * cols * 4);
            const WeightView wv{quant::GGML_TYPE_Q8_0, {mwb.get(), 0}};
            std::vector<std::vector<float>> one(cols), one_added(cols);
            for (size_t r = 0; r < cols; ++r) {
                const RowRun run[1] = {{1, 1}};
                g.matmul(wv, {xwb.get(), r * w}, {y1.get(), 0}, w, nout, 1, {run, 1}, Dtype::f16);
                one[r] = read(g, *y1, nout);
                g.write(*y1, 0, base.data() + r * nout, nout * 4);
                g.matmul_add(wv, {xwb.get(), r * w}, {y1.get(), 0}, w, nout, 1, {run, 1}, Dtype::f16);
                one_added[r] = read(g, *y1, nout);
            }
            for (size_t n = 1; n <= cols; ++n) {
                const RowRun run[1] = {{n, 1}};
                g.matmul(wv, {xwb.get(), 0}, {yw.get(), 0}, w, nout, n, {run, 1}, Dtype::f16);
                const std::vector<float> y = read(g, *yw, nout * n);
                for (size_t r = 0; r < n; ++r) alone = alone && std::memcmp(y.data() + r * nout, one[r].data(), nout * 4) == 0;
            }
            for (size_t n : {size_t(1), size_t(2), size_t(3), size_t(15), size_t(16), size_t(17), size_t(31), size_t(32), size_t(33), size_t(48), size_t(49), size_t(63), size_t(64)}) {
                const RowRun run[1] = {{n, 1}};
                g.write(*ya, 0, base.data(), nout * n * 4);
                g.matmul_add(wv, {xwb.get(), 0}, {ya.get(), 0}, w, nout, n, {run, 1}, Dtype::f16);
                const std::vector<float> y = read(g, *ya, nout * n);
                for (size_t r = 0; r < n; ++r) added = added && std::memcmp(y.data() + r * nout, one_added[r].data(), nout * 4) == 0;
                g.matmul_group({Projection(wv, {yw.get(), 0}, nout), Projection(wv, {yb.get(), 0}, nout), Projection(wv, {yc.get(), 0}, nout)},
                               {xwb.get(), 0}, w, n, {run, 1}, Dtype::f16);
                for (BufferPtr* out : {&yw, &yb, &yc}) {
                    const std::vector<float> yg = read(g, **out, nout * n);
                    for (size_t r = 0; r < n; ++r) grouped = grouped && std::memcmp(yg.data() + r * nout, one[r].data(), nout * 4) == 0;
                }
            }
        }
        require(alone, "every decode column of 1 to 64 generated rows is the column alone, bit for bit, rows 4096 and 1280 wide");
        require(added, "every decode column of the residual add at 1 to 64 rows is the column alone, bit for bit");
        require(grouped, "every decode column of a group of three projections at 1 to 64 rows is the column alone, bit for bit");
    }
    // A prompt row gives the same bits in calls of 1 to 21 rows, wherever the tile's edges fall.
    bool tiled = true;
    for (size_t r = 0; r < rows; ++r) {
        const RowRun run[1] = {{1, 2}};
        g.matmul({quant::GGML_TYPE_Q8_0, {mb.get(), 0}}, {xsb.get(), r * width}, {y1.get(), 0}, width, nout, 1, {run, 1}, Dtype::f16);
        single[r] = read(g, *y1, nout);
    }
    for (size_t n = 1; n <= rows; ++n) {
        const RowRun run[1] = {{n, 2}};
        g.matmul({quant::GGML_TYPE_Q8_0, {mb.get(), 0}}, {xsb.get(), 0}, {ys.get(), 0}, width, nout, n, {run, 1}, Dtype::f16);
        const std::vector<float> y = read(g, *ys, nout * n);
        for (size_t r = 0; r < n; ++r) tiled = tiled && std::memcmp(y.data() + r * nout, single[r].data(), nout * 4) == 0;
    }
    require(tiled, "a prompt row gives the same bits in calls of 1 to 21 rows");
    // Every pair of extents of one class: the same bits through a Q8_0 product in the F16 class.
    std::vector<std::vector<float>> out;
    const size_t four = 4;
    for (size_t e : row_classes::kExtents) {
        const RowRun run[1] = {{four, e}};
        g.matmul({quant::GGML_TYPE_Q8_0, {mb.get(), 0}}, {xsb.get(), 0}, {ys.get(), 0}, width, nout, four, {run, 1}, Dtype::f16);
        out.push_back(read(g, *ys, nout * four));
    }
    require(row_classes::same_within_classes(g, out, "a Q8_0 product in the F16 class") > 0, "the F16 row-class check compared pairs");
}

// A prompt's rows whose histories cross a change in the attention's chunks (256 positions, and 4096 where the count stops at 16), and at 16K where 40 rows fill three tiles: each row the same bits as that row alone.
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
        g.attention({qb.get(), 0}, 0, &v, 1, {ob.get(), 0}, n_head, n_head_kv, head_dim);
        const std::vector<float> together = read(g, *ob, rows * qw);
        bool alone = true;
        for (size_t b = 0; b < rows; ++b) {
            KVView one = all;
            one.length = hist + b;
            one.nq = 1;
            g.attention({qb.get(), b * qw}, 0, &one, 1, {o1.get(), 0}, n_head, n_head_kv, head_dim);
            alone = alone && std::memcmp(read(g, *o1, qw).data(), together.data() + b * qw, qw * 4) == 0;
        }
        require(alone, "attention rows across a chunk change after a history of " + std::to_string(hist) + " are each the row alone, bit for bit");
        s.abort();
    }
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
    require(pairs > 0, "the row-class check compared pairs");
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
        int16_products(*g);
        host_output(*g);
        attention_chunks(*g);
        failure();
    } catch (const std::exception& e) {
        std::cout << "FAIL threw: " << e.what() << "\n";
        ++failures;
    }
    std::cout << (failures ? "backend-rocm: FAIL\n" : "backend-rocm: PASS\n");
    return failures ? 1 : 0;
}
