#pragma once
#include <algorithm>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>
#include <cstdlib>

#include "backends/backend.hpp"
#include "core/fp16.hpp"
#include "format/gguf.hpp"
#include "quant/quant.hpp"

// Platform-agnostic intrinsics headers. MSVC uses <intrin.h> for __cpuid;
// GCC/Clang use <cpuid.h> (for __get_cpuid) and <immintrin.h> for AVX2.
// <immintrin.h> exists on all three; only the cpuid call differs.
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#include <immintrin.h>

namespace backend {

// CPU implementation of the Backend interface. Uses AVX2 fused dequant+FMA for
// Q8_0 matmuls when the host supports it, otherwise a scalar fallback.
class CpuBackend : public Backend {
public:
    CpuBackend() {
        unsigned hw = std::thread::hardware_concurrency();
        threads_ = (hw > 0) ? (int)hw : 4;
        if (threads_ > 64) threads_ = 64;
        avx2_ = has_avx2();   // detect once, not per row dot
        f16c_ = has_f16c();
        start_pool();
    }

    ~CpuBackend() override { stop_pool(); }

    void set_threads(int n) override {
        int t = (n > 0) ? n : 1;
        if (t == threads_) return;
        stop_pool();
        threads_ = t;
        start_pool();
    }

    int threads_available() const override { return threads_; }

    float dot_q8_0(const uint8_t* row, const float* x, size_t nblocks) override {
        return dot_row_impl(row, x, nblocks);
    }

    void matvec_q8_0(const uint8_t* data, const float* x, float* out,
                     size_t nblocks, size_t nout) override {
        const int nt = threads_;
        // Small problems are not worth waking the pool.
        if (nt <= 1 || nout < (size_t)nt * 8) {
            for (size_t o = 0; o < nout; o++)
                out[o] = dot_row_impl(data + o * nblocks * gguf::Q8_0_TYPESIZE, x, nblocks);
            return;
        }
        const size_t chunk = (nout + (size_t)nt - 1) / (size_t)nt;
        run_parallel([&](int w) {
            size_t start = (size_t)w * chunk;
            size_t end = std::min(nout, start + chunk);
            for (size_t o = start; o < end; o++)
                out[o] = dot_row_impl(data + o * nblocks * gguf::Q8_0_TYPESIZE, x, nblocks);
        });
    }

    // Run fn(0..threads_-1) across the pool: worker 0 is the calling thread, so
    // a single-threaded backend never touches the pool at all. Blocks until
    // every participant has returned, which is what lets the job be referenced
    // rather than copied.
    template <class F>
    void run_parallel(F&& fn) {
        if (threads_ <= 1) { fn(0); return; }
        std::function<void(int)> job(std::ref(fn));   // ref -> no heap alloc
        {
            std::lock_guard<std::mutex> lk(m_);
            job_ = &job;
            pending_ = threads_ - 1;
            epoch_++;
        }
        cv_work_.notify_all();
        fn(0);
        std::unique_lock<std::mutex> lk(m_);
        cv_done_.wait(lk, [&] { return pending_ == 0; });
        job_ = nullptr;
    }

    void matmul(uint32_t ggml_type, const uint8_t* data, const float* X, float* Y,
                size_t nin, size_t nout, size_t nbatch) override {
        // Q8_0 keeps its fused dequant+FMA row dot for the single-column case,
        // which is the decode path and is bandwidth bound rather than load
        // bound, so the extra dequant buffer would buy nothing there.
        if (nbatch == 1 && ggml_type == gguf::GGML_TYPE_Q8_0) {
            matvec_q8_0(data, X, Y, nin / gguf::Q8_0_BLOCK, nout);
            return;
        }
        if (nbatch == 1 && ggml_type == gguf::GGML_TYPE_Q4_K) {
            const size_t nb = nin / gguf::Q4_K_BLOCK;
            const size_t rowbytes = nb * gguf::Q4_K_TYPESIZE;
            const int nt = threads_;
            if (nt <= 1 || nout < (size_t)nt * 8) {
                for (size_t o = 0; o < nout; o++)
                    Y[o] = dot_row_q4_K(data + o * rowbytes, X, nb);
                return;
            }
            const size_t chunk = (nout + (size_t)nt - 1) / (size_t)nt;
            run_parallel([&](int w) {
                const size_t s0 = (size_t)w * chunk;
                const size_t e0 = std::min(nout, s0 + chunk);
                for (size_t o = s0; o < e0; o++)
                    Y[o] = dot_row_q4_K(data + o * rowbytes, X, nb);
            });
            return;
        }
        const quant::QuantType* qt = quant::Registry::instance().get(ggml_type);
        const bool f32 = ggml_type == gguf::GGML_TYPE_F32;
        if (!f32 && (!qt || !qt->dequantize || qt->block_size == 0))
            throw std::runtime_error("backend: no dequantizer for tensor type");
        const size_t nblocks = f32 ? 0 : nin / qt->block_size;
        const size_t rowbytes = f32 ? nin * sizeof(float) : nblocks * qt->type_size;

        // Rows dequantized together before walking the batch. This is the
        // fused kernel's row width, not a tuning constant: dot_f32_x4 shares
        // one activation load across exactly 4 rows, and grouping more buys
        // nothing while enlarging the dequantized working set.
        // A cache-BYTE budget was tried instead and measured worse at every
        // size (64/128/196/256 KB gave 22.37/22.04/23.68/21.12 tok/s against
        // 24.04 for a flat 4), because the knee follows the kernel width
        // rather than the working-set size.
        const size_t RB = (size_t)DOT_ROWS;

        auto do_rows = [&](int w, size_t o0, size_t o1) {
            std::vector<float>& buf = rowbuf_[(size_t)w];
            if (!f32 && buf.size() < RB * nin) buf.assign(RB * nin, 0.0f);
            for (size_t o = o0; o < o1; o += RB) {
                const size_t nr = std::min(RB, o1 - o);
                const float* r = f32 ? (const float*)(data + o * rowbytes) : buf.data();
                if (!f32)
                    for (size_t k = 0; k < nr; k++)
                        qt->dequantize(data + (o + k) * rowbytes, buf.data() + k * nin, nblocks);
                size_t b = 0;
                // Three activation columns at a time where the row block is
                // full, so weight loads amortise across all three.
                for (; b + 3 <= nbatch && nr == RB; b += 3) {
                    const float* xa = X + b * nin;
                    const float* xb3 = X + (b + 1) * nin;
                    const float* xc3 = X + (b + 2) * nin;
                    float* ya = Y + b * nout + o;
                    float* yb = Y + (b + 1) * nout + o;
                    float* yc = Y + (b + 2) * nout + o;
                    for (size_t k = 0; k + 4 <= nr; k += 4)
                        dot_f32_x4x3(r + k * nin, nin, xa, xb3, xc3, nin,
                                     ya + k, yb + k, yc + k);
                }
                for (; b + 2 <= nbatch && nr == RB; b += 2) {
                    const float* xa = X + b * nin;
                    const float* xb2 = X + (b + 1) * nin;
                    float* ya = Y + b * nout + o;
                    float* yb = Y + (b + 1) * nout + o;
                    for (size_t k = 0; k + 4 <= nr; k += 4)
                        dot_f32_x4x2(r + k * nin, nin, xa, xb2, nin, ya + k, yb + k);
                }
                for (; b < nbatch; b++) {
                    const float* x = X + b * nin;
                    float* y = Y + b * nout + o;
                    size_t k = 0;
                    for (; k + 4 <= nr; k += 4)
                        dot_f32_x4(r + k * nin, nin, x, nin, y + k);
                    for (; k < nr; k++)
                        y[k] = dot_f32(r + k * nin, x, nin);
                }
            }
        };
        const int nt = threads_;
        if (nt <= 1 || nout < (size_t)nt * 4) { do_rows(0, 0, nout); return; }
        const size_t per = (nout + (size_t)nt - 1) / (size_t)nt;
        const size_t chunk = (per + RB - 1) / RB * RB;
        run_parallel([&](int w) {
            const size_t s = (size_t)w * chunk;
            const size_t e = std::min(nout, s + chunk);
            if (s < e) do_rows(w, s, e);
        });
    }

    // Rows fused per activation load: the width dot_f32_x4 handles.
    static const int DOT_ROWS = 4;


    // Four rows against THREE activation columns: 7 loads per 12 FMAs, a ratio
    // of 0.58 against 0.75 for the two-column form. This needs 12 accumulators
    // plus 3 activation registers, which is exactly the 16 YMM budget, so it
    // sits on the spill boundary and is only worth keeping if measured faster.
    static void dot_f32_x4x3(const float* r, size_t stride,
                             const float* xa, const float* xb, const float* xc,
                             size_t n, float* outa, float* outb, float* outc) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
        __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
        __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
        const float* r0 = r;
        const float* r1 = r + stride;
        const float* r2 = r + 2 * stride;
        const float* r3 = r + 3 * stride;
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            const __m256 xv = _mm256_loadu_ps(xa + i);
            const __m256 yv = _mm256_loadu_ps(xb + i);
            const __m256 zv = _mm256_loadu_ps(xc + i);
            __m256 w = _mm256_loadu_ps(r0 + i);
            a0 = _mm256_fmadd_ps(w, xv, a0); b0 = _mm256_fmadd_ps(w, yv, b0); c0 = _mm256_fmadd_ps(w, zv, c0);
            w = _mm256_loadu_ps(r1 + i);
            a1 = _mm256_fmadd_ps(w, xv, a1); b1 = _mm256_fmadd_ps(w, yv, b1); c1 = _mm256_fmadd_ps(w, zv, c1);
            w = _mm256_loadu_ps(r2 + i);
            a2 = _mm256_fmadd_ps(w, xv, a2); b2 = _mm256_fmadd_ps(w, yv, b2); c2 = _mm256_fmadd_ps(w, zv, c2);
            w = _mm256_loadu_ps(r3 + i);
            a3 = _mm256_fmadd_ps(w, xv, a3); b3 = _mm256_fmadd_ps(w, yv, b3); c3 = _mm256_fmadd_ps(w, zv, c3);
        }
        alignas(32) float t[8];
        const __m256* accs[12] = { &a0, &a1, &a2, &a3, &b0, &b1, &b2, &b3, &c0, &c1, &c2, &c3 };
        const float* xs[3] = { xa, xb, xc };
        float* outs[3] = { outa, outb, outc };
        for (int k = 0; k < 12; k++) {
            _mm256_store_ps(t, *accs[k]);
            float v = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
            const size_t row = (size_t)(k & 3);
            const int col = k / 4;
            for (size_t j = i; j < n; j++) v += r[row * stride + j] * xs[col][j];
            outs[col][row] = v;
        }
    }

    // Four rows against TWO activation columns in one pass.
    // dot_f32_x4 costs 5 loads per 4 FMAs (one activation, four weights).
    // Holding two activation columns makes it 6 loads per 8 FMAs, so the
    // load:FMA ratio drops from 1.25 to 0.75 and the kernel stops being load
    // bound. Eight accumulators plus two activation registers still fit the
    // 16 YMM registers, so nothing spills.
    static void dot_f32_x4x2(const float* r, size_t stride,
                             const float* xa, const float* xb, size_t n,
                             float* outa, float* outb) {
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
        __m256 b0 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
        __m256 b2 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
        const float* r0 = r;
        const float* r1 = r + stride;
        const float* r2 = r + 2 * stride;
        const float* r3 = r + 3 * stride;
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            const __m256 xv = _mm256_loadu_ps(xa + i);
            const __m256 yv = _mm256_loadu_ps(xb + i);
            __m256 w = _mm256_loadu_ps(r0 + i);
            a0 = _mm256_fmadd_ps(w, xv, a0); b0 = _mm256_fmadd_ps(w, yv, b0);
            w = _mm256_loadu_ps(r1 + i);
            a1 = _mm256_fmadd_ps(w, xv, a1); b1 = _mm256_fmadd_ps(w, yv, b1);
            w = _mm256_loadu_ps(r2 + i);
            a2 = _mm256_fmadd_ps(w, xv, a2); b2 = _mm256_fmadd_ps(w, yv, b2);
            w = _mm256_loadu_ps(r3 + i);
            a3 = _mm256_fmadd_ps(w, xv, a3); b3 = _mm256_fmadd_ps(w, yv, b3);
        }
        alignas(32) float t[8];
        const __m256* accs[8] = { &a0, &a1, &a2, &a3, &b0, &b1, &b2, &b3 };
        for (int k = 0; k < 8; k++) {
            _mm256_store_ps(t, *accs[k]);
            float v = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
            const size_t row = (size_t)(k & 3);
            const float* xt = (k < 4) ? xa : xb;
            for (size_t j = i; j < n; j++) v += r[row * stride + j] * xt[j];
            ((k < 4) ? outa : outb)[row] = v;
        }
    }

    // Four dots against a SHARED activation vector, in one pass.
    // Calling dot_f32 four times costs 2 loads per FMA (one weight, one
    // activation), and Zen3 sustains 2 loads/cycle against 2 FMAs/cycle, so
    // that kernel is load bound at half of FMA peak. Loading x once and reusing
    // it across 4 rows costs 5 loads per 4 FMAs instead of 8.
    static void dot_f32_x4(const float* r, size_t stride, const float* x,
                           size_t n, float* out) {
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
        const float* r0 = r;
        const float* r1 = r + stride;
        const float* r2 = r + 2 * stride;
        const float* r3 = r + 3 * stride;
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            const __m256 xv = _mm256_loadu_ps(x + i);
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + i), xv, s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + i), xv, s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + i), xv, s2);
            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + i), xv, s3);
        }
        alignas(32) float t[8];
        const __m256* acc[4] = { &s0, &s1, &s2, &s3 };
        for (int k = 0; k < 4; k++) {
            _mm256_store_ps(t, *acc[k]);
            float v = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
            for (size_t j = i; j < n; j++) v += r[(size_t)k * stride + j] * x[j];
            out[k] = v;
        }
    }

    // Four independent accumulators. With a single accumulator every FMA
    // depends on the previous one, so the loop runs at FMA LATENCY (about 4
    // cycles) instead of FMA throughput (about 0.5), which is most of an order
    // of magnitude on this path. Splitting the chain also changes the
    // summation order, so results differ in the last bits.
    static float dot_f32(const float* a, const float* b, size_t n) {
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 32 <= n; i += 32) {
            s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i +  0), _mm256_loadu_ps(b + i +  0), s0);
            s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i +  8), _mm256_loadu_ps(b + i +  8), s1);
            s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
            s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
        }
        __m256 acc = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
        for (; i + 8 <= n; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float out = _mm_cvtss_f32(s);
        for (; i < n; i++) out += a[i] * b[i];
        return out;
    }

    void parallel_for(int n, const std::function<void(int)>& fn) override {
        if (n <= 0) return;
        const int nt = std::min(threads_, n);
        if (nt <= 1) { for (int i = 0; i < n; i++) fn(i); return; }
        const int chunk = (n + nt - 1) / nt;
        run_parallel([&](int w) {
            const int start = w * chunk;
            const int end = std::min(n, start + chunk);
            for (int i = start; i < end; i++) fn(i);
        });
    }

    void rms_norm(float* dst, const float* src, const float* w, size_t n, float eps) override {
        if (avx2_) {
            // Sum of squares (vectorized), then a vectorized weighted scale.
            __m256 acc = _mm256_setzero_ps();
            size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m256 x = _mm256_loadu_ps(src + i);
                acc = _mm256_fmadd_ps(x, x, acc);
            }
            __m256 h = _mm256_hadd_ps(acc, acc);
            h = _mm256_hadd_ps(h, h);
            float s = _mm256_cvtss_f32(h);
            s += _mm_cvtss_f32(_mm256_extractf128_ps(h, 1));
            for (; i < n; i++) s += src[i] * src[i];
            float r = 1.0f / std::sqrt(s / (float)n + eps);
            __m256 rv = _mm256_set1_ps(r);
            i = 0;
            for (; i + 8 <= n; i += 8) {
                __m256 x = _mm256_loadu_ps(src + i);
                __m256 wv = _mm256_loadu_ps(w + i);
                _mm256_storeu_ps(dst + i, _mm256_mul_ps(x, _mm256_mul_ps(rv, wv)));
            }
            for (; i < n; i++) dst[i] = src[i] * r * w[i];
            return;
        }
        float s = 0.0f;
        for (size_t i = 0; i < n; i++) s += src[i] * src[i];
        float r = 1.0f / std::sqrt(s / (float)n + eps);
        for (size_t i = 0; i < n; i++) dst[i] = src[i] * r * w[i];
    }

    void rope(float* x, const float* cos, const float* sin, int half) override {
        if (avx2_) {
            int i = 0;
            for (; i + 8 <= half; i += 8) {
                __m256 xa = _mm256_loadu_ps(x + i);
                __m256 xb = _mm256_loadu_ps(x + i + half);
                __m256 c = _mm256_loadu_ps(cos + i);
                __m256 s = _mm256_loadu_ps(sin + i);
                __m256 na = _mm256_fnmadd_ps(xb, s, _mm256_mul_ps(xa, c));
                __m256 nb = _mm256_fmadd_ps(xa, s, _mm256_mul_ps(xb, c));
                _mm256_storeu_ps(x + i, na);
                _mm256_storeu_ps(x + i + half, nb);
            }
            for (; i < half; i++) {
                int a = i, b = i + half;
                float xa = x[a], xb = x[b];
                x[a] = xa * cos[i] - xb * sin[i];
                x[b] = xa * sin[i] + xb * cos[i];
            }
            return;
        }
        for (int i = 0; i < half; i++) {
            int a = i, b = i + half;
            float xa = x[a], xb = x[b];
            x[a] = xa * cos[i] - xb * sin[i];
            x[b] = xa * sin[i] + xb * cos[i];
        }
    }

private:
    int threads_ = 1;
    bool avx2_ = false;
    bool f16c_ = false;

    // Persistent worker pool. The previous code created and joined
    // std::threads on every matvec call, which is once per matmul per layer per
    // token; at 36 layers that is thousands of thread creations per token.
    std::vector<std::thread> pool_;
    // Per-worker dequantized weight-row scratch for matmul_q8_0.
    std::vector<std::vector<float>> rowbuf_;
    std::mutex m_;
    std::condition_variable cv_work_, cv_done_;
    const std::function<void(int)>* job_ = nullptr;
    unsigned epoch_ = 0;
    int pending_ = 0;
    bool stop_ = false;

    void start_pool() {
        rowbuf_.assign((size_t)(threads_ > 0 ? threads_ : 1), std::vector<float>());
        stop_ = false;
        epoch_ = 0;
        pending_ = 0;
        for (int i = 1; i < threads_; i++)
            pool_.emplace_back([this, i] { worker(i); });
    }

    void stop_pool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            epoch_++;
        }
        cv_work_.notify_all();
        for (auto& t : pool_) if (t.joinable()) t.join();
        pool_.clear();
    }

    void worker(int idx) {
        unsigned seen = 0;
        for (;;) {
            const std::function<void(int)>* job = nullptr;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_work_.wait(lk, [&] { return stop_ || epoch_ != seen; });
                if (stop_) return;
                seen = epoch_;
                job = job_;
            }
            if (job) (*job)(idx);
            {
                std::lock_guard<std::mutex> lk(m_);
                if (--pending_ == 0) cv_done_.notify_one();
            }
        }
    }

    // F16C (hardware half<->float). Present on every AVX2 part in practice,
    // but detected separately because the ISA bits are independent.
    static bool has_f16c() {
#if defined(_MSC_VER)
        int info[4];
        __cpuid(info, 1);
        return (info[2] & (1 << 29)) != 0;   // ECX bit 29 = F16C
#elif defined(__GNUC__) || defined(__clang__)
        unsigned eax, ebx, ecx, edx;
        if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
        return (ecx & (1u << 29)) != 0;
#else
        return false;
#endif
    }

    static bool has_avx2() {
#if defined(_MSC_VER)
        int info[4];
        __cpuid(info, 0);
        int maxid = info[0];
        if (maxid < 7) return false;
        __cpuid(info, 7);
        return (info[1] & (1 << 5)) != 0; // EBX bit 5 = AVX2
#elif defined(__GNUC__) || defined(__clang__)
        unsigned eax, ebx, ecx, edx;
        if (!__get_cpuid(0, &eax, &ebx, &ecx, &edx)) return false;
        if (eax < 7) return false;
        __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
        return (ebx & (1u << 5)) != 0; // EBX bit 5 = AVX2
#else
        return false;
#endif
    }

    // Dot product of one Q8_0 row (nblocks blocks, nin = nblocks*32) with x.
    // Uses an AVX2 fused dequant+FMA path when available, else scalar.
    // Fused Q4_K row dot. The generic path dequantizes a whole row into f32
    // scratch and then dots it, which is why Q4_K decoded at half the speed of
    // Q8_0 despite a smaller file: it is compute bound on unpacking, not
    // bandwidth bound.
    //
    // No dequantized value is ever materialised here. A Q4_K sub-block value is
    // d*q - m with q a 4-bit unsigned nibble, so its contribution to the dot is
    //     sum_l (d*q_l - m) * x_l  =  d * sum_l(q_l * x_l)  -  m * sum_l(x_l)
    // and the two sums are plain FMA reductions over the nibbles and over x.
    float dot_row_q4_K(const uint8_t* row, const float* x, size_t nblocks) {
        float acc = 0.0f;
        for (size_t b = 0; b < nblocks; b++) {
            const uint8_t* p = row + b * gguf::Q4_K_TYPESIZE;
            const float d    = half_to_float((uint16_t)(p[0] | ((uint16_t)p[1] << 8)));
            const float dmin = half_to_float((uint16_t)(p[2] | ((uint16_t)p[3] << 8)));
            const uint8_t* sc = p + 4;
            const uint8_t* qs = p + 16;
            const float* xp = x + b * gguf::Q4_K_BLOCK;

            int is = 0;
            for (int j = 0; j < (int)gguf::Q4_K_BLOCK; j += 64) {
                uint8_t s, mm;
                quant::get_scale_min_k4(is + 0, sc, &s, &mm);
                const float d1 = d * (float)s, m1 = dmin * (float)mm;
                quant::get_scale_min_k4(is + 1, sc, &s, &mm);
                const float d2 = d * (float)s, m2 = dmin * (float)mm;

                if (avx2_) {
                    const __m128i lo_mask = _mm_set1_epi8(0x0F);
                    __m128i raw0 = _mm_loadu_si128((const __m128i*)(qs +  0));
                    __m128i raw1 = _mm_loadu_si128((const __m128i*)(qs + 16));
                    // low nibbles -> first 32 values, high nibbles -> next 32
                    __m128i lo0 = _mm_and_si128(raw0, lo_mask);
                    __m128i lo1 = _mm_and_si128(raw1, lo_mask);
                    __m128i hi0 = _mm_and_si128(_mm_srli_epi16(raw0, 4), lo_mask);
                    __m128i hi1 = _mm_and_si128(_mm_srli_epi16(raw1, 4), lo_mask);

                    __m256 qx_lo = _mm256_setzero_ps(), sx_lo = _mm256_setzero_ps();
                    __m256 qx_hi = _mm256_setzero_ps(), sx_hi = _mm256_setzero_ps();
                    const __m128i* lohalves[2] = { &lo0, &lo1 };
                    const __m128i* hihalves[2] = { &hi0, &hi1 };
                    for (int h = 0; h < 2; h++) {
                        const __m128i L = *lohalves[h];
                        const __m128i H = *hihalves[h];
                        for (int q = 0; q < 2; q++) {
                            const __m128i Lp = q ? _mm_srli_si128(L, 8) : L;
                            const __m128i Hp = q ? _mm_srli_si128(H, 8) : H;
                            const __m256 fl = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(Lp));
                            const __m256 fh = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(Hp));
                            const __m256 xl = _mm256_loadu_ps(xp + h * 16 + q * 8);
                            const __m256 xh = _mm256_loadu_ps(xp + 32 + h * 16 + q * 8);
                            qx_lo = _mm256_fmadd_ps(fl, xl, qx_lo);
                            sx_lo = _mm256_add_ps(xl, sx_lo);
                            qx_hi = _mm256_fmadd_ps(fh, xh, qx_hi);
                            sx_hi = _mm256_add_ps(xh, sx_hi);
                        }
                    }
                    acc += d1 * hsum256(qx_lo) - m1 * hsum256(sx_lo);
                    acc += d2 * hsum256(qx_hi) - m2 * hsum256(sx_hi);
                } else {
                    float qx1 = 0, sx1 = 0, qx2 = 0, sx2 = 0;
                    for (int l = 0; l < 32; l++) {
                        const float xa = xp[l], xb = xp[l + 32];
                        qx1 += (float)(qs[l] & 0xF) * xa; sx1 += xa;
                        qx2 += (float)(qs[l] >> 4)  * xb; sx2 += xb;
                    }
                    acc += d1 * qx1 - m1 * sx1;
                    acc += d2 * qx2 - m2 * sx2;
                }
                xp += 64; qs += 32; is += 2;
            }
        }
        return acc;
    }

    static float hsum256(__m256 v) {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        return _mm_cvtss_f32(s);
    }

    // f16 -> f32 using hardware F16C where present.
    float half_to_float(uint16_t h) const {
        if (f16c_) return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
        return f16_to_f32(h);
    }

    float dot_row_impl(const uint8_t* row, const float* x, size_t nblocks) {
        if (avx2_) {
            // Four independent accumulators. A single chained accumulator
            // serialised the loop at FMA latency, which also capped how many
            // loads could be in flight; decode is bandwidth bound, so fewer
            // outstanding loads means less memory-level parallelism and less
            // achieved bandwidth.
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
            // Software prefetch of the weight stream was measured here and
            // made no difference (4.10/4.14 against 4.12/4.12 tok/s): the
            // hardware prefetcher already keeps up with these sequential
            // streams. Not reinstated.
            for (size_t b = 0; b < nblocks; b++) {
                const uint8_t* y = row + b * gguf::Q8_0_TYPESIZE;
                // Hardware f16 convert. The scalar f16_to_f32 is a branchy
                // function (zero, subnormal, inf/nan cases) called once per
                // 34 bytes of weights, which is a lot of unpredictable control
                // flow in a loop whose job is to keep loads in flight.
                const uint16_t h = (uint16_t)(y[0] | ((uint16_t)y[1] << 8));
                __m256 dv;
                if (f16c_) {
                    dv = _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
                } else {
                    dv = _mm256_set1_ps(f16_to_f32(h));
                }
                const __m128i* p = (const __m128i*)(y + 2);
                __m128i a = _mm_loadu_si128(p);
                __m128i c = _mm_loadu_si128(p + 1);
                // sign-extend the 32 int8 to 4 groups of 8 int32 -> float
                __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a));
                __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(a, 8)));
                __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c));
                __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(c, 8)));
                const float* xp = x + b * gguf::Q8_0_BLOCK;
                s0 = _mm256_fmadd_ps(_mm256_mul_ps(f0, dv), _mm256_loadu_ps(xp), s0);
                s1 = _mm256_fmadd_ps(_mm256_mul_ps(f1, dv), _mm256_loadu_ps(xp + 8), s1);
                s2 = _mm256_fmadd_ps(_mm256_mul_ps(f2, dv), _mm256_loadu_ps(xp + 16), s2);
                s3 = _mm256_fmadd_ps(_mm256_mul_ps(f3, dv), _mm256_loadu_ps(xp + 24), s3);
            }
            __m256 acc = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            return _mm_cvtss_f32(s);
        }
        float acc = 0.0f;
        for (size_t b = 0; b < nblocks; b++) {
            const uint8_t* y = row + b * gguf::Q8_0_TYPESIZE;
            uint16_t d16 = (uint16_t)(y[0] | ((uint16_t)y[1] << 8));
            float d = f16_to_f32(d16);
            for (int j = 0; j < gguf::Q8_0_BLOCK; j++)
                acc += (float)(int8_t)y[2 + j] * d * x[b * gguf::Q8_0_BLOCK + j];
        }
        return acc;
    }
};

// Default CPU backend factory.
inline BackendPtr make_cpu_backend() { return std::make_shared<CpuBackend>(); }

} // namespace backend
