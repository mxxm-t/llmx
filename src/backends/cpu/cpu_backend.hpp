#pragma once
#include <algorithm>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <vector>

#include "backends/backend.hpp"
#include "core/fp16.hpp"
#include "format/gguf.hpp"

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

    void matmul_q8_0(const uint8_t* data, const float* X, float* Y,
                     size_t nblocks, size_t nout, size_t nbatch) override {
        if (nbatch == 1) { matvec_q8_0(data, X, Y, nblocks, nout); return; }
        const size_t nin = nblocks * gguf::Q8_0_BLOCK;
        auto do_rows = [&](size_t o0, size_t o1) {
            for (size_t o = o0; o < o1; o++) {
                const uint8_t* row = data + o * nblocks * gguf::Q8_0_TYPESIZE;
                for (size_t b = 0; b < nbatch; b++)
                    Y[b * nout + o] = dot_row_impl(row, X + b * nin, nblocks);
            }
        };
        const int nt = threads_;
        if (nt <= 1 || nout < (size_t)nt * 4) { do_rows(0, nout); return; }
        const size_t chunk = (nout + (size_t)nt - 1) / (size_t)nt;
        run_parallel([&](int w) {
            const size_t s = (size_t)w * chunk;
            const size_t e = std::min(nout, s + chunk);
            if (s < e) do_rows(s, e);
        });
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

    // Persistent worker pool. The previous code created and joined
    // std::threads on every matvec call, which is once per matmul per layer per
    // token; at 36 layers that is thousands of thread creations per token.
    std::vector<std::thread> pool_;
    std::mutex m_;
    std::condition_variable cv_work_, cv_done_;
    const std::function<void(int)>* job_ = nullptr;
    unsigned epoch_ = 0;
    int pending_ = 0;
    bool stop_ = false;

    void start_pool() {
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
    float dot_row_impl(const uint8_t* row, const float* x, size_t nblocks) {
        if (avx2_) {
            __m256 acc = _mm256_setzero_ps();
            for (size_t b = 0; b < nblocks; b++) {
                const uint8_t* y = row + b * gguf::Q8_0_TYPESIZE;
                float d = f16_to_f32((uint16_t)(y[0] | ((uint16_t)y[1] << 8)));
                __m256 dv = _mm256_set1_ps(d);
                const __m128i* p = (const __m128i*)(y + 2);
                __m128i a = _mm_loadu_si128(p);
                __m128i c = _mm_loadu_si128(p + 1);
                // sign-extend the 32 int8 to 4 groups of 8 int32 -> float
                __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a));
                __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(a, 8)));
                __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c));
                __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(c, 8)));
                const float* xp = x + b * gguf::Q8_0_BLOCK;
                acc = _mm256_fmadd_ps(_mm256_mul_ps(f0, dv), _mm256_loadu_ps(xp), acc);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(f1, dv), _mm256_loadu_ps(xp + 8), acc);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(f2, dv), _mm256_loadu_ps(xp + 16), acc);
                acc = _mm256_fmadd_ps(_mm256_mul_ps(f3, dv), _mm256_loadu_ps(xp + 24), acc);
            }
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
