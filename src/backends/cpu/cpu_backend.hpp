#pragma once
#include <algorithm>
#include <cmath>
#include <thread>
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
    }

    void set_threads(int n) override { threads_ = (n > 0) ? n : 1; }
    int threads_available() const override { return threads_; }

    float dot_q8_0(const uint8_t* row, const float* x, size_t nblocks) override {
        return dot_row_impl(row, x, nblocks);
    }

    void matvec_q8_0(const uint8_t* data, const float* x, float* out,
                     size_t nblocks, size_t nout) override {
        int nt = threads_;
        // Small problems aren't worth thread overhead.
        if (nt <= 1 || nout < (size_t)nt * 8) {
            for (size_t o = 0; o < nout; o++) {
                const uint8_t* row = data + o * nblocks * gguf::Q8_0_TYPESIZE;
                out[o] = dot_q8_0(row, x, nblocks);
            }
            return;
        }
        if (nt > (int)nout) nt = (int)nout;

        std::vector<std::thread> workers;
        workers.reserve((size_t)nt);
        size_t chunk = (nout + (size_t)nt - 1) / (size_t)nt;
        for (int w = 0; w < nt; w++) {
            size_t start = (size_t)w * chunk;
            size_t end = std::min(nout, start + chunk);
            if (start >= end) break;
            workers.emplace_back([&, start, end]() {
                for (size_t o = start; o < end; o++) {
                    const uint8_t* row = data + o * nblocks * gguf::Q8_0_TYPESIZE;
                    out[o] = dot_q8_0(row, x, nblocks);
                }
            });
        }
        for (auto& th : workers) th.join();
    }

    void rms_norm(float* dst, const float* src, const float* w, size_t n, float eps) override {
        float s = 0.0f;
        for (size_t i = 0; i < n; i++) s += src[i] * src[i];
        float r = 1.0f / std::sqrt(s / (float)n + eps);
        for (size_t i = 0; i < n; i++) dst[i] = src[i] * r * w[i];
    }

    void rope(float* x, const float* cos, const float* sin, int half) override {
        for (int i = 0; i < half; i++) {
            int a = i, b = i + half;
            float xa = x[a], xb = x[b];
            x[a] = xa * cos[i] - xb * sin[i];
            x[b] = xa * sin[i] + xb * cos[i];
        }
    }

private:
    int threads_ = 1;

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
        if (has_avx2()) {
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
