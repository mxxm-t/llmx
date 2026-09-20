// Does paging the KV cache cost CPU attention anything?
//
// vLLM pages because uniform blocks remove fragmentation and make prefix
// sharing a refcount. It accepts block-table indirection because GPU attention
// is bandwidth bound regardless. Our decode attention is 20.6% of a token and
// memory bound at about 21 GB/s per thread, so the indirection is not
// obviously free here. This measures it before the design is chosen.
//
// Two honesty requirements, both learned the hard way in this project:
//   - The KV must be COLD. An earlier attention benchmark reused one buffer
//     and reported 72 GB/s, above this machine's DRAM bandwidth, because it
//     was measuring cache. Here a pool of distinct KV sets is rotated so each
//     iteration touches memory the weight stream would have evicted.
//   - The block table must be SHUFFLED. Sequentially allocated blocks are
//     indistinguishable from a contiguous layout and would flatter paging.
//     Real allocation interleaves sequences and reuses freed blocks.
//
// Usage: paged_attn_bench [n_past] [iters] [heads] [kv_heads] [repeats]
// Each repeat times both arms back to back and the median paired ratio is
// reported, so drift lands on both arms and one slow repeat does not decide.
// Build: cl /O2 /arch:AVX2 /EHsc tools/paged_attn_bench.cpp
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#include <immintrin.h>

using Clock = std::chrono::steady_clock;

namespace {

constexpr int kHeadDim = 128;
int kHeads = 16;     // 0.6B; pass 32 for the 8B geometry
int kKvHeads = 8;

float dot(const float* a, const float* b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    __m256 s = _mm256_add_ps(s0, s1);
    __m128 lo = _mm256_castps256_ps128(s), hi = _mm256_extractf128_ps(s, 1);
    __m128 r = _mm_add_ps(lo, hi);
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    float out = _mm_cvtss_f32(r);
    for (; i < n; ++i) out += a[i] * b[i];
    return out;
}

void accumulate(float* acc, const float* v, float w, int n) {
    const __m256 wv = _mm256_set1_ps(w);
    int d = 0;
    for (; d + 8 <= n; d += 8)
        _mm256_storeu_ps(acc + d,
            _mm256_fmadd_ps(wv, _mm256_loadu_ps(v + d), _mm256_loadu_ps(acc + d)));
    for (; d < n; ++d) acc[d] += w * v[d];
}

// One decode step for one head, contiguous history: position t is at
// base + t*head_dim, exactly as HostKVCache lays it out today.
void attend_contiguous(const float* q, const float* K, const float* V,
                       int n_past, float* out, std::vector<float>& scores) {
    const int end = n_past + 1;
    const float scale = 1.0f / std::sqrt((float)kHeadDim);
    float mx = -1e30f;
    for (int t = 0; t < end; ++t) {
        scores[t] = dot(q, K + (size_t)t * kHeadDim, kHeadDim) * scale;
        mx = std::max(mx, scores[t]);
    }
    float sum = 0;
    for (int t = 0; t < end; ++t) { scores[t] = std::exp(scores[t] - mx); sum += scores[t]; }
    std::memset(out, 0, sizeof(float) * kHeadDim);
    for (int t = 0; t < end; ++t)
        accumulate(out, V + (size_t)t * kHeadDim, scores[t] / sum, kHeadDim);
}

// Same arithmetic, paged: logical position t lives in physical block
// table[t / bs] at offset t % bs. Iterated block by block so the block index
// is computed once per block rather than once per position.
void attend_paged(const float* q, const float* pool_k, const float* pool_v,
                  const int* table, int bs, int n_past, float* out,
                  std::vector<float>& scores) {
    const int end = n_past + 1;
    const float scale = 1.0f / std::sqrt((float)kHeadDim);
    float mx = -1e30f;
    for (int t0 = 0; t0 < end; t0 += bs) {
        const float* blk = pool_k + (size_t)table[t0 / bs] * bs * kHeadDim;
        const int n = std::min(bs, end - t0);
        for (int j = 0; j < n; ++j) {
            scores[t0 + j] = dot(q, blk + (size_t)j * kHeadDim, kHeadDim) * scale;
            mx = std::max(mx, scores[t0 + j]);
        }
    }
    float sum = 0;
    for (int t = 0; t < end; ++t) { scores[t] = std::exp(scores[t] - mx); sum += scores[t]; }
    std::memset(out, 0, sizeof(float) * kHeadDim);
    for (int t0 = 0; t0 < end; t0 += bs) {
        const float* blk = pool_v + (size_t)table[t0 / bs] * bs * kHeadDim;
        const int n = std::min(bs, end - t0);
        for (int j = 0; j < n; ++j)
            accumulate(out, blk + (size_t)j * kHeadDim, scores[t0 + j] / sum, kHeadDim);
    }
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    const int n_past = argc > 1 ? atoi(argv[1]) : 840;
    const int iters = argc > 2 ? atoi(argv[2]) : 200;
    if (argc > 3) kHeads = atoi(argv[3]);
    if (argc > 4) kKvHeads = atoi(argv[4]);
    const int repeats = argc > 5 ? atoi(argv[5]) : 5;
    if (kHeads <= 0 || kKvHeads <= 0 || kHeads % kKvHeads || repeats <= 0) {
        fprintf(stderr, "heads must be a positive multiple of kv_heads; repeats > 0\n");
        return 2;
    }
    const int end = n_past + 1;
    const size_t per_head = (size_t)end * kHeadDim;

    // Rotate over enough distinct KV sets to exceed any cache this machine has.
    const size_t set_bytes = per_head * kKvHeads * sizeof(float) * 2;
    const int sets = (int)std::max<size_t>(4, (256u << 20) / std::max<size_t>(set_bytes, 1));

    std::mt19937 rng(12345);
    std::vector<float> q(kHeadDim);
    for (auto& v : q) v = std::uniform_real_distribution<float>(-1, 1)(rng);
    std::vector<float> out(kHeadDim);
    std::vector<float> scores(end + 512);

    printf("decode attention, n_past=%d, heads %d/%d, %d KV sets (%.0f MB), %d iters/arm, %d repeats\n",
           n_past, kHeads, kKvHeads, sets, sets * set_bytes / 1048576.0, iters, repeats);

    std::vector<std::vector<float>> ck(sets), cv(sets);
    for (int s = 0; s < sets; ++s) {
        ck[s].resize(per_head * kKvHeads);
        cv[s].resize(per_head * kKvHeads);
        for (size_t i = 0; i < ck[s].size(); ++i) {
            ck[s][i] = std::sin((float)(i + s)) * 0.5f;
            cv[s][i] = std::cos((float)(i + s)) * 0.5f;
        }
    }
    const auto run_contig = [&] {
        auto t0 = Clock::now();
        for (int it = 0; it < iters; ++it)
            for (int h = 0; h < kHeads; ++h)
                attend_contiguous(q.data(),
                                  ck[it % sets].data() + (size_t)(h % kKvHeads) * per_head,
                                  cv[it % sets].data() + (size_t)(h % kKvHeads) * per_head,
                                  n_past, out.data(), scores);
        return std::chrono::duration<double>(Clock::now() - t0).count() * 1000.0 / iters;
    };

    // Block tables are shuffled so the walk is not accidentally sequential.
    for (int bs : {16, 32, 64, 128, 256}) {
        const int blocks_per_head = (end + bs - 1) / bs;
        const int total_blocks = blocks_per_head * kKvHeads;
        std::vector<std::vector<float>> pk(sets), pv(sets);
        std::vector<std::vector<int>> tables(sets);
        for (int s = 0; s < sets; ++s) {
            pk[s].assign((size_t)total_blocks * bs * kHeadDim, 0.0f);
            pv[s].assign((size_t)total_blocks * bs * kHeadDim, 0.0f);
            for (size_t i = 0; i < pk[s].size(); ++i) {
                pk[s][i] = std::sin((float)(i + s)) * 0.5f;
                pv[s][i] = std::cos((float)(i + s)) * 0.5f;
            }
            tables[s].resize(total_blocks);
            std::iota(tables[s].begin(), tables[s].end(), 0);
            std::shuffle(tables[s].begin(), tables[s].end(), rng);
        }
        const auto run_paged = [&] {
            auto t1 = Clock::now();
            for (int it = 0; it < iters; ++it)
                for (int h = 0; h < kHeads; ++h)
                    attend_paged(q.data(), pk[it % sets].data(), pv[it % sets].data(),
                                 tables[it % sets].data() + (size_t)(h % kKvHeads) * blocks_per_head,
                                 bs, n_past, out.data(), scores);
            return std::chrono::duration<double>(Clock::now() - t1).count() * 1000.0 / iters;
        };
        std::vector<double> ratios, c_ms, p_ms;
        for (int r = 0; r < repeats; ++r) {
            const double c = run_contig(), p = run_paged();
            c_ms.push_back(c);
            p_ms.push_back(p);
            ratios.push_back(p / c);
        }
        printf("  bs=%-4d contiguous %8.3f  paged %8.3f ms/step  median paired %+6.1f%%  (all:",
               bs, median(c_ms), median(p_ms), (median(ratios) - 1.0) * 100.0);
        for (double x : ratios) printf(" %+.1f%%", (x - 1.0) * 100.0);
        printf(")\n");
    }
    return 0;
}
