// The orders both modules share, so a value has one order wherever it is summed: the int8 class's one-column Q8_0 row order, which the row kernels (kernels/int16.gpu) and the fused delta step (kernels/f32.gpu) take, and the block's sum tree.
#pragma once
#include "backends/gpu/kernels/port.h"
#include "backends/gpu/kernels/quant.h"

namespace {

constexpr unsigned kQ8 = rq::Q8_0::bytes;   // a Q8_0 block's bytes, whose layout the word loads below take: the f16 scale, then 32 int8 values
static_assert(kQ8 == 34 && rq::Q8_0::values == 32, "the Q8_0 word loads assume 34-byte blocks of 32 values");

// The sum tree of N threads' values in lds[0, N), left in lds[0]: wave 0 folds the levels above a wave in registers, each adding the pair a shared-memory level of barriers would, then the rest by shuffles.
// The caller's barriers come before and after it, so the tree takes two where a barrier a level took two more on a 64-lane wave and three more on a 32-lane one (docs/src/backends-gpu.md).
template <int N>
__device__ inline void block_tree(float* lds) {
    if ((int)threadIdx.x < GPU_WAVE) {
        constexpr int M = N / GPU_WAVE;
        float u[M];
#pragma unroll
        for (int k = 0; k < M; ++k) u[k] = lds[threadIdx.x + GPU_WAVE * k];
#pragma unroll
        for (int h = M / 2; h > 0; h >>= 1)
#pragma unroll
            for (int k = 0; k < h; ++k) u[k] = u[k] + u[k + h];
        float a = u[0];
        for (int h = GPU_WAVE / 2; h > 0; h >>= 1) a += shuffle_down(a, h);
        if (threadIdx.x == 0) lds[0] = a;
    }
}
// The sum of each thread's value over a block of N threads, the same tree for every block and wave width.
template <int N>
__device__ inline float block_sum(float v, float* lds) {
    lds[threadIdx.x] = v;
    __syncthreads();
    block_tree<N>(lds);
    __syncthreads();
    const float r = lds[0];
    __syncthreads();
    return r;
}

// The dot of two packed words, two 16-bit values each or four 8-bit values each, added to s.
__device__ inline int word_dot(int a, int b, int s, short) {
#ifdef LLMX_DOT16
    return dot2_i16(a, b, s);
#else
    return s + (int)(short)(a & 0xffff) * (int)(short)(b & 0xffff) + (a >> 16) * (b >> 16);
#endif
}
__device__ inline int word_dot(int a, int b, int s, signed char) {
#if defined(LLMX_DOT16) || GPU_IMMA8
    return dot4_i8(a, b, s);
#else
    for (int i = 0; i < 4; ++i) s += (int)(signed char)((a >> (8 * i)) & 0xff) * (int)(signed char)((b >> (8 * i)) & 0xff);
    return s;
#endif
}

// v of lane (this lane xor m) across the whole wave, m a power of two: DPP for 1 and 2 and swizzles up to 16, which wait on no LDS queue as a permute does; the same partner, so a sum over them keeps its bits.
__device__ inline int lane_xor(int v, int m) {
    if (m == 1) return lane_xor_quad<1>(v);
    if (m == 2) return lane_xor_quad<2>(v);
    if (m == 4) return lane_xor_swizzle<4>(v);
    if (m == 8) return lane_xor_swizzle<8>(v);
    if (m == 16) return lane_xor_swizzle<16>(v);
    return shuffle_xor(v, m, warpSize);
}
__device__ inline float lane_xor(float v, int m) { return __int_as_float(lane_xor(__float_as_int(v), m)); }
// The sum of v over the wave by the xor tree, m from `from` down to 1, each level a lane's own value plus its partner's.
__device__ inline float wave_sum(float v, int from = warpSize / 2) {
#pragma unroll
    for (int m = from; m > 0; m /= 2) v += lane_xor(v, m);
    return v;
}

// The one-column build of the half-block kernel: NS steps of a lane's loads for each of R rows issued before any of their products, then each step's products in block order, as the other builds sum.
// The scheduling barrier keeps the loads together (docs/STATUS.md, ROCm decode rows); TAIL skips a step past the row in its loads and its products alike, so a lane's blocks keep their order.
constexpr int kSteps = 4;
// The steps of the builds for a row under two spans of blocks (64 on a 64-lane wave): such a row cannot fill a second step a load, and each step held costs registers, 71 against 39 at one row (docs/STATUS.md, ROCm decode steps).
constexpr int kNarrowSteps = 1;
template <int NS, int R, bool TAIL = false>
__device__ inline void half_steps(const unsigned* const (&w)[R], const signed char* q, const float* dx, unsigned& b, unsigned blocks, unsigned hl, float (&acc)[R]) {
    const unsigned span = warpSize / 2;
    unsigned t[NS][R][5];
    int4 xq[NS];
    float xd[NS];
#pragma unroll
    for (int d = 0; d < NS; ++d) {
        const unsigned bd = b + d * span, start = bd * kQ8 + 18 * hl, sh = start & 2, wi = start >> 2;
        if (TAIL && bd >= blocks) continue;
        xq[d] = *(const int4*)(q + (unsigned long long)bd * 32 + 16 * hl);
        xd[d] = dx[bd];
#pragma unroll
        for (int j = 0; j < R; ++j) {
            t[d][j][0] = w[j][wi];
            t[d][j][1] = w[j][wi + 1];
            t[d][j][2] = w[j][wi + 2];
            t[d][j][3] = w[j][wi + 3];
            t[d][j][4] = w[j][wi + 3 + (hl == 0 || sh ? 1 : 0)];
        }
    }
    schedule_barrier();
#pragma unroll
    for (int d = 0; d < NS; ++d) {
        const unsigned bd = b + d * span, start = bd * kQ8 + 18 * hl, sh = start & 2, s0 = sh + 2 * (1 - hl);
        // Both lanes of a block take or skip a step together, so the scale's exchange reads an active partner.
        if (TAIL && bd >= blocks) continue;
        const bool next = s0 == 4;
#pragma unroll
        for (int j = 0; j < R; ++j) {
            const unsigned t0 = t[d][j][0], t1 = t[d][j][1], t2 = t[d][j][2], t3 = t[d][j][3], t4 = t[d][j][4];
            const unsigned u0 = next ? t1 : t0, u1 = next ? t2 : t1, u2 = next ? t3 : t2, u3 = next ? t4 : t3, n = s0 & 3;
            const unsigned wq[4] = {byte_align(u1, u0, n), byte_align(u2, u1, n), byte_align(u3, u2, n),
                                    byte_align(t4, u3, n)};
            int s = word_dot((int)wq[0], xq[d].x, 0, (signed char)0);
            s = word_dot((int)wq[1], xq[d].y, s, (signed char)0);
            s = word_dot((int)wq[2], xq[d].z, s, (signed char)0);
            s = word_dot((int)wq[3], xq[d].w, s, (signed char)0);
            const int own = (int)(t0 >> (8 * sh)) & 0xffff;
            const int partner = pair_lane(own);
            const float dw = __half2float(__ushort_as_half((unsigned short)(hl ? partner : own)));
            acc[j] = fmaf(dw * xd[d], (float)s, acc[j]);
        }
    }
    b += NS * span;
}
// A lane's partial sums of R rows of an even block count over the twin column q, dx: two lanes a block, the lane's blocks in order, NS steps a round.
template <int R, int NS>
__device__ inline void half_rows(const unsigned* const (&w)[R], const signed char* q, const float* dx, unsigned blocks, float (&acc)[R]) {
    const unsigned lane = threadIdx.x % warpSize, hl = lane & 1, span = warpSize / 2;
#pragma unroll
    for (int j = 0; j < R; ++j) acc[j] = 0.0f;
    unsigned b = lane >> 1;
    while (b + (NS - 1) * span < blocks) half_steps<NS, R>(w, q, dx, b, blocks, hl, acc);
    if (b < blocks) half_steps<NS, R, true>(w, q, dx, b, blocks, hl, acc);
}
// The int8 class's sum of one Q8_0 row over one twin column, a wave the row: the bits of every build of the row kernels (kernels/int16.gpu, q8_0_half_one).
__device__ inline float q8_half_row(const unsigned char* row, const signed char* q, const float* dx, unsigned blocks) {
    const unsigned* const w[1] = {(const unsigned*)row};
    float acc[1];
    if (blocks < (unsigned)warpSize) half_rows<1, kNarrowSteps>(w, q, dx, blocks, acc);
    else half_rows<1, kSteps>(w, q, dx, blocks, acc);
    return wave_sum(acc[0]);
}

}  // namespace
