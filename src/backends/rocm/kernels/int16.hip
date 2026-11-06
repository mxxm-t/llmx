// The quantized products of the F16 and int8 classes (docs/PRECISION.md, rule 4 and section 5): activations as a block-int16 or block-int8 twin, Q8_0 rows decoded exactly, integer block dots and F32 sums.
// A row's result depends on that row alone: the row kernel for generated rows and the tile for prompt rows each fix their order, whatever the call holds.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "backends/rocm/kernels/quant.h"

namespace {

constexpr int kThreads = 256;
constexpr unsigned kQ8 = rq::Q8_0::bytes;   // a Q8_0 block's bytes, whose layout the word loads below take: the f16 scale, then 32 int8 values
static_assert(kQ8 == 34 && rq::Q8_0::values == 32, "the Q8_0 word loads assume 34-byte blocks of 32 values");

#ifdef LLMX_DOT16
typedef short short2v __attribute__((ext_vector_type(2)));
#endif

// The dot of two packed words, two 16-bit values each or four 8-bit values each, added to s.
__device__ inline int word_dot(int a, int b, int s, short) {
#ifdef LLMX_DOT16
    return __builtin_amdgcn_sdot2(__builtin_bit_cast(short2v, a), __builtin_bit_cast(short2v, b), s, false);
#else
    return s + (int)(short)(a & 0xffff) * (int)(short)(b & 0xffff) + (a >> 16) * (b >> 16);
#endif
}
__device__ inline int word_dot(int a, int b, int s, signed char) {
#ifdef LLMX_DOT16
    return __builtin_amdgcn_sdot4(a, b, s, false);
#else
    for (int i = 0; i < 4; ++i) s += (int)(signed char)((a >> (8 * i)) & 0xff) * (int)(signed char)((b >> (8 * i)) & 0xff);
    return s;
#endif
}


// v of lane (this lane xor m) across the whole wave, m a power of two: DPP for 1 and 2 and swizzles up to 16, which wait on no LDS queue as a permute does; the same partner, so a sum over them keeps its bits.
__device__ inline int lane_xor(int v, int m) {
    if (m == 1) return __builtin_amdgcn_mov_dpp(v, 0xB1, 0xF, 0xF, false);
    if (m == 2) return __builtin_amdgcn_mov_dpp(v, 0x4E, 0xF, 0xF, false);
    if (m == 4) return __builtin_amdgcn_ds_swizzle(v, 0x1F | (4 << 10));
    if (m == 8) return __builtin_amdgcn_ds_swizzle(v, 0x1F | (8 << 10));
    if (m == 16) return __builtin_amdgcn_ds_swizzle(v, 0x1F | (16 << 10));
    return __shfl_xor(v, m, warpSize);
}
__device__ inline float lane_xor(float v, int m) { return __int_as_float(lane_xor(__float_as_int(v), m)); }
// The sum of v over the wave by the xor tree, m from `from` down to 1, each level a lane's own value plus its partner's.
__device__ inline float wave_sum(float v, int from = warpSize / 2) {
#pragma unroll
    for (int m = from; m > 0; m /= 2) v += lane_xor(v, m);
    return v;
}
#if defined(__GFX9__)
constexpr bool kWave64 = true;
#else
constexpr bool kWave64 = false;
#endif
// The wave sums of a lane's NV values by the same xor tree, transposed: each level halves the values a lane holds, keeping the half its bit m selects plus its partner's copy, as wave_sum adds.
// Every sum so has wave_sum's bits; once one value is left the levels below are wave_sum's, and lane l holds sum k = l / (warpSize / NV).
template <int H, int NV>
__device__ inline void halve(float (&v)[NV], int lane, int m) {
    const bool up = lane & m;
#pragma unroll
    for (int j = 0; j < H; ++j) {
        const float give = up ? v[j] : v[j + H], keep = up ? v[j + H] : v[j];
        v[j] = keep + lane_xor(give, m);
    }
    if constexpr (H > 1) halve<H / 2>(v, lane, m / 2);
}
template <int NV>
__device__ inline float wave_sums(float (&v)[NV], int lane, int& k) {
    static_assert(NV >= 1 && NV <= 64 && (NV & (NV - 1)) == 0, "a power of two values, at most the widest wave's 64 lanes");
    // On a narrower wave the caller passes at most its lanes' values (store_sums), so warpSize / NV is never 0.
    if constexpr (NV > 1) halve<NV / 2>(v, lane, warpSize / 2);
    k = lane / (warpSize / NV);
    return wave_sum(v[0], warpSize / (2 * NV));
}

// A row kernel call: up to three projections of one twin, their output rows numbered one after another, projection p's from start[p] on.
struct RowCall {
    const unsigned char* w[3];
    float* y[3];
    unsigned long long nout[3];
    unsigned long long start[3];
    unsigned long long total;
    unsigned nproj;
    unsigned cluster;   // the lanes a K-type row takes (k_rows); Q8_0's row kernels take a wave a row
};
// Row o of a call: its weight row, its output for column 0 and its projection's output count.
struct Row {
    const unsigned char* w;
    float* y;
    unsigned long long nout;
};
// The kernels pass o through readfirstlane: it is one wave's, so the projection is chosen in scalar registers, which measured 3 to 4 percent faster on the 8B's rows.
// The projection's fields are picked by adding differences, not by selecting between fields, which the compiler folds into an indexed load that keeps a routed call's copy in scratch.
__device__ inline Row row_of(const RowCall& c, unsigned long long o, unsigned long long row_bytes) {
    const bool p1 = c.nproj > 1 && o >= c.start[1], p2 = c.nproj > 2 && o >= c.start[2];
    const auto pick = [&](unsigned long long a, unsigned long long b, unsigned long long d) { return a + (p1 ? b - a : 0) + (p2 ? d - b : 0); };
    const unsigned long long local = o - pick(0, c.start[1], c.start[2]);
    const unsigned char* w = (const unsigned char*)pick((unsigned long long)c.w[0], (unsigned long long)c.w[1], (unsigned long long)c.w[2]);
    float* y = (float*)pick((unsigned long long)c.y[0], (unsigned long long)c.y[1], (unsigned long long)c.y[2]);
    return {w + local * row_bytes, y + local, pick(c.nout[0], c.nout[1], c.nout[2])};
}

// A row build's columns: column c reads twin column x(c) and writes output column y(c), c itself for a plain call.
struct Cols {
    __device__ unsigned long long x(int c) const { return (unsigned long long)c; }
    __device__ unsigned long long y(int c) const { return (unsigned long long)c; }
};
// A grouped routed call's columns, one expert's entries e[0..n): column c reads entry e[c]'s twin column e[c] / per and writes entry e[c]'s outputs; a column past n reads entry e[0]'s and stores nothing.
template <int C>
struct EntryCols {
    const unsigned* e;
    unsigned long long xs[C];
    __device__ EntryCols(const unsigned* entries, int n, unsigned long long per) : e(entries) {
#pragma unroll
        for (int c = 0; c < C; ++c) xs[c] = entries[c < n ? c : 0] / per;
    }
    __device__ unsigned long long x(int c) const { return xs[c]; }
    __device__ unsigned long long y(int c) const { return e[c]; }
};

// Sum s of the wave's row j and column c, stored by its writer lane unless the column is past the call or the row past the end.
// The lane picks its row from the wave's rows r, which sit in scalar registers, so no lane reads the call's arguments again.
template <int R, typename M>
__device__ inline void store_sum(const Row (&r)[R], float s, int j, int c, bool writer, unsigned long long o, unsigned long long total, int ncols, int add, const M& m) {
    if (!writer || c >= ncols || o + j >= total) return;
#pragma unroll
    for (int i = 0; i < R; ++i)
        if (j == i) {
            float* y = r[i].y + m.y(c) * r[i].nout;
            *y = add ? *y + s : s;
        }
}
// The R * C sums of a wave's rows o to o + R - 1 and its columns by one transposed tree, each written by its own lane; a column past the call or a row past the end is summed and not stored.
template <int R, int C, typename M = Cols>
__device__ inline void store_sums(float (&acc)[R][C], const Row (&r)[R], int lane, unsigned long long o, unsigned long long total, int ncols, int add, const M& m = M{}) {
    float v[R * C];
#pragma unroll
    for (int j = 0; j < R; ++j)
#pragma unroll
        for (int c = 0; c < C; ++c) v[j * C + c] = acc[j][c];
    // A target that may run 32-lane waves sums a 64-value build in parts of 32, each sum keeping the one tree's bits; gfx9 runs only 64-lane waves, so it keeps one tree, decided at compile time so no dead path holds registers.
    constexpr int NV = R * C, NP = NV < 32 ? NV : 32;
    if constexpr (kWave64 || NV <= 32) {
        int k;
        const float s = wave_sums<NV>(v, lane, k);
        store_sum<R>(r, s, k / C, k % C, lane % (warpSize / NV) == 0, o, total, ncols, add, m);
    } else {
#pragma unroll
        for (int p = 0; p < NV; p += NP) {
            float part[NP];
#pragma unroll
            for (int i = 0; i < NP; ++i) part[i] = v[p + i];
            int k;
            const float s = wave_sums<NP>(part, lane, k);
            store_sum<R>(r, s, (p + k) / C, (p + k) % C, lane % (warpSize / NP) == 0, o, total, ncols, add, m);
        }
    }
}

// A wave computes R output rows for up to C generated rows: four lanes a block, eight values each, each lane's part scaled into F32 in block order and folded by wave_sums, the bits of one shuffle tree whatever C and R.
// With R rows each twin load serves every row and a row keeps its lanes and its order, so it has the bits of the one-row build (docs/STATUS.md, ROCm F16 decode rows).
template <int C, typename QT, int R, int G, typename M = Cols>
__device__ void q8_0_rows(const RowCall& rc, const QT* q, const float* dx, unsigned long long nin, int ncols, int add, const M& m = M{}) {
    const int lanes = warpSize, lane = threadIdx.x % lanes;
    const unsigned long long blocks = nin / 32, waves = (unsigned long long)gridDim.x * (kThreads / lanes);
    // A wave takes rows o, o + R * waves, ...: the host launches a wave for R rows, and this loop form measured four times faster than an early return on rows of 2048 and 4096 (docs/STATUS.md, ROCm step 3).
    for (unsigned long long o = R * ((unsigned long long)blockIdx.x * (kThreads / lanes) + threadIdx.x / lanes); o < rc.total; o += R * waves) {
        Row r[R];
#pragma unroll
        for (int j = 0; j < R; ++j) r[j] = row_of(rc, __builtin_amdgcn_readfirstlane((unsigned)(o + j < rc.total ? o + j : o)), blocks * kQ8);
        float acc[R][C];
#pragma unroll
        for (int j = 0; j < R; ++j)
#pragma unroll
            for (int c = 0; c < C; ++c) acc[j][c] = 0.0f;
        for (unsigned long long u = lane; u < blocks * 4; u += lanes) {
            const unsigned long long k = u / 4;
            const int part = (int)(u % 4);
            const unsigned ko = (unsigned)k, xo = ko * 32 + part * 8;
            constexpr int NW = sizeof(QT) == 2 ? 4 : 2;
            // A lane's eight weights as four 16-bit loads, the block being 2-byte aligned, and its scale, read raw.
            unsigned short sw[R], pw[R][4];
#pragma unroll
            for (int j = 0; j < R; ++j) {
                const unsigned char* block = r[j].w + k * kQ8;
                sw[j] = *(const unsigned short*)block;
                const unsigned short* wp = (const unsigned short*)(block + 2 + part * 8);
#pragma unroll
                for (int i = 0; i < 4; ++i) pw[j][i] = wp[i];
            }
            // Columns in groups of G, a column past the call reading column 0 and storing nothing, so no branch splits the loads.
            int xw[G][NW];
            float xd[G];
            const auto load = [&](int g) {
#pragma unroll
                for (int c = 0; c < G; ++c) {
                    // A build wider than its group takes one 32-bit lane offset for every column, where a pointer a column held it to one wave a SIMD.
                    // The narrower builds measured faster with a pointer a column (docs/STATUS.md, ROCm wide decode rows).
                    const unsigned long long col = m.x(g + c < ncols ? g + c : 0) * blocks;
                    const QT* qc = C > G ? q + col * 32 : q + (col + k) * 32 + part * 8;
                    const unsigned off = C > G ? xo : 0;
                    if constexpr (NW == 4) {
                        const int4 x = *(const int4*)(qc + off);
                        xw[c][0] = x.x, xw[c][1] = x.y, xw[c][2] = x.z, xw[c][3] = x.w;
                    } else {
                        const int2 x = *(const int2*)(qc + off);
                        xw[c][0] = x.x, xw[c][1] = x.y;
                    }
                    xd[c] = C > G ? dx[col + ko] : dx[col + k];
                }
            };
            // The weights as the twin's words, packed once for every column: four words of two 16-bit values, or two of four 8-bit values.
            float dw[R];
            int ww[R][NW];
            const auto unpack = [&] {
#pragma unroll
                for (int j = 0; j < R; ++j) {
                    dw[j] = __half2float(__ushort_as_half(sw[j]));
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const unsigned short pair = pw[j][i];
                        if constexpr (NW == 4)
                            ww[j][i] = ((int)(signed char)(pair & 0xff) & 0xffff) | ((int)(signed char)(pair >> 8) << 16);
                        else if (i % 2)
                            ww[j][i / 2] |= (int)pair << 16;
                        else
                            ww[j][i / 2] = pair;
                    }
                }
            };
            // In a build of one group the weights' and the group's loads issue together before anything waits on them; a wider build unpacks its weights first, which measured faster for it.
            if constexpr (C <= G) {
                load(0);
                __builtin_amdgcn_sched_barrier(0);
                unpack();
            } else {
                unpack();
                load(0);
            }
#pragma unroll
            for (int g = 0; g < C; g += G) {
                if (g) load(g);
#pragma unroll
                for (int c = 0; c < G; ++c)
#pragma unroll
                    for (int j = 0; j < R; ++j) {
                        // The integer sum is exact, so the word dots give the bits of any order.
                        int isum = 0;
#pragma unroll
                        for (int i = 0; i < NW; ++i) isum = word_dot(ww[j][i], xw[c][i], isum, QT(0));
                        acc[j][g + c] = fmaf(dw[j] * xd[c], (float)isum, acc[j][g + c]);
                    }
                // A group's loads wait for the group before it, which holds a wide build's registers to one group's twin.
                if constexpr (C > G) __builtin_amdgcn_sched_barrier(0);
            }
        }
        store_sums<R, C>(acc, r, lane, o, rc.total, ncols, add, m);
    }
}

// The int8 class's Q8_0 decode kernel in the Vulkan decode kernel's half-block order, for rows of an even block count: two lanes a block, the even lane handing the scale to the odd one.
// Each lane scales its integer dots in block order and the lanes fold by wave_sums' transposed tree, so every column has the same bits whatever C and R the call takes.
template <int C, int R, int G, typename M = Cols>
__device__ void q8_0_rows_half(const RowCall& rc, const signed char* q, const float* dx, unsigned long long nin, int ncols, int add, const M& m = M{}) {
    const int lanes = warpSize, lane = threadIdx.x % lanes;
    const unsigned long long blocks = nin / 32, waves = (unsigned long long)gridDim.x * (kThreads / lanes);
    const unsigned hl = lane & 1;
    // A wave takes R rows, so each twin load serves every row; a row keeps its lanes and its order, so it has the bits of the one-row build.
    for (unsigned long long o = R * ((unsigned long long)blockIdx.x * (kThreads / lanes) + threadIdx.x / lanes); o < rc.total; o += R * waves) {
        Row r[R];
        const unsigned* w[R];
#pragma unroll
        for (int j = 0; j < R; ++j) {
            r[j] = row_of(rc, __builtin_amdgcn_readfirstlane((unsigned)(o + j < rc.total ? o + j : o)), blocks * kQ8);
            // An even block count puts every row on a word boundary.
            w[j] = (const unsigned*)r[j].w;
        }
        float acc[R][C];
#pragma unroll
        for (int j = 0; j < R; ++j)
#pragma unroll
            for (int c = 0; c < C; ++c) acc[j][c] = 0.0f;
#pragma unroll 2
        for (unsigned b = lane >> 1; b < blocks; b += lanes / 2) {
            // The lane's bytes start sh bytes into word wi and its weights s0 bytes in, 4 meaning the next word.
            // The even lane's fifth word lies inside its block and the odd lane's, read only where its bytes do not start on a word, inside the block after it.
            const unsigned at = b * kQ8, start = at + 18 * hl, sh = start & 2, wi = start >> 2, s0 = sh + 2 * (1 - hl);
            // Columns in groups of G, a column past the call skipped: the first group's twin loads issue before the weights' and each group's before its products (docs/STATUS.md, ROCm wide decode rows).
            int4 xq[G];
            float xd[G];
            const auto load = [&](int g) {
#pragma unroll
                for (int c = 0; c < G; ++c) {
                    if (g + c >= ncols) continue;
                    // The column's base is the wave's and the lane's offset 32-bit, one register for every column.
                    const unsigned long long col = m.x(g + c) * blocks;
                    xq[c] = *(const int4*)(q + col * 32 + (b * 32 + 16 * hl));
                    xd[c] = dx[col + b];
                }
            };
            load(0);
            unsigned wq[R][4];
            float dw[R];
#pragma unroll
            for (int j = 0; j < R; ++j) {
                const unsigned t0 = w[j][wi], t1 = w[j][wi + 1], t2 = w[j][wi + 2], t3 = w[j][wi + 3], t4 = w[j][wi + 3 + (hl == 0 || sh ? 1 : 0)];
                const bool next = s0 == 4;
                const unsigned u0 = next ? t1 : t0, u1 = next ? t2 : t1, u2 = next ? t3 : t2, u3 = next ? t4 : t3, n = s0 & 3;
                wq[j][0] = __builtin_amdgcn_alignbyte(u1, u0, n);
                wq[j][1] = __builtin_amdgcn_alignbyte(u2, u1, n);
                wq[j][2] = __builtin_amdgcn_alignbyte(u3, u2, n);
                wq[j][3] = __builtin_amdgcn_alignbyte(t4, u3, n);
                const int own = (int)(t0 >> (8 * sh)) & 0xffff;
                const int partner = lane_xor(own, 1);
                dw[j] = __half2float(__ushort_as_half((unsigned short)(hl ? partner : own)));
            }
#pragma unroll
            for (int g = 0; g < C; g += G) {
                if (g) load(g);
#pragma unroll
                for (int c = 0; c < G; ++c) {
                    if (g + c >= ncols) continue;
#pragma unroll
                    for (int j = 0; j < R; ++j) {
                        int t = word_dot((int)wq[j][0], xq[c].x, 0, (signed char)0);
                        t = word_dot((int)wq[j][1], xq[c].y, t, (signed char)0);
                        t = word_dot((int)wq[j][2], xq[c].z, t, (signed char)0);
                        t = word_dot((int)wq[j][3], xq[c].w, t, (signed char)0);
                        acc[j][g + c] = fmaf(dw[j] * xd[c], (float)t, acc[j][g + c]);
                    }
                }
            }
        }
        store_sums<R, C>(acc, r, lane, o, rc.total, ncols, add, m);
    }
}

// The one-column build of the half-block kernel: NS steps of a lane's loads for each of R rows issued before any of their products, then each step's products in block order, as the other builds sum.
// The scheduling barrier keeps the loads together (docs/STATUS.md, ROCm decode rows); TAIL skips a step past the row in its loads and its products alike, so a lane's blocks keep their order.
constexpr int kSteps = 4;
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
    __builtin_amdgcn_sched_barrier(0);
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
            const unsigned wq[4] = {__builtin_amdgcn_alignbyte(u1, u0, n), __builtin_amdgcn_alignbyte(u2, u1, n), __builtin_amdgcn_alignbyte(u3, u2, n),
                                    __builtin_amdgcn_alignbyte(t4, u3, n)};
            int s = word_dot((int)wq[0], xq[d].x, 0, (signed char)0);
            s = word_dot((int)wq[1], xq[d].y, s, (signed char)0);
            s = word_dot((int)wq[2], xq[d].z, s, (signed char)0);
            s = word_dot((int)wq[3], xq[d].w, s, (signed char)0);
            const int own = (int)(t0 >> (8 * sh)) & 0xffff;
            const int partner = lane_xor(own, 1);
            const float dw = __half2float(__ushort_as_half((unsigned short)(hl ? partner : own)));
            acc[j] = fmaf(dw * xd[d], (float)s, acc[j]);
        }
    }
    b += NS * span;
}
// R rows a wave, each twin load serving every row, NS steps a round: a row keeps its lanes, its block order and its tree, so every build gives the one-row build's bits.
// PAIRED, for rows of at most 16 blocks on a 64-lane wave, takes each half of the wave as a row, whose tree from 32 lanes is the one-row build's since that tree's first level adds the idle half's +0.
template <int R, int NS, bool PAIRED = false>
__device__ void q8_0_half_one(const RowCall& rc, const signed char* q, const float* dx, unsigned long long nin, int add) {
    static_assert(!PAIRED || R == 1, "a paired wave holds one row a half");
    const int lanes = warpSize, lane = threadIdx.x % lanes, cl = PAIRED ? 32 : lanes, sl = lane % cl;
    const unsigned blocks = (unsigned)(nin / 32);
    const unsigned long long waves = (unsigned long long)gridDim.x * (kThreads / lanes), step = (PAIRED ? 2 : R) * waves;
    const unsigned hl = sl & 1, span = lanes / 2;
    for (unsigned long long base = (PAIRED ? 2 : R) * ((unsigned long long)blockIdx.x * (kThreads / lanes) + threadIdx.x / lanes); base < rc.total; base += step) {
        Row r[R];
        const unsigned* w[R];
#pragma unroll
        for (int j = 0; j < R; ++j) {
            const unsigned long long o = PAIRED ? base + lane / cl : base + j;
            r[j] = row_of(rc, PAIRED ? (o < rc.total ? o : base) : __builtin_amdgcn_readfirstlane((unsigned)(o < rc.total ? o : base)), blocks * kQ8);
            w[j] = (const unsigned*)r[j].w;
        }
        float acc[R];
#pragma unroll
        for (int j = 0; j < R; ++j) acc[j] = 0.0f;
        unsigned b = sl >> 1;
        while (b + (NS - 1) * span < blocks) half_steps<NS, R>(w, q, dx, b, blocks, hl, acc);
        if (b < blocks) half_steps<NS, R, true>(w, q, dx, b, blocks, hl, acc);
#pragma unroll
        for (int j = 0; j < R; ++j) {
            const unsigned long long o = PAIRED ? base + lane / cl : base + j;
            const float v = PAIRED ? wave_sum(acc[j] + 0.0f, 16) : wave_sum(acc[j]);
            if (sl == 0 && o < rc.total) *r[j].y = add ? *r[j].y + v : v;
        }
    }
}

constexpr int kTileOut = 64, kTileRows = 64, kNarrowRows = 16;   // a narrow routed tile's rows: a group of up to 16 entries

// A tile call: up to three projections of one X and one twin, projection p's output tiles from workgroup start[p] on, the inner dimension in parts of kper blocks (blockIdx.z the part).
// A split call writes part z of projection p at partials + parts * nrows * (outputs of the projections before p) + z * nrows * nout[p], which llmx_tile_reduce adds in part order.
struct TileCall {
    const unsigned char* w[3];
    float* y[3];
    unsigned long long nout[3];
    unsigned start[3];
    unsigned nproj;
    unsigned long long nin, nrows;
    int add;
    unsigned kper, parts;
    float* partials;
    unsigned xrow, xblock;   // the twin's blocks from one row to the next and from one block to the next (TwinMap, kernels/f32.hip)
};
// A routed tile call's groups (kernels/f32.hip, llmx_moe_group): group first + blockIdx.y a tile of its expert's rows, its columns its entries.
// The call never splits, so an entry's sums do not depend on the entries beside it.
struct TileGroups {
    const unsigned* tab;
    unsigned long long order0, per, first;
    unsigned long long stride[3];
};
// Blocks of 32 staged a barrier as the Vulkan tile stages them, four for the 8-bit twin and two for the 16-bit one, a step loading while the previous one computes but in the tall 16-bit build.
// Each value sits at a 32-bit byte offset from a uniform base, which the host bounds (`rocm_backend.cpp`, the tile launch).
template <typename T> __device__ inline T at(const void* base, unsigned off) { return *(const T*)((const unsigned char*)base + off); }
// WQ quant words as one load: their words are 4-byte aligned, which a global vector load takes.
template <int WQ> using QWords = unsigned __attribute__((ext_vector_type(WQ), aligned(4)));
template <typename QT, int RO> struct TileStep { static constexpr int n = sizeof(QT) == 1 && RO == 4 ? 4 : 2; static constexpr bool ahead = RO == 4 || sizeof(QT) == 1; };

// The prompt tile in the Vulkan tile's design: a block of 256 threads 64 outputs by 64 rows, each thread 4 by 4, the steps staged k-major in shared memory.
// Each output sums its blocks in order, acc = fma(dw * dx, isum, acc), so a row gives the same bits wherever the tile falls; routed, a block row is a group.
template <typename QT, int RO, bool ROUTED = false, int ROWS = kTileRows>
__device__ void q8_0_tile(const TileCall& c, const QT* q, const float* dx, const TileGroups& tg = TileGroups{}) {
    constexpr bool AHEAD = TileStep<QT, RO>::ahead;
    constexpr int STEP = TileStep<QT, RO>::n, WORDS = 32 * (int)sizeof(QT) / 4, OUTS = RO * 16;
    constexpr int WT = kThreads / OUTS, WQ = 32 / WT / 4;   // threads a weight row and the quant words each stages
    constexpr int J = ROWS / 16;   // a thread's rows
    const unsigned pi = c.nproj > 2 && blockIdx.x >= c.start[2] ? 2 : c.nproj > 1 && blockIdx.x >= c.start[1] ? 1 : 0;
    const unsigned long long gi = ROUTED ? tg.first + blockIdx.y : 0;
    const unsigned long long nout = c.nout[pi], nrows = ROUTED ? tg.tab[4 * gi + 2] : c.nrows, nin = c.nin;
    // A routed group takes the narrow tile when it fits its rows and the 64-row tile otherwise; a column's sums are the same in both.
    if (ROUTED && (!nrows || (nrows <= kNarrowRows) != (ROWS == kNarrowRows))) return;
    const unsigned* ent = ROUTED ? tg.tab + tg.order0 + tg.tab[4 * gi + 1] : nullptr;
    const unsigned char* W = c.w[pi] + (ROUTED ? tg.tab[4 * gi] * tg.stride[pi] : 0);
    __shared__ __attribute__((aligned(16))) int wq[STEP * WORDS][OUTS];
    __shared__ __attribute__((aligned(16))) int xw[STEP * WORDS][ROWS];
    __shared__ float wd[STEP][OUTS], xd[STEP][ROWS];
    const int t = threadIdx.x, tr = t / 16, tc = t % 16;   // outputs tr * RO.., rows tc * J..
    // The narrow routed build gives each wave one quad of the group's entries, rotated by workgroup since wave w runs on SIMD w; a quad holding no entry skips its dots.
    constexpr bool QUADS = ROUTED && ROWS == kNarrowRows;
    constexpr int TO = QUADS ? 1 : RO, TJ = QUADS ? 4 : J;   // a thread's outputs and rows
    const int quad = (int)((t / 64 + blockIdx.x + blockIdx.y) % 4), lane = t % 64;
    const auto out_at = [&](int i) { return QUADS ? lane : tr * RO + i; };
    const auto row_at = [&](int j) { return QUADS ? 4 * quad + j : tc * J + j; };
    const bool idle = QUADS && 4ull * quad >= nrows;
    const unsigned long long o0 = (unsigned long long)(blockIdx.x - c.start[pi]) * OUTS, b0 = ROUTED ? 0 : (unsigned long long)blockIdx.y * ROWS;
    const unsigned long long blocks = nin / 32, kbeg = (unsigned long long)blockIdx.z * c.kper;
    const unsigned long long kend = kbeg + c.kper < blocks ? kbeg + c.kper : blocks;
    float acc[TO][TJ];
    for (int i = 0; i < TO; ++i)
        for (int j = 0; j < TJ; ++j) acc[i][j] = 0.0f;
    // Each weight row takes WT threads, each WQ words of a block's quants (two threads a row on the tall tile, four on the other), and the twin's rows four threads, each a quarter of a block (the Vulkan tile's staging).
    // A stage's blocks are loaded into registers while the previous stage computes, and stored to shared memory after it.
    const int r = t / 4, seg = t % 4, wr = t / WT, ws = t % WT;
    const unsigned long long o = o0 + wr, b = b0 + r;
    const unsigned xr = ROUTED ? (b < nrows ? (unsigned)(ent[b] / tg.per) : 0u) : (unsigned)r;   // the twin row this thread stages
    unsigned wv[STEP][WQ + 1];   // the thread's quant words and one more: the word after where they straddle, the word before where not
    bool wok[STEP], al[STEP], xok[STEP];
    int xv[STEP][WORDS / 4];       // a quarter of the twin block
    float xs_[STEP];
    // The tile's first output and first row as uniform bases, so each thread's offsets from them are 32-bit bytes (`at`) and its loads take the base from scalar registers.
    const unsigned char* Wt = W + o0 * blocks * kQ8;
    const unsigned mis = (unsigned)((unsigned long long)Wt & 3);
    const unsigned* W4 = (const unsigned*)(Wt - mis);   // the words around the tile's weights, as a pointer into the same buffer so its loads stay global ones
    const QT* qt = q + b0 * c.xrow * 32;
    const float* dxt = dx + b0 * c.xrow;
    // A thread's quants sit at an offset of 0 or 2 in their 4-byte words, since a block is 34 bytes, so one vector load and one more word cover them.
    // Every load of a stage goes out before any is waited for, a load out of range reading block 0 and its zero chosen at the store.
    const auto load = [&](unsigned long long k0) {
#pragma unroll
        for (int s = 0; s < STEP; ++s) {
            const unsigned long long k = k0 + s;
            xok[s] = k < kend && b < nrows;
            wok[s] = k < kend && o < nout;
            const unsigned a = mis + (wok[s] ? ((unsigned)wr * (unsigned)blocks + (unsigned)k) * 34u : 0u) + 2u + ws * (4u * WQ), w = a & ~3u;
            al[s] = (a & 3) == 0;
            const QWords<WQ> v = at<QWords<WQ>>(W4, w);
#pragma unroll
            for (int i = 0; i < WQ; ++i) wv[s][i] = v[i];
            wv[s][WQ] = at<unsigned>(W4, al[s] ? w - 4u : w + 4u * WQ);
            const unsigned xb = xok[s] ? xr * c.xrow + (unsigned)k * c.xblock : 0u;
            if constexpr (sizeof(QT) == 2) {
                const int4 v = at<int4>(qt, xb * 64u + seg * 16u);
                xv[s][0] = v.x, xv[s][1] = v.y, xv[s][2] = v.z, xv[s][3] = v.w;
            } else {
                const int2 v = at<int2>(qt, xb * 32u + seg * 8u);
                xv[s][0] = v.x, xv[s][1] = v.y;
            }
            xs_[s] = at<float>(dxt, xb * 4u);
        }
    };
    // Widened to the twin's integer type as they are stored: each quant word to two words of two 16-bit values, or the words of bytes as they are.
    const auto store = [&]() {
#pragma unroll
        for (int s = 0; s < STEP; ++s) {
            const unsigned last = wv[s][WQ];
#pragma unroll
            for (int i = 0; i < WQ; ++i) {
                const unsigned next = i + 1 < WQ ? wv[s][i + 1] : last;
                const unsigned qw = wok[s] ? (al[s] ? wv[s][i] : __builtin_amdgcn_alignbit(next, wv[s][i], 16)) : 0;
                if constexpr (sizeof(QT) == 2) {
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const unsigned pair = (qw >> (16 * h)) & 0xffff;
                        wq[s * WORDS + ws * 2 * WQ + 2 * i + h][wr] = (int)((unsigned)(unsigned short)(short)(signed char)(pair & 0xff) |
                                                                    ((unsigned)(unsigned short)(short)(signed char)(pair >> 8) << 16));
                    }
                } else {
                    wq[s * WORDS + ws * WQ + i][wr] = (int)qw;
                }
            }
            const unsigned short sc = wok[s] ? (unsigned short)(al[s] ? last >> 16 : wv[s][0] & 0xffff) : 0;
            if (ws == 0) wd[s][wr] = __half2float(__ushort_as_half(sc));
#pragma unroll
            for (int w = 0; w < WORDS / 4; ++w)
                if (r < ROWS) xw[s * WORDS + seg * (WORDS / 4) + w][r] = xok[s] ? xv[s][w] : 0;
            if (seg == 0 && r < ROWS) xd[s][r] = xok[s] ? xs_[s] : 0.0f;
        }
    };
    if (AHEAD) load(kbeg);
    for (unsigned long long k0 = kbeg; k0 < kend; k0 += STEP) {
        if (!AHEAD) load(k0);
        store();
        __syncthreads();
        if (AHEAD && k0 + STEP < kend) load(k0 + STEP);
#pragma unroll
        for (int s = 0; s < STEP; ++s) {
            if (k0 + s >= kend || idle) break;
            int is[TO][TJ];
#pragma unroll
            for (int i = 0; i < TO; ++i)
#pragma unroll
                for (int j = 0; j < TJ; ++j) is[i][j] = 0;
#pragma unroll
            for (int w = 0; w < WORDS; ++w) {
                int av[TO];
                if constexpr (QUADS) {
                    av[0] = wq[s * WORDS + w][lane];
                } else {
#pragma unroll
                    for (int h = 0; h < RO / 4; ++h) {
                        const int4 a = *(const int4*)&wq[s * WORDS + w][tr * RO + 4 * h];
                        av[4 * h] = a.x, av[4 * h + 1] = a.y, av[4 * h + 2] = a.z, av[4 * h + 3] = a.w;
                    }
                }
                int bv[TJ];
                if constexpr (TJ == 4) {
                    const int4 b4 = *(const int4*)&xw[s * WORDS + w][row_at(0)];
                    bv[0] = b4.x, bv[1] = b4.y, bv[2] = b4.z, bv[3] = b4.w;
                } else {
                    for (int j = 0; j < TJ; ++j) bv[j] = xw[s * WORDS + w][row_at(j)];
                }
#pragma unroll
                for (int i = 0; i < TO; ++i)
#pragma unroll
                    for (int j = 0; j < TJ; ++j) is[i][j] = word_dot(av[i], bv[j], is[i][j], QT());
            }
#pragma unroll
            for (int i = 0; i < TO; ++i)
#pragma unroll
                for (int j = 0; j < TJ; ++j) acc[i][j] = fmaf(wd[s][out_at(i)] * xd[s][row_at(j)], (float)is[i][j], acc[i][j]);
        }
        __syncthreads();
    }
    unsigned long long before = 0;
    for (unsigned p = 0; p < pi; ++p) before += c.nout[p];
    for (int j = 0; j < TJ; ++j)
        for (int i = 0; i < TO; ++i) {
            const unsigned long long o = o0 + out_at(i), b = b0 + row_at(j);
            if (o >= nout || b >= nrows) continue;
            if (c.parts > 1) {
                c.partials[c.parts * nrows * before + blockIdx.z * nrows * nout + b * nout + o] = acc[i][j];
            } else {
                float* y = c.y[pi] + (ROUTED ? ent[b] : b) * nout + o;
                *y = c.add ? *y + acc[i][j] : acc[i][j];
            }
        }
}

// The twin products of the K-quants, Q5_0, Q5_1 and IQ4_NL: a lane takes eight values of one 32-value sub-block, whose term is ds * dot(q, x) - dm * sum(x) over the twin's integers.
// Every fused step is written as fmaf, the kernels compiling without contraction (CMakeLists.txt, LLMX_HIP_FLAGS).

// A row lane's share of a block (k_rows, the Vulkan row layout of matmul_row_one.glsl): `lanes` lanes a block, lane t holding `segments` runs, its run s from value xoff(t, s) of twin block xblock(t, s).
// decode gives a run's values as bytes, four a word; a type's KRaw names its lane as Lane, so KRaw stays the type's one decoder for the tile and the rows.

// Sub-block sub's 6-bit scale and min from a Q4_K or Q5_K block's 12 packed scale bytes, as rq::scale_min_k4 reads them.
__device__ inline void k45_scale_min(const unsigned (&sc)[3], unsigned sub, unsigned& s, unsigned& m) {
    const auto byte = [&](unsigned i) { return (sc[i >> 2] >> (8 * (i & 3))) & 0xffu; };
    if (sub < 4) {
        s = byte(sub) & 63, m = byte(sub + 4) & 63;
    } else {
        s = (byte(sub + 4) & 0xF) | ((byte(sub - 4) >> 6) << 4);
        m = (byte(sub + 4) >> 4) | ((byte(sub) >> 6) << 4);
    }
}
// Q4_K and Q5_K: eight lanes a block, lane t the 16 nibble bytes from 16 * (t % 2) of 64-value chunk c = t / 2, its low nibbles in sub-block 2c and its high nibbles in 2c + 1.
template <bool Q5> struct K45Lane {
    static constexpr int lanes = 8, segments = 2, values = 16;
    unsigned dm, sc[3], q[4], h[4];
    __device__ void load(const unsigned char* b, unsigned t) {
        const QWords<4> hd = *(const QWords<4>*)b, qw = *(const QWords<4>*)(b + (Q5 ? 48 : 16) + 32 * (t >> 1) + 16 * (t & 1));
        dm = hd.x, sc[0] = hd.y, sc[1] = hd.z, sc[2] = hd.w;
        q[0] = qw.x, q[1] = qw.y, q[2] = qw.z, q[3] = qw.w;
        if constexpr (Q5) {
            const QWords<4> hw = *(const QWords<4>*)(b + 16 + 16 * (t & 1));
            h[0] = hw.x, h[1] = hw.y, h[2] = hw.z, h[3] = hw.w;
        }
    }
    __device__ static unsigned xblock(unsigned t, int s) { return 2 * (t >> 1) + s; }
    __device__ static unsigned xoff(unsigned t, int) { return 16 * (t & 1); }
    __device__ void decode(unsigned t, int s, unsigned (&v)[4], float& ds, float& dmin) const {
        const unsigned sub = 2 * (t >> 1) + s;
        unsigned sv, mv;
        k45_scale_min(sc, sub, sv, mv);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[i] = (q[i] >> (4 * s)) & 0x0F0F0F0Fu;
            if constexpr (Q5) v[i] |= ((h[i] >> sub) & 0x01010101u) << 4;
        }
        ds = __half2float(__ushort_as_half((unsigned short)(dm & 0xffff))) * (float)sv;
        dmin = __half2float(__ushort_as_half((unsigned short)(dm >> 16))) * (float)mv;
    }
};
// Q6_K: sixteen lanes a block, lane t the 16 values of half t % 2 of 32-value block t / 2, which share one sub-scale; the block is 2-byte aligned, so its words come as 16-bit loads.
struct Q6KLane {
    static constexpr int lanes = 16, segments = 1, values = 16;
    unsigned lo[4], hi[4];
    unsigned short d;
    signed char sc;
    __device__ void load(const unsigned char* b, unsigned t) {
        const unsigned sub = t >> 1, n = sub >> 2, quarter = sub & 3, half = t & 1;
        const unsigned short* l = (const unsigned short*)(b + 64 * n + 32 * (quarter & 1) + 16 * half);
        const unsigned short* g = (const unsigned short*)(b + 128 + 32 * n + 16 * half);
#pragma unroll
        for (int i = 0; i < 4; ++i) lo[i] = l[2 * i] | ((unsigned)l[2 * i + 1] << 16), hi[i] = g[2 * i] | ((unsigned)g[2 * i + 1] << 16);
        sc = ((const signed char*)b)[192 + 8 * n + half + 2 * quarter];
        d = *(const unsigned short*)(b + 208);
    }
    __device__ static unsigned xblock(unsigned t, int) { return t >> 1; }
    __device__ static unsigned xoff(unsigned t, int) { return 16 * (t & 1); }
    __device__ void decode(unsigned t, int, unsigned (&v)[4], float& ds, float& dmin) const {
        const unsigned quarter = (t >> 1) & 3, ls = 4 * (quarter >> 1), hs = 2 * quarter;
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = ((lo[i] >> ls) & 0x0F0F0F0Fu) | (((hi[i] >> hs) & 0x03030303u) << 4);
        ds = __half2float(__ushort_as_half(d)) * (float)sc;
        dmin = 0.0f;
    }
};
// The fifth bits of a block's 32 values, bit j of h value j's, as bit 4 of each byte of eight words: words 0 to 3 hold values 0 to 15, words 4 to 7 values 16 to 31.
__device__ inline void fifth_bits32(unsigned h, unsigned (&v)[8]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const unsigned x = (h >> (4 * i)) & 0xF;
        v[i] |= ((x & 1) | ((x & 2) << 7) | ((x & 4) << 14) | ((x & 8) << 21)) << 4;
    }
}
// The 32 values of 16 nibble bytes l[0..3], low nibbles first (values 0 to 15), then high nibbles (16 to 31).
__device__ inline void nibbles32(const unsigned (&l)[4], unsigned (&v)[8]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = l[i] & 0x0F0F0F0Fu, v[4 + i] = (l[i] >> 4) & 0x0F0F0F0Fu;
}
// Q5_0: one lane a 22-byte block, 2-byte aligned, as 16-bit loads; its values less 16, taken out of the integer sum.
struct Q50Lane {
    static constexpr int lanes = 1, segments = 1, values = 32;
    unsigned short d;
    unsigned h, l[4];
    __device__ void load(const unsigned char* b, unsigned) {
        const unsigned short* w = (const unsigned short*)b;
        d = w[0], h = w[1] | ((unsigned)w[2] << 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) l[i] = w[3 + 2 * i] | ((unsigned)w[4 + 2 * i] << 16);
    }
    __device__ static unsigned xblock(unsigned, int) { return 0; }
    __device__ static unsigned xoff(unsigned, int) { return 0; }
    __device__ void decode(unsigned, int, unsigned (&v)[8], float& ds, float& dmin) const {
        nibbles32(l, v);
        fifth_bits32(h, v);
        ds = __half2float(__ushort_as_half(d));
        dmin = 0.0f;
    }
};
// Q5_1: one lane a 24-byte block, 4-byte aligned, its scale and min, fifth bits and nibbles as words; the min joins as a negative dm, so the term adds it.
struct Q51Lane {
    static constexpr int lanes = 1, segments = 1, values = 32;
    unsigned dm, h, l[4];
    __device__ void load(const unsigned char* b, unsigned) {
        const QWords<2> a = *(const QWords<2>*)b;
        const QWords<4> n = *(const QWords<4>*)(b + 8);
        dm = a.x, h = a.y, l[0] = n.x, l[1] = n.y, l[2] = n.z, l[3] = n.w;
    }
    __device__ static unsigned xblock(unsigned, int) { return 0; }
    __device__ static unsigned xoff(unsigned, int) { return 0; }
    __device__ void decode(unsigned, int, unsigned (&v)[8], float& ds, float& dmin) const {
        nibbles32(l, v);
        fifth_bits32(h, v);
        ds = __half2float(__ushort_as_half((unsigned short)(dm & 0xffff)));
        dmin = -__half2float(__ushort_as_half((unsigned short)(dm >> 16)));
    }
};

template <typename KT> struct KRaw;
template <typename KT> struct QuarterLane;
// Q4_K and Q5_K: the f16 scale and min word, the 12 packed scale bytes, the lane's eight nibble bytes and Q5_K's eight fifth-bit bytes, all 4-byte aligned since a block is a multiple of 16 bytes.
template <bool Q5> struct K45Raw {
    unsigned dm, sc[3], q[2], h[2];
    __device__ void load(const unsigned char* b, unsigned sub, unsigned part) {
        const unsigned* w = (const unsigned*)b;
        dm = w[0], sc[0] = w[1], sc[1] = w[2], sc[2] = w[3];
        const unsigned qo = (Q5 ? 48 : 16) + 32 * (sub >> 1) + 8 * part;
        q[0] = w[qo / 4], q[1] = w[qo / 4 + 1];
        if constexpr (Q5) h[0] = w[4 + 2 * part], h[1] = w[5 + 2 * part];
    }
    __device__ void decode(unsigned sub, unsigned, unsigned (&v)[2], float& ds, float& dmin) const {
        unsigned s, m;
        k45_scale_min(sc, sub, s, m);
        const unsigned sh = 4 * (sub & 1);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            v[i] = (q[i] >> sh) & 0x0F0F0F0Fu;
            if constexpr (Q5) v[i] |= ((h[i] >> sub) & 0x01010101u) << 4;
        }
        ds = __half2float(__ushort_as_half((unsigned short)(dm & 0xffff))) * (float)s;
        dmin = __half2float(__ushort_as_half((unsigned short)(dm >> 16))) * (float)m;
    }
};
template <> struct KRaw<rq::Q4_K> : K45Raw<false> { static constexpr int offset = 0, halves = 1; using Lane = K45Lane<false>; };
template <> struct KRaw<rq::Q5_K> : K45Raw<true> { static constexpr int offset = 0, halves = 1; using Lane = K45Lane<true>; };
// Q6_K: a 210-byte block is 2-byte aligned only, so its words come as 16-bit loads: the lane's eight low-nibble bytes and eight high-bit bytes, its 16 values' sub-scale and the block's scale.
// Its scale changes every 16 values, so the tile takes a sub-block as two halves of two parts, each with its own scale and term.
template <> struct KRaw<rq::Q6_K> {
    static constexpr int offset = 32, halves = 2;
    using Lane = Q6KLane;
    unsigned short lo[4], hi[4], d;
    signed char sc;
    __device__ void load(const unsigned char* b, unsigned sub, unsigned part) {
        const unsigned n = sub >> 2, quarter = sub & 3;
        const unsigned short* l = (const unsigned short*)(b + 64 * n + 32 * (quarter & 1) + 8 * part);
        const unsigned short* h = (const unsigned short*)(b + 128 + 32 * n + 8 * part);
#pragma unroll
        for (int i = 0; i < 4; ++i) lo[i] = l[i], hi[i] = h[i];
        sc = ((const signed char*)b)[192 + 8 * n + (part >> 1) + 2 * quarter];
        d = *(const unsigned short*)(b + 208);
    }
    __device__ void decode(unsigned sub, unsigned, unsigned (&v)[2], float& ds, float& dmin) const {
        const unsigned quarter = sub & 3, ls = 4 * (quarter >> 1), hs = 2 * quarter;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned l = lo[2 * i] | ((unsigned)lo[2 * i + 1] << 16), h = hi[2 * i] | ((unsigned)hi[2 * i + 1] << 16);
            v[i] = ((l >> ls) & 0x0F0F0F0Fu) | (((h >> hs) & 0x03030303u) << 4);
        }
        ds = __half2float(__ushort_as_half(d)) * (float)sc;
        dmin = 0.0f;
    }
};
// The fifth bits of a lane's eight values, bits 0 to 7 of h, as bit 4 of each byte of two words.
__device__ inline void fifth_bits(unsigned h, unsigned (&v)[2]) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const unsigned x = (h >> (4 * i)) & 0xF;
        v[i] |= ((x & 1) | ((x & 2) << 7) | ((x & 4) << 14) | ((x & 8) << 21)) << 4;
    }
}
// Q5_0: a 22-byte block, 2-byte aligned, its scale, 32 fifth bits and the lane's four nibble bytes' words as 16-bit loads; the values less 16, taken out of the integer sum.
template <> struct KRaw<rq::Q5_0> {
    static constexpr int offset = 16, halves = 1;
    using Lane = Q50Lane;
    unsigned short d, h[2], lo[4];
    __device__ void load(const unsigned char* b, unsigned, unsigned part) {
        const unsigned short* w = (const unsigned short*)b;
        d = w[0], h[0] = w[1], h[1] = w[2];
#pragma unroll
        for (int i = 0; i < 4; ++i) lo[i] = w[3 + 4 * (part & 1) + i];
    }
    __device__ void decode(unsigned, unsigned part, unsigned (&v)[2], float& ds, float& dmin) const {
        const unsigned sh = 4 * (part >> 1);
#pragma unroll
        for (int i = 0; i < 2; ++i) v[i] = ((lo[2 * i] | ((unsigned)lo[2 * i + 1] << 16)) >> sh) & 0x0F0F0F0Fu;
        fifth_bits(((h[0] | ((unsigned)h[1] << 16)) >> (8 * part)) & 0xff, v);
        ds = __half2float(__ushort_as_half(d));
        dmin = 0.0f;
    }
};
// Q5_1: a 24-byte block, 4-byte aligned, its scale and min, 32 fifth bits and the lane's nibble words; the min joins as a negative dm, so the term adds it.
template <> struct KRaw<rq::Q5_1> {
    static constexpr int offset = 0, halves = 1;
    using Lane = Q51Lane;
    unsigned dm, h, lo[2];
    __device__ void load(const unsigned char* b, unsigned, unsigned part) {
        const unsigned* w = (const unsigned*)b;
        dm = w[0], h = w[1], lo[0] = w[2 + 2 * (part & 1)], lo[1] = w[3 + 2 * (part & 1)];
    }
    __device__ void decode(unsigned, unsigned part, unsigned (&v)[2], float& ds, float& dmin) const {
        const unsigned sh = 4 * (part >> 1);
#pragma unroll
        for (int i = 0; i < 2; ++i) v[i] = (lo[i] >> sh) & 0x0F0F0F0Fu;
        fifth_bits((h >> (8 * part)) & 0xff, v);
        ds = __half2float(__ushort_as_half((unsigned short)(dm & 0xffff)));
        dmin = -__half2float(__ushort_as_half((unsigned short)(dm >> 16)));
    }
};
// IQ4_NL: an 18-byte block, 2-byte aligned, its scale and the lane's eight nibble bytes as 16-bit loads; each nibble's value from the format's grid as a signed byte.
template <> struct KRaw<rq::IQ4_NL> {
    static constexpr int offset = 0, halves = 1;
    using Lane = QuarterLane<rq::IQ4_NL>;   // one lane a block decodes 32 grid values and measured slower at every width (docs/STATUS.md)
    unsigned short d, lo[4];
    __device__ void load(const unsigned char* b, unsigned, unsigned part) {
        const unsigned short* w = (const unsigned short*)b;
        d = w[0];
#pragma unroll
        for (int i = 0; i < 4; ++i) lo[i] = w[1 + 4 * (part & 1) + i];
    }
    __device__ void decode(unsigned, unsigned part, unsigned (&v)[2], float& ds, float& dmin) const {
        const unsigned sh = 4 * (part >> 1);
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const unsigned l = lo[2 * i] | ((unsigned)lo[2 * i + 1] << 16);
            v[i] = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) v[i] |= ((unsigned)quant::IQ4_NL_VALUES[(l >> (8 * j + sh)) & 0xF] & 0xffu) << (8 * j);
        }
        ds = __half2float(__ushort_as_half(d));
        dmin = 0.0f;
    }
};
// The quarter form: four lanes a 32-value block, lane t its values 8t to 8t + 7 through the type's KRaw, the tile's loads and arithmetic.
// IQ4_NL's rows take it alone; a Q5_0 or Q5_1 row takes it or one lane a block by row_form (rocm_backend.cpp), whichever its lane units fill the cluster's steps fuller with.
template <typename KT> struct QuarterLane {
    static constexpr int lanes = 4, segments = 1, values = 8;
    KRaw<KT> raw;
    __device__ void load(const unsigned char* b, unsigned t) { raw.load(b, 0, t); }
    __device__ static unsigned xblock(unsigned, int) { return 0; }
    __device__ static unsigned xoff(unsigned t, int) { return 8 * t; }
    __device__ void decode(unsigned t, int, unsigned (&v)[2], float& ds, float& dmin) const { raw.decode(0, t, v, ds, dmin); }
};
// Whether a type's decoded bytes are signed, which its 16-bit words then extend.
template <typename KT> struct KSigned { static constexpr bool value = false; };
template <> struct KSigned<rq::IQ4_NL> { static constexpr bool value = true; };
// The eight byte values as the twin's words: two words of four bytes, or four of two 16-bit values.
template <typename QT> struct KWords { static constexpr int n = sizeof(QT) == 2 ? 4 : 2; };
template <typename QT, bool S>
__device__ inline void widen(const unsigned (&v)[2], int (&w)[KWords<QT>::n]) {
    if constexpr (sizeof(QT) == 2 && S) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int x = (int)v[i];
            w[2 * i] = (((x << 24) >> 24) & 0xffff) | (((x << 16) >> 24) << 16);
            w[2 * i + 1] = (((x << 8) >> 24) & 0xffff) | ((x >> 24) << 16);
        }
    } else if constexpr (sizeof(QT) == 2) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            w[2 * i] = (int)__builtin_amdgcn_perm(0u, v[i], 0x0c010c00u);
            w[2 * i + 1] = (int)__builtin_amdgcn_perm(0u, v[i], 0x0c030c02u);
        }
    } else {
        w[0] = (int)v[0], w[1] = (int)v[1];
    }
}
// A sub-block's term from its exact integer dot and twin sum.
template <typename KT>
__device__ inline float k_term(float ds, float dmin, int isum, int xsum) {
    if constexpr (KRaw<KT>::offset) return ds * (float)(isum - KRaw<KT>::offset * xsum);
    else return fmaf(ds, (float)isum, -(dmin * (float)xsum));
}

// The R * C sums of a cluster's rows o to o + R - 1, each stored by one lane: by the transposed tree where the cluster has a lane for every value, else value by value.
// Either way each sum has the bits of one xor tree over the cluster.
template <int R, int C, typename M>
__device__ inline void store_cluster_sums(float (&acc)[R][C], const Row (&r)[R], int lane, unsigned cl, unsigned long long o, unsigned long long total, int ncols, int add, const M& m) {
    constexpr int NV = R * C;
    float v[NV];
#pragma unroll
    for (int j = 0; j < R; ++j)
#pragma unroll
        for (int c = 0; c < C; ++c) v[j * C + c] = acc[j][c];
    const unsigned sl = lane & (cl - 1);
    if (cl >= NV) {
        if constexpr (NV > 1) halve<NV / 2>(v, lane, (int)cl / 2);
        const unsigned share = cl / NV;
        const float s = wave_sum(v[0], (int)share / 2);
        const int k = (int)(sl / share);
        store_sum<R>(r, s, k / C, k % C, sl % share == 0, o, total, ncols, add, m);
        return;
    }
#pragma unroll
    for (int i = 0; i < NV; ++i) {
        float s = v[i];
        for (unsigned d = cl / 2; d > 0; d /= 2) s += lane_xor(s, (int)d);
        store_sum<R>(r, s, i / C, i % C, sl == 0, o, total, ncols, add, m);
    }
}

// A cluster of rc.cluster lanes computes R output rows for up to C generated rows; each lane adds its runs' terms into F32 in block order, and the cluster folds them by one xor tree.
// The cluster and the lane follow the row's width and type alone (rocm_backend.cpp, row_cluster and row_form), so a row and a column keep their bits in every build.
template <typename KT, int C, typename QT, int R, int G, typename M = Cols, typename L = typename KRaw<KT>::Lane>
__device__ void k_rows(const RowCall& rc, const QT* q, const float* dx, unsigned long long nin, int ncols, int add, const M& m = M{}) {
    constexpr int NV = L::values / 4, NW = sizeof(QT) == 2 ? 2 * NV : NV;
    constexpr int ONES = sizeof(QT) == 2 ? 0x00010001 : 0x01010101;
    const int lanes = warpSize, lane = threadIdx.x % lanes;
    const unsigned cl = rc.cluster, sl = lane & (cl - 1), groups = lanes / cl;
    const unsigned long long blocks = nin / 32, kblocks = nin / KT::values, units = kblocks * L::lanes, row_bytes = kblocks * KT::bytes;
    const unsigned long long waves = (unsigned long long)gridDim.x * (kThreads / lanes), wave = (unsigned long long)blockIdx.x * (kThreads / lanes) + threadIdx.x / lanes;
    for (unsigned long long base = R * groups * wave; base < rc.total; base += R * groups * waves) {
        const unsigned long long o = base + R * (lane / cl);
        Row r[R];
#pragma unroll
        for (int j = 0; j < R; ++j) r[j] = row_of(rc, o + j < rc.total ? o + j : rc.total - 1, row_bytes);
        float acc[R][C];
#pragma unroll
        for (int j = 0; j < R; ++j)
#pragma unroll
            for (int c = 0; c < C; ++c) acc[j][c] = 0.0f;
        for (unsigned long long u = sl; u < units; u += cl) {
            const unsigned long long kb = u / L::lanes;
            const unsigned t = (unsigned)(u % L::lanes);
            L raw[R];
#pragma unroll
            for (int j = 0; j < R; ++j) raw[j].load(r[j].w + kb * KT::bytes, t);
#pragma unroll
            for (int s = 0; s < L::segments; ++s) {
                int ww[R][NW];
                float ds[R], dm[R];
#pragma unroll
                for (int j = 0; j < R; ++j) {
                    unsigned v[NV];
                    raw[j].decode(t, s, v, ds[j], dm[j]);
#pragma unroll
                    for (int i = 0; i < NV; i += 2) {
                        const unsigned pair[2] = {v[i], v[i + 1]};
                        int w2[KWords<QT>::n];
                        widen<QT, KSigned<KT>::value>(pair, w2);
#pragma unroll
                        for (int x = 0; x < KWords<QT>::n; ++x) ww[j][i / 2 * KWords<QT>::n + x] = w2[x];
                    }
                }
                const unsigned long long xb = kb * (KT::values / 32) + L::xblock(t, s);
                const unsigned xo = L::xoff(t, s);
#pragma unroll
                for (int g = 0; g < C; g += G) {
                    int xw[G][NW];
                    float xd[G];
#pragma unroll
                    for (int c = 0; c < G; ++c) {
                        const unsigned long long col = m.x(g + c < ncols ? g + c : 0) * blocks;
                        const int* qc = (const int*)(q + (col + xb) * 32 + xo);
                        if constexpr (NW % 4 == 0) {
#pragma unroll
                            for (int i = 0; i < NW; i += 4) {
                                const int4 x = *(const int4*)(qc + i);
                                xw[c][i] = x.x, xw[c][i + 1] = x.y, xw[c][i + 2] = x.z, xw[c][i + 3] = x.w;
                            }
                        } else {
                            static_assert(NW == 2, "a lane's twin words come as int4 or int2 loads");
                            const int2 x = *(const int2*)qc;
                            xw[c][0] = x.x, xw[c][1] = x.y;
                        }
                        xd[c] = dx[col + xb];
                    }
#pragma unroll
                    for (int c = 0; c < G; ++c) {
                        int xsum = 0;
#pragma unroll
                        for (int i = 0; i < NW; ++i) xsum = word_dot(ONES, xw[c][i], xsum, QT(0));
#pragma unroll
                        for (int j = 0; j < R; ++j) {
                            int isum = 0;
#pragma unroll
                            for (int i = 0; i < NW; ++i) isum = word_dot(ww[j][i], xw[c][i], isum, QT(0));
                            acc[j][g + c] = fmaf(xd[c], k_term<KT>(ds[j], dm[j], isum, xsum), acc[j][g + c]);
                        }
                    }
                }
            }
        }
        store_cluster_sums<R, C>(acc, r, lane, cl, o, rc.total, ncols, add, m);
    }
}

// The K-quants' prompt tile: q8_0_tile's 64 outputs by 64 rows and its staging, each weight row's twin block staged by four threads with the scale and min of its halves.
// Each output sums its blocks, and a block its halves, in order, acc = fma(dx, term, acc), so a row gives the same bits wherever the tile falls.
template <typename KT, typename QT, bool ROUTED = false, int ROWS = kTileRows>
__device__ void k_tile(const TileCall& c, const QT* q, const float* dx, const TileGroups& tg = TileGroups{}) {
    constexpr int STEP = TileStep<QT, 4>::n, WORDS = 32 * (int)sizeof(QT) / 4, OUTS = 64, NW = KWords<QT>::n, H = KRaw<KT>::halves;
    constexpr int ONES = sizeof(QT) == 2 ? 0x00010001 : 0x01010101, J = ROWS / 16;   // J a thread's rows
    const unsigned pi = c.nproj > 2 && blockIdx.x >= c.start[2] ? 2 : c.nproj > 1 && blockIdx.x >= c.start[1] ? 1 : 0;
    const unsigned long long gi = ROUTED ? tg.first + blockIdx.y : 0;
    const unsigned long long nout = c.nout[pi], nrows = ROUTED ? tg.tab[4 * gi + 2] : c.nrows, nin = c.nin;
    if (ROUTED && (!nrows || (nrows <= kNarrowRows) != (ROWS == kNarrowRows))) return;   // as q8_0_tile's groups
    const unsigned* ent = ROUTED ? tg.tab + tg.order0 + tg.tab[4 * gi + 1] : nullptr;
    const unsigned long long blocks = nin / 32, row_bytes = nin / KT::values * KT::bytes;
    __shared__ __attribute__((aligned(16))) int wq[STEP * WORDS][OUTS];
    __shared__ __attribute__((aligned(16))) int xw[STEP * WORDS][ROWS];
    __shared__ float wd[STEP][H][OUTS], wm[STEP][H][OUTS], xd[STEP][ROWS];
    const int t = threadIdx.x, tr = t / 16, tc = t % 16;
    // The narrow routed build gives each wave one quad of the group's entries, rotated by workgroup since wave w runs on SIMD w; a quad holding no entry skips its dots.
    constexpr bool QUADS = ROUTED && ROWS == kNarrowRows;
    constexpr int TO = QUADS ? 1 : 4, TJ = QUADS ? 4 : J;   // a thread's outputs and rows
    const int quad = (int)((t / 64 + blockIdx.x + blockIdx.y) % 4), lane = t % 64;
    const auto out_at = [&](int i) { return QUADS ? lane : tr * 4 + i; };
    const auto row_at = [&](int j) { return QUADS ? 4 * quad + j : tc * J + j; };
    const bool idle = QUADS && 4ull * quad >= nrows;
    const unsigned long long o0 = (unsigned long long)(blockIdx.x - c.start[pi]) * OUTS, b0 = ROUTED ? 0 : (unsigned long long)blockIdx.y * ROWS;
    const unsigned long long kbeg = (unsigned long long)blockIdx.z * c.kper;
    const unsigned long long kend = kbeg + c.kper < blocks ? kbeg + c.kper : blocks;
    float acc[TO][TJ];
    for (int i = 0; i < TO; ++i)
        for (int j = 0; j < TJ; ++j) acc[i][j] = 0.0f;
    // Weight row wr's twin block takes four threads, part ws each; the twin's row r four threads, a quarter of a block each.
    const int r = t / 4, seg = t % 4, wr = t / 4, ws = t % 4;
    const unsigned long long o = o0 + wr, b = b0 + r;
    const unsigned xr = ROUTED ? (b < nrows ? (unsigned)(ent[b] / tg.per) : 0u) : (unsigned)r;   // the twin row this thread stages
    const unsigned char* wrow = c.w[pi] + (ROUTED ? tg.tab[4 * gi] * tg.stride[pi] : 0) + (o < nout ? o : 0) * row_bytes;
    KRaw<KT> raw[STEP];
    bool wok[STEP], xok[STEP];
    int xv[STEP][WORDS / 4];
    float xs_[STEP];
    const QT* qt = q + b0 * c.xrow * 32;
    const float* dxt = dx + b0 * c.xrow;
    const auto load = [&](unsigned long long k0) {
#pragma unroll
        for (int s = 0; s < STEP; ++s) {
            const unsigned long long k = k0 + s;
            xok[s] = k < kend && b < nrows;
            wok[s] = k < kend && o < nout;
            const unsigned long long kk = wok[s] ? k : 0;
            raw[s].load(wrow + (kk / (KT::values / 32)) * KT::bytes, (unsigned)(kk % (KT::values / 32)), ws);
            const unsigned xb = xok[s] ? xr * c.xrow + (unsigned)k * c.xblock : 0u;
            if constexpr (sizeof(QT) == 2) {
                const int4 v = at<int4>(qt, xb * 64u + seg * 16u);
                xv[s][0] = v.x, xv[s][1] = v.y, xv[s][2] = v.z, xv[s][3] = v.w;
            } else {
                const int2 v = at<int2>(qt, xb * 32u + seg * 8u);
                xv[s][0] = v.x, xv[s][1] = v.y;
            }
            xs_[s] = at<float>(dxt, xb * 4u);
        }
    };
    const auto store = [&](unsigned long long k0) {
#pragma unroll
        for (int s = 0; s < STEP; ++s) {
            unsigned v[2];
            float ds, dm;
            raw[s].decode((unsigned)((k0 + s) % (KT::values / 32)), ws, v, ds, dm);
            int w[NW];
            widen<QT, KSigned<KT>::value>(v, w);
#pragma unroll
            for (int i = 0; i < NW; ++i) wq[s * WORDS + ws * NW + i][wr] = wok[s] ? w[i] : 0;
            if (ws % (4 / H) == 0) wd[s][ws / (4 / H)][wr] = wok[s] ? ds : 0.0f, wm[s][ws / (4 / H)][wr] = wok[s] ? dm : 0.0f;
#pragma unroll
            for (int w2 = 0; w2 < WORDS / 4; ++w2)
                if (r < ROWS) xw[s * WORDS + seg * (WORDS / 4) + w2][r] = xok[s] ? xv[s][w2] : 0;
            if (seg == 0 && r < ROWS) xd[s][r] = xok[s] ? xs_[s] : 0.0f;
        }
    };
    load(kbeg);
    for (unsigned long long k0 = kbeg; k0 < kend; k0 += STEP) {
        store(k0);
        __syncthreads();
        if (k0 + STEP < kend) load(k0 + STEP);
#pragma unroll
        for (int s = 0; s < STEP; ++s) {
            if (k0 + s >= kend || idle) break;
#pragma unroll
            for (int h = 0; h < H; ++h) {
                int is[TO][TJ], xsum[TJ];
#pragma unroll
                for (int j = 0; j < TJ; ++j) xsum[j] = 0;
#pragma unroll
                for (int i = 0; i < TO; ++i)
#pragma unroll
                    for (int j = 0; j < TJ; ++j) is[i][j] = 0;
#pragma unroll
                for (int w = h * (WORDS / H); w < (h + 1) * (WORDS / H); ++w) {
                    int av[TO];
                    if constexpr (QUADS) {
                        av[0] = wq[s * WORDS + w][lane];
                    } else {
                        const int4 a = *(const int4*)&wq[s * WORDS + w][tr * 4];
                        av[0] = a.x, av[1] = a.y, av[2] = a.z, av[3] = a.w;
                    }
                    int bv[TJ];
                    if constexpr (TJ == 4) {
                        const int4 b4 = *(const int4*)&xw[s * WORDS + w][row_at(0)];
                        bv[0] = b4.x, bv[1] = b4.y, bv[2] = b4.z, bv[3] = b4.w;
                    } else {
                        for (int j = 0; j < TJ; ++j) bv[j] = xw[s * WORDS + w][row_at(j)];
                    }
#pragma unroll
                    for (int j = 0; j < TJ; ++j) xsum[j] = word_dot(ONES, bv[j], xsum[j], QT());
#pragma unroll
                    for (int i = 0; i < TO; ++i)
#pragma unroll
                        for (int j = 0; j < TJ; ++j) is[i][j] = word_dot(av[i], bv[j], is[i][j], QT());
                }
#pragma unroll
                for (int i = 0; i < TO; ++i)
#pragma unroll
                    for (int j = 0; j < TJ; ++j)
                        acc[i][j] = fmaf(xd[s][row_at(j)], k_term<KT>(wd[s][h][out_at(i)], wm[s][h][out_at(i)], is[i][j], xsum[j]), acc[i][j]);
            }
        }
        __syncthreads();
    }
    unsigned long long before = 0;
    for (unsigned p = 0; p < pi; ++p) before += c.nout[p];
    for (int j = 0; j < TJ; ++j)
        for (int i = 0; i < TO; ++i) {
            const unsigned long long oo = o0 + out_at(i), bb = b0 + row_at(j);
            if (oo >= nout || bb >= nrows) continue;
            if (c.parts > 1) {
                c.partials[c.parts * nrows * before + blockIdx.z * nrows * nout + bb * nout + oo] = acc[i][j];
            } else {
                float* y = c.y[pi] + (ROUTED ? ent[bb] : bb) * nout + oo;
                *y = c.add ? *y + acc[i][j] : acc[i][j];
            }
        }
}

// A routed row call (backend.hpp matmul_experts): the row kernel's call over expert 0, each projection's bytes an expert, the entries' ids, the launch's first entry and the entries a twin column serves.
struct RoutedCall {
    RowCall c;
    unsigned long long stride[3];
    const unsigned* ids;
    unsigned long long first, per, n_expert;
};
// The call over expert id's rows, field by field, so the call stays in registers rather than a private copy.
__device__ inline RowCall with_expert(const RoutedCall& rc, unsigned id) {
    RowCall c;
#pragma unroll
    for (int p = 0; p < 3; ++p) {
        c.w[p] = rc.c.w[p] + id * rc.stride[p];
        c.y[p] = rc.c.y[p];
        c.nout[p] = rc.c.nout[p];
        c.start[p] = rc.c.start[p];
    }
    c.total = rc.c.total;
    c.nproj = rc.c.nproj;
    c.cluster = rc.c.cluster;
    return c;
}
// Entry first + blockIdx.y's call over its expert's rows into its own outputs, and its twin column; false for an entry without an expert, which writes nothing.
__device__ inline bool routed_entry(const RoutedCall& rc, RowCall& c, unsigned long long& col) {
    const unsigned long long e = rc.first + blockIdx.y;
    const unsigned id = rc.ids[e];
    if (id >= rc.n_expert) return false;
    c = with_expert(rc, id);
#pragma unroll
    for (int p = 0; p < 3; ++p) c.y[p] += e * rc.c.nout[p];
    col = e / rc.per;
    return true;
}
// Group first + blockIdx.y of a grouped call (kernels/f32.hip, llmx_moe_group): its expert's call over the expert's rows, its entries from word order0 of tab, and their count, 0 for an unused group.
__device__ inline int group_entries(const RoutedCall& rc, const unsigned* tab, unsigned long long order0, RowCall& c, const unsigned*& entries) {
    const unsigned long long g = rc.first + blockIdx.y;
    const int n = (int)tab[4 * g + 2];
    if (!n) return 0;
    const unsigned id = tab[4 * g];
    c = with_expert(rc, id);
    entries = tab + order0 + tab[4 * g + 1];
    return n;
}

} // namespace


// Every kernel here runs blocks of kThreads, which __launch_bounds__ tells the compiler, since its default of 1024 threads spilled the tile's prefetch to scratch.
// The row kernel's builds: R output rows a wave and G columns whose loads issue together, the host choosing by the call's columns and rows (rocm_backend.cpp, row_build).
#define LLMX_ROWS(name, C, QT, R, G)                                                                                                                   \
    extern "C" __global__ void __launch_bounds__(kThreads) name(RowCall c, const QT* q, const float* dx, unsigned long long nin, int ncols, int add) { \
        q8_0_rows<C, QT, R, G>(c, q, dx, nin, ncols, add);                                                                                             \
    }
LLMX_ROWS(llmx_q8_0_rows16_1, 1, short, 2, 1)
LLMX_ROWS(llmx_q8_0_rows16_2, 2, short, 2, 2)
LLMX_ROWS(llmx_q8_0_rows16_2_r4, 2, short, 4, 2)
LLMX_ROWS(llmx_q8_0_rows16_4, 4, short, 2, 4)
LLMX_ROWS(llmx_q8_0_rows16_4_r4, 4, short, 4, 4)
LLMX_ROWS(llmx_q8_0_rows16_8, 8, short, 2, 4)
LLMX_ROWS(llmx_q8_0_rows16_8_r4, 8, short, 4, 4)
LLMX_ROWS(llmx_q8_0_rows16_16, 16, short, 2, 4)
LLMX_ROWS(llmx_q8_0_rows8_1, 1, signed char, 2, 1)
LLMX_ROWS(llmx_q8_0_rows8_2, 2, signed char, 2, 2)
LLMX_ROWS(llmx_q8_0_rows8_2_r4, 2, signed char, 4, 2)
LLMX_ROWS(llmx_q8_0_rows8_4, 4, signed char, 2, 4)
LLMX_ROWS(llmx_q8_0_rows8_4_r4, 4, signed char, 4, 4)
LLMX_ROWS(llmx_q8_0_rows8_8, 8, signed char, 2, 4)
LLMX_ROWS(llmx_q8_0_rows8_8_r4, 8, signed char, 4, 4)
LLMX_ROWS(llmx_q8_0_rows8_16, 16, signed char, 4, 8)
#define LLMX_HALF(name, C, R, G)                                                                                                                                \
    extern "C" __global__ void __launch_bounds__(kThreads) name(RowCall c, const signed char* q, const float* dx, unsigned long long nin, int ncols, int add) { \
        q8_0_rows_half<C, R, G>(c, q, dx, nin, ncols, add);                                                                                                     \
    }
extern "C" __global__ void __launch_bounds__(kThreads) llmx_q8_0_half8_1(RowCall c, const signed char* q, const float* dx, unsigned long long nin, int ncols, int add) {
    q8_0_half_one<1, kSteps>(c, q, dx, nin, add);
}
extern "C" __global__ void __launch_bounds__(kThreads) llmx_q8_0_half8_1p(RowCall c, const signed char* q, const float* dx, unsigned long long nin, int ncols, int add) {
    q8_0_half_one<1, kSteps, true>(c, q, dx, nin, add);
}
extern "C" __global__ void __launch_bounds__(kThreads) llmx_q8_0_half8_1r2(RowCall c, const signed char* q, const float* dx, unsigned long long nin, int ncols, int add) {
    q8_0_half_one<2, 2>(c, q, dx, nin, add);
}
LLMX_HALF(llmx_q8_0_half8_2, 2, 2, 2)
LLMX_HALF(llmx_q8_0_half8_2_r4, 2, 4, 2)
LLMX_HALF(llmx_q8_0_half8_4, 4, 2, 4)
LLMX_HALF(llmx_q8_0_half8_4_r4, 4, 4, 4)
LLMX_HALF(llmx_q8_0_half8_8, 8, 2, 4)
LLMX_HALF(llmx_q8_0_half8_8_r4, 8, 4, 4)
LLMX_HALF(llmx_q8_0_half8_16, 16, 4, 4)

extern "C" __global__ void __launch_bounds__(kThreads) llmx_q8_0_tile16(TileCall c, const short* q, const float* dx) { q8_0_tile<short, 4>(c, q, dx); }
extern "C" __global__ void __launch_bounds__(kThreads) llmx_q8_0_tile8(TileCall c, const signed char* q, const float* dx) { q8_0_tile<signed char, 4>(c, q, dx); }

// The parts of a split tile call added in part order, plus y when adding (the Vulkan reduce's order, matmul_reduce.comp): a thread an output, projection p's outputs from workgroup c.start[p] on.
extern "C" __global__ void __launch_bounds__(kThreads) llmx_tile_reduce(TileCall c) {
    const unsigned pi = c.nproj > 2 && blockIdx.x >= c.start[2] ? 2 : c.nproj > 1 && blockIdx.x >= c.start[1] ? 1 : 0;
    const unsigned long long n = c.nout[pi] * c.nrows, local = (unsigned long long)(blockIdx.x - c.start[pi]) * kThreads + threadIdx.x;
    if (local >= n) return;
    unsigned long long before = 0;
    for (unsigned p = 0; p < pi; ++p) before += c.nout[p];
    const float* part = c.partials + c.parts * c.nrows * before + local;
    float s = c.add ? c.y[pi][local] : 0.0f;
    for (unsigned z = 0; z < c.parts; ++z) s += part[z * n];
    c.y[pi][local] = s;
}

extern "C" __global__ void __launch_bounds__(kThreads) llmx_q8_0_tile8_tall(TileCall c, const signed char* q, const float* dx) { q8_0_tile<signed char, 8>(c, q, dx); }
extern "C" __global__ void __launch_bounds__(kThreads) llmx_q8_0_tile16_tall(TileCall c, const short* q, const float* dx) { q8_0_tile<short, 8>(c, q, dx); }
// The routed prompt tiles: the 64-output tile over a group a block row (TileGroups), 64 rows a group or, narrow, 16.
extern "C" __global__ void __launch_bounds__(kThreads) llmx_q8_0_routed_tile16(TileCall c, TileGroups g, const short* q, const float* dx) { q8_0_tile<short, 4, true>(c, q, dx, g); }
extern "C" __global__ void __launch_bounds__(kThreads) llmx_q8_0_routed_tile8(TileCall c, TileGroups g, const signed char* q, const float* dx) { q8_0_tile<signed char, 4, true>(c, q, dx, g); }
extern "C" __global__ void __launch_bounds__(kThreads) llmx_q8_0_routed_narrow16(TileCall c, TileGroups g, const short* q, const float* dx) { q8_0_tile<short, 4, true, kNarrowRows>(c, q, dx, g); }
extern "C" __global__ void __launch_bounds__(kThreads) llmx_q8_0_routed_narrow8(TileCall c, TileGroups g, const signed char* q, const float* dx) { q8_0_tile<signed char, 4, true, kNarrowRows>(c, q, dx, g); }

// The twin types' row builds and tiles but Q8_0's, a type's set named by its type: the builds the host takes by a call's columns (rocm_backend.cpp, TwinKernels), two rows a wave.
#define LLMX_KROWS(name, KT, C, QT, G, L)                                                                                                               \
    extern "C" __global__ void __launch_bounds__(kThreads) name(RowCall c, const QT* q, const float* dx, unsigned long long nin, int ncols, int add) { \
        k_rows<KT, C, QT, 2, G, Cols, L>(c, q, dx, nin, ncols, add);                                                                                     \
    }
#define LLMX_KTILE(name, KT, QT) \
    extern "C" __global__ void __launch_bounds__(kThreads) name(TileCall c, const QT* q, const float* dx) { k_tile<KT, QT>(c, q, dx); }
#define LLMX_KTILE_ROUTED(name, KT, QT, ROWS) \
    extern "C" __global__ void __launch_bounds__(kThreads) name(TileCall c, TileGroups g, const QT* q, const float* dx) { k_tile<KT, QT, true, ROWS>(c, q, dx, g); }
// A routed build runs a type's one-column body over an entry's expert and twin column, so each entry has the bits of its expert's one-column product.
#define LLMX_ROUTED(name, QT, ...)                                                                                                       \
    extern "C" __global__ void __launch_bounds__(kThreads) name(RoutedCall rc, const QT* q, const float* dx, unsigned long long nin) { \
        RowCall c;                                                                                                                       \
        unsigned long long col;                                                                                                          \
        if (!routed_entry(rc, c, col)) return;                                                                                           \
        __VA_ARGS__(c, q + col * (nin / 32) * 32, dx + col * (nin / 32), nin, 1, 0);                                                     \
    }
LLMX_ROUTED(llmx_q8_0_routed16, short, q8_0_rows<1, short, 2, 1>)
LLMX_ROUTED(llmx_q8_0_routed8, signed char, q8_0_rows<1, signed char, 2, 1>)
extern "C" __global__ void __launch_bounds__(kThreads) llmx_q8_0_routed_half8(RoutedCall rc, const signed char* q, const float* dx, unsigned long long nin) {
    RowCall c;
    unsigned long long col;
    if (!routed_entry(rc, c, col)) return;
    q8_0_half_one<1, kSteps>(c, q + col * (nin / 32) * 32, dx + col * (nin / 32), nin, 0);
}
// A grouped build runs a row build of C columns over one expert's entries (EntryCols), a column an entry, so each entry has the bits of its one-column product and the expert's rows are read once a group.
#define LLMX_GROUPED(name, QT, C, ...)                                                                                                                                                 \
    extern "C" __global__ void __launch_bounds__(kThreads) name(RoutedCall rc, const unsigned* tab, unsigned long long order0, const QT* q, const float* dx, unsigned long long nin) { \
        RowCall c;                                                                                                                                                                     \
        const unsigned* e;                                                                                                                                                             \
        const int n = group_entries(rc, tab, order0, c, e);                                                                                                                            \
        if (!n) return;                                                                                                                                                                \
        const EntryCols<C> m(e, n, rc.per);                                                                                                                                            \
        __VA_ARGS__(c, q, dx, nin, n, 0, m);                                                                                                                                           \
    }
LLMX_GROUPED(llmx_q8_0_routed16_g2, short, 2, q8_0_rows<2, short, 2, 2, EntryCols<2>>)
LLMX_GROUPED(llmx_q8_0_routed16_g4, short, 4, q8_0_rows<4, short, 2, 4, EntryCols<4>>)
LLMX_GROUPED(llmx_q8_0_routed16_g8, short, 8, q8_0_rows<8, short, 2, 4, EntryCols<8>>)
LLMX_GROUPED(llmx_q8_0_routed8_g2, signed char, 2, q8_0_rows<2, signed char, 2, 2, EntryCols<2>>)
LLMX_GROUPED(llmx_q8_0_routed8_g4, signed char, 4, q8_0_rows<4, signed char, 2, 4, EntryCols<4>>)
LLMX_GROUPED(llmx_q8_0_routed8_g8, signed char, 8, q8_0_rows<8, signed char, 2, 4, EntryCols<8>>)
LLMX_GROUPED(llmx_q8_0_routed_half8_g2, signed char, 2, q8_0_rows_half<2, 2, 2, EntryCols<2>>)
LLMX_GROUPED(llmx_q8_0_routed_half8_g4, signed char, 4, q8_0_rows_half<4, 2, 4, EntryCols<4>>)
LLMX_GROUPED(llmx_q8_0_routed_half8_g8, signed char, 8, q8_0_rows_half<8, 2, 4, EntryCols<8>>)
#define LLMX_KTYPE(t, KT)                                   \
    LLMX_KROWS(llmx_##t##_rows16_1, KT, 1, short, 1, KRaw<KT>::Lane) \
    LLMX_KROWS(llmx_##t##_rows16_2, KT, 2, short, 2, KRaw<KT>::Lane) \
    LLMX_KROWS(llmx_##t##_rows16_4, KT, 4, short, 4, KRaw<KT>::Lane) \
    LLMX_KROWS(llmx_##t##_rows16_8, KT, 8, short, 4, KRaw<KT>::Lane) \
    LLMX_KROWS(llmx_##t##_rows16_16, KT, 16, short, 4, KRaw<KT>::Lane) \
    LLMX_KROWS(llmx_##t##_rows8_1, KT, 1, signed char, 1, KRaw<KT>::Lane) \
    LLMX_KROWS(llmx_##t##_rows8_2, KT, 2, signed char, 2, KRaw<KT>::Lane) \
    LLMX_KROWS(llmx_##t##_rows8_4, KT, 4, signed char, 4, KRaw<KT>::Lane) \
    LLMX_KROWS(llmx_##t##_rows8_8, KT, 8, signed char, 4, KRaw<KT>::Lane) \
    LLMX_KROWS(llmx_##t##_rows8_16, KT, 16, signed char, 4, KRaw<KT>::Lane) \
    LLMX_KTILE(llmx_##t##_tile16, KT, short)                \
    LLMX_KTILE(llmx_##t##_tile8, KT, signed char)           \
    LLMX_ROUTED(llmx_##t##_routed16, short, k_rows<KT, 1, short, 2, 1>) \
    LLMX_ROUTED(llmx_##t##_routed8, signed char, k_rows<KT, 1, signed char, 2, 1>)                      \
    LLMX_GROUPED(llmx_##t##_routed16_g2, short, 2, k_rows<KT, 2, short, 2, 2, EntryCols<2>>)            \
    LLMX_GROUPED(llmx_##t##_routed16_g4, short, 4, k_rows<KT, 4, short, 2, 4, EntryCols<4>>)            \
    LLMX_GROUPED(llmx_##t##_routed16_g8, short, 8, k_rows<KT, 8, short, 2, 4, EntryCols<8>>)            \
    LLMX_GROUPED(llmx_##t##_routed8_g2, signed char, 2, k_rows<KT, 2, signed char, 2, 2, EntryCols<2>>) \
    LLMX_GROUPED(llmx_##t##_routed8_g4, signed char, 4, k_rows<KT, 4, signed char, 2, 4, EntryCols<4>>) \
    LLMX_GROUPED(llmx_##t##_routed8_g8, signed char, 8, k_rows<KT, 8, signed char, 2, 4, EntryCols<8>>) \
    LLMX_KTILE_ROUTED(llmx_##t##_routed_tile16, KT, short, kTileRows)                                  \
    LLMX_KTILE_ROUTED(llmx_##t##_routed_tile8, KT, signed char, kTileRows)                             \
    LLMX_KTILE_ROUTED(llmx_##t##_routed_narrow16, KT, short, kNarrowRows)                              \
    LLMX_KTILE_ROUTED(llmx_##t##_routed_narrow8, KT, signed char, kNarrowRows)
LLMX_KTYPE(q4_k, rq::Q4_K)
LLMX_KTYPE(q5_k, rq::Q5_K)
LLMX_KTYPE(q6_k, rq::Q6_K)
LLMX_KTYPE(q5_0, rq::Q5_0)
LLMX_KTYPE(q5_1, rq::Q5_1)
LLMX_KTYPE(iq4_nl, rq::IQ4_NL)
// The quarter form's row, routed and grouped builds of Q5_0 and Q5_1, in the order LLMX_KTYPE gives those builds (rocm_backend.cpp, quarter_twins).
#define LLMX_KQUARTER(t, KT)                                                                                                       \
    LLMX_KROWS(llmx_##t##_quarter_rows16_1, KT, 1, short, 1, QuarterLane<KT>)                                                      \
    LLMX_KROWS(llmx_##t##_quarter_rows16_2, KT, 2, short, 2, QuarterLane<KT>)                                                      \
    LLMX_KROWS(llmx_##t##_quarter_rows16_4, KT, 4, short, 4, QuarterLane<KT>)                                                      \
    LLMX_KROWS(llmx_##t##_quarter_rows16_8, KT, 8, short, 4, QuarterLane<KT>)                                                      \
    LLMX_KROWS(llmx_##t##_quarter_rows16_16, KT, 16, short, 4, QuarterLane<KT>)                                                    \
    LLMX_KROWS(llmx_##t##_quarter_rows8_1, KT, 1, signed char, 1, QuarterLane<KT>)                                                 \
    LLMX_KROWS(llmx_##t##_quarter_rows8_2, KT, 2, signed char, 2, QuarterLane<KT>)                                                 \
    LLMX_KROWS(llmx_##t##_quarter_rows8_4, KT, 4, signed char, 4, QuarterLane<KT>)                                                 \
    LLMX_KROWS(llmx_##t##_quarter_rows8_8, KT, 8, signed char, 4, QuarterLane<KT>)                                                 \
    LLMX_KROWS(llmx_##t##_quarter_rows8_16, KT, 16, signed char, 4, QuarterLane<KT>)                                               \
    LLMX_ROUTED(llmx_##t##_quarter_routed16, short, k_rows<KT, 1, short, 2, 1, Cols, QuarterLane<KT>>)                             \
    LLMX_ROUTED(llmx_##t##_quarter_routed8, signed char, k_rows<KT, 1, signed char, 2, 1, Cols, QuarterLane<KT>>)                  \
    LLMX_GROUPED(llmx_##t##_quarter_routed16_g2, short, 2, k_rows<KT, 2, short, 2, 2, EntryCols<2>, QuarterLane<KT>>)            \
    LLMX_GROUPED(llmx_##t##_quarter_routed16_g4, short, 4, k_rows<KT, 4, short, 2, 4, EntryCols<4>, QuarterLane<KT>>)            \
    LLMX_GROUPED(llmx_##t##_quarter_routed16_g8, short, 8, k_rows<KT, 8, short, 2, 4, EntryCols<8>, QuarterLane<KT>>)            \
    LLMX_GROUPED(llmx_##t##_quarter_routed8_g2, signed char, 2, k_rows<KT, 2, signed char, 2, 2, EntryCols<2>, QuarterLane<KT>>) \
    LLMX_GROUPED(llmx_##t##_quarter_routed8_g4, signed char, 4, k_rows<KT, 4, signed char, 2, 4, EntryCols<4>, QuarterLane<KT>>) \
    LLMX_GROUPED(llmx_##t##_quarter_routed8_g8, signed char, 8, k_rows<KT, 8, signed char, 2, 4, EntryCols<8>, QuarterLane<KT>>)
LLMX_KQUARTER(q5_0, rq::Q5_0)
LLMX_KQUARTER(q5_1, rq::Q5_1)
