// The quantized products of the F16 and int8 classes (docs/PRECISION.md, rule 4 and section 5): activations as a block-int16 or block-int8 twin, Q8_0 rows decoded exactly, integer block dots and F32 sums.
// A row's result depends on that row alone: the row kernel for generated rows and the tile for prompt rows each fix their order, whatever the call holds.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace {

constexpr int kThreads = 256;

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
};
// Row o of a call: its weight row, its output for column 0 and its projection's output count.
struct Row {
    const unsigned char* w;
    float* y;
    unsigned long long nout;
};
// The kernels pass o through readfirstlane: it is one wave's, so the projection is chosen in scalar registers, which measured 3 to 4 percent faster on the 8B's rows.
__device__ inline Row row_of(const RowCall& c, unsigned long long o, unsigned long long row_bytes) {
    const unsigned p = c.nproj > 2 && o >= c.start[2] ? 2 : c.nproj > 1 && o >= c.start[1] ? 1 : 0;
    const unsigned long long local = o - (p == 2 ? c.start[2] : p == 1 ? c.start[1] : 0);
    const unsigned char* w = p == 2 ? c.w[2] : p == 1 ? c.w[1] : c.w[0];
    float* y = p == 2 ? c.y[2] : p == 1 ? c.y[1] : c.y[0];
    return {w + local * row_bytes, y + local, p == 2 ? c.nout[2] : p == 1 ? c.nout[1] : c.nout[0]};
}

// Sum s of the wave's row j and column c, stored by its writer lane unless the column is past the call or the row past the end.
// The lane picks its row from the wave's rows r, which sit in scalar registers, so no lane reads the call's arguments again.
template <int R>
__device__ inline void store_sum(const Row (&r)[R], float s, int j, int c, bool writer, unsigned long long o, unsigned long long total, int ncols, int add) {
    if (!writer || c >= ncols || o + j >= total) return;
#pragma unroll
    for (int i = 0; i < R; ++i)
        if (j == i) {
            float* y = r[i].y + (unsigned long long)c * r[i].nout;
            *y = add ? *y + s : s;
        }
}
// The R * C sums of a wave's rows o to o + R - 1 and its columns by one transposed tree, each written by its own lane; a column past the call or a row past the end is summed and not stored.
template <int R, int C>
__device__ inline void store_sums(float (&acc)[R][C], const Row (&r)[R], int lane, unsigned long long o, unsigned long long total, int ncols, int add) {
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
        store_sum<R>(r, s, k / C, k % C, lane % (warpSize / NV) == 0, o, total, ncols, add);
    } else {
#pragma unroll
        for (int p = 0; p < NV; p += NP) {
            float part[NP];
#pragma unroll
            for (int i = 0; i < NP; ++i) part[i] = v[p + i];
            int k;
            const float s = wave_sums<NP>(part, lane, k);
            store_sum<R>(r, s, (p + k) / C, (p + k) % C, lane % (warpSize / NP) == 0, o, total, ncols, add);
        }
    }
}

// A wave computes R output rows for up to C generated rows: four lanes a block, eight values each, each lane's part scaled into F32 in block order and folded by wave_sums, the bits of one shuffle tree whatever C and R.
// With R rows each twin load serves every row and a row keeps its lanes and its order, so it has the bits of the one-row build (docs/STATUS.md, ROCm F16 decode rows).
template <int C, typename QT, int R, int G>
__device__ void q8_0_rows(const RowCall& rc, const QT* q, const float* dx, unsigned long long nin, int ncols, int add) {
    const int lanes = warpSize, lane = threadIdx.x % lanes;
    const unsigned long long blocks = nin / 32, waves = (unsigned long long)gridDim.x * (kThreads / lanes);
    // A wave takes rows o, o + R * waves, ...: the host launches a wave for R rows, and this loop form measured four times faster than an early return on rows of 2048 and 4096 (docs/STATUS.md, ROCm step 3).
    for (unsigned long long o = R * ((unsigned long long)blockIdx.x * (kThreads / lanes) + threadIdx.x / lanes); o < rc.total; o += R * waves) {
        Row r[R];
#pragma unroll
        for (int j = 0; j < R; ++j) r[j] = row_of(rc, __builtin_amdgcn_readfirstlane((unsigned)(o + j < rc.total ? o + j : o)), blocks * 34);
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
                const unsigned char* block = r[j].w + k * 34;
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
                    const unsigned long long col = (unsigned long long)(g + c < ncols ? g + c : 0) * blocks;
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
        store_sums<R, C>(acc, r, lane, o, rc.total, ncols, add);
    }
}

// The int8 class's Q8_0 decode kernel in the half-block order of the Vulkan decode kernel (matmul_vec_q8_one.glsl, HALF), for rows of an even block count: a wave R output rows, two lanes a block, so a wave reads warpSize / 2 blocks a step.
// The even lane takes the block's bytes [0, 18), the scale and 16 weights, the odd lane bytes [18, 34), each as aligned words realigned in registers without a branch, and the even lane hands the scale to the odd one; a branch on the alignment waited on its loads and measured slower (docs/STATUS.md, ROCm decode products).
// Each lane scales its halves' integer dots into F32 in block order and the lanes fold by the transposed tree of wave_sums, the bits of one shuffle tree for every column whatever C and R the call takes.
template <int C, int R, int G>
__device__ void q8_0_rows_half(const RowCall& rc, const signed char* q, const float* dx, unsigned long long nin, int ncols, int add) {
    const int lanes = warpSize, lane = threadIdx.x % lanes;
    const unsigned long long blocks = nin / 32, waves = (unsigned long long)gridDim.x * (kThreads / lanes);
    const unsigned hl = lane & 1;
    // A wave takes R rows, so each twin load serves every row; a row keeps its lanes and its order, so it has the bits of the one-row build.
    for (unsigned long long o = R * ((unsigned long long)blockIdx.x * (kThreads / lanes) + threadIdx.x / lanes); o < rc.total; o += R * waves) {
        Row r[R];
        const unsigned* w[R];
#pragma unroll
        for (int j = 0; j < R; ++j) {
            r[j] = row_of(rc, __builtin_amdgcn_readfirstlane((unsigned)(o + j < rc.total ? o + j : o)), blocks * 34);
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
            const unsigned at = b * 34, start = at + 18 * hl, sh = start & 2, wi = start >> 2, s0 = sh + 2 * (1 - hl);
            // Columns in groups of G, a column past the call skipped: the first group's twin loads issue before the weights' and each group's before its products (docs/STATUS.md, ROCm wide decode rows).
            int4 xq[G];
            float xd[G];
            const auto load = [&](int g) {
#pragma unroll
                for (int c = 0; c < G; ++c) {
                    if (g + c >= ncols) continue;
                    // The column's base is the wave's and the lane's offset 32-bit, one register for every column.
                    const unsigned long long col = (unsigned long long)(g + c) * blocks;
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
        store_sums<R, C>(acc, r, lane, o, rc.total, ncols, add);
    }
}

// The one-column build of the half-block kernel: kSteps steps of a lane's loads, the weights' five words and the twin's, issued before any of their products, then each step's products in block order, so a row sums as the other builds sum it.
// The scheduling barrier keeps the loads together, which the compiler otherwise spreads to save registers; four steps hold a row of 4096 in one round of loads (docs/STATUS.md, ROCm decode rows).
constexpr int kSteps = 4;
template <int NS>
__device__ inline void half_steps(const unsigned* w, const signed char* q, const float* dx, unsigned& b, unsigned hl, float& acc) {
    const unsigned span = warpSize / 2;
    unsigned t[NS][5];
    int4 xq[NS];
    float xd[NS];
#pragma unroll
    for (int d = 0; d < NS; ++d) {
        const unsigned bd = b + d * span, start = bd * 34 + 18 * hl, sh = start & 2, wi = start >> 2;
        xq[d] = *(const int4*)(q + (unsigned long long)bd * 32 + 16 * hl);
        xd[d] = dx[bd];
        t[d][0] = w[wi];
        t[d][1] = w[wi + 1];
        t[d][2] = w[wi + 2];
        t[d][3] = w[wi + 3];
        t[d][4] = w[wi + 3 + (hl == 0 || sh ? 1 : 0)];
    }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int d = 0; d < NS; ++d) {
        const unsigned start = (b + d * span) * 34 + 18 * hl, sh = start & 2, s0 = sh + 2 * (1 - hl);
        const bool next = s0 == 4;
        const unsigned t0 = t[d][0], t1 = t[d][1], t2 = t[d][2], t3 = t[d][3], t4 = t[d][4];
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
        acc = fmaf(dw * xd[d], (float)s, acc);
    }
    b += NS * span;
}
__device__ void q8_0_half_one(const RowCall& rc, const signed char* q, const float* dx, unsigned long long nin, int add) {
    const int lanes = warpSize, lane = threadIdx.x % lanes;
    const unsigned long long blocks = nin / 32, waves = (unsigned long long)gridDim.x * (kThreads / lanes);
    const unsigned hl = lane & 1, span = lanes / 2;
    for (unsigned long long o = (unsigned long long)blockIdx.x * (kThreads / lanes) + threadIdx.x / lanes; o < rc.total; o += waves) {
        const Row r = row_of(rc, __builtin_amdgcn_readfirstlane((unsigned)o), blocks * 34);
        const unsigned* w = (const unsigned*)r.w;
        float acc = 0.0f;
        unsigned b = lane >> 1;
        while (b + (kSteps - 1) * span < blocks) half_steps<kSteps>(w, q, dx, b, hl, acc);
        while (b < blocks) half_steps<1>(w, q, dx, b, hl, acc);
        const float v = wave_sum(acc);
        if (lane == 0) *r.y = add ? *r.y + v : v;
    }
}

constexpr int kTileOut = 64, kTileRows = 64;

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
// Blocks of 32 staged a barrier, as the Vulkan tile stages them: four for the 8-bit twin, two for the 16-bit one, whose widened words are twice as many.
// A step loads while the previous one computes, but in the tall build over the 16-bit twin, which loads its step after the barrier: holding the next step's loads took it 124 VGPRs and after the barrier it takes 99 (docs/STATUS.md, ROCm tall tile staging).
// A value at a 32-bit byte offset from a uniform base, so the load takes the base from scalar registers and the offset from one vector register, with no 64-bit address arithmetic; the host bounds every offset (`rocm_backend.cpp`, the tile launch).
template <typename T> __device__ inline T at(const void* base, unsigned off) { return *(const T*)((const unsigned char*)base + off); }
// WQ quant words as one load: their words are 4-byte aligned, which a global vector load takes.
template <int WQ> using QWords = unsigned __attribute__((ext_vector_type(WQ), aligned(4)));
template <typename QT, int RO> struct TileStep { static constexpr int n = sizeof(QT) == 1 && RO == 4 ? 4 : 2; static constexpr bool ahead = RO == 4 || sizeof(QT) == 1; };

// The prompt tile (the Vulkan tile's design, matmul_tile_q.comp): a block of 256 threads 64 outputs by 64 rows, each thread 4 adjacent outputs by 4 adjacent rows, the blocks of a step staged k-major in shared memory, word w of every output and then of every row, the weights widened to the twin's integer type.
// A thread reads its four outputs' word and its four rows' word as two 16-byte reads and feeds 16 integer sums; each output sums its blocks in order, acc = fma(dw * dx, isum, acc), so a row gives the same bits wherever the tile falls.
template <typename QT, int RO>
__device__ void q8_0_tile(const TileCall& c, const QT* q, const float* dx) {
    constexpr bool AHEAD = TileStep<QT, RO>::ahead;
    constexpr int STEP = TileStep<QT, RO>::n, WORDS = 32 * (int)sizeof(QT) / 4, OUTS = RO * 16;
    constexpr int WT = kThreads / OUTS, WQ = 32 / WT / 4;   // threads a weight row and the quant words each stages
    const unsigned pi = c.nproj > 2 && blockIdx.x >= c.start[2] ? 2 : c.nproj > 1 && blockIdx.x >= c.start[1] ? 1 : 0;
    const unsigned char* W = c.w[pi];
    const unsigned long long nout = c.nout[pi], nrows = c.nrows, nin = c.nin;
    __shared__ __attribute__((aligned(16))) int wq[STEP * WORDS][OUTS];
    __shared__ __attribute__((aligned(16))) int xw[STEP * WORDS][kTileRows];
    __shared__ float wd[STEP][OUTS], xd[STEP][kTileRows];
    const int t = threadIdx.x, tr = t / 16, tc = t % 16;   // outputs tr * 4.., rows tc * 4..
    const unsigned long long o0 = (unsigned long long)(blockIdx.x - c.start[pi]) * OUTS, b0 = (unsigned long long)blockIdx.y * kTileRows;
    const unsigned long long blocks = nin / 32, kbeg = (unsigned long long)blockIdx.z * c.kper;
    const unsigned long long kend = kbeg + c.kper < blocks ? kbeg + c.kper : blocks;
    float acc[RO][4];
    for (int i = 0; i < RO; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    // Each weight row takes WT threads, each WQ words of a block's quants (two threads a row on the tall tile, four on the other), and the twin's rows four threads, each a quarter of a block (the Vulkan tile's staging).
    // A stage's blocks are loaded into registers while the previous stage computes, and stored to shared memory after it.
    const int r = t / 4, seg = t % 4, wr = t / WT, ws = t % WT;
    const unsigned long long o = o0 + wr, b = b0 + r;
    unsigned wv[STEP][WQ + 1];   // the thread's quant words and one more: the word after where they straddle, the word before where not
    bool wok[STEP], al[STEP], xok[STEP];
    int xv[STEP][WORDS / 4];       // a quarter of the twin block
    float xs_[STEP];
    // The tile's first output and first row as uniform bases, so each thread's offsets from them are 32-bit bytes (`at`) and its loads take the base from scalar registers.
    const unsigned char* Wt = W + o0 * blocks * 34;
    const unsigned mis = (unsigned)((unsigned long long)Wt & 3);
    const unsigned* W4 = (const unsigned*)(Wt - mis);   // the words around the tile's weights, as a pointer into the same buffer so its loads stay global ones
    const QT* qt = q + b0 * c.xrow * 32;
    const float* dxt = dx + b0 * c.xrow;
    // A thread's quants sit at an offset of 0 or 2 in their 4-byte words, since a block is 34 bytes: one vector load of WQ words and one word cover them, the word after where they straddle and the word before where not, which holds the scale in its high half; straddling, the first word holds it in its low half.
    // Every load of a stage goes out before any is waited for: a load out of range reads block 0 and its zero is chosen at the store, so no load sits under a branch the compiler would wait inside.
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
            const unsigned xb = xok[s] ? (unsigned)r * c.xrow + (unsigned)k * c.xblock : 0u;
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
            for (int w = 0; w < WORDS / 4; ++w) xw[s * WORDS + seg * (WORDS / 4) + w][r] = xok[s] ? xv[s][w] : 0;
            if (seg == 0) xd[s][r] = xok[s] ? xs_[s] : 0.0f;
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
            if (k0 + s >= kend) break;
            int is[RO][4];
#pragma unroll
            for (int i = 0; i < RO; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) is[i][j] = 0;
#pragma unroll
            for (int w = 0; w < WORDS; ++w) {
                int av[RO];
#pragma unroll
                for (int h = 0; h < RO / 4; ++h) {
                    const int4 a = *(const int4*)&wq[s * WORDS + w][tr * RO + 4 * h];
                    av[4 * h] = a.x, av[4 * h + 1] = a.y, av[4 * h + 2] = a.z, av[4 * h + 3] = a.w;
                }
                const int4 b = *(const int4*)&xw[s * WORDS + w][tc * 4];
                const int bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
                for (int i = 0; i < RO; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) is[i][j] = word_dot(av[i], bv[j], is[i][j], QT());
            }
#pragma unroll
            for (int i = 0; i < RO; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(wd[s][tr * RO + i] * xd[s][tc * 4 + j], (float)is[i][j], acc[i][j]);
        }
        __syncthreads();
    }
    unsigned long long before = 0;
    for (unsigned p = 0; p < pi; ++p) before += c.nout[p];
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < RO; ++i) {
            const unsigned long long o = o0 + tr * RO + i, b = b0 + tc * 4 + j;
            if (o >= nout || b >= nrows) continue;
            if (c.parts > 1) {
                c.partials[c.parts * nrows * before + blockIdx.z * nrows * nout + b * nout + o] = acc[i][j];
            } else {
                float* y = c.y[pi] + b * nout + o;
                *y = c.add ? *y + acc[i][j] : acc[i][j];
            }
        }
}

} // namespace


// Every kernel here runs blocks of kThreads, which __launch_bounds__ tells the compiler: its default assumes 1024 threads and caps a thread at 64 registers, which spilled the tile's prefetch to scratch.
// The row kernel's builds, the host taking the narrowest that holds a call's columns.
// R is the output rows a wave takes and G the columns whose loads issue together; a build named _r4 takes four rows where its namesake takes two, the host choosing between them by the call's rows (rocm_backend.cpp, row_build).
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
    q8_0_half_one(c, q, dx, nin, add);
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
