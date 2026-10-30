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

// Eight values of a twin block at `p` as ints: four words of two 16-bit values, or two words of four 8-bit values.
__device__ inline void eight(const short* p, int v[8]) {
    const int4 w = *(const int4*)p;
    const int words[4] = {w.x, w.y, w.z, w.w};
    for (int i = 0; i < 4; ++i) {
        v[2 * i] = (short)(words[i] & 0xffff);
        v[2 * i + 1] = (short)(words[i] >> 16);
    }
}
__device__ inline void eight(const signed char* p, int v[8]) {
    const int2 w = *(const int2*)p;
    const int words[2] = {w.x, w.y};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j) v[4 * i + j] = (signed char)((words[i] >> (8 * j)) & 0xff);
}

// The twin of `blocks` blocks of 32 floats, 32 threads a block, at L levels: d the block's scale and q the values scaled to L (docs/STATUS.md, the twin's rule), L 32767 for block-int16 and 127 for block-int8.
template <typename QT, int L>
__device__ void twin(const float* x, QT* q, float* d, unsigned long long blocks) {
    __shared__ float lds[kThreads];
    const unsigned long long b = (unsigned long long)blockIdx.x * (kThreads / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const float v = b < blocks ? x[b * 32 + lane] : 0.0f;
    lds[threadIdx.x] = fabsf(v);
    __syncthreads();
    const int base = threadIdx.x - lane;
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, lds[base + j]);
    if (b >= blocks) return;
    if (amax == 0.0f) {
        q[b * 32 + lane] = 0;
        if (lane == 0) d[b] = 0.0f;
        return;
    }
    int e = 0;
    const float m = frexpf(amax, &e);   // amax = m 2^e, m in [0.5, 1)
    (void)m;
    const int s = (amax >= 0x1p-100f && amax <= 0x1p100f) ? 0 : 1 - e;
    const float an = ldexpf(amax, s), xn = ldexpf(v, s);
    const float r = xn * ((float)L / an);
    float t = floorf(fabsf(r) + 0.5f);
    t = fminf(t, (float)L);
    q[b * 32 + lane] = (QT)(r < 0.0f ? -t : t);
    if (lane == 0) d[b] = ldexpf(an / (float)L, -s);
}

// A wave an output row o for up to C generated rows: four lanes a block, eight values each, so a wave reads 16 blocks a step in wide loads.
// Each lane's part is scaled into F32 in block order and the parts folded by one shuffle tree, the same for every column whatever C the call takes.
template <int C, typename QT>
__device__ void q8_0_rows(const unsigned char* W, const QT* q, const float* dx, float* Y, unsigned long long nin, unsigned long long nout,
                          int ncols, int add) {
    const int lanes = warpSize, lane = threadIdx.x % lanes;
    const unsigned long long blocks = nin / 32, waves = (unsigned long long)gridDim.x * (kThreads / lanes);
    // A wave takes rows o, o + waves, ...: the host launches a wave a row, and this loop form measured four times faster than an early return on rows of 2048 and 4096 (docs/STATUS.md, ROCm step 3).
    for (unsigned long long o = (unsigned long long)blockIdx.x * (kThreads / lanes) + threadIdx.x / lanes; o < nout; o += waves) {
        const unsigned char* w = W + o * blocks * 34;
        float acc[C];
        for (int c = 0; c < C; ++c) acc[c] = 0.0f;
        for (unsigned long long u = lane; u < blocks * 4; u += lanes) {
            const unsigned long long k = u / 4;
            const int part = (int)(u % 4);
            const unsigned char* block = w + k * 34;
            const float dw = __half2float(*(const __half*)block);
            // Eight weights as four 16-bit loads, the block being 2-byte aligned.
            const unsigned short* wp = (const unsigned short*)(block + 2 + part * 8);
            signed char wv[8];
            for (int i = 0; i < 4; ++i) {
                const unsigned short pair = wp[i];
                wv[2 * i] = (signed char)(pair & 0xff);
                wv[2 * i + 1] = (signed char)(pair >> 8);
            }
#pragma unroll
            for (int c = 0; c < C; ++c) {
                if (c >= ncols) break;
                const unsigned long long xb = (unsigned long long)c * blocks + k;
                int xv[8];
                eight(q + xb * 32 + part * 8, xv);
                int isum = 0;
                for (int i = 0; i < 8; ++i) isum += (int)wv[i] * xv[i];
                acc[c] = fmaf(dw * dx[xb], (float)isum, acc[c]);
            }
        }
#pragma unroll
        for (int c = 0; c < C; ++c) {
            if (c >= ncols) break;
            float v = acc[c];
            for (int m = lanes / 2; m > 0; m /= 2) v += __shfl_xor(v, m, lanes);
            if (lane == 0) {
                float* y = Y + (unsigned long long)c * nout + o;
                *y = add ? *y + v : v;
            }
        }
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
};
// Blocks of 32 staged a barrier, as the Vulkan tile stages them: four for the 8-bit twin, two for the 16-bit one, whose widened words are twice as many.
template <typename QT> struct TileStep { static constexpr int n = sizeof(QT) == 1 ? 4 : 2; };

// The prompt tile (the Vulkan tile's design, matmul_tile_q.comp): a block of 256 threads 64 outputs by 64 rows, each thread 4 adjacent outputs by 4 adjacent rows, the blocks of a step staged k-major in shared memory, word w of every output and then of every row, the weights widened to the twin's integer type.
// A thread reads its four outputs' word and its four rows' word as two 16-byte reads and feeds 16 integer sums; each output sums its blocks in order, acc = fma(dw * dx, isum, acc), so a row gives the same bits wherever the tile falls.
template <typename QT>
__device__ void q8_0_tile(const TileCall& c, const QT* q, const float* dx) {
    constexpr int STEP = TileStep<QT>::n, WORDS = 32 * (int)sizeof(QT) / 4;
    const unsigned pi = c.nproj > 2 && blockIdx.x >= c.start[2] ? 2 : c.nproj > 1 && blockIdx.x >= c.start[1] ? 1 : 0;
    const unsigned char* W = c.w[pi];
    const unsigned long long nout = c.nout[pi], nrows = c.nrows, nin = c.nin;
    __shared__ __attribute__((aligned(16))) int wq[STEP * WORDS][kTileOut];
    __shared__ __attribute__((aligned(16))) int xw[STEP * WORDS][kTileRows];
    __shared__ float wd[STEP][kTileOut], xd[STEP][kTileRows];
    const int t = threadIdx.x, tr = t / 16, tc = t % 16;   // outputs tr * 4.., rows tc * 4..
    const unsigned long long o0 = (unsigned long long)(blockIdx.x - c.start[pi]) * kTileOut, b0 = (unsigned long long)blockIdx.y * kTileRows;
    const unsigned long long blocks = nin / 32, kbeg = (unsigned long long)blockIdx.z * c.kper;
    const unsigned long long kend = kbeg + c.kper < blocks ? kbeg + c.kper : blocks;
    float acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    // Four threads a row, each a quarter of a block's 32 values (the Vulkan tile's staging): thread t / 4 of the outputs and t / 4 of the rows.
    // A stage's blocks are loaded into registers while the previous stage computes, and stored to shared memory after it.
    const int r = t / 4, seg = t % 4;
    const unsigned long long o = o0 + r, b = b0 + r;
    unsigned wv[STEP][2];          // eight quants: two words of bytes
    unsigned short wh[STEP];       // the block's f16 scale
    int xv[STEP][WORDS / 4];       // a quarter of the twin block
    float xs_[STEP];
    // The quarter's eight bytes sit at an offset of 0 or 2 in their 4-byte words, since a block is 34 bytes; two or three word loads and a shift realign them.
    const auto load = [&](unsigned long long k0) {
#pragma unroll
        for (int s = 0; s < STEP; ++s) {
            const unsigned long long k = k0 + s;
            const bool wok = k < kend && o < nout, xok = k < kend && b < nrows;
            if (wok) {
                const unsigned char* block = W + (o * blocks + k) * 34;
                const unsigned long long a = (unsigned long long)(block + 2 + seg * 8);
                const unsigned* p4 = (const unsigned*)(a & ~3ull);
                if ((a & 3) == 0) {
                    wv[s][0] = p4[0];
                    wv[s][1] = p4[1];
                } else {
                    const unsigned d0 = p4[0], d1 = p4[1], d2 = *(const unsigned short*)(p4 + 2);   // two bytes only: the block ends there
                    wv[s][0] = (d0 >> 16) | (d1 << 16);
                    wv[s][1] = (d1 >> 16) | (d2 << 16);
                }
                wh[s] = seg == 0 ? *(const unsigned short*)block : 0;
            } else {
                wv[s][0] = wv[s][1] = 0;
                wh[s] = 0;
            }
            if constexpr (sizeof(QT) == 2) {
                const int4 v = xok ? ((const int4*)(q + (b * blocks + k) * 32))[seg] : int4{0, 0, 0, 0};
                xv[s][0] = v.x, xv[s][1] = v.y, xv[s][2] = v.z, xv[s][3] = v.w;
            } else {
                const int2 v = xok ? ((const int2*)(q + (b * blocks + k) * 32))[seg] : int2{0, 0};
                xv[s][0] = v.x, xv[s][1] = v.y;
            }
            xs_[s] = seg == 0 && xok ? dx[b * blocks + k] : 0.0f;
        }
    };
    // Widened to the twin's integer type as they are stored: four words of two 16-bit values, or the two words of bytes as they are.
    const auto store = [&]() {
#pragma unroll
        for (int s = 0; s < STEP; ++s) {
            if constexpr (sizeof(QT) == 2) {
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int w = 0; w < 2; ++w) {
                        const unsigned pair = (wv[s][h] >> (16 * w)) & 0xffff;
                        wq[s * WORDS + seg * 4 + 2 * h + w][r] = (int)((unsigned)(unsigned short)(short)(signed char)(pair & 0xff) |
                                                                       ((unsigned)(unsigned short)(short)(signed char)(pair >> 8) << 16));
                    }
            } else {
                wq[s * WORDS + seg * 2][r] = (int)wv[s][0];
                wq[s * WORDS + seg * 2 + 1][r] = (int)wv[s][1];
            }
            if (seg == 0) wd[s][r] = __half2float(__ushort_as_half(wh[s]));
#pragma unroll
            for (int w = 0; w < WORDS / 4; ++w) xw[s * WORDS + seg * (WORDS / 4) + w][r] = xv[s][w];
            if (seg == 0) xd[s][r] = xs_[s];
        }
    };
    load(kbeg);
    for (unsigned long long k0 = kbeg; k0 < kend; k0 += STEP) {
        store();
        __syncthreads();
        if (k0 + STEP < kend) load(k0 + STEP);
#pragma unroll
        for (int s = 0; s < STEP; ++s) {
            if (k0 + s >= kend) break;
            int is[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) is[i][j] = 0;
#pragma unroll
            for (int w = 0; w < WORDS; ++w) {
                const int4 a = *(const int4*)&wq[s * WORDS + w][tr * 4];
                const int4 b = *(const int4*)&xw[s * WORDS + w][tc * 4];
                const int av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j) is[i][j] = word_dot(av[i], bv[j], is[i][j], QT());
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(wd[s][tr * 4 + i] * xd[s][tc * 4 + j], (float)is[i][j], acc[i][j]);
        }
        __syncthreads();
    }
    unsigned long long before = 0;
    for (unsigned p = 0; p < pi; ++p) before += c.nout[p];
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i) {
            const unsigned long long o = o0 + tr * 4 + i, b = b0 + tc * 4 + j;
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

extern "C" __global__ void __launch_bounds__(kThreads) llmx_twin16(const float* x, short* q, float* d, unsigned long long blocks) { twin<short, 32767>(x, q, d, blocks); }
extern "C" __global__ void __launch_bounds__(kThreads) llmx_twin8(const float* x, signed char* q, float* d, unsigned long long blocks) { twin<signed char, 127>(x, q, d, blocks); }

// Every kernel here runs blocks of kThreads, which __launch_bounds__ tells the compiler: its default assumes 1024 threads and caps a thread at 64 registers, which spilled the tile's prefetch to scratch.
// The row kernel's builds, the host taking the narrowest that holds a call's columns.
#define LLMX_ROWS(name, C, QT)                                                                                                          \
    extern "C" __global__ void __launch_bounds__(kThreads) name(const unsigned char* W, const QT* q, const float* dx, float* Y, unsigned long long nin,           \
                                    unsigned long long nout, int ncols, int add) {                                                      \
        q8_0_rows<C, QT>(W, q, dx, Y, nin, nout, ncols, add);                                                                           \
    }
LLMX_ROWS(llmx_q8_0_rows16_1, 1, short)
LLMX_ROWS(llmx_q8_0_rows16_2, 2, short)
LLMX_ROWS(llmx_q8_0_rows16_4, 4, short)
LLMX_ROWS(llmx_q8_0_rows16_8, 8, short)
LLMX_ROWS(llmx_q8_0_rows16_16, 16, short)
LLMX_ROWS(llmx_q8_0_rows8_1, 1, signed char)
LLMX_ROWS(llmx_q8_0_rows8_2, 2, signed char)
LLMX_ROWS(llmx_q8_0_rows8_4, 4, signed char)
LLMX_ROWS(llmx_q8_0_rows8_8, 8, signed char)
LLMX_ROWS(llmx_q8_0_rows8_16, 16, signed char)

extern "C" __global__ void __launch_bounds__(kThreads) llmx_q8_0_tile16(TileCall c, const short* q, const float* dx) { q8_0_tile<short>(c, q, dx); }
extern "C" __global__ void __launch_bounds__(kThreads) llmx_q8_0_tile8(TileCall c, const signed char* q, const float* dx) { q8_0_tile<signed char>(c, q, dx); }

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
