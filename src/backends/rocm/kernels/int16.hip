// The F16 class's quantized products (docs/PRECISION.md, rule 4): activations as a block-int16 twin, Q8_0 rows decoded exactly, integer block dots and F32 sums.
// A row's result depends on that row alone: the row kernel for generated rows and the tile for prompt rows each fix their order, whatever the call holds.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace {

constexpr int kThreads = 256;

#ifdef LLMX_DOT16
typedef short short2v __attribute__((ext_vector_type(2)));
// Two weights and two twin values as the 16-bit dot reads them, a weight widened into each half.
__device__ inline int dot2(signed char w0, signed char w1, int q, int s) {
    return __builtin_amdgcn_sdot2(__builtin_bit_cast(short2v, (int)((unsigned)(unsigned short)w0 | ((unsigned)(unsigned short)w1 << 16))),
                                  __builtin_bit_cast(short2v, q), s, false);
}
#endif

// The integer dot of a Q8_0 block's 32 values with a twin block's; at most 32 * 127 * 32767 in magnitude, inside an int, so either form gives the same sum.
__device__ inline int block_dot(const signed char* w, const short* q) {
    int s = 0;
#ifdef LLMX_DOT16
    for (int j = 0; j < 32; j += 2) s = dot2(w[j], w[j + 1], (int)((unsigned)(unsigned short)q[j] | ((unsigned)(unsigned short)q[j + 1] << 16)), s);
#else
    for (int j = 0; j < 32; ++j) s += (int)w[j] * (int)q[j];
#endif
    return s;
}

} // namespace

// The twin of `blocks` blocks of 32 floats, 32 threads a block: d the block's scale and q the values scaled to 32767 (docs/STATUS.md, the twin's rule).
extern "C" __global__ void llmx_twin16(const float* x, short* q, float* d, unsigned long long blocks) {
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
    const float r = xn * (32767.0f / an);
    float t = floorf(fabsf(r) + 0.5f);
    t = fminf(t, 32767.0f);
    q[b * 32 + lane] = (short)(r < 0.0f ? -t : t);
    if (lane == 0) d[b] = ldexpf(an / 32767.0f, -s);
}

// A wave an output row o for up to C generated rows: four lanes a block, eight values each, so a wave reads 16 blocks a step in wide loads.
// Each lane's part is scaled into F32 in block order and the parts folded by one shuffle tree, the same for every column whatever C the call takes.
template <int C>
__device__ void q8_0_rows16(const unsigned char* W, const short* q, const float* dx, float* Y, unsigned long long nin, unsigned long long nout,
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
            const int4 qv = *(const int4*)(q + xb * 32 + part * 8);
            const int words[4] = {qv.x, qv.y, qv.z, qv.w};
            int isum = 0;
            for (int i = 0; i < 4; ++i) {
                isum += (int)wv[2 * i] * (int)(short)(words[i] & 0xffff);
                isum += (int)wv[2 * i + 1] * (int)(short)(words[i] >> 16);
            }
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

// The row kernel's builds, the host taking the narrowest that holds a call's columns.
extern "C" __global__ void llmx_q8_0_rows16_1(const unsigned char* W, const short* q, const float* dx, float* Y, unsigned long long nin,
                                         unsigned long long nout, int ncols, int add) {
    q8_0_rows16<1>(W, q, dx, Y, nin, nout, ncols, add);
}
extern "C" __global__ void llmx_q8_0_rows16_2(const unsigned char* W, const short* q, const float* dx, float* Y, unsigned long long nin,
                                         unsigned long long nout, int ncols, int add) {
    q8_0_rows16<2>(W, q, dx, Y, nin, nout, ncols, add);
}
extern "C" __global__ void llmx_q8_0_rows16_4(const unsigned char* W, const short* q, const float* dx, float* Y, unsigned long long nin,
                                         unsigned long long nout, int ncols, int add) {
    q8_0_rows16<4>(W, q, dx, Y, nin, nout, ncols, add);
}
extern "C" __global__ void llmx_q8_0_rows16_8(const unsigned char* W, const short* q, const float* dx, float* Y, unsigned long long nin,
                                         unsigned long long nout, int ncols, int add) {
    q8_0_rows16<8>(W, q, dx, Y, nin, nout, ncols, add);
}
extern "C" __global__ void llmx_q8_0_rows16_16(const unsigned char* W, const short* q, const float* dx, float* Y, unsigned long long nin,
                                         unsigned long long nout, int ncols, int add) {
    q8_0_rows16<16>(W, q, dx, Y, nin, nout, ncols, add);
}

// The prompt tile: a block 16 outputs by 16 rows, one thread an (o, b), the weight and twin blocks staged in shared memory; each output sums its blocks in order, so a row gives the same bits wherever the tile falls.
extern "C" __global__ void llmx_q8_0_tile16(const unsigned char* W, const short* q, const float* dx, float* Y, unsigned long long nin,
                                            unsigned long long nout, unsigned long long nrows, int add) {
    __shared__ signed char ws[16][32];
    __shared__ float wd[16];
    __shared__ short xs[16][32];
    __shared__ float xd[16];
    const int to = threadIdx.x % 16, tb = threadIdx.x / 16;
    const unsigned long long o = (unsigned long long)blockIdx.x * 16 + to, b = (unsigned long long)blockIdx.y * 16 + tb;
    const unsigned long long blocks = nin / 32;
    float acc = 0.0f;
    for (unsigned long long k = 0; k < blocks; ++k) {
        // 256 threads stage 16 weight blocks and 16 twin blocks, two values each.
        const int r = threadIdx.x / 16, j = (threadIdx.x % 16) * 2;
        const unsigned long long ow = (unsigned long long)blockIdx.x * 16 + r, bx = (unsigned long long)blockIdx.y * 16 + r;
        if (ow < nout) {
            const unsigned char* block = W + (ow * blocks + k) * 34;
            ws[r][j] = (signed char)block[2 + j];
            ws[r][j + 1] = (signed char)block[3 + j];
            if (j == 0) wd[r] = __half2float(*(const __half*)block);
        }
        if (bx < nrows) {
            xs[r][j] = q[(bx * blocks + k) * 32 + j];
            xs[r][j + 1] = q[(bx * blocks + k) * 32 + j + 1];
            if (j == 0) xd[r] = dx[bx * blocks + k];
        }
        __syncthreads();
        if (o < nout && b < nrows) acc = fmaf(wd[to] * xd[tb], (float)block_dot(ws[to], xs[tb]), acc);
        __syncthreads();
    }
    if (o < nout && b < nrows) {
        float* y = Y + b * nout + o;
        *y = add ? *y + acc : acc;
    }
}
