// The dense F32 kernels of the ROCm backend (docs/ROCM.md, Delivery checkpoints, step 2) and the activation twins the quantized products read: every sum is F32 in a fixed order that depends on the row alone, never on how many rows a call holds.
// A block reduction folds lane partials in one tree whatever the wave width, so a row gives the same bits in any call.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace {

constexpr int kThreads = 256;

// The sum of each thread's value over the block, the same tree for every block and wave width: levels across waves in shared memory, the rest by shuffles in wave 0, each adding the pair the shared-memory level would.
__device__ float block_sum(float v, float* lds) {
    lds[threadIdx.x] = v;
    __syncthreads();
    int s = kThreads / 2;
    for (; s >= warpSize; s >>= 1) {
        if ((int)threadIdx.x < s) lds[threadIdx.x] += lds[threadIdx.x + s];
        __syncthreads();
    }
    if ((int)threadIdx.x < warpSize) {
        float t = lds[threadIdx.x];
        for (; s > 0; s >>= 1) t += __shfl_down(t, s);
        if (threadIdx.x == 0) lds[0] = t;
    }
    __syncthreads();
    const float r = lds[0];
    __syncthreads();
    return r;
}

// One row's RMS norm, the reciprocal root applied as the CPU applies it, x * (r * w).
__device__ void rms_norm_row(float* dst, const float* src, const float* w, int n, float eps, float* lds) {
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += kThreads) s = fmaf(src[i], src[i], s);
    const float r = 1.0f / sqrtf(block_sum(s, lds) / (float)n + eps);
    for (int i = threadIdx.x; i < n; i += kThreads) dst[i] = src[i] * (r * w[i]);
}

// The largest magnitude of the 32 lanes of a twin block, a value a lane.
__device__ inline float block_amax(float v) {
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor(a, o, 32));
    return a;
}

// Value v of a twin block whose 32 lanes give amax, at L levels: d the block's scale and q the value scaled to L (docs/STATUS.md, the twin's rule), L 32767 for block-int16 and 127 for block-int8; lane 0 writes d.
template <typename QT, int L>
__device__ inline void twin_value(float v, float amax, QT* q, float* d, int lane) {
    if (amax == 0.0f) {
        *q = 0;
        if (lane == 0) *d = 0.0f;
        return;
    }
    int e = 0;
    (void)frexpf(amax, &e);   // amax = m 2^e, m in [0.5, 1)
    const int s = (amax >= 0x1p-100f && amax <= 0x1p100f) ? 0 : 1 - e;
    const float an = ldexpf(amax, s), xn = ldexpf(v, s);
    const float r = xn * ((float)L / an);
    float t = floorf(fabsf(r) + 0.5f);
    t = fminf(t, (float)L);
    *q = (QT)(r < 0.0f ? -t : t);
    if (lane == 0) *d = ldexpf(an / (float)L, -s);
}

// Value v at element i of an output whose twin a later product reads, at the class `level` names (1 block-int16, 2 block-int8, 0 none); the 32 lanes of i's block call it together.
__device__ inline void twin_at(float v, unsigned long long i, void* q, float* d, int level) {
    if (!level) return;
    const float amax = block_amax(v);
    if (level == 2) twin_value<signed char, 127>(v, amax, (signed char*)q + i, d + i / 32, (int)(i % 32));
    else twin_value<short, 32767>(v, amax, (short*)q + i, d + i / 32, (int)(i % 32));
}

// The twin of `blocks` blocks of 32 floats, 32 threads a block.
template <typename QT, int L>
__device__ void twin(const float* x, QT* q, float* d, unsigned long long blocks) {
    const unsigned long long b = (unsigned long long)blockIdx.x * (kThreads / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const float v = b < blocks ? x[b * 32 + lane] : 0.0f;
    const float amax = block_amax(v);
    if (b < blocks) twin_value<QT, L>(v, amax, q + b * 32 + lane, d + b, lane);
}

constexpr int kNormThreads = 1024;   // the held norm's block: four times kThreads, so each thread scales a quarter of the values

// A block a row of n <= K * kThreads values: dst = RMSNorm(src) against w, with its twin at L levels beside it when L is not 0, the rows then dense and n a multiple of 32.
// Each of the block's kNormThreads threads loads its values and their weights at once, element t + kNormThreads j; the sum is rms_norm_row's, thread t < kThreads chaining elements t + kThreads k and the chains folded by block_sum's tree, so both give the same bits.
template <int K, typename QT, int L>
__device__ void rms_norm_held(float* __restrict__ dst, const float* __restrict__ src, const float* __restrict__ w, int n, unsigned long long stride, float eps,
                              QT* __restrict__ q, float* __restrict__ d) {
    constexpr int J = K * kThreads / kNormThreads;
    __shared__ float xs[K * kThreads];
    __shared__ float lds[kThreads];
    const int t = threadIdx.x;
    const unsigned long long r = blockIdx.x;
    const float* x = src + r * stride;
    float v[J], wv[J];
#pragma unroll
    for (int j = 0; j < J; ++j) {
        const int i = t + kNormThreads * j;
        v[j] = i < n ? x[i] : 0.0f;
        wv[j] = i < n ? w[i] : 0.0f;
        xs[i] = v[j];
    }
    __syncthreads();
    if (t < kThreads) {
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < K; ++k)
            if (t + kThreads * k < n) s = fmaf(xs[t + kThreads * k], xs[t + kThreads * k], s);
        lds[t] = s;
    }
    __syncthreads();
    int h = kThreads / 2;
    for (; h >= warpSize; h >>= 1) {
        if (t < h) lds[t] += lds[t + h];
        __syncthreads();
    }
    if (t < warpSize) {
        float u = lds[t];
        for (; h > 0; h >>= 1) u += __shfl_down(u, h);
        if (t == 0) lds[0] = u;
    }
    __syncthreads();
    const float rr = 1.0f / sqrtf(lds[0] / (float)n + eps);
    float* y = dst + r * stride;
#pragma unroll
    for (int j = 0; j < J; ++j) {
        const int i = t + kNormThreads * j;
        if (i >= n) break;
        const float o = v[j] * (rr * wv[j]);
        y[i] = o;
        if (L) {
            const float amax = block_amax(o);
            const unsigned long long g = r * (unsigned long long)n + i;
            twin_value<QT, L>(o, amax, q + g, d + g / 32, i % 32);
        }
    }
}

} // namespace

extern "C" __global__ void __launch_bounds__(kThreads) llmx_twin16(const float* x, short* q, float* d, unsigned long long blocks) { twin<short, 32767>(x, q, d, blocks); }
extern "C" __global__ void __launch_bounds__(kThreads) llmx_twin8(const float* x, signed char* q, float* d, unsigned long long blocks) { twin<signed char, 127>(x, q, d, blocks); }

// dst[i] += src[i].
extern "C" __global__ void llmx_add(float* dst, const float* src, unsigned long long n) {
    const unsigned long long i = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// dst[i] = silu(gate[i]) * up[i], with its twin at L levels beside it when L is not 0, n then a multiple of 32.
template <typename QT, int L>
__device__ void silu_mul(float* dst, const float* gate, const float* up, unsigned long long n, QT* q, float* d) {
    const unsigned long long i = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    float o = 0.0f;
    if (i < n) {
        const float g = gate[i];
        o = g / (1.0f + expf(-g)) * up[i];
        dst[i] = o;
    }
    if (L) {
        const float amax = block_amax(o);
        if (i < n) twin_value<QT, L>(o, amax, q + i, d + i / 32, (int)(i % 32));
    }
}
extern "C" __global__ void llmx_silu_mul(float* dst, const float* gate, const float* up, unsigned long long n, short* q, float* d) {
    silu_mul<short, 0>(dst, gate, up, n, q, d);
}
extern "C" __global__ void llmx_silu_mul_twin16(float* dst, const float* gate, const float* up, unsigned long long n, short* q, float* d) {
    silu_mul<short, 32767>(dst, gate, up, n, q, d);
}
extern "C" __global__ void llmx_silu_mul_twin8(float* dst, const float* gate, const float* up, unsigned long long n, signed char* q, float* d) {
    silu_mul<signed char, 127>(dst, gate, up, n, q, d);
}

// A block a row: dst row r = RMSNorm(src row r) against w, for any n.
extern "C" __global__ void llmx_rms_norm_rows(float* dst, const float* src, const float* w, int n, unsigned long long stride, float eps) {
    __shared__ float lds[kThreads];
    const unsigned long long r = blockIdx.x;
    rms_norm_row(dst + r * stride, src + r * stride, w, n, eps, lds);
}

// The held builds for rows of up to 4096 and 8192 values, plain and with the twin of either class.
#define LLMX_NORM(name, K, QT, L)                                                                                                       \
    extern "C" __global__ void __launch_bounds__(kNormThreads) name(float* dst, const float* src, const float* w, int n, unsigned long long stride, \
                                                                float eps, QT* q, float* d) {                                         \
        rms_norm_held<K, QT, L>(dst, src, w, n, stride, eps, q, d);                                                                   \
    }
LLMX_NORM(llmx_rms_norm_16, 16, short, 0)
LLMX_NORM(llmx_rms_norm_16_twin16, 16, short, 32767)
LLMX_NORM(llmx_rms_norm_16_twin8, 16, signed char, 127)
LLMX_NORM(llmx_rms_norm_32, 32, short, 0)
LLMX_NORM(llmx_rms_norm_32_twin16, 32, short, 32767)
LLMX_NORM(llmx_rms_norm_32_twin8, 32, signed char, 127)

// One head of up to kThreads values held a value a thread: its RMS norm when w is given, then pairs (i, i + half) rotated by the angles c and s, written to head and to out, the value of thread t at out[t].
// The sum and every product are those of the wider form in llmx_norm_rope, so both give the same bits.
__device__ void norm_rope_held(float* head, const float* from, int head_dim, int half, const float* w, float eps, const float* c, const float* s,
                               float* lds, float* hs, float* out) {
    const int t = threadIdx.x;
    const float v = t < head_dim ? from[t] : 0.0f;
    const float wt = w && t < head_dim ? w[t] : 0.0f;
    const float ct = t < half ? c[t] : 0.0f, st = t < half ? s[t] : 0.0f;
    float o = v;
    if (w) {
        const float r2 = 1.0f / sqrtf(block_sum(t < head_dim ? fmaf(v, v, 0.0f) : 0.0f, lds) / (float)head_dim + eps);
        o = v * (r2 * wt);
    }
    hs[t] = o;
    __syncthreads();
    if (t < half) {
        const float xb = hs[t + half];
        out[t] = fmaf(-xb, st, o * ct);
        out[t + half] = fmaf(o, st, xb * ct);
    } else if (t >= 2 * half && t < head_dim) {
        out[t] = o;
    }
    __syncthreads();
    if (t < head_dim) head[t] = out[t];
}

// A block a (row, head): the head's RMS norm when w is given, then pairs (i, i + half) rotated at the row's position.
// A head of up to kThreads values is held a value a thread (norm_rope_held), else read through global memory with the same sums and products.
extern "C" __global__ void llmx_norm_rope(float* dst, const float* src, unsigned long long dst_stride, unsigned long long src_stride,
                                          unsigned long long src_head_stride, int head_dim, int half, const float* w, float eps,
                                          const float* cos, const float* sin, const unsigned* pos) {
    __shared__ float lds[kThreads];
    __shared__ float hs[kThreads];
    __shared__ float os[kThreads];
    const unsigned long long r = blockIdx.x, h = blockIdx.y;
    float* head = dst + r * dst_stride + h * head_dim;
    const float* from = src + r * src_stride + h * src_head_stride;
    const float* c = cos + (unsigned long long)pos[r] * half;
    const float* s = sin + (unsigned long long)pos[r] * half;
    const int t = threadIdx.x;
    if (head_dim <= kThreads) return norm_rope_held(head, from, head_dim, half, w, eps, c, s, lds, hs, os);
    if (w) rms_norm_row(head, from, w, head_dim, eps, lds);
    else if (head != from)
        for (int i = t; i < head_dim; i += kThreads) head[i] = from[i];
    __syncthreads();
    for (int i = t; i < half; i += kThreads) {
        const float xa = head[i], xb = head[i + half];
        head[i] = fmaf(-xb, s[i], xa * c[i]);
        head[i + half] = fmaf(xa, s[i], xb * c[i]);
    }
}

// One value of a K or V row into its KV block, in the side's type (f16 when half).
__device__ inline void kv_store(void* cache, unsigned long long at, float x, int half) {
    if (half) ((__half*)cache)[at] = __float2half_rn(x);
    else ((float*)cache)[at] = x;
}

// The layer's q and k norm-rope and its KV write in one launch, heads of at most kThreads values (Backend::norm_rope_kv): a block a (row, head), the first n_head heads q's, in place.
// The rest are k's heads, normed and rotated in place and then written with v's head into the row's KV block, as llmx_norm_rope and llmx_kv_write compute them; k's and v's rows are dense, n_head_kv * dim apart.
extern "C" __global__ void llmx_norm_rope_kv(float* q, unsigned long long q_stride, int n_head, const float* q_w, float* k, const float* v, int n_head_kv,
                                             const float* k_w, int half, float eps, const float* cos, const float* sin, const unsigned* pos,
                                             void* kc, void* vc, const int* blocks, const int* slots, int bt, int k_half, int v_half) {
    __shared__ float lds[kThreads];
    __shared__ float hs[kThreads];
    __shared__ float os[kThreads];
    const int dim = 2 * half;
    const unsigned long long r = blockIdx.x;
    const int h = blockIdx.y, t = threadIdx.x;
    const float* c = cos + (unsigned long long)pos[r] * half;
    const float* s = sin + (unsigned long long)pos[r] * half;
    if (h < n_head) {
        float* head = q + r * q_stride + (unsigned long long)h * dim;
        return norm_rope_held(head, head, dim, half, q_w, eps, c, s, lds, hs, os);
    }
    const int kh = h - n_head;
    const unsigned long long in = (r * n_head_kv + kh) * dim;
    norm_rope_held(k + in, k + in, dim, half, k_w, eps, c, s, lds, hs, os);
    if (t >= dim) return;
    const unsigned long long out = (((unsigned long long)blocks[r] * n_head_kv + kh) * bt + slots[r]) * dim + t;
    kv_store(kc, out, os[t], k_half);
    kv_store(vc, out, v[in + t], v_half);
}

// Rows of an F32 table: dst row i = table row ids[i], `nin` floats each.
extern "C" __global__ void llmx_gather(float* dst, const float* table, const unsigned* ids, unsigned long long width) {
    const unsigned long long i = blockIdx.y, j = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    if (j < width) dst[i * width + j] = table[(unsigned long long)ids[i] * width + j];
}

// A block an output (o, b): Y[b * nout + o] = dot(W row o, X row b), added to Y with `add`; the dot is each lane's strided sum folded by block_sum.
extern "C" __global__ void llmx_matmul_f32(const float* W, const float* X, float* Y, unsigned long long nin, unsigned long long nout, int add) {
    __shared__ float lds[kThreads];
    const unsigned long long o = blockIdx.x, b = blockIdx.y;
    const float* w = W + o * nin;
    const float* x = X + b * nin;
    float s = 0.0f;
    for (unsigned long long i = threadIdx.x; i < nin; i += kThreads) s = fmaf(w[i], x[i], s);
    const float dot = block_sum(s, lds);
    if (threadIdx.x == 0) {
        float* y = Y + b * nout + o;
        *y = add ? *y + dot : dot;
    }
}

// A block a (row, head) of K and V rows into their KV blocks: row r lands in block blocks[r] at token slot slots[r], [kv_head][token][head_dim] in the side's type (1 for f16).
extern "C" __global__ void llmx_kv_write(void* kc, void* vc, const float* k, const float* v, const int* blocks, const int* slots,
                                         int heads, int dim, int bt, int k_half, int v_half) {
    const int r = blockIdx.x, h = blockIdx.y;
    const unsigned long long in = ((unsigned long long)r * heads + h) * dim;
    const unsigned long long out = (((unsigned long long)blocks[r] * heads + h) * bt + slots[r]) * dim;
    for (int d = threadIdx.x; d < dim; d += kThreads) {
        kv_store(kc, out + d, k[in + d], k_half);
        kv_store(vc, out + d, v[in + d], v_half);
    }
}

// Q8_0 rows decoded exactly, a block of 32 int8 values after its f16 scale: the dot of one block's values with x, then times the scale, summed over blocks.
__device__ inline float q8_0_block_dot(const unsigned char* block, const float* x) {
    const float d = __half2float(*(const __half*)block);
    const signed char* q = (const signed char*)(block + 2);
    float s = 0.0f;
    for (int j = 0; j < 32; ++j) s = fmaf((float)q[j], x[j], s);
    return d * s;
}

// A block an output (o, b), as llmx_matmul_f32 over Q8_0 rows: each lane sums its strided blocks, folded by block_sum.
extern "C" __global__ void llmx_matmul_q8_0(const unsigned char* W, const float* X, float* Y, unsigned long long nin, unsigned long long nout, int add) {
    __shared__ float lds[kThreads];
    const unsigned long long o = blockIdx.x, b = blockIdx.y, blocks = nin / 32;
    const unsigned char* w = W + o * blocks * 34;
    const float* x = X + b * nin;
    float s = 0.0f;
    for (unsigned long long k = threadIdx.x; k < blocks; k += kThreads) s += q8_0_block_dot(w + k * 34, x + k * 32);
    const float dot = block_sum(s, lds);
    if (threadIdx.x == 0) {
        float* y = Y + b * nout + o;
        *y = add ? *y + dot : dot;
    }
}

// Rows of a Q8_0 table decoded: dst row i = table row ids[i], `width` floats each.
extern "C" __global__ void llmx_gather_q8_0(float* dst, const unsigned char* table, const unsigned* ids, unsigned long long width) {
    const unsigned long long i = blockIdx.y, j = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    if (j >= width) return;
    const unsigned char* block = table + (unsigned long long)ids[i] * (width / 32) * 34 + (j / 32) * 34;
    dst[i * width + j] = __half2float(*(const __half*)block) * (float)((const signed char*)(block + 2))[j % 32];
}

// A row's attention chunks: as many as its history needs at 256 positions each, at most 16, each a whole number of 64-position tiles; a function of the history alone, so a row's chunks are the same in any call.
__device__ inline void attention_chunks(int end, int& count, int& length) {
    count = min(16, (end + 255) / 256);
    length = ((end + count - 1) / count + 63) / 64 * 64;
}

// A block a (row, head, chunk) of the split attention: 64 positions a tile, four threads a position's dot and one thread a dimension's accumulator, an online softmax over the chunk's tiles in order.
// The chunk's maximum, sum and accumulators go to `part` for llmx_attention_merge.
extern "C" __global__ void llmx_attention_chunk(const float* Q, float* part, const void* kc, const void* vc, const int* table, const int* firsts,
                                                const int* ends, int n_head, int ratio, int heads_kv, int dim, int bt, int k_half, int v_half,
                                                float scale) {
    __shared__ float sc[64];
    __shared__ float red[64];
    const int r = blockIdx.x, h = blockIdx.y, c = blockIdx.z, t = threadIdx.x;
    int count, length;
    attention_chunks(ends[r], count, length);
    if (c >= count) return;
    const int first = c * length, last = min(ends[r], first + length);
    const float* q = Q + ((unsigned long long)r * n_head + h) * dim;
    const int kvh = h / ratio;
    const int* blocks = table + firsts[r];
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (int base = first; base < last; base += 64) {
        const int p = t / 4, quarter = t % 4, pos = base + p;
        float s = 0.0f;
        if (pos < last) {
            const unsigned long long at = (((unsigned long long)blocks[pos / bt] * heads_kv + kvh) * bt + pos % bt) * dim;
            for (int d = quarter; d < dim; d += 4) {
                const float kd = k_half ? __half2float(((const __half*)kc)[at + d]) : ((const float*)kc)[at + d];
                s = fmaf(q[d], kd, s);
            }
        }
        s += __shfl_xor(s, 1, 4);
        s += __shfl_xor(s, 2, 4);
        if (quarter == 0) sc[p] = pos < last ? s * scale : -INFINITY;
        __syncthreads();
        // The tile's maximum and then its sum, each one tree over the 64 scores, which every thread reads.
        if (t < 64) red[t] = sc[t];
        __syncthreads();
        for (int w = 32; w > 0; w /= 2) {
            if (t < w) red[t] = fmaxf(red[t], red[t + w]);
            __syncthreads();
        }
        const float mt = fmaxf(m, red[0]);
        __syncthreads();
        if (t < 64) {
            sc[t] = expf(sc[t] - mt);
            red[t] = sc[t];
        }
        __syncthreads();
        for (int w = 32; w > 0; w /= 2) {
            if (t < w) red[t] += red[t + w];
            __syncthreads();
        }
        const float a = expf(m - mt);
        l = l * a + red[0];
        m = mt;
        __syncthreads();
        if (t < dim) {
            float v = acc * a;
            for (int j = 0; j < 64 && base + j < last; ++j) {
                const int pj = base + j;
                const unsigned long long at = (((unsigned long long)blocks[pj / bt] * heads_kv + kvh) * bt + pj % bt) * dim + t;
                const float vd = v_half ? __half2float(((const __half*)vc)[at]) : ((const float*)vc)[at];
                v = fmaf(sc[j], vd, v);
            }
            acc = v;
        }
        __syncthreads();
    }
    float* out = part + (((unsigned long long)r * n_head + h) * 16 + c) * (dim + 2);
    if (t < dim) out[2 + t] = acc;
    if (t == 0) {
        out[0] = m;
        out[1] = l;
    }
}

// Eight values of a K or V row from `i`, as floats: one 16-byte load of halves, or two of floats.
__device__ inline void load8(const void* base, unsigned long long i, bool half, float v[8]) {
    if (half) {
        const uint4 w = *(const uint4*)((const __half*)base + i);
        const unsigned words[4] = {w.x, w.y, w.z, w.w};
        for (int k = 0; k < 4; ++k) {
            v[2 * k] = __half2float(__ushort_as_half((unsigned short)(words[k] & 0xffff)));
            v[2 * k + 1] = __half2float(__ushort_as_half((unsigned short)(words[k] >> 16)));
        }
    } else {
        const float4 a = *(const float4*)((const float*)base + i), b = *(const float4*)((const float*)base + i + 4);
        v[0] = a.x, v[1] = a.y, v[2] = a.z, v[3] = a.w, v[4] = b.x, v[5] = b.y, v[6] = b.z, v[7] = b.w;
    }
}

// Causal attention of a prompt's rows (the Vulkan tile's design, attention_tile.comp): a block a tile of 32 query rows of one view and one head, eight lanes a row each holding 16 dimensions of the scaled query, K and V through shared memory 16 tokens at a time.
// A row's scores are partial dots over its lanes folded by a butterfly, its softmax runs online over the tiles from token 0 in order, so a row's result depends on its own history alone, whatever rows share its tile; heads 128 wide.
// `views` holds, a view each, its first row, rows, history, first block in `table` and first tile; `tiles` is their total.
extern "C" __global__ void llmx_attention_tile(const float* Q, float* O, const void* kc, const void* vc, const int* table, const int* views, int nviews,
                                               int n_head, int heads_kv, int bt, int k_half, int v_half, float scale) {
    constexpr int DIM = 128, TQ = 32, TK = 16, OWN = TK / 8, RUNS = DIM / 32;
    __shared__ float4 ks[TK * DIM / 4];
    __shared__ float4 vs[TK * DIM / 4];
    __shared__ float ps[TQ * TK];
    const int wg = blockIdx.x;
    int v = 0;
    for (int i = 1; i < nviews; ++i)
        if (wg >= views[5 * i + 4] * n_head) v = i;
    const int row0 = views[5 * v], nq = views[5 * v + 1], hist = views[5 * v + 2];
    const int* blocks = table + views[5 * v + 3];
    const int local = wg - views[5 * v + 4] * n_head, tile = local / n_head, h = local % n_head;
    const int kvh = h / (n_head / heads_kv);
    const int lid = threadIdx.x, r = lid >> 3, lane = lid & 7;
    const int b = tile * TQ + r;
    const bool live = b < nq;
    const unsigned long long qbase = (unsigned long long)(row0 + (b < nq ? b : nq - 1)) * n_head * DIM + (unsigned long long)h * DIM;
    float4 qv[RUNS], acc[RUNS];
    for (int k = 0; k < RUNS; ++k) {
        const int d = 32 * k + 4 * lane;
        qv[k] = make_float4(Q[qbase + d] * scale, Q[qbase + d + 1] * scale, Q[qbase + d + 2] * scale, Q[qbase + d + 3] * scale);
        acc[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    float m = -INFINITY, l = 0.0f;
    const int tile_end = min(hist + tile * TQ + TQ, hist + nq), my_end = hist + b + 1;
    for (int t0 = 0; t0 < tile_end; t0 += TK) {
        // Eight values a load, zero past the tile's last token so a masked score meets a finite value.
        for (int i = lid; i < TK * DIM / 8; i += kThreads) {
            const int j = i / (DIM / 8), d = (i % (DIM / 8)) * 8, t = t0 + j;
            float kv8[8] = {0, 0, 0, 0, 0, 0, 0, 0}, vv8[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            if (t < tile_end) {
                const unsigned long long at = (((unsigned long long)blocks[t / bt] * heads_kv + kvh) * bt + t % bt) * DIM + d;
                load8(kc, at, k_half, kv8);
                load8(vc, at, v_half, vv8);
            }
            const int s4 = (j * DIM + d) / 4;
            ks[s4] = make_float4(kv8[0], kv8[1], kv8[2], kv8[3]);
            ks[s4 + 1] = make_float4(kv8[4], kv8[5], kv8[6], kv8[7]);
            vs[s4] = make_float4(vv8[0], vv8[1], vv8[2], vv8[3]);
            vs[s4 + 1] = make_float4(vv8[4], vv8[5], vv8[6], vv8[7]);
        }
        __syncthreads();
        // Partial dots eight tokens at a time over the lane's dimensions, then the butterfly, which leaves token 8 g + lane's score in this lane.
        float sc[OWN];
        float mt = -INFINITY;
        for (int g = 0; g < OWN; ++g) {
            float part[8];
            for (int j = 0; j < 8; ++j) {
                float s = 0.0f;
                for (int k = 0; k < RUNS; ++k) {
                    const float4 kk = ks[(8 * g + j) * (DIM / 4) + 8 * k + lane];
                    s = fmaf(qv[k].x, kk.x, s);
                    s = fmaf(qv[k].y, kk.y, s);
                    s = fmaf(qv[k].z, kk.z, s);
                    s = fmaf(qv[k].w, kk.w, s);
                }
                part[j] = s;
            }
            for (int half = 4; half >= 1; half /= 2) {
                const bool upper = (lane & half) != 0;
                for (int j = 0; j < half; ++j) {
                    const float lo = part[j], hi = part[j + half];
                    part[j] = (upper ? hi : lo) + __shfl_xor(upper ? lo : hi, half, 8);
                }
            }
            sc[g] = t0 + 8 * g + lane < my_end ? part[0] : -INFINITY;
            mt = fmaxf(mt, sc[g]);
        }
        mt = fmaxf(mt, __shfl_xor(mt, 1, 8));
        mt = fmaxf(mt, __shfl_xor(mt, 2, 8));
        mt = fmaxf(mt, __shfl_xor(mt, 4, 8));
        const float mn = fmaxf(m, mt);
        // A row with nothing unmasked yet keeps m at -inf; its weights are zero rather than the NaN of inf - inf.
        const float corr = mn == -INFINITY ? 1.0f : expf(m - mn);
        float sum = 0.0f;
        for (int g = 0; g < OWN; ++g) {
            const float w = sc[g] == -INFINITY ? 0.0f : expf(sc[g] - mn);
            ps[r * TK + 8 * g + lane] = w;
            sum += w;
        }
        sum += __shfl_xor(sum, 1, 8);
        sum += __shfl_xor(sum, 2, 8);
        sum += __shfl_xor(sum, 4, 8);
        l = l * corr + sum;
        m = mn;
        for (int k = 0; k < RUNS; ++k) acc[k] = make_float4(acc[k].x * corr, acc[k].y * corr, acc[k].z * corr, acc[k].w * corr);
        __syncthreads();
        // The weighted sum of the tile's values.
        for (int j = 0; j < TK; ++j) {
            const float w = ps[r * TK + j];
            for (int k = 0; k < RUNS; ++k) {
                const float4 vv = vs[j * (DIM / 4) + 8 * k + lane];
                acc[k] = make_float4(fmaf(w, vv.x, acc[k].x), fmaf(w, vv.y, acc[k].y), fmaf(w, vv.z, acc[k].z), fmaf(w, vv.w, acc[k].w));
            }
        }
        __syncthreads();
    }
    if (live)
        for (int k = 0; k < RUNS; ++k) {
            const int d = 32 * k + 4 * lane;
            O[qbase + d] = acc[k].x / l;
            O[qbase + d + 1] = acc[k].y / l;
            O[qbase + d + 2] = acc[k].z / l;
            O[qbase + d + 3] = acc[k].w / l;
        }
}

// Causal attention of rows that attend one at a time (the Vulkan per-row design, attention_vec.comp), heads 128 wide: a block a (row, head, split), 16 streams of 16 lanes, each stream's softmax online and the streams merged in stream order.
// A row's splits are chunks of its own length; one split writes O and its twin (twin_at), several write partials for llmx_attention_vec_merge; a row whose end is 0 is the tile kernel's.
extern "C" __global__ void llmx_attention_vec(const float* Q, float* O, const void* kc, const void* vc, const int* table, const int* firsts, const int* ends,
                                              int n_head, int heads_kv, int bt, int k_half, int v_half, float scale, int nsplit, int chunk0, int max_parts,
                                              float* partials, void* tq, float* td, int tlevel) {
    constexpr int DIM = 128, LANES = 16, STREAMS = kThreads / LANES;
    __shared__ float ms[STREAMS], ls[STREAMS];
    __shared__ float accs[STREAMS * DIM];
    const int split = blockIdx.x % nsplit, rh = blockIdx.x / nsplit, r = rh / n_head, h = rh % n_head;
    const int len = ends[r];
    if (len == 0) return;
    const int kvh = h / (n_head / heads_kv), lid = threadIdx.x, stream = lid / LANES, d0 = (lid % LANES) * 8;
    const int* blocks = table + firsts[r];
    const unsigned long long qrow = ((unsigned long long)r * n_head + h) * DIM;
    float qv[8], acc[8], m = -INFINITY, l = 0.0f;
    for (int e = 0; e < 8; ++e) {
        qv[e] = Q[qrow + d0 + e];
        acc[e] = 0.0f;
    }
    int chunk = chunk0;
    while ((len + chunk - 1) / chunk > max_parts) chunk *= 2;
    const int start = min(split * chunk, len), end = min(len, start + chunk);
    for (int t = start + stream; t < end; t += STREAMS) {
        const unsigned long long at = (((unsigned long long)blocks[t / bt] * heads_kv + kvh) * bt + t % bt) * DIM + d0;
        float k8[8], v8[8];
        load8(kc, at, k_half, k8);
        load8(vc, at, v_half, v8);
        float part = 0.0f;
        for (int e = 0; e < 8; ++e) part += qv[e] * k8[e];
        for (int o = LANES / 2; o > 0; o /= 2) part += __shfl_xor(part, o, LANES);
        const float sc = part * scale, mn = fmaxf(m, sc);
        const float corr = expf(m - mn), w = expf(sc - mn);
        l = l * corr + w;
        for (int e = 0; e < 8; ++e) acc[e] = acc[e] * corr + w * v8[e];
        m = mn;
    }
    if (lid == stream * LANES) {
        ms[stream] = m;
        ls[stream] = l;
    }
    for (int e = 0; e < 8; ++e) accs[stream * DIM + d0 + e] = acc[e];
    __syncthreads();
    if (lid >= DIM) return;
    float M = ms[0];
    for (int i = 1; i < STREAMS; ++i) M = fmaxf(M, ms[i]);
    float L = 0.0f, A = 0.0f;
    // A stream that saw no token holds -inf and zeros and adds nothing; exp(-inf - -inf) must never be formed.
    for (int i = 0; i < STREAMS; ++i) {
        if (isinf(ms[i])) continue;
        const float c = expf(ms[i] - M);
        L += ls[i] * c;
        A += accs[i * DIM + lid] * c;
    }
    if (nsplit == 1) {
        const float r = A / L;
        O[qrow + lid] = r;
        twin_at(r, qrow + lid, tq, td, tlevel);
    } else {
        float* out = partials + ((unsigned long long)rh * nsplit + split) * (DIM + 2);
        out[2 + lid] = A;
        if (lid == 0) {
            out[0] = M;
            out[1] = L;
        }
    }
}

// The splits of one (row, head) merged (the Vulkan split merge's order, attention_merge.comp): the largest maximum, then the sums in split order, a split without tokens weighing nothing.
extern "C" __global__ void llmx_attention_vec_merge(const float* partials, float* O, const int* ends, int n_head, int nsplit, void* tq, float* td, int tlevel) {
    constexpr int DIM = 128;
    __shared__ float sw[kThreads], sl[kThreads];
    const int rh = blockIdx.x, r = rh / n_head, lid = threadIdx.x;
    if (ends[r] == 0) return;
    const float* base = partials + (unsigned long long)rh * nsplit * (DIM + 2);
    float M = -INFINITY;
    for (int i = 0; i < nsplit; ++i) M = fmaxf(M, base[i * (DIM + 2)]);
    if (lid < nsplit) {
        const float mi = base[lid * (DIM + 2)];
        sw[lid] = isinf(mi) ? 0.0f : expf(mi - M);
        sl[lid] = base[lid * (DIM + 2) + 1];
    }
    __syncthreads();
    if (lid >= DIM) return;
    float L = 0.0f, A = 0.0f;
    for (int i = 0; i < nsplit; ++i) {
        const float c = sw[i];
        if (c == 0.0f) continue;
        L += sl[i] * c;
        A += base[i * (DIM + 2) + 2 + lid] * c;
    }
    const float o = A / L;
    O[(unsigned long long)rh * DIM + lid] = o;
    twin_at(o, (unsigned long long)rh * DIM + lid, tq, td, tlevel);
}

// A block a (row, head): the row's chunks merged in order, each scaled to the largest maximum.
extern "C" __global__ void llmx_attention_merge(const float* part, float* out, const int* ends, int n_head, int dim) {
    const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    if (t >= dim || ends[r] == 0) return;   // an end of 0 marks a row the tile kernel answered
    int count, length;
    attention_chunks(ends[r], count, length);
    const float* p = part + ((unsigned long long)r * n_head + h) * 16 * (dim + 2);
    float mx = -INFINITY;
    for (int c = 0; c < count; ++c) mx = fmaxf(mx, p[c * (dim + 2)]);
    float l = 0.0f, acc = 0.0f;
    for (int c = 0; c < count; ++c) {
        const float a = expf(p[c * (dim + 2)] - mx);
        l = fmaf(p[c * (dim + 2) + 1], a, l);
        acc = fmaf(p[c * (dim + 2) + 2 + t], a, acc);
    }
    out[((unsigned long long)r * n_head + h) * dim + t] = acc / l;
}
