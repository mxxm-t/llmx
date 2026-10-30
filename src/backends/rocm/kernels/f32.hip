// The dense F32 kernels of the ROCm backend's first step (docs/ROCM.md, Delivery checkpoints, step 2): every sum is F32 in a fixed order that depends on the row alone, never on how many rows a call holds.
// A block reduction folds lane partials in one tree whatever the wave width, so a row gives the same bits in any call.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace {

constexpr int kThreads = 256;

// The sum of each thread's value over the block, the same tree for every block.
__device__ float block_sum(float v, float* lds) {
    lds[threadIdx.x] = v;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) lds[threadIdx.x] += lds[threadIdx.x + s];
        __syncthreads();
    }
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

} // namespace

// dst[i] += src[i].
extern "C" __global__ void llmx_add(float* dst, const float* src, unsigned long long n) {
    const unsigned long long i = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// dst[i] = silu(gate[i]) * up[i].
extern "C" __global__ void llmx_silu_mul(float* dst, const float* gate, const float* up, unsigned long long n) {
    const unsigned long long i = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        dst[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

// A block a row: dst row r = RMSNorm(src row r) against w.
extern "C" __global__ void llmx_rms_norm_rows(float* dst, const float* src, const float* w, int n, unsigned long long stride, float eps) {
    __shared__ float lds[kThreads];
    const unsigned long long r = blockIdx.x;
    rms_norm_row(dst + r * stride, src + r * stride, w, n, eps, lds);
}

// A block a (row, head): the head's RMS norm when w is given, then pairs (i, i + half) rotated at the row's position.
extern "C" __global__ void llmx_norm_rope(float* dst, const float* src, unsigned long long dst_stride, unsigned long long src_stride,
                                          unsigned long long src_head_stride, int head_dim, int half, const float* w, float eps,
                                          const float* cos, const float* sin, const unsigned* pos) {
    __shared__ float lds[kThreads];
    const unsigned long long r = blockIdx.x, h = blockIdx.y;
    float* head = dst + r * dst_stride + h * head_dim;
    const float* from = src + r * src_stride + h * src_head_stride;
    if (w) rms_norm_row(head, from, w, head_dim, eps, lds);
    else if (head != from)
        for (int i = threadIdx.x; i < head_dim; i += kThreads) head[i] = from[i];
    __syncthreads();
    const float* c = cos + (unsigned long long)pos[r] * half;
    const float* s = sin + (unsigned long long)pos[r] * half;
    for (int i = threadIdx.x; i < half; i += kThreads) {
        const float xa = head[i], xb = head[i + half];
        head[i] = fmaf(-xb, s[i], xa * c[i]);
        head[i + half] = fmaf(xa, s[i], xb * c[i]);
    }
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
        if (k_half) ((__half*)kc)[out + d] = __float2half_rn(k[in + d]);
        else ((float*)kc)[out + d] = k[in + d];
        if (v_half) ((__half*)vc)[out + d] = __float2half_rn(v[in + d]);
        else ((float*)vc)[out + d] = v[in + d];
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

// The chunk kernel's arithmetic for a tile of a view's rows that share their chunks (one count and length), and the query heads of one KV head: each 64-position tile of K, then of V, is read into shared memory once for every (row, head) pair of the block.
// Each pair's score, tile maximum, sum and accumulators are formed as llmx_attention_chunk forms them, the same operations in the same order, so a row gives the same bits in either kernel and in any tile.
// Block (tile, KV head, chunk); `tiles` gives each tile's first row and row count, at most 32 pairs a tile; F16 sides and heads up to 128 wide.
extern "C" __global__ void llmx_attention_tile(const float* Q, float* part, const __half* kc, const __half* vc, const int* table, const int* firsts,
                                               const int* ends, const int* tiles, int n_head, int ratio, int heads_kv, int dim, int bt, float scale) {
    __shared__ __half kv[64][130];   // a row padded by two halves, so the score phase's lanes on consecutive positions fall on different banks
    __shared__ float qs[32][128];
    __shared__ float sc[32][64];
    __shared__ float red[32][64];
    __shared__ float ms[32], ls[32], as[32];
    __shared__ int lasts[32];
    const int kvh = blockIdx.y, c = blockIdx.z, t = threadIdx.x;
    const int r0 = tiles[2 * blockIdx.x], nr = tiles[2 * blockIdx.x + 1], pairs = nr * ratio;
    int count, length;
    attention_chunks(ends[r0], count, length);
    if (c >= count) return;
    const int first = c * length;
    int tile_last = first;
    for (int i = 0; i < nr; ++i) tile_last = max(tile_last, min(ends[r0 + i], first + length));
    const int* blocks = table + firsts[r0];
    // Pair p is row r0 + p / ratio and head kvh * ratio + p % ratio.
    for (int e = t; e < pairs * dim; e += 256) {
        const int p = e / dim, x = e % dim;
        qs[p][x] = Q[((unsigned long long)(r0 + p / ratio) * n_head + kvh * ratio + p % ratio) * dim + x];
    }
    if (t < pairs) {
        ms[t] = -INFINITY;
        ls[t] = 0.0f;
        lasts[t] = min(ends[r0 + t / ratio], first + length);
    }
    // Accumulators: dimension t % 128 of pairs t / 128, t / 128 + 2, ...
    const int d = t % 128, pg = t / 128;
    float acc[16];
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
    const auto load = [&](const __half* src, int base) {
        for (int e = t; e < 64 * dim; e += 256) {
            const int j = e / dim, x = e % dim, pos = base + j;
            if (pos < tile_last) kv[j][x] = src[(((unsigned long long)blocks[pos / bt] * heads_kv + kvh) * bt + pos % bt) * dim + x];
        }
    };
    __syncthreads();
    for (int base = first; base < tile_last; base += 64) {
        load(kc, base);
        __syncthreads();
        // Scores: four partial sums over every fourth dimension, folded as the chunk kernel's two shuffles fold them.
        for (int e = t; e < pairs * 64; e += 256) {
            const int p = e / 64, j = e % 64, pos = base + j;
            float s = -INFINITY;
            if (pos < lasts[p]) {
                float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
                for (int x = 0; x < dim; x += 4) {
                    p0 = fmaf(qs[p][x], __half2float(kv[j][x]), p0);
                    p1 = fmaf(qs[p][x + 1], __half2float(kv[j][x + 1]), p1);
                    p2 = fmaf(qs[p][x + 2], __half2float(kv[j][x + 2]), p2);
                    p3 = fmaf(qs[p][x + 3], __half2float(kv[j][x + 3]), p3);
                }
                s = ((p0 + p1) + (p2 + p3)) * scale;
            }
            sc[p][j] = s;
            red[p][j] = s;
        }
        __syncthreads();
        load(vc, base);
        // Each pair's tile maximum and then its sum, the chunk kernel's trees over the 64 scores.
        for (int w = 32; w > 0; w /= 2) {
            for (int e = t; e < pairs * w; e += 256) red[e / w][e % w] = fmaxf(red[e / w][e % w], red[e / w][e % w + w]);
            __syncthreads();
        }
        // The running maximum taken with the tile's, kept in as[] until the sum is formed.
        if (t < pairs) as[t] = fmaxf(ms[t], red[t][0]);
        __syncthreads();
        for (int e = t; e < pairs * 64; e += 256) sc[e / 64][e % 64] = expf(sc[e / 64][e % 64] - as[e / 64]);
        __syncthreads();
        for (int e = t; e < pairs * 64; e += 256) red[e / 64][e % 64] = sc[e / 64][e % 64];
        __syncthreads();
        for (int w = 32; w > 0; w /= 2) {
            for (int e = t; e < pairs * w; e += 256) red[e / w][e % w] += red[e / w][e % w + w];
            __syncthreads();
        }
        if (t < pairs && base < lasts[t]) {
            const float a = expf(ms[t] - as[t]);
            ls[t] = ls[t] * a + red[t][0];
            ms[t] = as[t];
            as[t] = a;
        }
        __syncthreads();
        if (d < dim)
            for (int i = 0, p = pg; p < pairs; ++i, p += 2) {
                const int last = lasts[p];
                if (base >= last) continue;
                float v = acc[i] * as[p];
                for (int j = 0; j < 64 && base + j < last; ++j) v = fmaf(sc[p][j], __half2float(kv[j][d]), v);
                acc[i] = v;
            }
        __syncthreads();
    }
    if (d < dim)
        for (int i = 0, p = pg; p < pairs; ++i, p += 2)
            part[(((unsigned long long)(r0 + p / ratio) * n_head + kvh * ratio + p % ratio) * 16 + c) * (dim + 2) + 2 + d] = acc[i];
    if (t < pairs) {
        float* out = part + (((unsigned long long)(r0 + t / ratio) * n_head + kvh * ratio + t % ratio) * 16 + c) * (dim + 2);
        out[0] = ms[t];
        out[1] = ls[t];
    }
}

// A block a (row, head): the row's chunks merged in order, each scaled to the largest maximum.
extern "C" __global__ void llmx_attention_merge(const float* part, float* out, const int* ends, int n_head, int dim) {
    const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    if (t >= dim) return;
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
