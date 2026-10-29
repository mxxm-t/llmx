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

// A block a (row, head): causal attention over the row's `ends[r]` positions, whose blocks are table[firsts[r] ...], with an online softmax in position order.
// Each thread holds one dimension's accumulator; each score is a block sum, so the order follows the position alone.
extern "C" __global__ void llmx_attention(const float* Q, float* out, const void* kc, const void* vc, const int* table, const int* firsts,
                                          const int* ends, int n_head, int ratio, int heads_kv, int dim, int bt, int k_half, int v_half, float scale) {
    __shared__ float lds[kThreads];
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const unsigned long long qoff = ((unsigned long long)r * n_head + h) * dim;
    const float q = d < dim ? Q[qoff + d] : 0.0f;
    const int kvh = h / ratio;
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    const int* blocks = table + firsts[r];
    for (int t = 0; t < ends[r]; ++t) {
        const unsigned long long at = (((unsigned long long)blocks[t / bt] * heads_kv + kvh) * bt + t % bt) * dim + d;
        float kd = 0.0f, vd = 0.0f;
        if (d < dim) {
            kd = k_half ? __half2float(((const __half*)kc)[at]) : ((const float*)kc)[at];
            vd = v_half ? __half2float(((const __half*)vc)[at]) : ((const float*)vc)[at];
        }
        const float s = block_sum(q * kd, lds) * scale;
        const float mn = fmaxf(m, s);
        const float a = expf(m - mn), e = expf(s - mn);
        l = l * a + e;
        acc = acc * a + e * vd;
        m = mn;
    }
    if (d < dim) out[qoff + d] = acc / l;
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
