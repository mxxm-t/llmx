// The dense F32 kernels and the activation twins of the ROCm backend: every sum is F32 in a fixed order that depends on the row alone (docs/src/backends-rocm.md, Kernel notes).
// A block reduction folds lane partials in one tree whatever the wave width, so a row gives the same bits in any call.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "backends/rocm/kernels/quant.h"

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

// Where a twin's blocks lie (the host's TwinMap, rocm_backend.cpp): its rows in regions, region g from row first[g] and its blocks from base[g], row after row where pad[g] is 0, else block k of the region's rows together, pad[g] rows a block.
struct TwinMap {
    unsigned regions, blocks, shift, magic_lo, magic_hi;   // fb / blocks = fb * magic >> shift for every fb below 2^31
    unsigned first[8], base[8], pad[8];
    unsigned total;
};

// The twin block of block k of row b.
__device__ inline unsigned long long twin_block(const TwinMap& m, unsigned b, unsigned k) {
    unsigned g = 0;
    for (unsigned i = 1; i < m.regions; ++i) g = b >= m.first[i] ? i : g;
    const unsigned r = b - m.first[g];
    return m.base[g] + (m.pad[g] ? k * m.pad[g] + r : r * m.blocks + k);
}

// The twin block of the block numbered fb = b * blocks + k as the floats lie; a null map lays every row row after row, as the map-free builds take it.
__device__ inline unsigned long long twin_block(const TwinMap* m, unsigned long long fb) {
    if (!m) return fb;
    const unsigned b = (unsigned)((fb * ((unsigned long long)m->magic_hi << 32 | m->magic_lo)) >> m->shift);
    return twin_block(*m, b, (unsigned)fb - b * m->blocks);
}

// Value v at element i of an output whose twin a later product reads, at the class `level` names (1 block-int16, 2 block-int8, 0 none); the 32 lanes of i's block call it together.
__device__ inline void twin_at(float v, unsigned long long i, void* q, float* d, int level, const TwinMap* m) {
    if (!level) return;
    const float amax = block_amax(v);
    const unsigned long long b = twin_block(m, i / 32);
    if (level == 2) twin_value<signed char, 127>(v, amax, (signed char*)q + b * 32 + i % 32, d + b, (int)(i % 32));
    else twin_value<short, 32767>(v, amax, (short*)q + b * 32 + i % 32, d + b, (int)(i % 32));
}

// The twin of `blocks` blocks of 32 floats, 32 threads a block.
template <typename QT, int L>
__device__ void twin(const float* x, QT* q, float* d, unsigned long long blocks, const TwinMap* m) {
    const unsigned long long b = (unsigned long long)blockIdx.x * (kThreads / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const float v = b < blocks ? x[b * 32 + lane] : 0.0f;
    const float amax = block_amax(v);
    if (b < blocks) {
        const unsigned long long t = twin_block(m, b);
        twin_value<QT, L>(v, amax, q + t * 32 + lane, d + t, lane);
    }
}

constexpr int kNormThreads = 1024;   // the held norm's block: four times kThreads, so each thread scales a quarter of the values

// A block a row of n <= K * kThreads values: dst = RMSNorm(src) against w, with its twin at L levels beside it when L is not 0.
// The sum takes rms_norm_row's chains and block_sum's tree, so both give the same bits.
template <int K, typename QT, int L>
__device__ void rms_norm_held(float* __restrict__ dst, const float* __restrict__ src, const float* __restrict__ w, int n, unsigned long long stride, float eps,
                              QT* __restrict__ q, float* __restrict__ d, const TwinMap* m) {
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
            const unsigned long long b = !m ? (r * (unsigned long long)n + i) / 32 : twin_block(*m, (unsigned)r, (unsigned)i / 32);
            twin_value<QT, L>(o, amax, q + b * 32 + i % 32, d + b, i % 32);
        }
    }
}

} // namespace

// Each producer of a twin has a map-free build, which a call whose rows lie row after row takes, and a build reading the map (_major).
extern "C" __global__ void __launch_bounds__(kThreads) llmx_twin16(const float* x, short* q, float* d, unsigned long long blocks) {
    twin<short, 32767>(x, q, d, blocks, nullptr);
}
extern "C" __global__ void __launch_bounds__(kThreads) llmx_twin8(const float* x, signed char* q, float* d, unsigned long long blocks) {
    twin<signed char, 127>(x, q, d, blocks, nullptr);
}
extern "C" __global__ void __launch_bounds__(kThreads) llmx_twin16_major(const float* x, short* q, float* d, unsigned long long blocks, TwinMap m) {
    twin<short, 32767>(x, q, d, blocks, &m);
}
extern "C" __global__ void __launch_bounds__(kThreads) llmx_twin8_major(const float* x, signed char* q, float* d, unsigned long long blocks, TwinMap m) {
    twin<signed char, 127>(x, q, d, blocks, &m);
}

// dst[i] += src[i].
extern "C" __global__ void llmx_add(float* dst, const float* src, unsigned long long n) {
    const unsigned long long i = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

// A tensor group's sum (RocmCollective, docs/src/backends-rocm.md): each member's area holds this block of signals, then a staging slot for each member's partial rows.
// kGroupMembers is the host's backend::kMaxTensorWidth (rocm_collective.hpp).
constexpr int kGroupBlocks = 16, kGroupMembers = 16;
struct GroupSignal {
    unsigned epoch[kGroupBlocks];   // the last fast sum's epoch, per block, first in the area
    unsigned abort;                 // set by a member whose wait ran out, read by every member's waits
    unsigned start[kGroupBlocks][kGroupMembers], done[kGroupBlocks][kGroupMembers];   // [block][sender]: the epoch a sender's block reached
};
static_assert(sizeof(GroupSignal) <= 4096, "the host's kGroupSignalBytes");
struct GroupPeers {
    float* stage[kGroupMembers];      // each member's staging slots, cap floats a member
    unsigned* signal[kGroupMembers];  // each member's GroupSignal
    unsigned* status;                 // host memory, a word a member, set where its sum failed
    unsigned* fault;                  // host memory, this member's backend's fault word
};
static_assert(sizeof(GroupPeers) == 272, "the host's GroupPeers");  // in device memory, written once at join, so the launch does not grow with kGroupMembers

// The members' values at i added in member order, as sum_rows.comp adds them: member `rank`'s from its own rows, the others' from its staging.
__device__ inline float4 group_values(const float* own, const float* stage, unsigned long long cap, int members, int rank, unsigned long long i) {
    float4 t = *(const float4*)(rank == 0 ? own + i : stage + i);
    for (int j = 1; j < members; ++j) {
        const float4 v = *(const float4*)(j == rank ? own + i : stage + j * cap + i);
        t.x += v.x;
        t.y += v.y;
        t.z += v.z;
        t.w += v.w;
    }
    return t;
}
__device__ inline float group_value(const float* own, const float* stage, unsigned long long cap, int members, int rank, unsigned long long i) {
    float t = rank == 0 ? own[i] : stage[i];
    for (int j = 1; j < members; ++j) t += j == rank ? own[i] : stage[j * cap + i];
    return t;
}

// Thread j of the block tells member j its block reached `epoch` and waits until member j's block has told it the same, giving up at the deadline or once a member set abort.
__device__ bool group_meet(const GroupPeers* p, GroupSignal* self, int members, int rank, unsigned epoch, bool end, unsigned long long deadline, int* failed) {
    const int b = blockIdx.x, j = threadIdx.x;
    if (j < members && j != rank) {
        GroupSignal* peer = (GroupSignal*)p->signal[j];
        __hip_atomic_store(end ? &peer->done[b][rank] : &peer->start[b][rank], epoch, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        unsigned* slot = end ? &self->done[b][j] : &self->start[b][j];
        while (__hip_atomic_load(slot, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != epoch)
            if (__hip_atomic_load(&self->abort, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) || (unsigned long long)wall_clock64() > deadline) {
                __hip_atomic_store(failed, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
                break;
            }
    }
    __syncthreads();
    // Every thread reads peers' staging after the barrier, so each takes the acquire itself.
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    return !*failed;
}

// A failed wait: every member stops waiting, and the host learns it from the status and fault words.
__device__ void group_failed(const GroupPeers* p, int members, int rank) {
    if (threadIdx.x) return;
    for (int k = 0; k < members; ++k) __hip_atomic_store(&((GroupSignal*)p->signal[k])->abort, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(&p->status[rank], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(p->fault, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Member `rank`'s part of a group sum of n floats: out = the members' rows added in member order (flag 1), or out += them.
// Fast (flag 2) pushes its rows into every peer's staging and meets before and after the sum; a failed meeting leaves NaN in out.
extern "C" __global__ void __launch_bounds__(kThreads) llmx_group_sum(const GroupPeers* p, const float* own, float* out, unsigned long long n, unsigned long long cap,
                                                                      int members, int rank, int flags, unsigned long long timeout) {
    __shared__ int failed;
    const bool replace = flags & 1, fast = flags & 2;
    GroupSignal* self = (GroupSignal*)p->signal[rank];
    const float* stage = p->stage[rank];
    const unsigned long long units = (n + 3) / 4, step = (unsigned long long)gridDim.x * kThreads;
    const unsigned long long first = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    unsigned epoch = 0;
    unsigned long long deadline = 0;
    if (fast) {
        deadline = (unsigned long long)wall_clock64() + timeout;
        // Epoch 0 is what fresh signals hold, so a wrap skips it.
        epoch = self->epoch[blockIdx.x] + 1;
        if (!epoch) epoch = 1;
        if (!threadIdx.x) failed = 0;
        for (unsigned long long u = first; u < units; u += step) {
            const unsigned long long i = u * 4;
            if (i + 4 <= n) {
                const float4 v = *(const float4*)(own + i);
                for (int j = 0; j < members; ++j)
                    if (j != rank) *(float4*)(p->stage[j] + rank * cap + i) = v;
            } else {
                for (unsigned long long k = i; k < n; ++k)
                    for (int j = 0; j < members; ++j)
                        if (j != rank) p->stage[j][rank * cap + k] = own[k];
            }
        }
        __threadfence_system();
        __syncthreads();
        if (!group_meet(p, self, members, rank, epoch, false, deadline, &failed)) {
            group_failed(p, members, rank);
            for (unsigned long long u = first; u < units; u += step)
                for (unsigned long long k = u * 4; k < n && k < u * 4 + 4; ++k) out[k] = __builtin_nanf("");
            return;
        }
    }
    const bool aligned = ((unsigned long long)out & 15) == 0;
    for (unsigned long long u = first; u < units; u += step) {
        const unsigned long long i = u * 4;
        if (i + 4 <= n && aligned) {
            const float4 t = group_values(own, stage, cap, members, rank, i);
            float4* o = (float4*)(out + i);
            if (replace) {
                *o = t;
            } else {
                float4 r = *o;
                r.x += t.x;
                r.y += t.y;
                r.z += t.z;
                r.w += t.w;
                *o = r;
            }
        } else {
            for (unsigned long long k = i; k < n && k < i + 4; ++k) {
                const float t = group_value(own, stage, cap, members, rank, k);
                if (replace) out[k] = t;
                else out[k] += t;
            }
        }
    }
    if (fast) {
        __syncthreads();
        if (!group_meet(p, self, members, rank, epoch, true, deadline, &failed)) return group_failed(p, members, rank);
        if (!threadIdx.x) self->epoch[blockIdx.x] = epoch;
    }
}

// dst[i] = silu(gate[i]) * up[i], with its twin at L levels beside it when L is not 0, n then a multiple of 32.
template <typename QT, int L>
__device__ void silu_mul(float* dst, const float* gate, const float* up, unsigned long long n, QT* q, float* d, const TwinMap* m) {
    const unsigned long long i = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    float o = 0.0f;
    if (i < n) {
        const float g = gate[i];
        o = g / (1.0f + expf(-g)) * up[i];
        dst[i] = o;
    }
    if (L) {
        const float amax = block_amax(o);
        if (i < n) {
            const unsigned long long b = twin_block(m, i / 32);
            twin_value<QT, L>(o, amax, q + b * 32 + i % 32, d + b, (int)(i % 32));
        }
    }
}
extern "C" __global__ void llmx_silu_mul(float* dst, const float* gate, const float* up, unsigned long long n, short* q, float* d) {
    silu_mul<short, 0>(dst, gate, up, n, q, d, nullptr);
}
extern "C" __global__ void llmx_silu_mul_twin16(float* dst, const float* gate, const float* up, unsigned long long n, short* q, float* d) {
    silu_mul<short, 32767>(dst, gate, up, n, q, d, nullptr);
}
extern "C" __global__ void llmx_silu_mul_twin8(float* dst, const float* gate, const float* up, unsigned long long n, signed char* q, float* d) {
    silu_mul<signed char, 127>(dst, gate, up, n, q, d, nullptr);
}
extern "C" __global__ void llmx_silu_mul_twin16_major(float* dst, const float* gate, const float* up, unsigned long long n, short* q, float* d, TwinMap m) {
    silu_mul<short, 32767>(dst, gate, up, n, q, d, &m);
}
extern "C" __global__ void llmx_silu_mul_twin8_major(float* dst, const float* gate, const float* up, unsigned long long n, signed char* q, float* d, TwinMap m) {
    silu_mul<signed char, 127>(dst, gate, up, n, q, d, &m);
}

// A block a row: dst row r = RMSNorm(src row r) against w, for any n.
extern "C" __global__ void llmx_rms_norm_rows(float* dst, const float* src, const float* w, int n, unsigned long long stride, float eps) {
    __shared__ float lds[kThreads];
    const unsigned long long r = blockIdx.x;
    rms_norm_row(dst + r * stride, src + r * stride, w, n, eps, lds);
}

// The held builds for rows of up to 4096 and 8192 values, plain and with the twin of either class, the twin's map-free and the map's (_major).
#define LLMX_NORM(name, K, QT, L)                                                                                                       \
    extern "C" __global__ void __launch_bounds__(kNormThreads) name(float* dst, const float* src, const float* w, int n, unsigned long long stride, \
                                                                float eps, QT* q, float* d) {                                         \
        rms_norm_held<K, QT, L>(dst, src, w, n, stride, eps, q, d, nullptr);                                                          \
    }
#define LLMX_NORM_MAJOR(name, K, QT, L)                                                                                                 \
    extern "C" __global__ void __launch_bounds__(kNormThreads) name(float* dst, const float* src, const float* w, int n, unsigned long long stride, \
                                                                float eps, QT* q, float* d, TwinMap m) {                                \
        rms_norm_held<K, QT, L>(dst, src, w, n, stride, eps, q, d, &m);                                                               \
    }
LLMX_NORM(llmx_rms_norm_16, 16, short, 0)
LLMX_NORM(llmx_rms_norm_16_twin16, 16, short, 32767)
LLMX_NORM(llmx_rms_norm_16_twin8, 16, signed char, 127)
LLMX_NORM(llmx_rms_norm_32, 32, short, 0)
LLMX_NORM(llmx_rms_norm_32_twin16, 32, short, 32767)
LLMX_NORM(llmx_rms_norm_32_twin8, 32, signed char, 127)
LLMX_NORM_MAJOR(llmx_rms_norm_16_twin16_major, 16, short, 32767)
LLMX_NORM_MAJOR(llmx_rms_norm_16_twin8_major, 16, signed char, 127)
LLMX_NORM_MAJOR(llmx_rms_norm_32_twin16_major, 32, short, 32767)
LLMX_NORM_MAJOR(llmx_rms_norm_32_twin8_major, 32, signed char, 127)

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

// Rows of an F32 table: dst row i = table row ids[i], `nin` floats each, and a zero row for an id of `nrows` or more.
extern "C" __global__ void llmx_gather(float* dst, const float* table, const unsigned* ids, unsigned long long width, unsigned nrows) {
    const unsigned long long i = blockIdx.y, j = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    const unsigned id = ids[i];
    if (j < width) dst[i * width + j] = id < nrows ? table[(unsigned long long)id * width + j] : 0.0f;
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
    const float d = rq::half_at(block);
    const signed char* q = (const signed char*)(block + 2);
    float s = 0.0f;
    for (int j = 0; j < 32; ++j) s = fmaf((float)q[j], x[j], s);
    return d * s;
}

// The exact products' parts, a lane's strided sum of a row against x that exact() folds by block_sum: F32 rows, Q8_0 rows by blocks, and rows of a type decoded value by value (kernels/quant.h).
struct ExactF32 {
    static __device__ unsigned long long row_bytes(unsigned long long nin) { return nin * sizeof(float); }
    static __device__ float part(const unsigned char* row, const float* x, unsigned long long nin) {
        const float* w = (const float*)row;
        float s = 0.0f;
        for (unsigned long long i = threadIdx.x; i < nin; i += kThreads) s = fmaf(w[i], x[i], s);
        return s;
    }
};
struct ExactQ8 {
    static __device__ unsigned long long row_bytes(unsigned long long nin) { return nin / rq::Q8_0::values * rq::Q8_0::bytes; }
    static __device__ float part(const unsigned char* row, const float* x, unsigned long long nin) {
        float s = 0.0f;
        for (unsigned long long k = threadIdx.x; k < nin / rq::Q8_0::values; k += kThreads) s += q8_0_block_dot(row + k * rq::Q8_0::bytes, x + k * rq::Q8_0::values);
        return s;
    }
};
template <class T> struct ExactDecoded {
    static __device__ unsigned long long row_bytes(unsigned long long nin) { return nin / T::values * T::bytes; }
    static __device__ float part(const unsigned char* row, const float* x, unsigned long long nin) {
        float s = 0.0f;
        for (unsigned long long i = threadIdx.x; i < nin; i += kThreads) s = fmaf(T::value(row + i / T::values * T::bytes, (int)(i % T::values)), x[i], s);
        return s;
    }
};
// Output (o, b), a block an output: Y[b * nout + o] = dot(W row o, X row b), added to Y with `add`.
template <class D>
__device__ void exact(const unsigned char* W, const float* X, float* Y, unsigned long long nin, unsigned long long nout, int add, unsigned long long o, unsigned long long b) {
    __shared__ float lds[kThreads];
    const float dot = block_sum(D::part(W + o * D::row_bytes(nin), X + b * nin, nin), lds);
    if (threadIdx.x == 0) {
        float* y = Y + b * nout + o;
        *y = add ? *y + dot : dot;
    }
}
// A routed exact call (backend.hpp matmul_experts): the entries' ids, the launch's first entry, the entries an X row serves, the experts and an expert's bytes.
struct RoutedExact {
    const unsigned* ids;
    unsigned long long first, per, n_expert, expert_bytes;
};
// Entry e = first + blockIdx.y, output blockIdx.x: its expert's row against X row e / per into Y + e * nout, the one-row product's arithmetic; an entry without an expert writes nothing.
template <class D> __device__ void routed_exact(const unsigned char* W, const float* X, float* Y, unsigned long long nin, unsigned long long nout, RoutedExact r) {
    const unsigned long long e = r.first + blockIdx.y;
    const unsigned id = r.ids[e];
    if (id >= r.n_expert) return;
    exact<D>(W + id * r.expert_bytes, X + e / r.per * nin, Y + e * nout, nin, nout, 0, blockIdx.x, 0);
}
#define LLMX_EXACT(t, ...)                                                                                                                                         \
    extern "C" __global__ void llmx_matmul_##t(const unsigned char* W, const float* X, float* Y, unsigned long long nin, unsigned long long nout, int add) {     \
        exact<__VA_ARGS__>(W, X, Y, nin, nout, add, blockIdx.x, blockIdx.y);                                                                                    \
    }                                                                                                                                                              \
    extern "C" __global__ void llmx_routed_##t(const unsigned char* W, const float* X, float* Y, unsigned long long nin, unsigned long long nout, RoutedExact r) { \
        routed_exact<__VA_ARGS__>(W, X, Y, nin, nout, r);                                                                                                       \
    }
LLMX_EXACT(f32, ExactF32)
LLMX_EXACT(q8_0, ExactQ8)

// Rows of a table of type T decoded: dst row i = table row ids[i], `width` floats each, and a zero row for an id of `nrows` or more; offsets are 64-bit, since a lookup table passes 4 GiB.
template <class T> __device__ void gather_decoded(float* dst, const unsigned char* table, const unsigned* ids, unsigned long long width, unsigned nrows) {
    const unsigned long long i = blockIdx.y, j = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    if (j >= width) return;
    const unsigned id = ids[i];
    if (id >= nrows) {
        dst[i * width + j] = 0.0f;
        return;
    }
    const unsigned char* row = table + (unsigned long long)id * (width / T::values) * T::bytes;
    dst[i * width + j] = T::value(row + j / T::values * T::bytes, (int)(j % T::values));
}

#define LLMX_DECODED(t, T) \
    LLMX_EXACT(t, ExactDecoded<rq::T>) \
    extern "C" __global__ void llmx_gather_##t(float* dst, const unsigned char* table, const unsigned* ids, unsigned long long width, unsigned nrows) { \
        gather_decoded<rq::T>(dst, table, ids, width, nrows); \
    }
LLMX_DECODED(q4_k, Q4_K)
LLMX_DECODED(q5_k, Q5_K)
LLMX_DECODED(q6_k, Q6_K)
LLMX_DECODED(q5_0, Q5_0)
LLMX_DECODED(q5_1, Q5_1)
LLMX_DECODED(bf16, BF16)
LLMX_DECODED(iq4_nl, IQ4_NL)

extern "C" __global__ void llmx_gather_q8_0(float* dst, const unsigned char* table, const unsigned* ids, unsigned long long width, unsigned nrows) {
    gather_decoded<rq::Q8_0>(dst, table, ids, width, nrows);
}

// A row's attention chunks: as many as its history needs at 256 positions each, at most 16, each a whole number of 64-position tiles; a function of the history alone, so a row's chunks are the same in any call.
__device__ inline void attention_chunks(int end, int& count, int& length) {
    count = min(16, (end + 255) / 256);
    length = ((end + count - 1) / count + 63) / 64 * 64;
}

// A block a (row, head, chunk) of the split attention, an online softmax over the chunk's 64-position tiles in order, its state written to `part` for llmx_attention_merge.
// SPARSE reads the row's `stride` selected ids in place of its positions, an id outside the row's causal prefix skipped.
template <bool SPARSE>
__device__ inline void attention_chunk(const float* Q, float* part, const void* kc, const void* vc, const int* table, const int* firsts, const int* ends,
                                       int n_head, int ratio, int heads_kv, int dim, int bt, int k_half, int v_half, float scale, const unsigned* ids,
                                       int stride) {
    __shared__ float sc[64];
    __shared__ float red[64];
    __shared__ unsigned tok[64];
    const int r = blockIdx.x, h = blockIdx.y, c = blockIdx.z, t = threadIdx.x;
    int count, length;
    attention_chunks(SPARSE ? stride : ends[r], count, length);
    if (c >= count) return;
    const int first = c * length, last = min(SPARSE ? stride : ends[r], first + length);
    const float* q = Q + ((unsigned long long)r * n_head + h) * dim;
    const int kvh = h / ratio;
    const int* blocks = table + firsts[r];
    float m = -INFINITY, l = 0.0f, acc = 0.0f;
    for (int base = first; base < last; base += 64) {
        if (SPARSE && t < 64) {
            const unsigned id = base + t < last ? ids[(unsigned long long)r * stride + base + t] : ~0u;
            tok[t] = id < (unsigned)ends[r] ? id : ~0u;
        }
        if (SPARSE) __syncthreads();
        const int p = t / 4, quarter = t % 4;
        const bool valid = SPARSE ? tok[p] != ~0u : base + p < last;
        const int pos = SPARSE ? (valid ? (int)tok[p] : 0) : base + p;
        float s = 0.0f;
        if (valid) {
            const unsigned long long at = (((unsigned long long)blocks[pos / bt] * heads_kv + kvh) * bt + pos % bt) * dim;
            for (int d = quarter; d < dim; d += 4) {
                const float kd = k_half ? __half2float(((const __half*)kc)[at + d]) : ((const float*)kc)[at + d];
                s = fmaf(q[d], kd, s);
            }
        }
        s += __shfl_xor(s, 1, 4);
        s += __shfl_xor(s, 2, 4);
        if (quarter == 0) sc[p] = valid ? s * scale : -INFINITY;
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
        // A sparse tile of no selected id after none before leaves the chunk as it was; exp(-inf - -inf) must never be formed.
        if (SPARSE && mt == -INFINITY) continue;
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
        l = fmaf(l, a, red[0]);
        m = mt;
        __syncthreads();
        if (t < dim) {
            float v = acc * a;
            for (int j = 0; j < 64 && base + j < last; ++j) {
                if (SPARSE && tok[j] == ~0u) continue;
                const int pj = SPARSE ? (int)tok[j] : base + j;
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
extern "C" __global__ void llmx_attention_chunk(const float* Q, float* part, const void* kc, const void* vc, const int* table, const int* firsts,
                                                const int* ends, int n_head, int ratio, int heads_kv, int dim, int bt, int k_half, int v_half,
                                                float scale) {
    attention_chunk<false>(Q, part, kc, vc, table, firsts, ends, n_head, ratio, heads_kv, dim, bt, k_half, v_half, scale, nullptr, 0);
}
extern "C" __global__ void llmx_sparse_attention_chunk(const float* Q, float* part, const void* kc, const void* vc, const int* table, const int* firsts,
                                                       const int* ends, int n_head, int ratio, int heads_kv, int dim, int bt, int k_half, int v_half,
                                                       float scale, const unsigned* ids, int stride) {
    attention_chunk<true>(Q, part, kc, vc, table, firsts, ends, n_head, ratio, heads_kv, dim, bt, k_half, v_half, scale, ids, stride);
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

// Causal attention of a prompt's rows in the Vulkan tile's design: a block a tile of 32 query rows of one view and one head, heads 128 wide.
// A row's softmax runs online over the tiles from token 0 in order, so its result depends on its own history alone.
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
        l = fmaf(l, corr, sum);
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

// Causal attention of rows that attend one at a time in the Vulkan per-row design, heads 128 or 256 wide: a block a (row, head, split), its streams merged in stream order.
// One split writes O and its twin and several write partials for llmx_attention_vec_merge; SPARSE splits its selected ids by the stride alone.
template <int DIM, bool SPARSE>
__device__ inline void attention_vec(const float* Q, float* O, const void* kc, const void* vc, const int* table, const int* firsts, const int* ends,
                                     int n_head, int heads_kv, int bt, int k_half, int v_half, float scale, int nsplit, int chunk0, int max_parts,
                                     float* partials, void* tq, float* td, int tlevel, const TwinMap* tm, const unsigned* ids, int stride) {
    constexpr int LANES = DIM / 8, STREAMS = kThreads / LANES;
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
    while (!SPARSE && (len + chunk - 1) / chunk > max_parts) chunk *= 2;
    const int span = SPARSE ? stride : len, start = min(split * chunk, span), end = start + min(chunk, span - start);
    for (int slot = start + stream; slot < end; slot += STREAMS) {
        const unsigned id = SPARSE ? ids[(unsigned long long)r * stride + slot] : 0u;
        if (SPARSE && id >= (unsigned)len) continue;
        const int t = SPARSE ? (int)id : slot;
        const unsigned long long at = (((unsigned long long)blocks[t / bt] * heads_kv + kvh) * bt + t % bt) * DIM + d0;
        float k8[8], v8[8];
        load8(kc, at, k_half, k8);
        load8(vc, at, v_half, v8);
        float part = 0.0f;
        for (int e = 0; e < 8; ++e) part = fmaf(qv[e], k8[e], part);
        for (int o = LANES / 2; o > 0; o /= 2) part += __shfl_xor(part, o, LANES);
        const float sc = part * scale, mn = fmaxf(m, sc);
        const float corr = expf(m - mn), w = expf(sc - mn);
        // The fused products are written out, so where this is inlined cannot choose which product the compiler fuses.
        l = fmaf(l, corr, w);
        for (int e = 0; e < 8; ++e) acc[e] = fmaf(w, v8[e], acc[e] * corr);
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
        L = fmaf(ls[i], c, L);
        A = fmaf(accs[i * DIM + lid], c, A);
    }
    if (nsplit == 1) {
        const float r = SPARSE && !(L > 0.0f) ? 0.0f : A / L;
        O[qrow + lid] = r;
        twin_at(r, qrow + lid, tq, td, tlevel, tm);
    } else {
        float* out = partials + ((unsigned long long)rh * nsplit + split) * (DIM + 2);
        out[2 + lid] = A;
        if (lid == 0) {
            out[0] = M;
            out[1] = L;
        }
    }
}
extern "C" __global__ void llmx_attention_vec(const float* Q, float* O, const void* kc, const void* vc, const int* table, const int* firsts, const int* ends,
                                              int n_head, int heads_kv, int bt, int k_half, int v_half, float scale, int nsplit, int chunk0, int max_parts,
                                              float* partials, void* tq, float* td, int tlevel) {
    attention_vec<128, false>(Q, O, kc, vc, table, firsts, ends, n_head, heads_kv, bt, k_half, v_half, scale, nsplit, chunk0, max_parts, partials, tq, td, tlevel,
                              nullptr, nullptr, 0);
}
extern "C" __global__ void llmx_attention_vec_major(const float* Q, float* O, const void* kc, const void* vc, const int* table, const int* firsts,
                                                    const int* ends, int n_head, int heads_kv, int bt, int k_half, int v_half, float scale, int nsplit, int chunk0,
                                                    int max_parts, float* partials, void* tq, float* td, int tlevel, TwinMap tm) {
    attention_vec<128, false>(Q, O, kc, vc, table, firsts, ends, n_head, heads_kv, bt, k_half, v_half, scale, nsplit, chunk0, max_parts, partials, tq, td, tlevel, &tm,
                              nullptr, 0);
}
extern "C" __global__ void llmx_attention_vec256(const float* Q, float* O, const void* kc, const void* vc, const int* table, const int* firsts, const int* ends,
                                                 int n_head, int heads_kv, int bt, int k_half, int v_half, float scale, int nsplit, int chunk0, int max_parts,
                                                 float* partials, void* tq, float* td, int tlevel) {
    attention_vec<256, false>(Q, O, kc, vc, table, firsts, ends, n_head, heads_kv, bt, k_half, v_half, scale, nsplit, chunk0, max_parts, partials, tq, td, tlevel,
                              nullptr, nullptr, 0);
}
extern "C" __global__ void llmx_attention_vec256_major(const float* Q, float* O, const void* kc, const void* vc, const int* table, const int* firsts,
                                                       const int* ends, int n_head, int heads_kv, int bt, int k_half, int v_half, float scale, int nsplit, int chunk0,
                                                       int max_parts, float* partials, void* tq, float* td, int tlevel, TwinMap tm) {
    attention_vec<256, false>(Q, O, kc, vc, table, firsts, ends, n_head, heads_kv, bt, k_half, v_half, scale, nsplit, chunk0, max_parts, partials, tq, td, tlevel, &tm,
                              nullptr, 0);
}
#define LLMX_SPARSE_VEC(name, D) \
    extern "C" __global__ void name(const float* Q, float* O, const void* kc, const void* vc, const int* table, const int* firsts, const int* ends, int n_head, \
                                    int heads_kv, int bt, int k_half, int v_half, float scale, int nsplit, int chunk, float* partials, const unsigned* ids, \
                                    int stride) { \
        attention_vec<D, true>(Q, O, kc, vc, table, firsts, ends, n_head, heads_kv, bt, k_half, v_half, scale, nsplit, chunk, 0, partials, nullptr, nullptr, 0, \
                               nullptr, ids, stride); \
    }
LLMX_SPARSE_VEC(llmx_sparse_attention_vec, 128)
LLMX_SPARSE_VEC(llmx_sparse_attention_vec256, 256)

// The splits of one (row, head) merged (the Vulkan split merge's order, attention_merge.comp): the largest maximum, then the sums in split order, a split without tokens weighing nothing.
// SPARSE gives zeros for a row whose splits saw no selected id.
template <int DIM, bool SPARSE>
__device__ inline void attention_vec_merge(const float* partials, float* O, const int* ends, int n_head, int nsplit, void* tq, float* td, int tlevel, const TwinMap* tm) {
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
        L = fmaf(sl[i], c, L);
        A = fmaf(base[i * (DIM + 2) + 2 + lid], c, A);
    }
    const float o = SPARSE && !(L > 0.0f) ? 0.0f : A / L;
    O[(unsigned long long)rh * DIM + lid] = o;
    twin_at(o, (unsigned long long)rh * DIM + lid, tq, td, tlevel, tm);
}
extern "C" __global__ void llmx_attention_vec_merge(const float* partials, float* O, const int* ends, int n_head, int nsplit, void* tq, float* td, int tlevel) {
    attention_vec_merge<128, false>(partials, O, ends, n_head, nsplit, tq, td, tlevel, nullptr);
}
extern "C" __global__ void llmx_attention_vec_merge_major(const float* partials, float* O, const int* ends, int n_head, int nsplit, void* tq, float* td, int tlevel,
                                                          TwinMap tm) {
    attention_vec_merge<128, false>(partials, O, ends, n_head, nsplit, tq, td, tlevel, &tm);
}
extern "C" __global__ void llmx_attention_vec_merge256(const float* partials, float* O, const int* ends, int n_head, int nsplit, void* tq, float* td, int tlevel) {
    attention_vec_merge<256, false>(partials, O, ends, n_head, nsplit, tq, td, tlevel, nullptr);
}
extern "C" __global__ void llmx_attention_vec_merge256_major(const float* partials, float* O, const int* ends, int n_head, int nsplit, void* tq, float* td,
                                                             int tlevel, TwinMap tm) {
    attention_vec_merge<256, false>(partials, O, ends, n_head, nsplit, tq, td, tlevel, &tm);
}
extern "C" __global__ void llmx_sparse_attention_vec_merge(const float* partials, float* O, const int* ends, int n_head, int nsplit) {
    attention_vec_merge<128, true>(partials, O, ends, n_head, nsplit, nullptr, nullptr, 0, nullptr);
}
extern "C" __global__ void llmx_sparse_attention_vec_merge256(const float* partials, float* O, const int* ends, int n_head, int nsplit) {
    attention_vec_merge<256, true>(partials, O, ends, n_head, nsplit, nullptr, nullptr, 0, nullptr);
}

// A block a (row, head): the row's chunks merged in order, each scaled to the largest maximum.
// SPARSE takes the chunks of the row's `stride` ids, skips a chunk that saw none and gives zeros for a row of none.
template <bool SPARSE>
__device__ inline void attention_merge(const float* part, float* out, const int* ends, int n_head, int dim, int stride) {
    const int r = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    if (t >= dim || ends[r] == 0) return;   // an end of 0 marks a row the tile kernel answered
    int count, length;
    attention_chunks(SPARSE ? stride : ends[r], count, length);
    const float* p = part + ((unsigned long long)r * n_head + h) * 16 * (dim + 2);
    float mx = -INFINITY;
    for (int c = 0; c < count; ++c) mx = fmaxf(mx, p[c * (dim + 2)]);
    float l = 0.0f, acc = 0.0f;
    for (int c = 0; c < count; ++c) {
        if (SPARSE && isinf(p[c * (dim + 2)])) continue;
        const float a = expf(p[c * (dim + 2)] - mx);
        l = fmaf(p[c * (dim + 2) + 1], a, l);
        acc = fmaf(p[c * (dim + 2) + 2 + t], a, acc);
    }
    out[((unsigned long long)r * n_head + h) * dim + t] = SPARSE && !(l > 0.0f) ? 0.0f : acc / l;
}
extern "C" __global__ void llmx_attention_merge(const float* part, float* out, const int* ends, int n_head, int dim) {
    attention_merge<false>(part, out, ends, n_head, dim, 0);
}
extern "C" __global__ void llmx_sparse_attention_merge(const float* part, float* out, const int* ends, int n_head, int dim, int stride) {
    attention_merge<true>(part, out, ends, n_head, dim, stride);
}

// The qwen35 layers' ops (backend.hpp), each as the CPU computes it, the transcendental steps with expf.

// dst[i] = x[i] * sigmoid(g), g being element d of head h of row r's gate at r * gs + h * ghs + d, where i = (r * heads + h) * dim + d; dst may be x.
extern "C" __global__ void llmx_sigmoid_mul(float* dst, const float* x, const float* gate, unsigned long long n, unsigned long long width,
                                            unsigned long long dim, unsigned long long gs, unsigned long long ghs) {
    const unsigned long long i = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    const unsigned long long r = i / width, rem = i - r * width, h = rem / dim, d = rem - h * dim;
    dst[i] = x[i] * (1.0f / (1.0f + expf(-gate[r * gs + h * ghs + d])));
}

// A block a (row, head): x's RMS norm against w, times SiLU or, with `sig`, the sigmoid of z; dst may be x, since every value is read into the sum before the block's barrier.
extern "C" __global__ void llmx_gated_rms_norm(float* dst, const float* x, const float* z, const float* w, int dim, float eps, int sig) {
    __shared__ float lds[kThreads];
    const unsigned long long o = (unsigned long long)blockIdx.x * dim;
    float s = 0.0f;
    for (int i = threadIdx.x; i < dim; i += kThreads) s = fmaf(x[o + i], x[o + i], s);
    const float r = 1.0f / sqrtf(block_sum(s, lds) / (float)dim + eps);
    for (int i = threadIdx.x; i < dim; i += kThreads) {
        const float zz = z[o + i], g = sig ? 1.0f / (1.0f + expf(-zz)) : zz / (1.0f + expf(-zz));
        dst[o + i] = x[o + i] * (r * w[i]) * g;
    }
}

// A block a (row, head) of a rope over `count` streams: the head's RMS norm when w is given, then pair i rotated at stream of[i]'s position of the row, pos holding `count` a row.
// A thread reads only the values it writes, after the norm's barrier, so dst may be src over contiguous heads.
extern "C" __global__ void llmx_norm_rope_streams(float* dst, const float* src, unsigned long long dst_stride, unsigned long long src_stride,
                                                  unsigned long long src_head_stride, int head_dim, int half, const float* w, float eps,
                                                  const float* cos, const float* sin, const unsigned* pos, const unsigned char* of, int count) {
    __shared__ float lds[kThreads];
    const unsigned long long r = blockIdx.x, h = blockIdx.y;
    float* head = dst + r * dst_stride + h * head_dim;
    const float* from = src + r * src_stride + h * src_head_stride;
    float rs = 1.0f;
    if (w) {
        float s = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += kThreads) s = fmaf(from[i], from[i], s);
        rs = 1.0f / sqrtf(block_sum(s, lds) / (float)head_dim + eps);
    }
    for (int i = threadIdx.x; i < half; i += kThreads) {
        const float xa = w ? from[i] * (rs * w[i]) : from[i], xb = w ? from[i + half] * (rs * w[i + half]) : from[i + half];
        const unsigned long long t = (unsigned long long)pos[r * count + of[i]] * half + i;
        head[i] = fmaf(-xb, sin[t], xa * cos[t]);
        head[i + half] = fmaf(xa, sin[t], xb * cos[t]);
    }
    for (int i = 2 * half + threadIdx.x; i < head_dim; i += kThreads) head[i] = w ? from[i] * (rs * w[i]) : from[i];
}

// The linear attention's causal conv of four taps, then SiLU (backend.hpp, causal_conv_silu), a thread a channel, as the CPU's products in tap order, so a row is the same however its view is cut.
// The layout of `tab` is in docs/src/backends-rocm.md, Kernel notes.
extern "C" __global__ void llmx_causal_conv_silu(float* o, const float* x, const float* w, float* st, const unsigned* tab, unsigned C,
                                                 unsigned long long slot, unsigned long long carried, unsigned chunk) {
    const unsigned c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= C) return;
    const unsigned k = 1u + 5u * tab[0] + 2u * blockIdx.y, v = tab[k], b0 = tab[k + 1];
    const unsigned row0 = tab[1 + 5 * v], nq = tab[2 + 5 * v], len = tab[3 + 5 * v];
    const float* src = st + tab[4 + 5 * v] * slot + carried;
    const auto raw = [&](unsigned t) { return t < 3u ? (len + t >= 3u ? src[(unsigned long long)t * C + c] : 0.0f) : x[(unsigned long long)(row0 + t - 3u) * C + c]; };
    const float w0 = w[4ull * c], w1 = w[4ull * c + 1], w2 = w[4ull * c + 2], w3 = w[4ull * c + 3];
    float win0 = raw(b0), win1 = raw(b0 + 1), win2 = raw(b0 + 2);
    float keep0 = 0.0f, keep1 = 0.0f, keep2 = 0.0f;
    if (b0 == 0) keep0 = raw(nq), keep1 = raw(nq + 1), keep2 = raw(nq + 2);
    const unsigned end = nq < b0 + chunk ? nq : b0 + chunk;
    for (unsigned b = b0; b < end; ++b) {
        const float now = x[(unsigned long long)(row0 + b) * C + c];
        float acc = w0 * win0;
        acc = fmaf(w1, win1, acc);
        acc = fmaf(w2, win2, acc);
        acc = fmaf(w3, now, acc);
        o[(unsigned long long)(row0 + b) * C + c] = acc / (1.0f + expf(-acc));
        win0 = win1, win1 = win2, win2 = now;
    }
    if (b0 == 0) {
        float* d = st + tab[5 + 5 * v] * slot + carried + c;
        d[0] = keep0, d[C] = keep1, d[2ull * C] = keep2;
    }
}

// The sum over the eight lanes of a column or of a row, each step adding the same two partials, so every lane gets the same bits.
__device__ float lanes8(float x) {
    const float a = x + __shfl_xor(x, 1), b = a + __shfl_xor(a, 2);
    return b + __shfl_xor(b, 4);
}
// log(1 + y) without the cancellation of 1 + y for a small y.
__device__ float log1p_of(float y) {
    const float up = 1.0f + y, down = up + -1.0f;
    return up == 1.0f ? y : logf(up) * (y / down);
}

// The gated delta rule (backend.hpp, gated_delta_rule) in the Vulkan shader's order: a block a (view, V head, 32 V columns), every sum a lane's fused multiply-adds then lanes8.
// A sequence gives the same bits alone, batched or cut into passes, and a decay below 2^-126 is 0.
extern "C" __global__ void llmx_gated_delta_rule(float* o, const float* u, const float* alpha, const float* bb, const float* a, const float* dt_bias, float* st,
                                                 const unsigned* tab, unsigned C, unsigned k_heads, unsigned v_heads, unsigned k_dim, unsigned v_dim,
                                                 unsigned long long slot, unsigned blocks, float scale, float eps) {
    constexpr unsigned kRows = 16, kTb = 16;
    __shared__ float kn[kTb][128], qn[kTb][128], vs[kTb][32], betas[kTb], decays[kTb];
    const unsigned wg = blockIdx.x, per_view = v_heads * blocks;
    const unsigned v = wg / per_view, rest = wg - v * per_view, j = rest / blocks, blk = rest - j * blocks;
    const unsigned lid = threadIdx.x, part = lid & 7u, c = blk * 32u + (lid >> 3u);
    const bool col = c < v_dim;
    const unsigned qk = k_heads * k_dim, kh = j % k_heads;
    const unsigned row0 = tab[1 + 5 * v], nq = tab[2 + 5 * v];
    const bool fresh = tab[3 + 5 * v] == 0u;
    const unsigned long long matrix = (unsigned long long)j * k_dim * v_dim + c;
    const unsigned long long src = tab[4 + 5 * v] * slot + matrix, dst = tab[5 + 5 * v] * slot + matrix;
    float s[kRows];
    const unsigned g = lid >> 3u, tok = g >> 1u;
    const bool is_k = (g & 1u) != 0u;
    float m = 0.0f;
    bool have_m = false;
    for (unsigned t0 = 0; t0 < nq; t0 += kTb) {
        const unsigned n = nq - t0 < kTb ? nq - t0 : kTb;
        if (tok < n) {
            const unsigned long long in0 = (unsigned long long)(row0 + t0 + tok) * C + (is_k ? qk : 0u) + kh * k_dim;
            float x[kRows], ss = 0.0f;
#pragma unroll
            for (unsigned ii = 0; ii < kRows; ++ii) {
                const unsigned i = part + 8u * ii;
                x[ii] = i < k_dim ? u[in0 + i] : 0.0f;
                ss = fmaf(x[ii], x[ii], ss);
            }
            const float inv = 1.0f / sqrtf(lanes8(ss) + eps);
#pragma unroll
            for (unsigned ii = 0; ii < kRows; ++ii) {
                const float nv = x[ii] * inv;
                if (is_k) kn[tok][part + 8u * ii] = nv;
                else qn[tok][part + 8u * ii] = nv * scale;
            }
        }
        if (lid < n) {
            const unsigned long long r = row0 + t0 + lid;
            const float beta = 1.0f / (1.0f + expf(-bb[r * v_heads + j]));
            const float xg = alpha[r * v_heads + j] + dt_bias[j];
            const float sp = xg > 20.0f ? xg : log1p_of(expf(xg));
            const float decay = expf(a[j] * sp);
            betas[lid] = beta;
            decays[lid] = decay < 1.17549435e-38f ? 0.0f : decay;
        }
        for (unsigned idx = lid; idx < n * 32u; idx += 256u) {
            const unsigned t = idx >> 5u, cc = blk * 32u + (idx & 31u);
            vs[t][idx & 31u] = cc < v_dim ? u[(unsigned long long)(row0 + t0 + t) * C + 2u * qk + j * v_dim + cc] : 0.0f;
        }
        __syncthreads();
        if (t0 == 0) {
#pragma unroll
            for (unsigned ii = 0; ii < kRows; ++ii) {
                const unsigned i = part + 8u * ii;
                s[ii] = col && i < k_dim && !fresh ? st[src + (unsigned long long)i * v_dim] : 0.0f;
            }
        }
        if (col) {
            if (!have_m) {
                float mp = 0.0f;
                const float decay = decays[0];
#pragma unroll
                for (unsigned ii = 0; ii < kRows; ++ii) {
                    s[ii] = s[ii] * decay;
                    mp = fmaf(s[ii], kn[0][part + 8u * ii], mp);
                }
                m = lanes8(mp);
                have_m = true;
            }
            for (unsigned t = 0; t < n; ++t) {
                const unsigned long long r = row0 + t0 + t;
                const float d = betas[t] * (vs[t][lid >> 3u] + -m);
                float op = 0.0f;
#pragma unroll
                for (unsigned ii = 0; ii < kRows; ++ii) {
                    s[ii] = fmaf(kn[t][part + 8u * ii], d, s[ii]);
                    op = fmaf(s[ii], qn[t][part + 8u * ii], op);
                }
                float np = 0.0f;
                const bool next = t + 1u < n;
                if (next) {
                    const float decay = decays[t + 1u];
#pragma unroll
                    for (unsigned ii = 0; ii < kRows; ++ii) {
                        s[ii] = s[ii] * decay;
                        np = fmaf(s[ii], kn[t + 1u][part + 8u * ii], np);
                    }
                }
                const float res = lanes8(op);
                if (next) m = lanes8(np);
                if (part == 0u) o[r * v_heads * v_dim + j * v_dim + c] = res;
            }
            have_m = false;
        }
        __syncthreads();
    }
    if (col) {
#pragma unroll
        for (unsigned ii = 0; ii < kRows; ++ii) {
            const unsigned i = part + 8u * ii;
            if (i < k_dim) st[dst + (unsigned long long)i * v_dim] = s[ii];
        }
    }
}

// The Flash-Next stream ops (backend.hpp), each at its Vulkan kernel's op boundary and in its order, every value a function of its own row.

// A block a (row, group) and slice of rows: the group's RMS norm against its own weights, each thread's squares in order, then block_sum's tree.
// Every slice repeats the whole reduction, so a group's bits do not depend on its slices.
extern "C" __global__ void llmx_group_rms_norm(float* dst, const float* src, const float* w, unsigned groups, unsigned dim, float eps) {
    __shared__ float lds[kThreads];
    const unsigned long long o = (unsigned long long)blockIdx.x * dim;
    const float* wg = w + (unsigned long long)(blockIdx.x % groups) * dim;
    float s = 0.0f;
    for (unsigned i = threadIdx.x; i < dim; i += kThreads) s = fmaf(src[o + i], src[o + i], s);
    const float r = 1.0f / sqrtf(block_sum(s, lds) / (float)dim + eps);
    for (unsigned c = blockIdx.y; c * 1024u < dim; c += gridDim.y)
        for (unsigned i = c * 1024u + threadIdx.x; i < dim && i < (c + 1u) * 1024u; i += kThreads) dst[o + i] = src[o + i] * (r * wg[i]);
}

// dst[i] = silu(src[i] * scale); dst may be src.
extern "C" __global__ void llmx_silu_scaled(float* dst, const float* src, unsigned long long n, float scale) {
    const unsigned long long i = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    const float v = src[i] * scale;
    dst[i] = v / (1.0f + expf(-v));
}

// dst[r][d] = the sum over streams s in order of src[r][s][d] * sigmoid(gate[r][s][d]), divided by the streams.
extern "C" __global__ void llmx_mix_streams(float* dst, const float* src, const float* gate, unsigned long long n, unsigned streams, unsigned dim) {
    const unsigned long long i = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    const unsigned long long r = i / dim, d = i - r * dim;
    float sum = 0.0f;
    for (unsigned s = 0; s < streams; ++s) {
        const unsigned long long j = (r * streams + s) * dim + d;
        sum = fmaf(src[j], 1.0f / (1.0f + expf(-gate[j])), sum);
    }
    dst[i] = sum / (float)streams;
}

// dst[r][s][d] += block[r][d] * 2 sigmoid(gate[r][s] / streams), the residual streams in place; block b takes row and stream b / chunks, so blocks walk memory in order and no index divides 64 bits.
extern "C" __global__ void llmx_inject_streams(float* dst, const float* block, const float* gate, unsigned streams, unsigned dim) {
    const unsigned chunks = (dim + kThreads - 1) / kThreads, row = blockIdx.x / chunks, d = (blockIdx.x - row * chunks) * kThreads + threadIdx.x;
    if (d >= dim) return;
    const unsigned long long rs = row, r = row / streams, s = rs - r * streams, i = rs * dim + d;
    const float weight = 2.0f * (1.0f / (1.0f + expf(-(gate[r * streams + s] / (float)streams))));
    dst[i] = fmaf(block[r * dim + d], weight, dst[i]);
}

// A block a (row, group): g = dot(key, query) * scale over the group in block_sum's order, and the row's values times sigmoid(sign(g) sqrt(max(|g|, 1e-6))).
// A NaN score is kept as the CPU keeps it.
extern "C" __global__ void llmx_signed_sqrt_gate(float* dst, const float* key, const float* query, const float* value, unsigned groups, unsigned dim, float scale) {
    __shared__ float lds[kThreads];
    const unsigned long long o = (unsigned long long)blockIdx.x * dim, v0 = (unsigned long long)(blockIdx.x / groups) * dim;
    float s = 0.0f;
    for (unsigned i = threadIdx.x; i < dim; i += kThreads) s = fmaf(key[o + i], query[o + i], s);
    const float g = block_sum(s, lds) * scale;
    const float sign = g > 0.0f ? 1.0f : g < 0.0f ? -1.0f : g == 0.0f ? 0.0f : g, magnitude = fabsf(g) < 1e-6f ? 1e-6f : fabsf(g);
    const float factor = 1.0f / (1.0f + expf(-(sign * sqrtf(magnitude))));
    for (unsigned i = threadIdx.x; i < dim; i += kThreads) dst[o + i] = value[v0 + i] * factor;
}

// The dilated causal conv, its SiLU and the residual added (backend.hpp, dilated_conv_silu_add), a thread a channel, the taps in order, so a row is the same however its view is cut.
// Only a view's first chunk reads its carried raw rows, before it leaves the view's last raw rows in slot dst.
extern "C" __global__ void llmx_dilated_conv_silu_add(float* o, const float* x, const float* residual, const float* w, float* st, const unsigned* tab, unsigned C, unsigned taps,
                                                      unsigned dilation, unsigned history, unsigned long long slot, unsigned long long prefix, unsigned chunk) {
    const unsigned c = blockIdx.x * kThreads + threadIdx.x;
    if (c >= C) return;
    const unsigned k = 1u + 5u * tab[0] + 2u * blockIdx.y, v = tab[k], b0 = tab[k + 1];
    const unsigned row0 = tab[1 + 5 * v], nq = tab[2 + 5 * v], len = tab[3 + 5 * v];
    const float* src = st + tab[4 + 5 * v] * slot + prefix;
    const unsigned end = nq < b0 + chunk ? nq : b0 + chunk;
    for (unsigned b = b0; b < end; ++b) {
        float acc = 0.0f;
        for (unsigned j = 0; j < taps; ++j) {
            const unsigned lag = (taps - 1u - j) * dilation;
            float value;
            if (b >= lag) value = x[(unsigned long long)(row0 + b - lag) * C + c];
            else {
                const unsigned back = lag - b;
                value = len >= back ? src[(unsigned long long)(history - back) * C + c] : 0.0f;
            }
            acc = j == 0 ? w[(unsigned long long)c * taps] * value : fmaf(w[(unsigned long long)c * taps + j], value, acc);
        }
        const unsigned long long i = (unsigned long long)(row0 + b) * C + c;
        o[i] = residual[i] + acc / (1.0f + expf(-acc));
    }
    if (b0 == 0) {
        float* d = st + tab[5 + 5 * v] * slot + prefix + c;
        const unsigned missing = history - (history < nq ? history : nq);
        for (unsigned i = 0; i < history; ++i) {
            float value;
            if (nq >= history) value = x[(unsigned long long)(row0 + nq - history + i) * C + c];
            else if (i < missing) {
                const unsigned old = nq + i;
                value = len >= history - old ? src[(unsigned long long)old * C + c] : 0.0f;
            } else value = x[(unsigned long long)(row0 + i - missing) * C + c];
            d[(unsigned long long)i * C] = value;
        }
    }
}

// Expert routing (backend.hpp route_experts), a row a block of kRouteThreads: the softmax over the row's experts, then k picks of the largest probability, ties to the lower id, in Vulkan's order (moe_route.glsl).
// The reductions run through shared memory in one fixed tree, so a row's picks and weights do not depend on the wave size.
constexpr unsigned kRouteThreads = 64, kRouteExperts = 1024, kNoExpert = 0xffffffffu;
__device__ inline float route_max(float v, float* red) {
    const unsigned t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (unsigned s = kRouteThreads / 2; s; s /= 2) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}
__device__ inline float route_sum(float v, float* red) {
    const unsigned t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (unsigned s = kRouteThreads / 2; s; s /= 2) {
        if (t < s) red[t] = red[t] + red[t + s];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}
__device__ inline unsigned route_min(unsigned v, unsigned* red) {
    const unsigned t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (unsigned s = kRouteThreads / 2; s; s /= 2) {
        if (t < s) red[t] = min(red[t], red[t + s]);
        __syncthreads();
    }
    const unsigned r = red[0];
    __syncthreads();
    return r;
}
extern "C" __global__ void __launch_bounds__(kRouteThreads) llmx_moe_route(const float* scores, unsigned* ids, float* weights, unsigned n_expert, unsigned k, int normalize) {
    __shared__ float p[kRouteExperts], red[kRouteThreads], chosen[256];
    __shared__ unsigned redi[kRouteThreads], chosen_id[256];
    const unsigned t = threadIdx.x;
    const unsigned long long r = blockIdx.x;
    const float* s = scores + r * n_expert;
    float top = -INFINITY;
    for (unsigned e = t; e < n_expert; e += kRouteThreads) {
        p[e] = s[e];
        top = fmaxf(top, p[e]);
    }
    top = route_max(top, red);
    float part = 0.0f;
    for (unsigned e = t; e < n_expert; e += kRouteThreads) {
        const float v = expf(p[e] - top);
        p[e] = v;
        part += v;
    }
    const float sum = route_sum(part, red);
    // Probabilities that are not finite, from a NaN or infinite score or every score -inf, give no candidate.
    if (isnan(sum)) {
        for (unsigned j = t; j < k; j += kRouteThreads) {
            ids[r * k + j] = kNoExpert;
            weights[r * k + j] = 0.0f;
        }
        return;
    }
    // A probability below FLT_MIN counts as 0 under any float mode (backend.hpp kNoExpert).
    for (unsigned e = t; e < n_expert; e += kRouteThreads) {
        const float v = p[e] / sum;
        p[e] = v < __uint_as_float(0x00800000u) ? 0.0f : v;
    }
    __syncthreads();
    float total = 0.0f;
    for (unsigned j = 0; j < k; ++j) {
        // The largest untaken probability, then the lowest id holding it; a taken expert holds -1, below every probability.
        float bv = -2.0f;
        unsigned bi = kNoExpert;
        for (unsigned e = t; e < n_expert; e += kRouteThreads)
            if (p[e] > bv) bv = p[e], bi = e;
        const float m = route_max(bv, red);
        const unsigned id = route_min(bv == m ? bi : kNoExpert, redi);
        const float c = id < n_expert ? m : 0.0f;
        total += c;
        if (t == 0) {
            if (id < n_expert) p[id] = -1.0f;
            chosen[j] = c;
            chosen_id[j] = id;
        }
        __syncthreads();
    }
    for (unsigned j = t; j < k; j += kRouteThreads) {
        ids[r * k + j] = chosen_id[j];
        weights[r * k + j] = chosen_id[j] >= n_expert ? 0.0f : normalize ? chosen[j] / total : chosen[j];
    }
}

// y += the routed down projection's slots weighted in order (backend.hpp, matmul_experts_add) in Vulkan's order: slot 0's product is the sum, and each later one is rounded, then added.
// A slot without an expert adds nothing.
extern "C" __global__ void llmx_moe_combine(float* y, const float* part, const float* weights, const unsigned* ids, unsigned long long rows, unsigned long long n, unsigned k) {
    const unsigned long long i = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    if (i >= rows * n) return;
    const unsigned long long r = i / n, o = i - r * n;
    float s = 0.0f;
    for (unsigned j = 0; j < k; ++j)
        if (ids[r * k + j] != kNoExpert) {
            const float product = weights[r * k + j] * part[(r * k + j) * n + o];
            s = j == 0 ? product : s + product;
        }
    y[i] = y[i] + s;
}

// A grouped routed call's groups in the Vulkan grouping, a block an expert, the layout of `tab` in docs/src/backends-rocm.md, Kernel notes.
// The grouping orders the work only: an entry's outputs are its one-column product in any group.
extern "C" __global__ void llmx_moe_group(const unsigned* ids, unsigned* tab, unsigned entries, unsigned n_expert, unsigned max_groups, unsigned chunk) {
    __shared__ unsigned hist[1024], sa[kThreads], sb[kThreads];
    const unsigned e = blockIdx.x, t = threadIdx.x;
    // The sums of a and b over the block, by a tree.
    const auto total = [&](unsigned a, unsigned b, unsigned& ra, unsigned& rb) {
        sa[t] = a;
        sb[t] = b;
        __syncthreads();
        for (unsigned h = kThreads / 2; h > 0; h >>= 1) {
            if (t < h) {
                sa[t] += sa[t + h];
                sb[t] += sb[t + h];
            }
            __syncthreads();
        }
        ra = sa[0];
        rb = sb[0];
        __syncthreads();
    };
    for (unsigned x = t; x < n_expert; x += kThreads) hist[x] = 0;
    __syncthreads();
    for (unsigned i = t; i < entries; i += kThreads)
        if (ids[i] < n_expert) atomicAdd(&hist[ids[i]], 1u);
    __syncthreads();
    // This expert's first entry and first group follow the experts below it; expert 0's block also clears the groups past the last one used.
    unsigned f = 0, tl = 0;
    for (unsigned x = t; x < e; x += kThreads) {
        f += hist[x];
        tl += (hist[x] + chunk - 1) / chunk;
    }
    unsigned first, group0;
    total(f, tl, first, group0);
    const unsigned count = hist[e];
    if (e == 0) {
        unsigned all = 0, unused;
        for (unsigned x = t; x < n_expert; x += kThreads) all += (hist[x] + chunk - 1) / chunk;
        unsigned used;
        total(all, 0, used, unused);
        for (unsigned y = used + t; y < max_groups; y += kThreads) tab[4 * y] = tab[4 * y + 1] = tab[4 * y + 2] = tab[4 * y + 3] = 0;
    }
    if (count == 0) return;
    for (unsigned j = t * chunk; j < count; j += kThreads * chunk) {
        const unsigned y = group0 + j / chunk;
        tab[4 * y] = e;
        tab[4 * y + 1] = first + j;
        tab[4 * y + 2] = min(chunk, count - j);
        tab[4 * y + 3] = 0;
    }
    // Placement kThreads entries at a time: an entry's slot is the expert's cursor plus the earlier entries of this expert in the step, by a prefix sum.
    unsigned cursor = first;
    for (unsigned c0 = 0; c0 < entries; c0 += kThreads) {
        const unsigned i = c0 + t, m = i < entries && ids[i] == e ? 1u : 0u;
        sa[t] = m;
        __syncthreads();
        for (unsigned off = 1; off < kThreads; off <<= 1) {
            const unsigned v = t >= off ? sa[t - off] : 0u;
            __syncthreads();
            sa[t] += v;
            __syncthreads();
        }
        if (m) tab[4 * max_groups + cursor + sa[t] - 1] = i;
        cursor += sa[kThreads - 1];
        __syncthreads();
    }
}

// The indexer (backend.hpp, select_kv and pool_kv), Vulkan's design (kv_index_pool.comp, kv_index_scores_tile.comp, kv_index_select.comp and kv_index_select_wide.comp at 424ea29f3) with the CPU's order where one value is formed.
// A view's auxiliary blocks hold, each of block_floats, its tokens' raw rows (rows_floats) and then a pooled row a complete group.

// A block a complete group first + blockIdx.x: its raw rows summed in token order times 1 / compression, block_sum's RMS norm against w, the first 2 * half values rotated at the group's position, into its pooled row.
// With `streams` positions a group (GroupPositions) a pair turns at the one its stream names, counted from group `base`, else at the group's first token.
extern "C" __global__ void llmx_kv_index_pool(float* aux, const float* w, const float* cs, const float* sn, const int* blocks, const unsigned* gpos, const unsigned char* of,
                                              unsigned first, unsigned base, unsigned dim, unsigned half, unsigned compression, unsigned bt, unsigned long long block_floats,
                                              unsigned rows_floats, float eps, unsigned streams) {
    __shared__ float lds[kThreads], vals[kThreads];
    const unsigned g = first + blockIdx.x, t = threadIdx.x, start = g * compression;
    float v = 0.0f;
    if (t < dim) {
        for (unsigned i = 0; i < compression; ++i) {
            const unsigned tok = start + i;
            v += aux[(unsigned long long)blocks[tok / bt] * block_floats + (unsigned long long)(tok % bt) * dim + t];
        }
        v *= 1.0f / (float)compression;
    }
    const float r = 1.0f / sqrtf(block_sum(t < dim ? v * v : 0.0f, lds) / (float)dim + eps);
    const float n = t < dim ? (v * r) * w[t] : 0.0f;
    vals[t] = n;
    __syncthreads();
    if (t >= dim) return;
    float out = n;
    if (t < 2u * half) {
        const unsigned i = t < half ? t : t - half;
        const unsigned at = streams == 0u ? start : gpos[(unsigned long long)(g - base) * streams + (streams > 1u ? of[i] : 0u)];
        const unsigned long long ri = (unsigned long long)at * half + i;
        const float c = cs[ri], s = sn[ri], a = vals[i], b = vals[i + half];
        out = t < half ? a * c - b * s : b * c + a * s;
    }
    aux[(unsigned long long)blocks[start / bt] * block_floats + rows_floats + (unsigned long long)((start % bt) / compression) * dim + t] = out;
}

// A block scores eight groups of one tile for one query row, 32 lanes a group, each head's dot in slot order then a butterfly, a NaN dot kept as the CPU keeps it.
// Row r sees (hist + r + 1) / compression groups, its scores a row of `stride` apart.
extern "C" __global__ void llmx_kv_index_scores(const float* q, const float* aux, float* scores, const int* blocks, unsigned hist, unsigned stride, unsigned heads, unsigned dim,
                                                unsigned compression, unsigned bt, unsigned long long block_floats, unsigned rows_floats, float scale) {
    const unsigned r = blockIdx.y, lane = threadIdx.x & 31u, g = blockIdx.x * 8u + (threadIdx.x >> 5);
    if (g >= (hist + r + 1u) / compression) return;
    const unsigned t0 = g * compression;
    const float* key = aux + (unsigned long long)blocks[t0 / bt] * block_floats + rows_floats + (unsigned long long)((t0 % bt) / compression) * dim;
    const float* qr = q + (unsigned long long)r * heads * dim;
    float k[8];
#pragma unroll
    for (unsigned c = 0; c < 8u; ++c) k[c] = lane + 32u * c < dim ? key[lane + 32u * c] : 0.0f;
    float score = 0.0f;
    for (unsigned h = 0; h < heads; ++h) {
        float s = 0.0f;
#pragma unroll
        for (unsigned c = 0; c < 8u; ++c)
            if (lane + 32u * c < dim) s = fmaf(qr[h * dim + lane + 32u * c], k[c], s);
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor(s, o, 32);
        score += s < 0.0f ? 0.0f : s;
    }
    if (lane == 0u) scores[(unsigned long long)r * stride + g] = score * scale;
}

// The inclusive prefix sums of each thread's two counts over the block, Hillis and Steele's steps.
__device__ void scan2(unsigned& a, unsigned& b, unsigned* sa, unsigned* sb) {
    sa[threadIdx.x] = a;
    sb[threadIdx.x] = b;
    __syncthreads();
    for (unsigned step = 1; step < kThreads; step <<= 1) {
        const unsigned x = threadIdx.x >= step ? sa[threadIdx.x - step] : 0u, y = threadIdx.x >= step ? sb[threadIdx.x - step] : 0u;
        __syncthreads();
        sa[threadIdx.x] += x;
        sb[threadIdx.x] += y;
        __syncthreads();
    }
    a = sa[threadIdx.x];
    b = sb[threadIdx.x];
}

// Groups [first, end) of a row's scores in chronological order, those above `pre` and the earliest equal ones, `left + 1` in all, emitted after `emitted` groups with `equals` equal ones seen before.
// Scores order by their raw F32 bits, and with `all` every group is taken.
__device__ void emit_groups(const float* sc, unsigned* out, unsigned first, unsigned end, unsigned pre, unsigned left, bool all, unsigned compression, unsigned& emitted,
                            unsigned& equals, unsigned* sa, unsigned* sb) {
    for (unsigned start = first; start < end; start += kThreads) {
        const unsigned i = start + threadIdx.x, bits = i < end ? __float_as_uint(sc[i]) : 0u;
        unsigned a = i < end && (all || bits > pre) ? 1u : 0u, b = i < end && !all && bits == pre ? 1u : 0u;
        scan2(a, b, sa, sb);
        const unsigned room = left + 1u - min(equals, left + 1u), allowed = min(b, room);
        if (i < end && (all || bits > pre || (bits == pre && equals + b <= left + 1u))) {
            const unsigned slot = emitted + a + allowed - 1u;
            for (unsigned t = 0; t < compression; ++t) out[slot * compression + t] = i * compression + t;
        }
        emitted += sa[kThreads - 1] + min(sb[kThreads - 1], room);
        equals += sb[kThreads - 1];
        __syncthreads();
    }
}

// A block a row: eight 4-bit radix passes find the take-th largest score's bits, then the groups go out in order, the causal tail after them and UINT32_MAX to `stride`.
extern "C" __global__ void llmx_kv_index_select(const float* scores, unsigned* ids, unsigned hist, unsigned groups, unsigned compression, unsigned budget, unsigned stride) {
    __shared__ unsigned bins[16], sa[kThreads], sb[kThreads];
    __shared__ unsigned prefix, mask, rank;
    const unsigned r = blockIdx.x, lid = threadIdx.x, visible = hist + r + 1u, count = visible / compression, take = min(count, budget / compression);
    const float* sc = scores + (unsigned long long)r * groups;
    unsigned* out = ids + (unsigned long long)r * stride;
    for (unsigned i = lid; i < stride; i += kThreads) out[i] = 0xffffffffu;
    if (lid == 0) {
        prefix = 0u;
        mask = 0u;
        rank = take ? take - 1u : 0u;
    }
    __syncthreads();
    if (count > take) {
        for (int shift = 28; shift >= 0; shift -= 4) {
            if (lid < 16) bins[lid] = 0u;
            __syncthreads();
            for (unsigned i = lid; i < count; i += kThreads) {
                const unsigned bits = __float_as_uint(sc[i]);
                if ((bits & mask) == prefix) atomicAdd(&bins[(bits >> shift) & 15u], 1u);
            }
            __syncthreads();
            if (lid == 0) {
                for (int digit = 15; digit >= 0; --digit) {
                    if (rank < bins[digit]) {
                        prefix |= (unsigned)digit << shift;
                        break;
                    }
                    rank -= bins[digit];
                }
                mask |= 15u << shift;
            }
            __syncthreads();
        }
    }
    unsigned emitted = 0, equals = 0;
    emit_groups(sc, out, 0, count, prefix, rank, count == take, compression, emitted, equals, sa, sb);
    for (unsigned t = lid; t < visible % compression; t += kThreads) out[take * compression + t] = count * compression + t;
}

// llmx_kv_index_select over slices of 1024 groups, a block a (slice, row) of at most 16 rows and five launches by `stage`: histograms of 11, 11 and 10 bits into `state`, each slice's greater and equal counts, their emission.
// Per row `state` holds the three histograms (2048, 2048 and 1024 bins, zero between calls), then the bits and rank found; every row's slice counts follow the 16 rows.
constexpr unsigned kSelectSlice = 1024, kSelectRowWords = 5122, kSelectPrefix = 5120, kSelectRank = 5121, kSelectRows = 16;
__device__ const unsigned kSelectShift[3] = {21, 10, 0}, kSelectWidth[3] = {2048, 2048, 1024}, kSelectAt[3] = {0, 2048, 4096};

// The digit of pass k holding the rank left, from that pass's histogram, which counts only the scores matching the digits before it.
__device__ void select_pick(const unsigned* state, unsigned k, unsigned* bins, unsigned* sa, unsigned& prefix, unsigned& mask, unsigned& rank) {
    __shared__ unsigned found_prefix, found_mask, found_rank;
    const unsigned n = kSelectWidth[k], per = n / kThreads, lid = threadIdx.x;
    for (unsigned i = lid; i < n; i += kThreads) bins[i] = state[kSelectAt[k] + i];
    if (lid == 0) {
        found_prefix = prefix;
        found_mask = mask;
        found_rank = rank;
    }
    __syncthreads();
    // Thread t sums the bins of its run, the runs taken from the top digit down.
    const unsigned top = n - 1u - lid * per;
    unsigned sum = 0, unused = 0;
    for (unsigned j = 0; j < per; ++j) sum += bins[top - j];
    unsigned upto = sum;
    scan2(upto, unused, sa, sa + kThreads);
    const unsigned from = upto - sum;
    unsigned left = rank;
    if (left >= from && left < upto) {
        left -= from;
        for (unsigned j = 0; j < per; ++j) {
            const unsigned h = bins[top - j];
            if (left < h) {
                found_prefix = prefix | (top - j) << kSelectShift[k];
                found_mask = mask | (n - 1u) << kSelectShift[k];
                found_rank = left;
                break;
            }
            left -= h;
        }
    }
    __syncthreads();
    prefix = found_prefix;
    mask = found_mask;
    rank = found_rank;
    __syncthreads();
}

extern "C" __global__ void llmx_kv_index_select_wide(const float* scores, unsigned* ids, unsigned* state, unsigned hist, unsigned groups, unsigned compression, unsigned budget,
                                                     unsigned stride, unsigned stage, unsigned slices) {
    __shared__ unsigned bins[2048], sa[2 * kThreads];
    __shared__ unsigned greater_n, equal_n;
    const unsigned slice = blockIdx.x, r = blockIdx.y, lid = threadIdx.x, visible = hist + r + 1u, count = visible / compression, take = min(count, budget / compression);
    const float* sc = scores + (unsigned long long)r * groups;
    unsigned* out = ids + (unsigned long long)r * stride;
    unsigned* st = state + r * kSelectRowWords;
    const bool all = count == take;
    const unsigned first = slice * kSelectSlice, end = min(count, first + kSelectSlice);
    unsigned prefix = 0, mask = 0, rank = take ? take - 1u : 0u;
    if (stage < 3u) {
        // One digit's histogram of the scores that match the digits found before it.
        if (all) return;
        for (unsigned k = 0; k < stage; ++k) select_pick(st, k, bins, sa, prefix, mask, rank);
        const unsigned shift = kSelectShift[stage], n = kSelectWidth[stage];
        for (unsigned i = lid; i < n; i += kThreads) bins[i] = 0u;
        __syncthreads();
        for (unsigned i = first + lid; i < end; i += kThreads) {
            const unsigned bits = __float_as_uint(sc[i]);
            if ((bits & mask) == prefix) atomicAdd(&bins[(bits >> shift) & (n - 1u)], 1u);
        }
        __syncthreads();
        for (unsigned i = lid; i < n; i += kThreads)
            if (bins[i]) atomicAdd(&st[kSelectAt[stage] + i], bins[i]);
        return;
    }
    unsigned* counts = state + kSelectRows * kSelectRowWords + (unsigned long long)r * slices * 2u;
    if (stage == 3u) {
        // The take-th largest score's bits and the equals left to take, then each slice's greater and equal scores; the first slice also writes the row's tail and padding.
        if (!all)
            for (unsigned k = 0; k < 3u; ++k) select_pick(st, k, bins, sa, prefix, mask, rank);
        if (slice == 0) {
            if (lid == 0) {
                st[kSelectPrefix] = prefix;
                st[kSelectRank] = rank;
            }
            const unsigned tail = visible % compression;
            for (unsigned i = take * compression + lid; i < stride; i += kThreads)
                out[i] = i < take * compression + tail ? count * compression + (i - take * compression) : 0xffffffffu;
        }
        if (lid == 0) {
            greater_n = 0u;
            equal_n = 0u;
        }
        __syncthreads();
        for (unsigned i = first + lid; i < end; i += kThreads) {
            const unsigned bits = __float_as_uint(sc[i]);
            if (all || bits > prefix) atomicAdd(&greater_n, 1u);
            else if (bits == prefix) atomicAdd(&equal_n, 1u);
        }
        __syncthreads();
        if (lid == 0) {
            counts[2 * slice] = greater_n;
            counts[2 * slice + 1] = equal_n;
        }
        return;
    }
    // Stage 4: the slice's groups after the greater and the earliest equal ones of the slices before it; the slices also clear the histograms for the next call.
    const unsigned pre = st[kSelectPrefix], left = st[kSelectRank];
    unsigned ga = 0, eb = 0;
    for (unsigned s = lid; s < slice; s += kThreads) {
        ga += counts[2 * s];
        eb += counts[2 * s + 1];
    }
    sa[lid] = ga;
    sa[kThreads + lid] = eb;
    __syncthreads();
    for (unsigned k = kThreads / 2; k > 0; k >>= 1) {
        if (lid < k) {
            sa[lid] += sa[lid + k];
            sa[kThreads + lid] += sa[kThreads + lid + k];
        }
        __syncthreads();
    }
    unsigned emitted = sa[0] + min(sa[kThreads], left + 1u), equals = sa[kThreads];
    __syncthreads();
    for (unsigned i = slice * kThreads + lid; i < kSelectPrefix; i += slices * kThreads) st[i] = 0u;
    emit_groups(sc, out, first, end, pre, left, all, compression, emitted, equals, sa, sa + kThreads);
}

// The id of each row's largest logit, ties to the lowest (Backend::argmax_rows), or `n` where a slice saw a NaN or an infinity or the largest is -inf.
// Every step keeps the larger value and the lower id of equal values, so the id is the CPU's in any reduction order.
__device__ inline void argmax_pick(float& v, unsigned& id, float ov, unsigned oid) {
    if (ov > v || (ov == v && oid < id)) {
        v = ov;
        id = oid;
    }
}
__device__ inline void argmax_block(float& v, unsigned& id, bool& bad) {
    __shared__ float sv[kThreads];
    __shared__ unsigned si[kThreads];
    __shared__ int sb[kThreads];
    const int lid = threadIdx.x;
    sv[lid] = v;
    si[lid] = id;
    sb[lid] = bad;
    __syncthreads();
    for (int o = kThreads / 2; o > 0; o /= 2) {
        if (lid < o) {
            argmax_pick(sv[lid], si[lid], sv[lid + o], si[lid + o]);
            sb[lid] |= sb[lid + o];
        }
        __syncthreads();
    }
    v = sv[0];
    id = si[0];
    bad = sb[0] != 0;
}
__device__ inline unsigned argmax_id(float v, unsigned id, bool bad, unsigned n, const unsigned* after, unsigned r) {
    return bad || !(v > -INFINITY) || (after && after[r] >= n) ? n : id;
}
extern "C" __global__ void llmx_argmax_slices(const float* logits, unsigned* ids, float* parts, unsigned n, unsigned slice, unsigned nslice, const unsigned* after) {
    const unsigned r = blockIdx.x / nslice, sl = blockIdx.x % nslice, start = sl * slice, end = min(n, start + slice);
    const float* row = logits + (unsigned long long)r * n;
    float v = -INFINITY;
    unsigned id = UINT_MAX;
    bool bad = false;
    for (unsigned i = start + threadIdx.x; i < end; i += kThreads) {
        const float x = row[i];
        bad = bad || isnan(x) || (isinf(x) && x > 0.0f);
        argmax_pick(v, id, x, i);
    }
    argmax_block(v, id, bad);
    if (threadIdx.x) return;
    if (nslice == 1) {
        ids[r] = argmax_id(v, id, bad, n, after, r);
    } else {
        parts[2 * blockIdx.x] = bad ? NAN : v;
        parts[2 * blockIdx.x + 1] = __uint_as_float(id);
    }
}
extern "C" __global__ void llmx_argmax_merge(const float* parts, unsigned* ids, unsigned n, unsigned nslice, const unsigned* after) {
    const unsigned r = blockIdx.x;
    if (threadIdx.x) return;
    float v = -INFINITY;
    unsigned id = UINT_MAX;
    bool bad = false;
    for (unsigned s = 0; s < nslice; ++s) {
        const float x = parts[2 * (r * nslice + s)];
        bad = bad || isnan(x);
        argmax_pick(v, id, x, __float_as_uint(parts[2 * (r * nslice + s) + 1]));
    }
    ids[r] = argmax_id(v, id, bad, n, after, r);
}
