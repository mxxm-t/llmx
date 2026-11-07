// The dense F32 kernels and the activation twins of the ROCm backend: every sum is F32 in a fixed order that depends on the row alone (docs/src/backends-rocm.md, Kernel notes).
// A block reduction folds lane partials in one tree whatever the wave width, so a row gives the same bits in any call.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "backends/rocm/kernels/quant.h"

namespace {

constexpr int kThreads = 256;
#if defined(__GFX9__)
constexpr int kWave = 64;
#else
constexpr int kWave = 32;
#endif

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
__device__ inline unsigned long long twin_block(const TwinMap& m, unsigned long long fb) {
    const unsigned b = (unsigned)((fb * ((unsigned long long)m.magic_hi << 32 | m.magic_lo)) >> m.shift);
    return twin_block(m, b, (unsigned)fb - b * m.blocks);
}
__device__ inline unsigned long long twin_block(const TwinMap* m, unsigned long long fb) {
    return m ? twin_block(*m, fb) : fb;
}

// Value v at element i of an output, its twin block b, at the class `level` names (1 block-int16, 2 block-int8); the 32 lanes of i's block call it together.
__device__ inline void twin_put(float v, unsigned long long i, unsigned long long b, void* q, float* d, int level) {
    const float amax = block_amax(v);
    if (level == 2) twin_value<signed char, 127>(v, amax, (signed char*)q + b * 32 + i % 32, d + b, (int)(i % 32));
    else twin_value<short, 32767>(v, amax, (short*)q + b * 32 + i % 32, d + b, (int)(i % 32));
}

// Value v at element i of an output whose twin a later product reads, at the class `level` names (0 none); the 32 lanes of i's block call it together.
__device__ inline void twin_at(float v, unsigned long long i, void* q, float* d, int level, const TwinMap* m) {
    if (level) twin_put(v, i, twin_block(m, i / 32), q, d, level);
}
// The same with the map read where `major` is set: a pointer chosen at run time between the map and null would put the map in scratch.
__device__ inline void twin_at(float v, unsigned long long i, void* q, float* d, int level, const TwinMap& m, int major) {
    if (level) twin_put(v, i, major ? twin_block(m, i / 32) : i / 32, q, d, level);
}

// Four consecutive values o of twin block b at element i, its 32 values over 8 aligned lanes of one wave, by twin_value's rule after the max of the lane's four and an 8-lane xor tree, the four codes in one store.
// A max of the same 32 magnitudes is the same float in any grouping, so the codes and the scale are those of one value a lane.
template <typename QT, int L>
__device__ inline void twin_four(const float4 o, unsigned long long i, unsigned long long b, QT* q, float* d) {
    float a = fmaxf(fmaxf(fabsf(o.x), fabsf(o.y)), fmaxf(fabsf(o.z), fabsf(o.w)));
    for (int s = 4; s > 0; s >>= 1) a = fmaxf(a, __shfl_xor(a, s, 8));
    const int lane = (int)(i % 32);
    QT c[4];
    twin_value<QT, L>(o.x, a, &c[0], d + b, lane);
    twin_value<QT, L>(o.y, a, &c[1], d + b, 1);
    twin_value<QT, L>(o.z, a, &c[2], d + b, 1);
    twin_value<QT, L>(o.w, a, &c[3], d + b, 1);
    __builtin_memcpy(q + b * 32 + lane, c, sizeof c);
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

// n bytes from src to dst, in 16-byte words where both and n allow, a grid-stride loop.
extern "C" __global__ void llmx_copy(unsigned char* dst, const unsigned char* src, unsigned long long n) {
    const unsigned long long stride = (unsigned long long)gridDim.x * kThreads;
    unsigned long long i = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    if (!(((unsigned long long)dst | (unsigned long long)src | n) & 15)) {
        for (; i < n / 16; i += stride) ((uint4*)dst)[i] = ((const uint4*)src)[i];
        return;
    }
    for (; i < n; i += stride) dst[i] = src[i];
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
// A limit above 0 takes silu(min(g, limit)) * clamp(u, -limit, limit) in the CPU's order, a NaN passing through both (docs/DEEPSEEK4.md, section 4.2).
template <typename QT, int L>
__device__ void silu_mul(float* dst, const float* gate, const float* up, unsigned long long n, float limit, QT* q, float* d, const TwinMap* m) {
    const unsigned long long i = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    float o = 0.0f;
    if (i < n) {
        float g = gate[i], u = up[i];
        if (limit > 0.0f) {
            u = u < -limit ? -limit : u > limit ? limit : u;
            g = g > limit ? limit : g;
        }
        o = g / (1.0f + expf(-g)) * u;
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
extern "C" __global__ void llmx_silu_mul(float* dst, const float* gate, const float* up, unsigned long long n, float limit, short* q, float* d) {
    silu_mul<short, 0>(dst, gate, up, n, limit, q, d, nullptr);
}
extern "C" __global__ void llmx_silu_mul_twin16(float* dst, const float* gate, const float* up, unsigned long long n, float limit, short* q, float* d) {
    silu_mul<short, 32767>(dst, gate, up, n, limit, q, d, nullptr);
}
extern "C" __global__ void llmx_silu_mul_twin8(float* dst, const float* gate, const float* up, unsigned long long n, float limit, signed char* q, float* d) {
    silu_mul<signed char, 127>(dst, gate, up, n, limit, q, d, nullptr);
}
extern "C" __global__ void llmx_silu_mul_twin16_major(float* dst, const float* gate, const float* up, unsigned long long n, float limit, short* q, float* d, TwinMap m) {
    silu_mul<short, 32767>(dst, gate, up, n, limit, q, d, &m);
}
extern "C" __global__ void llmx_silu_mul_twin8_major(float* dst, const float* gate, const float* up, unsigned long long n, float limit, signed char* q, float* d, TwinMap m) {
    silu_mul<signed char, 127>(dst, gate, up, n, limit, q, d, &m);
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

// The exact products over F32 rows, Q8_0 rows by blocks and rows of a type decoded value by value (kernels/quant.h): an output is 256 strided chains, chain l over units l, l + 256 and so on in order from +0, folded by block_sum's pairs.
// units() and step() are a chain one unit at a time; a type whose `vector` is set loads four weights at once (load4) where the row is aligned to `align` bytes, `weight_bytes` a weight.
struct ExactF32 {
    static constexpr bool vector = true;
    static constexpr int weight_bytes = 4;
    static constexpr unsigned long long align = 16;
    static __device__ float4 load4(const unsigned char* row, unsigned long long u) { return ((const float4*)row)[u / 4]; }
    static __device__ unsigned long long row_bytes(unsigned long long nin) { return nin * sizeof(float); }
    static __device__ unsigned long long units(unsigned long long nin) { return nin; }
    static __device__ float step(const unsigned char* row, const float* x, unsigned long long u, float s) { return fmaf(((const float*)row)[u], x[u], s); }
    template <int O, int R> static __device__ void parts(const unsigned char* const* rows, const float* const* x, unsigned long long nin, float (&s)[O][R]) {
        for (unsigned long long i = threadIdx.x; i < nin; i += kThreads) {
            float w[O], v[R];
#pragma unroll
            for (int o = 0; o < O; ++o) w[o] = ((const float*)rows[o])[i];
#pragma unroll
            for (int r = 0; r < R; ++r) v[r] = x[r][i];
#pragma unroll
            for (int o = 0; o < O; ++o)
#pragma unroll
                for (int r = 0; r < R; ++r) s[o][r] = fmaf(w[o], v[r], s[o][r]);
        }
    }
};
struct ExactQ8 {
    static constexpr bool vector = false;
    static __device__ unsigned long long row_bytes(unsigned long long nin) { return nin / rq::Q8_0::values * rq::Q8_0::bytes; }
    static __device__ unsigned long long units(unsigned long long nin) { return nin / rq::Q8_0::values; }
    static __device__ float step(const unsigned char* row, const float* x, unsigned long long u, float s) {
        return s + q8_0_block_dot(row + u * rq::Q8_0::bytes, x + u * rq::Q8_0::values);
    }
    template <int O, int R> static __device__ void parts(const unsigned char* const* rows, const float* const* x, unsigned long long nin, float (&s)[O][R]) {
        for (unsigned long long k = threadIdx.x; k < nin / rq::Q8_0::values; k += kThreads)
#pragma unroll
            for (int o = 0; o < O; ++o)
#pragma unroll
                for (int r = 0; r < R; ++r) s[o][r] += q8_0_block_dot(rows[o] + k * rq::Q8_0::bytes, x[r] + k * rq::Q8_0::values);
    }
};
template <class T> struct ExactDecoded {
    static constexpr bool vector = __is_same(T, rq::BF16);
    static constexpr int weight_bytes = 2;
    static constexpr unsigned long long align = 8;
    static __device__ float4 load4(const unsigned char* row, unsigned long long u) {
        const uint2 b = ((const uint2*)row)[u / 4];
        return make_float4(__uint_as_float(b.x << 16), __uint_as_float(b.x & 0xffff0000u), __uint_as_float(b.y << 16), __uint_as_float(b.y & 0xffff0000u));
    }
    static __device__ unsigned long long row_bytes(unsigned long long nin) { return nin / T::values * T::bytes; }
    static __device__ unsigned long long units(unsigned long long nin) { return nin; }
    static __device__ float step(const unsigned char* row, const float* x, unsigned long long u, float s) {
        return fmaf(T::value(row + u / T::values * T::bytes, (int)(u % T::values)), x[u], s);
    }
    template <int O, int R> static __device__ void parts(const unsigned char* const* rows, const float* const* x, unsigned long long nin, float (&s)[O][R]) {
        for (unsigned long long i = threadIdx.x; i < nin; i += kThreads) {
            float w[O], v[R];
#pragma unroll
            for (int o = 0; o < O; ++o) w[o] = T::value(rows[o] + i / T::values * T::bytes, (int)(i % T::values));
#pragma unroll
            for (int r = 0; r < R; ++r) v[r] = x[r][i];
#pragma unroll
            for (int o = 0; o < O; ++o)
#pragma unroll
                for (int r = 0; r < R; ++r) s[o][r] = fmaf(w[o], v[r], s[o][r]);
        }
    }
};
// A lane's weight bytes a batch of steps: 128 measured slower than 64 on the 2560 x 512 F32 router, more loads in flight than memory takes (docs/STATUS.md).
constexpr int kBatchBytes = 64;
// U whole steps of a lane's chains from unit u on, every load issued before the first product (held there by a scheduling barrier), so the wave waits on memory once for all of them.
template <class D, int J, int U> __device__ inline void vector_steps(const unsigned char* row, const float* X, unsigned long long u, float (&s)[J]) {
    float4 a[U][J / 4], b[U][J / 4];
#pragma unroll
    for (int q = 0; q < U; ++q)
#pragma unroll
        for (int k = 0; k < J / 4; ++k) a[q][k] = D::load4(row, u + 4 * k + q * kThreads), b[q][k] = ((const float4*)X)[(u + 4 * k + q * kThreads) / 4];
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int q = 0; q < U; ++q)
#pragma unroll
        for (int k = 0; k < J / 4; ++k) {
            s[4 * k] = fmaf(a[q][k].x, b[q][k].x, s[4 * k]);
            s[4 * k + 1] = fmaf(a[q][k].y, b[q][k].y, s[4 * k + 1]);
            s[4 * k + 2] = fmaf(a[q][k].z, b[q][k].z, s[4 * k + 2]);
            s[4 * k + 3] = fmaf(a[q][k].w, b[q][k].w, s[4 * k + 3]);
        }
}
// The whole steps from step m on in batches of U, then the rest in halving batches, each step once and in order.
template <class D, int J, int U>
__device__ inline void vector_run(const unsigned char* row, const float* X, unsigned long long first, unsigned long long full, unsigned long long m, float (&s)[J]) {
    for (; m + U <= full; m += U) vector_steps<D, J, U>(row, X, first + m * kThreads, s);
    if constexpr (U > 1) vector_run<D, J, U / 2>(row, X, first, full, m, s);
}
// One row's outputs, a wave each, wave w of block b taking output b * (256 / the wave) + w: lane t holds chains J t to J t + J - 1 (J = 256 / the wave), each over its units in order.
// The steps every lane fills go in batches with their loads ahead of their products (vector_run); block_sum's pairs then fold them, levels 128 to J by shuffles and those below J in registers.
template <class D>
__device__ inline float exact_sum(const unsigned char* W, const float* X, unsigned long long nin, unsigned long long o) {
    constexpr int J = kThreads / kWave;
    const int t = threadIdx.x % kWave;
    const unsigned char* row = W + o * D::row_bytes(nin);
    const unsigned long long n = D::units(nin), full = n / kThreads, first = (unsigned long long)J * t;
    float s[J];
#pragma unroll
    for (int j = 0; j < J; ++j) s[j] = 0.0f;
    bool done = false;
    if constexpr (D::vector)
        if (((unsigned long long)row % D::align | (unsigned long long)X % 16) == 0) {
            done = true;
            vector_run<D, J, kBatchBytes / (J * D::weight_bytes)>(row, X, first, full, 0, s);
        }
    if (!done) {
        for (unsigned long long m = 0; m < full; ++m)
#pragma unroll
            for (int j = 0; j < J; ++j) s[j] = D::step(row, X, first + j + m * kThreads, s[j]);
    }
#pragma unroll
    for (int j = 0; j < J; ++j)
        if (first + j + full * kThreads < n) s[j] = D::step(row, X, first + j + full * kThreads, s[j]);
#pragma unroll
    for (int h = kThreads / 2; h >= J; h >>= 1)
#pragma unroll
        for (int j = 0; j < J; ++j) s[j] += __shfl_down(s[j], h / J, kWave);
#pragma unroll
    for (int h = J / 2; h > 0; h >>= 1)
#pragma unroll
        for (int j = 0; j < h; ++j) s[j] += s[j + h];
    return s[0];
}
// exact_sum's output written by the wave's first lane.
template <class D>
__device__ inline void exact_row(const unsigned char* W, const float* X, float* Y, unsigned long long nin, unsigned long long o, int add) {
    const float s = exact_sum<D>(W, X, nin, o);
    if (threadIdx.x % kWave == 0) Y[o] = add ? Y[o] + s : s;
}
// vector_steps over C rows of X, nin apart: the weights loaded once for all of them, each row's chains stepped as exact_wave steps its one.
template <class D, int J, int U, int C>
__device__ inline void vector_steps_cols(const unsigned char* row, const float* const (&X)[C], unsigned long long u, float (&s)[C][J]) {
    float4 a[U][J / 4], b[C][U][J / 4];
#pragma unroll
    for (int q = 0; q < U; ++q)
#pragma unroll
        for (int k = 0; k < J / 4; ++k) {
            a[q][k] = D::load4(row, u + 4 * k + q * kThreads);
#pragma unroll
            for (int c = 0; c < C; ++c) b[c][q][k] = ((const float4*)X[c])[(u + 4 * k + q * kThreads) / 4];
        }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int c = 0; c < C; ++c)
#pragma unroll
        for (int q = 0; q < U; ++q)
#pragma unroll
            for (int k = 0; k < J / 4; ++k) {
                s[c][4 * k] = fmaf(a[q][k].x, b[c][q][k].x, s[c][4 * k]);
                s[c][4 * k + 1] = fmaf(a[q][k].y, b[c][q][k].y, s[c][4 * k + 1]);
                s[c][4 * k + 2] = fmaf(a[q][k].z, b[c][q][k].z, s[c][4 * k + 2]);
                s[c][4 * k + 3] = fmaf(a[q][k].w, b[c][q][k].w, s[c][4 * k + 3]);
            }
}
template <class D, int J, int U, int C>
__device__ inline void vector_run_cols(const unsigned char* row, const float* const (&X)[C], unsigned long long first, unsigned long long full, unsigned long long m, float (&s)[C][J]) {
    for (; m + U <= full; m += U) vector_steps_cols<D, J, U, C>(row, X, first + m * kThreads, s);
    if constexpr (U > 1) vector_run_cols<D, J, U / 2, C>(row, X, first, full, m, s);
}
// exact_wave over up to C rows of X at once, a wave an output reading its weight row once: each row's output the bits of exact_wave on that row alone; rows past `nrows` repeat the last and are not written.
template <class D, int C>
__device__ void exact_cols(const unsigned char* W, const float* X, float* Y, unsigned long long nin, unsigned long long nout, unsigned long long nrows, int add) {
    constexpr int J = kThreads / kWave;
    const int t = threadIdx.x % kWave;
    const unsigned long long o = (unsigned long long)blockIdx.x * (kThreads / kWave) + threadIdx.x / kWave;
    if (o >= nout) return;
    const unsigned char* row = W + o * D::row_bytes(nin);
    const unsigned long long n = D::units(nin), full = n / kThreads, first = (unsigned long long)J * t;
    const float* x[C];
    bool aligned = (unsigned long long)row % D::align == 0;
#pragma unroll
    for (int c = 0; c < C; ++c) {
        x[c] = X + (c < (int)nrows ? c : (int)nrows - 1) * nin;
        aligned = aligned && (unsigned long long)x[c] % 16 == 0;
    }
    float s[C][J];
#pragma unroll
    for (int c = 0; c < C; ++c)
#pragma unroll
        for (int j = 0; j < J; ++j) s[c][j] = 0.0f;
    bool done = false;
    if constexpr (D::vector)
        if (aligned) {
            done = true;
            vector_run_cols<D, J, kBatchBytes / (J * D::weight_bytes), C>(row, x, first, full, 0, s);
        }
    if (!done) {
        for (unsigned long long m = 0; m < full; ++m)
#pragma unroll
            for (int c = 0; c < C; ++c)
#pragma unroll
                for (int j = 0; j < J; ++j) s[c][j] = D::step(row, x[c], first + j + m * kThreads, s[c][j]);
    }
#pragma unroll
    for (int c = 0; c < C; ++c)
#pragma unroll
        for (int j = 0; j < J; ++j)
            if (first + j + full * kThreads < n) s[c][j] = D::step(row, x[c], first + j + full * kThreads, s[c][j]);
#pragma unroll
    for (int c = 0; c < C; ++c) {
#pragma unroll
        for (int h = kThreads / 2; h >= J; h >>= 1)
#pragma unroll
            for (int j = 0; j < J; ++j) s[c][j] += __shfl_down(s[c][j], h / J, kWave);
#pragma unroll
        for (int h = J / 2; h > 0; h >>= 1)
#pragma unroll
            for (int j = 0; j < h; ++j) s[c][j] += s[c][j + h];
        if (t == 0 && c < (int)nrows) Y[c * nout + o] = add ? Y[c * nout + o] + s[c][0] : s[c][0];
    }
}
template <class D>
__device__ void exact_wave(const unsigned char* W, const float* X, float* Y, unsigned long long nin, unsigned long long nout, int add) {
    const unsigned long long o = (unsigned long long)blockIdx.x * (kThreads / kWave) + threadIdx.x / kWave;
    if (o < nout) exact_row<D>(W, X, Y, nin, o, add);
}
// kRowsOut outputs by kRowsIn rows a block, each output the bits of exact_wave: every lane keeps its strided chain for each, folded by block_sum's tree.
// Sixteen threads an output take the tree's levels 128 to 16 from shared memory and 8 to 1 by shuffles; rows and outputs past the call repeat the last and are not written.
constexpr int kRowsOut = 8, kRowsIn = 8, kRowsChunk = kThreads / 16;
template <class D>
__device__ void exact_rows(const unsigned char* W, const float* X, float* Y, unsigned long long nin, unsigned long long nout, unsigned long long nrows, int add) {
    __shared__ float lds[kRowsChunk][kThreads + 16];
    const unsigned long long o0 = (unsigned long long)blockIdx.x * kRowsOut, b0 = (unsigned long long)blockIdx.y * kRowsIn;
    const unsigned char* rows[kRowsOut];
    const float* x[kRowsIn];
#pragma unroll
    for (int o = 0; o < kRowsOut; ++o) rows[o] = W + (o0 + o < nout ? o0 + o : nout - 1) * D::row_bytes(nin);
#pragma unroll
    for (int r = 0; r < kRowsIn; ++r) x[r] = X + (b0 + r < nrows ? b0 + r : nrows - 1) * nin;
    float s[kRowsOut][kRowsIn];
#pragma unroll
    for (int o = 0; o < kRowsOut; ++o)
#pragma unroll
        for (int r = 0; r < kRowsIn; ++r) s[o][r] = 0.0f;
    D::template parts<kRowsOut, kRowsIn>(rows, x, nin, s);
    const int t = threadIdx.x, v = t / 16, j = t % 16;
#pragma unroll
    for (int c0 = 0; c0 < kRowsOut * kRowsIn; c0 += kRowsChunk) {
#pragma unroll
        for (int c = 0; c < kRowsChunk; ++c) lds[c][t] = s[(c0 + c) / kRowsIn][(c0 + c) % kRowsIn];
        __syncthreads();
        // Lane l's level-64 sum is (p[l] + p[l + 128]) + (p[l + 64] + p[l + 192]); levels 32 and 16 pair lanes j, j + 16, j + 32 and j + 48, which this thread holds.
        const float* p = lds[v];
        float q[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int l = j + 16 * k;
            q[k] = (p[l] + p[l + 128]) + (p[l + 64] + p[l + 192]);
        }
        float a = (q[0] + q[2]) + (q[1] + q[3]);
#pragma unroll
        for (int h = 8; h > 0; h >>= 1) a += __shfl_down(a, h, 16);
        if (j == 0) {
            const unsigned long long o = o0 + (c0 + v) / kRowsIn, b = b0 + (c0 + v) % kRowsIn;
            if (o < nout && b < nrows) {
                float* y = Y + b * nout + o;
                *y = add ? *y + a : a;
            }
        }
        __syncthreads();
    }
}
// A routed exact call (backend.hpp matmul_experts): the entries' ids, the launch's first entry, the entries an X row serves, the experts and an expert's bytes.
struct RoutedExact {
    const unsigned* ids;
    unsigned long long first, per, n_expert, expert_bytes;
};
// Entry e = first + blockIdx.y: its expert's rows against X row e / per into Y + e * nout through the one-row build, a wave an output; an entry without an expert writes nothing.
template <class D> __device__ void routed_exact(const unsigned char* W, const float* X, float* Y, unsigned long long nin, unsigned long long nout, RoutedExact r) {
    const unsigned long long e = r.first + blockIdx.y;
    const unsigned id = r.ids[e];
    if (id >= r.n_expert) return;
    exact_wave<D>(W + id * r.expert_bytes, X + e / r.per * nin, Y + e * nout, nin, nout, 0);
}
#define LLMX_EXACT(t, ...)                                                                                                                                         \
    extern "C" __global__ void __launch_bounds__(kThreads) llmx_matmul_##t(const unsigned char* W, const float* X, float* Y, unsigned long long nin, unsigned long long nout, int add) { \
        exact_wave<__VA_ARGS__>(W, X, Y, nin, nout, add);                                        \
    }                                                                                                                                                              \
    extern "C" __global__ void __launch_bounds__(kThreads) llmx_routed_##t(const unsigned char* W, const float* X, float* Y, unsigned long long nin, unsigned long long nout, RoutedExact r) { \
        routed_exact<__VA_ARGS__>(W, X, Y, nin, nout, r);                                                                                                       \
    }                                                                                                                                                              \
    extern "C" __global__ void __launch_bounds__(kThreads) llmx_matmul_rows_##t(const unsigned char* W, const float* X, float* Y, unsigned long long nin, unsigned long long nout,          \
                                                    unsigned long long nrows, int add) {                                                                          \
        exact_rows<__VA_ARGS__>(W, X, Y, nin, nout, nrows, add);                                                                                                \
    }
LLMX_EXACT(f32, ExactF32)
LLMX_EXACT(q8_0, ExactQ8)
#define LLMX_COLS(t, C, D)     extern "C" __global__ void __launch_bounds__(kThreads) llmx_matmul_cols##C##_##t(const unsigned char* W, const float* X, float* Y, unsigned long long nin, unsigned long long nout,                                                     unsigned long long nrows, int add) {                                                                                  exact_cols<D, C>(W, X, Y, nin, nout, nrows, add);                                                                                                           }
LLMX_COLS(f32, 2, ExactF32)
LLMX_COLS(f32, 4, ExactF32)

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
LLMX_DECODED(mxfp4, MXFP4)

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

// Sparse attention_vec over G heads of one KV head, a block a (row, G heads, split), each selected token's keys and values loaded once for the G heads.
// Each head keeps attention_vec's streams, chains and merge, so its output has the bits of attention_vec<DIM, true>.
template <int DIM, int G>
__device__ inline void sparse_vec_grouped(const float* Q, float* O, const void* kc, const void* vc, const int* table, const int* firsts, const int* ends, int n_head,
                                          int heads_kv, int bt, int k_half, int v_half, float scale, int nsplit, int chunk, float* partials, const unsigned* ids,
                                          int stride) {
    constexpr int LANES = DIM / 8, STREAMS = kThreads / LANES;
    __shared__ float ms[G][STREAMS], ls[G][STREAMS];
    __shared__ float accs[STREAMS * DIM];
    const int split = blockIdx.x % nsplit, rg = blockIdx.x / nsplit, r = rg / (n_head / G), h0 = rg % (n_head / G) * G, kvh = h0 / (n_head / heads_kv);
    const int len = ends[r];
    if (len == 0) return;
    const int lid = threadIdx.x, stream = lid / LANES, d0 = (lid % LANES) * 8;
    const int* blocks = table + firsts[r];
    float qv[G][8], acc[G][8], m[G], l[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const unsigned long long qrow = ((unsigned long long)r * n_head + h0 + g) * DIM;
        for (int e = 0; e < 8; ++e) {
            qv[g][e] = Q[qrow + d0 + e];
            acc[g][e] = 0.0f;
        }
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }
    const int start = min(split * chunk, stride), end = start + min(chunk, stride - start);
    for (int slot = start + stream; slot < end; slot += STREAMS) {
        const unsigned id = ids[(unsigned long long)r * stride + slot];
        if (id >= (unsigned)len) continue;
        const int t = (int)id;
        const unsigned long long at = (((unsigned long long)blocks[t / bt] * heads_kv + kvh) * bt + t % bt) * DIM + d0;
        float k8[8], v8[8];
        load8(kc, at, k_half, k8);
        load8(vc, at, v_half, v8);
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float part = 0.0f;
            for (int e = 0; e < 8; ++e) part = fmaf(qv[g][e], k8[e], part);
            for (int o = LANES / 2; o > 0; o /= 2) part += __shfl_xor(part, o, LANES);
            const float sc = part * scale, mn = fmaxf(m[g], sc);
            const float corr = expf(m[g] - mn), w = expf(sc - mn);
            l[g] = fmaf(l[g], corr, w);
            for (int e = 0; e < 8; ++e) acc[g][e] = fmaf(w, v8[e], acc[g][e] * corr);
            m[g] = mn;
        }
    }
    if (lid == stream * LANES)
#pragma unroll
        for (int g = 0; g < G; ++g) {
            ms[g][stream] = m[g];
            ls[g][stream] = l[g];
        }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        for (int e = 0; e < 8; ++e) accs[stream * DIM + d0 + e] = acc[g][e];
        __syncthreads();
        if (lid < DIM) {
            const unsigned long long rh = (unsigned long long)r * n_head + h0 + g;
            float M = ms[g][0];
            for (int i = 1; i < STREAMS; ++i) M = fmaxf(M, ms[g][i]);
            float L = 0.0f, A = 0.0f;
            for (int i = 0; i < STREAMS; ++i) {
                if (isinf(ms[g][i])) continue;
                const float c = expf(ms[g][i] - M);
                L = fmaf(ls[g][i], c, L);
                A = fmaf(accs[i * DIM + lid], c, A);
            }
            if (nsplit == 1) {
                O[rh * DIM + lid] = !(L > 0.0f) ? 0.0f : A / L;
            } else {
                float* out = partials + (rh * nsplit + split) * (DIM + 2);
                out[2 + lid] = A;
                if (lid == 0) {
                    out[0] = M;
                    out[1] = L;
                }
            }
        }
        __syncthreads();
    }
}
#define LLMX_SPARSE_VEC_GROUPED(name, D, G) \
    extern "C" __global__ void __launch_bounds__(kThreads) name(const float* Q, float* O, const void* kc, const void* vc, const int* table, const int* firsts, \
                                                                const int* ends, int n_head, int heads_kv, int bt, int k_half, int v_half, float scale, int nsplit, \
                                                                int chunk, float* partials, const unsigned* ids, int stride) { \
        sparse_vec_grouped<D, G>(Q, O, kc, vc, table, firsts, ends, n_head, heads_kv, bt, k_half, v_half, scale, nsplit, chunk, partials, ids, stride); \
    }
LLMX_SPARSE_VEC_GROUPED(llmx_sparse_attention_vec_g4, 128, 4)
LLMX_SPARSE_VEC_GROUPED(llmx_sparse_attention_vec256_g4, 256, 4)

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
// Each of the stream and gate ops below also writes its output's twin at the class tlevel names, laid out by tm where major is set (twin_at); the host asks for it only for outputs of whole blocks a row, so a block's 32 lanes run together.
extern "C" __global__ void llmx_sigmoid_mul(float* dst, const float* x, const float* gate, unsigned long long n, unsigned long long width,
                                            unsigned long long dim, unsigned long long gs, unsigned long long ghs, void* tq, float* td, int tlevel, TwinMap tm, int major) {
    const unsigned long long i = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    const unsigned long long r = i / width, rem = i - r * width, h = rem / dim, d = rem - h * dim;
    const float v = x[i] * (1.0f / (1.0f + expf(-gate[r * gs + h * ghs + d])));
    dst[i] = v;
    twin_at(v, i, tq, td, tlevel, tm, major);
}

// A block a (row, head): x's RMS norm against w, times SiLU or, with `sig`, the sigmoid of z; dst may be x, since every value is read into the sum before the block's barrier.
extern "C" __global__ void llmx_gated_rms_norm(float* dst, const float* x, const float* z, const float* w, int dim, float eps, int sig, void* tq, float* td, int tlevel, TwinMap tm, int major) {
    __shared__ float lds[kThreads];
    const unsigned long long o = (unsigned long long)blockIdx.x * dim;
    float s = 0.0f;
    for (int i = threadIdx.x; i < dim; i += kThreads) s = fmaf(x[o + i], x[o + i], s);
    const float r = 1.0f / sqrtf(block_sum(s, lds) / (float)dim + eps);
    for (int i = threadIdx.x; i < dim; i += kThreads) {
        const float zz = z[o + i], g = sig ? 1.0f / (1.0f + expf(-zz)) : zz / (1.0f + expf(-zz));
        const float v = x[o + i] * (r * w[i]) * g;
        dst[o + i] = v;
        twin_at(v, o + i, tq, td, tlevel, tm, major);
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
// A sequence gives the same bits alone, batched or cut into passes, and with kTb tokens staged a barrier in either build (docs/src/backends-rocm.md); a decay below 2^-126 is 0.
template <unsigned kTb>
__device__ void gated_delta_rule(float* o, const float* u, const float* alpha, const float* bb, const float* a, const float* dt_bias, float* st,
                                 const unsigned* tab, unsigned C, unsigned k_heads, unsigned v_heads, unsigned k_dim, unsigned v_dim,
                                 unsigned long long slot, unsigned blocks, float scale, float eps) {
    constexpr unsigned kRows = 16;
    static_assert(kTb >= 1 && kTb <= 16, "a block of 256 threads stages at most 16 tokens' q and k rows");
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
// The decode build stages 4 tokens, for calls whose every view holds at most 8 rows, as Vulkan's short build does; the prompt build stages 16.
// Both are bounded to 256 threads and two waves a SIMD, so a lane keeps its state rows in registers rather than scratch (docs/src/backends-rocm.md).
#define LLMX_DELTA_RULE(name, tb)                                                                                                                                       \
    extern "C" __global__ void __launch_bounds__(kThreads, 2) name(float* o, const float* u, const float* alpha, const float* bb, const float* a, const float* dt_bias, float* st, const unsigned* tab, \
                                    unsigned C, unsigned k_heads, unsigned v_heads, unsigned k_dim, unsigned v_dim, unsigned long long slot, unsigned blocks,            \
                                    float scale, float eps) {                                                                                                           \
        gated_delta_rule<tb>(o, u, alpha, bb, a, dt_bias, st, tab, C, k_heads, v_heads, k_dim, v_dim, slot, blocks, scale, eps);                                       \
    }
LLMX_DELTA_RULE(llmx_gated_delta_rule, 16)
LLMX_DELTA_RULE(llmx_gated_delta_rule_short, 4)

// The Flash-Next stream ops (backend.hpp), each at its Vulkan kernel's op boundary and in its order, every value a function of its own row.

// A block a (row, group) and slice of rows: the group's RMS norm against its own weights, each thread's squares in order, then block_sum's tree.
// Every slice repeats the whole reduction, so a group's bits do not depend on its slices; with `four` (dst, src and w on 16 bytes, dim a multiple of 4) a thread scales four values a chunk, one load and store each, their twin by twin_four.
extern "C" __global__ void llmx_group_rms_norm(float* dst, const float* src, const float* w, unsigned groups, unsigned dim, float eps, void* tq, float* td, int tlevel, TwinMap tm, int major,
                                               int four) {
    __shared__ float lds[kThreads];
    const unsigned long long o = (unsigned long long)blockIdx.x * dim;
    const float* wg = w + (unsigned long long)(blockIdx.x % groups) * dim;
    float s = 0.0f;
    for (unsigned i = threadIdx.x; i < dim; i += kThreads) s = fmaf(src[o + i], src[o + i], s);
    const float r = 1.0f / sqrtf(block_sum(s, lds) / (float)dim + eps);
    if (four) {
        for (unsigned c = blockIdx.y; c * 1024u < dim; c += gridDim.y) {
            const unsigned i = c * 1024u + 4u * threadIdx.x;
            // A block's 8 lanes are all in or all out of the row, since dim is a multiple of 32 wherever a twin is asked for.
            if (i >= dim) continue;
            const float4 x = *(const float4*)(src + o + i), g = *(const float4*)(wg + i);
            const float4 v = make_float4(x.x * (r * g.x), x.y * (r * g.y), x.z * (r * g.z), x.w * (r * g.w));
            *(float4*)(dst + o + i) = v;
            if (tlevel) {
                const unsigned long long b = major ? twin_block(tm, (o + i) / 32) : (o + i) / 32;
                if (tlevel == 2) twin_four<signed char, 127>(v, o + i, b, (signed char*)tq, td);
                else twin_four<short, 32767>(v, o + i, b, (short*)tq, td);
            }
        }
        return;
    }
    for (unsigned c = blockIdx.y; c * 1024u < dim; c += gridDim.y)
        for (unsigned i = c * 1024u + threadIdx.x; i < dim && i < (c + 1u) * 1024u; i += kThreads) {
            const float v = src[o + i] * (r * wg[i]);
            dst[o + i] = v;
            twin_at(v, o + i, tq, td, tlevel, tm, major);
        }
}

// dst[i] = silu(src[i] * scale); dst may be src.
extern "C" __global__ void llmx_silu_scaled(float* dst, const float* src, unsigned long long n, float scale, void* tq, float* td, int tlevel, TwinMap tm, int major) {
    const unsigned long long i = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    const float v = src[i] * scale, y = v / (1.0f + expf(-v));
    dst[i] = y;
    twin_at(y, i, tq, td, tlevel, tm, major);
}

// dst[r][d] = the sum over streams s in order of src[r][s][d] * sigmoid(gate[r][s][d]), divided by the streams.
extern "C" __global__ void llmx_mix_streams(float* dst, const float* src, const float* gate, unsigned long long n, unsigned streams, unsigned dim, void* tq, float* td, int tlevel, TwinMap tm, int major) {
    const unsigned long long i = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    const unsigned long long r = i / dim, d = i - r * dim;
    float sum = 0.0f;
    for (unsigned s = 0; s < streams; ++s) {
        const unsigned long long j = (r * streams + s) * dim + d;
        sum = fmaf(src[j], 1.0f / (1.0f + expf(-gate[j])), sum);
    }
    const float v = sum / (float)streams;
    dst[i] = v;
    twin_at(v, i, tq, td, tlevel, tm, major);
}

// dst[r][s][d] += block[r][d] * 2 sigmoid(gate[r][s] / streams), the residual streams in place; block b takes row and stream b / chunks, so blocks walk memory in order and no index divides 64 bits.
extern "C" __global__ void llmx_inject_streams(float* dst, const float* block, const float* gate, unsigned streams, unsigned dim) {
    const unsigned chunks = (dim + kThreads - 1) / kThreads, row = blockIdx.x / chunks, d = (blockIdx.x - row * chunks) * kThreads + threadIdx.x;
    if (d >= dim) return;
    const unsigned long long rs = row, r = row / streams, s = rs - r * streams, i = rs * dim + d;
    const float weight = 2.0f * (1.0f / (1.0f + expf(-(gate[r * streams + s] / (float)streams))));
    dst[i] = fmaf(block[r * dim + d], weight, dst[i]);
}

// A block a (row, group): g = dot(key, query) * scale over the group in block_sum's order, and the row's values times sigmoid(sign(g) sqrt(max(|g|, 1e-6))), sign(0) being 0, or with `kept` the zero's own sign.
// A NaN score is kept as the CPU keeps it.
extern "C" __global__ void llmx_signed_sqrt_gate(float* dst, const float* key, const float* query, const float* value, unsigned groups, unsigned dim, float scale, unsigned kept) {
    __shared__ float lds[kThreads];
    const unsigned long long o = (unsigned long long)blockIdx.x * dim, v0 = (unsigned long long)(blockIdx.x / groups) * dim;
    float s = 0.0f;
    for (unsigned i = threadIdx.x; i < dim; i += kThreads) s = fmaf(key[o + i], query[o + i], s);
    const float g = block_sum(s, lds) * scale;
    const float sign = g > 0.0f ? 1.0f : g < 0.0f ? -1.0f : g == 0.0f ? (kept ? copysignf(1.0f, g) : 0.0f) : g, magnitude = fabsf(g) < 1e-6f ? 1e-6f : fabsf(g);
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
// Each reduction is the 64-pair tree, lane t adding lane t + s at level s, so a row's picks and weights do not depend on the wave size (docs/src/backends-rocm.md).
constexpr unsigned kRouteThreads = 64, kRouteExperts = 1024, kNoExpert = 0xffffffffu;
template <typename T, typename Op>
__device__ inline T route_tree(T v, T* red, Op op) {
    const unsigned t = threadIdx.x;
    for (unsigned s = kRouteThreads / 2; s; s /= 2) {
        if (s >= (unsigned)warpSize) {
            red[t] = v;
            __syncthreads();
            v = op(v, red[t ^ s]);
            __syncthreads();
        } else {
            v = op(v, __shfl_xor(v, (int)s, warpSize));
        }
    }
    return v;
}
extern "C" __global__ void __launch_bounds__(kRouteThreads) llmx_moe_route(const float* scores, unsigned* ids, float* weights, unsigned n_expert, unsigned k, int normalize) {
    __shared__ float p[kRouteExperts], red[kRouteThreads], chosen[256];
    __shared__ unsigned redi[kRouteThreads], chosen_id[256];
    const unsigned t = threadIdx.x;
    const unsigned long long r = blockIdx.x;
    const float* s = scores + r * n_expert;
    const auto larger = [](float a, float b) { return fmaxf(a, b); };
    float top = -INFINITY;
    for (unsigned e = t; e < n_expert; e += kRouteThreads) {
        p[e] = s[e];
        top = fmaxf(top, p[e]);
    }
    top = route_tree(top, red, larger);
    float part = 0.0f;
    for (unsigned e = t; e < n_expert; e += kRouteThreads) {
        const float v = expf(p[e] - top);
        p[e] = v;
        part += v;
    }
    // Both partners of a level add the same two values, so every lane holds the tree's lane-0 sum.
    const float sum = route_tree(part, red, [](float a, float b) { return a + b; });
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
    float total = 0.0f;
    for (unsigned j = 0; j < k; ++j) {
        // The largest untaken probability, then the lowest id holding it; a taken expert holds -1, below every probability.
        float bv = -2.0f;
        unsigned bi = kNoExpert;
        for (unsigned e = t; e < n_expert; e += kRouteThreads)
            if (p[e] > bv) bv = p[e], bi = e;
        const float m = route_tree(bv, red, larger);
        const unsigned id = route_tree(bv == m ? bi : kNoExpert, redi, [](unsigned a, unsigned b) { return min(a, b); });
        const float c = id < n_expert ? m : 0.0f;
        total += c;
        if (id < n_expert && id % kRouteThreads == t) p[id] = -1.0f;
        if (j % kRouteThreads == t) {
            chosen[j] = c;
            chosen_id[j] = id;
        }
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

// The fused decode moe_step's first launch (rocm_backend.cpp moe_step): the router's and row gate's F32 rows and the int8 twin of h, then the last block out routes as llmx_moe_route does on a 64-lane wave.
#include <cstddef>

namespace {

// v of lane (this lane xor m) of a 64-lane wave: DPP for 1 and 2, swizzles up to 16; the partner route_tree takes, so its sums keep their bits.
__device__ inline unsigned rm_xlane(unsigned v, int m) {
    if (m == 1) return __builtin_amdgcn_mov_dpp(v, 0xB1, 0xF, 0xF, false);
    if (m == 2) return __builtin_amdgcn_mov_dpp(v, 0x4E, 0xF, 0xF, false);
    if (m == 4) return __builtin_amdgcn_ds_swizzle(v, 0x1F | (4 << 10));
    if (m == 8) return __builtin_amdgcn_ds_swizzle(v, 0x1F | (8 << 10));
    if (m == 16) return __builtin_amdgcn_ds_swizzle(v, 0x1F | (16 << 10));
    return __shfl_xor(v, m, 64);
}
template <typename Op>
__device__ inline float rm_tree(float v, Op op) {
#pragma unroll
    for (int m = 32; m > 0; m >>= 1) v = op(v, __uint_as_float(rm_xlane(__float_as_uint(v), m)));
    return v;
}

// The most experts mf_route takes, J = n_expert / 64 of them a lane (rocm_backend.cpp kStepExperts).
constexpr unsigned kStepExperts = 512;

// llmx_moe_route's picks and weights for one row of 64 J experts and k picks on one wave, a pick the largest key of probability bits over the id's complement; the scores are read past this compute unit's cache.
template <unsigned J>
__device__ inline void mf_route(const float* s, unsigned* ids, float* weights, unsigned k, int normalize) {
    const unsigned t = threadIdx.x % 64;
    float p[J];
    float top = -INFINITY;
#pragma unroll
    for (unsigned j = 0; j < J; ++j) p[j] = __hip_atomic_load(s + t + j * 64, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#pragma unroll
    for (unsigned j = 0; j < J; ++j) top = fmaxf(top, p[j]);
    top = rm_tree(top, [](float a, float b) { return fmaxf(a, b); });
    float part = 0.0f;
#pragma unroll
    for (unsigned j = 0; j < J; ++j) {
        const float v = expf(p[j] - top);
        p[j] = v;
        part += v;
    }
    const float sum = rm_tree(part, [](float a, float b) { return a + b; });
    if (isnan(sum)) {
        if (t < k) ids[t] = kNoExpert, weights[t] = 0.0f;
        return;
    }
    unsigned long long key[J];
#pragma unroll
    for (unsigned j = 0; j < J; ++j) {
        const float v = p[j] / sum;
        const float c = v < __uint_as_float(0x00800000u) ? 0.0f : v;
        key[j] = ((unsigned long long)__float_as_uint(c) << 32) | (0xffffffffu - (t + j * 64));
    }
#pragma unroll
    for (unsigned a = 0; a < J; ++a)
#pragma unroll
        for (unsigned b = 0; b + 1 < J - a; ++b)
            if (key[b] < key[b + 1]) { const unsigned long long x = key[b]; key[b] = key[b + 1]; key[b + 1] = x; }
    float total = 0.0f, cw = 0.0f;
    unsigned cid = kNoExpert;
    for (unsigned q = 0; q < k; ++q) {
        unsigned long long m = key[0];
#pragma unroll
        for (int x = 32; x > 0; x >>= 1) {
            const unsigned lo = rm_xlane((unsigned)m, x), hi = rm_xlane((unsigned)(m >> 32), x);
            const unsigned long long o = ((unsigned long long)hi << 32) | lo;
            m = o > m ? o : m;
        }
        const float c = __uint_as_float((unsigned)(m >> 32));
        total += c;
        if (key[0] == m) {
#pragma unroll
            for (unsigned j = 0; j + 1 < J; ++j) key[j] = key[j + 1];
            key[J - 1] = 0;
        }
        if (q == t) cw = c, cid = 0xffffffffu - (unsigned)m;
    }
    if (t < k) {
        ids[t] = cid;
        weights[t] = normalize ? cw / total : cw;
    }
}

struct MfRoute {
    const unsigned char* w;
    const float* x;
    float* scores;
    unsigned long long nin, nout;
    const unsigned char* gw;
    float* rg;
    signed char* hq;
    float* hd;
    unsigned long long hblocks;
    unsigned* ids;
    float* weights;
    unsigned n_expert, k;
    int normalize;
    unsigned* counter;
    unsigned rows, pad;
};
static_assert(offsetof(MfRoute, gw) == 40 && offsetof(MfRoute, ids) == 80 && offsetof(MfRoute, n_expert) == 96 && offsetof(MfRoute, counter) == 112 && offsetof(MfRoute, rows) == 120 && sizeof(MfRoute) == 128,
              "host packs MfRoute by these offsets");

} // namespace

// Blocks 0 to nout / 4 - 1 four router rows each, the next the row gate; the first blocks also the twin of h, 8 twin blocks each; the last block out routes.
extern "C" __global__ void __launch_bounds__(256) llmx_moe_step_route(MfRoute a) {
    __shared__ unsigned last;
    const unsigned b = blockIdx.x, w = threadIdx.x / 64, rb = (unsigned)(a.nout / 4);
    // Each row of h on its own, in the one-row step's order, so a row keeps its bits at any row count.
    for (unsigned r = 0; r < a.rows; ++r)
        if (b < rb) exact_row<ExactF32>(a.w, a.x + r * a.nin, a.scores + r * a.nout, a.nin, 4ull * b + w, 0);
    if (b >= rb)
        for (unsigned r = w; r < a.rows; r += 4) exact_row<ExactF32>(a.gw, a.x + r * a.nin, a.rg + r, a.nin, 0, 0);
    for (unsigned long long tb = (unsigned long long)b * 8 + threadIdx.x / 32; tb < a.hblocks; tb += (unsigned long long)gridDim.x * 8) {
        const int lane = threadIdx.x % 32;
        const float v = a.x[tb * 32 + lane];
        twin_value<signed char, 127>(v, block_amax(v), a.hq + tb * 32 + lane, a.hd + tb, lane);
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) last = atomicAdd(a.counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (threadIdx.x == 0) *a.counter = 0;
    static_assert(kStepExperts == 8 * 64, "mf_route's builds below take 1 to 8 keys a lane");
    for (unsigned r = w; r < a.rows; r += 4) {
        const float* s = a.scores + r * a.nout;
        unsigned* ids = a.ids + r * a.k;
        float* wt = a.weights + r * a.k;
        switch (a.n_expert / 64) {
        case 1: mf_route<1>(s, ids, wt, a.k, a.normalize); break;
        case 2: mf_route<2>(s, ids, wt, a.k, a.normalize); break;
        case 3: mf_route<3>(s, ids, wt, a.k, a.normalize); break;
        case 4: mf_route<4>(s, ids, wt, a.k, a.normalize); break;
        case 5: mf_route<5>(s, ids, wt, a.k, a.normalize); break;
        case 6: mf_route<6>(s, ids, wt, a.k, a.normalize); break;
        case 7: mf_route<7>(s, ids, wt, a.k, a.normalize); break;
        default: mf_route<8>(s, ids, wt, a.k, a.normalize); break;
        }
    }
}

// DeepSeek-V4.1's layer ops (docs/DEEPSEEK4.md, section 4.2) as first forms, in the CPU backend's order where one value is formed, so a value without an exponential is the CPU's bits.

// The sum of squares of n floats as the CPU's rms_norm_raw forms it: eight lanes of multiply-adds over whole groups of eight, their pairs added as two hadds leave them, then the tail's multiply-adds.
// Every thread of the block receives the sum.
__device__ float cpu_squares(const float* x, unsigned n, float* lds) {
    const unsigned t = threadIdx.x, whole = n / 8u * 8u;
    if (t < 8u) {
        float a = 0.0f;
        for (unsigned i = t; i < whole; i += 8u) a = fmaf(x[i], x[i], a);
        lds[t] = a;
    }
    __syncthreads();
    if (t == 0) {
        float s = (lds[0] + lds[1]) + (lds[2] + lds[3]);
        s += (lds[4] + lds[5]) + (lds[6] + lds[7]);
        for (unsigned i = whole; i < n; ++i) s = fmaf(x[i], x[i], s);
        lds[8] = s;
    }
    __syncthreads();
    const float s = lds[8];
    __syncthreads();
    return s;
}
__device__ inline float sigmoid_of(float v) { return 1.0f / (1.0f + expf(-v)); }

// A block a row: pre, post and comb from the row's (2 + hc) * hc mixes, the stream's squares summed by block_sum and the rest on one thread in the CPU's order; hc is at most 16.
extern "C" __global__ void llmx_hc_coeffs(float* pre, float* post, float* comb, const float* mixes, const float* x, const float* scale, const float* base, unsigned hc,
                                          unsigned dim, float norm_eps, float eps, unsigned iters) {
    __shared__ float lds[kThreads], c[256];
    const unsigned long long r = blockIdx.x;
    const unsigned width = hc * dim, mix = (hc + 2u) * hc;
    const float* xr = x + r * width;
    float sq = 0.0f;
    for (unsigned i = threadIdx.x; i < width; i += kThreads) sq = fmaf(xr[i], xr[i], sq);
    const float rs = 1.0f / sqrtf(block_sum(sq, lds) / (float)width + norm_eps);
    if (threadIdx.x) return;
    const float* m = mixes + r * mix;
    for (unsigned k = 0; k < hc; ++k) {
        pre[r * hc + k] = sigmoid_of(fmaf(m[k] * rs, scale[0], base[k])) + eps;
        post[r * hc + k] = 2.0f * sigmoid_of(fmaf(m[hc + k] * rs, scale[1], base[hc + k]));
    }
    for (unsigned i = 0; i < hc; ++i) {
        float peak = -INFINITY;
        for (unsigned j = 0; j < hc; ++j) {
            c[i * hc + j] = fmaf(m[2u * hc + i * hc + j] * rs, scale[2], base[2u * hc + i * hc + j]);
            peak = fmaxf(peak, c[i * hc + j]);
        }
        float sum = 0.0f;
        for (unsigned j = 0; j < hc; ++j) sum += c[i * hc + j] = expf(c[i * hc + j] - peak);
        for (unsigned j = 0; j < hc; ++j) c[i * hc + j] = c[i * hc + j] / sum + eps;
    }
    for (unsigned round = 0; round < iters; ++round) {
        if (round)
            for (unsigned i = 0; i < hc; ++i) {
                float sum = 0.0f;
                for (unsigned j = 0; j < hc; ++j) sum += c[i * hc + j];
                for (unsigned j = 0; j < hc; ++j) c[i * hc + j] /= sum + eps;
            }
        for (unsigned j = 0; j < hc; ++j) {
            float sum = 0.0f;
            for (unsigned i = 0; i < hc; ++i) sum += c[i * hc + j];
            for (unsigned i = 0; i < hc; ++i) c[i * hc + j] /= sum + eps;
        }
    }
    for (unsigned i = 0; i < hc * hc; ++i) comb[r * hc * hc + i] = c[i];
}

// A block a row: u = sum over k in order of pre[k] X_k, a thread a channel, then its RMS norm with the CPU's sums and products, so the row is the CPU's bits.
extern "C" __global__ void llmx_hc_collapse_norm(float* dst, const float* x, const float* pre, const float* w, unsigned hc, unsigned dim, float eps) {
    __shared__ float lds[16];
    const unsigned long long r = blockIdx.x;
    const float* xr = x + r * hc * dim;
    const float* p = pre + r * hc;
    float* u = dst + r * dim;
    for (unsigned d = threadIdx.x; d < dim; d += kThreads) {
        float sum = p[0] * xr[d];
        for (unsigned k = 1; k < hc; ++k) sum = fmaf(p[k], xr[(unsigned long long)k * dim + d], sum);
        u[d] = sum;
    }
    __syncthreads();
    const float rs = 1.0f / sqrtf(cpu_squares(u, dim, lds) / (float)dim + eps);
    for (unsigned d = threadIdx.x; d < dim; d += kThreads) u[d] = u[d] * (rs * w[d]);
}

// A thread a (row, channel): X'_k = post[k] f + sum over j in order of comb[j][k] X_j with the CPU's multiply-adds, every old stream read before a new one is written, so dst may be the stream.
extern "C" __global__ void llmx_hc_expand(float* dst, const float* f, const float* x, const float* post, const float* comb, unsigned hc, unsigned dim, unsigned long long n) {
    const unsigned long long i = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    const unsigned long long r = i / dim, d = i - r * dim, width = (unsigned long long)hc * dim;
    const float* c = comb + r * hc * hc;
    float old[16], out[16];
    for (unsigned j = 0; j < hc; ++j) old[j] = x[r * width + j * dim + d];
    const float fv = f[r * dim + d];
    for (unsigned k = 0; k < hc; ++k) {
        float sum = c[k] * old[0];
        for (unsigned j = 1; j < hc; ++j) sum = fmaf(c[j * hc + k], old[j], sum);
        out[k] = fmaf(post[r * hc + k], fv, sum);
    }
    for (unsigned k = 0; k < hc; ++k) dst[r * width + k * dim + d] = out[k];
}

// The storage formats of quant/model_formats.hpp, a thread a block, each step the CPU's: exact comparisons for the codes, correctly rounded quotients and one multiply of code and scale.
__device__ inline int format_ceil_log2(float v) {
    const unsigned u = __float_as_uint(v);
    return (int)((u >> 23) & 0xFFu) - 127 + ((u & 0x7FFFFFu) != 0u);
}
__device__ inline float format_pow2(int e) { return __uint_as_float((unsigned)(e + 127) << 23); }
__device__ inline float e4m3_value(unsigned c) {
    const unsigned e = (c >> 3) & 15u, m = c & 7u;
    const float v = e == 15u && m == 7u ? NAN : e ? ldexpf((float)(8u + m), (int)e - 10) : ldexpf((float)m, -9);
    return c & 0x80u ? -v : v;
}
__device__ inline unsigned e4m3_code(float v) {
    const unsigned sign = signbit(v) ? 0x80u : 0u;
    if (isnan(v)) return 0x7Fu;
    const float a = fabsf(v);
    if (a >= 464.0f) return sign | 0x7Eu;
    if (a < 0.015625f) return sign | (unsigned)rintf(a * 512.0f);
    int e;
    frexpf(a, &e);
    const unsigned t = (unsigned)rintf(ldexpf(a, 4 - e));
    return sign | (((unsigned)(e + 6) << 3) + (t - 8u));
}
__device__ inline float e2m1_value(unsigned c) {
    const float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float v = magnitudes[c & 7u];
    return c & 8u ? -v : v;
}
__device__ inline unsigned e2m1_code(float v) {
    const float a = fabsf(v);
    const unsigned m = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.0f);
    return (signbit(v) ? 8u : 0u) | m;
}
__device__ inline float format_clamp(float v, float limit) { return v < -limit ? -limit : (v > limit ? limit : v); }
// Format 0 is fp8_power, 1 fp4_power and 2 fp4_e4m3; the inverse maxima come from the host as the CPU rounds them.
extern "C" __global__ void llmx_model_format_qdq(float* x, unsigned long long blocks, unsigned format, float inv_fp8, float inv_fp4) {
    const unsigned long long blk = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    if (blk >= blocks) return;
    const unsigned n = format == 2u ? 16u : 32u;
    float* b = x + blk * n;
    float amax = 0.0f;
    for (unsigned i = 0; i < n; ++i) amax = fmaxf(amax, fabsf(b[i]));
    const float s = format == 0u ? format_pow2(format_ceil_log2(fmaxf(amax, 1e-4f) * inv_fp8))
                  : format == 1u ? format_pow2(format_ceil_log2(fmaxf(amax, 6.0f * 0x1p-126f) * inv_fp4))
                                 : e4m3_value(e4m3_code(fmaxf(amax, 6.0f * 0x1p-9f) / 6.0f));
    for (unsigned i = 0; i < n; ++i)
        b[i] = format == 0u ? e4m3_value(e4m3_code(format_clamp(b[i] / s, 448.0f))) * s : e2m1_value(e2m1_code(format_clamp(b[i] / s, 6.0f))) * s;
}

// A block a row: the `picks` best of `items` scores by score, then lower index, written by ascending index, mode 0 topk_rows and mode 1 candidate_blocks, as Vulkan's topk_rows.comp.
// The picks' least key is found bit by bit from the top over counts of the row, so the workspace is the block's at any row length.
__device__ inline unsigned select_key(float s) {
    const unsigned u = __float_as_uint(s == 0.0f ? 0.0f : s);
    return u >> 31 ? ~u : u | 0x80000000u;
}
__device__ inline float select_score(const float* row, unsigned i, unsigned mode, unsigned n, unsigned block_size, unsigned pinned) {
    if (mode == 0u) return row[i];
    if (i == pinned) return INFINITY;
    float best = -INFINITY;
    const unsigned end = min(n, (i + 1u) * block_size);
    for (unsigned j = i * block_size; j < end; ++j) best = fmaxf(best, row[j]);
    return best;
}
__device__ unsigned block_count(unsigned c, unsigned* red) {
    red[threadIdx.x] = c;
    __syncthreads();
    for (unsigned k = kThreads / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) red[threadIdx.x] += red[threadIdx.x + k];
        __syncthreads();
    }
    const unsigned total = red[0];
    __syncthreads();
    return total;
}
// The inclusive prefix of v over the block's threads, and its total.
__device__ unsigned block_prefix(unsigned v, unsigned& total, unsigned* scan) {
    const unsigned t = threadIdx.x;
    scan[t] = v;
    __syncthreads();
    for (unsigned k = 1; k < kThreads; k <<= 1) {
        const unsigned add = t >= k ? scan[t - k] : 0u;
        __syncthreads();
        scan[t] += add;
        __syncthreads();
    }
    const unsigned s = scan[t];
    total = scan[kThreads - 1];
    __syncthreads();
    return s;
}
__device__ unsigned select_count(const float* row, unsigned items, unsigned mode, unsigned n, unsigned block_size, unsigned pinned, unsigned key, bool strict, unsigned* red) {
    unsigned c = 0;
    for (unsigned i = threadIdx.x; i < items; i += kThreads) {
        const unsigned k = select_key(select_score(row, i, mode, n, block_size, pinned));
        c += strict ? k > key : k >= key;
    }
    return block_count(c, red);
}
extern "C" __global__ void llmx_topk_rows(unsigned* ids, const float* scores, const unsigned* lens, unsigned n, unsigned stride, unsigned picks, unsigned offset,
                                            unsigned mode, unsigned block_size, unsigned items) {
    __shared__ unsigned red[kThreads], scan[kThreads];
    const unsigned long long r = blockIdx.x;
    const float* row = scores + r * n;
    unsigned* out = ids + r * stride;
    const unsigned len = lens[r], t = threadIdx.x;
    const unsigned pinned = mode == 1u && len && (len - 1u) / block_size < items ? (len - 1u) / block_size : kNoExpert;
    unsigned threshold = 0;
    if (picks)
        for (int bit = 31; bit >= 0; --bit) {
            const unsigned candidate = threshold | (1u << bit);
            if (select_count(row, items, mode, n, block_size, pinned, candidate, false, red) >= picks) threshold = candidate;
        }
    const unsigned need = picks - (picks ? select_count(row, items, mode, n, block_size, pinned, threshold, true, red) : 0u);
    unsigned ties = 0, kept = 0;
    for (unsigned first = 0; first < items; first += kThreads) {
        const unsigned i = first + t;
        const bool live = i < items && picks;
        const float s = i < items ? select_score(row, i, mode, n, block_size, pinned) : -INFINITY;
        const unsigned key = i < items ? select_key(s) : 0u;
        unsigned tie_total, written_total;
        const unsigned tie = live && key == threshold;
        const unsigned tie_before = ties + block_prefix(tie, tie_total, scan) - tie;
        const bool picked = live && (key > threshold || (tie && tie_before < need));
        const bool written = picked && (mode == 0u || s > -INFINITY);
        const unsigned at = block_prefix(written, written_total, scan);
        if (written) out[kept + at - 1u] = mode == 0u ? (i < len ? i + offset : kNoExpert) : i;
        ties += tie_total;
        kept += written_total;
    }
    for (unsigned i = kept + t; i < stride; i += kThreads) out[i] = kNoExpert;
}

// A block a group, a thread a channel: the kv values weighted by the softmax of their scores over the group's rows in the CPU's order, then the pooled row's RMS norm with the CPU's sums and products.
extern "C" __global__ void llmx_compress_pool(float* out, const float* kv, const float* score, const float* w, unsigned ratio, unsigned dim, float eps) {
    __shared__ float lds[16];
    const unsigned long long g = blockIdx.x, base = g * ratio * dim;
    float* o = out + g * dim;
    for (unsigned d = threadIdx.x; d < dim; d += kThreads) {
        float peak = -INFINITY;
        for (unsigned i = 0; i < ratio; ++i) peak = fmaxf(peak, score[base + (unsigned long long)i * dim + d]);
        float sum = 0.0f;
        for (unsigned i = 0; i < ratio; ++i) sum += expf(score[base + (unsigned long long)i * dim + d] - peak);
        float acc = 0.0f;
        for (unsigned i = 0; i < ratio; ++i) {
            const unsigned long long at = base + (unsigned long long)i * dim + d;
            acc = fmaf(kv[at], expf(score[at] - peak) / sum, acc);
        }
        o[d] = acc;
    }
    __syncthreads();
    const float rs = 1.0f / sqrtf(cpu_squares(o, dim, lds) / (float)dim + eps);
    for (unsigned d = threadIdx.x; d < dim; d += kThreads) o[d] = o[d] * (rs * w[d]);
}

// The pooled rows a row reads (the host's pooled table): rows[2r] is the offset of its view's block ids in `rows`, rows[2r + 1] its position plus one.
// Group g's row starts after its block's token rows, at its slot among the block's groups.
__device__ inline unsigned long long pooled_at(const unsigned* rows, unsigned r, unsigned g, unsigned tokens, unsigned bt, unsigned long long block_floats, unsigned rows_floats,
                                               unsigned width) {
    const unsigned t = g * tokens;
    return (unsigned long long)rows[rows[2u * r] + t / bt] * block_floats + rows_floats + (unsigned long long)((t % bt) / tokens) * width;
}

// A thread a (row, position): sum over heads of w[h] w_scale ReLU(q_h dot key) with the CPU's multiply-add chains, key the last dim floats of the pooled row, -inf where the row cannot see it.
extern "C" __global__ void llmx_indexer_scores(float* out, const float* q, const float* aux, const unsigned* rows, const float* w, const unsigned* blocks, unsigned n,
                                               unsigned heads, unsigned dim, float w_scale, unsigned tokens, unsigned bt, unsigned long long block_floats, unsigned rows_floats,
                                               unsigned width, unsigned block_k, unsigned block_size) {
    const unsigned t = blockIdx.x * kThreads + threadIdx.x, r = blockIdx.y;
    if (t >= n) return;
    float* o = out + (unsigned long long)r * n + t;
    bool reached = t < rows[2u * r + 1u] / tokens;
    if (reached && block_k) {
        reached = false;
        for (unsigned b = 0; b < block_k && !reached; ++b) reached = blocks[(unsigned long long)r * block_k + b] == t / block_size;
    }
    if (!reached) {
        *o = -INFINITY;
        return;
    }
    const float* key = aux + pooled_at(rows, r, t, tokens, bt, block_floats, rows_floats, width) + width - dim;
    float sum = 0.0f;
    for (unsigned h = 0; h < heads; ++h) {
        const float* qh = q + ((unsigned long long)r * heads + h) * dim;
        float dot = 0.0f;
        for (unsigned d = 0; d < dim; ++d) dot = fmaf(qh[d], key[d], dot);
        sum = fmaf(fmaxf(dot, 0.0f), w[(unsigned long long)r * heads + h] * w_scale, sum);
    }
    *o = sum;
}

// A thread a float of the call's groups, each group's first float in the auxiliary plane given by the host, copied from the rows in order.
extern "C" __global__ void llmx_pool_write(float* aux, const float* src, const unsigned long long* at, unsigned width, unsigned long long n) {
    const unsigned long long i = (unsigned long long)blockIdx.x * kThreads + threadIdx.x;
    if (i >= n) return;
    const unsigned long long g = i / width;
    aux[at[g] + (i - g * width)] = src[i];
}

// A block a (row, head): attention over the kv rows its ids name, then the pooled rows its entry ids name, one row serving as K and V, under one softmax whose denominator takes exp(sink - max).
// Each dot and the output's sums are the CPU's multiply-add chains in its order, and an id that names nothing is skipped; at most 1024 rows are named.
extern "C" __global__ void llmx_sparse_attention_sink(float* out, const float* q, const float* kv, const unsigned* ids, const float* sink, const float* aux, const unsigned* rows,
                                                      const unsigned* entries, unsigned heads, unsigned dim, unsigned k, unsigned entry_k, float scale, unsigned kv_rows,
                                                      unsigned tokens, unsigned bt, unsigned long long block_floats, unsigned rows_floats, unsigned width) {
    __shared__ float pr[1024], red[kThreads];
    __shared__ const float* at[1024];
    __shared__ float denominator;
    const unsigned rh = blockIdx.x, r = rh / heads, h = rh - r * heads, named = k + entry_k;
    const unsigned visible = rows ? rows[2u * r + 1u] / tokens : 0u;
    const float* qr = q + (unsigned long long)rh * dim;
    float peak = -1e30f;
    for (unsigned t = threadIdx.x; t < named; t += kThreads) {
        const unsigned j = t < k ? ids[(unsigned long long)r * k + t] : entries[(unsigned long long)r * entry_k + t - k];
        const float* row = nullptr;
        if (t < k && j < kv_rows) row = kv + (unsigned long long)j * dim;
        else if (t >= k && j < visible) row = aux + pooled_at(rows, r, j, tokens, bt, block_floats, rows_floats, width);
        at[t] = row;
        if (!row) continue;
        float dot = 0.0f;
        for (unsigned d = 0; d < dim; ++d) dot = fmaf(qr[d], row[d], dot);
        pr[t] = dot * scale;
        peak = fmaxf(peak, pr[t]);
    }
    red[threadIdx.x] = peak;
    __syncthreads();
    for (unsigned s = kThreads / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    peak = red[0];
    for (unsigned t = threadIdx.x; t < named; t += kThreads)
        if (at[t]) pr[t] = expf(pr[t] - peak);
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (unsigned t = 0; t < named; ++t)
            if (at[t]) sum += pr[t];
        denominator = sum + expf(sink[h] - peak);
    }
    __syncthreads();
    for (unsigned d = threadIdx.x; d < dim; d += kThreads) {
        float o = 0.0f;
        for (unsigned t = 0; t < named; ++t)
            if (at[t]) o = fmaf(pr[t], at[t][d], o);
        out[(unsigned long long)rh * dim + d] = o / denominator;
    }
}

// A thread a row: route_experts' sqrt_softplus rule in the CPU's order, s = sqrt(softplus(x)), the k picks by s + bias, highest first and ties to the lower id, weighed s / (sum + 1e-20) when normalizing more than one, then times scale.
// A row whose s + bias is not finite everywhere gives k picks of kNoExpert with weights 0.
__device__ inline float softplus_root(float v) { return sqrtf(v > 20.0f ? v : log1pf(expf(v))); }
extern "C" __global__ void llmx_route_sqrt_softplus(const float* scores, const float* bias, unsigned* ids, float* weights, unsigned long long rows, unsigned n_expert,
                                                    unsigned k, int normalize, float scale) {
    const unsigned long long r = (unsigned long long)blockIdx.x * 64 + threadIdx.x;
    if (r >= rows) return;
    const float* x = scores + r * n_expert;
    unsigned* id = ids + r * k;
    float* wt = weights + r * k;
    bool finite = true;
    for (unsigned e = 0; e < n_expert; ++e) {
        const float s = softplus_root(x[e]), b = bias ? s + bias[e] : s;
        finite = finite && isfinite(b);
    }
    if (!finite) {
        for (unsigned j = 0; j < k; ++j) id[j] = kNoExpert, wt[j] = 0.0f;
        return;
    }
    float total = 0.0f;
    for (unsigned j = 0; j < k; ++j) {
        unsigned best = n_expert;
        float best_b = 0.0f;
        for (unsigned e = 0; e < n_expert; ++e) {
            bool taken = false;
            for (unsigned i = 0; i < j && !taken; ++i) taken = id[i] == e;
            if (taken) continue;
            const float s = softplus_root(x[e]), b = bias ? s + bias[e] : s;
            if (best == n_expert || b > best_b) best = e, best_b = b;
        }
        id[j] = best;
        total += softplus_root(x[best]);
    }
    const bool divide = normalize && k > 1u;
    for (unsigned j = 0; j < k; ++j) {
        const float s = softplus_root(x[id[j]]);
        wt[j] = (divide ? s / (total + 1e-20f) : s) * scale;
    }
}

// A block a (row, head) of norm_rope_partial's other forms: the head's RMS norm in the CPU's order when w is given, then each pair from `offset` on rotated by the CPU's multiply-adds, turning back when inverse.
// A thread reads its values before the barrier and writes them after it, so dst may be src over contiguous heads.
extern "C" __global__ void llmx_norm_rope_form(float* dst, const float* src, unsigned long long dst_stride, unsigned long long src_stride, unsigned long long src_head_stride,
                                               unsigned head_dim, unsigned half, const float* w, float eps, const float* cos, const float* sin, const unsigned* pos,
                                               const unsigned char* of, unsigned count, unsigned adjacent, unsigned offset, unsigned inverse) {
    __shared__ float lds[16];
    const unsigned long long r = blockIdx.x, h = blockIdx.y;
    float* head = dst + r * dst_stride + h * head_dim;
    const float* from = src + r * src_stride + h * src_head_stride;
    const float rs = w ? 1.0f / sqrtf(cpu_squares(from, head_dim, lds) / (float)head_dim + eps) : 1.0f;
    const auto value = [&](unsigned i) { return w ? from[i] * (rs * w[i]) : from[i]; };
    float va[8], vb[8];
    unsigned n = 0;
    for (unsigned j = threadIdx.x; j < half; j += kThreads, ++n) {
        va[n] = value(offset + (adjacent ? 2u * j : j));
        vb[n] = value(offset + (adjacent ? 2u * j + 1u : j + half));
    }
    float rest[8];
    unsigned m = 0;
    for (unsigned i = threadIdx.x; i < head_dim; i += kThreads)
        if (i < offset || i >= offset + 2u * half) rest[m++] = value(i);
    __syncthreads();
    n = 0;
    for (unsigned j = threadIdx.x; j < half; j += kThreads, ++n) {
        const unsigned long long t = (unsigned long long)pos[r * count + (count > 1u ? of[j] : 0u)] * half + j;
        const float c = cos[t], s = inverse ? -sin[t] : sin[t];
        head[offset + (adjacent ? 2u * j : j)] = fmaf(-vb[n], s, va[n] * c);
        head[offset + (adjacent ? 2u * j + 1u : j + half)] = fmaf(va[n], s, vb[n] * c);
    }
    m = 0;
    for (unsigned i = threadIdx.x; i < head_dim; i += kThreads)
        if (i < offset || i >= offset + 2u * half) head[i] = rest[m++];
}

// The linear-attention decode step (rocm_backend.cpp delta_step), field for field with the host's DeltaStepArgs.
struct DeltaStepArgs {
    float* o;                 // the delta rule's rows, then their gated norm in place, rows x v_heads x vd
    const float* x;           // the qkv product's rows, the conv's input, rows x C
    const float* cw;          // the conv's weights, C x 4
    const float* mixed;       // the products' input rows, rows x E
    const unsigned char* wa;  // the F32 alpha and beta matrices, v_heads x E each
    const unsigned char* wb;
    float* ao;                // the alpha and beta products' rows, rows x v_heads each
    float* bo;
    const float* a;
    const float* dt_bias;
    float* st;                // the layer's state slots
    const unsigned* tab;      // rocm_backend.cpp state_table
    const float* z;           // the gate product's rows, rows x v_heads x vd
    const float* nw;          // the norm's weight, vd
    unsigned* counter;        // a counter a (view, K head), zero between launches
    unsigned C, k_heads, v_heads, E;
    unsigned long long slot, carried;
    float scale, eps, neps;
    void* tq;
    float* td;
    int tlevel, major;
    TwinMap tm;
    unsigned kd, vd;          // the K and V head widths, multiples of 4 up to 128
};

// A block a (view, V head j), each of its 32 lane groups four adjacent V columns and each of a group's eight lanes 16 K rows, so a block holds the whole head at kd, vd <= 128.
// It runs exact_sum's alpha and beta products, llmx_causal_conv_silu's conv, the delta rule as llmx_gated_delta_rule_short takes it and llmx_gated_rms_norm's sigmoid gate, each output their bits, for at most 8 rows a view.
extern "C" __global__ void __launch_bounds__(kThreads, 1) llmx_delta_step(DeltaStepArgs A) {
    constexpr unsigned kTb = 4, kRows = 16, kMaxQ = 8, P = 4, W = 128;
    const unsigned kd = A.kd, vd = A.vd;
    __shared__ float kn[kTb][128], qn[kTb][128], vs[kTb][W], betas[kTb], decays[kTb];
    __shared__ float cqk[kMaxQ][256], cv[kMaxQ][W], ab[2 * kMaxQ], red[kThreads], osh[kMaxQ][W];
    __shared__ unsigned last;
    const unsigned C = A.C, k_heads = A.k_heads, v_heads = A.v_heads;
    const unsigned v = blockIdx.x / v_heads, j = blockIdx.x - v * v_heads;
    const unsigned lid = threadIdx.x, part = lid & 7u, c = (lid >> 3u) * P;
    const bool col = c < vd;
    const unsigned qk = k_heads * kd, kh = j % k_heads;
    // The table through the constant address space, so its reads are scalar loads.
    const __attribute__((address_space(4))) unsigned* tab = (const __attribute__((address_space(4))) unsigned*)A.tab;
    const unsigned row0 = tab[1 + 5 * v], nq = tab[2 + 5 * v], len = tab[3 + 5 * v];
    const bool fresh = len == 0u;
    const unsigned long long matrix = (unsigned long long)j * kd * vd + c;
    const unsigned long long src = tab[4 + 5 * v] * A.slot + matrix, dst = tab[5 + 5 * v] * A.slot + matrix;
    float* st = A.st;
    // The state's loads first, four adjacent columns a load, so their latency runs under the products'; a fresh slot's are dropped.
    float s[P][kRows];
#pragma unroll
    for (unsigned ii = 0; ii < kRows; ++ii) {
        const bool on = !fresh && col && part + 8u * ii < kd;
        const float4 q = on ? *(const float4*)&st[src + (unsigned long long)(part + 8u * ii) * vd] : float4{0.0f, 0.0f, 0.0f, 0.0f};
        s[0][ii] = fresh ? 0.0f : q.x, s[1][ii] = fresh ? 0.0f : q.y, s[2][ii] = fresh ? 0.0f : q.z, s[3][ii] = fresh ? 0.0f : q.w;
    }
    // Alpha and beta of head j for every row of the view, a wave a product.
    for (unsigned p = lid / kWave; p < 2u * nq; p += kThreads / kWave) {
        const unsigned long long r = row0 + p / 2u;
        const float sum = exact_sum<ExactF32>((p & 1u) ? A.wb : A.wa, A.mixed + r * A.E, A.E, j);
        if (lid % kWave == 0) {
            ab[p] = sum;
            ((p & 1u) ? A.bo : A.ao)[r * v_heads + j] = sum;
        }
    }
    // The conv of head kh's q and k channels and of head j's V channels, as llmx_causal_conv_silu.
    const float* csrc = A.st + tab[4 + 5 * v] * A.slot + A.carried;
    float* cdst = A.st + tab[5 + 5 * v] * A.slot + A.carried;
    {
        const unsigned ch = lid < 128u ? kh * kd + lid : qk + kh * kd + (lid - 128u);
        const auto raw = [&](unsigned t, unsigned cc) { return t < 3u ? (len + t >= 3u ? csrc[(unsigned long long)t * C + cc] : 0.0f) : A.x[(unsigned long long)(row0 + t - 3u) * C + cc]; };
        const auto conv = [&](unsigned cc, float* out, unsigned stride) {
            const float w0 = A.cw[4ull * cc], w1 = A.cw[4ull * cc + 1], w2 = A.cw[4ull * cc + 2], w3 = A.cw[4ull * cc + 3];
            float win0 = raw(0, cc), win1 = raw(1, cc), win2 = raw(2, cc);
            for (unsigned b = 0; b < nq; ++b) {
                const float now = A.x[(unsigned long long)(row0 + b) * C + cc];
                float acc = w0 * win0;
                acc = fmaf(w1, win1, acc);
                acc = fmaf(w2, win2, acc);
                acc = fmaf(w3, now, acc);
                out[b * stride] = acc / (1.0f + expf(-acc));
                win0 = win1, win1 = win2, win2 = now;
            }
        };
        const bool qkon = (lid & 127u) < kd;
        if (qkon) conv(ch, &cqk[0][lid], 256);
        else for (unsigned b = 0; b < nq; ++b) cqk[b][lid] = 0.0f;
        if (lid < vd) {
            const unsigned cc = 2u * qk + j * vd + lid;
            const float keep0 = raw(nq, cc), keep1 = raw(nq + 1, cc), keep2 = raw(nq + 2, cc);
            conv(cc, &cv[0][lid], W);
            float* d = cdst + cc;
            d[0] = keep0, d[C] = keep1, d[2ull * C] = keep2;
        }
        // The q and k channels' carried rows are written by the last of the V heads that read them, once every one has read them.
        __threadfence();
        __syncthreads();
        if (lid == 0) last = atomicAdd(A.counter + v * k_heads + kh, 1u) == v_heads / k_heads - 1u;
        __syncthreads();
        if (last && qkon) {
            __threadfence();
            const float keep0 = raw(nq, ch), keep1 = raw(nq + 1, ch), keep2 = raw(nq + 2, ch);
            float* d = cdst + ch;
            d[0] = keep0, d[C] = keep1, d[2ull * C] = keep2;
        }
        if (last && lid == 0) A.counter[v * k_heads + kh] = 0;
    }
    // The delta rule, kTb rows a stage, reading the conv and the products from shared memory.
    const unsigned g = lid >> 3u, tok = g >> 1u;
    const bool is_k = (g & 1u) != 0u;
    float m[P];
    for (unsigned t0 = 0; t0 < nq; t0 += kTb) {
        const unsigned n = nq - t0 < kTb ? nq - t0 : kTb;
        if (tok < n) {
            const float* in0 = &cqk[t0 + tok][is_k ? 128u : 0u];
            float xx[kRows], ss = 0.0f;
#pragma unroll
            for (unsigned ii = 0; ii < kRows; ++ii) {
                xx[ii] = in0[part + 8u * ii];
                ss = fmaf(xx[ii], xx[ii], ss);
            }
            const float inv = 1.0f / sqrtf(lanes8(ss) + A.eps);
#pragma unroll
            for (unsigned ii = 0; ii < kRows; ++ii) {
                const float nv = xx[ii] * inv;
                if (is_k) kn[tok][part + 8u * ii] = nv;
                else qn[tok][part + 8u * ii] = nv * A.scale;
            }
        }
        if (lid < n) {
            const float beta = 1.0f / (1.0f + expf(-ab[2u * (t0 + lid) + 1u]));
            const float xg = ab[2u * (t0 + lid)] + A.dt_bias[j];
            const float sp = xg > 20.0f ? xg : log1p_of(expf(xg));
            const float decay = expf(A.a[j] * sp);
            betas[lid] = beta;
            decays[lid] = decay < 1.17549435e-38f ? 0.0f : decay;
        }
        for (unsigned idx = lid; idx < n * W; idx += kThreads) vs[idx / W][idx % W] = cv[t0 + idx / W][idx % W];
        __syncthreads();
        {
            const float decay = decays[0];
#pragma unroll
            for (unsigned k = 0; k < P; ++k) {
                float mp = 0.0f;
#pragma unroll
                for (unsigned ii = 0; ii < kRows; ++ii) {
                    s[k][ii] = s[k][ii] * decay;
                    mp = fmaf(s[k][ii], kn[0][part + 8u * ii], mp);
                }
                m[k] = lanes8(mp);
            }
        }
        for (unsigned t = 0; t < n; ++t) {
            const bool next = t + 1u < n;
#pragma unroll
            for (unsigned k = 0; k < P; ++k) {
                const float d = betas[t] * (vs[t][c + k] + -m[k]);
                float op = 0.0f;
#pragma unroll
                for (unsigned ii = 0; ii < kRows; ++ii) {
                    s[k][ii] = fmaf(kn[t][part + 8u * ii], d, s[k][ii]);
                    op = fmaf(s[k][ii], qn[t][part + 8u * ii], op);
                }
                float np = 0.0f;
                if (next) {
                    const float decay = decays[t + 1u];
#pragma unroll
                    for (unsigned ii = 0; ii < kRows; ++ii) {
                        s[k][ii] = s[k][ii] * decay;
                        np = fmaf(s[k][ii], kn[t + 1u][part + 8u * ii], np);
                    }
                }
                const float res = lanes8(op);
                if (next) m[k] = lanes8(np);
                if (part == 0u && col) osh[t0 + t][c + k] = res;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned ii = 0; ii < kRows; ++ii)
#pragma unroll
        for (unsigned k = 0; k < P; ++k)
            if (col && part + 8u * ii < kd) st[dst + k + (unsigned long long)(part + 8u * ii) * vd] = s[k][ii];
    // The gated norm of head j's rows from shared memory, as llmx_gated_rms_norm with its sigmoid gate, and their twin.
    for (unsigned b = 0; b < nq; ++b) {
        const unsigned long long o = ((unsigned long long)(row0 + b) * v_heads + j) * vd;
        const float xv = lid < vd ? osh[b][lid] : 0.0f;
        const float sq = lid < vd ? fmaf(xv, xv, 0.0f) : 0.0f;
        const float rr = 1.0f / sqrtf(block_sum(sq, red) / (float)vd + A.neps);
        if (lid < vd) {
            const float zz = A.z[o + lid], gg = 1.0f / (1.0f + expf(-zz));
            const float val = xv * (rr * A.nw[lid]) * gg;
            A.o[o + lid] = val;
            twin_at(val, o + lid, A.tq, A.td, A.tlevel, A.tm, A.major);
        }
    }
}
