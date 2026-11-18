// The only file naming vendor intrinsics: the kernels reach every lane exchange, byte operation, integer dot, scoped atomic, fence and clock through these wrappers.
// Under HIP each wrapper expands to exactly the builtin the kernels called before it, so moving a call behind one changes no code object.
#pragma once

#if defined(__HIP__)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// The lanes of a wave the code is compiled for: 64 on gfx9, 32 on the targets compiled for wave32.
#if defined(__GFX9__)
#define GPU_WAVE 64
#else
#define GPU_WAVE 32
#endif
// 1 where dot4_i8 is the one instruction the code is compiled with: gfx9 and CUDA; the plain module's other targets leave out the kernels that need it.
#if defined(__GFX9__)
#define GPU_DOT4 1
#else
#define GPU_DOT4 0
#endif
// 1 where the int8 class runs in one module on the four-way dot and the 8-bit integer tensor cores: never under HIP, whose plain module keeps the scalar loop and whose dot module (LLMX_DOT16) takes dot4_i8.
#define GPU_IMMA8 0
// The launch bounds of the dense int8 tiles, which HIP leaves at the block size so gfx9 keeps its code.
#define LLMX_TILE8_BOUNDS(blocks) __launch_bounds__(kThreads)
// The one-shot group sum at five waves a SIMD, so the 16 blocks each of a 16-rank group stay resident together on one 60-compute-unit gfx906 (docs/src/backends-gpu.md).
#define LLMX_ONE_SHOT_BOUNDS __launch_bounds__(kThreads) __attribute__((amdgpu_waves_per_eu(5)))
// A kernel's by-value struct parameter, which HIP leaves as it is so gfx9 keeps its code.
#define LLMX_GRID_CONST

// The value of lane (this lane xor m), or of lane (this lane + d), within groups of width lanes, the whole wave where no width is given.
template <typename T> __device__ __forceinline__ auto shuffle_xor(T v, int m) { return __shfl_xor(v, m); }
template <typename T> __device__ __forceinline__ auto shuffle_xor(T v, int m, int width) { return __shfl_xor(v, m, width); }
template <typename T> __device__ __forceinline__ auto shuffle_down(T v, int d) { return __shfl_down(v, d); }
template <typename T> __device__ __forceinline__ auto shuffle_down(T v, int d, int width) { return __shfl_down(v, d, width); }
// The integer sum over each aligned group of W lanes, which every lane of the group returns.
template <int W> __device__ __forceinline__ int segment_sum(int v) {
    for (int o = W / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, W);
    return v;
}

// The value of lane (this lane xor M) for M = 1 or 2 by DPP quad_perm within each group of four lanes: selector 0xB1 is [1,0,3,2] (xor 1), 0x4E is [2,3,0,1] (xor 2).
// Both pass row mask 0xF, bank mask 0xF and bound_ctrl false, so every lane writes and reads its partner.
template <int M, typename T> __device__ __forceinline__ auto lane_xor_quad(T v) {
    static_assert(M == 1 || M == 2, "quad_perm exchanges cover xor 1 and 2");
    static_assert(__is_same(T, int) || __is_same(T, unsigned), "lane exchanges move a 32-bit integer's bits; a float goes through lane_xor");
    return __builtin_amdgcn_mov_dpp(v, M == 1 ? 0xB1 : 0x4E, 0xF, 0xF, false);
}
// The value of lane (this lane xor 1) where only that pair of lanes is known to run the exchange.
template <typename T> __device__ __forceinline__ auto pair_lane(T v) { return lane_xor_quad<1>(v); }
// The value of lane (this lane xor m), m below W, where only this lane's group of W lanes is known to run the exchange; the partner is in the group, so the wave's exchange gives it.
template <int W, typename T> __device__ __forceinline__ auto group_xor(T v, int m) { return __shfl_xor(v, m); }
// The value of lane (this lane xor M) for M = 4, 8 or 16 by ds_swizzle's bit mode within groups of 32 lanes: offset 0x1F | (M << 10) is and-mask 0x1F, or-mask 0, xor-mask M.
template <int M, typename T> __device__ __forceinline__ auto lane_xor_swizzle(T v) {
    static_assert(M == 4 || M == 8 || M == 16, "swizzle exchanges cover xor 4, 8 and 16");
    static_assert(__is_same(T, int) || __is_same(T, unsigned), "lane exchanges move a 32-bit integer's bits; a float goes through lane_xor");
    return __builtin_amdgcn_ds_swizzle(v, 0x1F | (M << 10));
}

// v of the first active lane, the same in every lane, so the compiler may keep it in a scalar register.
template <typename T> __device__ __forceinline__ auto first_lane(T v) { return __builtin_amdgcn_readfirstlane(v); }
// A bit a lane of the wave, set where that lane's p is nonzero.
__device__ __forceinline__ unsigned long long wave_ballot(int p) { return __ballot(p); }

// Bytes n to n + 3 of the 64-bit value hi:lo, lo the low word, n taken mod 4 (v_alignbyte).
__device__ __forceinline__ unsigned byte_align(unsigned hi, unsigned lo, unsigned n) { return __builtin_amdgcn_alignbyte(hi, lo, n); }
// Bits s to s + 31 of the 64-bit value hi:lo, lo the low word, s taken mod 32 (v_alignbit); the kernels pass s = 16.
__device__ __forceinline__ unsigned bit_align(unsigned hi, unsigned lo, unsigned s) { return __builtin_amdgcn_alignbit(hi, lo, s); }
// Byte i of the result is selector byte i over the bytes of hi:lo, lo bytes 0-3: 0-7 that byte, 8, 9, 10 and 11 the top bit of byte 1, 3, 5 and 7 spread over 8 bits, 12 gives 0x00 and 13 or more 0xFF (v_perm_b32).
// The kernels pass the selectors 0x0B090A08, 0x08010A00, 0x09030B02, 0x0C010C00 and 0x0C030C02, and selectors of bytes 0-7 alone as table lookups into (0x0C080604, 0x03020100) and (0xF4F8FAFC, 0xFDFEFF00).
__device__ __forceinline__ unsigned byte_perm(unsigned hi, unsigned lo, unsigned sel) { return __builtin_amdgcn_perm(hi, lo, sel); }

// s plus the four products of a's and b's signed bytes, or of their two signed 16-bit halves, without saturation (v_dot4_i32_i8, v_dot2_i32_i16).
__device__ __forceinline__ int dot4_i8(int a, int b, int s) { return __builtin_amdgcn_sdot4(a, b, s, false); }
__device__ __forceinline__ int dot2_i16(int a, int b, int s) {
    typedef short halves __attribute__((ext_vector_type(2)));
    return __builtin_amdgcn_sdot2(__builtin_bit_cast(halves, a), __builtin_bit_cast(halves, b), s, false);
}
// s plus the two products of a's and b's binary16 halves, in F32 (v_dot2_f32_f16).
__device__ __forceinline__ float dot2_f16(unsigned a, unsigned b, float s) {
    typedef _Float16 halves __attribute__((ext_vector_type(2)));
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(halves, a), __builtin_bit_cast(halves, b), s, false);
}

// No instruction is scheduled across this point.
__device__ __forceinline__ void schedule_barrier() { __builtin_amdgcn_sched_barrier(0); }
// A pause of the wave in a spin wait, about 64 clocks.
__device__ __forceinline__ void spin_pause() { __builtin_amdgcn_s_sleep(1); }
// A clock at a constant rate whatever the shader clock, for deadlines.
__device__ __forceinline__ auto wall_clock() { return wall_clock64(); }

// Atomic loads and stores at a stated order and scope: block is the workgroup, device the whole device, system every device and the host.
template <typename P> __device__ __forceinline__ auto atomic_load_relaxed_block(P p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP); }
template <typename P> __device__ __forceinline__ auto atomic_load_relaxed_device(P p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT); }
template <typename P> __device__ __forceinline__ auto atomic_load_acquire_device(P p) { return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT); }
template <typename P> __device__ __forceinline__ auto atomic_load_relaxed_system(P p) { return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); }
template <typename P> __device__ __forceinline__ auto atomic_load_acquire_system(P p) { return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM); }
template <typename P, typename V> __device__ __forceinline__ void atomic_store_relaxed_block(P p, V v) { __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP); }
template <typename P, typename V> __device__ __forceinline__ void atomic_store_relaxed_system(P p, V v) { __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM); }
template <typename P, typename V> __device__ __forceinline__ void atomic_store_release_system(P p, V v) { __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM); }
// An acquire fence at device scope.
__device__ __forceinline__ void fence_acquire_device() { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent"); }
// An acquire fence at system scope.
__device__ __forceinline__ void fence_acquire_system() { __atomic_thread_fence(__ATOMIC_ACQUIRE); }
// A grid barrier of resident blocks in three steps, each called by every thread: stores_sync before the arrive, grid_arrive and grid_wait by the whole first wave, acquire_sync after.
// On gfx9 before gfx940 the barriers wait on LDS alone and the counter goes on the scalar path; elsewhere they are __syncthreads and device-scope atomics (docs/src/backends-gpu.md, port.h).
#if defined(__GFX9__) && !defined(__gfx940__) && !defined(__gfx941__) && !defined(__gfx942__) && !defined(__gfx950__)
__device__ __forceinline__ void lds_sync() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
__device__ __forceinline__ void stores_sync() { asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
__device__ __forceinline__ void grid_arrive(unsigned* bar) {
    unsigned one = 1u;
    asm volatile("s_atomic_add %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "+s"(one) : "s"(bar) : "memory");
}
__device__ __forceinline__ void grid_wait(unsigned* bar, unsigned target) {
    unsigned cur;
    do asm volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(cur) : "s"(bar) : "memory");
    while (cur < target);
}
__device__ __forceinline__ void acquire_sync() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n\tbuffer_wbinvl1_vol" ::: "memory"); }
#else
__device__ __forceinline__ void lds_sync() { __syncthreads(); }
__device__ __forceinline__ void stores_sync() { __threadfence(); __syncthreads(); }
__device__ __forceinline__ void grid_arrive(unsigned* bar) { if (threadIdx.x == 0) atomicAdd(bar, 1u); }
__device__ __forceinline__ void grid_wait(unsigned* bar, unsigned target) {
    if (threadIdx.x == 0) while (atomic_load_acquire_device(bar) < target) spin_pause();
}
__device__ __forceinline__ void acquire_sync() { __syncthreads(); fence_acquire_device(); }
#endif
// The device library's tanhf.
__device__ __forceinline__ float gpu_tanhf(float x) { return tanhf(x); }
// A constexpr host table read in device code, which HIP reads in place.
#define GPU_TABLE(a) (a)

// Pointer qualifiers: memory no lane writes during the launch, which the compiler may read through scalar loads, and global memory.
#define GPU_CONSTANT_SPACE __attribute__((address_space(4)))
#define GPU_GLOBAL_SPACE __attribute__((address_space(1)))
// N 32-bit words aligned to 4 bytes, loaded and stored as one vector.
template <int N> using gpu_words = unsigned __attribute__((ext_vector_type(N), aligned(4)));
// N 32-bit words aligned to their whole size.
template <int N> using gpu_vector = unsigned __attribute__((ext_vector_type(N)));
// The vector at p in one volatile load, so a spin on it sees each peer's store.
template <int N> __device__ __forceinline__ gpu_vector<N> load_volatile(const gpu_vector<N>* p) { return *(const volatile gpu_vector<N>*)p; }

#elif defined(__CUDACC__)

#include <cuda_fp16.h>
#include <type_traits>
#define LLMX_PORT_MODEL __host__ __device__ __forceinline__ constexpr

#elif defined(LLMX_PORT_HOST_MODEL)

// The host build of the mappings' arithmetic, which tests/gpu_port.cpp checks against models of the instructions.
#define LLMX_PORT_MODEL inline constexpr

#else
#error "port.h has wrappers for HIP and CUDA alone"
#endif

#if !defined(__HIP__)

// An AMD byte selector as a PTX prmt selector: bytes 0-7 as they are, 8-11 the sign of byte 1, 3, 5 or 7 (nibble bit 3), and the bytes that select 12 (zero) or 13 and more (0xFF) masked in after.
struct PortPrmt { unsigned nibbles, clear, ones; };
LLMX_PORT_MODEL PortPrmt port_prmt(unsigned sel) {
    PortPrmt r{0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        const unsigned s = (sel >> (8 * i)) & 0xffu;
        if (s < 8u) r.nibbles |= s << (4 * i);
        else if (s < 12u) r.nibbles |= (8u | (2u * (s - 8u) + 1u)) << (4 * i);
        else if (s == 12u) r.clear |= 0xffu << (8 * i);
        else r.ones |= 0xffu << (8 * i);
    }
    return r;
}
// Two signed 16-bit products summed in wrapping 32-bit arithmetic, as the AMD instruction without clamp.
LLMX_PORT_MODEL int port_dot2_i16(int a, int b, int s) {
    const unsigned lo = (unsigned)((int)(short)a * (int)(short)b), hi = (unsigned)((int)(short)(a >> 16) * (int)(short)(b >> 16));
    return (int)((unsigned)s + lo + hi);
}
// The lanes of a 32-lane warp in lane's segment of `width` lanes, width a power of two, the whole warp from 32 up.
LLMX_PORT_MODEL unsigned port_segment_mask(int width, unsigned lane) {
    return width >= 32 ? 0xffffffffu : ((1u << width) - 1u) << (lane & ~(unsigned)(width - 1));
}

#endif

#if defined(__CUDACC__)

// Under CUDA a wave is a warp of 32 lanes; an exchange of a width names that width's lanes alone, since a kernel may run a loop whose trip count differs between the segments of one warp.
#define GPU_WAVE 32
#define GPU_DOT4 1

#define LLMX_FULL_WARP 0xffffffffu
__device__ __forceinline__ unsigned lane_index() {
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    return lane;
}
__device__ __forceinline__ unsigned segment_lanes(int width) { return port_segment_mask(width, lane_index()); }
template <typename T> __device__ __forceinline__ auto shuffle_xor(T v, int m) { return __shfl_xor_sync(LLMX_FULL_WARP, v, m); }
template <typename T> __device__ __forceinline__ auto shuffle_xor(T v, int m, int width) { return __shfl_xor_sync(segment_lanes(width), v, m, width); }
template <typename T> __device__ __forceinline__ auto shuffle_down(T v, int d) { return __shfl_down_sync(LLMX_FULL_WARP, v, d); }
template <typename T> __device__ __forceinline__ auto shuffle_down(T v, int d, int width) { return __shfl_down_sync(segment_lanes(width), v, d, width); }
// Before sm_80, which lacks redux.sync, a segment's sum is a shuffle tree, the same exact integer sum.
template <int W> __device__ __forceinline__ int segment_sum(int v) {
#if __CUDA_ARCH__ >= 800
    return __reduce_add_sync(segment_lanes(W), v);
#else
    for (int m = W / 2; m > 0; m /= 2) v += __shfl_xor_sync(segment_lanes(W), v, m, W);
    return v;
#endif
}
template <int M, typename T> __device__ __forceinline__ auto lane_xor_quad(T v) {
    static_assert(M == 1 || M == 2, "quad_perm exchanges cover xor 1 and 2");
    static_assert(__is_same(T, int) || __is_same(T, unsigned), "lane exchanges move a 32-bit integer's bits; a float goes through lane_xor");
    return __shfl_xor_sync(LLMX_FULL_WARP, v, M);
}
template <int M, typename T> __device__ __forceinline__ auto lane_xor_swizzle(T v) {
    static_assert(M == 4 || M == 8 || M == 16, "swizzle exchanges cover xor 4, 8 and 16");
    static_assert(__is_same(T, int) || __is_same(T, unsigned), "lane exchanges move a 32-bit integer's bits; a float goes through lane_xor");
    return __shfl_xor_sync(LLMX_FULL_WARP, v, M);
}
// The pair's exchange names the pair alone: the 8-bit rows hand a block's scale to the odd lane in a loop whose trip count differs between pairs, and a full-warp mask there is undefined.
// It waits for every named lane, so lanes already past the loop met it from the wave sum's shuffles and both exchanged wrong values (docs/STATUS.md, CUDA backend step 4).
template <typename T> __device__ __forceinline__ auto pair_lane(T v) { return __shfl_xor_sync(segment_lanes(2), v, 1); }
// A group's exchange names the group alone: the delta rule sums a token's eight lanes in a branch the other tokens' lanes of the warp skip (docs/STATUS.md, CUDA backend step 5).
template <int W, typename T> __device__ __forceinline__ auto group_xor(T v, int m) { return __shfl_xor_sync(segment_lanes(W), v, m, W); }
template <typename T> __device__ __forceinline__ auto first_lane(T v) { return __shfl_sync(LLMX_FULL_WARP, v, 0); }
__device__ __forceinline__ unsigned long long wave_ballot(int p) { return __ballot_sync(LLMX_FULL_WARP, p); }

__device__ __forceinline__ unsigned byte_align(unsigned hi, unsigned lo, unsigned n) { return __funnelshift_r(lo, hi, (n & 3u) * 8u); }
__device__ __forceinline__ unsigned bit_align(unsigned hi, unsigned lo, unsigned s) { return __funnelshift_r(lo, hi, s); }
// prmt is written out because __byte_perm keeps three bits of each selector nibble, which drops the sign mode selectors 8 to 11 need.
__device__ __forceinline__ unsigned byte_perm(unsigned hi, unsigned lo, unsigned sel) {
    const PortPrmt p = port_prmt(sel);
    unsigned r;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(lo), "r"(hi), "r"(p.nibbles));
    return (r & ~(p.clear | p.ones)) | p.ones;
}

__device__ __forceinline__ int dot4_i8(int a, int b, int s) { return __dp4a(a, b, s); }
__device__ __forceinline__ int dot2_i16(int a, int b, int s) { return port_dot2_i16(a, b, s); }
__device__ __forceinline__ float dot2_f16(unsigned a, unsigned b, float s) {
    const auto h = [](unsigned v, int i) { return __half2float(__ushort_as_half((unsigned short)(v >> (16 * i)))); };
    return fmaf(h(a, 1), h(b, 1), fmaf(h(a, 0), h(b, 0), s));
}
// Every CUDA target has the four-way dot (GPU_DOT4, above), and from sm_80 the 8-bit integer tensor cores, so the int8 class takes both in its one module.
// Before sm_80 the tensor-core products below are four-way dots over exchanged fragments: the same wrapping int32 sums, slower.
#define GPU_IMMA8 1
// The dense int8 tiles' blocks an SM: two hold a tile to 128 registers, which the tensor-core fragments pass otherwise (docs/STATUS.md, CUDA int8 tiles of every type on the tensor cores).
#define LLMX_TILE8_BOUNDS(blocks) __launch_bounds__(kThreads, blocks)
#define LLMX_ONE_SHOT_BOUNDS __launch_bounds__(kThreads)
// A kernel's by-value struct parameter read in place, since a kernel taking its address otherwise copies it to local memory (docs/STATUS.md, CUDA prompt gap).
#define LLMX_GRID_CONST const __grid_constant__
// d += the 16 by 8 product of a (16 rows by 32 signed bytes) and b (32 signed bytes by 8 columns), exact in wrapping int32 (mma m16n8k32 s8).
// Lane l holds a's rows l / 4 and l / 4 + 8 at words l % 4 and l % 4 + 4, b's column l / 4 at the same words, and d's rows l / 4 and l / 4 + 8 at columns 2 * (l % 4) and the next.
__device__ __forceinline__ void mma_s8(int (&d)[4], const int (&a)[4], const int (&b)[2]) {
#if __CUDA_ARCH__ < 800
    const int g = (int)lane_index() / 4, c = 2 * ((int)lane_index() % 4);
    for (int w = 0; w < 4; ++w) {
        const int r0 = __shfl_sync(LLMX_FULL_WARP, a[0], 4 * g + w), r0h = __shfl_sync(LLMX_FULL_WARP, a[2], 4 * g + w);
        const int r8 = __shfl_sync(LLMX_FULL_WARP, a[1], 4 * g + w), r8h = __shfl_sync(LLMX_FULL_WARP, a[3], 4 * g + w);
        const int c0 = __shfl_sync(LLMX_FULL_WARP, b[0], 4 * c + w), c0h = __shfl_sync(LLMX_FULL_WARP, b[1], 4 * c + w);
        const int c1 = __shfl_sync(LLMX_FULL_WARP, b[0], 4 * c + 4 + w), c1h = __shfl_sync(LLMX_FULL_WARP, b[1], 4 * c + 4 + w);
        d[0] = __dp4a(r0h, c0h, __dp4a(r0, c0, d[0]));
        d[1] = __dp4a(r0h, c1h, __dp4a(r0, c1, d[1]));
        d[2] = __dp4a(r8h, c0h, __dp4a(r8, c0, d[2]));
        d[3] = __dp4a(r8h, c1h, __dp4a(r8, c1, d[3]));
    }
#else
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}
// d += the 16 by 8 product over 16 signed bytes (mma m16n8k16 s8): a0 holds a's row l / 4 and a1 its row l / 4 + 8 at word l % 4, b b's column l / 4 at that word, d as above.
__device__ __forceinline__ void mma_s8_k16(int (&d)[4], int a0, int a1, int b) {
#if __CUDA_ARCH__ < 800
    const int g = (int)lane_index() / 4, c = 2 * ((int)lane_index() % 4);
    for (int w = 0; w < 4; ++w) {
        const int r0 = __shfl_sync(LLMX_FULL_WARP, a0, 4 * g + w), r8 = __shfl_sync(LLMX_FULL_WARP, a1, 4 * g + w);
        const int c0 = __shfl_sync(LLMX_FULL_WARP, b, 4 * c + w), c1 = __shfl_sync(LLMX_FULL_WARP, b, 4 * c + 4 + w);
        d[0] = __dp4a(r0, c0, d[0]);
        d[1] = __dp4a(r0, c1, d[1]);
        d[2] = __dp4a(r8, c0, d[2]);
        d[3] = __dp4a(r8, c1, d[3]);
    }
#else
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a0), "r"(a1), "r"(b));
#endif
}
// Four 8 by 16-byte matrices from shared memory, lane l giving row l % 8 of matrix l / 8; r[j] gets matrix j's row l / 4 at bytes 4 * (l % 4) on (ldmatrix x4).
// Before sm_75, which lacks ldmatrix, each lane takes matrix j's row l / 4 address from lane 8 j + l / 4 and loads its word, the same bytes.
__device__ __forceinline__ void ldsm_x4(int (&r)[4], const void* row) {
#if __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned)__cvta_generic_to_shared(row)));
#else
    const unsigned l = lane_index();
    for (int j = 0; j < 4; ++j) {
        const char* p = (const char*)__shfl_sync(LLMX_FULL_WARP, (unsigned long long)row, 8 * j + l / 4);
        r[j] = *(const int*)(p + 4 * (l % 4));
    }
#endif
}

__device__ __forceinline__ void schedule_barrier() {}
__device__ __forceinline__ void spin_pause() { __nanosleep(64); }
// The global timer, in nanoseconds; the host reads its rate as 1 GHz.
__device__ __forceinline__ unsigned long long wall_clock() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Scoped atomics as PTX loads and stores: cta is the block, gpu the device, sys every device and the host.
#define LLMX_PTX_LOAD(fn, op)                                                                                       \
    template <typename P> __device__ __forceinline__ auto fn(P p) {                                                 \
        using T = std::remove_cv_t<std::remove_pointer_t<P>>;                                                       \
        static_assert(sizeof(T) == 4 || sizeof(T) == 8, "atomics cover 32 and 64-bit words");                       \
        if constexpr (sizeof(T) == 4) { unsigned v; asm volatile(op ".b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return (T)v; } \
        else { unsigned long long v; asm volatile(op ".b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return (T)v; } \
    }
#define LLMX_PTX_STORE(fn, op)                                                                                      \
    template <typename P, typename V> __device__ __forceinline__ void fn(P p, V v) {                                \
        using T = std::remove_cv_t<std::remove_pointer_t<P>>;                                                       \
        static_assert(sizeof(T) == 4 || sizeof(T) == 8, "atomics cover 32 and 64-bit words");                       \
        if constexpr (sizeof(T) == 4) asm volatile(op ".b32 [%0], %1;" :: "l"(p), "r"((unsigned)(T)v) : "memory");  \
        else asm volatile(op ".b64 [%0], %1;" :: "l"(p), "l"((unsigned long long)(T)v) : "memory");                 \
    }
LLMX_PTX_LOAD(atomic_load_relaxed_block, "ld.relaxed.cta")
LLMX_PTX_LOAD(atomic_load_relaxed_device, "ld.relaxed.gpu")
LLMX_PTX_LOAD(atomic_load_acquire_device, "ld.acquire.gpu")
LLMX_PTX_LOAD(atomic_load_relaxed_system, "ld.relaxed.sys")
LLMX_PTX_LOAD(atomic_load_acquire_system, "ld.acquire.sys")
LLMX_PTX_STORE(atomic_store_relaxed_block, "st.relaxed.cta")
LLMX_PTX_STORE(atomic_store_relaxed_system, "st.relaxed.sys")
LLMX_PTX_STORE(atomic_store_release_system, "st.release.sys")
#undef LLMX_PTX_LOAD
#undef LLMX_PTX_STORE
__device__ __forceinline__ void fence_acquire_device() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }
__device__ __forceinline__ void fence_acquire_system() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
// The grid barrier's steps (the HIP section has their contract).
__device__ __forceinline__ void lds_sync() { __syncthreads(); }
__device__ __forceinline__ void stores_sync() { __threadfence(); __syncthreads(); }
__device__ __forceinline__ void grid_arrive(unsigned* bar) { if (threadIdx.x == 0) atomicAdd(bar, 1u); }
__device__ __forceinline__ void grid_wait(unsigned* bar, unsigned target) {
    if (threadIdx.x == 0) while (atomic_load_acquire_device(bar) < target) spin_pause();
}
__device__ __forceinline__ void acquire_sync() { __syncthreads(); fence_acquire_device(); }
// CUDA reads no host array in device code, so each table read there is a device copy made from it at compile time.
template <typename T, unsigned N> struct gpu_table_of { T v[N]; };
template <typename T, unsigned N> constexpr gpu_table_of<T, N> gpu_table_copy(const T (&a)[N]) {
    gpu_table_of<T, N> t{};
    for (unsigned i = 0; i < N; ++i) t.v[i] = a[i];
    return t;
}
template <const auto& A> __device__ constexpr auto gpu_table_v = gpu_table_copy(A);
#define GPU_TABLE(a) (gpu_table_v<a>.v)

#define GPU_CONSTANT_SPACE
#define GPU_GLOBAL_SPACE
// The vector's lanes by index and, up to four, by name, laid out as N words at 4-byte alignment.
template <int N> struct gpu_words_of {
    unsigned w_[N];
    __device__ unsigned& operator[](int i) { return w_[i]; }
    __device__ unsigned operator[](int i) const { return w_[i]; }
};
template <> struct gpu_words_of<2> {
    union { unsigned w_[2]; struct { unsigned x, y; }; };
    __device__ unsigned& operator[](int i) { return w_[i]; }
    __device__ unsigned operator[](int i) const { return w_[i]; }
};
template <> struct gpu_words_of<3> {
    union { unsigned w_[3]; struct { unsigned x, y, z; }; };
    __device__ unsigned& operator[](int i) { return w_[i]; }
    __device__ unsigned operator[](int i) const { return w_[i]; }
};
template <> struct gpu_words_of<4> {
    union { unsigned w_[4]; struct { unsigned x, y, z, w; }; };
    __device__ unsigned& operator[](int i) { return w_[i]; }
    __device__ unsigned operator[](int i) const { return w_[i]; }
};
template <int N> using gpu_words = gpu_words_of<N>;
// Four words aligned to 16 bytes, the one width the kernels take whole, and its volatile load.
struct alignas(16) gpu_vector4 {
    union { unsigned w_[4]; struct { unsigned x, y, z, w; }; };
};
template <int N> using gpu_vector = std::enable_if_t<N == 4, gpu_vector4>;
__device__ __forceinline__ gpu_vector4 load_volatile(const gpu_vector4* p) {
    gpu_vector4 r;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p) : "memory");
    return r;
}

// tanh as (1 - e) / (1 + e) of e = expf(-2|x|) with an IEEE divide, since CUDA's tanhf flushes subnormals in its reciprocal and the kernels keep them (tests/cuda_sass.py).
__device__ __forceinline__ float gpu_tanhf(float x) {
    const float e = expf(-2.0f * fabsf(x));
    return copysignf((1.0f - e) / (1.0f + e), x);
}

#endif
