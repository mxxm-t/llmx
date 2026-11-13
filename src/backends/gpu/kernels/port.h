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

// The value of lane (this lane xor m), or of lane (this lane + d), within groups of width lanes, the whole wave where no width is given.
template <typename T> __device__ __forceinline__ auto shuffle_xor(T v, int m) { return __shfl_xor(v, m); }
template <typename T> __device__ __forceinline__ auto shuffle_xor(T v, int m, int width) { return __shfl_xor(v, m, width); }
template <typename T> __device__ __forceinline__ auto shuffle_down(T v, int d) { return __shfl_down(v, d); }
template <typename T> __device__ __forceinline__ auto shuffle_down(T v, int d, int width) { return __shfl_down(v, d, width); }

// The value of lane (this lane xor M) for M = 1 or 2 by DPP quad_perm within each group of four lanes: selector 0xB1 is [1,0,3,2] (xor 1), 0x4E is [2,3,0,1] (xor 2).
// Both pass row mask 0xF, bank mask 0xF and bound_ctrl false, so every lane writes and reads its partner.
template <int M, typename T> __device__ __forceinline__ auto lane_xor_quad(T v) {
    static_assert(M == 1 || M == 2, "quad_perm exchanges cover xor 1 and 2");
    return __builtin_amdgcn_mov_dpp(v, M == 1 ? 0xB1 : 0x4E, 0xF, 0xF, false);
}
// The value of lane (this lane xor M) for M = 4, 8 or 16 by ds_swizzle's bit mode within groups of 32 lanes: offset 0x1F | (M << 10) is and-mask 0x1F, or-mask 0, xor-mask M.
template <int M, typename T> __device__ __forceinline__ auto lane_xor_swizzle(T v) {
    static_assert(M == 4 || M == 8 || M == 16, "swizzle exchanges cover xor 4, 8 and 16");
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

#define LLMX_FULL_WARP 0xffffffffu
__device__ __forceinline__ unsigned segment_lanes(int width) {
    unsigned lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));
    return port_segment_mask(width, lane);
}
template <typename T> __device__ __forceinline__ auto shuffle_xor(T v, int m) { return __shfl_xor_sync(LLMX_FULL_WARP, v, m); }
template <typename T> __device__ __forceinline__ auto shuffle_xor(T v, int m, int width) { return __shfl_xor_sync(segment_lanes(width), v, m, width); }
template <typename T> __device__ __forceinline__ auto shuffle_down(T v, int d) { return __shfl_down_sync(LLMX_FULL_WARP, v, d); }
template <typename T> __device__ __forceinline__ auto shuffle_down(T v, int d, int width) { return __shfl_down_sync(segment_lanes(width), v, d, width); }
template <int M, typename T> __device__ __forceinline__ auto lane_xor_quad(T v) {
    static_assert(M == 1 || M == 2, "quad_perm exchanges cover xor 1 and 2");
    return __shfl_xor_sync(LLMX_FULL_WARP, v, M);
}
template <int M, typename T> __device__ __forceinline__ auto lane_xor_swizzle(T v) {
    static_assert(M == 4 || M == 8 || M == 16, "swizzle exchanges cover xor 4, 8 and 16");
    return __shfl_xor_sync(LLMX_FULL_WARP, v, M);
}
template <typename T> __device__ __forceinline__ auto first_lane(T v) { return __shfl_sync(LLMX_FULL_WARP, v, 0); }
__device__ __forceinline__ unsigned long long wave_ballot(int p) { return __ballot_sync(LLMX_FULL_WARP, p); }

__device__ __forceinline__ unsigned byte_align(unsigned hi, unsigned lo, unsigned n) { return __funnelshift_r(lo, hi, (n & 3u) * 8u); }
__device__ __forceinline__ unsigned bit_align(unsigned hi, unsigned lo, unsigned s) { return __funnelshift_r(lo, hi, s); }
__device__ __forceinline__ unsigned byte_perm(unsigned hi, unsigned lo, unsigned sel) {
    const PortPrmt p = port_prmt(sel);
    return (__byte_perm(lo, hi, p.nibbles) & ~(p.clear | p.ones)) | p.ones;
}

__device__ __forceinline__ int dot4_i8(int a, int b, int s) { return __dp4a(a, b, s); }
__device__ __forceinline__ int dot2_i16(int a, int b, int s) { return port_dot2_i16(a, b, s); }

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

#endif
