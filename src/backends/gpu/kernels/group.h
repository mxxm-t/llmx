// The tensor group's device-side layout and stop rules, shared by the sum kernels (kernels/f32.gpu) and the sublayer entry's take (kernels/int16.gpu).
#pragma once
#include "backends/gpu/kernels/port.h"

namespace {
// A tensor group's sum (GpuCollective, docs/src/backends-gpu.md): each member's area holds this block of signals, then a staging slot for each member's partial rows.
// kGroupMembers is the host's backend::kMaxTensorWidth, kGroupBlocks, kChainBlocks and kChainUnits the host's of the same names (collective_core.hpp).
constexpr int kGroupBlocks = 32, kGroupMembers = 16, kChainBlocks = 16;
constexpr unsigned long long kChainUnits = 4096;   // float4 units a chain's largest chunk
constexpr unsigned long long kChainDepth = 8, kChainLeast = 256;   // chunks a chain block aims for, and the fewest float4 units a chunk
// A chain's chunk: halved from kChainUnits until each of the F blocks a side gets kChainDepth chunks, so a member starts on a chunk while its predecessor sends the next.
__device__ inline unsigned long long group_chain_units(unsigned long long units, unsigned F) {
    unsigned long long cu = kChainUnits;
    while (cu > kChainLeast && (units + cu - 1) / cu < kChainDepth * F) cu /= 2;
    return cu;
}
struct GroupSignal {
    unsigned epoch[kGroupBlocks];   // the last sum's epoch in a chain, per block, first in the area
    unsigned abort;                 // set by a member whose wait ran out, read by every member's waits
    unsigned done[3][kGroupBlocks][kGroupMembers];   // [meeting][block][sender]: the epoch a sender's block reached, at a chain's meeting or a draws sum's first or second
    unsigned ahead[kChainBlocks], back[kChainBlocks];     // the chunks a chain's predecessor and successor passed this member so far, per chain block
    unsigned passed[2][kChainBlocks];                     // this member's own count of the same, ahead then back, which every member keeps equal
};
static_assert(sizeof(GroupSignal) <= 8192, "the host's kGroupSignalBytes");
struct GroupPeers {
    float* stage[kGroupMembers];      // each member's staging slots, cap floats a member, two parity halves of them and under peer writes its two draw slots
    unsigned* signal[kGroupMembers];  // each member's GroupSignal
    unsigned* status;                 // host memory, a word a member, set where its sum failed
    unsigned* fault;                  // host memory, this member's backend's fault word
    unsigned* cancel;                 // host memory, set by the host where a peer's sum will never launch
    unsigned* flush;                  // this member's word whose write makes its own stores visible to peers' reads of its memory (ROCm's HDP flush), or null where none is needed
};
static_assert(sizeof(GroupPeers) == 288, "the host's GroupPeers");  // in device memory, written once at join, so the launch does not grow with kGroupMembers

// A fast one-shot's staging slot holds the signalling NaN kGroupArmed until its sender's value lands; a sender's value of those bits goes as the quiet kGroupQuiet, which any add makes of it (group_probe checks that on each device).
constexpr unsigned kGroupArmed = 0x7f800001u, kGroupQuiet = 0x7fc00001u;
constexpr unsigned kGroupSpinChecks = 64;   // spins of a slot between two looks at the stop words
typedef gpu_vector<4> GroupWord4;
__device__ inline unsigned group_sane(float v) {
    const unsigned x = __float_as_uint(v);
    return x == kGroupArmed ? kGroupQuiet : x;
}
__device__ inline float4 group_sane4(float4 v) {
    return make_float4(__uint_as_float(group_sane(v.x)), __uint_as_float(group_sane(v.y)), __uint_as_float(group_sane(v.z)), __uint_as_float(group_sane(v.w)));
}

// Whether a wait gives up: a member set abort, or, read once a tick, the host set cancel or a probe's deadline passed.
__device__ inline bool group_stop(const GroupPeers* p, GroupSignal* self, unsigned long long deadline, unsigned long long tick, unsigned long long* next, int* failed) {
    if (atomic_load_relaxed_system(&self->abort)) {
        atomic_store_relaxed_block(failed, 1);
        return true;
    }
    // The host's cancel word is read once a tick, so a long wait does not stream reads of host memory.
    const unsigned long long now = (unsigned long long)wall_clock();
    if (now < *next) return false;
    *next = now + tick;
    if (now > deadline || atomic_load_relaxed_system(p->cancel)) {
        atomic_store_relaxed_block(failed, 1);
        return true;
    }
    return false;
}

// A failed wait: every member stops waiting, and the host learns it from the status and fault words.
__device__ inline void group_failed(const GroupPeers* p, int members, int rank) {
    if (threadIdx.x) return;
    for (int k = 0; k < members; ++k) atomic_store_relaxed_system(&((GroupSignal*)p->signal[k])->abort, 1u);
    atomic_store_relaxed_system(&p->status[rank], 1u);
    atomic_store_relaxed_system(p->fault, 1u);
}

// Floats [i, i + count) of a slot armed again, count at most 4; the group kernels keep their tails rolled loops, so cuda-sass reads no unrolled tail as a split store.
__device__ inline void group_arm_unit(float* slot, unsigned long long i, unsigned long long n) {
    if (i + 4 <= n) {
        store_vector((GroupWord4*)(slot + i), GroupWord4{kGroupArmed, kGroupArmed, kGroupArmed, kGroupArmed});
        return;
    }
#pragma unroll 1
    for (unsigned long long k = i; k < n && k < i + 4; ++k) ((unsigned*)slot)[k] = kGroupArmed;
}

// Unit i of n floats (i a multiple of 4) from 16-byte aligned rows, the floats past n zero; and stored to them, the floats past n left alone.
__device__ inline float4 group_load(const float* src, unsigned long long i, unsigned long long n) {
    if (i + 4 <= n) return *(const float4*)(src + i);
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned long long k = i; k < n; ++k) v[k - i] = src[k];
    return make_float4(v[0], v[1], v[2], v[3]);
}
__device__ inline void group_store(float* dst, unsigned long long i, unsigned long long n, float4 v) {
    if (i + 4 <= n) {
        store_vector((gpu_vector<4>*)(dst + i), gpu_vector<4>{__float_as_uint(v.x), __float_as_uint(v.y), __float_as_uint(v.z), __float_as_uint(v.w)});
        return;
    }
    const float w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll 1
    for (unsigned long long k = i; k < n && k < i + 4; ++k) dst[k] = w[k - i];
}
// out at unit i = t (replace) or out + t, as the one-shot writes it.
__device__ inline void group_out(float* out, unsigned long long i, unsigned long long n, float4 t, bool replace, bool aligned) {
    if (i + 4 <= n && aligned) {
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
        return;
    }
    const float w[4] = {t.x, t.y, t.z, t.w};
#pragma unroll 1
    for (unsigned long long k = i; k < n && k < i + 4; ++k) {
        if (replace) out[k] = w[k - i];
        else out[k] += w[k - i];
    }
}
// Waits until no float below n of unit i of a peer's slot is armed, giving up as group_stop says or once a thread of the block has, then takes the unit and arms it again.
__device__ inline bool group_take(const GroupPeers* p, GroupSignal* self, float* slot, unsigned long long i, unsigned long long n, unsigned long long deadline, unsigned long long tick,
                                  unsigned long long* next, int* failed, float4* got) {
    GroupWord4 w;
    for (unsigned s = 1;; ++s) {
        if (i + 4 <= n) {
            w = load_volatile((const GroupWord4*)(slot + i));
        } else {
            unsigned v[4] = {0u, 0u, 0u, 0u};
            for (unsigned long long k = i; k < n; ++k) v[k - i] = ((const volatile unsigned*)slot)[k];
            w = GroupWord4{v[0], v[1], v[2], v[3]};
        }
        if (w.x != kGroupArmed && w.y != kGroupArmed && w.z != kGroupArmed && w.w != kGroupArmed) break;
        if (s % kGroupSpinChecks == 0 && (atomic_load_relaxed_block(failed) || group_stop(p, self, deadline, tick, next, failed))) return false;
    }
    *got = make_float4(__uint_as_float(w.x), __uint_as_float(w.y), __uint_as_float(w.z), __uint_as_float(w.w));
    group_arm_unit(slot, i, n);
    return true;
}

// The one-shot's take of unit i: the members' rows added in member order into out, each peer's unit armed again once read; false when a wait gave up.
__device__ __forceinline__ bool group_take_unit(const GroupPeers* p, GroupSignal* self, const float* own, float* mine, float* out, unsigned long long i, unsigned long long n, unsigned long long cap, int members,
                                       int rank, bool replace, bool aligned, unsigned long long deadline, unsigned long long tick, unsigned long long* next, int* failed) {
    float4 t = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    bool met = true;
    for (int j = 0; j < members && met; ++j) {
        float4 v;
        if (j == rank) v = group_load(own, i, n);
        else met = group_take(p, self, mine + j * cap, i, n, deadline, tick, next, failed, &v);
        if (!j) {
            t = v;
        } else {
            t.x += v.x;
            t.y += v.y;
            t.z += v.z;
            t.w += v.w;
        }
    }
    if (met) group_out(out, i, n, t, replace, aligned);
    return met;
}
} // namespace
