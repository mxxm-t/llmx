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
    unsigned done[kGroupBlocks][kGroupMembers];   // [block][sender]: the epoch a sender's block reached
    unsigned ahead[kChainBlocks], back[kChainBlocks];     // the chunks a chain's predecessor and successor passed this member so far, per chain block
    unsigned passed[2][kChainBlocks];                     // this member's own count of the same, ahead then back, which every member keeps equal
};
static_assert(sizeof(GroupSignal) <= 8192, "the host's kGroupSignalBytes");
struct GroupPeers {
    float* stage[kGroupMembers];      // each member's staging slots, cap floats a member
    unsigned* signal[kGroupMembers];  // each member's GroupSignal
    unsigned* status;                 // host memory, a word a member, set where its sum failed
    unsigned* fault;                  // host memory, this member's backend's fault word
    unsigned* cancel;                 // host memory, set by the host where a peer's sum will never launch
};
static_assert(sizeof(GroupPeers) == 280, "the host's GroupPeers");  // in device memory, written once at join, so the launch does not grow with kGroupMembers

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
} // namespace
