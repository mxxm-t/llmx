#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace backend {
namespace detail {

class PrefillPlacement {
    std::array<uintptr_t, 6> targets_{};
#if defined(_WIN32)
    struct Participant {
        DWORD_PTR old = 0;
        DWORD thread = 0;
        bool changed = false;
        bool verified = false;
    };
    std::array<Participant, 6> participants_{};
#endif
public:
    explicit PrefillPlacement(bool enabled) {
#if defined(_WIN32)
        if (!enabled || GetActiveProcessorGroupCount() != 1) return;
        DWORD_PTR allowed = 0, system = 0;
        if (!GetProcessAffinityMask(GetCurrentProcess(), &allowed, &system) ||
            !allowed || (allowed & ~system)) return;
        DWORD bytes = 0;
        if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes) ||
            GetLastError() != ERROR_INSUFFICIENT_BUFFER || !bytes) return;
        std::vector<unsigned char> storage(bytes);
        if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(storage.data()), &bytes) ||
            bytes > storage.size()) return;
        std::vector<uintptr_t> eligible;
        DWORD_PTR seen = 0;
        int efficiency = -1;
        for (size_t offset = 0; offset < bytes;) {
            if (bytes - offset < 8) return;
            const auto* info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(storage.data() + offset);
            if (info->Size < offsetof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Processor.GroupMask) + sizeof(GROUP_AFFINITY) ||
                info->Size > bytes - offset || info->Relationship != RelationProcessorCore) return;
            const auto& core = info->Processor;
            if (core.GroupCount != 1 || core.GroupMask[0].Group != 0) return;
            if (efficiency < 0) efficiency = core.EfficiencyClass;
            if (efficiency != core.EfficiencyClass) return;
            const DWORD_PTR mask = core.GroupMask[0].Mask;
            if (!mask || (mask & seen)) return;
            seen |= mask;
            const uintptr_t usable = mask & allowed;
            if (usable) eligible.push_back(usable & (~usable + 1));
            offset += info->Size;
        }
        if ((allowed & ~seen) || eligible.size() < targets_.size()) return;
        std::sort(eligible.begin(), eligible.end());
        std::copy_n(eligible.begin(), targets_.size(), targets_.begin());
#else
        (void)enabled;
#endif
    }

    PrefillPlacement(const PrefillPlacement&) = delete;
    PrefillPlacement& operator=(const PrefillPlacement&) = delete;

    bool enabled() const { return targets_[0] != 0; }

    void apply(int worker) {
#if defined(_WIN32)
        const uintptr_t target = targets_[size_t(worker)];
        GROUP_AFFINITY before{};
        if (!target || !GetThreadGroupAffinity(GetCurrentThread(), &before) ||
            before.Group != 0 || (target & before.Mask) != target) return;
        auto& participant = participants_[size_t(worker)];
        participant.thread = GetCurrentThreadId();
        participant.old = SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(target));
        participant.changed = participant.old != 0;
        if (!participant.changed) return;
        GROUP_AFFINITY after{};
        participant.verified = GetThreadGroupAffinity(GetCurrentThread(), &after) &&
            after.Group == 0 && after.Mask == target;
#else
        (void)worker;
#endif
    }

    bool applied() const {
#if defined(_WIN32)
        return std::all_of(participants_.begin(), participants_.end(),
            [](const Participant& p) { return p.changed && p.verified; });
#else
        return false;
#endif
    }

    bool needs_restore() const {
#if defined(_WIN32)
        return std::any_of(participants_.begin(), participants_.end(),
            [](const Participant& p) { return p.changed; });
#else
        return false;
#endif
    }

    void restore(int worker) {
#if defined(_WIN32)
        auto& participant = participants_[size_t(worker)];
        if (!participant.changed) return;
        if (GetCurrentThreadId() != participant.thread)
            throw std::runtime_error("CPU prefill restore participant changed");
        if (!SetThreadAffinityMask(GetCurrentThread(), participant.old))
            throw std::runtime_error("CPU prefill placement restore failed");
        participant.changed = false;
        GROUP_AFFINITY after{};
        if (!GetThreadGroupAffinity(GetCurrentThread(), &after) ||
            after.Group != 0 || after.Mask != participant.old)
            throw std::runtime_error("CPU prefill restored mask verification failed");
#else
        (void)worker;
#endif
    }
};

}
}
