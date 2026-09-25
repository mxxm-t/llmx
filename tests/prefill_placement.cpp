#define NOMINMAX
#include <windows.h>
#include <atomic>
#include <vector>
#include <string>
#include <iostream>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

namespace hooks {
const auto real_set = &SetThreadAffinityMask;
const auto real_thread = &GetThreadGroupAffinity;
const auto real_process = &GetProcessAffinityMask;
const auto real_topology = &GetLogicalProcessorInformationEx;
const auto real_groups = &GetActiveProcessorGroupCount;
enum Mode { Normal, MultiGroup, TopologyFail, ProcessFail, SetupThrow,
            QueryOneFail, ApplyOneFail, RestoreOnce, RestoreAlways, RestoreQueryFail };
std::atomic<int> mode{Normal}, applies{0}, restores{0}, apply_ok{0}, restore_ok{0};
std::atomic<int> failed_restores{0};
thread_local int worker = -1, restore_failures = 0;
thread_local DWORD_PTR old = 0;
thread_local bool active = false, just_restored = false;
bool simulated = false;
int core_count = 6;
thread_local DWORD_PTR simulated_mask = 63;
// Synthetic topology exercises failure paths without changing real OS affinity.
BOOL raw_thread(HANDLE h, PGROUP_AFFINITY a) {
    if (!simulated) return real_thread(h, a);
    *a = {}; a->Mask = simulated_mask;
    return TRUE;
}
DWORD_PTR raw_set(HANDLE h, DWORD_PTR target) {
    if (!simulated) return real_set(h, target);
    if (!target || (target & ~((DWORD_PTR(1) << core_count) - 1))) return 0;
    const DWORD_PTR previous = simulated_mask;
    simulated_mask = target;
    return previous;
}
WORD WINAPI group_count() { return mode == MultiGroup ? 2 : (simulated ? 1 : real_groups()); }
BOOL WINAPI process_mask(HANDLE h, PDWORD_PTR a, PDWORD_PTR s) {
    if (mode == ProcessFail) { SetLastError(ERROR_ACCESS_DENIED); return FALSE; }
    if (simulated) { *a = *s = (DWORD_PTR(1) << core_count) - 1; return TRUE; }
    return real_process(h, a, s);
}
BOOL WINAPI topology_info(LOGICAL_PROCESSOR_RELATIONSHIP kind,
                         PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX data, PDWORD bytes) {
    if (mode == SetupThrow) throw std::bad_alloc();
    if (mode == TopologyFail) { SetLastError(ERROR_NOT_SUPPORTED); return FALSE; }
    if (simulated) {
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX> cores(size_t(core_count), SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX{});
        for (int i = 0; i < core_count; ++i) {
            cores[size_t(i)].Relationship = RelationProcessorCore;
            cores[size_t(i)].Size = sizeof(cores[0]);
            cores[size_t(i)].Processor.GroupCount = 1;
            cores[size_t(i)].Processor.GroupMask[0].Mask = DWORD_PTR(1) << i;
        }
        const DWORD required = DWORD(cores.size() * sizeof(cores[0]));
        if (!data || *bytes < required) {
            *bytes = required; SetLastError(ERROR_INSUFFICIENT_BUFFER); return FALSE;
        }
        std::memcpy(data, cores.data(), required); *bytes = required;
        return TRUE;
    }
    return real_topology(kind, data, bytes);
}
BOOL WINAPI thread_mask(HANDLE h, PGROUP_AFFINITY a) {
    if ((mode == QueryOneFail && worker == 3) || (mode == RestoreQueryFail && just_restored)) {
        SetLastError(ERROR_ACCESS_DENIED); return FALSE;
    }
    return raw_thread(h, a);
}
DWORD_PTR WINAPI set_mask(HANDLE h, DWORD_PTR target) {
    const bool restoring = active && target == old;
    if (restoring) ++restores; else ++applies;
    const bool fail = restoring ? (mode == RestoreAlways ||
        (mode == RestoreOnce && restore_failures++ == 0)) : (mode == ApplyOneFail && worker == 3);
    GROUP_AFFINITY before{};
    if (!raw_thread(h, &before)) throw std::runtime_error("test raw mask read failed");
    DWORD_PTR result = 0;
    if (fail) { if (restoring) ++failed_restores; SetLastError(ERROR_ACCESS_DENIED); }
    else result = raw_set(h, target);
    if (result) {
        if (restoring) { ++restore_ok; active = false; just_restored = true; }
        else { ++apply_ok; old = result; active = true; just_restored = false; }
    }
    return result;
}
void reset_thread(int id) { worker = id; old = 0; active = false; just_restored = false; restore_failures = 0; }
}
#define GetActiveProcessorGroupCount hooks::group_count
#define GetProcessAffinityMask hooks::process_mask
#define GetLogicalProcessorInformationEx hooks::topology_info
#define GetThreadGroupAffinity hooks::thread_mask
#define SetThreadAffinityMask hooks::set_mask
#include "backends/cpu/cpu_backend.hpp"
#undef GetActiveProcessorGroupCount
#undef GetProcessAffinityMask
#undef GetLogicalProcessorInformationEx
#undef GetThreadGroupAffinity
#undef SetThreadAffinityMask

static size_t cases = 0, exact_values = 0;
static void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
using Masks = std::vector<std::pair<WORD, DWORD_PTR>>;
static Masks masks(backend::CpuBackend& cpu) {
    Masks result(size_t(cpu.threads_available()));
    cpu.run_parallel([&](int w) {
        GROUP_AFFINITY a{};
        require(hooks::raw_thread(GetCurrentThread(), &a), "raw thread query");
        result[size_t(w)] = {a.Group, a.Mask};
    });
    return result;
}
static void reset(backend::CpuBackend& cpu, hooks::Mode mode) {
    cpu.run_parallel([](int w) { hooks::reset_thread(w); });
    hooks::mode = mode;
    hooks::applies = 0; hooks::restores = 0; hooks::apply_ok = 0;
    hooks::restore_ok = 0; hooks::failed_restores = 0;
}
static void rescue(backend::CpuBackend& cpu, const Masks& original) {
    hooks::mode = hooks::Normal;
    cpu.run_parallel([&](int w) {
        require(hooks::raw_set(GetCurrentThread(), original[size_t(w)].second) != 0, "manual rescue failed");
        hooks::reset_thread(w);
    });
    require(masks(cpu) == original, "manual rescue mask mismatch");
}
struct RescueMasks {
    backend::CpuBackend& cpu;
    Masks original;
    explicit RescueMasks(backend::CpuBackend& value) : cpu(value), original(masks(value)) {}
    ~RescueMasks() {
        try {
            original.resize(size_t(cpu.threads_available()));
            if (masks(cpu) != original) rescue(cpu, original);
        }
        catch (...) { std::terminate(); }
    }
};
static void record(const std::string& name) {
    ++cases;
    std::cout << (hooks::simulated ? "synthetic: " : "native: ") << name << " [ok]\n";
}
static void exact(const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), "matrix output differs");
    exact_values += a.size();
}
struct Matrix {
    size_t n = 64, rows = 49, batch = 3;
    std::vector<float> weights, x;
    std::vector<uint8_t> q8;
    Matrix() : weights(rows * n), x(batch * n), q8(rows * 2 * 34) {
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i * 13 % 53) - 26) / 31.0f;
        for (size_t i = 0; i < rows; ++i) quant::quantize_row_q8_0(weights.data() + i * n, q8.data() + i * 68, 2);
        input(1);
    }
    void input(int seed) { for (size_t i = 0; i < x.size(); ++i) x[i] = float((int(i) * 7 + seed * 11) % 37 - 18) / float(seed + 19); }
    std::vector<float> run(backend::CpuBackend& cpu, bool f32) {
        std::vector<float> y(rows * batch + 2, 123456.0f);
        const void* rows_ptr = f32 ? (const void*)weights.data() : (const void*)q8.data();
        const size_t bytes = f32 ? weights.size() * sizeof(float) : q8.size();
        const auto wb = cpu.adopt(rows_ptr, bytes);
        const auto xb = cpu.adopt(x.data(), x.size() * sizeof(float));
        const auto yb = cpu.adopt(y.data(), y.size() * sizeof(float));
        cpu.matmul(f32 ? gguf::GGML_TYPE_F32 : gguf::GGML_TYPE_Q8_0,
            {wb.get(), 0}, {xb.get(), 0}, {yb.get(), 1}, n, rows, batch);
        require(y.front() == 123456.0f && y.back() == 123456.0f, "output guards changed");
        for (size_t i = 1; i + 1 < y.size(); ++i) require(std::isfinite(y[i]), "nonfinite output");
        return {y.begin() + 1, y.end() - 1};
    }
};
struct BodyError { int worker; };

static void lifecycle() {
    backend::CpuBackend cpu; cpu.set_threads(6);
    if (hooks::simulated) cpu.run_parallel([](int) { hooks::simulated_mask = 63; });
    RescueMasks cleanup(cpu);
    const Masks original = masks(cpu); Masks bound;
    reset(cpu, hooks::Normal);
    int calls = 0;
    const DWORD caller = GetCurrentThreadId();
    cpu.run_prefill([&] {
        ++calls; require(GetCurrentThreadId() == caller, "body changed caller");
        bound = masks(cpu); DWORD_PTR seen = 0;
        for (size_t i = 0; i < bound.size(); ++i) {
            require(bound[i].first == 0 && bound[i].second && !(bound[i].second & (bound[i].second - 1)) && !(seen & bound[i].second) &&
                    (bound[i].second & original[i].second) == bound[i].second, "body not wholly on distinct allowed singleton masks");
            seen |= bound[i].second;
        }
    });
    require(calls == 1 && hooks::apply_ok == 6 && hooks::restore_ok == 6 && masks(cpu) == original, "phase transition count/restoration");
    record("whole-phase-six-participants");
    Matrix matrix;
    for (int threads : {1, 6, 4, 6}) for (bool f32 : {false, true}) {
        cpu.set_threads(threads); auto saved = masks(cpu); matrix.input(threads + int(f32));
        const auto reference = matrix.run(cpu, f32);
        reset(cpu, hooks::Normal); calls = 0;
        cpu.run_prefill([&] { ++calls; exact(reference, matrix.run(cpu, f32)); });
        require(calls == 1 && masks(cpu) == saved, "reconfigured phase not restored");
        require(hooks::apply_ok == (threads == 6 ? 6 : 0) && hooks::restore_ok == hooks::apply_ok, "phase eligibility count");
        record(std::string(f32 ? "f32-" : "q8-") + std::to_string(threads));
    }
    const Masks refreshed = masks(cpu);
    for (auto mode : {hooks::MultiGroup, hooks::TopologyFail, hooks::ProcessFail, hooks::QueryOneFail, hooks::ApplyOneFail}) {
        reset(cpu, mode); calls = 0;
        cpu.run_prefill([&] { ++calls; require(masks(cpu) == refreshed && hooks::apply_ok == hooks::restore_ok, "body entered with partial placement"); });
        require(calls == 1 && masks(cpu) == refreshed, "fallback did not preserve original masks");
        if (mode == hooks::ApplyOneFail) require(hooks::apply_ok > 0, "partial-apply failure path not exercised");
        record("clean-fallback-" + std::to_string(mode));
    }
    reset(cpu, hooks::Normal);
    const DWORD_PTR restricted = refreshed[0].second & ~bound[0].second;
    require(restricted && hooks::raw_set(GetCurrentThread(), restricted), "restrict caller failed");
    const Masks custom = masks(cpu); calls = 0;
    cpu.run_prefill([&] { ++calls; require(masks(cpu) == custom, "restricted participant caused partial/widened placement"); });
    require(calls == 1 && masks(cpu) == custom && hooks::apply_ok == hooks::restore_ok, "restricted fallback cleanup");
    require(hooks::raw_set(GetCurrentThread(), refreshed[0].second), "restore test caller restriction");
    record("restricted-mask-all-or-clean-fallback");

    reset(cpu, hooks::SetupThrow); calls = 0; bool caught = false;
    try { cpu.run_prefill([&] { ++calls; }); } catch (const std::bad_alloc&) { caught = true; }
    require(caught && !calls && !hooks::apply_ok && masks(cpu) == refreshed, "setup failure mutated or invoked body");
    record("setup-throw-before-mutation");
    for (int failing : {-1, 0, 3}) {
        reset(cpu, hooks::Normal); calls = 0; caught = false; std::atomic<int> finished{0};
        try { cpu.run_prefill([&] {
            ++calls;
            if (failing == -1) throw BodyError{-1};
            cpu.run_parallel([&](int w) { ++finished; if (w == failing) throw BodyError{w}; });
        }); } catch (const BodyError& e) { caught = e.worker == failing; }
        require(caught && calls == 1 && masks(cpu) == refreshed && hooks::apply_ok == hooks::restore_ok, "body exception cleanup");
        if (failing != -1) require(finished == 6, "body task escaped before all workers finished");
        record("body-error-" + std::to_string(failing));
    }
    for (auto mode : {hooks::RestoreOnce, hooks::RestoreAlways, hooks::RestoreQueryFail}) {
        reset(cpu, mode); calls = 0; caught = false;
        try { cpu.run_prefill([&] { ++calls; }); }
        catch (const std::exception& e) { caught = std::string(e.what()).find("restore") != std::string::npos; }
        require(caught && calls == 1, "cleanup failure hidden");
        if (mode == hooks::RestoreAlways) {
            require(masks(cpu) != refreshed && hooks::failed_restores >= 6, "persistent refusal falsely recovered");
            rescue(cpu, refreshed);
        } else require(masks(cpu) == refreshed, "recoverable cleanup not restored");
        record("restore-error-" + std::to_string(mode));
        reset(cpu, hooks::Normal); calls = 0;
        cpu.run_prefill([&] { ++calls; });
        require(calls == 1 && masks(cpu) == refreshed, "reuse after verified restoration/test rescue");
        record("reuse-after-verified-restore-or-test-rescue");
    }
    reset(cpu, hooks::RestoreOnce); caught = false;
    try { cpu.run_prefill([] { throw BodyError{-1}; }); }
    catch (const std::exception& e) { caught = std::string(e.what()).find("restore") != std::string::npos; }
    require(caught && masks(cpu) == refreshed, "cleanup error precedence or retry");
    record("dual-body-and-cleanup-failure");
    reset(cpu, hooks::Normal); calls = 0; cpu.run_prefill([&] { ++calls; });
    require(calls == 1 && masks(cpu) == refreshed, "final scope reuse");
    record("final-reuse");
}

static void fallback(const std::string& name) {
    backend::CpuBackend cpu; cpu.set_threads(6);
    if (hooks::simulated) cpu.run_parallel([](int) {
        hooks::simulated_mask = (DWORD_PTR(1) << hooks::core_count) - 1;
    });
    RescueMasks cleanup(cpu);
    reset(cpu, hooks::Normal);
    const auto original = masks(cpu);
    int calls = 0;
    cpu.run_prefill([&] { ++calls; require(masks(cpu) == original, "fallback changes masks"); });
    require(calls == 1 && hooks::applies == 0 && hooks::restores == 0 && masks(cpu) == original,
            "unsupported topology must preserve scheduler placement");
    record(name);
}

int main() {
    try {
        hooks::simulated = true;
        for (int cores : {2, 4}) {
            hooks::core_count = cores;
            fallback("insufficient-cores-" + std::to_string(cores));
        }
        hooks::core_count = 6;
        lifecycle();
        hooks::simulated = false;
        hooks::mode = hooks::Normal;
        backend::detail::PrefillPlacement topology(true);
        if (topology.enabled()) lifecycle();
        else fallback("actual-topology-ineligible");
        std::cout << "Windows prefill placement: " << cases << " cases, " << exact_values
                  << " exact values; persistent injected failures use test-only rescue\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
