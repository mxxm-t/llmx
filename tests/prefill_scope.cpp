#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>
#include <stdexcept>
#include <thread>
#include "model/arch_qwen.hpp"

static thread_local bool before_scope = false, fail_body_allocation = false;

void* operator new(std::size_t size) {
    if (fail_body_allocation) {
        fail_body_allocation = false;
        throw std::bad_alloc();
    }
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
#if defined(__GNUC__)
__attribute__((noinline))
#endif
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { ::operator delete(p); }

namespace {
size_t contracts = 0, model_calls = 0, exact_values = 0;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

struct BodyError { int worker; };

void exact(const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float)),
            "scope changed model output");
    for (float v : b) require(std::isfinite(v), "nonfinite model output");
    exact_values += a.size();
}

void check_contract() {
    backend::CpuBackend cpu;
    const auto caller = std::this_thread::get_id();
    int calls = 0;
    cpu.backend::Backend::run_prefill([&] {
        require(std::this_thread::get_id() == caller, "default scope changed caller");
        ++calls;
    });
    require(calls == 1, "default scope did not complete exactly once");
    bool caught = false;
    try { cpu.backend::Backend::run_prefill([] { throw BodyError{17}; }); }
    catch (const BodyError& e) { caught = e.worker == 17; }
    require(caught, "default scope lost exception");
    ++contracts;

    for (int threads : {1, 6, 4, 6}) {
        cpu.set_threads(threads);
        std::vector<std::thread::id> before(size_t(threads), caller), after = before;
        cpu.run_parallel([&](int w) { before[size_t(w)] = std::this_thread::get_id(); });
        calls = 0;
        cpu.run_prefill([&] {
            ++calls;
            require(std::this_thread::get_id() == caller, "scope changed caller");
            cpu.set_threads(threads);
            if (threads == 1) cpu.set_threads(0);
            bool nested = false, changed = false;
            int inner_calls = 0;
            try { cpu.run_prefill([&] { ++inner_calls; }); }
            catch (const std::runtime_error&) { nested = true; }
            try { cpu.set_threads(threads == 6 ? 4 : 6); }
            catch (const std::runtime_error&) { changed = true; }
            require(nested && inner_calls == 0, "nested callback ran");
            require(changed && cpu.threads_available() == threads, "active thread count changed");
            cpu.run_parallel([&](int w) { after[size_t(w)] = std::this_thread::get_id(); });
        });
        require(calls == 1 && before == after, "scope count or worker identities changed");
        ++contracts;

        for (int failing : {0, threads - 1}) {
            std::atomic<int> entered{0}, completed{0};
            caught = false;
            try {
                cpu.run_prefill([&] {
                    cpu.run_parallel([&](int worker) {
                        ++entered;
                        while (entered.load() != threads) std::this_thread::yield();
                        ++completed;
                        if (worker == failing) throw BodyError{worker};
                    });
                });
            } catch (const BodyError& e) {
                caught = e.worker == failing;
                require(completed.load() == threads, "scope error escaped before dispatch drained");
            }
            require(caught, "scope lost worker exception");
            std::atomic<int> work{0};
            cpu.run_prefill([&] { cpu.parallel_for(37, [&](int) { ++work; }); });
            require(work.load() == 37, "scope or pool not reusable after error");
            ++contracts;
        }
        caught = false;
        try { cpu.run_prefill([] { throw BodyError{31}; }); }
        catch (const BodyError& e) { caught = e.worker == 31; }
        require(caught, "scope lost direct body exception");
        calls = 0;
        cpu.run_prefill([&] { ++calls; });
        require(calls == 1, "scope guard remained active after direct error");
        ++contracts;
    }
}

gguf::GGUFModel fixture(bool tied) {
    gguf::GGUFModel m;
    for (const auto& kv : std::vector<std::pair<std::string, uint64_t>>{
            {"block_count", 1}, {"embedding_length", 8}, {"feed_forward_length", 12},
            {"attention.head_count", 2}, {"attention.head_count_kv", 1},
            {"attention.key_length", 4}, {"context_length", 32}}) {
        gguf::MetaValue v; v.vtype = gguf::V_UINT32; v.u = kv.second;
        m.kv.push_back({"qwen3." + kv.first, v});
    }
    auto add = [&](const std::string& name, std::vector<uint64_t> shape, bool norm = false) {
        size_t count = 1;
        for (uint64_t d : shape) count *= size_t(d);
        const size_t offset = m.blob.size();
        m.blob.resize(offset + count * sizeof(float));
        for (size_t i = 0; i < count; ++i) {
            const float v = norm ? 1.0f : float(int((i * 17 + m.tensors.size() * 3) % 29) - 14) / 64.0f;
            std::memcpy(m.blob.data() + offset + i * sizeof(float), &v, sizeof(v));
        }
        m.tensors.push_back({name, std::move(shape), gguf::GGML_TYPE_F32, 0});
        m.offsets.push_back(offset);
    };
    add("token_embd.weight", {8, 16});
    add("output_norm.weight", {8}, true);
    for (const char* name : {"attn_norm", "ffn_norm"})
        add(std::string("blk.0.") + name + ".weight", {8}, true);
    for (const char* name : {"attn_q_norm", "attn_k_norm"})
        add(std::string("blk.0.") + name + ".weight", {4}, true);
    add("blk.0.attn_q.weight", {8, 8});
    add("blk.0.attn_k.weight", {8, 4});
    add("blk.0.attn_v.weight", {8, 4});
    add("blk.0.attn_output.weight", {8, 8});
    add("blk.0.ffn_gate.weight", {8, 12});
    add("blk.0.ffn_up.weight", {8, 12});
    add("blk.0.ffn_down.weight", {12, 8});
    if (!tied) add("output.weight", {8, 16});
    return m;
}

struct ObservedCpu : backend::CpuBackend {
    int entries = 0, bodies = 0;
    bool inside = false, passthrough = false, check_graph = false, fail_allocation = false;
    size_t early_allocs = 0;
    std::vector<size_t> batches;

    // Activations are one backend allocation, so counting the calls is exact.
    // Matching their byte size instead stopped detecting anything the moment
    // the nine vectors became one arena.
    backend::BufferPtr alloc(size_t bytes, backend::Memory where) override {
        if (before_scope) ++early_allocs;
        return backend::CpuBackend::alloc(bytes, where);
    }

    void run_prefill(const std::function<void()>& work) override {
        before_scope = false;
        ++entries;
        auto body = [&] {
            ++bodies;
            inside = true;
            fail_body_allocation = fail_allocation;
            try { work(); }
            catch (...) { inside = false; fail_body_allocation = false; throw; }
            inside = false;
            fail_body_allocation = false;
        };
        if (passthrough) backend::Backend::run_prefill(std::ref(body));
        else backend::CpuBackend::run_prefill(std::ref(body));
    }

    void matmul_group(std::initializer_list<backend::Projection> p, backend::CSlice x,
                      size_t nin, size_t nbatch, backend::RowRuns = {}) override {
        if (check_graph) {
            require(inside, "model projection outside prefill scope");
            if (p.size() == 3) batches.push_back(nbatch);
        }
        backend::CpuBackend::matmul_group(p, x, nin, nbatch);
    }

    void matmul(uint32_t type, backend::CSlice data, backend::CSlice x, backend::Slice y,
                size_t nin, size_t nout, size_t nbatch, backend::RowRuns = {}) override {
        if (check_graph) require(inside, "model matmul outside prefill scope");
        backend::CpuBackend::matmul(type, data, x, y, nin, nout, nbatch);
    }
};

void check_model(bool tied) {
    const auto weights = fixture(tied);
    auto cpu = std::make_shared<ObservedCpu>(), reference = std::make_shared<ObservedCpu>();
    cpu->set_threads(6); reference->set_threads(6); reference->passthrough = true;
    infer::Model model(weights, cpu), control(weights, reference);
    model.set_ubatch(2); control.set_ubatch(2);
    bool caught = false;
    try { model.prefill({}); } catch (const std::runtime_error&) { caught = true; }
    require(caught && cpu->entries == 0 && model.n_tokens() == 0, "empty prompt entered scope");

    const std::vector<uint32_t> first{1, 2, 3};
    const size_t initial_early = cpu->early_allocs;
    caught = false;
    cpu->fail_allocation = true;
    before_scope = true;
    try { model.prefill(first); } catch (const std::bad_alloc&) { caught = true; }
    before_scope = false;
    require(cpu->early_allocs == initial_early, "initial batch buffers allocated before scope entry");
    require(caught && cpu->entries == 1 && cpu->bodies == 1 && !cpu->inside && model.n_tokens() == 0,
            "first body allocation failure escaped scope contract");
    cpu->fail_allocation = false;

    for (const auto& prompt : {std::vector<uint32_t>{1, 2, 3}, {4}, {7, 6, 5, 4, 3, 2, 1}}) {
        model.reset(); control.reset();
        const int entries = cpu->entries, bodies = cpu->bodies;
        cpu->batches.clear(); cpu->check_graph = true;
        const size_t early = cpu->early_allocs;
        before_scope = true;
        const auto actual = model.prefill(prompt);
        before_scope = false;
        require(cpu->early_allocs == early, "batch buffers allocated before scope entry");
        exact(control.prefill(prompt), actual);
        require(cpu->entries == entries + 1 && cpu->bodies == bodies + 1, "prefill not wrapped exactly once");
        std::vector<size_t> expected;
        for (size_t i = 0; i < prompt.size(); i += 2) expected.push_back(std::min(size_t(2), prompt.size() - i));
        require(cpu->batches == expected && model.n_tokens() == int(prompt.size()), "physical batches changed");
        exact(control.prefill({8}), model.prefill({8}));
        require(cpu->entries == entries + 2 && cpu->bodies == bodies + 2, "follow-up scope count");
        cpu->check_graph = false;
        exact(control.step(9), model.step(9));
        require(cpu->entries == entries + 2, "decode entered prefill scope");
        model_calls += 2;
    }
}
}

int main() {
    try {
        quant::register_builtins();
        check_contract();
        check_model(false); check_model(true);
        std::cout << "prefill scope: " << contracts << " contracts, " << model_calls
                  << " model scopes, " << exact_values << " exact finite logits\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
