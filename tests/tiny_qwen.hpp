#pragma once
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "backends/cpu/cpu_backend.hpp"
#include "format/gguf.hpp"

// A Qwen3-shaped F32 model with deterministic weights, 8 wide with two query heads over one KV head, for the native tests that run a forward pass.
// Without `tied` it has an output projection of its own.
inline gguf::GGUFModel tiny_qwen(int layers, uint64_t context, bool tied) {
    gguf::GGUFModel m;
    for (const auto& kv : std::vector<std::pair<std::string, uint64_t>>{
            {"block_count", uint64_t(layers)}, {"embedding_length", 8}, {"feed_forward_length", 12},
            {"attention.head_count", 2}, {"attention.head_count_kv", 1},
            {"attention.key_length", 4}, {"context_length", context}}) {
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
    for (int l = 0; l < layers; ++l) {
        const std::string pre = "blk." + std::to_string(l) + ".";
        for (const char* name : {"attn_norm", "ffn_norm"}) add(pre + name + ".weight", {8}, true);
        for (const char* name : {"attn_q_norm", "attn_k_norm"}) add(pre + name + ".weight", {4}, true);
        add(pre + "attn_q.weight", {8, 8});
        add(pre + "attn_k.weight", {8, 4});
        add(pre + "attn_v.weight", {8, 4});
        add(pre + "attn_output.weight", {8, 8});
        add(pre + "ffn_gate.weight", {8, 12});
        add(pre + "ffn_up.weight", {8, 12});
        add(pre + "ffn_down.weight", {12, 8});
    }
    if (!tied) add("output.weight", {8, 16});
    return m;
}

// A CPU backend that fails once where a test arms it and counts what the passes ask of it, for the tests of a failed pass on these models.
// `fail_output` fails the next output projection, the only 16-row matmul, after every layer's KV has been written; `fail_attention` fails the Nth attention from now, after its layer's KV has been written.
// `histories` collects the committed length each attention finds for the pass's first sequence, which is how a test reads back the history of this backend's storage.
struct FailingCpu : backend::CpuBackend {
    bool fail_output = false;
    int fail_attention = 0;
    int outputs = 0;
    int syncs = 0, submits = 0, waits = 0, reads = 0;
    backend::Ticket last_wait = 0;
    std::vector<size_t> histories;
    void matmul(uint32_t type, backend::CSlice data, backend::CSlice x, backend::Slice y,
                size_t nin, size_t nout, size_t nbatch, backend::RowRuns runs = {}) override {
        if (nout == 16) {
            ++outputs;
            if (fail_output) { fail_output = false; throw std::runtime_error("injected"); }
        }
        backend::CpuBackend::matmul(type, data, x, y, nin, nout, nbatch, runs);
    }
    void attention(backend::CSlice q, size_t layer, const backend::KVView* views, size_t n_views, backend::Slice out,
                   int n_head, int n_head_kv, int head_dim) override {
        if (n_views) histories.push_back(views[0].length);
        if (fail_attention && --fail_attention == 0) throw std::runtime_error("injected");
        backend::CpuBackend::attention(q, layer, views, n_views, out, n_head, n_head_kv, head_dim);
    }
    void sync() noexcept override { ++syncs; backend::CpuBackend::sync(); }
    backend::Ticket submit() override { ++submits; return backend::CpuBackend::submit(); }
    void wait(backend::Ticket t) noexcept override {
        ++waits;
        last_wait = t;
        backend::CpuBackend::wait(t);
    }
    void read(const backend::Buffer& src, size_t off, void* dst, size_t bytes) override {
        ++reads;
        backend::CpuBackend::read(src, off, dst, bytes);
    }
};
