#pragma once
// A CPU backend that plays a device loading weights: it copies what it adopts, its buffers hide their host bytes, and a copy or write stays outstanding until wait or sync, so a test sees a buffer freed before its upload retired (`premature`).
// Failures can be injected at the Nth adoption, allocation or write, and at the cache.
#include <memory>
#include <stdexcept>
#include "backends/cpu/cpu_backend.hpp"

struct LoadingState {
    int adoptions = 0, allocations = 0, weights = 0, writes = 0, drains = 0, releases = 0, premature = 0;
    bool pending = false;
};

// Model loading sees a device-like buffer whose outstanding upload ends only at wait or sync.
struct LoadingBuffer : backend::Buffer {
    backend::BufferPtr storage;
    std::shared_ptr<LoadingState> state;
    LoadingBuffer(backend::BufferPtr b, std::shared_ptr<LoadingState> s)
        : storage(std::move(b)), state(std::move(s)) {}
    ~LoadingBuffer() override {
        ++state->releases;
        if (state->pending) ++state->premature;
    }
    size_t size() const override { return storage->size(); }
    const void* host_ptr() const override { return nullptr; }
};

struct LoadingBackend : backend::CpuBackend {
    std::shared_ptr<LoadingState> state = std::make_shared<LoadingState>();
    int fail_adopt = 0, fail_alloc = 0, fail_write = 0;
    bool fail_cache = false;
    // It plays a device, whose buffers hide their host bytes, so it copies what it adopts.
    bool reads_in_place() const override { return false; }
    backend::BufferPtr adopt(const void* src, size_t bytes) override {
        if (++state->adoptions == fail_adopt) throw std::runtime_error("injected adoption failure");
        auto storage = backend::CpuBackend::alloc(bytes, backend::Memory::device);
        backend::CpuBackend::write(*storage, 0, src, bytes);
        auto buffer = std::make_shared<LoadingBuffer>(std::move(storage), state);
        state->pending = true;
        return buffer;
    }
    backend::BufferPtr alloc(size_t bytes, backend::Memory where) override {
        if (++state->allocations == fail_alloc) throw std::runtime_error("injected window allocation failure");
        auto buffer = std::make_shared<LoadingBuffer>(backend::CpuBackend::alloc(bytes, where), state);
        state->pending = true;
        return buffer;
    }
    // A loader's weight storage, filled by write; the write's copy is outstanding until wait or sync.
    backend::BufferPtr alloc_weight(size_t bytes) override {
        ++state->weights;
        return std::make_shared<LoadingBuffer>(backend::CpuBackend::alloc(bytes, backend::Memory::device), state);
    }
    void write(backend::Buffer& dst, size_t off, const void* src, size_t bytes) override {
        if (++state->writes == fail_write) throw std::runtime_error("injected write failure");
        auto* wrapped = dynamic_cast<LoadingBuffer*>(&dst);
        backend::CpuBackend::write(wrapped ? *wrapped->storage : dst, off, src, bytes);
        state->pending = true;
    }
    // A copy is outstanding as a write is, and one out of host memory it reads in place, which a streamed load makes in place of a write, counts as one.
    void copy(backend::Buffer& dst, size_t dst_off, const backend::Buffer& src, size_t src_off, size_t bytes) override {
        auto* wrapped = dynamic_cast<LoadingBuffer*>(&dst);
        const auto* from = dynamic_cast<const LoadingBuffer*>(&src);
        if (!from && ++state->writes == fail_write) throw std::runtime_error("injected write failure");
        backend::CpuBackend::copy(wrapped ? *wrapped->storage : dst, dst_off, from ? *from->storage : src, src_off, bytes);
        state->pending = true;
    }
    void sync() noexcept override { ++state->drains; state->pending = false; }
    void wait(backend::Ticket) noexcept override { sync(); }
    std::unique_ptr<backend::KVStorage> kv_alloc(size_t layers, size_t heads, size_t dim, size_t tokens,
                                               backend::KVType k, backend::KVType v) override {
        if (fail_cache) throw std::runtime_error("injected cache allocation failure");
        return backend::CpuBackend::kv_alloc(layers, heads, dim, tokens, k, v);
    }
};
