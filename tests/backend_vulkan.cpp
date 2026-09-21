// Vulkan backend (docs/VULKAN.md). Sub-step 1: storage and submission over
// a real device. Buffers round-trip through adopt, copy, write and read;
// allocations come back zeroed; host-visible memory is readable in place
// after a wait; tickets are monotonic and retire in order. Sub-step 2: the
// elementwise kernels, gather, embed and the norms against the CPU backend
// on random inputs, with the bounds below fixed before the first run: bit
// exact where the arithmetic is the same operation in the same order (add,
// gather, embed), and a stated relative tolerance where a transcendental or
// a reduction order differs. Exits 77, which CTest reports as skipped, when
// there is no loader or no device.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include "backends/cpu/cpu_backend.hpp"
#include "backends/vulkan/vulkan_backend.hpp"
#include "quant/quant.hpp"

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

std::vector<float> uniform(size_t n, uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

// The same op on both backends over the same inputs. Inputs are adopted
// (the CPU keeps the pointer, the device uploads a copy); outputs are
// allocated on each and the device's is read back.
struct Pair {
    backend::CpuBackend cpu;
    backend::Backend& vk;
    explicit Pair(backend::Backend& v) : vk(v) { cpu.set_threads(1); }
    struct In {
        backend::BufferPtr c, v;
        backend::CSlice cs() const { return {c.get(), 0}; }
        backend::CSlice vs() const { return {v.get(), 0}; }
    };
    struct Out {
        backend::BufferPtr c, v;
        size_t n;
        backend::Slice cs() const { return {c.get(), 0}; }
        backend::Slice vs() const { return {v.get(), 0}; }
    };
    In in(const void* data, size_t bytes) {
        return {cpu.adopt(data, bytes), vk.adopt(data, bytes)};
    }
    In in(const std::vector<float>& f) { return in(f.data(), f.size() * sizeof(float)); }
    Out out(size_t floats) {
        return {cpu.alloc(floats * sizeof(float), backend::Memory::device),
                vk.alloc(floats * sizeof(float), backend::Memory::device), floats};
    }
    // Both results, with the device's read back.
    std::pair<std::vector<float>, std::vector<float>> results(const Out& o) {
        std::vector<float> a(o.n), b(o.n);
        cpu.read(*o.c, 0, a.data(), o.n * sizeof(float));
        vk.read(*o.v, 0, b.data(), o.n * sizeof(float));
        return {a, b};
    }
};

size_t exact(const std::vector<float>& a, const std::vector<float>& b, const char* what) {
    require(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0, what);
    for (float v : b) require(std::isfinite(v), "nonfinite device output");
    return a.size();
}

size_t close(const std::vector<float>& a, const std::vector<float>& b, double rel, const char* what) {
    require(a.size() == b.size(), what);
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(b[i]), "nonfinite device output");
        require(std::fabs((double)a[i] - b[i]) <= rel * (1.0 + std::fabs((double)a[i])), what);
    }
    return a.size();
}

size_t check_kernels(backend::Backend& vk) {
    Pair p(vk);
    size_t values = 0;

    // add: same operation in the same order, so exact.
    {
        const size_t n = 100003;
        const auto dst0 = uniform(n, 1), src = uniform(n, 2);
        Pair::In s = p.in(src);
        Pair::Out d = p.out(n);
        p.cpu.write(*d.c, 0, dst0.data(), n * sizeof(float));
        p.vk.write(*d.v, 0, dst0.data(), n * sizeof(float));
        p.cpu.add(d.cs(), s.cs(), n);
        p.vk.add(d.vs(), s.vs(), n);
        auto r = p.results(d);
        values += exact(r.first, r.second, "add differs");
    }
    // silu_mul: exp differs between libm and the device, so a tolerance.
    {
        const size_t n = 50001;
        const auto g = uniform(n, 3, -6.0f, 6.0f), u = uniform(n, 4);
        Pair::In gi = p.in(g), ui = p.in(u);
        Pair::Out d = p.out(n);
        p.cpu.silu_mul(d.cs(), gi.cs(), ui.cs(), n);
        p.vk.silu_mul(d.vs(), gi.vs(), ui.vs(), n);
        auto r = p.results(d);
        values += close(r.first, r.second, 1e-6, "silu_mul differs beyond 1e-6");
    }
    // gather_rows: copies, so exact; a row beyond the source is refused.
    {
        const size_t width = 37, rows = 5;
        const auto src = uniform(width * rows, 5);
        const uint32_t pick[4] = {4, 1, 1, 0};
        Pair::In s = p.in(src);
        Pair::Out d = p.out(width * 4);
        p.cpu.gather_rows(d.cs(), s.cs(), width, pick, 4);
        p.vk.gather_rows(d.vs(), s.vs(), width, pick, 4);
        auto r = p.results(d);
        values += exact(r.first, r.second, "gather_rows differs");
        const uint32_t beyond[1] = {5};
        bool rejected = false;
        try { p.vk.gather_rows(d.vs(), s.vs(), width, beyond, 1); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "gather beyond the source accepted");
    }
    // rms_norm_rows and rms_norm: the sum of squares is reduced in a
    // different order, so a tolerance; dst aliasing src is exercised.
    {
        const size_t rows = 7, n = 1000, stride = 1013;
        const auto src = uniform(rows * stride, 6), w = uniform(n, 7, 0.5f, 1.5f);
        Pair::In s = p.in(src), wi = p.in(w);
        Pair::Out d = p.out(rows * stride);
        p.cpu.rms_norm_rows(d.cs(), s.cs(), wi.cs(), rows, n, stride, 1e-6f);
        p.vk.rms_norm_rows(d.vs(), s.vs(), wi.vs(), rows, n, stride, 1e-6f);
        auto r = p.results(d);
        for (size_t row = 0; row < rows; ++row) {
            std::vector<float> a(r.first.begin() + row * stride, r.first.begin() + row * stride + n);
            std::vector<float> b(r.second.begin() + row * stride, r.second.begin() + row * stride + n);
            values += close(a, b, 1e-5, "rms_norm_rows differs beyond 1e-5");
        }
        Pair::Out one = p.out(n);
        p.cpu.write(*one.c, 0, src.data(), n * sizeof(float));
        p.vk.write(*one.v, 0, src.data(), n * sizeof(float));
        p.cpu.rms_norm(one.cs(), one.cs(), wi.cs(), n, 1e-6f);
        p.vk.rms_norm(one.vs(), one.vs(), wi.vs(), n, 1e-6f);
        auto q = p.results(one);
        values += close(q.first, q.second, 1e-5, "rms_norm in place differs beyond 1e-5");
    }
    // norm_rope_rows: positions per row out of order, a padded stride.
    {
        const size_t rows = 5, heads = 3, half = 64, dim = 2 * half, stride = heads * dim + 8, table = 12;
        const auto x = uniform(rows * stride, 8), w = uniform(dim, 9, 0.5f, 1.5f);
        std::vector<float> cs(table * half), sn(table * half);
        for (size_t t = 0; t < table; ++t)
            for (size_t i = 0; i < half; ++i) {
                const double f = std::pow(10000.0, -2.0 * double(i) / double(dim));
                cs[t * half + i] = float(std::cos(double(t) * f));
                sn[t * half + i] = float(std::sin(double(t) * f));
            }
        const uint32_t pos[rows] = {5, 2, 9, 0, 11};
        Pair::In wi = p.in(w), ci = p.in(cs), si = p.in(sn);
        Pair::Out d = p.out(rows * stride);
        p.cpu.write(*d.c, 0, x.data(), x.size() * sizeof(float));
        p.vk.write(*d.v, 0, x.data(), x.size() * sizeof(float));
        p.cpu.norm_rope_rows(d.cs(), rows, stride, heads, wi.cs(), 1e-6f, ci.cs(), si.cs(), half, pos);
        p.vk.norm_rope_rows(d.vs(), rows, stride, heads, wi.vs(), 1e-6f, ci.vs(), si.vs(), half, pos);
        auto r = p.results(d);
        values += close(r.first, r.second, 1e-5, "norm_rope_rows differs beyond 1e-5");
        const uint32_t beyond[1] = {12};
        bool rejected = false;
        try { p.vk.norm_rope_rows(d.vs(), 1, stride, heads, wi.vs(), 1e-6f, ci.vs(), si.vs(), half, beyond); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "position beyond the table accepted");
    }
    // embed: F32 rows are copies and Q8_0 rows are a half scale times a
    // small integer, exact in float, so both are exact.
    {
        const size_t nin = 96, nrows = 10;
        const auto table = uniform(nin * nrows, 10);
        const uint32_t ids[3] = {3, 9, 0};
        Pair::In t = p.in(table);
        Pair::Out d = p.out(nin * 3);
        p.cpu.embed(d.cs(), gguf::GGML_TYPE_F32, t.cs(), nin, nrows, ids, 3);
        p.vk.embed(d.vs(), gguf::GGML_TYPE_F32, t.vs(), nin, nrows, ids, 3);
        auto r = p.results(d);
        values += exact(r.first, r.second, "embed F32 differs");

        std::vector<uint8_t> q(nrows * (nin / gguf::Q8_0_BLOCK) * gguf::Q8_0_TYPESIZE);
        for (size_t row = 0; row < nrows; ++row)
            quant::quantize_row_q8_0(table.data() + row * nin,
                                     q.data() + row * (nin / gguf::Q8_0_BLOCK) * gguf::Q8_0_TYPESIZE,
                                     nin / gguf::Q8_0_BLOCK);
        Pair::In tq = p.in(q.data(), q.size());
        Pair::Out dq = p.out(nin * 3);
        p.cpu.embed(dq.cs(), gguf::GGML_TYPE_Q8_0, tq.cs(), nin, nrows, ids, 3);
        p.vk.embed(dq.vs(), gguf::GGML_TYPE_Q8_0, tq.vs(), nin, nrows, ids, 3);
        auto rq = p.results(dq);
        values += exact(rq.first, rq.second, "embed Q8_0 differs");
        const uint32_t beyond[1] = {10};
        bool rejected = false;
        try { p.vk.embed(d.vs(), gguf::GGML_TYPE_F32, t.vs(), nin, nrows, beyond, 1); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "embedding row beyond the table accepted");
    }
    // matmul: F32 and Q8_0 over odd sizes and batch widths that fall
    // inside, on and past the eight-column chunk. The reduction order
    // differs from the CPU's, so a tolerance.
    for (size_t nin : {size_t(256), size_t(224)}) {
        // 256 is eight blocks, the word-wide path; 224 is seven, the 16-bit path.
        const size_t nout = 67;
        const auto wf = uniform(nin * nout, 11);
        std::vector<uint8_t> wq(nout * (nin / gguf::Q8_0_BLOCK) * gguf::Q8_0_TYPESIZE);
        for (size_t row = 0; row < nout; ++row)
            quant::quantize_row_q8_0(wf.data() + row * nin,
                                     wq.data() + row * (nin / gguf::Q8_0_BLOCK) * gguf::Q8_0_TYPESIZE,
                                     nin / gguf::Q8_0_BLOCK);
        Pair::In wfi = p.in(wf), wqi = p.in(wq.data(), wq.size());
        for (size_t nbatch : {size_t(1), size_t(3), size_t(8), size_t(13)}) {
            const auto x = uniform(nbatch * nin, 12 + (uint32_t)nbatch);
            Pair::In xi = p.in(x);
            for (int q = 0; q < 2; ++q) {
                const uint32_t type = q ? gguf::GGML_TYPE_Q8_0 : gguf::GGML_TYPE_F32;
                const Pair::In& wi = q ? wqi : wfi;
                Pair::Out d = p.out(nbatch * nout);
                p.cpu.matmul(type, wi.cs(), xi.cs(), d.cs(), nin, nout, nbatch);
                p.vk.matmul(type, wi.vs(), xi.vs(), d.vs(), nin, nout, nbatch);
                auto r = p.results(d);
                values += close(r.first, r.second, 1e-4, q ? "Q8_0 matmul differs beyond 1e-4"
                                                            : "F32 matmul differs beyond 1e-4");
            }
        }
        bool rejected = false;
        try { p.vk.matmul(gguf::GGML_TYPE_Q4_0, wqi.vs(), wqi.vs(), p.out(8).vs(), nin, 1, 1); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "unsupported matrix type accepted");
    }
    // Decode bandwidth of the row kernel on a Qwen3-8B-sized projection,
    // reported and not asserted: 4096 x 4096 Q8_0 is 17 MiB per column.
    {
        const size_t n = 4096;
        std::vector<uint8_t> wq(n * (n / gguf::Q8_0_BLOCK) * gguf::Q8_0_TYPESIZE);
        for (size_t i = 0; i < wq.size(); ++i) wq[i] = uint8_t(i * 7 + 3);
        const auto x = uniform(n, 13);
        const auto w = vk.adopt(wq.data(), wq.size());
        const auto xb = vk.adopt(x.data(), x.size() * sizeof(float));
        const auto y = vk.alloc(n * sizeof(float), backend::Memory::device);
        vk.matmul(gguf::GGML_TYPE_Q8_0, {w.get(), 0}, {xb.get(), 0}, {y.get(), 0}, n, n, 1);
        vk.sync();
        const int iters = 50;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i)
            vk.matmul(gguf::GGML_TYPE_Q8_0, {w.get(), 0}, {xb.get(), 0}, {y.get(), 0}, n, n, 1);
        vk.sync();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
        std::cout << "backend-vulkan: Q8_0 matvec 4096x4096 " << ms << " ms, "
                  << (double)wq.size() / ms / 1e6 << " GB/s\n";
    }
    return values;
}

std::vector<uint8_t> pattern(size_t bytes, uint32_t seed) {
    std::vector<uint8_t> v(bytes);
    uint32_t x = seed;
    for (size_t i = 0; i < bytes; ++i) {
        x = x * 1664525u + 1013904223u;
        v[i] = uint8_t(x >> 24);
    }
    return v;
}
}

int main() {
    backend::BackendPtr b;
    try {
        b = backend::make_vulkan_backend(0);
    } catch (const backend::VulkanUnavailable& e) {
        std::cout << "backend-vulkan: skipped: " << e.what() << "\n";
        return 77;
    }
    try {
        std::cout << "backend-vulkan: " << backend::vulkan_device_name(*b) << "\n";
        size_t checks = 0;

        // Allocations are zeroed, on the device and in host-visible memory.
        const size_t mib = 1u << 20;
        const auto zero = b->alloc(mib);
        std::vector<uint8_t> out(mib, 0xff);
        b->read(*zero, 0, out.data(), mib);
        for (uint8_t v : out) require(v == 0, "device allocation not zeroed");
        require(zero->host_ptr() == nullptr, "device memory reported a host address");
        const auto visible = b->alloc(4096, backend::Memory::host_visible);
        require(visible->host_ptr() != nullptr, "host-visible memory has no host address");
        b->sync();
        for (size_t i = 0; i < 4096; ++i)
            require(((const uint8_t*)visible->host_ptr())[i] == 0, "host-visible allocation not zeroed");
        checks += 2;

        // Adopt uploads a copy; the source may change afterwards.
        std::vector<uint8_t> src = pattern(3 * mib + 12345, 1);
        const std::vector<uint8_t> kept = src;
        const auto adopted = b->adopt(src.data(), src.size());
        std::fill(src.begin(), src.end(), uint8_t(0));
        out.assign(kept.size(), 0);
        b->read(*adopted, 0, out.data(), kept.size());
        require(out == kept, "adopted bytes differ after upload");
        checks += 1;

        // Copy within the device at odd offsets, then read a window.
        const auto dst = b->alloc(kept.size() + 100);
        b->copy(*dst, 100, *adopted, 0, kept.size());
        b->copy(*dst, 0, *adopted, 7, 100);
        out.assign(kept.size() + 100, 0);
        b->read(*dst, 0, out.data(), out.size());
        require(std::memcmp(out.data() + 100, kept.data(), kept.size()) == 0, "device copy differs");
        require(std::memcmp(out.data(), kept.data() + 7, 100) == 0, "offset copy differs");
        std::vector<uint8_t> window(1000);
        b->read(*dst, 100 + 2 * mib + 3, window.data(), window.size());
        require(std::memcmp(window.data(), kept.data() + 2 * mib + 3, window.size()) == 0,
                "windowed read differs");
        checks += 3;

        // Write from the host into device memory, then into host-visible
        // memory, and a copy whose result the host reads in place.
        const std::vector<uint8_t> patch = pattern(777, 2);
        b->write(*dst, 5000, patch.data(), patch.size());
        b->read(*dst, 5000, window.data(), patch.size());
        require(std::memcmp(window.data(), patch.data(), patch.size()) == 0, "device write differs");
        b->write(*visible, 8, patch.data(), 1000);
        b->copy(*visible, 2000, *adopted, 4096, 2000);
        const backend::Ticket t = b->submit();
        b->wait(t);
        require(std::memcmp((const uint8_t*)visible->host_ptr() + 8, patch.data(), 1000) == 0,
                "host-visible write differs");
        require(std::memcmp((const uint8_t*)visible->host_ptr() + 2000, kept.data() + 4096, 2000) == 0,
                "copy into host-visible memory not visible after wait");
        checks += 3;

        // Tickets are monotonic and every one retires.
        const backend::Ticket t1 = b->submit(), t2 = b->submit();
        require(t2 > t1 && t1 > t, "tickets are not monotonic");
        b->wait(t2);
        b->wait(t1);
        b->sync();
        b->copy(*dst, 0, *adopted, 0, 64);
        b->sync();
        b->read(*dst, 0, window.data(), 64);
        require(std::memcmp(window.data(), kept.data(), 64) == 0, "work before sync not retired");
        checks += 2;

        // Empty allocations and ranges outside an allocation.
        const auto empty = b->alloc(0);
        require(empty->size() == 0, "empty allocation has a size");
        b->read(*empty, 0, window.data(), 0);
        bool rejected = false;
        try { b->read(*dst, kept.size() + 100 - 10, window.data(), 11); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "read past the allocation accepted");
        rejected = false;
        try { b->copy(*dst, 0, *adopted, kept.size(), 1); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "copy past the source accepted");
        checks += 2;

        quant::register_builtins();
        const size_t values = check_kernels(*b);
        std::cout << "backend-vulkan: " << checks << " storage and submission checks; "
                  << values << " kernel outputs against the CPU backend\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
