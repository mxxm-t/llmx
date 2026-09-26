// Byte-identity harness for the CPU kernels: hashes every output of the matmul, group, expert and gather paths over types, shapes, thread counts and both decode dot modes.
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "backends/cpu/cpu_backend.hpp"

using backend::CpuBackend;

static uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*)p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static std::vector<uint8_t> packed(uint32_t type, size_t rows, size_t nin, std::mt19937& rng) {
    if (type == gguf::GGML_TYPE_F32) {
        std::vector<uint8_t> out(rows * nin * 4);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (size_t i = 0; i < rows * nin; ++i) { float v = u(rng); std::memcpy(&out[i * 4], &v, 4); }
        return out;
    }
    const quant::QuantType* qt = quant::Registry::instance().get(type);
    const size_t blocks = rows * nin / qt->block_size;
    std::vector<uint8_t> out(blocks * qt->type_size);
    if (qt->quantize) {
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        std::vector<float> f(rows * nin);
        for (float& v : f) v = u(rng);
        qt->quantize(f.data(), out.data(), blocks);
        return out;
    }
    for (uint8_t& b : out) b = uint8_t(rng());
    for (size_t b = 0; b < blocks; ++b) {
        uint8_t* p = out.data() + b * qt->type_size;
        if (type == gguf::GGML_TYPE_Q6_K) { p[208] = 0x00; p[209] = 0x14; }
        else { p[0] = 0x00; p[1] = 0x14; p[2] = 0x00; p[3] = 0x10; }
    }
    return out;
}

static size_t cases = 0;
static void out(const std::string& name, const std::vector<float>& v) {
    std::printf("%s %016llx\n", name.c_str(), (unsigned long long)fnv(v.data(), v.size() * sizeof(float)));
    ++cases;
}

int main() {
    const uint32_t types[] = {gguf::GGML_TYPE_F32, gguf::GGML_TYPE_Q8_0, gguf::GGML_TYPE_Q4_0, gguf::GGML_TYPE_Q4_1,
                              gguf::GGML_TYPE_Q4_K, gguf::GGML_TYPE_Q5_K, gguf::GGML_TYPE_Q6_K};
    for (uint32_t type : types) {
        const bool k = type == gguf::GGML_TYPE_Q4_K || type == gguf::GGML_TYPE_Q5_K || type == gguf::GGML_TYPE_Q6_K;
        std::vector<size_t> nins = k ? std::vector<size_t>{256, 4096} : std::vector<size_t>{96, 1024, 4096};
        if (type == gguf::GGML_TYPE_F32) nins = {100, 1024};
        for (size_t nin : nins) for (size_t nout : {5, 37, 300}) for (int threads : {1, 6, 16}) for (int d8 : {1, 0}) for (int huge : {0, 1}) try {
            std::mt19937 rng(type * 7919u + (unsigned)(nin * 31 + nout));
            const size_t n_expert = 4, kk = 2, rows = 4;
            const auto w = packed(type, n_expert * nout, nin, rng);
            const auto w2 = packed(type, nout + 3, nin, rng);
            std::uniform_real_distribution<float> u(-1.0f, 1.0f);
            std::vector<float> x(8 * nin);
            for (size_t i = 0; i < x.size(); ++i) x[i] = u(rng) * (huge ? (i % 7 == 0 ? 3e37f : 1e30f) : (i % 97 == 0 ? 40.0f : 1.0f));
            CpuBackend cpu;
            cpu.set_threads(threads);
            cpu.set_decode_activations8(d8 != 0);
            const auto wb = cpu.adopt(w.data(), w.size()), w2b = cpu.adopt(w2.data(), w2.size());
            const auto xb = cpu.adopt(x.data(), x.size() * sizeof(float));
            const std::string tag = std::to_string(type) + "/" + std::to_string(nin) + "/" + std::to_string(nout) + "/t" +
                                    std::to_string(threads) + "/d" + std::to_string(d8) + "/h" + std::to_string(huge);
            auto run = [&](const char* name, size_t nbatch, backend::RowRuns runs) {
                std::vector<float> y(nbatch * nout, -7.0f);
                const auto yb = cpu.adopt(y.data(), y.size() * sizeof(float));
                cpu.matmul(type, {wb.get(), 0}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, nbatch, runs);
                out(tag + "/" + name, y);
                std::vector<float> ya(nbatch * nout, 0.25f);
                const auto yab = cpu.adopt(ya.data(), ya.size() * sizeof(float));
                cpu.matmul_add(type, {wb.get(), 0}, {xb.get(), 0}, {yab.get(), 0}, nin, nout, nbatch, runs);
                out(tag + "/" + name + "+add", ya);
            };
            const backend::RowRun dec3[3] = {{1, 1}, {2, 1}, {3, 1}}, pr5[1] = {{5, 512}}, mix4[3] = {{1, 1}, {3, 512}, {4, 1}};
            run("one", 1, {});
            run("dec3", 3, {dec3, 3});
            run("pr5", 5, {pr5, 1});
            run("mix4", 4, {mix4, 3});
            run("nr3", 3, {});
            {
                std::vector<float> ya(2 * nout, -1.0f), yb2(2 * (nout + 3), -1.0f);
                const auto a = cpu.adopt(ya.data(), ya.size() * sizeof(float)), b = cpu.adopt(yb2.data(), yb2.size() * sizeof(float));
                const backend::RowRun dec2[2] = {{1, 1}, {2, 1}};
                cpu.matmul_group({{type, {wb.get(), 0}, {a.get(), 0}, nout}, {type, {w2b.get(), 0}, {b.get(), 0}, nout + 3}},
                                 {xb.get(), 0}, nin, 2, {dec2, 2});
                out(tag + "/group", ya);
                out(tag + "/group2", yb2);
            }
            {
                std::vector<float> ids(rows * kk), weights(rows * kk);
                std::vector<float> scores(rows * n_expert);
                for (float& v : scores) v = u(rng);
                const auto sb = cpu.adopt(scores.data(), scores.size() * sizeof(float));
                const auto ib = cpu.adopt(ids.data(), ids.size() * sizeof(float)), gb = cpu.adopt(weights.data(), weights.size() * sizeof(float));
                cpu.route_experts({sb.get(), 0}, rows, n_expert, kk, true, {ib.get(), 0}, {gb.get(), 0});
                out(tag + "/route", ids);
                out(tag + "/routew", weights);
                const backend::Backend::Routing routing{{ib.get(), 0}, {gb.get(), 0}, kk, n_expert};
                const backend::RowRun rd[4] = {{1, 1}, {2, 1}, {3, 1}, {4, 1}}, rm[3] = {{1, 1}, {3, 512}, {4, 1}};
                for (int m = 0; m < 2; ++m) {
                    const backend::RowRuns runs = m ? backend::RowRuns{rm, 3} : backend::RowRuns{rd, 4};
                    std::vector<float> up(rows * kk * nout, -3.0f), y(rows * nout, 0.5f);
                    const auto ub = cpu.adopt(up.data(), up.size() * sizeof(float)), yb = cpu.adopt(y.data(), y.size() * sizeof(float));
                    cpu.matmul_experts({{type, {wb.get(), 0}, {ub.get(), 0}, nout}}, {xb.get(), 0}, nin, rows, routing, runs);
                    out(tag + "/experts" + std::to_string(m), up);
                    cpu.matmul_experts_add(type, {wb.get(), 0}, {xb.get(), 0}, {yb.get(), 0}, nin, nout, rows, routing, runs);
                    out(tag + "/experts_add" + std::to_string(m), y);
                }
            }
        } catch (const std::exception& e) {
            std::printf("%u/%zu/%zu/t%d/d%d/h%d threw %s%c", type, nin, nout, threads, d8, huge, e.what(), 10);
        }
    }
    {
        CpuBackend cpu;
        std::vector<float> src(9 * 13), dst(4 * 13, 0.0f);
        for (size_t i = 0; i < src.size(); ++i) src[i] = (float)i * 0.5f;
        const uint32_t pick[4] = {8, 0, 3, 3};
        const auto s = cpu.adopt(src.data(), src.size() * sizeof(float)), d = cpu.adopt(dst.data(), dst.size() * sizeof(float));
        cpu.gather_rows({d.get(), 0}, {s.get(), 0}, 13, pick, 4);
        out("gather", dst);
        const uint32_t beyond[1] = {9};
        try { cpu.gather_rows({d.get(), 0}, {s.get(), 0}, 13, beyond, 1); std::printf("gather-beyond accepted\n"); }
        catch (const std::exception& e) { std::printf("gather-beyond refused\n"); }
    }
    std::fprintf(stderr, "%zu cases\n", cases);
    return 0;
}
