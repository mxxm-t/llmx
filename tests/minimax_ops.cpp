// minimax-ops: the CPU forms of MiniMax-M3.1's layer ops against the math of docs/MINIMAX.md in double.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/cpu/cpu_backend.hpp"

namespace {

using backend::BufferPtr;
using backend::CpuBackend;

size_t checked = 0;
const double U = std::ldexp(1.0, -24);
const int kThreads[] = {1, 2, 3, 5, 8};

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checked;
}

void refused(const std::function<void()>& run, const std::string& what, const std::string& text) {
    try {
        run();
    } catch (const std::exception& e) {
        require(std::string(e.what()) == text, what + " refused with \"" + e.what() + "\"");
        return;
    }
    throw std::runtime_error(what + " was not refused");
}

uint32_t bits(float x) {
    uint32_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}

uint64_t state = 0x243F6A8885A308D3ull;
uint32_t draw() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return uint32_t(state >> 32);
}

BufferPtr upload(CpuBackend& cpu, const std::vector<float>& v) {
    BufferPtr b = cpu.alloc(std::max<size_t>(v.size(), 1) * sizeof(float), backend::Memory::device);
    if (!v.empty()) cpu.write(*b, 0, v.data(), v.size() * sizeof(float));
    return b;
}

std::vector<float> download(CpuBackend& cpu, const backend::Buffer& b, size_t n) {
    std::vector<float> v(n);
    cpu.read(b, 0, v.data(), n * sizeof(float));
    return v;
}

// route_experts' sigmoid rule against docs/MINIMAX.md (The forward pass) in double: scores sigmoid(x), picks by score + bias with ties to the lower id, weights the unbiased scores over their sum times the scale.
void sigmoid_routing() {
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    const float level[] = {-inf, -20.0f, -30.0f, -2.5f, -0.0f, 0.0f, 0.75f, 3.0f, 12.0f, 31.0f, inf};
    const size_t n_expert = 24, rows = 9;
    std::vector<float> x(rows * n_expert), bias(n_expert);
    for (float& v : x) v = level[draw() % 11];
    for (float& v : bias) v = float(int(draw() % 9) - 4) / 64.0f;
    for (size_t e = 0; e < n_expert; ++e) x[n_expert + e] = 1.0f;
    for (size_t e = 0; e < n_expert; ++e) x[4 * n_expert + e] = float(int(draw() % 2001) - 1000) / 100.0f;
    x[2 * n_expert + 5] = nan;
    CpuBackend cpu;
    BufferPtr xb = upload(cpu, x), bb = upload(cpu, bias);
    for (size_t k : {1, 8, 24})
        for (bool with_bias : {false, true})
            for (bool normalize : {false, true}) {
                const float scale = 2.0f;
                const backend::RouteRule rule = backend::RouteRule::sigmoid(with_bias ? backend::CSlice{bb.get(), 0} : backend::CSlice{}, scale, normalize);
                auto run = [&](size_t first, size_t n) {
                    BufferPtr ib = cpu.alloc(n * k * 4, backend::Memory::device), wb = cpu.alloc(n * k * 4, backend::Memory::device);
                    cpu.route_experts({xb.get(), first * n_expert}, n, n_expert, k, rule, {ib.get(), 0}, {wb.get(), 0});
                    std::vector<float> out = download(cpu, *ib, n * k), w = download(cpu, *wb, n * k);
                    out.insert(out.end(), w.begin(), w.end());
                    return out;
                };
                cpu.set_threads(1);
                const std::vector<float> got = run(0, rows);
                const std::string name = "route_experts sigmoid k " + std::to_string(k) + (with_bias ? " with a bias" : "") + (normalize ? " normalized" : "");
                for (size_t r = 0; r < rows; ++r) {
                    std::vector<double> s(n_expert);
                    std::vector<float> key(n_expert);
                    bool finite = true;
                    for (size_t e = 0; e < n_expert; ++e) {
                        s[e] = 1.0 / (1.0 + std::exp(-double(x[r * n_expert + e])));
                        const float b = float(s[e]) + (with_bias ? bias[e] : 0.0f);
                        finite = finite && std::isfinite(b);
                        key[e] = b;
                    }
                    std::vector<uint32_t> order(n_expert);
                    for (size_t i = 0; i < n_expert; ++i) order[i] = uint32_t(i);
                    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return key[a] > key[b]; });
                    double total = 0.0;
                    for (size_t j = 0; j < k; ++j) total += s[order[j]];
                    for (size_t j = 0; j < k; ++j) {
                        const uint32_t id = bits(got[r * k + j]);
                        const float w = got[rows * k + r * k + j];
                        if (!finite) {
                            require(id == CpuBackend::kNoExpert && bits(w) == 0, name + " row " + std::to_string(r) + " without a candidate");
                            continue;
                        }
                        require(id == order[j], name + " row " + std::to_string(r) + " pick " + std::to_string(j));
                        const double want = (normalize ? s[order[j]] / total : s[order[j]]) * scale;
                        require(std::fabs(w - want) <= (k + 8) * U * std::fabs(want) + 1e-44, name + " row " + std::to_string(r) + " weight " + std::to_string(j));
                    }
                    const std::vector<float> one = run(r, 1);
                    require(!std::memcmp(one.data(), got.data() + r * k, k * 4) && !std::memcmp(one.data() + k, got.data() + rows * k + r * k, k * 4),
                            name + " of row " + std::to_string(r) + " alone");
                }
                for (int t : kThreads) {
                    cpu.set_threads(t);
                    const std::vector<float> again = run(0, rows);
                    require(!std::memcmp(again.data(), got.data(), got.size() * 4), name + " at " + std::to_string(t) + " threads");
                }
            }
    refused([] { backend::RouteRule::sigmoid({}, 2.0f, true).softmax_only("device"); }, "a device without sigmoid scores given them",
            "device: route_experts has no sigmoid scores");
}

// silu_mul's quick_gelu form against docs/MINIMAX.md (The forward pass) in double: g' = min(g, L), u' = clamp(u, -L, L), (u' + 1) * g' * sigmoid(1.702 g'), a limit of 0 clamping nothing and 1 to 8 threads bit for bit.
void quick_gelu() {
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    const float special[] = {-inf, -200.0f, -7.5f, -7.0f, -1.0f, -0.0f, 0.0f, 1e-30f, 0.5f, 6.999f, 7.0f, 7.25f, 90.0f, inf, nan};
    const size_t n = 4099;
    std::vector<float> g(n), u(n);
    for (size_t i = 0; i < n; ++i) {
        g[i] = float(int(draw() % 24001) - 12000) / 1000.0f;
        u[i] = float(int(draw() % 24001) - 12000) / 1000.0f;
    }
    for (size_t i = 0; i < 15; ++i)
        for (size_t j = 0; j < 15; ++j) {
            g[i * 15 + j] = special[i];
            u[i * 15 + j] = special[j];
        }
    CpuBackend cpu;
    BufferPtr gb = upload(cpu, g), ub = upload(cpu, u), ob = cpu.alloc(n * 4, backend::Memory::device);
    for (float limit : {7.0f, 0.0f}) {
        const std::string name = "silu_mul quick_gelu at limit " + std::to_string(limit);
        cpu.set_threads(1);
        cpu.silu_mul({ob.get(), 0}, {gb.get(), 0}, {ub.get(), 0}, n, limit, {}, backend::GluForm::quick_gelu);
        const std::vector<float> got = download(cpu, *ob, n);
        for (size_t i = 0; i < n; ++i) {
            double gd = g[i], ud = u[i];
            if (limit > 0.0f) {
                ud = ud < -limit ? -limit : ud > limit ? limit : ud;
                gd = gd > limit ? limit : gd;
            }
            // The GLU half is rounded to float as the op computes it, so an underflow to zero meets an infinite up as the op's does.
            const double want = (ud + 1.0) * double(float(gd / (1.0 + std::exp(-1.702 * gd))));
            const std::string at = name + " at " + std::to_string(g[i]) + ", " + std::to_string(u[i]);
            if (std::isnan(want)) require(std::isnan(got[i]), at + " is NaN");
            else if (std::isinf(want)) require(got[i] == float(want), at + " is " + std::to_string(want));
            else require(std::fabs(got[i] - want) <= (8 + 2 * std::fabs(1.702 * gd)) * U * std::fabs(want) + 1e-37, at + ": " + std::to_string(got[i]) + " against " + std::to_string(want));
        }
        for (int t : kThreads) {
            cpu.set_threads(t);
            cpu.silu_mul({ob.get(), 0}, {gb.get(), 0}, {ub.get(), 0}, n, limit, {}, backend::GluForm::quick_gelu);
            require(!std::memcmp(download(cpu, *ob, n).data(), got.data(), n * 4), name + " at " + std::to_string(t) + " threads");
        }
    }
}

} // namespace

int main() {
    try {
        sigmoid_routing();
        quick_gelu();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << std::endl;
        return 1;
    }
    std::cout << "minimax-ops: " << checked << " checks passed" << std::endl;
    return 0;
}
