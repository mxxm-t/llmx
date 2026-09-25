// The CPU decode dots over quantized activations (backends/cpu/q8_dots.hpp), 8-bit or 16-bit by type, against a double-precision reference fed the same quantized activations, for every type they take.
// Also that a decode row computes the same alone and beside others, and grouped projections the same as separate ones, bit for bit.
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "backends/cpu/cpu_backend.hpp"
#include "quant/quant.hpp"

namespace {
void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
}

struct ActivationChecks {
    size_t blocks = 0, values = 0, tiny = 0, ties = 0;
};

// Compare the represented value with its integer neighbours, without computing the encoder's rounding formula.
template <class Rows>
void check_activation_block(const float* x, const Rows& out, size_t block, int levels,
                            const std::array<int, 32>* exact, int sign, ActivationChecks& checks) {
    double top = 0.0;
    for (size_t i = 0; i < 32; ++i) top = std::max(top, std::fabs(double(x[i])));
    const double d = out.d[block];
    auto check = [&](bool ok, size_t lane, const char* what) {
        if (ok) return;
        char message[256];
        std::snprintf(message, sizeof(message), "activation q%d block=%zu lane=%zu x=%a top=%a d=%a q=%d: %s",
                      levels == 127 ? 8 : 16, block, lane, double(x[lane]), top, d,
                      int(out.q[block * 32 + lane]), what);
        throw std::runtime_error(message);
    };
    check(std::isfinite(d) && (top ? d > 0.0 : d == 0.0), 0, "invalid scale");
    const bool tiny = top && !std::isfinite(static_cast<float>(double(levels) / top));
    const double previous = std::nextafter(out.d[block], 0.0f);
    if (tiny) {
        check(double(levels) * d >= top && double(levels) * previous < top, 0,
              "scale is not the smallest positive float covering the block");
        ++checks.tiny;
    }
    // The unchanged SIMD path rounds both the reciprocal/product and its stored scale in float.
    const double slack = tiny ? 0.0 : 4.0 * std::numeric_limits<float>::epsilon() * top +
                                          double(levels) * (d - previous);
    int32_t sum = 0;
    for (size_t i = 0; i < 32; ++i) {
        const int q = out.q[block * 32 + i];
        check(q >= -levels && q <= levels, i, "packed value outside symmetric range");
        check(x[i] == 0.0f ? q == 0 : (x[i] > 0.0f ? q >= 0 : q <= 0), i,
              "zero or sign was not preserved");
        const double error = std::fabs(double(x[i]) - double(q) * d);
        check(error <= 0.5 * d + slack, i, "reconstruction exceeds nearest-step bound");
        for (const int neighbour : {q - 1, q + 1}) {
            if (neighbour < -levels || neighbour > levels) continue;
            const double alternative = std::fabs(double(x[i]) - double(neighbour) * d);
            check(error <= alternative + 2.0 * slack, i, "adjacent reconstruction is closer");
            if (tiny && d && error == alternative) {
                check(q % 2 == 0, i, "exact midpoint did not select even integer");
                ++checks.ties;
            }
        }
        if (exact) check(q == sign * (*exact)[i], i, "exact-scale control changed");
        sum += q;
        ++checks.values;
    }
    check(out.sum[block] == sum, 0, "sum differs from the packed integers");
    ++checks.blocks;
}

template <class Rows, class Quantize>
ActivationChecks check_activation_range(int levels, Quantize quantize) {
    ActivationChecks checks;
    Rows out;
    backend::q8::size_rows(2, 64, out);
    auto run = [&](const std::array<float, 32>& input, const std::array<int, 32>* exact = nullptr) {
        std::vector<float> storage(129, 19.0f);
        float* x = storage.data() + 1;
        for (size_t i = 0; i < 32; ++i) { x[32 + i] = input[i]; x[64 + i] = -input[i]; }
        const auto original = storage;
        std::fill(out.q.begin(), out.q.end(), 25);
        std::fill(out.d.begin(), out.d.end(), -77.0f);
        std::fill(out.sum.begin(), out.sum.end(), -991);
        // The selected blocks cross a row boundary; the neighbouring blocks must stay untouched.
        quantize(x, 1, 3, out);
        for (size_t b : {size_t(0), size_t(3)}) {
            require(out.d[b] == -77.0f && out.sum[b] == -991, "activation metadata guard changed");
            for (size_t i = 0; i < 32; ++i)
                require(out.q[b * 32 + i] == 25, "activation packed guard changed");
        }
        require(std::memcmp(storage.data(), original.data(), storage.size() * sizeof(float)) == 0,
                "activation quantization changed its input");
        check_activation_block(x + 32, out, 1, levels, exact, 1, checks);
        check_activation_block(x + 64, out, 2, levels, exact, -1, checks);
    };

    std::vector<float> tops;
    auto add = [&](float top) { if (top > 0.0f && std::isfinite(top)) tops.push_back(top); };
    for (int exponent = -149; exponent <= 127; ++exponent) {
        const float top = std::ldexp(1.0f, exponent);
        add(std::nextafter(top, 0.0f)); add(top);
        add(std::nextafter(top, std::numeric_limits<float>::infinity()));
    }
    const float threshold = static_cast<float>(double(levels) / std::numeric_limits<float>::max());
    add(std::nextafter(threshold, 0.0f)); add(threshold);
    add(std::nextafter(threshold, std::numeric_limits<float>::infinity()));
    add(std::numeric_limits<float>::max());
    std::sort(tops.begin(), tops.end());
    tops.erase(std::unique(tops.begin(), tops.end()), tops.end());
    run({});
    for (float top : tops) {
        std::array<float, 32> input{};
        input[0] = top;
        run(input);
        input.fill(top);
        run(input);
        input[0] = top; input[1] = -top; input[2] = 0.0f; input[3] = -0.0f;
        for (size_t i = 4; i < input.size(); ++i)
            input[i] = static_cast<float>(double(top) * (int((i * 19) % 63) - 31) / 32.0);
        run(input);
    }
    for (int exponent : {-148, -140, -128, -120, -100, 0, 100}) {
        const float step = std::ldexp(1.0f, exponent);
        std::array<float, 32> input{};
        std::array<int, 32> exact{};
        input[0] = levels * step; input[1] = -input[0];
        exact[0] = levels; exact[1] = -levels;
        for (size_t i = 2; i < input.size(); ++i) {
            const int lower = int((i - 2) / 2), sign = i % 2 ? -1 : 1;
            input[i] = static_cast<float>(sign * (double(lower) + 0.5) * step);
            exact[i] = sign * (lower + lower % 2);
        }
        run(input, &exact);
    }
    return checks;
}

// Packed rows of a type: the block quantizers from floats, the K-quants from a byte pattern with small half scales.
std::vector<uint8_t> packed(uint32_t type, size_t rows, size_t nin, std::mt19937& rng) {
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

// Reference for ordinary blocks with a finite float reciprocal; tiny blocks use the reconstruction checks above.
void rounded(const float* x, size_t n, float levels, std::vector<double>& q, std::vector<double>& d) {
    q.assign(n, 0.0);
    d.assign(n / 32, 0.0);
    for (size_t b = 0; b < n / 32; ++b) {
        float top = 0.0f;
        for (size_t i = 0; i < 32; ++i) top = std::max(top, std::fabs(x[b * 32 + i]));
        const float step = top / levels, inv = top > 0.0f ? levels / top : 0.0f;
        d[b] = step;
        for (size_t i = 0; i < 32; ++i) q[b * 32 + i] = std::nearbyint(x[b * 32 + i] * inv);
    }
}

size_t check_type(uint32_t type, size_t nin, std::mt19937& rng) {
    const quant::QuantType* qt = quant::Registry::instance().get(type);
    const size_t rows = 37, cols = 3;
    const auto w = packed(type, rows, nin, rng);
    std::vector<float> wf(rows * nin);
    qt->dequantize(w.data(), wf.data(), rows * nin / qt->block_size);
    // Ordinary values, and one with a large outlier per block, which is where a block's rounding is coarsest.
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> x(cols * nin);
    for (size_t i = 0; i < x.size(); ++i) x[i] = u(rng) * (i % 97 == 0 ? 40.0f : 1.0f);

    backend::CpuBackend cpu;
    cpu.set_threads(4);
    const auto wb = cpu.adopt(w.data(), w.size());
    const auto xb = cpu.adopt(x.data(), x.size() * sizeof(float));
    std::vector<float> alone(rows * cols), together(rows * cols), grouped(2 * rows * cols);
    for (size_t c = 0; c < cols; ++c) {
        const auto xc = cpu.adopt(x.data() + c * nin, nin * sizeof(float));
        const auto yc = cpu.adopt(alone.data() + c * rows, rows * sizeof(float));
        cpu.matmul(type, {wb.get(), 0}, {xc.get(), 0}, {yc.get(), 0}, nin, rows, 1);
    }
    // Three generated tokens in one call, as a server's decode pass has them.
    const backend::RowRun decode[3] = {{1, 1}, {2, 1}, {3, 1}};
    const auto yt = cpu.adopt(together.data(), together.size() * sizeof(float));
    cpu.matmul(type, {wb.get(), 0}, {xb.get(), 0}, {yt.get(), 0}, nin, rows, cols, {decode, 3});
    require(std::memcmp(alone.data(), together.data(), alone.size() * sizeof(float)) == 0, "a decode row differs beside others");
    const auto yg = cpu.adopt(grouped.data(), grouped.size() * sizeof(float));
    cpu.matmul_group({{type, {wb.get(), 0}, {yg.get(), 0}, rows}, {type, {wb.get(), 0}, {yg.get(), rows * cols}, rows}},
                     {xb.get(), 0}, nin, cols, {decode, 3});
    for (size_t c = 0; c < cols; ++c)
        for (size_t o = 0; o < rows; ++o)
            require(grouped[c * rows + o] == together[c * rows + o] && grouped[rows * cols + c * rows + o] == together[c * rows + o],
                    "grouped projections differ from separate ones");

    // The same columns as a prompt's rows, through the prompt dots for the K-quants and the float path for the rest: alone, and beside a generated token, where the prompt's rows compute as alone and the token as a decode row.
    std::vector<float> prompt(rows * cols), mixed(rows * cols);
    const backend::RowRun as_prompt[1] = {{cols, 512}}, beside[2] = {{1, 1}, {cols, 512}};
    const auto yp = cpu.adopt(prompt.data(), prompt.size() * sizeof(float)), ym = cpu.adopt(mixed.data(), mixed.size() * sizeof(float));
    cpu.matmul(type, {wb.get(), 0}, {xb.get(), 0}, {yp.get(), 0}, nin, rows, cols, {as_prompt, 1});
    cpu.matmul(type, {wb.get(), 0}, {xb.get(), 0}, {ym.get(), 0}, nin, rows, cols, {beside, 2});
    require(std::memcmp(mixed.data(), alone.data(), rows * sizeof(float)) == 0, "a decode row differs beside a prompt");
    require(std::memcmp(mixed.data() + rows, prompt.data() + rows, (cols - 1) * rows * sizeof(float)) == 0,
            "a prompt's rows differ beside a generated token");

    std::vector<double> q, d;
    for (size_t c = 0; c < cols; ++c) {
        rounded(x.data() + c * nin, nin, backend::q8::reads16(type) ? 32767.0f : 127.0f, q, d);
        for (size_t o = 0; o < rows; ++o) {
            double ref = 0.0, mag = 0.0;
            for (size_t i = 0; i < nin; ++i) {
                const double t = (double)wf[o * nin + i] * q[i] * d[i / 32];
                ref += t;
                mag += std::fabs(t);
            }
            // A prompt's K-quant rows at least kPromptDotsFrom wide meet these activations through the prompt dots; narrower ones, and the other types, take the float path on the unrounded ones.
            const bool kquant = type == gguf::GGML_TYPE_Q4_K || type == gguf::GGML_TYPE_Q5_K || type == gguf::GGML_TYPE_Q6_K;
            const bool dots = kquant && nin >= backend::CpuBackend::kPromptDotsFrom;
            for (const float got : dots ? std::vector<float>{alone[c * rows + o], prompt[c * rows + o]} : std::vector<float>{alone[c * rows + o]})
                require(std::isfinite(got) && std::fabs(got - ref) <= 1e-5 * mag + 1e-30,
                        "type " + std::to_string(type) + " nin " + std::to_string(nin) + ": " + std::to_string(got) +
                        " against " + std::to_string(ref));
        }
    }
    return rows * cols;
}
// A generated token's routed entries through the same dots: every entry equals a one-column matmul of its expert's matrix and its token's row, bit for bit, and the down projection their weighted sum.
size_t check_experts(uint32_t type, std::mt19937& rng) {
    const quant::QuantType* qt = quant::Registry::instance().get(type);
    const size_t n_expert = 3, k = 2, rows = 6, entries = rows * k, nin = 256, nout = 8;
    const size_t stride = nout * (nin / qt->block_size) * qt->type_size;
    const auto w = packed(type, n_expert * nout, nin, rng);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> x(rows * nin), x2(entries * nin), ids(entries), weights(entries, 0.5f);
    for (float& v : x) v = u(rng);
    for (float& v : x2) v = u(rng);
    for (size_t e = 0; e < entries; ++e) {
        const uint32_t id = (uint32_t)((e * 7 + e / k) % n_expert);
        std::memcpy(&ids[e], &id, sizeof(id));
    }
    backend::CpuBackend cpu;
    cpu.set_threads(4);
    const auto wb = cpu.adopt(w.data(), w.size());
    const auto xb = cpu.adopt(x.data(), x.size() * sizeof(float)), x2b = cpu.adopt(x2.data(), x2.size() * sizeof(float));
    const auto ib = cpu.adopt(ids.data(), ids.size() * sizeof(float)), gb = cpu.adopt(weights.data(), weights.size() * sizeof(float));
    const backend::Backend::Routing routing{{ib.get(), 0}, {gb.get(), 0}, k, n_expert};
    const backend::RowRun decode[6] = {{1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}, {6, 1}};
    const backend::RowRuns runs{decode, 6};
    std::vector<float> up(entries * nout), down(rows * nout, 0.0f);
    const auto ub = cpu.adopt(up.data(), up.size() * sizeof(float));
    cpu.matmul_experts({{type, {wb.get(), 0}, {ub.get(), 0}, nout}}, {xb.get(), 0}, nin, rows, routing, runs);
    for (size_t e = 0; e < entries; ++e) {
        uint32_t id;
        std::memcpy(&id, &ids[e], sizeof(id));
        std::vector<float> y(nout), yd(nout);
        const auto xe = cpu.adopt(x.data() + (e / k) * nin, nin * sizeof(float)), ye = cpu.adopt(y.data(), nout * sizeof(float));
        cpu.matmul(type, {wb.get(), id * stride / sizeof(float)}, {xe.get(), 0}, {ye.get(), 0}, nin, nout, 1);
        require(std::memcmp(y.data(), up.data() + e * nout, nout * sizeof(float)) == 0,
                "type " + std::to_string(type) + ": a routed entry differs from its expert's matmul");
        const auto xd = cpu.adopt(x2.data() + e * nin, nin * sizeof(float)), yb = cpu.adopt(yd.data(), nout * sizeof(float));
        cpu.matmul(type, {wb.get(), id * stride / sizeof(float)}, {xd.get(), 0}, {yb.get(), 0}, nin, nout, 1);
        for (size_t o = 0; o < nout; ++o) down[(e / k) * nout + o] += 0.5f * yd[o];
    }
    // The same rows as a prompt, whose entries go through their expert's rows several at a time: each prompt row computes the same beside generated tokens as in a call of its own, and each generated token as alone.
    const backend::RowRun prompt[1] = {{rows, 512}}, mixed[3] = {{2, 1}, {5, 512}, {6, 1}};
    std::vector<float> ups(entries * nout), upm(entries * nout);
    const auto usb = cpu.adopt(ups.data(), ups.size() * sizeof(float)), umb = cpu.adopt(upm.data(), upm.size() * sizeof(float));
    cpu.matmul_experts({{type, {wb.get(), 0}, {usb.get(), 0}, nout}}, {xb.get(), 0}, nin, rows, routing, {prompt, 1});
    cpu.matmul_experts({{type, {wb.get(), 0}, {umb.get(), 0}, nout}}, {xb.get(), 0}, nin, rows, routing, {mixed, 3});
    for (size_t e = 0; e < entries; ++e) {
        const size_t token = e / k;
        const std::vector<float>& want = token >= 2 && token < 5 ? ups : up;
        require(std::memcmp(want.data() + e * nout, upm.data() + e * nout, nout * sizeof(float)) == 0,
                "type " + std::to_string(type) + ": a routed entry differs beside other kinds of row");
    }
    // A prompt's products on the prompt dots stay within the quantized reference's bound of the decode dots'.
    for (size_t i = 0; i < entries * nout; ++i)
        require(std::fabs(ups[i] - up[i]) <= 1e-5f * (1.0f + std::fabs(up[i])) * 64.0f,
                "type " + std::to_string(type) + ": a prompt's routed entry strays from a generated token's");
    std::vector<float> y(rows * nout, 0.0f);
    const auto yb = cpu.adopt(y.data(), y.size() * sizeof(float));
    cpu.matmul_experts_add(type, {wb.get(), 0}, {x2b.get(), 0}, {yb.get(), 0}, nin, nout, rows, routing, runs);
    for (size_t i = 0; i < rows * nout; ++i)
        require(std::fabs(y[i] - down[i]) <= 1e-6f * (1.0f + std::fabs(down[i])),
                "type " + std::to_string(type) + ": a routed down projection differs from its experts' matmuls");
    return entries;
}
}  // namespace

int main() {
    try {
        const auto a8 = check_activation_range<backend::q8::Rows>(127, backend::q8::quantize);
        const auto a16 = check_activation_range<backend::q8::Rows16>(32767, backend::q8::quantize16);
        std::printf("activation range: %zu blocks, %zu packed values, %zu tiny blocks, %zu exact tiny ties; input and output guards passed\n",
                    a8.blocks + a16.blocks, a8.values + a16.values, a8.tiny + a16.tiny, a8.ties + a16.ties);
        std::mt19937 rng(7);
        size_t n = 0;
        for (uint32_t type : {gguf::GGML_TYPE_Q8_0, gguf::GGML_TYPE_Q4_0, gguf::GGML_TYPE_Q4_1,
                              gguf::GGML_TYPE_Q4_K, gguf::GGML_TYPE_Q5_K, gguf::GGML_TYPE_Q6_K})
            for (size_t nin : {size_t(256), size_t(2048), size_t(4096)}) n += check_type(type, nin, rng);
        size_t routed = 0;
        for (uint32_t type : {gguf::GGML_TYPE_Q8_0, gguf::GGML_TYPE_Q4_0, gguf::GGML_TYPE_Q4_1,
                              gguf::GGML_TYPE_Q4_K, gguf::GGML_TYPE_Q5_K, gguf::GGML_TYPE_Q6_K})
            routed += check_experts(type, rng);
        std::printf("q8 dots: %zu rows against the reference, alone, beside others and grouped; %zu routed entries against their experts\n",
                    n, routed);
        return 0;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
