#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "core/bf16.hpp"
#include "model/arch/registry.hpp"
#include "model/runtime.hpp"
#include "model/place.hpp"
#include "tiny_qwen.hpp"
#include "matrix_precision.hpp"

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
float from_bits(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}
uint32_t bits_of(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof bits);
    return bits;
}
// The mathematical value of a positive BF16 encoding, with 0x7f80 the finite overflow neighbour 2^128.
double value(uint32_t b) {
    const int e = int(b >> 7);
    return e ? std::ldexp(1.0 + double(b & 127u) / 128.0, e - 127) : std::ldexp(double(b), -133);
}
void conversions() {
    for (uint32_t b = 0; b < 0x7f80u; ++b) {
        for (uint32_t sign : {0u, 0x80000000u}) {
            const uint16_t h = uint16_t(b | (sign >> 16));
            require(bits_of(bf16_to_f32(h)) == (uint32_t(h) << 16), "BF16 widening changed bits");
            require(f32_to_bf16(from_bits(uint32_t(h) << 16)) == h, "BF16 exact round trip");
            for (uint32_t tail : {0u, 0x7fffu, 0x8000u, 0x8001u, 0xffffu}) {
                const float f = from_bits((b << 16) | tail);
                const double lo = double(f) - value(b), hi = value(b + 1) - double(f);
                const uint32_t chosen = hi < lo || (hi == lo && (b & 1u)) ? b + 1 : b;
                require(f32_to_bf16(from_bits(bits_of(f) | sign)) == uint16_t(chosen | (sign >> 16)),
                        "BF16 nearest neighbour or ties-to-even");
            }
        }
    }
    for (uint32_t sign : {0u, 0x80000000u}) {
        require(f32_to_bf16(from_bits(sign | 0x7f800000u)) == uint16_t((sign >> 16) | 0x7f80u), "BF16 infinity");
        for (uint32_t payload : {1u, 0x8000u, 0x400000u, 0x7fffffu}) {
            const uint16_t h = f32_to_bf16(from_bits(sign | 0x7f800000u | payload));
            require((h & 0x7f80u) == 0x7f80u && (h & 127u) && (h & 0x8000u) == (sign >> 16), "BF16 NaN or sign lost");
        }
    }
}

// A packed Q4_0 unit row selects one input beside a larger block peak. F32, BF16 and the 16-bit block dot give distinct answers.
void arithmetic() {
    backend::CpuBackend cpu;
    cpu.set_threads(1);
    std::vector<uint8_t> w(18, 0x88);
    w[0] = 0; w[1] = 0x3c; w[3] = 0x89;
    std::vector<float> x(3 * 32), y(9), z(9), ids(3, 0), gains(3, 0.5f);
    for (size_t r = 0; r < 3; ++r) {
        x[r * 32] = 2.0f;
        x[r * 32 + 1] = r == 1 ? -1.005859375f : 1.005859375f;
    }
    const auto wb = cpu.adopt(w.data(), w.size()), xb = cpu.adopt(x.data(), x.size() * sizeof(float));
    const auto yb = cpu.adopt(y.data(), y.size() * sizeof(float)), zb = cpu.adopt(z.data(), z.size() * sizeof(float));
    const auto ib = cpu.adopt(ids.data(), ids.size() * sizeof(float)), gb = cpu.adopt(gains.data(), gains.size() * sizeof(float));
    const backend::Backend::Routing routing{{ib.get(), 0}, {gb.get(), 0}, 1, 1};
    for (size_t n : {std::numeric_limits<size_t>::max(), size_t(64)}) {
        bool refused = false;
        try {
            cpu.matmul(quant::GGML_TYPE_Q4_0, {wb.get(), 0}, {xb.get(), 0}, {yb.get(), 0}, n, 1, 3, {}, backend::Dtype::bf16);
        } catch (const std::runtime_error&) { refused = true; }
        require(refused && y[0] == 0.0f, "BF16 conversion accepted an invalid input extent");
    }

    const backend::RowRun prompt[1] = {{3, 512}}, decode[1] = {{3, 1}}, mixed[2] = {{1, 1}, {3, 512}};
    for (int threads : {1, 3}) {
        cpu.set_threads(threads);
        for (const backend::RowRuns runs : {backend::RowRuns{prompt, 1}, {decode, 1}, {mixed, 2}}) {
            for (auto dtype : {backend::Dtype::bf16, backend::Dtype::f32, backend::Dtype::bf16, backend::Dtype::f32}) {
                auto check = [&](const std::vector<float>& out, size_t offset, float scale) {
                    if (!offset) {
                        require(testq::take_matrix_paths(cpu) == std::vector<std::string>{dtype == backend::Dtype::bf16 ? "bf16" : "f32"}, "matrix witness differs from arithmetic");
                        require(testq::take_matrix_paths(cpu).empty(), "matrix witness leaked into the next measurement");
                    }
                    for (size_t r = 0; r < 3; ++r) {
                        const float expected = dtype == backend::Dtype::f32 ? x[r * 32 + 1] : (r == 1 ? -1.0078125f : 1.0078125f);
                        require(out[offset + r] == scale * expected, "matrix call did not apply requested arithmetic");
                    }
                };
                cpu.matmul(quant::GGML_TYPE_Q4_0, {wb.get(), 0}, {xb.get(), 0}, {yb.get(), 0}, 32, 1, 3, runs, dtype);
                check(y, 0, 1);
                cpu.matmul_logits(quant::GGML_TYPE_Q4_0, {wb.get(), 0}, {xb.get(), 0}, {yb.get(), 0}, 32, 1, 3, runs, dtype);
                check(y, 0, 1);
                cpu.matmul_group({{quant::GGML_TYPE_Q4_0, {wb.get(), 0}, {yb.get(), 0}, 1},
                                  {quant::GGML_TYPE_Q4_0, {wb.get(), 0}, {yb.get(), 3}, 1}}, {xb.get(), 0}, 32, 3, runs, dtype);
                check(y, 0, 1); check(y, 3, 1);
                std::fill(z.begin(), z.end(), 0.0f);
                cpu.matmul_add(quant::GGML_TYPE_Q4_0, {wb.get(), 0}, {xb.get(), 0}, {zb.get(), 0}, 32, 1, 3, runs, dtype);
                check(z, 0, 1);
                cpu.matmul_experts({{quant::GGML_TYPE_Q4_0, {wb.get(), 0}, {yb.get(), 0}, 1}}, {xb.get(), 0}, 32, 3, routing, runs, dtype);
                check(y, 0, 1);
                std::fill(z.begin(), z.end(), 0.0f);
                cpu.matmul_experts_add(quant::GGML_TYPE_Q4_0, {wb.get(), 0}, {xb.get(), 0}, {zb.get(), 0}, 32, 1, 3, routing, runs, dtype);
                check(z, 0, 0.5f);
            }
        }
    }
    cpu.matmul(quant::GGML_TYPE_Q4_0, {wb.get(), 0}, {xb.get(), 0}, {yb.get(), 0}, 32, 1, 3, {decode, 1}, backend::Dtype::f16);
    require(testq::take_matrix_paths(cpu) == std::vector<std::string>{"block-int16"}, "integer path missing from witness");
    const float block = float(std::nearbyint(double(x[1]) * 32767.0 / 2.0) * (2.0 / 32767.0));
    require(std::fabs(y[0] - block) < 2e-7f && y[0] != x[1] && y[0] != 1.0078125f, "16-bit activation path was not witnessed");
}

void exact(const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), "shared backend or split changed model policy");
}
std::shared_ptr<backend::CpuBackend> cpu() {
    auto b = std::make_shared<backend::CpuBackend>();
    b->set_threads(1);
    return b;
}
struct Capabilities : backend::CpuBackend {
    std::vector<backend::Dtype> supported;
    explicit Capabilities(std::vector<backend::Dtype> dtypes) : supported(std::move(dtypes)) { set_threads(1); }
    std::vector<backend::Dtype> native_dtypes() const override { return supported; }
    bool emulates_dtype(backend::Dtype) const override { return false; }
};
struct Emulated : Capabilities {
    Emulated() : Capabilities({backend::Dtype::f32}) {}
    bool emulates_dtype(backend::Dtype dtype) const override { return dtype == backend::Dtype::bf16; }
};
// A device that runs int8, as an MI50 that prefers the integer dot does, and one that does not without being the CPU.
struct Eight : Capabilities {
    Eight() : Capabilities({backend::Dtype::f16, backend::Dtype::f32, backend::Dtype::int8}) {}
    bool is_cpu() const override { return false; }
};
struct Sixteen : Capabilities {
    Sixteen() : Capabilities({backend::Dtype::f16, backend::Dtype::f32}) {}
    bool is_cpu() const override { return false; }
};
// int8 is never chosen by auto, runs where a device lists it, and a device that does not takes the next wider dtype it has, f16, else f32, with the warning (docs/PRECISION.md, rule 2).
void int8_resolution() {
    using D = backend::Dtype;
    require(std::string(backend::dtype_name(D::int8)) == "int8", "int8 name");
    const auto eight = std::make_shared<Eight>(), other = std::make_shared<Eight>();
    const auto sixteen = std::make_shared<Sixteen>();
    const auto full = std::make_shared<Capabilities>(std::vector<D>{D::f32});
    require(infer::resolve_dtype(D::bf16, {eight}, {}).effective == D::f16 && infer::resolve_dtype(D::f32, {eight}, {}).effective == D::f16,
            "auto chose int8");
    const auto record = infer::resolve_dtype(D::bf16, {eight, other}, {"vulkan:0", "vulkan:1"}, D::int8);
    require(record.effective == D::int8 && record.devices.size() == 2 && record.devices[1].how == "native" && record.devices[1].effective == D::int8 &&
                record.describe().find("warning") == std::string::npos,
            "int8 not resolved where every device lists it");
    const auto mixed = infer::resolve_dtype(D::bf16, {eight, cpu(), sixteen, full}, {"vulkan:0", "cpu", "vulkan:1", "other"}, D::int8);
    require(mixed.effective == D::int8 && mixed.devices[0].how == "native" && mixed.devices[0].effective == D::int8 &&
                mixed.devices[1].how == "fallback" && mixed.devices[1].effective == D::f16 && mixed.devices[2].how == "fallback" &&
                mixed.devices[2].effective == D::f16 && mixed.devices[3].how == "fallback" && mixed.devices[3].effective == D::f32,
            "int8 not widened to the next dtype a device has");
    const std::string line = mixed.describe();
    require(line.find("cpu fallback to f16: ") != std::string::npos && line.find("other fallback to f32: ") != std::string::npos &&
                line.find("(warning: emulated or wider fallback)") != std::string::npos,
            "int8 fallback not reported per device");
    // Where no device runs int8, the run is the dtype they widened to.
    require(infer::resolve_dtype(D::bf16, {cpu(), sixteen}, {"cpu", "vulkan:0"}, D::int8).effective == D::f16, "a run without int8 not named f16");
}

void resolution() {
    using D = backend::Dtype;
    const auto half = std::make_shared<Capabilities>(std::vector<D>{D::f16, D::f32});
    const auto brain = std::make_shared<Capabilities>(std::vector<D>{D::bf16, D::f16, D::f32});
    const auto brain_only = std::make_shared<Capabilities>(std::vector<D>{D::bf16, D::f32});
    const auto full = std::make_shared<Capabilities>(std::vector<D>{D::f32});
    require(infer::resolve_dtype(D::bf16, {brain}, {}).effective == D::bf16, "auto discarded supported declaration");
    require(infer::resolve_dtype(D::f32, {brain}, {}).effective == D::bf16, "declared F32 did not select preferred auto dtype");
    require(infer::resolve_dtype(D::bf16, {half}, {}).effective == D::f16, "auto did not use preferred supported dtype");
    require(infer::resolve_dtype(D::bf16, {brain, half}, {}).effective == D::f16, "auto ignored mixed device intersection");
    require(infer::resolve_dtype(D::bf16, {brain_only, half}, {}).effective == D::f32, "no common narrow dtype did not use F32");
    require(infer::resolve_dtype(D::bf16, {half, full}, {}).effective == D::f32, "auto ignored F32-only device");
    const auto actual = cpu();
    const auto native = actual->native_dtypes();
    require(actual->dtype_path(D::f32) == "f32", "F32 catalog claims a narrower path");
    const std::string half_paths = "block-int16 (Q4/Q5/Q6/MXFP4 decode, wide K-quant and routed Q4/Q5/Q6 prompts), f32 (other products, routers)";
    const std::string brain_paths = "bf16 (matrix inputs), f32 (routers)";
    require(actual->dtype_path(D::f16) == half_paths, "CPU catalog hides packed or wider matrix families");
    require(actual->dtype_path(D::bf16) == brain_paths, "BF16 catalog hides retained F32 routers");
    require(actual->native_dtypes() == native, "path catalog changed native policy preferences");
    require(testq::take_matrix_paths(*actual).empty(), "path catalog fabricated an execution witness");
    const auto half_record = infer::resolve_dtype(D::bf16, {half}, {"cpu"}, D::f16);
    require(half_record.devices[0].paths == half_paths &&
            half_record.describe().find("cpu native: " + half_paths) != std::string::npos, "placement lost CPU path families");
    const auto brain_record = infer::resolve_dtype(D::bf16, {actual}, {"cpu"}, D::bf16);
    require(brain_record.devices[0].how == "emulated" && brain_record.devices[0].paths == brain_paths,
            "emulated BF16 record hides its matrix families");
    const auto record = infer::resolve_dtype(D::bf16, {full}, {"cpu"});
    require(record.devices.size() == 1 && record.devices[0].how == "native" && record.devices[0].paths == "f32", "dtype record describes another path");
    require(record.describe() == "dtype: auto -> f32 (model declares bf16); cpu native: f32\n", "dtype record text");
    for (const std::vector<backend::BackendPtr>& invalid : {std::vector<backend::BackendPtr>{}, {nullptr}}) {
        bool refused = false;
        try { infer::resolve_dtype(D::bf16, invalid, {}); } catch (const std::runtime_error&) { refused = true; }
        require(refused, "dtype resolver accepted missing devices");
    }
    const auto emulated = std::make_shared<Emulated>();
    const auto explicit_full = infer::resolve_dtype(D::bf16, {brain}, {}, D::f32);
    require(explicit_full.effective == D::f32 && explicit_full.requested == D::f32, "explicit F32 followed auto");
    const auto fallback = infer::resolve_dtype(D::bf16, {brain_only}, {"only-bf16"}, D::f16);
    require(fallback.effective == D::f32 && fallback.devices[0].how == "fallback" && fallback.devices[0].effective == D::f32,
            "unsupported F16 silently became BF16");
    require(fallback.describe().find("warning:") != std::string::npos, "fallback warning missing");
    const auto mix = infer::resolve_dtype(D::f32, {brain, emulated, full}, {"native", "emulated", "full"}, D::bf16);
    require(mix.effective == D::bf16 && mix.devices[0].how == "native" && mix.devices[1].how == "emulated" && mix.devices[2].how == "fallback",
            "mixed device implementation modes");
    require(mix.devices[0].effective == D::bf16 && mix.devices[1].effective == D::bf16 && mix.devices[2].effective == D::f32,
            "mixed device fallback changed other policies");
    require(infer::resolve_dtype(D::bf16, {emulated}, {}).effective == D::f32, "emulation entered auto preference");
    int8_resolution();
    const auto file = tiny_qwen(2, 128, false);
    auto weights = infer::gguf_weights(file);
    require(weights.declared_dtype == D::bf16, "GGUF architecture default missing");
    infer::ModelOptions bf16, f32;
    bf16.dtype = D::bf16; f32.dtype = D::f32;
    infer::PlacementRequest request; request.names = {"mock"}; request.ubatch = 2;
    auto placed = infer::place_model(weights, {brain}, request, f32);
    infer::Model expected(weights, brain, bf16), other(weights, full, f32);
    expected.set_ubatch(2); other.set_ubatch(2);
    const std::vector<uint32_t> prompt{3, 1, 4, 1, 5};
    const auto want = expected.prefill(prompt), different = other.prefill(prompt);
    require(want != different, "resolver application fixture does not distinguish policies");
    exact(placed.model->prefill(prompt), want);
    require(placed.dtype.effective == D::bf16, "placed record differs from applied policy");
    request.dtype = D::f32;
    auto forced = infer::place_model(weights, {brain}, request, bf16);
    exact(forced.model->prefill(prompt), different);
    request.dtype = D::bf16; request.names = {"emulated", "full"}; request.shares = {1, 1};
    request.fit_kv = true;
    auto mixed = infer::place_model(weights, {emulated, full}, request, f32);
    require(mixed.model->kv_allocated_bytes() > 0, "dtype placement discarded the fitted KV backing");
    infer::ModelOptions per_device = f32;
    per_device.device_dtypes = {D::bf16, D::f32};
    infer::Placement split; split.mixer_device = split.ffn_device = {0, 1}; split.output_device = 1;
    infer::Model reference(weights, {cpu(), cpu()}, split, per_device);
    reference.set_ubatch(2);
    exact(mixed.model->prefill(prompt), reference.prefill(prompt));
    require(mixed.model->take_matrix_paths() == std::vector<std::vector<std::string>>{{"bf16"}, {"f32"}}, "mixed fallback policy not executed");
    for (int id : {9, 2, 6}) exact(mixed.model->step(id), reference.step(id));
    per_device.device_dtypes.pop_back();
    bool refused = false;
    try { infer::Model invalid(weights, {cpu(), cpu()}, split, per_device); } catch (const std::runtime_error& e) {
        refused = std::string(e.what()) == "inference: dtype policy does not cover every device";
    }
    require(refused, "partial per-device policy accepted");
}

void models(bool routed) {
    auto file = routed ? tiny_qwen_moe(2, 128, false) : tiny_qwen(2, 128, false);
    const auto weights = infer::gguf_weights(file);
    const auto shared = cpu(), other = cpu();
    infer::ModelOptions f32, bf16;
    f32.dtype = backend::Dtype::f32;
    bf16.dtype = backend::Dtype::bf16;
    f32.kv_k = f32.kv_v = bf16.kv_k = bf16.kv_v = backend::KVType::f32;
    infer::Model a(weights, shared, f32), b(weights, shared, bf16);
    infer::Model refa(weights, cpu(), f32), refb(weights, cpu(), bf16);
    infer::Placement p;
    p.mixer_device = {0, 1}; p.ffn_device = {1, 0}; p.output_device = 1;
    infer::Model split(weights, {shared, other}, p, bf16);
    a.set_ubatch(2); b.set_ubatch(2); refa.set_ubatch(2); refb.set_ubatch(2); split.set_ubatch(2);
    const std::vector<uint32_t> prompt{3, 1, 4, 1, 5};
    auto ra = refa.prefill(prompt), rb = refb.prefill(prompt);
    require(ra != rb, "model dtype test did not distinguish arithmetic");
    const std::vector<std::string> narrow = routed ? std::vector<std::string>{"f32", "bf16"} : std::vector<std::string>{"bf16"};
    const auto witness = [](infer::Model& model, const std::vector<std::vector<std::string>>& expected) {
        require(model.take_matrix_paths() == expected, "model or split witness differs from completed work");
    };
    // Read only after all three have used the shared backend, in another order.
    exact(a.prefill(prompt), ra);
    exact(b.prefill(prompt), rb);
    exact(split.prefill(prompt), rb);
    witness(b, {narrow}); witness(a, {{"f32"}}); witness(split, {narrow, narrow});
    witness(a, {{}}); witness(b, {{}}); witness(split, {{}, {}});
    for (int token : {9, 2, 6, 5}) {
        ra = refa.step(token); rb = refb.step(token);
        exact(b.step(token), rb);
        exact(a.step(token), ra);
        exact(split.step(token), rb);
        witness(split, {narrow, narrow}); witness(a, {{"f32"}}); witness(b, {narrow});
    }
}

struct FailingHead : backend::CpuBackend {
    bool fail = false;
    void matmul_logits(uint32_t type, backend::CSlice data, backend::CSlice x, backend::Slice y,
                       size_t nin, size_t nout, size_t nbatch, backend::RowRuns runs, backend::Dtype dtype) override {
        CpuBackend::matmul_logits(type, data, x, y, nin, nout, nbatch, runs, dtype);
        if (fail) throw std::runtime_error("injected head failure");
    }
};
void failed_witness() {
    const auto file = tiny_qwen(2, 128, false);
    const auto weights = infer::gguf_weights(file);
    auto shared = std::make_shared<FailingHead>();
    shared->set_threads(1);
    infer::ModelOptions bf16, f32;
    bf16.dtype = backend::Dtype::bf16; f32.dtype = backend::Dtype::f32;
    infer::Model a(weights, shared, bf16), b(weights, shared, f32);
    const std::vector<uint32_t> prompt{3, 1, 4};
    // Direct backend evidence belongs to its caller, even across a model's failed stage.
    const float one = 1;
    const auto w = shared->adopt(&one, sizeof(one)), x = shared->adopt(&one, sizeof(one)), y = shared->alloc(sizeof(one), backend::Memory::device);
    shared->matmul(quant::GGML_TYPE_F32, {w.get(), 0}, {x.get(), 0}, {y.get(), 0}, 1, 1, 1, {}, backend::Dtype::f32);
    shared->fail = true;
    bool refused = false;
    try { a.prefill(prompt); } catch (const std::runtime_error& e) { refused = std::string(e.what()) == "injected head failure"; }
    require(refused, "head failure did not reach stage cleanup");
    shared->fail = false;
    b.prefill(prompt);
    require(b.take_matrix_paths() == std::vector<std::vector<std::string>>{{"f32"}}, "failed model leaked paths to another model");
    require(a.take_matrix_paths() == std::vector<std::vector<std::string>>{{"bf16"}}, "failed model lost its dispatched paths");
    require(testq::take_matrix_paths(*shared) == std::vector<std::string>{"f32"}, "model consumed direct backend evidence");
    require(testq::take_matrix_paths(*shared).empty(), "direct backend evidence did not clear");
    a.prefill(prompt);
    require(a.take_matrix_paths() == std::vector<std::vector<std::string>>{{"bf16"}}, "failed stage did not restore witness capture");
}
}

int main() {
    try {
        conversions(); arithmetic(); resolution(); models(false); models(true); failed_witness();
        std::cout << "dtype: exact BF16 conversions, six matrix calls, model isolation and split pass\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
