// The qwen3vl_merger vision module (model/arch/qwen3vl_vision.hpp) and the encoder runner (model/encoder.hpp) on tiny mmproj models built in memory.
// Every key and tensor the reader refuses is refused with its text, the registry refuses another architecture or projector, and the plan, its inputs and an encode hold their shapes.
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/cpu/cpu_backend.hpp"
#include "model/arch/registry.hpp"

namespace {

int checks = 0;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

void refused(const std::function<void()>& f, const std::string& text, const std::string& what) {
    try {
        f();
    } catch (const std::exception& e) {
        require(std::string(e.what()).find(text) != std::string::npos, what + ": refused with \"" + e.what() + "\", not \"" + text + "\"");
        return;
    }
    throw std::runtime_error(what + ": not refused");
}

constexpr uint64_t D = 32, F = 64, H = 2, P = 16, O = 48, SIDE = 4, L = 2;

void add_tensor(gguf::GGUFModel& m, const std::string& name, std::vector<uint64_t> shape) {
    size_t count = 1;
    for (uint64_t d : shape) count *= size_t(d);
    const size_t at = m.blob.size();
    m.blob.resize(at + count * sizeof(float));
    for (size_t i = 0; i < count; ++i) {
        const float v = float(int((i * 17 + m.tensors.size() * 7) % 29) - 14) / 64.0f;
        std::memcpy(m.blob.data() + at + i * sizeof(float), &v, sizeof(v));
    }
    m.tensors.push_back({name, std::move(shape), quant::GGML_TYPE_F32, 0});
    m.offsets.push_back(at);
}

gguf::MetaValue u32(uint64_t v) {
    gguf::MetaValue m;
    m.vtype = gguf::V_UINT32;
    m.u = v;
    return m;
}
gguf::MetaValue str(const std::string& s) {
    gguf::MetaValue m;
    m.vtype = gguf::V_STRING;
    m.s = s;
    return m;
}
gguf::MetaValue boolean(bool b) {
    gguf::MetaValue m;
    m.vtype = gguf::V_BOOL;
    m.b = b;
    return m;
}
gguf::MetaValue floats(std::vector<float> v) {
    gguf::MetaValue m;
    m.vtype = gguf::V_ARRAY;
    m.u = gguf::V_FLOAT32;
    for (float f : v) {
        gguf::MetaValue e;
        e.vtype = gguf::V_FLOAT32;
        std::memcpy(&e.fb, &f, sizeof(f));
        m.arr.push_back(e);
    }
    return m;
}
gguf::MetaValue bools(std::vector<bool> v) {
    gguf::MetaValue m;
    m.vtype = gguf::V_ARRAY;
    m.u = gguf::V_BOOL;
    for (bool b : v) m.arr.push_back(boolean(b));
    return m;
}

void set(gguf::GGUFModel& m, const std::string& key, gguf::MetaValue v) {
    for (auto& kv : m.kv)
        if (kv.first == key) {
            kv.second = std::move(v);
            return;
        }
    m.kv.push_back({key, std::move(v)});
}
void erase(gguf::GGUFModel& m, const std::string& key) {
    for (size_t i = 0; i < m.kv.size(); ++i)
        if (m.kv[i].first == key) {
            m.kv.erase(m.kv.begin() + std::ptrdiff_t(i));
            return;
        }
}
void erase_tensor(gguf::GGUFModel& m, const std::string& name) {
    for (size_t i = 0; i < m.tensors.size(); ++i)
        if (m.tensors[i].name == name) {
            m.tensors.erase(m.tensors.begin() + std::ptrdiff_t(i));
            m.offsets.erase(m.offsets.begin() + std::ptrdiff_t(i));
            return;
        }
}

// Two blocks 32 wide of two heads, an MLP of 64, a 4 by 4 learned grid and 48-wide rows; `edit` changes the file before it is read.
gguf::GGUFModel tiny(const std::function<void(gguf::GGUFModel&)>& edit = {}) {
    gguf::GGUFModel m;
    set(m, "general.architecture", str("clip"));
    set(m, "clip.projector_type", str("qwen3vl_merger"));
    set(m, "clip.use_gelu", boolean(true));
    set(m, "clip.vision.embedding_length", u32(D));
    set(m, "clip.vision.feed_forward_length", u32(F));
    set(m, "clip.vision.block_count", u32(L));
    set(m, "clip.vision.attention.head_count", u32(H));
    set(m, "clip.vision.patch_size", u32(P));
    set(m, "clip.vision.spatial_merge_size", u32(2));
    set(m, "clip.vision.projection_dim", u32(O));
    set(m, "clip.vision.image_mean", floats({0.5f, 0.5f, 0.5f}));
    set(m, "clip.vision.image_std", floats({0.5f, 0.5f, 0.5f}));
    set(m, "clip.vision.is_deepstack_layers", bools(std::vector<bool>(L, false)));
    add_tensor(m, "v.patch_embd.weight", {P, P, 3, D});
    add_tensor(m, "v.patch_embd.weight.1", {P, P, 3, D});
    add_tensor(m, "v.patch_embd.bias", {D});
    add_tensor(m, "v.position_embd.weight", {D, SIDE * SIDE});
    for (uint64_t l = 0; l < L; ++l) {
        const std::string n = "v.blk." + std::to_string(l) + ".";
        add_tensor(m, n + "ln1.weight", {D});
        add_tensor(m, n + "ln1.bias", {D});
        add_tensor(m, n + "attn_qkv.weight", {D, 3 * D});
        add_tensor(m, n + "attn_qkv.bias", {3 * D});
        add_tensor(m, n + "attn_out.weight", {D, D});
        add_tensor(m, n + "attn_out.bias", {D});
        add_tensor(m, n + "ln2.weight", {D});
        add_tensor(m, n + "ln2.bias", {D});
        add_tensor(m, n + "ffn_up.weight", {D, F});
        add_tensor(m, n + "ffn_up.bias", {F});
        add_tensor(m, n + "ffn_down.weight", {F, D});
        add_tensor(m, n + "ffn_down.bias", {D});
    }
    add_tensor(m, "v.post_ln.weight", {D});
    add_tensor(m, "v.post_ln.bias", {D});
    add_tensor(m, "mm.0.weight", {4 * D, 4 * D});
    add_tensor(m, "mm.0.bias", {4 * D});
    add_tensor(m, "mm.2.weight", {4 * D, O});
    add_tensor(m, "mm.2.bias", {O});
    if (edit) edit(m);
    return m;
}

// The encoder over a file on a backend, the file kept alive beside it.
struct Opened {
    gguf::GGUFModel file;
    infer::VisionWeights weights;
    std::unique_ptr<infer::VisionEncoder> encoder;
};
std::unique_ptr<Opened> open(gguf::GGUFModel m, backend::Backend& b) {
    auto o = std::make_unique<Opened>();
    o->file = std::move(m);
    o->weights = infer::vision_weights(o->file);
    o->encoder = std::make_unique<infer::VisionEncoder>(o->weights.arch, o->weights.tensors, b, backend::Dtype::f32);
    return o;
}

// A CPU without one of the encoder's ops.
struct LackingCpu : backend::CpuBackend {
    backend::Op lack;
    explicit LackingCpu(backend::Op op) : lack(op) {}
    bool implements(backend::Op op) const override { return op != lack; }
};

void refusals() {
    backend::CpuBackend cpu;
    const auto reads = [&](const std::function<void(gguf::GGUFModel&)>& edit) { return [&cpu, edit] { open(tiny(edit), cpu); }; };
    refused(reads([](gguf::GGUFModel& m) { set(m, "general.architecture", str("siglip")); }), "unsupported vision encoder general.architecture", "another architecture");
    refused(reads([](gguf::GGUFModel& m) { set(m, "clip.projector_type", str("gemma3")); }), "unsupported vision encoder clip.projector_type", "another projector");
    refused(reads([](gguf::GGUFModel& m) { erase(m, "clip.projector_type"); }), "unsupported vision encoder clip.projector_type", "no projector");
    for (const char* key : {"clip.vision.embedding_length", "clip.vision.feed_forward_length", "clip.vision.block_count", "clip.vision.attention.head_count",
                            "clip.vision.patch_size", "clip.vision.spatial_merge_size", "clip.vision.projection_dim"})
        refused(reads([key](gguf::GGUFModel& m) { erase(m, key); }), std::string("missing metadata ") + key, std::string("no ") + key);
    for (const char* key : {"clip.vision.image_mean", "clip.vision.image_std"}) {
        refused(reads([key](gguf::GGUFModel& m) { erase(m, key); }), std::string("invalid metadata ") + key, std::string("no ") + key);
        refused(reads([key](gguf::GGUFModel& m) { set(m, key, floats({0.5f, 0.5f})); }), std::string("invalid metadata ") + key, std::string("two values of ") + key);
    }
    refused(reads([](gguf::GGUFModel& m) { set(m, "clip.vision.image_std", floats({0.5f, 0.0f, 0.5f})); }), "invalid metadata clip.vision.image_std", "a zero std");
    refused(reads([](gguf::GGUFModel& m) { set(m, "clip.use_gelu", boolean(false)); }), "unsupported metadata clip.use_gelu", "no GELU");
    refused(reads([](gguf::GGUFModel& m) { erase(m, "clip.use_gelu"); }), "unsupported metadata clip.use_gelu", "GELU unstated");
    refused(reads([](gguf::GGUFModel& m) { set(m, "clip.vision.is_deepstack_layers", bools({false, true})); }), "deepstack layers are not supported",
            "a deepstack layer");
    refused(reads([](gguf::GGUFModel& m) { set(m, "clip.vision.spatial_merge_size", u32(3)); }), "clip.vision.spatial_merge_size (only 2", "a merge of 3");
    refused(reads([](gguf::GGUFModel& m) { set(m, "clip.vision.attention.head_count", u32(16)); }), "a multiple of 4 is required", "heads of 2");
    refused(reads([](gguf::GGUFModel& m) { erase_tensor(m, "v.position_embd.weight"); }), "missing tensor v.position_embd.weight", "no positions");
    refused(reads([](gguf::GGUFModel& m) { erase_tensor(m, "v.patch_embd.weight.1"); }), "missing tensor v.patch_embd.weight.1", "one temporal kernel");
    refused(reads([](gguf::GGUFModel& m) { erase_tensor(m, "v.blk.1.ffn_down.bias"); }), "missing tensor v.blk.1.ffn_down.bias", "a missing bias");
    refused(reads([](gguf::GGUFModel& m) {
                erase_tensor(m, "v.patch_embd.weight");
                add_tensor(m, "v.patch_embd.weight", {P, P, 2, D});
            }),
            "incompatible tensor layout v.patch_embd.weight", "a two-channel kernel");
    refused(reads([](gguf::GGUFModel& m) {
                erase_tensor(m, "mm.2.weight");
                add_tensor(m, "mm.2.weight", {4 * D, O + 1});
            }),
            "incompatible tensor layout mm.2.weight", "a merger of another width");
    refused(reads([](gguf::GGUFModel& m) {
                erase_tensor(m, "v.position_embd.weight");
                add_tensor(m, "v.position_embd.weight", {D, 15});
            }),
            "the learned positions are not a square grid", "positions of 15");
    for (backend::Op op : {backend::Op::layer_norm, backend::Op::gelu, backend::Op::add_bias, backend::Op::mix_rows, backend::Op::rope_streams,
                           backend::Op::attention_full}) {
        LackingCpu lacking(op);
        refused([&] { open(tiny(), lacking); }, std::string("needs ") + backend::op_name(op), std::string("a device without ") + backend::op_name(op));
    }
}

void plan_and_encode() {
    backend::CpuBackend cpu;
    cpu.set_threads(2);
    auto o = open(tiny(), cpu);
    const infer::VisionArchitecture& a = o->encoder->architecture();
    const infer::EncoderPlan& p = o->encoder->plan();
    require(p.roles.size() == 4 + 12 * L + 6 && p.role_ids == p.roles.size(), "the plan's roles");
    require(p.heads == H && p.head_dim == D / H && p.out_width == O, "the plan's attention and rows");
    require(p.slots[p.input_slot] == 3 * P * P, "a patch's input row");
    require(a.tokens(14, 20) == 70 && a.positions() == infer::VisionPositions::grid, "tokens and positions");
    const infer::ImageTokenTexts t = a.token_texts();
    require(t.start == "<|vision_start|>" && t.pad == "<|image_pad|>" && t.end == "<|vision_end|>", "the template's image tokens");
    const image::Params pp = a.preprocess();
    require(pp.patch == P && pp.merge == 2 && pp.min_pixels == 65536 && pp.max_pixels == 16777216 && pp.mean[1] == 0.5f && pp.std[2] == 0.5f,
            "the preprocessing parameters");
    // Inputs: positions in merge-block order, taps on the 4 by 4 grid with aligned corners, each patch's weights summing to one.
    const infer::EncodeInputs in = a.inputs(4, 6);
    require(in.positions.size() == 48 && in.taps.size() == 96 && in.weights.size() == 96, "an image's inputs");
    require(in.positions[0] == 0 && in.positions[1] == 0 && in.positions[2] == 0 && in.positions[3] == 1 && in.positions[4] == 1 && in.positions[5] == 0 &&
                in.positions[8] == 0 && in.positions[9] == 2,
            "patch positions in merge-block order");
    for (size_t k = 0; k < 24; ++k) {
        float sum = 0.0f;
        for (int j = 0; j < 4; ++j) {
            sum += in.weights[4 * k + size_t(j)];
            require(in.taps[4 * k + size_t(j)] < SIDE * SIDE, "a tap past the grid");
        }
        require(std::fabs(sum - 1.0f) < 1e-6f, "a patch's tap weights");
    }
    refused([&] { a.inputs(3, 4); }, "not whole merge blocks", "an odd grid");
    // An encode: rows of the plan's width, finite, the same bits encoded again and on one thread.
    image::Prepared img;
    img.grid_h = 4;
    img.grid_w = 6;
    img.patches.resize(24 * 3 * P * P);
    for (size_t i = 0; i < img.patches.size(); ++i) img.patches[i] = float(int(i % 23) - 11) / 11.0f;
    const std::vector<float> rows = o->encoder->encode(img), again = o->encoder->encode(img);
    require(rows.size() == 6 * O, "an image's rows");
    for (float v : rows) require(std::isfinite(v), "a row that is not finite");
    require(std::memcmp(rows.data(), again.data(), rows.size() * sizeof(float)) == 0, "an image encoded again");
    backend::CpuBackend one;
    one.set_threads(1);
    auto o1 = open(tiny(), one);
    const std::vector<float> single = o1->encoder->encode(img);
    require(std::memcmp(rows.data(), single.data(), rows.size() * sizeof(float)) == 0, "an image on one thread");
    img.patches.pop_back();
    refused([&] { o->encoder->encode(img); }, "do not fill its grid", "patches short of the grid");
}

}  // namespace

int main() {
    try {
        refusals();
        plan_and_encode();
    } catch (const std::exception& e) {
        std::cerr << "arch-qwen3vl-vision: FAIL: " << e.what() << "\n";
        return 1;
    }
    std::cout << "arch-qwen3vl-vision: " << checks << " checks passed\n";
    return 0;
}
