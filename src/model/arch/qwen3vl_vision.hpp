#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "backends/backend.hpp"
#include "format/gguf.hpp"
#include "image/image.hpp"
#include "model/arch/metadata.hpp"
#include "model/encoder.hpp"
#include "model/weights.hpp"

// The vision tower of the Qwen 3.5, 3.6 and 3.8 families and Qwen3.8-Flash-Next, read from an mmproj GGUF whose projector is qwen3vl_merger (docs/VISION.md, section 3).

namespace infer::qwen3vl_vision {

struct Config {
    size_t width = 0, mlp = 0, blocks = 0, heads = 0, patch = 0, merge = 0, out = 0;
    size_t side = 0;   // the learned position grid's side
    float eps = 1e-6f;
    float mean[3] = {0, 0, 0}, std[3] = {0, 0, 0};
};

// The processor's pixel bounds, which the GGUF does not carry: the documented defaults of every Qwen 3.5 and 3.8 checkpoint's preprocessor_config.json.
inline constexpr uint64_t kMinPixels = 65536, kMaxPixels = 16777216;
// The vision rope's base, which HF fixes and the GGUF does not carry.
inline constexpr double kRopeTheta = 10000.0;
// smart_resize refuses an aspect ratio past this, which bounds the longest side of a grid a pixel bound allows.
inline constexpr double kMaxAspect = 200.0;

// Every key the math reads, and the refusal of each value it does not implement.
inline Config read_config(const gguf::GGUFModel& m) {
    using namespace metadata;
    Config c;
    c.width = size_t(integer(m, "clip.vision.embedding_length"));
    c.mlp = size_t(integer(m, "clip.vision.feed_forward_length"));
    c.blocks = size_t(integer(m, "clip.vision.block_count"));
    c.heads = size_t(integer(m, "clip.vision.attention.head_count"));
    c.patch = size_t(integer(m, "clip.vision.patch_size"));
    c.merge = size_t(integer(m, "clip.vision.spatial_merge_size"));
    c.out = size_t(integer(m, "clip.vision.projection_dim"));
    c.eps = float(real(m, "clip.vision.attention.layer_norm_epsilon", 1e-6f));
    three_floats(m, "clip.vision.image_mean", c.mean);
    three_floats(m, "clip.vision.image_std", c.std);
    for (int i = 0; i < 3; ++i)
        if (!(c.std[i] > 0)) throw std::runtime_error("inference: invalid metadata clip.vision.image_std");
    if (!boolean(m, "clip.use_gelu", false)) throw std::runtime_error("inference: unsupported metadata clip.use_gelu (the merger's GELU is required)");
    if (const auto* d = m.find("clip.vision.is_deepstack_layers")) {
        if (d->vtype != gguf::V_ARRAY || d->u != gguf::V_BOOL) throw std::runtime_error("inference: invalid metadata clip.vision.is_deepstack_layers");
        for (const auto& v : d->arr)
            if (v.b) throw std::runtime_error("inference: unsupported metadata clip.vision.is_deepstack_layers (deepstack layers are not supported)");
    }
    if (c.merge != 2) throw std::runtime_error("inference: unsupported metadata clip.vision.spatial_merge_size (only 2 is supported)");
    if (c.width % c.heads || (c.width / c.heads) % 4) throw std::runtime_error("inference: unsupported vision head width (a multiple of 4 is required)");
    if ((c.width / c.heads) / 2 > 128) throw std::runtime_error("inference: unsupported vision head width (at most 256)");
    // The learned positions are a square grid, whose side their table's rows give.
    for (const auto& t : m.tensors)
        if (t.name == "v.position_embd.weight" && t.ne.size() >= 2) c.side = size_t(std::llround(std::sqrt(double(t.ne[1]))));
    if (c.side < 2) throw std::runtime_error("inference: missing tensor v.position_embd.weight");
    return c;
}

// The roles, by id: the patch embedding, the learned positions, each block's twelve, the merger's six.
enum : uint16_t { patch0, patch1, patch_bias, pos_table, kBlockBase };
inline constexpr uint16_t kPerBlock = 12;
enum : uint16_t { ln1_w, ln1_b, qkv_w, qkv_b, out_w, out_b, ln2_w, ln2_b, up_w, up_b, down_w, down_b };
enum : uint16_t { post_w, post_b, mm0_w, mm0_b, mm2_w, mm2_b, kMerger };

// The arena's slots, the floats one patch takes in each.
enum : size_t { s_pixels, s_x, s_h, s_q, s_k, s_v, s_attn, s_mlp, s_merge, s_out, kSlots };

class Module final : public VisionArchitecture {
public:
    explicit Module(Config c) : c_(c) {}

    // N tokens bound smart_resize's max_pixels at N merged patches of (patch * merge)^2 pixels; an image below min_pixels is still raised to it, as HF's processor raises it, and any count from 1 is taken.
    image::Params preprocess(std::optional<uint64_t> max_tokens) const override {
        image::Params p;
        p.patch = uint32_t(c_.patch);
        p.merge = uint32_t(c_.merge);
        for (int i = 0; i < 3; ++i) {
            p.mean[i] = c_.mean[i];
            p.std[i] = c_.std[i];
        }
        p.min_pixels = kMinPixels;
        p.max_pixels = kMaxPixels;
        if (max_tokens) {
            const uint64_t side = uint64_t(c_.patch) * c_.merge;
            if (!*max_tokens) throw std::runtime_error("vision: an image takes at least 1 token, not 0");
            if (*max_tokens > std::numeric_limits<uint64_t>::max() / (side * side)) throw std::runtime_error("vision: " + std::to_string(*max_tokens) + " tokens an image is past any image's size");
            p.max_pixels = *max_tokens * side * side;
        }
        return p;
    }
    uint64_t max_tokens(const image::Params& p) const override { return p.max_pixels / (uint64_t(p.patch) * p.merge * p.patch * p.merge); }
    // smart_resize raises an image to min_pixels or above, and shrinks one to sides floored to the factor, each at least half its real-valued target, so to max_pixels / 4 at the least.
    uint64_t min_tokens(const image::Params& p) const override {
        const uint64_t side = uint64_t(p.patch) * p.merge;
        return std::max<uint64_t>(1, std::min(p.min_pixels, p.max_pixels / 4) / (side * side));
    }
    size_t tokens(uint32_t grid_h, uint32_t grid_w) const override { return size_t(grid_h) * grid_w / (c_.merge * c_.merge); }
    // smart_resize keeps an image within max_pixels, except where it raises one to min_pixels with its sides ceiled, or floors one past the aspect bound to a side of one factor.
    EncodeBound bound(const image::Params& p) const override {
        const double f = double(p.patch) * p.merge, lo = double(p.min_pixels), hi = double(p.max_pixels);
        const double raised = (std::sqrt(lo * kMaxAspect) + f) * (std::sqrt(lo / kMaxAspect) + f), floored = std::sqrt(hi * kMaxAspect) * f;
        EncodeBound b;
        b.patches = size_t(std::ceil(std::max({hi, raised, floored}) / (double(p.patch) * p.patch)));
        b.arena_patches = b.patches;
        const size_t crossing = size_t(std::sqrt(double(kPieceProducts) / double(2 * c_.width)));
        b.attention_rows = std::max(attention_rows(std::min(b.patches, crossing), c_.width), attention_rows(std::min(b.patches, crossing + 1), c_.width));
        return b;
    }
    VisionPositions positions() const override { return VisionPositions::grid; }
    ImageTokenTexts token_texts() const override { return {"<|vision_start|>", "<|image_pad|>", "<|vision_end|>"}; }

    EncoderPlan plan(const TensorIndex& tensors) const override {
        const size_t D = c_.width, F = c_.mlp, M2 = c_.merge * c_.merge, in = c_.patch * c_.patch * 3;
        const TensorView& pos = tensors.view(tensors.at("v.position_embd.weight"));
        if (pos.shape.size() < 2 || c_.side * c_.side != pos.shape[1]) throw std::runtime_error("inference: the learned positions are not a square grid");
        for (const char* name : {"v.patch_embd.weight", "v.patch_embd.weight.1"}) {
            const TensorView& t = tensors.view(tensors.at(name));
            if (t.shape.size() != 4 || t.shape[0] != c_.patch || t.shape[1] != c_.patch || t.shape[2] != 3 || t.shape[3] != D)
                throw std::runtime_error(std::string("inference: incompatible tensor layout ") + name);
        }
        EncoderPlan p;
        const auto role = [&](uint16_t id, const std::string& name, RoleKind kind, uint64_t rin, uint64_t rout) {
            Role r{id, Part::embed, kind, name, "", rin, rout};
            p.roles.push_back(r);
        };
        role(patch0, "v.patch_embd.weight", RoleKind::matrix, in, D);
        role(patch1, "v.patch_embd.weight.1", RoleKind::matrix, in, D);
        role(patch_bias, "v.patch_embd.bias", RoleKind::norm, D, 1);
        role(pos_table, "v.position_embd.weight", RoleKind::table, D, c_.side * c_.side);
        for (size_t l = 0; l < c_.blocks; ++l) {
            const uint16_t b = uint16_t(kBlockBase + l * kPerBlock);
            const std::string n = "v.blk." + std::to_string(l) + ".";
            role(b + ln1_w, n + "ln1.weight", RoleKind::norm, D, 1);
            role(b + ln1_b, n + "ln1.bias", RoleKind::norm, D, 1);
            role(b + qkv_w, n + "attn_qkv.weight", RoleKind::matrix, D, 3 * D);
            role(b + qkv_b, n + "attn_qkv.bias", RoleKind::norm, 3 * D, 1);
            role(b + out_w, n + "attn_out.weight", RoleKind::matrix, D, D);
            role(b + out_b, n + "attn_out.bias", RoleKind::norm, D, 1);
            role(b + ln2_w, n + "ln2.weight", RoleKind::norm, D, 1);
            role(b + ln2_b, n + "ln2.bias", RoleKind::norm, D, 1);
            role(b + up_w, n + "ffn_up.weight", RoleKind::matrix, D, F);
            role(b + up_b, n + "ffn_up.bias", RoleKind::norm, F, 1);
            role(b + down_w, n + "ffn_down.weight", RoleKind::matrix, F, D);
            role(b + down_b, n + "ffn_down.bias", RoleKind::norm, D, 1);
        }
        const uint16_t m = merger_base();
        role(m + post_w, "v.post_ln.weight", RoleKind::norm, D, 1);
        role(m + post_b, "v.post_ln.bias", RoleKind::norm, D, 1);
        role(m + mm0_w, "mm.0.weight", RoleKind::matrix, M2 * D, M2 * D);
        role(m + mm0_b, "mm.0.bias", RoleKind::norm, M2 * D, 1);
        role(m + mm2_w, "mm.2.weight", RoleKind::matrix, M2 * D, c_.out);
        role(m + mm2_b, "mm.2.bias", RoleKind::norm, c_.out, 1);
        p.role_ids = size_t(m + kMerger);
        p.ops = {backend::Op::layer_norm, backend::Op::gelu, backend::Op::add_bias, backend::Op::mix_rows, backend::Op::rope_streams,
                 backend::Op::attention_full};
        p.slots.assign(kSlots, 0);
        p.slots[s_pixels] = in;
        p.slots[s_x] = p.slots[s_h] = p.slots[s_q] = p.slots[s_k] = p.slots[s_v] = p.slots[s_attn] = D;
        p.slots[s_mlp] = F;
        p.slots[s_merge] = D;                         // a token's 4 D, a quarter a patch
        p.slots[s_out] = (c_.out + M2 - 1) / M2;      // a token's output, a quarter a patch rounded up
        p.input_slot = s_pixels;
        p.output_slot = s_out;
        p.heads = c_.heads;
        p.head_dim = D / c_.heads;
        p.out_width = c_.out;
        const size_t half = p.head_dim / 2;
        p.tables = {max_side() * half, max_side() * half};
        return p;
    }

    // The rope's cos and sin, as HF's two streams lay them out.
    void fill_tables(std::vector<std::vector<float>>& t) const override { fill_stream_rope(t[0], t[1], c_.width / c_.heads, kRopeTheta); }

    // Each patch's row and column and its four bilinear taps of the learned grid with aligned corners, in merge-block order, as HF's vision utilities compute them in F32.
    EncodeInputs inputs(uint32_t gh, uint32_t gw) const override {
        if (gh % c_.merge || gw % c_.merge) throw std::runtime_error("inference: an image grid that is not whole merge blocks");
        if (gh > max_side() || gw > max_side()) throw std::runtime_error("inference: an image grid past the rope table");
        EncodeInputs in;
        const size_t n = size_t(gh) * gw, M = c_.merge;
        in.positions.resize(2 * n);
        in.taps.resize(4 * n);
        in.weights.resize(4 * n);
        size_t k = 0;
        for (size_t by = 0; by < gh / M; ++by)
            for (size_t bx = 0; bx < gw / M; ++bx)
                for (size_t iy = 0; iy < M; ++iy)
                    for (size_t ix = 0; ix < M; ++ix, ++k) {
                        const size_t r = by * M + iy, c = bx * M + ix;
                        in.positions[2 * k] = uint32_t(r);
                        in.positions[2 * k + 1] = uint32_t(c);
                        uint32_t ht[2], wt[2];
                        float hw[2], ww[2];
                        bilinear_taps(r, gh, c_.side, true, ht, hw);
                        bilinear_taps(c, gw, c_.side, true, wt, ww);
                        for (int a = 0; a < 2; ++a)
                            for (int b = 0; b < 2; ++b) {
                                in.taps[4 * k + size_t(2 * a + b)] = uint32_t(ht[a] * c_.side + wt[b]);
                                in.weights[4 * k + size_t(2 * a + b)] = hw[a] * ww[b];
                            }
                    }
        return in;
    }

    // The patch embedding, then each block as its projections, its attention in runs of query rows of at most kPieceProducts multiply-adds, and the rest, then the merger.
    size_t pieces(size_t patches) const override { return 2 + c_.blocks * (2 + attention_runs(patches, c_.width)); }

    void encode(const EncodeStep& s, size_t piece) const override {
        backend::Backend& b = s.b;
        const size_t D = c_.width, F = c_.mlp, n = s.patches, M2 = c_.merge * c_.merge, T = n / M2;
        const size_t in = c_.patch * c_.patch * 3;
        const backend::RowRun run{n, n}, merged_run{T, T};
        const backend::RowRuns runs{&run, 1}, merged{&merged_run, 1};
        const Weight* w = s.w;
        const size_t per_block = 2 + attention_runs(n, D);
        if (piece == 0) {
            // The patch embedding: both temporal kernels on the one frame, the bias, then the learned positions' taps.
            b.matmul(w[patch0].operand(), s.slot(s_pixels), s.slot(s_x), in, D, n, runs, s.dtype);
            b.matmul_add(w[patch1].operand(), s.slot(s_pixels), s.slot(s_x), in, D, n, runs, s.dtype);
            b.add_bias(s.slot(s_x), w[patch_bias].slice(), n, D);
            b.mix_rows(s.slot(s_x), w[pos_table].slice(), D, c_.side * c_.side, s.inputs->taps.data(), s.inputs->weights.data(), n, 4);
            return;
        }
        if (piece == pieces(n) - 1) {
            // The merger: the norm over each patch, then four consecutive patches read as one token's row.
            const Weight* mw = w + merger_base();
            b.layer_norm_rows(s.slot(s_h), s.slot(s_x), mw[post_w].slice(), mw[post_b].slice(), n, D, D, c_.eps);
            b.matmul(mw[mm0_w].operand(), s.slot(s_h), s.slot(s_merge), M2 * D, M2 * D, T, merged, s.dtype);
            b.add_bias(s.slot(s_merge), mw[mm0_b].slice(), T, M2 * D);
            b.gelu(s.slot(s_merge), s.slot(s_merge), T * M2 * D, backend::GeluForm::erf);
            b.matmul(mw[mm2_w].operand(), s.slot(s_merge), s.slot(s_out), M2 * D, c_.out, T, merged, s.dtype);
            b.add_bias(s.slot(s_out), mw[mm2_b].slice(), T, c_.out);
            return;
        }
        const size_t l = (piece - 1) / per_block, part = (piece - 1) % per_block;
        const Weight* lw = w + kBlockBase + l * kPerBlock;
        const backend::Slice q = s.slot(s_q), k = s.slot(s_k), v = s.slot(s_v);
        if (part == 0) {
            // The first head_dim / 4 frequencies turn by the patch's row, the rest by its column.
            b.layer_norm_rows(s.slot(s_h), s.slot(s_x), lw[ln1_w].slice(), lw[ln1_b].slice(), n, D, D, c_.eps);
            b.matmul_group({{{lw[qkv_w].type, row_block(lw[qkv_w], 0)}, q, D}, {{lw[qkv_w].type, row_block(lw[qkv_w], D)}, k, D},
                            {{lw[qkv_w].type, row_block(lw[qkv_w], 2 * D)}, v, D}},
                           s.slot(s_h), D, n, runs, s.dtype);
            const backend::CSlice bias = lw[qkv_b].slice();
            bias_rope_kv(s, q, k, v, bias, {bias.buffer, bias.offset + D}, {bias.buffer, bias.offset + 2 * D}, n, D, c_.heads, {s.tables[0].get(), 0}, {s.tables[1].get(), 0},
                         backend::RopeForm{});
            return;
        }
        if (part + 1 < per_block) {
            attention_run(s, q, s.slot(s_attn), n, part - 1, D, c_.heads);
            return;
        }
        // The projection and its bias before the residual add, in HF's order.
        b.matmul(lw[out_w].operand(), s.slot(s_attn), s.slot(s_h), D, D, n, runs, s.dtype);
        b.add_bias(s.slot(s_h), lw[out_b].slice(), n, D);
        b.add(s.slot(s_x), s.slot(s_h), n * D);
        b.layer_norm_rows(s.slot(s_h), s.slot(s_x), lw[ln2_w].slice(), lw[ln2_b].slice(), n, D, D, c_.eps);
        b.matmul(lw[up_w].operand(), s.slot(s_h), s.slot(s_mlp), D, F, n, runs, s.dtype);
        b.add_bias(s.slot(s_mlp), lw[up_b].slice(), n, F);
        b.gelu(s.slot(s_mlp), s.slot(s_mlp), n * F, backend::GeluForm::tanh);
        b.matmul(lw[down_w].operand(), s.slot(s_mlp), s.slot(s_h), F, D, n, runs, s.dtype);
        b.add_bias(s.slot(s_h), lw[down_b].slice(), n, D);
        b.add(s.slot(s_x), s.slot(s_h), n * D);
    }

private:
    uint16_t merger_base() const { return uint16_t(kBlockBase + c_.blocks * kPerBlock); }
    // The longest grid side, in patches, the default pixel bound allows under smart_resize's aspect limit.
    size_t max_side() const { return size_t(std::sqrt(kMaxAspect * double(kMaxPixels))) / c_.patch + 1; }

    Config c_;
};

inline std::shared_ptr<const VisionArchitecture> open(const gguf::GGUFModel& m) { return std::make_shared<Module>(read_config(m)); }

}  // namespace infer::qwen3vl_vision
