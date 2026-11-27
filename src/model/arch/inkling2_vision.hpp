#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

// The vision tower of Inkling-2, read from an mmproj GGUF whose projector is inkling2v (docs/VISION.md, section 3).

namespace infer::inkling2_vision {

struct Config {
    size_t width = 0, mlp = 0, blocks = 0, heads = 0, patch = 0, merge = 0, out = 0, hidden = 0;
    size_t side = 0;   // the learned position grid's side
    float eps = 1e-5f;
    float mean[3] = {0, 0, 0}, std[3] = {0, 0, 0};
};

// The vision rope's base, which HF fixes and the GGUF does not carry.
inline constexpr double kRopeTheta = 10000.0;
// An image's tokens at most, the model's practical bound and the default, so 1080p and 1440p screenshots fit (docs/VISION.md, section 12).
// The processor never resizes, so the bound refuses a larger image rather than shrinking it; it also bounds the rope table and the attention's rows.
inline constexpr uint64_t kMaxTokens = 4096;

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
    c.eps = float(real(m, "clip.vision.attention.layer_norm_epsilon", 1e-5f));
    three_floats(m, "clip.vision.image_mean", c.mean);
    three_floats(m, "clip.vision.image_std", c.std);
    for (int i = 0; i < 3; ++i)
        if (!(c.std[i] > 0)) throw std::runtime_error("inference: invalid metadata clip.vision.image_std");
    if (c.merge != 2) throw std::runtime_error("inference: unsupported metadata clip.vision.spatial_merge_size (only 2 is supported)");
    if (!c.heads || c.width % c.heads || (c.width / c.heads) % 4) throw std::runtime_error("inference: unsupported vision head width (a multiple of 4 is required)");
    if ((c.width / c.heads) / 2 > 128) throw std::runtime_error("inference: unsupported vision head width (at most 256)");
    // The learned positions are a square grid, whose side their table's rows give; the merger's hidden width is its first matrix's outputs.
    for (const auto& t : m.tensors) {
        if (t.name == "v.position_embd.weight" && t.ne.size() >= 2) c.side = size_t(std::llround(std::sqrt(double(t.ne[1]))));
        if (t.name == "mm.1.weight" && t.ne.size() == 2) c.hidden = size_t(t.ne[1]);
    }
    if (c.side < 2) throw std::runtime_error("inference: missing tensor v.position_embd.weight");
    if (!c.hidden) throw std::runtime_error("inference: missing tensor mm.1.weight");
    return c;
}

// The roles, by id: the patch embedding, the learned positions, the class row's two vectors, the norm before the blocks, each block's eighteen, the merger's two.
enum : uint16_t { patch_w, pos_table, cls, cls_pos, pre_w, pre_b, kBlockBase };
inline constexpr uint16_t kPerBlock = 18;
enum : uint16_t { ln1_w, ln1_b, q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b, ls1, ln2_w, ln2_b, up_w, up_b, down_w, down_b, ls2 };
static_assert(ls2 + 1 == kPerBlock, "a block's roles");
enum : uint16_t { mm1_w, mm2_w, kMerger };

// The arena's slots, the floats one row takes in each; the merger's hidden rows reuse s_mlp, a token's hidden row a quarter of four patches' F.
enum : size_t { s_pixels, s_x, s_h, s_q, s_k, s_v, s_attn, s_mlp, s_out, kSlots };

class Module final : public VisionArchitecture {
public:
    explicit Module(Config c) : c_(c) {}

    // Inkling2ImageProcessor never resizes: an image is padded at its bottom and right to whole tokens, composited onto its checkerboard, and refused past the token bound.
    image::Params preprocess(std::optional<uint64_t> max_tokens) const override {
        image::Params p;
        p.patch = uint32_t(c_.patch);
        p.merge = uint32_t(c_.merge);
        for (int i = 0; i < 3; ++i) {
            p.mean[i] = c_.mean[i];
            p.std[i] = c_.std[i];
        }
        p.fit = image::Fit::pad_end;
        p.pad_value = -1.0f;
        p.matte = image::Matte::checker;
        p.max_rows = kMaxTokens;
        if (max_tokens) {
            if (!*max_tokens) throw std::runtime_error("vision: an image takes at least 1 token, not 0");
            if (*max_tokens > kMaxTokens) throw std::runtime_error("vision: an image takes at most " + std::to_string(kMaxTokens) + " tokens, not " + std::to_string(*max_tokens));
            p.max_rows = *max_tokens;
        }
        return p;
    }
    uint64_t max_tokens(const image::Params& p) const override { return p.max_rows; }
    uint64_t min_tokens(const image::Params&) const override { return 1; }
    size_t tokens(uint32_t grid_h, uint32_t grid_w) const override { return size_t(grid_h) * grid_w / (c_.merge * c_.merge); }
    // An image of at most max_rows tokens has that many merge blocks of patches; its attention runs over them and the class row, whose arena row the runner adds (EncoderPlan::extra_rows).
    EncodeBound bound(const image::Params& p) const override {
        EncodeBound b;
        b.patches = size_t(p.max_rows) * c_.merge * c_.merge;
        b.arena_patches = b.patches;
        const size_t n = b.patches + 1, crossing = size_t(std::sqrt(double(kPieceProducts) / double(2 * c_.width)));
        b.attention_rows = std::max(attention_rows(std::min(n, crossing), c_.width), attention_rows(std::min(n, crossing + 1), c_.width));
        return b;
    }
    VisionPositions positions() const override { return VisionPositions::text; }
    ImageTokenTexts token_texts() const override { return {"<|content_image|>", "<|unused_200054|>", "<|end_message|>"}; }

    EncoderPlan plan(const TensorIndex& tensors) const override {
        const size_t D = c_.width, F = c_.mlp, H = c_.hidden, M2 = c_.merge * c_.merge, in = c_.patch * c_.patch * 3;
        const TensorView& pos = tensors.view(tensors.at("v.position_embd.weight"));
        if (pos.shape.size() < 2 || c_.side * c_.side != pos.shape[1]) throw std::runtime_error("inference: the learned positions are not a square grid");
        const TensorView& pe = tensors.view(tensors.at("v.patch_embd.weight"));
        if (pe.shape.size() != 4 || pe.shape[0] != c_.patch || pe.shape[1] != c_.patch || pe.shape[2] != 3 || pe.shape[3] != D)
            throw std::runtime_error("inference: incompatible tensor layout v.patch_embd.weight");
        EncoderPlan p;
        const auto role = [&](uint16_t id, const std::string& name, RoleKind kind, uint64_t rin, uint64_t rout) { p.roles.push_back(Role{id, Part::embed, kind, name, "", rin, rout}); };
        role(patch_w, "v.patch_embd.weight", RoleKind::matrix, in, D);
        role(pos_table, "v.position_embd.weight", RoleKind::table, D, c_.side * c_.side);
        role(cls, "v.class_embd", RoleKind::norm, D, 1);
        role(cls_pos, "v.class_position_embd.weight", RoleKind::norm, D, 1);
        role(pre_w, "v.pre_ln.weight", RoleKind::norm, D, 1);
        role(pre_b, "v.pre_ln.bias", RoleKind::norm, D, 1);
        for (size_t l = 0; l < c_.blocks; ++l) {
            const uint16_t b = block_base(l);
            const std::string n = "v.blk." + std::to_string(l) + ".";
            role(b + ln1_w, n + "ln1.weight", RoleKind::norm, D, 1);
            role(b + ln1_b, n + "ln1.bias", RoleKind::norm, D, 1);
            role(b + q_w, n + "attn_q.weight", RoleKind::matrix, D, D);
            role(b + q_b, n + "attn_q.bias", RoleKind::norm, D, 1);
            role(b + k_w, n + "attn_k.weight", RoleKind::matrix, D, D);
            role(b + k_b, n + "attn_k.bias", RoleKind::norm, D, 1);
            role(b + v_w, n + "attn_v.weight", RoleKind::matrix, D, D);
            role(b + v_b, n + "attn_v.bias", RoleKind::norm, D, 1);
            role(b + out_w, n + "attn_out.weight", RoleKind::matrix, D, D);
            role(b + out_b, n + "attn_out.bias", RoleKind::norm, D, 1);
            role(b + ls1, n + "ls1.weight", RoleKind::norm, D, 1);
            role(b + ln2_w, n + "ln2.weight", RoleKind::norm, D, 1);
            role(b + ln2_b, n + "ln2.bias", RoleKind::norm, D, 1);
            role(b + up_w, n + "ffn_up.weight", RoleKind::matrix, D, F);
            role(b + up_b, n + "ffn_up.bias", RoleKind::norm, F, 1);
            role(b + down_w, n + "ffn_down.weight", RoleKind::matrix, F, D);
            role(b + down_b, n + "ffn_down.bias", RoleKind::norm, D, 1);
            role(b + ls2, n + "ls2.weight", RoleKind::norm, D, 1);
        }
        const uint16_t m = merger_base();
        role(m + mm1_w, "mm.1.weight", RoleKind::matrix, M2 * D, H);
        role(m + mm2_w, "mm.2.weight", RoleKind::matrix, H, c_.out);
        p.role_ids = size_t(m + kMerger);
        p.ops = {backend::Op::layer_norm, backend::Op::gelu, backend::Op::add_bias, backend::Op::mix_rows, backend::Op::rope_streams, backend::Op::rope_forms,
                 backend::Op::attention_full};
        p.slots.assign(kSlots, 0);
        p.slots[s_pixels] = in;
        p.slots[s_x] = p.slots[s_h] = p.slots[s_q] = p.slots[s_k] = p.slots[s_v] = p.slots[s_attn] = D;
        p.slots[s_mlp] = std::max(F, (H + M2 - 1) / M2);
        p.slots[s_out] = (c_.out + M2 - 1) / M2;   // a token's output, a quarter a patch rounded up
        p.input_slot = s_pixels;
        p.output_slot = s_out;
        p.heads = c_.heads;
        p.head_dim = D / c_.heads;
        p.out_width = c_.out;
        p.extra_rows = 1;   // the class row, after the patches
        const size_t half = p.head_dim / 2;
        p.tables = {(max_side() + 1) * half, (max_side() + 1) * half};
        return p;
    }

    // The rope's cos and sin, a row a position from 0 to the longest side, the frequencies as HF's two axes lay them out.
    void fill_tables(std::vector<std::vector<float>>& t) const override { fill_stream_rope(t[0], t[1], c_.width / c_.heads, kRopeTheta); }

    // Each patch's column and row from 1 and its four bilinear taps of the learned grid at half-cell centres, in merge-block order; the class row last at (0, 0).
    EncodeInputs inputs(uint32_t gh, uint32_t gw) const override {
        if (gh % c_.merge || gw % c_.merge) throw std::runtime_error("inference: an image grid that is not whole merge blocks");
        if (gh > max_side() || gw > max_side()) throw std::runtime_error("inference: an image grid past the rope table");
        EncodeInputs in;
        const size_t n = size_t(gh) * gw, M = c_.merge;
        in.positions.assign(2 * (n + 1), 0);
        in.taps.resize(4 * n);
        in.weights.resize(4 * n);
        size_t k = 0;
        for (size_t by = 0; by < gh / M; ++by)
            for (size_t bx = 0; bx < gw / M; ++bx)
                for (size_t iy = 0; iy < M; ++iy)
                    for (size_t ix = 0; ix < M; ++ix, ++k) {
                        const size_t r = by * M + iy, c = bx * M + ix;
                        in.positions[2 * k] = uint32_t(c + 1);
                        in.positions[2 * k + 1] = uint32_t(r + 1);
                        uint32_t ht[2], wt[2];
                        float hw[2], ww[2];
                        bilinear_taps(r, gh, c_.side, false, ht, hw);
                        bilinear_taps(c, gw, c_.side, false, wt, ww);
                        for (int a = 0; a < 2; ++a)
                            for (int b = 0; b < 2; ++b) {
                                in.taps[4 * k + size_t(2 * a + b)] = uint32_t(ht[a] * c_.side + wt[b]);
                                in.weights[4 * k + size_t(2 * a + b)] = hw[a] * ww[b];
                            }
                    }
        return in;
    }

    // The embeddings and the norm before the blocks, then each block as its projections, its attention in runs of query rows over the patches and the class row, and the rest, then the merger.
    size_t pieces(size_t patches) const override { return 2 + c_.blocks * (2 + attention_runs(patches + 1, c_.width)); }

    void encode(const EncodeStep& s, size_t piece) const override {
        backend::Backend& b = s.b;
        const size_t D = c_.width, F = c_.mlp, H = c_.hidden, n = s.patches, rows = n + 1, M2 = c_.merge * c_.merge, T = n / M2;
        const size_t in = c_.patch * c_.patch * 3;
        const backend::RowRun run{n, n}, all_run{rows, rows}, merged_run{T, T};
        const backend::RowRuns runs{&run, 1}, all{&all_run, 1}, merged{&merged_run, 1};
        const Weight* w = s.w;
        const size_t per_block = 2 + attention_runs(rows, D);
        if (piece == 0) {
            // The patch embedding, which has no bias, and the learned positions' taps; the class row, its two vectors summed; then the norm over every row into the residual stream.
            const backend::Slice h = s.slot(s_h), x = s.slot(s_x), cls_row{h.buffer, h.offset + n * D};
            b.matmul(w[patch_w].operand(), s.slot(s_pixels), h, in, D, n, runs, s.dtype);
            b.mix_rows(h, w[pos_table].slice(), D, c_.side * c_.side, s.inputs->taps.data(), s.inputs->weights.data(), n, 4);
            const backend::CSlice c0 = w[cls].slice();
            b.copy(*h.buffer, cls_row.offset * sizeof(float), *c0.buffer, c0.offset * sizeof(float), D * sizeof(float));
            b.add(cls_row, w[cls_pos].slice(), D);
            b.layer_norm_rows(x, h, w[pre_w].slice(), w[pre_b].slice(), rows, D, D, c_.eps);
            return;
        }
        if (piece == pieces(n) - 1) {
            // The merger reads the patch rows only, four consecutive patches as one token's row, with no norm before it and no biases.
            const Weight* mw = w + merger_base();
            b.matmul(mw[mm1_w].operand(), s.slot(s_x), s.slot(s_mlp), M2 * D, H, T, merged, s.dtype);
            b.gelu(s.slot(s_mlp), s.slot(s_mlp), T * H, backend::GeluForm::erf);
            b.matmul(mw[mm2_w].operand(), s.slot(s_mlp), s.slot(s_out), H, c_.out, T, merged, s.dtype);
            return;
        }
        const size_t l = (piece - 1) / per_block, part = (piece - 1) % per_block;
        const Weight* lw = w + block_base(l);
        const backend::Slice q = s.slot(s_q), k = s.slot(s_k), v = s.slot(s_v);
        if (part == 0) {
            // Each interleaved pair of the first head_dim / 4 turns by its row's column, the rest by its row's row.
            b.layer_norm_rows(s.slot(s_h), s.slot(s_x), lw[ln1_w].slice(), lw[ln1_b].slice(), rows, D, D, c_.eps);
            b.matmul_group({{lw[q_w].operand(), q, D}, {lw[k_w].operand(), k, D}, {lw[v_w].operand(), v, D}}, s.slot(s_h), D, rows, all, s.dtype);
            bias_rope_kv(s, q, k, v, lw[q_b].slice(), lw[k_b].slice(), lw[v_b].slice(), rows, D, c_.heads, {s.tables[0].get(), 0}, {s.tables[1].get(), 0},
                         backend::RopeForm{true});
            return;
        }
        if (part + 1 < per_block) {
            attention_run(s, q, s.slot(s_attn), rows, part - 1, D, c_.heads);
            return;
        }
        // Layer scale multiplies each branch's output, its bias added first, before the residual add, in HF's order.
        b.matmul(lw[out_w].operand(), s.slot(s_attn), s.slot(s_h), D, D, rows, all, s.dtype);
        b.add_bias(s.slot(s_h), lw[out_b].slice(), rows, D, lw[ls1].slice());
        b.add(s.slot(s_x), s.slot(s_h), rows * D);
        b.layer_norm_rows(s.slot(s_h), s.slot(s_x), lw[ln2_w].slice(), lw[ln2_b].slice(), rows, D, D, c_.eps);
        b.matmul(lw[up_w].operand(), s.slot(s_h), s.slot(s_mlp), D, F, rows, all, s.dtype);
        b.add_bias(s.slot(s_mlp), lw[up_b].slice(), rows, F);
        b.gelu(s.slot(s_mlp), s.slot(s_mlp), rows * F, backend::GeluForm::erf);
        b.matmul(lw[down_w].operand(), s.slot(s_mlp), s.slot(s_h), F, D, rows, all, s.dtype);
        b.add_bias(s.slot(s_h), lw[down_b].slice(), rows, D, lw[ls2].slice());
        b.add(s.slot(s_x), s.slot(s_h), rows * D);
    }

private:
    uint16_t block_base(size_t l) const { return uint16_t(kBlockBase + l * kPerBlock); }
    uint16_t merger_base() const { return block_base(c_.blocks); }
    // The longest grid side, in patches, kMaxTokens allows: a strip one token high.
    size_t max_side() const { return size_t(kMaxTokens) * c_.merge; }

    Config c_;
};

inline std::shared_ptr<const VisionArchitecture> open(const gguf::GGUFModel& m) { return std::make_shared<Module>(read_config(m)); }

}  // namespace infer::inkling2_vision
