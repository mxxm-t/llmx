#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

// The vision tower of DeepSeek-V4.1, read from an mmproj GGUF whose projector is deepseek41v (docs/VISION.md, section 3).

namespace infer::deepseek41_vision {

struct Config {
    size_t width = 0, mlp = 0, blocks = 0, heads = 0, patch = 0, merge = 0, out = 0;
    uint64_t min_pixels = 0;
    float eps = 1e-6f;
    float mean[3] = {0, 0, 0}, std[3] = {0, 0, 0};
};

// The processor's default bound on an image block's rows (vision_max_n_token), which the GGUF does not carry.
inline constexpr uint64_t kDefaultRows = 1024;
// The most rows --image-max-tokens may ask of a block, which sizes the rope tables for the longest grid side such a block allows.
inline constexpr uint64_t kMaxRows = 4096;
// The vision rope's base, which the official code fixes and the GGUF does not carry.
inline constexpr double kRopeTheta = 10000.0;
// The gray an image is padded with, as the processor's ImageOps.pad call gives it.
inline constexpr uint8_t kFill = 127;

inline Config read_config(const gguf::GGUFModel& m) {
    using namespace metadata;
    Config c;
    c.width = size_t(integer(m, "clip.vision.embedding_length"));
    c.mlp = size_t(integer(m, "clip.vision.feed_forward_length"));
    c.blocks = size_t(integer(m, "clip.vision.block_count"));
    c.heads = size_t(integer(m, "clip.vision.attention.head_count"));
    c.patch = size_t(integer(m, "clip.vision.patch_size"));
    c.merge = size_t(integer(m, "clip.vision.projector.scale_factor"));
    c.out = size_t(integer(m, "clip.vision.projection_dim"));
    c.min_pixels = uint64_t(integer(m, "clip.vision.image_min_pixels"));
    c.eps = float(real(m, "clip.vision.attention.layer_norm_epsilon", 1e-6f));
    const auto three = [&](const char* key, float* dst) {
        const auto* v = m.find(key);
        if (!v || v->vtype != gguf::V_ARRAY || v->u != gguf::V_FLOAT32 || v->arr.size() != 3) throw std::runtime_error(std::string("inference: invalid metadata ") + key);
        for (size_t i = 0; i < 3; ++i) std::memcpy(&dst[i], &v->arr[i].fb, sizeof(float));
    };
    three("clip.vision.image_mean", c.mean);
    three("clip.vision.image_std", c.std);
    // The fused normalization equals the processor's two steps once rounded to BF16 only for a mean and std of one half.
    for (int i = 0; i < 3; ++i)
        if (c.mean[i] != 0.5f || c.std[i] != 0.5f) throw std::runtime_error("inference: unsupported metadata clip.vision.image_mean or image_std (0.5 is required)");
    if (!boolean(m, "clip.use_silu", false)) throw std::runtime_error("inference: unsupported metadata clip.use_silu (the SwiGLU feed-forward is required)");
    if (!c.width || !c.mlp || !c.blocks || !c.heads || !c.patch || !c.out || !c.min_pixels) throw std::runtime_error("inference: invalid deepseek41v metadata");
    if (c.merge != 3) throw std::runtime_error("inference: unsupported metadata clip.vision.projector.scale_factor (only 3 is supported)");
    if (c.width % c.heads || (c.width / c.heads) % 4) throw std::runtime_error("inference: unsupported vision head width (a multiple of 4 is required)");
    if ((c.width / c.heads) / 2 > 128) throw std::runtime_error("inference: unsupported vision head width (at most 256)");
    return c;
}

// The roles, by id: the patch embedding, each block's nine, the final norm, the aligner's four and the block's three marker rows.
enum : uint16_t { patch_w, patch_b, kBlockBase };
inline constexpr uint16_t kPerBlock = 9;
enum : uint16_t { ln1_w, qkv_w, qkv_b, out_w, out_b, ln2_w, gate_w, up_w, down_w };
enum : uint16_t { post_w, mm1_w, mm1_b, mm2_w, mm2_b, img_start, img_end, img_newline, kTail };
// The multiply-adds of one piece's attention at most, as the Qwen module bounds them (docs/VISION.md, section 10).
inline constexpr size_t kPieceProducts = size_t(1) << 37;

// The arena's slots, the floats one patch takes in each; the aligner reuses s_mlp and s_qkv once the blocks are done, and a grid asks for the patch rows its aligner needs there (EncodeInputs::arena_patches).
enum : size_t { s_pixels, s_x, s_h, s_qkv, s_attn, s_mlp, s_out, kSlots };
// The tables: the rope's cos and sin, then the zeros the unfold's pad rows are copied from, a token's 9 rows.
enum : size_t { t_cos, t_sin, t_zeros, kTables };

class Module final : public VisionArchitecture {
public:
    explicit Module(Config c) : c_(c) {}

    // N tokens bound the block's rows, padded_size solving an image down to the largest that fits.
    image::Params preprocess(std::optional<uint64_t> max_tokens) const override {
        image::Params p;
        p.patch = uint32_t(c_.patch);
        p.merge = uint32_t(c_.merge);
        for (int i = 0; i < 3; ++i) {
            p.mean[i] = c_.mean[i];
            p.std[i] = c_.std[i];
        }
        p.fit = image::Fit::padded;
        p.min_pixels = c_.min_pixels;
        p.max_rows = kDefaultRows;
        p.fill = kFill;
        if (max_tokens) {
            if (*max_tokens < kSmallestBlock)
                throw std::runtime_error("vision: an image takes at least " + std::to_string(kSmallestBlock) + " tokens, not " + std::to_string(*max_tokens));
            if (*max_tokens > kMaxRows) throw std::runtime_error("vision: an image takes at most " + std::to_string(kMaxRows) + " tokens, not " + std::to_string(*max_tokens));
            p.max_rows = *max_tokens;
        }
        return p;
    }
    uint64_t max_tokens(const image::Params& p) const override { return p.max_rows; }
    // A block of one token's row, the smallest any grid gives.
    uint64_t min_tokens(const image::Params&) const override { return kSmallestBlock; }
    size_t tokens(uint32_t grid_h, uint32_t grid_w) const override { return size_t(image::padded_rows(uint64_t(grid_h) * c_.patch, uint64_t(grid_w) * c_.patch, uint32_t(c_.patch), uint32_t(c_.merge))); }
    // A block of R rows holds an h by w grid of tokens with h * (w + 1) + 2 = R at most, so at most R - 3 tokens of merge^2 patches; each term of inputs' arena grows with them.
    EncodeBound bound(const image::Params& p) const override {
        const size_t R = size_t(p.max_rows), T = R - 3, K = c_.merge * c_.merge, O = c_.out;
        const auto rows = [](size_t floats, size_t per) { return (floats + per - 1) / per; };
        EncodeBound b;
        b.patches = K * T;
        b.arena_patches = std::max({b.patches, rows(K * T * c_.width, 2 * c_.mlp), rows((T + 3) * O, 2 * c_.mlp), rows(T * O, 3 * c_.width), rows(R * O, (O + 1) / 2)});
        const size_t crossing = size_t(std::sqrt(double(kPieceProducts) / double(2 * c_.width)));
        b.attention_rows = std::max(attention_rows(std::min(b.patches, crossing)), attention_rows(std::min(b.patches, crossing + 1)));
        return b;
    }
    VisionPositions positions() const override { return VisionPositions::text; }
    ImageTokenTexts token_texts() const override { return {"", "<\xef\xbd\x9c" "deepseek_image\xef\xbd\x9c>", ""}; }

    EncoderPlan plan(const TensorIndex& tensors) const override {
        const size_t D = c_.width, F = c_.mlp, in = c_.patch * c_.patch * 3, K = c_.merge * c_.merge;
        const TensorView& pe = tensors.view(tensors.at("v.patch_embd.weight"));
        if (pe.shape.size() != 4 || pe.shape[0] != c_.patch || pe.shape[1] != c_.patch || pe.shape[2] != 3 || pe.shape[3] != D)
            throw std::runtime_error("inference: incompatible tensor layout v.patch_embd.weight");
        EncoderPlan p;
        const auto role = [&](uint16_t id, const std::string& name, RoleKind kind, uint64_t rin, uint64_t rout) {
            p.roles.push_back(Role{id, Part::embed, kind, name, "", rin, rout});
        };
        role(patch_w, "v.patch_embd.weight", RoleKind::matrix, in, D);
        role(patch_b, "v.patch_embd.bias", RoleKind::norm, D, 1);
        for (size_t l = 0; l < c_.blocks; ++l) {
            const uint16_t b = uint16_t(kBlockBase + l * kPerBlock);
            const std::string n = "v.blk." + std::to_string(l) + ".";
            role(b + ln1_w, n + "ln1.weight", RoleKind::norm, D, 1);
            role(b + qkv_w, n + "attn_qkv.weight", RoleKind::matrix, D, 3 * D);
            role(b + qkv_b, n + "attn_qkv.bias", RoleKind::norm, 3 * D, 1);
            role(b + out_w, n + "attn_out.weight", RoleKind::matrix, D, D);
            role(b + out_b, n + "attn_out.bias", RoleKind::norm, D, 1);
            role(b + ln2_w, n + "ln2.weight", RoleKind::norm, D, 1);
            role(b + gate_w, n + "ffn_gate.weight", RoleKind::matrix, D, F);
            role(b + up_w, n + "ffn_up.weight", RoleKind::matrix, D, F);
            role(b + down_w, n + "ffn_down.weight", RoleKind::matrix, F, D);
        }
        const uint16_t t = tail();
        role(t + post_w, "v.post_ln.weight", RoleKind::norm, D, 1);
        role(t + mm1_w, "mm.1.weight", RoleKind::matrix, K * D, c_.out);
        role(t + mm1_b, "mm.1.bias", RoleKind::norm, c_.out, 1);
        role(t + mm2_w, "mm.2.weight", RoleKind::matrix, c_.out, c_.out);
        role(t + mm2_b, "mm.2.bias", RoleKind::norm, c_.out, 1);
        role(t + img_start, "v.token_embd.img_start", RoleKind::norm, c_.out, 1);
        role(t + img_end, "v.token_embd.img_end", RoleKind::norm, c_.out, 1);
        role(t + img_newline, "v.image_newline", RoleKind::norm, c_.out, 1);
        // torch's unfold lays a token's inputs out channel by channel, d * 9 + k; regrouped, they are the 9 patch rows one after another.
        p.regrouped = {Regroup{uint16_t(t + mm1_w), K}};
        p.role_ids = size_t(t + kTail);
        p.ops = {backend::Op::gelu, backend::Op::add_bias, backend::Op::rope_streams, backend::Op::attention_full};
        p.slots.assign(kSlots, 0);
        p.slots[s_pixels] = in;
        p.slots[s_x] = p.slots[s_h] = p.slots[s_attn] = D;
        p.slots[s_qkv] = 3 * D;
        p.slots[s_mlp] = 2 * F;
        p.slots[s_out] = (c_.out + 1) / 2;
        p.input_slot = s_pixels;
        p.output_slot = s_out;
        p.heads = c_.heads;
        p.head_dim = D / c_.heads;
        p.out_width = c_.out;
        p.tables.assign(kTables, 0);
        p.tables[t_cos] = p.tables[t_sin] = max_side() * (p.head_dim / 2);
        p.tables[t_zeros] = K * D;
        return p;
    }

    // The rope's cos and sin as the Qwen module lays them out: column j of row p turns by p times frequency j mod head_dim / 4, the first quarter by the patch's row and the rest by its column.
    void fill_tables(std::vector<std::vector<float>>& t) const override {
        const size_t hd = c_.width / c_.heads, half = hd / 2, quarter = hd / 4;
        std::vector<float> inv(quarter);
        for (size_t k = 0; k < quarter; ++k) inv[k] = float(1.0 / std::pow(kRopeTheta, double(2 * k) / double(half)));
        const size_t rows = t[t_cos].size() / half;
        for (size_t p = 0; p < rows; ++p)
            for (size_t j = 0; j < half; ++j) {
                const float a = float(p) * inv[j % quarter];
                t[t_cos][p * half + j] = std::cos(a);
                t[t_sin][p * half + j] = std::sin(a);
            }
    }

    // Each patch's row and column in row order; the unfold's patch rows, token by token in rows of the zero-padded grid, its pad runs, the block's rows of the aligner's output and its three markers, and the arena the aligner needs.
    // A grid thinner than 3 patches is padded to whole tokens as the official aligner pads it.
    EncodeInputs inputs(uint32_t gh, uint32_t gw) const override {
        const size_t M = c_.merge, K = M * M, O = c_.out;
        if (gh > max_side() || gw > max_side()) throw std::runtime_error("inference: an image grid past the rope table");
        const size_t n = size_t(gh) * gw, h = (gh + M - 1) / M, w = (gw + M - 1) / M, T = h * w;
        EncodeInputs in;
        // The unfold and mm.2's rows with the markers in s_mlp, mm.1's in s_qkv and the block in s_out, each in patch rows of its slot.
        const auto rows = [](size_t floats, size_t per) { return (floats + per - 1) / per; };
        in.arena_patches = std::max({n, rows(K * T * c_.width, 2 * c_.mlp), rows((T + 3) * O, 2 * c_.mlp), rows(T * O, 3 * c_.width), rows(tokens(gh, gw) * O, (O + 1) / 2)});
        in.positions.resize(2 * n);
        for (size_t r = 0, k = 0; r < gh; ++r)
            for (size_t c = 0; c < gw; ++c, ++k) {
                in.positions[2 * k] = uint32_t(r);
                in.positions[2 * k + 1] = uint32_t(c);
            }
        in.unfold.resize(K * T);
        for (size_t ty = 0; ty < h; ++ty)
            for (size_t tx = 0; tx < w; ++tx)
                for (size_t k = 0; k < K; ++k) {
                    const size_t i = (ty * w + tx) * K + k, r = ty * M + k / M, c = tx * M + k % M;
                    const bool pad = r >= gh || c >= gw;
                    in.unfold[i] = pad ? 0 : uint32_t(r * gw + c);
                    if (!pad) continue;
                    if (!in.zero_runs.empty() && in.zero_runs[in.zero_runs.size() - 2] + in.zero_runs.back() == i) ++in.zero_runs.back();
                    else in.zero_runs.insert(in.zero_runs.end(), {uint32_t(i), 1});
                }
        // The aligner's T rows are followed by the start, end and newline rows.
        in.layout.push_back(uint32_t(T));
        for (size_t ty = 0; ty < h; ++ty) {
            for (size_t tx = 0; tx < w; ++tx) in.layout.push_back(uint32_t(ty * w + tx));
            in.layout.push_back(uint32_t(T + 2));
        }
        in.layout.push_back(uint32_t(T + 1));
        return in;
    }

    // The patch embedding, then each block as its projections, its attention in runs of query rows, and the rest, then the final norm and the aligner.
    size_t pieces(size_t patches) const override { return 2 + c_.blocks * (2 + attention_runs(patches)); }

    void encode(const EncodeStep& s, size_t piece) const override {
        backend::Backend& b = s.b;
        const size_t D = c_.width, F = c_.mlp, n = s.patches, hd = D / c_.heads;
        const backend::RowRun run{n, n};
        const backend::RowRuns runs{&run, 1};
        const Weight* w = s.w;
        const size_t per_block = 2 + attention_runs(n);
        if (piece == 0) {
            b.matmul(w[patch_w].operand(), s.slot(s_pixels), s.slot(s_x), c_.patch * c_.patch * 3, D, n, runs, s.dtype);
            b.add_bias(s.slot(s_x), w[patch_b].slice(), n, D);
            return;
        }
        if (piece == pieces(n) - 1) {
            aligner(s);
            return;
        }
        const size_t l = (piece - 1) / per_block, part = (piece - 1) % per_block;
        const Weight* lw = w + kBlockBase + l * kPerBlock;
        const backend::Slice qkv = s.slot(s_qkv), q = qkv, k = at(qkv, n * D), v = at(qkv, 2 * n * D);
        if (part == 0) {
            uint8_t of[128];
            for (size_t j = 0; j < hd / 2; ++j) of[j] = j < hd / 4 ? 0 : 1;
            const backend::CSlice cos{s.tables[t_cos].get(), 0}, sin{s.tables[t_sin].get(), 0};
            b.rms_norm_rows(s.slot(s_h), s.slot(s_x), lw[ln1_w].slice(), n, D, D, c_.eps);
            b.matmul_group({{{lw[qkv_w].type, row_block(lw[qkv_w], 0)}, q, D}, {{lw[qkv_w].type, row_block(lw[qkv_w], D)}, k, D},
                            {{lw[qkv_w].type, row_block(lw[qkv_w], 2 * D)}, v, D}},
                           s.slot(s_h), D, n, runs, s.dtype);
            const backend::CSlice bias = lw[qkv_b].slice();
            b.add_bias(q, bias, n, D);
            b.add_bias(k, {bias.buffer, bias.offset + D}, n, D);
            b.add_bias(v, {bias.buffer, bias.offset + 2 * D}, n, D);
            b.norm_rope_partial(q, q, n, D, hd, c_.heads, hd, hd, {}, 0.0f, cos, sin, s.inputs->positions.data(), backend::RopeForm{}, backend::RopeStreams{2, of});
            b.norm_rope_partial(k, k, n, D, hd, c_.heads, hd, hd, {}, 0.0f, cos, sin, s.inputs->positions.data(), backend::RopeForm{}, backend::RopeStreams{2, of});
            b.kv_write(0, s.view, 1, k, v);
            return;
        }
        if (part + 1 < per_block) {
            const size_t rows = attention_rows(n), first = (part - 1) * rows, m = std::min(rows, n - first);
            const backend::KVView view{s.view->storage, s.view->blocks, s.view->n_blocks, n - m, m, m, false};
            b.attention(at(q, first * D), 0, &view, 1, at(s.slot(s_attn), first * D), int(c_.heads), int(c_.heads), int(hd));
            return;
        }
        b.matmul(lw[out_w].operand(), s.slot(s_attn), s.slot(s_h), D, D, n, runs, s.dtype);
        b.add_bias(s.slot(s_h), lw[out_b].slice(), n, D);
        b.add(s.slot(s_x), s.slot(s_h), n * D);
        b.rms_norm_rows(s.slot(s_h), s.slot(s_x), lw[ln2_w].slice(), n, D, D, c_.eps);
        const backend::Slice gate = s.slot(s_mlp), up = at(gate, n * F);
        b.matmul_group({{lw[gate_w].operand(), gate, F}, {lw[up_w].operand(), up, F}}, s.slot(s_h), D, n, runs, s.dtype);
        b.silu_mul(gate, gate, up, n * F, 0.0f, runs);
        b.matmul_add(lw[down_w].operand(), gate, s.slot(s_x), F, D, n, runs, s.dtype);
    }

private:
    // A block always holds a start, one token, its newline and an end.
    static constexpr uint64_t kSmallestBlock = 4;

    // The final norm, the 3 by 3 unfold of the zero-padded grid as 9 patch rows a token, the aligner's two layers with the erf GELU between, then the block laid out with its markers.
    void aligner(const EncodeStep& s) const {
        backend::Backend& b = s.b;
        const size_t D = c_.width, K = c_.merge * c_.merge, O = c_.out, n = s.patches;
        const EncodeInputs& in = *s.inputs;
        const size_t T = in.unfold.size() / K, rows = in.layout.size();
        const backend::RowRun run{T, T};
        const backend::RowRuns runs{&run, 1};
        const Weight* t = s.w + tail();
        const backend::Slice unf = s.slot(s_mlp), m1 = s.slot(s_qkv), m2 = s.slot(s_mlp);
        b.rms_norm_rows(s.slot(s_h), s.slot(s_x), t[post_w].slice(), n, D, D, c_.eps);
        b.gather_rows(unf, s.slot(s_h), D, in.unfold.data(), in.unfold.size());
        for (size_t i = 0; i < in.zero_runs.size(); i += 2)
            b.copy(*unf.buffer, (unf.offset + size_t(in.zero_runs[i]) * D) * sizeof(float), *s.tables[t_zeros], 0, size_t(in.zero_runs[i + 1]) * D * sizeof(float));
        b.matmul(t[mm1_w].operand(), unf, m1, K * D, O, T, runs, s.dtype);
        b.add_bias(m1, t[mm1_b].slice(), T, O);
        b.gelu(m1, m1, T * O, backend::GeluForm::erf);
        b.matmul(t[mm2_w].operand(), m1, m2, O, O, T, runs, s.dtype);
        b.add_bias(m2, t[mm2_b].slice(), T, O);
        for (size_t i = 0; i < 3; ++i)
            b.copy(*m2.buffer, (m2.offset + (T + i) * O) * sizeof(float), *t[img_start + i].data, 0, O * sizeof(float));
        b.gather_rows(s.slot(s_out), m2, O, in.layout.data(), rows);
    }

    uint16_t tail() const { return uint16_t(kBlockBase + c_.blocks * kPerBlock); }
    size_t attention_rows(size_t n) const { return std::max<size_t>(1, std::min(n, kPieceProducts / std::max<size_t>(1, 2 * n * c_.width))); }
    size_t attention_runs(size_t n) const { return (n + attention_rows(n) - 1) / attention_rows(n); }
    // The longest grid side a block of kMaxRows rows allows: one row of kMaxRows - 3 tokens.
    size_t max_side() const { return size_t(kMaxRows - 3) * c_.merge; }
    static backend::Slice at(backend::Slice s, size_t floats) { return {s.buffer, s.offset + floats}; }
    static backend::CSlice row_block(const Weight& w, size_t first) {
        const size_t bytes = quant::row_bytes(w.type, w.nin, first);
        if (bytes % sizeof(float)) throw std::runtime_error("inference: a vision projection's rows do not start on a float boundary");
        return {w.data.get(), bytes / sizeof(float)};
    }

    Config c_;
};

inline std::shared_ptr<const VisionArchitecture> open(const gguf::GGUFModel& m) { return std::make_shared<Module>(read_config(m)); }

}  // namespace infer::deepseek41_vision
