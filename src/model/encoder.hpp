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
#include "image/image.hpp"
#include "model/architecture.hpp"
#include "model/layer_split.hpp"
#include "model/weights.hpp"

// A vision encoder: the contract a module of a vision model family implements, and the runner that opens its weights on a backend and encodes one image a call (docs/VISION.md, sections 3 and 9).

namespace infer {

// How an encoder's rows are placed among a prompt's positions: Qwen's three components, the patch's row and column beside the image's start, or one text position a row.
enum class VisionPositions : uint8_t { grid, text };

// The texts of the tokens a chat template marks an image with, read from the text model's vocabulary.
struct ImageTokenTexts {
    std::string start, pad, end;
};

// A matrix role whose input axis the runner reorders once on a host copy: input d * groups + k of each row becomes k * (in / groups) + d, so a row's inputs run group by group.
struct Regroup {
    uint16_t role = 0;
    size_t groups = 0;
};

// What an encoder declares of its weights: its roles, those regrouped, the ops some backends lack, the floats a patch takes in each arena slot and which hold its input and output rows.
// It also gives its attention's heads, the width of an image token's row, the floats of each table it computes and the rows it attends over beyond the patches, such as a class row after them.
struct EncoderPlan {
    size_t role_ids = 0;
    std::vector<Role> roles;
    std::vector<Regroup> regrouped;
    std::vector<backend::Op> ops;
    std::vector<size_t> slots;
    size_t input_slot = 0, output_slot = 0;
    size_t heads = 0, head_dim = 0;
    size_t out_width = 0;
    std::vector<size_t> tables;
    size_t extra_rows = 0;
};

// What an image gives its encoder besides its patches, computed on the host from its grid: each patch's position streams and its learned position taps and weights.
// A module that regroups patches gives the rows it gathers, the runs of those it zeroes as (first, count) pairs, the rows its output is laid out from, and the patch rows its arena must hold, if more than the patches.
struct EncodeInputs {
    std::vector<uint32_t> positions;
    std::vector<uint32_t> taps;
    std::vector<float> weights;
    std::vector<uint32_t> unfold, zero_runs, layout;
    size_t arena_patches = 0;
};

// One image's call of the module's math: the backend and its arena, the weights by role id, the image's patches and grid, its inputs, its non-causal attention view and the activation dtype.
struct EncodeStep {
    backend::Backend& b;
    backend::Buffer* arena;
    const size_t* offsets;   // bytes, per slot
    const Weight* w;
    size_t patches;
    uint32_t grid_h, grid_w;
    const EncodeInputs* inputs;
    const backend::KVView* view;
    const backend::BufferPtr* tables;
    backend::Dtype dtype;
    backend::Slice slot(size_t i) const { return {arena, offsets[i] / sizeof(float)}; }
};

// The most any image under one preprocessing asks of the runner: its patches, the patch rows its arena holds (EncodeInputs::arena_patches) and the query rows of one attention piece.
struct EncodeBound {
    size_t patches = 0, arena_patches = 0, attention_rows = 0;
};

// A vision model family's module: its preprocessing, the tokens and positions an image takes, its template's token texts, the plan of its file's tensors, its tables and its math as backend ops.
class VisionArchitecture {
public:
    virtual ~VisionArchitecture() = default;
    // Its preprocessing; with `max_tokens`, its resize bound the one that keeps an image within that many tokens, refused where its images always take more.
    virtual image::Params preprocess(std::optional<uint64_t> max_tokens) const = 0;
    // The most and the fewest tokens an image resized under `p` takes.
    virtual uint64_t max_tokens(const image::Params& p) const = 0;
    virtual uint64_t min_tokens(const image::Params& p) const = 0;
    virtual size_t tokens(uint32_t grid_h, uint32_t grid_w) const = 0;
    // What the largest image under `p` asks of the runner, which the fit charges (VisionEncoder::memory).
    virtual EncodeBound bound(const image::Params& p) const = 0;
    virtual VisionPositions positions() const = 0;
    virtual ImageTokenTexts token_texts() const = 0;
    virtual EncoderPlan plan(const TensorIndex& tensors) const = 0;
    // The tables come sized as the plan says.
    virtual void fill_tables(std::vector<std::vector<float>>& tables) const = 0;
    virtual EncodeInputs inputs(uint32_t grid_h, uint32_t grid_w) const = 0;
    // An image of `patches` patches encodes in pieces, run in order, each of a bounded amount of work, so a server runs other work between them (docs/VISION.md, section 10).
    virtual size_t pieces(size_t patches) const = 0;
    virtual void encode(const EncodeStep& s, size_t piece) const = 0;
};

// The pieces the vision modules share: the bound of one attention piece, a fused projection's row blocks, the two-stream rope over a patch's position and the attention runs.

// The multiply-adds of one piece's attention at most, about 60 ms on an MI50, so a server's replies wait at most that long for an image's encode (docs/VISION.md, section 10).
inline constexpr size_t kPieceProducts = size_t(1) << 37;

// A piece's attention rows over n rows `width` wide: as many as keep its dots and weighted sums over all n keys within kPieceProducts, at least one, at most n.
inline size_t attention_rows(size_t n, size_t width) { return std::max<size_t>(1, std::min(n, kPieceProducts / std::max<size_t>(1, 2 * n * width))); }
inline size_t attention_runs(size_t n, size_t width) { return (n + attention_rows(n, width) - 1) / attention_rows(n, width); }

// A matrix weight from row `first` on, which starts on a float boundary in every type the encoders read.
inline backend::CSlice row_block(const Weight& w, size_t first) {
    const size_t bytes = quant::row_bytes(w.type, w.nin, first);
    if (bytes % sizeof(float)) throw std::runtime_error("inference: a vision projection's rows do not start on a float boundary");
    return {w.data.get(), bytes / sizeof(float)};
}

// The rope's cos and sin of a head `head_dim` wide: column j of row p turns by p times frequency j mod head_dim / 4, as two streams of head_dim / 4 frequencies each lay them out.
inline void fill_stream_rope(std::vector<float>& cos, std::vector<float>& sin, size_t head_dim, double theta) {
    const size_t half = head_dim / 2, quarter = head_dim / 4;
    std::vector<float> inv(quarter);
    for (size_t k = 0; k < quarter; ++k) inv[k] = float(1.0 / std::pow(theta, double(2 * k) / double(half)));
    const size_t rows = cos.size() / half;
    for (size_t p = 0; p < rows; ++p)
        for (size_t j = 0; j < half; ++j) {
            const float a = float(p) * inv[j % quarter];
            cos[p * half + j] = std::cos(a);
            sin[p * half + j] = std::sin(a);
        }
}

// One axis's two bilinear taps into a learned grid `side` positions long for position `index` of `size`, clamped at its border, in F32 as HF's vision utilities compute them.
// With aligned corners the axis's ends meet the grid's; otherwise its positions sit at half-cell centres, as F.interpolate's align_corners=False places them.
inline void bilinear_taps(size_t index, size_t size, size_t side, bool align_corners, uint32_t taps[2], float w[2]) {
    const float src = align_corners ? float(index) * float(side - 1) / float(size > 1 ? size - 1 : 1) : (float(index) + 0.5f) * float(side) / float(size) - 0.5f;
    const float fl = std::floor(src);
    for (int o = 0; o < 2; ++o) {
        const int64_t raw = int64_t(fl) + o;
        taps[o] = uint32_t(std::min<int64_t>(std::max<int64_t>(raw, 0), int64_t(side - 1)));
        w[o] = std::max(1.0f - std::fabs(src - fl - float(o)), 0.0f);
    }
}

// The biases of q, k and v added over n rows `width` wide, q and k turned by the rope of fill_stream_rope, the first head_dim / 4 frequencies by a row's first position and the rest by its second, and k and v written to the view.
inline void bias_rope_kv(const EncodeStep& s, backend::Slice q, backend::Slice k, backend::Slice v, backend::CSlice qb, backend::CSlice kb, backend::CSlice vb, size_t n, size_t width,
                         size_t heads, backend::CSlice cos, backend::CSlice sin, backend::RopeForm form) {
    const size_t hd = width / heads;
    if (hd / 2 > 128) throw std::logic_error("inference: a vision head wider than 256");
    uint8_t of[128];
    for (size_t j = 0; j < hd / 2; ++j) of[j] = j < hd / 4 ? 0 : 1;
    s.b.add_bias(q, qb, n, width);
    s.b.add_bias(k, kb, n, width);
    s.b.add_bias(v, vb, n, width);
    s.b.norm_rope_partial(q, q, n, width, hd, heads, hd, hd, {}, 0.0f, cos, sin, s.inputs->positions.data(), form, backend::RopeStreams{2, of});
    s.b.norm_rope_partial(k, k, n, width, hd, heads, hd, hd, {}, 0.0f, cos, sin, s.inputs->positions.data(), form, backend::RopeStreams{2, of});
    s.b.kv_write(0, s.view, 1, k, v);
}

// Run `run` of attention_runs(n, width): its query rows over every key, a non-causal view of all n keys whose rows are the run's, each row's output that of the whole call.
inline void attention_run(const EncodeStep& s, backend::Slice q, backend::Slice out, size_t n, size_t run, size_t width, size_t heads) {
    const size_t rows = attention_rows(n, width), first = run * rows, m = std::min(rows, n - first);
    const backend::KVView view{s.view->storage, s.view->blocks, s.view->n_blocks, n - m, m, m, false};
    s.b.attention({q.buffer, q.offset + first * width}, 0, &view, 1, {out.buffer, out.offset + first * width}, int(heads), int(heads), int(width / heads));
}

// The runner: an encoder's weights adopted on one backend, its tables, and the scratch of one encode, its arena and its one layer of F32 K and V, grown to the largest image it has encoded and never shrunk.
class VisionEncoder {
public:
    VisionEncoder(std::shared_ptr<const VisionArchitecture> arch, const std::vector<TensorView>& views, backend::Backend& b, backend::Dtype dtype)
        : arch_(std::move(arch)), b_(b), dtype_(dtype) {
        const TensorIndex index(views);
        plan_ = arch_->plan(index);
        for (backend::Op op : plan_.ops)
            if (!b_.implements(op)) throw std::runtime_error(std::string("inference: the vision encoder needs ") + backend::op_name(op) + ", which its device does not implement");
        w_.resize(plan_.role_ids);
        for (const Role& role : plan_.roles) {
            const TensorView& t = index.view(index.at(role.name));
            // A convolution's kernel stored [kw, kh, c, out] is the matrix [kw * kh * c, out] row for row, so its leading axes fold into one.
            std::vector<uint64_t> shape = t.shape;
            if (role.kind == RoleKind::matrix && shape.size() > 2) {
                uint64_t in = 1;
                for (size_t d = 0; d + 1 < shape.size(); ++d) in = backend::size_mul(in, shape[d]);
                shape = {in, shape.back()};
            }
            if (!role_fits(role, shape, t.type)) throw std::runtime_error("inference: incompatible tensor layout " + t.name);
            if (!t.data) throw std::runtime_error("inference: tensor " + t.name + " is not in memory");
            if (role.kind != RoleKind::norm && role.kind != RoleKind::table && !b_.supports_type(t.type))
                throw std::runtime_error("inference: the vision encoder's device does not read the type of " + t.name);
            const void* data = t.data;
            for (const Regroup& g : plan_.regrouped)
                if (g.role == role.id) data = regroup(t, role, g.groups);
            w_[role.id] = Weight{t.type, b_.adopt(data, t.bytes), size_t(role.in), size_t(role.out)};
        }
        tables_.resize(plan_.tables.size());
        for (size_t i = 0; i < tables_.size(); ++i) tables_[i].assign(plan_.tables[i], 0.0f);
        arch_->fill_tables(tables_);
        for (const auto& t : tables_) table_buffers_.push_back(b_.adopt(t.data(), t.size() * sizeof(float)));
    }

    const VisionArchitecture& architecture() const { return *arch_; }
    const EncoderPlan& plan() const { return plan_; }
    backend::Backend& backend() const { return b_; }

    // What the runner keeps on its device for images under `p`, which the fit charges beside the embedding (EncoderMemory): each weight as adopted, its tables, its arena and its K and V at the bound in whole blocks of `block`.
    // Its passes run their products over every patch and their attention one piece at a time, so it prices those as two widths.
    static EncoderMemory memory(const VisionArchitecture& arch, const std::vector<TensorView>& views, const image::Params& p, size_t block) {
        const TensorIndex index(views);
        const EncoderPlan plan = arch.plan(index);
        const EncodeBound bound = arch.bound(p);
        const size_t rows = backend::size_add(bound.patches, plan.extra_rows), arena = std::max(bound.arena_patches, rows);
        EncoderMemory m;
        backend::PassWidths products, attention;
        for (const Role& role : plan.roles) {
            const TensorView& t = index.view(index.at(role.name));
            const bool product = role.kind == RoleKind::matrix;
            m.weights.push_back(Matrix{t.type, size_t(role.in), size_t(role.out), t.bytes, product});
            if (product) {
                products.product_in = std::max(products.product_in, size_t(role.in));
                products.product_out = std::max(products.product_out, size_t(role.out));
            }
        }
        for (size_t floats : plan.tables) m.buffers.push_back(backend::size_mul(floats, sizeof(float)));
        m.buffers.push_back(arena_bytes(plan, arena));
        const size_t kv = backend::size_mul(backend::size_mul(backend::blocks_for(arena, block), block), backend::size_mul(plan.heads * plan.head_dim, sizeof(float)));
        m.buffers.insert(m.buffers.end(), {kv, kv});
        products.float_out = float_product_out(m.weights);
        products.rows = rows;
        attention.rows = bound.attention_rows;
        attention.heads = plan.heads;
        attention.head_dim = plan.head_dim;
        attention.dense_attention = true;
        attention.context = rows;
        m.widths = {products, attention};
        return m;
    }
    // Makes the arena and the K and V of the largest image under `p` at once, as the fit charged them, so no encode grows them beside the buffers they replace.
    void hold(const image::Params& p) {
        const EncodeBound bound = arch_->bound(p);
        reserve(std::max(bound.arena_patches, backend::size_add(bound.patches, plan_.extra_rows)));
    }

    // One image's rows, tokens(grid) rows of out_width floats: its pieces in order; one image at a time, so its rows are its own whatever else is encoded.
    std::vector<float> encode(const image::Prepared& image) {
        const size_t count = begin(image);
        for (size_t i = 0; i < count; ++i) piece(i);
        return rows();
    }
    // An image's encode begun: its patches written and its inputs computed; it returns the pieces, which piece() runs in order before rows() reads the result.
    // A new begin() abandons an encode not finished, whose scratch it reuses.
    size_t begin(const image::Prepared& image) {
        const size_t n = size_t(image.grid_h) * image.grid_w;
        const size_t patch_floats = plan_.slots.at(plan_.input_slot);
        if (!n || image.patches.size() != backend::size_mul(n, patch_floats)) throw std::runtime_error("inference: an image's patches do not fill its grid");
        pieces_ = next_ = 0;
        inputs_ = arch_->inputs(image.grid_h, image.grid_w);
        const size_t rows = backend::size_add(n, plan_.extra_rows);
        reserve(std::max(rows, inputs_.arena_patches));
        b_.write(*arena_, offsets_[plan_.input_slot], image.patches.data(), image.patches.size() * sizeof(float));
        blocks_.resize(backend::blocks_for(rows, b_.kv_layout().block_tokens));
        for (size_t i = 0; i < blocks_.size(); ++i) blocks_[i] = int32_t(i);
        view_ = backend::KVView{kv_.get(), blocks_.data(), blocks_.size(), 0, rows, rows, false};
        patches_ = n;
        grid_h_ = image.grid_h;
        grid_w_ = image.grid_w;
        pieces_ = arch_->pieces(n);
        return pieces_;
    }
    void piece(size_t i) {
        if (i != next_ || i >= pieces_) throw std::logic_error("inference: an encoder piece out of order");
        arch_->encode(EncodeStep{b_, arena_.get(), offsets_.data(), w_.data(), patches_, grid_h_, grid_w_, &inputs_, &view_, table_buffers_.data(), dtype_}, i);
        ++next_;
    }
    std::vector<float> rows() {
        std::vector<float> out(row_floats());
        b_.read(*arena_, offsets_[plan_.output_slot], out.data(), out.size() * sizeof(float));
        pieces_ = next_ = 0;
        return out;
    }
    // After the last piece, its rows copied on the device into host-visible storage and that work submitted, so no caller waits for the device: copied() reads them once the ticket has retired.
    backend::Ticket copy_rows() {
        const size_t bytes = backend::size_mul(row_floats(), sizeof(float));
        if (!out_ || out_->size() < bytes) out_ = b_.alloc(bytes, backend::Memory::host_visible);
        b_.copy(*out_, 0, *arena_, offsets_[plan_.output_slot], bytes);
        copied_ = bytes / sizeof(float);
        pieces_ = next_ = 0;
        return b_.submit();
    }
    // The rows copy_rows copied, read in place once its ticket has retired and raised; any thread may read them, while no later encode has copied its own.
    std::vector<float> copied() const {
        const float* p = static_cast<const float*>(out_->host_ptr());
        return std::vector<float>(p, p + copied_);
    }

private:
    // A host copy of a 16- or 32-bit matrix with each row's inputs regrouped, kept for as long as the backend may read it in place.
    const void* regroup(const TensorView& t, const Role& role, size_t groups) {
        size_t unit = 0;
        if (t.type == quant::GGML_TYPE_F32) unit = 4;
        else if (t.type == quant::GGML_TYPE_F16 || t.type == quant::GGML_TYPE_BF16) unit = 2;
        if (role.kind != RoleKind::matrix || !unit) throw std::runtime_error("inference: the vision encoder regroups only F32, F16 or BF16 matrices, not " + t.name);
        const size_t in = size_t(role.in), out = size_t(role.out), per = groups ? in / groups : 0;
        if (!groups || in % groups || t.bytes != backend::size_mul(backend::size_mul(in, out), unit)) throw std::runtime_error("inference: incompatible tensor layout " + t.name);
        std::vector<uint8_t>& copy = regrouped_.emplace_back(t.bytes);
        const uint8_t* src = static_cast<const uint8_t*>(t.data);
        for (size_t o = 0; o < out; ++o)
            for (size_t d = 0; d < per; ++d)
                for (size_t k = 0; k < groups; ++k) std::memcpy(&copy[(o * in + k * per + d) * unit], &src[(o * in + d * groups + k) * unit], unit);
        return copy.data();
    }
    size_t row_floats() const {
        if (!pieces_ || next_ != pieces_) throw std::logic_error("inference: an encoder's rows read before its last piece");
        return backend::size_mul(arch_->tokens(grid_h_, grid_w_), plan_.out_width);
    }
    // An arena of `patches` rows: its slots one after another.
    static size_t arena_bytes(const EncoderPlan& plan, size_t patches) {
        size_t bytes = 0;
        for (size_t floats : plan.slots) bytes = backend::size_add(bytes, backend::size_mul(backend::size_mul(floats, patches), sizeof(float)));
        return bytes;
    }
    // Grows the arena and the K and V to `patches` rows; the encoder's K and V are scratch of one encode, F32 always (docs/VISION.md, section 3).
    void reserve(size_t patches) {
        if (patches <= capacity_) return;
        offsets_.assign(plan_.slots.size(), 0);
        for (size_t i = 1; i < plan_.slots.size(); ++i)
            offsets_[i] = backend::size_add(offsets_[i - 1], backend::size_mul(backend::size_mul(plan_.slots[i - 1], patches), sizeof(float)));
        arena_ = b_.alloc(arena_bytes(plan_, patches));
        kv_ = b_.kv_alloc(1, plan_.heads, plan_.head_dim, patches, backend::KVType::f32, backend::KVType::f32);
        capacity_ = patches;
    }

    std::shared_ptr<const VisionArchitecture> arch_;
    backend::Backend& b_;
    backend::Dtype dtype_;
    EncoderPlan plan_;
    std::vector<Weight> w_;
    std::vector<std::vector<uint8_t>> regrouped_;
    std::vector<std::vector<float>> tables_;
    std::vector<backend::BufferPtr> table_buffers_;
    backend::BufferPtr arena_;
    std::unique_ptr<backend::KVStorage> kv_;
    std::vector<size_t> offsets_;
    size_t capacity_ = 0;
    // The encode begun: its image's size, inputs and view, its pieces and the next one to run.
    EncodeInputs inputs_;
    std::vector<int32_t> blocks_;
    backend::KVView view_{};
    size_t patches_ = 0, pieces_ = 0, next_ = 0;
    uint32_t grid_h_ = 0, grid_w_ = 0;
    // The host-visible storage copy_rows fills, grown to the largest image's rows, and the floats it last copied.
    backend::BufferPtr out_;
    size_t copied_ = 0;
};

}  // namespace infer
