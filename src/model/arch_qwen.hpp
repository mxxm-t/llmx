#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <stdexcept>
#include <limits>

#include "format/gguf.hpp"
#include "core/fp16.hpp"
#include "quant/quant.hpp"
#include "backends/backend.hpp"
#include "model/kv_cache.hpp"
#include "backends/cpu/cpu_backend.hpp"

// Qwen3-style transformer forward pass, from scratch. The compute primitives
// (matmul, attention, RMSNorm, RoPE) are delegated to a backend::Backend, so the
// same model code runs on CPU now and other backends later.
// Dense matrices use supported block quants or F32; normalization weights are F32.
// Tensor ne[0] is the input dimension, with each output row contiguous.
// Attention projection width is n_head*head_dim and need not equal n_embd.
// An absent output.weight ties the output projection to token_embd.weight.

namespace infer {

struct QwenConfig {
    int n_layer = 0;
    int n_embd = 0;
    int n_ff = 0;
    int n_head = 0;
    int n_head_kv = 0;
    int head_dim = 0;
    int context_length = 4096;
    float rope_theta = 10000.0f;
    float rms_eps = 1e-6f;
};

inline QwenConfig load_config(const gguf::GGUFModel& m) {
    QwenConfig c;
    auto find = [&](const std::string& k) -> const gguf::MetaValue* {
        const gguf::MetaValue* found = nullptr;
        for (const auto& kv : m.kv) {
            if (kv.first == k) {
                if (found) throw std::runtime_error("inference: duplicate metadata " + k);
                found = &kv.second;
            }
        }
        return found;
    };
    auto integer = [&](const std::string& k, int fallback = 0) -> int {
        const auto* v = find(k);
        if (!v) {
            if (fallback) return fallback;
            throw std::runtime_error("inference: missing metadata " + k);
        }
        uint64_t n;
        if (v->vtype == gguf::V_UINT32 || v->vtype == gguf::V_UINT64) {
            n = v->u;
        } else if (v->vtype == gguf::V_INT32 || v->vtype == gguf::V_INT64) {
            if (v->i <= 0) throw std::runtime_error("inference: invalid positive integer " + k);
            n = uint64_t(v->i);
        } else {
            throw std::runtime_error("inference: invalid integer type " + k);
        }
        if (!n || n > uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("inference: integer outside supported range " + k);
        return int(n);
    };
    auto real = [&](const std::string& k, float fallback) -> double {
        const auto* v = find(k);
        if (!v) return fallback;
        double n;
        if (v->vtype == gguf::V_FLOAT32) {
            float value;
            std::memcpy(&value, &v->fb, sizeof(value));
            n = value;
        } else if (v->vtype == gguf::V_FLOAT64) {
            n = v->f64;
        } else {
            throw std::runtime_error("inference: invalid floating-point type " + k);
        }
        if (!std::isfinite(n) || n <= 0 || n > std::numeric_limits<float>::max())
            throw std::runtime_error("inference: invalid positive float " + k);
        const float value = float(n);
        if (value == 0) throw std::runtime_error("inference: float underflow " + k);
        return n;
    };
    auto option = [&](const std::string& k, const std::string& supported) {
        const auto* v = find(k);
        if (v && (v->vtype != gguf::V_STRING || v->s != supported))
            throw std::runtime_error("inference: unsupported metadata " + k);
    };
    option("general.architecture", "qwen3");
    option("qwen3.tensor_data_layout", "reference");
    option("qwen3.rope.scaling.type", "none");
    if (real("qwen3.rope.scaling.factor", 1) != 1 ||
        real("qwen3.rope.scale_linear", 1) != 1)
        throw std::runtime_error("inference: scaled RoPE is unsupported");

    c.n_layer = integer("qwen3.block_count");
    c.n_embd = integer("qwen3.embedding_length");
    c.n_ff = integer("qwen3.feed_forward_length");
    c.n_head = integer("qwen3.attention.head_count");
    c.n_head_kv = integer("qwen3.attention.head_count_kv", c.n_head);
    if (c.n_head % c.n_head_kv)
        throw std::runtime_error("inference: head count must be divisible by KV head count");
    if (find("qwen3.attention.key_length")) {
        c.head_dim = integer("qwen3.attention.key_length");
    } else {
        if (c.n_embd % c.n_head)
            throw std::runtime_error("inference: embedding width does not determine an integral head width");
        c.head_dim = c.n_embd / c.n_head;
    }
    if (c.head_dim <= 0 || c.head_dim % 2 ||
        c.n_head > std::numeric_limits<int>::max() / c.head_dim)
        throw std::runtime_error("inference: invalid attention projection dimensions");
    if (integer("qwen3.attention.value_length", c.head_dim) != c.head_dim ||
        integer("qwen3.rope.dimension_count", c.head_dim) != c.head_dim)
        throw std::runtime_error("inference: value and rotary widths must equal key width");
    c.context_length = integer("qwen3.context_length", c.context_length);
    c.rope_theta = float(real("qwen3.rope.freq_base", c.rope_theta));
    c.rms_eps = float(real("qwen3.attention.layer_norm_rms_epsilon", c.rms_eps));
    const uint64_t kv_width = uint64_t(c.n_head_kv) * c.head_dim;
    const auto max_floats = std::vector<float>().max_size();
    if (uint64_t(c.context_length) > max_floats / kv_width ||
        uint64_t(c.context_length) > max_floats / uint64_t(c.n_head))
        throw std::runtime_error("inference: context storage exceeds allocation limit");
    return c;
}

// A weight resolved once at load: type, storage and dimensions. Resolving per
// call meant rebuilding "blk.N." and hashing a tensor name for every
// projection of every layer of every token; the forward pass indexes layers_
// instead. It is also what lets a device backend recognize a weight across
// calls, which is the prerequisite for residency (docs/DEVICE-EXECUTION.md).
struct Weight {
    uint32_t type = 0;
    // A handle, not a pointer: the backend decides where the bytes live. The
    // model never dereferences it: slice() below names a location, and only a
    // backend turns that into an address.
    backend::BufferPtr data;
    size_t nin = 0, nout = 0;
    // Normalization weights are F32 by validation, so this is the whole row.
    backend::CSlice slice() const { return {data.get(), 0}; }
};

class Model;

struct LayerWeights {
    Weight attn_norm, attn_q_norm, attn_k_norm;
    Weight attn_q, attn_k, attn_v, attn_output;
    Weight ffn_norm, ffn_gate, ffn_up, ffn_down;
};

// One request's history in a model's cache: its block table and committed
// length, and the ticket of the last pass that touched it, which is what a
// release waits on rather than draining the device (docs/EXECUTION.md).
// Made by Model::make_sequence so it is bound to that model's pool and block
// size. Movable, not copyable; the server keeps one per request.
class Sequence {
public:
    Sequence() = default;
    size_t length() const { return kv_.length(); }
private:
    friend class Model;
    explicit Sequence(KVSequence kv) : kv_(std::move(kv)), bound_(true) {}
    KVSequence kv_;
    backend::Ticket last_ = 0;
    bool bound_ = false;
};

// One pass in flight: the activation arena, the host-visible logits rows and
// the ticket of its submission. Storage is allocated by the first forward
// that needs it and grows to the largest pass seen. Two contexts are what
// let a scheduler keep one pass on the device while it reads another's
// logits; the CLI has one. Plain data that Model fills.
struct ExecContext {
    // Row i of the logits the last forward produced, in entry order, valid
    // until the next forward through this context. The first read waits on
    // the pass's ticket, so forward itself never blocks: a caller with two
    // contexts submits the next pass before it reads this one.
    const float* logits(size_t i) {
        if (!logits_buf || i >= n_logits)
            throw std::out_of_range("inference: no such logits row");
        if (pending) { backend->wait(ticket); pending = false; }
        const void* p = logits_buf->host_ptr();
        if (!p) throw std::runtime_error("inference: logits are not host visible");
        return (const float*)p + i * width;
    }
    size_t n_logits = 0;
    size_t width = 0;
    backend::Ticket ticket = 0;
    backend::Backend* backend = nullptr;
    bool pending = false;

    static constexpr size_t kSlots = 9;
    backend::BufferPtr arena, logits_buf;
    size_t rows = 0, logit_rows = 0;
    size_t offset[kSlots] = {0};
    std::vector<uint32_t> ids, pos, pick;
    std::vector<backend::KVView> views;
};

// What one sequence contributes to a pass: `n` tokens appended to `seq`, and
// whether the logits after its last token are wanted. A prefill microbatch
// is one entry with many tokens, a decode batch is many entries with one,
// and the two mix freely. A sequence appears in a batch at most once.
struct BatchEntry {
    Sequence* seq;
    const uint32_t* ids;
    size_t n;
    bool want_logits;
};

class Model {
public:
    // Construct the model over a GGUF model, using the given backend (defaults
    // to the CPU backend). The model owns a reference to the model data, which
    // must outlive the Model.
    explicit Model(const gguf::GGUFModel& m,
                   backend::BackendPtr backend = backend::make_cpu_backend())
        : m_(&m), b_(std::move(backend)) {
        if (!b_) throw std::runtime_error("inference: missing backend");
        quant::register_builtins(); // populate the quant registry (idempotent)
        cfg = load_config(m);
        // The attention projection width is n_head*head_dim, which only equals
        // n_embd by coincidence on some models (Qwen3-8B: 32*128 == 4096).
        // Qwen3-0.6B/1.7B/4B have head_dim 128 with a smaller n_embd.
        q_dim_ = cfg.n_head * cfg.head_dim;

        if (m.offsets.size() != m.tensors.size())
            throw std::runtime_error("inference: tensor storage count mismatch");
        for (size_t i = 0; i < m.tensors.size(); i++) {
            const auto& t = m.tensors[i];
            if (!tindex_.emplace(t.name, i).second)
                throw std::runtime_error("inference: duplicate tensor " + t.name);
            if (t.ne.size() > 4)
                throw std::runtime_error("inference: invalid tensor rank " + t.name);
            const uint64_t bytes = t.data_size();
            if (m.offsets[i] % alignof(float) || m.offsets[i] > m.blob.size() ||
                bytes > m.blob.size() - m.offsets[i])
                throw std::runtime_error("inference: invalid tensor storage " + t.name);
        }

        // Tied embeddings: models without a separate output.weight reuse
        // token_embd.weight as the output projection (same [n_embd, n_vocab]
        // layout), so the head is just a matvec against the embedding matrix.
        out_name_ = tindex_.count("output.weight") ? "output.weight" : "token_embd.weight";
        resolve_tensors();

        // Budget: the whole context. The backend turns tokens into blocks and
        // bytes; storage is backed on demand, so a short chat does not
        // allocate it.
        kv_storage_ = b_->kv_alloc(cfg.n_layer, cfg.n_head_kv, cfg.head_dim,
                                   (size_t)cfg.context_length);
        kv_pool_.configure(kv_storage_->max_blocks());
        seq_ = make_sequence();

        // Precompute the RoPE cos/sin table for every position up to the
        // context length. Indexed as [pos*(head_dim/2) + i]. The backend
        // reads it through adopted buffers, so the host vectors stay alive
        // for the model's lifetime; on CPU that is the same memory.
        int half = cfg.head_dim / 2;
        rope_cos_.assign((size_t)cfg.context_length * half, 0.0f);
        rope_sin_.assign((size_t)cfg.context_length * half, 0.0f);
        for (int pos = 0; pos < cfg.context_length; pos++) {
            for (int i = 0; i < half; i++) {
                float fre = std::pow(cfg.rope_theta, -2.0f * (float)i / (float)cfg.head_dim);
                rope_cos_[(size_t)pos * half + i] = std::cos((float)pos * fre);
                rope_sin_[(size_t)pos * half + i] = std::sin((float)pos * fre);
            }
        }
        rope_cos_buf_ = b_->adopt(rope_cos_.data(), rope_cos_.size() * sizeof(float));
        rope_sin_buf_ = b_->adopt(rope_sin_.data(), rope_sin_.size() * sizeof(float));
    }

    // Sequences hold the pool's address; moving the model would leave them
    // pointing at the old one. Nothing moves a Model today.
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    void set_threads(int n) { b_->set_threads(n); }
    int threads_available() const { return b_->threads_available(); }
    // 0 keeps the default. Sets how a prompt is chunked; storage follows the
    // passes actually run.
    void set_ubatch(int n) { if (n > 0) ubatch_ = n; }

    int n_tokens() const { return (int)seq_.length(); }
    int head_dim() const { return cfg.head_dim; }
    int context_length() const { return cfg.context_length; }

    // A fresh history over this model's cache.
    Sequence make_sequence() {
        return Sequence(KVSequence(&kv_pool_, b_->kv_layout().block_tokens));
    }

    // One pass over every entry: each sequence's tokens go through the graph
    // at their own positions and attend through their own history, and the
    // logits after the last token of every entry that wants them land in
    // the context, in entry order. The pass is one submission. It is one
    // transaction as well: every sequence commits its tokens only once the
    // pass is submitted, and a failure before that leaves every history as
    // it was. The context reads the logits after waiting on the ticket.
    void forward(ExecContext& ctx, const BatchEntry* entries, size_t n_entries) {
        if (!entries || !n_entries) throw std::runtime_error("inference: empty batch");
        size_t rows = 0, want = 0;
        for (size_t e = 0; e < n_entries; ++e) {
            const BatchEntry& en = entries[e];
            if (!en.seq || !en.seq->bound_)
                throw std::runtime_error("inference: batch entry without a sequence");
            if (!en.ids || !en.n)
                throw std::runtime_error("inference: batch entry without tokens");
            // The RoPE table is precomputed for [0, context_length); a row
            // past it would read off the end.
            if (en.n > (size_t)cfg.context_length ||
                en.seq->length() > (size_t)cfg.context_length - en.n)
                throw std::runtime_error("inference: context length exceeded (" +
                                         std::to_string(cfg.context_length) + " tokens)");
            rows += en.n;
            want += en.want_logits ? 1 : 0;
        }
        ensure(ctx, rows, want);
        ctx.ids.resize(rows);
        ctx.pos.resize(rows);
        ctx.pick.resize(want);
        ctx.views.resize(n_entries);

        // Blocks are taken for every entry before anything runs. A sequence
        // listed twice fails here, since its second prepare finds the first
        // still pending.
        size_t prepared = 0;
        try {
            for (; prepared < n_entries; ++prepared)
                entries[prepared].seq->kv_.prepare(entries[prepared].n);
        } catch (...) {
            for (size_t e = 0; e < prepared; ++e) entries[e].seq->kv_.abort();
            throw;
        }
        size_t r = 0, w = 0;
        for (size_t e = 0; e < n_entries; ++e) {
            const BatchEntry& en = entries[e];
            const size_t len = en.seq->length();
            for (size_t b = 0; b < en.n; ++b) {
                ctx.ids[r + b] = en.ids[b];
                ctx.pos[r + b] = (uint32_t)(len + b);
            }
            ctx.views[e] = en.seq->kv_.view(kv_storage_.get());
            r += en.n;
            if (en.want_logits) ctx.pick[w++] = (uint32_t)(r - 1);
        }

        try {
            body(ctx, rows, n_entries);
            if (want) {
                // The rows that want logits are not contiguous once entries
                // mix, so they are compacted first and the head runs once
                // over exactly those rows.
                const size_t E = (size_t)cfg.n_embd;
                b_->gather_rows(slot(ctx, 1), slot(ctx, 0), E, ctx.pick.data(), want);
                b_->rms_norm_rows(slot(ctx, 1), slot(ctx, 1), output_norm_.slice(),
                                  want, E, E, cfg.rms_eps);
                b_->matmul(output_.type, output_.slice(), slot(ctx, 1),
                           {ctx.logits_buf.get(), 0}, output_.nin, output_.nout, want);
            }
            ctx.ticket = b_->submit();
        } catch (...) {
            retire();
            for (size_t e = 0; e < n_entries; ++e) entries[e].seq->kv_.abort();
            throw;
        }
        ctx.n_logits = want;
        ctx.backend = b_.get();
        ctx.pending = want > 0;
        for (size_t e = 0; e < n_entries; ++e) {
            entries[e].seq->kv_.commit();
            entries[e].seq->last_ = ctx.ticket;
        }
    }

    // Start a new history. Blocks return to the pool; their storage is
    // retained. Every pass ends in a submit or, on failure, a sync, so the
    // sequence's last ticket covers everything that could still be touching
    // a block: this waits for that and no more.
    void reset(Sequence& s) {
        b_->wait(s.last_);
        s.kv_.reset();
    }

    // The single-sequence entry points the CLI uses: one sequence and one
    // context owned here, and one entry per pass.

    // Run one token through the model (prefill or continue). Returns logits
    // over the full vocabulary.
    std::vector<float> step(int token_id) {
        const uint32_t id = (uint32_t)token_id;
        const BatchEntry entry{&seq_, &id, 1, true};
        forward(ctx_, &entry, 1);
        return row(ctx_, 0);
    }

    // Process a whole prompt with matrix-matrix matmuls instead of one token at
    // a time. Each weight row is then reused across the batch, which is the
    // difference between prefill being compute bound and paying the entire
    // weight stream once per token. Only the final token's logits are needed,
    // so only the last pass asks for them.
    std::vector<float> prefill(const std::vector<uint32_t>& ids) {
        if (ids.empty()) throw std::runtime_error("inference: empty prompt");
        // The prompt is one transaction across its microbatches: a failure in
        // any of them restores the history from before the call.
        const size_t start = seq_.length();
        auto work = [&] {
            // Sized to the largest chunk this prompt will use, inside the
            // scope, so a short prompt does not allocate scratch for a full
            // ubatch (at n_ff 12288 a 512-wide gate/up/ffn is about 25 MB each).
            ensure(ctx_, std::min((size_t)ubatch(), ids.size()), 1);
            size_t i = 0;
            while (i < ids.size()) {
                const size_t B = std::min((size_t)ubatch(), ids.size() - i);
                const BatchEntry entry{&seq_, ids.data() + i, B, i + B == ids.size()};
                forward(ctx_, &entry, 1);
                i += B;
            }
        };
        try {
            b_->run_prefill(std::ref(work));
        } catch (...) {
            retire();
            seq_.kv_.truncate(start);
            throw;
        }
        return row(ctx_, 0);
    }

    void reset() { reset(seq_); }

    // Allocated is what the backend backs; used is the committed history.
    // The gap is the paging cost in memory (docs/KV-CACHE.md).
    size_t kv_allocated_bytes() const { return kv_storage_->allocated_bytes(); }
    size_t kv_peak_bytes() const { return kv_storage_->peak_bytes(); }
    size_t kv_used_bytes() const {
        return seq_.length() * cfg.n_layer * 2 * cfg.n_head_kv * cfg.head_dim * sizeof(float);
    }

private:
    const gguf::GGUFModel* m_;
    backend::BackendPtr b_;
    QwenConfig cfg;
    int q_dim_ = 0;
    int ubatch_ = 512;   // default matches llama.cpp
    std::string out_name_;
    std::unordered_map<std::string, size_t> tindex_;
    std::vector<LayerWeights> layers_;
    Weight token_embd_, output_norm_, output_;
    std::unique_ptr<backend::KVStorage> kv_storage_;
    BlockPool kv_pool_;
    std::vector<float> rope_cos_, rope_sin_;
    backend::BufferPtr rope_cos_buf_, rope_sin_buf_;
    Sequence seq_;
    ExecContext ctx_;

    backend::CSlice rope_cos() const { return {rope_cos_buf_.get(), 0}; }
    backend::CSlice rope_sin() const { return {rope_sin_buf_.get(), 0}; }

    const gguf::TensorInfo& tensor(const std::string& name) const {
        auto it = tindex_.find(name);
        if (it == tindex_.end()) throw std::runtime_error("inference: missing tensor " + name);
        return m_->tensors[it->second];
    }

    // Validate every tensor this architecture needs and resolve it to a
    // Weight in the same pass, so a resolved handle is well-formed by
    // construction and the forward pass never looks a tensor up by name.
    void resolve_tensors() {
        const auto& embedding = tensor("token_embd.weight");
        if (embedding.ne.size() < 2 || !embedding.ne[1] ||
            embedding.ne[1] > uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("inference: invalid vocabulary dimension");
        const uint64_t vocab = embedding.ne[1];
        auto check = [&](const std::string& name, uint64_t input, uint64_t output,
                         bool norm = false) -> Weight {
            const auto& t = tensor(name);
            bool valid = !t.ne.empty() && t.ne[0] == input;
            if (norm) {
                valid = valid && t.type == gguf::GGML_TYPE_F32;
            } else {
                valid = valid && t.ne.size() >= 2 && t.ne[1] == output;
            }
            for (size_t d = norm ? 1 : 2; d < t.ne.size(); ++d) valid = valid && t.ne[d] == 1;
            if (!valid) throw std::runtime_error("inference: incompatible tensor layout " + name);
            // adopt, not copy: the payload is already resident and the
            // GGUF model outlives this one by contract.
            const size_t i = tindex_.at(t.name);
            return Weight{t.type, b_->adopt(m_->tensor_data(i), m_->tensor_bytes(i)),
                          (size_t)input, (size_t)output};
        };
        token_embd_ = check("token_embd.weight", cfg.n_embd, vocab);
        output_ = check(out_name_, cfg.n_embd, vocab);
        output_norm_ = check("output_norm.weight", cfg.n_embd, 1, true);
        const uint64_t kv_width = uint64_t(cfg.n_head_kv) * cfg.head_dim;
        layers_.resize(cfg.n_layer);
        for (int l = 0; l < cfg.n_layer; ++l) {
            const std::string pre = "blk." + std::to_string(l) + ".";
            LayerWeights& w = layers_[l];
            w.attn_norm   = check(pre + "attn_norm.weight", cfg.n_embd, 1, true);
            w.attn_q_norm = check(pre + "attn_q_norm.weight", cfg.head_dim, 1, true);
            w.attn_k_norm = check(pre + "attn_k_norm.weight", cfg.head_dim, 1, true);
            w.attn_q      = check(pre + "attn_q.weight", cfg.n_embd, q_dim_);
            w.attn_k      = check(pre + "attn_k.weight", cfg.n_embd, kv_width);
            w.attn_v      = check(pre + "attn_v.weight", cfg.n_embd, kv_width);
            w.attn_output = check(pre + "attn_output.weight", q_dim_, cfg.n_embd);
            w.ffn_norm    = check(pre + "ffn_norm.weight", cfg.n_embd, 1, true);
            w.ffn_gate    = check(pre + "ffn_gate.weight", cfg.n_embd, cfg.n_ff);
            w.ffn_up      = check(pre + "ffn_up.weight", cfg.n_embd, cfg.n_ff);
            w.ffn_down    = check(pre + "ffn_down.weight", cfg.n_ff, cfg.n_embd);
        }
    }

    // Physical batch: how many tokens go through ONE forward pass of the
    // graph. This is llama.cpp's n_ubatch (-ub), not n_batch: it sets the GEMM
    // width and the scratch buffer sizes. llmx has no logical batch, since
    // there is one sequence and no queue; that distinction only starts to
    // matter with the multi-user server in ROADMAP #7.
    int ubatch() const { return ubatch_; }

    // One backend allocation holding the nine activations of a pass, each at
    // a 64-byte boundary so the AVX2 kernels see the alignment they saw when
    // every vector was its own allocation. Device allocators handle a few
    // large blocks far better than many small ones, and resizing is one
    // call. The caller only publishes the result once this returns, so an
    // allocation that throws leaves the previous arena intact.
    backend::BufferPtr alloc_arena(const size_t (&counts)[ExecContext::kSlots],
                                   size_t (&offsets)[ExecContext::kSlots]) const {
        size_t total = 0;
        for (size_t i = 0; i < ExecContext::kSlots; ++i) {
            offsets[i] = total;
            const size_t bytes = counts[i] * sizeof(float);
            if (bytes / sizeof(float) != counts[i] || total > (size_t)-1 - bytes - 63)
                throw std::runtime_error("inference: activation arena size overflows");
            total = (total + bytes + 63) / 64 * 64;
        }
        return b_->alloc(total);
    }

    // Storage for a pass of `rows` rows with `want` logits rows: grown when
    // a pass needs more than the context holds, never shrunk. Each is
    // allocated whole before it replaces what the context had.
    void ensure(ExecContext& ctx, size_t rows, size_t want) {
        auto mul = [](size_t a, size_t b) {
            if (b && a > (size_t)-1 / b)
                throw std::runtime_error("inference: activation arena size overflows");
            return a * b;
        };
        if (!ctx.arena || ctx.rows < rows) {
            const size_t KV = (size_t)cfg.n_head_kv * cfg.head_dim;
            const size_t counts[ExecContext::kSlots] = {
                mul(rows, (size_t)cfg.n_embd),   // x
                mul(rows, (size_t)cfg.n_embd),   // h
                mul(rows, (size_t)q_dim_),       // q
                mul(rows, KV),                   // k
                mul(rows, KV),                   // v
                mul(rows, (size_t)q_dim_),       // attn
                mul(rows, (size_t)cfg.n_ff),     // gate
                mul(rows, (size_t)cfg.n_ff),     // up
                mul(rows, (size_t)cfg.n_ff),     // ffn
            };
            size_t offsets[ExecContext::kSlots];
            backend::BufferPtr arena = alloc_arena(counts, offsets);
            ctx.arena = std::move(arena);
            std::copy(offsets, offsets + ExecContext::kSlots, ctx.offset);
            ctx.rows = rows;
        }
        if (want && (!ctx.logits_buf || ctx.logit_rows < want)) {
            // The head writes here and the host reads it in place once the
            // pass has retired: the one point per pass that must be host
            // visible, and the one wait per pass.
            backend::BufferPtr logits = b_->alloc(
                mul(mul(want, output_.nout), sizeof(float)), backend::Memory::host_visible);
            ctx.logits_buf = std::move(logits);
            ctx.logit_rows = want;
            ctx.width = output_.nout;
        }
    }

    static backend::Slice slot(const ExecContext& ctx, size_t i) {
        return {ctx.arena.get(), ctx.offset[i] / sizeof(float)};
    }

    static std::vector<float> row(ExecContext& ctx, size_t i) {
        const float* p = ctx.logits(i);
        return std::vector<float>(p, p + ctx.width);
    }

    // The graph over `rows` rows and the views of the entries they came
    // from. Slots: 0 x, 1 h, 2 q, 3 k, 4 v, 5 attn, 6 gate, 7 up, 8 ffn.
    void body(ExecContext& ctx, size_t rows, size_t n_views) {
        const size_t E = (size_t)cfg.n_embd, half = (size_t)cfg.head_dim / 2;
        const size_t KV = (size_t)cfg.n_head_kv * cfg.head_dim;
        const backend::Slice x = slot(ctx, 0), h = slot(ctx, 1), q = slot(ctx, 2),
                             k = slot(ctx, 3), v = slot(ctx, 4), attn = slot(ctx, 5),
                             gate = slot(ctx, 6), up = slot(ctx, 7), ffn = slot(ctx, 8);

        b_->embed(x, token_embd_.type, token_embd_.slice(), token_embd_.nin,
                  token_embd_.nout, ctx.ids.data(), rows);

        for (int l = 0; l < cfg.n_layer; l++) {
            const LayerWeights& w = layers_[l];

            b_->rms_norm_rows(h, x, w.attn_norm.slice(), rows, E, E, cfg.rms_eps);

            b_->matmul_group({projection(w.attn_q, q),
                              projection(w.attn_k, k),
                              projection(w.attn_v, v)}, h, E, rows);

            b_->norm_rope_rows(q, rows, (size_t)q_dim_, cfg.n_head,
                               w.attn_q_norm.slice(), cfg.rms_eps,
                               rope_cos(), rope_sin(), half, ctx.pos.data());
            b_->norm_rope_rows(k, rows, KV, cfg.n_head_kv,
                               w.attn_k_norm.slice(), cfg.rms_eps,
                               rope_cos(), rope_sin(), half, ctx.pos.data());

            b_->kv_write(l, ctx.views.data(), n_views, k, v);
            b_->attention(q, l, ctx.views.data(), n_views, attn,
                          cfg.n_head, cfg.n_head_kv, cfg.head_dim);

            matmul(w.attn_output, attn, h, rows);
            b_->add(x, h, rows * E);

            b_->rms_norm_rows(h, x, w.ffn_norm.slice(), rows, E, E, cfg.rms_eps);

            b_->matmul_group({projection(w.ffn_gate, gate),
                              projection(w.ffn_up, up)}, h, E, rows);
            b_->silu_mul(ffn, gate, up, rows * (size_t)cfg.n_ff);
            matmul(w.ffn_down, ffn, h, rows);
            b_->add(x, h, rows * E);
        }
    }

    // A block returns to the pool only once the backend has retired every
    // submission that touched it (docs/KV-CACHE.md). The CPU backend is eager
    // so this costs nothing; on a device, releasing a block while a write to
    // it is still queued hands a later sequence someone else's history. The
    // callers are the exception paths, where a failed pass has ops queued
    // behind no ticket, so this drains rather than waits; reset() has a
    // ticket and waits on it. sync() cannot throw for the same reason.
    void retire() noexcept { b_->sync(); }

    // The buffer is passed by raw pointer, not by handle: three projections
    // per layer per token is nearly two hundred refcount pairs a token if a
    // shared pointer is copied here instead.
    static backend::Projection projection(const Weight& w, backend::Slice out) {
        return {w.type, {w.data.get(), 0}, out, w.nout};
    }

    // Batched matmul. The backend dispatches on the quant type, so every
    // block format takes the same path; there is no per-type branch here.
    void matmul(const Weight& w, backend::CSlice X, backend::Slice Y, size_t nbatch) {
        b_->matmul(w.type, w.slice(), X, Y, w.nin, w.nout, nbatch);
    }
};

} // namespace infer
