#pragma once
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <thread>
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
    // model never dereferences it except through f32() below, which only the
    // host path uses for normalization rows.
    backend::BufferPtr data;
    size_t nin = 0, nout = 0;
    // Normalization weights are F32 by validation, so this is the whole row.
    const float* f32() const {
        return reinterpret_cast<const float*>(data->host_ptr());
    }
};

struct LayerWeights {
    Weight attn_norm, attn_q_norm, attn_k_norm;
    Weight attn_q, attn_k, attn_v, attn_output;
    Weight ffn_norm, ffn_gate, ffn_up, ffn_down;
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

        // buffers
        {
            const size_t KV = (size_t)cfg.n_head_kv * cfg.head_dim;
            const size_t counts[kArenaSlots] = {
                (size_t)cfg.n_embd, (size_t)cfg.n_embd, (size_t)q_dim_, KV, KV,
                (size_t)q_dim_, (size_t)cfg.n_ff, (size_t)cfg.n_ff, (size_t)cfg.n_ff};
            decode_arena_ = alloc_arena(counts, decode_offset_);
        }

        // Budget: the whole context. The backend turns tokens into blocks and
        // bytes; storage is backed on demand, so a short chat does not
        // allocate it.
        kv_storage_ = b_->kv_alloc(cfg.n_layer, cfg.n_head_kv, cfg.head_dim,
                                   (size_t)cfg.context_length);
        kv_pool_.configure(kv_storage_->max_blocks());
        kv_seq_ = KVSequence(&kv_pool_, b_->kv_layout().block_tokens);

        // Precompute the RoPE cos/sin table for every position up to the
        // context length. Indexed as [pos*(head_dim/2) + i].
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
    }

    // One sequence owns one pool through a raw pointer; moving the model
    // would leave it pointing at the old pool. Nothing moves a Model today.
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    void set_threads(int n) { b_->set_threads(n); }
    int threads_available() const { return b_->threads_available(); }
    // 0 keeps the default. Changing it invalidates the scratch buffers.
    void set_ubatch(int n) {
        if (n > 0 && n != ubatch_) { ubatch_ = n; arena_.reset(); arena_batch_ = 0; }
    }

    int n_tokens() const { return n_tokens_; }
    int head_dim() const { return cfg.head_dim; }
    int context_length() const { return cfg.context_length; }

    // Run one token through the model (prefill or continue). Returns logits
    // over the full vocabulary.
    std::vector<float> step(int token_id) {
        int pos = n_tokens_;
        // The RoPE table is precomputed for [0, context_length); stepping past
        // it would read off the end of rope_cos_/rope_sin_.
        if (pos >= cfg.context_length)
            throw std::runtime_error("inference: context length exceeded (" +
                                     std::to_string(cfg.context_length) + " tokens)");

        // The whole step is one transaction: blocks are taken first, and the
        // committed length and token position advance together only after
        // the logits exist. A failure anywhere leaves the previous history.
        kv_seq_.prepare(1);
        std::vector<float> logits;
        try {
            step_body(token_id, pos);
            b_->rms_norm(h(), x(), output_norm_.f32(), cfg.n_embd, cfg.rms_eps);
            logits.assign(output_.nout, 0.0f);
            matvec(output_, h(), logits.data());
        } catch (...) {
            kv_seq_.abort();
            throw;
        }
        kv_seq_.commit();
        n_tokens_++;
        return logits;
    }

    void step_body(int token_id, int pos) {
        const backend::KVView view = kv_seq_.view(kv_storage_.get());

        // embedding. The id lives in a member slot rather than on the stack.
        // step() is a very large inline body in a header, and adding one local
        // to it shifted the generated layout enough to cost 8% of 0.6B prefill
        // for work measured at 0.0014 ms (docs/STATUS.md).
        embed_id_ = (uint32_t)token_id;
        b_->embed(x(), token_embd_.type, *token_embd_.data, token_embd_.nin,
                  token_embd_.nout, &embed_id_, 1);

        for (int l = 0; l < cfg.n_layer; l++) {
            const LayerWeights& w = layers_[l];

            // attn norm
            b_->rms_norm(h(), x(), w.attn_norm.f32(), cfg.n_embd, cfg.rms_eps);

            // q,k,v projections
            b_->matmul_group({projection(w.attn_q, q()),
                              projection(w.attn_k, kv()),
                              projection(w.attn_v, v())},
                             h(), cfg.n_embd, 1);

            // per-head q/k norms + rope
            {
                const size_t half = cfg.head_dim / 2;
                const float* cs = rope_cos_.data() + (size_t)pos * half;
                const float* sn = rope_sin_.data() + (size_t)pos * half;
                b_->norm_rope_rows(q(), 1, 0, cfg.n_head,
                                   w.attn_q_norm.f32(), cfg.rms_eps, cs, sn, half);
                b_->norm_rope_rows(kv(), 1, 0, cfg.n_head_kv,
                                   w.attn_k_norm.f32(), cfg.rms_eps, cs, sn, half);
            }

            b_->kv_write(l, view, (size_t)pos, kv(), v(), 1);
            b_->attention(q(), l, view, attn(),
                          cfg.n_head, cfg.n_head_kv, cfg.head_dim, 1);

            // attn_output projection + residual
            std::memset(h(), 0, (size_t)cfg.n_embd * sizeof(float));
            matvec(w.attn_output, attn(), h());
            b_->add(x(), h(), cfg.n_embd);

            // ffn norm
            b_->rms_norm(h(), x(), w.ffn_norm.f32(), cfg.n_embd, cfg.rms_eps);

            // gate/up (SwiGLU). Buffers are members: allocating these per layer
            // per token cost 108 heap allocations of n_ff floats on a 36-layer
            // model, every token.
            b_->matmul_group({projection(w.ffn_gate, gate()),
                              projection(w.ffn_up, up())},
                             h(), cfg.n_embd, 1);
            b_->silu_mul(ffn(), gate(), up(), cfg.n_ff);
            // down projection + residual
            std::memset(h(), 0, (size_t)cfg.n_embd * sizeof(float));
            matvec(w.ffn_down, ffn(), h());
            b_->add(x(), h(), cfg.n_embd);
        }
    }

    // Process a whole prompt with matrix-matrix matmuls instead of one token at
    // a time. Each weight row is then reused across the batch, which is the
    // difference between prefill being compute bound and paying the entire
    // weight stream once per token.
    // Only the final token's logits are ever needed, so the vocab projection
    // stays a single matvec rather than B of them.
    std::vector<float> prefill(const std::vector<uint32_t>& ids) {
        if (ids.empty()) throw std::runtime_error("inference: empty prompt");
        std::vector<float> logits;
        // The prompt is one transaction across its microbatches: a failure in
        // any of them restores the history from before the call.
        const int start = n_tokens_;
        auto work = [&] {
            ensure_batch_buffers(ids.size());
            size_t i = 0;
            while (i < ids.size()) {
                const int B = (int)std::min((size_t)ubatch(), ids.size() - i);
                const bool last = (i + (size_t)B == ids.size());
                forward_batch(&ids[i], B, last ? &logits : nullptr);
                i += (size_t)B;
            }
        };
        try {
            b_->run_prefill(std::ref(work));
        } catch (...) {
            kv_seq_.truncate((size_t)start);
            n_tokens_ = start;
            throw;
        }
        return logits;
    }

    // Start a new conversation. Blocks return to the pool; their storage is
    // retained for the next history.
    void reset() {
        kv_seq_.reset();
        n_tokens_ = 0;
    }

    // Allocated is what the backend backs; used is the committed history.
    // The gap is the paging cost in memory (docs/KV-CACHE.md).
    size_t kv_allocated_bytes() const { return kv_storage_->allocated_bytes(); }
    size_t kv_peak_bytes() const { return kv_storage_->peak_bytes(); }
    size_t kv_used_bytes() const {
        return (size_t)n_tokens_ * cfg.n_layer * 2 * cfg.n_head_kv * cfg.head_dim * sizeof(float);
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

    uint32_t embed_id_ = 0;
    static const size_t kArenaSlots = 9;
    backend::BufferPtr decode_arena_;
    size_t decode_offset_[kArenaSlots] = {0};
    backend::BufferPtr arena_;
    size_t arena_batch_ = 0;
    size_t arena_offset_[kArenaSlots] = {0};
    std::unique_ptr<backend::KVStorage> kv_storage_;
    BlockPool kv_pool_;
    KVSequence kv_seq_;
    std::vector<float> rope_cos_, rope_sin_;
    int n_tokens_ = 0;
    const gguf::TensorInfo& tensor(const std::string& name) const {
        auto it = tindex_.find(name);
        if (it == tindex_.end()) throw std::runtime_error("inference: missing tensor " + name);
        return m_->tensors[it->second];
    }
    const uint8_t* tensor_data(const std::string& name) const {
        return m_->tensor_data(tindex_.at(name));
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


    // Sized to the largest chunk this prompt will actually use, so a short
    // prompt does not allocate scratch for a full ubatch (at n_ff 12288 a
    // 512-wide gate/up/ffn is about 25 MB each).
    // Readiness is published only once every buffer exists: the new set is
    // built aside and swapped in together, so a failed allocation part way
    // leaves the old set intact and a retry allocates again.
    // One backend allocation holding a set of activations, each at a 64-byte
    // boundary so the AVX2 kernels see the alignment they saw when every
    // vector was its own allocation. Device allocators handle a few large
    // blocks far better than many small ones, and resizing is one call.
    backend::BufferPtr alloc_arena(const size_t (&counts)[kArenaSlots],
                                   size_t (&offsets)[kArenaSlots]) const {
        size_t total = 0;
        for (size_t i = 0; i < kArenaSlots; ++i) {
            offsets[i] = total;
            const size_t bytes = counts[i] * sizeof(float);
            if (bytes / sizeof(float) != counts[i] || total > (size_t)-1 - bytes - 63)
                throw std::runtime_error("inference: activation arena size overflows");
            total = (total + bytes + 63) / 64 * 64;
        }
        auto arena = b_->alloc(total);
        std::memset(arena->mutable_host_ptr(), 0, total);
        return arena;
    }

    static float* slot_of(const backend::BufferPtr& arena, size_t offset) {
        return (float*)((uint8_t*)arena->mutable_host_ptr() + offset);
    }

    float* x() const { return slot_of(decode_arena_, decode_offset_[0]); }
    float* h() const { return slot_of(decode_arena_, decode_offset_[1]); }
    float* q() const { return slot_of(decode_arena_, decode_offset_[2]); }
    float* kv() const { return slot_of(decode_arena_, decode_offset_[3]); }
    float* v() const { return slot_of(decode_arena_, decode_offset_[4]); }
    float* attn() const { return slot_of(decode_arena_, decode_offset_[5]); }
    float* gate() const { return slot_of(decode_arena_, decode_offset_[6]); }
    float* up() const { return slot_of(decode_arena_, decode_offset_[7]); }
    float* ffn() const { return slot_of(decode_arena_, decode_offset_[8]); }

    void ensure_batch_buffers(size_t want) {
        const size_t B = std::min((size_t)ubatch(), std::max<size_t>(want, 1));
        if (arena_ && arena_batch_ >= B) return;
        const size_t KV = (size_t)cfg.n_head_kv * cfg.head_dim;
        const size_t counts[kArenaSlots] = {
            B * (size_t)cfg.n_embd,   // xb
            B * (size_t)cfg.n_embd,   // hb
            B * (size_t)q_dim_,       // qb
            B * KV,                   // kb
            B * KV,                   // vb
            B * (size_t)q_dim_,       // attnb
            B * (size_t)cfg.n_ff,     // gateb
            B * (size_t)cfg.n_ff,     // upb
            B * (size_t)cfg.n_ff,     // ffnb
        };
        arena_ = alloc_arena(counts, arena_offset_);
        arena_batch_ = B;
    }

    float* slot(size_t i) const { return slot_of(arena_, arena_offset_[i]); }
    float* xb() const { return slot(0); }
    float* hb() const { return slot(1); }
    float* qb() const { return slot(2); }
    float* kb() const { return slot(3); }
    float* vb() const { return slot(4); }
    float* attnb() const { return slot(5); }
    float* gateb() const { return slot(6); }
    float* upb() const { return slot(7); }
    float* ffnb() const { return slot(8); }

    void forward_batch(const uint32_t* ids, int B, std::vector<float>* out_logits) {
        const int pos0 = n_tokens_;
        if (pos0 + B > cfg.context_length)
            throw std::runtime_error("inference: context length exceeded (" +
                                     std::to_string(cfg.context_length) + " tokens)");
        kv_seq_.prepare((size_t)B);
        try {
            forward_batch_body(ids, B, pos0);
            if (out_logits) {
                b_->rms_norm(h(), xb() + (size_t)(B - 1) * cfg.n_embd,
                             output_norm_.f32(), cfg.n_embd, cfg.rms_eps);
                out_logits->assign(output_.nout, 0.0f);
                matvec(output_, h(), out_logits->data());
            }
        } catch (...) {
            kv_seq_.abort();
            throw;
        }
        kv_seq_.commit();
        n_tokens_ += B;
    }

    void forward_batch_body(const uint32_t* ids, int B, int pos0) {
        const backend::KVView view = kv_seq_.view(kv_storage_.get());
        const int E = cfg.n_embd, HD = cfg.head_dim;
        const size_t half = (size_t)HD / 2;
        const size_t KV = (size_t)cfg.n_head_kv * HD;

        b_->embed(xb(), token_embd_.type, *token_embd_.data, token_embd_.nin,
                  token_embd_.nout, ids, (size_t)B);

        for (int l = 0; l < cfg.n_layer; l++) {
            const LayerWeights& w = layers_[l];

            b_->rms_norm_rows(hb(), xb(), w.attn_norm.f32(),
                              (size_t)B, (size_t)E, (size_t)E, cfg.rms_eps);

            b_->matmul_group({projection(w.attn_q, qb()),
                              projection(w.attn_k, kb()),
                              projection(w.attn_v, vb())}, hb(), E, B);

            const float* cs = rope_cos_.data() + (size_t)pos0 * half;
            const float* sn = rope_sin_.data() + (size_t)pos0 * half;
            b_->norm_rope_rows(qb(), (size_t)B, (size_t)q_dim_, cfg.n_head,
                               w.attn_q_norm.f32(), cfg.rms_eps, cs, sn, half);
            b_->norm_rope_rows(kb(), (size_t)B, KV, cfg.n_head_kv,
                               w.attn_k_norm.f32(), cfg.rms_eps, cs, sn, half);

            b_->kv_write(l, view, (size_t)pos0, kb(), vb(), (size_t)B);
            b_->attention(qb(), l, view, attnb(),
                          cfg.n_head, cfg.n_head_kv, cfg.head_dim, B);

            matmul(w.attn_output, attnb(), hb(), B);
            b_->add(xb(), hb(), (size_t)B * E);

            b_->rms_norm_rows(hb(), xb(), w.ffn_norm.f32(),
                              (size_t)B, (size_t)E, (size_t)E, cfg.rms_eps);

            b_->matmul_group({projection(w.ffn_gate, gateb()),
                              projection(w.ffn_up, upb())}, hb(), E, B);
            b_->silu_mul(ffnb(), gateb(), upb(), (size_t)B * cfg.n_ff);
            matmul(w.ffn_down, ffnb(), hb(), B);
            b_->add(xb(), hb(), (size_t)B * E);
        }
    }

    // The buffer is passed by raw pointer, not by handle: three projections
    // per layer per token is nearly two hundred refcount pairs a token if a
    // shared pointer is copied here instead.
    static backend::Projection projection(const Weight& w, float* out) {
        return {w.type, w.data.get(), out, w.nout};
    }

    // Batched matmul. The backend dispatches on the quant type, so every
    // block format takes the same path; there is no per-type branch here.
    void matmul(const Weight& w, const float* X, float* Y, int nbatch) {
        b_->matmul(w.type, *w.data, X, Y, w.nin, w.nout, (size_t)nbatch);
    }

    // out = W^T x for a single column. Same backend entry point as the
    // batched form, so there is one dispatch path and one place that knows
    // about quant types.
    void matvec(const Weight& w, const float* x, float* out) {
        b_->matmul(w.type, *w.data, x, out, w.nin, w.nout, 1);
    }

};

} // namespace infer
