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

#include "format/gguf.hpp"
#include "core/fp16.hpp"
#include "quant/quant.hpp"
#include "backends/backend.hpp"
#include "backends/cpu/cpu_backend.hpp"

// Qwen3-style transformer forward pass, from scratch. The compute primitives
// (quantized matmul, RMSNorm, RoPE) are delegated to a backend::Backend, so the
// same model code runs on CPU now and other backends later.
// Supports dense Q8_0 / F32 tensors only:
//   token_embd.weight        Q8_0 [n_embd, n_vocab]
//   output.weight            Q8_0 [n_embd, n_vocab]
//   output_norm.weight       F32  [n_embd]
//   per layer l:
//     blk.l.attn_norm.weight F32  [n_embd]
//     blk.l.attn_q.weight    Q8_0 [n_embd, n_embd]
//     blk.l.attn_k.weight    Q8_0 [n_embd, n_embd*n_head_kv/n_head]
//     blk.l.attn_v.weight    Q8_0 [n_embd, n_embd*n_head_kv/n_head]
//     blk.l.attn_output      Q8_0 [n_embd, n_embd]
//     blk.l.attn_q_norm      F32  [head_dim]
//     blk.l.attn_k_norm      F32  [head_dim]
//     blk.l.ffn_norm         F32  [n_embd]
//     blk.l.ffn_gate         Q8_0 [n_embd, n_ff]
//     blk.l.ffn_up           Q8_0 [n_embd, n_ff]
//     blk.l.ffn_down         Q8_0 [n_ff, n_embd]
// Matrices are stored with ne[0] = input dim fastest: row o occupies bytes
// [o*nin/32*34, (o+1)*nin/32*34).

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
    auto getu = [&](const std::string& k) -> uint64_t {
        for (const auto& kv : m.kv) {
            if (kv.first == k) {
                if (kv.second.vtype == gguf::V_UINT32) return kv.second.u;
                if (kv.second.vtype == gguf::V_INT32)  return (uint64_t)kv.second.i;
                if (kv.second.vtype == gguf::V_UINT64) return kv.second.u;
                if (kv.second.vtype == gguf::V_INT64)  return (uint64_t)kv.second.i;
            }
        }
        return 0;
    };
    auto getf = [&](const std::string& k) -> float {
        for (const auto& kv : m.kv) {
            if (kv.first == k) {
                if (kv.second.vtype == gguf::V_FLOAT32) { float x; std::memcpy(&x, &kv.second.fb, 4); return x; }
                if (kv.second.vtype == gguf::V_FLOAT64) return (float)kv.second.f64;
            }
        }
        return 0.0f;
    };
    c.n_layer = (int)getu("qwen3.block_count");
    c.n_embd  = (int)getu("qwen3.embedding_length");
    c.n_ff    = (int)getu("qwen3.feed_forward_length");
    c.n_head  = (int)getu("qwen3.attention.head_count");
    c.n_head_kv = (int)getu("qwen3.attention.head_count_kv");
    int kl = (int)getu("qwen3.attention.key_length");
    if (kl <= 0) kl = c.n_embd / c.n_head;
    c.head_dim = kl;
    c.context_length = (int)getu("qwen3.context_length");
    if (c.context_length <= 0) c.context_length = 4096;
    c.rope_theta = getf("qwen3.rope.freq_base");
    if (c.rope_theta <= 0.0f) c.rope_theta = 10000.0f;
    c.rms_eps = getf("qwen3.attention.layer_norm_rms_epsilon");
    if (c.rms_eps <= 0.0f) c.rms_eps = 1e-6f;
    return c;
}

class Model {
public:
    // Construct the model over a GGUF model, using the given backend (defaults
    // to the CPU backend). The model owns a reference to the model data, which
    // must outlive the Model.
    explicit Model(const gguf::GGUFModel& m,
                   backend::BackendPtr backend = backend::make_cpu_backend())
        : m_(&m), b_(std::move(backend)) {
        quant::register_builtins(); // populate the quant registry (idempotent)
        cfg = load_config(m);
        if (cfg.n_layer <= 0 || cfg.n_embd <= 0) throw std::runtime_error("inference: incomplete Qwen3 config in metadata");
        if (cfg.n_head_kv <= 0) cfg.n_head_kv = cfg.n_head;
        ratio_ = cfg.n_head / cfg.n_head_kv;
        // The attention projection width is n_head*head_dim, which only equals
        // n_embd by coincidence on some models (Qwen3-8B: 32*128 == 4096).
        // Qwen3-0.6B/1.7B/4B have head_dim 128 with a smaller n_embd.
        q_dim_ = cfg.n_head * cfg.head_dim;

        for (size_t i = 0; i < m.tensors.size(); i++) tindex_[m.tensors[i].name] = i;

        // Tied embeddings: models without a separate output.weight reuse
        // token_embd.weight as the output projection (same [n_embd, n_vocab]
        // layout), so the head is just a matvec against the embedding matrix.
        out_name_ = tindex_.count("output.weight") ? "output.weight" : "token_embd.weight";
        if (!tindex_.count(out_name_))
            throw std::runtime_error("inference: missing output projection tensor");

        // buffers
        x_.assign(cfg.n_embd, 0.0f);
        h_.assign(cfg.n_embd, 0.0f);
        q_.assign(q_dim_, 0.0f);
        kv_.assign((size_t)cfg.n_head_kv * cfg.head_dim, 0.0f);
        v_.assign((size_t)cfg.n_head_kv * cfg.head_dim, 0.0f);
        attn_.assign(q_dim_, 0.0f);
        gate_.assign(cfg.n_ff, 0.0f);
        up_.assign(cfg.n_ff, 0.0f);
        ffn_.assign(cfg.n_ff, 0.0f);
        scores_.assign((size_t)cfg.n_head * (size_t)cfg.context_length, 0.0f);

        k_cache_.resize(cfg.n_layer);
        v_cache_.resize(cfg.n_layer);

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

    void set_threads(int n) { b_->set_threads(n); }

    int n_tokens() const { return n_tokens_; }
    int head_dim() const { return cfg.head_dim; }

    // Run one token through the model (prefill or continue). Returns logits
    // over the full vocabulary.
    std::vector<float> step(int token_id) {
        int pos = n_tokens_;
        // The RoPE table is precomputed for [0, context_length); stepping past
        // it would read off the end of rope_cos_/rope_sin_.
        if (pos >= cfg.context_length)
            throw std::runtime_error("inference: context length exceeded (" +
                                     std::to_string(cfg.context_length) + " tokens)");

        // embedding
        dequant_row(tensor("token_embd.weight"), token_id, x_.data());

        for (int l = 0; l < cfg.n_layer; l++) {
            const std::string pre = "blk." + std::to_string(l) + ".";

            // attn norm
            b_->rms_norm(h_.data(), x_.data(),
                         (const float*)tensor_data(pre + "attn_norm.weight"),
                         cfg.n_embd, cfg.rms_eps);

            // q,k,v projections
            matvec(tensor(pre + "attn_q.weight"), h_.data(), q_.data(), cfg.n_embd, (size_t)q_dim_);
            matvec(tensor(pre + "attn_k.weight"), h_.data(), kv_.data(), cfg.n_embd, (size_t)cfg.n_head_kv * cfg.head_dim);
            matvec(tensor(pre + "attn_v.weight"), h_.data(), v_.data(), cfg.n_embd, (size_t)cfg.n_head_kv * cfg.head_dim);

            // per-head q/k norms
            const float* qnorm = (const float*)tensor_data(pre + "attn_q_norm.weight");
            const float* knorm = (const float*)tensor_data(pre + "attn_k_norm.weight");
            for (int h = 0; h < cfg.n_head; h++)
                b_->rms_norm(q_.data() + h * cfg.head_dim, q_.data() + h * cfg.head_dim, qnorm, cfg.head_dim, cfg.rms_eps);
            for (int h = 0; h < cfg.n_head_kv; h++)
                b_->rms_norm(kv_.data() + h * cfg.head_dim, kv_.data() + h * cfg.head_dim, knorm, cfg.head_dim, cfg.rms_eps);

            // rope
            {
                int half = cfg.head_dim / 2;
                for (int h = 0; h < cfg.n_head; h++)
                    b_->rope(q_.data() + h * cfg.head_dim,
                             rope_cos_.data() + (size_t)pos * half,
                             rope_sin_.data() + (size_t)pos * half, half);
                for (int h = 0; h < cfg.n_head_kv; h++)
                    b_->rope(kv_.data() + h * cfg.head_dim,
                             rope_cos_.data() + (size_t)pos * half,
                             rope_sin_.data() + (size_t)pos * half, half);
            }

            // store k,v in cache
            k_cache_[l].insert(k_cache_[l].end(), kv_.begin(), kv_.end());
            v_cache_[l].insert(v_cache_[l].end(), v_.begin(), v_.end());

            // attention: each q-head writes only its own attn_ slice, so heads
            // can be processed in parallel.
            const float* kcache = k_cache_[l].data();
            const float* vcache = v_cache_[l].data();
            attend_heads(kcache, vcache);

            // attn_output projection + residual
            std::fill(h_.begin(), h_.end(), 0.0f);
            matvec(tensor(pre + "attn_output.weight"), attn_.data(), h_.data(), (size_t)q_dim_, cfg.n_embd);
            for (int i = 0; i < cfg.n_embd; i++) x_[i] += h_[i];

            // ffn norm
            b_->rms_norm(h_.data(), x_.data(),
                         (const float*)tensor_data(pre + "ffn_norm.weight"),
                         cfg.n_embd, cfg.rms_eps);

            // gate/up (SwiGLU). Buffers are members: allocating these per layer
            // per token cost 108 heap allocations of n_ff floats on a 36-layer
            // model, every token.
            matvec(tensor(pre + "ffn_gate.weight"), h_.data(), gate_.data(), cfg.n_embd, cfg.n_ff);
            matvec(tensor(pre + "ffn_up.weight"),   h_.data(), up_.data(),   cfg.n_embd, cfg.n_ff);
            for (int i = 0; i < cfg.n_ff; i++) {
                float g = gate_[i] / (1.0f + std::exp(-gate_[i])); // SiLU
                ffn_[i] = g * up_[i];
            }
            // down projection + residual
            std::fill(h_.begin(), h_.end(), 0.0f);
            matvec(tensor(pre + "ffn_down.weight"), ffn_.data(), h_.data(), cfg.n_ff, cfg.n_embd);
            for (int i = 0; i < cfg.n_embd; i++) x_[i] += h_[i];
        }

        // final norm + output projection
        b_->rms_norm(h_.data(), x_.data(),
                     (const float*)tensor_data("output_norm.weight"),
                     cfg.n_embd, cfg.rms_eps);
        const gguf::TensorInfo& out_t = tensor(out_name_);
        size_t n_vocab = out_t.ne[1];
        std::vector<float> logits(n_vocab);
        matvec(out_t, h_.data(), logits.data(), cfg.n_embd, n_vocab);

        n_tokens_++;
        return logits;
    }

    // Process a whole prompt with matrix-matrix matmuls instead of one token at
    // a time. Each weight row is then reused across the batch, which is the
    // difference between prefill being compute bound and paying the entire
    // weight stream once per token.
    // Only the final token's logits are ever needed, so the vocab projection
    // stays a single matvec rather than B of them.
    std::vector<float> prefill(const std::vector<uint32_t>& ids) {
        if (ids.empty()) throw std::runtime_error("inference: empty prompt");
        ensure_batch_buffers();
        std::vector<float> logits;
        size_t i = 0;
        while (i < ids.size()) {
            const int B = (int)std::min((size_t)prefill_chunk(), ids.size() - i);
            const bool last = (i + (size_t)B == ids.size());
            forward_batch(&ids[i], B, last ? &logits : nullptr);
            i += (size_t)B;
        }
        return logits;
    }

    // Clear the KV cache (start a new conversation).
    void reset() {
        n_tokens_ = 0;
        for (auto& c : k_cache_) c.clear();
        for (auto& c : v_cache_) c.clear();
    }

private:
    const gguf::GGUFModel* m_;
    backend::BackendPtr b_;
    QwenConfig cfg;
    int ratio_ = 1;
    int q_dim_ = 0;
    std::string out_name_;
    std::unordered_map<std::string, size_t> tindex_;

    std::vector<float> x_, h_, q_, kv_, v_, attn_;
    std::vector<float> gate_, up_, ffn_, scores_;
    std::vector<float> xb_, hb_, qb_, kb_, vb_, attnb_, gateb_, upb_, ffnb_;
    std::vector<std::vector<float>> k_cache_, v_cache_;
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

    // Attention across all q-heads. Each head reads its group's k/v cache and
    // writes only its own attn_ slice, so heads are independent. Dispatched
    // through the backend's pool rather than creating threads per token.
    void attend_heads(const float* kcache, const float* vcache) {
        b_->parallel_for(cfg.n_head, [&](int hq) {
            attend_head(hq, kcache, vcache);
        });
    }

    // Attention for one q-head. Reads q_ (head hq), the KV cache for its group,
    // writes only attn_[hq*head_dim ..]. Thread-safe (no shared writes).
    void attend_head(int hq, const float* kcache, const float* vcache) {
        int hk = hq / ratio_;
        const float* qh = q_.data() + hq * cfg.head_dim;
        int seq = n_tokens_ + 1;
        int stride = cfg.n_head_kv * cfg.head_dim;
        // Per-head scratch, preallocated to the context length. Heads run
        // concurrently, so each owns its own row and nothing is shared.
        float* scores = scores_.data() + (size_t)hq * (size_t)cfg.context_length;
        // softmax(QK^T / sqrt(head_dim)) -- without this the scores are
        // sqrt(head_dim)x too large and the softmax collapses to near one-hot.
        const float scale = 1.0f / std::sqrt((float)cfg.head_dim);
        float maxs = -1e30f;
        for (int t = 0; t < seq; t++) {
            const float* kt = kcache + (size_t)t * stride + (size_t)hk * cfg.head_dim;
            float s = 0.0f;
            for (int d = 0; d < cfg.head_dim; d++) s += qh[d] * kt[d];
            s *= scale;
            scores[t] = s;
            if (s > maxs) maxs = s;
        }
        float sum = 0.0f;
        for (int t = 0; t < seq; t++) { scores[t] = std::exp(scores[t] - maxs); sum += scores[t]; }
        float* oh = attn_.data() + hq * cfg.head_dim;
        std::fill(oh, oh + cfg.head_dim, 0.0f);
        for (int t = 0; t < seq; t++) {
            const float* vt = vcache + (size_t)t * stride + (size_t)hk * cfg.head_dim;
            float w = scores[t] / sum;
            for (int d = 0; d < cfg.head_dim; d++) oh[d] += w * vt[d];
        }
    }

    // Chunk size for batched prefill. Larger chunks mean fewer passes over the
    // weights, but the activation block (B * n_embd floats) has to stay in
    // cache or it is re-streamed for every weight row. LLMX_PREFILL_CHUNK
    // overrides it for A/B measurement.
    int prefill_chunk() const {
        static const int v = [] {
            const char* e = std::getenv("LLMX_PREFILL_CHUNK");
            int n = e ? std::atoi(e) : 0;
            return (n > 0) ? n : 128;
        }();
        return v;
    }

    void ensure_batch_buffers() {
        if (!xb_.empty()) return;
        const size_t B = (size_t)prefill_chunk();
        const size_t KV = (size_t)cfg.n_head_kv * cfg.head_dim;
        xb_.assign(B * cfg.n_embd, 0.0f);
        hb_.assign(B * cfg.n_embd, 0.0f);
        qb_.assign(B * (size_t)q_dim_, 0.0f);
        kb_.assign(B * KV, 0.0f);
        vb_.assign(B * KV, 0.0f);
        attnb_.assign(B * (size_t)q_dim_, 0.0f);
        gateb_.assign(B * cfg.n_ff, 0.0f);
        upb_.assign(B * cfg.n_ff, 0.0f);
        ffnb_.assign(B * cfg.n_ff, 0.0f);
    }

    void forward_batch(const uint32_t* ids, int B, std::vector<float>* out_logits) {
        const int pos0 = n_tokens_;
        if (pos0 + B > cfg.context_length)
            throw std::runtime_error("inference: context length exceeded (" +
                                     std::to_string(cfg.context_length) + " tokens)");
        const int E = cfg.n_embd, HD = cfg.head_dim, half = HD / 2;
        const size_t KV = (size_t)cfg.n_head_kv * HD;

        for (int b = 0; b < B; b++)
            dequant_row(tensor("token_embd.weight"), ids[b], xb_.data() + (size_t)b * E);

        for (int l = 0; l < cfg.n_layer; l++) {
            const std::string pre = "blk." + std::to_string(l) + ".";

            const float* anorm = (const float*)tensor_data(pre + "attn_norm.weight");
            for (int b = 0; b < B; b++)
                b_->rms_norm(hb_.data() + (size_t)b * E, xb_.data() + (size_t)b * E, anorm, E, cfg.rms_eps);

            matmul(tensor(pre + "attn_q.weight"), hb_.data(), qb_.data(), E, (size_t)q_dim_, B);
            matmul(tensor(pre + "attn_k.weight"), hb_.data(), kb_.data(), E, KV, B);
            matmul(tensor(pre + "attn_v.weight"), hb_.data(), vb_.data(), E, KV, B);

            const float* qn = (const float*)tensor_data(pre + "attn_q_norm.weight");
            const float* kn = (const float*)tensor_data(pre + "attn_k_norm.weight");
            for (int b = 0; b < B; b++) {
                float* q = qb_.data() + (size_t)b * q_dim_;
                float* k = kb_.data() + (size_t)b * KV;
                for (int h = 0; h < cfg.n_head; h++)
                    b_->rms_norm(q + h * HD, q + h * HD, qn, HD, cfg.rms_eps);
                for (int h = 0; h < cfg.n_head_kv; h++)
                    b_->rms_norm(k + h * HD, k + h * HD, kn, HD, cfg.rms_eps);
                const float* cs = rope_cos_.data() + (size_t)(pos0 + b) * half;
                const float* sn = rope_sin_.data() + (size_t)(pos0 + b) * half;
                for (int h = 0; h < cfg.n_head; h++)    b_->rope(q + h * HD, cs, sn, half);
                for (int h = 0; h < cfg.n_head_kv; h++) b_->rope(k + h * HD, cs, sn, half);
            }

            k_cache_[l].insert(k_cache_[l].end(), kb_.begin(), kb_.begin() + (size_t)B * KV);
            v_cache_[l].insert(v_cache_[l].end(), vb_.begin(), vb_.begin() + (size_t)B * KV);

            const float* kc = k_cache_[l].data();
            const float* vc = v_cache_[l].data();
            b_->parallel_for(cfg.n_head, [&](int hq) {
                for (int b = 0; b < B; b++) attend_head_batch(hq, b, pos0 + b, kc, vc);
            });

            matmul(tensor(pre + "attn_output.weight"), attnb_.data(), hb_.data(), (size_t)q_dim_, E, B);
            for (size_t j = 0; j < (size_t)B * E; j++) xb_[j] += hb_[j];

            const float* fnorm = (const float*)tensor_data(pre + "ffn_norm.weight");
            for (int b = 0; b < B; b++)
                b_->rms_norm(hb_.data() + (size_t)b * E, xb_.data() + (size_t)b * E, fnorm, E, cfg.rms_eps);

            matmul(tensor(pre + "ffn_gate.weight"), hb_.data(), gateb_.data(), E, cfg.n_ff, B);
            matmul(tensor(pre + "ffn_up.weight"),   hb_.data(), upb_.data(),   E, cfg.n_ff, B);
            for (size_t j = 0; j < (size_t)B * cfg.n_ff; j++) {
                const float g = gateb_[j] / (1.0f + std::exp(-gateb_[j])); // SiLU
                ffnb_[j] = g * upb_[j];
            }
            matmul(tensor(pre + "ffn_down.weight"), ffnb_.data(), hb_.data(), cfg.n_ff, E, B);
            for (size_t j = 0; j < (size_t)B * E; j++) xb_[j] += hb_[j];
        }

        n_tokens_ += B;

        if (out_logits) {
            b_->rms_norm(h_.data(), xb_.data() + (size_t)(B - 1) * E,
                         (const float*)tensor_data("output_norm.weight"), E, cfg.rms_eps);
            const gguf::TensorInfo& ot = tensor(out_name_);
            const size_t n_vocab = (size_t)ot.ne[1];
            out_logits->assign(n_vocab, 0.0f);
            matvec(ot, h_.data(), out_logits->data(), (size_t)E, n_vocab);
        }
    }

    // Attention for one q-head of one batch row, causal over [0, pos].
    void attend_head_batch(int hq, int b, int pos, const float* kc, const float* vc) {
        const int hk = hq / ratio_, HD = cfg.head_dim;
        const int seq = pos + 1;
        const int stride = cfg.n_head_kv * HD;
        const float* qh = qb_.data() + (size_t)b * q_dim_ + (size_t)hq * HD;
        float* scores = scores_.data() + (size_t)hq * (size_t)cfg.context_length;
        const float scale = 1.0f / std::sqrt((float)HD);
        float maxs = -1e30f;
        for (int t = 0; t < seq; t++) {
            const float* kt = kc + (size_t)t * stride + (size_t)hk * HD;
            float s = 0.0f;
            for (int d = 0; d < HD; d++) s += qh[d] * kt[d];
            s *= scale;
            scores[t] = s;
            if (s > maxs) maxs = s;
        }
        float sum = 0.0f;
        for (int t = 0; t < seq; t++) { scores[t] = std::exp(scores[t] - maxs); sum += scores[t]; }
        float* oh = attnb_.data() + (size_t)b * q_dim_ + (size_t)hq * HD;
        std::fill(oh, oh + HD, 0.0f);
        for (int t = 0; t < seq; t++) {
            const float* vt = vc + (size_t)t * stride + (size_t)hk * HD;
            const float w = scores[t] / sum;
            for (int d = 0; d < HD; d++) oh[d] += w * vt[d];
        }
    }

    // Batched form of matvec. Q8_0 uses the backend's batched kernel; other
    // types fall back to one matvec per row, correct but with no reuse gain.
    void matmul(const gguf::TensorInfo& t, const float* X, float* Y,
                size_t nin, size_t nout, int nbatch) {
        if (t.type == gguf::GGML_TYPE_Q8_0) {
            b_->matmul_q8_0(m_->tensor_data(tindex_.at(t.name)), X, Y,
                            nin / gguf::Q8_0_BLOCK, nout, (size_t)nbatch);
            return;
        }
        for (int b = 0; b < nbatch; b++)
            matvec(t, X + (size_t)b * nin, Y + (size_t)b * nout, nin, nout);
    }

    // Dequantize row `r` of a quantized matrix (nin fastest) into `out`,
    // dispatching on the tensor's type via the quant registry.
    void dequant_row(const gguf::TensorInfo& t, size_t r, float* out) const {
        size_t nin = (size_t)t.ne[0];
        const quant::QuantType* qt = quant::Registry::instance().get(t.type);
        if (!qt || !qt->dequantize) throw std::runtime_error("unsupported tensor type in dequant_row");
        const uint8_t* base = m_->tensor_data(tindex_.at(t.name)) +
            r * (nin / qt->block_size) * qt->type_size;
        qt->dequantize(base, out, nin / qt->block_size);
    }

    // out = W^T x for quantized W [nin, nout] (row o at o*nin/blk*tsz).
    // Q8_0 uses the backend's fused AVX2 matvec; other types use a correct
    // generic path: dequantize each row to f32 then f32 dot. The generic path
    // is slow-but-correct; a fused kernel per type is a follow-up.
    void matvec(const gguf::TensorInfo& t, const float* x, float* out, size_t nin, size_t nout) {
        const uint8_t* data = m_->tensor_data(tindex_.at(t.name));
        if (t.type == gguf::GGML_TYPE_Q8_0) {
            size_t nblocks = nin / gguf::Q8_0_BLOCK;
            b_->matvec_q8_0(data, x, out, nblocks, nout);
            return;
        }
        const quant::QuantType* qt = quant::Registry::instance().get(t.type);
        if (!qt || !qt->dequantize) throw std::runtime_error("unsupported tensor type in matvec");
        std::vector<float> row(nin);
        for (size_t o = 0; o < nout; o++) {
            qt->dequantize(data + o * (nin / qt->block_size) * qt->type_size,
                           row.data(), nin / qt->block_size);
            float acc = 0.0f;
            for (size_t i = 0; i < nin; i++) acc += row[i] * x[i];
            out[o] = acc;
        }
    }
};

} // namespace infer
