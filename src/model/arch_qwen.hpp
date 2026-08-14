#pragma once
#include <cstdint>
#include <cstring>
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

        for (size_t i = 0; i < m.tensors.size(); i++) tindex_[m.tensors[i].name] = i;

        // buffers
        x_.assign(cfg.n_embd, 0.0f);
        h_.assign(cfg.n_embd, 0.0f);
        q_.assign(cfg.n_embd, 0.0f);
        kv_.assign((size_t)cfg.n_head_kv * cfg.head_dim, 0.0f);
        v_.assign((size_t)cfg.n_head_kv * cfg.head_dim, 0.0f);
        attn_.assign(cfg.n_embd, 0.0f);
        row_.assign(std::max(cfg.n_embd, cfg.n_ff), 0.0f);

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

        // embedding
        dequant_row(tensor("token_embd.weight"), token_id, x_.data());

        for (int l = 0; l < cfg.n_layer; l++) {
            const std::string pre = "blk." + std::to_string(l) + ".";

            // attn norm
            b_->rms_norm(h_.data(), x_.data(),
                         (const float*)tensor_data(pre + "attn_norm.weight"),
                         cfg.n_embd, cfg.rms_eps);

            // q,k,v projections
            matvec(tensor(pre + "attn_q.weight"), h_.data(), q_.data(), cfg.n_embd, cfg.n_embd);
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
            matvec(tensor(pre + "attn_output.weight"), attn_.data(), h_.data(), cfg.n_embd, cfg.n_embd);
            for (int i = 0; i < cfg.n_embd; i++) x_[i] += h_[i];

            // ffn norm
            b_->rms_norm(h_.data(), x_.data(),
                         (const float*)tensor_data(pre + "ffn_norm.weight"),
                         cfg.n_embd, cfg.rms_eps);

            // gate/up (SwiGLU)
            std::vector<float> gate(cfg.n_ff), up(cfg.n_ff), ffn(cfg.n_ff);
            matvec(tensor(pre + "ffn_gate.weight"), h_.data(), gate.data(), cfg.n_embd, cfg.n_ff);
            matvec(tensor(pre + "ffn_up.weight"),   h_.data(), up.data(),   cfg.n_embd, cfg.n_ff);
            for (int i = 0; i < cfg.n_ff; i++) {
                float g = gate[i] / (1.0f + std::exp(-gate[i])); // SiLU
                ffn[i] = g * up[i];
            }
            // down projection + residual
            std::fill(h_.begin(), h_.end(), 0.0f);
            matvec(tensor(pre + "ffn_down.weight"), ffn.data(), h_.data(), cfg.n_ff, cfg.n_embd);
            for (int i = 0; i < cfg.n_embd; i++) x_[i] += h_[i];
        }

        // final norm + output projection
        b_->rms_norm(h_.data(), x_.data(),
                     (const float*)tensor_data("output_norm.weight"),
                     cfg.n_embd, cfg.rms_eps);
        size_t n_vocab = tensor("output.weight").ne[1];
        std::vector<float> logits(n_vocab);
        matvec(tensor("output.weight"), h_.data(), logits.data(), cfg.n_embd, n_vocab);

        n_tokens_++;
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
    std::unordered_map<std::string, size_t> tindex_;

    std::vector<float> x_, h_, q_, kv_, v_, attn_, row_;
    std::vector<std::vector<float>> k_cache_, v_cache_;
    std::vector<float> rope_cos_, rope_sin_;
    int n_tokens_ = 0;

    const gguf::TensorInfo& tensor(const std::string& name) const {
        auto it = tindex_.find(name);
        if (it == tindex_.end()) throw std::runtime_error("inference: missing tensor " + name);
        return m_->tensors[it->second];
    }
    const uint8_t* tensor_data(const std::string& name) const {
        return m_->data[tindex_.at(name)].data();
    }

    // Attention across all q-heads, parallelized over heads. Each head reads
    // its group's k/v cache and writes only its own attn_ slice (no sharing).
    void attend_heads(const float* kcache, const float* vcache) {
        int nh = cfg.n_head;
        int nt = (int)b_->threads_available();
        if (nt > nh) nt = nh;
        if (nt <= 1) {
            for (int hq = 0; hq < nh; hq++)
                attend_head(hq, kcache, vcache);
        } else {
            std::vector<std::thread> workers;
            workers.reserve((size_t)nt);
            int chunk = (nh + nt - 1) / nt;
            for (int w = 0; w < nt; w++) {
                int start = w * chunk;
                int end = std::min(nh, start + chunk);
                if (start >= end) break;
                workers.emplace_back([&, start, end, kcache, vcache]() {
                    for (int hq = start; hq < end; hq++)
                        attend_head(hq, kcache, vcache);
                });
            }
            for (auto& th : workers) th.join();
        }
    }

    // Attention for one q-head. Reads q_ (head hq), the KV cache for its group,
    // writes only attn_[hq*head_dim ..]. Thread-safe (no shared writes).
    void attend_head(int hq, const float* kcache, const float* vcache) {
        int hk = hq / ratio_;
        const float* qh = q_.data() + hq * cfg.head_dim;
        int seq = n_tokens_ + 1;
        int stride = cfg.n_head_kv * cfg.head_dim;
        std::vector<float> scores(seq);
        float maxs = -1e30f;
        for (int t = 0; t < seq; t++) {
            const float* kt = kcache + (size_t)t * stride + (size_t)hk * cfg.head_dim;
            float s = 0.0f;
            for (int d = 0; d < cfg.head_dim; d++) s += qh[d] * kt[d];
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

    // Dequantize row `r` of a quantized matrix (nin fastest) into `out`,
    // dispatching on the tensor's type via the quant registry.
    void dequant_row(const gguf::TensorInfo& t, size_t r, float* out) const {
        size_t nin = (size_t)t.ne[0];
        const quant::QuantType* qt = quant::Registry::instance().get(t.type);
        if (!qt || !qt->dequantize) throw std::runtime_error("unsupported tensor type in dequant_row");
        const uint8_t* base = m_->data[tindex_.at(t.name)].data() +
            r * (nin / qt->block_size) * qt->type_size;
        qt->dequantize(base, out, nin / qt->block_size);
    }

    // out = W^T x for quantized W [nin, nout] (row o at o*nin/blk*tsz).
    // Q8_0 uses the backend's fused AVX2 matvec; other types use a correct
    // generic path: dequantize each row to f32 then f32 dot. The generic path
    // is slow-but-correct; a fused kernel per type is a follow-up.
    void matvec(const gguf::TensorInfo& t, const float* x, float* out, size_t nin, size_t nout) {
        const uint8_t* data = m_->data[tindex_.at(t.name)].data();
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
