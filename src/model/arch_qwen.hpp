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
#include "model/host_kv_cache.hpp"
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
        validate_tensors();

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

        cache_ = HostKVCache(cfg.n_layer, cfg.n_head_kv, cfg.head_dim, cfg.context_length);

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
    int threads_available() const { return b_->threads_available(); }
    // 0 keeps the default. Changing it invalidates the scratch buffers.
    void set_ubatch(int n) { if (n > 0 && n != ubatch_) { ubatch_ = n; xb_.clear(); } }

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

        cache_.reserve((size_t)pos + 1, (size_t)n_tokens_);

        // embedding
        dequant_row(tensor("token_embd.weight"), token_id, x_.data());

        for (int l = 0; l < cfg.n_layer; l++) {
            const std::string pre = "blk." + std::to_string(l) + ".";

            // attn norm
            b_->rms_norm(h_.data(), x_.data(),
                         (const float*)tensor_data(pre + "attn_norm.weight"),
                         cfg.n_embd, cfg.rms_eps);

            // q,k,v projections
            b_->matmul_group({projection(pre + "attn_q.weight", q_.data(), size_t(q_dim_)),
                              projection(pre + "attn_k.weight", kv_.data(), size_t(cfg.n_head_kv * cfg.head_dim)),
                              projection(pre + "attn_v.weight", v_.data(), size_t(cfg.n_head_kv * cfg.head_dim))},
                             h_.data(), cfg.n_embd, 1);

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
            cache_.write(l, kv_.data(), v_.data(), (size_t)pos, 1);

            // attention: each q-head writes only its own attn_ slice, so heads
            // can be processed in parallel.
            const float* kcache = cache_.keys(l);
            const float* vcache = cache_.values(l);
            b_->attention(q_.data(), kcache, vcache, attn_.data(),
                          cfg.n_head, cfg.n_head_kv, cfg.head_dim, pos, 1, cache_.head_stride());

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
            b_->matmul_group({projection(pre + "ffn_gate.weight", gate_.data(), cfg.n_ff),
                              projection(pre + "ffn_up.weight", up_.data(), cfg.n_ff)},
                             h_.data(), cfg.n_embd, 1);
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
        std::vector<float> logits;
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
        b_->run_prefill(std::ref(work));
        return logits;
    }

    // Clear the KV cache (start a new conversation).
    void reset() {
        n_tokens_ = 0;

    }

private:
    const gguf::GGUFModel* m_;
    backend::BackendPtr b_;
    QwenConfig cfg;
    int q_dim_ = 0;
    int ubatch_ = 512;   // default matches llama.cpp
    std::string out_name_;
    std::unordered_map<std::string, size_t> tindex_;

    std::vector<float> x_, h_, q_, kv_, v_, attn_;
    std::vector<float> gate_, up_, ffn_;
    std::vector<float> xb_, hb_, qb_, kb_, vb_, attnb_, gateb_, upb_, ffnb_;
    HostKVCache cache_;
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

    void validate_tensors() const {
        const auto& embedding = tensor("token_embd.weight");
        if (embedding.ne.size() < 2 || !embedding.ne[1] ||
            embedding.ne[1] > uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("inference: invalid vocabulary dimension");
        const uint64_t vocab = embedding.ne[1];
        auto check = [&](const std::string& name, uint64_t input, uint64_t output, bool norm = false) {
            const auto& t = tensor(name);
            bool valid = !t.ne.empty() && t.ne[0] == input;
            if (norm) {
                valid = valid && t.type == gguf::GGML_TYPE_F32;
            } else {
                valid = valid && t.ne.size() >= 2 && t.ne[1] == output;
            }
            for (size_t d = norm ? 1 : 2; d < t.ne.size(); ++d) valid = valid && t.ne[d] == 1;
            if (!valid) throw std::runtime_error("inference: incompatible tensor layout " + name);
        };
        check("token_embd.weight", cfg.n_embd, vocab);
        check(out_name_, cfg.n_embd, vocab);
        check("output_norm.weight", cfg.n_embd, 1, true);
        const uint64_t kv_width = uint64_t(cfg.n_head_kv) * cfg.head_dim;
        for (int l = 0; l < cfg.n_layer; ++l) {
            const std::string pre = "blk." + std::to_string(l) + ".";
            check(pre + "attn_norm.weight", cfg.n_embd, 1, true);
            check(pre + "attn_q_norm.weight", cfg.head_dim, 1, true);
            check(pre + "attn_k_norm.weight", cfg.head_dim, 1, true);
            check(pre + "attn_q.weight", cfg.n_embd, q_dim_);
            check(pre + "attn_k.weight", cfg.n_embd, kv_width);
            check(pre + "attn_v.weight", cfg.n_embd, kv_width);
            check(pre + "attn_output.weight", q_dim_, cfg.n_embd);
            check(pre + "ffn_norm.weight", cfg.n_embd, 1, true);
            check(pre + "ffn_gate.weight", cfg.n_embd, cfg.n_ff);
            check(pre + "ffn_up.weight", cfg.n_embd, cfg.n_ff);
            check(pre + "ffn_down.weight", cfg.n_ff, cfg.n_embd);
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
    void ensure_batch_buffers(size_t want) {
        const size_t B = std::min((size_t)ubatch(), std::max<size_t>(want, 1));
        if (xb_.size() >= B * (size_t)cfg.n_embd) return;
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
        cache_.reserve((size_t)pos0 + (size_t)B, (size_t)n_tokens_);
        const int E = cfg.n_embd, HD = cfg.head_dim, half = HD / 2;
        const size_t KV = (size_t)cfg.n_head_kv * HD;
        const int nt = b_->threads_available();
        // Avoid dispatching short elementwise stages when there are fewer
        // than two rows per worker; small-batch timing regressed without this.
        const auto for_rows = [&](const auto& fn) {
            if (nt <= 1 || B / nt < 2) {
                for (int b = 0; b < B; ++b) fn(b);
            } else {
                b_->parallel_for(B, fn);
            }
        };

        for (int b = 0; b < B; b++)
            dequant_row(tensor("token_embd.weight"), ids[b], xb_.data() + (size_t)b * E);

        for (int l = 0; l < cfg.n_layer; l++) {
            const std::string pre = "blk." + std::to_string(l) + ".";

            const float* anorm = (const float*)tensor_data(pre + "attn_norm.weight");
            for_rows([&](int b) {
                b_->rms_norm(hb_.data() + (size_t)b * E, xb_.data() + (size_t)b * E, anorm, E, cfg.rms_eps);
            });

            b_->matmul_group({projection(pre + "attn_q.weight", qb_.data(), size_t(q_dim_)),
                              projection(pre + "attn_k.weight", kb_.data(), KV),
                              projection(pre + "attn_v.weight", vb_.data(), KV)}, hb_.data(), E, B);

            const float* qn = (const float*)tensor_data(pre + "attn_q_norm.weight");
            const float* kn = (const float*)tensor_data(pre + "attn_k_norm.weight");
            for_rows([&](int b) {
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
            });

            cache_.write(l, kb_.data(), vb_.data(), (size_t)pos0, (size_t)B);

            const float* kc = cache_.keys(l);
            const float* vc = cache_.values(l);
            b_->attention(qb_.data(), kc, vc, attnb_.data(),
                          cfg.n_head, cfg.n_head_kv, cfg.head_dim, pos0, B, cache_.head_stride());

            matmul(tensor(pre + "attn_output.weight"), attnb_.data(), hb_.data(), (size_t)q_dim_, E, B);
            for (size_t j = 0; j < (size_t)B * E; j++) xb_[j] += hb_[j];

            const float* fnorm = (const float*)tensor_data(pre + "ffn_norm.weight");
            for_rows([&](int b) {
                b_->rms_norm(hb_.data() + (size_t)b * E, xb_.data() + (size_t)b * E, fnorm, E, cfg.rms_eps);
            });

            b_->matmul_group({projection(pre + "ffn_gate.weight", gateb_.data(), cfg.n_ff),
                              projection(pre + "ffn_up.weight", upb_.data(), cfg.n_ff)}, hb_.data(), E, B);
            for_rows([&](int b) {
                const size_t end = (size_t)(b + 1) * cfg.n_ff;
                for (size_t j = (size_t)b * cfg.n_ff; j < end; j++) {
                    const float g = gateb_[j] / (1.0f + std::exp(-gateb_[j]));
                    ffnb_[j] = g * upb_[j];
                }
            });
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

    backend::Projection projection(const std::string& name, float* out, size_t rows) const {
        const auto& t = tensor(name);
        return {t.type, m_->tensor_data(tindex_.at(t.name)), out, rows};
    }

    // Batched matmul. The backend dispatches on the quant type, so every
    // block format takes the same path; there is no per-type branch here.
    void matmul(const gguf::TensorInfo& t, const float* X, float* Y,
                size_t nin, size_t nout, int nbatch) {
        b_->matmul(t.type, m_->tensor_data(tindex_.at(t.name)), X, Y,
                   nin, nout, (size_t)nbatch);
    }

    // Dequantize row `r` of a quantized matrix (nin fastest) into `out`,
    // dispatching on the tensor's type via the quant registry.
    void dequant_row(const gguf::TensorInfo& t, size_t r, float* out) const {
        size_t nin = (size_t)t.ne[0];
        if (t.type == gguf::GGML_TYPE_F32) {
            std::memcpy(out, m_->tensor_data(tindex_.at(t.name)) + r * nin * sizeof(float), nin * sizeof(float));
            return;
        }
        const quant::QuantType* qt = quant::Registry::instance().get(t.type);
        if (!qt || !qt->dequantize) throw std::runtime_error("unsupported tensor type in dequant_row");
        const uint8_t* base = m_->tensor_data(tindex_.at(t.name)) +
            r * (nin / qt->block_size) * qt->type_size;
        qt->dequantize(base, out, nin / qt->block_size);
    }

    // out = W^T x for a single column. Same backend entry point as the
    // batched form, so there is one dispatch path and one place that knows
    // about quant types.
    void matvec(const gguf::TensorInfo& t, const float* x, float* out,
                size_t nin, size_t nout) {
        b_->matmul(t.type, m_->tensor_data(tindex_.at(t.name)), x, out, nin, nout, 1);
    }

};

} // namespace infer
