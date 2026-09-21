#pragma once
#include <algorithm>
#include <memory>
#include <functional>
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

// Where each tensor role runs, as an index into the model's backends. Per
// role rather than per layer, so a layer's attention and its feed-forward
// block can sit on different devices; that is what expert offload needs
// later (docs/EXECUTION.md). Empty means everything on device 0.
struct Placement {
    std::vector<int> attn_device, ffn_device;
    int embed_device = 0, output_device = 0;
};

// One request's history in a model's cache: a block table per storage and
// the committed length, and per device the ticket of the last pass that
// touched it, which is what a release waits on rather than draining the
// device (docs/EXECUTION.md). Made by Model::make_sequence so it is bound to
// that model's pools and block sizes. Movable, not copyable; the server
// keeps one per request.
class Sequence {
public:
    Sequence() = default;
    size_t length() const { return kv_.empty() ? 0 : kv_[0].length(); }
private:
    friend class Model;
    std::vector<KVSequence> kv_;
    std::vector<backend::Ticket> last_;
    const Model* owner_ = nullptr;
};

// One pass in flight: an activation arena per device, the host-visible
// logits rows on the output device, the staging vector a crossing goes
// through, and the tickets of its submissions. Storage is allocated by the
// first forward that needs it and grows to the largest pass seen. Two
// contexts are what let a scheduler keep one pass on the device while it
// reads another's logits; the CLI has one. Plain data that Model fills.
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
    struct Scratch {
        backend::BufferPtr arena;
        size_t rows = 0;
        size_t offset[kSlots] = {0};
    };
    std::vector<Scratch> scratch;              // per device
    backend::BufferPtr logits_buf;
    size_t logit_rows = 0;
    std::vector<uint32_t> ids, pos, pick;
    std::vector<std::vector<backend::KVView>> views;   // per storage, per entry
    std::vector<backend::Ticket> tickets;      // per device
    std::vector<float> staging;
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
    // Construct the model over a GGUF model on one backend (defaults to the
    // CPU backend). The model owns a reference to the model data, which must
    // outlive the Model.
    explicit Model(const gguf::GGUFModel& m,
                   backend::BackendPtr backend = backend::make_cpu_backend())
        : Model(m, std::vector<backend::BackendPtr>{std::move(backend)}, Placement{}) {}

    // Construct over several backends with a placement of every role.
    Model(const gguf::GGUFModel& m, std::vector<backend::BackendPtr> backends,
          Placement placement)
        : m_(&m), place_(std::move(placement)) {
        if (backends.empty()) throw std::runtime_error("inference: missing backend");
        for (const auto& b : backends)
            if (!b) throw std::runtime_error("inference: missing backend");
        quant::register_builtins(); // populate the quant registry (idempotent)
        cfg = load_config(m);
        // The attention projection width is n_head*head_dim, which only equals
        // n_embd by coincidence on some models (Qwen3-8B: 32*128 == 4096).
        // Qwen3-0.6B/1.7B/4B have head_dim 128 with a smaller n_embd.
        q_dim_ = cfg.n_head * cfg.head_dim;

        if (place_.attn_device.empty() && place_.ffn_device.empty()) {
            place_.attn_device.assign((size_t)cfg.n_layer, 0);
            place_.ffn_device.assign((size_t)cfg.n_layer, 0);
        }
        if (place_.attn_device.size() != (size_t)cfg.n_layer ||
            place_.ffn_device.size() != (size_t)cfg.n_layer)
            throw std::runtime_error("inference: placement does not cover every layer");
        auto device_index = [&](int d) {
            if (d < 0 || (size_t)d >= backends.size())
                throw std::runtime_error("inference: placement names a device the model does not have");
            return (size_t)d;
        };
        devices_.reserve(backends.size());
        for (auto& b : backends) {
            devices_.push_back(std::make_unique<Device>());
            devices_.back()->b = std::move(b);
            devices_.back()->local_layer.assign((size_t)cfg.n_layer, -1);
        }
        device_index(place_.embed_device);
        device_index(place_.output_device);
        devices_[(size_t)place_.embed_device]->used = true;
        devices_[(size_t)place_.output_device]->used = true;
        for (int l = 0; l < cfg.n_layer; ++l) {
            Device& a = *devices_[device_index(place_.attn_device[(size_t)l])];
            a.local_layer[(size_t)l] = a.attn_layers++;
            a.used = true;
            devices_[device_index(place_.ffn_device[(size_t)l])]->used = true;
        }

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

        // Each device that runs attention gets a storage for exactly its
        // layers, with its own block size and pool. Budget: the whole
        // context; storage is backed on demand, so a short chat does not
        // allocate it.
        for (auto& dp : devices_) {
            Device& d = *dp;
            if (!d.attn_layers) continue;
            d.storage = d.b->kv_alloc((size_t)d.attn_layers, cfg.n_head_kv, cfg.head_dim,
                                      (size_t)cfg.context_length);
            d.pool.configure(d.storage->max_blocks());
            d.storage_index = (int)storages_.size();
            storages_.push_back(&d);
        }
        seq_ = make_sequence();

        // Precompute the RoPE cos/sin table for every position up to the
        // context length. Indexed as [pos*(head_dim/2) + i]. Every device
        // that runs attention reads it through an adopted buffer, so the
        // host vectors stay alive for the model's lifetime; on CPU that is
        // the same memory.
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
        for (Device* d : storages_) {
            d->rope_cos = d->b->adopt(rope_cos_.data(), rope_cos_.size() * sizeof(float));
            d->rope_sin = d->b->adopt(rope_sin_.data(), rope_sin_.size() * sizeof(float));
        }
    }

    // Sequences hold the pools' addresses; moving the model would leave them
    // pointing at the old ones. Nothing moves a Model today.
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // CPU worker counts, applied to every backend; a device backend ignores
    // them. The count reported is device 0's, which is the host when a
    // model spans a CPU and a device.
    void set_threads(int n) { for (auto& d : devices_) d->b->set_threads(n); }
    int threads_available() const { return devices_[0]->b->threads_available(); }
    // 0 keeps the default. Sets how a prompt is chunked; storage follows the
    // passes actually run.
    void set_ubatch(int n) { if (n > 0) ubatch_ = n; }

    int n_tokens() const { return (int)seq_.length(); }
    int head_dim() const { return cfg.head_dim; }
    int context_length() const { return cfg.context_length; }
    const Placement& placement() const { return place_; }

    // A second history with the same committed tokens as `src`, sharing
    // every full block and copying the partial tail on each storage. The
    // fork inherits the tickets of the passes that wrote what it shares.
    // Shared blocks are read-only from now on: a sequence truncated into one
    // cannot append and has to be forked instead.
    Sequence fork(const Sequence& src) {
        if (src.owner_ != this) throw std::runtime_error("inference: sequence of another model");
        Sequence f;
        f.kv_.reserve(storages_.size());
        for (size_t s = 0; s < storages_.size(); ++s) {
            KVSequence::Tail tail;
            f.kv_.push_back(src.kv_[s].fork(tail));
            if (tail.to >= 0)
                storages_[s]->b->kv_copy(*storages_[s]->storage, tail.from, tail.to);
        }
        f.last_ = src.last_;
        f.owner_ = this;
        return f;
    }

    // A fresh history over this model's cache: one table per storage.
    Sequence make_sequence() {
        Sequence s;
        s.kv_.reserve(storages_.size());
        for (Device* d : storages_)
            s.kv_.emplace_back(&d->pool, d->b->kv_layout().block_tokens);
        s.last_.assign(devices_.size(), 0);
        s.owner_ = this;
        return s;
    }

    // One pass over every entry: each sequence's tokens go through the graph
    // at their own positions and attend through their own history, and the
    // logits after the last token of every entry that wants them land in
    // the context, in entry order. The residual stream crosses to another
    // device wherever the placement changes, through the context's staging
    // vector. The pass is one submission per device. It is one transaction
    // as well: every sequence commits its tokens only once the pass is
    // submitted, and a failure before that leaves every history as it was.
    // The context reads the logits after waiting on the output device.
    void forward(ExecContext& ctx, const BatchEntry* entries, size_t n_entries) {
        if (!entries || !n_entries) throw std::runtime_error("inference: empty batch");
        size_t rows = 0, want = 0;
        for (size_t e = 0; e < n_entries; ++e) {
            const BatchEntry& en = entries[e];
            if (!en.seq || en.seq->owner_ != this)
                throw std::runtime_error("inference: batch entry without a sequence of this model");
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
        ctx.views.resize(storages_.size());
        for (auto& v : ctx.views) v.resize(n_entries);

        // Blocks are taken on every storage for every entry before anything
        // runs. A sequence listed twice fails here, since its second prepare
        // finds the first still pending.
        size_t prepared = 0;
        try {
            for (; prepared < n_entries * storages_.size(); ++prepared)
                entries[prepared / storages_.size()].seq->kv_[prepared % storages_.size()]
                    .prepare(entries[prepared / storages_.size()].n);
        } catch (...) {
            for (size_t i = 0; i < prepared; ++i)
                entries[i / storages_.size()].seq->kv_[i % storages_.size()].abort();
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
            for (size_t s = 0; s < storages_.size(); ++s)
                ctx.views[s][e] = en.seq->kv_[s].view(storages_[s]->storage.get());
            r += en.n;
            if (en.want_logits) ctx.pick[w++] = (uint32_t)(r - 1);
        }

        try {
            const size_t E = (size_t)cfg.n_embd;
            size_t cur = (size_t)place_.embed_device;
            devices_[cur]->b->embed(slot(ctx, cur, 0), token_embd_.type, token_embd_.slice(),
                                    token_embd_.nin, token_embd_.nout, ctx.ids.data(), rows);
            for (int l = 0; l < cfg.n_layer; l++) {
                const size_t a = (size_t)place_.attn_device[(size_t)l];
                if (a != cur) { cross(ctx, cur, a, rows * E); cur = a; }
                attention_half(ctx, cur, l, rows, n_entries);
                const size_t f = (size_t)place_.ffn_device[(size_t)l];
                if (f != cur) { cross(ctx, cur, f, rows * E); cur = f; }
                ffn_half(ctx, cur, l, rows);
            }
            const size_t o = (size_t)place_.output_device;
            if (o != cur) { cross(ctx, cur, o, rows * E); cur = o; }
            if (want) {
                // The rows that want logits are not contiguous once entries
                // mix, so they are compacted first and the head runs once
                // over exactly those rows.
                backend::Backend& b = *devices_[cur]->b;
                b.gather_rows(slot(ctx, cur, 1), slot(ctx, cur, 0), E, ctx.pick.data(), want);
                b.rms_norm_rows(slot(ctx, cur, 1), slot(ctx, cur, 1), output_norm_.slice(),
                                want, E, E, cfg.rms_eps);
                b.matmul(output_.type, output_.slice(), slot(ctx, cur, 1),
                         {ctx.logits_buf.get(), 0}, output_.nin, output_.nout, want);
            }
            for (size_t d = 0; d < devices_.size(); ++d)
                if (devices_[d]->used) ctx.tickets[d] = devices_[d]->b->submit();
        } catch (...) {
            retire();
            for (size_t e = 0; e < n_entries; ++e)
                for (auto& kv : entries[e].seq->kv_) kv.abort();
            throw;
        }
        ctx.n_logits = want;
        ctx.backend = devices_[(size_t)place_.output_device]->b.get();
        ctx.ticket = ctx.tickets[(size_t)place_.output_device];
        ctx.pending = want > 0;
        for (size_t e = 0; e < n_entries; ++e) {
            for (auto& kv : entries[e].seq->kv_) kv.commit();
            for (size_t d = 0; d < devices_.size(); ++d)
                if (devices_[d]->used) entries[e].seq->last_[d] = ctx.tickets[d];
        }
    }

    // Start a new history. Blocks return to every pool; their storage is
    // retained. Every pass ends in a submit or, on failure, a sync, so the
    // sequence's last tickets cover everything that could still be touching
    // a block: this waits for those and no more.
    void reset(Sequence& s) {
        if (s.owner_ != this)
            throw std::runtime_error("inference: sequence of another model");
        for (size_t d = 0; d < devices_.size(); ++d)
            if (devices_[d]->used) devices_[d]->b->wait(s.last_[d]);
        for (auto& kv : s.kv_) kv.reset();
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
            scoped(0, work);
        } catch (...) {
            retire();
            for (auto& kv : seq_.kv_) kv.truncate(start);
            throw;
        }
        return row(ctx_, 0);
    }

    void reset() { reset(seq_); }

    // Allocated is what the backends back; used is the committed history.
    // The gap is the paging cost in memory (docs/KV-CACHE.md).
    size_t kv_allocated_bytes() const {
        size_t n = 0;
        for (Device* d : storages_) n += d->storage->allocated_bytes();
        return n;
    }
    size_t kv_peak_bytes() const {
        size_t n = 0;
        for (Device* d : storages_) n += d->storage->peak_bytes();
        return n;
    }
    size_t kv_used_bytes() const {
        return seq_.length() * cfg.n_layer * 2 * cfg.n_head_kv * cfg.head_dim * sizeof(float);
    }

private:
    // One backend and what the placement put on it. A pool is not movable,
    // because sequences hold its address, so devices live behind pointers.
    struct Device {
        backend::BackendPtr b;
        bool used = false;
        int attn_layers = 0;
        int storage_index = -1;
        std::vector<int> local_layer;            // model layer -> layer in storage
        std::unique_ptr<backend::KVStorage> storage;
        BlockPool pool;
        backend::BufferPtr rope_cos, rope_sin;
    };

    const gguf::GGUFModel* m_;
    Placement place_;
    std::vector<std::unique_ptr<Device>> devices_;
    std::vector<Device*> storages_;              // the devices that run attention
    QwenConfig cfg;
    int q_dim_ = 0;
    int ubatch_ = 512;   // the conventional default
    std::string out_name_;
    std::unordered_map<std::string, size_t> tindex_;
    std::vector<LayerWeights> layers_;
    Weight token_embd_, output_norm_, output_;
    std::vector<float> rope_cos_, rope_sin_;
    Sequence seq_;
    ExecContext ctx_;

    const gguf::TensorInfo& tensor(const std::string& name) const {
        auto it = tindex_.find(name);
        if (it == tindex_.end()) throw std::runtime_error("inference: missing tensor " + name);
        return m_->tensors[it->second];
    }

    // Validate every tensor this architecture needs and resolve it to a
    // Weight in the same pass, so a resolved handle is well-formed by
    // construction and the forward pass never looks a tensor up by name.
    // Each weight is adopted by the backend that hosts its role.
    void resolve_tensors() {
        const auto& embedding = tensor("token_embd.weight");
        if (embedding.ne.size() < 2 || !embedding.ne[1] ||
            embedding.ne[1] > uint64_t(std::numeric_limits<int>::max()))
            throw std::runtime_error("inference: invalid vocabulary dimension");
        const uint64_t vocab = embedding.ne[1];
        auto check = [&](size_t device, const std::string& name, uint64_t input,
                         uint64_t output, bool norm = false) -> Weight {
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
            return Weight{t.type, devices_[device]->b->adopt(m_->tensor_data(i), m_->tensor_bytes(i)),
                          (size_t)input, (size_t)output};
        };
        const size_t ed = (size_t)place_.embed_device, od = (size_t)place_.output_device;
        token_embd_ = check(ed, "token_embd.weight", cfg.n_embd, vocab);
        output_ = check(od, out_name_, cfg.n_embd, vocab);
        output_norm_ = check(od, "output_norm.weight", cfg.n_embd, 1, true);
        const uint64_t kv_width = uint64_t(cfg.n_head_kv) * cfg.head_dim;
        layers_.resize(cfg.n_layer);
        for (int l = 0; l < cfg.n_layer; ++l) {
            const std::string pre = "blk." + std::to_string(l) + ".";
            const size_t a = (size_t)place_.attn_device[(size_t)l];
            const size_t f = (size_t)place_.ffn_device[(size_t)l];
            LayerWeights& w = layers_[l];
            w.attn_norm   = check(a, pre + "attn_norm.weight", cfg.n_embd, 1, true);
            w.attn_q_norm = check(a, pre + "attn_q_norm.weight", cfg.head_dim, 1, true);
            w.attn_k_norm = check(a, pre + "attn_k_norm.weight", cfg.head_dim, 1, true);
            w.attn_q      = check(a, pre + "attn_q.weight", cfg.n_embd, q_dim_);
            w.attn_k      = check(a, pre + "attn_k.weight", cfg.n_embd, kv_width);
            w.attn_v      = check(a, pre + "attn_v.weight", cfg.n_embd, kv_width);
            w.attn_output = check(a, pre + "attn_output.weight", q_dim_, cfg.n_embd);
            w.ffn_norm    = check(f, pre + "ffn_norm.weight", cfg.n_embd, 1, true);
            w.ffn_gate    = check(f, pre + "ffn_gate.weight", cfg.n_embd, cfg.n_ff);
            w.ffn_up      = check(f, pre + "ffn_up.weight", cfg.n_embd, cfg.n_ff);
            w.ffn_down    = check(f, pre + "ffn_down.weight", cfg.n_ff, cfg.n_embd);
        }
    }

    // Physical batch: how many tokens go through ONE forward pass of the
    // graph. This is the physical batch (-ub), not a logical one: it sets the GEMM
    // width and the scratch buffer sizes. llmx has no logical batch, since
    // there is one sequence and no queue; that distinction only starts to
    // matter with the multi-user server in ROADMAP #7.
    int ubatch() const { return ubatch_; }

    // The CPU prefill scope is per backend, so a prompt enters one on every
    // device it runs on, nested. A device backend's scope is the default
    // and just runs the body.
    void scoped(size_t d, const std::function<void()>& work) {
        while (d < devices_.size() && !devices_[d]->used) ++d;
        if (d >= devices_.size()) { work(); return; }
        devices_[d]->b->run_prefill([&] { scoped(d + 1, work); });
    }

    // One backend allocation holding the nine activations of a pass, each at
    // a 64-byte boundary so the AVX2 kernels see the alignment they saw when
    // every vector was its own allocation. Device allocators handle a few
    // large blocks far better than many small ones, and resizing is one
    // call. The caller only publishes the result once this returns, so an
    // allocation that throws leaves the previous arena intact.
    backend::BufferPtr alloc_arena(backend::Backend& b,
                                   const size_t (&counts)[ExecContext::kSlots],
                                   size_t (&offsets)[ExecContext::kSlots]) const {
        size_t total = 0;
        for (size_t i = 0; i < ExecContext::kSlots; ++i) {
            offsets[i] = total;
            const size_t bytes = counts[i] * sizeof(float);
            if (bytes / sizeof(float) != counts[i] || total > (size_t)-1 - bytes - 63)
                throw std::runtime_error("inference: activation arena size overflows");
            total = (total + bytes + 63) / 64 * 64;
        }
        return b.alloc(total);
    }

    // Storage for a pass of `rows` rows with `want` logits rows, on every
    // device the placement uses: grown when a pass needs more than the
    // context holds, never shrunk. Each is allocated whole before it
    // replaces what the context had.
    void ensure(ExecContext& ctx, size_t rows, size_t want) {
        auto mul = [](size_t a, size_t b) {
            if (b && a > (size_t)-1 / b)
                throw std::runtime_error("inference: activation arena size overflows");
            return a * b;
        };
        ctx.scratch.resize(devices_.size());
        ctx.tickets.resize(devices_.size(), 0);
        for (size_t d = 0; d < devices_.size(); ++d) {
            ExecContext::Scratch& sc = ctx.scratch[d];
            if (!devices_[d]->used || (sc.arena && sc.rows >= rows)) continue;
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
            backend::BufferPtr arena = alloc_arena(*devices_[d]->b, counts, offsets);
            sc.arena = std::move(arena);
            std::copy(offsets, offsets + ExecContext::kSlots, sc.offset);
            sc.rows = rows;
        }
        if (want && (!ctx.logits_buf || ctx.logit_rows < want)) {
            // The head writes here and the host reads it in place once the
            // pass has retired: the one point per pass that must be host
            // visible, and the one wait per pass.
            backend::BufferPtr logits = devices_[(size_t)place_.output_device]->b->alloc(
                mul(mul(want, output_.nout), sizeof(float)), backend::Memory::host_visible);
            ctx.logits_buf = std::move(logits);
            ctx.logit_rows = want;
            ctx.width = output_.nout;
        }
    }

    static backend::Slice slot(const ExecContext& ctx, size_t device, size_t i) {
        const ExecContext::Scratch& sc = ctx.scratch[device];
        return {sc.arena.get(), sc.offset[i] / sizeof(float)};
    }

    static std::vector<float> row(ExecContext& ctx, size_t i) {
        const float* p = ctx.logits(i);
        return std::vector<float>(p, p + ctx.width);
    }

    // The residual stream moves from one device's x slot to another's,
    // through host memory: a read, which waits for the source, then a
    // write, which is enqueued on the destination. Once per placement
    // boundary per pass; `n_embd * rows` floats, a few kilobytes on a
    // decode token.
    void cross(ExecContext& ctx, size_t from, size_t to, size_t floats) {
        const size_t bytes = floats * sizeof(float);
        ctx.staging.resize(floats);
        const backend::Slice src = slot(ctx, from, 0), dst = slot(ctx, to, 0);
        devices_[from]->b->read(*src.buffer, src.offset * sizeof(float), ctx.staging.data(), bytes);
        devices_[to]->b->write(*dst.buffer, dst.offset * sizeof(float), ctx.staging.data(), bytes);
    }

    // Slots: 0 x, 1 h, 2 q, 3 k, 4 v, 5 attn, 6 gate, 7 up, 8 ffn.
    void attention_half(ExecContext& ctx, size_t dev, int l, size_t rows, size_t n_views) {
        Device& d = *devices_[dev];
        backend::Backend& b = *d.b;
        const LayerWeights& w = layers_[(size_t)l];
        const size_t E = (size_t)cfg.n_embd, half = (size_t)cfg.head_dim / 2;
        const size_t KV = (size_t)cfg.n_head_kv * cfg.head_dim;
        const backend::Slice x = slot(ctx, dev, 0), h = slot(ctx, dev, 1), q = slot(ctx, dev, 2),
                             k = slot(ctx, dev, 3), v = slot(ctx, dev, 4), attn = slot(ctx, dev, 5);
        const size_t layer = (size_t)d.local_layer[(size_t)l];
        const std::vector<backend::KVView>& views = ctx.views[(size_t)d.storage_index];

        b.rms_norm_rows(h, x, w.attn_norm.slice(), rows, E, E, cfg.rms_eps);

        b.matmul_group({projection(w.attn_q, q),
                        projection(w.attn_k, k),
                        projection(w.attn_v, v)}, h, E, rows);

        const backend::Backend::RopeArgs rope{{d.rope_cos.get(), 0}, {d.rope_sin.get(), 0},
                                              half, ctx.pos.data(), cfg.rms_eps};
        b.norm_rope_kv(q, (size_t)q_dim_, cfg.n_head, w.attn_q_norm.slice(),
                       k, v, KV, cfg.n_head_kv, w.attn_k_norm.slice(),
                       rope, rows, layer, views.data(), n_views);
        b.attention(q, layer, views.data(), n_views, attn,
                    cfg.n_head, cfg.n_head_kv, cfg.head_dim);

        b.matmul_add(w.attn_output.type, w.attn_output.slice(), attn, x,
                     w.attn_output.nin, w.attn_output.nout, rows);
    }

    void ffn_half(ExecContext& ctx, size_t dev, int l, size_t rows) {
        backend::Backend& b = *devices_[dev]->b;
        const LayerWeights& w = layers_[(size_t)l];
        const size_t E = (size_t)cfg.n_embd;
        const backend::Slice x = slot(ctx, dev, 0), h = slot(ctx, dev, 1),
                             gate = slot(ctx, dev, 6), up = slot(ctx, dev, 7), ffn = slot(ctx, dev, 8);

        b.rms_norm_rows(h, x, w.ffn_norm.slice(), rows, E, E, cfg.rms_eps);

        b.matmul_group({projection(w.ffn_gate, gate),
                        projection(w.ffn_up, up)}, h, E, rows);
        b.silu_mul(ffn, gate, up, rows * (size_t)cfg.n_ff);
        b.matmul_add(w.ffn_down.type, w.ffn_down.slice(), ffn, x,
                     w.ffn_down.nin, w.ffn_down.nout, rows);
    }

    // A block returns to the pool only once the backend has retired every
    // submission that touched it (docs/KV-CACHE.md). The CPU backend is eager
    // so this costs nothing; on a device, releasing a block while a write to
    // it is still queued hands a later sequence someone else's history. The
    // callers are the exception paths, where a failed pass has ops queued
    // behind no ticket, so this drains every device rather than waiting;
    // reset() has tickets and waits on them. sync() cannot throw for the
    // same reason.
    void retire() noexcept {
        for (auto& d : devices_) if (d->used) d->b->sync();
    }

    // The buffer is passed by raw pointer, not by handle: three projections
    // per layer per token is nearly two hundred refcount pairs a token if a
    // shared pointer is copied here instead.
    static backend::Projection projection(const Weight& w, backend::Slice out) {
        return {w.type, {w.data.get(), 0}, out, w.nout};
    }
};

} // namespace infer
