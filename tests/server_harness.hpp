#pragma once
// What the scheduler's native tests share: the synthetic Q8_0 model and a hybrid one served over CPU backends or a device, requests and their replies, a run of waves through one scheduler, and replies compared bit for bit.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "backends/devices.hpp"
#include "model/runtime.hpp"
#include "model/place.hpp"
#include "model/arch/registry.hpp"
#include "quant/quant.hpp"
#include "server/scheduler.hpp"

inline size_t checks = 0;

inline void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error(what);
    ++checks;
}

struct Shape {
    int layers, embd, ff, heads, kv_heads, head_dim, vocab;
};
// Small enough that a thousand-token reply takes a fraction of a second on the CPU.
inline const Shape kCpu{2, 64, 128, 4, 2, 16, 64};
// Heads of 128, which take the device's wide attention for a prompt and its per-row attention for a generated token.
inline const Shape kDevice{2, 256, 512, 2, 1, 128, 256};
// The CPU shape with a layer for each of up to four stages, for the cases that split it over several CPUs.
inline const Shape kSplit{4, 64, 128, 4, 2, 16, 64};

// A model with a token list of `vocab` tokens and no end token, so an uncapped reply runs to what the request may hold.
inline gguf::GGUFModel with_tokens(gguf::GGUFModel m, int vocab) {
    gguf::MetaValue tokens;
    tokens.vtype = gguf::V_ARRAY;
    tokens.u = gguf::V_STRING;
    for (int i = 0; i < vocab; ++i) {
        gguf::MetaValue t;
        t.vtype = gguf::V_STRING;
        t.s = "t" + std::to_string(i);
        tokens.arr.push_back(t);
    }
    m.kv.push_back({"tokenizer.ggml.tokens", tokens});
    return m;
}

// The synthetic model, served.
inline gguf::GGUFModel served(const Shape& s) {
    return with_tokens(infer::synthetic_model({s.layers, s.embd, s.ff, s.heads, s.kv_heads, s.head_dim, s.vocab, 20260925u}), s.vocab);
}

// A qwen35 model whose layers alternate linear and full attention, so each request holds a recurrent state slot beside its KV blocks (docs/QWEN35.md).
struct HybridShape {
    int layers, embd, ff, heads, kv_heads, head_dim, rope_dim, k_heads, v_heads, state_k, state_v, vocab;
};
// The CPU shape's widths, with four layers for up to four stages, two of them linear attention.
inline const HybridShape kHybrid{4, 64, 128, 4, 2, 16, 8, 2, 4, 16, 16, 64};
// The same with V heads of 32, so a tensor group of two splits it whole: each member's V heads of a tile are one Q8_0 block of the linear attention's output projection.
inline const HybridShape kHybridEven{4, 64, 128, 4, 2, 16, 8, 2, 4, 16, 32, 64};

// A hybrid model with random weights, Q8_0 matrices as the synthetic model has, so its prompt and decode rows take different CPU paths, and F32 norms and linear-attention tables; a context of 4096, which the cases' pools bound first.
// With `mtp`, an MTP block after the layers whose weights follow theirs, so the model without its drafter gives the bits of the file without the block (docs/SPECULATIVE.md, section 7).
inline gguf::GGUFModel served_hybrid(const HybridShape& s, bool mtp = false) {
    gguf::GGUFModel m;
    gguf::MetaValue arch;
    arch.vtype = gguf::V_STRING;
    arch.s = "qwen35";
    m.kv.push_back({"general.architecture", arch});
    for (const auto& kv : std::vector<std::pair<std::string, int>>{
             {"block_count", s.layers + (mtp ? 1 : 0)}, {"embedding_length", s.embd}, {"feed_forward_length", s.ff}, {"attention.head_count", s.heads},
             {"attention.head_count_kv", s.kv_heads}, {"attention.key_length", s.head_dim}, {"attention.value_length", s.head_dim},
             {"rope.dimension_count", s.rope_dim}, {"context_length", 4096}, {"ssm.conv_kernel", 4}, {"ssm.state_size", s.state_k},
             {"ssm.group_count", s.k_heads}, {"ssm.time_step_rank", s.v_heads}, {"ssm.inner_size", s.v_heads * s.state_v},
             {"full_attention_interval", 2}}) {
        gguf::MetaValue v;
        v.vtype = gguf::V_UINT32;
        v.u = (uint64_t)kv.second;
        m.kv.push_back({"qwen35." + kv.first, v});
    }
    if (mtp) {
        gguf::MetaValue v;
        v.vtype = gguf::V_UINT32;
        v.u = 1;
        m.kv.push_back({"qwen35.nextn_predict_layers", v});
    }
    gguf::MetaValue sections;
    sections.vtype = gguf::V_ARRAY;
    sections.u = gguf::V_INT32;
    for (int n : {s.rope_dim / 4, s.rope_dim / 2 - s.rope_dim / 4, 0, 0}) {
        gguf::MetaValue e;
        e.vtype = gguf::V_INT32;
        e.i = n;
        sections.arr.push_back(e);
    }
    m.kv.push_back({"qwen35.rope.dimension_sections", sections});
    std::mt19937 rng(20260930u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    // A tensor of shape [nin, rest...] with values scale * u + offset, u uniform in [-1, 1]; `q8` quantizes it to Q8_0 by rows of nin.
    auto add = [&](const std::string& name, std::vector<uint64_t> ne, bool q8, float scale = 1.0f, float offset = 0.0f) {
        size_t rows = 1;
        for (size_t d = 1; d < ne.size(); ++d) rows *= (size_t)ne[d];
        const size_t nin = (size_t)ne[0];
        std::vector<float> values(nin * rows);
        for (float& v : values) v = offset + scale * dist(rng);
        std::vector<uint8_t> bytes;
        if (q8) {
            const size_t blocks = nin / quant::Q8_0_BLOCK;
            bytes.resize(rows * blocks * quant::Q8_0_TYPESIZE);
            for (size_t r = 0; r < rows; ++r) quant::quantize_row_q8_0(values.data() + r * nin, bytes.data() + r * blocks * quant::Q8_0_TYPESIZE, blocks);
        } else {
            bytes.resize(values.size() * sizeof(float));
            std::memcpy(bytes.data(), values.data(), bytes.size());
        }
        gguf::TensorInfo t;
        t.name = name;
        t.ne = std::move(ne);
        t.type = q8 ? quant::GGML_TYPE_Q8_0 : quant::GGML_TYPE_F32;
        m.tensors.push_back(std::move(t));
        m.add_tensor_data(bytes);
    };
    const uint64_t E = (uint64_t)s.embd, F = (uint64_t)s.ff, D = (uint64_t)s.head_dim, HV = (uint64_t)s.v_heads, DV = (uint64_t)s.state_v;
    const uint64_t C = 2 * (uint64_t)(s.k_heads * s.state_k) + HV * DV;
    add("token_embd.weight", {E, (uint64_t)s.vocab}, true);
    add("output.weight", {E, (uint64_t)s.vocab}, true);
    add("output_norm.weight", {E}, false, 0.1f, 1.0f);
    for (int l = 0; l < s.layers; ++l) {
        const std::string pre = "blk." + std::to_string(l) + ".";
        add(pre + "attn_norm.weight", {E}, false, 0.1f, 1.0f);
        add(pre + "post_attention_norm.weight", {E}, false, 0.1f, 1.0f);
        if (l % 2) {
            add(pre + "attn_q.weight", {E, 2 * (uint64_t)s.heads * D}, true);
            add(pre + "attn_k.weight", {E, (uint64_t)s.kv_heads * D}, true);
            add(pre + "attn_v.weight", {E, (uint64_t)s.kv_heads * D}, true);
            add(pre + "attn_q_norm.weight", {D}, false, 0.1f, 1.0f);
            add(pre + "attn_k_norm.weight", {D}, false, 0.1f, 1.0f);
            add(pre + "attn_output.weight", {(uint64_t)s.heads * D, E}, true);
        } else {
            add(pre + "attn_qkv.weight", {E, C}, true);
            add(pre + "attn_gate.weight", {E, HV * DV}, true);
            add(pre + "ssm_alpha.weight", {E, HV}, true);
            add(pre + "ssm_beta.weight", {E, HV}, true);
            add(pre + "ssm_conv1d.weight", {4, C}, false, 0.5f);
            add(pre + "ssm_a", {HV}, false, 0.4f, -0.5f);
            add(pre + "ssm_dt.bias", {HV}, false, 0.5f);
            add(pre + "ssm_norm.weight", {DV}, false, 0.1f, 1.0f);
            add(pre + "ssm_out.weight", {HV * DV, E}, true);
        }
        add(pre + "ffn_gate.weight", {E, F}, true);
        add(pre + "ffn_up.weight", {E, F}, true);
        add(pre + "ffn_down.weight", {F, E}, true);
    }
    if (mtp) {
        const std::string pre = "blk." + std::to_string(s.layers) + ".";
        add(pre + "nextn.eh_proj.weight", {2 * E, E}, true);
        add(pre + "nextn.enorm.weight", {E}, false, 0.1f, 1.0f);
        add(pre + "nextn.hnorm.weight", {E}, false, 0.1f, 1.0f);
        add(pre + "nextn.shared_head_norm.weight", {E}, false, 0.1f, 1.0f);
        add(pre + "attn_norm.weight", {E}, false, 0.1f, 1.0f);
        add(pre + "post_attention_norm.weight", {E}, false, 0.1f, 1.0f);
        add(pre + "attn_q.weight", {E, 2 * (uint64_t)s.heads * D}, true);
        add(pre + "attn_k.weight", {E, (uint64_t)s.kv_heads * D}, true);
        add(pre + "attn_v.weight", {E, (uint64_t)s.kv_heads * D}, true);
        add(pre + "attn_q_norm.weight", {D}, false, 0.1f, 1.0f);
        add(pre + "attn_k_norm.weight", {D}, false, 0.1f, 1.0f);
        add(pre + "attn_output.weight", {(uint64_t)s.heads * D, E}, true);
        add(pre + "ffn_gate.weight", {E, F}, true);
        add(pre + "ffn_up.weight", {E, F}, true);
        add(pre + "ffn_down.weight", {F, E}, true);
    }
    return with_tokens(std::move(m), s.vocab);
}

// A prompt of n ids starting with `first`, the rest a fixed walk through the vocabulary.
inline std::vector<uint32_t> prompt_of(uint32_t first, size_t n, uint32_t vocab) {
    std::vector<uint32_t> p(n);
    for (size_t i = 0; i < n; ++i) p[i] = i ? (uint32_t)((first * 13 + i * 7) % vocab) : first;
    return p;
}

using Reply = std::vector<server::Request::Token>;

// One request: its prompt, and a cap, or none (0), which makes it uncapped as a compatible route's request without max_tokens is; a stop string, which ends it where its text first holds it; and a temperature above 0 with a seed for a sampled one.
struct Req {
    std::vector<uint32_t> prompt;
    int cap = 0;
    std::string stop = {};
    float temp = 0.0f;
    uint64_t seed = 0;
};

inline server::SampleParams params_of(const Req& r) {
    server::SampleParams p;
    p.temp = r.temp;
    p.seed = r.seed;
    p.logprobs = true;
    p.top_logprobs = 5;
    p.until_limit = r.cap == 0;
    if (r.cap) p.max_tokens = r.cap;
    if (!r.stop.empty()) p.stop = {r.stop};
    return p;
}

// Every token a request's channel gives until it ends.
inline Reply collect(server::Request& r) {
    Reply out;
    server::Request::Token t;
    for (;;) {
        const auto got = r.next(t, server::Request::Clock::now() + std::chrono::seconds(120));
        if (got == server::Request::Next::end) break;
        require(got == server::Request::Next::id, "a channel gave nothing for 120 seconds");
        out.push_back(t);
    }
    return out;
}

// A request's reply, which must run to its length or its stop string.
inline Reply drain(server::Request& r) {
    Reply out = collect(r);
    require(r.finish() == "length" || r.finish() == "stop", "a request ended with " + r.finish() + " " + r.error());
    return out;
}

// A stop string that ends a request whose reply is `ids` with its token `at`: the text of its tokens up to `at`, from the latest start whose text first appears there.
inline std::string stop_at(const bpe::Tokenizer& tok, const std::vector<uint32_t>& ids, size_t at) {
    std::string whole;
    std::vector<size_t> ends;
    for (uint32_t id : ids) {
        whole += tok.decode({id});
        ends.push_back(whole.size());
    }
    for (size_t from = at + 1; from-- > 0;) {
        const size_t begin = from ? ends[from - 1] : 0;
        const std::string s = whole.substr(begin, ends[at] - begin);
        if (whole.find(s) == begin) return s;
    }
    throw std::runtime_error("no stop string ends the reply at its token " + std::to_string(at));
}

// Once every request has ended, the blocks the scheduler holds reserved are the donors' blocks, never more than a pool has.
inline void ledger(const server::Scheduler::Stats& s, const infer::Model& model, const std::string& what) {
    require(s.reserved.size() == model.kv_pools() && s.donor_blocks.size() == model.kv_pools(), what + ": the ledger does not name every pool");
    for (size_t p = 0; p < s.reserved.size(); ++p)
        require(s.reserved[p] <= model.kv_pool_blocks(p) && s.reserved[p] == s.donor_blocks[p],
                what + ": pool " + std::to_string(p) + " holds " + std::to_string(s.reserved[p]) + " blocks reserved and its donors " +
                std::to_string(s.donor_blocks[p]) + ", of " + std::to_string(model.kv_pool_blocks(p)));
}

// Runs `waves` through one scheduler over `model` with `passes` passes in flight, the scheduler's own number when 0, `host` bytes of host memory for evicted donors and, with a `proposer`, up to `draft_max` drafts a verify, `priced` by the passes' measured cost or else by the decode columns alone: a wave is fully queued before any pass retires, and every request is drained before the next wave starts.
// The replies come in submission order; the scheduler's counters at the end go to `stats`.
inline std::vector<Reply> serve(infer::Model& model, const bpe::Tokenizer& tok, size_t max_seqs,
                         const std::vector<std::vector<Req>>& waves, server::Scheduler::Stats* stats = nullptr, size_t passes = 0,
                         size_t host = 0, infer::spec::Proposer* proposer = nullptr, size_t draft_max = 0, bool priced = false) {
    server::Scheduler sched(model, tok, max_seqs, 64, passes, false, host, proposer, draft_max, priced);
    std::mutex submitting;
    // The first pass may start while a wave is queued, but cannot retire and advance its request ahead of the rest.
    sched.on_retire = [&](const server::Scheduler::Retired&) { std::lock_guard<std::mutex> lock(submitting); };
    std::thread runner([&] { sched.run(); });
    std::vector<Reply> replies;
    try {
        for (const auto& wave : waves) {
            std::vector<std::shared_ptr<server::Request>> handles;
            {
                std::lock_guard<std::mutex> lock(submitting);
                for (const Req& r : wave) handles.push_back(sched.submit(r.prompt, params_of(r)));
            }
            for (auto& h : handles) replies.push_back(drain(*h));
            ledger(sched.stats(), model, "after a wave");
        }
        if (stats) *stats = sched.stats();
    } catch (...) {
        sched.stop();
        runner.join();
        throw;
    }
    sched.stop();
    runner.join();
    return replies;
}

// A reply against the same request's reply alone: every id, logprob and top entry equal, bit for bit; a reply that ended early, a cancelled one's, may be a prefix of it.
inline void same(const Reply& alone, const Reply& got, const std::string& what, bool prefix = false) {
    const size_t n = std::min(alone.size(), got.size());
    for (size_t i = 0; i < n; ++i) {
        bool eq = alone[i].id == got[i].id && alone[i].logprob == got[i].logprob && alone[i].top.size() == got[i].top.size();
        for (size_t j = 0; eq && j < alone[i].top.size(); ++j)
            eq = alone[i].top[j].id == got[i].top[j].id && alone[i].top[j].logprob == got[i].top[j].logprob;
        if (!eq) {
            char detail[160];
            std::snprintf(detail, sizeof detail, ": token %zu of %zu differs from the request alone (id %u, logprob %.9g; alone id %u, logprob %.9g)",
                          i, alone.size(), got[i].id, (double)got[i].logprob, alone[i].id, (double)alone[i].logprob);
            require(false, what + detail);
        }
    }
    require(alone.size() == got.size() || (prefix && got.size() < alone.size()),
            what + ": " + std::to_string(got.size()) + " tokens against " + std::to_string(alone.size()) + " alone");
}

// A fresh model over the backends `backends` makes, with a pool of `pool` tokens and prompts taken `ubatch` tokens a pass (the default when 0).
using Make = std::function<std::unique_ptr<infer::Model>(size_t pool, int ubatch)>;

// A hybrid model holds `state_slots` recurrent states, which bounds the requests a scheduler over it runs at once, and `checkpoints` states kept for prefix reuse; the others hold none.
// With `marks` the model marks that many sequences at once for verifies of up to `mark_rows` rows, and with `drafter` it loads the embedded drafter its file carries.
// With a `width` above 1 the backends form tensor groups of that many, the stages (docs/TENSOR-SPLIT.md).
inline Make on(const gguf::GGUFModel& weights, std::function<std::vector<backend::BackendPtr>()> backends, size_t state_slots = 8, size_t checkpoints = 0,
               size_t marks = 0, size_t mark_rows = 0, bool drafter = false, size_t width = 1) {
    return [&weights, backends, state_slots, checkpoints, marks, mark_rows, drafter, width](size_t pool, int ubatch) {
        infer::ModelOptions options;
        options.kv_tokens = pool;
        options.state_slots = state_slots;
        options.checkpoint_slots = checkpoints;
        options.mark_slots = marks;
        options.mark_rows = mark_rows;
        std::vector<backend::BackendPtr> b = backends();
        infer::PlacementRequest request;
        request.drafter = drafter;
        for (size_t i = 0; i < b.size(); ++i) request.names.push_back("device " + std::to_string(i));
        if (b.size() > width) request.shares.assign(b.size() / width, 1);
        request.width = width;
        request.ubatch = ubatch;
        return std::move(infer::place_model(infer::gguf_weights(weights), std::move(b), request, options).model);
    };
}

inline std::vector<backend::BackendPtr> cpus(size_t n) {
    std::vector<backend::BackendPtr> v;
    for (size_t i = 0; i < n; ++i) {
        auto c = std::make_shared<backend::CpuBackend>();
        c->set_threads(1);
        v.push_back(c);
    }
    return v;
}

// A CPU backend whose copies into or out of its host-visible memory throw once `fail_read` or `fail_write` is set, as a copy to or from host memory that fails does (Model::save_host, Model::restore_host); every other copy runs.
struct FailingCopies : backend::CpuBackend {
    std::atomic<bool> fail_read{false}, fail_write{false};
    std::mutex m;
    std::vector<const backend::Buffer*> host;
    backend::BufferPtr alloc(size_t bytes, backend::Memory where) override {
        backend::BufferPtr b = CpuBackend::alloc(bytes, where);
        std::lock_guard<std::mutex> lock(m);
        if (where == backend::Memory::host_visible) host.push_back(b.get());
        return b;
    }
    void copy(backend::Buffer& dst, size_t dst_off, const backend::Buffer& src, size_t src_off, size_t bytes) override {
        {
            std::lock_guard<std::mutex> lock(m);
            const auto in = [&](const backend::Buffer* b) { return std::find(host.begin(), host.end(), b) != host.end(); };
            if (fail_read && in(&dst)) throw std::runtime_error("injected read failure");
            if (fail_write && in(&src)) throw std::runtime_error("injected write failure");
        }
        CpuBackend::copy(dst, dst_off, src, src_off, bytes);
    }
};

// A CPU backend that runs `hook` whenever the model submits its work, which under the scheduler happens only inside a pass's stages, so a case acts in the scheduler's own thread while that pass is in flight.
struct Hooked : backend::CpuBackend {
    std::function<void()> hook;
    backend::Ticket submit() override {
        if (hook) hook();
        return CpuBackend::submit();
    }
};

inline std::vector<std::shared_ptr<Hooked>> hooked(size_t n) {
    std::vector<std::shared_ptr<Hooked>> v;
    for (size_t i = 0; i < n; ++i) {
        v.push_back(std::make_shared<Hooked>());
        v.back()->set_threads(1);
    }
    return v;
}

// The CPU's 128-token blocks, of which a case's pool holds a given number.
inline constexpr size_t kBlock = 128;

// Once the scheduler has stopped, the model's own history takes the whole pool of `tokens`, so every block has come back, a block of a request whose handle is still held included.
inline void whole_pool_free(infer::Model& model, size_t tokens, uint32_t vocab, const std::string& what) {
    try {
        model.prefill(prompt_of(7, tokens, vocab));
    } catch (const std::exception& e) {
        require(false, what + ": a prompt as long as the pool failed once the scheduler had stopped (" + e.what() + ")");
    }
    model.reset();
}
