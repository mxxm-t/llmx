#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "quant/types.hpp"
#include "backends/backend.hpp"
#include "model/weights.hpp"
#include "model/architecture.hpp"

// The graph pieces more than one architecture runs, as backend ops over the weights and arena slots each is handed; none names a tensor or a metadata key (docs/ADDING-AN-ARCHITECTURE.md, One owner).

namespace infer::blocks {

// An image row's (start, row, column) interleaved over `half` rotated pairs as HF's Qwen 3.5 interleaves its MRoPE sections (ModelPlan::image_streams).
// Pair j takes the row where j mod 3 is 1 inside three times the second section, the column where it is 2 inside three times the third, and the start elsewhere.
inline std::vector<uint8_t> interleaved_streams(const std::vector<int>& sections, size_t half) {
    std::vector<uint8_t> of(half, 0);
    for (size_t j = 0; j < half; ++j) {
        if (j % 3 == 1 && j < 3 * (size_t)sections[1]) of[j] = 1;
        if (j % 3 == 2 && j < 3 * (size_t)sections[2]) of[j] = 2;
    }
    return of;
}

// A routed layer's part issues the routed products, and a grouped expert call needs mixed-type support too when both declared projections exist and their storage tags differ; both are the part's the gate role runs in.
inline void routed_ops(LayerPlan& layer, const TensorIndex& tensors, uint16_t gate, uint16_t up) {
    const TensorView *g = nullptr, *u = nullptr;
    Part part = Part::ffn;
    for (const Role& role : layer.roles) {
        if (role.id != gate && role.id != up) continue;
        if (role.id == gate) part = role.part;
        const auto tensor = tensors.find(role.name);
        if (tensor) (role.id == gate ? g : u) = &tensors.view(*tensor);
    }
    layer.ops.push_back({part, backend::Op::routed_experts});
    if (g && u && g->type != u->type) layer.ops.push_back({part, backend::Op::mixed_experts});
}

// A drafter's head, `output` over the `nin`-wide rows `x`: one device's logits and their argmax as the next ids, or on a tensor group the member's vocabulary rows and their candidate, which Model::draft_record sums and picks.
inline void draft_head(const DraftStep& s, const Weight& output, backend::CSlice x, size_t nin) {
    const size_t n = output.nout;
    s.b.matmul_logits(output.operand(), x, s.logits, nin, n, s.rows, s.runs, s.dtype);
    if (s.width > 1) s.b.argmax_part(s.partial, s.logits, s.rows, n, s.first, s.member, s.width);
    else s.b.argmax_rows(s.next, s.logits, s.rows, n, s.id);
}

// Declare how role `id` of `roles` splits over a tensor group (Role::shard); a role the list lacks is the plan's error.
inline void shard(std::vector<Role>& roles, uint16_t id, Axis axis, std::vector<ShardSection> sections) {
    for (Role& role : roles)
        if (role.id == id) {
            role.shard = {axis, std::move(sections)};
            return;
        }
    throw std::logic_error("inference: no role to split with id " + std::to_string(id));
}

// A routed layer's split: each expert's hidden rows, the gate and up stacks by the `ff` rows of each of their `experts` matrices and the down stack by as many columns, the shared expert likewise as one expert; the router stays whole.
// The down stack splits on whole blocks of its type, so a member holds the blocks covering its share and its gate and up stacks a row for each covered column, zero where another member owns it (ShardSection::align).
inline void shard_experts(std::vector<Role>& roles, const TensorIndex& tensors, uint16_t gate, uint16_t up, uint16_t down, uint64_t ff, uint64_t experts) {
    uint64_t block = 1;
    for (const Role& role : roles)
        if (role.id == down)
            if (const auto tensor = tensors.find(role.name))
                if (const quant::StorageType* type = quant::storage_type(tensors.view(*tensor).type)) block = type->block_size;
    shard(roles, gate, Axis::rows, {{ff, 1, experts, "rows of each expert", false, block}});
    shard(roles, up, Axis::rows, {{ff, 1, experts, "rows of each expert", false, block}});
    shard(roles, down, Axis::columns, {{ff, 1, 1, "expert columns", false, block}});
}

// The SwiGLU block's split: the gate and up projections by their `ff` rows, the down projection by as many columns.
inline void shard_swiglu(std::vector<Role>& roles, uint16_t gate, uint16_t up, uint16_t down, uint64_t ff) {
    shard(roles, gate, Axis::rows, {{ff, 1, 1, "feed-forward rows"}});
    shard(roles, up, Axis::rows, {{ff, 1, 1, "feed-forward rows"}});
    shard(roles, down, Axis::columns, {{ff, 1, 1, "feed-forward columns"}});
}

// A weight's product into `out`, the buffer passed by raw pointer, not by handle, so building one copies no shared pointer on the per-token path.
inline backend::Projection projection(const Weight& w, backend::Slice out) {
    return {w.operand(), out, w.nout};
}

// A part's last projection, whose output joins the residual: added to it on one device, and on a tensor group written as the member's partial rows, which the runtime sums into every member's residual (Step::partial).
inline void join(const Step& s, const Weight& w, backend::CSlice in) {
    if (s.width > 1) s.b.matmul(w.operand(), in, s.partial, w.nin, w.nout, s.rows, s.runs, s.dtype);
    else s.b.matmul_add(w.operand(), in, s.x, w.nin, w.nout, s.rows, s.runs, s.dtype);
}

// The rows of `table` the ids name, into the residual.
inline void embed(const Step& s, const Weight& table, const uint32_t* ids) {
    s.b.embed(s.x, table.operand(), table.nin, table.nout, ids, s.rows);
}

// The SwiGLU feed-forward block over the normed rows `h`: the residual gains down(silu(gate h) * up h), through the slots `g`, `u` and `act`, each as wide as the block.
inline void swiglu(const Step& s, const Weight& gate, const Weight& up, const Weight& down,
                   backend::Slice h, backend::Slice g, backend::Slice u, backend::Slice act) {
    s.b.matmul_group({projection(gate, g), projection(up, u)}, h, gate.nin, s.rows, s.runs, s.dtype);
    s.b.silu_mul(act, g, u, s.rows * gate.nout, 0.0f, s.runs);
    join(s, down, act);
}

// A layer's shared expert: its SwiGLU weights and the vector gating its row, null where it has none.
struct SharedExpert {
    const Weight &gate, &up, &down;
    const Weight* row_gate;
    float limit = 0.0f;  // the SwiGLU clamp, 0 for none
};

// Routed experts over the normed rows `h` as one backend op (Backend::moe_step), picked by `rule` and clamped at `limit`, then `shared` gated into `row_gate`, joining the residual or a member's partial rows.
// `g`, `u` and `act` hold k routed rows a token and the shared expert's after rows * k of them; `zero`, a row of zeros, clears the residual first, and a member's partial rows are cleared from the step's own.
inline void moe_step(const Step& s, const Weight& router, const Weight& gate, const Weight& up, const Weight& down, size_t k, const backend::RouteRule& rule,
                     backend::Slice h, backend::Slice g, backend::Slice u, backend::Slice act, backend::Slice scores, backend::Slice ids, backend::Slice weights,
                     const SharedExpert* shared = nullptr, backend::Slice row_gate = {}, const backend::Buffer* zero = nullptr, float limit = 0.0f) {
    const size_t routed = s.rows * k * gate.nout;
    const auto past = [&](backend::Slice slot) { return backend::Slice{slot.buffer, slot.offset + routed}; };
    backend::MoeShared sh{};
    if (shared)
        sh = {projection(shared->gate, past(g)), projection(shared->up, past(u)), shared->down.operand(), past(act),
              shared->row_gate ? projection(*shared->row_gate, row_gate) : backend::Projection{}, shared->limit};
    const backend::MoeStep m{projection(router, scores), k, rule, ids, weights, projection(gate, g), projection(up, u), down.operand(), act,
                             shared ? &sh : nullptr, zero ? zero : s.width > 1 ? s.zero : nullptr, s.scratch, limit};
    s.b.moe_step(m, h, s.width > 1 ? s.partial : s.x, router.nin, s.rows, s.runs, s.dtype);
}

// An MTP block's input (docs/SPECULATIVE.md, section 7): `pair` holds 2 * rows rows of E, the tokens' rows then the target's before them, normed in place by `enorm` and `hnorm`, joined token first through `side` and projected by `eh_proj`.
inline void nextn_input(const Step& s, const Weight& enorm, const Weight& hnorm, const Weight& eh_proj, backend::Slice pair, backend::Slice side,
                        backend::Slice out, float eps) {
    const size_t E = enorm.nin, n = s.rows;
    const backend::Slice prev{pair.buffer, pair.offset + n * E};
    s.b.rms_norm_rows(pair, pair, enorm.slice(), n, E, E, eps);
    s.b.rms_norm_rows(prev, prev, hnorm.slice(), n, E, E, eps);
    backend::Slice in = pair;
    if (n > 1) {
        std::vector<uint32_t> order(2 * n);
        for (size_t r = 0; r < n; ++r) {
            order[2 * r] = (uint32_t)r;
            order[2 * r + 1] = (uint32_t)(n + r);
        }
        s.b.gather_rows(side, pair, E, order.data(), 2 * n);
        in = side;
    }
    s.b.matmul(eh_proj.operand(), in, out, eh_proj.nin, eh_proj.nout, n, s.runs, s.dtype);
}

// An n-gram hash's fields, read by ngram_rows; `fill_id` stands in before the start and, when `fill_cuts`, for every older token once it appears in text.
// A nonempty `token_map` maps every id, the fill's included, into [0, mapped_vocab) before the fold.
struct NgramHash {
    int ngram_size = 0, heads_per_ngram = 0, fill_id = 0;
    std::vector<uint64_t> multipliers, head_offsets, head_vocab_sizes;
    std::vector<uint32_t> token_map;
    uint32_t mapped_vocab = 0;
    bool fill_cuts = true;
};

// The table rows of a token chunk by the n-gram hash: previous is the oldest-first tail immediately before current, and output is token-major, `heads_per_ngram` rows for each order past the first.
inline size_t ngram_rows(const NgramHash& h, uint32_t vocab_size,
                         const uint32_t* previous, size_t previous_count,
                         const uint32_t* current, size_t current_count,
                         int32_t* output, size_t output_capacity) {
    if (!vocab_size || h.fill_id < 0 || uint32_t(h.fill_id) >= vocab_size)
        throw std::runtime_error("inference: n-gram token id is outside the vocabulary");
    if (h.ngram_size < 2 || h.heads_per_ngram <= 0 ||
        size_t(h.ngram_size) != h.multipliers.size())
        throw std::runtime_error("inference: invalid n-gram metadata");
    const bool mapped = !h.token_map.empty();
    if (mapped != (h.mapped_vocab != 0))
        throw std::runtime_error("inference: an n-gram token map and its range come together");
    if (mapped && h.token_map.size() != vocab_size)
        throw std::runtime_error("inference: the n-gram token map does not cover the vocabulary");
    const size_t max_previous = size_t(h.ngram_size - 1);
    if (previous_count > max_previous || (previous_count && !previous) || (current_count && !current))
        throw std::runtime_error("inference: invalid n-gram token history");
    const size_t heads = size_t(h.ngram_size - 1) * size_t(h.heads_per_ngram);
    if (h.head_offsets.size() != heads || h.head_vocab_sizes.size() != heads)
        throw std::runtime_error("inference: invalid n-gram head metadata");
    if (current_count && heads > std::numeric_limits<size_t>::max() / current_count)
        throw std::runtime_error("inference: n-gram output size overflow");
    const size_t required = current_count * heads;
    if (required > output_capacity || (required && !output))
        throw std::runtime_error("inference: n-gram output buffer is too small");

    const uint64_t largest_token = uint64_t((mapped ? h.mapped_vocab : vocab_size) - 1);
    for (uint64_t multiplier : h.multipliers) {
        if (multiplier > uint64_t(std::numeric_limits<int64_t>::max()) ||
            (largest_token && multiplier > uint64_t(std::numeric_limits<int64_t>::max()) / largest_token))
            throw std::runtime_error("inference: n-gram token hash product exceeds int64 range");
    }
    const auto valid = [&](uint32_t id) { return id < vocab_size && (!mapped || h.token_map[id] < h.mapped_vocab); };
    if (!valid(uint32_t(h.fill_id)))
        throw std::runtime_error("inference: n-gram token id is outside the vocabulary");
    for (size_t i = 0; i < previous_count; ++i)
        if (!valid(previous[i]))
            throw std::runtime_error("inference: n-gram token id is outside the vocabulary");
    for (size_t i = 0; i < current_count; ++i)
        if (!valid(current[i]))
            throw std::runtime_error("inference: n-gram token id is outside the vocabulary");

    const auto map = [&](uint32_t id) -> uint64_t { return mapped ? h.token_map[id] : id; };
    const size_t ngram = size_t(h.ngram_size);
    const size_t per_ngram = size_t(h.heads_per_ngram);
    const size_t history_count = previous_count;
    for (size_t i = 0; i < current_count; ++i) {
        auto token_before = [&](size_t distance) -> uint32_t {
            const size_t prefix_count = history_count + i;
            if (distance > prefix_count) return uint32_t(h.fill_id);
            const size_t index = prefix_count - distance;
            return index < history_count ? previous[index] : current[index - history_count];
        };
        uint64_t mixed = map(current[i]) * h.multipliers[0];
        bool cut = false;
        for (size_t s = 1; s < ngram; ++s) {
            const uint32_t token = cut ? uint32_t(h.fill_id) : token_before(s);
            cut = cut || (h.fill_cuts && token == uint32_t(h.fill_id));
            mixed ^= map(token) * h.multipliers[s];
            const size_t first_head = (s - 1) * per_ngram;
            for (size_t g = 0; g < per_ngram; ++g) {
                const size_t head = first_head + g;
                output[i * heads + head] = int32_t(
                    mixed % h.head_vocab_sizes[head] + h.head_offsets[head]);
            }
        }
    }
    return required;
}


struct NgramGateWeights {
    const Weight &key, &value, &norm_key, &norm_query;
};

struct NgramGateScratch {
    backend::Slice key, value, query;
};

// The n-gram gate over the gathered rows `embeddings`, `embedding_dim` wide: the H streams' keys and the value projected, the keys and stream rows normed, the value gated into `gated`.
// Each stream gates by its dot's signed square root, a zero dot's sign as `zero` gives it.
inline void ngram_gate(const Step& s, const NgramGateWeights& w, backend::CSlice embeddings, size_t embedding_dim, size_t E, size_t H, float eps,
                       backend::Slice gated, const NgramGateScratch& tmp, backend::ZeroSign zero = backend::ZeroSign::none) {
    s.b.matmul_group({projection(w.key, tmp.key), projection(w.value, tmp.value)},
                     embeddings, embedding_dim, s.rows, s.runs, s.dtype);
    s.b.group_rms_norm(tmp.key, tmp.key, w.norm_key.slice(), s.rows, H, E, eps, s.runs);
    s.b.group_rms_norm(tmp.query, s.x, w.norm_query.slice(), s.rows, H, E, eps, s.runs);
    s.b.signed_sqrt_gate(gated, tmp.key, tmp.query, tmp.value, s.rows, H, E, s.runs, zero);
}

// The head: the rows that want logits are not contiguous once entries mix, so they are compacted into `rows` first, then normed and projected once over exactly those rows.
inline void head(const HeadStep& s, const Weight& norm, const Weight& out, float eps, backend::Slice rows) {
    const size_t E = out.nin;
    s.b.gather_rows(rows, s.x, E, s.pick, s.want);
    s.b.rms_norm_rows(rows, rows, norm.slice(), s.want, E, E, eps);
    s.b.matmul_logits(out.operand(), rows, s.logits, out.nin, out.nout, s.want, s.head_runs, s.dtype);
}

} // namespace infer::blocks
