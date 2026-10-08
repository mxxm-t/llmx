#pragma once
#include <cstddef>
#include <cstdint>
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

// A grouped expert call needs mixed-type support only when both declared projections exist and their storage tags differ; the requirement is the part's the gate role runs in.
inline void routed_ops(LayerPlan& layer, const TensorIndex& tensors, uint16_t gate, uint16_t up) {
    const TensorView *g = nullptr, *u = nullptr;
    Part part = Part::ffn;
    for (const Role& role : layer.roles) {
        if (role.id != gate && role.id != up) continue;
        if (role.id == gate) part = role.part;
        const auto tensor = tensors.find(role.name);
        if (tensor) (role.id == gate ? g : u) = &tensors.view(*tensor);
    }
    if (g && u && g->type != u->type) layer.ops.push_back({part, backend::Op::mixed_experts});
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

// A routed layer's split: each expert's hidden rows, the gate and up stacks by the `ff` rows of each of their `experts` matrices and the down stack by as many columns; the router stays whole on every member, so every member routes alike.
// A shared expert is split the same way, as one expert.
// The down stack splits on whole blocks of its storage type, so a member holds the blocks that cover its even share of the columns, and its gate and up stacks a row for each covered column, zero where the column is another member's: that row's SiLU product is 0, so each column counts once, by its owner (ShardSection::align).
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
    return {w.type, {w.data.get(), 0}, out, w.nout};
}

// A part's last projection, whose output joins the residual: added to it on one device, and on a tensor group written as the member's partial rows, which the runtime sums into every member's residual (Step::partial).
// With `add`, a part's second such projection, after its routed experts, it adds to the partial rows they wrote.
inline void join(const Step& s, const Weight& w, backend::CSlice in, bool add = false) {
    if (s.width > 1 && !add) s.b.matmul(w.type, w.slice(), in, s.partial, w.nin, w.nout, s.rows, s.runs, s.dtype);
    else s.b.matmul_add(w.type, w.slice(), in, s.width > 1 ? s.partial : s.x, w.nin, w.nout, s.rows, s.runs, s.dtype);
}

// The rows of `table` the ids name, into the residual.
inline void embed(const Step& s, const Weight& table, const uint32_t* ids) {
    s.b.embed(s.x, table.type, table.slice(), table.nin, table.nout, ids, s.rows);
}

// The SwiGLU feed-forward block over the normed rows `h`: the residual gains down(silu(gate h) * up h), through the slots `g`, `u` and `act`, each as wide as the block.
// With `row_gate`, one value a row, the down projection reads its row scaled by sigmoid of that value; that block is a shared expert's, which follows its layer's routed experts (join's `add`).
inline void swiglu(const Step& s, const Weight& gate, const Weight& up, const Weight& down,
                   backend::Slice h, backend::Slice g, backend::Slice u, backend::Slice act, const backend::Slice* row_gate = nullptr) {
    s.b.matmul_group({projection(gate, g), projection(up, u)}, h, gate.nin, s.rows, s.runs, s.dtype);
    s.b.silu_mul(act, g, u, s.rows * gate.nout, s.runs);
    if (row_gate) s.b.sigmoid_mul(act, act, *row_gate, s.rows, gate.nout, 1, 1, 0, s.runs);
    join(s, down, act, row_gate != nullptr);
}

// Routed experts over the normed rows `h`: the router's scores, a softmax's top k renormalized when `norm`, each token's k experts' SwiGLU through the slots `g`, `u` and `act`, each k expert rows a token wide, and the weighted sum of their down projections added to the residual.
// `scores` holds a row's n_expert scores, and `ids` and `weights` its k choices.
// On a tensor group each member holds its share of every expert's hidden rows and adds its weighted sum to its partial rows, which the runtime hands this part cleared (Step::partial).
inline void routed_experts(const Step& s, const Weight& router, const Weight& gate, const Weight& up, const Weight& down, size_t k, bool norm,
                           backend::Slice h, backend::Slice g, backend::Slice u, backend::Slice act,
                           backend::Slice scores, backend::Slice ids, backend::Slice weights) {
    backend::Backend& b = s.b;
    const size_t E = router.nin, n_expert = router.nout, ff = gate.nout;
    b.matmul(router.type, router.slice(), h, scores, E, n_expert, s.rows, s.runs, backend::Dtype::f32);
    b.route_experts(scores, s.rows, n_expert, k, norm, ids, weights);
    const backend::Backend::Routing routing{ids, weights, k, n_expert};
    b.matmul_experts({projection(gate, g), projection(up, u)}, h, E, s.rows, routing, s.runs, s.dtype);
    // The routed down projection reads the SiLU's output as k entries a token row, each of its token's prompt.
    std::vector<backend::RowRun>& entry_runs = *s.scratch;
    entry_runs.clear();
    for (size_t i = 0; i < s.runs.n; ++i) entry_runs.push_back(backend::RowRun{s.runs.runs[i].end * k, s.runs.runs[i].extent});
    b.silu_mul(act, g, u, s.rows * k * ff, {entry_runs.data(), entry_runs.size()});
    b.matmul_experts_add(down.type, down.slice(), act, s.width > 1 ? s.partial : s.x, ff, E, s.rows, routing, s.runs, s.dtype);
}

// An MTP block's input (docs/SPECULATIVE.md, section 7): `pair` holds 2 * rows rows of E, the tokens' rows then the target's rows before them, which are normed in place by `enorm` and `hnorm`, put side by side a row each, token first, through `side` (a single row is already so), and projected by `eh_proj` into `out`.
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
    s.b.matmul(eh_proj.type, eh_proj.slice(), in, out, eh_proj.nin, eh_proj.nout, n, s.runs, s.dtype);
}

// The head: the rows that want logits are not contiguous once entries mix, so they are compacted into `rows` first, then normed and projected once over exactly those rows.
inline void head(const HeadStep& s, const Weight& norm, const Weight& out, float eps, backend::Slice rows) {
    const size_t E = out.nin;
    s.b.gather_rows(rows, s.x, E, s.pick, s.want);
    s.b.rms_norm_rows(rows, rows, norm.slice(), s.want, E, E, eps);
    s.b.matmul_logits(out.type, out.slice(), rows, s.logits, out.nin, out.nout, s.want, s.head_runs, s.dtype);
}

} // namespace infer::blocks
