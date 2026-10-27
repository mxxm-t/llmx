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
    s.b.silu_mul(act, g, u, s.rows * gate.nout, s.runs);
    join(s, down, act);
}

// A layer's shared expert: its SwiGLU weights and the vector gating its row, null where it has none.
struct SharedExpert {
    const Weight &gate, &up, &down;
    const Weight* row_gate;
};

// Routed experts over the normed rows `h` as one backend op (Backend::moe_step), renormalized when `norm`, then `shared` scaled by sigmoid of its row gate into `row_gate`, all joining the residual or a member's partial rows.
// `g`, `u` and `act` hold k routed rows a token and the shared expert's after rows * k of them; `zero`, a row of zeros, clears the residual first.
inline void moe_step(const Step& s, const Weight& router, const Weight& gate, const Weight& up, const Weight& down, size_t k, bool norm,
                     backend::Slice h, backend::Slice g, backend::Slice u, backend::Slice act, backend::Slice scores, backend::Slice ids, backend::Slice weights,
                     const SharedExpert* shared = nullptr, backend::Slice row_gate = {}, const backend::Buffer* zero = nullptr) {
    const size_t routed = s.rows * k * gate.nout;
    const auto past = [&](backend::Slice slot) { return backend::Slice{slot.buffer, slot.offset + routed}; };
    backend::MoeShared sh{};
    if (shared)
        sh = {projection(shared->gate, past(g)), projection(shared->up, past(u)), shared->down.operand(), past(act),
              shared->row_gate ? projection(*shared->row_gate, row_gate) : backend::Projection{}};
    const backend::MoeStep m{projection(router, scores), k, norm, ids, weights, projection(gate, g), projection(up, u), down.operand(), act,
                             shared ? &sh : nullptr, zero, s.scratch};
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

// The head: the rows that want logits are not contiguous once entries mix, so they are compacted into `rows` first, then normed and projected once over exactly those rows.
inline void head(const HeadStep& s, const Weight& norm, const Weight& out, float eps, backend::Slice rows) {
    const size_t E = out.nin;
    s.b.gather_rows(rows, s.x, E, s.pick, s.want);
    s.b.rms_norm_rows(rows, rows, norm.slice(), s.want, E, E, eps);
    s.b.matmul_logits(out.operand(), rows, s.logits, out.nin, out.nout, s.want, s.head_runs, s.dtype);
}

} // namespace infer::blocks
