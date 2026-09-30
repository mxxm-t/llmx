#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "backends/backend.hpp"
#include "model/weights.hpp"
#include "model/architecture.hpp"

// The graph pieces more than one architecture runs, as backend ops over the weights and arena slots each is handed; none names a tensor or a metadata key (docs/ADDING-AN-ARCHITECTURE.md, One owner).

namespace infer::blocks {

// A weight's product into `out`, the buffer passed by raw pointer, not by handle, so building one copies no shared pointer on the per-token path.
inline backend::Projection projection(const Weight& w, backend::Slice out) {
    return {w.type, {w.data.get(), 0}, out, w.nout};
}

// The rows of `table` the ids name, into the residual.
inline void embed(const Step& s, const Weight& table, const uint32_t* ids) {
    s.b.embed(s.x, table.type, table.slice(), table.nin, table.nout, ids, s.rows);
}

// The SwiGLU feed-forward block over the normed rows `h`: the residual gains down(silu(gate h) * up h), through the slots `g`, `u` and `act`, each as wide as the block.
// With `row_gate`, one value a row, the down projection reads its row scaled by sigmoid of that value.
inline void swiglu(const Step& s, const Weight& gate, const Weight& up, const Weight& down,
                   backend::Slice h, backend::Slice g, backend::Slice u, backend::Slice act, const backend::Slice* row_gate = nullptr) {
    s.b.matmul_group({projection(gate, g), projection(up, u)}, h, gate.nin, s.rows, s.runs);
    s.b.silu_mul(act, g, u, s.rows * gate.nout, s.runs);
    if (row_gate) s.b.sigmoid_mul(act, act, *row_gate, s.rows, gate.nout, 1, 1, 0, s.runs);
    s.b.matmul_add(down.type, down.slice(), act, s.x, down.nin, down.nout, s.rows, s.runs);
}

// Routed experts over the normed rows `h`: the router's scores, a softmax's top k renormalized when `norm`, each token's k experts' SwiGLU through the slots `g`, `u` and `act`, each k expert rows a token wide, and the weighted sum of their down projections added to the residual.
// `scores` holds a row's n_expert scores, and `ids` and `weights` its k choices.
inline void routed_experts(const Step& s, const Weight& router, const Weight& gate, const Weight& up, const Weight& down, size_t k, bool norm,
                           backend::Slice h, backend::Slice g, backend::Slice u, backend::Slice act,
                           backend::Slice scores, backend::Slice ids, backend::Slice weights) {
    backend::Backend& b = s.b;
    const size_t E = router.nin, n_expert = router.nout, ff = gate.nout;
    b.matmul(router.type, router.slice(), h, scores, E, n_expert, s.rows, s.runs);
    b.route_experts(scores, s.rows, n_expert, k, norm, ids, weights);
    const backend::Backend::Routing routing{ids, weights, k, n_expert};
    b.matmul_experts({projection(gate, g), projection(up, u)}, h, E, s.rows, routing, s.runs);
    // The routed down projection reads the SiLU's output as k entries a token row, each of its token's prompt.
    std::vector<backend::RowRun>& entry_runs = *s.scratch;
    entry_runs.clear();
    for (size_t i = 0; i < s.runs.n; ++i) entry_runs.push_back(backend::RowRun{s.runs.runs[i].end * k, s.runs.runs[i].extent});
    b.silu_mul(act, g, u, s.rows * k * ff, {entry_runs.data(), entry_runs.size()});
    b.matmul_experts_add(down.type, down.slice(), act, s.x, ff, E, s.rows, routing, s.runs);
}

// The head: the rows that want logits are not contiguous once entries mix, so they are compacted into `rows` first, then normed and projected once over exactly those rows.
inline void head(const HeadStep& s, const Weight& norm, const Weight& out, float eps, backend::Slice rows) {
    const size_t E = out.nin;
    s.b.gather_rows(rows, s.x, E, s.pick, s.want);
    s.b.rms_norm_rows(rows, rows, norm.slice(), s.want, E, E, eps);
    s.b.matmul_logits(out.type, out.slice(), rows, s.logits, out.nin, out.nout, s.want, s.head_runs);
}

} // namespace infer::blocks
