#pragma once
#include <cstddef>
#include <cstdint>

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
inline void swiglu(const Step& s, const Weight& gate, const Weight& up, const Weight& down,
                   backend::Slice h, backend::Slice g, backend::Slice u, backend::Slice act) {
    s.b.matmul_group({projection(gate, g), projection(up, u)}, h, gate.nin, s.rows, s.runs);
    s.b.silu_mul(act, g, u, s.rows * gate.nout, s.runs);
    s.b.matmul_add(down.type, down.slice(), act, s.x, down.nin, down.nout, s.rows, s.runs);
}

// The head: the rows that want logits are not contiguous once entries mix, so they are compacted into `rows` first, then normed and projected once over exactly those rows.
inline void head(const HeadStep& s, const Weight& norm, const Weight& out, float eps, backend::Slice rows) {
    const size_t E = out.nin;
    s.b.gather_rows(rows, s.x, E, s.pick, s.want);
    s.b.rms_norm_rows(rows, rows, norm.slice(), s.want, E, E, eps);
    s.b.matmul_logits(out.type, out.slice(), rows, s.logits, out.nin, out.nout, s.want, s.head_runs);
}

} // namespace infer::blocks
