#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "format/file_reader.hpp"
#include "format/gguf.hpp"
#include "model/arch/dflash.hpp"

// DeepSeek-V4.1's DSpark drafter (docs/DEEPSEEK4.md, section 4.7): a `dflash` sidecar paired into the target's blocks, its plan, its context rows from the taps and its block of drafts.
// model.hpp includes it after the class, whose members it defines.
namespace infer::deepseek4 {

// Stage `st` runs the trunk's roles under ids past the pass's and the earlier stages', and the drafter's own roles follow the last stage's.
inline uint16_t stage_role(size_t st, uint16_t role) { return uint16_t(role_count * (1 + st) + role); }
enum DraftRole : uint16_t { main_proj, main_norm, draft_norm, markov_embed, markov_head, draft_roles };
inline uint16_t draft_role(const Config& c, DraftRole r) { return uint16_t(role_count * (1 + size_t(c.n_nextn)) + r); }

// The drafter's arena slots after slot_widths': the taps, one plane of a pass's rows each, and the Markov head's rows.
enum : size_t { tap_slot = route_bias + 1, markov_slot = route_bias + 2 };

// DFly2's roles past its stages and its correction's attention, which takes a stage's role ids after the last stage (docs/SPECULATIVE.md, section 12).
enum Dfly2Role : uint16_t {
    fusion_proj, fusion_enorm, fusion_hnorm, fusion_gate, dfly2_norm, corr_hidden_norm, corr_embed_norm, corr_ffn_gate, corr_ffn_up, corr_ffn_down, corr_block_gate,
    corr_gate_bias, corr_rel_bias, corr_block_sinks, dfly2_roles
};
inline uint16_t dfly2_role(const Config& c, Dfly2Role r) { return uint16_t(role_count * (2 + size_t(c.n_nextn)) + r); }

// DFly2's slots after slot_widths': each target layer's picks of the rows, a plane a stage, the carried normed rows and picks (ModelPlan::draft_h), the block's final-normed rows,
// each correction row's previous token, and the correction's gates with their bias after them.
enum : size_t { pair_planes = route_bias + 1, carry_h, carry_picks, block_h, prev_ids, block_gates };

// A tap (model.py:1263-1265 at e5decd76): the mean of the streams entering the layer after its lookup, plane i of the taps' slot.
// The planes hold the pass's rows one after another, so a crossing of the pass's rows carries every plane.
inline void DeepSeek41::tap(const Step& s) const {
    const size_t H = size_t(cfg_.hc_count), E = size_t(cfg_.n_embd), n = backend::size_mul(s.rows, H);
    const size_t i = size_t(std::find(cfg_.taps.begin(), cfg_.taps.end(), s.layer) - cfg_.taps.begin());
    if (i == cfg_.taps.size()) throw std::logic_error("inference: a DeepSeek-V4.1 tap on a layer the drafter does not read");
    if (!s.indices || s.indices_size < n) throw std::runtime_error("inference: the DeepSeek-V4.1 tap's index scratch is too small");
    const backend::Slice plane = s.plane(tap_slot, i, E);
    std::fill(s.indices, s.indices + s.rows, 0u);
    s.b.gather_rows(plane, {s.tables[zeros].get(), 0}, E, s.indices, s.rows);
    static thread_local std::vector<float> weights;
    weights.assign(n, 1.0f / float(H));
    for (size_t r = 0; r < n; ++r) s.indices[r] = uint32_t(r);
    s.b.mix_rows(plane, s.x, E, n, s.indices, weights.data(), s.rows, H);
}

// The drafter's stages are trunk blocks of the window alone with their own expert counts and clamps, each run where the split puts it, and their window rings are three rings of one state slot.
inline void DeepSeek41::plan_drafter(const TensorIndex& tensors, ModelPlan& p) const {
    const Config& c = cfg_;
    if (!c.n_nextn) throw std::runtime_error("inference: the file carries no DSpark stages for an embedded drafter");
    const uint64_t E = uint64_t(c.n_embd), HC = uint64_t(c.hc_count), R = HC * E, MIX = (2 + HC) * HC;
    const uint64_t D = uint64_t(c.head_dim), Q = uint64_t(c.q_lora_rank), HD = uint64_t(c.n_head) * D, G = uint64_t(c.out_groups);
    const uint64_t GR = G * uint64_t(c.out_rank), F = uint64_t(c.n_ff_exp), X = uint64_t(c.draft_experts), W = uint64_t(c.window);
    const size_t stages = size_t(c.n_nextn), T = c.taps.size(), B = size_t(c.draft_block), used = size_t(c.draft_used);
    LayerPlan d;
    if (!c.dfly2) {
        d.cache = Cache::entries;
        d.headroom = size_t(W);
        d.state_tail = size_t(stages * 2 * W * D);
        d.native = LayerPlan::Widths{0, backend::size_mul(stages * 2 * size_t(W), window_words(D))};
    }
    const auto add = [&](uint16_t id, RoleKind kind, const std::string& name, uint64_t in, uint64_t out = 1, uint64_t experts = 0) {
        d.roles.push_back({id, Part::draft, kind, name, "", in, out, experts});
        tensors.at(name);
    };
    add(token_embd, RoleKind::gather, "token_embd.weight", E, p.vocab);
    add(output, RoleKind::matrix, "output.weight", E, p.vocab);
    for (size_t st = 0; st < stages; ++st) {
        const std::string pre = "blk." + std::to_string(size_t(c.n_layer) + st) + ".";
        const auto role = [&](uint16_t id, RoleKind kind, const char* name, uint64_t in, uint64_t out = 1, uint64_t experts = 0) {
            add(stage_role(st, id), kind, pre + name, in, out, experts);
        };
        role(attn_norm, RoleKind::norm, "attn_norm.weight", E);
        role(hc_attn_fn, RoleKind::table, "hc_attn_fn.weight", R, MIX);
        role(hc_attn_base, RoleKind::norm, "hc_attn_base.weight", MIX);
        role(hc_attn_scale, RoleKind::norm, "hc_attn_scale.weight", 3);
        role(attn_sinks, RoleKind::norm, "attn_sinks.weight", uint64_t(c.n_head));
        role(attn_q_a, RoleKind::matrix, "attn_q_a.weight", E, Q);
        role(attn_q_a_norm, RoleKind::norm, "attn_q_a_norm.weight", Q);
        role(attn_q_b, RoleKind::matrix, "attn_q_b.weight", Q, HD);
        role(attn_kv, RoleKind::matrix, "attn_kv.weight", E, D);
        role(attn_kv_a_norm, RoleKind::norm, "attn_kv_a_norm.weight", D);
        role(attn_output_a, RoleKind::matrix, "attn_output_a.weight", HD / G, GR);
        role(attn_output_b, RoleKind::matrix, "attn_output_b.weight", GR, E);
        role(ffn_norm, RoleKind::norm, "ffn_norm.weight", E);
        role(hc_ffn_fn, RoleKind::table, "hc_ffn_fn.weight", R, MIX);
        role(hc_ffn_base, RoleKind::norm, "hc_ffn_base.weight", MIX);
        role(hc_ffn_scale, RoleKind::norm, "hc_ffn_scale.weight", 3);
        role(ffn_gate_inp, RoleKind::matrix, "ffn_gate_inp.weight", E, X);
        role(exp_probs_b, RoleKind::norm, "exp_probs_b.bias", X);
        role(ffn_gate_exps, RoleKind::experts, "ffn_gate_exps.weight", E, F, X);
        role(ffn_up_exps, RoleKind::experts, "ffn_up_exps.weight", E, F, X);
        role(ffn_down_exps, RoleKind::experts, "ffn_down_exps.weight", F, E, X);
        role(ffn_gate_shexp, RoleKind::matrix, "ffn_gate_shexp.weight", E, F);
        role(ffn_up_shexp, RoleKind::matrix, "ffn_up_shexp.weight", E, F);
        role(ffn_down_shexp, RoleKind::matrix, "ffn_down_shexp.weight", F, E);
        blocks::routed_ops(d, tensors, stage_role(st, ffn_gate_exps), stage_role(st, ffn_up_exps));
        shard_block(d.roles, tensors, stage_role(st, 0), X);
        d.stages.push_back({stage_role(st, 0), stage_role(st + 1, 0)});
        if (c.swiglu_exp[size_t(c.n_layer) + st] != 0.0f || c.swiglu_shexp[size_t(c.n_layer) + st] != 0.0f) d.ops.push_back({Part::draft, backend::Op::silu_mul_clamp});
    }
    using backend::Op;
    for (Op op : {Op::hc_coeffs, Op::hc_collapse_norm, Op::hc_expand, Op::norm_rope_partial, Op::rope_forms, Op::model_format_qdq, Op::model_format_pack,
                  Op::sparse_attention_sink, Op::sqrt_softplus_routing, Op::embed_ids, Op::argmax_rows})
        d.ops.push_back({Part::draft, op});
    if (c.dfly2) return plan_dfly2(tensors, p, d);
    const std::string first = "blk." + std::to_string(c.n_layer) + ".nextn.", last = "blk." + std::to_string(size_t(c.n_layer) + stages - 1) + ".nextn.";
    const TensorView& markov = tensors.view(tensors.at(last + "markov_embed.weight"));
    if (markov.shape.empty() || !markov.shape[0]) throw std::runtime_error("inference: the DSpark Markov table has no rank");
    const uint64_t K = markov.shape[0];
    add(draft_role(c, main_proj), RoleKind::matrix, first + "main_proj.weight", T * E, E);
    add(draft_role(c, main_norm), RoleKind::norm, first + "main_norm.weight", E);
    add(draft_role(c, draft_norm), RoleKind::norm, last + "output_norm.weight", E);
    add(draft_role(c, markov_embed), RoleKind::gather, last + "markov_embed.weight", K, p.vocab);
    add(draft_role(c, markov_head), RoleKind::matrix, last + "markov_head.weight", K, p.vocab);
    // A group's member computes its vocabulary rows of the head and the Markov bias, and the members' candidates pick each draft (Model::draft_record).
    for (uint16_t id : {uint16_t(output), draft_role(c, markov_head)}) blocks::shard_vocab(d.roles, id, p.vocab);
    // A tap layer's mixer writes its plane, which crosses with the residual from the first tap on (ModelPlan::Carried).
    for (int t : c.taps) {
        p.layers[size_t(t)].kind |= writes_tap;
        p.layers[size_t(t)].ops.push_back({Part::mixer, Op::mix_rows});
    }
    p.slots.insert(p.slots.end(), {size_t(T * E), size_t(K)});
    p.carried.push_back({tap_slot, c.taps.front() + 1});
    // A draft's rows each read the window and the block; its experts and the taps' interleave may be wider than the trunk's slots.
    p.slots[window_ids] = size_t(W) + B;
    p.slots[head_x] = std::max(p.slots[head_x], size_t(T * E));
    p.slots[scores] = std::max(p.slots[scores], size_t(X));
    for (size_t slot : {picks, pick_weights}) p.slots[slot] = std::max(p.slots[slot], used);
    for (size_t slot : {gates, ups, acts}) p.slots[slot] = std::max(p.slots[slot], (used + 1) * size_t(F));
    p.index_width = std::max({p.index_width, size_t(W) + B, T});
    p.role_ids = draft_role(c, draft_roles);
    p.draft_block = B;
    p.draft_window = size_t(W);
    p.draft_sums = 2 * stages;
    // Apart from the head a draft's rows carry their streams' coefficients, its ids sit in the window's ids until its attention writes them, and the context rows' stages read the collapsed rows.
    p.draft_carried = {{pre_ffn}};
    p.draft_ids = window_ids;
    p.draft_rows_x = collapsed;
    p.drafter = std::move(d);
}

// The context rows of a pass (model.py:1035-1053 at e5decd76): the taps of each row side by side through main_proj and main_norm, then each stage's window row from it in that stage's ring at the row's position.
inline void DeepSeek41::draft_rows(const DraftRowsStep& s) const {
    const Config& c = cfg_;
    if (c.dfly2) return dfly2_rows(s);
    const size_t E = size_t(c.n_embd), T = c.taps.size(), rows = s.rows;
    const auto runs = [&](size_t part) { return s.phase < 0 || size_t(s.phase) == part; };
    if (runs(0)) {
        if (!s.indices || s.indices_size < backend::size_mul(rows, T)) throw std::runtime_error("inference: the DSpark context rows' index scratch is too small");
        for (size_t r = 0; r < rows; ++r)
            for (size_t t = 0; t < T; ++t) s.indices[r * T + t] = uint32_t(t * rows + r);
        s.b.gather_rows(s.slot(head_x), s.slot(tap_slot), E, s.indices, rows * T);
        const Weight& fc = s.w[draft_role(c, main_proj)];
        s.b.matmul(fc.operand(), s.slot(head_x), s.slot(collapsed), fc.nin, fc.nout, rows, s.runs, s.dtype);
        s.b.rms_norm_rows(s.slot(collapsed), s.slot(collapsed), s.w[draft_role(c, main_norm)].slice(), rows, E, E, c.rms_eps, s.runs);
    }
    for (size_t st = 0; st < size_t(c.n_nextn); ++st) {
        if (!runs(1 + st)) continue;
        const Rings r = rings(s);
        const Weight* w = s.w + stage_role(st, 0);
        s.b.matmul(w[attn_kv].operand(), s.slot(collapsed), s.slot(window_kv), w[attn_kv].nin, w[attn_kv].nout, rows, s.runs, s.dtype);
        window_rows(s, w);
        move_rings(s, r, st, true);
        for (size_t e = 0, row = 0; e < s.n_views; row += s.states[e].nq, ++e) store_rows(s, r, s.states[e], st, row, s.states[e].nq);
    }
}

// A block of drafts for each sequence (model.py:1131-1165 at e5decd76): the last pick then the noise token at the next positions through every stage, each row reading the window before the block and the whole block.
// The target's head reads the drafter's final norm, and each row's Markov bias from its input id comes before its argmax picks the next row's input; a group's parts are each stage's two sublayers, then a part a draft (DraftStep).
inline void DeepSeek41::draft(const DraftStep& s) const {
    const Config& c = cfg_;
    if (c.dfly2) return dfly2_draft(s);
    const size_t E = size_t(c.n_embd), H = size_t(c.hc_count), R = s.rows, B = s.ids, n = backend::size_mul(R, B), sums = 2 * size_t(c.n_nextn);
    if (s.stride != R || (s.phase >= 0 && size_t(s.phase) >= sums + B)) throw std::logic_error("inference: a DSpark draft is one call over its sequences");
    if (!s.indices || s.indices_size < backend::size_mul(n, H)) throw std::runtime_error("inference: the DSpark draft's index scratch is too small");
    const auto runs = [&](size_t part) { return s.phase < 0 || size_t(s.phase) == part; };
    const backend::RowRun run{n, 1};
    Step x = s;
    x.rows = n;
    x.runs = {&run, 1};
    const backend::Slice rows = s.slot(collapsed);
    using Lookup = DraftStep::Lookup;
    // The table's rows, the ids' then the noise token's, R + 1 rows of `collapsed`, which is ModelPlan::draft_rows_x and crosses where the table sits apart.
    if (runs(0) && s.lookup != Lookup::crossed) {
        const Weight& table = s.w[token_embd];
        s.b.embed_ids(rows, table.operand(), E, table.nout, s.id, R);
        s.b.embed({rows.buffer, rows.offset + R * E}, table.operand(), E, table.nout, &c.noise_id, 1);
    }
    if (s.lookup == Lookup::only) return;
    if (runs(0)) {
        for (size_t i = 0; i < n * H; ++i) s.indices[i] = (i / H) % B ? uint32_t(R) : uint32_t(i / H / B);
        s.b.gather_rows(x.x, rows, E, s.indices, n * H);
        std::fill(s.indices, s.indices + n, 0u);
        s.b.gather_rows(s.slot(pre_ffn), {s.tables[first_pre].get(), 0}, H, s.indices, n);
    }
    for (size_t st = 0; st < size_t(c.n_nextn); ++st) {
        const Weight* w = s.w + stage_role(st, 0);
        if (runs(2 * st)) attend(x, w, {st, B, true});
        if (runs(2 * st + 1)) ffn_body(x, w, size_t(c.draft_used), c.swiglu_exp[size_t(c.n_layer) + st], c.swiglu_shexp[size_t(c.n_layer) + st],
                                         backend::RouteRule::sqrt_softplus(w[exp_probs_b].slice(), c.expert_scale, true));
    }
    // The logits of draft j of sequence i in row j * R + i of the member's vocabulary rows, then each draft's Markov bias and pick in order.
    const Weight &out = s.w[output], &m1 = s.w[draft_role(c, markov_embed)], &m2 = s.w[draft_role(c, markov_head)];
    const size_t V = out.nout;
    if (runs(sums)) {
        s.b.hc_collapse_norm(rows, x.x, s.slot(pre_ffn), s.w[draft_role(c, draft_norm)].slice(), n, H, E, c.rms_eps);
        for (size_t i = 0; i < n; ++i) s.indices[i] = uint32_t(i % R * B + i / R);
        s.b.gather_rows(s.slot(block_out), rows, E, s.indices, n);
        s.b.matmul_logits(out.operand(), s.slot(block_out), s.logits, E, V, n, x.runs, s.dtype);
    }
    for (size_t j = 0; j < B; ++j) {
        if (!runs(sums + j)) continue;
        const backend::CSlice ids = j ? backend::CSlice{s.next.buffer, s.next.offset + (j - 1) * R} : s.id;
        const backend::Slice logits{s.logits.buffer, s.logits.offset + j * R * V};
        s.b.embed_ids(s.slot(markov_slot), m1.operand(), m1.nin, m1.nout, ids, R);
        s.b.matmul_add(m2.operand(), s.slot(markov_slot), logits, m2.nin, V, R, {}, s.dtype);
        if (s.width > 1) s.b.argmax_part(s.partial, logits, R, V, s.first, s.member, s.width);
        else s.b.argmax_rows({s.next.buffer, s.next.offset + j * R}, logits, R, V, ids);
    }
}

// A DFly2 target layer's picks of each row, plane st of pair_planes for its stage, T = index_top_k ids a row before UINT32_MAX: the indexer's where it listed T, else listed on the host.
// Where every visible entry is picked a row reads its first (position + 1) / ratio entries in order (picks_all), as the layer's own read does.
inline void DeepSeek41::pair_picks(const Step& s) const {
    const Config& c = cfg_;
    const size_t st = size_t(std::find(c.taps.begin(), c.taps.end(), s.layer) - c.taps.begin()), T = size_t(c.index_top_k), rows = s.rows;
    if (st == c.taps.size() || !s.views) throw std::logic_error("inference: DFly2 picks on a layer the drafter does not read");
    const size_t ratio = size_t(c.compress_ratios.at(size_t(s.layer)));
    size_t visible = 0;
    for (size_t e = 0; e < s.n_views; ++e) visible = std::max(visible, (s.views[e].length + s.views[e].nq) / ratio);
    const size_t k = std::min(T, visible);
    const bool listed = k && !picks_all(visible, ratio);
    const backend::Slice plane = s.plane(pair_planes, st, T), picked = s.slot(index_picks);
    if (listed && k == T) return s.b.copy(*plane.buffer, plane.offset * sizeof(float), *picked.buffer, picked.offset * sizeof(float), rows * T * sizeof(float));
    static thread_local std::vector<uint32_t> ids;
    ids.assign(rows * T, UINT32_MAX);
    // Entries count by a row's index in its history, not its rope position, which an image holds back (the trunk's own read, sparse_attention_sink).
    for (size_t e = 0, r = 0; e < s.n_views && !listed; r += s.views[e].nq, ++e)
        for (size_t i = 0; i < s.views[e].nq; ++i)
            for (size_t j = 0, n = std::min(T, (s.views[e].length + i + 1) / ratio); j < n; ++j) ids[(r + i) * T + j] = uint32_t(j);
    s.b.write(*plane.buffer, plane.offset * sizeof(float), ids.data(), ids.size() * sizeof(uint32_t));
    for (size_t r = 0; r < rows && listed; ++r)
        s.b.copy(*plane.buffer, (plane.offset + r * T) * sizeof(float), *picked.buffer, (picked.offset + r * k) * sizeof(float), k * sizeof(float));
}

// DFly2's plan past the stages plan_drafter gave `d`: no cache of its own, each stage reading its target layer's ring and picks as they stand (ModelPlan::draft_pairs) in the compressed rope.
// Then the fusion of the anchor with the target's normed row before it, the final norm and the correction, over a block of anchor 1 whose rows see the whole block.
inline void DeepSeek41::plan_dfly2(const TensorIndex& tensors, ModelPlan& p, LayerPlan& d) const {
    const Config& c = cfg_;
    const uint64_t E = uint64_t(c.n_embd), D = uint64_t(c.head_dim), Q = uint64_t(c.q_lora_rank), heads = uint64_t(c.n_head), HD = heads * D, G = uint64_t(c.out_groups);
    const uint64_t GR = G * uint64_t(c.out_rank), F = uint64_t(c.n_ff_exp), X = uint64_t(c.draft_experts);
    const size_t stages = size_t(c.n_nextn), A = size_t(c.draft_block), W = size_t(c.window), T = size_t(c.index_top_k), used = size_t(c.draft_used), H = size_t(c.hc_count);
    const auto add = [&](uint16_t id, RoleKind kind, const std::string& name, uint64_t in, uint64_t out = 1) {
        d.roles.push_back({id, Part::draft, kind, name, "", in, out, 0});
        tensors.at(name);
    };
    const auto width = [&](const std::string& name, size_t axis) {
        const TensorView& v = tensors.view(tensors.at(name));
        if (v.shape.size() <= axis || !v.shape[axis]) throw std::runtime_error("inference: the DFly2 tensor " + name + " has no such axis");
        return v.shape[axis];
    };
    if (A < 2 || !T) throw std::runtime_error("inference: a DFly2 block holds the anchor and at least one draft, read beside the target's picks");
    const std::string first = "blk." + std::to_string(c.n_layer) + ".nextn.", last = "blk." + std::to_string(size_t(c.n_layer) + stages - 1) + ".nextn.";
    const uint64_t CF = width(last + "corr.ffn_gate.weight", 1), S = width(last + "corr.block_rel_bias.weight", 0);
    if (S < A) throw std::runtime_error("inference: the DFly2 block is wider than its correction's place bias");
    if (width(last + "corr.block_gate_bias.weight", 0) != S) throw std::runtime_error("inference: the DFly2 correction's gate bias and place bias differ in places");
    d.cache = Cache::none;
    d.kind = reads_compressed;
    add(output_norm, RoleKind::norm, "output_norm.weight", E);
    add(dfly2_role(c, fusion_proj), RoleKind::matrix, first + "fusion.eh_proj.weight", 2 * E, E);
    add(dfly2_role(c, fusion_enorm), RoleKind::norm, first + "fusion.enorm.weight", E);
    add(dfly2_role(c, fusion_hnorm), RoleKind::norm, first + "fusion.hnorm.weight", E);
    add(dfly2_role(c, fusion_gate), RoleKind::norm, first + "fusion.gate.weight", E);
    add(dfly2_role(c, dfly2_norm), RoleKind::norm, last + "output_norm.weight", E);
    const auto corr = [&](uint16_t role, RoleKind kind, const char* name, uint64_t in, uint64_t out = 1) { add(stage_role(stages, role), kind, last + "corr." + name, in, out); };
    corr(attn_q_a, RoleKind::matrix, "attn_q_a.weight", 2 * E, Q);
    corr(attn_q_a_norm, RoleKind::norm, "attn_q_a_norm.weight", Q);
    corr(attn_q_b, RoleKind::matrix, "attn_q_b.weight", Q, HD);
    corr(attn_kv, RoleKind::matrix, "attn_kv.weight", 2 * E, D);
    corr(attn_kv_a_norm, RoleKind::norm, "attn_kv_a_norm.weight", D);
    corr(attn_sinks, RoleKind::norm, "attn_sinks.weight", heads);
    corr(attn_output_a, RoleKind::matrix, "attn_output_a.weight", HD / G, GR);
    corr(attn_output_b, RoleKind::matrix, "attn_output_b.weight", GR, E);
    const auto own = [&](Dfly2Role role, RoleKind kind, const char* name, uint64_t in, uint64_t out = 1) { add(dfly2_role(c, role), kind, last + "corr." + name, in, out); };
    own(corr_hidden_norm, RoleKind::norm, "hidden_norm.weight", E);
    own(corr_embed_norm, RoleKind::norm, "embed_norm.weight", E);
    own(corr_ffn_gate, RoleKind::matrix, "ffn_gate.weight", 2 * E, CF);
    own(corr_ffn_up, RoleKind::matrix, "ffn_up.weight", 2 * E, CF);
    own(corr_ffn_down, RoleKind::matrix, "ffn_down.weight", CF, E);
    own(corr_block_gate, RoleKind::matrix, "block_gate.weight", 2 * E, heads);
    own(corr_gate_bias, RoleKind::table, "block_gate_bias.weight", S, heads);
    own(corr_rel_bias, RoleKind::table, "block_rel_bias.weight", S, heads);
    own(corr_block_sinks, RoleKind::norm, "block_sinks.weight", heads);
    for (backend::Op op : {backend::Op::add_bias, backend::Op::tap_weighted_rows}) d.ops.push_back({Part::draft, op});
    // Each target layer leaves its rows' picks for the drafter after its attention (DeepSeek41::mixer).
    for (int t : c.taps) p.layers[size_t(t)].kind |= writes_tap;
    if (p.slots.size() != pair_planes) throw std::logic_error("inference: DeepSeek-V4.1's slots are not as the DFly2 plan expects");
    p.slots.insert(p.slots.end(), {stages * T, p.residual, stages * T, size_t(E), 1, 2 * size_t(heads)});
    // A block's rows each read the window, then their block apart; the correction's rows are the normed hidden and token rows side by side.
    p.slots[window_ids] = W + A + size_t(S);
    for (size_t slot : {size_t(head_x), size_t(collapsed)}) p.slots[slot] = std::max(p.slots[slot], size_t(2 * E));
    p.slots[scores] = std::max(p.slots[scores], size_t(X));
    for (size_t slot : {size_t(picks), size_t(pick_weights)}) p.slots[slot] = std::max(p.slots[slot], used);
    for (size_t slot : {size_t(gates), size_t(ups), size_t(acts)}) p.slots[slot] = std::max({p.slots[slot], (used + 1) * size_t(F), size_t(CF)});
    p.index_width = std::max({p.index_width, W + A, size_t(S), H, stages, size_t(heads), size_t(2)});
    p.role_ids = dfly2_role(c, dfly2_roles);
    p.draft_block = A - 1;
    p.draft_anchor = 1;
    p.draft_causal = false;
    p.draft_window = W;
    p.draft_sums = 2 * stages;
    p.draft_pairs.assign(c.taps.begin(), c.taps.end());
    // A split carries the target layers' picks to the head, and a draft's rows cross between its stages' devices with their streams' coefficients and picks.
    p.carried.push_back({pair_planes, c.taps.front() + 1});
    p.draft_carried = {{pre_ffn}, {pair_planes}};
    p.draft_ids = prev_ids;
    p.draft_rows_x = collapsed;
    p.draft_h = carry_h;
    p.draft_picks = stages * T;
    p.draft_picks_slot = carry_picks;
    p.drafter = std::move(d);
}

// DFly2's context rows of a pass: the target's final-normed rows, each in its residual-wide row of carry_h, and each row's picks of every target layer side by side in carry_picks.
// Model::draft_context carries each entry's last row of both into its slot's record.
inline void DeepSeek41::dfly2_rows(const DraftRowsStep& s) const {
    const Config& c = cfg_;
    const size_t E = size_t(c.n_embd), H = size_t(c.hc_count), stages = size_t(c.n_nextn), T = size_t(c.index_top_k), rows = s.rows;
    if (s.phase > 0) throw std::logic_error("inference: DFly2's context rows run on the head's device");
    if (!s.indices || s.indices_size < backend::size_mul(rows, std::max(H, stages))) throw std::runtime_error("inference: the DFly2 context rows' index scratch is too small");
    s.b.hc_collapse_norm(s.slot(collapsed), s.x, s.slot(pre_ffn), s.w[output_norm].slice(), rows, H, E, c.rms_eps);
    for (size_t q = 0; q < rows * H; ++q) s.indices[q] = uint32_t(q / H);
    s.b.gather_rows(s.slot(carry_h), s.slot(collapsed), E, s.indices, rows * H);
    for (size_t r = 0; r < rows; ++r)
        for (size_t st = 0; st < stages; ++st) s.indices[r * stages + st] = uint32_t(st * rows + r);
    s.b.gather_rows(s.slot(carry_picks), s.plane(pair_planes, 0, T), T, s.indices, rows * stages);
}

// A DFly2 block for each sequence (docs/SPECULATIVE.md, section 12): the anchor fused with the target's normed row before it, then mask rows, through the stages, each reading its pair's window and picks.
// Then the final norm and the correction, pass j giving draft j from block row j + 1; rows are sequence-major, A = ids + 1 a sequence, and parts are the stages' sublayers, then a pass a draft.
inline void DeepSeek41::dfly2_draft(const DraftStep& s) const {
    const Config& c = cfg_;
    const size_t E = size_t(c.n_embd), H = size_t(c.hc_count), R = s.rows, B = s.ids, A = B + 1, n = backend::size_mul(R, A), stages = size_t(c.n_nextn), sums = 2 * stages;
    const size_t T = size_t(c.index_top_k), heads = size_t(c.n_head), residual = H * E, carried = residual + stages * T;
    if (s.width > 1) throw std::runtime_error("inference: the DFly2 drafter does not run on a tensor group yet");
    if (s.stride != R || A != size_t(c.draft_block) || (s.phase >= 0 && size_t(s.phase) >= sums + B) || !s.prev_rows)
        throw std::logic_error("inference: a DFly2 draft is one block call over its sequences, reading their carried rows");
    using Lookup = DraftStep::Lookup;
    // The fusion (Lookup::only where stage 0 runs apart, on the head with the carried rows) and the correction read the ids; a stage apart reads none.
    const bool fuses = (s.phase <= 0 && s.lookup != Lookup::crossed), corrects = s.phase < 0 || size_t(s.phase) >= sums;
    if ((fuses || corrects) && (s.next.buffer != s.id.buffer || s.next.offset != s.id.offset + R)) throw std::logic_error("inference: DFly2's drafts must follow their anchors' ids");
    if (!s.indices || s.indices_size < std::max(backend::size_mul(n, H), backend::size_mul(A, heads))) throw std::runtime_error("inference: the DFly2 draft's index scratch is too small");
    for (size_t e = 0; s.states && e < s.n_views; ++e)
        if (s.states[e].nq != A) throw std::logic_error("inference: a DFly2 block's views hold its anchor and drafts");
    const auto runs = [&](size_t part) { return s.phase < 0 || size_t(s.phase) == part; };
    const backend::RowRun run{n, 1}, lead{R, 1};
    Step x = s;
    x.rows = n;
    x.runs = {&run, 1};
    const Weight& table = s.w[token_embd];
    const size_t V = table.nout;
    const backend::Slice staged = s.slot(head_x), cat = s.slot(collapsed), rows = s.slot(block_out);
    const auto w = [&](Dfly2Role r) -> const Weight& { return s.w[dfly2_role(c, r)]; };
    if (runs(0) && fuses) {
        // The anchors' rows, then each sequence's carried normed row, side by side and normed, through eh_proj; the gated sum joins the anchor's row (dfly2_reference.py, draft).
        s.b.embed_ids(staged, table.operand(), E, V, s.id, R);
        for (size_t i = 0; i < R; ++i) {
            const size_t from = s.prev.offset + size_t(s.prev_rows[i]) * carried;
            s.b.copy(*staged.buffer, (staged.offset + (R + i) * E) * sizeof(float), *s.prev.buffer, from * sizeof(float), E * sizeof(float));
            for (size_t st = 0; st < stages; ++st) {
                const backend::Slice plane = x.plane(pair_planes, st, T);
                for (size_t j = 0; j < A; ++j)
                    s.b.copy(*plane.buffer, (plane.offset + (i * A + j) * T) * sizeof(float), *s.prev.buffer, (from + residual + st * T) * sizeof(float), T * sizeof(float));
            }
        }
        for (size_t i = 0; i < R; ++i) s.indices[2 * i] = uint32_t(i), s.indices[2 * i + 1] = uint32_t(R + i);
        s.b.gather_rows(cat, staged, E, s.indices, 2 * R);
        s.b.rms_norm_rows(cat, cat, w(fusion_enorm).slice(), R, E, 2 * E, c.rms_eps);
        s.b.rms_norm_rows(at(cat, E), at(cat, E), w(fusion_hnorm).slice(), R, E, 2 * E, c.rms_eps);
        const Weight& proj = w(fusion_proj);
        s.b.matmul(proj.operand(), cat, staged, proj.nin, proj.nout, R, {&lead, 1}, s.dtype);
        s.b.embed_ids(rows, table.operand(), E, V, s.id, R);
        s.b.embed(at(rows, R * E), table.operand(), E, V, &c.noise_id, 1);
        for (size_t i = 0; i < R; ++i) s.indices[i] = uint32_t(i);
        s.b.tap_weighted_rows(rows, staged, E, s.indices, R, 1, w(fusion_gate).slice(), {s.tables[zeros].get(), 0}, 1, E, true);
        // Every stream of a row starts as its block row, and its first pre is the one-hot the trunk's first block reads.
        for (size_t q = 0; q < n * H; ++q) s.indices[q] = (q / H) % A ? uint32_t(R) : uint32_t(q / H / A);
        s.b.gather_rows(x.x, rows, E, s.indices, n * H);
        std::fill(s.indices, s.indices + n, 0u);
        s.b.gather_rows(s.slot(pre_ffn), {s.tables[first_pre].get(), 0}, H, s.indices, n);
    }
    if (s.lookup == Lookup::only) return;
    for (size_t st = 0; st < stages; ++st) {
        const Weight* ws = s.w + stage_role(st, 0);
        if (runs(2 * st)) attend(x, ws, {0, A, false, true, x.plane(pair_planes, st, T), T});
        if (runs(2 * st + 1)) ffn_body(x, ws, size_t(c.draft_used), c.swiglu_exp[size_t(c.n_layer) + st], c.swiglu_shexp[size_t(c.n_layer) + st],
                                         backend::RouteRule::sqrt_softplus(ws[exp_probs_b].slice(), c.expert_scale, true));
    }
    const Weight* wc = s.w + stage_role(stages, 0);
    const Weight& head = s.w[output];
    const size_t S = w(corr_rel_bias).nin;
    const backend::Slice read_gates = s.slot(block_gates), bias = at(read_gates, n * heads);
    const BlockRead block{w(corr_block_sinks).slice(), w(corr_rel_bias).slice(), read_gates, S};
    for (size_t j = 0; j < B; ++j) {
        if (!runs(sums + j)) continue;
        // The final-normed rows h and their correction norm, once; row k's previous token is the anchor for k of 0 or 1, else draft k - 2, which pass k - 2 picked.
        if (!j) {
            s.b.hc_collapse_norm(s.slot(block_h), x.x, s.slot(pre_ffn), w(dfly2_norm).slice(), n, H, E, c.rms_eps);
            s.b.rms_norm_rows(staged, s.slot(block_h), w(corr_hidden_norm).slice(), n, E, E, c.rms_eps);
        }
        for (size_t i = 0; i < R; ++i)
            for (size_t k = 0; k < A; ++k) s.indices[i * A + k] = k < 2 || k > j + 1 ? uint32_t(i) : uint32_t((k - 1) * R + i);
        s.b.gather_rows(s.slot(prev_ids), s.id, 1, s.indices, n);
        s.b.embed_ids(at(staged, n * E), table.operand(), E, V, s.slot(prev_ids), n);
        s.b.rms_norm_rows(at(staged, n * E), at(staged, n * E), w(corr_embed_norm).slice(), n, E, E, c.rms_eps);
        for (size_t r = 0; r < n; ++r) s.indices[2 * r] = uint32_t(r), s.indices[2 * r + 1] = uint32_t(n + r);
        s.b.gather_rows(cat, staged, E, s.indices, 2 * n);
        // The block read's gate, tanh of its projection plus the bias of the row's place in the block, the bias moved to places x heads first.
        const Weight& gate = w(corr_block_gate);
        s.b.matmul(gate.operand(), cat, read_gates, gate.nin, gate.nout, n, x.runs, s.dtype);
        for (size_t k = 0; k < A; ++k)
            for (size_t h = 0; h < heads; ++h) s.indices[k * heads + h] = uint32_t(h * S + k);
        s.b.gather_rows(bias, w(corr_gate_bias).slice(), 1, s.indices, A * heads);
        s.b.add_bias(read_gates, bias, R, A * heads);
        window_attention(x, wc, {0, A, false, true, x.plane(pair_planes, stages - 1, T), T, &block});
        // h + the SwiGLU delta + the attention's output, then the target's head over row j + 1 alone.
        Step y = x;
        y.x = rows;
        blocks::swiglu(y, w(corr_ffn_gate), w(corr_ffn_up), w(corr_ffn_down), cat, s.slot(gates), s.slot(ups), s.slot(acts));
        s.b.add(rows, s.slot(block_h), n * E);
        for (size_t i = 0; i < R; ++i) s.indices[i] = uint32_t(i * A + j + 1);
        s.b.gather_rows(s.slot(query), rows, E, s.indices, R);
        const backend::Slice logits = at(s.logits, j * R * V);
        s.b.matmul_logits(head.operand(), s.slot(query), logits, E, V, R, {&lead, 1}, s.dtype);
        s.b.argmax_rows(at(s.next, j * R), logits, R, V, j ? backend::CSlice{s.next.buffer, s.next.offset + (j - 1) * R} : s.id);
    }
}

// A DSpark sidecar, `general.architecture` dflash, rewritten into the target's blocks after tokenizer equality (docs/DEEPSEEK4.md, section 4.7); any other drafter file is left as it is.
// Its stages follow the target's layers with `nextn` globals and keys, its experts not the target's (F14, F21), and a DFly2 sidecar (docs/SPECULATIVE.md, section 12) joins the same way.
inline void normalize_drafter(const gguf::GGUFModel& target, gguf::GGUFModel& side) {
    if (!dflash::sidecar(side)) return;
    const std::string p = "deepseek41.", q = "dflash.";
    const auto* kind = side.find(q + "kind");
    const bool dfly2 = kind && kind->vtype == gguf::V_STRING && kind->s == "dfly2";
    const std::string name = dfly2 ? "DFly2" : "DSpark";
    const auto refuse = [&](const std::string& text) { return std::runtime_error("inference: " + name + " sidecar " + text); };
    if (kind && !dfly2) throw refuse("of unsupported dflash.kind");
    const int layers = metadata::integer(target, p + "block_count");
    if (metadata::count(target, p + "nextn_predict_layers", 0)) throw refuse("beside a target that carries drafter stages");
    const int stages = metadata::integer(side, q + "block_count");
    // Every key it shares with the target the target's, but the counts and clamps its stages have of their own; DFly2's keys under other names are checked below.
    std::set<std::string> own{"block_count", "expert_count", "expert_used_count", "swiglu_clamp_exp", "swiglu_clamp_shexp", "attention.compress_ratios",
                              "block_size", "target_layers", "kind", "attention.causal", "target_kv_layers", "target_index_layers"};
    if (dfly2) own.insert({"feed_forward_length", "rope.freq_base", "hyper_connection.single_pass", "sample_from_anchor", "has_confidence_head"});
    const gguf::MetaValue noise = dflash::check_sidecar(target, side, p, own, name);
    if (dfly2) {
        if (metadata::boolean(side, q + "attention.causal", true)) throw refuse("must attend over its block non-causally");
        if (metadata::counts(side, q + "target_kv_layers") != metadata::counts(side, q + "target_index_layers"))
            throw refuse("must read each stage's latents and picks from one target layer");
        // The converter writes the stages' expert width again, and the base of the compressed rope that the picked target KV and the drafter's own attention share.
        const std::pair<const char*, const char*> restated[] = {{"feed_forward_length", "expert_feed_forward_length"}, {"rope.freq_base", "attention.compress_rope_freq_base"}};
        for (const auto& [key, of] : restated) {
            const auto *v = side.find(q + key), *t = target.find(p + of);
            if (v && (!t || !gguf::detail::equal_value(*t, *v))) throw refuse("metadata " + q + key + " differs from the target's " + p + of);
        }
        if (!metadata::boolean(side, q + "hyper_connection.single_pass", false)) throw refuse("must mix its hyper-connections in one pass, as DeepSeek-V4.1's stages do");
        if (metadata::boolean(side, q + "sample_from_anchor", true)) throw refuse("must draft past its anchor, the block's first row");
        if (metadata::boolean(side, q + "has_confidence_head", false)) throw refuse("carries a confidence head the drafter does not run");
    } else if (side.find(q + "target_kv_layers") || side.find(q + "target_index_layers") || side.find(q + "attention.causal")) {
        throw refuse("carries DFly2 keys without dflash.kind dfly2");
    }
    for (int r : metadata::counts(side, q + "attention.compress_ratios"))
        if (r) throw refuse("stage reads compressed entries");

    std::vector<std::pair<std::string, gguf::MetaValue>> kv;
    for (const auto& item : side.kv)
        if (item.first.compare(0, q.size(), q)) kv.push_back(item);
    for (const auto& item : target.kv)
        if (!item.first.compare(0, p.size(), p)) kv.push_back(item);
    const auto value = [&](const std::string& key) -> gguf::MetaValue& {
        for (auto& item : kv)
            if (item.first == key) return item.second;
        kv.push_back({key, {}});
        return kv.back().second;
    };
    const auto number = [&](const std::string& key, uint64_t n) {
        gguf::MetaValue& v = value(key);
        v = {};
        v.vtype = gguf::V_UINT32;
        v.u = n;
    };
    const auto copy = [&](const std::string& key, const std::string& from) {
        const auto* v = side.find(from);
        if (!v) throw refuse("lacks " + from);
        value(key) = *v;
    };
    value("general.architecture").s = "deepseek41";
    number(p + "block_count", uint64_t(layers + stages));
    number(p + "nextn_predict_layers", uint64_t(stages));
    for (const char* key : {"swiglu_clamp_exp", "swiglu_clamp_shexp"}) {
        gguf::MetaValue& v = value(p + key);
        const auto* extra = side.find(q + key);
        if (v.vtype != gguf::V_ARRAY || !extra || extra->vtype != gguf::V_ARRAY || extra->arr.size() != size_t(stages)) throw refuse("clamps " + std::string(key));
        v.arr.insert(v.arr.end(), extra->arr.begin(), extra->arr.end());
    }
    copy(p + "nextn.expert_count", q + "expert_count");
    copy(p + "nextn.expert_used_count", q + "expert_used_count");
    copy(p + "nextn.block_size", q + "block_size");
    copy(p + "nextn.target_layers", q + (dfly2 ? "target_kv_layers" : "target_layers"));
    value(p + "nextn.noise_token_id") = noise;
    if (dfly2) value(p + "nextn.kind") = *kind;

    // Its stages after the target's layers, and its globals in the first and last stage.
    const std::string first = "blk." + std::to_string(layers) + ".nextn.", last = "blk." + std::to_string(layers + stages - 1) + ".nextn.";
    std::vector<std::pair<std::string, std::string>> globals{{"output_norm.weight", last + "output_norm.weight"}};
    if (!dfly2)
        globals.insert(globals.end(), {{"fc.weight", first + "main_proj.weight"},
                                       {"enc.output_norm.weight", first + "main_norm.weight"},
                                       {"markov_w1.weight", last + "markov_embed.weight"},
                                       {"markov_w2.weight", last + "markov_head.weight"},
                                       {"conf_proj.weight", last + "confidence.weight"}});
    std::vector<std::string> names;
    for (const auto& tensor : side.tensors) {
        if (const auto block = gguf::block_of(tensor.name)) {
            if (*block >= size_t(stages)) throw refuse("tensor " + tensor.name + " is past its stages");
            const size_t dot = tensor.name.find('.', 4);
            names.push_back("blk." + std::to_string(size_t(layers) + *block) + tensor.name.substr(dot));
            continue;
        }
        static const std::set<std::string> own_globals{
            "fusion.eh_proj.weight",         "fusion.enorm.weight",         "fusion.hnorm.weight",       "fusion.gate.weight",       "corr.hidden_norm.weight",
            "corr.embed_norm.weight",        "corr.ffn_gate.weight",        "corr.ffn_up.weight",        "corr.ffn_down.weight",     "corr.attn_q_a.weight",
            "corr.attn_q_a_norm.weight",     "corr.attn_q_b.weight",        "corr.attn_kv.weight",       "corr.attn_kv_a_norm.weight", "corr.attn_sinks.weight",
            "corr.attn_output_a.weight",     "corr.attn_output_b.weight",   "corr.block_gate.weight",    "corr.block_gate_bias.weight", "corr.block_rel_bias.weight",
            "corr.block_sinks.weight"};
        if (dfly2 && own_globals.count(tensor.name)) {
            names.push_back((tensor.name.compare(0, 7, "fusion.") ? last : first) + tensor.name);
            continue;
        }
        const auto g = std::find_if(globals.begin(), globals.end(), [&](const auto& m) { return m.first == tensor.name; });
        if (g == globals.end()) throw refuse("unsupported tensor " + tensor.name);
        names.push_back(g->second);
    }
    // Validate the joined geometry before any header changes.
    gguf::GGUFModel header;
    header.kv = kv;
    uint32_t vocab = 0;
    for (const auto& tensor : target.tensors)
        if (tensor.name == "token_embd.weight" && tensor.ne.size() >= 2) vocab = uint32_t(tensor.ne[1]);
    (void)read_config(header, p, vocab);

    // A stage's experts that begin as the target's same block's do were copied from the trunk (the converter's F21): the first bytes of each stack are compared.
    const auto sample = [](const gguf::GGUFModel& m, size_t i, size_t n) {
        std::vector<uint8_t> out(std::min(n, m.tensor_bytes(i)));
        if (m.segments.empty()) {
            std::copy_n(m.blob.data() + m.offsets[i], out.size(), out.begin());
        } else {
            const format::FileSpan span = m.span(i);
            if (format::FileReader(span.file).read(span.offset, out.data(), out.size()) != out.size()) throw std::runtime_error("inference: short read of " + span.file);
        }
        return out;
    };
    for (size_t i = 0; i < side.tensors.size(); ++i) {
        const auto block = gguf::block_of(side.tensors[i].name);
        if (!block || side.tensors[i].name.find("_exps.") == std::string::npos) continue;
        const std::string twin = "blk." + std::to_string(*block) + side.tensors[i].name.substr(side.tensors[i].name.find('.', 4));
        for (size_t j = 0; j < target.tensors.size(); ++j) {
            if (target.tensors[j].name != twin || target.tensors[j].type != side.tensors[i].type) continue;
            const size_t n = std::min({size_t(65536), target.tensor_bytes(j), side.tensor_bytes(i)});
            if (sample(target, j, n) == sample(side, i, n)) throw refuse("experts " + side.tensors[i].name + " are the target's " + twin);
        }
    }
    side.kv.swap(kv);
    for (size_t i = 0; i < names.size(); ++i) side.tensors[i].name.swap(names[i]);
}

} // namespace infer::deepseek4
