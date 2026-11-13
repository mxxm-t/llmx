#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "format/file_reader.hpp"
#include "format/gguf.hpp"

// DeepSeek-V4.1's DSpark drafter (docs/DEEPSEEK4.md, section 4.7): a `dflash` sidecar paired into the target's blocks, its plan, its context rows from the taps and its block of drafts.
// model.hpp includes it after the class, whose members it defines.
namespace infer::deepseek4 {

// Stage `st` runs the trunk's roles under ids past the pass's and the earlier stages', and the drafter's own roles follow the last stage's.
inline uint16_t stage_role(size_t st, uint16_t role) { return uint16_t(role_count * (1 + st) + role); }
enum DraftRole : uint16_t { main_proj, main_norm, draft_norm, markov_embed, markov_head, draft_roles };
inline uint16_t draft_role(const Config& c, DraftRole r) { return uint16_t(role_count * (1 + size_t(c.n_nextn)) + r); }

// The drafter's arena slots after slot_widths': the taps, one plane of a pass's rows each, and the Markov head's rows.
enum : size_t { tap_slot = candidate_ids + 1, markov_slot = candidate_ids + 2 };

// A tap (model.py:1263-1265 at e5decd76): the mean of the streams entering the layer after its lookup, plane i of the taps' slot.
// The planes hold the pass's rows one after another, so a crossing of the pass's rows carries every plane.
inline void DeepSeek41::tap(const Step& s) const {
    const size_t H = size_t(cfg_.hc_count), E = size_t(cfg_.n_embd), n = backend::size_mul(s.rows, H);
    const size_t i = size_t(std::find(cfg_.taps.begin(), cfg_.taps.end(), s.layer) - cfg_.taps.begin());
    if (i == cfg_.taps.size()) throw std::logic_error("inference: a DeepSeek-V4.1 tap on a layer the drafter does not read");
    if (!s.indices || s.indices_size < n) throw std::runtime_error("inference: the DeepSeek-V4.1 tap's index scratch is too small");
    backend::Slice plane = s.slot(tap_slot);
    plane.offset += backend::size_mul(i * s.rows, E);
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
    d.cache = Cache::entries;
    d.headroom = size_t(W);
    d.state_tail = size_t(stages * 2 * W * D);
    d.native = LayerPlan::Widths{0, backend::size_mul(stages * 2 * size_t(W), window_words(D))};
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
    for (uint16_t id : {uint16_t(output), draft_role(c, markov_head)}) blocks::shard(d.roles, id, Axis::rows, {{p.vocab, 1, 1, "vocabulary rows"}});
    using backend::Op;
    for (Op op : {Op::hc_coeffs, Op::hc_collapse_norm, Op::hc_expand, Op::norm_rope_partial, Op::rope_forms, Op::model_format_qdq, Op::model_format_pack,
                  Op::sparse_attention_sink, Op::sqrt_softplus_routing, Op::embed_ids, Op::argmax_rows})
        d.ops.push_back({Part::draft, op});
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
    const size_t E = size_t(c.n_embd), H = size_t(c.hc_count), R = s.rows, B = s.ids, n = backend::size_mul(R, B), sums = 2 * size_t(c.n_nextn);
    if (s.stride != R || (s.phase >= 0 && size_t(s.phase) >= sums + B)) throw std::logic_error("inference: a DSpark draft is one call over its sequences");
    if (!s.indices || s.indices_size < backend::size_mul(n, H)) throw std::runtime_error("inference: the DSpark draft's index scratch is too small");
    const auto runs = [&](size_t part) { return s.phase < 0 || size_t(s.phase) == part; };
    const backend::RowRun run{n, 1};
    Step x = s;
    x.rows = n;
    x.runs = {&run, 1};
    const backend::Slice rows = s.slot(collapsed);
    if (runs(0)) {
        const Weight& table = s.w[token_embd];
        s.b.embed_ids(rows, table.operand(), E, table.nout, s.id, R);
        s.b.embed({rows.buffer, rows.offset + R * E}, table.operand(), E, table.nout, &c.noise_id, 1);
        for (size_t i = 0; i < n * H; ++i) s.indices[i] = (i / H) % B ? uint32_t(R) : uint32_t(i / H / B);
        s.b.gather_rows(x.x, rows, E, s.indices, n * H);
        std::fill(s.indices, s.indices + n, 0u);
        s.b.gather_rows(s.slot(pre_ffn), {s.tables[first_pre].get(), 0}, H, s.indices, n);
    }
    for (size_t st = 0; st < size_t(c.n_nextn); ++st) {
        const Weight* w = s.w + stage_role(st, 0);
        if (runs(2 * st)) attend(x, w, {st, B, true});
        if (runs(2 * st + 1)) ffn_body(x, w, size_t(c.draft_used), c.swiglu_exp[size_t(c.n_layer) + st], c.swiglu_shexp[size_t(c.n_layer) + st]);
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

// A DSpark sidecar, `general.architecture` dflash, rewritten into the target's blocks after tokenizer equality (docs/DEEPSEEK4.md, section 4.7); any other drafter file is left as it is.
// Its stages follow the target's layers, its globals take `nextn` names in its first and last stage, its keys are the target's with the clamps extended and its own under `nextn`, and its experts must not be the target's (F14, F21).
inline void normalize_drafter(const gguf::GGUFModel& target, gguf::GGUFModel& side) {
    const auto* arch = side.find("general.architecture");
    if (!arch || arch->vtype != gguf::V_STRING || arch->s != "dflash") return;
    const auto refuse = [](const std::string& text) { return std::runtime_error("inference: DSpark sidecar " + text); };
    const std::string p = "deepseek41.", q = "dflash.";
    const int layers = metadata::integer(target, p + "block_count");
    if (metadata::count(target, p + "nextn_predict_layers", 0)) throw refuse("beside a target that carries drafter stages");
    const int stages = metadata::integer(side, q + "block_count");
    // Every key it shares with the target the target's, but the counts and clamps its stages have of their own.
    const std::set<std::string> own{"block_count", "expert_count", "expert_used_count", "swiglu_clamp_exp", "swiglu_clamp_shexp", "attention.compress_ratios",
                                    "block_size", "target_layers"};
    for (const auto& item : side.kv) {
        if (item.first.compare(0, q.size(), q) || own.count(item.first.substr(q.size()))) continue;
        const auto* t = target.find(p + item.first.substr(q.size()));
        if (!t || !gguf::detail::equal_value(*t, item.second)) throw refuse("metadata " + item.first + " differs from the target's");
    }
    for (int r : metadata::counts(side, q + "attention.compress_ratios"))
        if (r) throw refuse("stage reads compressed entries");
    const auto* noise = side.find("tokenizer.ggml.mask_token_id");
    if (!noise) throw refuse("names no tokenizer.ggml.mask_token_id for its noise rows");

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
    copy(p + "nextn.target_layers", q + "target_layers");
    value(p + "nextn.noise_token_id") = *noise;

    // Its stages after the target's layers, and its globals in the first and last stage.
    const std::string first = "blk." + std::to_string(layers) + ".nextn.", last = "blk." + std::to_string(layers + stages - 1) + ".nextn.";
    const std::vector<std::pair<std::string, std::string>> globals{{"fc.weight", first + "main_proj.weight"},
                                                                   {"enc.output_norm.weight", first + "main_norm.weight"},
                                                                   {"output_norm.weight", last + "output_norm.weight"},
                                                                   {"markov_w1.weight", last + "markov_embed.weight"},
                                                                   {"markov_w2.weight", last + "markov_head.weight"},
                                                                   {"conf_proj.weight", last + "confidence.weight"}};
    std::vector<std::string> names;
    for (const auto& tensor : side.tensors) {
        if (const auto block = gguf::block_of(tensor.name)) {
            if (*block >= size_t(stages)) throw refuse("tensor " + tensor.name + " is past its stages");
            const size_t dot = tensor.name.find('.', 4);
            names.push_back("blk." + std::to_string(size_t(layers) + *block) + tensor.name.substr(dot));
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
