#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/bf16.hpp"
#include "format/gguf.hpp"
#include "model/arch/blocks.hpp"
#include "model/architecture.hpp"
#include "model/arch/inkling2/config.hpp"
#include "model/rope.hpp"

// Inkling-2-Mini: the plan of the inkling2_loop file's tensors over the loop's execution layers, and its math over the residual bundle (docs/INKLING2.md).
namespace infer::inkling2 {

// Role ids are local to this architecture; every pass of the loop repeats its layers' ids and names.
enum Role : uint16_t {
    token_embd, token_embd_norm, output_norm, output_res_query, output,
    attn_norm, attn_q_inp_norm, attn_k_inp_norm, attn_v_inp_norm, attn_res_query,
    attn_q, attn_k, attn_v, attn_gate, attn_output, attn_sink_proj, attn_rel, attn_q_clip, attn_rel_clip, attn_rel_bias,
    attn_k_conv, attn_v_conv, attn_k_conv_proj, attn_v_conv_proj, attn_post_conv, loop_msg_query,
    loop_q_ctrl_down, loop_q_ctrl_forget, loop_q_ctrl_gain, loop_k_ctrl_down, loop_k_ctrl_forget, loop_k_ctrl_gain,
    loop_v_ctrl_down, loop_v_ctrl_forget, loop_v_ctrl_gain, loop_ffn_ctrl_down, loop_ffn_ctrl_forget, loop_ffn_ctrl_gain,
    ffn_norm, ffn_res_query, ffn_gate_inp, exp_probs_b, ffn_gate_exps, ffn_up_exps, ffn_down_exps, ffn_post_conv,
    role_count
};

// An execution layer's kind, the flags LayerPlan::kind hands back.
enum Kind : uint8_t {
    global = 1, block_boundary = 2, second_pass = 4, third_pass = 8, opens_pass = 16, reads_block_query = 32, in_loop = 64
};

// The rows of the residual bundle: the hidden row and the states of the attention-residual blocks (docs/INKLING2.md, section 3).
constexpr size_t kBundleRows = 6;

// Arena slots after slot 0, the residual bundle; `ins` holds the q, k and v inputs as planes of the pass's rows.
enum Slot : size_t { mix = 1, ins, msg, pair_norm, low, gains, q_rows, k_rows, v_rows, gate_rows, sink_rows, rel_rows, k_dyn, v_dyn, attn_rows, out_rows,
                     ffn_in, glu_gate, glu_up, act_rows, router_scores, expert_ids, expert_weights, log_scale, head_rows };

// The tables: a zero row, a ones row, then each physical layer's relative bias table as [R][extent].
enum Table : size_t { zeros, ones, rel_tables };

// The unweighted RMS norm of the attention residuals and the messages, whose eps HF fixes rather than reads (Inkling2LoopUnweightedRMSNorm).
constexpr float kResidualEps = 1e-6f;

class Inkling2 final : public Architecture {
public:
    Inkling2(const Config& cfg, const rope::Request& request) : cfg_(cfg), rope_(rope::none(request)) {
        check_config(cfg_);
        const std::vector<int> run = execution_layers(cfg_);
        const int loop_layers = cfg_.loop_end - cfg_.loop_start;
        for (size_t e = 0; e < run.size(); ++e) {
            const bool loop = run[e] >= cfg_.loop_start && run[e] < cfg_.loop_end;
            run_.push_back({run[e], loop ? (int(e) - cfg_.loop_start) / loop_layers : 0});
        }
        // The bundle holds h, the blocks and, from the slot after the loop's last block, the previous pass's messages: its later blocks and its output (docs/INKLING2.md, section 3).
        if (blocks_of(cfg_.n_layer) + 1 > kBundleRows || message_slot() + loop_blocks() > kBundleRows)
            throw std::runtime_error("inference: the Inkling-2 blocks and loop messages do not fit the residual bundle");
    }

    // The plan of the file's tensors, every execution layer with its physical layer's roles; a tensor a layer's kind does not read is refused, and every role's tensor must exist.
    ModelPlan layout(const TensorIndex& tensors) const {
        const Config& c = cfg_;
        const uint64_t E = uint64_t(c.n_embd), X = uint64_t(c.n_expert), F = uint64_t(c.n_ff), R = uint64_t(c.rel_length);
        const uint64_t K = uint64_t(c.conv_taps), r = uint64_t(c.gate_rank);
        ModelPlan p;
        p.role_ids = role_count;
        p.vocab = size_t(c.vocab_unpadded);
        // The head's rows past the unpadded vocabulary belong to no token (nonzero in the real files), so the head reads only the unpadded prefix.
        p.pass = {{token_embd, Part::embed, RoleKind::gather, "token_embd.weight", "", E, uint64_t(c.vocab)},
                  {token_embd_norm, Part::embed, RoleKind::norm, "token_embd_norm.weight", "", E},
                  {output_norm, Part::head, RoleKind::norm, "output_norm.weight", "", E},
                  {output_res_query, Part::head, RoleKind::norm, "output_res_query.weight", "", E},
                  {output, Part::head, RoleKind::matrix, "output.weight", "", E, uint64_t(c.vocab_unpadded)}};
        p.pass.back().tensor_rows = uint64_t(c.vocab);
        for (const infer::Role& role : p.pass) tensors.at(role.name);
        const std::vector<int> run = execution_layers(c);
        const size_t loop_layers = size_t(c.loop_end - c.loop_start);
        for (size_t e = 0; e < run.size(); ++e) {
            const int l = run[e];
            const bool glob = !c.sliding[size_t(l)], loop = l >= c.loop_start && l < c.loop_end;
            const size_t pass = loop ? (e - size_t(c.loop_start)) / loop_layers : 0;
            const uint64_t H = uint64_t(c.heads[size_t(l)]), HK = uint64_t(c.kv_heads[size_t(l)]), D = uint64_t(glob ? c.head_dim : c.head_dim_swa);
            const uint64_t extent = uint64_t(glob ? c.rel_extent : c.window), W = uint64_t(c.window);
            LayerPlan layer;
            layer.kind = uint8_t((glob ? global : 0) | (l % c.block_size == 0 ? block_boundary : 0) | (pass == 1 ? second_pass : 0) | (pass == 2 ? third_pass : 0) |
                                 (loop && l == c.loop_start ? opens_pass : 0) | (l > 0 ? reads_block_query : 0) | (loop ? in_loop : 0));
            layer.routed = true;
            // Every layer keeps entries: a sliding layer its [k|v] ring of the window and its convolution rings, a full-attention layer its convolution rings and one pooled [k|v] row a position.
            // Each ring holds the rows its op reads plus the headroom a rewind by length may undo, kVerifyRows rows or the window if smaller (docs/INKLING2.md, section 4).
            layer.cache = Cache::entries;
            layer.headroom = std::min(size_t(W), kVerifyRows);
            const size_t rings = size_t(K - 1 + layer.headroom) * size_t(2 * HK * D + 2 * E);
            if (glob) {
                layer.kv_pool = 1;
                layer.kv_pool_width = size_t(2 * HK * D);
                layer.kv_pool_heads = size_t(HK);
                layer.state_tail = rings;
            } else {
                layer.state_tail = size_t(W - 1 + layer.headroom) * size_t(2 * HK * D) + rings;
            }
            // The rings and pooled rows stay F32 until an f16 format is calibrated, so the declared native format is the F32 widths (docs/INKLING2.md, section 4).
            layer.native = LayerPlan::Widths{layer.kv_pool_width, layer.state_tail};
            // The feed-forward part's convolution ring is in the layer's slot.
            layer.ffn_reads_state = true;
            const std::string pre = "blk." + std::to_string(l) + ".";
            const auto add = [&](uint16_t id, Part part, RoleKind kind, const char* name, uint64_t in, uint64_t out = 1, uint64_t experts = 0) {
                layer.roles.push_back({id, part, kind, pre + name, "", in, out, experts});
                tensors.at(layer.roles.back().name);
            };
            const auto absent = [&](bool read, std::initializer_list<const char*> names) {
                if (read) return;
                for (const char* name : names)
                    if (tensors.find(pre + name)) throw std::runtime_error("inference: a tensor its layer's kind does not read " + pre + name);
            };
            absent(!glob, {"attn_norm.weight"});
            absent(glob, {"attn_q_inp_norm.weight", "attn_k_inp_norm.weight", "attn_v_inp_norm.weight", "attn_k_conv_proj.weight", "attn_v_conv_proj.weight"});
            absent(l > 0, {"attn_res_query.weight"});
            absent(loop, {"loop_msg_query.weight", "loop_q_ctrl_down.weight", "loop_q_ctrl_forget.weight", "loop_q_ctrl_gain.weight", "loop_k_ctrl_down.weight",
                          "loop_k_ctrl_forget.weight", "loop_k_ctrl_gain.weight", "loop_v_ctrl_down.weight", "loop_v_ctrl_forget.weight", "loop_v_ctrl_gain.weight",
                          "loop_ffn_ctrl_down.weight", "loop_ffn_ctrl_forget.weight", "loop_ffn_ctrl_gain.weight"});
            if (glob) {
                add(attn_q_inp_norm, Part::mixer, RoleKind::norm, "attn_q_inp_norm.weight", E);
                add(attn_k_inp_norm, Part::mixer, RoleKind::norm, "attn_k_inp_norm.weight", E);
                add(attn_v_inp_norm, Part::mixer, RoleKind::norm, "attn_v_inp_norm.weight", E);
            } else {
                add(attn_norm, Part::mixer, RoleKind::norm, "attn_norm.weight", E);
            }
            // A full-attention layer's block query holds a row for each of the three block states it reads.
            if (l > 0) add(attn_res_query, Part::mixer, glob ? RoleKind::table : RoleKind::norm, "attn_res_query.weight", E, glob ? 3 : 1);
            add(attn_q, Part::mixer, RoleKind::matrix, "attn_q.weight", E, H * D);
            add(attn_k, Part::mixer, RoleKind::matrix, "attn_k.weight", E, HK * D);
            add(attn_v, Part::mixer, RoleKind::matrix, "attn_v.weight", E, HK * D);
            add(attn_gate, Part::mixer, RoleKind::matrix, "attn_gate.weight", E, H * D);
            add(attn_output, Part::mixer, RoleKind::matrix, "attn_output.weight", H * D, E);
            add(attn_sink_proj, Part::mixer, RoleKind::matrix, "attn_sink_proj.weight", E, H);
            add(attn_rel, Part::mixer, RoleKind::matrix, "attn_rel.weight", E, H * R);
            add(attn_q_clip, Part::mixer, RoleKind::table, "attn_q_clip.weight", D, H);
            add(attn_rel_clip, Part::mixer, RoleKind::table, "attn_rel_clip.weight", R, H);
            // The file holds a position's R values together; fill_tables turns it into the [R][extent] table attention_rel_sink reads.
            add(attn_rel_bias, Part::mixer, RoleKind::source, "attn_rel_bias.weight", R, extent);
            add(attn_k_conv, Part::mixer, RoleKind::table, "attn_k_conv.weight", K, HK * D);
            add(attn_v_conv, Part::mixer, RoleKind::table, "attn_v_conv.weight", K, HK * D);
            if (glob) {
                add(attn_k_conv_proj, Part::mixer, RoleKind::matrix, "attn_k_conv_proj.weight", E, HK * K);
                add(attn_v_conv_proj, Part::mixer, RoleKind::matrix, "attn_v_conv_proj.weight", E, HK * K);
            }
            add(attn_post_conv, Part::mixer, RoleKind::table, "attn_post_conv.weight", K, E);
            if (loop) {
                add(loop_msg_query, Part::mixer, RoleKind::table, "loop_msg_query.weight", E, 4);
                add(loop_q_ctrl_down, Part::mixer, RoleKind::matrix, "loop_q_ctrl_down.weight", 2 * E, r);
                add(loop_q_ctrl_forget, Part::mixer, RoleKind::matrix, "loop_q_ctrl_forget.weight", r, E);
                add(loop_q_ctrl_gain, Part::mixer, RoleKind::matrix, "loop_q_ctrl_gain.weight", r, E);
                add(loop_k_ctrl_down, Part::mixer, RoleKind::matrix, "loop_k_ctrl_down.weight", 2 * E, r);
                add(loop_k_ctrl_forget, Part::mixer, RoleKind::matrix, "loop_k_ctrl_forget.weight", r, E);
                add(loop_k_ctrl_gain, Part::mixer, RoleKind::matrix, "loop_k_ctrl_gain.weight", r, E);
                add(loop_v_ctrl_down, Part::mixer, RoleKind::matrix, "loop_v_ctrl_down.weight", 2 * E, r);
                add(loop_v_ctrl_forget, Part::mixer, RoleKind::matrix, "loop_v_ctrl_forget.weight", r, E);
                add(loop_v_ctrl_gain, Part::mixer, RoleKind::matrix, "loop_v_ctrl_gain.weight", r, E);
            }
            add(ffn_norm, Part::ffn, RoleKind::norm, "ffn_norm.weight", E);
            add(ffn_res_query, Part::ffn, RoleKind::norm, "ffn_res_query.weight", E);
            add(ffn_gate_inp, Part::ffn, RoleKind::matrix, "ffn_gate_inp.weight", E, X);
            add(exp_probs_b, Part::ffn, RoleKind::norm, "exp_probs_b.bias", X);
            // ffn_gate_up_exps holds each expert's gate rows, then its up rows: two runs of one stack.
            for (uint16_t half : {ffn_gate_exps, ffn_up_exps}) {
                add(half, Part::ffn, RoleKind::experts, "ffn_gate_up_exps.weight", E, F, X);
                layer.roles.back().first_row = half == ffn_up_exps ? F : 0;
                layer.roles.back().tensor_rows = 2 * F;
            }
            add(ffn_down_exps, Part::ffn, RoleKind::experts, "ffn_down_exps.weight", F, E, X);
            add(ffn_post_conv, Part::ffn, RoleKind::table, "ffn_post_conv.weight", K, E);
            if (loop) {
                add(loop_ffn_ctrl_down, Part::ffn, RoleKind::matrix, "loop_ffn_ctrl_down.weight", 2 * E, r);
                add(loop_ffn_ctrl_forget, Part::ffn, RoleKind::matrix, "loop_ffn_ctrl_forget.weight", r, E);
                add(loop_ffn_ctrl_gain, Part::ffn, RoleKind::matrix, "loop_ffn_ctrl_gain.weight", r, E);
            }
            using backend::Op;
            for (Op op : {Op::copy_rows, Op::query_mix, Op::l2_clip, Op::depthwise_conv, Op::depthwise_conv_ring, Op::attention_rel_sink, Op::group_rms_norm, Op::sigmoid_mul})
                layer.ops.push_back({Part::mixer, op});
            for (Op op : {Op::copy_rows, Op::query_mix, Op::depthwise_conv, Op::depthwise_conv_ring, Op::glu_softcap, Op::expert_window, Op::sigmoid_routing})
                layer.ops.push_back({Part::ffn, op});
            if (loop) {
                layer.ops.push_back({Part::mixer, Op::tanh_gate_mix});
                layer.ops.push_back({Part::ffn, Op::tanh_gate_mix});
            }
            blocks::routed_ops(layer, tensors, ffn_gate_exps, ffn_up_exps);
            // A tensor group splits the attention by heads, each member's q heads reading its KV heads, and the experts' hidden rows (docs/INKLING2.md).
            shard_heads(layer, H, HK, D, R, K, glob);
            blocks::shard_experts(layer.roles, tensors, ffn_gate_exps, ffn_up_exps, ffn_down_exps, F, X);
            p.layers.push_back(std::move(layer));
        }
        blocks::shard_vocab(p.pass, output, p.vocab);
        // A group sums the attention's and the experts' output, which each member then convolves and joins into its bundle (join).
        p.partial = size_t(E);
        p.pass_ops = {{Part::embed, backend::Op::copy_rows}, {Part::head, backend::Op::query_mix}, {Part::head, backend::Op::softcap}};
        p.context_length = size_t(c.context_length);
        p.residual = size_t(kBundleRows * E);
        // An image's rows replace its pad rows after the embedding norm, in bundle row 0 with the other five zero, as HF scatters them into the normed embeddings.
        p.embed_width = size_t(E);
        p.text_position_images = c.image_id;
        size_t H = 0, KD = 0, KK = 0;
        for (int l = 0; l < c.n_layer; ++l) {
            const size_t D = size_t(c.sliding[size_t(l)] ? c.head_dim_swa : c.head_dim);
            H = std::max(H, size_t(c.heads[size_t(l)]));
            KD = std::max(KD, size_t(c.kv_heads[size_t(l)]) * D);
            KK = std::max(KK, size_t(c.kv_heads[size_t(l)]) * size_t(K));
        }
        const size_t HD = H * size_t(std::max(c.head_dim, c.head_dim_swa)), k = size_t(c.n_expert_used);
        p.slots = {p.residual, 3 * E, 3 * E, 4 * E, 2 * E, 3 * r, 6 * E, HD, KD, KD, HD, H, H * R, KK, KK, HD, E, E, k * F, k * F, k * F, X, k, k, 1, p.residual};
        p.tables = {size_t(E), size_t(std::max(2 * E, uint64_t(KD)))};
        for (int l = 0; l < c.n_layer; ++l) p.tables.push_back(size_t(R) * size_t(c.sliding[size_t(l)] ? c.window : c.rel_extent));
        backend::PassWidths& w = p.widths;
        w.heads = size_t(*std::max_element(c.heads.begin(), c.heads.end()));
        w.head_dim = size_t(std::max(c.head_dim, c.head_dim_swa));
        w.relative_attention = true;
        w.routed = size_t(c.n_expert_used);
        w.expert_in = size_t(F);
        w.expert_out = std::max(size_t(E), size_t(2 * F));   // the down's rows, or the gate-up stack's whole rows its windows are read from
        w.product_in = std::max(size_t(2 * E), w.heads * w.head_dim);
        w.product_out = std::max({w.heads * w.head_dim, size_t(2 * F), size_t(E)});
        return p;
    }

    ModelPlan plan(const TensorIndex& tensors) const override { return layout(tensors); }

    const rope::Policy& rope_policy() const override { return rope_; }

    // The zero and ones rows, then each physical layer's relative bias turned from the file's [extent][R] into [R][extent], from the first execution layer that reads it.
    void fill_tables(std::vector<std::vector<float>>& tables, const std::vector<TensorView>& sources) const override {
        std::fill(tables[zeros].begin(), tables[zeros].end(), 0.0f);
        std::fill(tables[ones].begin(), tables[ones].end(), 1.0f);
        if (sources.size() != run_.size()) throw std::logic_error("inference: the Inkling-2 relative bias sources are not one an execution layer");
        const size_t R = size_t(cfg_.rel_length);
        for (size_t e = 0; e < run_.size(); ++e) {
            const int l = run_[e].layer;
            if (run_[e].pass) continue;
            const TensorView& t = sources[e];
            std::vector<float>& out = tables[rel_tables + size_t(l)];
            const size_t n = out.size(), extent = n / R, width = t.type == quant::GGML_TYPE_BF16 ? 2 : t.type == quant::GGML_TYPE_F32 ? 4 : 0;
            if (!width || !t.data || t.bytes != n * width) throw std::runtime_error("inference: unsupported tensor type " + t.name + ", which Inkling-2 reads on the host as BF16 or F32");
            for (size_t d = 0; d < extent; ++d)
                for (size_t r = 0; r < R; ++r) {
                    const size_t i = d * R + r;
                    float v;
                    if (width == 2) {
                        uint16_t b;
                        std::memcpy(&b, t.data + 2 * i, 2);
                        v = bf16_to_f32(b);
                    } else {
                        std::memcpy(&v, t.data + 4 * i, 4);
                    }
                    out[r * extent + d] = v;
                }
        }
    }

    bool embeds_fed() const override { return true; }

    // The token's row in bundle row 0 and zeros in the other five, from ids past the table, then the embedding norm in place.
    void embed(const Step& s, const uint32_t* ids) const override {
        const Weight& table = s.w[token_embd];
        const size_t E = size_t(cfg_.n_embd);
        std::vector<uint32_t> six(s.rows * kBundleRows, uint32_t(table.nout));
        for (size_t r = 0; r < s.rows; ++r) six[r * kBundleRows] = ids[r];
        s.b.embed(s.x, table.operand(), E, table.nout, six.data(), six.size());
        // A fed row's bundle row 0 takes its token's row by the id a device holds (Step::fed), over the placeholder's.
        for (size_t i = 0; i < s.n_fed; ++i)
            for (size_t j = 0; j < s.fed[i].count; ++j)
                s.b.embed_ids({s.x.buffer, s.x.offset + (s.fed[i].row + j) * kBundleRows * E}, table.operand(), E, table.nout,
                              {s.fed[i].ids.buffer, s.fed[i].ids.offset + j}, 1);
        s.b.rms_norm_rows(s.x, s.x, s.w[token_embd_norm].slice(), s.rows, E, kBundleRows * E, cfg_.rms_eps);
    }

    // The full-attention layers' log scale of each row's position, 1 + alpha log(max((pos + 1) / floor, 1)) as HF computes it in F32.
    void pass_values(const Step& s) const override {
        std::vector<float> scale(s.rows);
        for (size_t r = 0; r < s.rows; ++r)
            scale[r] = 1.0f + cfg_.log_scale_alpha * std::log(std::max(float(s.pos[r] + 1) / float(cfg_.log_scale_floor), 1.0f));
        s.b.write(*s.arena, s.offsets[log_scale], scale.data(), scale.size() * sizeof(float));
    }

    // The attention part: inputs mixed from the bundle (gated by the messages in later loop passes), clipped queries, convolved keys and values, relative attention with its sink, gate, output, its convolution, residual.
    void mixer(const Step& s) const override {
        backend::Backend& b = s.b;
        const Weight* w = s.w;
        const Config& c = cfg_;
        const int l = run_.at(size_t(s.layer)).layer;
        const bool glob = s.kind & global, msgs = s.kind & (second_pass | third_pass);
        // On a tensor group a member runs its share of the heads (shard_heads).
        const size_t E = size_t(c.n_embd), S = kBundleRows * E, rows = s.rows, H = size_t(c.heads[size_t(l)]) / s.width, HK = kv_share(size_t(c.kv_heads[size_t(l)]), s.width);
        const size_t D = size_t(glob ? c.head_dim : c.head_dim_swa), HD = H * D, KD = HK * D, R = size_t(c.rel_length), K = size_t(c.conv_taps);
        const size_t nq = glob ? 3 : 1;
        // The message mix of the later passes and the residual mix, in one call where both run; a first-pass sliding layer's one mix is normed into its plane in the call.
        const Weight* norms[3] = {glob ? &w[attn_q_inp_norm] : &w[attn_norm], glob ? &w[attn_k_inp_norm] : &w[attn_norm], glob ? &w[attn_v_inp_norm] : &w[attn_norm]};
        const bool normed = l != 0 && !msgs && (!glob || rows == 1);
        const backend::QueryMixRows message{s.slot(msg), bundle(s, message_slot()), w[loop_msg_query].slice(), loop_blocks(), 4};
        backend::QueryMixRows residual{normed ? plane(s, ins, 0, E) : s.slot(mix), bundle(s, 0), w[attn_res_query].slice(), 1 + blocks_of(l), nq};
        if (normed)
            for (size_t j = 0; j < nq; ++j) residual.norm[j] = norms[j]->slice();
        if (normed) residual.norm_eps = c.rms_eps;
        // One row needs no packing: a first-pass norm reads its mix plane, or layer 0's bundle row, where it lies.
        const bool one = rows == 1;
        if (l == 0) {
            if (msgs) b.query_mix({message}, rows, E, S, kResidualEps);
            if (msgs || !one) b.copy_rows(s.slot(mix), E, bundle(s, 0), S, rows, E, false);
        } else if (msgs) {
            b.query_mix({message, residual}, rows, E, S, kResidualEps);
        } else {
            b.query_mix({residual}, rows, E, S, kResidualEps);
        }
        // The q, k and v inputs as planes: a full-attention layer's three mixes, a sliding layer's one, each gated by its message in the later passes, then normed.
        const size_t inputs = glob || msgs ? 3 : 1;
        // The later passes and a first-pass global layer normed at one row have all three planes ready before any product, so their products are one group.
        const bool merged = msgs || (normed && glob);
        const Weight* gates[3][3] = {{&w[loop_q_ctrl_down], &w[loop_q_ctrl_forget], &w[loop_q_ctrl_gain]},
                                     {&w[loop_k_ctrl_down], &w[loop_k_ctrl_forget], &w[loop_k_ctrl_gain]},
                                     {&w[loop_v_ctrl_down], &w[loop_v_ctrl_forget], &w[loop_v_ctrl_gain]}};
        // The q plane's products follow its norm, and the k and v planes' products follow both their norms as one group of two inputs, so each reads the twin its norm wrote in one of the device's two twin slots.
        // In the later passes one call norms the three planes into one twin, so all their products are one group of three inputs, which a device runs as one launch.
        const auto own = [&](const Weight& wt, size_t slot, backend::CSlice x) {
            backend::Projection p = blocks::projection(wt, s.slot(slot));
            p.in = x;
            return p;
        };
        const auto project = [&](size_t j, backend::CSlice in) {
            // A device takes a group's products three a launch, or six in one (pair builds), so each wide projection leads a three with a narrow one, the threes near equal; one input plane feeds k and v too.
            if (j == 0) {
                const backend::Projection q = blocks::projection(w[attn_q], s.slot(q_rows)), sink = blocks::projection(w[attn_sink_proj], s.slot(sink_rows)),
                                          rel = blocks::projection(w[attn_rel], s.slot(rel_rows)), gate = blocks::projection(w[attn_gate], s.slot(gate_rows));
                const backend::CSlice kin = plane(s, ins, 1, E), vin = plane(s, ins, 2, E);
                if (inputs == 1)
                    b.matmul_group({q, sink, blocks::projection(w[attn_k], s.slot(k_rows)), gate, rel, blocks::projection(w[attn_v], s.slot(v_rows))}, in, E, rows, s.runs, s.dtype);
                else if (!merged)
                    b.matmul_group({q, sink, rel, gate}, in, E, rows, s.runs, s.dtype);
                else if (glob)
                    b.matmul_group({q, sink, rel, gate, own(w[attn_k], k_rows, kin), own(w[attn_k_conv_proj], k_dyn, kin), own(w[attn_v], v_rows, vin), own(w[attn_v_conv_proj], v_dyn, vin)}, in, E,
                                   rows, s.runs, s.dtype);
                else
                    b.matmul_group({q, sink, rel, gate, own(w[attn_k], k_rows, kin), own(w[attn_v], v_rows, vin)}, in, E, rows, s.runs, s.dtype);
            } else if (j == 2 && !merged) {
                const backend::CSlice kin = plane(s, ins, 1, E);
                if (glob)
                    b.matmul_group({own(w[attn_k], k_rows, kin), own(w[attn_k_conv_proj], k_dyn, kin), own(w[attn_v], v_rows, in), own(w[attn_v_conv_proj], v_dyn, in)}, in, E, rows, s.runs,
                                   s.dtype);
                else
                    b.matmul_group({own(w[attn_k], k_rows, kin), own(w[attn_v], v_rows, in)}, in, E, rows, s.runs, s.dtype);
            }
        };
        const auto mixed = [&](size_t j) { return offset(s.slot(mix), (glob ? j : 0) * E); };
        // A plane's gated mix with its down step and products, normed into its input plane; the three planes' items are one call (tanh_gate_mix), which a device may run as one launch.
        const auto gated = [&](size_t j) {
            return gate_item(s, j, plane(s, ins, j, E), mixed(j), nq * E, offset(s.slot(msg), j * E), norms[j]->slice(), *gates[j][0], *gates[j][1], *gates[j][2]);
        };
        for (size_t j = 0; j < inputs; ++j) {
            const backend::Slice in = plane(s, ins, j, E), src = mixed(j);
            // A single mix is already rows E apart, so it is normed straight into its plane; several are packed first unless there is one row.
            // The three planes' mixes are one call before any product, which a device runs as one launch keeping one twin of the three planes (the q products write none of their inputs).
            if (msgs) {
                if (j == 0) b.tanh_gate_mix({gated(0), gated(1), gated(2)}, rows, E, E, nq * E, 4 * E, s.runs, c.rms_eps);
            } else {
                const bool packed = nq > 1 && !one;
                if (packed) b.copy_rows(in, E, src, nq * E, rows, E, false);
                const backend::CSlice from = packed ? backend::CSlice(in) : l == 0 && one ? backend::CSlice(bundle(s, 0)) : backend::CSlice(src);
                if (!normed) b.rms_norm_rows(in, from, norms[j]->slice(), rows, E, E, c.rms_eps);
            }
            project(j, in);
        }
        const backend::CSlice scale = glob ? backend::CSlice(s.slot(log_scale)) : backend::CSlice{};
        // The slot's rings: a sliding layer's [k|v] window first, then the k, v, post-attention and post-feed-forward convolution rings (Layout::state_tail).
        const Rings g = rings(l, s.width);
        const float divisor = glob ? 3.0f : std::sqrt(5.0f);
        // The k and v convs take one call, the convolved k clipped and v normed per head as its post step.
        backend::DepthwiseConv kc = conv(w[attn_k_conv], KD, K, glob ? backend::CSlice(s.slot(k_dyn)) : backend::CSlice{}, glob ? HK : 0, divisor, g.conv, g.rows);
        backend::DepthwiseConv vc = conv(w[attn_v_conv], KD, K, glob ? backend::CSlice(s.slot(v_dyn)) : backend::CSlice{}, glob ? HK : 0, divisor, g.conv + g.rows * KD, g.rows);
        kc.post = backend::ConvPost::l2_clip, kc.post_heads = HK, kc.post_limit = c.qk_clip;
        vc.post = backend::ConvPost::rms_norm, vc.post_heads = HK, vc.post_limit = c.rms_eps, vc.post_w = {s.tables[ones].get(), 0};
        b.depthwise_conv({{s.slot(k_rows), s.slot(k_rows), kc}, {s.slot(v_rows), s.slot(v_rows), vc}}, s.state_layer, s.states, s.n_views);
        const backend::CSlice table{s.tables[rel_tables + size_t(l)].get(), 0};
        const float att_scale = 1.0f / std::sqrt(float(D));
        const backend::Backend::OutputGate out_gate{s.slot(gate_rows), rows, HD, D, {}};
        const backend::RelClip clip{w[attn_q_clip].slice(), w[attn_rel_clip].slice(), scale, rows, c.qk_clip, c.rel_clip};
        if (glob)
            b.attention_rel_sink(s.slot(attn_rows), s.slot(q_rows), s.slot(k_rows), s.slot(v_rows), s.slot(rel_rows), table, s.slot(sink_rows), H, HK, D, R,
                                 size_t(c.rel_extent), 0, att_scale, s.kv_layer, nullptr, s.views, s.n_views, 0, 0, s.runs, out_gate, clip);
        else
            b.attention_rel_sink(s.slot(attn_rows), s.slot(q_rows), s.slot(k_rows), s.slot(v_rows), s.slot(rel_rows), table, s.slot(sink_rows), H, HK, D, R,
                                 size_t(c.window), size_t(c.window), att_scale, s.state_layer, s.states, nullptr, s.n_views, 0, g.window, s.runs, out_gate, clip);
        // On a tensor group the output projection writes the member's partial rows and the rest waits for their sum (join); one device hands it to the post conv as its input's product.
        if (s.width > 1)
            b.matmul(w[attn_output].operand(), s.slot(attn_rows), s.partial, HD, E, rows, s.runs, s.dtype);
        else
            close_mixer(s, s.slot(out_rows), {w[attn_output].operand(), s.slot(attn_rows), s.slot(out_rows), HD, s.runs, s.dtype});
    }

    // The feed-forward part: its input mixed from the bundle (and gated by the feed-forward message), normed, the sigmoid-routed softcap experts, their convolution, then the residual.
    // On a tensor group the experts write the member's partial rows and the rest waits for their sum (join).
    void ffn(const Step& s) const override {
        backend::Backend& b = s.b;
        const Weight* w = s.w;
        const Config& c = cfg_;
        const size_t E = size_t(c.n_embd), S = kBundleRows * E, rows = s.rows;
        if (s.width == 1) b.query_mix({ffn_mix(s)}, rows, E, S, kResidualEps);
        const bool msgs = s.kind & (second_pass | third_pass);
        // The message gates the mix in place; the step norms it into ffn_in (MoeStep::input).
        if (msgs)
            b.tanh_gate_mix({gate_item(s, 0, s.slot(mix), s.slot(mix), E, offset(s.slot(msg), 3 * E), {}, w[loop_ffn_ctrl_down], w[loop_ffn_ctrl_forget], w[loop_ffn_ctrl_gain])},
                            rows, E, E, E, 4 * E, s.runs, c.rms_eps);
        const backend::MoeInput in{s.slot(mix), w[ffn_norm].slice(), c.rms_eps};
        const backend::Projection gate = blocks::projection(w[ffn_gate_exps], s.slot(glu_gate)), up = blocks::projection(w[ffn_up_exps], s.slot(glu_up));
        const size_t k = size_t(c.n_expert_used);
        const backend::MoeStep m{blocks::projection(w[ffn_gate_inp], s.slot(router_scores)), k,
                                 backend::RouteRule::sigmoid(w[exp_probs_b].slice(), c.expert_scale, true, 1e-20f), s.slot(expert_ids), s.slot(expert_weights), gate, up,
                                 w[ffn_down_exps].operand(), s.slot(act_rows), nullptr, s.tables[zeros].get(), s.scratch,
                                 {c.gate_softcap, backend::GluForm::softcap, c.up_softcap}, &in};
        // One device hands the step to the post-FFN conv as the product forming its input, so a device may run the conv in the step's store.
        if (s.width > 1)
            b.moe_step(m, s.slot(ffn_in), s.partial, E, rows, s.runs, s.dtype);
        else
            close_ffn(s, s.slot(out_rows), {{}, {}, s.slot(out_rows), E, s.runs, s.dtype, &m, s.slot(ffn_in)});
        blocks::report_routes(s, k, s.slot(expert_ids), s.slot(router_scores), w[ffn_gate_inp].nout);
    }

    // On a tensor group, the summed rows of the part just run, the attention's or the experts', closed as one device closes its own.
    void join(const Step& s) const override {
        if (s.joins == Part::mixer) close_mixer(s, s.partial);
        else close_ffn(s, s.partial);
    }

    // The head: the wanted rows' bundles gathered, then one row product whose source mixes them over every block state and norms them, its store the softcapped logits.
    void head(const HeadStep& s) const override {
        const size_t E = size_t(cfg_.n_embd), S = kBundleRows * E;
        const Weight& out = s.w[output];
        s.b.gather_rows(s.slot(head_rows), s.x, S, s.pick, s.want);
        const backend::RowMix mix{{s.slot(out_rows), s.slot(head_rows), s.w[output_res_query].slice(), 1 + blocks_of(cfg_.n_layer), 1}, E, S, kResidualEps};
        const backend::RowNorm norm{s.slot(out_rows), s.slot(out_rows), s.w[output_norm].slice(), E, E, cfg_.rms_eps};
        backend::RowSource src;
        src.row[0] = s.slot(out_rows), src.mix = &mix, src.norm[0] = &norm;
        backend::RowSink logits;
        logits.store = backend::RowStore::logits, logits.cap = cfg_.logit_softcap;
        s.b.rows_product(src, {{{out.operand(), s.logits, out.nout}, logits}}, out.nin, s.want, s.head_runs, s.dtype);
    }

private:
    // Where a layer's rings start in its slot and how many rows each holds: the [k|v] window ring of a sliding layer, then the convolution rings.
    struct Rings {
        size_t window = 0, conv = 0, rows = 0;
    };
    // A convolution over a ring of the slot: (x + sum of taps, static plus optional per-head dynamic ones) / divisor.
    static backend::DepthwiseConv conv(const Weight& w, size_t channels, size_t taps, backend::CSlice dyn, size_t dyn_heads, float divisor, size_t offset, size_t rows) {
        backend::DepthwiseConv d;
        d.w = w.slice();
        d.channels = channels;
        d.taps = taps;
        d.residual_from = backend::ConvResidual::input;
        d.divisor = divisor;
        d.dyn = dyn;
        d.dyn_heads = dyn_heads;
        d.dyn_stride = dyn_heads * taps;
        d.history = backend::ConvHistory::ring;
        d.ring_offset = offset;
        d.ring_rows = rows;
        return d;
    }
    // A tensor group member's slot holds the rings of its KV heads in the same order.
    Rings rings(int l, size_t width) const {
        const Config& c = cfg_;
        const size_t headroom = std::min(size_t(c.window), kVerifyRows), KD = member_kd(l, width);
        Rings g;
        g.rows = size_t(c.conv_taps) - 1 + headroom;
        if (c.sliding[size_t(l)]) {
            g.window = size_t(c.window) - 1 + headroom;
            g.conv = g.window * 2 * KD;
        }
        return g;
    }
    // The KV width of physical layer l a member of a group of `width` keeps.
    size_t member_kd(int l, size_t width) const {
        return kv_share(size_t(cfg_.kv_heads[size_t(l)]), width) * size_t(cfg_.sliding[size_t(l)] ? cfg_.head_dim_swa : cfg_.head_dim);
    }
    // A layer's attention split by heads: q, gate, sink and relative rows and their clips by query head, k, v, their convolutions and dynamic taps by KV head, the output by columns.
    static void shard_heads(LayerPlan& layer, uint64_t H, uint64_t HK, uint64_t D, uint64_t R, uint64_t K, bool glob) {
        const ShardSection kv{HK, D, 1, "KV heads", true};
        blocks::shard(layer.roles, attn_q, Axis::rows, {{H, D, 1, "heads"}});
        blocks::shard(layer.roles, attn_gate, Axis::rows, {{H, D, 1, "heads"}});
        blocks::shard(layer.roles, attn_sink_proj, Axis::rows, {{H, 1, 1, "heads"}});
        blocks::shard(layer.roles, attn_rel, Axis::rows, {{H, R, 1, "heads"}});
        blocks::shard(layer.roles, attn_q_clip, Axis::rows, {{H, 1, 1, "heads"}});
        blocks::shard(layer.roles, attn_rel_clip, Axis::rows, {{H, 1, 1, "heads"}});
        for (uint16_t id : {attn_k, attn_v, attn_k_conv, attn_v_conv}) blocks::shard(layer.roles, id, Axis::rows, {kv});
        if (glob)
            for (uint16_t id : {attn_k_conv_proj, attn_v_conv_proj}) blocks::shard(layer.roles, id, Axis::rows, {{HK, K, 1, "KV heads", true}});
        blocks::shard(layer.roles, attn_output, Axis::columns, {{H, D, 1, "heads"}});
    }
    // The attention part after its output rows `y`, which `from` forms where it has a product: their convolution, which writes the residual stream itself.
    // It is added to the stream or, at a block boundary, once the layer's input is stored as the next block state, written in its place.
    void close_mixer(const Step& s, backend::CSlice y, const backend::ConvProduct& from = {}) const {
        backend::Backend& b = s.b;
        const Config& c = cfg_;
        const int l = run_.at(size_t(s.layer)).layer;
        const size_t E = size_t(c.n_embd), S = kBundleRows * E, rows = s.rows;
        const Rings g = rings(l, s.width);
        const bool boundary = s.kind & block_boundary;
        if (boundary) b.copy_rows(bundle(s, 1 + size_t(l / c.block_size)), S, bundle(s, 0), S, rows, E, false);
        backend::DepthwiseConv post = conv(s.w[attn_post_conv], E, size_t(c.conv_taps), {}, 0, std::sqrt(5.0f), g.conv + 2 * g.rows * member_kd(l, s.width), g.rows);
        post.out_stride = S;
        post.accumulate = !boundary;
        // On a tensor group (join) the feed-forward's mix of the stream follows the conv in its call, on the member that runs that part.
        if (s.width > 1) return b.depthwise_conv({{bundle(s, 0), y, post, from}}, s.state_layer, s.states, s.n_views, {ffn_mix(s), E, S, kResidualEps});
        b.depthwise_conv({{bundle(s, 0), y, post, from}}, s.state_layer, s.states, s.n_views);
    }
    // The feed-forward part's mix of the residual stream's block states into its input.
    backend::QueryMixRows ffn_mix(const Step& s) const {
        const int l = run_.at(size_t(s.layer)).layer;
        return {s.slot(mix), bundle(s, 0), s.w[ffn_res_query].slice(), 2 + size_t(l / cfg_.block_size), 1};
    }
    // The feed-forward part after its experts' rows `y`, which `from` forms where it has a product: their convolution, then the residual.
    // The last loop layer of a pass before the last also leaves that pass's messages and restarts the next pass from the loop's input.
    void close_ffn(const Step& s, backend::CSlice y, const backend::ConvProduct& from = {}) const {
        backend::Backend& b = s.b;
        const Config& c = cfg_;
        const Exec& x = run_.at(size_t(s.layer));
        const int l = x.layer;
        const size_t E = size_t(c.n_embd), S = kBundleRows * E, rows = s.rows, K = size_t(c.conv_taps), KD = member_kd(l, s.width);
        const Rings g = rings(l, s.width);
        backend::DepthwiseConv post = conv(s.w[ffn_post_conv], E, K, {}, 0, std::sqrt(5.0f), g.conv + 2 * g.rows * KD + g.rows * E, g.rows);
        const bool closes = l == c.loop_end - 1 && x.pass + 1 < c.loop_count;
        if (!closes) {
            // The conv adds itself to the residual stream.
            post.out_stride = S;
            post.accumulate = true;
            b.depthwise_conv({{bundle(s, 0), y, post, from}}, s.state_layer, s.states, s.n_views);
            return;
        }
        b.depthwise_conv({{s.slot(out_rows), y, post, from}}, s.state_layer, s.states, s.n_views);
        // The pass's output and its blocks after the first become the messages, and the next pass starts from the loop's first block state, the loop's input.
        const size_t first = 1 + size_t(c.loop_start / c.block_size), at = message_slot(), lb = loop_blocks();
        b.copy_rows(bundle(s, at + lb - 1), S, bundle(s, 0), S, rows, E, false);
        b.copy_rows(bundle(s, at + lb - 1), S, s.slot(out_rows), E, rows, E, true);
        for (size_t i = 0; i + 1 < lb; ++i) b.copy_rows(bundle(s, at + i), S, bundle(s, first + 1 + i), S, rows, E, false);
        b.copy_rows(bundle(s, 0), S, bundle(s, first), S, rows, E, false);
    }
    // The block states before physical layer l: one for each block boundary below it.
    size_t blocks_of(int l) const { return size_t((l + cfg_.block_size - 1) / cfg_.block_size); }
    size_t loop_blocks() const { return size_t((cfg_.loop_end - cfg_.loop_start) / cfg_.block_size); }
    size_t message_slot() const { return 1 + size_t(cfg_.loop_end / cfg_.block_size); }
    // Bundle row k of the residual's first row; rows are kBundleRows hidden widths apart.
    backend::Slice bundle(const Step& s, size_t k) const { return offset(s.x, k * size_t(cfg_.n_embd)); }
    static backend::Slice offset(backend::Slice x, size_t floats) { return {x.buffer, x.offset + floats}; }
    // Plane j of a slot: the pass's rows of `width` floats, after j such planes.
    static backend::Slice plane(const Step& s, size_t slot, size_t j, size_t width) { return offset(s.slot(slot), j * s.rows * width); }

    // The message gate's g = down [norm(h) ; norm(m)] into plane j of `low`, as a gate mix item's down step; h and m are rows h_stride and msg's 4E apart, and the items share one scratch norm row.
    backend::NormRows gate_down(const Step& s, const Weight& down, backend::CSlice h, size_t h_stride, backend::CSlice m, size_t j) const {
        return {s.slot(pair_norm), h, {m, 1, h_stride, 4 * size_t(cfg_.n_embd)}, {down.operand(), plane(s, low, j, size_t(cfg_.gate_rank)), down.nout, s.dtype}};
    }
    // The message gate's forget (k 0) or gain (k 1) product of gate_down's g in plane j of `low`, into plane 2j + k of `gains`; its prompt rows take a GPU's wide tile (WeightView::wide).
    backend::Projection gate_product(const Step& s, const Weight& wt, size_t j, size_t k) const {
        backend::Projection p = blocks::projection(wt, plane(s, gains, 2 * j + k, size_t(cfg_.n_embd)));
        p.in = plane(s, low, j, size_t(cfg_.gate_rank));
        p.weight.wide = true;
        return p;
    }
    // The message gate, (1 - tanh(forget g)) h + tanh(gain g) m with gate j's down step and products, which the call computes (GateMixItem), into the contiguous rows `out`.
    backend::GateMixItem gate_item(const Step& s, size_t j, backend::Slice out, backend::CSlice h, size_t h_stride, backend::CSlice m, backend::CSlice norm, const Weight& down,
                                   const Weight& forget, const Weight& gain) const {
        const size_t E = size_t(cfg_.n_embd);
        backend::GateMixItem it{out, h, m, plane(s, gains, 2 * j, E), plane(s, gains, 2 * j + 1, E), norm};
        it.forget = gate_product(s, forget, j, 0), it.gain = gate_product(s, gain, j, 1), it.rank = size_t(cfg_.gate_rank), it.dtype = s.dtype;
        it.down = gate_down(s, down, h, h_stride, m, j), it.down_w = {s.tables[ones].get(), 0};
        return it;
    }

    // An execution layer's physical layer and loop pass.
    struct Exec {
        int layer = 0, pass = 0;
    };
    Config cfg_;
    rope::Policy rope_;
    std::vector<Exec> run_;
};

inline std::shared_ptr<const Architecture> open(const gguf::GGUFModel& m, const std::string&, const rope::Request& request) {
    return std::make_shared<const Inkling2>(read_config(m), request);
}

} // namespace infer::inkling2
