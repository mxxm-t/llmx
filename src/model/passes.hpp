#pragma once
#include "model/runtime.hpp"

// A pass, the one owner of how it runs: its plan, the stages of its layers over the devices, the crossings of the residual between devices, a streamed layer's split, the embedded drafter's rows after the last stage, and the storage of the context it runs in.
// Members of infer::Model, declared in its class (model/runtime.hpp), which includes this file after it.

namespace infer {

// One pass over every entry: each sequence's tokens at their own positions through their own history, the logits after each wanting entry's last token landing in the context in entry order.
// Its stages run in a row, each submitting its own work and committing the blocks of the storage it writes; a failure anywhere returns every history to where the pass found it.
inline void Model::forward(ExecContext& ctx, const BatchEntry* entries, size_t n_entries) {
    if (ctx.slots) throw std::logic_error("inference: a context reserved for passes runs them through begin_pass");
    if (ctx.passes.empty()) ctx.passes.resize(1);
    Pass& p = ctx.passes[0];
    begin(ctx, p, entries, n_entries);
    try {
        for (size_t s = 0; s < stages_.size(); ++s) run_stage(ctx, p, s);
    } catch (...) {
        roll_back(p);
        throw;
    }
    finish(ctx, p);
}

// Size a fresh context once, before any pass, for `slots` passes in flight, which above one need a pipelined placement, of up to `rows` rows each, with their handoff buffers and `logit_rows` rows of logits the caller hands out (begin_pass's logits_base); a reservation that fails leaves the context fresh, so a smaller one may follow.
// The context is frozen from then on: begin_pass refuses a pass that needs more before any work, nothing is replaced while passes are in flight, and forward refuses it.
inline void Model::reserve_passes(ExecContext& ctx, size_t slots, size_t rows, size_t logit_rows) {
    if (ctx.slots || !ctx.scratch.empty()) throw std::logic_error("inference: reserve_passes takes a fresh context, once");
    if (!slots || !rows) throw std::logic_error("inference: reserve_passes needs a slot and a row");
    if (slots > 1 && !pipelined_) throw std::logic_error("inference: passes in flight need a pipelined placement");
    ExecContext reserved;
    ensure(reserved, rows, logit_rows, handoffs(slots));
    reserved.passes.assign(slots, Pass{});
    reserved.slots = slots;
    reserved.pass_rows = rows;
    ctx = std::move(reserved);
}

// Plan a pass in `slot` of a reserved context, its entries' tokens copied here and its wanting rows written from logits row `logits_base` on, and put its sequences in flight until end_pass or abort_pass.
// A sequence in flight or listed twice, a slot in use or beyond the reservation, and more rows or logits rows than reserved are refused, with nothing changed.
inline void Model::begin_pass(ExecContext& ctx, size_t slot, const BatchEntry* entries, size_t n_entries, size_t logits_base) {
    if (!ctx.slots) throw std::logic_error("inference: begin_pass needs a context reserved for passes");
    if (slot >= ctx.slots) throw std::logic_error("inference: a pass slot beyond the reservation");
    Pass& p = ctx.passes[slot];
    if (p.in_flight) throw std::logic_error("inference: a pass slot already in flight");
    begin(ctx, p, entries, n_entries, logits_base);
    for (size_t e = 0; e < n_entries; ++e) entries[e].seq->in_flight_ = true;
    p.handoff = slot;
    p.in_flight = true;
}

// Stage s of the pass in `slot`, which must be the stage after the last one run: its storage reserved, the residual embedded or received, its layers, the head or the handoff out, its submissions and the storage's commit.
// A failure aborts the pass, as abort_pass does, before it is rethrown; other passes in flight go on.
inline void Model::run_pass_stage(ExecContext& ctx, size_t slot, size_t s) {
    Pass& p = in_flight(ctx, slot);
    if (s != p.ran || s >= stages_.size()) throw std::logic_error("inference: a pass's stages run in order, each once");
    try {
        run_stage(ctx, p, s);
    } catch (...) {
        roll_back(p);
        release(p);
        throw;
    }
    ++p.ran;
}

// A stage that waits on its devices (stage_waits) in three calls, for a caller that records it on a thread of that stage's while its own thread goes on.
// prepare_pass_stage, on the thread that owns the histories and the pools, takes the stage's blocks; record_pass_stage, on the stage's thread, does the device work, touching only this pass, the stage's devices and their tickets; commit_pass_stage, on the owning thread again, commits the histories and counts the stage.
// Nothing is undone by a failure of the first two: the caller abandons the pass with abort_pass on the owning thread.
inline void Model::prepare_pass_stage(ExecContext& ctx, size_t slot, size_t s) {
    Pass& p = in_flight(ctx, slot);
    if (s != p.ran || s >= stages_.size()) throw std::logic_error("inference: a pass's stages run in order, each once");
    if (!stage_waits(s)) throw std::logic_error("inference: a stage of one device is recorded whole, by run_pass_stage");
    group_prepare(p, s);
}

inline void Model::record_pass_stage(ExecContext& ctx, size_t slot, size_t s) {
    Pass& p = in_flight(ctx, slot);
    const Stage& st = stages_[s];
    // The stage's evidence of the matrix paths it took stays with this model, as run_stage keeps it.
    struct Paths {
        const std::vector<std::unique_ptr<Device>>& devices;
        const std::vector<size_t>& touches;
        size_t width;
        void swap() const noexcept {
            for (size_t d : touches)
                for (size_t m = d; m < d + width; ++m) devices[m]->b->swap_matrix_paths(devices[m]->matrix_paths);
        }
        ~Paths() { swap(); }
    } paths{devices_, st.touches, width_};
    paths.swap();
    group_record(ctx, p, s);
}

inline void Model::commit_pass_stage(ExecContext& ctx, size_t slot) {
    Pass& p = in_flight(ctx, slot);
    stage_commit(ctx, p, p.ran);
    ++p.ran;
}

// Row i of the pass's wanting rows, in entry order, once its last stage has run; this waits on the pass's own ticket, never on a later pass's.
inline const float* Model::pass_logits(ExecContext& ctx, size_t slot, size_t i) {
    const Pass& p = in_flight(ctx, slot);
    if (p.ran < stages_.size()) throw std::logic_error("inference: the logits of a pass before its last stage");
    if (i >= p.want) throw std::out_of_range("inference: no such logits row");
    devices_[(size_t)place_.output_device]->b->wait(p.sent);
    for (size_t m = 1; m < p.sent_members.size(); ++m) devices_[(size_t)place_.output_device + m]->b->wait(p.sent_members[m]);
    const void* host = ctx.logits_host.data() ? ctx.logits_host.data() : ctx.logits_buf->host_ptr();
    if (!host) throw std::runtime_error("inference: logits are not host visible");
    return (const float*)host + (p.logits_base + i) * ctx.width;
}

// The pass in `slot` is done once its last stage has run: its sequences leave flight with the tokens committed, and the slot, its handoff buffers and the logits rows it was given are free for the next pass.
inline void Model::end_pass(ExecContext& ctx, size_t slot) {
    Pass& p = in_flight(ctx, slot);
    if (p.ran < stages_.size()) throw std::logic_error("inference: a pass ends after its last stage, and abort_pass abandons one before");
    release(p);
}

// Abandon the pass in `slot`, run or not: every device drained, then only its entries' histories back to where it found them in every storage, and its sequences out of flight.
inline void Model::abort_pass(ExecContext& ctx, size_t slot) {
    Pass& p = in_flight(ctx, slot);
    roll_back(p);
    release(p);
}

// A pass's plan: its rows in entry order, their positions after each history, and the rows the head reads, from logits row `logits_base` on, with nothing reserved yet, since each stage reserves the blocks of the storage it writes.
// A context reserved for passes is not grown: a pass that needs more rows or logits rows than it holds is refused here.
inline void Model::begin(ExecContext& ctx, Pass& p, const BatchEntry* entries, size_t n_entries, size_t logits_base) {
    if (!entries || !n_entries) throw std::runtime_error("inference: empty batch");
    size_t rows = 0, want = 0;
    for (size_t e = 0; e < n_entries; ++e) {
        const BatchEntry& en = entries[e];
        if (!en.seq || en.seq->owner_ != this)
            throw std::runtime_error("inference: batch entry without a sequence of this model");
        if (en.seq->in_flight_) throw std::logic_error("inference: a sequence already in flight");
        if (!en.ids || !en.n)
            throw std::runtime_error("inference: batch entry without tokens");
        // The position tables cover [0, context_length); a row past them would read off the end.
        if (en.n > plan_.context_length ||
            history(*en.seq) > plan_.context_length - en.n)
            throw std::runtime_error("inference: context length exceeded (" +
                                     std::to_string(plan_.context_length) + " tokens)");
        rows += en.n;
        want += en.want_logits ? (en.every_logits ? en.n : 1) : 0;
    }
    // None is in flight, so the first entry found marked is listed twice; the marks come off again here.
    size_t marked = 0;
    for (; marked < n_entries && !entries[marked].seq->in_flight_; ++marked) entries[marked].seq->in_flight_ = true;
    for (size_t e = 0; e < marked; ++e) entries[e].seq->in_flight_ = false;
    if (marked < n_entries) throw std::logic_error("inference: a sequence listed twice in a pass");
    if (ctx.slots && (rows > ctx.pass_rows || want > ctx.logit_rows || logits_base > ctx.logit_rows - want))
        throw std::logic_error("inference: a pass beyond the rows or logits rows reserve_passes reserved");
    // Every entry holds a state slot from its first pass on, so a pass never runs short of one, and a keep entry a checkpoint slot; a pass that cannot take them all, is refused or fails to plan takes none.
    size_t fresh = 0, keeps = 0;
    for (size_t e = 0; state_layers_ && e < n_entries; ++e) {
        fresh += !entries[e].seq->state_.held();
        keeps += entries[e].keep;
    }
    if (fresh > slots_.available()) throw std::runtime_error("inference: every recurrent state slot is held");
    if (keeps > slots_.kept_available()) throw std::runtime_error("inference: every checkpoint slot is held");
    for (size_t e = 0; e < n_entries; ++e) {
        const Sequence::Mark& m = entries[e].seq->mark_;
        if (!m.held()) continue;
        if (m.ran) throw std::logic_error("inference: a marked sequence takes one pass before its retract");
        if (entries[e].n > options_.mark_rows) throw std::logic_error("inference: a pass after a mark beyond the rows a mark saves");
        if (entries[e].keep) throw std::logic_error("inference: a checkpoint in a pass after a mark");
    }
    for (size_t e = 0; keeps && e < n_entries; ++e)
        if (entries[e].keep && !whole_blocks(history(*entries[e].seq) + entries[e].n))
            throw std::logic_error("inference: a checkpoint at a position of whole blocks in every storage");
    if (!ctx.slots) ensure(ctx, rows, want, handoffs(1));
    p.entries.assign(entries, entries + n_entries);
    p.start.resize(n_entries);
    p.rows = rows;
    p.want = want;
    p.handoff = 0;
    p.logits_base = logits_base;
    p.ran = 0;
    p.ids.resize(rows);
    p.pos.resize(rows);
    p.pick.resize(want);
    p.runs.resize(n_entries);
    p.head_runs.clear();
    p.views.resize(storages_.size() * width_);
    for (auto& v : p.views) v.resize(n_entries);
    if (state_layers_) {
        p.states.resize(devices_.size());
        for (auto& v : p.states) v.resize(n_entries);
    }
    p.kept.clear();
    p.kept.resize(n_entries);
    size_t r = 0, w = 0;
    for (size_t e = 0; e < n_entries; ++e) {
        const BatchEntry& en = entries[e];
        const size_t len = history(*en.seq);
        p.start[e] = len;
        for (size_t b = 0; b < en.n; ++b) {
            p.ids[r + b] = en.ids[b];
            p.pos[r + b] = (uint32_t)(len + b);
        }
        const size_t extent = en.extent ? en.extent : en.n;
        p.runs[e] = backend::RowRun{r + en.n, extent};
        // The head reads one row per entry as a generated token's, or every row of a scored text as its prompt's.
        if (en.want_logits && en.every_logits) {
            for (size_t b = 0; b < en.n; ++b) p.pick[w++] = (uint32_t)(r + b);
            p.head_runs.push_back(backend::RowRun{w, extent});
        }
        r += en.n;
        if (en.want_logits && !en.every_logits) {
            p.pick[w++] = (uint32_t)(r - 1);
            p.head_runs.push_back(backend::RowRun{w, 1});
        }
    }
    p.long_runs = false;
    for (size_t e = 0; e < n_entries; ++e) p.long_runs = p.long_runs || streams(p, e);
    // Last, once nothing can fail: the checks above left a free slot for each.
    for (size_t e = 0; fresh && e < n_entries; ++e) entries[e].seq->state_.take(slots_);
    for (size_t e = 0; e < n_entries; ++e) entries[e].seq->mark_.ran = entries[e].seq->mark_.held();
    for (size_t e = 0; keeps && e < n_entries; ++e)
        if (entries[e].keep) p.kept[e] = Checkpoint(slots_, slots_.acquire_kept(), p.start[e] + entries[e].n);
}

// Stage s of a pass: its storage's blocks reserved, the residual embedded or received from the stage before, its layers, then the head after the last stage or the residual sent on, its submissions, and the commit of its length and its storage.
inline void Model::run_stage(ExecContext& ctx, Pass& p, size_t s) {
    const Stage& st = stages_[s];
    // Backends may be shared by models used in turn.
    // Keep each stage's evidence with its model, including failures, while preserving direct backend evidence.
    struct Paths {
        const std::vector<std::unique_ptr<Device>>& devices;
        const std::vector<size_t>& touches;
        size_t width;
        void swap() const noexcept {
            for (size_t d : touches)
                for (size_t m = d; m < d + width; ++m) devices[m]->b->swap_matrix_paths(devices[m]->matrix_paths);
        }
        ~Paths() { swap(); }
    } paths{devices_, st.touches, width_};
    paths.swap();
    if (width_ > 1) {
        group_stage(ctx, p, s);
        return;
    }
    Device& home = *devices_[st.device];
    const int storage = home.storage_index;
    for (size_t e = 0; storage >= 0 && e < p.entries.size(); ++e) {
        KVSequence& kv = p.entries[e].seq->kv_[(size_t)storage];
        kv.prepare(p.entries[e].n);
        p.views[(size_t)storage][e] = kv.view(home.storage.get());
        p.views[(size_t)storage][e].extent = p.runs[e].extent;
    }
    // A state is read from where the history this stage has committed left it, the live slot or a checkpoint, and written to the live slot, or for a keep entry to its checkpoint's.
    for (size_t e = 0; home.states && e < p.entries.size(); ++e) {
        const Sequence& q = *p.entries[e].seq;
        const size_t src = q.from_[s] == Sequence::kLive ? q.state_.slot() : q.from_[s];
        const size_t dst = p.kept[e].held() ? p.kept[e].slot() : q.state_.slot();
        p.states[st.device][e] = backend::StateView{home.states.get(), src, dst, q.stage_length(s), p.entries[e].n};
    }
    size_t cur = st.device;
    const backend::RowRuns all{p.runs.data(), p.runs.size()};
    if (s == 0) {
        cur = (size_t)place_.embed_device;
        arch_->embed(part(ctx, cur, pass_.data(), 0, 0, p.rows, all), p.ids.data());
    } else if (p.at != cur) {
        receive(ctx, p.at, p.handoff, p.sent, cur, 0, p.rows);
    }
    for (int l = st.first; l < st.end; l++) {
        if (st.device != cur) { cross(ctx, cur, st.device, 0, p.rows); cur = st.device; }
        arch_->mixer(mixer_part(ctx, p, cur, l));
        if (plan_.layers[(size_t)l].cache == Cache::state) save(ctx, p, cur, l);
        if (p.long_runs && stream_device_[(size_t)l] == (int)cur) {
            ffn_split(ctx, p, cur, l);
            continue;
        }
        const size_t f = (size_t)place_.ffn_device[(size_t)l];
        if (f != cur) { cross(ctx, cur, f, 0, p.rows); cur = f; }
        arch_->ffn(part(ctx, cur, home_[(size_t)l].data(), plan_.layers[(size_t)l].kind, 0, p.rows, all));
    }
    if (s + 1 < stages_.size()) {
        // A residual already where the next stage runs stays there.
        if (cur != stages_[s + 1].device) send(ctx, cur, p.handoff, 0, p.rows);
        p.at = cur;
    } else {
        const size_t o = (size_t)place_.output_device;
        if (o != cur) { cross(ctx, cur, o, 0, p.rows); cur = o; }
        if (plan_.drafter) draft_context(ctx, p, s);
        if (p.want)
            arch_->head(HeadStep{part(ctx, cur, pass_.data(), 0, 0, p.rows, all), p.pick.data(), p.want,
                                 backend::RowRuns{p.head_runs.data(), p.head_runs.size()},
                                 {ctx.logits_buf.get(), p.logits_base * plan_.vocab}});
    }
    end_stage(ctx, p, s, cur);
}

// Stage s on a tensor group (docs/TENSOR-SPLIT.md, section 4.3): every member runs each part on its shards over its own residual, the group's collective sums the members' partial rows into every member's residual after each part, the residual comes in to every member and leaves from the first, and after the last stage each member's slice of the logits rows is gathered into the context's.
// The placement holds the embedding on the first stage's group, the head on the last's and every feed-forward block beside its mixer, and a group runs no layer that keeps a state (the constructor's checks).
inline void Model::group_stage(ExecContext& ctx, Pass& p, size_t s) {
    group_prepare(p, s);
    group_record(ctx, p, s);
    stage_commit(ctx, p, s);
}

// What a group's stage takes of its histories and their pool before any device work: each entry's blocks and its views on every member.
inline void Model::group_prepare(Pass& p, size_t s) {
    const size_t g = stages_[s].device, W = width_;
    const int storage = devices_[g]->storage_index;
    for (size_t e = 0; storage >= 0 && e < p.entries.size(); ++e) {
        KVSequence& kv = p.entries[e].seq->kv_[(size_t)storage];
        kv.prepare(p.entries[e].n);
        for (size_t m = 0; m < W; ++m) {
            backend::KVView& v = p.views[(size_t)storage * W + m][e];
            v = kv.view(devices_[g + m]->storage.get());
            v.extent = p.runs[e].extent;
        }
    }
}

// A group's stage on its devices, through its last submission: it touches the pass, the stage's devices and the context's tickets of those devices, and no history or pool.
inline void Model::group_record(ExecContext& ctx, Pass& p, size_t s) {
    const Stage& st = stages_[s];
    const size_t g = st.device, W = width_;
    const backend::RowRuns all{p.runs.data(), p.runs.size()};
    backend::Collective& sum = *ctx.collectives[g];
    std::vector<backend::Slice> x(W);
    for (size_t m = 0; m < W; ++m) x[m] = slot(ctx, g + m, 0);
    for (size_t m = 0; m < W; ++m) {
        if (s == 0) arch_->embed(part(ctx, g + m, pass_row(m), 0, 0, p.rows, all), p.ids.data());
        else receive(ctx, p.at, p.handoff, p.sent, g + m, 0, p.rows);
    }
    for (int l = st.first; l < st.end; l++) {
        for (size_t m = 0; m < W; ++m) {
            Step step = mixer_part(ctx, p, g + m, l);
            step.width = W;
            step.partial = sum.partial(m);
            arch_->mixer(step);
        }
        sum.sum_into(x, p.rows, plan_.residual);
        for (size_t m = 0; m < W; ++m) {
            Step step = part(ctx, g + m, home_row(m, (size_t)l), plan_.layers[(size_t)l].kind, 0, p.rows, all);
            step.width = W;
            step.partial = sum.partial(m);
            arch_->ffn(step);
        }
        sum.sum_into(x, p.rows, plan_.residual);
    }
    if (s + 1 < stages_.size()) {
        send(ctx, g, p.handoff, 0, p.rows);
        p.at = g;
    } else if (p.want) {
        // Each member's vocabulary rows of every row that wants logits, each copied by its member into its slice of the rows in host memory the members import, behind no wait; the pass's logits wait for every member's submission.
        const size_t V = plan_.vocab;
        for (size_t m = 0, at = 0; m < W; ++m) {
            HeadStep head{part(ctx, g + m, pass_row(m), 0, 0, p.rows, all), p.pick.data(), p.want,
                          backend::RowRuns{p.head_runs.data(), p.head_runs.size()}, {ctx.member_logits[g + m].get(), 0}};
            head.width = W;
            arch_->head(head);
            const size_t n = head_rows(m);
            for (size_t r = 0; r < p.want; ++r)
                devices_[g + m]->b->copy(*ctx.member_rows[g + m], ((p.logits_base + r) * V + at) * sizeof(float), *ctx.member_logits[g + m],
                                         r * n * sizeof(float), n * sizeof(float));
            at += n;
        }
    }
    stage_submit(ctx, p, s, g);
}

// The end of stage s, the residual or the head last on device `cur`: every device the stage recorded on submits, each member of a tensor group, the pass's ticket that of `cur`, and each entry's history commits the stage.
inline void Model::end_stage(ExecContext& ctx, Pass& p, size_t s, size_t cur) {
    stage_submit(ctx, p, s, cur);
    stage_commit(ctx, p, s);
}

// The stage's submissions, every device it recorded on and each member of a tensor group, the pass's ticket that of `cur`.
inline void Model::stage_submit(ExecContext& ctx, Pass& p, size_t s, size_t cur) {
    for (size_t d : stages_[s].touches)
        for (size_t m = d; m < d + width_; ++m) ctx.tickets[m] = devices_[m]->b->submit();
    p.sent = ctx.tickets[cur];
    p.sent_members.assign(ctx.tickets.begin() + (std::ptrdiff_t)cur, ctx.tickets.begin() + (std::ptrdiff_t)(cur + width_));
}

// Each entry's history commits the stage, on the thread that owns the histories.
inline void Model::stage_commit(ExecContext& ctx, Pass& p, size_t s) {
    const Stage& st = stages_[s];
    const int storage = devices_[st.device]->storage_index;
    for (size_t e = 0; e < p.entries.size(); ++e) {
        Sequence& q = *p.entries[e].seq;
        if (storage >= 0) q.kv_[(size_t)storage].commit();
        else q.length_[s] += p.entries[e].n;
        for (size_t d : st.touches)
            for (size_t m = d; m < d + width_; ++m) q.last_[m] = ctx.tickets[m];
        if (!state_layers_) continue;
        q.from_[s] = p.kept[e].held() ? p.kept[e].slot() : Sequence::kLive;
        // Once every stage holds it, the checkpoint is the sequence's, and the one it replaces goes: a later write of that slot is enqueued after every read of it on each device's stream.
        if (s + 1 == stages_.size() && p.kept[e].held()) q.kept_ = std::move(p.kept[e]);
    }
}

// An embedded drafter's context rows of the pass, on the head's device after the last stage (Architecture::draft_rows): each entry's first row reads the row its sequence carries, from where its history left it on the last stage, or a zero row for an empty history.
// Then each entry's last normed row is carried into the slot its state is written to, and a marked entry's rows are saved for its retract (save_h).
inline void Model::draft_context(ExecContext& ctx, Pass& p, size_t s) {
    const size_t o = (size_t)place_.output_device;
    Device& d = *devices_[o];
    const size_t E = plan_.residual, n = p.entries.size();
    ctx.carry.resize(n);
    for (size_t e = 0; e < n; ++e) {
        const Sequence& q = *p.entries[e].seq;
        const size_t src = q.from_[s] == Sequence::kLive ? q.state_.slot() : q.from_[s];
        ctx.carry[e] = q.stage_length(s) ? backend::CSlice{d.carry.get(), src * E} : backend::CSlice{d.zero.get(), 0};
    }
    const backend::RowRuns all{p.runs.data(), p.runs.size()};
    DraftRowsStep step{part(ctx, o, drafter_.data(), plan_.drafter->kind, 0, p.rows, all), p.ids.data(), ctx.carry.data()};
    step.views = p.views[(size_t)d.storage_index].data();
    step.n_views = n;
    step.kv_layer = drafter_kv_;
    step.pos = p.pos.data();
    arch_->draft_rows(step);
    const backend::Slice hn = slot(ctx, o, plan_.draft_h);
    for (size_t e = 0; e < n; ++e) {
        const Sequence& q = *p.entries[e].seq;
        const size_t dst = p.kept[e].held() ? p.kept[e].slot() : q.state_.slot();
        d.b->copy(*d.carry, dst * E * sizeof(float), *hn.buffer, (hn.offset + (p.runs[e].end - 1) * E) * sizeof(float), E * sizeof(float));
    }
    save_h(ctx, p);
}

// After the last stage: where the logits are and the ticket that says they are ready, the pass's own head's.
inline void Model::finish(ExecContext& ctx, const Pass& p) {
    ctx.n_logits = p.want;
    ctx.backend = devices_[(size_t)place_.output_device]->b.get();
    ctx.ticket = p.sent;
    ctx.member_waits.clear();
    for (size_t m = 1; m < p.sent_members.size(); ++m) ctx.member_waits.push_back({devices_[(size_t)place_.output_device + m]->b.get(), p.sent_members[m]});
    ctx.pending = p.want > 0;
}

// A failed pass: every device drained, then every entry's histories back to where the pass found them, or on a model that keeps a state, whose live state the pass may have written, to its checkpoint (rewind); the blocks and the checkpoint slots its stages reserved or wrote returned.
inline void Model::roll_back(Pass& p) noexcept {
    retire();
    for (size_t e = 0; e < p.entries.size(); ++e) rewind(*p.entries[e].seq, p.start[e]);
    p.kept.clear();
}

// One backend allocation holds the activation slots of a pass, each aligned to 64 bytes.
// Device allocators handle a few large blocks far better than many small ones, and resizing is one call.
// The caller only publishes the result once this returns, so an allocation that throws leaves the previous arena intact.
inline backend::BufferPtr Model::alloc_arena(backend::Backend& b, const std::vector<size_t>& counts, std::vector<size_t>& offsets) const {
    size_t total = 0;
    offsets.resize(counts.size());
    for (size_t i = 0; i < counts.size(); ++i) {
        offsets[i] = total;
        const size_t bytes = counts[i] * sizeof(float);
        if (bytes / sizeof(float) != counts[i] || total > (size_t)-1 - bytes - 63)
            throw std::runtime_error("inference: activation arena size overflows");
        total = (total + bytes + 63) / 64 * 64;
    }
    return b.alloc(total);
}

// Storage for a pass of `rows` rows with `want` logits rows, on every device the placement uses, with `buffers` handoff buffers on each device a crossing leaves (handoffs): grown when a pass needs more rows than the context holds, never shrunk.
// Each is allocated whole before it replaces what the context had.
inline void Model::ensure(ExecContext& ctx, size_t rows, size_t want, size_t buffers) {
    auto mul = [](size_t a, size_t b) {
        if (b && a > (size_t)-1 / b)
            throw std::runtime_error("inference: activation arena size overflows");
        return a * b;
    };
    ctx.scratch.resize(devices_.size());
    ctx.tickets.resize(devices_.size(), 0);
    // The run list a part rebuilds and a streamed layer's groups hold at most a run per entry, and so per row, so no part grows them.
    ctx.entry_runs.reserve(rows);
    ctx.part_runs.reserve(rows);
    for (size_t d = 0; d < devices_.size(); ++d) {
        ExecContext::Scratch& sc = ctx.scratch[d];
        if (!devices_[d]->used || (sc.arena && sc.rows >= rows)) continue;
        std::vector<size_t> counts(plan_.slots.size()), offsets;
        for (size_t i = 0; i < counts.size(); ++i) counts[i] = mul(rows, plan_.slots[i]);
        backend::BufferPtr arena = alloc_arena(*devices_[d]->b, counts, offsets);
        // A pass through this context may still run on the arena being replaced: nothing else waits for a pass that wanted no logits.
        if (sc.arena) devices_[d]->b->wait(ctx.tickets[d]);
        sc.arena = std::move(arena);
        sc.offset = std::move(offsets);
        sc.rows = rows;
    }
    // The host-visible buffers a crossing leaves each sending device through.
    size_t used = 0;
    for (const auto& d : devices_) used += d->used;
    if (used > 1 && ctx.handoff_rows < rows) {
        std::vector<std::vector<backend::BufferPtr>> handoff(devices_.size());
        const size_t bytes = mul(mul(rows, plan_.residual), sizeof(float));
        for (size_t d = 0; d < devices_.size(); ++d)
            for (size_t i = 0; devices_[d]->sends && i < buffers; ++i)
                handoff[d].push_back(devices_[d]->b->alloc(bytes, backend::Memory::host_visible));
        for (size_t d = 0; d < ctx.handoff.size(); ++d)
            if (devices_[d]->used) devices_[d]->b->wait(ctx.tickets[d]);
        ctx.handoff = std::move(handoff);
        ctx.handoff_rows = rows;
    }
    if (want && (!ctx.logits_buf || ctx.logit_rows < want)) {
        // The head writes here and the host reads it in place once the pass has retired: the one point per pass that must be host visible, and the one wait per pass.
        // On a tensor split it is host memory each member of the head's group imports, so each copies its slice of the rows there.
        const size_t o = (size_t)place_.output_device, bytes = mul(mul(want, plan_.vocab), sizeof(float));
        if (ctx.logits_buf)
            for (size_t m = o; m < o + width_; ++m) devices_[m]->b->wait(ctx.tickets[m]);
        if (width_ == 1) {
            ctx.logits_buf = devices_[o]->b->alloc(bytes, backend::Memory::host_visible);
        } else {
            core::HostPages pages(bytes);
            std::vector<backend::BufferPtr> views(devices_.size());
            for (size_t m = o; m < o + width_; ++m) {
                views[m] = devices_[m]->b->wrap_host(pages.data(), pages.size());
                if (!views[m]) throw std::runtime_error("inference: a device of the head's tensor group cannot import the host memory of the logits rows");
            }
            ctx.logits_buf = views[o];
            ctx.member_rows = std::move(views);
            ctx.logits_host = std::move(pages);
        }
        ctx.logit_rows = want;
        ctx.width = plan_.vocab;
    }
    if (width_ == 1) return;
    // A tensor group's collective for the rows the arenas now hold, and each member of the head's group its slice of the logits rows.
    ctx.collectives.resize(devices_.size());
    ctx.member_logits.resize(devices_.size());
    for (size_t g = 0; g < devices_.size(); g += width_) {
        if (!devices_[g]->used || (ctx.collectives[g] && ctx.collective_rows >= rows)) continue;
        std::vector<backend::Backend*> members;
        for (size_t m = g; m < g + width_; ++m) {
            members.push_back(devices_[m]->b.get());
            devices_[m]->b->wait(ctx.tickets[m]);
        }
        ctx.collectives[g] = devices_[g]->b->join(members, rows, plan_.residual);
        if (!ctx.collectives[g]) throw std::runtime_error("inference: the backends of a tensor group have no cross-device sum");
    }
    ctx.collective_rows = std::max(ctx.collective_rows, rows);
    const size_t o = (size_t)place_.output_device;
    for (size_t m = 0; want && m < width_; ++m) {
        backend::BufferPtr& b = ctx.member_logits[o + m];
        const size_t bytes = mul(mul(want, head_rows(m)), sizeof(float));
        if (b && b->size() >= bytes) continue;
        devices_[o + m]->b->wait(ctx.tickets[o + m]);
        b = devices_[o + m]->b->alloc(bytes, backend::Memory::host_visible);
    }
}

// The residual stream moves from one device's x slot to another's through host memory, `rows` rows from `base`; a few kilobytes on a decode token.
// `send` copies them into the source's host-visible handoff buffer inside the source's own work, so they outlast the source moving on to its next pass, and the submission that carries the copy says when they are there.
inline void Model::send(ExecContext& ctx, size_t from, size_t handoff, size_t base, size_t rows) {
    const size_t E = plan_.residual;
    const backend::Slice x = slot(ctx, from, 0);
    devices_[from]->b->copy(*ctx.handoff[from][handoff], base * E * sizeof(float), *x.buffer,
                            (x.offset + base * E) * sizeof(float), rows * E * sizeof(float));
}

// `receive` waits for that submission and writes the rows into the destination's residual, enqueued there.
inline void Model::receive(ExecContext& ctx, size_t from, size_t handoff, backend::Ticket sent, size_t to, size_t base, size_t rows) {
    const size_t E = plan_.residual;
    devices_[from]->b->wait(sent);
    const backend::Slice x = slot(ctx, to, 0);
    const uint8_t* rows_out = (const uint8_t*)ctx.handoff[from][handoff]->host_ptr() + base * E * sizeof(float);
    devices_[to]->b->write(*x.buffer, (x.offset + base * E) * sizeof(float), rows_out, rows * E * sizeof(float));
}

// A crossing inside a stage, both halves at once.
inline void Model::cross(ExecContext& ctx, size_t from, size_t to, size_t base, size_t rows) {
    send(ctx, from, 0, base, rows);
    receive(ctx, from, 0, devices_[from]->b->submit(), to, base, rows);
}

// A streamed layer in a pass with long runs: consecutive entries alike form a group, the long ones run where the residual is on the layer's streamed row, with each window role's bytes written into its window once, and the rest on the host through a crossing each way.
// The residual ends where it started, on the layer's mixer device.
inline void Model::ffn_split(ExecContext& ctx, const Pass& p, size_t dev, int l) {
    const std::vector<Weight>& home = home_[(size_t)l];
    const std::vector<Weight>& streamed = stream_[(size_t)l];
    const size_t host = (size_t)place_.ffn_device[(size_t)l];
    const uint8_t kind = plan_.layers[(size_t)l].kind;
    bool copied = false;
    for (size_t e = 0, base = 0; e < p.runs.size();) {
        const bool on_device = streams(p, e);
        ctx.part_runs.clear();
        size_t end = base;
        for (; e < p.runs.size() && streams(p, e) == on_device; ++e) {
            end = p.runs[e].end;
            ctx.part_runs.push_back(backend::RowRun{end - base, p.runs[e].extent});
        }
        const backend::RowRuns runs{ctx.part_runs.data(), ctx.part_runs.size()};
        if (on_device) {
            if (!copied) {
                backend::Backend& b = *devices_[dev]->b;
                for (const Role& role : plan_.layers[(size_t)l].roles)
                    if (role.stream == Stream::window)
                        b.write(*streamed[role.id].data, 0, home[role.id].data->host_ptr(), home[role.id].data->size());
                copied = true;
            }
            arch_->ffn(part(ctx, dev, streamed.data(), kind, base, end - base, runs));
        } else {
            cross(ctx, dev, host, base, end - base);
            arch_->ffn(part(ctx, host, home.data(), kind, base, end - base, runs));
            cross(ctx, host, dev, base, end - base);
        }
        base = end;
    }
}

// A call of a part on device `dev`: `rows` rows of the residual from row `base` with their runs, and the row of weights `w` by role id, the layer's kind with it.
inline Step Model::part(ExecContext& ctx, size_t dev, const Weight* w, uint8_t kind, size_t base, size_t rows, backend::RowRuns runs) const {
    const ExecContext::Scratch& sc = ctx.scratch[dev];
    const Device& d = *devices_[dev];
    return Step{*d.b, sc.arena.get(), sc.offset.data(), {sc.arena.get(), sc.offset[0] / sizeof(float) + base * plan_.residual},
                rows, runs, w, kind, nullptr, 0, 0, nullptr, d.tables.data(), &ctx.entry_runs, nullptr, 0, options_.device_dtypes.empty() ? options_.dtype : options_.device_dtypes[dev]};
}

// Layer l's mixer over every row of the pass, with the cache views of its device's storage when the layer keeps KV, and the rows' positions.
inline Step Model::mixer_part(ExecContext& ctx, const Pass& p, size_t dev, int l) const {
    const Device& d = *devices_[dev];
    const LayerPlan& layer = plan_.layers[(size_t)l];
    Step s = part(ctx, dev, home_row(d.member, (size_t)l), layer.kind, 0, p.rows, {p.runs.data(), p.runs.size()});
    if (layer.cache == Cache::kv) {
        s.views = p.views[(size_t)d.storage_index * width_ + d.member].data();
        s.n_views = p.entries.size();
        s.kv_layer = (size_t)d.local_layer[(size_t)l];
    } else if (layer.cache == Cache::state) {
        s.states = p.states[dev].data();
        s.n_views = p.entries.size();
        s.state_layer = (size_t)d.local_layer[(size_t)l];
    }
    s.pos = p.pos.data();
    return s;
}

} // namespace infer
