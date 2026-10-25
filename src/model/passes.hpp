#pragma once
#include "model/runtime.hpp"
#include "model/stage_lanes.hpp"

// A pass, the one owner of how it runs: its plan, its layers' stages over the devices, the residual's crossings, a streamed layer's split, the embedded drafter's rows and the storage of the context it runs in.
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

// Size a fresh context once, before any pass, for `slots` passes in flight (above one they need a pipelined placement) of up to `rows` rows each, with their handoff buffers and `logit_rows` rows of logits (begin_pass's logits_base).
// A failed reservation leaves the context fresh, and from then on it is frozen: begin_pass refuses a pass that needs more before any work, and forward refuses it too.
inline void Model::reserve_passes(ExecContext& ctx, size_t slots, size_t rows, size_t logit_rows, size_t apart_rows) {
    if (ctx.slots || !ctx.scratch.empty()) throw std::logic_error("inference: reserve_passes takes a fresh context, once");
    if (!slots || !rows) throw std::logic_error("inference: reserve_passes needs a slot and a row");
    if (apart_rows && !paces()) throw std::logic_error("inference: a pass apart needs stages of one device each, none on the host");
    if (slots > 1 && !pipelined_ && !apart_rows) throw std::logic_error("inference: passes in flight need a pipelined placement");
    ExecContext reserved;
    ensure(reserved, rows, logit_rows, handoffs(slots), apart_rows);
    reserved.passes.assign(slots, Pass{});
    reserved.slots = slots;
    reserved.pass_rows = rows;
    ctx = std::move(reserved);
}

// Plan a pass in `slot` of a reserved context, its tokens copied and its wanting rows written from logits row `logits_base` on, its sequences in flight until end_pass or abort_pass; what it refuses changes nothing.
// A sequence in flight continues its prompt only in a pass of its own, each of its passes in flight holding it alone, past its first stage and not wanting its logits (docs/MULTI-DEVICE.md, phase 4).
inline void Model::begin_pass(ExecContext& ctx, size_t slot, const BatchEntry* entries, size_t n_entries, size_t logits_base, bool apart) {
    if (!ctx.slots) throw std::logic_error("inference: begin_pass needs a context reserved for passes");
    if (slot >= ctx.slots) throw std::logic_error("inference: a pass slot beyond the reservation");
    Pass& p = ctx.passes[slot];
    if (p.in_flight) throw std::logic_error("inference: a pass slot already in flight");
    size_t rows = 0;
    bool continues = false;
    for (size_t e = 0; entries && e < n_entries; ++e) {
        rows += entries[e].n;
        const Sequence* q = entries[e].seq;
        if (!q || !q->flights_) continue;
        if (n_entries != 1) throw std::logic_error("inference: a sequence in flight continues in a pass of its own");
        if (q->mark_.held()) throw std::logic_error("inference: a marked sequence in flight continued");
        for (const Pass& o : ctx.passes)
            for (size_t i = 0; o.in_flight && i < o.entries.size(); ++i) {
                if (o.entries[i].seq != q) continue;
                if (o.entries.size() != 1) throw std::logic_error("inference: a sequence continued while its pass in flight holds other sequences");
                if (!o.ran) throw std::logic_error("inference: a sequence continued before its pass in flight committed its first stage");
                if (o.entries[i].want_logits) throw std::logic_error("inference: a sequence continued while its pass in flight wants its logits");
            }
        continues = true;
    }
    if (apart && rows > ctx.apart_rows) throw std::logic_error("inference: a pass apart beyond the rows reserve_passes reserved for one");
    begin(ctx, p, entries, n_entries, logits_base, continues);
    p.apart = apart;
    for (size_t e = 0; e < n_entries; ++e) ++entries[e].seq->flights_;
    p.handoff = slot;
    p.in_flight = true;
}

// Stage s of the pass in `slot` in one call, as the scheduler's inline and host stages, the layer split check and the tests take it: the stage after the last one run, its storage, residual, layers, head or handoff, submissions and commit.
// A failure aborts the pass alone, after `before_undo`, which a caller recording other stages on threads uses to quiet them since undoing drains every device; a later pass of its sequence is the caller's to abort.
inline void Model::run_pass_stage(ExecContext& ctx, size_t slot, size_t s, const std::function<void()>& before_undo) {
    Pass& p = in_flight(ctx, slot);
    if (s != p.ran || s >= stages_.size()) throw std::logic_error("inference: a pass's stages run in order, each once");
    at_stage(p, s);
    try {
        run_stage(ctx, p, s);
    } catch (...) {
        if (before_undo) before_undo();
        roll_back(p);
        release(p);
        throw;
    }
    ++p.ran;
}

// A stage that waits on its devices (stage_waits: a tensor group's, and past the first a device's on a pipelined split) in three calls, for a caller that records it on the stage's own thread.
// prepare_pass_stage and commit_pass_stage run on the thread that owns the histories and pools, record_pass_stage on the stage's thread; a failure of the first two undoes nothing, so the caller abandons the pass with abort_pass.
inline void Model::prepare_pass_stage(ExecContext& ctx, size_t slot, size_t s) {
    Pass& p = in_flight(ctx, slot);
    if (s != p.ran || s >= stages_.size()) throw std::logic_error("inference: a pass's stages run in order, each once");
    if (stage_on_host(s) && !stage_grouped(s)) throw std::logic_error("inference: a stage on the host is recorded whole, by run_pass_stage");
    prepare_stage(p, s);
}

// With a `pace` a stage of one device is fed in pieces (Pace).
inline void Model::record_pass_stage(ExecContext& ctx, size_t slot, size_t s, Pace* pace) { record_stage(ctx, in_flight(ctx, slot), s, pace); }

// What stage s of a pass takes of its histories and pool before any device work, on the thread that owns them.
inline void Model::prepare_stage(Pass& p, size_t s) {
    if (width_ > 1) group_prepare(p, s);
    else single_prepare(p, s);
}

// Stage s of a pass recorded and submitted on its devices, the thread's for as long as the stage's waits hold it; the matrix paths it takes stay with this model.
inline void Model::record_stage(ExecContext& ctx, Pass& p, size_t s, Pace* pace) {
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
    // A pass recorded at a boundary is told once it has retired, by the piece after it; after the last piece, or a failure, none follows, so it is settled here.
    const auto settle = [&] {
        if (!pace || !pace->within) return;
        devices_[st.device]->b->wait(pace->within);
        pace->within = 0;
        pace->passed();
    };
    try {
        if (width_ > 1) group_record(ctx, p, s, pace);
        else single_record(ctx, p, s, pace);
    } catch (...) {
        settle();
        throw;
    }
    settle();
}

// The same for a pass begun apart, from a Pace's `between`, on the thread and inside the recording of another pass's stage s: the stage's devices are that recording's already.
inline void Model::record_pass_within(ExecContext& ctx, size_t slot, size_t s) {
    Pass& p = in_flight(ctx, slot);
    if (!p.apart) throw std::logic_error("inference: only a pass begun apart is recorded within another's stage");
    if (width_ > 1) group_record(ctx, p, s, nullptr);
    else single_record(ctx, p, s, nullptr);
}

// Whether recording stage s of the pass in `slot` would wait for nothing: the stage before it has retired on its device, asked without waiting.
inline bool Model::stage_ready(ExecContext& ctx, size_t slot, size_t s) {
    const Pass& p = in_flight(ctx, slot);
    return s == 0 || p.at == stages_[s].device || devices_[p.at]->b->done(p.sent);
}

inline void Model::commit_pass_stage(ExecContext& ctx, size_t slot) {
    Pass& p = in_flight(ctx, slot);
    stage_commit(p, p.ran);
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
// Later passes of its sequences go first, newest first, so the history ends where this pass found it; their slots are returned, abandoned too.
inline std::vector<size_t> Model::abort_pass(ExecContext& ctx, size_t slot) {
    Pass& p = in_flight(ctx, slot);
    std::vector<size_t> later;
    for (size_t k = 0; k < ctx.passes.size(); ++k) {
        const Pass& o = ctx.passes[k];
        if (k == slot || !o.in_flight) continue;
        for (size_t e = 0; e < p.entries.size(); ++e)
            if (o.entries.size() == 1 && o.entries[0].seq == p.entries[e].seq && o.start[0] > p.start[e]) later.push_back(k);
    }
    std::sort(later.begin(), later.end(), [&](size_t a, size_t b) { return ctx.passes[a].start[0] > ctx.passes[b].start[0]; });
    for (size_t k : later) {
        roll_back(ctx.passes[k]);
        release(ctx.passes[k]);
    }
    roll_back(p);
    release(p);
    return later;
}

// A pass's plan: its rows in entry order, their positions after each history, and the rows the head reads, from logits row `logits_base` on, with nothing reserved yet, since each stage reserves the blocks of the storage it writes.
// A context reserved for passes is not grown: a pass that needs more rows or logits rows than it holds is refused here; a sequence in flight is refused unless begin_pass found it `continues`.
inline void Model::begin(ExecContext& ctx, Pass& p, const BatchEntry* entries, size_t n_entries, size_t logits_base, bool continues) {
    if (!entries || !n_entries) throw std::runtime_error("inference: empty batch");
    size_t rows = 0, want = 0;
    for (size_t e = 0; e < n_entries; ++e) {
        const BatchEntry& en = entries[e];
        if (!en.seq || en.seq->owner_ != this)
            throw std::runtime_error("inference: batch entry without a sequence of this model");
        if (en.seq->flights_ && !continues) throw std::logic_error("inference: a sequence already in flight");
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
    for (size_t e = 0; e < n_entries; ++e)
        for (size_t f = 0; f < e; ++f)
            if (entries[f].seq == entries[e].seq) throw std::logic_error("inference: a sequence listed twice in a pass");
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
    for (size_t e = 0; keeps && e < n_entries; ++e) {
        if (entries[e].keep && !whole_blocks(history(*entries[e].seq) + entries[e].n))
            throw std::logic_error("inference: a checkpoint at a position of whole blocks in every storage");
        if (entries[e].keep) refuse_media(*entries[e].seq, history(*entries[e].seq) + entries[e].n, "a checkpoint");
    }
    // A pass holding an image's rows gives every row the plan's three position streams; any other keeps one.
    bool image_rows = false;
    for (size_t e = 0; e < n_entries; ++e) image_rows = image_rows || meets_media(*entries[e].seq, history(*entries[e].seq), entries[e].n);
    if (!ctx.slots) ensure(ctx, rows, want, handoffs(1));
    p.entries.assign(entries, entries + n_entries);
    p.start.resize(n_entries);
    p.rows = rows;
    p.want = want;
    p.handoff = 0;
    p.logits_base = logits_base;
    p.ran = 0;
    p.ids.resize(rows);
    p.previous_tokens.resize(backend::size_mul(n_entries, token_width_));
    p.image_rows = image_rows;
    p.pos.resize(image_rows ? backend::size_mul(rows, 3) : rows);
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
        for (size_t b = 0; b < en.n; ++b) p.ids[r + b] = en.ids[b];
        positions(*en.seq, len, en.n, image_rows, p.pos.data() + r * (image_rows ? 3 : 1));
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

// Stage s of a pass: its storage's blocks reserved, the residual embedded or received, its layers, then the head after the last stage or the residual sent on, its submissions, and the commit of its length and storage.
inline void Model::run_stage(ExecContext& ctx, Pass& p, size_t s) {
    prepare_stage(p, s);
    record_stage(ctx, p, s, nullptr);
    stage_commit(p, s);
}

// A prompt's chunks through a pipelined split (Model::prefill), each device taking its chunks in order and each stage that waits on its devices (stage_waits) recording on a thread of its own.
// This thread begins, prepares and commits every stage and records the rest; when a stage may take a chunk is docs/src/model-runtime.md's prefill.
inline void Model::run_chunks(ExecContext& ctx, size_t chunks, const std::function<BatchEntry(size_t)>& chunk) {
    const size_t S = stages_.size();
    if (ctx.passes.size() < S) ctx.passes.resize(S);
    // Each stage's hand-off under `m`: the chunk given to its lane, those it has recorded and its failure.
    struct Lane {
        std::optional<size_t> todo;
        std::deque<size_t> done;
        std::exception_ptr error;
    };
    std::mutex m;
    std::condition_variable cv;   // this thread's wake, as a lane ends a record
    std::vector<std::unique_ptr<Lane>> lanes(S);
    std::vector<size_t> next(S, 0), committed(S, 0);
    std::vector<char> busy(S, 0);
    std::exception_ptr failed;
    std::optional<StageLanes> threads;
    try {
        threads.emplace();
        for (size_t s = 0; s < S; ++s) {
            if (!stage_waits(s)) continue;
            lanes[s] = std::make_unique<Lane>();
            Lane* lane = lanes[s].get();
            threads->add(s, m);
            StageLanes::Policy policy;
            policy.ready = [lane] { return lane->todo.has_value(); };
            policy.step = [this, &ctx, &cv, lane, s, S](std::unique_lock<std::mutex>& lk) {
                const size_t c = *lane->todo;
                lane->todo.reset();
                lk.unlock();
                std::exception_ptr error;
                try { record_stage(ctx, ctx.passes[c % S], s, nullptr); } catch (...) { error = std::current_exception(); }
                lk.lock();
                if (error) lane->error = error;
                else lane->done.push_back(c);
                cv.notify_all();
            };
            threads->start(s, std::move(policy));
        }
        const auto ready = [&](size_t s) {
            const size_t c = next[s];
            if (c >= chunks || busy[s] || committed[s] != c) return false;
            if (s > 0 && committed[s - 1] <= c) return false;
            if (s + 1 < S && committed[s + 1] + 2 <= c) return false;
            return s > 0 || committed[S - 1] + S > c;
        };
        while (committed[S - 1] < chunks) {
            bool moved = false;
            // A round takes each stage ready at its start once, the first stage first, as a step of the software pipeline does.
            std::vector<char> go(S);
            for (size_t s = 0; s < S; ++s) go[s] = ready(s);
            for (size_t s = 0; s < S; ++s) {
                if (!go[s]) continue;
                const size_t c = next[s]++;
                Pass& p = ctx.passes[c % S];
                if (s == 0) {
                    const BatchEntry entry = chunk(c);
                    begin(ctx, p, &entry, 1);
                    p.handoff = c % 2;
                }
                prepare_stage(p, s);
                if (lanes[s]) {
                    {
                        std::lock_guard<std::mutex> lk(m);
                        lanes[s]->todo = c;
                        busy[s] = 1;
                    }
                    threads->notify(s);
                } else {
                    record_stage(ctx, p, s, nullptr);
                    stage_commit(p, s);
                    ++committed[s];
                }
                moved = true;
            }
            std::vector<std::pair<size_t, size_t>> finished;
            {
                std::unique_lock<std::mutex> lk(m);
                const auto arrived = [&] {
                    for (const auto& l : lanes)
                        if (l && (l->error || !l->done.empty())) return true;
                    return false;
                };
                if (!moved && !arrived()) {
                    if (std::none_of(busy.begin(), busy.end(), [](char b) { return b != 0; }))
                        throw std::logic_error("inference: a pipelined prefill with no stage ready and none recording");
                    cv.wait(lk, arrived);
                }
                for (size_t s = 0; s < S; ++s) {
                    if (!lanes[s]) continue;
                    Lane& lane = *lanes[s];
                    if (lane.error) {
                        if (!failed) failed = lane.error;
                        lane.error = nullptr;
                        busy[s] = 0;
                    }
                    for (; !lane.done.empty(); lane.done.pop_front()) {
                        finished.push_back({s, lane.done.front()});
                        busy[s] = 0;
                    }
                }
            }
            if (failed) break;
            for (const auto& [s, c] : finished) {
                stage_commit(ctx.passes[c % S], s);
                ++committed[s];
            }
        }
    } catch (...) {
        if (!failed) failed = std::current_exception();
    }
    // The lanes end their current records and stop before the prompt's transaction is undone or finished.
    threads.reset();
    if (failed) std::rethrow_exception(failed);
    finish(ctx, ctx.passes[(chunks - 1) % S]);
}

// Each entry's history at stage s where the pass found it, before the stage takes anything: an earlier pass of a continued prompt has run the stage first.
inline void Model::at_stage(const Pass& p, size_t s) const {
    for (size_t e = 0; e < p.entries.size(); ++e)
        if (p.entries[e].seq->stage_length(s) != p.start[e])
            throw std::logic_error("inference: a pass at a stage its sequence's earlier pass has not run, or whose history went back");
}

// What a stage of one device takes of its histories and their pool before any device work: each entry's blocks and its view, and the slots its state is read from and written to.
inline void Model::single_prepare(Pass& p, size_t s) {
    at_stage(p, s);
    const Stage& st = stages_[s];
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
    prepare_tokens(p, s);
}

// Start reading a layer's requested rows on the group's first member, as its stage begins (WeightRows::start), so the layers before it record and run meanwhile.
inline void Model::start_lookup(ExecContext& ctx, const Pass& p, size_t group, int l) {
    const auto& lookup = plan_.layers[size_t(l)].lookup;
    if (!lookup || !home_[size_t(l)][lookup->role].row_source) return;
    ExecContext::Scratch& owner = p.apart ? ctx.apart[group] : ctx.scratch[group];
    auto& prepared = owner.lookups[size_t(l)];
    prepared.view = {};
    Step step = mixer_part(ctx, p, group, l);
    step.indices = prepared.ids.data();
    step.indices_size = prepared.ids.size();
    arch_->lookup_ids(step);
    home_[size_t(l)][lookup->role].row_source->start(prepared.ids.data(), backend::size_mul(p.rows, lookup->rows), *prepared.host);
}

// Take a layer's rows read since start_lookup, just before the layer, and upload the packed rows to each member's own buffer.
// The remapped IDs live in that first member's context scratch until the backends consume their gather calls.
inline void Model::finish_lookup(ExecContext& ctx, const Pass& p, size_t group, size_t width, int l) {
    const auto& lookup = plan_.layers[size_t(l)].lookup;
    if (!lookup || !home_[size_t(l)][lookup->role].row_source) return;
    ExecContext::Scratch& owner = p.apart ? ctx.apart[group] : ctx.scratch[group];
    auto& prepared = owner.lookups[size_t(l)];
    const auto rows = home_[size_t(l)][lookup->role].row_source->finish(*prepared.host);
    for (size_t i = 0; i < rows.requested; ++i) prepared.ids[i] = uint32_t(rows.remap[i]);
    const size_t bytes = backend::size_mul(rows.unique, rows.row_bytes);
    for (size_t m = 0; m < width; ++m) {
        const size_t dev = group + m;
        ExecContext::Scratch& member = p.apart ? ctx.apart[dev] : ctx.scratch[dev];
        auto& target = member.lookups[size_t(l)];
        target.view = {};
        devices_[dev]->b->write(*target.data, 0, rows.data, bytes);
        target.view = {{target.data.get(), 0}, prepared.ids.data(), rows.unique};
    }
}

// A stage of one device on its devices, through its submissions: it touches the pass, the devices the stage records on and their tickets, and no history or pool.
// With a `pace` each part ends a piece: what is recorded is submitted, that piece or the one before it waited for, and the caller's `between` runs.
inline void Model::single_record(ExecContext& ctx, Pass& p, size_t s, Pace* pace) {
    const Stage& st = stages_[s];
    size_t cur = st.device;
    const backend::RowRuns all{p.runs.data(), p.runs.size()};
    for (int l = st.first; l < st.end; ++l) start_lookup(ctx, p, st.device, l);
    const auto piece = [&](int l, bool ffn) {
        if (!pace) return;
        backend::Backend& b = *devices_[st.device]->b;
        const backend::Ticket t = b.submit();
        if (pace->within) {
            b.wait(pace->within);
            pace->within = 0;
            pace->passed();
        }
        b.wait(pace->ahead ? pace->fed : t);
        pace->fed = t;
        pace->layer = l;
        pace->ffn = ffn;
        pace->between();
    };
    if (s == 0) {
        cur = (size_t)place_.embed_device;
        arch_->embed(part(ctx, cur, pass_.data(), 0, 0, p.rows, all, p.apart), p.ids.data());
        if (p.image_rows) write_media(ctx, p, cur);
    } else if (p.at != cur) {
        receive(ctx, p.at, p.handoff, p.sent, cur, 0, p.rows, p.apart);
    }
    for (int l = st.first; l < st.end; l++) {
        if (st.device != cur) { cross(ctx, cur, st.device, 0, p.rows); cur = st.device; }
        finish_lookup(ctx, p, cur, 1, l);
        arch_->mixer(mixer_part(ctx, p, cur, l));
        if (plan_.layers[(size_t)l].cache == Cache::state) save(ctx, p, cur, l);
        if (p.long_runs && stream_device_[(size_t)l] == (int)cur) {
            ffn_split(ctx, p, cur, l);
            continue;
        }
        piece(l, false);
        const size_t f = (size_t)place_.ffn_device[(size_t)l];
        if (f != cur) { cross(ctx, cur, f, 0, p.rows); cur = f; }
        arch_->ffn(part(ctx, cur, home_[(size_t)l].data(), plan_.layers[(size_t)l].kind, 0, p.rows, all, p.apart));
        piece(l, true);
    }
    if (s + 1 < stages_.size()) {
        // A residual already where the next stage runs stays there.
        if (cur != stages_[s + 1].device) send(ctx, cur, p.handoff, 0, p.rows, p.apart);
        p.at = cur;
    } else {
        const size_t o = (size_t)place_.output_device;
        if (o != cur) { cross(ctx, cur, o, 0, p.rows); cur = o; }
        if (plan_.drafter) draft_context(ctx, p, s);
        if (p.want)
            arch_->head(HeadStep{part(ctx, cur, pass_.data(), 0, 0, p.rows, all, p.apart), p.pick.data(), p.want,
                                 backend::RowRuns{p.head_runs.data(), p.head_runs.size()},
                                 {ctx.logits_buf.get(), p.logits_base * plan_.vocab}});
    }
    stage_submit(ctx, p, s, cur);
}

// What a group's stage takes of its histories and their pool before any device work: each entry's blocks and its views on every member, and its state's slots, the same on every member, each of which keeps the state of its own heads there.
inline void Model::group_prepare(Pass& p, size_t s) {
    at_stage(p, s);
    const size_t g = stages_[s].device, W = width_;
    const int storage = devices_[g]->storage_index;
    for (size_t e = 0; devices_[g]->states && e < p.entries.size(); ++e) {
        const Sequence& q = *p.entries[e].seq;
        const size_t src = q.from_[s] == Sequence::kLive ? q.state_.slot() : q.from_[s];
        const size_t dst = p.kept[e].held() ? p.kept[e].slot() : q.state_.slot();
        for (size_t m = 0; m < W; ++m) p.states[g + m][e] = backend::StateView{devices_[g + m]->states.get(), src, dst, q.stage_length(s), p.entries[e].n};
    }
    for (size_t e = 0; storage >= 0 && e < p.entries.size(); ++e) {
        KVSequence& kv = p.entries[e].seq->kv_[(size_t)storage];
        kv.prepare(p.entries[e].n);
        for (size_t m = 0; m < W; ++m) {
            backend::KVView& v = p.views[(size_t)storage * W + m][e];
            v = kv.view(devices_[g + m]->storage.get());
            v.extent = p.runs[e].extent;
        }
    }
    prepare_tokens(p, s);
}

// Snapshot the token-history owner's preceding IDs on the history thread, before its stage may be recorded on a worker.
inline void Model::prepare_tokens(Pass& p, size_t s) const {
    if (s != token_stage_) return;
    for (size_t e = 0; e < p.entries.size(); ++e) {
        const backend::StateView& view = p.states[stages_[s].device][e];
        uint32_t* out = p.previous_tokens.data() + e * token_width_;
        if (!view.length) std::fill_n(out, token_width_, token_padding_);
        else std::copy_n(token_tails_.data() + view.src * token_width_, token_width_, out);
    }
}

// A group's stage on its devices, through its last submission: it touches the pass, the stage's devices and their tickets, and no history or pool.
// With a `pace` each sum ends a piece: that sum or the one before it is waited for on every member, and the caller's `between` runs.
inline void Model::group_record(ExecContext& ctx, Pass& p, size_t s, Pace* pace) {
    const Stage& st = stages_[s];
    const size_t g = st.device, W = width_;
    const backend::RowRuns all{p.runs.data(), p.runs.size()};
    backend::Collective& sum = *ctx.collectives[g];
    const auto piece = [&](int l, bool ffn) {
        if (!pace) return;
        if (pace->within) {
            devices_[g]->b->wait(pace->within);
            pace->within = 0;
            pace->passed();
        }
        sum.wait(pace->ahead ? 1 : 0);
        pace->layer = l;
        pace->ffn = ffn;
        pace->between();
    };
    std::vector<backend::Slice> x(W);
    for (size_t m = 0; m < W; ++m) x[m] = slot(ctx, g + m, 0, p.apart);
    for (size_t m = 0; m < W; ++m) {
        if (s == 0) {
            arch_->embed(part(ctx, g + m, pass_row(m), 0, 0, p.rows, all, p.apart), p.ids.data());
            if (p.image_rows) write_media(ctx, p, g + m);
        } else {
            receive(ctx, p.at, p.handoff, p.sent, g + m, 0, p.rows, p.apart);
        }
    }
    for (int l = st.first; l < st.end; ++l) start_lookup(ctx, p, g, l);
    for (int l = st.first; l < st.end; l++) {
        finish_lookup(ctx, p, g, W, l);
        for (size_t m = 0; m < W; ++m) {
            Step step = mixer_part(ctx, p, g + m, l);
            step.width = W;
            step.partial = sum.partial(m);
            arch_->mixer(step);
            if (plan_.layers[(size_t)l].cache == Cache::state) save(ctx, p, g + m, l);
        }
        sum.sum_into(x, p.rows, plan_.residual);
        piece(l, false);
        for (size_t m = 0; m < W; ++m) {
            Step step = part(ctx, g + m, home_row(m, (size_t)l), plan_.layers[(size_t)l].kind, 0, p.rows, all, p.apart);
            step.width = W;
            step.partial = sum.partial(m);
            if (has_experts(plan_.layers[(size_t)l])) clear_partial(ctx, g + m, step.partial, p.rows);
            arch_->ffn(step);
        }
        sum.sum_into(x, p.rows, plan_.residual);
        piece(l, true);
    }
    if (s + 1 == stages_.size() && plan_.drafter) draft_context(ctx, p, s);
    if (s + 1 < stages_.size()) {
        send(ctx, g, p.handoff, 0, p.rows, p.apart);
        p.at = g;
    } else if (p.want) {
        // Each member's vocabulary rows of every row that wants logits, each copied by its member into its slice of the rows in host memory the members import, behind no wait; the pass's logits wait for every member's submission.
        const size_t V = plan_.vocab;
        for (size_t m = 0, at = 0; m < W; ++m) {
            HeadStep head{part(ctx, g + m, pass_row(m), 0, 0, p.rows, all, p.apart), p.pick.data(), p.want,
                          backend::RowRuns{p.head_runs.data(), p.head_runs.size()}, {ctx.member_logits[g + m].get(), 0}};
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

// A member's partial rows cleared, `rows` of them, before a part whose routed experts add their weighted sum to them (blocks::routed_experts): a copy of zero rows the context keeps on the member.
inline void Model::clear_partial(ExecContext& ctx, size_t dev, backend::Slice partial, size_t rows) {
    devices_[dev]->b->copy(*partial.buffer, partial.offset * sizeof(float), *ctx.zeros[dev], 0, rows * plan_.residual * sizeof(float));
}

// The stage's submissions, every device it recorded on and each member of a tensor group, the pass's ticket that of `cur`.
inline void Model::stage_submit(ExecContext& ctx, Pass& p, size_t s, size_t cur) {
    p.tickets.resize(devices_.size(), 0);
    for (size_t d : stages_[s].touches)
        for (size_t m = d; m < d + width_; ++m) p.tickets[m] = ctx.tickets[m] = devices_[m]->b->submit();
    p.sent = p.tickets[cur];
    p.sent_members.assign(p.tickets.begin() + (std::ptrdiff_t)cur, p.tickets.begin() + (std::ptrdiff_t)(cur + width_));
}

// Each entry's history commits the stage, on the thread that owns the histories.
inline void Model::stage_commit(Pass& p, size_t s) {
    const Stage& st = stages_[s];
    const int storage = devices_[st.device]->storage_index;
    for (size_t e = 0; e < p.entries.size(); ++e) {
        Sequence& q = *p.entries[e].seq;
        if (s == token_stage_) {
            const backend::StateView& view = p.states[st.device][e];
            const size_t first = e ? p.runs[e - 1].end : 0;
            advance_tokens(token_tails_.data() + view.dst * token_width_, p.previous_tokens.data() + e * token_width_,
                           view.length, p.ids.data() + first, p.entries[e].n);
            if (q.mark_.held()) std::copy_n(p.ids.data() + first, p.entries[e].n, marked_tokens_.data() + q.mark_.hold.buffer() * options_.mark_rows);
        }
        if (storage >= 0) q.kv_[(size_t)storage].commit();
        else q.length_[s] += p.entries[e].n;
        for (size_t d : st.touches)
            for (size_t m = d; m < d + width_; ++m) q.last_[m] = p.tickets[m];
        if (!state_layers_) continue;
        q.from_[s] = p.kept[e].held() ? p.kept[e].slot() : Sequence::kLive;
        // Once every stage holds it, the checkpoint is the sequence's, and the one it replaces goes: a later write of that slot is enqueued after every read of it on each device's stream.
        if (s + 1 == stages_.size() && p.kept[e].held()) q.kept_ = std::move(p.kept[e]);
    }
}

// An embedded drafter's context rows of the pass, on the head's device after the last stage (Architecture::draft_rows): each entry's first row reads the row its sequence carries, or a zero row for an empty history.
// Then each entry's last normed row is carried into the slot its state is written to, a marked entry's rows are saved for its retract (save_h), and on a tensor group every member of the head's group does so over its own KV heads.
inline void Model::draft_context(ExecContext& ctx, Pass& p, size_t s) {
    const size_t E = plan_.residual, n = p.entries.size();
    const backend::RowRuns all{p.runs.data(), p.runs.size()};
    ctx.carry.resize(n);
    for (size_t m = 0; m < width_; ++m) {
        const size_t o = (size_t)place_.output_device + m;
        Device& d = *devices_[o];
        for (size_t e = 0; e < n; ++e) {
            const Sequence& q = *p.entries[e].seq;
            const size_t src = q.from_[s] == Sequence::kLive ? q.state_.slot() : q.from_[s];
            ctx.carry[e] = q.stage_length(s) ? backend::CSlice{d.carry.get(), src * E} : backend::CSlice{d.zero.get(), 0};
        }
        DraftRowsStep step{part(ctx, o, drafter_row(m), plan_.drafter->kind, 0, p.rows, all, p.apart), p.ids.data(), ctx.carry.data()};
        step.width = width_;
        step.views = p.views[(size_t)d.storage_index * width_ + m].data();
        step.n_views = n;
        step.kv_layer = drafter_kv_;
        step.pos = p.pos.data();
        arch_->draft_rows(step);
        const backend::Slice hn = slot(ctx, o, plan_.draft_h, p.apart);
        for (size_t e = 0; e < n; ++e) {
            const Sequence& q = *p.entries[e].seq;
            const size_t dst = p.kept[e].held() ? p.kept[e].slot() : q.state_.slot();
            d.b->copy(*d.carry, dst * E * sizeof(float), *hn.buffer, (hn.offset + (p.runs[e].end - 1) * E) * sizeof(float), E * sizeof(float));
        }
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

// A failed pass: every device drained, every entry's histories back to where the pass found them (rewind, to the checkpoint on a model that keeps a state), and the blocks and checkpoint slots its stages took returned.
inline void Model::roll_back(Pass& p) noexcept {
    retire();
    for (size_t e = 0; e < p.entries.size(); ++e) rewind(*p.entries[e].seq, p.start[e]);
    p.kept.clear();
}

// One backend allocation holds the activation slots of a pass, each aligned to 64 bytes, since device allocators handle a few large blocks far better than many small ones.
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

// Storage for a pass of `rows` rows with `want` logits rows, on every device the placement uses, with `buffers` handoff buffers on each device a crossing leaves (handoffs): grown when a pass needs more rows, never shrunk.
// Each is allocated whole before it replaces what the context had.
inline void Model::ensure(ExecContext& ctx, size_t rows, size_t want, size_t buffers, size_t apart_rows) {
    auto mul = [](size_t a, size_t b) {
        if (b && a > (size_t)-1 / b)
            throw std::runtime_error("inference: activation arena size overflows");
        return a * b;
    };
    ctx.scratch.resize(devices_.size());
    ctx.tickets.resize(devices_.size(), 0);
    // The run list a part rebuilds and a streamed layer's groups hold at most a run per entry, and so per row, so no part grows them.
    ctx.part_runs.reserve(rows);
    const auto lookups = [&](size_t device, size_t n) {
        std::vector<ExecContext::Scratch::LookupScratch> result;
        for (size_t l = 0; l < plan_.layers.size(); ++l) {
            const auto& lookup = plan_.layers[l].lookup;
            if (!lookup || size_t(place_.mixer_device[l]) != device - device % width_) continue;
            const auto& source = home_[l][lookup->role].row_source;
            if (!source) continue;
            const size_t count = mul(n, lookup->rows);
            if (count > UINT32_MAX) throw std::length_error("inference: lookup staging exceeds row ID range");
            if (result.empty()) result.resize(plan_.layers.size());
            auto& target = result[l];
            if (device % width_ == 0) {
                target.host = std::make_unique<WeightRows::Scratch>(*source, count);
                target.ids.resize(count);
            }
            target.data = devices_[device]->b->alloc(mul(count, source->row_bytes()));
        }
        return result;
    };
    for (size_t d = 0; d < devices_.size(); ++d) {
        ExecContext::Scratch& sc = ctx.scratch[d];
        if (!devices_[d]->used || (sc.arena && sc.rows >= rows)) continue;
        std::vector<size_t> counts(plan_.slots.size()), offsets;
        for (size_t i = 0; i < counts.size(); ++i) counts[i] = mul(rows, plan_.slots[i]);
        backend::BufferPtr arena = alloc_arena(*devices_[d]->b, counts, offsets);
        std::vector<uint32_t> indices(mul(rows, plan_.index_width));
        auto prepared = lookups(d, rows);
        // A pass through this context may still run on the arena being replaced: nothing else waits for a pass that wanted no logits.
        if (sc.arena) devices_[d]->b->wait(ctx.tickets[d]);
        sc.arena = std::move(arena);
        sc.offset = std::move(offsets);
        sc.indices = std::move(indices);
        sc.lookups = std::move(prepared);
        sc.rows = rows;
        sc.entry_runs.reserve(rows);
    }
    ctx.apart.resize(devices_.size());
    for (size_t d = 0; apart_rows && d < devices_.size(); ++d) {
        ExecContext::Scratch& sc = ctx.apart[d];
        if (!devices_[d]->used) continue;
        std::vector<size_t> counts(plan_.slots.size());
        for (size_t i = 0; i < counts.size(); ++i) counts[i] = mul(apart_rows, plan_.slots[i]);
        std::vector<uint32_t> indices(mul(apart_rows, plan_.index_width));
        auto prepared = lookups(d, apart_rows);
        sc.arena = alloc_arena(*devices_[d]->b, counts, sc.offset);
        sc.indices = std::move(indices);
        sc.lookups = std::move(prepared);
        sc.rows = apart_rows;
        sc.entry_runs.reserve(apart_rows);
    }
    ctx.apart_rows = apart_rows;
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
    ctx.zeros.resize(devices_.size());
    const bool clears = std::any_of(plan_.layers.begin(), plan_.layers.end(), has_experts) || (plan_.drafter && has_experts(*plan_.drafter));
    for (size_t g = 0; g < devices_.size(); g += width_) {
        if (!devices_[g]->used || (ctx.collectives[g] && ctx.collective_rows >= rows)) continue;
        std::vector<backend::Backend*> members;
        for (size_t m = g; m < g + width_; ++m) {
            members.push_back(devices_[m]->b.get());
            devices_[m]->b->wait(ctx.tickets[m]);
        }
        ctx.collectives[g] = devices_[g]->b->join(members, rows, plan_.residual);
        if (!ctx.collectives[g]) throw std::runtime_error("inference: the backends of a tensor group have no cross-device sum");
        // An allocation is zero-filled, and nothing writes these rows.
        for (size_t m = g; clears && m < g + width_; ++m) ctx.zeros[m] = devices_[m]->b->alloc(mul(mul(rows, plan_.residual), sizeof(float)));
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
inline void Model::send(ExecContext& ctx, size_t from, size_t handoff, size_t base, size_t rows, bool apart) {
    const size_t E = plan_.residual;
    const backend::Slice x = slot(ctx, from, 0, apart);
    devices_[from]->b->copy(*ctx.handoff[from][handoff], base * E * sizeof(float), *x.buffer,
                            (x.offset + base * E) * sizeof(float), rows * E * sizeof(float));
}

// `receive` waits for that submission and writes the rows into the destination's residual, enqueued there.
inline void Model::receive(ExecContext& ctx, size_t from, size_t handoff, backend::Ticket sent, size_t to, size_t base, size_t rows, bool apart) {
    const size_t E = plan_.residual;
    devices_[from]->b->wait(sent);
    const backend::Slice x = slot(ctx, to, 0, apart);
    const uint8_t* rows_out = (const uint8_t*)ctx.handoff[from][handoff]->host_ptr() + base * E * sizeof(float);
    devices_[to]->b->write(*x.buffer, (x.offset + base * E) * sizeof(float), rows_out, rows * E * sizeof(float));
}

// A crossing inside a stage, both halves at once.
inline void Model::cross(ExecContext& ctx, size_t from, size_t to, size_t base, size_t rows) {
    send(ctx, from, 0, base, rows);
    receive(ctx, from, 0, devices_[from]->b->submit(), to, base, rows);
}

// A streamed layer in a pass with long runs: consecutive alike entries form a group, the long ones run on the layer's streamed row with each window role's bytes written into its window once, the rest on the host through a crossing each way.
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
inline Step Model::part(ExecContext& ctx, size_t dev, const Weight* w, uint8_t kind, size_t base, size_t rows, backend::RowRuns runs, bool apart) const {
    ExecContext::Scratch& sc = apart ? ctx.apart[dev] : ctx.scratch[dev];
    const Device& d = *devices_[dev];
    Step step{*d.b, sc.arena.get(), sc.offset.data(), {sc.arena.get(), sc.offset[0] / sizeof(float) + base * plan_.residual},
                rows, runs, w, kind, nullptr, 0, 0, nullptr, d.tables.data(), &sc.entry_runs, nullptr, 0, options_.device_dtypes.empty() ? options_.dtype : options_.device_dtypes[dev]};
    step.indices = sc.indices.data();
    step.indices_size = sc.indices.size();
    return step;
}

// Layer l's mixer over every row of the pass, with the cache views of its device's storage when the layer keeps KV, and the rows' positions.
inline Step Model::mixer_part(ExecContext& ctx, const Pass& p, size_t dev, int l) const {
    const Device& d = *devices_[dev];
    const LayerPlan& layer = plan_.layers[(size_t)l];
    Step s = part(ctx, dev, home_row(d.member, (size_t)l), layer.kind, 0, p.rows, {p.runs.data(), p.runs.size()}, p.apart);
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
    if (p.image_rows) s.streams = backend::RopeStreams{3, plan_.image_streams.data()};
    if (layer.lookup && home_[(size_t)l][layer.lookup->role].row_source) {
        const ExecContext::Scratch& sc = p.apart ? ctx.apart[dev] : ctx.scratch[dev];
        s.lookup = &sc.lookups[size_t(l)].view;
    }
    if (layer.token_history) {
        s.tokens = p.ids.data();
        s.previous_tokens = p.previous_tokens.data();
        s.previous_count = token_width_;
    }
    return s;
}

// Each image's rows the pass reads written over their pad rows of the residual on embedding device `dev`, right after the embedding.
// An image's rows are uploaded there once, then copied for each entry's part of each image (docs/VISION.md, section 4).
inline void Model::write_media(ExecContext& ctx, const Pass& p, size_t dev) {
    backend::Backend& b = *devices_[dev]->b;
    const ExecContext::Scratch& sc = p.apart ? ctx.apart[dev] : ctx.scratch[dev];
    const size_t W = plan_.residual, x = sc.offset[0];
    for (size_t e = 0, r = 0; e < p.entries.size(); r += p.entries[e++].n) {
        Sequence& q = *p.entries[e].seq;
        const size_t len = p.start[e], n = p.entries[e].n;
        for (Sequence::Media& m : q.media_) {
            const size_t from = std::max(len, m.run.first), to = std::min(len + n, m.run.first + m.run.rows);
            if (from >= to) continue;
            if (!m.on[dev]) {
                m.on[dev] = b.alloc(m.run.encoded->size() * sizeof(float));
                b.write(*m.on[dev], 0, m.run.encoded->data(), m.run.encoded->size() * sizeof(float));
            }
            b.copy(*sc.arena, x + (r + from - len) * W * sizeof(float), *m.on[dev], (from - m.run.first) * W * sizeof(float), (to - from) * W * sizeof(float));
        }
    }
}

} // namespace infer
