# Sequence history, state checkpoints and speculative decoding (planned)

A design, approved by the user on 2026-09-30 as agreed with XDEV and not yet built: one owner for where a sequence's history can be re-entered, checkpoints of the recurrent state on top of it, and speculative decoding on top of both.
Its first user is prefix reuse for the hybrid models, the qwen35 plan's step 8c ([QWEN35](QWEN35.md), [STATUS](STATUS.md)), because production serves Qwen 3.5, 3.6 and 3.8 with none: every follow-up chat turn reads the whole conversation again.
The speculative decoding plan the user approved on 2026-09-26 is taken as input; section 3 says what it keeps and what changes, and section 4 what the user's mx-llama.cpp history teaches.
The rules every part keeps: output is byte-identical with every feature here on and off, greedy and seeded, alone and among other requests, on one device and on a layer split; kernels follow a row's class, never the batch; one owner per rule ([ARCHITECTURE](ARCHITECTURE.md), Each concern has one owner).

## 1. One owner of a sequence's history (planned)

**What a history is.**
A `Sequence` holds a committed length L and, in every storage its model keeps, what the rows [0, L) left there.
The storage kinds, each declared by a module's plan (`LayerPlan::cache`), and each shortened and shared its own way:

| kind | holds | exact re-entry points | who has it |
|---|---|---|---|
| paged KV | a row per position, in blocks | every length | qwen3, qwen3moe, qwen35's full-attention layers, the MTP layer (section 3) |
| window cache | paged KV kept only for the last W rows | every length within the window | DFlash and DSpark drafters |
| recurrent state | one fixed-size state a slot, the state after the last row read | 0, a checkpoint, or a row inside a mark | qwen35's linear-attention layers |
| ring and compressor tails | the last rows, kept by committed length with k_max headroom | every length within the headroom | DeepSeek V4.x (its own plan) |

**The owner** is the model runtime (`model/runtime.hpp`, its history operations in `model/history.hpp`, and `model/kv_cache.hpp`): `infer::Model` and `infer::Sequence`.
The server and the CLI ask for the operations below and never touch a block, a slot or a storage; the module declares its cache kinds and names no operation.

**The operations**, each one call on `Model`:

| operation | call | paged KV | recurrent state | cost |
|---|---|---|---|---|
| extend | a pass's commit (today) | rows appended | updated in the live slot | the pass |
| checkpoint at p | an entry ending at p with `BatchEntry::keep` | nothing: rows below L are never rewritten | the entry writes its state into a free slot (StateView `dst`), which becomes the checkpoint at p; the next pass reads it (`src`) and writes the live slot | no copy, no host wait |
| fork at p | `fork(src, p)` | whole blocks below p shared, as today | the fork's first pass reads src's checkpoint at p in place and writes the fork's own live slot | no copy |
| mark (from step 3) | `mark(seq)` | nothing | the live slot is kept as the mark at L; later passes write a fresh slot and save their recurrent inputs, up to k_max + 1 rows | no copy |
| retract to L | `retract(seq, L)`, returns the length reached | blocks past L returned | inside a mark's saved rows: the recurrence rerun from the mark over them into the live slot, enqueued before the next pass; else its checkpoint at or below L, else 0 | KV free; state: a rerun of at most k_max + 1 rows, or the caller's recompute |
| reset | `reset(seq)` (today) | every block returned | the live slot, the mark and every checkpoint returned | none |

- `retract` is the only call that shortens a history, for every caller: a rejected draft, a donor trimmed at park, a failed pass, a paused request.
  It returns the longest length at or below L that every storage holds exactly, and the caller recomputes the rest in the row classes that first computed it, as the exact resume already does ([SERVER](SERVER.md), An exact resume).
  On a model that keeps no state it always reaches L.
- `fork(src, p)` takes a p that is whole blocks in every storage and, on a model that keeps a state, a checkpoint of src; the scheduler picks p from `Sequence::checkpoints()`.
- A failed pass drains every device's pending work, then retracts each of its sequences to where it began: KV reaches that length, but the live state it wrote is gone, so a stateful sequence reaches its mark or its latest checkpoint, and 0 when it has neither, every storage made coherent at that one length.
  The pass reports the length reached, and the caller recomputes from there to its committed history without sampling, as a resume does; a call refused before it runs changes nothing.
  That replaces the rule that a failed pass leaves a state lost until a reset (the qwen35 plan's decision 6), since a checkpoint or a mark is never a slot a pass writes.
- A checkpoint's or a mark's slot is read-only once taken, counted by reference like a shared block (a fork reading it holds one), and returned only after the tickets of every pass that read it have retired, the rule blocks and state slots already keep.
- Refused, with nothing changed: a sequence in flight; a fork or a checkpoint at a position that is not whole blocks in every storage; a fork of a stateful sequence where it has no checkpoint; a second mark; `keep` in a verify entry.

**Exactness.**
Within a row class, a recurrent row's output and the state after it depend only on the state before it and the row's inputs ([QWEN35](QWEN35.md), Row classes), and a checkpoint position is whole blocks in every storage (the model refuses block sizes that do not nest, so today a multiple of the largest) and a multiple of the chunk grid's 64, so it lies on the grid.
So a history continued from a checkpoint, a fork or a rerun from a mark gives the bits of the history never stopped, provided its rows [0, p) were computed in the classes the new request would compute them in.
**Row classes have one owner:** `Model::row_class(extent)`, the tuple, over every op and stage the rows pass through, of the arithmetic each dispatch owner chooses for that extent with the loaded model's types, widths and effective per-device types: the matmul and attention kernels by `RowRun` extent, the recurrence's form, and the streamed path by `Placement::stream_from`.
Two classes are equal only where every element is proved the same arithmetic; a backend name or a matrix-path label is not enough, since two paths can both report f32 and sum in another order, and anything not proved equal is a separate class.
A fork, a take-back and a restore take only the longest prefix whose recorded classes all equal the new request's, since a changed row changes every state after it; two prompts past every threshold share rows, and a prompt's rows and a reply's decode rows share none unless their classes are proved equal (decision 4).

**On a layer split** each device's `StateStorage` holds its own layers' part of every slot, at the same slot index on every device, and each stage's KV storage its own blocks.
A checkpoint named in a pass is valid once the pass's last stage has run (`end_pass`), and `abort_pass` returns its slot; once phase 4 gives a sequence per-storage progress, it is valid once every stage has passed p.
Nothing crosses between devices: every stage writes and reads only its own part.

**Slots.** One slot pool per model, each slot live, a checkpoint or a mark by use.
`ModelOptions::state_slots` live slots and `ModelOptions::checkpoint_slots` more, fitted at load and never grown, are counted apart, so admission never waits on a live slot: when a live slot becomes a checkpoint (`Model::keep`), it moves to the checkpoint side only where that side has room, and the live side has one more to give.
A sequence holds one checkpoint at most, a newer one replacing it, which is all prefix reuse needs; a mark (step 3) is a slot of its own.

## 2. Checkpoint storage and policy (planned)

**Where.** On the device, as more slots of each state storage beside the live ones, so a restore or a fork reads a checkpoint in place.
A host tier behind it keeps what the device slots cannot (Host tier, below, step 2b): a 27B checkpoint crosses in 13 to 31 ms at 5 to 12 GB/s, against about 4 s of recompute per 1000 tokens.

**Size.** A checkpoint is one sequence's state, F32, plus the KV blocks of [0, p) it pins, which the ledger already counts as the donor's blocks:

| model | state per checkpoint | KV per token (f16) |
|---|---|---|
| Qwen3.5-0.8B, 2B | 19.3 MiB | 12 KiB |
| Qwen3.5-4B, 9B | 50.3 MiB | 32 KiB |
| Qwen3.6-27B, Qwen3.8-27B | 149.6 MiB | 64 KiB, plus 4 KiB with MTP |
| Qwen3.6-35B-A3B | 62.8 MiB | 20 KiB |
| Qwen3.5-122B-A10B | 149.1 MiB | 24 KiB |

**How many.** `serve --state-checkpoints N` sets the checkpoint slots; by default the fit gives the fewer of `--max-seqs` and what fits in a quarter of the room left after the weights, the activations and the live slots, for example about 4 for the 27B Q8_0 at 16 live slots on one MI50; 0 turns checkpoints off, and an explicit N the devices cannot hold is refused before anything is allocated.
`chat` keeps one (two slots); `generate`, `logits`, `perplexity` and `bench` keep none.
**The fit (8a)** counts the live and checkpoint slots on every stage that holds state layers before the KV budget, so the budget is what they leave, and the startup line gives live slots, checkpoint slots, KV tokens and the KV tokens the checkpoints cost; a drafter's marks, saved rows and their copies per pass slot come after the budget and never lower it (section 3).

**Where a checkpoint goes.** Batch assembly is the one owner, one checkpoint a request at first:
- c = the largest position that is whole blocks in every storage at or below min(S, E - 1), E the prompt's length and S its stable prefix: the tokens of the conversation rendered without the generation prompt (`chat::ChatFormat::render` with `add_generation_prompt` false, taken only as far as those ids really prefix the prompt's ids after rendering and tokenization) for a chat request, the whole prompt for a text.
- On the Qwen 3.5 and 3.6 templates the next turn drops the reply's reasoning and diverges right after the assistant header, so its prompt starts with S; on the 3.8 templates with `reasoning_content` passed back it keeps the whole reply, still after S, and the reply's rows are decode rows, which the new prompt's class does not share anyway.
- No checkpoint when c is 0 or at or below the fork the request started from.
- The slice that reaches c ends at c and carries `keep`, so a request pays at most one pass boundary for it; the prompt goes on in the next pass.
- Message-boundary checkpoints come with the host tier (below); every-N checkpoints stay a later option, kept only on a measured gain.
- The reference server cuts a prompt 4 + ubatch and 4 tokens before its end and keeps up to 32 host copies a slot, 256 tokens apart (section 4, lesson 23); S is where the next turn actually diverges, whatever the template, and eviction goes by donor age, never by spacing (lesson 17).

**Prefix reuse for a model that keeps a state.**
A donor is its KV blocks of [0, c), its checkpoint at c, its tokens and the row classes of [0, c).
At finish such a request retracts to its latest checkpoint, returning its live slot and the rows past c, and one with no checkpoint is released as today.
`best_donor` takes, for each donor, its largest checkpoint at or below the matched tokens (whole blocks, never the last token) over rows whose classes are equal, and the request forks there and computes the rest.
A paused request takes its own donor back whole where it survives: its live slot, at its full length, becomes that donor's checkpoint without a copy when the checkpoint side has a free slot; otherwise it keeps its prompt checkpoint and the resume recomputes from there, not from 0.

**Eviction.** `make_room` stays the one owner, over KV blocks and checkpoint slots together, the slots as one more pool whose unit is a slot.
A request's checkpoint is optional: it takes a free slot, else the oldest donor goes with its blocks, else it is skipped; nothing is paused for one, and an active request's checkpoint is never taken.
On a model that keeps a state, a donor without a checkpoint cannot be continued, whatever KV blocks it holds, so a donor whose checkpoint is taken is dropped with its blocks.
`/v1/health` adds checkpoints held and the forks that read one.

**Host tier (step 2b, agreed with the other developer and approved by the user 2026-10-01).**
The device keeps each conversation's latest checkpoint, forked in place; host memory keeps what the device cannot, so an edited earlier message and more conversations than the device slots hold are resumed rather than recomputed.
- **Write-back:** a finished donor's checkpoint and KV blocks are copied to host memory after its last pass, and evicting the donor from the device then drops the device copy only.
  The source blocks and slot stay held until the copy's ticket retires, by the retirement rule blocks and slots keep today; a separate transfer queue or thread comes only if measured to overlap.
- **Promotion:** `best_donor` searches both tiers by tokens and row classes; a host hit uploads the state and the blocks the request needs into free device slots and blocks before its first pass (27B: 149.6 MiB of state and 64 KiB a token, about 70 ms for an 8k conversation at 10 GB/s, against about 32 s of recompute).
- **Message-boundary checkpoints:** a state-only checkpoint, held on the host, at each user message's start where that start is a token prefix of the final rendered prompt, checked by tokens and row classes rather than assumed, since rendering a prefix can rewrite earlier tokens, and only at whole-block positions; its KV is a prefix of the donor's own blocks and is never stored twice.
  Each is written through a staging ring of two device slots; an edited message resumes at the checkpoint before it.
- **Owners:** the scheduler decides which donor is kept, written back, promoted or dropped, through `make_room` over device blocks, device slots and host bytes, by donor age in every tier; `Model` and the backends' storages own the snapshot, copy and restore, so the scheduler never sees a KV or state layout.
- **Bounds:** `--host-cache-bytes` caps the tier, its default a fraction of `core::host_memory_available` at start; the cap counts pending copies and staging, admission to the tier is checked against the host's current headroom, and old entries are evicted before a new one waits.
  A copy that fails or is cancelled leaves a valid device donor or no entry, never a partly written host entry that a request could hit.
- **Every model:** for qwen3 and qwen3moe the tier is KV prefix offload, the same entries without a state.
- **Owner identity:** the tier belongs to one loaded `Model`, and an entry is restored only through it, which fixes the weights, the architecture, the placement, each device's resolved dtype, the cache types and the KV and state layouts; row classes do not carry the dtype, so no entry is ever matched across models, and reloading or replacing the model drops every entry.
  Copies keep the storages' bytes as they are, never converted to an activation dtype.
- **Publication:** a written-back entry can be hit only once every stage's copy has retired; a promotion that fails releases what it took on the devices and leaves the host entry valid.
- **Exactness:** a copy keeps the bytes and the row classes, so a resumed or promoted history gives the CLI's bits.
  Tests: a host round trip of KV and state on every placement, mixed ones included; promotion beside a pass in flight; eviction and cancellation while a copy is pending; injected host allocation and transfer failures, at write-back and at promotion; and a model of another dtype or placement on the same backends refused every entry.
- **Two parts:** (a) whole-donor write-back and promotion, for every model; (b) the message-boundary checkpoints with their token-prefix checks and their staging slots, which (a) does not allocate, after (a) has been measured.

**Idle re-prefill (step 2c, agreed with the other developer and approved by the user 2026-10-01).**
A reply's rows are decode rows, which a follow-up's prompt class does not share, so a follow-up that keeps the reply reads it again (about 140 tokens a turn in step 2's comparison, reasoning off).
- When a request finishes and the next turn's render keeps its reply, the scheduler queues an idle job: fork the donor at its checkpoint, feed the reply and the closing tokens the next render adds as prompt rows, and keep a checkpoint at the new stable prefix, replacing the donor's once the job completes; the old donor stays until then.
- A second trigger, while the reply is generated (agreed with the other developer 2026-10-01, at the user's request for exact reuse without idle time): chunks of the reply's accepted tokens, never drafts, are fed the same way on the shadow fork in passes with room under the round's existing budget, so under load or with clients that answer at once most of the reply is read by its end; only a token prefix the next turn's render is known to keep (reasoning removed, closing tokens rewritten) is fed, and where that prefix cannot be established the idle trigger alone acts.
  The shadow's state slot, its KV blocks for the reply span and its copies in flight are charged to the admission and budget owner before a chunk is scheduled, a chunk is bounded and cancelled at a pass boundary, and spare rows are not taken as spare latency: the per-token and tail latency of every active request is measured with the chunk size.
- Only where `Model::row_class` is one class over every extent the next prompt can have, from the history's length up, the streamed transition (`Placement::stream_from`) included; elsewhere the job is not queued.
- Idle only: it takes passes no request wants and is cancelled at the next pass boundary when a request needs its slot or rows; a submitted pass is not preempted, so a new request's wait on it is measured and bounded; a cancellation or an error leaves the old donor and no leaked slot.
- Owners: the scheduler decides and cancels; `Model` forks, extends and keeps as today; no new storage kind.

## 3. Speculative decoding on top (planned)

**Kept from the 2026-09-26 plan:**
- Every drafter is a proposer; the verify, the acceptance rule, the rollback, the scheduler integration, the loader path and the flags are shared.
- **Verify:** a request with k drafts is one entry [last pick, d1 ... dk], k + 1 rows, `want_logits` and `every_logits`, extent 1, so every row takes the decode kernels; k = 0 is today's decode entry.
- **Acceptance** (`infer::accept`, decided with the qwen35 plan's question 11): row i is sampled with the request's own sampler and generator exactly as without drafts, and draft d(i+1) is kept only if it equals that pick; so ids, logprobs, the end cause and the number of draws equal the run without drafts, greedy and seeded.
- **Proposers draft deterministically**, argmax with ties to the lowest id, and never touch a request's generator.
- **A drafter never changes the target's history**: its forks, donors, KV budget and `token_limit` are the same with it loaded; it fits in the room the no-drafter fit leaves, or is refused at load with the numbers.
- **MTP rows indexed by the token they read:** the MTP layer is one more attention layer in the target's KV storage, with the target's length, blocks, forks and retract, and the row reading h(i-1) and t(i) at index i.
- **Flags:** `--drafter off|embedded|lookup|PATH` and `--draft-max N` on `generate`, `chat`, `serve` and `bench --model`, off by default, honoured on every backend or refused; no request field.
- **Drafts never stall, pause or evict anything**; a request drafts only where `spec::draft_length` (one owner, over the pass cost model) finds a gain.
- **Not doing:** the ratio test, relaxed or typical acceptance, trees, a drafter lowering the KV budget, cross-request n-gram pools, drafting while the request's verify is in flight, speculation in `logits` or `perplexity`.

**The proposer interface** (`inference/spec.hpp`):

```
class Proposer {
    virtual Reads reads() const = 0;             // nothing, the committed tokens, the target's final rows, or rows entering listed layers
    virtual size_t block() const = 0;            // the most drafts one call gives
    virtual void draft(const Req& r, size_t k, std::vector<uint32_t>& out) = 0;
    virtual void settle(const Req& r, size_t kept) = 0;   // its own history retracted to the kept tokens
};
```

- **Lookup:** the request's own tokens, on the host, no storage.
- **Draft model:** a second `Model`; its sequence goes through that model's own mark and retract, so a hybrid draft model (Qwen3.5-0.8B for the 27B) rolls back as the target does.
- **In-model drafters** (MTP, DFlash, DSpark): a model-layer `Drafter` computes its context rows inside the target's pass into storages of the same `Sequence` (the MTP layer, window caches), so the target's retract covers them; its draft pass returns ids only.
- **A new kind** (an EAGLE-style head, a DeepSeek MTP) is one proposer class and, if it reads target rows, one `Drafter`; nothing else changes.

**One round**, the same in `generate` and in the scheduler:
1. `mark(seq)` at L0, where the model keeps a state (on a KV-only model it is a no-op).
2. The proposer drafts d1 ... dk; the verify entry runs.
3. `infer::accept` samples row i (the history after x0, the last pick, and d1 ... di) for i = 0, 1, ... in order, one draw a sampled row as without drafts, emits each pick y_i, and goes on only while the request has not ended and y_i = d(i+1); no row after the one that ended the request or missed its draft is sampled.
4. `retract(seq, n)`, n being the history the run without drafts would hold at that point: L0 + 1 for x0, plus one for each draft kept and fed, a draft that is an end token or completes a stop text never counted, and the final pick fed or not by the caller's own rule, as today; KV blocks past n are returned and the state rerun from the mark over the kept saved rows, or, when every fed row was kept, the mark's slot returned and nothing rerun.
5. `settle` on the proposer.

The saved inputs cover the conv's raw rows and the recurrence's inputs of every linear-attention layer on every stage, so the rerun restores both parts of the state; it reads only the kept mark and the saved inputs and writes only the live slot, so it is exact, as the per-token recurrence gives the same bits for the same state and inputs, and it can be retried.

**Rules of the round,** each from a failure the reference fork met (section 4):
- `retract` takes effect when it is called, in the ledger, the slot roles and the KV lengths; a second retract composes with it, and nothing is armed for a later pass (lesson 5).
- The kept rows end at the first end token or stop text among them, and retract takes that count, never the draft's; tests put an end token and a stop text at the first draft, at each later one and at the pick after them (lesson 8).
- Rows recomputed after a retract are history: they are committed, never sampled or verified again (lesson 7).
- A proposer's state for a request (the lookup index, the MTP block's last h) is keyed by the sequence and follows its reset, fork and retract (lesson 14).
- A draft whose best logit or whose inputs are not finite ends drafting for that request until its reset, and acceptance is counted per request and gated, since a wrong drafter shows only as low acceptance (lesson 13).
- A request drafting nothing runs today's decode entry, with no mark and no saved rows (lesson 11).

**The scheduler:** retract runs right after a pass's logits are read, in the same round, before `park`, `pause` or any fork, so no rule is needed for a pending restore.
A mark takes a slot only from free ones; a request without one drafts nothing that round.
Draft rows count in the pass's ubatch budget; the logits rows become 2 x (max_seqs + 64), draft rows in flight capped at 64; `Request::kRowsWaiting` becomes at least k_max + 1; a draft pass runs per drafter device, in that device's formation order, and is retired like a stage.
A request is in at most one pass at a time, so with passes in flight its next draft waits for its verify.

**What changed from the 2026-09-26 plan, and why:**
- **The verify-slot pool and the qwen35 restore become the mark,** from the one slot pool: one owner and one fit count for live slots, checkpoints and marks, instead of a pool per purpose.
- **`Model::retract` returns the length it reached,** so one call serves a rejected draft, a donor trimmed to its checkpoint, a failed pass and a pause, and a caller recomputes the rest by the exact resume's rule.
- **A fork of a stateful sequence reads the checkpoint in place,** where the plan copied one; a checkpoint is written by the pass that reaches it, with no copy and no kernel change.
- **A failed pass goes back to the latest checkpoint or mark** instead of always losing the state.
- **Order:** 8c first, as production needs it and it builds the owner; the verify then covers qwen35 from its first step, since retract already restores a state; MTP comes before draft models and sidecars, as the gain the user asked for next; the reference measurement (the old step 0) runs beside steps 1 to 4 and gates only their speed criteria.

## 4. Lessons from mx-llama.cpp

The user's gfx906 fork of llama.cpp built MTP, draft models, DFlash, DSpark, recurrent rollback and context checkpoints, and measured them on the same cards.
Read, not copied: each lesson below is taken as a rule or a gate, and llmx implements its own design.
Sources are commits of that fork, its notes (`tp-notes/research/`, at 2993afb7d1) and its development log for Qwen3.8-Flash-Next MTP.

| # | lesson | taken as | source |
|---|---|---|---|
| 1 | Verify rows computed by kernels chosen by the verify's width changed greedy output; verifying drafts that were all rejected reproduced it | verify rows are extent 1; synthetic all-rejected proposers are a standing test | f49ed86916; `qwen36-mtp-tensor-tps2.md`; 35B-A3B at 4 drafts in `mtp-deferred-prefill-default.md` |
| 2 | Restoring inside a multi-token verify changed the trajectory where the verify ran the recurrence otherwise than decode | the rerun and the verify run the decode's per-token recurrence | b4c3674d38 |
| 3 | Rollback tests that restored to the prefill boundary passed broken code; a generation-cycle test (decode, roll back from the new tail, replay, compare, repeat) failed from its first cycle | generation cycles, several sequences at once and the serving gate in steps 2 and 3 | development log, "bounded recurrent rollback - SOLVED" (2026-08-30); 93697608e |
| 4 | Every writer of recurrent state must share the reader's slot mapping, and every place a sequence is made, copied, kept or dropped must carry it; converting a subset desynchronised silently | slot roles live in `Sequence` and change only in `Model`'s history calls; kernels touch only their `StateView` slots | the same entry; 340504803, 968ca8e84 |
| 5 | A rollback armed for a later pass was overwritten by a second rewind, and output degraded later (a 728-character run of spaces) | `retract` takes effect at once | 432efe7705, 39a560407b |
| 6 | A failed decode left the snapshot ring advanced past unwritten state | slot roles commit at `end_pass`, and `abort_pass` restores them | ca51c15651, e5c09d548d |
| 7 | Replayed tokens after a checkpoint restore were verified again; batch-dependent logits on Vulkan rejected them and the request looped on one position | recomputed rows are never sampled again | 4b1f6f811c, a4770544b5 |
| 8 | Accepted drafts past an end token entered rollback and the statistics | kept rows end at the first end token or stop text | 31d304b92f |
| 9 | Speculative checkpoints serialized to the host took about 600 ms of an 825 ms round (6.2 t/s against 32.4 without drafts); on the device, 41.5 t/s | checkpoints and marks are device slots | 175b66c51f, 82bacc5475 |
| 10 | Per-token snapshots beat restoring and decoding again: Flash-Next MTP 29.8 to 42.4 t/s on a layer split | a rejected draft never runs the model again; decision 5 weighs rerun against snapshots | 7c5afc123d, 2ca450b71c |
| 11 | Rollback support slowed plain decode with no drafts (37.7 to 33.7 t/s with 4 snapshot rows forced) until single-token steps took the normal path | no mark without drafts; the k = 0 gate | `qwen36-mtp-adaptive-disable.md`, `qwen36-mtp-recurrent-single-token-fastpath.md` |
| 12 | Mirroring prompt rows to the MTP context through the host took 47.4 s of a 100 KB prompt; deferring them gained 15 to 28 percent of prefill | MTP context rows computed inside the target's pass | `mtp-runtime-recommendations.md`, `qwen36-mtp-process-timing.md`, `mtp-deferred-prefill-default.md` |
| 13 | Hidden rows read from the wrong device were garbage (about -2.7e36) or NaN on long prompts, and acceptance collapsed with no error | drafter inputs are the pass's own rows; a non-finite draft stops drafting; acceptance gated | `qwen36-mtp-tensor-split-mirrored-read.md`, `qwen36-mtp-adaptive-disable.md` |
| 14 | The MTP drafter had no reset, so a request's first drafts continued the previous request's hidden state | drafter state is the sequence's, reset, forked and retracted with it | development log, 2026-08-30 06:45; fc6b4a2ae2 |
| 15 | A drafter's checkpoint and the target's ended at different positions on reuse | in-model drafter storages are the same `Sequence`; a draft model's sequence forks or restarts beside it | 79a4eb0567 |
| 16 | A final-prompt snapshot made an identical prompt fast (12.6 to 2.0 s), but a divergent prompt then missed its uncached control | reuse only at a checkpoint over class-equal rows | `docs/development/staged-sequence-major-research.md` at d0b080eadc |
| 17 | Spacing eviction dropped the near-end checkpoint the next request resumed from | one checkpoint at S; eviction by donor age | 89722330a4 |
| 18 | Context checkpoints of DeepSeek V4's cache crashed a tensor split; with them off, drafting served | every storage kind's checkpoint, fork and retract tested on each placement it runs on | development log, 2026-09-14 23:22 |
| 19 | On DeepSeek V4.1 each verify row cost about 20 ms and acceptance fell 0.55, 0.27, 0.14 by position; one draft broke even | `draft_length` prices each row by the pass cost model and drafts 0 where it does not pay | development log, 2026-09-13 (DSpark n1 to n3) |
| 20 | Qwen3.6-27B-MTP Q8_0: 0.83 of drafts kept at 2 (+62 percent) and 0.70 at 4 (+72) on a layer split, +9 on a tensor split; 6 drafts slower, 8 changed output; 35B-A3B flat to +1.4; Q4_1 not exact | step 4 on the dense 27B first; the default depth from measurement | `qwen36-27b-mtp-path-matrix.md`, `qwen36-mtp-draft-depth-sweep.md`, `qwen36-a3b-mtp.md` |
| 21 | Adaptive depth was neutral with a draft model and cut 35B-A3B MTP from 57.3 to 32.2 t/s; disabling after one miss was too eager | one acceptance average a request, no fast adaptation | `adaptive-spec-depth.md` (6183cea399), `mtp-runtime-recommendations.md` |
| 22 | A graph whose shape followed the number of snapshots cost prefill 2.5 times (421 against 1047 t/s) | a checkpoint changes no kernel or dispatch shape, only a `StateView` slot | 2e740434ea |
| 23 | The reference server checkpoints 4 + ubatch and 4 tokens before a prompt's end, host-resident | S from the template, device slots | `tools/server/server-context.cpp` (`checkpoint_offsets`) |
| 24 | The MTP block's carried h started as stale bytes, so repeated requests drafted differently (228/140 against 226/141 drafted/kept) until it was zeroed at position 0; a zero h at any later position cut acceptance from 0.518 to 0.328 | the carried row lives in the slot pool and follows every history call; only a sequence's first row reads a zero h (section 7) | 017a5d3f8f, a186706304, 7393b6c88e |

## 5. Order of work (planned)

Each step is a branch off main, landed as at most two commits, with every command byte-identical to main where the step's feature is off, and `llmx-split-check` bit-identical on the splits it runs.

| # | branch | brings | gate |
|---|---|---|---|
| 1 | `fix/server-row-class` | `Model::row_class`; `best_donor`, first admissions and take-backs compare classes by it, for every model (SERVER, Open gaps, closed) | `server-resume` and `server-passes-cpu` unchanged; a follow-up turn's ids equal `generate` on its full prompt, on Qwen3-0.6B and 8B Q8_0, CPU, one MI50 and a split; `tools/server_mix_check.py` |
| 2 | `feat/qwen35-checkpoints` (8c) | section 1's checkpoint, fork, retract to a checkpoint and failed-pass rules over KV and states (no mark yet), the one slot pool, `--state-checkpoints` and the fit, section 2's policy in the scheduler, `chat`'s checkpoint | `arch-qwen35`: a fork at a checkpoint equals a fresh sequence fed the same tokens bit for bit, alone and beside others, and over four CPU stages; retract and a failed pass back to a checkpoint; generation cycles that decode, retract from the new tail and continue, equal to never retracting, with four sequences at once (lesson 3); `server-resume`, `server-passes` and `server-passes-cpu` with hybrid donors; the `qwen35` component: a six-turn conversation, every turn's ids equal to `generate` on its full prompt, reused tokens growing each turn; `tools/server_mix_check.py` on Qwen3.5-0.8B, 9B and Qwen3.6-27B Q8_0 on one MI50 and the Radeon VII (0.8B and 9B); the use: time to first token of each turn of a 27B conversation reaching 8k tokens, against main, and the KV budget each fits |
| 2b | `feat/host-cache`, in two parts | section 2's host tier: (a) write-back, promotion, `--host-cache-bytes`, its owners and bounds; (b) message-boundary checkpoints | the exactness tests of the host tier above; per-turn time to first token, prompt tokens re-read, host bytes held and bytes transferred, and the inter-token latency of unrelated active requests, against main and the reference server with its checkpoints on, on Qwen3.8-27B Q8_0: a conversation reaching about 8k tokens, the same with an edited earlier message, and more conversations than the device tier holds |
| 2c | `feat/idle-reprefill` | section 2's idle re-prefill, both triggers | a follow-up continuing from re-prefilled rows equal to `generate` on its full prompt, bit for bit, on one device, a split and a placement with streamed experts; the job cancelled at every pass boundary; follow-up time to first token and tokens read with reasoning off against main and the reference server, and unrelated requests' inter-token latency while jobs run |
| 3 | `feat/spec-verify` | multi-row `step`, `mark` and the saved-row rerun, `infer::accept`, `spec::Proposer`, lookup, `spec::draft_length`, the round in `generate` and `chat`, `--drafter off\|lookup`, `--draft-max`, test-only synthetic proposers; qwen3, qwen3moe and qwen35 targets | the 2026-09-26 plan's step 1 gates, plus the hybrid fixture: synthetic proposers rejecting at j = 0, 1, 2 and k, every token and logprob equal to the run without drafts, greedy and seeded, on the CPU, one MI50 and the Radeon VII; drafts past an end token and a second retract before a pass (lessons 5 and 8); retracts repeated at intermediate positions of a long generation; a rerun that fails, injected |
| 4 | `feat/qwen35-mtp` | the embedded MTP proposer as section 7 plans it: the MTP layer indexed by token, its context rows in the target's pass, its carried row in the slot pool, the on-device draft chain, the output device on a split, `--drafter embedded` | section 7's gates: the 2026-09-26 plan's step 4 gates (identity on and off, a loaded drafter at k = 0 giving the logits of none, decode within 3 percent and pp512 and pp16384 within 2 percent at k = 0, acceptance within the margin of the exact reference build on Qwen3.6-27B-MTP and Qwen3.8-27B Q8_0), plus the history calls carrying the MTP rows, the assembled tiny reference and a long-context cell |
| 5 | `feat/spec-server` | steps 3 and 4 in the scheduler (section 3), after the layer split's final gate | `server-spec` CTest, `tools/server_mix_check.py` drafts on against off at P = 1 and P = S, `tools/server_load.py` at 1 to 64 users |
| 6 | `feat/spec-drafters` | sidecar sources, `spec::pair`, draft models, `llmx-drafter-pack` | the 2026-09-26 plan's step 2 gates |
| 7 | `feat/spec-dflash2`, then DSpark, then qwen4exp MTP | as the 2026-09-26 plan's steps 5 to 7 | theirs |

Step 2 is the first user of sections 1 and 2 and nothing in it is specific to chat; step 3 adds the mark with its first caller, and steps 3 to 7 reuse the owner otherwise unchanged.

## 6. Decisions (planned)

Each recommendation was agreed with XDEV on 2026-09-30, with its clarifications written in, and the user approved them all that day ("design is settled, start").

1. **Checkpoint position.** At the largest block boundary at or below the stable prefix, a position whole in every storage, recomputing at most one of the largest blocks (64 tokens on Vulkan, 128 on the CPU; an estimated 0.2 s on the 27B), or at the exact position with a new op copying the partial KV block.
   **Recommendation:** the block boundary; the exact position only if a measurement shows the block matters.
2. **Checkpoint slots and flag** (the qwen35 plan's question 7). `--state-checkpoints N`, by default the fewer of `--max-seqs` and what fits in a quarter of the room after the live slots, 0 for none.
   **Recommendation:** approve, with that name.
3. **How a checkpoint is written.** The slice reaching c ends at c and its state goes into the checkpoint slot, at most one more pass boundary a request and no kernel change, or an in-kernel store at a named row with no cut.
   **Recommendation:** the cut, on the recurrence's global chunk grid, gated on time to first token; the named row only if the extra boundary measures there.
4. **A previous reply's rows on a follow-up turn** (step 1). The reply was computed as decode rows (extent 1) and the next turn's prompt holds the same tokens as prompt rows, which the kernels may compute otherwise (on Vulkan the row kernel against the tile past its crossover, on the CPU the decode dots against the prompt dots).
   - A, one rule: a fork never crosses classes, so a follow-up recomputes the reply's rows as prompt rows, about 1.3 s per 1000 reply tokens on the 8B on one MI50, and every server follow-up equals the CLI bit for bit.
   - B, today's reuse: fork the reply's rows by tokens; free, but a follow-up differs from the CLI in its last bits and can flip a greedy token (lesson 16 is that failure).
   - C, the class function decides: a fork takes the longest prefix of a reply's rows whose complete class (section 1) is proved equal to the prompt's, and recomputes the rest. No current model has decode and prompt classes proved equal, so C acts as A for replies until a backend proves one; it then becomes free with no scheduler change. Tests hold the threshold edges (a short prompt below a tile crossover) and per-device type overrides.
   - Scope: the Qwen3 and Qwen 3.5 and 3.6 templates drop a reply's reasoning, so a thinking reply's rows are never reused under any option; the cost falls on replies without reasoning and on text continuations. A hybrid model's checkpoint sits at S, before the reply, under every option.
   **Recommendation:** C, conservative: classes are separate unless proved equal; B is rejected, since its speed does not buy the exactness it gives up.
5. **Partial rollback of a recurrent state.** A rerun of the recurrence alone from the mark over saved inputs (at most k_max + 1 rows, about 16 MB of saved inputs a sequence on the 27B at k = 3, an estimated, unmeasured 1 to 2 ms a rejected round there, mostly reading and writing the 149.6 MiB state once), or a state snapshot after every verify row, (k_max + 1) x 149.6 MiB a drafting sequence on the 27B (748 MiB at k = 4). mx-llama.cpp's snapshot ring won against restoring and decoding the whole model again (lesson 10); the rerun does not decode the model again either.
   **Recommendation:** the rerun first; if it measures above 5 percent of a decode step, the alternatives are measured, never an inexact path taken. Marks, saved rows and their copies per pass slot are counted in the drafter's fit after the KV budget.
6. **Host tier for checkpoints.** **Recommendation:** not in step 2; measured once step 2 shows how often device slots evict a donor that a later turn wanted.
7. **Order.** 8c first (steps 1 and 2: checkpoint, fork and retract to a checkpoint only), then the verify with lookup and the mark (step 3), then the idle re-prefill (2c) and the host tier's whole-donor part (2b a), then qwen35 MTP in the CLI (step 4), the message-boundary checkpoints (2b b) after the host tier's own measurement without holding step 4, then speculation in the server after the layer split's final gate.
   **Recommendation:** approve; if MTP should reach the server before that gate, step 5 runs on today's round at one pass in flight and the gate reruns once passes in flight land.
8. **The rest of the 2026-09-26 recommendations** (its questions 3 to 6 and 8 to 19: the `draft.` namespace, gate thresholds, the Radeon VII gate, the reference builds, the V4.1 DSpark file and placement, drafter rows under reuse, drafter memory, DFlash, DeepSeek and qwen4exp MTP, the single-user comparison, draft models on the server, coupled drafting, the `--drafter` default); its question 17, the verify-slot pool, is replaced by the mark.
   **Recommendation:** carry them over as approved, each step checking again the assumptions it relies on; they are decisions, not measurements revalidated today.

## 7. Step 4: the embedded MTP proposer (planned)

Proposed on 2026-10-01 for review with XDEV and the user's approval: how step 4 builds MTP for qwen35 on sections 1 to 3 and on step 3's code, changing none of their decisions.
Sources: the MTP block in [QWEN35](QWEN35.md); the user's mx-llama.cpp history (section 4, lesson 24 and the notes cited below, at `test/best-stack-mtp-20260902`, f2a54df595); vLLM's `vllm/model_executor/models/qwen3_5_mtp.py` and its EAGLE-family proposer `vllm/v1/spec_decode/llm_base_proposer.py` (main at 08e03df9, read 2026-10-01); and the GGUF headers of the MTP files on the Linux MI50 machine.

**The files.**
Qwen3.6-27B-MTP Q8_0 and Q4_1 and Qwen3.8-27B Q8_0 and UD-Q8_K_XL each hold one block, `blk.64`, of 15 tensors: `nextn.eh_proj` [10240, 5120], the three `nextn.` norms, and a full-attention layer with a dense FFN under the decoder layers' names; none has `nextn.embed_tokens` or `nextn.shared_head_head`, so the block reads the target's `token_embd` and `output`.
The Q4_1 file's block is Q8_0 throughout; the UD-Q8_K_XL's `eh_proj`, `attn_q`, `attn_k` and `attn_v` are BF16; the Qwen3.6-35B-A3B-MTP files' block is MoE with a BF16 router and shared-expert gate.
At Q8_0 the block's matrices take 430 MiB and the head 1.26 GiB, so a draft step reads about 1.68 GiB, 6.3 percent of a 27B decode step's weights; a context row reads only `eh_proj`, `attn_k` and `attn_v`, 63.75 MiB a pass, about 0.23 percent of a decode step and of a prompt row's multiply-adds; the MTP layer's KV is 4 KiB a token at f16.

**What a row computes.**
Row j of the MTP layer reads h(j-1), the target's row after `output_norm`, and t(j), at the target's rotary position j.
- A **context row**, one for every row a target pass feeds: u = `eh_proj` [RMSNorm(e(t(j)); `enorm`), RMSNorm(h(j-1); `hnorm`)], then `attn_norm`, K and V with the k norm and the partial rope, written into the MTP layer's KV row j.
  Attention, the FFN and the head are skipped: K and V depend on the layer's input alone, and nothing reads a context row's output (the user's history's KV-only replay, fa9c6ee0f2; vLLM pads with such rows).
- A **draft row**: the whole block, attention over the MTP rows 0 to j, the FFN, `nextn.shared_head_norm`, the target's head and the argmax, ties to the lowest id.
  Draft step m > 1 reads the previous draft as t and the previous draft row's output after `shared_head_norm` as h (vLLM's `propose` loop; the user's history after upstream 166fe29492).
- Row 0 reads a zero h (decision 1), as the user's history does since 017a5d3f8f.

**Owners**, each once:

| concern | owner |
|---|---|
| the block's metadata and refusals | `qwen35::read_config`, as today (one block at most) |
| the block's roles, planned only when a drafter is asked for | `Qwen35::plan` given the request, filling `ModelPlan::drafter`, a `LayerPlan` of `Part::draft` roles whose cache is KV and which carries one row a slot |
| the block's math | `Qwen35::draft_rows` (context rows) and `Qwen35::draft` (one draft row for each drafting sequence); the two norms, the concatenation and `eh_proj` in `blocks::nextn_input` beside the other shared pieces |
| the contract | `Architecture::draft_rows` and `Architecture::draft`, none by default; `Part::draft` |
| the MTP layer's KV | one more KV layer in the output device's storage, with the target's block table, length, fork, retract and budget |
| the carried row | E floats a slot on the output device, at the slot indices the state storages use, read at a pass's src slot and written at its dst slot; a sequence with no history reads a zero h whatever its slot holds, NaN included, as the state ops read a zero state; the h rows a mark saves sit in the mark's saved buffer, allocated, counted and released with the state layers' saved inputs by the same owner |
| context rows in a pass | `Model::run_stage` after the last stage, on the output device, before the head; with the block loaded the final norm runs over every row the pass feeds, not only the rows `blocks::head` gathers for logits, once a row, and the head reads the logits rows' normed values from that result rather than norming them again; its cost and the off and k = 0 byte identity are in the k = 0 gate |
| the draft chain | `Model::draft(seq, last, k, out)` in `model/history.hpp`, beside the other history operations |
| the proposer | `spec::Embedded` in `inference/spec.hpp`, whose `draft` calls `Model::draft` with the history's last token |
| two new ops | `Backend::argmax_rows` (ties to the lowest id, an id past the vocabulary where the best logit is not finite) and `Backend::embed` reading ids from a device buffer, on the CPU and Vulkan, refused at load by name where a backend lacks them; an invalid id is handled on the device: `embed` writes a zero row for it and never reads the table there, and `argmax_rows` gives an invalid id for every later step of that request, so one request's invalid draft neither reads out of bounds nor touches another request's rows |
| loading and the fit | `PlacementRequest::drafter` through `place_model` and `plan_model`, the placement deciding where the block's roles sit and the proposer none; the fit counts the block, an embedding copy only where no copy of `token_embd` is resident on the output device already (a tied head or the embedding there is reused, never adopted or counted twice), the MTP KV, the carried rows and the mark's saved h rows after the KV budget, or refuses with the numbers |
| the flag | `--drafter embedded` in `Drafts` (`cli/main.cpp`) on `generate` and `chat`, and on `bench --model`, which loads the block and runs its context rows without drafting (the k = 0 gate); a file without an MTP block refused, naming the file |

The round (`infer::generate`), `infer::accept`, `spec::draft_length`, `spec::Acceptance`, `Model::mark`, `Model::retract` and `Model::rerun` are step 3's and gain no MTP code; `Proposer::reads`, `block` and `settle` stay deferred, since the MTP proposer keeps nothing of its own.

**One round**, history L, the carried row holding h(L-1), last pick y:
1. `mark(seq)`, as in step 3.
2. `Model::draft(seq, y, k, out)`: one submission on the output device takes the KV blocks of positions L to L + k - 1 in its storage and commits no length; draft row L reads the carried row and y, row L + m - 1 the previous row's normed output and draft m - 1; `argmax_rows` writes each draft id on the device and `embed` reads it for the next step; the host reads k ids once and keeps those before the first invalid one.
   The blocks it takes are taken and returned through the KV cache's owner like a pass's, so a failed draft, an allocation failure among them included, returns them though no length was committed.
   The rows it wrote lie past the committed length and are overwritten by the verify before anything reads them.
3. The verify, `Model::step([y, d1 ... dk])`: after the last stage, the context rows L to L + k, row L from the carried row and y, row L + i from the pass's h(L + i - 1) and d(i); the pass writes h(L + k) into the dst slot's carried row and, under the mark, saves the pass's h rows as `LayerPlan::saved` inputs.
4. `infer::accept`, then `retract(seq, n)`: the MTP rows past n go with the target's blocks, and the rerun that restores the recurrent state also copies h(n-1) from the saved rows into the live slot's carried row.

A round drafting nothing, a plain decode step and every prompt slice compute their context rows the same way, the first from the carried row, so the MTP layer always holds every row its tokens determine, as vLLM keeps its drafter's KV in step at K = 0.

**Rollback and reuse need no MTP code.**
The MTP KV is the target's storage, so retract, fork, reset and a failed pass treat it as one more layer.
The carried row is a slot's, so a checkpoint at c carries h(c-1), a fork at c reads it, a mark's rerun restores it, a failed pass returns to the mark's or the checkpoint's, and a reset returns it; lessons 14, 15 and 24 are the failures this rules out.
A failed draft pass changes no committed length, and the round's retract to L returns its blocks and the mark.

**On a layer split** the block runs on the output device, the last stage's on a pipelined split: the target's normed rows, the head and the carried row are there, so nothing crosses for drafting (lesson 13).
`token_embd` is adopted there too where the embedding sits on another device, 1.26 GiB on the 27B Q8_0, counted by the fit (decision 5).
The draft chain is one submission on that device; in the CLI the other stages wait for it, and in the server (step 5) passes in flight use them.

**Row classes.**
A context row takes its entry's extent, so a prompt's MTP rows are prompt rows and a reply's are decode rows, and a fork shares a donor's MTP rows under the rule that already shares its target rows (section 1); the block's matmuls choose kernels by extent as every op does.
Draft rows are extent 1 and never history.

**Gates**, beside the 2026-09-26 plan's step 4 gates in section 5:
- Identity: tokens and logprobs with `--drafter embedded` equal to `--drafter off`, greedy and seeded, through `generate` and `chat`, on the CPU, one MI50, a two-MI50 split and the Radeon VII (tiny fixtures there, since a 27B does not fit), on Qwen3.6-27B-MTP Q8_0 and Q4_1 and Qwen3.8-27B Q8_0; with the block loaded at k = 0, the logits of no drafter bit for bit.
- History: on the tiny MTP fixture, the drafts and draft logits of a sequence forked at a checkpoint, of one retracted at every position of a generation cycle, of a reset one and of one continued after a failed pass equal those of a fresh sequence fed the same tokens, bit for bit, alone and beside others and over four CPU stages; a first pass whose slot holds NaN reads a zero h; a draft past the vocabulary or not finite ends drafting for its request.
- The two ops: NaN, infinity, an invalid id and ties in one request's row beside a valid request, which drafts as it does alone, the invalid one reading no table row; a draft's block reservation failing, part way through a block and at an allocation, returning every block it took.
- The block's math: the tiny fixture's draft logits at steps 1 and 2 against the assembled HF reference (decision 7), within the F32 bounds, from prompts of one and two tokens, so row 0 and the first carried row are both reached, and from longer ones.
- Acceptance: per position at `--draft-max` 1 and 3 on a fixed prompt set, within the margin of the reference build (decision 8), on Qwen3.6-27B-MTP Q8_0 and Qwen3.8-27B Q8_0, with the counts and denominators of every position kept; a draft length above 1 is what checks the h of step 2.
- Rollback: the user's history found rollback the main cost of MTP, so a round's rollback is timed on its own on the 27B Q8_0 on one MI50 and a two-MI50 split: a partial rejection (the mark's rerun of every state layer over the kept rows, the MTP rows' retract and the carried row's copy) at most 2 percent of a decode step, a full acceptance and the mark itself no measurable cost; timed as completed work, queued device execution and synchronization included, not host submission alone, against a decode step at the same placement, precision and clocks, at every rejection position with the kept prefix's length recorded, so a cheap short rerun cannot stand for the long one; if the rerun misses, it runs as one submission over every state layer, fused inside the mark's saved-state and rerun owner with its failure and history checks kept, before step 4 merges.
- Speed and memory: decode at the default depth against off and against the reference build with its MTP, on one MI50 and a two-MI50 split, with the memory the block takes; at k = 0 decode within 3 percent and pp512 and pp16384 within 2 percent; and one cell after a 16k-token prompt, since vLLM reports acceptance falling with context and decode slower with MTP from 16K (its issue 47602, Qwen3.6-27B: 0.935, 0.829, 0.715 by position at 2K against 0.721, 0.512, 0.395 at 30K; decode +129 percent at 2K, -14 percent at 16K).
- The use: a measured decode gain on the dense 27B, or it does not merge.

**Decisions**, each with options and a recommendation:
1. **Row 0**, which has no h(-1).
   - A: a zero h, attended like any row, as the user's history does; no kernel change; the acceptance gate's reference computes the same function.
   - B: a first row in the attention view, so the MTP layer attends rows 1 to j as vLLM's does; a bound in the CPU attention and the three Vulkan attention kernels.
   **Recommendation:** A; B only if the acceptance gate misses at short prompts.
2. **Rotary position** of row j: the target's j (the user's history), or j - 1 (vLLM, and the qwen35 plan's earlier text); rope is relative, so the scores differ only in rounding.
   **Recommendation:** j, reusing the pass's positions with no shifted list; this replaces "rotary position i - 1" in the qwen35 plan.
3. **The carried row.**
   - A: a row a slot on the output device, carried by the slot pool's history calls.
   - B: a row a sequence outside the pool, with its own reset, fork, checkpoint and retract code, which is how lessons 14, 15 and 24 happened.
   - C: none, recomputing the row before a fork as vLLM recomputes the last block on a prefix hit (its issue 38182: the hit rate fell from about 92 to 71 percent), which a model that keeps a state cannot do without a checkpoint a block earlier.
   **Recommendation:** A.
4. **The draft chain.**
   - A: one submission of k steps on the device, with `argmax_rows` and `embed` from device ids.
   - B: a host loop, one wait a step; the user's history measured about 12.3 ms a call of up to four steps on six MI50s, and an h fetch of 46.6 ms against 3.2 ms of draft decode on a tensor split.
   **Recommendation:** A, as the 2026-09-26 plan approved.
5. **The embedding on a split's output device.**
   - A: `token_embd` adopted there as well, counted by the fit.
   - B: embedding rows crossing with the residual for context rows, and a host lookup and upload for every draft step.
   **Recommendation:** A.
6. **Depth.**
   One `--draft-max` default for every drafter, 3 today; the user's history found the 27B Q8_0's best at 2 to 4 on a layer split (+52.6 to +89.6 percent, 0.867, 0.676 and 0.511 kept by position at 3) and output changes at 8 from its kernels, which extent-1 verifies rule out here.
   **Recommendation:** sweep 1 to 4 on the 27B Q8_0 on one MI50 and a split and keep one default, chosen from the workloads' tradeoff with every depth's results kept; a default per drafter only if the evidence asks for it.
7. **The assembled tiny MTP reference** (the qwen35 plan's question 11): HF drops `mtp.*`, so HF's own full-attention `Qwen3_5DecoderLayer`, with the fc, the three norms and the head written around it in the reference environment, run on the tiny fixture with an MTP block.
   It is labelled an assembled HF reference, not HF's own MTP; its packages, code and fixture are pinned, and it holds each convention on its own: the concatenation's order, the target's h after `output_norm`, the h of step 2 after `shared_head_norm` and the rotary position; step 1 alone does not qualify the chain.
   **Recommendation:** approve; it is the only check of the block's math against an external reference.
8. **The acceptance reference build**: the user's mx-llama.cpp family at `test/best-stack-mtp-20260902` (f2a54df595), which holds the post-norm h (166fe29492) and the zero carrier at position 0 (017a5d3f8f); the June notes' figures used the pre-norm h and are history.
   Before measuring, the exact commit, build, flags, model files and prompts are frozen and recorded, and no other reference is chosen after a miss.
   **Recommendation:** approve, the margin read as 5 percentage points of acceptance at each position, with counts and denominators kept.
9. **Scope.**
   Step 4 is the CLI and the dense qwen35 files; the qwen35moe block runs through the same code with the MoE FFN and is gated once the 16-bit branch reads its BF16 router (the user's history measured the 35B-A3B flat to +1.4 percent on a layer split); the server is step 5; the sidecar form the qwen35 plan's step 9 named moves to step 6 with the other sidecar sources.
   None of the three is marked supported from the shared code alone, only once its own gate passes.
   **Recommendation:** approve.

XDEV reviewed this section at 26c40e54 on 2026-10-01, agreed with the owners and the nine recommendations, and asked for the amendments now written in: invalid drafts handled on the device before the next embedding read, the draft's blocks returned through the cache's owner on failure, a resident embedding reused, the carried row's saved rows with the other saved state, one- and two-token prompts and each convention held by the assembled reference, and the reference build frozen before measuring with the margin in percentage points.
The user approved it on 2026-10-02 ("Mtp is yes if xdev accepted").
