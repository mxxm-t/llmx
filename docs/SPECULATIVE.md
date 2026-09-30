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

**The owner** is the model runtime (`model/runtime.hpp`, `model/kv_cache.hpp`): `infer::Model` and `infer::Sequence`.
The server and the CLI ask for the operations below and never touch a block, a slot or a storage; the module declares its cache kinds and names no operation.

**The operations**, each one call on `Model`:

| operation | call | paged KV | recurrent state | cost |
|---|---|---|---|---|
| extend | a pass's commit (today) | rows appended | updated in the live slot | the pass |
| checkpoint at p | an entry ending at p with `BatchEntry::keep` | nothing: rows below L are never rewritten | the entry writes its state into a free slot (StateView `dst`), which becomes the checkpoint at p; the next pass reads it (`src`) and writes the live slot | no copy, no host wait |
| fork at p | `fork(src, p)` | whole blocks below p shared, as today | the fork's first pass reads src's checkpoint at p in place and writes the fork's own live slot | no copy |
| mark (from step 3) | `mark(seq)` | nothing | the live slot is kept as the mark at L; later passes write a fresh slot and save their recurrent inputs, up to k_max + 1 rows | no copy |
| retract to L | `retract(seq, L)`, returns the length reached | blocks past L returned | inside a mark's saved rows: the recurrence rerun from the mark over them into the live slot, enqueued before the next pass; else the latest checkpoint at or below L, else 0 | KV free; state: a rerun of at most k_max + 1 rows, or the caller's recompute |
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
`ModelOptions::state_slots` is the total, fitted at load and never grown; `ModelOptions::live_slots` of them stay available for live use, so admission never waits on a slot: when a live slot becomes a checkpoint or a mark, a free slot takes its place on the live side in the same call, or the change is not made.

## 2. Checkpoint storage and policy (planned)

**Where.** On the device, as more slots of each state storage beside the live ones, so a restore or a fork reads a checkpoint in place.
A host tier, for donors the device slots cannot keep, comes only if measured to pay: a 27B checkpoint crosses in 13 to 31 ms at 5 to 12 GB/s, against about 4 s of recompute per 1000 tokens.

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
- Message-boundary and every-N checkpoints are later options, each kept only on a measured gain.
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

## 5. Order of work (planned)

Each step is a branch off main, landed as at most two commits, with every command byte-identical to main where the step's feature is off, and `llmx-split-check` bit-identical on the splits it runs.

| # | branch | brings | gate |
|---|---|---|---|
| 1 | `fix/server-row-class` | `Model::row_class`; `best_donor`, first admissions and take-backs compare classes by it, for every model (SERVER, Open gaps, closed) | `server-resume` and `server-passes-cpu` unchanged; a follow-up turn's ids equal `generate` on its full prompt, on Qwen3-0.6B and 8B Q8_0, CPU, one MI50 and a split; `tools/server_mix_check.py` |
| 2 | `feat/qwen35-checkpoints` (8c) | section 1's checkpoint, fork, retract to a checkpoint and failed-pass rules over KV and states (no mark yet), the one slot pool, `--state-checkpoints` and the fit, section 2's policy in the scheduler, `chat`'s checkpoint | `arch-qwen35`: a fork at a checkpoint equals a fresh sequence fed the same tokens bit for bit, alone and beside others, and over four CPU stages; retract and a failed pass back to a checkpoint; generation cycles that decode, retract from the new tail and continue, equal to never retracting, with four sequences at once (lesson 3); `server-resume`, `server-passes` and `server-passes-cpu` with hybrid donors; the `qwen35` component: a six-turn conversation, every turn's ids equal to `generate` on its full prompt, reused tokens growing each turn; `tools/server_mix_check.py` on Qwen3.5-0.8B, 9B and Qwen3.6-27B Q8_0 on one MI50 and the Radeon VII (0.8B and 9B); the use: time to first token of each turn of a 27B conversation reaching 8k tokens, against main, and the KV budget each fits |
| 3 | `feat/spec-verify` | multi-row `step`, `mark` and the saved-row rerun, `infer::accept`, `spec::Proposer`, lookup, `spec::draft_length`, the round in `generate` and `chat`, `--drafter off\|lookup`, `--draft-max`, test-only synthetic proposers; qwen3, qwen3moe and qwen35 targets | the 2026-09-26 plan's step 1 gates, plus the hybrid fixture: synthetic proposers rejecting at j = 0, 1, 2 and k, every token and logprob equal to the run without drafts, greedy and seeded, on the CPU, one MI50 and the Radeon VII; drafts past an end token and a second retract before a pass (lessons 5 and 8); retracts repeated at intermediate positions of a long generation; a rerun that fails, injected |
| 4 | `feat/qwen35-mtp` | the embedded MTP proposer: the MTP layer indexed by token, the in-pass rows, the on-device draft chain, the output device on a split | the 2026-09-26 plan's step 4 gates: identity on and off, a loaded drafter at k = 0 giving the logits of none, decode within 3 percent and pp512 and pp16384 within 2 percent at k = 0, acceptance within the margin of the exact reference build on Qwen3.6-27B-MTP and Qwen3.8-27B Q8_0 |
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
7. **Order.** 8c first (steps 1 and 2: checkpoint, fork and retract to a checkpoint only), then the verify with lookup and the mark, then qwen35 MTP in the CLI, then speculation in the server after the layer split's final gate.
   **Recommendation:** approve; if MTP should reach the server before that gate, step 5 runs on today's round at one pass in flight and the gate reruns once passes in flight land.
8. **The rest of the 2026-09-26 recommendations** (its questions 3 to 6 and 8 to 19: the `draft.` namespace, gate thresholds, the Radeon VII gate, the reference builds, the V4.1 DSpark file and placement, drafter rows under reuse, drafter memory, DFlash, DeepSeek and qwen4exp MTP, the single-user comparison, draft models on the server, coupled drafting, the `--drafter` default); its question 17, the verify-slot pool, is replaced by the mark.
   **Recommendation:** carry them over as approved, each step checking again the assumptions it relies on; they are decisions, not measurements revalidated today.
