# Cache and context plan

One plan for where a conversation's cache lives and how it moves, in place of one fix per incident.
Approved to build on 2026-10-13, the nine questions decided as section 6 records; step 1 is the first built.
Names in sections 4 to 6 may not exist yet.
Each statement about today's code is marked **[v]**, verified by reading the code or a log for this plan, or **[i]**, inferred from those; **[r]** marks what the reviewer read and this plan takes from the review.
Read at main e982e13dd; production's log and health at revision 7cf01359f on 2026-10-13.
Revision 6, the decisions recorded. Revision 5 came after the reviews of 2026-10-13; both reviewers cleared revision 3 for the owner, revision 4 holds their text corrections and this one adds step 10, the next context window prepared before a cut.

## 1. What the user is promised (planned)

Each promise is a statement a test can check, in terms of tokens, bytes and measured rates, never of a device.
"Message boundary" is the end of the last message a turn shares with the turn before; "restore" is a copy from host memory or a read from disk onto the device; "the tiers" are the device pool, the host cache and the disk cache as the user sized them.

- **P1, reuse.** A conversation's next turn computes no row that a restore point of it already holds, wherever that point lives: on the device, in host memory, on disk, or in work the server has in flight for that conversation. Rows past its newest restore point are new work, and the server says how many of them the read-ahead would have prepared.
- **P2, other users.** While the tiers can hold both, a request of another conversation never makes a conversation's next turn recompute rows. It may cost that turn one restore, whose wait is finite: it ends at a stated timeout, whatever has or has not been measured, and the turn then computes its rows as if the restore were not there.
- **P2b, the same the other way.** Nor does a conversation's turn make another conversation's request lose what it holds, under the same clause; where the tiers cannot hold both, P3's ranking says who loses.
- **P3, a holder always.** From a turn's end on, a conversation's rows up to its last message boundary have at least one holder that survives any other request's admission, as long as the tiers can hold them together. Where they cannot, the one of least worth by the one ranking goes, and P8 applies.
- **P4, restart.** With kept files written by a build of the same numerics, or proved (`--disk-cache-adopt proved`), a restart costs the next turn a read, not a recompute. Across builds of other numerics without proof it costs a recompute, said at start.
- **P5, the cut.** A prompt longer than the context is cut at the same place turn after turn where a step's cut point exists; where none lies a whole step past the leading block, the cut is at the first point that fits and moves with the prompt. The turn after a cut pays one re-read of a known size, said in the log, and the turns after it reuse as P1 says. Where the server had spare room and time before the cut, part of that re-read is already prepared (step 10); under saturation it is not, and no cut is promised to cost nothing.
- **P6, replies.** Background work reaches the device only in bounded pieces admitted by the scheduler: copies, writes, reads and the read-ahead of another conversation's reply. With pacing on, what those pieces may add to a reply's gap is a stated budget inside the pacing target; with `--no-pacing`, background work never takes a pass a waiting or generating request could use. The promise is over what the scheduler admits, not over a file system that stalls. Clarified by the two reviewers on 2026-10-13, for work a request asks for itself: "A request's own draft-history preparation may run without a background grant only while no other queued, paused, prompting or generating request could use a stage it touches; it runs in bounded pieces, checks that demand before each piece, and charges its full time to that request." With other demand the background budget applies unchanged.
- **P7, background work.** Work the server does ahead of a request (reading a reply again, writing files, proving files) never takes what a waiting request needs, and giving way never loses cache.
- **P8, nothing silent.** Every loss of a restore point, every restore refused, every background job started, given way or handed over is printed once with its reason and counted in `/v1/health`.
- **P9, identity.** A reply is the same bytes whichever path its rows took. This holds today and every step keeps it.
- **P10, edits.** An edited or regenerated turn reuses up to the newest restore point at or below the last message it leaves unchanged.

**What "reuse" promises when recomputing would be cheaper.** Revision 2 promised reuse and also let the server choose recomputing where the measured rates made it cheaper; the two disagree. This revision takes the reviewer's reading as the default: a history that is already reusable is preserved while the tiers have the capacity, and a restore that is available is taken. Recomputing in place of an available restore is not done unless the owner accepts it (question 8); if accepted, it is a decision the server reports as its own kind, never as a loss.
A turn's rows therefore fall into five kinds, which the server and the scenario tell apart: **reused**; **new work** (rows no restore point ever held, among them what a skipped or unfinished read-ahead would have prepared); **evicted** (a restore point the tiers could not hold); **failed restore** (a restore that was started and failed or timed out); and, only under question 8, **chosen recomputation**.

## 2. The parts as they are today

One paragraph and one owner each.

- **The context rule.** `Scheduler::prompt_room` gives the rows a prompt may take beside its reply: the limit less the asked tokens, or less an eighth of the limit for a reply without a cap (`kReplyRoom`). `Api::fit` cuts a prompt past it under `--context-overflow shift`, in steps of half the limit counted from the prompt's start (`kCutTo`, `first_kept`, `cut_rows` in `src/server/context_cut.hpp`). **[v]** Production cut 136725 tokens of 53 messages on three turns running. **[v]**
- **The device pool.** Room on the device is three things: blocks, per cache pool (`pools_`, `reserved_`; a split has a pool a device or stage **[r]**), state checkpoint slots, which `checkpoint_room_held` takes from donors **[v]**, and mark slots for drafts **[r]**. `infer::Model` (`src/model/history.hpp`) owns sequences, forks, checkpoints and blocks, and counts a block a fork shares once, under a refcount **[r]**. The scheduler's ledger does not: a request reserves blocks for its whole history and its reply room (`Growth::entry` in `src/server/policy.hpp`: capped, what it may still generate; uncapped, a step of 256 tokens, taken again as it grows, `Scheduler::grow`), a donor keeps the blocks of its whole length (`park`), and each is added to `reserved_` whatever it shares. **[v]**
- **Message boundaries.** `Scheduler::keep_boundary` keeps a state alone in host memory at a message's end (`bounds_`); it serves only with a history that holds its rows (`best_bound`, `holds`). **[v]**
- **The host tier.** `Scheduler::write_back` copies a donor leaving the device into host slabs (`host_`), within `--host-cache-bytes` and the host's reserve (`Model::alloc_host`, `host_admits`); it prefers conversations that came back (`back`), and `promote` copies an entry back to the device for a request. Room is made by `release_written`, `drop_oldest_bound`, `drop_oldest_host`. **[v]**
- **The disk tier and its store.** `DiskTier` (`src/server/disk_tier.hpp`) owns which files a history lacks and the reads and writes in flight; `DiskIndex` the paths; `DiskStore` the bytes, published whole. The scheduler starts writes in two places: `write_ahead` (unwritten host entries past three quarters of the tier) and `flush_one` (a server idle five seconds under keep, and the stop). **[v]**
- **Restore from disk.** `Scheduler::prefetch` looks once at each waiting request for a path on disk longer than what memory holds, `start_read` takes host room and starts the read, `finish_read` makes a host entry or a boundary, and the request waits with a bound (`disk_wait_`, `disk_until_`). At most two reads at once (`kDiskReads`). **[v]**
- **Jobs.** After a reply the routes give the ids the next turn will begin with (`Scheduler::follow`); a job is a request of no generated tokens that reads them ahead through `enter` and parks as a donor with a boundary (`complete_jobs`). A job gives way at a pass boundary to a waiting request that does not fit (`yield_jobs`, which releases its sequence and blocks) or is handed to the turn it read for (`hand_jobs`). **[v]**
- **Admission.** `Scheduler::enter` is the one place a request gets its source and its room: its own donor, the best donor, host entry, boundary or running job by `shareable`; then `make_room` (`src/server/policy.hpp`), the one rule of who gives up blocks, donors oldest first and the forked one last. A donor the request forks and that must also give its blocks is consumed: dropped with no host copy, the fork holding its rows. **[v]**
- **Pacing where it meets them.** Generated rows ride the passes that read prompts or go in passes apart between a prompt pass's pieces (`Scheduler::rides`, the stage recorder). A job's prompt rows are prompt rows like any other to this rule. **[v]** Copies to and from host memory are enqueued behind passes under a hold (`quieted`). **[v]**

### Where one conversation's cache can be, and every move

Places: **A** an active request's sequence; **J** a running job's sequence; **D** a device donor; **B** a boundary's state in host memory; **H** a host entry; **F** files on disk; **R** a read in flight.
The table is the audit of 2026-10-12 with three rows added (a donor consumed, a job giving way, the cut); each row names who decides.

| Move | Decided by, on what rule | Where it can drop the cache |
|---|---|---|
| A to D, a turn ends or is paused | `park`, `pause`: whole blocks and, with a state, a checkpoint | no checkpoint or no whole block: released |
| D, H, B or J to A, a fork | `enter` with `shareable`; `Model::fork` | promotion or the state's slot failing falls back to a weaker source |
| D to nothing, consumed by its fork | `enter`, when the fork's reservation and the donor's do not fit together | right for the rows the fork took; the donor's rows past the fork point go with no copy (N14), and the fork may be a job, whose giving way is three rows down **[v]** |
| A to J, the read-ahead | `follow`, `enter` | as any admission |
| J to D and B, the job done | `complete_jobs`, `keep_boundary` | the boundary refused room |
| J to nothing, the job gives way | `yield_jobs`, when a waiting request does not fit | rows released, no copy, no line **[v]** |
| J to D, handed over | `hand_jobs`, only to the turn it read for, only a state the job kept | otherwise it gives way |
| D to H, room takes a donor | `make_room`, `drop_donor`, `write_back` | superseded, nothing to copy, larger than the tier, refused for copies in use, copy failed |
| H to F | `write_ahead` past three quarters unwritten | the entry can go before its file is whole |
| D to F | `flush_one`, idle five seconds under keep, or the stop | the donor can go first |
| H or B to nothing | `release_written` (file whole, no loss), `drop_oldest_bound`, `drop_oldest_host` | unwritten entries lost |
| F to R to H or B | `prefetch`, `start_read`, `finish_read` | room or memory refused; a state alone losing its rows before admission |
| H to D | `promote` | device room or a checkpoint slot not to be had |
| F to nothing | `DiskTier` for the cap, the age limit, a clean exit without keep | by design |
| anything at a cut | `Api::fit` | the rows before the cut are of no use to the cut prompt |

Seven places, fifteen moves, decided in more than a dozen functions by rules that do not share one notion of what a history is worth or of who else holds it.

## 3. Where the map breaks a promise (planned)

### The audit of 2026-10-12, seven gaps

| Gap | Promise | State |
|---|---|---|
| 1. The longest boundary chosen whether or not its rows exist | P1, P10 | landed 2026-10-13 (`best_bound`) |
| 2. A state read alone evicting the host copy it is for | P1 | landed 2026-10-13 (`read_source`); two windows left, N8 |
| 3. A request arriving while a job reads its conversation loses the job's rows | P1, P7 | landed 2026-10-12 (`hand_jobs`), for the turn the job read for only |
| 4. A donor taken before it is safe on host or disk | P3 | open; a first step exists unpushed (a history the host tier cannot keep written at its turn's end) |
| 5. Jobs never read from disk | P4 | open |
| 6. Two requests of one history each read it from disk | P2 | open |
| 7. Losses not counted | P8 | approved, at its gate (`lost_before_written`) |

### Production since 2026-10-11

- **S1, 2026-10-13, the top scenario.** A 147k-token chat, an unrelated 27k-token request after its turn. The chat's donor and its read-ahead job could not both be reserved (157.7k and 151k of 262144, shared blocks counted twice), so the donor was consumed and the job became the only holder; the next turn did not fit beside the job, the job gave way and its rows were released; the turn read 9.9 GiB from disk in 10 s, promoted it, and read 4434 rows in 46 s: 57 s to the first token where the turn before took 2.9 s **[v]**. The chain is true in the code **[v, r]**; that it is what ran at 08:01 is inferred, and two things only a trace settles: whether `hand_jobs` ran for that turn, and whether the job had a checkpoint then. The unrelated request generated at 3.7 tok/s for four minutes, riding the job's prompt passes **[v, the cause i]**. Nothing was printed and `lost_before_written` stayed 0 **[v]**. Breaks P1, P3, P6, P7, P8.
- **S2, 2026-10-12, the 35-minute re-reads.** A 255k-token chat read from its first token on most turns: a boundary was counted as a history, so no read from disk was started. Fixed (`prefetch`). P1.
- **S3, 2026-10-12.** The same chat's history, larger than the host tier, taken before an idle moment had written it. Gap 4. P3.
- **S4, 2026-10-12.** The next message arriving while the job read: the job gave way and the turn read everything, twice that day. Gap 3, fixed for that turn; S1 is the same loss for any other waiting request.
- **S5, 2026-10-13.** A request stalled by its own reply's job did not grow in the round the job gave way. Fixed. P6, P7.
- **S6, 2026-10-13.** After a restart onto a build of other numerics the kept files were not adopted, said in one line at start, and the chat's first turn read 123958 tokens in 682 s **[v]**. The same turn was a cut one, and nothing told the two costs apart. P4's clause, P5, P8. That figure is not what preparing a cut would save: a restart onto other numerics invalidates prepared work as it does kept files, so step 10 keeps restart and cut apart in its baseline and its outcome.

### Found reading the moves

- **N1.** Shared blocks are reserved once a holder, so a conversation longer than half the pool can never have two holders on the device, though it would cost a few blocks. The root of S1. P3. **[v, r]**
- **N2.** A job that gives way drops its rows with no host copy, where a donor leaving for the same room is written back. P3, P7. **[v]**
- **N3.** `prefetch` and `enter` count what memory holds differently: `enter` sees a running job and an active request, `prefetch` does not, and reads from disk what the device holds. P2. **[v, r]**
- **N4.** A job's prompt rows pace another user's reply. P6. **[v]**
- **N5.** Jobs print nothing and their loss is not counted. P8. **[v]**
- **N6.** A boundary can outlive every copy of its rows and then holds its 150 MiB (on the 27B) for nothing; production held seven, and nothing says which of them still have rows. **[v]**
- **N7.** `write_back` refuses a conversation that did not come back the room of ones that did, with no line; the refused donor is lost. Counted once gap 7 lands. P8.
- **N8.** The host copy a state read relies on is not pinned between the read's end and admission, nor is a device donor in that role. P1. **[v]**
- **N9.** `enter`'s fallback does not try the next boundary down when promotion fails. P1. **[v]**
- **N10.** A read's wait has no bound until a prompt pass has been measured (`disk_until_` is the end of time). P2. **[v]**
- **N11.** Nothing weighs a restore against recomputing: `prefetch` reads whenever disk holds more than memory. Right on a slow device; on a fast one with a slow disk a short history is cheaper to recompute. Whether the server should then recompute is question 8, not a defect. **[i]**
- **N12.** Five functions pick a victim (`make_room`, `write_back`, `release_written`, `drop_oldest_bound`, `drop_oldest_host`), each by its own order: age, superseded, came back, on disk. None knows whether the victim is the last holder of its history. P3. **[v]**
- **N14, from the scenario, 2026-10-13.** A donor consumed by its fork loses its rows past the fork point: the fork holds the donor's rows only up to where it forked, and the rest go with no host copy. After an edit of an earlier turn the original conversation's next turn reused 768 of the 1152 rows its donor had held, on both models; the edit had forked that donor at 768 (640 on the hybrid model) and consumed it. P1, P3, P10 for the branch left. Closed by step 3 where the donor was consumed for room alone, which it then is not; where a donor is still consumed, for the donor count or a checkpoint slot, by step 7's last-holder rule, which keeps or demotes what no other holder has. **[v]**
- **N15, from the scenario, 2026-10-13.** On a model that keeps a state and without a disk tier, an edit does not reuse up to the last unchanged message: of a conversation's boundaries only a fixed number stay in host memory (`kBoundaries`, the first, the newest and those spaced most evenly between), so the state at that message had been thinned and the next one down served, 640 of 768. P10. Closed by step 7: which states stay is the ranking's, by worth within the host tier's room, not a fixed count. **[v]**
- **N16, from the scenario, 2026-10-13.** A job gives way to a request that would fit once a donor were evicted: the test for "does not fit" counts free blocks only (`fits_free`), so once finished requests' donors fill the pool every arrival makes the running job give way, and it starts again. Below half the pool no turn loses rows by it; the job's rows are read twice. P7. Closed by step 4, where giving way parks the job's rows instead of releasing them, and where the test for whether a request waits is admission's own. **[v]**
- **N17, from the scenario, 2026-10-13.** On a model that keeps a state placed over two devices, a host tier of 1 GiB keeps almost no state at a message boundary: host memory is taken in whole slabs a device, so a boundary's state costs a slab on each device, and the tier's room goes to the copies. Edits and cuts there reuse nothing (0 of 768 and of 384 rows, where one device reuses 640 and 384), while a plain next turn reuses as elsewhere. It was silent until step 1; the server now says "a message boundary at N tokens was not kept in host memory (the host tier has no room it may give)", and says the same of a boundary a request asked for that found no checkpoint slot or no state. P5, P10 by capacity; whether a state should cost a whole slab a device is the host pool's question, for step 7's ranking to weigh. **[v]**
- **N13.** Room on the device is counted in three ledgers that do not know each other: blocks, checkpoint slots and marks. A donor at its checkpoint holds a slot as surely as blocks, and a victim chosen for blocks may be one a slot already decided. P3. **[r]**

## 4. The target (planned)

### One model, before any code shape

The lean shape: two functions over the holders as they are, no new type.
The containers stay (donors, host entries, boundaries, disk paths, reads in flight, running jobs and requests).
A record a conversation was this plan's first shape and is withdrawn: nothing identifies a conversation today, a history is found by its tokens, and an edited turn is a branch, so such a record would be a trie kept in step with seven containers for one caller.
It is reached for only if step 6 shows the question cannot be one function over the containers.

- **One question for reuse.** A function over the holders returning the longest restore point of a prompt that some holder serves, and its cheapest holder. Admission, the read from disk and a job all ask it. It answers N3 and gaps 1 and 5 by itself.
- **What is selected is kept until it is used.** Selecting is not enough for N8, N9 or gap 6. The rows, the state and the copy tickets of the selected restore are retained through admission and promotion by the pins and tickets that exist (`pinned_bound_`, `promoting_`, `read_source`, the model's copy tickets, the disk tier's pins), or the selection is validated again before any use and the next candidate down is tried. A read in flight shared by several waiters is released by none of them while another needs it, and the last waiter's leaving drains the read before its buffer is freed.
- **One ranking of what stays.** A function giving a holder's worth: its tokens at the measured prompt rate, whether and how lately its conversation came back, against its bytes in the tier it sits in. `make_room`, `write_back` and the tiers' room-making ask it for victims. A long conversation that came back is not the first to go for a short one that did not.
- **One count of room.** The model's count is the truth, and it is not a sum over holders: two forks sharing a hundred blocks would each give back none, and the pool still holds a hundred. So, per pool: what is taken is the model's unique occupied blocks (it already counts references, `BlockPool::refs` **[v]**) plus the room promised and not yet occupied (a reply's cap, a growth step). A victim's gain is a different quantity, the blocks its removal would free now, and `make_room` evaluates it again after each victim it simulates: two forks of one prefix free nothing singly and the prefix together. That evaluation is one read-only query of the model, the blocks a set of sequences would free together; the policy simulation models it as a set. Checkpoint slots and marks are dimensions of their own with the same two questions, never numbers added to blocks. Held by: two forks removed in either order, growth, a failed admission rolled back, and a split of unequal pools. N1, N13.
- **One rule of last holders.** No move drops the last holder of a restore point while another victim would do. When it must, the point is demoted first (device to host, host to disk) while the next tier has the capacity, and dropped only where it has not, said and counted as evicted. A file being written is not a holder until it is whole. Background work is never a last holder: a job that gives way becomes a donor, or was never started, and what a job never read is new work, not a loss. N2, N12, gap 4, S1.
- **One rule for background work.** Jobs, writes, proofs and reads for jobs reach the device in bounded pieces the scheduler admits, inside a stated budget of the pacing target, and take room only that no waiting request needs. P6, P7, N4.
- **Every wait is finite.** A request waiting for a restore has a timeout that does not depend on anything having been measured: a constant with its reason until the whole restore path (queueing, allocation, read, validation, upload) has been measured on this server, then that measurement with a margin. At the timeout the request goes on and computes; the read is left to finish for a later turn or is cancelled and drained. Bytes over a rate is an estimate the server may print, never the bound. N10.
- **Everything said.** Each move in the table prints once and counts once, through one function. P8.

**The context rule keeps its owners.** The room a prompt may take stays the scheduler's (`prompt_room`, `token_limit`), the cut the route's (`Api::fit`, `context_cut.hpp`). The cache's one concern with the context is step 9: entries a cut orphaned are ranked out, and the cut says its cost. Step 3 does not touch what fits a context.

**What differs by hardware is only the measured rates.** On a fast GPU recomputing is cheap, so a history is worth less against its bytes and the same ranking keeps other things; whether a cheap recompute may replace an available restore is question 8. With the cache on the CPU the device pool is host memory and there is no host tier; there are fewer places and the rules are the same. On a small device a conversation meets the cut early, and P5 carries most of the promise. Over a split the pools are several, a conversation's rows lie on all of them and a host copy spans them; "once" means once in each pool. No constant in the model is a device's.

### Whether the code is gathered

- Today the cache's decisions are between 1100 and 1400 of `src/server/scheduler.hpp`'s 3542 lines **[i, r: from the functions' line ranges]**, interleaved with the round, the passes and pacing. `policy.hpp` (496 lines) holds the pure rules, `disk_tier.hpp`, `disk_index.hpp` and `disk_store.hpp` (2206) the disk, `history.hpp` (835) the model's side.
- **Proposed: gathered, as a pure move after the scenario test and before any behaviour step.** One file, `src/server/reuse.hpp`, owning the holders and the moves between places: `donors_`, `host_`, `bounds_`, `disk_reads_` and `write_back`, `promote`, `keep_boundary`, `prefetch`, `start_read`, `finish_read`, the `best_` and `drop_` families, `release_written`, `write_ahead`, `flush_one`. No signature changes in the move; it is held by identical replies and counters on the scenario test and the suite.
- **What stays in the scheduler:** the round, the passes, pacing, the order of admission and the requests' and jobs' lifetimes. So `enter` stays and asks `reuse.hpp` for its source and for room, and `grow`, `hand_jobs`, `yield_jobs`, `follow_up` and `complete_jobs` stay.
- **What stays where it is:** the model's mechanics, the disk tier, index and store, and the pure rules in `policy.hpp`, which the audit found sound.
- **Sized honestly:** the move is one large change with no behaviour in it; it makes every later step a change to one file. It is not required for the promises, which steps 3 to 9 could close in place; it is proposed because the victim choices cannot share one ranking while they stay scattered among the round's code.

## 5. The scenario the plan is held to (planned)

One script at three sizes, in two shapes.

- **The single conversation:** a chat of many turns whose history passes half the pool; after every turn an unrelated request arrives while the read-ahead runs; an edit of an earlier message; a restart over kept files; a prompt that passes the context and is cut, then two more turns.
- **Many users at the cap:** N conversations taking turns with the pool, the host tier and the disk tier each too small for all of them, a job each, two reads at once and the pacing between them. Held to: the server stays usable, the one of least worth goes, and each turn of a conversation the tiers still hold keeps P1 and P2.
- **Reported per turn:** its rows by kind (reused, new work, evicted, failed restore, chosen recomputation), the restore taken (none, host, disk) with its bytes and the time of each part of its path, time to first token, the other requests' tokens per second and the largest gap background work added, and every counter.
- **Held to:** P1 (no row computed that a restore point held: evicted, failed and chosen rows are zero wherever the script leaves the tiers the capacity; new work is exactly what the script left unprepared), P2, P2b and P3, P4, P5 (the re-read's size as computed from the cut), P6 (the gap budget), P8 (the kinds add up to the prompt), P9 (each reply its reply on a fresh model).
- **The next window prepared, for step 10:** a conversation crossing two cuts, on the synthetic and the hybrid model and over a split, each reply and its logits those of the shortened prompt on a fresh model. Variations: a changed `max_tokens`, an edit of an earlier turn, tool groups, no stable step boundary (the moving fallback), a sudden large next input, a prepared job handed over incomplete, a cancellation, and a competing user. Reported against the same run with preparation off: background rows and time, the peak of extra device, host and disk use, prepared rows used and discarded, the rows the cut turn still computes, time to first token, and the other user's gap and rate. A restart is not in this run; its cost is measured in the single conversation's restart leg.
- **Lifetime cases beside it, for step 6:** a read completing and a donor or host copy evicted before the waiter is admitted; one of two waiters of one read leaving; the last waiter leaving while the read runs; a copy or a read failing after selection. Each ends with the next candidate used or the rows computed, the ledgers balanced and nothing freed while in use.
- **Room cases beside it, for step 3:** two forks of one prefix removed in either order, growth of one of them, an admission that fails after room was taken, and a split of unequal pools.
- **Where:** in CTest on the synthetic Q8_0 model and the hybrid one, on one CPU and over a split, counts only, no clock; by a runner on Qwen3.8-27B Q8_0 on one card and a tensor group of two with a tool that drives a real server, where time to first token is held to the measured complete restore path with its declared margin and interference budget, and the other replies' rate to the pacing target; the finite timeout and its fallback are tested separately, on a read made to stall.
- **How it starts:** it is written first and records each promise broken today as a named expectation; a step that closes one flips it, and none may break another.

## 6. Ordered steps (planned)

Each is small enough for one gate, and each is measured by the scenario's per-turn report.

| Step | Closes | Deletes | Gated by |
|---|---|---|---|
| 0. The loss counter lands (at its gate) | part of P8 | nothing | its own gate |
| 1. The scenario test, and every move said and counted through one function, jobs included | P8, N5; shows the rest and settles S1's two open points | the scattered print and count calls | counts |
| 1b. The scenario's legs that need a disk tier, tests only: a restart over kept files, two followers of one history on disk, a history the host tier cannot keep, many users at the cap | nothing; it holds later steps | nothing | counts; before step 3 for many users at the cap, which holds the room count, and before step 6 for the rest |
| 2. The move into `reuse.hpp`, no behaviour or signature change | nothing; one place | 1100 to 1400 lines from the scheduler, moved | identity |
| 3. Room counted once: per pool, the model's unique occupied blocks plus room promised; a victim's gain marginal and evaluated again after each simulated victim, by one read-only query of the model (the blocks a set of sequences would free together), so the sharing truth is the model's and no graph of it is kept in the scheduler; slots and marks as their own dimensions | N1, N13, N14 for room, the root of S1 | the consumption of a donor for room alone (it stays for the donor count and the slots) | the consume path gone from S1's shape; the room cases |
| 4. A job is never a last holder: giving way parks it as a donor, or, with no spare room, it does not start | N2, N16, S1, S4 for every request, P7 | `yield_jobs`' release; `hand_jobs` folds into the park | the loss counter at zero across a give-way |
| 5. Background work admitted in bounded pieces inside a stated budget of the pacing target, where passes are formed | P6, N4 | nothing; one rule | the gap counters and the other reply's rate |
| 6. One question for reuse, asked by admission, the read from disk and jobs; the selected restore kept or validated again with the next candidate tried; a read in flight shared, by the existing pins and tickets | N3, N8, N9, gaps 5 and 6 | `prefetch`'s own count, `rows_held`, the separate boundary search | reuse counts; one read for two followers; the lifetime cases |
| 7. One ranking and the last-holder rule for every tier: demote while the next tier has the capacity | P2b, P3, N6, N7, N12, N14 for the count and the slots, N15, gap 4 | the five victim orders, the fixed count of boundaries, the three-quarters rule, the idle timer as the only device-to-disk write | evicted rows zero where capacity allows |
| 8. Every wait finite: a timeout and its fallback that need no measurement, the whole restore path measured and reported. At the timeout the request goes on without the scheduler's thread joining the stalled read; the read keeps its buffer until it retires, and timed-out reads count within the existing limits on reads and their memory (`kDiskReads`, the host tier's room), so they do not pile up. Only if question 8 is yes: recomputing chosen over an available restore, reported as its own kind | P2, N10; N11 only under question 8 | the unbounded wait | a stalled read ending at its timeout with the rows computed; the path's parts in the report |
| 9. The cut and the restart each said with their cost; entries a cut orphaned ranked out | P5, P4's clause | nothing | the lines |
| 10. The next context window prepared before a cut, opportunistic and bounded | part of P5's re-read, where room and time allow | nothing | the two-cut scenario against the same run with preparation off |

Step 3 comes before step 4 of necessity: a job parked on giving way would double the room while blocks are counted twice.
Step 3 restates the policy simulation's invariant, no pool over-reserved, in the new terms, so `server-passes` keeps holding it; every admission that fits today still fits, and pauses and consumptions become fewer.
Step 5 is independent of the reuse code and closes what another user sees today, so it comes before step 6.

**Step 10, the next context window prepared before a cut.** It is the same lifecycle and pacing as the read-ahead of a reply, built from the earlier steps' owners and rules, not another cache or scheduler. Its conditions:
- **Who computes it.** The owner of the cut and the render, `Api::fit` with `first_kept`, `cut_rows` and the routes' rendering, computes the candidate shortened prompt with its system and leading content, tool groups, template variables and exact ids. No cut policy goes into the scheduler. A pure planning operation is taken out of `fit` only where the new caller needs it, and the foreground cut does not change.
- **What is prepared.** Only a stable prefix of one likely next window, where a valid cut point exists and there is spare capacity and time. Its cache is a distinct shortened history, matched by the existing rules of exact prefix, row classes and numerics. It is recomputed, never spliced: the old rows after the dropped text, and a recurrent state, still encode the old context, and moving positions does not make them the shortened prompt's. Only a leading prefix that the ordinary reuse rules allow is reused.
- **By what.** The planned background jobs, the one count of room, the retention and the hand-over of steps 3 to 7. `Scheduler::follow` today means the current conversation's next turn; the candidate is a second kind of ids given to the same machinery.
- **What it may not do.** Evict the conversation's current history, take pass time outside the reply budget of step 5, change the foreground cut, or stall it.
- **How it ends.** It is extended as the stable text grows. At the actual cut only the matching prepared prefix is used and the rest computed. A next input that is unexpectedly long, edits an earlier turn, changes the reply room, the template or the system text, or meets the fallback cut elsewhere, falls back to ordinary matching.
- **Its limits.** At most one candidate a history, within the limits on jobs and room. Without spare room or enough lead time it is skipped, and the foreground work that remains is reported. Preparing a whole window is real compute and additional cache: it cannot promise that a cut costs nothing under saturation.
- **Its baseline.** The 123958 rows in 682 s of S6 followed a restart onto other numerics as well as a cut and is not what this step removes; restart-invalidated work and cut work are measured apart.

**From the old list.**
- The turn-end write (built, local): it is the last-holder rule for one case, a history the host tier cannot keep. It becomes part of step 7, or lands before it as it is if the owner wants that loss closed sooner.
- The bounded wait of an admission for a write: inside step 7, and only if the scenario's measurement after the demotion rule still shows evicted rows; its wait would be finite by step 8's rule.
- The faster disk reader: after the plan. It shortens a restore (production read 9.9 GiB in 10 s) and changes no promise.

**The nine questions, decided on 2026-10-13.** The owner delegated them to the two reviewers' common answers, which the coordinator recorded as the decision.

1. **No.** One conversation's read-ahead does not slow another user's reply beyond a defined budget of bounded pieces inside the pacing target.
2. **Yes.** With no spare room the read-ahead is skipped for that turn; the rows it would have prepared are new work, reported as such.
3. **Yes.** Where the tiers cannot hold every conversation the one of least worth loses, worth being the measured cost to recompute weighed by whether and how lately it came back; a new request's admission stays bounded whatever old cache is worth.
4. **Yes.** A history the host tier cannot keep is written to disk while serving, also without `--disk-cache-keep`, by the existing write machinery inside the background budget; a file is a holder only once whole.
5. **Deferred** to step 7's measurement, and only with a real timeout and fallback.
6. **Yes.** The cache's code moves into one file after step 1, a pure relocation with no behaviour or signature change and no new identity or index.
7. **Deferred.** The restart default changes only once a proof's correctness, coverage and cost at real history sizes are established; no silent adoption of unproved state.
8. **No.** An available restore is taken and a reusable history kept while the tiers have room; recomputing by choice is not built. If it is ever proposed it is for restores from disk only and goes back to the owner.
9. **Yes.** The next context window is prepared before a cut as step 10, opportunistic and bounded, merged only on the two-cut scenario's measurement against the same run with it off.

The questions as they were put, with the recommendations they were decided on:

| | Question | Developer | Gating reviewer | Returning reviewer |
|---|---|---|---|---|
| 1 | May the read-ahead of one conversation's reply ever slow another user's reply below the pacing target? | no | no | no beyond a defined budget of bounded pieces; the promise is over what the scheduler admits, not a file system that stalls |
| 2 | When the device has no spare room, may the read-ahead be skipped for that turn, so the next turn reads its own new rows (46 s in S1) rather than anything being evicted for it? | yes | yes; it is step 4's second half | acceptable, the rows it would have prepared being new work and reported as such, not a loss |
| 3 | When the tiers together cannot hold every conversation, is the one of least worth the one to lose, worth being what it would cost to recompute weighed by whether and how lately it came back? | yes | yes, by that weighting rather than age alone, admission and progress staying bounded apart from what is retained | yes to a measured worth ranking, with request admission and progress still bounded independently of retaining valuable old cache |
| 4 | May a history the host tier cannot keep be written to disk while the server serves, also without `--disk-cache-keep`? | yes | yes; a file is a holder only when whole | yes to early use of the existing write machinery; a durable holder only on complete publication |
| 5 | May an admission wait for a save, with a finite timeout, where that is measured cheaper than recomputing? | decide after step 7's measurement | the same | only with a real timeout and fallback; an estimate from bandwidth is not the bound |
| 6 | Is the move of the cache's code into one file wanted before the behaviour steps? | yes | yes, after step 1, as a pure move | yes to a pure relocation with no behaviour or signature change, and not a new conversation identity or index |
| 7 | Should a restart onto a build of other numerics prove and adopt the kept files by default, where today it needs `--disk-cache-adopt proved`? | decide with the measured cost of a proof at production's size | a default needs the proof's coverage judged sufficient and its cost measured, not its speed alone | defer the default until proof correctness, coverage and cost are established; speed alone does not justify accepting cache from changed numerics |
| 8 | When a history can be restored but recomputing it is measured cheaper, should the server recompute? If no, an available restore is always taken and a reusable history always kept while the tiers have room. If yes, the server may recompute instead and reports each such turn as chosen recomputation. | no, until the scenario shows a turn it would shorten | no by default; if ever accepted, only for restores from disk, a copy from the device or host memory never being dearer than recomputing its rows | no by default; allowed only as a separately reported decision the owner accepts |
| 9 | Should the server prepare the next context window ahead of a cut when it has spare room and time, knowing it costs background compute and extra cache and that a cut under saturation still pays its re-read? | yes, as the last step, measured against the same run with it off | not stated | yes, as step 10 of this plan, opportunistic and bounded |
