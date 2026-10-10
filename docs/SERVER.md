# The server (ROADMAP #7)

Design for the multi-user front-end: what it needs, how it drives the model
layer that [EXECUTION](EXECUTION.md) built for it, what it gates on, and
the order the pieces land in. It follows the same rules as everything else
here: no external libraries, a knob is a flag, nothing is claimed until it
is measured against the single-sequence path and the reference.

## What it needs

- **One model, many requests.** Weights are loaded once and shared read-only;
  every request owns a `Sequence`, its own history in the KV pool, and no
  request's reset or cancel touches another's blocks
  ([ARCHITECTURE](ARCHITECTURE.md), "KV state and concurrent execution").
- **Continuous batching.** A device saturates through batch size, so the
  scheduler forms passes carrying a share of ready decoding requests
  plus slices of prompts or histories being recomputed. On a pipelined
  split several passes can be in flight. The model's pass API
  (`Model::begin_pass`, [EXECUTION](EXECUTION.md)) takes that batch: a
  prefill microbatch is one entry with many tokens, a decode step is many
  entries with one token each, and the two mix in one pass.
- **One submitter per device.** A `Backend` is driven by one thread at a
  time. The scheduler is that thread, but for a tensor group's stage
  while several passes are in flight, which a thread of that stage
  records (The round, below); connection threads only queue
  requests and drain token streams. This is a contract, not a lock.
- **Streaming.** Tokens leave as they are sampled, over a plain HTTP/1.1
  response with chunked transfer or server-sent events, so a browser, curl
  or a client library reads them without a client of its own.
- **Prefix reuse without sharing mutable state.** A request whose prompt
  repeats the tokens of a finished request's history shares its full KV
  blocks read-only through `Model::fork` at the length of those blocks,
  which copies nothing, and prefills only what follows.
  Finished requests stay a while as donors; the donor with the longest
  run of matching full blocks is found by comparing tokens, not hashes,
  since a server holds at most `max_seqs` donors for one model
  ([KV-CACHE](KV-CACHE.md), "Prefix sharing"). A run is counted in the
  largest block of the model's storages (a device and the CPU in one
  placement can differ), which is whole in every storage because a model
  refuses block sizes that do not nest. A live request's growing history
  is never shared.
  Only rows computed in the classes the request computes them in are shared (`Model::row_class`, [SPECULATIVE](SPECULATIVE.md), section 1): at a first admission its prompt at the prompt's extent, so a follow-up turn shares the previous turn's prompt blocks and computes the previous reply, which decode computed, as rows of its own prompt, and every reply equals the CLI's for the same prompt.
- **A memory budget that admits, not crashes.** The KV pool holds `--ctx-size` tokens in total, the model context by default, or the most whole blocks the devices hold beside the weights, a pass's activations and the recurrent states where they cannot hold that, fitted and backed whole as the server loads (`fitted_kv`), so no pass grows the cache and an admitted request never meets a device out of memory; a model that leaves no room for one block is refused before the server listens.
  A capped request is admitted when the pool can hold its prompt and its `max_tokens`, an uncapped one (a compatible route without `max_tokens`) when it can hold its prompt and a growth step, reserving more as it generates; otherwise it waits in the queue, and past `--max-queue` queued requests a new one is refused with 503, paused requests not counted.
  A request's donor is chosen before room is made for it, and the other donors are evicted, oldest first, when it needs their blocks.
  If the pool is still short, its donor is consumed: the request forks it and the donor goes, so the blocks they share are reserved once rather than for each, and a follow-up turn keeps the history it repeats however many donors fill the pool.
  A request that shares every full block of its donor, a follow-up turn or a resume, consumes that donor before any other is evicted, since all the donor holds beyond what the request keeps is a partial last block; the other donors go only if that does not make room.
  A donor is consumed only when the pool is short, so while there is room it stays for other requests sharing its prefix.
  Nothing is evicted for a request that would still not fit: it waits with every donor in place.
- **Room by first admission.** One function, `make_room`, decides who gives up blocks for whom, and a request's place is its first admission, never renumbered; only its place among the paused changes, when it gives the pool's edge up (Turns at the pool's edge, below).
  A request takes donors first, oldest first; only a request that grows then pauses uncapped requests admitted after it, the latest first, each giving up its reservation and, if that is still short, the donor its history became.
  A capped request is never paused, and a request is never paused for its own growth: one that cannot grow sits out the pass with its cache as it is (a stall) and asks again before the next; what it lacks is held by capped requests or requests admitted before it, and the stall ends when one of them ends or is paused, or by an uncapped request admitted after it that a pass in flight holds, and the stall ends once that pass has retired.
  Paused requests wait apart from the queue, in order of first admission: they do not count against `--max-queue`, a client that leaves ends its paused request at the next round, its donor going with it when it holds less than a block, which no fork can share, and they hold nothing but an evictable donor.
  Each round with a free slot gives the growth steps that fall due their room before anything resumes or is admitted, so a newer request never takes the room an older request's step needs in the same pass.
  While a request is stalled nothing resumes or is admitted; paused requests resume in the order of their line, which is first admission but for one that gave the pool's edge up (below), and stop at the first that does not fit, and new requests are admitted only once none is paused.
  `/v1/health` counts the requests paused now, the passes requests sat out and those whose room waited on a request in flight, the tokens resumes recomputed and the resumes that took their donor back.
- **Turns at the pool's edge.** Uncapped requests that cannot all hold their room do not wait for one another's end: they take the room in turn (`Growth::turn`, `Growth::yields`, `Scheduler::yields`).
  When a running uncapped request's growth step falls due while an uncapped request is paused, and it has held the edge for a turn, it is paused in that one's place: its history is kept whole, it goes to the back of the line of the paused, and the one paused longest resumes, holds the edge for a turn of its own and yields the same way.
  A turn is counted in growth steps, not passes, because a step falls due on a whole block, where a model that keeps a state can have its history kept whole; a pause inside a block is recomputed.
  Its length comes from two times the scheduler measures: a change of turn copies one history to host memory and the other back, timed from the yield to the first pass retired that carries the request the turn went to, and a turn is the steps that make the last change `kTurnShare`, a tenth, of the turn (a constant: at histories of 14k tokens together 5 percent makes a turn 14 to 20 s, 10 percent 8 to 10 s and 20 percent 8 s, one step, so more buys nothing), by the time the running request took for its last step; never under one step, and one step until a change has been timed. The change's time is the smaller of the last two timed (`Growth::change`), so one slow change, the first one among them, which allocates the tier's slabs, does not lengthen the next turn, and two in a row do.
  On Qwen3.8-27B Q8_0 over a tensor group of two MI50s a change of turn between histories of 14k tokens together is 0.7 to 1.0 s and a step of 256 tokens is 8 s, so a turn is one or two steps, and a waiting stream moves again after 8 to 16 s where it waited 87 s for the other's end; at 200k tokens a change is about 10 s by the bytes and a turn about 100 s.
  The request that used to run undisturbed pays for it: there its longest wait for a token went from 2.1 s to 9.5 s and it ended 73 s later, 227 s against 154 s.
  A request yields only where its history can be kept whole meanwhile, in a host tier with room for it and, on a model that keeps a state, a free checkpoint slot, and only where the one waiting then fits; otherwise it keeps the edge as before, so a server without a host tier (`--host-cache-bytes 0`) behaves as it did. A server where turns were possible or asked for and will not happen says so in one line as it starts (`no_turns` in `src/server/api.hpp`): one whose host memory, given or by default, does not hold the longest history a request may have, with both sizes, or whose model keeps a state with no checkpoint slot. A server given a tier that holds a history takes turns wherever its caches are and says nothing, and so does one with its caches on the CPU at its default, where there is nothing to size.
  Each copy a request leaves in host memory at a pause supersedes the one its pause before left, a prefix nothing forks while the longer one exists, so two requests taking turns hold two copies there and no kept conversation of another user gives its place to their old ones.
  Capped requests are never paused, a request alone at the edge is untouched, and each request still ends at the limit with the text it gives alone. `/v1/health` counts the turns given up (`pressure.since_start.turns`).
- **An exact resume.** A request keeps its prompt and what it generated, never rewritten, and a record of how each stretch of its history was computed, its row classes: the extent each stretch took (`BatchEntry`), which chooses a device's kernels and, with experts streamed, whether a routed layer runs on the device.
  At its first admission that is the forked prefix as its donor recorded it, the rest of the prompt at the prompt's extent, and the generated tokens at extent 1; a donor keeps the classes of the history it holds.
  A resumed request whose own donor, the one its pause left, is still there because nothing evicted it takes that donor back whole, its partial last block included, and recomputes nothing: the rows it continues from are the ones it computed itself. Its history goes to that donor even when it holds less than a full block, which only it can take.
  Otherwise it forks only rows computed the way its own were, from a donor sharing part of its history, an identical request's or one holding a prefix it forked at its first admission, and recomputes what its cache lacks in the classes that first computed it, its generated tokens as entries of extent 1, which take the decode kernels however many rows they carry.
  So a paused request gives the logits, bit for bit, that it gives when never paused, on the CPU, on a device and on a layer split, and a request that never pauses runs exactly as before.
  The price of a recompute is time: recomputed generated tokens cost what decode rows cost, at most 64 a request a pass (`kReplayRows`), each counting as ubatch / 64 tokens of the pass's budget, at least one, so 8 of a 512-token slice, where a re-prefill as prompt rows cost less and changed the values.
- **Dependency-free transport.** HTTP/1.1 over BSD sockets and Winsock,
  request parsing, chunked responses, JSON in and out through
  `core/json.hpp`. No TLS: the server sits behind a reverse proxy when it
  faces a network, which is what every runtime's built-in server assumes.

## Structure

```
server/
  http.hpp       listen, accept, parse one request, write a response or a
                 chunked stream; blocking sockets, one thread per connection
  scheduler.hpp  the request queue, admission, batch assembly, the rounds
                 over the model's pass API, sampling, token channels
  policy.hpp     the policy core: the pools' blocks, the growth rule,
                 make_room, the round's stages, the decode share and the
                 logits rows, as free functions
  api.hpp        the routes and their JSON: /v1/generate, /v1/chat,
                 /v1/tokenize, /v1/detokenize, /v1/health, /v1/models,
                 /v1/chat/completions, /v1/completions
cli/main.cpp     `llmx serve <model.gguf> [--host H] [--port N] [--device D]
                 [--max-seqs N] [--max-queue N] [--ctx-size N] [--ubatch N]
                 [--cache-type-k T] [--cache-type-v T] [--threads N]
                 [--layer-shares A,B] [--n-cpu-moe N] [--cpu-moe]
                 [--moe-stream-from N] [--load-mode M] [--passes N] [--timing]`
```

`server/` sits above `inference/` in the layering: it uses the model, the tokenizer, the sampler and the chat template renderer, and adds scheduling and transport.
It does not drive `infer::generate`, which runs one sequence to its end: the scheduler advances requests through shared passes and has its own per-token end check (end of text, which a request's `ignore_eos` keeps out of the draw, any of its stop texts, its token limit), over the shared sampler (`infer::sample`, with the defaults and ranges of `infer::Sampling`) and `Tokenizer::is_eos`.
The directory is created with its first working route, not before.

### Threads

```
accept thread ---> connection thread (one per socket)
                     parse request, validate, tokenize, render the chat
                     template, enqueue a Request, then wait on the
                     request's token channel and write chunks until done,
                     looking at the socket every 100 ms meanwhile
scheduler thread   the only thread that calls the model, pushes to the
                   channels and keeps the ledger
stage threads      one for each tensor group's stage while several passes
                   are in flight: each records the one stage of the one
                   pass the round hands it, and decides nothing
sampling threads   up to four, the scheduler's own: they draw a retiring
                   pass's rows beside the scheduler thread and touch
                   nothing else
```

A `Request` carries the prompt ids, the sampling parameters, a `Sequence`, the stop conditions and a channel: a mutex, a condition variable and a deque of sampled tokens that the connection thread drains, each an id with, for a request that asks for log-probabilities, the logits row they come from or the values themselves (`Request::Token`, Log-probabilities below).
Cancellation is a flag the connection thread sets once its client has gone, which it learns in one of two ways: a write to the client fails, or a look at the socket finds the connection closed.
The thread waits on the channel for at most 100 ms at a time and looks at the socket whenever 100 ms have passed since its last look, token or not, so a departed client is noticed within about 100 ms wherever its request is: waiting in the queue, prefilling, building a whole reply or streaming.
The look takes no byte and never blocks: an end of stream or a reset from the client is gone, and nothing to read, data waiting or only urgent (out-of-band) data is present.
So a client that shuts only its sending side after the request is taken as gone, since that arrives as the same end of stream, and one that has sent bytes past its request is taken as present until a write to it fails.
A client taken as gone gets no answer, not even an error: its request is cancelled and the connection closes, so a client that shut only its sending side and still reads sees the end of the connection.
The scheduler sees the flag at its next round: a queued or paused request ends wherever it waits, and an active one not in flight ends there, its history kept as a donor as a finished request's is.
One in flight completes its pass and is not sampled, since no one reads its token, and ends once that pass has retired, its cache consistent, so its history is kept as a donor too; a reset returns its blocks only once the last pass that read them has retired.

### The round

The scheduler thread repeats a round over the model's pass API (`reserve_passes`, `begin_pass`, `run_pass_stage`, `pass_logits`, `end_pass`, `abort_pass`, [EXECUTION](EXECUTION.md)) in one context reserved at start for its passes in flight, each in a slot with its own handoff buffers, sized for at most one row per decoding request plus a ubatch of prompt or replay rows. Each new pass takes the ready decoders selected by `decode_share`, and each request wants one logits row at most.
On a pipelined layer split it keeps one pass in flight per stage, so every stage works on some pass while the host samples one and forms the next; elsewhere it keeps one.
`--passes N` sets another number, which a placement that is not pipelined refuses above one, and passes whose handoff buffers do not fit the memory are dropped at start, one at a time, with a line on stderr, never silently.
The policy the round follows is in `server/policy.hpp`, free functions over plain data that the `server-passes` CTest runs the round over with a simulated executor: `Pools` for what positions take in the pools' blocks, `Growth` for what a request reserves at admission and as it grows, `make_room` for room, `round_steps` for which stages a round records and which passes it retires, `decode_share` for how many decode entries a pass takes, `prompt_slice` for how many rows a prompt slice takes, and `logit_rows`, `take_rows` and `give_rows` for the logits rows a context reserves and where a pass's rows go.

```
round:
  collect: take the stages the stage threads finished: a recorded one
           counts, a failed one fails its pass alone
  drop:    end the queued and paused requests whose client left,
           wherever they wait, and look again as admission reaches each
  advance: from the last stage down to stage 1, each stage records the
           oldest pass waiting for it (run_pass_stage), so every device
           runs its passes in formation order and a pass advances a stage
           at most a round; a stage on the host, which computes as it is
           recorded, waits for the end of the round
           a tensor group's stage, whose every sum its members wait
           on each other for, is handed to its stage thread and takes
           no other pass until that one is collected
  retire:  every pass whose last stage an earlier round recorded, oldest
           first: pass_logits waits on that pass's own ticket, the
           sampling threads and the scheduler thread draw its wanting
           rows in place, each with its request's own sampler state, and
           the ids go to their channels in entry order, each, when the
           request asked for logprobs, with a copy of its logits row, or
           with the values themselves once its reader has fallen behind;
           end_pass releases the sequences from flight; return the pass's
           logits rows; stages already submitted and committed their
           histories; a request cancelled in flight is not
           sampled; finish on EOS, a stop string or max_tokens, and keep
           the history as a donor when it holds a full block, else release
  cancel:  the active requests whose client left and that no pass in
           flight holds end, kept as donors as a finished request is
  -- the rest waits for a free slot, and a request in flight is never
     paused, parked or given room
  grow:    an uncapped decoding request not in flight whose next token
           passes its reservation reserves another step, the earliest
           admitted first, with what make_room gives it: donors, then
           pausing uncapped requests admitted after it, latest first; if
           that is not enough, or it would pause a request in flight, it
           sits out this pass (a stall)
  admit:   unless a request sat out the pass, and while active <
           max_seqs: the paused requests in their line's order, then, once none
           is paused, the queue in order; for each, take a resumed
           request's own donor if it is still there, else find the donor
           sharing the longest run of full blocks (for a resumed request,
           of rows computed as its own were); if make_room finds room for
           the history plus max_tokens, or an uncapped request's history
           plus a growth step (the other donors, oldest first, then
           consuming that donor; its own donor, or a donor whose full
           blocks the request all shares, is consumed first), take it,
           take its own donor back whole, fork the donor at the shared
           blocks or make a fresh sequence; else stop, evicting nothing;
           the cache's length is all the progress there is
  form:    while a slot is free and a request has rows to add: one entry
           per decoding request that is neither in flight nor stalled,
           whose cache lacks only its last sampled id, up to the decode
           share, those that left flight earliest first; then for every
           other request not in flight a slice of the next stretch its
           cache lacks, at that stretch's extent, in order of first
           admission, each slice at most prompt_slice rows (for a
           request alone on a split, nothing else active, queued or
           paused after this round's admissions, about a 2 * stages-th
           of what it lacks, 128 at least, so the stages read it
           together; once company comes, the rows that bring what it
           read alone back to a whole ubatch, before any other prompt's
           slice and in a pass no other prompt shares), until the pass
           holds
           ubatch tokens, the generated
           tokens a resume recomputes at most 64 a request, each counting
           ubatch / 64, at least 1; the entry that
           ends a request's history wants logits, the others do not; the
           pass takes a run of logits rows (take_rows), begin_pass puts
           its sequences in flight, and its first stage is recorded, on
           the host at the end of the round
  host:    the stages on the host the round advanced, last stage first,
           then the new passes' first stages
  repeat while a pass is in flight or any request is active or paused;
  otherwise block on the queue
```

The decode share is the decoding requests, those in flight included, over the passes, rounded up, once the passes fill the stages; with fewer passes than stages every ready decoder goes.
So however the requests arrived, the passes in flight carry about as many decode rows each: four users on a two-stage split make two passes of two, where taking every ready decoder would leave them in whatever passes their prompts landed in.
A decoder the share holds back waits only while every slot is taken, and once the passes fill the stages a pass retires about every round, so it takes the next pass, ahead of the decoders that left flight after it.

A failed pass is abandoned in the model (`abort_pass`, which drains every device and returns each of its histories to where the pass found it), and its requests end with the error and give their blocks back; the other passes in flight go on, since their rows sit in their own storages, handoff buffers and logits rows.
A round the scheduler's own rules cannot go on with, a pass that came out empty with nothing in flight or a pass that found no logits rows, abandons every pass in flight and ends every active request with the error, rather than leave the loop to spin; the oldest request sits a pass out only while a capped request holds room, so with nothing in flight some active request always has rows to add.
Stopping abandons every pass in flight first and then ends everything, so the ledger and every pool are left at zero.
With one pass in flight a round that records a stage before the last one only relays, and the pass's logits are read the round after its last stage, which is the order `Model::forward` runs a pass in.

Prefill of a long prompt is chunked at `ubatch`, so a 16k prompt does not
stall the decoding requests for a whole pass: they advance one token a
pass while the prompt goes through in slices. That is the reason the
prompt slice comes after the decode entries and is bounded by what is left
of `ubatch`.
What `--ubatch` trades when serving, by placement, is measured in `docs/USAGE.md` (Physical batch): a decoder's token waits for the pass its row rides, so fewer prompt rows a pass shorten the longest gap and cost output; the scheduler has no rule of its own for it, a cap on the prompt rows of a pass that carries decoders having measured as a trade on tensor groups and a loss on a layer split.

Layer split phase 3's step 0 (`docs/STATUS-2026-09.md`) timed the host time between a pass's logits and the next pass on Qwen3-8B-Q8_0 on one MI50: 0.26 to 0.71 ms a row greedy and at the defaults at 1 to 32 sequences, nearly all of it sampling and about half of a greedy row the copy out of the mapped logits, which is 7 to 14 percent of a greedy pass at 8 to 32 sequences; the 25 microseconds an earlier timing build recorded did not hold.
For the same requests a second pass in flight would not hide that time, because the next pass's tokens come from this one, so one device keeps one pass in flight; on a split the passes in flight carry different requests, and one is sampled while the stages run the others.
Layer split phase 3's step 4 takes the copy out and shares the rest: each row is read in place from the pass's mapped logits, copied only for a request that asks for log-probabilities, and a pass's rows are drawn on the sampling threads beside the scheduler thread (Sampling, below).
The host also spends the recording of each pass, 0.7 milliseconds at one sequence and 1.7 at eight on Qwen3-0.6B-Q8_0 and 2.6 to 5.2 at one to 32 on the 8B, which no second pass of the same requests hides; only a recorded pass replayed with new inputs would, and that is a backend change noted in STATUS, not a scheduler one.

#### A generating request beside a prompt (`--generating-share`)

At share 0, the default, a generating request's row rides the pass that reads a prompt's rows and waits for that pass on every device, so beside a long prompt its tokens come a prompt pass apart.
With a share above 0, on one device or one tensor group (`Model::paces` with one stage; `check_share` refuses the flag by name elsewhere), the round forms two kinds of pass.
A pass apart holds generated rows only, in a second arena on every device (`ExecContext::apart`, reserved by `reserve_passes` and counted by the fit as `PlacementRequest::apart_rows`), with its own slots and its own logits rows, and `round_steps` keeps the kinds apart: a stage holds one pass of each kind at most, each kind in its formation order.
The prompt's pass is fed to its stage in pieces by the stage's thread (`record_pass_stage` with a `Pace`): a piece ends at each part on a single device, a layer's attention or its feed-forward block, and at each sum on a tensor group (`Collective::wait`).
Between two pieces that thread records a waiting pass apart on the same devices (`record_pass_within`) when its rows have arrived (`stage_ready`, which asks a ticket without waiting, `Backend::done`) and the stage's share says it is due.

The share is kept by each stage's thread as time owed (`StageShare` in `server/policy.hpp`): a piece earns the generated rows their share of its time, a pass of theirs spends the rest of its own, and they are due while nothing is owed back, so over any stretch they have their share of the stage and no more, however long a pass of theirs is against a piece.
A pass is priced as the least of the last eight a stage measured (`StageShare::price`): what the rows are owed stops at one pass at that price, so rows that arrive after a quiet stretch take two passes in a row at most, and no pass spends more than two, so one pass held up on the host costs them a bounded wait.
The time a stage stands still at a boundary for the scheduler thread's own device work (below) is not the prompt's either, so the generated rows spend it (`StageShare::held`), and what they owe for it stops at two passes, so a long copy to host memory does not keep them waiting long.
While generated rows come or one of their passes waits for its share, each piece is waited for before the next is fed, so a pass recorded at a boundary runs at once and is measured; once none has come for `StageShare::kQuiet` pieces and none waits, the device holds one piece ahead again and a prompt alone is read as without a share.
A pass recorded behind a piece held ahead is not measured and spends the price.

Generated rows still ride a prompt's pass where that costs them no more than their share would (`rides`): the prompt slice's time times the share is at most a pass of their own at that price, which one slow pass does not move.
The line the scheduler fits to its passes' times is not that price: under a share a pass apart retires only once it was due, so its time between retirements holds its wait.
While rows ride, no pass apart is measured, so after eight prompt passes ridden in a row one pass goes apart and is measured (`RideRun`): a ride entered on a price that no longer holds ends there, and one that pays costs a pass apart in nine.
So the short last slice of a prompt, or a short prompt, takes them along, and `/v1/health` counts those passes (`rode`) beside the passes apart and the ones recorded between pieces.
A pass's cost is counted from the last pass of its own kind, since passes apart retire past the others.

The scheduler thread's own work on a stage's devices, a mark, a retract, the drafter's chain, a history copied to or from host memory, waits for the stage's thread at a piece boundary and holds it there (`hold`, `Held`), not for the whole stage.
A request cancelled, a pass failed and a stop inside a piece end as they do without a share: the pass in the stage is abandoned once its thread has returned, and a pass apart recorded within it fails alone.

A layer split refuses a share: a pass fed in pieces stays in flight for its stage's device time and a sequence is in one pass at a time, so the prompt's next slice could not enter the first stage while its last is in the second, and the prompt would lose the overlap of the stages (docs/MULTI-DEVICE.md, Prefill beside decode).

`serve --timing` times the rounds for `/v1/health` over devices made to time their work, each dispatch between two timestamps (Protocol below): the round's period and the thread's time in it, recording, relaying (the uploads that carry a residual into its next stage), sampling and forming passes, apart from where it was held, on the source's ticket in `receive`, on staging, on a free command slot and on the logits; each stage's idle share; and the device-bound rate, the rows the passes carried over the busiest stage's device time.
A stage on a device is timed by its timestamps, read every 32 rounds, which waits for the device's queue, and a stage on the host by the thread's own time in it.
The timestamps and those readings slow serving, so throughput is read from a server without `--timing`, and the timed figures from one with it on the same load.

### Sampling

Sampling is per request, on the host, from the logits row the pass returns for that entry: the existing `inference/sampler.hpp` with the request's own temperature, top-k, top-p, penalty and seeded RNG, so a request with `seed` set is reproducible regardless of what it was batched with.
The sampler ranks tokens by score with a tie going to the lower id and takes no sum in an order its selection leaves, so a seeded request gives the tokens `generate` gives with the same settings and seed.
It reads the row in place in the pass's mapped logits, and the scheduler copies a row only for a request that asks for log-probabilities (below).
A retiring pass's rows are drawn on the scheduler's sampling threads (`core/job_threads.hpp`): four, or one fewer than the CPUs the process may use where that is fewer, beside the scheduler thread, which waits for the pass's logits, hands them the rows and waits for every draw before the pass ends and its logits rows come back.
Each draw touches only its request's generator and row copy, and a request wants one row a pass, so no draw depends on another, on the thread that takes it or on the order they run in; the scheduler thread then pushes the tokens in entry order, and it alone pushes to the channels, keeps the ledger and calls the backends.
A pass of one row is drawn on the scheduler thread without waking another.
A `top_k` of 0, which is what the compatible routes' -1 becomes, ranks only the best tokens, 64 at first and more as the nucleus `top_p` keeps needs them, and with `top_p` 1 ranks none.
A field the request leaves out takes the default of `infer::Sampling`, the one the CLI's flag starts from, except `max_tokens` on the compatible routes, where leaving it out means no cap; a value outside the range `infer::Sampling` gives the field is refused, as the CLI refuses it, but for the `top_k` of -1 that the compatible routes take as 0.
Greedy requests give the text the CLI gives for the same prompt, which is the first correctness gate below.
A request's `ignore_eos`, the CLI's `--ignore-eos`, is one rule of the sampler (`infer::Sampling::ignore_eos`): the id that ends a reply (`Tokenizer::is_eos`) is passed over by every sampling step, greedy included, and a draw leaves it out before top-k, top-p and the softmax, so the reply runs to its token limit, the context's for an uncapped request, while a stop text still ends it.
The masked token does not exist for the draw, so the repetition penalty cannot bring it back and top-k and top-p count only the other tokens.
The mask changes the draw, not the logits row: whatever reads the row reads the model's own distribution, the end token's share included.

### Drafts

With `--drafter lookup`, `--drafter embedded` or a file of MTP blocks beside the model (`--drafter PATH`, which drafts as `embedded` does) a decoding request drafts the tokens after its last pick and verifies them in the same pass as the other requests' rows (docs/SPECULATIVE.md, section 3), and every reply is the one it gives without drafts.
The pass being formed asks each decoding request it takes for drafts (`Scheduler::propose`): `spec::draft_length` gives the count from the request's acceptance, the tokens it may still generate, the positions its reservation holds and the decode columns the pass has left (`Scheduler::draft_columns`), the rows the device's decode kernels read each weight once for (`Backend::decode_columns`) less a row for each decoding request in the pass, every draft a lone decoder asks, at most 64 draft rows a pass, and none while every seat is taken and a request is queued for one. A request paused or queued for room takes no draft away: it waits for the running requests to end or grow, which their drafts bring sooner.
A request with drafts marks its history (`Model::mark`) and takes one entry, its last pick then the drafts, extent 1, every row's logits; one whose mark is refused, or whose proposer has nothing, takes a decode entry.
Within those, a pass drafts only what pays: the scheduler measures its passes of generated tokens, a base and a row in milliseconds, each by the time the server gave it, from its formation through its retract or, with passes in flight, since the pass before it retired, the drafter's chains apart, and those chains a step, a chain counted once for each pass in flight beside the one being formed, since they all wait behind it (`spec::PassTimes`), and a request drafts where the drafts it is expected to keep, its own share kept at each position weighed with the server's, pay for their rows at the rate the pass gives without drafts, at the depth whose tokens a millisecond are most (`spec::draft_depths`); until passes of different widths have been measured every request drafts what the columns allow, so the price is learned, and a server long at one width measures it again.
On Qwen3.6-27B Q8_0 on one MI50 a pass costs about 10 ms a row past a few rows, so the columns alone made four or more users drafting slower than without drafts (docs/STATUS-2026-09.md).
The drafts stay inside the room its reservation already holds, so they never grow the ledger, pause, stall or evict anything.
The pass asks for every request's drafts at once (`spec::Proposer::draft_all`): the embedded drafter's chains run as one batch on the head's device as the pass is formed (`Model::draft` over the requests' sequences), each step reading the drafter's weights and the head once for all of them, lookup's on the host.
The rows are drawn in order on the sampling pool as `infer::accept` takes them, each with the request's own settings, history and generator, stopping at the first pick that ends the request or differs from its draft, so the draws are those of the run without drafts; with log-probabilities each token's row is copied for the channel while the reader keeps up and its values computed once it has fallen behind (`Request::kRowsWaiting`), as for one row.
Right after the pass ends, before anything parks, pauses or forks the request, its history goes back to the last pick and the drafts its picks kept (`Model::retract`, which on a model that keeps a state reruns the kept rows from the mark), and a request cancelled in flight keeps its last pick alone.
Jobs, replays and resumes never draft; a resumed request drafts once it decodes again.
The model holds marks for up to as many requests as can draft in one pass (`server::draft_marks`, half the decode columns, at most `--max-seqs`, at least one): on a model that keeps a state each is a state slot and the saved rows of a verify of `--draft-max` + 1 rows, and drafting takes only the room the fit without it leaves: the KV budget and the state checkpoints are fitted without the embedded drafter or a mark, and the drafter and one mark must fit beside that budget, which stays as it is: automatic checkpoints give way to them, the most that still fit, which the start line counts, while a `--state-checkpoints` given by number does not, and where nothing makes room the server is refused with the numbers, a smaller `--ctx-size` leaving it; the other marks take only what is left (`PlacementRequest::fit_marks`); the requests drafting at once are at most the marks verifies in flight leave free (`Model::mark_slots`); on Qwen3.6-27B Q8_0 on one MI50 the drafter with eight marks counted before the budget had left 1536 of 8192 KV tokens and no checkpoint; the host tier copies an embedded drafter's carried row with the checkpoint slot it sits beside.
`/v1/health` counts by draft position the drafts verifies fed and those they kept.

### A prompt larger than the context

A prompt that does not fit what one request may hold (`Scheduler::token_limit`) with what it asks to generate is refused with 413, or, under `--context-overflow shift`, cut by the route before the scheduler sees it (`Api::fit` in `src/server/api.hpp`, over `src/server/context_cut.hpp`); the scheduler only ever takes a prompt that fits.

- **The index.** A cut point is a row offset after which a prompt may begin again, the row after a turn's end, found by one scan of the prompt's ids for the tokens the tokenizer's metadata marks as ends (`cut_points`); it is a row offset and not a text offset, so a part that is not text changes nothing in it. A rendered conversation has one a message; where the scan finds another number than there are messages, a template that ends turns otherwise or a user who typed the token's text, the chat routes count each message's end by rendering the conversation up to it (`Api::message_ends`).
- **The cut.** The leading block stays, the system message or a text prompt's rows up to its first cut point, and rows after it go up to the first cut point a whole number of steps past it, the fewest steps after which the prompt fits (`first_kept`, `cut_rows`). A step is half the limit (`kCutTo`). On the chat routes a kept window starts at a user message, so an assistant's tool calls stay with their results, and the kept messages are rendered again; a text prompt without a cut point that serves keeps its first `kCutLead` rows, 64 or half a step, and loses whole steps of rows.
- **Why steps from the start.** Counted from the prompt's start, the cut falls at the same message while a conversation grows within a step, so the kept window is read again once a step and reused on the turns between. Counted from its end, it would move every turn and every turn would read the window again.
- **What a cut costs.** The turn that cuts reads everything past the leading block again. No saved state serves it: a recurrent state at any row holds every row before it, and on a model without one the kept rows' positions and what they attend to both change; a message boundary's state is the state after the dropped messages. Only the leading block is forked, as any shared prefix is.
- **What the client sees.** A `context` object in the reply (`tokens`, `limit`, and after a cut `dropped_messages`, `dropped_tokens`, `at_marker`), and a line on the server.

### Log-probabilities

A request may ask for the log-probability of each token it is sent and for the most likely tokens at each position, at most 20.
Each value is the log-softmax of the logits row the sampler reads for that token: the model's own distribution, before the repetition penalty, the temperature, top-k and top-p, which is what the compatible APIs report.
So a token's value does not depend on how it was sampled, and a greedy token's is the largest at its position unless a repetition penalty moved it.
For a request that asks, the scheduler sends that row with the id on the request's token channel (`Request::Token`), and `Request::next` computes the values in the thread that reads the channel, with `inference/logprobs.hpp`, the functions perplexity scores with, in double and rounded once to float.
A row costs a pass of `exp` over the vocabulary, about a millisecond for Qwen3's 151936 tokens, so no pass of the batch waits for it, and the native tests read the values from the channel where the routes do.
Once 8 tokens wait on a channel with their rows (`Request::kRowsWaiting`), as when a client stops reading its stream, the pass's draws compute the next tokens' values from the rows in place, on the sampling threads or the scheduler thread, and the scheduler thread sends them in the row's place: the same values, and a request then holds at most ten rows however far its reader falls behind.
A row the reader has finished with goes back to the request for a later pass to fill, and a cancelled request's rows are dropped unread.
Every value is written as the shortest decimal that reads back as that float.
In the native shape a value JSON has no number for, minus infinity for a token given no probability, is `null`; the compatible shapes type the field as a number, so they write -9999 for any value below it, minus infinity included, and for a NaN.
A request that asks gets the ids and text it gets without them, and a request that does not gets no byte of them.

- `/v1/generate` and `/v1/chat` take `"logprobs": true` and `"top_logprobs": k`.
  A whole reply adds `"logprobs": [v, ...]` beside `ids` and, with `k` above 0, `"top_logprobs": [[{"id", "logprob"}, ...], ...]`, one list per token, most likely first.
  A streamed token's event adds `"logprob"` and, with `k` above 0, `"top_logprobs"`.
- `/v1/chat/completions` takes the same two fields.
  The choice's `logprobs` is `{"content": [{"token", "logprob", "bytes", "top_logprobs": [{"token", "logprob", "bytes"}, ...]}, ...], "refusal": null}`, and a streamed chunk carries its token's entry, a chunk without a token `null`.
- `/v1/completions` takes `"logprobs": k`, the count of most likely tokens to list.
  The choice's `logprobs` is `{"tokens", "token_logprobs", "top_logprobs", "text_offset"}`: `top_logprobs` maps each listed token's text to its value and holds the sampled token too, and `text_offset` is the characters of the reply's text before the character a token starts in.
- A token's text is its own when its bytes are whole UTF-8, and otherwise `bytes:` followed by each byte as `\xNN`, as the compatible APIs spell a token that splits a character; `bytes` carries them exactly.
- A `top_logprobs` or completions `logprobs` above 20, a `top_logprobs` above 0 without `logprobs` true, and a `logprobs` of another type are refused with 400; a null field is taken as absent.

They are what shows a difference in logits before a greedy token flips, which ids alone show only when a token changes.

### Protocol

```
POST /v1/generate    {"prompt": "...", "max_tokens": 64, "temperature": 0.8,
                      "top_k": 40, "top_p": 0.95, "seed": 0, "stop": ["..."],
                      "ignore_eos": false, "stream": true,
                      "logprobs": false, "top_logprobs": 0}
POST /v1/chat        {"messages": [{"role": "user", "content": "..."}], ...}
                     the model's chat template renders the prompt; an assistant message may carry "reasoning_content"
POST /v1/tokenize    {"text": "..."} or {"messages": [...]}
                     -> {"tokens": [ids], "count": n}
POST /v1/detokenize  {"tokens": [ids]} -> {"text": "..."}
GET  /v1/health      {"status": "ok", "server": {...}, "precision": {...},
                      "requests": {...}, "reuse": {...}, "pressure": {...},
                      "reread": {...}, "drafting": {...}, "passes": {...}}
                     each group has "now" (true at this moment) and
                      "since_start" (counters), units in the names;
                      with --timing also "timing": {"rounds", "round_ms",
                      "recording_ms", "relaying_ms", "sampling_ms",
                      "assembly_ms", "receive_wait_ms", "staging_wait_ms",
                      "open_wait_ms", "logits_wait_ms", "stage_idle_share": [..],
                      "device_bound_rows_per_s"}, each time a mean a round
GET  /v1/live        {"status": "ok"}, answered without the scheduler
GET  /v1/models      {"object": "list", "data": [{"id": "...", "object": "model", ...}]}
POST /v1/chat/completions   the OpenAI clients' shape over the same scheduler
POST /v1/completions        request: one parse, one request, one drain loop
```

The compatible routes exist so existing tools connect without a client of their own: they list `/v1/models`, send its `id` back as the model, and stream `/v1/chat/completions` as chunks with the role in the first delta, `finish_reason` in the last and `data: [DONE]` after.
They are a JSON mapping in the routes file over the scheduler the native routes use, with llmx's own knobs (`top_k`, `penalty`, `seed`, `ignore_eos`) accepted as extra fields and the synonyms the clients send (`max_completion_tokens`, `repetition_penalty`) beside them, validated before anything reaches the model, and they cost a request exactly what a native one costs.
What the shape cannot carry, token ids and the `eos` finish, stays on the native routes; the compatible replies carry the reused-prefix count as `timings.cache_n`.
A sampling field takes the range the CLI's flag for it takes, both read from beside the sampler's parameters, so the CLI and the server refuse the same values, with the compatible routes also accepting `top_k: -1` as no top-k, `seed: -1` as no explicit seed, and an omitted or -1 token limit as uncapped.

A streaming response is `text/event-stream`: one `data:` line per token with the id and the decoded text, and a final `data: [DONE]`.
The server holds a character split across tokens until its bytes complete, and replaces each byte that starts no valid UTF-8 character with U+FFFD, since JSON carries only characters.
The CLI writes each token's bytes as they come.
The `precision` object in `/v1/health` is the placement record passed at startup: `requested`, `declared`, `effective` and `devices`, each with `device`, `effective`, `how` and `paths`. A device may report F32 fallback while another implements the requested dtype. An entirely F32 run reports F32 as the overall effective dtype. The API serializes it without resolving precision or treating the listed paths as execution witnesses.

The model's name in `/v1/health`, `/v1/models` and every compatible reply is its file name, which on Linux can hold bytes that are not UTF-8, and an error message can carry text from outside the request, such as a chat template's from the model file or a backend's failure.
Both get the same U+FFFD repair as generated text, so every reply is UTF-8.
A non-streaming request gets one JSON object with the text, the ids and the counts.
Errors are JSON with an HTTP status: 400 for a bad request (a text the tokenizer cannot encode and a token id outside the vocabulary included), 413 for a prompt and its `max_tokens` past what one request may hold, unless `--context-overflow shift` cuts it as A prompt larger than the context, above, has it (the model context or the KV pool, whichever is smaller) or a body past 64 MiB, 503 when the queue is full.
A conversation the chat template raises on is a bad request, answered 400 with the template's message; a template the renderer refuses stops `serve` before it listens, since no chat request could be answered.
A model whose layers keep a recurrent state, a `qwen35` file, has a state only at the end of what it has read, so its prefix reuse goes through checkpoints, states kept at a position in slots of their own ([SPECULATIVE](SPECULATIVE.md), section 2): each request keeps one where its prompt's last whole block ends within what a follow-up turn would begin with, the conversation rendered without the generation prompt for a chat route and the whole prompt for a text, the slice that reaches it ending there.
A finished request's donor is its history back to that checkpoint, which a follow-up turn forks, reading the state in place; a paused request keeps its whole history as its checkpoint where a slot is free, taking it back on resuming, and its prompt's checkpoint otherwise.
A checkpoint takes a free slot, else the oldest donor holding one goes (`make_room`), else it is left out; nothing is paused for one.
`--state-checkpoints N` sets the slots, by default the fewer of `--max-seqs` and the most that take at most a quarter of the KV budget's room, and 0 keeps no checkpoint, so a follow-up turn recomputes its history and a paused request resumes from its start; each admitted request also holds one of the `--max-seqs` state slots the load reserves.
A donor the devices evict, to admit a request, to make a growth step's room or for a checkpoint slot, is first copied to host memory, its whole blocks up to its checkpoint on a model that keeps one ([SPECULATIVE](SPECULATIVE.md), section 2, Host tier), within `--host-cache-bytes`, by default what `--max-seqs` histories take at the most one request may hold, within half of the host's free memory once the model is loaded, and none where every cache is on the CPU, and leaving the host the reserve the fit keeps on it, the room going by the ranking stated once in [DISK-TIER](DISK-TIER.md) (What the host tier does now).
A donor whose conversation did not come back, its request having forked nothing the tiers kept, takes only the room that ranking gives it: users taking turns over more conversations than the tier holds would otherwise evict, each time, the one needed next.
A paused request's own donor counts as one whose conversation came back, whatever its request forked: the request resumes, and a copy the tier refused would be recomputed at that resume.
A request matching a copy in host memory better than any donor on the devices promotes it, taking its blocks and checkpoint slot as a first admission takes room and writing it back before its first pass, then forks it, so more conversations than the devices hold resume rather than recompute; the copies keep the bytes the storages held, so the reply is the one a fresh server gives.
The copies are enqueued on the devices' own streams between passes, into host memory the model keeps in 64 MiB slabs once it has first allocated them, and the scheduler thread does not wait for them: an enqueue takes about a millisecond, a slab's first allocation about 30 ms while the tier fills, and dropping a copy from host memory waits for its copies to retire, which they long have by then, while a device spends the copy's own time on its queue.
On a model that keeps a state, a conversation's message boundaries are kept in host memory too ([SPECULATIVE](SPECULATIVE.md), section 2, Host tier): as a chat reply's job completes, its state alone, at the start of the next user message, about 150 MiB on a 27B model, in the room the copies leave, by the same ranking, each conversation keeping four, its first and its newest among them, or with a disk tier every one, the older going to disk as the host tier's room takes them; a request that edits or regenerates an earlier turn forks the boundary before that turn's message with the blocks of a later history of the same conversation, on the devices or promoted from host memory, and reads only from there, and one that finds no boundary there keeps one where that message starts as it reads its prompt, so the next edit of the message forks it.
The slabs alive, idle or holding a copy, stay within `--host-cache-bytes`: before a copy allocates a slab, idle slabs on other devices are freed, and a copy they cannot make room for is not kept.
Under the host tier a disk tier keeps what host memory lets go ([DISK-TIER](DISK-TIER.md)), off unless `--disk-cache-bytes` gives it room: once the copies and boundaries not yet on disk fill three quarters of the host tier, the one the host tier would drop next is written to the server's own directory under `--disk-cache-dir` while it stays in host memory, bit for bit and only what its history lacks there, its new blocks as segments and its state, each file named by digests of its tokens and row classes, so room the host tier needs later releases entries already on disk at once and no request waits for a write.
A waiting request that a history on disk shares more with than anything in memory has its files read back into host memory, within the read bounds of [DISK-TIER](DISK-TIER.md) (Restore: disk to host to devices), keeps its place while later requests that fit go ahead, and is then promoted and forked as from the host tier; a read that fails its checksums deletes the entry and the request computes its history.
Until the model file's digest is known and the store's directory and write probe are done, which takes about twenty seconds for a 27 GB file not hashed before, the server serves without the disk tier, and `/v1/health` says so (`reuse.disk.now.ready`).
The disk keeps its room by deleting files from a history's end down, its earlier states and then conversations that did not come back first, deletes entries unused past `--disk-cache-max-age`, keeps `--disk-cache-floor` free on the file system, stops writing after a failed write until that room is back, and leaves nothing behind at a clean exit unless `--disk-cache-keep` has it write what memory holds for the next server of the same model and numerics fingerprint, which adopts it.
A reply's rows are decode rows, which a follow-up turn's prompt rows do not share, so a chat reply is read again as prompt rows for the next turn ([SPECULATIVE](SPECULATIVE.md), section 2, Idle re-prefill): the chat routes give the scheduler the ids the next turn begins with (`Scheduler::follow`), the conversation with the reply as a client of the route sends it back (`reply_sent_back`: the content alone on `/v1/chat/completions`, whose clients do not return `reasoning_content`, the whole text on `/v1/chat`), rendered as a following turn renders it (`chat::stable_prefix`), every 32 tokens as far as two continuations of the reply agree and, once it has ended, in full before the reply's last chunk goes out, so the scheduler has them before a client can answer.
The scheduler reads them in a job, an internal request whose prompt is those ids to their last whole block: it forks the history sharing most with them, the request itself while it runs and its donor once it has ended, reads the rest as prompt rows of the class every longer prompt takes, in slices ending on whole blocks, and becomes a donor beside the one it forked, so the follow-up turn forks past the reply while a regenerated reply, the prompt sent again, still forks the earlier one.
The job's donor supersedes the conversation's other donors, the one it forked, the request's and every one whose tokens its own begin with, such as the previous turn's job's: they go first when room is needed, the donor count included, before any donor of another conversation, and the devices evict them without copying them to host memory, so a conversation holds one full copy in each tier and the earlier ones stay, for a regenerated reply, only while room allows.
On a model that keeps a state a job keeps its state at each whole block it reaches, and a running job is a source like a donor, so a follow-up that comes before the job completes forks what it has read.
A job begun while its reply was written takes at most 64 rows of a pass's leftover budget, and one begun after it only passes formed while no request is active, in flight in another pass or not; it takes a seat and room as a first admission does, and gives both back at the next pass boundary when a waiting request cannot be admitted from free seats and blocks alone or a request cannot grow, starting again once nothing waits.
Jobs run only where rows are one class from some extent up to the token limit (`Model::row_class`), on a model that keeps a state only with checkpoint slots, and only for next turns at least that long.
A message that carries `reasoning_content` as a string is taken as sent, its content and its reasoning unchanged.
An assistant message without it, or with it null, is kept as `chat` keeps its own replies (`chat::ChatFormat::assistant`): split by `chat::assistant_turn`, the text after its last `</think>` the content and the reasoning before it `reasoning_content`, only when the template reads `reasoning_content` and does not split a turn at `</think>` itself, as the Qwen 3.8 templates do not, and rendered whole, as sent, under every other template.
So the server renders a conversation as `chat` does, and each model sees earlier reasoning in the form its template was written for, whichever way a client sends it back ([inference-chat](src/inference-chat.md) states the rule).
A reply goes the other way through `chat::ReplySplit`: on `/v1/chat/completions` the text inside the `<think>` the rendered prompt left open (`chat::opens_reasoning`) or the reply opened itself, up to the first `</think>`, is `reasoning_content` and the rest `content`, split as the tokens stream so every delta carries its part, and a reply without `<think>` is `content` byte for byte; the native `/v1/chat` keeps the text whole.
A request's `chat_template_kwargs` reach the template as variables (`chat::TemplateVar`), so a client sets `enable_thinking` as the templates read it.
A stream whose pass fails has already sent its 200 head, so it ends with one `data:` event holding the error in the route's error shape, without `data: [DONE]`.

`/v1/tokenize` and `/v1/detokenize` expose the model's tokenizer on the connection thread, without the scheduler, and give the ids and text `llmx tokenize` and `llmx detokenize` give.
A `text` is tokenized as it is, with no chat template, and the text of a special token such as `<|im_start|>` reads as that token, as it does in a prompt; `messages` in its place are rendered first as `/v1/chat` renders them, an assistant message's `reasoning_content` and the template's refusal of a conversation included, so the ids are the ones a chat request with those messages reads.
No start or end token is added, since no route adds one to a prompt, so `add_special`, which clients of other servers send true to count one, is not read, and the count is always what a generating route reads.
Neither route passes through the scheduler, so both answer while the queue is full, and a text past the context is counted rather than refused.
Detokenized text gets the U+FFFD repair of generated text, so the ids of a whole reply give back its text.

### What is not in the first version

- A prefix index that outlives the process, spans models or holds more
  than a few dozen donors: the hashed key of KV-CACHE is for that scale.
- Speculative decoding, grammars, logit bias, embeddings endpoints.
- TLS, authentication, rate limiting: the reverse proxy's job.

### Open gaps

None: forks across row classes, the last, closed with `Model::row_class` ([SPECULATIVE](SPECULATIVE.md), step 1), so a first admission forks only rows computed as it computes them.

## Gates

- **Correctness.** A greedy request through the server gives the same
  token ids as `llmx generate --temp 0` for the same prompt on the same
  backend, alone and while three other requests decode beside it; the
  kv-cache test already holds a two-entry `forward` to the entries run
  alone. A forked prefix continues exactly as a fresh sequence fed the
  same history, since a fork takes only rows of the request's own classes.
  A cancelled request returns its blocks and the others
  finish unchanged. All on the CPU and on the device.
  With log-probabilities, a request's values repeat from run to run, and a request run alone gets the values it gets while three others run beside it.
  A request paused and resumed gets the ids and every value it gets alone, where it never pauses (`server-resume`, the `server` component's uncapped checks, `tools/server_mix_check.py --uncapped`).
- **Serving performance.** The figures a serving runtime is judged by, measured by `tools/server_load.py` through streaming requests at 1, 4, 8, 16 and more concurrent requests of the same shape: time to first token and inter-token latency at the median and the 99th percentile, decoded tokens per second and requests per second.
  The tool also runs a sweep of Poisson request rates, prompts of an exact token length and replies of a fixed one, and reports time per output token, end-to-end latency and total tokens per second, the load and figures a reference serving benchmark reports.
  Against the reference runtime's server under the same load, same model, same card, both in the same minutes.
  The bar is not parity: the reference's server is the weaker of the serving runtimes at concurrency and the one that can run on this hardware, so llmx must beat it by a wide margin on every figure, and the margin is what is reported.
  A batch of one must not cost more than the CLI's decode.
- **Long prompts under load.** A 16k prompt admitted while eight requests
  decode: the decoders' per-token latency during its prefill is recorded,
  since chunked prefill is what bounds it.

## Order of work

| # | Step | Test |
|---|---|---|
| 1 | `server/http.hpp`: listen, accept, parse a request, write a response and a chunked stream, on Windows and Linux (**done**) | The `http` CTest starts the server on a port, sends requests with the runtime's own client code and checks the responses, including a split chunk boundary |
| 2 | `server/scheduler.hpp`: queue, admission by pool budget, batch assembly with chunked prefill, sampling per request, channels, cancellation; `llmx serve` with `/v1/generate` and `/v1/health` (**done**; the throughput gate is open) | Greedy equality with the CLI alone and beside three decoders, cancellation, refusals (the `server` Python component, synthetic and real, CPU and device); the throughput gate at 1, 4, 8, 16 measured with `tools/server_load.py`: 119, 89, 81 and 103 percent of the reference's server on the device, the shortfall at 4 and 8 being the per-view dispatches of the device's batched attention path |
| 3 | `/v1/chat` through the template renderer; `/v1/models` (**done**) | A chat turn through the server in the `server` component |
| 3a | One dispatch per layer for `kv_write`, `attention` and `norm_rope_kv` over every view of a batch (**done**: a view table per dispatch, `shaders/views.glsl`) | The backend-vulkan checks over two views and the device HF gate; the throughput gate again: 118, 112, 109 and 125 percent of the reference's server at 1, 4, 8 and 16: above it at every concurrency, short of the wide margin the serving gate above asks for |
| 4 | Prefix reuse: finished requests kept as donors, the longest shared run of full blocks forked on admission (**done**) | The `server` component: a prompt repeating a 247-token excerpt with a different ending reuses the first request's blocks and its greedy text equals the CLI's; a 1995-token prefix on Qwen3-0.6B-Q8_0 costs 1.53 s the first time and 0.16 s with a donor on the device (8B: 8.74 s to 0.72 s; CPU 0.6B: 6.95 s to 0.95 s) |
| 5 | The second execution context, if measured to help (**measured, not added**) | The host gap between passes is 0.26 to 0.71 ms a row on the 8B, 7 to 14 percent of a greedy pass at 8 to 32 sequences on one MI50 (layer split phase 3's step 0), and the next pass's tokens come from this one; see the round above |
| 6 | The compatible routes: `/v1/chat/completions`, `/v1/completions`, `/v1/models` in the OpenAI clients' shape (**done**) | The `server` component: greedy equality with the CLI through `/v1/completions` whole and streamed, usage counts, the role in the first chat chunk and the finish reason in the last, text content parts, the refusals' shape; CPU and device |
| 7 | `/v1/tokenize` and `/v1/detokenize`, and `messages` rendered by the chat template in place of a text (**done**) | The `server` component against `llmx tokenize` and `llmx detokenize` on the synthetic model and the Q8_0 fixture: text beyond ASCII, special tokens, an empty text, ids ending inside a character, a reply's ids giving back its text, the chat fixture's goldens under the file's template and a chat request reading the same count, the refusals |
| 8 | Log-probabilities on every generating route, in the compatible shapes and a native one (**done**) | The `logprobs` CTest: the log-softmax against a double-precision reference and the scheduler's channel against a second model's logits, read at once or left to fall behind; the `server` component: each route's shape whole and streamed, the ids unchanged, the values repeating byte for byte and equal alone and four at a time, greedy's token the most likely, and a reply that does not ask byte-identical to one that never names them |
| 9 | An exact resume: each request's row classes, a resume taking its own donor back whole when it survived, or forking only rows of its own classes and recomputing the rest in them (**done**) | `server-resume`: uncapped requests paused beside others give every id and value they give alone, on the CPU, a two-CPU split and a device, a follow-up turn's forked reply rows and a prefix of another extent included; a donor taken back recomputing nothing, a follow-up turn's among them, part of a history kept after its donor went, and a paused request cancelled; the `server` component's uncapped checks by value; `tools/server_mix_check.py --uncapped` checks the same by value, run before the merge on one MI50, a split over three MI50s and the Radeon VII (`docs/STATUS.md`) |
| 10 | Room by first admission: `make_room` as the one owner of who gives up blocks for whom, growth that stalls rather than pausing itself, paused requests apart from the queue (**done**) | A property test of `make_room`, now part of `server-passes`: random admissions, growth, pauses, cancellations and ends over two pools whose blocks are 4 and 8 or 8 and 16 tokens, the ledger adding up, the oldest request never refused room younger requests or donors hold, no empty pass while requests are active, every request ending; `server-resume` unchanged in its values, with its stall, waiting and fork-by-class cases on the scheduler itself |
| 11 | The rounds over the model's pass API with one pass in flight, and the policy core as free functions: the pools' blocks, the growth rule, `make_room`, the round's stages and the logits rows (layer split phase 3, step 2; **merged**) | `server-passes`: the policy core by hand, then the scheduler's round over it in random schedules of a simulated executor over 1 to 4 stages and 1 to 2S pass slots, with arrivals, growth, pauses, cancellations, failures and stops (`docs/STATUS-2026-09.md`, layer split phase 3); `server-resume`: a request cancelled and a stop from inside a pass's stage on one to three CPU stages; every reply byte-identical to the step before, alone, together and against the CLI (`server`, `server-resume`, `tools/server_mix_check.py`) |
| 12 | A pass in flight per stage on a pipelined split (`--passes`), the decode share, the host's stages after the devices', a cancelled request in flight not sampled and kept, a failure ending its own pass alone, `--timing` and the health fields, and a 16-slot command ring (layer split phase 3, step 3; **merged**) | `server-passes` over the new round; `server-passes-cpu`: the scheduler on one to three CPUs at P = 1, S, S + 1 and 2S, every reply its reply alone, pauses and a plan waiting on a request in flight, a cancel, a failure and a stop from inside a stage, and the passes replayed through `Model::forward`; `llmx-split-check`'s passes in flight; `tools/server_mix_check.py --passes --logprobs` (`docs/STATUS-2026-09.md`, layer split phase 3) |
| 13 | A retiring pass's rows read in place from its mapped logits, a row copied only for log-probabilities (layer split phase 3, step 4) | `server-passes-cpu`: the steady load without log-probabilities, drawn in place, gives every id it gives alone; every reply byte-identical to the step before, greedy and seeded, alone, together and against the CLI (`server`, `server-resume`, `logprobs`, `tools/server_mix_check.py --ids`, with `--sampled` for the seeded replies; `docs/STATUS-2026-09.md`, layer split phase 3) |
| 15 | A first admission forks only rows of the classes it computes them in (`Model::row_class`; [SPECULATIVE](SPECULATIVE.md), step 1) | `server-resume`: a follow-up turn, on the CPU and on a device, and a prompt sharing a block of a prompt of another tile split, each equal to the same prompt on a fresh model; `backend-group` and `backend-vulkan`: rows of every pair of extents of one class give the same bits |
| 14 | A retiring pass's rows drawn on the scheduler's sampling threads beside the scheduler thread (layer split phase 3, step 4) | `sampling-pool`: each index once, the threads side by side, an exception rethrown once every call has returned; TSan over the pool; every reply byte-identical to the step before, as for step 13 (`docs/STATUS-2026-09.md`, layer split phase 3) |

## The grouped `/v1/health` and help pages (2026-10-07)

`/v1/health` is one flat object of about 45 fields in the order they were added, and the `serve` help page is two long groups.
This plan regroups both for a person reading them, keeps every value, renames no flag and changes no default.
`docs/USAGE.md` (`/v1/health` and `/v1/live`) lists the fields as the server prints them; this section is the design, and the mapping from the flat names for anyone who read them.

### The shape

Every group that holds numbers splits them the same way: `now` for what describes the server at this moment, `since_start` for counters that only rise.
A name carries its unit where it has one: `_bytes`, `_tokens`, `_ms`, `_s`.

```json
{
  "status": "ok",
  "server": {"version": "0.9.0+abc1234", "numerics": "0123456789abcdef", "uptime_s": 8123,
             "model": "Qwen3-8B-Q8_0.gguf", "context_tokens": 40960, "devices": ["vulkan:0", "vulkan:1"]},
  "precision": {"requested": "auto", "declared": "f16", "effective": "f16", "devices": [{"device": "vulkan:0", "how": "...", "paths": "...", "effective": "f16"}]},
  "requests": {
    "now": {"active": 3, "queued": 0, "paused": 0},
    "limits": {"active": 16, "queued": 64},
    "since_start": {"finished": 912, "prompt_tokens": 481203, "generated_tokens": 90112}
  },
  "reuse": {
    "since_start": {"forks": 311, "tokens": 205112},
    "device": {"now": {"entries": 4, "state_checkpoints": 0}},
    "host": {"now": {"entries": 12, "bytes": 913047552, "limit_bytes": 17179869184},
             "since_start": {"promotions": 40, "bytes_moved": 4093640704}},
    "disk": {"now": {"entries": 90, "bytes": 8011472896, "limit_bytes": 214748364800, "in_flight": 0, "ready": true, "writing": true},
             "since_start": {"hits": 7, "bytes_read": 612368384, "bytes_written": 9100574720, "waits": 7, "wait_ms": 412,
                             "errors": 0, "dropped_for_cap": 0, "lost_before_written": 0}},
    "boundaries": {"now": {"entries": 14}, "since_start": {"hits": 21}}
  },
  "pressure": {"since_start": {"pauses": 0, "stalls": 0, "waits": 0, "recomputed_tokens": 0, "resumes_taking_history_back": 0}},
  "reread": {"since_start": {"jobs": 55, "rows": 31040, "cancelled": 3}},
  "drafting": {"since_start": {"drafted": 4096, "kept": 2780,
               "by_position": [{"position": 1, "drafted": 1024, "kept": 901}]}},
  "passes": {"limit": 2, "in_flight": 1, "sampling_threads": 3}
}
```

`timing` appears beside `passes` only under `--timing`, with the keys it has now and `stage_idle` renamed `stage_idle_share` (a fraction per stage).
`drafting.by_position` is empty with no drafter, and a server has one drafter, so the counts are by draft position and not by drafter.

### Where each old field goes

| Old | New | Source |
|---|---|---|
| `status` | `status` | constant |
| `model` | `server.model` | `Config::model_name` |
| `dtype` | `precision` | unchanged object, `Config::dtype` |
| `active`, `queued`, `paused` | `requests.now.active`, `.queued`, `.paused` | `Stats` |
| `donors` | `reuse.device.now.entries` | `Stats::donors` |
| `checkpoints` | `reuse.device.now.state_checkpoints` | `Stats::checkpoints` |
| `prefix_hits`, `prefix_tokens` | `reuse.since_start.forks`, `.tokens` | `Stats`; every admission that forked a shared history, whatever tier it came from, so they sit above the tiers |
| `host_donors`, `host_bytes` | `reuse.host.now.entries`, `.bytes` | `Stats` |
| `host_hits`, `host_bytes_moved` | `reuse.host.since_start.promotions`, `.bytes_moved` | `Stats` |
| `boundaries`, `boundary_hits` | `reuse.boundaries.now.entries`, `.since_start.hits` | `Stats` |
| `disk_entries`, `disk_bytes` | `reuse.disk.now.entries`, `.bytes` | `Stats` |
| `disk_ready`, `disk_writing` | `reuse.disk.now.ready`, `.writing` | `Stats` |
| `disk_hits`, `disk_bytes_read`, `disk_bytes_written` | `reuse.disk.since_start.hits`, `.bytes_read`, `.bytes_written` | `Stats` |
| `disk_waits`, `disk_wait_ms`, `disk_errors` | `reuse.disk.since_start.waits`, `.wait_ms`, `.errors` | `Stats` |
| `disk_capped` | `reuse.disk.since_start.dropped_for_cap` | `Stats` |
| `host_unwritten` | `reuse.disk.since_start.lost_before_written` | `Stats` |
| `pauses`, `stalls`, `waits`, `recomputed`, `taken_back` | `pressure.since_start.pauses`, `.stalls`, `.waits`, `.recomputed_tokens`, `.resumes_taking_history_back` | `Stats` |
| `reprefills`, `reprefill_rows`, `reprefill_cancels` | `reread.since_start.jobs`, `.rows`, `.cancelled` | `Stats` |
| `drafted`, `kept` | `drafting.since_start.by_position[]` and their sums | `Stats` |
| `passes`, `in_flight` | `passes.limit`, `passes.in_flight` | `Stats` |
| `timing` | `timing` | `Stats::timing` |

New, with where each comes from:

| New | Source |
|---|---|
| `server.version`, `server.numerics` | `LLMX_VERSION_STRING` and the first 16 characters of `LLMX_NUMERICS`, as `--version` prints them |
| `server.uptime_s` | seconds since the `Api` was made (`started_` is already kept) |
| `server.context_tokens` | `Model::context_length()`, as `/v1/models` gives it |
| `server.devices` | the names in `Config::dtype.devices` |
| `requests.limits.*` | `Config::max_seqs`, `Config::max_queue` |
| `requests.since_start.*` | three atomic counters in the scheduler, added where a client's request that was admitted ends: requests finished, prompt tokens, generated tokens; a re-read job (`reread`) is not counted |
| `reuse.host.now.limit_bytes` | the scheduler's `host_cap_` |
| `reuse.disk.now.limit_bytes` | the disk tier's `cap()` |
| `passes.sampling_threads` | `Stats::samplers`, already kept and not printed |
| `reuse.disk.now.in_flight` | `Stats::disk_in_flight`, already kept and not printed |

The new counters are plain reads and atomic increments; nothing in the scheduler's logic changes.

### Liveness

Chosen: a second route, `GET /v1/live`, answered from the HTTP layer without touching the scheduler, for a load balancer that polls often.
It returns `{"status": "ok"}` and nothing else; `active`, `paused` and `in_flight`, which the scheduler keeps in atomics (`active_count_`, `paused_count_`, `in_flight_`) that need no lock, could be added if a balancer wants more than a constant.
The ground for a route of its own is that a probe that must not wait on the scheduler should not share a path with one that does.
`/v1/health` takes the scheduler's lock, which the scheduler holds in admission and eviction, where it waits for the recorders to be quiet, and in the disk flush, so a health answer can wait up to a stage's time and a poll with a short timeout can fail on a healthy loaded server (`docs/src/server.md`).
`/v1/live` says the process and its listener are up and cannot tell a wedged scheduler; a monitor that must tell that polls `/v1/health` with a timeout longer than a stage.
The alternative is `GET /v1/health?brief`, one route and no new name, which returns only `status` before taking the lock; the HTTP layer already splits the query off the target, so it costs a comparison, but it shares a path with the call that waits.

### The help pages

The `serve` page is regrouped into Server, Limits, Prefix cache (the host and disk flags together), Speculative decoding and Execution, each flag one or two short lines with its default last, and three examples: one GPU, two GPUs with the disk cache kept, and drafting.
Internal words get a plain rendering that is still true: "passes in flight" becomes "batches the devices work on at once", "states a recurrent model keeps" becomes "saved conversation states for models with recurrent layers, so a follow-up turn skips re-reading its prompt", "routed layers" becomes "mixture-of-experts layers".
The `generate`, `chat` and `bench` pages keep their groups and gain the same shape where a group runs past about ten lines (Generation, Sampling, Drafting, Execution).
No flag is renamed, added or removed and no default changes; `tests/cli.py` reads every flag of every page, so a flag lost from a page fails it.

### What breaks

The flat names go in one change with their readers: `tools/server_load.py`, `tools/server_mix_check.py`, `tests/server.py`, `tests/qwen35.py`, `tests/drafters.py`, `tests/server_mix_tool.py`, `tests/common.py` (it reads only `status`) and the fields named in `docs/USAGE.md`, `docs/DISK-TIER.md`, `docs/SPECULATIVE.md` and this file.
There is no copy of the old flat fields beside the new ones: two shapes would be two things to keep true, and the readers are all in this tree.
Recorded benchmark scripts under `docs/benchmarks/` read the old shape and are history.
A person or script outside the tree reading the old names must move, among them the operator's own readiness polls (`disk_ready` and `disk_writing` tell a restart that the disk tier is ready), which move in the same change; the release notes say so.
`tests/server_resume.cpp` reads the `Stats` struct and not the JSON, so it is unaffected while the struct keeps its fields.
