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
- **A memory budget that admits, not crashes.** The KV pool holds `--kv-tokens` tokens in total, by default every whole block the devices' free memory holds beside everything else (the model's context on the CPU, or where a device's free memory is the host's or unknown), or the most whole blocks the devices hold beside the weights, a pass's activations and the recurrent states where they cannot hold a given one; a request holds at most the smallest of `--ctx-size`, the model's context and the pool (`Scheduler::request_limit`), fitted and backed whole as the server loads (`fitted_kv`), so no pass grows the cache and an admitted request never meets a device out of memory; a model that leaves no room for one block is refused before the server serves.
  A capped request is admitted when the pool can hold its prompt and its `max_tokens`, an uncapped one (a compatible route without `max_tokens`) when it can hold its prompt and a growth step, reserving more as it generates; otherwise it waits in the queue, and past `--max-queue` queued requests a new one is refused with 503, paused requests not counted.
  A request's donor is chosen before room is made for it, and the other donors are evicted, oldest first, when it needs their blocks.
  If the pool is still short, its donor is consumed: the request forks it and the donor goes, so the blocks they share are reserved once rather than for each, and a follow-up turn keeps the history it repeats however many donors fill the pool.
  A request that shares every full block of its donor, a follow-up turn or a resume, consumes that donor before any other is evicted, since all the donor holds beyond what the request keeps is a partial last block; the other donors go only if that does not make room.
  A donor is consumed only when the pool is short, so while there is room it stays for other requests sharing its prefix.
  Nothing is evicted for a request that would still not fit: it waits with every donor in place.
- **Room by first admission.** One function, `make_room`, decides who gives up blocks for whom, and a request's place is its first admission, never renumbered. Room is counted once, from the model's own block references: what is free is each pool's free blocks less what the running requests promised and have not taken, and what victims free is asked of the model for the set together, so a fork beside the donor it shares needs only its own blocks and two forks of one prefix free it only together; only its place among the paused changes, when it gives the pool's edge up (Turns at the pool's edge, below).
  A request takes donors first, oldest first; only a request that grows then pauses uncapped requests admitted after it, the latest first, each giving up its reservation and, if that is still short, the donor its history became.
  A capped request is never paused, and a request is never paused for its own growth: one that cannot grow sits out the pass with its cache as it is (a stall) and asks again before the next; what it lacks is held by capped requests or requests admitted before it, and the stall ends when one of them ends or is paused, or by an uncapped request admitted after it that a pass in flight holds, and the stall ends once that pass has retired.
  Paused requests wait apart from the queue, in order of first admission: they do not count against `--max-queue`, a client that leaves ends its paused request at the next round, its donor going with it when it holds less than a block, which no fork can share, and they hold nothing but an evictable donor.
  Each round with a free slot gives the growth steps that fall due their room before anything resumes or is admitted, so a newer request never takes the room an older request's step needs in the same pass.
  While a request is stalled nothing resumes or is admitted; everything that waits stands in one line (`admit_waiting`): the requests paused for another's growth first, oldest first, then those that gave the pool's edge up (below) and the queued ones merged by the number each took when it began to wait, a yielder at its yield and a queued request at its submission; admission walks it and stops at the first that does not fit. So without turns the order is what it was, the paused and then the queue, and with them a queued request is behind only what waited before it.
  `/v1/health` counts the requests paused now, the passes requests sat out and those whose room waited on a request in flight, the tokens resumes recomputed and the resumes that took their donor back.
- **Turns at the pool's edge.** Uncapped requests that cannot all hold their room do not wait for one another's end: they take the room in turn (`Growth::turn`, `Growth::yields`, `Scheduler::yields`).
  When a running uncapped request's growth step falls due while an uncapped request is paused, and it has held the edge for a turn, it is paused in that one's place: its history is kept whole, it goes to the back of the line of the paused, and the one paused longest resumes, holds the edge for a turn of its own and yields the same way.
  A turn is counted in growth steps, not passes, because a step falls due on a whole block, where a model that keeps a state can have its history kept whole; a pause inside a block is recomputed.
  Its length comes from two times the scheduler measures: a change of turn copies one history to host memory and the other back, timed from the yield to the first pass retired that carries the request the turn went to, and a turn is the steps that make the last change `kTurnShare`, a tenth, of the turn (a constant: at histories of 14k tokens together 5 percent makes a turn 14 to 20 s, 10 percent 8 to 10 s and 20 percent 8 s, one step, so more buys nothing), by the time the running request took for its last step; never under one step, and one step until a change has been timed. The change's time is the smaller of the last two timed (`Growth::change`), so one slow change, the first one among them, which allocates the tier's slabs, does not lengthen the next turn, and two in a row do.
  On Qwen3.8-27B Q8_0 over a tensor group of two MI50s a change of turn between histories of 14k tokens together is 0.7 to 1.0 s and a step of 256 tokens is 8 s, so a turn is one or two steps, and a waiting stream moves again after 8 to 16 s where it waited 87 s for the other's end; at 200k tokens a change is about 10 s by the bytes and a turn about 100 s.
  The request that used to run undisturbed pays for it: there its longest wait for a token went from 2.1 s to 9.5 s and it ended 73 s later, 227 s against 154 s.
  A request yields only where its history can be kept whole meanwhile: the copy it would leave is one the host tier keeps now, by the rule that makes the copy (`host_room`: the tier's room it may give and the host's memory for the slabs it takes anew), and, on a model that keeps a state, a free checkpoint slot, and only where the one waiting then fits; otherwise it keeps the edge as before, so a server without a host tier (`--host-cache-bytes 0`) behaves as it did. A server where turns were possible or asked for and will not happen says so in one line as it starts (`no_turns` in `src/server/api.hpp`): one whose host memory, given or by default, does not hold the longest history a request may have, with both sizes and, where it holds some, the history length up to which a request still yields (longer ones keep the edge, shorter ones take turns), or whose model keeps a state with no checkpoint slot. A server given a tier that holds a history takes turns wherever its caches are and says nothing, and so does one with its caches on the CPU at its default, where there is nothing to size.
  Each copy a request leaves in host memory at a pause supersedes the one its pause before left, a prefix nothing forks while the longer one exists, so two requests taking turns hold two copies there and no kept conversation of another user gives its place to their old ones.
  A request that arrives while two take turns stands in the same line as they do, behind the one then paused and ahead of the next to yield, so it is admitted at the next change of turn if it fits beside the request that takes the edge, and no stream of arrivals passes a paused request, each arrival standing behind it; what it reserves comes out of the room the two rotate in, and where that leaves the waiting one no room the holder keeps the edge until the capped request ends; a queued request with a smaller number than the turn's receiver is admitted first, and where it did not fit beside the holder and fits once the holder has yielded it takes the freed room and both edge requests wait until it ends, which a capped request's limit bounds; an uncapped request admitted ahead of the turn's receiver takes its turn in the rotation.
  Capped requests are never paused, a request alone at the edge is untouched, and each request still ends at the limit with the text it gives alone. `/v1/health` counts the turns given up (`pressure.since_start.turns`) and those kept because the host had no memory for the copy (`.turns_without_memory`), whose run the log says as it starts and as it ends; a turn given up while the copy could not be made would lose the history it leaves, so every change of turn would compute the other request's history again (production, 2026-10-17: a reply paused 20 times, nothing taken back, 67456 tokens recomputed).
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
  cache.hpp      the cache behind one owner the scheduler calls: device
                 donors, message boundaries, host entries, the disk
                 tier's side, the restores, parking and the jobs
  policy.hpp     the policy core: the pools' blocks, the growth rule,
                 make_room, the round's stages, the decode share and the
                 logits rows, as free functions
  api.hpp        the routes and their JSON: /v1/generate, /v1/chat,
                 /v1/tokenize, /v1/detokenize, /v1/health, /v1/models,
                 /v1/chat/completions, /v1/completions
cli/main.cpp     `llmx serve <model.gguf> [--host H] [--port N] [--device D]
                 [--max-seqs N] [--max-queue N] [--ctx-size N] [--kv-tokens N] [--ubatch N]
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
                   channels and counts the room
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
On a pipelined layer split it reserves a pass slot per stage and one more and keeps one pass in flight per stage, so every stage works on some pass while the host samples one and forms the next; the one more is used only while prompt rows and replies overlap, so a prompt's next chunk continues while a pass of replies and its chunk before are in flight (`passes_in_flight`, The service rule, below); elsewhere it keeps one. More slots (`--passes`) add no chunk passes: while both wait the round keeps the replies' passes and one chunk pass a stage in flight, so a reply's step meets one chunk at every stage, as the chunk budget plans; at `--passes` equal to the stages one chunk pass is left, and a stage idles half a step, since a prompt in flight continues only in passes holding it alone.
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
           max_seqs: what waits in its one line, the paused for another's
           growth, then yielders and the queue by number; for each, take a resumed
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
  form:    while a slot is free and a request has rows to add, by the
           service rule (below): with only replies or only prompt work,
           one entry per decoding request that is neither in flight nor
           stalled, whose cache lacks only its last sampled id, up to the
           decode share, those that left flight earliest first, then a
           slice of the next stretch each prompt's cache lacks, at that
           stretch's extent, in the share rule's order, until the pass
           holds ubatch tokens (a lone prompt on a split about a 2S-th of
           what it lacks, at least 128 rows, so every stage reads;
           lone_slice), the generated tokens a resume recomputes
           at most 64 a request, each counting ubatch / 64, at least 1;
           while both wait, on a split a pass of the replies packed by
           the decode columns or a chunk of one prompt, on one pass in
           flight the replies and one prompt's chunk; a prompt in flight
           continues only in a pass of its own; the entry that ends a
           request's history wants logits, the others do not; the pass
           takes a run of logits rows (take_rows), begin_pass puts its
           sequences in flight, and its first stage is recorded, on the
           host at the end of the round
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
Stopping abandons every pass in flight first and then ends everything, so every pool is left with nothing taken or promised.
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

`serve --timing` times the rounds for `/v1/health` over devices made to time their work, each dispatch between two timestamps (Protocol below): the round's period and the thread's time in it, recording, relaying (the uploads that carry a residual into its next stage), sampling, forming passes, the drafters' chains as completed work (`drafting_ms`) and the verifies' retracts (`settling_ms`), apart from where it was held, on the source's ticket in `receive`, on staging, on a free command slot and on the logits; each stage's idle share; and the device-bound rate, the rows the passes carried over the busiest stage's device time.
A stage on a device is timed by its timestamps, read every 32 rounds, which waits for the device's queue, and a stage on the host by the thread's own time in it.
The timestamps and those readings slow serving, so throughput is read from a server without `--timing`, and the timed figures from one with it on the same load.

### The price table

Every pass's every stage is timed as its devices ran it, at no cost to the round (`ServiceRecord` in `server/service_record.hpp`).
The stage that records it posts the submission that ends it (`Model::stage_sent`) with when its recording began, and a thread of the record's own asks that submission whether it has retired, every millisecond while one waits and not at all otherwise (`Backend::done`).
A stage's time runs from the latest of its recording's start, the end of the stage's previous pass and the end of the pass's previous stage, to its own end: the time a pass behind it waits for it.
On a stage on the host, which computes as it records, that is its recording.
The time is split into the host's recording of the stage, of it the waits for a lookup table's rows (`WeightRows::finish`) and in a tensor group's sums, which the model keeps for the pass's last stage recorded (`Model::stage_host_waits`), and the devices' time after the recording ended.

Each stage keeps a table of measured prices (`PriceTable` in `server/policy.hpp`), keyed by what a pass holds (`PassWork`): rows of extent 1 (decode entries, verifies and a resume's recomputed tokens) by their count and the depth band of the deepest, and prompt slices by their row class (`Model::row_class` of the prompt's extent, which picks the tiles and their split), depth band and rows in steps of 16.
A depth band is a doubling of the history from 4096 tokens (`depth_band`): band 0 below 4096, band 6 from 128k to 256k.
A pass of decode rows alone prices its decode cell and a pass of one slice its slice's cell, each cell keeping a forgetting mean and its last measure, a decode cell's mean starting at its second measure, since its first pays its width's one-time costs such as kernel builds made on first use, so `/v1/health` shows a decode cell measured once with its last measure and a mean (`ms`) of 0 until its second measure, which is no free pass; beside decode rows the slice takes what remains of the pass once their price is taken off, so a prompt riding the replies' pass learns its price, and other passes of several parts price nothing and only measure the prediction.
A pass is predicted as its parts' prices added, each measured as a pass of its own, so a pass of several parts is predicted high by their fixed costs; rows not measured take the nearest measured rows of their class and band, a slice the next above it or the largest below scaled by its rows.
Decode rows in a band not yet measured take the nearest band below, so a reply crossing into a deeper band, as one does every doubling of its history, is still priced and no whole-ubatch chunk passes beside it.

The table's search gives the rows of a slice that stays within a stage time (`PriceTable::chunk_rows`): twice the largest slice measured within it while nothing above that is over, the bracket's middle in steps of 16 while it is wider than a step, and half the smallest over where nothing is within, or its rows scaled to the time where that is less, as for a band measured only at a whole ubatch read alone, each by the cells' last measures, so an overrun acts at once.
A band not measured starts at the largest slice within of the nearest band below that has one, and a class's first slice is one step.
The service rule sizes its chunks by it (The service rule, below), and `/v1/health` shows it under `service` beside each stage's measures (docs/USAGE.md, `/v1/health` and `/v1/live`).
The record also counts the rows passes carried by kind and the prompt rows by depth band, and each request's wait for admission, its time from reading to its first token and the gaps between its tokens, a step's gap spread over the tokens it published; the health gives their percentiles over the last 1024 of each, and each live request its own.

### The service rule

One rule forms the passes on every placement (`Scheduler::form_round` over `Scheduler::plan`), replacing the pacing modes, the pieces, the passes apart and the short prompts it grew from (docs/STATUS.md, the serving rework).
While only replies wait, or only prompts, passes are formed as the round above says, the decode share over the passes in flight and the prompts' slices in the share rule's order up to the ubatch.
While a prompt also waits, on a split the replies go in as few passes as the decode kernels' columns hold them, never every slot, each reply in one of them (`packed_passes`, `packed_share`), and each other free slot takes a chunk pass of one prompt, which continues in its own pass once its chunk before has run the first stage, up to one chunk pass a stage (`passes_in_flight`); a slot no prompt can use goes to the replies.
On one pass in flight, one device or one tensor group, the chunk rides the replies' pass as a second class.
Prompt work is the clients' prompts, and with none waiting the read-again jobs', which so take only what no prompt wants.
A request at an image without rows spends its prompt turn on a piece of that image's encode (docs/VISION.md, section 10), and beside replies a replies' pass meets as many pieces as the chunk budget holds at a piece's measured time, one at least (`encode_pieces`); a piece's time is what a replies' pass that waited behind pieces took beyond its predicted time, or the scheduler's own time running them where the first stage records on its thread, whichever is longer, shared among them.

While prompt work waits, the rule aims to keep every reply at `kReplyTokensPerSecond`, 8 tokens a second, a product objective and not a guaranteed rate: about twice a reader's silent reading speed, the same on any model and placement, where only the stages' measured prices differ and the stage count is a parameter.
The gap between a reply's steps is the largest that holds the objective: 1000 ms over 8 times the fewest tokens a step among the replies, each reply's tokens a step the mean of its last 8 steps (`StepTokens`), a verify's kept drafts and its pick, and 1 until it has 8, so one lucky verify does not widen the gap.
The gap is at most `kMaxStepGapMs`, 500 ms, however many tokens a step brings, so a reply whose drafts keep many tokens a step is still delivered without long pauses; it is a reader's property too, not the hardware's.
A reply's pass meets one chunk at every stage, so its gap is the stages times a chunk and its own pass: a chunk takes at most the gap over the stages less the replies' pass (`chunk_budget`, `chunk_plan`), and its rows are those the price table gives within that (`PriceTable::chunk_rows`, the slowest stage's).
Where the replies' pass alone leaves nothing, the replies cannot reach the objective; the health counts the pass as unmet, and a prompt then takes a chunk of one stage's share of the gap only once it has gone `kPromptWaitMs` without rows, so it still reads at least a chunk each bound.
While the objective is unmet, the shortest prompt whose remaining rows fit one pass and whose whole remaining cost and the replies' pass fit within the step gap's cap by the measured prices is quick (`quick_prompt`): it runs whole in a pass shared with the replies, every other replies' pass at most, so a newcomer's short prompt never waits for the bound and every second replies' pass stays the replies' alone; while its price is not yet known it rides as the 16-row learning step, and prices come from `PriceTable::slice_ms`, a line through the measured slices rather than a scaling that inflates a slice's fixed cost.
Until the replies' pass has a measured price, a chunk is one small step (`kPieceStep` rows), and on one pass in flight the replies' pass goes alone first, so it is measured before a chunk rides it; a class's first slice in a new band starts small (The price table, above).
`/v1/health` shows the objective, the step gap's maximum, the last gap chosen and the passes formed with it met, unmet and capped by that maximum under `service.objective`.

The disk's work takes only what nothing waits on (`Scheduler::write_turn`): a turn-end write beyond the host tier goes whole while nothing runs or waits, a piece's time at a time beside prompts alone, and beside replies only for a donor a request waits for; a donor whose write starved so is lost when it goes, as the health's moves count.

### Prompt shares

A share rule divides prompt time among the ready prompts, which `next_prompt` picks by from what each has left at the table's prices, how long it has waited without rows and the time it has had (`ReadyPrompt`, `Share`).
The server reads the least priced work left first, and any prompt without rows for `kPromptWaitMs` (10 s) first of all, the earliest admitted of equals, the rule the replays below chose (docs/STATUS.md, the serving rework).
An ordering rule takes the first by its order: today's order (a prompt of at most a quarter of a ubatch left first, else the oldest), the oldest first, the least priced work left first with any prompt idle for a bound first and waiting weighed against work, or the least time had first.
A dividing rule gives each prompt a part of the time (the oldest a fixed part and the next in arrival order the rest, equal parts, or the oldest a floor and the least remaining of the others the rest) as credit over the chunks given (`charge_share`), the next chunk going to the largest credit among the prompts given a part.

The rule is chosen by replaying production's requests and synthetic loads through the scheduler's own policy (`llmx-server-passes-test --replay FILE` or `--load NAME`, `tests/server_passes.cpp`).
The replay runs the simulated round of `server-passes` with each stage's time at a price model (`Prices`: a replies' pass, a chunk pass's fixed time, a prompt row by its depth and a further row class), the passes formed by the service rule with each candidate share, the prices it decides by measured from its own stages into a `PriceTable` per stage, and the ledger checked at every event.
Its loads are the trace (one JSON object a line with the arrival, the prompt, reused and generated tokens and the finish, a request following the one it reuses at least half of or arrives within 3 s of the end of), the owner's case, short turns behind a long prompt, two long prompts, bursts beside four replies, a stream at 70 and 90 percent load beside a long prompt, a long prompt into a stream, and users in a closed loop.
For each load and rule it prints the time to first token and the slowdown against the same prompt alone by the rows read (under 1k, 1k to 8k, 8k to 32k, from 32k), the worst wait of a ready prompt for rows, the longest prompt's first token against alone, the gaps between tokens and the chunk passes a thousand prompt rows took; the score is the mean p95 slowdown over the classes present.
`--rules` names the candidates (A0 today's order, A1 the oldest first, A2 and A3 the oldest keeping three quarters and a half, A4 equal parts, A5 least remaining with aging, A6 least remaining with the oldest's floor, A7 least attained), `--rate` (the replies' objective in tokens a second), `--columns`, `--stages`, `--passes`, `--max-seqs`, `--ubatch`, `--pool` and `--hours` set the rule and the bench, and `--row-scale`, `--fixed-scale` and `--class-ms` move the prices, so a choice that holds at one price only shows.
Every CTest run replays the owner's case at a smaller scale under three rules (`replay_smoke`).

### Sampling

Sampling is per request, on the host, from the logits row the pass returns for that entry: the existing `inference/sampler.hpp` with the request's own temperature, top-k, top-p, penalty and seeded RNG, so a request with `seed` set is reproducible regardless of what it was batched with.
The sampler ranks tokens by score with a tie going to the lower id and takes no sum in an order its selection leaves, so a seeded request gives the tokens `generate` gives with the same settings and seed.
It reads the row in place in the pass's mapped logits, and the scheduler copies a row only for a request that asks for log-probabilities (below).
A retiring pass's rows are drawn on the scheduler's sampling threads (`core/job_threads.hpp`): four, or one fewer than the CPUs the process may use where that is fewer, beside the scheduler thread, which waits for the pass's logits, hands them the rows and waits for every draw before the pass ends and its logits rows come back.
Each draw touches only its request's generator and row copy, and a request wants one row a pass, so no draw depends on another, on the thread that takes it or on the order they run in; the scheduler thread then pushes the tokens in entry order, and it alone pushes to the channels, keeps the ledger and calls the backends.
A pass of one row is drawn on the scheduler thread without waking another.
A `top_k` of 0, which is what the compatible routes' -1 becomes, ranks only the best tokens, 64 at first and more as the nucleus `top_p` keeps needs them, and with `top_p` 1 ranks none.
The additive penalties `presence_penalty` and `frequency_penalty` (OpenAI's definitions, -2 to 2) are subtracted from a generated token's score once and once for each time it was generated, after the multiplicative `penalty` and before the temperature; the draw sees the penalized scores, the log-probabilities the model's own.
`min_p` (0 to 1) cuts the tokens kept by top-k and then top-p to those whose weight at the temperature is at least `min_p` times the likeliest token's, the order other runtimes use (top-k, top-p, min-p): the nucleus is measured on the top-k set as without `min_p`, and `min_p` removes the ranking's tail from it.
At their neutral values (0, 0, 0) the three change nothing, no bit of the draw or the generator's state.
A field the request leaves out takes the default of `infer::Sampling`, the one the CLI's flag starts from, except `max_tokens` on the compatible routes, where leaving it out means no cap; a value outside the range `infer::Sampling` gives the field is refused, as the CLI refuses it, but for the `top_k` of -1 that the compatible routes take as 0.
Greedy requests give the text the CLI gives for the same prompt, which is the first correctness gate below.
A request's `ignore_eos`, the CLI's `--ignore-eos`, is one rule of the sampler (`infer::Sampling::ignore_eos`): the id that ends a reply (`Tokenizer::is_eos`) is passed over by every sampling step, greedy included, and a draw leaves it out before top-k, top-p and the softmax, so the reply runs to its token limit, the context's for an uncapped request, while a stop text still ends it.
The masked token does not exist for the draw, so the repetition penalty cannot bring it back and top-k and top-p count only the other tokens.
The mask changes the draw, not the logits row: whatever reads the row reads the model's own distribution, the end token's share included.

### Drafts

With `--drafter lookup`, `--drafter embedded`, `--drafter lookup,embedded` or a file of MTP blocks beside the model (`--drafter PATH` or `--drafter lookup,PATH`, which draft as `embedded` does) a decoding request drafts the tokens after its last pick and verifies them in the same pass as the other requests' rows (docs/SPECULATIVE.md, section 3), and every reply is the one it gives without drafts.
The pass being formed asks each decoding request it takes for drafts (`Scheduler::propose`): `spec::draft_length` gives the count from the request's acceptance, the tokens it may still generate, the positions its reservation holds and the decode columns the pass has left (`Scheduler::draft_columns`), the rows the device's decode kernels read each weight once for (`Backend::decode_columns`) less a row for each decoding request in the pass, every draft a lone decoder asks, at most the draft rows a pass reserves, 64 or `--draft-max` a request where that is less (`Scheduler::draft_rows`), and none while every seat is taken and a request is queued for one. A request paused or queued for room takes no draft away: it waits for the running requests to end or grow, which their drafts bring sooner.
A request with drafts marks its history (`Model::mark`) and takes one entry, its last pick then the drafts, extent 1, every row's logits; one whose mark is refused, or whose proposer has nothing, takes a decode entry.
Within those, a pass drafts only what pays: the scheduler measures its passes of generated tokens, a base and a row in milliseconds, each by the time the server gave it, from its formation through its retract or, with passes in flight, since the pass before it retired, the drafter's chains apart, and those chains a step, a chain counted once for each pass in flight beside the one being formed, since they all wait behind it (`spec::PassTimes`), and a request drafts where the drafts it is expected to keep, its own share kept at each position weighed with the server's, pay for what their rows add to the pass at the rate the pass gives without drafts, at the depth whose tokens a millisecond are most (`spec::draft_depths`); a pass of a width is priced by the price table (The price table, below), its slowest stage's cell of those decode rows at the deepest decoder's band, or between two measured cells the line through them, brought to the server's time by the decode passes' mean time past their table prices, past the band's last measured width or before its first that width's price continued on the row's slope, and by the base and row only where the band has no cell, so a step in the cost is priced as measured and a draft past the measured widths never looks cheaper than the widest measured pass; the columns and the free marks then go to the drafts the price kept, those with the largest expected share kept first, so a request admitted earlier with a draft it seldom keeps does not take them from a better one; until passes of different widths have been measured every request drafts what the columns allow, so the price is learned, and a server long at one width measures it again.
On Qwen3.6-27B Q8_0 on one MI50 a pass costs about 10 ms a row past a few rows, so the columns alone made four or more users drafting slower than without drafts (docs/STATUS-2026-09.md).
The drafts stay inside the room its reservation already holds, so they never grow it, pause, stall or evict anything.
The pass asks for every request's drafts at once (`spec::Proposer::draft_all`): the embedded drafter's chains run as one batch on the head's device (`Model::draft` over the requests' sequences), each step reading the drafter's weights and the head once for all of them, lookup's on the host.
Where the head's stage records on a thread of its own the batch is queued there and the scheduler thread goes on: its requests sit out the pass being formed and verify their drafts in the first pass formed after the chain has run, so a sampled reply waits one chain between its verifies and no pass waits for the scheduler.
Where `lookup` is listed ahead of a drafter that runs a model (`--drafter lookup,embedded`), a request whose own history convinces drafts from it and is left out of that batch (`spec::lookup_first`, docs/SPECULATIVE.md, section 8), its acceptance kept apart from the drafter's (`spec::Record`) and its drafts priced without a chain; a pass whose requests all draft so runs no chain.
The rows are drawn in order on the sampling pool as `infer::accept` takes them, each with the request's own settings, history and generator, stopping at the first pick that ends the request or differs from its draft, so the draws are those of the run without drafts; with log-probabilities each token's row is copied for the channel while the reader keeps up and its values computed once it has fallen behind (`Request::kRowsWaiting`), as for one row.
Right after the pass ends, before anything parks, pauses or forks the request, its history goes back to the last pick and the drafts its picks kept (`Model::retract`, which on a model that keeps a state reruns the kept rows from the mark), and a request cancelled in flight keeps its last pick alone.
Jobs, replays and resumes never draft; a resumed request drafts once it decodes again.
The model holds marks for up to as many requests as can draft in one pass (`server::draft_marks`, half the decode columns, at most `--max-seqs`, at least one): on a model that keeps a state, each mark is a state slot and saved rows for a verify of `--draft-max` + 1 rows. The fit treats the pool differently depending on how its budget is set. When `--kv-tokens` is omitted and device memory can size an automatic pool, the fit includes the embedded drafter and its first mark before choosing the pool. With an explicit `--kv-tokens` limit, the no-drafter fit chooses the pool first; the drafter and first mark must fit beside that pool without lowering it. In either case, automatic checkpoint slots can be reduced to make room; a `--state-checkpoints` count given by number is held, and the server refuses with the budget and checkpoint count if the drafter still does not fit. Lower `--kv-tokens` or use fewer `--state-checkpoints` to leave room. `--ctx-size` limits one request and does not set the shared pool. The model's own context of `--draft-max` + 1 rows, which a verify's state rerun and the drafter's chains take beside the passes, and the chains' ids and logits for every mark are taken as the server starts and counted in the fit, so its figures in `/v1/health` include them. The other marks take only what is left (`PlacementRequest::fit_marks`); the requests drafting at once are at most the marks verifies in flight leave free (`Model::mark_slots`); on Qwen3.6-27B Q8_0 on one MI50 the drafter with eight marks counted before the budget had left 1536 of 8192 KV tokens and no checkpoint; the host tier copies an embedded drafter's carried row with the checkpoint slot it sits beside.
`/v1/health` counts by draft position the drafts verifies fed and those they kept.

### A prompt larger than the context

A prompt that does not fit what one request may hold (`Scheduler::token_limit`) beside its reply is refused with 413, or, under `--context-overflow shift`, cut by the route before the scheduler sees it (`Api::fit` in `src/server/api.hpp`, over `src/server/context_cut.hpp`); the scheduler only ever takes a prompt that fits.

- **The room.** What fits is one rule with one owner, `Scheduler::prompt_room`: the rows a prompt may take are the limit less the tokens the request asks for, or, where it names none, less an eighth of the limit (`kReplyRoom`). A route cuts to it and `submit` refuses past it (`TooLong`, with the numbers `too_long` gives), so a caller that reaches `submit` without a route is held to it too. The eighth is a fraction so that it means the same at any context: with cuts of half the limit an uncapped chat is cut when it passes seven eighths, keeps about three eighths, and is cut again half the limit of new rows later. It has no floor in tokens: at a small context no count is right, and a client that wants another room sends `max_tokens`. A capped request's room is its own cap, never lowered for it.

- **The index.** A cut point is a row offset after which a prompt may begin again, the row after a turn's end, found by one scan of the prompt's ids for the tokens the tokenizer's metadata marks as ends (`cut_points`); it is a row offset and not a text offset, so a part that is not text changes nothing in it. A rendered conversation has one a message, besides those a template writes before its messages, such as the system block of reasoning instructions the Qwen 3.8 templates add where a conversation has no system message, which one render of the leading system message and one user message counts (`Api::message_ends`); the scan then places every message's end. Where it finds another number, as when a run of tool results shares one end token, a template ends turns otherwise or a user typed the token's text, the chat routes place only the ends the cut asks about, each by rendering the conversation up to that message as text and counting the prompt's ids whose text lies within it (`MessageEnds`); the ends grow with the messages, so `first_kept` finds its two candidates by halving and asks for at most about twice the binary logarithm of the message count, four more besides, where rendering every part grew with the square of the conversation, 37 to 39 s a turn on a chat of 158 messages. A part the template refuses to render alone, such as the system message under a template that asks for a user message, as the Qwen 3.5 templates and the official 3.6 and 3.8 templates do, ends at the last end token before the next part's end, or with that part where the prompt has none, so such a conversation is cut as any other and not answered 500. The cut's line says how many renders placed the ends.
- **The cut.** The leading block stays, the system message or a text prompt's rows up to its first cut point, and rows after it go up to the first cut point a whole number of steps past it, the fewest steps after which the prompt fits (`first_kept`, `cut_rows`). A step is half the limit (`kCutTo`). Where no whole step serves, none having a cut point past the leading block (a system message of three eighths of the limit or more beside an uncapped chat) or none leaving a window that fits before the prompt's end, the cut is the first cut point after the leading block that leaves at most the room, a user start on the chat routes; on the chat routes a kept window starts at a user message, so an assistant's tool calls stay with their results, and the kept messages are rendered again; a text prompt without a cut point that serves keeps its first `kCutLead` rows, 64 or half a step, and loses whole steps of rows.
- **An agent's turn.** Where the leading block and the newest turn alone do not fit, as an agent's turn of many tool calls and results can, the earlier turns go and so do the newest turn's oldest exchanges, each an assistant message with every result after it, so parallel calls and all their results go or stay together and no result is kept without its call (`cut_inside`, F2DEV and XDEV, 2026-10-20). The system message and the turn's user message stay, and the same steps pick the exchange the window starts at, counted from the user message, so successive calls of one agent cut at the same exchange while the turn grows within a step. A prompt is refused under `shift` only where the leading block and the newest user message alone do not fit, or the newest exchange beside them does not; no message is cut inside.
- **Why steps from the start.** Counted from the prompt's start, the cut falls at the same message while a conversation grows within a step, so the kept window is read again once a step and reused on the turns between. Counted from its end, it would move every turn and every turn would read the window again.
- **What a cut costs.** The turn that cuts reads everything past the leading block again, about three eighths of the limit for an uncapped chat, 98000 tokens at 262144. No saved state serves it: a recurrent state at any row holds every row before it, and on a model without one the kept rows' positions and what they attend to both change; a message boundary's state is the state after the dropped messages. Only the leading block is forked, as any shared prefix is. The turn says what it pays as it is admitted (`server: cache: a cut turn reads R rows again beside S reused`, with the seconds at the measured prompt rate once a pass has measured it), and the conversation's uncut history ranks first to go (docs/CACHE-PLAN.md, section 4).
- **What the client sees.** A `context` object in the reply (`tokens`, `limit`, and after a cut `dropped_messages`, `dropped_tokens`, `at_marker`), a line on the server, and the cuts counted in `/v1/health` (`requests.since_start.shifts`, `shifted_tokens`).

### A turn that reuses little

A first admission of a prompt of 4096 tokens or more (`kMissTokens`) that forks under half of it says why, once, with numbers and no text: `server: cache: a turn of M tokens reused N; the closest held history agrees for P tokens (message K of L, its role), limited by LIMIT at A` (`Cache::note_miss`, over `reuse_miss` in `src/server/policy.hpp`).
- **P.** Every history the cache holds in memory, device donors, host entries and message boundaries, is compared with the prompt token by token, each compare stopping at its first difference, row classes left out; the closest is the one P names. Files on disk are not read.
- **The limit.** `tokens`: the closest history differs at P, in its first block where P is under a block. `classes`: its rows were computed in other row classes from A. `state`: on a model that keeps a recurrent state, a history is forked only at a state it keeps, its checkpoint, a host entry's end or a boundary of the same tokens, and the furthest within the agreement is at A. `room`: the history held A rows the turn could have forked but did not, given up for room or out of the source's reach.
- **K.** On the chat routes the message the difference falls in, from the end tokens' positions alone (`Api::message_spans`, no render); where a conversation has fewer end tokens than messages, as where tool results share one, and on `/v1/completions`, none.
- **Health.** `reuse.misses`: `since_start`, the count, and `last`, the last one's `prompt`, `reused`, `agreed`, `limit`, `at`, `message` (counted from 1), `messages` and `role`, or null before the first.

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
                      "requests": {...}, "work": {...}, "reuse": {...}, "pressure": {...},
                      "reread": {...}, "drafting": {...}, "passes": {...}}
                     each group has "now" (true at this moment) and
                      "since_start" (counters), units in the names;
                      with --timing also "timing": {"rounds", "round_ms",
                      "recording_ms", "relaying_ms", "sampling_ms",
                      "assembly_ms", "drafting_ms", "settling_ms",
                      "receive_wait_ms", "staging_wait_ms",
                      "open_wait_ms", "logits_wait_ms", "stage_idle_share": [..],
                      "device_bound_rows_per_s"}, each time a mean a round
GET  /v1/live        {"status": "ok"}, answered without the scheduler
GET  /v1/models      {"object": "list", "data": [{"id": "...", "object": "model", ...}]}
POST /v1/chat/completions   the OpenAI clients' shape over the same scheduler
POST /v1/completions        request: one parse, one request, one drain loop
```

The compatible routes exist so existing tools connect without a client of their own: they list `/v1/models`, send its `id` back as the model, and stream `/v1/chat/completions` as chunks with the role in the first delta, `finish_reason` in the last and `data: [DONE]` after.
They are a JSON mapping in the routes file over the scheduler the native routes use, with llmx's own knobs (`top_k`, `penalty`, `seed`, `ignore_eos`) accepted as extra fields and the synonyms the clients send (`max_completion_tokens`, `repetition_penalty`) beside them, validated before anything reaches the model, and they cost a request exactly what a native one costs.
What the shape cannot carry, token ids and the `eos` finish, stays on the native routes; the compatible replies carry the reused-prefix count as `timings.cache_n` and `usage.prompt_tokens_details.cached_tokens`.
A field a request sends that llmx cannot honour is refused with a 400 naming it, not ignored (`Api::refuse_unsupported`): `tools` and a `tool_choice` other than `"none"` on `/v1/completions`, which renders no chat template, what the chat routes cannot honour of their tools (Tool calls, below), `response_format` other than text, `logit_bias`, `best_of` above 1, `echo`, `suffix`, `reasoning_effort`, `n` above 1 and a prompt array of several entries; `user`, `metadata`, `store`, `service_tier`, `model` and the `stream_options` other than `include_usage` are accepted and ignored.
The role `developer` renders as `system`, and an assistant message with `tool_calls` may leave its content null.
Where a request leaves a sampling field out, llmx's defaults apply (temperature 0.8, `top_k` 40, `top_p` 0.95) and not the OpenAI API's (temperature 1, `top_p` 1), which USAGE.md states beside the fields.
On these routes the stop text is left out of the reply (`StopTrim`): the end of the text so far that could still grow into a stop text is held back, whole and streamed, and released when the reply ends without one; the scheduler's own end check and the native routes keep the stop text in the reply.
The text leaves the stop text out but the tokens do not: the stop text's tokens stay in `usage.completion_tokens`, in the log-probabilities and in the completions route's `text_offset`.
A sampling field takes the range the CLI's flag for it takes, both read from beside the sampler's parameters, so the CLI and the server refuse the same values, with the compatible routes also accepting `top_k: -1` as no top-k, `seed: -1` as no explicit seed, and an omitted or -1 token limit as uncapped.

A streaming response is `text/event-stream`: one `data:` line per token with the id and the decoded text, and a final `data: [DONE]`.
The server holds a character split across tokens until its bytes complete, and replaces each byte that starts no valid UTF-8 character with U+FFFD, since JSON carries only characters.
The CLI writes each token's bytes as they come.
The topology object in /v1/health reports the runtime's realized placement. Its mode is single, layer, tensor or staged_tensor; mixed_roles independently marks roles placed on different backend groups, as with CPU expert offload. Devices gives backend-vector indices, canonical device labels, whether the model uses them and their tensor-group member number. Stages are the ordered mixer-layer runs with half-open layer ranges and the backend indices of each stage's tensor-group members. embedding_members and head_members identify the endpoint groups. Each layers entry reports its mixer and feed-forward groups, whether it is routed, and an expert_stream target when the runtime admitted streaming for that layer. The target is eligible for prompts from min_prompt_tokens; it does not describe the path currently running. Backend names are labels from the resolved device list, not guesses about physical GPU numbering.
The `precision` object in `/v1/health` is the placement record passed at startup: `requested`, `declared`, `effective` and `devices`, each with `device`, `effective`, `how` and `paths`. A device may report F32 fallback while another implements the requested dtype. An entirely F32 run reports F32 as the overall effective dtype. The API serializes it without resolving precision or treating the listed paths as execution witnesses.

The model's name in `/v1/health`, `/v1/models` and every compatible reply is its file name, which on Linux can hold bytes that are not UTF-8, and an error message can carry text from outside the request, such as a chat template's from the model file or a backend's failure.
Both get the same U+FFFD repair as generated text, so every reply is UTF-8.
A non-streaming request gets one JSON object with the text, the ids and the counts.
Errors are JSON with an HTTP status: 400 for a bad request (a text the tokenizer cannot encode and a token id outside the vocabulary included), 413 for a prompt and its `max_tokens` past what one request may hold, unless `--context-overflow shift` cuts it as A prompt larger than the context, above, has it (the model context or the KV pool, whichever is smaller) or a body past 64 MiB, 503 when the queue is full.
A conversation the chat template raises on is a bad request, answered 400 with the template's message; a template the renderer refuses stops `serve` before it serves, since no chat request could be answered.
A model whose layers keep a recurrent state, a `qwen35` file, has a state only at the end of what it has read, so its prefix reuse goes through checkpoints, states kept at a position in slots of their own ([SPECULATIVE](SPECULATIVE.md), section 2): each request keeps one where its prompt's last whole block ends within what a follow-up turn would begin with, the conversation rendered without the generation prompt for a chat route and the whole prompt for a text, the slice that reaches it ending there.
A finished request's donor is its history back to that checkpoint, which a follow-up turn forks, reading the state in place; a paused request keeps its whole history as its checkpoint where a slot is free, taking it back on resuming, and its prompt's checkpoint otherwise.
A checkpoint takes a free slot, else the oldest donor holding one goes (`make_room`), else it is left out; nothing is paused for one.
`--state-checkpoints N` sets the slots, by default the fewer of `--max-seqs` and the most that take at most a quarter of the KV budget's room, and 0 keeps no checkpoint, so a follow-up turn recomputes its history and a paused request resumes from its start; each admitted request also holds one of the `--max-seqs` state slots the load reserves.
A donor the devices evict, to admit a request, to make a growth step's room or for a checkpoint slot, is first copied to host memory, its whole blocks up to its checkpoint on a model that keeps one ([SPECULATIVE](SPECULATIVE.md), section 2, Host tier), within the prefixes' share of `--host-cache-bytes`, what a lookup table's hot rows leave of it ([USAGE](USAGE.md), Bounded lookup tables), by default what `--max-seqs` histories take at the most one request may hold, within half of the host's free memory once the model is loaded less those rows, and none where every cache is on the CPU, and leaving the host a reserve fixed at the start, a twentieth of what it then had free, below which idle slabs are given back, the room going by the ranking stated once in [DISK-TIER](DISK-TIER.md) (What the host tier does now).
A donor whose conversation did not come back, its request having forked nothing the tiers kept, takes only the room that ranking gives it: users taking turns over more conversations than the tier holds would otherwise evict, each time, the one needed next.
A paused request's own donor counts as one whose conversation came back, whatever its request forked: the request resumes, and a copy the tier refused would be recomputed at that resume.
A request matching a copy in host memory better than any donor on the devices promotes it, taking its blocks and checkpoint slot as a first admission takes room and writing it back before its first pass, then forks it, so more conversations than the devices hold resume rather than recompute; the copies keep the bytes the storages held, so the reply is the one a fresh server gives.
The copies are enqueued on the devices' own streams between passes, into host memory the model keeps in 64 MiB slabs once it has first allocated them, and the scheduler thread does not wait for them: an enqueue takes about a millisecond, a slab's first allocation about 30 ms while the tier fills, and dropping a copy from host memory waits for its copies to retire, which they long have by then, while a device spends the copy's own time on its queue.
On a model that keeps a state, a conversation's message boundaries are kept in host memory too ([SPECULATIVE](SPECULATIVE.md), section 2, Host tier): as a chat reply's job completes, its state alone, at the start of the next user message, about 150 MiB on a 27B model, in the room the copies leave, by the same ranking, each conversation keeping four, its first and its newest among them, or with a disk tier every one, the older going to disk as the host tier's room takes them; a request that edits or regenerates an earlier turn forks the boundary before that turn's message with the blocks of a later history of the same conversation, on the devices or promoted from host memory, and reads only from there, and one that finds no boundary there keeps one where that message starts as it reads its prompt, so the next edit of the message forks it.
The slabs alive, idle or holding a copy, stay within `--host-cache-bytes`: before a copy allocates a slab, idle slabs on other devices are freed, and a copy they cannot make room for is not kept.
Under the host tier a disk tier keeps what host memory lets go ([DISK-TIER](DISK-TIER.md)), off unless `--disk-cache-bytes` gives it room: once the copies and boundaries not yet on disk fill three quarters of the host tier, the one the host tier would drop next is written to the server's own directory under `--disk-cache-dir` while it stays in host memory, bit for bit and only what its history lacks there, its new blocks as segments and its state, each file named by digests of its tokens and row classes, so room the host tier needs later releases entries already on disk at once and no request waits for a write.
A donor the host tier cannot hold at all is written from its turn's end on, while requests run and with or without `--disk-cache-keep`, a segment at a time while a request runs or waits: the blocks a piece's time (`kPieceMs`) copies at the measured cost a block, the copy's wait included, one block before any is measured, each segment a file of its own, so no copy off the devices is one long hold. Beside replies a segment goes only at background work's turn, after `kBackgroundGaps` reply-only gaps, as a read-ahead job's piece does; beside prompts alone a segment a round; with no request it goes whole, and the donor a request waits for goes first and whole. A finished donor trimmed to its next turn's prefix stops a write of it in flight past that prefix, which is written again.
A request whose room would take such a donor before its files are whole waits for them, its write then going in whole segments, and a request whose restore from host memory finds no room waits for the room, each at most 15 s (`kWaitBound`, step 8 of [CACHE-PLAN](CACHE-PLAN.md)), or for a write the time waited and the write's remaining bytes at its measured rate (the write in flight's own once it has run a second), 1.5 times and 2 s more, never longer than computing the rows again, after which it goes on and the rows lost are said; `/v1/health` counts the waits under `reuse.disk.since_start.room_waits`.
A conversation that came back and left the devices for room is read back by a job, from host memory or from disk, once room is spare again, so its next turn computes only its new rows.
A waiting request that a history on disk shares more with than anything in memory has its files read back within the read bounds of [DISK-TIER](DISK-TIER.md) (Restore: disk to host to devices). A history that fits the host tier is read there, then promoted and forked; a history with blocks larger than the whole host tier streams directly into its device storage without a whole host copy. The request keeps its place while later requests that fit go ahead; a read that fails its checksums deletes the entry and the request computes its history.
Until the model file's digest is known and the store's directory and write probe are done, which takes about twenty seconds for a 27 GB file not hashed before, the server serves without the disk tier, and `/v1/health` says so (`reuse.disk.now.ready`).
The disk keeps its room by deleting files from a history's end down, its earlier states and then conversations that did not come back first, deletes entries unused past `--disk-cache-max-age`, keeps `--disk-cache-floor` free on the file system, stops writing after a failed write until that room is back, and leaves nothing behind at a clean exit unless `--disk-cache-keep` has it write what memory holds for the next server of the same model and numerics fingerprint, which adopts it.
A reply's rows are decode rows, which a follow-up turn's prompt rows do not share, so a chat reply is read again as prompt rows for the next turn ([SPECULATIVE](SPECULATIVE.md), section 2, Idle re-prefill): the chat routes give the scheduler the ids the next turn begins with (`Scheduler::follow`), the conversation with the reply as a client of the route sends it back (`reply_sent_back`: the content alone on `/v1/chat/completions`, whose clients do not return `reasoning_content`, the whole text on `/v1/chat`), rendered as a following turn renders it (`chat::stable_prefix`), every 32 tokens as far as two continuations of the reply agree and, once it has ended, in full before the reply's last chunk goes out, so the scheduler has them before a client can answer.
The scheduler reads them in a job, an internal request whose prompt is those ids to their last whole block: it forks the history sharing most with them, the request itself while it runs and its donor once it has ended, reads the rest as prompt rows of the class every longer prompt takes, in slices ending on whole blocks, and becomes a donor beside the one it forked, so the follow-up turn forks past the reply while a regenerated reply, the prompt sent again, still forks the earlier one.
The job's donor supersedes the conversation's other donors, the one it forked, the request's and every one whose tokens its own begin with, such as the previous turn's job's: they go first when room is needed, the donor count included, before any donor of another conversation, and the devices evict them without copying them to host memory, so a conversation holds one full copy in each tier and the earlier ones stay, for a regenerated reply, only while room allows.
On a model that keeps a state a job keeps its state at each whole block it reaches, and a running job is a source like a donor, so a follow-up that comes before the job completes forks what it has read.
A job's rows are the lowest-priority prompt rows (promise P6 in [CACHE-PLAN](CACHE-PLAN.md)): a job forms rows only in a round where no reply decodes and no client prompt is read, never rides a replies' pass, and a job chunk already in flight finishes but continues into no further pass while a reply decodes (`Scheduler::replying`); with nothing else to do it takes a pass's whole budget. A job has no one waiting for it, so the trade the service rule makes for a waiting prompt's first token buys nothing that would justify slowing a reply; the job reads between turns, while the user reads the reply. On production a job taking two 16-row chunk passes of about 65 ms beside every decode pass had held a reply at 156k tokens of context to about 6 tokens a second.
Once a finished chat request's next turn is given whole, its donor keeps only the whole blocks that next turn's ids share with it (`Cache::trim`), so a reply's reasoning the next render leaves out holds no room and is neither copied nor written; a model that keeps a state keeps the donor whole on the devices, ending at its checkpoint, and copies and writes it only to that prefix; paused donors, jobs' donors and a request whose next turn is never given, a raw continuation, are kept whole.
A job starts only into spare room, giving up no donor but the one it forks (question 2 of [CACHE-PLAN](CACHE-PLAN.md)), and gives its seat and room back at the next pass boundary when a waiting request cannot be admitted beside it by admission's own test or a request cannot grow; whether it gives way, outgrows its room as its ids grow or stops because its ids changed, its rows become a donor (`park_job`), and it starts again from that donor once nothing waits. Waiting jobs start oldest first, but one still reading its history from disk, or refused room while requests run, holds back no younger job that needs neither.
A waiting job goes once a newer job's ids, or a request admitted, begin with all of its own, or a newer job reads for the same request; conversations that only share a start, as under one system prompt, keep their jobs; and a waiting job asks the restore question a queued request asks, so one starting again reads its history from disk rather than from its first row.
Jobs run only where rows are one class from some extent up to the token limit (`Model::row_class`), on a model that keeps a state only with checkpoint slots, and only for next turns at least that long.
A message that carries `reasoning_content` as a string is taken as sent, its content and its reasoning unchanged.
An assistant message without it, or with it null, is kept as `chat` keeps its own replies (`chat::ChatFormat::assistant`): split by `chat::assistant_turn`, the text after its last `</think>` the content and the reasoning before it `reasoning_content`, only when the template reads `reasoning_content` and does not split a turn at `</think>` itself, as the Qwen 3.8 templates do not, and rendered whole, as sent, under every other template.
So the server renders a conversation as `chat` does, and each model sees earlier reasoning in the form its template was written for, whichever way a client sends it back ([inference-chat](src/inference-chat.md) states the rule).
A reply goes the other way through `chat::ReplySplit`: on `/v1/chat/completions` the text inside the `<think>` the rendered prompt left open (`chat::opens_reasoning`) or the reply opened itself, up to the first `</think>`, is `reasoning_content` and the rest `content`, split as the tokens stream so every delta carries its part, and a reply without `<think>` is `content` byte for byte; the native `/v1/chat` keeps the text whole.
A request's `chat_template_kwargs` reach the template as variables (`chat::TemplateVar`), so a client sets `enable_thinking` as the templates read it.
Their numbers are read as transformers reads them, a number written whole an int and any other a float (`chat::template_value`), so a value written `1.0` reaches the template as the float 1.0 where it reached it as the integer 1 before 2026-10-15, and a whole number of 2^53 or more is refused with 400.
An assistant message's `tool_calls`, and a message's `tool_call_id` and `name`, reach the template as the client sends them, but for one rule: arguments given as JSON text that holds an object are that object, which the newer Qwen templates need to render a call at all ([inference-chat](src/inference-chat.md)); a null or absent content beside calls is empty.
A stream whose pass fails has already sent its 200 head, so it ends with one `data:` event holding the error in the route's error shape, without `data: [DONE]`.

`/v1/tokenize` and `/v1/detokenize` expose the model's tokenizer on the connection thread, without the scheduler, and give the ids and text `llmx tokenize` and `llmx detokenize` give.
A `text` is tokenized as it is, with no chat template, and the text of a special token such as `<|im_start|>` reads as that token, as it does in a prompt; `messages` in its place are rendered first as `/v1/chat` renders them, an assistant message's `reasoning_content` and the template's refusal of a conversation included, so the ids are the ones a chat request with those messages reads.
No start or end token is added, since no route adds one to a prompt, so `add_special`, which clients of other servers send true to count one, is not read, and the count is always what a generating route reads.
Neither route passes through the scheduler, so both answer while the queue is full, and a text past the context is counted rather than refused.
Detokenized text gets the U+FFFD repair of generated text, so the ids of a whole reply give back its text.

### Tool calls

A chat request's `tools`, function tools as the compatible API declares them, reach the template as its `tools` variable (`Api::tools_of`, through `Api::template_vars`), where `tool_choice` is `"auto"`, its default, so the template's own tool block is rendered as transformers renders it; `"none"` is the request without tools, its prompt and reply byte for byte those of the same request with no `tools`.
On `/v1/chat/completions` the reply's tool calls are then taken out of its content (`chat::ToolCallSplit`, after `chat::ReplySplit` has taken the reasoning off, so a `<tool_call>` inside the reasoning stays reasoning), in the format the template writes, read once from its text (`chat::ChatFormat::tool_format`): a JSON object in `<tool_call>`, `{"name": ..., "arguments": ...}`, as the Qwen2.5 and Qwen3 templates write it, or a function and its parameters in `<tool_call>`, `<function=NAME>` and `<parameter=KEY>`, as the Qwen3-Coder and every Qwen 3.5, 3.6 and 3.8 template write it; the renderer fixture records each pinned template's format from Jinja's own parse, and `chat-template` holds the reading to it.
A block becomes a call once its opening parses, `{"name": "...", "arguments": ` or `<function=...>`, so a stream can name the call at once: in the JSON format its arguments are the model's text up to the object's closing brace, valid JSON or not, as the compatible API documents that a client must check; in the function format they are built from its parameters, each value written by the JSON types its schema admits (`chat::tool_params`), read through `type`, `enum`, `const`, `anyOf`, `oneOf`, `allOf`, a `$ref` into the parameters' `$defs` or `definitions`, `properties` and `items`: as a string where it admits only strings (and null); as JSON where it admits no string and the text is JSON; and where it admits a string and another type, names no type, or the request does not declare the parameter or its tool, as a client that lists large tools by name and gives their schemas on request leaves them out, as JSON where the text is a JSON object or array and as a string otherwise.
A value whose text is not JSON is a string under every schema.
A block whose opening never parses stays in the content as the model wrote it, and a reply cut by its length inside a call ends with the arguments it has.
The whitespace that joins the text to a call is not content, and a suffix that may still open or close a call is held back until the next token shows which, as the reasoning's closing tag is.
The reply gives `tool_calls` (`{"id", "type": "function", "function": {"name", "arguments"}}`, the id `call_` with the server's start, the reply's number and the call's index) with a null `content` where it has no text, and `finish_reason` `tool_calls` unless it was cut by its length; a stream gives a call's first piece with its `index`, `id`, `type` and name and its later pieces with their `index` and the next part of `arguments`.
`/v1/chat` renders the tools the same way and keeps its reply's text whole, as it keeps the reasoning whole, so its clients send the text back and the template renders it as the model wrote it.
The loop's next request sends the calls back with the tools' results: an assistant message's `tool_calls`, and a tool result's `tool_call_id` and `name`, reach the template as sent, arguments given as JSON text that holds an object as that object (step 1, in `chat::context`).
The reply read again for the next turn (`reply_sent_back`, `chat::reply_message`) is the content and the calls with their arguments as the reply gave them, so the next request forks past the reply; after a reply of calls the next turn is probed with a tool's result rather than a user message (`chat::stable_prefix`), and no ids are given while a call is still being written.
`parallel_tool_calls` false asks for one call a reply: on either chat route the request is cancelled as its first call closes (`chat::ToolCallSplit` with one call, read on the tokens as they come), the tokens it draws after are left out of the reply and its usage, and the compatible reply holds that call with `finish_reason` `tool_calls`, while `/v1/chat`'s text ends with the token that closes the call and its `finish` is `stop`; the request's line says it was cancelled, and its history is kept as a cancelled request's is.
Refused with 400, each naming its field and why: `tool_choice` `"required"` or naming a function, which a prompt cannot honour and constrained decoding would; any other `tool_choice` but `"auto"` and `"none"`; a `parallel_tool_calls` that is not a boolean; `tools` that are not function tools with a name, and `function.strict` true; and tools for a template that writes no `<tool_call>`.

### What is not in the first version

- A prefix index that outlives the process, spans models or holds more
  than a few dozen donors: the hashed key of KV-CACHE is for that scale.
- Speculative decoding, grammars, logit bias, embeddings endpoints.
- TLS, authentication, rate limiting: the reverse proxy's job.

### Open gaps

None: forks across row classes, the last, closed with `Model::row_class` ([SPECULATIVE](SPECULATIVE.md), step 1), so a first admission forks only rows computed as it computes them.

## Startup

`serve` listens before it loads its model, so a monitor sees how the start goes (`server::Startup`, `server::Front`).
While it starts, `GET /v1/health` answers 503, so a readiness check on 200 keeps its meaning, with `"status": "starting"`, the `server` and `requests` objects in their shapes and a `startup` object; `GET /v1/live` answers 200, the process and its listener being up, and every other route answers 503 in its clients' shape.
`startup` gives `started_unix`, `elapsed_s` (the start's seconds so far, fixed once it is ready or has failed), the `phase` it is in and `phases`, each phase reached so far in order with its `state` (`running`, `done` or `failed`) and its `seconds`; a phase that reads also gives `bytes_done` of `bytes_total`, `bytes_per_s` over the phase so far, and `files`, each file read with its `bytes_done` of `bytes_total`.
The phases are `devices` (the backends made), `metadata` (the files' headers, the tokenizer and the chat format), `fit` (the KV budget and the split), `allocate` (the model built, every weight's storage allocated), `weights` (the weights read and uploaded; a mapped load warms them before `allocate`), `vision` with `--mmproj`, and `ready` (the scheduler and its disk tier made); the loader reports its phases through its one progress callback (`infer::LoadStep`), and the server only renders them.
Once it serves, the routes answer as ever, `/v1/health` keeps `startup` with every phase `done` and `phase` `ready`, and the disk tier's digest of the model's files and its adoption of a kept directory, which run after the server listens, are under `reuse.disk.now.digest` and `reuse.disk.now.adopt` ([DISK-TIER](DISK-TIER.md), Surface).
A model path with nothing there is refused before the server listens; a SIGTERM while it starts ends the process at once, which closes the listener; a load that fails exits nonzero with its error, and its health says `failed` with the error until then.

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

## The cache's moves

A conversation's cache lives on the device (an active request's or a job's sequence, a donor), in host memory (a copy, or a message boundary's state alone) and on disk, and every move between those places is said and counted in one function, `Cache::moved` (`src/server/cache.hpp`), by kind (`server::Move`).
`/v1/health` gives the counts since start under `reuse.moves.since_start`, one field a kind, and `reread.since_start.rows_released`, the rows jobs had read and released as they left the active set, only the rows `park` cannot keep: a job keeps what it read as a donor.
A kind whose move the request's own line already says, or that loses nothing, is counted without a line.

| Kind | The move | Its line |
|---|---|---|
| `parked` | a request's or a job's history is kept as a donor (one too short to keep is not counted) | none |
| `forked` | a request takes rows from a holder | none; the request's line gives the tokens reused |
| `consumed` | a donor is dropped by the request or job that forked it, the two not fitting together | `server: cache: a donor of N tokens consumed by the job that forked M of it` |
| `consumed_lost` | rows past the fork point that only that donor held | `server: cache: rows A to B of a history left memory with the donor a fork consumed` |
| `job_started`, `job_done` | a reply's read-ahead job begins and completes | `server: cache: a job started, rows A to B to read`; none |
| `job_gave_way` | a job gives its seat and reservation back for a waiting request, or stops as its ids outgrow its room or change, its rows kept as a donor (`park_job`), or released where no donor can be kept | `server: cache: a job gave way, its N rows kept as a donor, its state kept at K` (or `outgrew its room as its ids grew`, `stopped, its ids changed`), or `server: cache: a job gave way, N rows released, its state kept at K, M of them held elsewhere` |
| `job_not_handed` | at that give-way, why no waiting request took the job over | `server: cache: a job of N rows was not handed over (why)` |
| `job_lost` | rows the job alone held, released where no donor could be kept | `server: cache: rows A to B of a history left memory with the job that alone held them` |
| `job_handed` | a job's rows taken over by the request it read for | `server: cache: a job was handed to the request it read for, its N rows kept as a donor, its state kept at K` |
| `to_host`, `not_to_host` | a donor leaving the device is copied to host memory, or is not | `server: a donor of N tokens kept in host memory, ...`; `... was not kept in host memory (why)`, or `a history of N tokens left the devices with no copy: ...` |
| `promoted`, `promote_failed` | a host copy goes back to the device, or cannot | `server: a donor of N tokens promoted from host memory, ...`; `... was not promoted from host memory (why)` |
| `host_dropped`, `host_lost` | a host copy released with its files whole or superseded, or with no file; without a disk tier one with nothing superseding it is a loss too, given to the test's hook | none; `server: cache: a copy of N tokens left host memory with no file` |
| `boundary_kept`, `boundary_not_kept` | a state kept at a message's end, or not: the host tier without room, no checkpoint slot for the one a request asked for, or no state found once its pass retired | `server: a message boundary at N tokens kept in host memory, ...`; `... was not kept in host memory (why)`; `server: cache: the message boundary asked at N tokens was not kept (no checkpoint slot for it)`; `server: cache: the message boundary a request asked for was not kept (why, at N)` |
| `boundary_dropped`, `boundary_lost` | a state released with its file whole or by the ranking, or beside a disk tier with no file; one whose file is whole is counted and is no loss | none; `server: cache: the state at N tokens left host memory with no file` |
| `read_started`, `read_done`, `read_refused`, `read_failed` | a history read from disk for a waiting request: one outcome a read, whatever its files | none; `server: a history of N tokens read from disk, ...`; `... was not read back (why); its request reads its prompt`; `server: cache: the read of N tokens from disk failed; its request reads its prompt` |
| `write_started` | a write to disk begun, a turn-end segment, a host copy's file, a state or a flush's file | none |
| `write_whole`, `write_failed` | a file published in the disk tier's index, the moment it becomes a holder, or a write that failed; one the server stopped itself is neither | none; `server: cache: N writes to disk failed` |
| `disk_removed` | a file leaving the index, with the tier's reason: a file of this build covering it, an unclaimed write, the cap, its age, its build's failed proof, no whole path reaching it, its read failing | `server: cache: N files left the disk (why)` |

`reuse.disk.since_start.lost_before_written` stays what it was, the restore points released while no file held them beside a disk tier; `job_lost`, `consumed_lost` and a `not_to_host` that keeps nothing count rows that left the device with their only holder, with or without a disk tier, and name them to `Cache::on_move` with their history.
The disk tier's outcomes are taken where the scheduler takes in the tier's completions (`Cache::sync_disk` from `DiskTier::figures`), never on an I/O thread.


## What a restore cost

A request that waits for a read from disk, or takes a copy back from host memory (`promote`), keeps what that cost on itself (`Request::Restore`, filled where each part happens, in `server/cache.hpp`) and says it at the end of its line.
- Its completed restores, with the tokens and bytes they gave it and the last one's kind, `host` or `disk` (a host copy read back from disk).
- Its waits for reads, with those that reached their bound (`timeouts`) and those whose read failed. A wait that reaches its bound or a read that fails completes no restore: no tokens, no bytes, no promotion.
- A request that waits or restores more than once sums its parts.

The parts sit in two spans of the request.
- In `queued_ms`: `wait_ms`, its wait for the read, from its start to its end (`end_wait`); `hold_ms`, which is 0 now that a promotion's copies are queued on the stages' threads rather than made under a hold of every recorder; and `enqueue_ms`, the copies' enqueue (`Model::restore_host`).
- In `prompt_ms`: `first_pass_ms`, the request's next pass after a promotion, which waits for the copies on the device and does its own work beside other requests; it is first-pass latency, never copy time.
- A part that ends after the request's first token, as a resume's promotion and its next pass do, belongs to neither span: it is `after_first_ms`, said on its own.
- The line gives each span as those parts and its remainder, so a restored turn adds up to its time to first token; a remainder below 0 is printed as it is, and would be a fault of the record.

Beside the spans, overlapping them and so not addends:
- `read_ms`, the read-ready latency from the store's call to `finish_read`, the store's queue and the scheduler's observation included; it overlaps the wait.
- `slabs_ms`, the read's slabs taken and zero-filled (`alloc_host`) before that call.
- `copies_ms`, from the enqueue to the first round boundary at which every ticket of the promotion has retired, asked once a round through `Backend::done` (`copies_retired`, `Model::copies_retired`) while such copies are outstanding; a device backend answers without waiting, as `stage_ready` relies on, and the CPU's copies have retired at submission; it overlaps the first pass.

A read that several requests wait on is shared: each request's record carries its latency, and the server counts it once.
A history streamed from disk straight to the devices (`start_stream`, docs/DISK-TIER.md) is a read with no slabs and no promotion: its `read_ms` runs from its start until its last copy has retired (`stream_round`), on each request waiting for it and once in the server's totals, and it completes a restore from disk for each of those requests; `hold_ms`, `enqueue_ms` and `copies_ms` stay with promotions, and its first pass is timed as a promotion's.
`/v1/health`'s `reuse.restore.since_start` counts each physical read and promotion once: `reads`, `promotions`, `copies` (promotions whose copies were seen retired), `from_host` and `from_disk` (restores requests completed), `timeouts`, and each part's sum and largest (`read_ms`, `slabs_ms`, `hold_ms`, `enqueue_ms`, `copies_ms` and their `_largest_ms`).
A message boundary's state forked with a request (`Model::fork` with a state) is not timed here; its copies are a state's, small beside a history's.

## Live work counters

`work.since_start.prefilled_tokens` counts client prompt positions in successfully retired passes, including prompt rows recomputed after a pause. A cached prefix contributes no rows; background read-again jobs, generated history replay, drafter calls and speculative verification rows contribute none. `work.since_start.generated_tokens` counts output tokens successfully pushed to clients' request channels, including the first output token and accepted speculative tokens, each once. EOS and rejected drafts contribute none; output already published before a later error stays counted. Both counters start at zero with each scheduler and are available without `--timing`.

The scheduler owns both: `retire` intersects each non-decoding client entry's rows with its prompt positions after `Model::end_pass`, and `step` increments its counter after each successful `Request::push`, so a later exception preserves the count of tokens already published. `Scheduler::Stats` reads their atomics and `Api::health` serializes them. The request counters under `requests.since_start` retain their separate meaning: admitted requests that ended, with their whole prompt including reused tokens.

Counter differences divided by the actual time between health samples give live aggregate throughput over that wall-clock interval, including concurrency and idle time. This is not a per-request or per-kernel PP/TG speed. A long pass reveals its prompt work only when it retires; zero means no counted work crossed these boundaries during the interval, even if a pass is still running. Restarted, decreasing or missing counters and missing samples must break a derived rate rather than imply zero work.

`requests.live` gives a cached per-request snapshot collected on demand at a scheduler round boundary: at most 64 records, active requests first, then paused and queued requests, with `total` and `omitted` preserving how many are not shown. Each record has an opaque server-local string ID, its phase, prompt and reused-token counts, retired prompt work, published-token count and elapsed milliseconds since receipt. A request encoding an image is in phase `encode` and adds `image`: `index` (from 1) and `count` among its images, the image's `rows`, `pieces_done` of `pieces_total` and `elapsed_ms` since the image's encode began, wall time with any passes between its pieces, 0 of 0 in 0 ms while another request's image holds the encoder; a pending poll ends a request's run of encoder pieces, so its sample is as fresh during an encode as between passes. No text or token IDs are exposed, and completed requests leave the snapshot. For one ID, pp/s and tg/s are the respective counter changes times 1000 divided by the change in `sample_elapsed_ms`, the monotonic snapshot time since scheduler creation. A new ID or a repeated timestamp supplies no rate observation; recovery prefill counts work again and pauses contribute elapsed time. Each health poll requests a refresh and returns the previous sample; concurrent polls coalesce, and polling never wakes the scheduler. A busy or waiting scheduler can therefore return stale samples, and the first poll may return an empty cache. Emptying the client request set clears cached records before the scheduler waits, while idle polls preserve the sample time. These are observed request work rates, with long passes visible on retirement, rather than kernel timings.

`requests.live.work` contains cumulative `prefilled_tokens` and `generated_tokens` with the same meaning as `work.since_start`, sampled at that request snapshot's `sample_elapsed_ms`. Differences between these totals divided by the snapshot time difference give aggregate rates over the same interval as the request rows. The totals include completed requests and requests omitted by the record cap; summing visible rows cannot recover them. Top-level `work.since_start` remains available at health-read time, which can describe a different interval. A requested boundary updates completed work even when no request row was ever sampled; once idle work is unchanged, its counters and timestamp stay fixed. If no natural scheduler boundary follows a poll, the older cached sample can persist until work resumes, so a repeated timestamp still supplies no new rate observation.

## Health dashboard

The separate [health dashboard](../tools/health_dashboard/README.md) collects
`/v1/health` and serves charts on its own port. It uses the Python standard
library, keeps bounded history in memory and sends no inference requests.
When the API exposes `requests.live.work`, sampled changes show live aggregate
client prompt retirement and output publication using totals captured beside
the per-request snapshot. Rates use actual snapshot times over a rolling window;
a long pass is unobserved until retirement, and zero does not establish idle
hardware. Older live APIs with `work.since_start` use collector-time samples;
the oldest APIs show a completed-work fallback whose prompt totals include
cached tokens and advance at request completion. Missing live fields remain
unavailable. Neither mode measures per-request or per-kernel speed.
Runtime-reported topology describes backend assignments when available; neither
placement nor cache reservations are physical GPU memory telemetry.


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
    "host": {"now": {"entries": 12, "bytes": 913047552, "limit_bytes": 17179869184, "allocated_bytes": 17179869184},
             "since_start": {"promotions": 40, "yielded_bytes": 0, "bytes_moved": 4093640704}},
    "disk": {"now": {"entries": 90, "bytes": 8011472896, "limit_bytes": 214748364800, "in_flight": 0, "ready": true, "writing": true},
             "since_start": {"hits": 7, "bytes_read": 612368384, "bytes_written": 9100574720, "waits": 7, "wait_ms": 412,
                             "errors": 0, "refused": 0, "dropped_for_cap": 0, "lost_before_written": 0}},
    "boundaries": {"now": {"entries": 14}, "since_start": {"hits": 21}},
    "misses": {"since_start": 1, "last": {"prompt": 222191, "reused": 0, "agreed": 221984, "limit": "state", "at": 0, "message": 71, "messages": 72, "role": "assistant"}}
  },
  "pressure": {"since_start": {"pauses": 0, "stalls": 0, "waits": 0, "recomputed_tokens": 0, "resumes_taking_history_back": 0}},
  "reread": {"since_start": {"jobs": 55, "rows": 31040, "cancelled": 3, "handed": 2}},
  "drafting": {"sources": ["lookup", "embedded"], "since_start": {"drafted": 4096, "kept": 2780,
               "by_position": [{"position": 1, "drafted": 1024, "kept": 901}],
               "own_history": {"drafted": 512, "kept": 480},
               "by_source": [{"source": "proposer", "by_position": [[512, 421, 512]]}, {"source": "history 8-15", "by_position": [[120, 101, 120]]},
                             {"source": "history 16-31", "by_position": []}, {"source": "history 32+", "by_position": [[392, 379, 392]]}]}},
  "passes": {"limit": 2, "in_flight": 1, "sampling_threads": 3}
}
```

`timing` appears beside `passes` only under `--timing`, with the keys it has now and `stage_idle` renamed `stage_idle_share` (a fraction per stage).
`drafting.sources` lists the `--drafter` sources in the order tried, as the flag spells them, and is empty with no drafter.
`drafting.by_position` is empty with no drafter, and a server has one drafter, so the counts are by draft position and not by drafter; `own_history` counts those of them the requests' own histories gave ahead of a drafter that runs a model.
`by_source` gives the same by the acceptance each is priced by, the drafter's and the history drafts' by the length of their match: by position, the drafts fed, those kept and those reached, fed with every draft before them kept, the counts a position's chance is computed from (docs/SPECULATIVE.md, section 3).

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
| `host_allocated`, `host_yielded` | `reuse.host.now.allocated_bytes`, `.since_start.yielded_bytes` | `Stats` |
| `host_hits`, `host_bytes_moved` | `reuse.host.since_start.promotions`, `.bytes_moved` | `Stats` |
| `boundaries`, `boundary_hits` | `reuse.boundaries.now.entries`, `.since_start.hits` | `Stats` |
| `disk_entries`, `disk_bytes` | `reuse.disk.now.entries`, `.bytes` | `Stats` |
| `disk_ready`, `disk_writing` | `reuse.disk.now.ready`, `.writing` | `Stats` |
| `disk_hits`, `disk_bytes_read`, `disk_bytes_written` | `reuse.disk.since_start.hits`, `.bytes_read`, `.bytes_written` | `Stats` |
| `disk_waits`, `disk_wait_ms`, `disk_errors` | `reuse.disk.since_start.waits`, `.wait_ms`, `.errors` | `Stats` |
| `disk_capped` | `reuse.disk.since_start.dropped_for_cap` | `Stats` |
| `disk_unproved` | `reuse.disk.now.unproved` | `Stats` |
| `disk_proved`, `disk_disproved` | `reuse.disk.since_start.proved`, `.disproved` | `Stats` |
| `lost_before_written` | `reuse.disk.since_start.lost_before_written` | `Stats` |
| `disk_refused` | `reuse.disk.since_start.refused` | `Stats` |
| `pauses`, `stalls`, `waits`, `recomputed`, `taken_back` | `pressure.since_start.pauses`, `.stalls`, `.waits`, `.recomputed_tokens`, `.resumes_taking_history_back` | `Stats` |
| `turns_without_memory` | `pressure.since_start.turns_without_memory` | `Stats` |
| `reprefills`, `reprefill_rows`, `reprefill_cancels` | `reread.since_start.jobs`, `.rows`, `.cancelled` | `Stats` |
| `reprefill_handed` | `reread.since_start.handed` | `Stats` |
| `drafted`, `kept` | `drafting.since_start.by_position[]` and their sums | `Stats` |
| `own_drafted`, `own_kept` | `drafting.since_start.own_history` | `Stats` |
| (new) | `drafting.since_start.by_source` | `Stats::by_source` |
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
