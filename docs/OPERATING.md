# Running a server

This page is for the person who starts `llmx serve` and keeps it running: setups to copy, what `/v1/health` tells you and what to do when something looks wrong.
[USAGE](USAGE.md) lists every flag and every health field, and [SERVER](SERVER.md) explains the scheduler behind them.
Nothing here is a speed claim; the figures belong to the measurement records in [STATUS](STATUS.md).

## Setups

Each command serves one model on `127.0.0.1:8080`, which only the machine itself can reach.
`--host 0.0.0.0` listens on every interface; there is no TLS or authentication, so put a reverse proxy in front of a server that faces a network.

**One GPU**

```
llmx serve model.gguf --device vulkan:0 --port 8080
```

**Two GPUs as a layer split.**
The model is split into a first and a second half of its layers, fitted to each device's free memory, and `--passes` defaults to one batch per device of the split so both devices work at once.
`--layer-shares 1,1` forces equal halves.

```
llmx serve model.gguf --device vulkan:0,vulkan:1
```

**The disk cache, kept across restarts.**
Conversations the devices drop go to host memory, and what host memory drops goes to disk.
With `--disk-cache-keep` the server writes what memory holds before it exits and the next server of the same model file and build adopts it (another build's too under `--disk-cache-adopt proved`, below), so a conversation continues after a restart without reading its prompt again.
The final write takes at least 20 seconds and longer for a large cache: the server prints its bound as the write starts, and a stop that allows less loses what is not yet written.
A cache of tens of gigabytes needs more than the 10 seconds `docker stop` waits by default, so set the time well above the bound.

```
llmx serve model.gguf --device vulkan:0,vulkan:1 --disk-cache-bytes 107374182400 --disk-cache-dir /var/cache/llmx --disk-cache-keep
docker stop -t 120 CONTAINER
```

A server adopts entries only when the model, every file of it, the build's numerics, the devices, the precision, the cache types, the KV block sizes and the placement are all those of the server that wrote them, and only conversations used within `--disk-cache-max-age` (24 hours by default), with the older files they stand on.
Change any of those flags between two servers and the second adopts nothing; its start line says which part differed (Troubleshooting, below).
An upgrade usually changes the build's numerics, so by default the first server of a new build starts with an empty cache.
`--disk-cache-adopt proved` keeps the conversations across an upgrade: the new build takes the old build's entries of the same model and placement and proves each conversation as it comes back, by computing its first block and its last stretch and comparing them with the files.
Conversations of two turns and more come back in about the time of their last turn, and one-turn conversations on a model with recurrent layers are read again.
Where the new build does compute other bits, the first conversation to come back pays about twice its prompt, once, and the old build's entries are deleted.
The comparison does not cover the rows between the two stretches, as `--disk-cache-adopt` in [USAGE](USAGE.md) spells out; leave the default, `exact`, where every served bit must be the running build's own.

**Drafting.**
The server proposes several tokens and checks them in one pass, which gives the same reply and can be faster where the passes have room.
`lookup` needs nothing in the model file, `embedded` uses the MTP block a file carries, and a path names a file of MTP blocks beside the model; sources are tried in the order listed, so `lookup,embedded` takes the reply's own history first and the block where it has nothing, the form to run on a file that carries the block (docs/USAGE.md, `generate`).
The server drafts only while its measured cost finds a gain, so it goes off as more requests decode.

```
llmx serve model.gguf --device vulkan:0 --drafter lookup --draft-max 4
```

**A generating request beside a long prompt.**
On one GPU or one tensor group a request that is already writing its reply keeps writing while a long prompt is read: the server serves it between the steps of the prompt's batch, about every 100 ms where the device's steps allow, and the prompt's reading takes longer for it.
The start line `server: pacing: automatic` says so, and `pacing` in `/v1/health` counts the gaps and how many were longer than the 100 ms ; the `pacing` rows of the health table in docs/USAGE.md say what each counter means.
On the CPU, with experts on the CPU, and on a layer split a prompt goes first instead: the writing request gets a token each time a batch of the prompt has run, which beside a long prompt is one or two seconds a token, and the start line says `pacing: off` with the reason.
`--no-pacing` chooses that everywhere.
`--generating-share X` replaces the automatic rule by an account that aims the requests that are writing at the fraction X of the device's time, with bounded credit and charge and no guarantee over every stretch; a layer split refuses the flag.
Its trade, as typical costs and not as bounds:

- the prompt's reading takes about 1 / (1 - X) of the time it takes alone, so twice as long at 0.5 and a third longer at 0.25, and only while a request is writing;
- the writing requests keep about X of the speed they have alone, and their longest pause is about (1 - X) / X of one of their own steps, where at 0 it is a whole prompt batch.

Choose the smallest share that makes a reply readable while a prompt arrives: 0.25 to 0.33 where long prompts are common and a wait for the first token matters, 0.5 where people read replies as they are written and long prompts are rare.
With nothing writing, a prompt is read at full speed in every mode; pacing, automatic or by a share, holds a small second arena on every device, which `--no-pacing` gives back.
With many users the share also lowers total output, most where prompts are long, because a batch of prompt rows uses a device better than a batch of a few generated rows.

One measured case, to read the trade from (a 27B model on a tensor group of two GPUs with `--drafter embedded`, which then drafted from the block alone, as `embedded` does now, one request writing while a 50000-token prompt is read; the record with its conditions and the many-user tables is in [STATUS](STATUS.md), A generating request keeps a share of a device beside a prompt):

| share | the writing request, tokens a second | its longest pause | the prompt's read |
|---|---|---|---|
| 0 | 2.2 | 2.5 s | 182 s |
| 0.25 | 20.6 | 0.4 s | 238 s |
| 0.33 | 26.2 | 0.25 s | 265 s |
| 0.5 | 38.0 | 0.2 to 0.5 s | 350 s |

Alone, that request writes about 60 tokens a second.
`passes.since_start` in health shows it at work (Reading health, below).

```
llmx serve model.gguf --device vulkan:0,vulkan:1 --tensor-width 2 --generating-share 0.33
```

**Many users.**
`--max-seqs` is how many requests run at once, `--max-queue` how many wait for a place before a new one gets a 503, `--kv-tokens` the conversation memory (KV cache) they all share, by default all the devices' free memory holds, and `--ctx-size` the most one request may hold, the model's context by default.
More requests at once need a larger pool: a request that sets a token limit waits in the queue until the pool can hold it, and one without a limit is paused when the pool runs out and reads its history again when room returns (`pressure` in health).
The server fits the budget to the devices at load and prints what it took; the default follows the free memory at start, so a server sharing its devices, or a production start that should take the same pool each time, gives `--kv-tokens`.

```
llmx serve model.gguf --device vulkan:0,vulkan:1 --max-seqs 32 --max-queue 128 --kv-tokens 65536
```

## Reading health

`GET /v1/health` returns groups of numbers; every group that counts things has `now`, what is true at this moment, and `since_start`, counters that only rise from zero when the server starts.
USAGE (`/v1/health` and `/v1/live`) lists every field with its unit and meaning.
Look at these five first:

1. `requests.now.queued` against `requests.limits.queued`, and `requests.now.active` against `requests.limits.active`.
   A server with `active` at its limit and `queued` above 0 is full; when `queued` reaches its limit, new requests get a 503.
2. `requests.now.paused` and `pressure.since_start.pauses`.
   A paused request ran out of KV memory and waits to resume; pauses that keep rising mean the load needs more `--kv-tokens` or fewer `--max-seqs`.
3. `reuse.since_start.forks` and `.tokens` against `requests.since_start.finished` and `.prompt_tokens`.
   `forks` is how many requests started from a shared history and `tokens` how many prompt tokens that saved; if your clients send conversations, `forks` close to zero means the prefixes are not being found (Troubleshooting).
4. `reuse.host.now.bytes` against `reuse.host.now.limit_bytes`, and the same for `reuse.disk.now`.
   A tier at its limit is working as designed: it makes room for a newer entry by dropping others.
5. `reuse.disk.now.ready` and `reuse.disk.since_start.errors`, when a disk cache is configured.
   `ready` is false for a while after start (the server reads the model file's digest first), and `errors` above 0 means a write or read failed; after a failed write the server stops writing until a check a minute later finds the floor and a tenth of the cap free.

`GET /v1/live` answers `{"status": "ok"}` without asking the scheduler anything; use it for a load balancer's liveness probe.
`/v1/health` can wait up to the time of a stage when the scheduler is busy, so give a monitor that polls it a timeout longer than that.
A server that answers `/v1/live` and not `/v1/health` within a long timeout has a scheduler that is not making progress.

## Troubleshooting

**The first token takes a long time.**
1. Is the request waiting for a place?
   `requests.now.queued` above 0 with `active` at its limit means every place is taken; the wait is the queue, and `--max-seqs` or more devices change it.
2. If it is not queued, the prompt is being read.
   A prompt is read in slices of `--ubatch` tokens, beside the other requests' decode rows, so a long prompt takes as many passes as it has slices.
3. Is the prompt new each time?
   A request starts from a kept history only for the whole KV blocks its prompt shares with it from the first token on (blocks of 128 tokens on the CPU, 64 on a Vulkan device).
   A client that changes the first part of the prompt every turn (a timestamp or a request id near the top, a system message that varies) shares nothing after the change.
   Send one request twice and read `reused_tokens` in the reply (`timings.cache_n` on the compatible routes); it is 0 when nothing was shared, and `reuse.since_start.forks` rises by one when something was.
4. After a restart, did the server adopt what the old one kept?
   The start line `disk cache in DIR, ..., N entries adopted` says how many; a line `a kept directory was not adopted, its X differing from this server's; its N entries, M MiB, computed again where a turn needs them` names what differed and what it costs, among `model`, `numerics`, `compiler`, `flags`, `shaders`, `libm` and `layout`.
   With `--disk-cache-adopt proved` a difference in `numerics`, `compiler`, `flags`, `shaders` or `libm` alone does not stop adoption: the start line then says how many of the entries are another build's, and `reuse.disk.now.unproved` counts them down as conversations come back, but does not fall to zero on its own: a file a path uses in part is not claimed and stays unproved until it ages out.
   `layout` covers the devices, precision, cache types, KV block sizes and placement, so a different `--device`, `--dtype`, `--layer-shares`, `--cache-type-k` or `--cache-type-v` stops adoption.
   A different `--ctx-size`, or a pool below the model's context, changes it only when what a request may hold crosses a point where the server's row classes change, so raising a budget usually keeps the cache, and a pool past the context always does.
   Other reasons for adopting nothing: the previous server did not run with `--disk-cache-keep`, it was killed before it finished writing, or its entries were older than `--disk-cache-max-age`.
   A disk cache is also not used until `reuse.disk.now.ready` is true.

**Generation slows while another request's long prompt is read.**
A decoding request's row rides the same pass as the prompt slice, and its token comes when that pass has left the last stage, so the longest gap between its tokens is about one pass of `--ubatch` prompt rows.
A smaller `--ubatch` shortens the gap and reads prompts slower; USAGE (Physical batch) has the measured trade.
Check `requests.now.active` to see how many are sharing the pass.
If `pressure.since_start.stalls` or `pauses` rise too, requests are also waiting for KV room, and the cure is `--kv-tokens`.

**Memory or disk is filling.**
The KV cache is fitted and backed whole at load, so a request does not grow it.
Uncapped requests that together want more than the pool take turns at its edge where host memory can hold the history of the one that yields (81 KiB a token on Qwen3.8-27B Q8_0, about 15 GiB at a 200k context), and on a model that keeps a state a checkpoint slot is free. Each request is judged by its own history: with less host memory (`--host-cache-bytes`) than the longest a request may have the server says so as it starts, naming the history length up to which requests still take turns, and a request with a longer history keeps the edge until it ends; with none (`--host-cache-bytes 0`) no request yields.
What grows is the kept histories, and each tier has a cap: host memory at `reuse.host.now.limit_bytes` (`--host-cache-bytes`) and the disk at `reuse.disk.now.limit_bytes` (`--disk-cache-bytes`).
Compare `bytes` with `limit_bytes` in each; a tier at its cap makes room by dropping entries.
On disk, entries unused for longer than `--disk-cache-max-age` are deleted, and the server keeps `--disk-cache-floor` bytes of the file system free after every write (the larger of 16 GiB and a twentieth of the disk by default).
A server without `--disk-cache-keep` removes its directory when it exits and the next server removes the directory of one that crashed.
`reuse.disk.since_start.dropped_for_cap` counts entries deleted to stay under the cap, and `lost_before_written` the histories and message boundaries released before any file held them, whether host memory gave up their room or a history left the devices with no copy made; a disk cache that loses many of these is too small, or too slow, for the rate conversations end.

**The server refuses requests.**
- 503: the queue is full.
  `requests.now.queued` equals `requests.limits.queued`.
  Raise `--max-queue`, raise `--max-seqs` if the devices have the room, or send fewer requests.
- 413: the prompt plus `max_tokens` is more than what a request may hold (`server.request_tokens`: the smallest of `--ctx-size`, the model's context and the pool), or a prompt sent without `max_tokens` leaves its reply less than an eighth of that; the message gives the prompt's tokens and the limit. For a chat front end that sends the whole conversation every turn and does not trim it, start the server with `--context-overflow shift`: it drops the oldest turns in steps of half the limit, before a reply would run out of room and not only when the prompt overflows, says so in each reply's `context` object and counts it in health (`requests.since_start.shifts`), and the turn that drops them waits for the kept window to be read again.
  Shorten the prompt, lower `max_tokens`, or raise `--ctx-size` up to the model's context.
- 400: the body is not valid; the reply's `error` names the problem.
- No answer at all: `GET /v1/live`.
  If that fails the process is down or the listener is gone; if it answers and `/v1/health` does not, the scheduler is busy or stuck.
