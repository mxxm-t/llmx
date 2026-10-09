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
With `--disk-cache-keep` the server writes what memory holds before it exits and the next server of the same model file and build adopts it, so a conversation continues after a restart without reading its prompt again.
The final write takes at least 20 seconds and longer for a large cache: the server prints its bound as the write starts, and a stop that allows less loses what is not yet written.
A cache of tens of gigabytes needs more than the 10 seconds `docker stop` waits by default, so set the time well above the bound.

```
llmx serve model.gguf --device vulkan:0,vulkan:1 --disk-cache-bytes 107374182400 --disk-cache-dir /var/cache/llmx --disk-cache-keep
docker stop -t 120 CONTAINER
```

A server adopts entries only when the model file, the build's numerics, the devices, the precision, the cache types, the KV block sizes and the placement are all those of the server that wrote them, and only conversations used within `--disk-cache-max-age` (24 hours by default), with the older files they stand on.
Change any of those flags between two servers and the second adopts nothing; its start line says which part differed (Troubleshooting, below).

**Drafting.**
The server proposes several tokens and checks them in one pass, which gives the same reply and can be faster where the passes have room.
`lookup` needs nothing in the model file, `embedded` uses the MTP block a file carries, and a path names a file of MTP blocks beside the model.
The server drafts only while its measured cost finds a gain, so it goes off as more requests decode.

```
llmx serve model.gguf --device vulkan:0 --drafter lookup --draft-max 4
```

**A generating request beside a long prompt.**
By default a prompt goes first: a request that is already writing its reply gets a token each time a batch of the prompt has run, which beside a long prompt is one or two seconds a token.
`--generating-share X` keeps the fraction X of the device's time for the requests that are writing while a prompt is read, on one GPU or on one tensor group; a layer split refuses the flag.
The trade is direct, and you choose it:

- the prompt's reading takes about 1 / (1 - X) of the time it takes alone, so twice as long at 0.5 and a third longer at 0.25, and only while a request is writing;
- the writing requests keep about X of the speed they have alone, and their longest pause is about (1 - X) / X of one of their own steps, where at 0 it is a whole prompt batch.

Choose the smallest share that makes a reply readable while a prompt arrives: 0.25 to 0.33 where long prompts are common and a wait for the first token matters, 0.5 where people read replies as they are written and long prompts are rare.
With nothing writing, a prompt is read at full speed at any share, and the flag costs a small second arena on every device.
With many users the share also lowers total output, most where prompts are long, because a batch of prompt rows uses a device better than a batch of a few generated rows.

One measured case, to read the trade from (a 27B model on a tensor group of two GPUs with `--drafter embedded`, one request writing while a 50000-token prompt is read; the record with its conditions and the many-user tables is in [STATUS](STATUS.md), A generating request keeps a share of a device beside a prompt):

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
`--max-seqs` is how many requests run at once, `--max-queue` how many wait for a place before a new one gets a 503, and `--ctx-size` the conversation memory (KV cache) they all share.
More requests at once need a larger `--ctx-size`: a request that sets a token limit waits in the queue until the pool can hold it, and one without a limit is paused when the pool runs out and reads its history again when room returns (`pressure` in health).
The server fits the budget to the devices at load and prints what it took.

```
llmx serve model.gguf --device vulkan:0,vulkan:1 --max-seqs 32 --max-queue 128 --ctx-size 65536
```

## Reading health

`GET /v1/health` returns groups of numbers; every group that counts things has `now`, what is true at this moment, and `since_start`, counters that only rise from zero when the server starts.
USAGE (`/v1/health` and `/v1/live`) lists every field with its unit and meaning.
Look at these five first:

1. `requests.now.queued` against `requests.limits.queued`, and `requests.now.active` against `requests.limits.active`.
   A server with `active` at its limit and `queued` above 0 is full; when `queued` reaches its limit, new requests get a 503.
2. `requests.now.paused` and `pressure.since_start.pauses`.
   A paused request ran out of KV memory and waits to resume; pauses that keep rising mean the load needs more `--ctx-size` or fewer `--max-seqs`.
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
   The start line `disk cache in DIR, ..., N entries adopted` says how many; a line `a kept directory was not adopted, its X differing from this server's` names what differed, among `model`, `numerics`, `compiler`, `flags`, `shaders`, `libm` and `layout`.
   `layout` covers the devices, precision, cache types, KV block sizes and placement, so a different `--device`, `--dtype`, `--layer-shares`, `--cache-type-k` or `--cache-type-v` stops adoption.
   A different `--ctx-size` changes it only when the new budget crosses a point where the server's row classes change, so raising a budget usually keeps the cache.
   Other reasons for adopting nothing: the previous server did not run with `--disk-cache-keep`, it was killed before it finished writing, or its entries were older than `--disk-cache-max-age`.
   A disk cache is also not used until `reuse.disk.now.ready` is true.

**Generation slows while another request's long prompt is read.**
A decoding request's row rides the same pass as the prompt slice, and its token comes when that pass has left the last stage, so the longest gap between its tokens is about one pass of `--ubatch` prompt rows.
A smaller `--ubatch` shortens the gap and reads prompts slower; USAGE (Physical batch) has the measured trade.
Check `requests.now.active` to see how many are sharing the pass.
If `pressure.since_start.stalls` or `pauses` rise too, requests are also waiting for KV room, and the cure is `--ctx-size`.

**Memory or disk is filling.**
The KV cache is fitted and backed whole at load, so a request does not grow it.
What grows is the kept histories, and each tier has a cap: host memory at `reuse.host.now.limit_bytes` (`--host-cache-bytes`) and the disk at `reuse.disk.now.limit_bytes` (`--disk-cache-bytes`).
Compare `bytes` with `limit_bytes` in each; a tier at its cap makes room by dropping entries.
On disk, entries unused for longer than `--disk-cache-max-age` are deleted, and the server keeps `--disk-cache-floor` bytes of the file system free after every write (the larger of 16 GiB and a twentieth of the disk by default).
A server without `--disk-cache-keep` removes its directory when it exits and the next server removes the directory of one that crashed.
`reuse.disk.since_start.dropped_for_cap` counts entries deleted to stay under the cap, and `lost_before_written` the copies host memory gave up room for before any file held them; a disk cache that loses many of these is too small, or too slow, for the rate conversations end.

**The server refuses requests.**
- 503: the queue is full.
  `requests.now.queued` equals `requests.limits.queued`.
  Raise `--max-queue`, raise `--max-seqs` if the devices have the room, or send fewer requests.
- 413: the prompt plus `max_tokens` is more than the smaller of the model's context (`server.context_tokens`) and the KV budget (`--ctx-size`); the message gives the prompt's tokens and the limit. For a chat front end that sends the whole conversation every turn and does not trim it, start the server with `--context-overflow shift`: it drops the oldest turns in steps of half the limit and says so in each reply's `context` object, and the turn that drops them waits for the kept window to be read again.
  Shorten the prompt, lower `max_tokens`, or raise `--ctx-size`.
- 400: the body is not valid; the reply's `error` names the problem.
- No answer at all: `GET /v1/live`.
  If that fails the process is down or the listener is gone; if it answers and `/v1/health` does not, the scheduler is busy or stuck.
