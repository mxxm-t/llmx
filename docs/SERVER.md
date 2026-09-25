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
  throughput design is one forward pass per scheduler iteration carrying
  every active request's next token plus a slice of some new request's
  prompt, not one pass per request. `Model::forward` already takes that
  batch: a prefill microbatch is one entry with many tokens, a decode step
  is many entries with one token each, and the two mix in one pass.
- **One submitter per device.** A `Backend` is driven by one thread at a
  time. The scheduler is that thread; connection threads only queue
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
- **A memory budget that admits, not crashes.** The KV pool holds
  `--ctx-size` tokens in total, the model context by default. A capped
  request is admitted when the pool can hold its prompt and its
  `max_tokens`, an uncapped one (a compatible route without `max_tokens`)
  when it can hold its prompt and a growth step, reserving more as it
  generates; otherwise it waits in the queue, and past `--max-queue`
  waiting requests a new one is refused with 503.
  A request's donor is chosen before room is made for it, and the other donors are evicted, oldest first, when it needs their blocks.
  If the pool is still short, its donor is consumed: the request forks it and the donor goes, so the blocks they share are reserved once rather than for each, and a follow-up turn keeps the history it repeats however many donors fill the pool.
  A request that shares every full block of its donor, a follow-up turn or a resume, consumes that donor before any other is evicted, since all the donor holds beyond what the request keeps is a partial last block; the other donors go only if that does not make room.
  A donor is consumed only when the pool is short, so while there is room it stays for other requests sharing its prefix.
  When an uncapped request cannot grow even with every donor evicted, the latest admitted uncapped request is paused: its history becomes a donor and it is queued again at the front, resuming from those blocks unless another request needed them.
  A capped request is never paused, and a request that cannot be admitted is not started.
- **Dependency-free transport.** HTTP/1.1 over BSD sockets and Winsock,
  request parsing, chunked responses, JSON in and out through
  `core/json.hpp`. No TLS: the server sits behind a reverse proxy when it
  faces a network, which is what every runtime's built-in server assumes.

## Structure

```
server/
  http.hpp       listen, accept, parse one request, write a response or a
                 chunked stream; blocking sockets, one thread per connection
  scheduler.hpp  the request queue, admission, batch assembly, the forward
                 loop, sampling, token channels
  api.hpp        the routes and their JSON: /v1/generate, /v1/chat,
                 /v1/health, /v1/models, /v1/chat/completions,
                 /v1/completions
cli/main.cpp     `llmx serve <model.gguf> [--host H] [--port N] [--device D]
                 [--max-seqs N] [--max-queue N] [--ctx-size N] [--ubatch N]
                 [--cache-type-k T] [--cache-type-v T] [--threads N]
                 [--layer-shares A,B] [--n-cpu-moe N] [--cpu-moe]
                 [--moe-stream-from N]`
```

`server/` sits above `inference/` in the layering: it uses the model, the tokenizer, the sampler and the chat template renderer, and adds scheduling and transport.
It does not drive `infer::generate`, which runs one sequence to its end: the scheduler advances every request a pass and has its own per-token end check (end of text, any of a request's stop texts, its token limit), over the shared sampler (`infer::sample`, with the defaults and ranges of `infer::Sampling`) and `Tokenizer::is_eos`.
The directory is created with its first working route, not before.

### Threads

```
accept thread ---> connection thread (one per socket)
                     parse request, validate, tokenize, render the chat
                     template, enqueue a Request, then wait on the
                     request's token channel and write chunks until done,
                     looking at the socket every 100 ms meanwhile
scheduler thread   the only caller of Model::forward for its devices
```

A `Request` carries the prompt ids, the sampling parameters, a `Sequence`, the stop conditions and a channel: a mutex, a condition variable and a deque of sampled ids that the connection thread drains.
Cancellation is a flag the connection thread sets once its client has gone, which it learns in one of two ways: a write to the client fails, or a look at the socket finds the connection closed.
The thread waits on the channel for at most 100 ms at a time and looks at the socket whenever 100 ms have passed since its last look, token or not, so a departed client is noticed within about 100 ms wherever its request is: waiting in the queue, prefilling, building a whole reply or streaming.
The look takes no byte and never blocks: an end of stream or a reset from the client is gone, and nothing to read, data waiting or only urgent (out-of-band) data is present.
So a client that shuts only its sending side after the request is taken as gone, since that arrives as the same end of stream, and one that has sent bytes past its request is taken as present until a write to it fails.
A client taken as gone gets no answer, not even an error: its request is cancelled and the connection closes, so a client that shut only its sending side and still reads sees the end of the connection.
The scheduler sees the flag at its next iteration, after the pass in flight: a queued request leaves the queue wherever it waits, and an active one is dropped from the batch and its sequence released, which returns its blocks once the last pass that read them has retired.

### The scheduler loop

```
loop:
  drop:    end the queued requests whose client left, wherever they
           wait, and look again as admission reaches each one
  admit:   while the queue has a request and active < max_seqs: find
           the donor sharing the longest run of full blocks; if the pool
           can hold the prompt plus max_tokens, or an uncapped request's
           prompt plus a growth step (dropping the other donors, oldest
           first, then consuming that donor, to make room; a donor whose
           full blocks the request all shares is consumed first), take
           it, fork the donor at the shared blocks or make a fresh
           sequence, and mark the request "prefilling" with an offset
           into its prompt
  grow:    an uncapped decoding request whose next token passes its
           reservation reserves another step, dropping donors first, or
           else the latest admitted uncapped request is paused
  assemble: one entry per decoding request with its last sampled id;
           then prompt slices from prefilling requests, in queue order,
           until the pass holds ubatch tokens; a request whose slice ends
           its prompt wants logits, the others do not
  run:     Model::forward(ctx, entries, n); ctx.logits() waits
  sample:  per entry that wanted logits, the request's own sampler state;
           push the id to its channel; commit the sequence; finish on EOS,
           a stop string or max_tokens, and keep the history as a donor
           when it holds a full block, else release
  repeat while any request is active; otherwise block on the queue
```

Prefill of a long prompt is chunked at `ubatch`, so a 16k prompt does not
stall the decoding requests for a whole pass: they advance one token per
iteration while the prompt goes through in slices. That is the reason the
prompt slice comes after the decode entries and is bounded by what is left
of `ubatch`.

Two execution contexts would be the overlap of host sampling with the device's next pass.
On the device with Qwen3-0.6B-Q8_0 (step 5) a timing build recorded about 25 microseconds of host time between a pass's logits and the next `forward` whatever the batch, against a pass of 5 to 30 milliseconds, so the server runs one context.
A later per-row timing of the sampler with the scheduler's row copy gave 0.25 ms a row greedy on the EPYC 7262 under load, which does not agree with it; layer split phase 3's step 0 (`docs/STATUS.md`) times both again on a quiet host.
What the host does spend per pass is the recording of the pass itself, 0.7 milliseconds at one sequence and 1.7 at eight, which no second context hides because the next pass's tokens come from this one; only a recorded pass replayed with new inputs would, and that is a backend change noted in STATUS, not a scheduler one.

### Sampling

Sampling is per request, on the host, from the logits row the pass returns for that entry: the existing `inference/sampler.hpp` with the request's own temperature, top-k, top-p, penalty and seeded RNG, so a request with `seed` set is reproducible regardless of what it was batched with.
A field the request leaves out takes the default of `infer::Sampling`, the one the CLI's flag starts from, except `max_tokens` on the compatible routes, where leaving it out means no cap; a value outside the range `infer::Sampling` gives the field is refused, as the CLI refuses it, but for the `top_k` of -1 that the compatible routes take as 0.
Greedy requests give the text the CLI gives for the same prompt, which is the first correctness gate below.

### Protocol

```
POST /v1/generate    {"prompt": "...", "max_tokens": 64, "temperature": 0.8,
                      "top_k": 40, "top_p": 0.95, "seed": 0, "stop": ["..."],
                      "stream": true}
POST /v1/chat        {"messages": [{"role": "user", "content": "..."}], ...}
                     the model's chat template renders the prompt
GET  /v1/health      {"status": "ok", "model": "...", "active": n, "queued": m,
                      "donors": d, "prefix_hits": h, "prefix_tokens": t,
                      "pauses": p}
GET  /v1/models      {"object": "list", "data": [{"id": "...", "object": "model", ...}]}
POST /v1/chat/completions   the OpenAI clients' shape over the same scheduler
POST /v1/completions        request: one parse, one request, one drain loop
```

The compatible routes exist so existing tools connect without a client of their own: they list `/v1/models`, send its `id` back as the model, and stream `/v1/chat/completions` as chunks with the role in the first delta, `finish_reason` in the last and `data: [DONE]` after.
They are a JSON mapping in the routes file over the scheduler the native routes use, with llmx's own knobs (`top_k`, `penalty`, `seed`) accepted as extra fields and the synonyms the clients send (`max_completion_tokens`, `repetition_penalty`) beside them, validated before anything reaches the model, and they cost a request exactly what a native one costs.
What the shape cannot carry, token ids and the `eos` finish, stays on the native routes; the compatible replies carry the reused-prefix count as `timings.cache_n`.
A sampling field takes the range the CLI's flag for it takes, both read from beside the sampler's parameters, so the CLI and the server refuse the same values, except that the compatible routes take a `top_k` of -1, which clients send for no top-k, as 0.

A streaming response is `text/event-stream`: one `data:` line per token with the id and the decoded text, and a final `data: [DONE]`.
The server holds a character split across tokens until its bytes complete, and replaces each byte that starts no valid UTF-8 character with U+FFFD, since JSON carries only characters.
The CLI writes each token's bytes as they come.
The model's name in `/v1/health`, `/v1/models` and every compatible reply is its file name, which on Linux can hold bytes that are not UTF-8, and an error message can carry text from outside the request, such as a chat template's from the model file or a backend's failure.
Both get the same U+FFFD repair as generated text, so every reply is UTF-8.
A non-streaming request gets one JSON object with the text, the ids and the counts.
Errors are JSON with an HTTP status: 400 for a bad request, 413 for a prompt past the context, 503 when the queue is full.
A stream whose pass fails has already sent its 200 head, so it ends with one `data:` event holding the error in the route's error shape, without `data: [DONE]`.

### What is not in the first version

- A prefix index that outlives the process, spans models or holds more
  than a few dozen donors: the hashed key of KV-CACHE is for that scale.
- Speculative decoding, grammars, logit bias, embeddings endpoints.
- TLS, authentication, rate limiting: the reverse proxy's job.

## Gates

- **Correctness.** A greedy request through the server gives the same
  token ids as `llmx generate --temp 0` for the same prompt on the same
  backend, alone and while three other requests decode beside it; the
  kv-cache test already holds a two-entry `forward` to the entries run
  alone. A forked prefix continues exactly as a fresh sequence fed the
  same history. A cancelled request returns its blocks and the others
  finish unchanged. All on the CPU and on the device.
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
| 5 | The second execution context, if measured to help (**measured, not added**) | The host gap between passes is about 25 microseconds at 1, 4, 8 and 16 sequences on the device, against passes of 5 to 31 milliseconds, which a later per-row timing of the sampler does not agree with; see the scheduler loop above |
| 6 | The compatible routes: `/v1/chat/completions`, `/v1/completions`, `/v1/models` in the OpenAI clients' shape (**done**) | The `server` component: greedy equality with the CLI through `/v1/completions` whole and streamed, usage counts, the role in the first chat chunk and the finish reason in the last, text content parts, the refusals' shape; CPU and device |
